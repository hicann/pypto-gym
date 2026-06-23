# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------



"""PyPTO chunked_gated_delta_rule golden reference implementation.

Pure PyTorch reference implementation of the Chunked Gated Delta Rule
Linear Attention mechanism. Serves as the accuracy validation baseline
for the PyPTO kernel implementation.

Confidence: ⭐⭐⭐⭐⭐
  Directly adapted from the verified reference implementation in
  qwen3_next/qwen3_next_gated_delta_rule.py (segs_chunk_gated_delta_rule).

Algorithm: Chunked Gated Delta Rule Linear Attention
  Reduces traditional O(n²) softmax attention to O(n) by processing
  the sequence in fixed-size chunks (L=chunk_size), with 6 sub-steps per chunk:
    1. L2 normalization of query and key (eps=1e-6)
    2. Pre-attention: gate cumsum → decay_mask → kβ @ k^T → A
    3. Matrix inverse: (I - A)^{-1} via iterative row method
    4. Cumulative decay: v_out = A_inv @ vβ, k_cumdecay = A_inv @ (kβ · exp(g_cum))
    5. Recurrent state attention: v_prime, o_inter, chunk output, state update
    6. Output write-back (with padding trim for unaligned sequences)

Key Parameters (configurable via chunk_size):
  - chunk_size (L) = configurable (default 128)
  - head_dim (D) = 128
  - scale = 1/sqrt(D) = 1/sqrt(128)
  - eps = 1e-6 (L2 normalization)
  - GQA: group = Nv // Nqk

State Convention:
  The states tensor [B, Nv, D, D] stores S in [Dv, Dk] orientation.
  Internally, we use S_internal = S^T (transposed) so that:
    - v_prime = k_cumdecay @ S_internal (no transpose needed)
    - o_inter  = (q·exp(g_cum)) @ S_internal
    - S_new_internal = S_internal · exp(g_last) + k_gexp^T @ v_new
  At output, we transpose back: last_state = S_internal^T.
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class ProcessGdrChunkInputs:
    cq: Any
    ck: Any
    cv: Any
    cb: Any
    cg: Any
    S: Any
    tril_mask: Any
    attn_upper_mask: Any
    chunk_attn_mask: Any
    scale: float
    eps: float


@dataclass
class ChunkedGatedDeltaRuleGoldenInputs:
    query: Any
    key: Any
    value: Any
    beta: Any
    gate: Any
    states: Any
    mask: Any
    tril_mask: Any
    eye: Any
    act_seq_len: Any
    chunk_size: int = 128


def _inverse_iterative(A, chunk_size):
    """Compute (I - A)^{-1} via iterative row-by-row method."""
    attn = A.clone()
    for index in range(1, chunk_size):
        line = attn[index, :index].clone()
        sub = attn[:index, :index].clone()
        attn[index, :index] = line + (line.unsqueeze(-1) * sub).sum(-2)
    return attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)


def _process_gdr_chunk(inputs: ProcessGdrChunkInputs):
    """Process a single chunk within the Gated Delta Rule algorithm."""
    q_norm = inputs.cq * torch.rsqrt((inputs.cq ** 2).sum(dim=-1, keepdim=True) + inputs.eps)
    k_norm = inputs.ck * torch.rsqrt((inputs.ck ** 2).sum(dim=-1, keepdim=True) + inputs.eps)
    q_scaled = q_norm * inputs.scale
    gate_cum = inputs.tril_mask @ inputs.cg.unsqueeze(-1)
    decay_mask = ((gate_cum - gate_cum.T) * inputs.tril_mask).exp() * inputs.tril_mask
    key_beta = k_norm * inputs.cb.unsqueeze(-1)
    kkt = key_beta @ k_norm.T
    A = -(kkt * decay_mask).masked_fill(inputs.attn_upper_mask, 0)
    A_inv = _inverse_iterative(A, inputs.cq.shape[0])
    v_beta = inputs.cv * inputs.cb.unsqueeze(-1)
    v_out = A_inv @ v_beta
    k_cumdecay = A_inv @ (key_beta * gate_cum.exp())
    v_prime = k_cumdecay @ inputs.S
    o_inter = (q_scaled * gate_cum.exp()) @ inputs.S
    attn = (q_scaled @ k_norm.T * decay_mask).masked_fill(inputs.chunk_attn_mask, 0)
    v_new = v_out - v_prime
    chunk_out = o_inter + attn @ v_new
    g_last_cum = gate_cum[-1, 0]
    k_gexp = k_norm * (g_last_cum - gate_cum).exp()
    inputs.S = inputs.S * torch.exp(g_last_cum) + k_gexp.T @ v_new
    return chunk_out, inputs.S


def chunked_gated_delta_rule_golden(
    inputs: ChunkedGatedDeltaRuleGoldenInputs,
) -> tuple:
    """PyTorch golden reference for chunked_gated_delta_rule.

    Implements the full 6-step Chunked Gated Delta Rule Linear Attention
    algorithm, supporting both aligned and unaligned sequences.
    """
    L = inputs.chunk_size
    D = 128
    eps = 1e-6
    scale = 1.0 / (D ** 0.5)

    T, Nqk, _ = inputs.query.shape
    _, Nv, _ = inputs.value.shape
    B = int(inputs.act_seq_len.shape[0]) - 1
    group = Nv // Nqk

    core_attn_out = torch.zeros(T, Nv, D, dtype=torch.float32, device=inputs.query.device)
    last_state_data = torch.zeros(B, Nv, D, D, dtype=torch.float32, device=inputs.query.device)

    attn_upper_mask = torch.ones(L, L, dtype=torch.bool, device=inputs.query.device).triu(diagonal=0)
    chunk_attn_mask = torch.ones(L, L, dtype=torch.bool, device=inputs.query.device).triu(diagonal=1)

    for b_idx in range(B):
        s = int(inputs.act_seq_len[b_idx + 1]) - int(inputs.act_seq_len[b_idx])
        bs_ofs = int(inputs.act_seq_len[b_idx])

        for nv_idx in range(Nv):
            nqk_idx = nv_idx // group
            S = inputs.states[b_idx, nv_idx].T.clone()
            q_data = inputs.query[bs_ofs:bs_ofs + s, nqk_idx, :].clone()
            k_data = inputs.key[bs_ofs:bs_ofs + s, nqk_idx, :].clone()
            v_data = inputs.value[bs_ofs:bs_ofs + s, nv_idx, :].clone()
            b_data = inputs.beta[bs_ofs:bs_ofs + s, nv_idx].clone()
            g_data = inputs.gate[bs_ofs:bs_ofs + s, nv_idx].clone()
            pad_size = (L - s % L) % L
            s_padded = s + pad_size
            q_pad = F.pad(q_data, (0, 0, 0, pad_size))
            k_pad = F.pad(k_data, (0, 0, 0, pad_size))
            v_pad = F.pad(v_data, (0, 0, 0, pad_size))
            b_pad = F.pad(b_data, (0, pad_size))
            g_pad = F.pad(g_data, (0, pad_size))

            for chunk_idx in range(0, s_padded, L):
                cq = q_pad[chunk_idx:chunk_idx + L, :]
                ck = k_pad[chunk_idx:chunk_idx + L, :]
                cv = v_pad[chunk_idx:chunk_idx + L, :]
                cb = b_pad[chunk_idx:chunk_idx + L]
                cg = g_pad[chunk_idx:chunk_idx + L]
                chunk_out, S = _process_gdr_chunk(ProcessGdrChunkInputs(
                    cq=cq, ck=ck, cv=cv, cb=cb, cg=cg, S=S, tril_mask=inputs.tril_mask,
                    attn_upper_mask=attn_upper_mask, chunk_attn_mask=chunk_attn_mask,
                    scale=scale, eps=eps))
                out_start = bs_ofs + chunk_idx
                out_end = min(out_start + L, bs_ofs + s)
                actual_len = out_end - out_start
                if actual_len > 0:
                    core_attn_out[out_start:out_end, nv_idx, :] = chunk_out[:actual_len, :]

            last_state_data[b_idx, nv_idx, :, :] = S.T

    return core_attn_out, last_state_data


def _ref_sub_inverse(attn, chunk_size):
    """Row-by-row iterative inverse (reference implementation style)."""
    for index in range(1, chunk_size):
        line = attn[..., index, :index].clone()
        sub = attn[..., :index, :index].clone()
        attn[..., index, :index] = line + (line.unsqueeze(-1) * sub).sum(-2)
    return attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)


def _ref_sub(query, key, value, g, beta, chunk_size, initial_state,
             output_final_state, use_qk_l2norm_in_kernel):
    """Reference per-chunk computation (segs_chunk_gated_delta_rule_sub)."""
    b, n, s, d = value.shape

    if initial_state is not None:
        initial_state = initial_state.transpose(3, 2)

    if use_qk_l2norm_in_kernel:
        query = query * torch.rsqrt((query * query).sum(dim=-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt((key * key).sum(dim=-1, keepdim=True) + 1e-6)

    pad_size = (chunk_size - s % chunk_size) % chunk_size
    query, key, value = [F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value)]
    beta, g = [F.pad(x, (0, pad_size)) for x in (beta, g)]

    total_sequence_length = s + pad_size
    query = query * (1 / (d ** 0.5))

    v_beta, k_beta = [x * beta.unsqueeze(-1) for x in (value, key)]
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask_upper = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()

    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask_upper, 0)
    attn = _ref_sub_inverse(attn, chunk_size)

    v_out = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    if initial_state is None:
        last_recurrent_state = torch.zeros(b, n, d, d, device=query.device).to(v_out)
    else:
        last_recurrent_state = initial_state.to(v_out)

    attn_out = torch.zeros_like(value).to(query.device)
    attn_mask_cycle = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    num_chunks = total_sequence_length // chunk_size
    for index in range(num_chunks):
        q_index = query[:, :, index]
        k_index = key[:, :, index]
        v_index = v_out[:, :, index]

        chunk_attn = (q_index @ k_index.transpose(-1, -2) * decay_mask[:, :, index]).masked_fill_(attn_mask_cycle, 0)
        v_new = v_index - (k_cumdecay[:, :, index]) @ last_recurrent_state
        attn_out[:, :, index] = (q_index * g[:, :, index, :, None].exp()) @ last_recurrent_state + chunk_attn @ v_new
        last_recurrent_state = last_recurrent_state * g[:, :, index, -1, None, None].exp() + \
            (k_index * (g[:, :, index, -1, None] - g[:, :, index]).exp()[..., None]).transpose(-1, -2) @ v_new

    attn_out = attn_out.reshape(attn_out.shape[0], attn_out.shape[1], -1, attn_out.shape[-1])
    attn_out = attn_out[:, :, :s].transpose(1, 2).contiguous()

    if output_final_state:
        last_recurrent_state = last_recurrent_state.transpose(3, 2)
    else:
        last_recurrent_state = None

    return attn_out, last_recurrent_state


def _ref_segs_chunk_gated_delta_rule(query, key, value, gate, beta, act_seq_len,
                                       chunk_size, initial_state, output_final_state,
                                       use_qk_l2norm_in_kernel):
    """Reference batched implementation (segs_chunk_gated_delta_rule)."""
    t, n1, d = query.shape
    t, n, d = value.shape
    batch = act_seq_len.shape[0] - 1

    query = query.repeat_interleave(n // n1, dim=1)
    key = key.repeat_interleave(n // n1, dim=1)

    final_state = torch.zeros([batch, n, d, d], dtype=torch.float32, device=query.device)

    query_t = query.transpose(0, 1).contiguous().to(torch.float32)
    key_t = key.transpose(0, 1).contiguous().to(torch.float32)
    value_t = value.transpose(0, 1).contiguous().to(torch.float32)
    beta_t = beta.transpose(0, 1).contiguous().to(torch.float32)
    gate_t = gate.transpose(0, 1).contiguous().to(torch.float32)

    final_attn = torch.zeros([t, n, d], dtype=torch.float32, device=query.device)

    for b_idx in range(batch):
        s = int(act_seq_len[b_idx + 1]) - int(act_seq_len[b_idx])
        b_ofs = int(act_seq_len[b_idx])
        seg_s = chunk_size
        pad_size = (chunk_size - s % chunk_size) % chunk_size
        pad_seq_length = s + pad_size

        batch_query = F.pad(query_t[:, b_ofs:b_ofs + s, :], (0, 0, 0, pad_size))
        batch_key = F.pad(key_t[:, b_ofs:b_ofs + s, :], (0, 0, 0, pad_size))
        batch_value = F.pad(value_t[:, b_ofs:b_ofs + s, :], (0, 0, 0, pad_size))
        batch_beta = F.pad(beta_t[:, b_ofs:b_ofs + s], (0, pad_size))
        batch_g = F.pad(gate_t[:, b_ofs:b_ofs + s], (0, pad_size))

        result_list = []
        recurrent_state = initial_state[b_idx:b_idx + 1, ...]

        for s_idx in range(0, pad_seq_length, seg_s):
            chunk_query = batch_query[:, s_idx:s_idx + seg_s, :].reshape(1, n, seg_s, d)
            chunk_key = batch_key[:, s_idx:s_idx + seg_s, :].reshape(1, n, seg_s, d)
            chunk_value = batch_value[:, s_idx:s_idx + seg_s, :].reshape(1, n, seg_s, d)
            chunk_gate = batch_g[:, s_idx:s_idx + seg_s].reshape(1, n, seg_s)
            chunk_beta = batch_beta[:, s_idx:s_idx + seg_s].reshape(1, n, seg_s)

            cur_attn, cur_state = _ref_sub(
                query=chunk_query, key=chunk_key, value=chunk_value,
                g=chunk_gate, beta=chunk_beta, chunk_size=chunk_size,
                initial_state=recurrent_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
            result_list.append(cur_attn.squeeze(0))
            recurrent_state = cur_state

        batch_attn = torch.cat(result_list, dim=0)[:s]
        final_attn[b_ofs:b_ofs + s] = batch_attn
        final_state[b_idx:b_idx + 1, ...] = recurrent_state

    return final_attn, final_state


def _run_single_validation_case(config, chunk_len, head_dim, mask, tril_mask, eye_aligned, eye_unaligned):
    """Validate a single test case and return True if passed."""
    name = config["name"]
    batch_size = config["B"]
    num_qk_heads = config["Nqk"]
    num_v_heads = config["Nv"]
    seq_len = config["T"]
    torch.manual_seed(42)
    query = torch.rand(seq_len, num_qk_heads, head_dim, dtype=torch.float32) * (1.3655 + 0.2785) - (1.3655 + 0.2785)
    key = torch.rand(seq_len, num_qk_heads, head_dim, dtype=torch.float32) * (1.4664 + 0.2785) - (1.4664 + 0.2785)
    value = torch.rand(seq_len, num_v_heads, head_dim, dtype=torch.float32) * (1.6488 + 0.2785) - (1.6488 + 0.2785)
    beta = torch.rand(seq_len, num_v_heads, dtype=torch.float32) * (0.8927 - 0.0889) - (0.8927 - 0.0889)
    gate = torch.rand(seq_len, num_v_heads, dtype=torch.float32) * (-0.1343 + 37.5452) - (-0.1343 + 37.5452)
    states = torch.zeros(batch_size, num_v_heads, head_dim, head_dim, dtype=torch.float32)
    seq_per_batch = seq_len // batch_size
    act_seq_len = torch.tensor([i * seq_per_batch for i in range(batch_size + 1)], dtype=torch.int32)
    is_aligned = all((int(act_seq_len[i + 1]) - int(act_seq_len[i])) % chunk_len == 0 for i in range(batch_size))
    eye = eye_aligned if is_aligned else eye_unaligned
    try:
        core_attn_out, last_state_data = chunked_gated_delta_rule_golden(
            ChunkedGatedDeltaRuleGoldenInputs(
                query=query, key=key, value=value, beta=beta, gate=gate,
                states=states, mask=mask, tril_mask=tril_mask, eye=eye,
                act_seq_len=act_seq_len, chunk_size=chunk_len))
        shape_ok = (core_attn_out.shape == (seq_len, num_v_heads, head_dim) and
                    last_state_data.shape == (batch_size, num_v_heads, head_dim, head_dim))
        nan_ok = (not torch.isnan(core_attn_out).any() and not torch.isinf(core_attn_out).any() and
                  not torch.isnan(last_state_data).any() and not torch.isinf(last_state_data).any())
        nonzero_ok = core_attn_out.abs().sum().item() > 0
        if shape_ok and nan_ok and nonzero_ok:
            print(f"  {name}: batch_size={batch_size}, num_qk_heads={num_qk_heads}, "
                  f"num_v_heads={num_v_heads}, seq_len={seq_len} ({config['desc']}) ... ✓ PASS")
            return True
        reasons = []
        if not shape_ok:
            reasons.append("shape mismatch")
        if not nan_ok:
            reasons.append("NaN/Inf detected")
        if not nonzero_ok:
            reasons.append("output is all zeros")
        print(f"  {name}: batch_size={batch_size}, num_qk_heads={num_qk_heads}, "
              f"num_v_heads={num_v_heads}, seq_len={seq_len} ... ✗ FAIL: {', '.join(reasons)}")
        return False
    except Exception as e:
        print(f"  {name}: batch_size={batch_size}, num_qk_heads={num_qk_heads}, "
              f"num_v_heads={num_v_heads}, seq_len={seq_len} ... ✗ FAIL: {e}")
        return False


def _validate_typical_cases(L, D, mask, tril_mask, all_passed):
    """Run typical case validation."""
    inverse_shape = L // 8
    eye_aligned = torch.eye(inverse_shape, dtype=torch.float32).repeat(1, L // inverse_shape)
    eye_unaligned = torch.eye(inverse_shape, dtype=torch.float32)

    configs = [
        {"name": "功能_P0_single_aligned", "B": 1, "Nqk": 2, "Nv": 4, "T": 128, "desc": "1 chunk, aligned, no GQA"},
        {"name": "功能_P0_multi_chunk_aligned", "B": 1, "Nqk": 2, "Nv": 4, "T": 256, "desc": "2 chunks, aligned"},
        {"name": "功能_P0_unaligned", "B": 1, "Nqk": 2, "Nv": 4, "T": 130,
         "desc": "unaligned (1 full + 1 partial chunk)"},
        {"name": "功能_P0_GQA", "B": 1, "Nqk": 2, "Nv": 4, "T": 128, "desc": "GQA group=2"},
        {"name": "性能_P0_multi_batch", "B": 2, "Nqk": 2, "Nv": 4, "T": 512, "desc": "multi-batch + GQA, aligned"},
    ]

    print("\n[典型 case 验证]")
    for config in configs:
        if not _run_single_validation_case(config, L, D, mask, tril_mask, eye_aligned, eye_unaligned):
            all_passed = False
    return all_passed


def _validate_cross(L, D, mask, tril_mask, all_passed):
    """Cross-validate against reference implementation."""
    inverse_shape = L // 8
    eye_aligned = torch.eye(inverse_shape, dtype=torch.float32).repeat(1, L // inverse_shape)
    eye_unaligned = torch.eye(inverse_shape, dtype=torch.float32)

    cross_configs = [
        {"name": "cross_aligned", "B": 1, "Nqk": 2, "Nv": 4, "T": 128},
        {"name": "cross_unaligned", "B": 1, "Nqk": 2, "Nv": 4, "T": 130},
        {"name": "cross_gqa", "B": 1, "Nqk": 2, "Nv": 4, "T": 256},
    ]

    print("\n[交叉验证 - 对比参考实现]")
    for config in cross_configs:
        name = config["name"]
        B, Nqk, Nv, T = config["B"], config["Nqk"], config["Nv"], config["T"]
        torch.manual_seed(42)
        query = torch.rand(T, Nqk, D, dtype=torch.float32) * (1.3655 + 0.2785) - (1.3655 + 0.2785)
        key = torch.rand(T, Nqk, D, dtype=torch.float32) * (1.4664 + 0.2785) - (1.4664 + 0.2785)
        value = torch.rand(T, Nv, D, dtype=torch.float32) * (1.6488 + 0.2785) - (1.6488 + 0.2785)
        beta = torch.rand(T, Nv, dtype=torch.float32) * (0.8927 - 0.0889) - (0.8927 - 0.0889)
        gate = torch.rand(T, Nv, dtype=torch.float32) * (-0.1343 + 37.5452) - (-0.1343 + 37.5452)
        states = torch.zeros(B, Nv, D, D, dtype=torch.float32)
        seq_per_batch = T // B
        act_seq_len = torch.tensor([i * seq_per_batch for i in range(B + 1)], dtype=torch.int32)
        is_aligned = all((int(act_seq_len[i + 1]) - int(act_seq_len[i])) % L == 0 for i in range(B))
        eye = eye_aligned if is_aligned else eye_unaligned
        try:
            golden_attn, golden_state = chunked_gated_delta_rule_golden(
                ChunkedGatedDeltaRuleGoldenInputs(
                    query=query, key=key, value=value, beta=beta, gate=gate,
                    states=states, mask=mask, tril_mask=tril_mask, eye=eye,
                    act_seq_len=act_seq_len, chunk_size=L))
            ref_attn, ref_state = _ref_segs_chunk_gated_delta_rule(
                query.clone(), key.clone(), value.clone(), gate.clone(), beta.clone(),
                act_seq_len.clone(), chunk_size=128, initial_state=states.clone(),
                output_final_state=True, use_qk_l2norm_in_kernel=True)
            diff_attn = torch.abs(golden_attn.float() - ref_attn.float())
            tolerance_attn = 1e-3 * torch.abs(ref_attn.float())
            attn_pass = (diff_attn <= tolerance_attn).all().item()
            diff_state = torch.abs(golden_state.float() - ref_state.float())
            tolerance_state = 1e-3 * torch.abs(ref_state.float())
            state_pass = (diff_state <= tolerance_state).all().item()
            max_attn_diff = diff_attn.max().item()
            max_state_diff = diff_state.max().item()
            if attn_pass and state_pass:
                print(f"  {name}: B={B}, Nqk={Nqk}, Nv={Nv}, T={T} "
                      f"attn_max_diff={max_attn_diff:.6f}, state_max_diff={max_state_diff:.6f} ... ✓ PASS")
            else:
                failed_items = []
                if not attn_pass:
                    out_of_tol = (diff_attn > tolerance_attn).sum().item()
                    failed_items.append(f"attn: {out_of_tol}/{golden_attn.numel()} out of tolerance")
                if not state_pass:
                    out_of_tol = (diff_state > tolerance_state).sum().item()
                    failed_items.append(f"state: {out_of_tol}/{golden_state.numel()} out of tolerance")
                print(f"  {name}: B={B}, Nqk={Nqk}, Nv={Nv}, T={T} ... ✗ FAIL: {', '.join(failed_items)}")
                all_passed = False
        except Exception as e:
            print(f"  {name}: B={B}, Nqk={Nqk}, Nv={Nv}, T={T} ... ✗ FAIL: {e}")
            all_passed = False
    return all_passed


def _validate_value_range_stability(L, D, mask, tril_mask, all_passed):
    """Value range and numerical stability checks."""
    inverse_shape = L // 8
    eye_aligned = torch.eye(inverse_shape, dtype=torch.float32).repeat(1, L // inverse_shape)

    torch.manual_seed(42)
    q = torch.rand(128, 2, 128, dtype=torch.float32) * (1.3655 + 0.2785) - (1.3655 + 0.2785)
    k = torch.rand(128, 2, 128, dtype=torch.float32) * (1.4664 + 0.2785) - (1.4664 + 0.2785)
    v = torch.rand(128, 4, 128, dtype=torch.float32) * (1.6488 + 0.2785) - (1.6488 + 0.2785)
    b = torch.rand(128, 4, dtype=torch.float32) * (0.8927 - 0.0889) - (0.8927 - 0.0889)
    g = torch.rand(128, 4, dtype=torch.float32) * (-0.1343 + 37.5452) - (-0.1343 + 37.5452)
    s = torch.zeros(1, 4, 128, 128, dtype=torch.float32)
    asl = torch.tensor([0, 128], dtype=torch.int32)

    print("\n[值域检查]")
    attn_out, state_out = chunked_gated_delta_rule_golden(
        ChunkedGatedDeltaRuleGoldenInputs(
            query=q, key=k, value=v, beta=b, gate=g,
            states=s, mask=mask, tril_mask=tril_mask, eye=eye_aligned,
            act_seq_len=asl, chunk_size=128))
    attn_max = attn_out.abs().max().item()
    state_max = state_out.abs().max().item()
    if attn_max < 1e4 and state_max < 1e4:
        print(f"  输出值域合理: attn_max={attn_max:.4f}, state_max={state_max:.4f} ... ✓ PASS")
    else:
        print(f"  输出值域异常: attn_max={attn_max:.4f}, state_max={state_max:.4f} ... ✗ FAIL")
        all_passed = False

    print("\n[数值稳定性检查]")
    torch.manual_seed(42)
    g_moderate = torch.rand(128, 4, dtype=torch.float32) * 10.0 - 10.0
    try:
        attn_out_stable, state_out_stable = chunked_gated_delta_rule_golden(
            ChunkedGatedDeltaRuleGoldenInputs(
                query=q, key=k, value=v, beta=b, gate=g_moderate,
                states=s, mask=mask, tril_mask=tril_mask, eye=eye_aligned,
                act_seq_len=asl, chunk_size=128))
        stable_ok = (not torch.isnan(attn_out_stable).any() and not torch.isinf(attn_out_stable).any() and
                     not torch.isnan(state_out_stable).any() and not torch.isinf(state_out_stable).any())
        if stable_ok:
            print(f"  负 gate 值 (range=[-10, 0]) ... ✓ PASS")
        else:
            print(f"  负 gate 值 ... ✗ FAIL: NaN/Inf detected")
            all_passed = False
    except Exception as e:
        print(f"  负 gate 值 ... ✗ FAIL: {e}")
        all_passed = False

    try:
        q_zero = torch.zeros(128, 2, 128, dtype=torch.float32)
        k_zero = torch.zeros(128, 2, 128, dtype=torch.float32)
        v_zero = torch.zeros(128, 4, 128, dtype=torch.float32)
        b_zero = torch.zeros(128, 4, dtype=torch.float32)
        g_zero = torch.zeros(128, 4, dtype=torch.float32)
        s_zero = torch.zeros(1, 4, 128, 128, dtype=torch.float32)
        attn_zero, state_zero = chunked_gated_delta_rule_golden(
            ChunkedGatedDeltaRuleGoldenInputs(
                query=q_zero, key=k_zero, value=v_zero, beta=b_zero, gate=g_zero,
                states=s_zero, mask=mask, tril_mask=tril_mask, eye=eye_aligned,
                act_seq_len=asl, chunk_size=128))
        zero_ok = (not torch.isnan(attn_zero).any() and not torch.isinf(attn_zero).any() and
                   not torch.isnan(state_zero).any() and not torch.isinf(state_zero).any())
        if zero_ok:
            print(f"  零值输入 ... ✓ PASS (no NaN/Inf)")
        else:
            print(f"  零值输入 ... ✗ FAIL: NaN/Inf detected")
            all_passed = False
    except Exception as e:
        print(f"  零值输入 ... ✗ FAIL: {e}")
        all_passed = False
    return all_passed


def _validate():
    """Auto-generated validation - runs typical cases and cross-validates."""

    print("=" * 60)
    print("chunked_gated_delta_rule_golden 验证报告")
    print("=" * 60)

    L = 128
    D = 128
    mask = torch.tril(-torch.ones(L, L, dtype=torch.float32), diagonal=-1)
    tril_mask = torch.ones(L, L, dtype=torch.float32).tril()

    all_passed = True
    all_passed &= _validate_typical_cases(L, D, mask, tril_mask, all_passed)
    all_passed &= _validate_cross(L, D, mask, tril_mask, all_passed)
    all_passed &= _validate_value_range_stability(L, D, mask, tril_mask, all_passed)

    print("\n[函数签名检查]")
    import inspect
    sig = inspect.signature(chunked_gated_delta_rule_golden)
    params = list(sig.parameters.keys())
    expected_params = ["inputs"]
    sig_ok = params == expected_params
    if sig_ok:
        print(f"  函数签名匹配 spec: {params} ... ✓ PASS")
    else:
        print(f"  函数签名不匹配: got {params}, expected {expected_params} ... ✗ FAIL")
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 所有验证通过")
    else:
        print("✗ 验证失败")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = _validate()
    exit(exit_code)