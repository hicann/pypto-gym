# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
PyPTO flash_attention_score golden reference implementation.

Flash Attention Score with PSE (Positional Score Encoding) and Dropout,
operating with GQA (Grouped Query Attention). Uses online-softmax algorithm
(Flash Attention style) with tiled computation over KV blocks.

Reference implementation constraints:
  ALLOWED:
    - torch.matmul, elementwise ops (+ - * /)
    - torch.amax / torch.sum over named dim with explicit dim=
    - torch.exp, torch.maximum
    - explicit Python loops over batch / head / group / q_block / kv_block on HOST
    - torch.transpose(t, dim0, dim1) instead of .T / .t()
    - explicit torch.reshape before matmul/ops
  DISALLOWED:
    - .T / .t() / any implicit transpose form
    - torch.cumsum, torch.masked_fill, torch.tril, torch.triu, torch.flip

Confidence: (4/5) — known paper algorithm (FlashAttention, Dao et al. 2022)
with user-provided golden serving as baseline. Validated on concrete shapes.

This file is pure torch — NO PyPTO imports.
"""

import os
from typing import Any
import torch
from dataclasses import dataclass


BLOCK_SIZE_Q = 320
BLOCK_SIZE_KV = 320


@dataclass
class FlashAttentionInputs:
    """Input container for flash_attention_score.

    All tensors share the same device. Attn_mask values: 0 = valid, 1 = invalid.
    """
    query: torch.Tensor       # [B, N, Sq, D] bf16
    key: torch.Tensor         # [B, N_kv, Skv, D] bf16
    value: torch.Tensor       # [B, N_kv, Skv, D] bf16
    atten_mask: torch.Tensor  # [Sq, Skv] int32/bool/bf16, 0=valid, 1=invalid
    pse: torch.Tensor         # [B, N, Sq, Skv] bf16 — positional score encoding
    drop_mask: torch.Tensor   # [Sq, Skv] bf16 — dropout mask (binary 0/1)
    pse_type: int             # 1 = (scores+pse)*scale, 2 = scores*scale+pse
    keep_prob: float          # dropout keep probability
    scale_value: float        # attention scale, typically 1/sqrt(D)


@dataclass
class _ProcessKvBlockInputs:
    q_block_fp32: Any
    k_block_2d: Any
    kv_head_idx: Any
    b_idx: Any
    kv_start: Any
    cur_block_size: Any
    query: Any
    key: Any
    value: Any
    pse: Any
    atten_mask_fp32: Any
    drop_mask: Any
    pse_type: Any
    scale: Any
    n_idx: Any
    q_start: Any
    cur_q_size: Any
    mi_update: Any
    li_update: Any
    oi_update: Any
    kv_block_idx: Any
    d: Any
    skv: Any


def _process_kv_block(inputs: _ProcessKvBlockInputs):
    cur_block_size_loc = inputs.cur_block_size
    k_block_fp32 = inputs.k_block_2d.float()
    k_block_transposed = torch.transpose(k_block_fp32, 1, 0)
    scores = torch.matmul(inputs.q_block_fp32, k_block_transposed)
    pse_block_2d = inputs.pse[
        inputs.b_idx, inputs.n_idx,
        inputs.q_start:inputs.q_start + inputs.cur_q_size,
        inputs.kv_start:inputs.kv_start + cur_block_size_loc
    ].reshape(inputs.cur_q_size, cur_block_size_loc)
    pse_fp32 = pse_block_2d.float()
    if inputs.pse_type == 1:
        scores_with_pse = scores + pse_fp32
        scores_scaled = scores_with_pse * inputs.scale
    else:
        scores_scaled_val = scores * inputs.scale
        scores_scaled = scores_scaled_val + pse_fp32
    mask_block = inputs.atten_mask_fp32[
        inputs.q_start:inputs.q_start + inputs.cur_q_size,
        inputs.kv_start:inputs.kv_start + cur_block_size_loc]
    valid_mask = (mask_block + (-1.0)) * (-1.0)
    m_ij = torch.amax(scores_scaled, dim=-1, keepdim=True)
    s_ij_sub_m = scores_scaled - m_ij
    p_ij = torch.exp(s_ij_sub_m)
    p_ij = p_ij * valid_mask
    drop_mask_block = inputs.drop_mask[
        inputs.q_start:inputs.q_start + inputs.cur_q_size,
        inputs.kv_start:inputs.kv_start + cur_block_size_loc]
    p_ij = p_ij * drop_mask_block
    l_ij = torch.sum(p_ij, dim=-1, keepdim=True)
    v_block_2d = inputs.value[inputs.b_idx, inputs.kv_head_idx,
                       inputs.kv_start:inputs.kv_start + cur_block_size_loc, :].reshape(cur_block_size_loc, inputs.d)
    v_block_fp32 = v_block_2d.float()
    o_ij = torch.matmul(p_ij, v_block_fp32)
    if inputs.kv_block_idx == 0:
        mi_update = m_ij
        li_update = l_ij
        oi_update = o_ij
    else:
        mi_new = torch.maximum(inputs.mi_update, m_ij)
        alpha = torch.exp(inputs.mi_update - mi_new)
        beta = torch.exp(m_ij - mi_new)
        li_update = alpha * inputs.li_update + beta * l_ij
        oi_update = alpha * inputs.oi_update + beta * o_ij
        mi_update = mi_new
    return mi_update, li_update, oi_update


@dataclass
class _MoveTensorsToDeviceOutputs:
    query: Any
    key: Any
    value: Any
    atten_mask: Any
    pse: Any
    drop_mask: Any


def _move_tensors_to_device(inputs, npu):
    """Select device and return tensors on the correct device."""
    if npu:
        import torch_npu
        torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", "0")))
        query = inputs.query.npu()              # [B, N, Sq, D] bf16
        key = inputs.key.npu()                  # [B, N_kv, Skv, D] bf16
        value = inputs.value.npu()              # [B, N_kv, Skv, D] bf16
        atten_mask = inputs.atten_mask.npu()    # [Sq, Skv] int32/bool/bf16
        pse = inputs.pse.npu()                  # [B, N, Sq, Skv] bf16
        drop_mask = inputs.drop_mask.npu()      # [Sq, Skv] bf16
    else:
        query = inputs.query              # [B, N, Sq, D] bf16
        key = inputs.key                  # [B, N_kv, Skv, D] bf16
        value = inputs.value              # [B, N_kv, Skv, D] bf16
        atten_mask = inputs.atten_mask    # [Sq, Skv] int32/bool/bf16
        pse = inputs.pse                  # [B, N, Sq, Skv] bf16
        drop_mask = inputs.drop_mask      # [Sq, Skv] bf16
    return _MoveTensorsToDeviceOutputs(query, key, value, atten_mask, pse, drop_mask)


@dataclass
class _InitOutputsAndSymbolsOutputs:
    output: Any
    softmax_max: Any
    softmax_sum: Any
    group: Any
    scale: Any
    num_blocks_kv: Any
    num_blocks_q: Any


def _init_outputs_and_symbols(query, key, inputs, b, n, sq, d, n_kv, skv):
    """Allocate output tensors and compute derived symbols."""
    scale_value = inputs.scale_value  # float scalar
    group = n // n_kv                  # GQA group size (e.g. 32/8=4)
    scale = scale_value                # 1/sqrt(D)
    output = torch.zeros(
        b, n, sq, d, dtype=torch.bfloat16, device=query.device)  # [B, N, Sq, D] bf16
    softmax_max = torch.zeros(
        b, n, sq, 1, dtype=torch.float32, device=query.device)   # [B, N, Sq, 1] fp32
    softmax_sum = torch.zeros(
        b, n, sq, 1, dtype=torch.float32, device=query.device)   # [B, N, Sq, 1] fp32
    num_blocks_kv = (skv + BLOCK_SIZE_KV - 1) // BLOCK_SIZE_KV
    num_blocks_q = (sq + BLOCK_SIZE_Q - 1) // BLOCK_SIZE_Q
    return _InitOutputsAndSymbolsOutputs(output, softmax_max, softmax_sum, group, scale, num_blocks_kv, num_blocks_q)


@dataclass
class _ProcessQBlockInputs:
    query: Any
    key: Any
    value: Any
    pse: Any
    atten_mask_fp32: Any
    drop_mask: Any
    b_idx: Any
    kv_head_idx: Any
    group_idx: Any
    group: Any
    q_block_idx: Any
    num_blocks_q: Any
    num_blocks_kv: Any
    sq: Any
    skv: Any
    d: Any
    scale: Any
    pse_type: Any
    keep_prob: Any
    mi_update_init: Any
    li_update_init: Any
    oi_update_init: Any
    output: Any
    softmax_max: Any
    softmax_sum: Any


def _process_q_block(inputs: _ProcessQBlockInputs):
    """Process a single Q-block: extract Q, run KV loop, write outputs."""
    n_idx = inputs.kv_head_idx * inputs.group + inputs.group_idx   # inputs.query head index
    q_start = inputs.q_block_idx * BLOCK_SIZE_Q
    cur_q_size = min(BLOCK_SIZE_Q, inputs.sq - q_start)
    q_block_2d = inputs.query[inputs.b_idx, n_idx, q_start:q_start + cur_q_size, :].reshape(cur_q_size, inputs.d)
    mi_update, li_update, oi_update = inputs.mi_update_init, inputs.li_update_init, inputs.oi_update_init

    for kv_block_idx in range(inputs.num_blocks_kv):
        kv_start = kv_block_idx * BLOCK_SIZE_KV
        cur_block_size = min(BLOCK_SIZE_KV, inputs.skv - kv_start)
        k_block_2d = inputs.key[inputs.b_idx, inputs.kv_head_idx,
                         kv_start:kv_start + cur_block_size, :].reshape(cur_block_size, inputs.d)
        q_block_fp32 = q_block_2d.float()
        mi_update, li_update, oi_update = _process_kv_block(_ProcessKvBlockInputs(
            q_block_fp32, k_block_2d, inputs.kv_head_idx, inputs.b_idx, kv_start,
            cur_block_size, inputs.query, inputs.key, inputs.value, inputs.pse, inputs.atten_mask_fp32,
            inputs.drop_mask, inputs.pse_type, inputs.scale, n_idx, q_start, cur_q_size,
            mi_update, li_update, oi_update, kv_block_idx, inputs.d, inputs.skv))

    o_final = oi_update / li_update                     # [cur_q, D] fp32
    o_final_bf16 = o_final.to(torch.bfloat16)           # [cur_q, D] bf16
    inputs.output[inputs.b_idx, n_idx, q_start:q_start + cur_q_size, :] = o_final_bf16.reshape(cur_q_size, inputs.d)
    inputs.softmax_max[inputs.b_idx, n_idx, q_start:q_start + cur_q_size, :] = mi_update.reshape(cur_q_size, 1)
    l_out = li_update                                   # [cur_q, 1] fp32
    if inputs.keep_prob < 1.0:
        l_out = l_out * (1.0 / inputs.keep_prob)               # [cur_q, 1] fp32
    inputs.softmax_sum[inputs.b_idx, n_idx, q_start:q_start + cur_q_size, :] = l_out.reshape(cur_q_size, 1)


def flash_attention_score_golden(inputs: FlashAttentionInputs, npu: bool = False) -> tuple:
    """PyPTO-friendly golden reference for Flash Attention Score with PSE and Dropout.

    Implements the online-softmax algorithm with GQA (Grouped Query Attention).
    Normalized for PyPTO compatibility:
      - ZERO .T / .t() calls — explicit torch.transpose
      - Shape comments on every intermediate tensor
      - Explicit reshape before matmul/operations
      - All intermediate tensors named
      - Module boundaries marked with # ===== (X) description =====

    Args:
        inputs: FlashAttentionInputs dataclass.
        npu: If True, move tensors to NPU, run golden on NPU hardware,
            then move outputs back to CPU for bit-exact matching with
            the kernel (which also runs on NPU). Default False (CPU).

    Returns:
        output:       [B, N, Sq, D] bf16  — attention output
        softmax_max:  [B, N, Sq, 1] fp32  — per-row online-softmax max
        softmax_sum:  [B, N, Sq, 1] fp32  — per-row online-softmax sum
    """
    result = _move_tensors_to_device(inputs, npu)
    query = result.query
    key = result.key
    value = result.value
    atten_mask = result.atten_mask
    pse = result.pse
    drop_mask = result.drop_mask
    pse_type = inputs.pse_type
    keep_prob = inputs.keep_prob
    b, n, sq, d = query.shape
    _, n_kv, skv, _ = key.shape

    symbols = _init_outputs_and_symbols(query, key, inputs, b, n, sq, d, n_kv, skv)
    output = symbols.output
    softmax_max = symbols.softmax_max
    softmax_sum = symbols.softmax_sum
    group = symbols.group
    scale = symbols.scale
    num_blocks_kv = symbols.num_blocks_kv
    num_blocks_q = symbols.num_blocks_q

    atten_mask_fp32 = atten_mask.float()              # [Sq, Skv] fp32
    dev = query.device

    for b_idx in range(b):
        for kv_head_idx in range(n_kv):
            for group_idx in range(group):
                for q_block_idx in range(num_blocks_q):
                    mi_u = torch.full((min(BLOCK_SIZE_Q, sq - q_block_idx * BLOCK_SIZE_Q), 1),
                                      float('-inf'), dtype=torch.float32, device=dev)
                    li_u = torch.zeros(min(BLOCK_SIZE_Q, sq - q_block_idx * BLOCK_SIZE_Q), 1,
                                       dtype=torch.float32, device=dev)
                    oi_u = torch.zeros(min(BLOCK_SIZE_Q, sq - q_block_idx * BLOCK_SIZE_Q), d,
                                       dtype=torch.float32, device=dev)
                    _process_q_block(_ProcessQBlockInputs(
                        query, key, value, pse, atten_mask_fp32, drop_mask,
                        b_idx, kv_head_idx, group_idx, group, q_block_idx,
                        num_blocks_q, num_blocks_kv, sq, skv, d, scale,
                        pse_type, keep_prob, mi_u, li_u, oi_u,
                        output, softmax_max, softmax_sum))

    if npu:
        output = output.cpu()
        softmax_max = softmax_max.cpu()
        softmax_sum = softmax_sum.cpu()
    return output, softmax_max, softmax_sum


def _run_p0_validation():
    """Validate golden with P0 target shape."""
    B, N, N_kv = 1, 32, 8
    Sq, Skv, D = 2048, 2048, 128
    device = "cpu"
    group = N // N_kv
    scale_value = 1.0 / (D ** 0.5)
    print(f"\n[典型 case 验证] P0: B={B}, N={N}, N_kv={N_kv}, Sq={Sq}, Skv={Skv}, D={D}")
    print(f"  group={group}, scale={scale_value:.6f}, pse_type=1, keep_prob=1.0")
    print(f"  BLOCK_SIZE_Q={BLOCK_SIZE_Q}, BLOCK_SIZE_KV={BLOCK_SIZE_KV}")
    query = torch.randn(B, N, Sq, D, dtype=torch.bfloat16, device=device) * 0.1
    key = torch.randn(B, N_kv, Skv, D, dtype=torch.bfloat16, device=device) * 0.1
    value = torch.randn(B, N_kv, Skv, D, dtype=torch.bfloat16, device=device) * 0.1
    atten_mask = torch.zeros(Sq, Skv, dtype=torch.int32, device=device)
    pse = torch.randn(B, N, Sq, Skv, dtype=torch.bfloat16, device=device) * 0.01
    drop_mask = torch.ones(Sq, Skv, dtype=torch.bfloat16, device=device)
    inputs_p0 = FlashAttentionInputs(
        query=query, key=key, value=value,
        atten_mask=atten_mask, pse=pse, drop_mask=drop_mask,
        pse_type=1, keep_prob=1.0, scale_value=scale_value,
    )
    print("  Running golden on P0 shape...")
    output, softmax_max, softmax_sum = flash_attention_score_golden(inputs_p0)
    print(f"\n  Output shapes:")
    print(f"    output:       {list(output.shape)}       expected [1, 32, 2048, 128]")
    print(f"    softmax_max:  {list(softmax_max.shape)}  expected [1, 32, 2048, 1]")
    print(f"    softmax_sum:  {list(softmax_sum.shape)}  expected [1, 32, 2048, 1]")
    assert output.shape == (B, N, Sq, D), f"output shape mismatch: {output.shape} != {(B, N, Sq, D)}"
    assert softmax_max.shape == (B, N, Sq, 1), f"softmax_max shape mismatch"
    assert softmax_sum.shape == (B, N, Sq, 1), f"softmax_sum shape mismatch"
    print("  ✓ Shape checks PASSED")
    print(f"\n  Dtypes:")
    print(f"    output dtype:       {output.dtype}       expected torch.bfloat16")
    print(f"    softmax_max dtype:  {softmax_max.dtype}  expected torch.float32")
    print(f"    softmax_sum dtype:  {softmax_sum.dtype}  expected torch.float32")
    assert output.dtype == torch.bfloat16, f"output dtype mismatch"
    assert softmax_max.dtype == torch.float32, f"softmax_max dtype mismatch"
    assert softmax_sum.dtype == torch.float32, f"softmax_sum dtype mismatch"
    print("  ✓ Dtype checks PASSED")
    return output, softmax_max, softmax_sum


def _run_small_shape_validation():
    """Validate golden with small shape (tests partial-block handling)."""
    B2, N2, N_kv2 = 1, 4, 2
    Sq2, Skv2, D2 = 128, 128, 64
    device = "cpu"
    scale2 = 1.0 / (D2 ** 0.5)
    print(f"\n[泛化 case 验证] Small shape: B={B2}, N={N2}, N_kv={N_kv2}, Sq={Sq2}, Skv={Skv2}, D={D2}")
    print(f"  BLOCK_SIZE_Q={BLOCK_SIZE_Q}, BLOCK_SIZE_KV={BLOCK_SIZE_KV}")
    print(f"  Note: Sq/Skv < BLOCK_SIZE: tests partial-block (tail) handling")
    query2 = torch.randn(B2, N2, Sq2, D2, dtype=torch.bfloat16, device=device) * 0.1
    key2 = torch.randn(B2, N_kv2, Skv2, D2, dtype=torch.bfloat16, device=device) * 0.1
    value2 = torch.randn(B2, N_kv2, Skv2, D2, dtype=torch.bfloat16, device=device) * 0.1
    atten_mask2 = torch.zeros(Sq2, Skv2, dtype=torch.int32, device=device)
    pse2 = torch.randn(B2, N2, Sq2, Skv2, dtype=torch.bfloat16, device=device) * 0.01
    drop_mask2 = torch.ones(Sq2, Skv2, dtype=torch.bfloat16, device=device)
    inputs_small = FlashAttentionInputs(
        query=query2, key=key2, value=value2,
        atten_mask=atten_mask2, pse=pse2, drop_mask=drop_mask2,
        pse_type=1, keep_prob=1.0, scale_value=scale2,
    )
    output2, max2, sum2 = flash_attention_score_golden(inputs_small)
    assert output2.shape == (B2, N2, Sq2, D2), f"small shape output mismatch"
    assert max2.shape == (B2, N2, Sq2, 1), f"small shape max mismatch"
    assert sum2.shape == (B2, N2, Sq2, 1), f"small shape sum mismatch"
    nan2 = torch.isnan(output2).any().item()
    inf2 = torch.isinf(output2).any().item()
    print(f"  Output shape: {list(output2.shape)} — NaN: {nan2}, Inf: {inf2}")
    assert not nan2 and not inf2, "small shape produced NaN/Inf"
    print("  ✓ Small shape PASSED")


def _validate():
    """Validate the golden function with concrete test shapes.

    Runs on:
      - P0 target shape: (B=1, N=32, N_kv=8, Sq=2048, Skv=2048, D=128)
      - Small shape: (B=1, N=4, N_kv=2, Sq=128, Skv=128, D=64)
    Checks shapes, NaN/Inf, and numerical reasonableness.
    """
    print("=" * 60)
    print("flash_attention_score_golden 验证报告")
    print("=" * 60)

    torch.manual_seed(42)

    output, softmax_max, softmax_sum = _run_p0_validation()

    nan_output = torch.isnan(output).any().item()
    inf_output = torch.isinf(output).any().item()
    nan_max = torch.isnan(softmax_max).any().item()
    inf_max = torch.isinf(softmax_max).any().item()
    nan_sum = torch.isnan(softmax_sum).any().item()
    inf_sum = torch.isinf(softmax_sum).any().item()

    print(f"\n  NaN/Inf check:")
    print(f"    output has NaN:       {nan_output},  Inf: {inf_output}")
    print(f"    softmax_max has NaN:  {nan_max},  Inf: {inf_max}")
    print(f"    softmax_sum has NaN:  {nan_sum},  Inf: {inf_sum}")

    assert not nan_output, "output contains NaN"
    assert not inf_output, "output contains Inf"
    assert not nan_max, "softmax_max contains NaN"
    assert not inf_max, "softmax_max contains -Inf (initial state not updated)"
    assert not nan_sum, "softmax_sum contains NaN"
    assert not inf_sum, "softmax_sum contains Inf"
    print("  ✓ NaN/Inf checks PASSED")

    sum_min = softmax_sum.min().item()
    sum_max = softmax_sum.max().item()
    sum_mean = softmax_sum.mean().item()
    print(f"\n  softmax_sum stats:")
    print(f"    min:  {sum_min:.6f}")
    print(f"    max:  {sum_max:.6f}")
    print(f"    mean: {sum_mean:.6f}")
    assert sum_min > 0, f"softmax_sum min ({sum_min}) should be positive"
    print("  ✓ softmax_sum positivity PASSED")

    _run_small_shape_validation()

    print("\n" + "=" * 60)
    print("所有验证通过")
    print("=" * 60)


if __name__ == "__main__":
    _validate()
