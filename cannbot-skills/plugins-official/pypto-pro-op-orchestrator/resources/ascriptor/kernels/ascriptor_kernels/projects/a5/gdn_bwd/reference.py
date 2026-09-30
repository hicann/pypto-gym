# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent Torch reference for the GDN backward: the staged decomposition each kernel
stage answers to, and the saved forward state the backward consumes."""

import torch

# ----------------------------------------------------------------------------------------------------
# formulas.py
# Independent Torch formulas for the preserved BF16-cube GDN backward ABI.
# ----------------------------------------------------------------------------------------------------

L = 64
D = 128


def strict_lower(device):
    return torch.tril(torch.ones(L, L, dtype=torch.float32, device=device), diagonal=-1)


def lower_eq(device):
    return torch.tril(torch.ones(L, L, dtype=torch.float32, device=device))


def reverse_cumsum(x):
    return torch.flip(torch.cumsum(torch.flip(x, dims=(-1,)), dim=-1), dims=(-1,))


def f32(x):
    return x if x.dtype == torch.float32 else x.float()


def bf16_cube_operand(x):
    return x.to(torch.bfloat16).float()


def bf16_cube_mm(lhs, rhs):
    """A5 cube contract for this plan: bf16 inputs, fp32 accumulation/output."""
    return bf16_cube_operand(lhs) @ bf16_cube_operand(rhs)


def generate_inputs(B=1, H=2, C=2, scale=0.05):
    query = (torch.randn(B, H, C, L, D, dtype=torch.float32) * scale).to(torch.bfloat16)
    key = (torch.randn(B, H, C, L, D, dtype=torch.float32) * scale).to(torch.bfloat16)
    value = (torch.randn(B, H, C, L, D, dtype=torch.float32) * scale).to(torch.bfloat16)
    beta = torch.rand(B, H, C, L, dtype=torch.float32)
    g = torch.log(torch.sigmoid(torch.randn(B, H, C, L, dtype=torch.float32)))
    grad_output = (torch.randn(B, H, C, L, D, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    grad_final_state = torch.randn(B, H, D, D, dtype=torch.float32) * 0.1

    g_cumsum = g.cumsum(dim=-1)
    decay_mask = (g_cumsum.unsqueeze(-1) - g_cumsum.unsqueeze(-2)).exp() * lower_eq(query.device)
    k = key.float()
    v = value.float()
    k_beta = k * beta.unsqueeze(-1)
    preprocess_attn = -(bf16_cube_mm(k_beta, k.transpose(-1, -2)) * decay_mask * strict_lower(query.device))
    wu_attn_fp32 = torch.linalg.inv(torch.eye(L, dtype=torch.float32, device=query.device) - preprocess_attn)
    wu_attn_bf16 = wu_attn_fp32.to(torch.bfloat16)
    v_beta = (v * beta.unsqueeze(-1)).to(torch.bfloat16).float()
    k_beta_g = (k_beta * g_cumsum.exp().unsqueeze(-1)).to(torch.bfloat16).float()
    value_wu = bf16_cube_mm(wu_attn_bf16, v_beta).to(torch.bfloat16)
    k_cumdecay = bf16_cube_mm(wu_attn_bf16, k_beta_g).to(torch.bfloat16)
    state_after_history, v_new_history, k_weighted_history, exp_delta_history = state_history_forward(key, g_cumsum, value_wu, k_cumdecay)

    return (
        query, key, value, beta, g, grad_output,
        g_cumsum, decay_mask, wu_attn_bf16, value_wu, k_cumdecay,
        state_after_history, v_new_history, k_weighted_history, grad_final_state,
        exp_delta_history,
    )


def state_history_forward(key, g_cumsum, value_wu, k_cumdecay):
    B, H, C, _, _ = key.shape
    state = torch.zeros(B, H, D, D, dtype=torch.bfloat16, device=key.device)
    states = []
    v_news = []
    k_weighteds = []
    exp_deltas = []
    for c in range(C):
        k_i = key[:, :, c].float()
        g_i = g_cumsum[:, :, c].float()
        v_new = (value_wu[:, :, c].float() - bf16_cube_mm(k_cumdecay[:, :, c], state)).to(torch.bfloat16)
        v_news.append(v_new)
        exp_g_last = g_i[:, :, -1].exp()
        exp_delta = (g_i[:, :, -1].unsqueeze(-1) - g_i).exp()
        exp_deltas.append(exp_delta)
        k_weighted_bf16 = (k_i * exp_delta.unsqueeze(-1)).to(torch.bfloat16)
        k_weighted = k_weighted_bf16.float()
        k_weighteds.append(k_weighted_bf16)
        state = (state.float() * exp_g_last[:, :, None, None] + bf16_cube_mm(k_weighted.transpose(-1, -2), v_new)).to(torch.bfloat16)
        states.append(state)
    return (
        torch.stack(states, dim=2),
        torch.stack(v_news, dim=2),
        torch.stack(k_weighteds, dim=2),
        torch.stack(exp_deltas, dim=2),
    )


def core_scan_bwd(query, key, grad_output, g_cumsum, decay_mask, k_cumdecay, state_after_history, v_new_history, k_weighted_history, grad_final_state=None, exp_delta_history=None):
    B, H, C, _, _ = query.shape
    q = bf16_cube_operand(query)
    k = bf16_cube_operand(key)
    g_cumsum = g_cumsum.float()
    decay_mask = decay_mask.float()
    k_cumdecay = bf16_cube_operand(k_cumdecay)
    state_after_history = bf16_cube_operand(state_after_history)
    v_new_history = bf16_cube_operand(v_new_history)
    k_weighted_history = bf16_cube_operand(k_weighted_history)
    exp_delta_history = None if exp_delta_history is None else exp_delta_history.float()
    final_state = state_after_history[:, :, C - 1]
    d_state = torch.zeros_like(final_state) if grad_final_state is None else grad_final_state.float()
    d_core = bf16_cube_operand(grad_output)

    d_q = torch.zeros_like(q)
    d_k = torch.zeros_like(k)
    d_g_cumsum = torch.zeros_like(g_cumsum)
    d_decay_mask = torch.zeros_like(decay_mask)
    d_value_wu = torch.zeros(B, H, C, L, D, dtype=torch.bfloat16, device=query.device)
    d_k_cumdecay = torch.zeros(B, H, C, L, D, dtype=torch.bfloat16, device=query.device)

    for c in range(C - 1, -1, -1):
        state = torch.zeros_like(final_state) if c == 0 else state_after_history[:, :, c - 1]
        q_i = q[:, :, c]
        k_i = k[:, :, c]
        d_out = d_core[:, :, c]
        g_i = g_cumsum[:, :, c]
        score_qk = bf16_cube_mm(q_i, k_i.transpose(-1, -2))
        attn = score_qk * decay_mask[:, :, c]
        v_new = v_new_history[:, :, c]
        exp_g_i = g_i.exp()
        q_exp = q_i * exp_g_i.unsqueeze(-1)
        exp_g_last = g_i[:, :, -1].exp()
        exp_delta = exp_delta_history[:, :, c] if exp_delta_history is not None else (g_i[:, :, -1].unsqueeze(-1) - g_i).exp()
        k_weighted = k_weighted_history[:, :, c]

        d_q_exp = bf16_cube_mm(d_out, state.transpose(-1, -2))
        d_state_in = bf16_cube_mm(q_exp.transpose(-1, -2), d_out) + d_state * exp_g_last[:, :, None, None]
        d_attn = bf16_cube_mm(d_out, v_new.transpose(-1, -2))
        d_v_new = bf16_cube_mm(attn.transpose(-1, -2), d_out)
        d_exp_g_last = (d_state * state).sum(dim=(-1, -2))
        d_k_weighted = bf16_cube_mm(v_new, d_state.transpose(-1, -2))
        d_v_new = d_v_new + bf16_cube_mm(k_weighted, d_state)

        d_value_wu[:, :, c] = d_value_wu[:, :, c] + d_v_new
        d_v_prime = -d_v_new
        d_k_cumdecay[:, :, c] = d_k_cumdecay[:, :, c] + bf16_cube_mm(d_v_prime, state.transpose(-1, -2))
        d_state_in = d_state_in + bf16_cube_mm(k_cumdecay[:, :, c].transpose(-1, -2), d_v_prime)

        d_score_qk = d_attn * decay_mask[:, :, c]
        d_decay_mask[:, :, c] = d_decay_mask[:, :, c] + d_attn * score_qk
        d_q[:, :, c] = d_q[:, :, c] + bf16_cube_mm(d_score_qk, k_i) + d_q_exp * exp_g_i.unsqueeze(-1)
        d_k[:, :, c] = d_k[:, :, c] + bf16_cube_mm(d_score_qk.transpose(-1, -2), q_i)
        d_g_cumsum[:, :, c] = d_g_cumsum[:, :, c] + (d_q_exp * q_i).sum(dim=-1) * exp_g_i

        d_k[:, :, c] = d_k[:, :, c] + d_k_weighted * exp_delta.unsqueeze(-1)
        d_delta = (d_k_weighted * k_i).sum(dim=-1) * exp_delta
        d_g_cumsum[:, :, c] = d_g_cumsum[:, :, c] - d_delta
        d_g_cumsum[:, :, c, -1] = d_g_cumsum[:, :, c, -1] + d_exp_g_last * exp_g_last + d_delta.sum(dim=-1)
        d_state = d_state_in

    return d_q, d_k, d_g_cumsum, d_decay_mask, d_value_wu, d_k_cumdecay


def scan_local_bwd(query, key, grad_output, decay_mask, v_new_history):
    B, H, C, _, _ = query.shape
    q = bf16_cube_operand(query)
    k = bf16_cube_operand(key)
    d_core = bf16_cube_operand(grad_output)
    decay_mask = decay_mask.float()
    v_new_history = bf16_cube_operand(v_new_history)

    d_score_tmp = torch.zeros(B, H, C, L, L, dtype=torch.bfloat16, device=query.device)
    d_v_attn_tmp = torch.zeros(B, H, C, L, D, dtype=torch.float32, device=query.device)
    d_decay_mask = torch.zeros_like(decay_mask)
    d_decay_masked = torch.zeros_like(decay_mask)

    for c in range(C):
        q_i = q[:, :, c]
        k_i = k[:, :, c]
        d_out = d_core[:, :, c]
        score_qk = bf16_cube_mm(q_i, k_i.transpose(-1, -2))
        v_new = v_new_history[:, :, c]
        d_attn = bf16_cube_mm(d_out, v_new.transpose(-1, -2))
        attn = score_qk * decay_mask[:, :, c]

        d_score_tmp[:, :, c] = (d_attn * decay_mask[:, :, c]).to(torch.bfloat16)
        d_v_attn_tmp[:, :, c] = bf16_cube_mm(attn.transpose(-1, -2), d_out)
        d_decay_mask[:, :, c] = d_attn * score_qk
        d_decay_masked[:, :, c] = d_decay_mask[:, :, c] * decay_mask[:, :, c]

    return d_score_tmp, d_v_attn_tmp, d_decay_mask, d_decay_masked


def _scan_state_vec_prelude(q_i, g_i, exp_delta_i=None):
    """Vector gate setup before the cube/vec/cube/vec state body."""
    exp_g_i = g_i.exp()
    q_exp = q_i * exp_g_i.unsqueeze(-1)
    exp_g_last = g_i[:, :, -1].exp()
    exp_delta = exp_delta_i.float() if exp_delta_i is not None else (g_i[:, :, -1].unsqueeze(-1) - g_i).exp()
    return exp_g_i, q_exp, exp_g_last, exp_delta


def _scan_state_cube0(d_out, state, q_exp, v_new, k_weighted, d_state):
    """First cube wave: products that do not need d_v_prime."""
    d_q_exp = bf16_cube_mm(d_out, state.transpose(-1, -2))
    d_state_q = bf16_cube_mm(q_exp.transpose(-1, -2), d_out)
    d_k_weighted = bf16_cube_mm(v_new, d_state.transpose(-1, -2))
    d_v_state = bf16_cube_mm(k_weighted, d_state)
    return d_q_exp, d_state_q, d_k_weighted, d_v_state


def _scan_state_vec0(d_v_attn, d_v_state):
    """First vector wave: expose d_v_prime for the second cube wave."""
    d_v_new = d_v_attn + d_v_state
    d_v_prime = -d_v_new
    return d_v_new, d_v_prime


def _scan_state_cube1(d_score, q_i, k_i, d_v_prime, state, k_cumdecay_i):
    """Second cube wave: score products plus d_v_prime-dependent products."""
    d_q_score = bf16_cube_mm(d_score, k_i)
    d_k_score = bf16_cube_mm(d_score.transpose(-1, -2), q_i)
    d_k_cumdecay_i = bf16_cube_mm(d_v_prime, state.transpose(-1, -2))
    d_state_k = bf16_cube_mm(k_cumdecay_i.transpose(-1, -2), d_v_prime)
    return d_q_score, d_k_score, d_k_cumdecay_i, d_state_k


def _scan_state_vec1(q_i, k_i, state, d_state, exp_g_i, exp_g_last, exp_delta, d_q_exp, d_state_q, d_k_weighted, d_q_score, d_k_score, d_state_k):
    """Final vector wave: accumulate public grads and the recurrent d_state."""
    d_exp_g_last = (d_state * state).sum(dim=(-1, -2))

    d_q_i = d_q_score + d_q_exp * exp_g_i.unsqueeze(-1)
    d_k_i = d_k_score
    d_g_i = (d_q_exp * q_i).sum(dim=-1) * exp_g_i

    d_k_i = d_k_i + d_k_weighted * exp_delta.unsqueeze(-1)
    d_delta = (d_k_weighted * k_i).sum(dim=-1) * exp_delta
    d_g_i = d_g_i - d_delta
    d_g_i[:, :, -1] = d_g_i[:, :, -1] + d_exp_g_last * exp_g_last + d_delta.sum(dim=-1)

    d_state_next = d_state_q + d_state * exp_g_last[:, :, None, None]
    d_state_next = d_state_next + d_state_k
    return d_q_i, d_k_i, d_g_i, d_state_next


def scan_state_bwd(query, key, grad_output, g_cumsum, k_cumdecay, state_after_history, grad_final_state, d_score_tmp, v_new_history, k_weighted_history, d_v_attn_tmp, exp_delta_history=None):
    B, H, C, _, _ = query.shape
    q = bf16_cube_operand(query)
    k = bf16_cube_operand(key)
    g_cumsum = g_cumsum.float()
    k_cumdecay = bf16_cube_operand(k_cumdecay)
    state_after_history = bf16_cube_operand(state_after_history)
    v_new_history = bf16_cube_operand(v_new_history)
    k_weighted_history = bf16_cube_operand(k_weighted_history)
    exp_delta_history = None if exp_delta_history is None else exp_delta_history.float()
    final_state = state_after_history[:, :, C - 1]
    d_state = torch.zeros_like(final_state) if grad_final_state is None else grad_final_state.float()
    d_core = bf16_cube_operand(grad_output)

    d_q = torch.zeros_like(q)
    d_k = torch.zeros_like(k)
    d_g_cumsum = torch.zeros_like(g_cumsum)
    d_value_wu = torch.zeros(B, H, C, L, D, dtype=torch.bfloat16, device=query.device)
    d_k_cumdecay = torch.zeros(B, H, C, L, D, dtype=torch.bfloat16, device=query.device)

    for c in range(C - 1, -1, -1):
        state = torch.zeros_like(final_state) if c == 0 else state_after_history[:, :, c - 1]
        q_i = q[:, :, c]
        k_i = k[:, :, c]
        d_out = d_core[:, :, c]
        g_i = g_cumsum[:, :, c]
        exp_delta_i = exp_delta_history[:, :, c] if exp_delta_history is not None else None
        exp_g_i, q_exp, exp_g_last, exp_delta = _scan_state_vec_prelude(q_i, g_i, exp_delta_i)
        k_weighted = k_weighted_history[:, :, c]
        d_score = d_score_tmp[:, :, c].float()
        v_new = v_new_history[:, :, c]
        d_v_attn = d_v_attn_tmp[:, :, c].float()

        d_q_exp, d_state_q, d_k_weighted, d_v_state = _scan_state_cube0(
            d_out, state, q_exp, v_new, k_weighted, d_state
        )
        d_v_new, d_v_prime = _scan_state_vec0(d_v_attn, d_v_state)
        d_value_wu[:, :, c] = d_v_new

        d_q_score, d_k_score, d_k_cumdecay_i, d_state_k = _scan_state_cube1(
            d_score, q_i, k_i, d_v_prime, state, k_cumdecay[:, :, c]
        )
        d_k_cumdecay[:, :, c] = d_k_cumdecay_i

        d_q_i, d_k_i, d_g_i, d_state = _scan_state_vec1(
            q_i, k_i, state, d_state, exp_g_i, exp_g_last, exp_delta,
            d_q_exp, d_state_q, d_k_weighted, d_q_score, d_k_score, d_state_k,
        )
        d_q[:, :, c] = d_q_i
        d_k[:, :, c] = d_k_i
        d_g_cumsum[:, :, c] = d_g_i

    return d_q, d_k, d_g_cumsum, d_value_wu, d_k_cumdecay


def scan_local_pre_bwd(query, key, grad_output, g_cumsum, decay_mask, state_after_history, v_new_history):
    B, H, C, _, _ = query.shape
    q = bf16_cube_operand(query)
    k = bf16_cube_operand(key)
    d_core = bf16_cube_operand(grad_output)
    g_cumsum = g_cumsum.float()
    decay_mask = decay_mask.float()
    state_after_history = bf16_cube_operand(state_after_history)
    v_new_history = bf16_cube_operand(v_new_history)

    d_q = torch.zeros_like(q)
    d_k_score = torch.zeros_like(k)
    d_g_q = torch.zeros_like(g_cumsum)
    d_state_q = torch.zeros(B, H, C, D, D, dtype=torch.float32, device=query.device)
    d_v_attn_tmp = torch.zeros(B, H, C, L, D, dtype=torch.float32, device=query.device)
    d_decay_mask = torch.zeros_like(decay_mask)

    final_state = state_after_history[:, :, C - 1]
    for c in range(C):
        state = torch.zeros_like(final_state) if c == 0 else state_after_history[:, :, c - 1]
        q_i = q[:, :, c]
        k_i = k[:, :, c]
        d_out = d_core[:, :, c]
        g_i = g_cumsum[:, :, c]
        exp_g_i, q_exp, _, _ = _scan_state_vec_prelude(q_i, g_i)
        score_qk = bf16_cube_mm(q_i, k_i.transpose(-1, -2))
        v_new = v_new_history[:, :, c]
        d_attn = bf16_cube_mm(d_out, v_new.transpose(-1, -2))
        attn = score_qk * decay_mask[:, :, c]
        d_score = d_attn * decay_mask[:, :, c]

        d_q_exp = bf16_cube_mm(d_out, state.transpose(-1, -2))
        d_q_score = bf16_cube_mm(d_score, k_i)
        d_k_score[:, :, c] = bf16_cube_mm(d_score.transpose(-1, -2), q_i)
        d_q[:, :, c] = d_q_score + d_q_exp * exp_g_i.unsqueeze(-1)
        d_g_q[:, :, c] = (d_q_exp * q_i).sum(dim=-1) * exp_g_i
        if c != 0:
            d_state_q[:, :, c] = bf16_cube_mm(q_exp.transpose(-1, -2), d_out)

        d_v_attn_tmp[:, :, c] = bf16_cube_mm(attn.transpose(-1, -2), d_out)
        d_decay_mask[:, :, c] = d_attn * score_qk

    return d_q, d_k_score, d_g_q, d_state_q, d_v_attn_tmp, d_decay_mask


def scan_pre_bwd(query, key, grad_output, g_cumsum, state_after_history, d_score_tmp):
    B, H, C, _, _ = query.shape
    q = bf16_cube_operand(query)
    k = bf16_cube_operand(key)
    d_core = bf16_cube_operand(grad_output)
    g_cumsum = g_cumsum.float()
    state_after_history = bf16_cube_operand(state_after_history)
    d_score_tmp = d_score_tmp.float()

    d_q = torch.zeros_like(q)
    d_k_score = torch.zeros_like(k)
    d_g_q = torch.zeros_like(g_cumsum)
    d_state_q = torch.zeros(B, H, C, D, D, dtype=torch.float32, device=query.device)

    final_state = state_after_history[:, :, C - 1]
    for c in range(C):
        state = torch.zeros_like(final_state) if c == 0 else state_after_history[:, :, c - 1]
        q_i = q[:, :, c]
        k_i = k[:, :, c]
        d_out = d_core[:, :, c]
        g_i = g_cumsum[:, :, c]
        exp_g_i, q_exp, _, _ = _scan_state_vec_prelude(q_i, g_i)
        d_score = d_score_tmp[:, :, c]

        d_q_exp = bf16_cube_mm(d_out, state.transpose(-1, -2))
        d_q_score = bf16_cube_mm(d_score, k_i)
        d_k_score[:, :, c] = bf16_cube_mm(d_score.transpose(-1, -2), q_i)
        d_q[:, :, c] = d_q_score + d_q_exp * exp_g_i.unsqueeze(-1)
        d_g_q[:, :, c] = (d_q_exp * q_i).sum(dim=-1) * exp_g_i
        if c != 0:
            d_state_q[:, :, c] = bf16_cube_mm(q_exp.transpose(-1, -2), d_out)

    return d_q, d_k_score, d_g_q, d_state_q


def scan_state_recur_bwd(key, g_cumsum, k_cumdecay, state_after_history, grad_final_state, v_new_history, k_weighted_history, d_v_attn_tmp, d_k_score_tmp, d_g_q_tmp, d_state_q_tmp, exp_delta_history=None):
    B, H, C, _, _ = key.shape
    k = bf16_cube_operand(key)
    g_cumsum = g_cumsum.float()
    k_cumdecay = bf16_cube_operand(k_cumdecay)
    state_after_history = bf16_cube_operand(state_after_history)
    v_new_history = bf16_cube_operand(v_new_history)
    k_weighted_history = bf16_cube_operand(k_weighted_history)
    exp_delta_history = None if exp_delta_history is None else exp_delta_history.float()
    final_state = state_after_history[:, :, C - 1]
    d_state = torch.zeros_like(final_state) if grad_final_state is None else grad_final_state.float()

    d_k = torch.zeros_like(k)
    d_g_cumsum = torch.zeros_like(g_cumsum)
    d_value_wu = torch.zeros(B, H, C, L, D, dtype=torch.bfloat16, device=key.device)
    d_k_cumdecay = torch.zeros(B, H, C, L, D, dtype=torch.bfloat16, device=key.device)

    for c in range(C - 1, -1, -1):
        state = torch.zeros_like(final_state) if c == 0 else state_after_history[:, :, c - 1]
        k_i = k[:, :, c]
        g_i = g_cumsum[:, :, c]
        exp_g_last = g_i[:, :, -1].exp()
        exp_delta = exp_delta_history[:, :, c] if exp_delta_history is not None else (g_i[:, :, -1].unsqueeze(-1) - g_i).exp()
        k_weighted = k_weighted_history[:, :, c]
        v_new = v_new_history[:, :, c]
        d_v_attn = d_v_attn_tmp[:, :, c].float()
        d_k_score = d_k_score_tmp[:, :, c].float()
        d_g_i = d_g_q_tmp[:, :, c].float()
        d_state_q = d_state_q_tmp[:, :, c].float()

        d_exp_g_last = (d_state * state).sum(dim=(-1, -2))
        d_k_weighted = bf16_cube_mm(v_new, d_state.transpose(-1, -2))
        d_v_state = bf16_cube_mm(k_weighted, d_state)
        d_v_new = d_v_attn + d_v_state
        d_value_wu[:, :, c] = d_v_new

        d_v_prime = -d_v_new
        d_k_cumdecay[:, :, c] = bf16_cube_mm(d_v_prime, state.transpose(-1, -2))
        d_state_k = bf16_cube_mm(k_cumdecay[:, :, c].transpose(-1, -2), d_v_prime)

        d_k[:, :, c] = d_k_score + d_k_weighted * exp_delta.unsqueeze(-1)
        d_delta = (d_k_weighted * k_i).sum(dim=-1) * exp_delta
        d_g_i = d_g_i - d_delta
        d_g_i[:, :, -1] = d_g_i[:, :, -1] + d_exp_g_last * exp_g_last + d_delta.sum(dim=-1)
        d_g_cumsum[:, :, c] = d_g_i
        d_state = d_state_q + d_state * exp_g_last[:, :, None, None]
        d_state = d_state + d_state_k

    return d_k, d_g_cumsum, d_value_wu, d_k_cumdecay


def scan_split_bwd(query, key, grad_output, g_cumsum, decay_mask, k_cumdecay, state_after_history, v_new_history, k_weighted_history, grad_final_state=None, exp_delta_history=None):
    local = scan_local_bwd(query, key, grad_output, decay_mask, v_new_history)
    d_score, d_v_attn_tmp, d_decay_mask, d_decay_masked = local
    d_q, d_k, d_g_cumsum, d_value_wu, d_k_cumdecay = scan_state_bwd(
        query, key, grad_output, g_cumsum, k_cumdecay, state_after_history,
        grad_final_state, d_score, v_new_history, k_weighted_history,
        d_v_attn_tmp, exp_delta_history,
    )
    return d_q, d_k, d_g_cumsum, d_decay_mask, d_value_wu, d_k_cumdecay, d_decay_masked


def wu_recompute_bwd(key, value, beta, g_cumsum, wu_attn_bf16, d_value_wu, d_k_cumdecay):
    k = bf16_cube_operand(key)
    v = bf16_cube_operand(value)
    beta_f = beta.float()
    g_cumsum = g_cumsum.float()
    wu_attn = bf16_cube_operand(wu_attn_bf16)
    d_value_wu = d_value_wu.float()
    d_k_cumdecay = d_k_cumdecay.float()

    exp_g_cumsum = g_cumsum.exp()
    k_beta = k * beta_f.unsqueeze(-1)
    v_beta = v * beta_f.unsqueeze(-1)
    k_beta_g = k_beta * exp_g_cumsum.unsqueeze(-1)

    d_wu = bf16_cube_mm(d_value_wu, v_beta.transpose(-1, -2)) + bf16_cube_mm(d_k_cumdecay, k_beta_g.transpose(-1, -2))
    d_wu = d_wu.to(torch.bfloat16)
    d_v_beta = bf16_cube_mm(wu_attn.transpose(-1, -2), d_value_wu)
    d_k_beta_g = bf16_cube_mm(wu_attn.transpose(-1, -2), d_k_cumdecay)
    d_k_beta = d_k_beta_g * exp_g_cumsum.unsqueeze(-1)
    d_g_cumsum = (d_k_beta_g * k_beta).sum(dim=-1) * exp_g_cumsum
    return d_wu, d_v_beta, d_k_beta, d_g_cumsum


def inverse_preprocess_bwd(key, beta, decay_mask, wu_attn_bf16, d_wu):
    k = bf16_cube_operand(key)
    beta_f = beta.float()
    decay_mask = decay_mask.float()
    wu_attn = bf16_cube_operand(wu_attn_bf16)
    d_wu = d_wu.float()
    k_beta = k * beta_f.unsqueeze(-1)
    preprocess_score = bf16_cube_mm(k_beta, k.transpose(-1, -2))

    d_attn_pre = bf16_cube_mm(bf16_cube_mm(wu_attn.transpose(-1, -2), d_wu), wu_attn.transpose(-1, -2))
    d_score_pre = -d_attn_pre * decay_mask * strict_lower(key.device)
    d_decay_mask = -d_attn_pre * preprocess_score * strict_lower(key.device)
    d_decay_masked = d_decay_mask * decay_mask
    d_k_beta = bf16_cube_mm(d_score_pre, k)
    d_k = bf16_cube_mm(d_score_pre.transpose(-1, -2), k_beta)
    return d_k, d_k_beta, d_decay_mask, d_decay_masked


def finalize_grads(key, value, beta, decay_mask, d_q, d_k_core, d_g_core, d_decay_core, d_v_beta, d_k_beta_wu, d_g_wu, d_k_pre, d_k_beta_pre, d_decay_pre):
    k = f32(key)
    v = f32(value)
    beta_f = beta.float()
    decay_mask = decay_mask.float()
    d_k = d_k_core.float() + d_k_pre.float()
    d_g_cumsum = d_g_core.float() + d_g_wu.float()
    d_decay_mask = d_decay_core.float() + d_decay_pre.float()
    d_k_beta = d_k_beta_wu.float() + d_k_beta_pre.float()
    d_v_beta = d_v_beta.float()

    d_g_cumsum = d_g_cumsum + (d_decay_mask * decay_mask).sum(dim=-1) - (d_decay_mask * decay_mask).sum(dim=-2)
    d_v = d_v_beta * beta_f.unsqueeze(-1)
    d_beta = (d_v_beta * v).sum(dim=-1)
    d_k = d_k + d_k_beta * beta_f.unsqueeze(-1)
    d_beta = d_beta + (d_k_beta * k).sum(dim=-1)
    return d_q.float(), d_k, d_v, d_beta, reverse_cumsum(d_g_cumsum)


def tail_fused_bwd(key, value, beta, g_cumsum, decay_mask, wu_attn_bf16, scan_outputs):
    d_q, d_k_core, d_g_core, d_decay_core, d_value_wu, d_k_cumdecay = scan_outputs[:6]
    d_wu, d_v_beta, d_k_beta_wu, d_g_wu = wu_recompute_bwd(
        key, value, beta, g_cumsum, wu_attn_bf16, d_value_wu, d_k_cumdecay
    )
    d_k_pre, d_k_beta_pre, d_decay_pre, _d_decay_pre_masked = inverse_preprocess_bwd(key, beta, decay_mask, wu_attn_bf16, d_wu)
    return finalize_grads(
        key, value, beta, decay_mask,
        d_q, d_k_core, d_g_core, d_decay_core,
        d_v_beta, d_k_beta_wu, d_g_wu,
        d_k_pre, d_k_beta_pre, d_decay_pre,
    )


def factor_pack(key, value, beta, g_cumsum, wu_attn_bf16):
    k = bf16_cube_operand(key)
    v = bf16_cube_operand(value)
    beta_f = beta.float()
    exp_g_cumsum = g_cumsum.float().exp()
    k_beta = k * beta_f.unsqueeze(-1)
    return (
        bf16_cube_operand(wu_attn_bf16),
        v * beta_f.unsqueeze(-1),
        k_beta,
        k_beta * exp_g_cumsum.unsqueeze(-1),
        bf16_cube_mm(k_beta, k.transpose(-1, -2)),
    )


def tail_from_factors_exact(key, value, beta, g_cumsum, decay_mask, factors, scan_outputs):
    wu_attn, v_beta, k_beta, k_beta_g, preprocess_score = factors
    d_q, d_k_core, d_g_core, d_decay_core, d_value_wu, d_k_cumdecay = scan_outputs[:6]
    k = bf16_cube_operand(key)
    exp_g_cumsum = g_cumsum.float().exp()
    d_wu = bf16_cube_mm(d_value_wu, v_beta.transpose(-1, -2)) + bf16_cube_mm(d_k_cumdecay, k_beta_g.transpose(-1, -2))
    d_wu = d_wu.to(torch.bfloat16)
    d_v_beta = bf16_cube_mm(wu_attn.transpose(-1, -2), d_value_wu)
    d_k_beta_g = bf16_cube_mm(wu_attn.transpose(-1, -2), d_k_cumdecay)
    d_k_beta_wu = d_k_beta_g * exp_g_cumsum.unsqueeze(-1)
    d_g_wu = (d_k_beta_g * k_beta).sum(dim=-1) * exp_g_cumsum
    d_attn_pre = bf16_cube_mm(bf16_cube_mm(wu_attn.transpose(-1, -2), d_wu), wu_attn.transpose(-1, -2))
    d_score_pre = -d_attn_pre * decay_mask.float() * strict_lower(key.device)
    d_decay_pre = -d_attn_pre * preprocess_score * strict_lower(key.device)
    d_k_beta_pre = bf16_cube_mm(d_score_pre, k)
    d_k_pre = bf16_cube_mm(d_score_pre.transpose(-1, -2), k_beta)
    return finalize_grads(
        key, value, beta, decay_mask,
        d_q, d_k_core, d_g_core, d_decay_core,
        d_v_beta, d_k_beta_wu, d_g_wu,
        d_k_pre, d_k_beta_pre, d_decay_pre,
    )


def oracle(*args):
    (
        query, key, value, beta, _g, grad_output,
        g_cumsum, decay_mask, wu_attn_bf16, _value_wu, k_cumdecay,
        state_after_history, v_new_history, k_weighted_history, grad_final_state,
        *extra_saved,
    ) = args
    exp_delta_history = extra_saved[0] if extra_saved else None
    scan = core_scan_bwd(
        query, key, grad_output, g_cumsum, decay_mask,
        k_cumdecay, state_after_history, v_new_history, k_weighted_history,
        grad_final_state, exp_delta_history,
    )
    return tail_fused_bwd(key, value, beta, g_cumsum, decay_mask, wu_attn_bf16, scan)


# ----------------------------------------------------------------------------------------------------
# reference.py
# Independent backward references with project-local saved-state generation.
# ----------------------------------------------------------------------------------------------------

INPUTS = ("query", "key", "value", "beta", "g", "grad_output", "g_cumsum", "decay_mask", "wu_attn_bf16", "value_wu", "k_cumdecay",
          "state_after_history", "v_new_history", "k_weighted_history", "grad_final_state", "exp_delta_history")
OUTPUTS = ("d_query", "d_key", "d_value", "d_beta", "d_g")
STAGES = {"local": ("d_score_tmp", "d_v_attn_tmp", "d_decay_core", "d_decay_core_masked"),
          "state": ("d_q_core", "d_k_core", "d_g_core", "d_value_wu", "d_k_cumdecay"),
          "wu": ("d_wu", "d_v_beta", "d_k_beta_wu", "d_g_wu"),
          "inverse": ("d_k_pre", "d_k_beta_pre", "d_decay_pre", "d_decay_pre_masked"),
          "finalize": ("d_key", "d_value", "d_beta", "d_g")}


def make_inputs(case):
    p = case["parameters"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(case["seed"])
        values = dict(zip(INPUTS, generate_inputs(p["B"], p["H"], p["C"], scale=p.get("scale", 0.05))))
        if p.get("gate") == "slow":
            values["g"] = -torch.rand_like(values["g"]) * 0.03
            _prepare_saved(values)
        if p.get("gradient") == "zero":
            values["grad_output"].zero_()
            values["grad_final_state"].zero_()
    return {name: value.contiguous() for name, value in values.items()}


def _prepare_saved(values):
    # Preserve backward source ABI: beta is rounded into the BF16 cube operand before preprocessing.
    k, v, beta, g = (values[name] for name in ("key", "value", "beta", "g"))
    gc = g.cumsum(-1)
    decay = (gc[..., :, None] - gc[..., None, :]).exp() * torch.ones(64, 64).tril()
    kbeta = k.float() * beta[..., None]
    strict = -(bf16_cube_mm(kbeta, k.float().transpose(-1, -2)) * decay * torch.ones(64, 64).tril(-1))
    wu = torch.linalg.inv(torch.eye(64) - strict).bfloat16()
    value_wu = bf16_cube_mm(wu, v.float() * beta[..., None]).bfloat16()
    key_decay = bf16_cube_mm(wu, kbeta * gc.exp()[..., None]).bfloat16()
    saved = state_history_forward(k, gc, value_wu, key_decay)
    values.update(g_cumsum=gc, decay_mask=decay, wu_attn_bf16=wu, value_wu=value_wu, k_cumdecay=key_decay,
                  state_after_history=saved[0], v_new_history=saved[1], k_weighted_history=saved[2], exp_delta_history=saved[3])


def reference(inputs):
    return dict(zip(OUTPUTS, oracle(*(inputs[name] for name in INPUTS))))


def reference_stages(inputs):
    q, k, v, b, gc = (inputs[name] for name in ("query", "key", "value", "beta", "g_cumsum"))
    local = scan_local_bwd(q, k, inputs["grad_output"], inputs["decay_mask"], inputs["v_new_history"])
    state = scan_state_bwd(q, k, inputs["grad_output"], gc, inputs["k_cumdecay"], inputs["state_after_history"], inputs["grad_final_state"],
                           local[0], inputs["v_new_history"], inputs["k_weighted_history"], local[1], inputs["exp_delta_history"])
    wu = wu_recompute_bwd(k, v, b, gc, inputs["wu_attn_bf16"], state[3], state[4])
    inv = inverse_preprocess_bwd(k, b, inputs["decay_mask"], inputs["wu_attn_bf16"], wu[0])
    final = finalize_grads(k, v, b, inputs["decay_mask"], state[0], state[1], state[2], local[2], wu[1], wu[2], wu[3], inv[0], inv[1], inv[2])[1:]
    return {name: value for stage, values in (("local", local), ("state", state), ("wu", wu), ("inverse", inv), ("finalize", final))
            for name, value in zip(STAGES[stage], values)}


