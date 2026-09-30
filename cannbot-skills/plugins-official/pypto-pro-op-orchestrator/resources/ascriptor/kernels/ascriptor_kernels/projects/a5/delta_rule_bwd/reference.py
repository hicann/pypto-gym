# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent Torch reference for the Delta Rule backward: the staged decomposition each
kernel stage answers to, and the saved forward state the backward consumes."""

import builtins
from typing import Optional, Tuple

import torch

# ----------------------------------------------------------------------------------------------------
# stages.py
# Independent BF16-cube backward formulas and saved-state recurrence.
#
# Adapted from projects/a5/delta_rule_bwd/refs/common.py; pure Torch, no DSL imports.
# ----------------------------------------------------------------------------------------------------

L = 64
D = 128
SIZE = 64
BLOCK = 16
NUM_BLOCKS = 4
ATOL = 2e-3
RTOL = 2e-3

def strict_lower(device):
    return torch.tril(torch.ones(L, L, dtype=torch.float32, device=device), diagonal=-1)

def lower_eq(device):
    """The intra-chunk causal keep: GDN's `decay_mask` at g == 0."""
    return torch.tril(torch.ones(L, L, dtype=torch.float32, device=device))

def f32(x):
    return x if x.dtype == torch.float32 else x.float()

def bf16_cube_operand(x):
    return x.to(torch.bfloat16).float()

def bf16_cube_mm(lhs, rhs):
    """A5 cube contract for this plan: bf16 inputs, fp32 accumulation/output."""
    return bf16_cube_operand(lhs) @ bf16_cube_operand(rhs)

def state_history_forward(key, value_wu, k_cumdecay):
    """Per-chunk post-update state and `v_new`; gdn's at g == 0 (no decay, k_weighted == key)."""
    B, H, C, _, _ = key.shape
    state = torch.zeros(B, H, D, D, dtype=torch.bfloat16, device=key.device)
    states = []
    v_news = []
    for c in range(C):
        k_i = key[:, :, c].float()
        v_new = (value_wu[:, :, c].float() - bf16_cube_mm(k_cumdecay[:, :, c], state)).to(torch.bfloat16)
        v_news.append(v_new)
        state = (state.float() + bf16_cube_mm(k_i.transpose(-1, -2), v_new)).to(torch.bfloat16)
        states.append(state)
    return torch.stack(states, dim=2), torch.stack(v_news, dim=2)

def core_scan_bwd(query, key, grad_output, k_cumdecay, state_after_history, v_new_history, grad_final_state=None):
    """Monolithic reverse-C scan backward (gdn's `core_scan_bwd` at g == 0)."""
    B, H, C, _, _ = query.shape
    q = bf16_cube_operand(query)
    k = bf16_cube_operand(key)
    k_cumdecay = bf16_cube_operand(k_cumdecay)
    state_after_history = bf16_cube_operand(state_after_history)
    v_new_history = bf16_cube_operand(v_new_history)
    causal = lower_eq(query.device)
    final_state = state_after_history[:, :, C - 1]
    d_state = torch.zeros_like(final_state) if grad_final_state is None else grad_final_state.float()
    d_core = bf16_cube_operand(grad_output)

    d_q = torch.zeros_like(q)
    d_k = torch.zeros_like(k)
    d_value_wu = torch.zeros(B, H, C, L, D, dtype=torch.bfloat16, device=query.device)
    d_k_cumdecay = torch.zeros(B, H, C, L, D, dtype=torch.bfloat16, device=query.device)

    for c in range(C - 1, -1, -1):
        state = torch.zeros_like(final_state) if c == 0 else state_after_history[:, :, c - 1]
        q_i = q[:, :, c]
        k_i = k[:, :, c]
        d_out = d_core[:, :, c]
        score_qk = bf16_cube_mm(q_i, k_i.transpose(-1, -2))
        attn = score_qk * causal
        v_new = v_new_history[:, :, c]

        d_q_inter = bf16_cube_mm(d_out, state.transpose(-1, -2))
        d_state_in = bf16_cube_mm(q_i.transpose(-1, -2), d_out) + d_state  # no exp(g_last) decay
        d_attn = bf16_cube_mm(d_out, v_new.transpose(-1, -2))
        d_v_new = bf16_cube_mm(attn.transpose(-1, -2), d_out)
        d_k_from_state = bf16_cube_mm(v_new, d_state.transpose(-1, -2))
        d_v_new = d_v_new + bf16_cube_mm(k_i, d_state)  # k_weighted == k_i

        d_value_wu[:, :, c] = d_value_wu[:, :, c] + d_v_new
        d_v_prime = -d_v_new
        d_k_cumdecay[:, :, c] = d_k_cumdecay[:, :, c] + bf16_cube_mm(d_v_prime, state.transpose(-1, -2))
        d_state_in = d_state_in + bf16_cube_mm(k_cumdecay[:, :, c].transpose(-1, -2), d_v_prime)

        d_score_qk = d_attn * causal
        d_q[:, :, c] = d_q[:, :, c] + bf16_cube_mm(d_score_qk, k_i) + d_q_inter
        d_k[:, :, c] = d_k[:, :, c] + bf16_cube_mm(d_score_qk.transpose(-1, -2), q_i)
        d_k[:, :, c] = d_k[:, :, c] + d_k_from_state  # exp_delta == 1
        d_state = d_state_in

    return d_q, d_k, d_value_wu, d_k_cumdecay

def scan_local_bwd(query, key, grad_output, v_new_history):
    """BHC-parallel local products (gdn's `scan_local_bwd` at g == 0).

    Drops the `d_decay_mask` / `d_decay_masked` outputs (gate-gradient only).
    """
    B, H, C, _, _ = query.shape
    q = bf16_cube_operand(query)
    k = bf16_cube_operand(key)
    d_core = bf16_cube_operand(grad_output)
    v_new_history = bf16_cube_operand(v_new_history)
    causal = lower_eq(query.device)

    d_score_tmp = torch.zeros(B, H, C, L, L, dtype=torch.bfloat16, device=query.device)
    d_v_attn_tmp = torch.zeros(B, H, C, L, D, dtype=torch.float32, device=query.device)

    for c in range(C):
        q_i = q[:, :, c]
        k_i = k[:, :, c]
        d_out = d_core[:, :, c]
        score_qk = bf16_cube_mm(q_i, k_i.transpose(-1, -2))
        v_new = v_new_history[:, :, c]
        d_attn = bf16_cube_mm(d_out, v_new.transpose(-1, -2))
        attn = score_qk * causal

        d_score_tmp[:, :, c] = (d_attn * causal).to(torch.bfloat16)
        d_v_attn_tmp[:, :, c] = bf16_cube_mm(attn.transpose(-1, -2), d_out)

    return d_score_tmp, d_v_attn_tmp

def scan_state_bwd(query, key, grad_output, k_cumdecay, state_after_history, grad_final_state, d_score_tmp, v_new_history, d_v_attn_tmp):
    """BH-parallel reverse-C state bridge (gdn's `scan_state_bwd` at g == 0).

    Consumes `key` directly (k_weighted == key), drops `g_cumsum`/`exp_delta`,
    and emits no `d_g`.  Arranged around the same cube/vec/cube/vec body.
    """
    B, H, C, _, _ = query.shape
    q = bf16_cube_operand(query)
    k = bf16_cube_operand(key)
    k_cumdecay = bf16_cube_operand(k_cumdecay)
    state_after_history = bf16_cube_operand(state_after_history)
    v_new_history = bf16_cube_operand(v_new_history)
    final_state = state_after_history[:, :, C - 1]
    d_state = torch.zeros_like(final_state) if grad_final_state is None else grad_final_state.float()
    d_core = bf16_cube_operand(grad_output)

    d_q = torch.zeros_like(q)
    d_k = torch.zeros_like(k)
    d_value_wu = torch.zeros(B, H, C, L, D, dtype=torch.bfloat16, device=query.device)
    d_k_cumdecay = torch.zeros(B, H, C, L, D, dtype=torch.bfloat16, device=query.device)

    for c in range(C - 1, -1, -1):
        state = torch.zeros_like(final_state) if c == 0 else state_after_history[:, :, c - 1]
        q_i = q[:, :, c]
        k_i = k[:, :, c]
        d_out = d_core[:, :, c]
        d_score = d_score_tmp[:, :, c].float()
        v_new = v_new_history[:, :, c]
        d_v_attn = d_v_attn_tmp[:, :, c].float()

        # cube0: products that do not need d_v_prime
        d_q_inter = bf16_cube_mm(d_out, state.transpose(-1, -2))
        d_state_q = bf16_cube_mm(q_i.transpose(-1, -2), d_out)   # q_exp == q_i
        d_k_from_state = bf16_cube_mm(v_new, d_state.transpose(-1, -2))
        d_v_state = bf16_cube_mm(k_i, d_state)                   # k_weighted == k_i
        # vec0: expose d_v_prime
        d_v_new = d_v_attn + d_v_state
        d_value_wu[:, :, c] = d_v_new
        d_v_prime = -d_v_new
        # cube1: score products + d_v_prime-dependent products
        d_q_score = bf16_cube_mm(d_score, k_i)
        d_k_score = bf16_cube_mm(d_score.transpose(-1, -2), q_i)
        d_k_cumdecay[:, :, c] = bf16_cube_mm(d_v_prime, state.transpose(-1, -2))
        d_state_k = bf16_cube_mm(k_cumdecay[:, :, c].transpose(-1, -2), d_v_prime)
        # vec1: accumulate public grads and the recurrent d_state
        d_q[:, :, c] = d_q_score + d_q_inter
        d_k[:, :, c] = d_k_score + d_k_from_state
        d_state = d_state_q + d_state + d_state_k               # no exp(g_last) decay

    return d_q, d_k, d_value_wu, d_k_cumdecay

def scan_split_bwd(query, key, grad_output, k_cumdecay, state_after_history, v_new_history, grad_final_state=None):
    """Wire the BHC-parallel local stage into the BH-parallel state stage."""
    d_score, d_v_attn_tmp = scan_local_bwd(query, key, grad_output, v_new_history)
    d_q, d_k, d_value_wu, d_k_cumdecay = scan_state_bwd(
        query, key, grad_output, k_cumdecay, state_after_history,
        grad_final_state, d_score, v_new_history, d_v_attn_tmp,
    )
    return d_q, d_k, d_value_wu, d_k_cumdecay

def wu_recompute_bwd(key, value, beta, wu_attn_bf16, d_value_wu, d_k_cumdecay):
    """WY-representation backward (gdn's `wu_recompute_bwd` at g == 0).

    Drops the `exp(g_cumsum)` weight on the key path and the `d_g_cumsum` output.
    """
    k = bf16_cube_operand(key)
    v = bf16_cube_operand(value)
    beta_f = beta.float()
    wu_attn = bf16_cube_operand(wu_attn_bf16)
    d_value_wu = d_value_wu.float()
    d_k_cumdecay = d_k_cumdecay.float()

    k_beta = k * beta_f.unsqueeze(-1)
    v_beta = v * beta_f.unsqueeze(-1)

    d_wu = bf16_cube_mm(d_value_wu, v_beta.transpose(-1, -2)) + bf16_cube_mm(d_k_cumdecay, k_beta.transpose(-1, -2))
    d_wu = d_wu.to(torch.bfloat16)
    d_v_beta = bf16_cube_mm(wu_attn.transpose(-1, -2), d_value_wu)
    d_k_beta = bf16_cube_mm(wu_attn.transpose(-1, -2), d_k_cumdecay)  # no exp(g_cumsum) factor
    return d_wu, d_v_beta, d_k_beta

def inverse_preprocess_bwd(key, beta, wu_attn_bf16, d_wu):
    """Inverse block backward `dA = W.T @ dW @ W.T` (gdn's at g == 0).

    `decay_mask * strict_lower` collapses to `strict_lower`; drops `d_decay_mask`
    and the `preprocess_score` it needed.
    """
    k = bf16_cube_operand(key)
    beta_f = beta.float()
    wu_attn = bf16_cube_operand(wu_attn_bf16)
    d_wu = d_wu.float()
    k_beta = k * beta_f.unsqueeze(-1)

    d_attn_pre = bf16_cube_mm(bf16_cube_mm(wu_attn.transpose(-1, -2), d_wu), wu_attn.transpose(-1, -2))
    d_score_pre = -d_attn_pre * strict_lower(key.device)
    # d_k (d_k_pre) and d_k_beta (d_k_beta_pre) are bf16 GM bridges to finalize_bwd
    # (the fixpipe casts fp32 L0C -> bf16 on store); model that bf16 rounding here.
    d_k_beta = bf16_cube_mm(d_score_pre, k).to(torch.bfloat16)
    d_k = bf16_cube_mm(d_score_pre.transpose(-1, -2), k_beta).to(torch.bfloat16)
    return d_k, d_k_beta

def finalize_grads(key, value, beta, d_q, d_k_core, d_v_beta, d_k_beta_wu, d_k_pre, d_k_beta_pre):
    """Final gradient accumulation (gdn's `finalize_grads` at g == 0).

    No `decay_mask`, no `d_g` / reverse-cumsum path.
    """
    k = f32(key)
    v = f32(value)
    beta_f = beta.float()
    d_k = d_k_core.float() + d_k_pre.float()
    d_k_beta = d_k_beta_wu.float() + d_k_beta_pre.float()
    d_v_beta = d_v_beta.float()

    d_v = d_v_beta * beta_f.unsqueeze(-1)
    d_beta = (d_v_beta * v).sum(dim=-1)
    d_k = d_k + d_k_beta * beta_f.unsqueeze(-1)
    d_beta = d_beta + (d_k_beta * k).sum(dim=-1)
    return d_q.float(), d_k, d_v, d_beta

def tail_fused_bwd(key, value, beta, wu_attn_bf16, scan_outputs):
    d_q, d_k_core, d_value_wu, d_k_cumdecay = scan_outputs[:4]
    d_wu, d_v_beta, d_k_beta_wu = wu_recompute_bwd(key, value, beta, wu_attn_bf16, d_value_wu, d_k_cumdecay)
    d_k_pre, d_k_beta_pre = inverse_preprocess_bwd(key, beta, wu_attn_bf16, d_wu)
    return finalize_grads(key, value, beta, d_q, d_k_core, d_v_beta, d_k_beta_wu, d_k_pre, d_k_beta_pre)

def oracle(*args):
    (
        query, key, value, beta, grad_output,
        wu_attn_bf16, _value_wu, k_cumdecay,
        state_after_history, v_new_history, grad_final_state,
    ) = args
    scan = core_scan_bwd(
        query, key, grad_output, k_cumdecay, state_after_history,
        v_new_history, grad_final_state,
    )
    return tail_fused_bwd(key, value, beta, wu_attn_bf16, scan)

# ----------------------------------------------------------------------------------------------------
# reference.py
# Self-contained backward state preparation and independent gradient references.
#
# The source backward training fixture rounds key*beta BEFORE preprocess matmul.
# This is an explicit precision variant; it is not the forward default's post-MM beta.
# ----------------------------------------------------------------------------------------------------

L, D = 64, 128
INPUT_ORDER = ("query", "key", "value", "beta", "grad_output", "wu_attn_bf16", "value_wu",
               "k_cumdecay", "state_after_history", "v_new_history", "grad_final_state")
OUTPUTS = ("d_query", "d_key", "d_value", "d_beta")


def prepare_saved(key: torch.Tensor, value: torch.Tensor, beta: torch.Tensor) -> dict:
    k_beta = key.float() * beta.unsqueeze(-1)
    a = -(bf16_cube_mm(k_beta, key.float().transpose(-1, -2)) * strict_lower(key.device))
    w = torch.linalg.inv(torch.eye(L) - a).bfloat16().contiguous()
    value_wu = bf16_cube_mm(w, value.float() * beta.unsqueeze(-1)).bfloat16()
    key_wu = bf16_cube_mm(w, k_beta).bfloat16()
    history, v_new = state_history_forward(key, value_wu, key_wu)
    return {"wu_attn_bf16": w, "value_wu": value_wu, "k_cumdecay": key_wu,
            "state_after_history": history, "v_new_history": v_new}


def make_inputs(case: dict) -> dict:
    p = case["parameters"]
    shape = (p["B"], p["H"], p["C"], L, D)
    rng = torch.Generator().manual_seed(case["seed"])
    result = {name: (torch.randn(shape, generator=rng) * p.get("input_scale", 0.05)).bfloat16()
              for name in ("query", "key", "value")}
    result["beta"] = torch.rand(shape[:-1], generator=rng)
    result["grad_output"] = (torch.randn(shape, generator=rng) * 0.1).bfloat16()
    result["grad_final_state"] = torch.randn(shape[:2] + (D, D), generator=rng) * p.get("final_gradient_scale", 0.1)
    result.update(prepare_saved(result["key"], result["value"], result["beta"]))
    return result




def reference(inputs: dict) -> dict:
    return dict(zip(OUTPUTS, oracle(*(inputs[name] for name in INPUT_ORDER)), strict=True))


def reference_stages(inputs: dict) -> dict:
    q, k, v, beta, do = (inputs[name] for name in ("query", "key", "value", "beta", "grad_output"))
    local = scan_local_bwd(q, k, do, inputs["v_new_history"])
    scan = scan_state_bwd(q, k, do, inputs["k_cumdecay"], inputs["state_after_history"],
                                inputs["grad_final_state"], local[0], inputs["v_new_history"], local[1])
    wu = wu_recompute_bwd(k, v, beta, inputs["wu_attn_bf16"], scan[2], scan[3])
    inverse = inverse_preprocess_bwd(k, beta, inputs["wu_attn_bf16"], wu[0])
    final = finalize_grads(k, v, beta, scan[0], scan[1], wu[1], wu[2], inverse[0], inverse[1])
    names = ("d_score_tmp", "d_v_attn_tmp", "d_q_core", "d_k_core", "d_value_wu", "d_k_cumdecay",
             "d_wu", "d_v_beta", "d_k_beta_wu", "d_k_pre", "d_k_beta_pre", *OUTPUTS)
    return dict(zip(names, (*local, *scan, *wu, *inverse, *final), strict=True))
