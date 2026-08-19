#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PyPTO engram golden reference implementation.

正向计算链（7 步，逐头 m∈[0,m_h)）:
    Step1  K^(m) = E @ W_k^(m)                         (linear, cube)
    Step2  K̂^(m) = RMSNorm(K^(m), γ_k^(m))             (rms_norm, vector)
    Step3  Q̂^(m) = RMSNorm(h^(m), γ_q^(m))             (rms_norm, vector)
    Step4  s^(m) = (1/√H) · Σ_h K̂_h · Q̂_h              (scaled_dot, vector reduction)
    Step5  g^(m) = σ(sign(s)·√max(|s|,c))              (signed_sqrt_gate, vector)
    Step6  V     = E @ W_v                              (linear, cube)
    Step7  O     = g · V                                (broadcast mul, vector)

反向沿 Step7→Step1 逆向求导，输出 6 个梯度：
    grad_hidden_states, grad_embeddings, grad_key_proj_weights,
    grad_value_proj_weights, grad_key_gamma, grad_query_gamma.

device 精度约定（对齐 kernel）：matmul 走 BF16 操作数（FP32 acc），vector 计算
（RMSNorm / gate / reduce）走 FP32；CPU 模式下一切沿用输入原 dtype。
本文件用 _v / _m 两个 helper 统一这套分派，避免每个算子重复 if is_npu 分支。
"""

import os
import logging

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

_DEVICE = None
_HAS_NPU = False

try:
    import torch_npu  # noqa: F401  (import side-effect: registers NPU backend)
    _HAS_NPU = torch.npu.is_available() and torch.npu.device_count() > 0
except ImportError as e:
    raise ImportError(
        "torch_npu is not installed. Please install it first:\n"
        "  pip install torch_npu\n"
        "Or use the pypto-environment-setup skill to set up the full NPU environment."
    ) from e


def _get_device() -> torch.device:
    """返回本进程使用的 device（卡号由 TILE_FWK_DEVICE_ID 指定，无 NPU 回退 CPU）。"""
    global _DEVICE
    if _DEVICE is None:
        if not _HAS_NPU:
            _DEVICE = torch.device("cpu")
        else:
            device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "4"))
            torch.npu.set_device(device_id)
            _DEVICE = torch.device(f"npu:{device_id}")
    return _DEVICE


# ── device 精度 helper ──────────────────────────────────────────────
# _v: vector 计算精度（NPU→FP32，CPU→原）；_m: matmul 操作数精度（NPU→BF16，CPU→原）
def _is_npu(t: torch.Tensor) -> bool:
    return t.device.type == "npu"


def _v(*tensors):
    """vector 计算精度：NPU 升 FP32，CPU 原样。单入参返回单 tensor，多入参返回元组。"""
    out = [t.float() if _is_npu(t) else t for t in tensors]
    return out[0] if len(out) == 1 else tuple(out)


def _m(*tensors):
    """matmul 操作数精度：NPU 降 BF16，CPU 原样。单入参返回单 tensor，多入参返回元组。"""
    out = [t.to(torch.bfloat16) if _is_npu(t) else t for t in tensors]
    return out[0] if len(out) == 1 else tuple(out)


# ==========================================================================
# Forward (including cache)
# ==========================================================================

def linear_torch(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """线性投影: y = x @ weight（无偏置）。NPU matmul 走 BF16（FP32 acc）。"""
    x, weight = _m(x, weight)
    return x @ weight


def rms_norm_torch(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm: x̂ = (x / rms) · γ，rms = √(mean(x², -1) + eps)。NPU 走 FP32。"""
    x, gamma = _v(x, gamma)
    d = x.shape[-1]
    rms = torch.sqrt(x.pow(2).sum(dim=-1, keepdim=True) / d + eps)
    return x / rms * gamma


def signed_sqrt_gate(score: torch.Tensor, clamp_value: float = 1e-6) -> torch.Tensor:
    """g = σ(sign(s)·√max(|s|, c))，返回 [b, s, 1]。NPU 走 FP32。"""
    s = _v(score)
    logits = torch.sign(s) * torch.sqrt(s.abs().clamp_min(clamp_value))
    return torch.sigmoid(logits).unsqueeze(-1)


def engram_forward_with_cache(
    hidden_states: torch.Tensor,
    embeddings: torch.Tensor,
    key_proj_weights: torch.Tensor,
    value_proj_weights: torch.Tensor,
    key_gamma: torch.Tensor,
    query_gamma: torch.Tensor,
    clamp_value: float = 1e-6,
    eps: float = 1e-6,
):
    """前向，返回 (value_out, cache)。cache 供 backward 消费。

    Returns:
        value_out: [b, s, m, h]  主输出（Step7: O = g · V）
        cache:     {keys [b,s,m,h], scores [b,s,m], gates [b,s,m], value [b,s,h]}
"""
    b, s, m, h = hidden_states.shape
    scale = h ** -0.5

    keys_list, scores_list, gates_list = [], [], []
    for m_i in range(m):
        key = linear_torch(embeddings, key_proj_weights[m_i])                       # Step1
        normed_key, _, _ = _rms_norm_fwd(key, key_gamma[m_i], eps)                  # Step2
        hidden_m = hidden_states[:, :, m_i, :]
        normed_query, _, _ = _rms_norm_fwd(hidden_m, query_gamma[m_i], eps)         # Step3
        score = (normed_key * normed_query).sum(dim=-1) * scale                     # Step4
        gate = signed_sqrt_gate(score, clamp_value)                                 # Step5
        keys_list.append(key)
        scores_list.append(score)
        gates_list.append(gate)

    keys = torch.stack(keys_list, dim=2)                  # [b, s, m, h]
    scores = torch.stack(scores_list, dim=2)              # [b, s, m]
    gates = torch.stack(gates_list, dim=2).squeeze(-1)    # [b, s, m]
    value = linear_torch(embeddings, value_proj_weights)  # Step6 [b, s, h]
    value_out = gates.unsqueeze(-1) * value.unsqueeze(2)  # Step7 [b, s, m, h]

    return value_out, {
        "keys": keys, "scores": scores, "gates": gates, "value": value,
    }


# ==========================================================================
# Backward
# ==========================================================================

def linear_backward(grad_output: torch.Tensor, x: torch.Tensor, weight: torch.Tensor):
    """linear_torch 反向: grad_x = grad_output @ weight^T; grad_weight = x^T @ grad_output。
    NPU matmul 走 BF16，结果落回 grad_output 原生 dtype。"""
    out_dt = grad_output.dtype
    go, ww, xx = _m(grad_output, weight, x)
    grad_x = F.linear(go, ww).to(out_dt)
    x_2d = xx.reshape(-1, xx.shape[-1])
    grad_2d = go.reshape(-1, go.shape[-1])
    grad_weight = (x_2d.T @ grad_2d).to(out_dt)
    return grad_x, grad_weight


def _rms_norm_fwd(x, gamma, eps=1e-6):
    """RMSNorm forward，返回 (normed, rms, n) — rms/n 供 backward 复用，避免重算。"""
    x, gamma = _v(x, gamma)
    d = x.shape[-1]
    rms = torch.sqrt(x.pow(2).sum(dim=-1, keepdim=True) / d + eps)
    n = x / rms
    return n * gamma, rms, n


def _rms_norm_bwd(grad_output, gamma, rms, n, out_dt):
    """RMSNorm backward，接受预计算 rms/n（来自 _rms_norm_fwd，避免重算）。"""
    gamma, grad_output = _v(gamma, grad_output)
    grad_n = grad_output * gamma
    d = grad_n.shape[-1]
    inner = (grad_n * n).sum(dim=-1, keepdim=True) / d
    grad_x = ((grad_n - n * inner) / rms).to(out_dt)
    grad_gamma = (grad_output * n).flatten(0, -2).sum(dim=0)
    return grad_x, grad_gamma


def signed_sqrt_gate_backward(grad_output: torch.Tensor, score: torch.Tensor,
                              gate: torch.Tensor, clamp_value: float = 1e-6):
    """signed_sqrt_gate 反向。NPU 走 FP32，grad_score 落点 FP32。"""
    gate, go, score = _v(gate.squeeze(-1), grad_output.squeeze(-1), score)
    sigmoid_grad = gate * (1.0 - gate)
    mask = torch.where(score.abs() > clamp_value,
                       torch.ones_like(score), torch.zeros_like(score))
    logits_grad = mask / (2.0 * score.abs().clamp_min(clamp_value).sqrt() + 1e-12)
    return go * sigmoid_grad * logits_grad


def engram_backward_golden(
    grad_output: torch.Tensor,      # [b, s, m, h]
    hidden_states: torch.Tensor,    # [b, s, m, h]
    embeddings: torch.Tensor,       # [b, s, de]
    key_proj_weights: torch.Tensor,  # [m, de, h]
    value_proj_weights: torch.Tensor,  # [de, h]
    key_gamma: torch.Tensor,        # [m, h]
    query_gamma: torch.Tensor,      # [m, h]
    scores: torch.Tensor,           # [b, s, m]
    gates: torch.Tensor,            # [b, s, m]
    keys: torch.Tensor,             # [b, s, m, h]
    value: torch.Tensor,            # [b, s, h]
    clamp_value: float = 1e-6,
    eps: float = 1e-6,
):
    """反向，返回 6 个梯度。NPU: vector 走 FP32、matmul 走 BF16、跨头累加用 FP32 acc。"""
    b, s, m, h = hidden_states.shape
    is_npu = _is_npu(hidden_states)
    out_dt = grad_output.dtype
    acc_dt = torch.float32 if is_npu else out_dt   # 跨头累加器 / gamma 梯度落点 dtype
    scale = h ** -0.5

    # ---- Step7 逆向: O = g · V ----
    gof, vf, gf = _v(grad_output, value, gates)
    grad_gates = (gof * vf.unsqueeze(2)).sum(dim=-1, keepdim=True)     # [b, s, m, 1]
    grad_value = (gof * gf.unsqueeze(-1)).sum(dim=2).to(out_dt)        # [b, s, h]

    # ---- Step6 逆向: V = E @ W_v ----
    grad_emb_from_v, grad_value_proj_weights = linear_backward(grad_value, embeddings, value_proj_weights)
    grad_embeddings = grad_emb_from_v.to(acc_dt).clone()               # 跨头 FP32 累加器（NPU）

    grad_hidden_states = torch.zeros_like(hidden_states)
    grad_key_proj_weights = torch.zeros_like(key_proj_weights)
    grad_key_gamma = torch.zeros_like(key_gamma, dtype=acc_dt)
    grad_query_gamma = torch.zeros_like(query_gamma, dtype=acc_dt)

    for m_i in range(m):
        # ---- Step5 逆向: signed_sqrt_gate ----
        grad_score_m = signed_sqrt_gate_backward(
            grad_gates[:, :, m_i, :], scores[:, :, m_i], gates[:, :, m_i], clamp_value)
        # ---- Step4/3/2: RMSNorm forward + backward ----
        hidden_m = hidden_states[:, :, m_i, :]
        key_m = keys[:, :, m_i, :]
        normed_key_m, key_rms_m, key_n = _rms_norm_fwd(key_m, key_gamma[m_i], eps)
        normed_query_m, query_rms_m, query_n = _rms_norm_fwd(hidden_m, query_gamma[m_i], eps)
        gsf = grad_score_m.unsqueeze(-1)
        grad_normed_key_m = gsf * scale * normed_query_m
        grad_normed_query_m = gsf * scale * normed_key_m
        # Step3: query RMSNorm backward（复用 query_rms/query_n）
        grad_hidden_m, grad_query_gamma_m = _rms_norm_bwd(
            grad_normed_query_m, query_gamma[m_i], query_rms_m, query_n, out_dt)
        grad_hidden_states[:, :, m_i, :] = grad_hidden_m
        grad_query_gamma[m_i] = grad_query_gamma_m
        # Step2: key RMSNorm backward（复用 key_rms/key_n）
        grad_key_m, grad_key_gamma_m = _rms_norm_bwd(
            grad_normed_key_m, key_gamma[m_i], key_rms_m, key_n, out_dt)
        grad_key_gamma[m_i] = grad_key_gamma_m
        # ---- Step1 逆向: K = E @ W_k ----
        grad_emb_from_k, grad_wk_m = linear_backward(
            grad_key_m, embeddings, key_proj_weights[m_i])
        grad_embeddings += grad_emb_from_k.to(acc_dt)
        grad_key_proj_weights[m_i] = grad_wk_m

    return (
        grad_hidden_states.to(out_dt),
        grad_embeddings.to(out_dt),
        grad_key_proj_weights.to(out_dt),
        grad_value_proj_weights.to(out_dt),
        grad_key_gamma.to(out_dt),
        grad_query_gamma.to(out_dt),
    )
