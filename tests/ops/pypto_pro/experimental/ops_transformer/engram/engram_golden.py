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
"""PyPTO-Pro engram_backward golden reference implementation.

Engram Gated Memory 算子（正向 engram_v4）的**反向传播** golden。

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

精度契约:
    - 所有计算均沿用输入 tensor 的 dtype，不在 golden 内部主动升精度或降精度。
    - 输出梯度保持对应输入计算产生的 dtype。
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
    """返回本进程使用的 device。

    设备卡号统一通过环境变量 TILE_FWK_DEVICE_ID 读取（运行前
    `export TILE_FWK_DEVICE_ID=<id>` 设置），未设置时默认使用卡 0。
    仅在无 NPU 硬件（device_count() == 0）时回退 CPU。
    """
    global _DEVICE
    if _DEVICE is None:
        if not _HAS_NPU:
            _DEVICE = torch.device("cpu")
        else:
            device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
            torch.npu.set_device(device_id)
            _DEVICE = torch.device(f"npu:{device_id}")
    return _DEVICE


# ==========================================================================
# Forward (including cache)
# ==========================================================================

def linear_torch(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """线性投影前向: y = x @ weight （无偏置）。

    Args:
        x:      [..., D_in]
        weight: [D_in, D_out]
    Returns:
        y: [..., D_out]，dtype 与输入计算一致。

    device 分派（对齐 kernel cube matmul "操作数全 BF16、L0C FP32 acc"）:
        - NPU → 全操作数转 BF16 做 matmul（FP32 acc），算完转回 x 原生 dtype；
        - CPU → 原 dtype 直接 matmul，逐字不变。
    """
    if x.device.type == "npu":
        out_dt = x.dtype
        return (x.to(torch.bfloat16) @ weight.to(torch.bfloat16)).to(out_dt)
    return x @ weight


def rms_norm_torch(
    x: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """RMSNorm 前向: x̂ = (x / rms) · γ，rms = √(mean(x², -1) + eps)。

    计算和输出沿用输入 dtype。

    device 分派（对齐 kernel: vf_rmsnorm_fwd 输出 ptnr32 是 UB 内 FP32 中间量，不降 BF16）:
        - NPU → 输入 cast BF16→FP32，全程 FP32，normed 落点保持 FP32
          （kernel: load ptnr16 BF16 → cast norm32 FP32 → vf_rmsnorm_fwd → FP32 ptnr32，
           直接喂 vf_scaled_dot，不降级）；
        - CPU → 原手写 pow/mean/sqrt 拼接，逐字不变（FP64 真值）。
    """
    if x.device.type == "npu":
        xf = x.float()
        gammaf = gamma.to(x.device).float()
        rms = torch.sqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)  # FP32
        return (xf / rms) * gammaf  # FP32（对齐 kernel ptnr32，不降级）
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)  # [..., 1]
    return x / rms * gamma


def signed_sqrt_gate(score: torch.Tensor, clamp_value: float = 1e-6) -> torch.Tensor:
    """Signed Sqrt 门控前向: g = σ(sign(s)·√max(|s|, c))。

    Args:
        score: [b, s]
    Returns:
        gate:  [b, s, 1] （末维 unsqueeze，供沿头维 stack）。输出沿用 score dtype。

    device 分派（对齐 kernel: sign/sqrt/clamp/sigmoid 全 FP32 内部）:
        - NPU → 升 FP32 计算，算完转回 score 原生 dtype；
        - CPU → 原 dtype 直接算，逐字不变。
    """
    out_dt = score.dtype
    if score.device.type == "npu":
        sf = score.float()
        logits = torch.sign(sf) * torch.sqrt(sf.abs().clamp_min(clamp_value))  # FP32
        gate = torch.sigmoid(logits)  # FP32
        return gate.to(out_dt).unsqueeze(-1)  # [b, s, 1]
    logits = torch.sign(score) * torch.sqrt(score.abs().clamp_min(clamp_value))  # [b, s]
    gate = torch.sigmoid(logits)  # [b, s]
    return gate.unsqueeze(-1)  # [b, s, 1]


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
    """前向传播，返回 (主输出, 中间量 cache dict)。

    中间量 cache 供 engram_backward_golden 消费；normed_keys 和 normed_queries
    不缓存，由 backward 根据原始输入重算。
    语义与根 engram_golden.py 的 engram_forward_with_cache 完全一致。

    Returns:
        value_out: [b, s, m, h]  主输出（Step7: O = g · V）
        cache:     dict，仅包含 backward 需要的前向中间结果 keys/scores/gates/value。
    """
    b, s, m, h = hidden_states.shape

    keys_list = []
    scores_list = []
    gates_list = []

    for m in range(m):
        # Step1: Key 投影  K^(m) = E @ W_k^(m)
        key = linear_torch(embeddings, key_proj_weights[m])
        # Step2: 公式：Key RMSNorm  K̂^(m) = RMSNorm(K^(m), γ_k^(m))
        normed_key = rms_norm_torch(key, key_gamma[m], eps)
        # Step3: 公式：Query RMSNorm  Q̂^(m) = RMSNorm(h^(m), γ_q^(m))
        normed_query = rms_norm_torch(hidden_states[:, :, m, :], query_gamma[m], eps)
        # Step4: 缩放点积  s^(m) = (1/√H) · Σ_h K̂_h · Q̂_h
        # 对齐 kernel: FP32 reduce（normed_key/normed_query 升 FP32，sum 在 FP32，
        # 1/√H 也在 FP32 构造，避免 BF16 标量通路丢精度）。score 落点 FP32。
        if normed_key.device.type == "npu":
            scale = float(h) ** -0.5
            score = (normed_key.float() * normed_query.float()).sum(dim=-1) * scale
        else:
            score = (normed_key * normed_query).sum(dim=-1) * (h ** -0.5)
        # Step5: Signed Sqrt 门控  g^(m) = σ(sign(s)·√max(|s|,c))
        gate = signed_sqrt_gate(score, clamp_value)

        keys_list.append(key)
        scores_list.append(score)
        gates_list.append(gate)

    # 沿头维 stack
    keys = torch.stack(keys_list, dim=2)  # [b, s, m, h]
    scores_stacked = torch.stack(
        [s.unsqueeze(-1) for s in scores_list], dim=2
    )  # [b, s, m, 1]
    gates = torch.stack(gates_list, dim=2)  # [b, s, m, 1]

    # Step6: Value 投影  V = E @ W_v
    value = linear_torch(embeddings, value_proj_weights)  # [b, s, h]

    # Step7: 门控输出  O = g · V （V 沿头维广播）
    value_out = gates * value.unsqueeze(2)  # [b, s, m, h]

    cache = {
        "keys": keys,
        "scores": scores_stacked,
        "gates": gates,
        "value": value,
    }

    return value_out, cache


# ==========================================================================
# Backward
# ==========================================================================

def linear_backward(
    grad_output: torch.Tensor,  # [..., D_out] 上游梯度 ∂L/∂y
    x: torch.Tensor,  # [..., D_in] 前向输入
    weight: torch.Tensor,  # [D_in, D_out] 权重
):
    """linear_torch 的反向传播（与根 golden 一致）。

    正向: y = x @ weight  (weight 形状 [D_in, D_out]，无偏置)
    反向:
        grad_x      = grad_output @ weight^T   [..., D_in]
        grad_weight = x^T @ grad_output         [D_in, D_out]  (沿 batch reduction)

    device 分派（对齐 kernel Nest1/Nest2/Nest3 cube matmul：操作数全 BF16、L0C FP32 acc）:
        - NPU → 两个 matmul 全部操作数转 BF16（grad_output / x / weight 都 .to(bfloat16)），
          FP32 acc，算完转回 out_dt（grad_output 原生 dtype）。
        - CPU → 原 dtype 直接 matmul，逐字不变。
    """
    out_dt = grad_output.dtype

    if grad_output.device.type == "npu":
        # 公式：grad_x = grad_output @ weight^T  (F.linear(a, b) = a @ b^T; BF16 in, FP32 acc)
        go = grad_output.to(torch.bfloat16)
        ww = weight.to(torch.bfloat16)
        grad_x = F.linear(go, ww).to(out_dt)
        # grad_weight = x^T @ grad_output  （将 batch 维展平后做 reduction matmul）
        xx = x.to(torch.bfloat16)
        x_2d = xx.reshape(-1, xx.shape[-1])  # [N, D_in]  BF16
        grad_2d = go.reshape(-1, go.shape[-1])  # [N, D_out] BF16
        grad_weight = (x_2d.T @ grad_2d).to(out_dt)  # [D_in, D_out]
    else:
        # 公式：grad_x = grad_output @ weight^T  (F.linear(a, b) = a @ b^T)
        grad_x = F.linear(grad_output, weight)

        # grad_weight = x^T @ grad_output  （将 batch 维展平后做 reduction matmul）
        x_2d = x.reshape(-1, x.shape[-1])  # [N, D_in]
        grad_2d = grad_output.reshape(-1, grad_output.shape[-1])  # [N, D_out]
        grad_weight = x_2d.T @ grad_2d  # [D_in, D_out]

    return grad_x, grad_weight


def rms_norm_backward(
    grad_output: torch.Tensor,  # [..., D] 上游梯度 ∂L/∂x̂
    x: torch.Tensor,  # [..., D] 前向输入
    gamma: torch.Tensor,  # [D] 缩放参数
    eps: float = 1e-6,
):
    """rms_norm_torch 的反向传播（与根 golden 一致，不主动改变计算 dtype）。

    正向: rms = √(mean(x²)+ε);  n = x / rms;  x̂_d = n_d · γ_d
    反向:
        grad_n    = grad_x̂ · γ
        grad_x    = (1/rms) · (grad_n − n · mean(grad_n · n, -1))
        grad_γ    = Σ_{batch,seq} grad_x̂ · n   （沿所有非最后一维 reduction）

    device 分派（对齐 kernel vf_rmsbw_* 3-pass：sq/inv_rms/rmean/grad_x 全 FP32 内部）:
        - NPU → 全程 FP32 计算；grad_x 落点 = grad_output 原生 dtype（kernel:
          grad_key_ws/grad_hidden_states 存 BF16）；grad_gamma 落点 FP32
          （kernel 进 FP32 gamma workspace 累加，host 端再 sum→cast）。
        - CPU → 原 dtype 直接算，逐字不变（FP64 真值）。
    """
    out_dt = grad_output.dtype
    if grad_output.device.type == "npu":
        xf = x.float()                          # FP32
        gf = gamma.float()                      # FP32
        gxf = grad_output.float()               # FP32
        # 前向中间值（重算，FP32）
        rms = torch.sqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)  # FP32 [..., 1]
        n = xf / rms                            # FP32 [..., D]
        # 公式：∂L/∂n = ∂L/∂x̂ · γ
        grad_n = gxf * gf                       # FP32
        # 公式：∂L/∂x = (1/RMS) · (∂L/∂n − n · mean(∂L/∂n · n))
        inner = (grad_n * n).mean(dim=-1, keepdim=True)  # FP32 [..., 1]
        grad_x = ((grad_n - n * inner) / rms).to(out_dt)  # 落点 BF16（对齐 kernel ws）
        # 公式：∂L/∂γ = Σ_{batch,seq} ∂L/∂x̂ · n   （FP32 累加，对齐 kernel gamma workspace）
        grad_gamma = (gxf * n).flatten(0, -2).sum(dim=0)  # FP32 [D]
        return grad_x, grad_gamma

    # 前向中间值（重算，与根 golden 一致）
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)  # [..., 1]
    n = x / rms  # [..., D]

    # 公式：∂L/∂n = ∂L/∂x̂ · γ
    grad_n = grad_output * gamma  # [..., D]

    # 公式：∂L/∂x = (1/RMS) · (∂L/∂n − n · mean(∂L/∂n · n))
    inner = (grad_n * n).mean(dim=-1, keepdim=True)  # [..., 1]
    grad_x = (grad_n - n * inner) / rms  # [..., D]

    # 公式：∂L/∂γ = Σ_{batch,seq} ∂L/∂x̂ · n
    grad_gamma = (grad_output * n).flatten(0, -2).sum(dim=0)  # [D]

    return grad_x, grad_gamma


def signed_sqrt_gate_backward(
    grad_output: torch.Tensor,  # [b, s, 1] 上游梯度 ∂L/∂g
    score: torch.Tensor,  # [b, s] 前向输入分数
    gate: torch.Tensor,  # [b, s, 1] 前向输出门控值
    clamp_value: float = 1e-6,
):
    """signed_sqrt_gate 的反向传播（与根 golden 一致）。

    正向: logits = sign(s)·√max(|s|,c);  g = σ(logits)
    反向:
        dg/dlogits   = g(1−g)
        dlogits/ds   = 1/(2√|s|)  if |s|>c  else 0
        grad_s       = grad_g · g(1−g) · dlogits/ds

    注: d/ds[sign(s)·√|s|] = sign(s)·sign(s)·1/(2√|s|) = 1/(2√|s|)（sign²=1），
        故根 golden 的 logits_grad = mask/(2·sqrt_abs) 无显式 sign 因子，正确。

    device 分派（对齐 kernel vf_gate_bw：g(1−g) / |s| / sqrt / mask/denom 全 FP32）:
        - NPU → 升 FP32 计算，grad_score 落点 FP32（喂 Step4 scaled_dot，kernel 也是 FP32）；
        - CPU → 原 dtype 直接算，逐字不变。
    """
    if grad_output.device.type == "npu":
        gate_s = gate.float().squeeze(-1)  # FP32 [b, s]
        go_f = grad_output.float().squeeze(-1)  # FP32
        sf = score.float()  # FP32
        # 公式：dg/dlogits = g(1−g)
        sigmoid_grad = gate_s * (1.0 - gate_s)  # FP32
        # dlogits/ds: 仅在 |s| > clamp_value 时有梯度（含 +1e-12 防零）
        mask = torch.where(
            sf.abs() > clamp_value,
            torch.ones_like(sf),
            torch.zeros_like(sf),
        )  # FP32
        sqrt_abs = sf.abs().clamp_min(clamp_value).sqrt()  # FP32
        logits_grad = mask / (2.0 * sqrt_abs + 1e-12)  # FP32
        # 公式：∂L/∂s = ∂L/∂g · g(1−g) · dlogits/ds
        grad_score = go_f * sigmoid_grad * logits_grad  # FP32
        return grad_score

    # 公式：dg/dlogits = g(1−g)
    gate_squeezed = gate.squeeze(-1)  # [b, s]
    sigmoid_grad = gate_squeezed * (1.0 - gate_squeezed)  # [b, s]

    # dlogits/ds: 仅在 |s| > clamp_value 时有梯度（与根 golden 一致，含 +1e-12 防零）
    mask = torch.where(
        score.abs() > clamp_value,
        torch.ones_like(score),
        torch.zeros_like(score),
    )  # [b, s]
    sqrt_abs = score.abs().clamp_min(clamp_value).sqrt()  # [b, s]
    logits_grad = mask / (2.0 * sqrt_abs + 1e-12)  # [b, s]

    # 公式：∂L/∂s = ∂L/∂g · g(1−g) · dlogits/ds
    grad_score = grad_output.squeeze(-1) * sigmoid_grad * logits_grad  # [b, s]

    return grad_score


def engram_backward_golden(
    grad_output: torch.Tensor,  # [b, s, m, h] 上游梯度 ∂L/∂O
    # ---- 前向输入 ----
    hidden_states: torch.Tensor,  # [b, s, m, h]
    embeddings: torch.Tensor,  # [b, s, de]
    key_proj_weights: torch.Tensor,  # [m, de, h]
    value_proj_weights: torch.Tensor,  # [de, h]
    key_gamma: torch.Tensor,  # [m, h]
    query_gamma: torch.Tensor,  # [m, h]
    # ---- 前向中间结果 ----
    scores: torch.Tensor,  # [b, s, m, 1] Step4 输出 s
    gates: torch.Tensor,  # [b, s, m, 1] Step5 输出 g
    keys: torch.Tensor,  # [b, s, m, h] Step1 输出 K
    value: torch.Tensor,  # [b, s, h] Step6 输出 V
    # ---- 超参数 ----
    clamp_value: float = 1e-6,
    eps: float = 1e-6,
):
    """Engram Gated Memory 算子的反向传播 golden 实现。

    与根 engram_golden.py 中的 engram_backward_golden 同名同签名，返回 6 个梯度：
        (grad_hidden_states, grad_embeddings, grad_key_proj_weights,
         grad_value_proj_weights, grad_key_gamma, grad_query_gamma)

    精度: 不主动改变 13 个输入 tensor 的 dtype，6 个输出保持对应计算结果的 dtype。
    """
    b, s, m, h = hidden_states.shape
    de = embeddings.shape[-1]
    is_npu = hidden_states.device.type == "npu"
    out_dt = grad_output.dtype

    # ---- Step 7 逆向: O = g · V ----
    # 对齐 kernel vf_grad_value_accum / vf_grad_gate_accum：FP32 内部 reduce。
    if is_npu:
        gof = grad_output.float()                       # FP32
        vf = value.float().unsqueeze(2)                 # FP32
        gf32 = gates.float()                            # FP32
        # 公式：∂L/∂g = Σ_h ∂L/∂O · V   → [b, s, m, 1]   (FP32)
        grad_gates = (gof * vf).sum(dim=-1, keepdim=True)
        # 公式：∂L/∂V = Σ_m ∂L/∂O · g   → [b, s, h]   (FP32)
        grad_value_f = (gof * gf32).sum(dim=2)          # FP32
        # grad_value 落点 BF16（对齐 kernel grad_value_ws，喂 cube linear_backward）
        grad_value = grad_value_f.to(out_dt)
    else:
        grad_gates = (grad_output * value.unsqueeze(2)).sum(dim=-1, keepdim=True)  # [b, s, m, 1]
        grad_value = (grad_output * gates).sum(dim=2)  # [b, s, h]

    # ---- Step 6 逆向: V = E @ W_v ----
    grad_emb_from_v, grad_value_proj_weights = linear_backward(
        grad_value, embeddings, value_proj_weights
    )

    # 初始化 embeddings 梯度累加器（跨头累加）。
    # 对齐 kernel Scheme A: grad_emb_acc 是 FP32 累加器（Nest1 value + 每头 key 都
    # FP32 atomic-add 进同一 slot），host 最后 cast BF16。BF16 原地累加会在 BS=1
    # 分量抵消时放大误差（catastrophic cancellation），FP32 累加根治。
    if is_npu:
        grad_embeddings = grad_emb_from_v.float().clone()    # FP32 累加器
    else:
        grad_embeddings = grad_emb_from_v.clone()                # 原生 dtype（CPU FP64）

    # 初始化其他梯度
    grad_hidden_states = torch.zeros_like(hidden_states)
    grad_key_proj_weights = torch.zeros_like(key_proj_weights)
    grad_key_gamma = torch.zeros_like(key_gamma, dtype=torch.float32) if is_npu \
        else torch.zeros_like(key_gamma)
    grad_query_gamma = torch.zeros_like(query_gamma, dtype=torch.float32) if is_npu \
        else torch.zeros_like(query_gamma)

    for m in range(m):
        # ---- Step 5 逆向: signed_sqrt_gate ----
        score_m = scores[:, :, m, 0]  # [b, s]
        gate_m = gates[:, :, m, :]  # [b, s, 1]
        grad_gate_m = grad_gates[:, :, m, :]  # [b, s, 1]

        grad_score_m = signed_sqrt_gate_backward(
            grad_gate_m, score_m, gate_m, clamp_value
        )  # NPU: FP32 [b, s] ; CPU: 原 dtype

        # ---- Step 4 逆向: s = (1/√H) · Σ_h K̂_h · Q̂_h ----
        hidden_m = hidden_states[:, :, m, :]  # [b, s, h]
        key_m = keys[:, :, m, :]  # [b, s, h]
        # 对齐 kernel vf_rmsnorm_fwd：normed 保持 FP32（UB 内 ptnr32 中间量，不降级）。
        normed_key_m = rms_norm_torch(key_m, key_gamma[m], eps)
        normed_query_m = rms_norm_torch(hidden_m, query_gamma[m], eps)

        if is_npu:
            # 对齐 kernel vf_scaled_dot：FP32（gs·(1/√H)·partner，1/√H 在 FP32 通路构造）。
            scale = float(h) ** -0.5
            gsf = grad_score_m.unsqueeze(-1)  # FP32 [b, s, 1]
            # 公式：∂L/∂K̂ = ∂L/∂s · (1/√H) · Q̂
            grad_normed_key_m = gsf * scale * normed_query_m  # FP32 [b, s, h]
            # 公式：∂L/∂Q̂ = ∂L/∂s · (1/√H) · K̂
            grad_normed_query_m = gsf * scale * normed_key_m  # FP32 [b, s, h]
        else:
            scale = h ** -0.5
            # 公式：∂L/∂K̂ = ∂L/∂s · (1/√H) · Q̂
            grad_normed_key_m = grad_score_m.unsqueeze(-1) * scale * normed_query_m  # [b, s, h]
            # 公式：∂L/∂Q̂ = ∂L/∂s · (1/√H) · K̂
            grad_normed_query_m = grad_score_m.unsqueeze(-1) * scale * normed_key_m  # [b, s, h]

        # ---- Step 3 逆向: Q̂ = RMSNorm(h^(m), γ_q^(m)) ----
        # 对齐 kernel vf_rmsbw_*：FP32 内部；grad_x 落点 BF16（kernel grad_hidden_states BF16）；
        # grad_gamma 落点 FP32（kernel FP32 gamma workspace RMW，host 端 sum→cast）。
        grad_hidden_m, grad_query_gamma_m = rms_norm_backward(
            grad_normed_query_m, hidden_m, query_gamma[m], eps
        )
        grad_hidden_states[:, :, m, :] = grad_hidden_m
        grad_query_gamma[m] = grad_query_gamma_m

        # ---- Step 2 逆向: K̂ = RMSNorm(K^(m), γ_k^(m)) ----
        grad_key_m, grad_key_gamma_m = rms_norm_backward(
            grad_normed_key_m, key_m, key_gamma[m], eps
        )
        grad_key_gamma[m] = grad_key_gamma_m

        # ---- Step 1 逆向: K^(m) = E @ W_k^(m) ----
        # 对齐 kernel Nest1/Nest3 cube：grad_key_ws BF16 in、FP32 acc；grad_emb 进 FP32 累加器。
        grad_emb_from_k, grad_wk_m = linear_backward(
            grad_key_m, embeddings, key_proj_weights[m]
        )
        # 跨头累加：NPU 用 FP32 累加器（对齐 kernel Scheme A FP32 atomic-add 进 grad_emb_acc）。
        if is_npu:
            grad_embeddings += grad_emb_from_k.float()
        else:
            grad_embeddings += grad_emb_from_k  # 跨头累加
        grad_key_proj_weights[m] = grad_wk_m

    return (
        grad_hidden_states.to(grad_output.dtype),       # [b, s, m, h]
        grad_embeddings.to(grad_output.dtype),          # [b, s, de]
        grad_key_proj_weights.to(grad_output.dtype),    # [m, de, h]
        grad_value_proj_weights.to(grad_output.dtype),  # [de, h]
        grad_key_gamma.to(grad_output.dtype),           # [m, h]
        grad_query_gamma.to(grad_output.dtype),         # [m, h]
    )
