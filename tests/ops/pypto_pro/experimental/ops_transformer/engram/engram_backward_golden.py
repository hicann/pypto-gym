#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

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

数学真值来源（唯一）:
    D:\\Iteration\\July_pyptopro\\engram_backward_golden.py 的
    engram_backward_golden() 及其调用的 linear_backward / rms_norm_backward /
    signed_sqrt_gate_backward。本文件的 7 步逆向逻辑与 3 个基础反传**逐行移植**，
    公式与计算顺序完全一致，不优化、不重排。

精度契约:
    - 所有计算均沿用输入 tensor 的 dtype，不在 golden 内部主动升精度或降精度。
    - 输出梯度保持对应输入计算产生的 dtype。

自包含说明:
    - 根 engram_backward_golden.py 顶部 `from engram_forward_golden import ...` 会
      失败（该文件不存在）。本 golden **完全自包含**：三个前向原语
      (linear_torch / rms_norm_torch / signed_sqrt_gate) 与 engram_forward_with_cache
      就地实现，语义与 SPEC §1.1 Step1-Step7 及根 golden 注释一致。
    - 不依赖 pypto / pypto_pro。

置信度: ★★★★★
    - 7 步逆向 + 3 基础反传为根 golden 逐行移植（数学真值）。
    - 全部使用 torch 标准算子（matmul / F.linear / sum / sqrt / sigmoid / abs /
      clamp / mean），在 NPU 上可跑。
    - _validate 内置 autograd 数值梯度交叉验证，独立确认 backward 数学正确性。

运行:
    python custom/engram_backward/engram_backward_golden.py
导出:
    engram_backward_golden(...)   —— 供 test_engram_backward.py 调用
    engram_forward_with_cache(...)—— 供测试夹具造前向中间量
    linear_torch / rms_norm_torch / signed_sqrt_gate —— 前向原语
    linear_backward / rms_norm_backward / signed_sqrt_gate_backward —— 基础反传
    _get_device()                 —— 供 test 复用设备号
    _make_inputs(device)          —— 供 profiling --factory 使用
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
            device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "1"))
            torch.npu.set_device(device_id)
            _DEVICE = torch.device(f"npu:{device_id}")
    return _DEVICE


# ==========================================================================
# 自包含前向辅助（供测试夹具造前向中间量；语义与 SPEC §1.1 / 根 golden 一致）
# ==========================================================================

def linear_torch(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """线性投影前向: y = x @ weight （无偏置）。

    Args:
        x:      [..., D_in]
        weight: [D_in, D_out]
    Returns:
        y: [..., D_out]，dtype 与输入计算一致。
    """
    return x @ weight


def rms_norm_torch(
    x: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """RMSNorm 前向: x̂ = (x / rms) · γ，rms = √(mean(x², -1) + eps)。

    计算和输出沿用输入 dtype。
    """
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)  # [..., 1]
    return x / rms * gamma


def signed_sqrt_gate(score: torch.Tensor, clamp_value: float = 1e-6) -> torch.Tensor:
    """Signed Sqrt 门控前向: g = σ(sign(s)·√max(|s|, c))。

    Args:
        score: [b, s]
    Returns:
        gate:  [b, s, 1] （末维 unsqueeze，供沿头维 stack）。输出沿用 score dtype。
    """
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
    语义与根 engram_backward_golden.py 的 engram_forward_with_cache 完全一致。

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
# 基础算子反向（与根 engram_backward_golden.py 逐行一致）
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

    说明: 此处不主动改变输入 dtype，运算沿用调用者提供的 dtype。
    """
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
    """
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
    """
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


# ==========================================================================
# 主算子反向（golden）—— 根 engram_backward_golden 逐行移植
# ==========================================================================

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

    与根 engram_backward_golden.py 同名同签名，返回 6 个梯度：
        (grad_hidden_states, grad_embeddings, grad_key_proj_weights,
         grad_value_proj_weights, grad_key_gamma, grad_query_gamma)

    精度: 不主动改变 13 个输入 tensor 的 dtype，6 个输出保持对应计算结果的 dtype。
    """
    b, s, m, h = hidden_states.shape
    de = embeddings.shape[-1]

    # ---- Step 7 逆向: O = g · V ----
    # 公式：∂L/∂g = Σ_h ∂L/∂O · V   → [b, s, m, 1]
    grad_gates = (grad_output * value.unsqueeze(2)).sum(dim=-1, keepdim=True)  # [b, s, m, 1]
    # 公式：∂L/∂V = Σ_m ∂L/∂O · g   → [b, s, h]
    grad_value = (grad_output * gates).sum(dim=2)  # [b, s, h]

    # ---- Step 6 逆向: V = E @ W_v ----
    grad_emb_from_v, grad_value_proj_weights = linear_backward(
        grad_value, embeddings, value_proj_weights
    )

    # 初始化 embeddings 梯度累加器（跨头累加）
    grad_embeddings = grad_emb_from_v.clone()

    # 初始化其他梯度
    grad_hidden_states = torch.zeros_like(hidden_states)
    grad_key_proj_weights = torch.zeros_like(key_proj_weights)
    grad_key_gamma = torch.zeros_like(key_gamma)
    grad_query_gamma = torch.zeros_like(query_gamma)

    for m in range(m):
        # ---- Step 5 逆向: signed_sqrt_gate ----
        score_m = scores[:, :, m, 0]  # [b, s]
        gate_m = gates[:, :, m, :]  # [b, s, 1]
        grad_gate_m = grad_gates[:, :, m, :]  # [b, s, 1]

        grad_score_m = signed_sqrt_gate_backward(
            grad_gate_m, score_m, gate_m, clamp_value
        )  # [b, s]

        # ---- Step 4 逆向: s = (1/√H) · Σ_h K̂_h · Q̂_h ----
        hidden_m = hidden_states[:, :, m, :]  # [b, s, h]
        key_m = keys[:, :, m, :]  # [b, s, h]
        # Forward cache does not retain these large tensors; recompute them for backward.
        normed_key_m = rms_norm_torch(key_m, key_gamma[m], eps)
        normed_query_m = rms_norm_torch(hidden_m, query_gamma[m], eps)
        scale = h ** -0.5

        # 公式：∂L/∂K̂ = ∂L/∂s · (1/√H) · Q̂
        grad_normed_key_m = grad_score_m.unsqueeze(-1) * scale * normed_query_m  # [b, s, h]
        # 公式：∂L/∂Q̂ = ∂L/∂s · (1/√H) · K̂
        grad_normed_query_m = grad_score_m.unsqueeze(-1) * scale * normed_key_m  # [b, s, h]

        # ---- Step 3 逆向: Q̂ = RMSNorm(h^(m), γ_q^(m)) ----
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
        grad_emb_from_k, grad_wk_m = linear_backward(
            grad_key_m, embeddings, key_proj_weights[m]
        )
        grad_embeddings += grad_emb_from_k  # 跨头累加
        grad_key_proj_weights[m] = grad_wk_m

    return (
        grad_hidden_states.to(grad_output.dtype),       # [B, S, M, H]
        grad_embeddings.to(grad_output.dtype),          # [B, S, De]
        grad_key_proj_weights.to(grad_output.dtype),    # [M, De, H]
        grad_value_proj_weights.to(grad_output.dtype),  # [De, H]
        grad_key_gamma.to(grad_output.dtype),            # [M, H]
        grad_query_gamma.to(grad_output.dtype),          # [M, H]
    )


# ==========================================================================
# 输入构造（供 profiling --factory 使用）
# ==========================================================================

# P0 验证 shape（与 SPEC.md §12 典型配置一致；M_H=16 编译期常量）
#   功能_P0_H1280: b=2,s=1024,m_h=16,h=1280,de=512  -> m=B·S=2048
#   功能_P0_H2560: b=2,s=1024,m_h=16,h=2560,de=1024 -> m=B·S=2048 (h-split 可选)
P0_SHAPES = {
    "func_p0_h1280": {"b": 2, "s": 1024, "m_h": 16, "h": 1280, "de": 512},
    "func_p0_h2560": {"b": 2, "s": 1024, "m_h": 16, "h": 2560, "de": 1024},
}


def _make_case(device, shapes, dtype=torch.float16, seed=42):
    """构造一组完整 backward 输入：先造 6 个前向输入 + 跑前向得中间量 + grad_output。

    gamma 初始化为 1（与根 golden verify 一致），权重 ×0.5 防止 FP16 matmul 溢出。
    返回 (args_list, kwargs_dict)，args_list 顺序与 engram_backward_golden 签名一致。
    """
    b, s, m_h, h, de = shapes["b"], shapes["s"], shapes["m_h"], shapes["h"], shapes["de"]
    torch.manual_seed(seed)

    hidden_states = torch.randn(b, s, m_h, h, dtype=dtype, device=device)
    embeddings = torch.randn(b, s, de, dtype=dtype, device=device)
    key_proj_weights = torch.randn(m_h, de, h, dtype=dtype, device=device) * 0.5
    value_proj_weights = torch.randn(de, h, dtype=dtype, device=device) * 0.5
    key_gamma = torch.ones(m_h, h, dtype=dtype, device=device)
    query_gamma = torch.ones(m_h, h, dtype=dtype, device=device)

    clamp_value = 1e-6
    eps = 1e-6

    # 前向（无 grad）得反向所需中间量
    with torch.no_grad():
        _, cache = engram_forward_with_cache(
            hidden_states, embeddings, key_proj_weights, value_proj_weights,
            key_gamma, query_gamma, clamp_value, eps,
        )

    grad_output = torch.randn(b, s, m_h, h, dtype=dtype, device=device)

    args = [
        grad_output,
        hidden_states, embeddings, key_proj_weights, value_proj_weights,
        key_gamma, query_gamma,
        cache["scores"], cache["gates"],
        cache["keys"], cache["value"],
    ]
    kwargs = {"clamp_value": clamp_value, "eps": eps}
    return args, kwargs


def _make_inputs(device):
    """构造所有 P0 典型输入，供验证和性能采集共用。

    Returns:
        [(case_name, args_list, kwargs_dict), ...]
          - args_list 顺序与 engram_backward_golden 签名一致（11 tensor）
          - kwargs_dict 含 clamp_value / eps 标量
    """
    cases = []
    for name, shapes in P0_SHAPES.items():
        args, kwargs = _make_case(device, shapes)
        cases.append((name, args, kwargs))
    return cases


# ==========================================================================
# 验证
# ==========================================================================

def _summarize(name: str, t: torch.Tensor):
    """打印单个梯度 tensor 的 shape/norm/finite/NaN/Inf 摘要。"""
    has_nan = bool(torch.isnan(t).any().item())
    has_inf = bool(torch.isinf(t).any().item())
    finite = bool(torch.isfinite(t).all().item())
    norm = float(t.norm().item()) if finite else float("nan")
    log.info(
        "    %-24s shape=%s dtype=%s norm=%.6e finite=%s nan=%s inf=%s",
        name, tuple(t.shape), t.dtype, norm, finite, has_nan, has_inf,
    )
    return finite and not has_nan and not has_inf


def _fp16_contract_sanity(device):
    """FP16 契约自检：构造 FP16 输入 → 前向得中间量 → 反向 → 打印 6 梯度摘要。

    用 m_h=16 的较小 shape（B=2,s=64,h=128,de=128），快速验证 FP16 路径跑通且输出 sane。
    """
    log.info("[FP16 契约自检]  (forward_with_cache → engram_backward_golden)")
    shapes = {"b": 2, "s": 64, "m_h": 16, "h": 128, "de": 128}
    args, kwargs = _make_case(device, shapes, dtype=torch.float16)
    grads = engram_backward_golden(*args, **kwargs)
    names = [
        "grad_hidden_states",
        "grad_embeddings",
        "grad_key_proj_weights",
        "grad_value_proj_weights",
        "grad_key_gamma",
        "grad_query_gamma",
    ]
    all_sane = True
    for name, g in zip(names, grads):
        ok = _summarize(name, g)
        all_sane = all_sane and ok
    tag = "PASS" if all_sane else "FAIL"
    log.info("    -> 6 梯度 sane 检查 ... %s", tag)
    return all_sane


def _autograd_cross_check(device):
    """autograd 数值梯度交叉验证（FP32，独立确认 backward 数学正确性）。

    流程：
      1) FP32 输入 requires_grad → engram_forward_with_cache → O.backward(grad_output)
         → torch autograd 给出 6 个输入的"真"梯度（独立路径，不经手动 backward）。
      2) 同一份数据 detached → engram_forward_with_cache 得 cache →
         engram_backward_golden(grad_output, ..., cache 中间量) → 6 个手动梯度。
      3) 逐项对比 max abs diff / max rel diff。

    说明：两边使用相同 dtype 运行，故应高度一致。唯一理论差异来自
          signed_sqrt_gate_backward 分母的 +1e-12 防零小量，对 |s|=O(1) 的随机输入可忽略。
    """
    log.info("[autograd 数值梯度交叉验证]  (FP32, manual backward vs torch.autograd)")
    b, s, m_h, h, de = 2, 32, 4, 64, 64  # 小 shape，跑得快
    torch.manual_seed(0)
    clamp_value, eps = 1e-6, 1e-6

    # 共享原始数据（FP32）
    hs_data = torch.randn(b, s, m_h, h, dtype=torch.float32, device=device)
    emb_data = torch.randn(b, s, de, dtype=torch.float32, device=device)
    kpw_data = torch.randn(m_h, de, h, dtype=torch.float32, device=device) * 0.5
    vpw_data = torch.randn(de, h, dtype=torch.float32, device=device) * 0.5
    kg_data = torch.ones(m_h, h, dtype=torch.float32, device=device)
    qg_data = torch.ones(m_h, h, dtype=torch.float32, device=device)
    grad_output = torch.randn(b, s, m_h, h, dtype=torch.float32, device=device)

    # ---- 路径 A: torch autograd（forward_with_cache 内部全 torch 算子可微）----
    hs_a = hs_data.clone().requires_grad_(True)
    emb_a = emb_data.clone().requires_grad_(True)
    kpw_a = kpw_data.clone().requires_grad_(True)
    vpw_a = vpw_data.clone().requires_grad_(True)
    kg_a = kg_data.clone().requires_grad_(True)
    qg_a = qg_data.clone().requires_grad_(True)
    value_out_a, _ = engram_forward_with_cache(
        hs_a, emb_a, kpw_a, vpw_a, kg_a, qg_a, clamp_value, eps
    )
    value_out_a.backward(grad_output)
    auto_grads = [hs_a.grad, emb_a.grad, kpw_a.grad, vpw_a.grad, kg_a.grad, qg_a.grad]

    # ---- 路径 b: 手动 engram_backward_golden（同数据 detached 造中间量）----
    with torch.no_grad():
        _, cache = engram_forward_with_cache(
            hs_data, emb_data, kpw_data, vpw_data, kg_data, qg_data,
            clamp_value, eps,
        )
        manual_grads = engram_backward_golden(
            grad_output,
            hs_data, emb_data, kpw_data, vpw_data, kg_data, qg_data,
            cache["scores"], cache["gates"],
            cache["keys"], cache["value"],
            clamp_value, eps,
        )

    names = [
        "hidden_states", "embeddings", "key_proj_weights",
        "value_proj_weights", "key_gamma", "query_gamma",
    ]
    atol_thr, rtol_thr = 1e-4, 1e-3
    all_pass = True
    for name, g_m, g_a in zip(names, manual_grads, auto_grads):
        if g_a is None:
            log.info("    %-20s autograd grad is None ... SKIP", name)
            continue
        # 仅在比较阶段统一到 FP32；golden 计算路径保持输入原生 dtype。
        diff = (g_m.float() - g_a.float()).abs()
        max_abs = float(diff.max().item())
        denom = g_a.float().abs().clamp_min(1e-8)
        max_rel = float((diff / denom).max().item())
        ok = (max_abs <= atol_thr) and (max_rel <= rtol_thr)
        all_pass = all_pass and ok
        log.info(
            "    %-20s max_abs=%.3e max_rel=%.3e ... %s",
            name, max_abs, max_rel, "PASS" if ok else "FAIL",
        )
    log.info("    -> 容差阈值 atol=%s rtol=%s ... %s",
             atol_thr, rtol_thr, "PASS" if all_pass else "FAIL")
    return all_pass


def _validate():
    """自动验证：FP16 契约自检 + autograd 数值梯度交叉验证。"""
    device = _get_device()
    log.info("=" * 70)
    log.info("engram_backward_golden 验证报告")
    log.info("=" * 70)
    log.info("Device: %s  (NPU available: %s)", device, _HAS_NPU)

    ok1 = _fp16_contract_sanity(device)
    ok2 = _autograd_cross_check(device)

    log.info("=" * 70)
    if ok1 and ok2:
        log.info("所有验证通过")
    else:
        log.info("验证完成（FP16 自检=%s, autograd 交叉验证=%s）",
                 "PASS" if ok1 else "FAIL", "PASS" if ok2 else "FAIL")
    log.info("=" * 70)


if __name__ == "__main__":
    _validate()
