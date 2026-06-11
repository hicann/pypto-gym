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

"""PyPTO mhc_pre golden reference implementation.

Golden 参考实现 - MHC Pre-processing 算子

算子说明:
  mhc_pre 是一个多头注意力（Multi-Head Context）前处理算子，用于将输入特征进行
  归一化、矩阵变换和分流处理。

计算步骤:
  1. RMSNorm: 对输入进行 Root Mean Square Layer Normalization
  2. MatMul: 与权重矩阵进行矩阵乘法（使用 F.linear）
  3. Split: 将结果分流为三路（Pre、Post、Res）
  4. 三分支处理:
     - Branch Pre: 生成加权输入 h_in（bfloat16）
     - Branch Post: 生成后处理门控 h_post（float32）
     - Branch Res: 生成组合门控 h_res（float32）

输入规格:
  - x: [B*S, N, D] bfloat16（规格表标注为 [B, S, N, D]）
  - phi: [N²+2N, N*D] float32
  - alpha: [3] float32
  - bias: [N²+2N] float32

输出规格:
  - h_in: [B*S, D] bfloat16
  - h_post: [B*S, N] float32
  - h_res: [B*S, N, N] float32

参数:
  - norm_eps: float32, 默认 1e-6, RMSNorm 的 epsilon
  - hc_eps: float32, 默认 1e-6, sigmoid 输出的精度保护
"""

import torch
import torch.nn.functional as F
from typing import Tuple

# ─────────────────────────────────────────────
# Golden 参考实现（纯 torch）
# ─────────────────────────────────────────────


def mhc_pre_golden(
    x: torch.Tensor,
    phi: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
    norm_eps: float = 1e-6,
    hc_eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MHC Pre-processing 算子的 PyTorch 参考实现。

    Args:
        x: 输入特征 tensor，shape [B*S, N, D]，dtype bfloat16
            其中 B*S 为动态轴（1024/2048/4096）
            注：B*S 是 Batch size * Sequence length 的组合轴
        phi: 权重矩阵，shape [N²+2N, N*D]，dtype float32
        alpha: 缩放系数，shape [3]，dtype float32
               alpha[0] 用于 Branch Pre, alpha[1] 用于 Branch Post, alpha[2] 用于 Branch Res
        bias: 偏置向量，shape [N²+2N]，dtype float32
        norm_eps: RMSNorm 的 epsilon，避免方差为零时除零，默认 1e-6
        hc_eps: sigmoid 输出的精度保护，避免饱和区，默认 1e-6

    Returns:
        h_in: 加权输入，shape [B*S, D]，dtype bfloat16
        h_post: 后处理门控信号，shape [B*S, N]，dtype float32
        h_res: 组合门控信号，shape [B*S, N, N]，dtype float32

    计算流程（基于用户参考实现）:
        Step 1 - Reshape & Float:
            T, N, D = x.shape
            x_flat = x.reshape(T, N*D).float()

        Step 2 - RMSNorm:
            inv_rms = rsqrt(mean(x_flat²) + norm_eps)

        Step 3 - MatMul with Normalization:
            h_mix = F.linear(x_flat, phi.float())
            weight = h_mix * inv_rms

        Step 4 - Split & Unflatten:
            h_pre, h_post, h_res = weight.split([N, N, N*N], dim=-1)
            
        Step 5 - Branch Pre:
            h_res = h_res.unflatten(-1, (N, N))
            h_pre = sigmoid(h_pre * alpha[0] + bias[:N]) + hc_eps

        Step 6 - Branch Post:
            h_post = 2 * sigmoid(h_post * alpha[1] + bias[N:2*N])

        Step 7 - Branch Res: 
            h_res = h_res * alpha[2] + bias[2*N:].view(N, N)

        Step 8 - Weighted Sum:
            y = sum(h_pre.unsqueeze(-1) * x_flat.unflatten(-1, (N, -1)), dim=2)
    """
    # 获取形状信息
    T, N, D = x.shape
    ND = N * D

    # 1. Reshape & Float: [B*S, N, D] -> [B*S, N*D]
    x_flat = x.reshape(T, ND).float()

    # 2. RMSNorm: 计算 inv_rms
    inv_rms = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + norm_eps)

    # 3. MatMul with Normalization: 使用 F.linear
    h_mix = F.linear(x_flat, phi.float())
    weight = h_mix * inv_rms

    # 4. Split & Unflatten
    h_pre, h_post, h_res = weight.split([N, N, N*N], dim=-1)
    

    # 三分支处理
    # 5. Branch Pre: [B*S, N]
    # 注意：bias[:N] 是 [N]，会自动广播到 [B*S, N]
    print("h_pre.shape:", h_pre.shape)
    print("bias[:N].shape:", bias[:N].shape)
    h_res = h_res.unflatten(-1, (N, N))
    h_pre = F.sigmoid(h_pre * alpha[0] + bias[:N].unsqueeze(0)) + hc_eps
    # x_flat: [B*S, N*D] -> unflatten -> [B*S, N, D]
    # h_pre: [B*S, N] -> unsqueeze(-1) -> [B*S, N, 1]
    # 相乘后在 dim=1 (N 维度) 求和 -> [B*S, D]
    y = torch.sum(h_pre.unsqueeze(-1) * x_flat.unflatten(dim=-1, sizes=(N, -1)), dim=1)


    # 6. Branch Post: [B*S, N]
    h_post = 2 * F.sigmoid(h_post * alpha[1] + bias[N:2*N].unsqueeze(0))

    # 7. Branch Res: [B*S, N, N]
    # bias[2*N:].view(N, N) 是 [N, N]，会广播到 [B*S, N, N]
    h_res = h_res * alpha[2] + bias[2*N:].view(N, N).unsqueeze(0)

    # 8. Weighted Sum: 生成 h_in

    return y.bfloat16(), h_post, h_res


# ==========================================
# 验证
# ==========================================

# 定义常量
_N = 8
_D = 5120
_N_SQ_P2N = 80  # N² + 2N = 64 + 16 = 80


def _gen_validate_inputs(bs):
    """Generate input tensors for validation at given batch size."""
    x = torch.randn(bs, _N, _D, dtype=torch.bfloat16)
    phi = torch.randn(_N_SQ_P2N, _N * _D, dtype=torch.float32)
    alpha = torch.randn(3, dtype=torch.float32)
    bias = torch.randn(_N_SQ_P2N, dtype=torch.float32)
    return x, phi, alpha, bias


def _verify_outputs(h_in, h_post, h_res, bs):
    """Verify output shapes and dtypes."""
    assert h_in.shape == (bs, _D), f"h_in shape 错误: 期望 {(bs, _D)}, 实际 {h_in.shape}"
    assert h_post.shape == (bs, _N), f"h_post shape 错误: 期望 {(bs, _N)}, 实际 {h_post.shape}"
    assert h_res.shape == (bs, _N, _N), f"h_res shape 错误: 期望 {(bs, _N, _N)}, 实际 {h_res.shape}"
    assert h_in.dtype == torch.bfloat16, f"h_in dtype 错误: 期望 bfloat16, 实际 {h_in.dtype}"
    assert h_post.dtype == torch.float32, f"h_post dtype 错误: 期望 float32, 实际 {h_post.dtype}"
    assert h_res.dtype == torch.float32, f"h_res dtype 错误: 期望 float32, 实际 {h_res.dtype}"


def _validate_typical_cases():
    """Run typical case validation for bs=1024, 2048, 4096."""
    for bs in [1024, 2048, 4096]:
        x, phi, alpha, bias = _gen_validate_inputs(bs)
        h_in, h_post, h_res = mhc_pre_golden(x, phi, alpha, bias)
        _verify_outputs(h_in, h_post, h_res, bs)
        print(f"  性能_P0 (bs={bs}): \u2713 PASS")


def _validate_generalization_cases():
    """Run generalization case validation."""
    bs = 3000
    x, phi, alpha, bias = _gen_validate_inputs(bs)
    h_in, h_post, h_res = mhc_pre_golden(x, phi, alpha, bias)
    _verify_outputs(h_in, h_post, h_res, bs)
    print(f"  泛化 case (bs={bs}): \u2713 PASS")


def _validate_value_range():
    """Check sigmoid output value range."""
    bs = 128
    x, phi, alpha, bias = _gen_validate_inputs(bs)
    h_in, h_post, h_res = mhc_pre_golden(x, phi, alpha, bias, hc_eps=1e-6)
    assert h_post.min() >= 0, f"h_post 最小值错误: 期望 >= 0, 实际 {h_post.min()}"
    assert h_post.max() <= 2.0, f"h_post 最大值错误: 期望 <= 2.0, 实际 {h_post.max()}"
    print(f"  sigmoid 输出值域检查: \u2713 PASS (h_post in [{h_post.min():.6f}, {h_post.max():.6f}])")


def _validate_numerical_stability():
    """Check numerical stability with large and small inputs."""
    bs = 128
    x, phi, alpha, bias = _gen_validate_inputs(bs)

    # 大值输入
    x_big = x * 100
    h_in, h_post, h_res = mhc_pre_golden(x_big, phi, alpha, bias)
    assert not torch.isnan(h_in).any(), "h_in 包含 NaN"
    assert not torch.isinf(h_in).any(), "h_in 包含 Inf"
    assert not torch.isnan(h_post).any(), "h_post 包含 NaN"
    assert not torch.isinf(h_post).any(), "h_post 包含 Inf"
    assert not torch.isnan(h_res).any(), "h_res 包含 NaN"
    assert not torch.isinf(h_res).any(), "h_res 包含 Inf"
    print(f"  大值输入 (x*=100): \u2713 PASS (无 NaN/Inf)")

    # 小值输入
    x_small = x * 1e-6
    h_in, h_post, h_res = mhc_pre_golden(x_small, phi, alpha, bias)
    assert not torch.isnan(h_in).any(), "h_in 包含 NaN"
    assert not torch.isinf(h_in).any(), "h_in 包含 Inf"
    assert not torch.isnan(h_post).any(), "h_post 包含 NaN"
    assert not torch.isinf(h_post).any(), "h_post 包含 Inf"
    assert not torch.isnan(h_res).any(), "h_res 包含 NaN"
    assert not torch.isinf(h_res).any(), "h_res 包含 Inf"
    print(f"  小值输入 (x*=1e-6): \u2713 PASS (无 NaN/Inf)")


def _validate_signature():
    """Check function signature."""
    import inspect
    sig = inspect.signature(mhc_pre_golden)
    params = list(sig.parameters.keys())
    expected_params = ['x', 'phi', 'alpha', 'bias', 'norm_eps', 'hc_eps']
    assert params == expected_params, f"函数签名错误: 期望 {expected_params}, 实际 {params}"
    assert sig.parameters['norm_eps'].default == 1e-6, "norm_eps 默认值错误"
    assert sig.parameters['hc_eps'].default == 1e-6, "hc_eps 默认值错误"
    print(f"  函数签名: \u2713 PASS")
    print(f"  参数列表: {params}")
    print(f"  默认参数: norm_eps={sig.parameters['norm_eps'].default}, hc_eps={sig.parameters['hc_eps'].default}")


def _validate():
    """自动生成的验证函数 - 运行时动态生成验证报告"""

    print("=" * 60)
    print("mhc_pre_golden 验证报告")
    print("=" * 60)

    # -- 1. 典型 case 验证（来自算子规格中的典型配置）--
    print("\n[典型 case 验证]")
    _validate_typical_cases()

    # -- 2. 泛化 case 验证（来自算子规格中的动态轴范围）--
    print("\n[泛化 case 验证]")
    _validate_generalization_cases()

    # -- 3. 值域检查（从公式推导）--
    print("\n[值域检查]")
    _validate_value_range()

    # -- 4. 数值稳定性检查 --
    print("\n[数值稳定性检查]")
    _validate_numerical_stability()

    # -- 5. 函数签名检查 --
    print("\n[函数签名检查]")
    _validate_signature()

    print("\n" + "=" * 60)
    print("\u2705 所有验证通过")
    print("=" * 60)


if __name__ == "__main__":
    _validate()
