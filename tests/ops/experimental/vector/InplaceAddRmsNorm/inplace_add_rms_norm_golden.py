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

"""PyPTO inplace_add_rms_norm golden reference implementation.

算子: inplace_add_rms_norm (融合 elementwise add + RMSNorm，inplace 写回)

数学公式:
    x_add = x1 + x2                                  # [B,S,H]
    ms    = mean(x_add^2, dim=-1, keepdim=True)      # [B,S,1]
    rstd  = 1 / sqrt(ms + eps)                       # [B,S,1]
    y     = x_add * rstd * gamma                     # [B,S,H]

Inplace 写回（核心语义）:
    x1.copy_(y.to(bf16))         # x1 buffer 接收归一化结果
    x2.copy_(x_add.to(bf16))     # x2 buffer 接收 add 中间结果
    rstd 是新建的 [B,S,1] bf16 输出 tensor

调用前后 x1.data_ptr() / x2.data_ptr() 不变。
"""

import torch


# ─────────────────────────────────────────────
# Golden 参考实现（纯 torch）
# ─────────────────────────────────────────────

def inplace_add_rms_norm_golden(
    x1: torch.Tensor,
    x2: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1.0e-6,
):
    """PyTorch 参考实现 (inplace 语义).

    Args:
        x1:     [B, S, H] bfloat16. 输入 + inplace 输出 (写回归一化结果).
        x2:     [B, S, H] bfloat16. 输入 + inplace 输出 (写回 x1+x2).
        gamma:  [H]       bfloat16. RMSNorm 缩放权重 (只读).
        eps:    float, 数值稳定项.

    Returns:
        (final_y, x_add, rstd):
          - final_y: alias of x1 (after inplace overwrite), [B,S,H] bf16.
          - x_add:   alias of x2 (after inplace overwrite), [B,S,H] bf16.
          - rstd:    新建 tensor, [B,S,1] bf16.

    Inplace 行为:
        函数返回后, x1 / x2 的 data_ptr 不变, 内容被覆写。
    """
    # 内部使用 fp32 计算以获得与目标精度一致的 bf16 结果
    x1_fp32 = x1.to(torch.float32)
    x2_fp32 = x2.to(torch.float32)
    gamma_fp32 = gamma.to(torch.float32)

    # 1) elementwise add
    x_add_fp32 = x1_fp32 + x2_fp32                        # [B,S,H]
    ms_fp32 = (x_add_fp32 * x_add_fp32).mean(dim=-1, keepdim=True)  # [B,S,1]
    rstd_fp32 = torch.rsqrt(ms_fp32 + eps)                # [B,S,1]
    y_fp32 = x_add_fp32 * rstd_fp32 * gamma_fp32          # [B,S,H]

    # 写回 bf16
    y_bf16 = y_fp32.to(torch.bfloat16)
    x_add_bf16 = x_add_fp32.to(torch.bfloat16)
    rstd_bf16 = rstd_fp32.to(torch.bfloat16)

    # ── 核心 inplace 语义 ──
    # x1 buffer 写归一化结果; x2 buffer 写 add 中间结果
    x1.copy_(y_bf16)
    x2.copy_(x_add_bf16)
    # rstd 是新建的输出 tensor (独立 buffer)
    rstd = rstd_bf16.contiguous()

    # final_y / x_add 直接以 x1 / x2 别名返回, 便于测试比对
    return x1, x2, rstd


# ==========================================
# 验证
# ==========================================

def _ref_compute(x1_orig, x2_orig, gamma, eps):
    """独立的、不做 inplace 的参考计算路径，用于交叉比对。"""
    x1f = x1_orig.to(torch.float32)
    x2f = x2_orig.to(torch.float32)
    gf = gamma.to(torch.float32)
    x_add = x1f + x2f
    ms = (x_add ** 2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(ms + eps)
    y = x_add * rstd * gf
    return (
        y.to(torch.bfloat16),
        x_add.to(torch.bfloat16),
        rstd.to(torch.bfloat16),
    )


def _check_case(name, B, S, H=7168, eps=1.0e-6, atol=0.001, rtol=0.001, seed=0):
    torch.manual_seed(seed)
    x1 = torch.randn(B, S, H, dtype=torch.bfloat16) * 0.1
    x2 = torch.randn(B, S, H, dtype=torch.bfloat16) * 0.1
    gamma = torch.randn(H, dtype=torch.bfloat16)

    # 备份用于参考计算 (确保 ref 不被 inplace 影响)
    x1_orig = x1.clone()
    x2_orig = x2.clone()

    p1, p2 = x1.data_ptr(), x2.data_ptr()

    final_y, x_add_view, rstd = inplace_add_rms_norm_golden(x1, x2, gamma, eps)

    # ── 检查 1: data_ptr 不变 (inplace) ──
    assert x1.data_ptr() == p1, f"{name}: x1 data_ptr changed (inplace 失败)"
    assert x2.data_ptr() == p2, f"{name}: x2 data_ptr changed (inplace 失败)"
    # ── 检查 2: 返回值是 alias ──
    assert final_y.data_ptr() == x1.data_ptr(), f"{name}: final_y 不是 x1 的 alias"
    assert x_add_view.data_ptr() == x2.data_ptr(), f"{name}: x_add 不是 x2 的 alias"
    # rstd 是新建 tensor, 不能 alias 任何输入
    assert rstd.data_ptr() not in (p1, p2), f"{name}: rstd 应为新建 tensor"

    # ── 检查 3: shape ──
    assert final_y.shape == (B, S, H), f"{name}: final_y shape {final_y.shape}"
    assert x_add_view.shape == (B, S, H), f"{name}: x_add shape {x_add_view.shape}"
    assert rstd.shape == (B, S, 1), f"{name}: rstd shape {rstd.shape}"

    # ── 检查 4: dtype ──
    assert final_y.dtype == torch.bfloat16
    assert x_add_view.dtype == torch.bfloat16
    assert rstd.dtype == torch.bfloat16

    # ── 检查 5: 内容覆写正确性 (与独立计算路径对比) ──
    ref_y, ref_xadd, ref_rstd = _ref_compute(x1_orig, x2_orig, gamma, eps)
    torch.testing.assert_close(final_y.float(), ref_y.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(x_add_view.float(), ref_xadd.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(rstd.float(), ref_rstd.float(), atol=atol, rtol=rtol)

    # ── 检查 6: 无 NaN / Inf ──
    assert torch.isfinite(final_y.float()).all(), f"{name}: final_y has NaN/Inf"
    assert torch.isfinite(x_add_view.float()).all(), f"{name}: x_add has NaN/Inf"
    assert torch.isfinite(rstd.float()).all(), f"{name}: rstd has NaN/Inf"
    # rstd > 0
    assert (rstd.float() > 0).all(), f"{name}: rstd should be positive"

    print(f"  {name:24s} B={B:3d} S={S:5d} H={H} ... PASS")


def _validate():
    print("=" * 60)
    print("inplace_add_rms_norm_golden 验证报告")
    print("=" * 60)

    # -- 1. 典型 case 验证 (SPEC §12 P0 配置) --
    print("\n[典型 case 验证]")
    _check_case("功能_P0", B=1, S=16, seed=0)
    _check_case("性能_P0_a", B=16, S=128, seed=42)
    _check_case("性能_P0_b", B=8, S=128, seed=0)
    _check_case("性能_P0_c", B=64, S=128, seed=1)
    _check_case("性能_P0_d", B=144, S=1, seed=2)

    # -- 2. 泛化 case (动态轴边界采样) --
    print("\n[泛化 case 验证]")
    _check_case("B*S=1024_min", B=1, S=1024, seed=3)  # B=1 边界
    _check_case("B*S=8192_max", B=32, S=256, seed=4)
    _check_case("S=1_B=16", B=16, S=1, seed=5)

    # -- 3. 数值稳定性 --
    print("\n[数值稳定性检查]")
    # 全零输入: rstd = 1/sqrt(eps) ≈ 1000
    torch.manual_seed(0)
    B, S, H = 1, 4, 7168
    x1 = torch.zeros(B, S, H, dtype=torch.bfloat16)
    x2 = torch.zeros(B, S, H, dtype=torch.bfloat16)
    gamma = torch.ones(H, dtype=torch.bfloat16)
    final_y, x_add, rstd = inplace_add_rms_norm_golden(x1, x2, gamma, eps=1e-6)
    assert torch.isfinite(rstd.float()).all(), "全零输入 rstd 应有限"
    assert (rstd.float() > 0).all(), "rstd 必须为正"
    print(f"  全零输入                 rstd≈{rstd.float().mean().item():.2f} ... PASS")

    # 大值输入
    x1 = torch.full((1, 4, 7168), 1.0, dtype=torch.bfloat16)
    x2 = torch.full((1, 4, 7168), 1.0, dtype=torch.bfloat16)
    gamma = torch.ones(7168, dtype=torch.bfloat16)
    final_y, x_add, rstd = inplace_add_rms_norm_golden(x1, x2, gamma, eps=1e-6)
    # x_add 应全部 ≈ 2; ms = 4; rstd = 0.5; y = 2 * 0.5 * 1 = 1
    assert torch.allclose(x_add.float(), torch.full_like(x_add.float(), 2.0), atol=0.01)
    assert torch.allclose(rstd.float(), torch.full_like(rstd.float(), 0.5), atol=0.01)
    assert torch.allclose(final_y.float(), torch.full_like(final_y.float(), 1.0), atol=0.01)
    print(f"  常量输入 (1+1=2)         y=1.0 / rstd=0.5 ... PASS")

    # -- 4. inplace 强语义二次确认 --
    print("\n[inplace 语义检查]")
    torch.manual_seed(7)
    x1 = torch.randn(2, 8, 7168, dtype=torch.bfloat16) * 0.1
    x2 = torch.randn(2, 8, 7168, dtype=torch.bfloat16) * 0.1
    gamma = torch.randn(7168, dtype=torch.bfloat16)
    x1_orig = x1.clone()
    x2_orig = x2.clone()
    p1, p2 = x1.data_ptr(), x2.data_ptr()
    final_y, x_add, rstd = inplace_add_rms_norm_golden(x1, x2, gamma)
    # x1 内容必须已变 (覆写)
    assert not torch.equal(x1, x1_orig), "x1 必须被覆写"
    assert not torch.equal(x2, x2_orig), "x2 必须被覆写"
    # data_ptr 必须不变
    assert x1.data_ptr() == p1
    assert x2.data_ptr() == p2
    # x2 内容应等于 x1_orig + x2_orig (cast bf16)
    expected_xadd = (x1_orig.float() + x2_orig.float()).to(torch.bfloat16)
    torch.testing.assert_close(x2.float(), expected_xadd.float(), atol=0.001, rtol=0.001)
    print(f"  data_ptr 不变 + 内容覆写 + x2==x1_orig+x2_orig ... PASS")

    print("\n" + "=" * 60)
    print("所有验证通过")
    print("=" * 60)


if __name__ == "__main__":
    _validate()
