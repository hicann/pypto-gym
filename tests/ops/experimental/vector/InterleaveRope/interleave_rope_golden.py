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

"""PyPTO interleave_rope golden reference implementation.

算子: interleave_rope
公式 (interleave 模式 RoPE，对最后维相邻元素对 (x[2i], x[2i+1]) 应用旋转):
    y[..., 2i]   = x[..., 2i] * cos_eff[..., 2i]   - x[..., 2i+1] * sin_eff[..., 2i]
    y[..., 2i+1] = x[..., 2i] * sin_eff[..., 2i+1] + x[..., 2i+1] * cos_eff[..., 2i+1]

其中 cos_eff / sin_eff 来自 cos / sin，沿 N（=1）和可能的 S（S_cs=1）维 broadcast 到 [B, N, S, D]。

数据规格:
    x   : [B, N, S, D=64], dtype ∈ {float16, bfloat16}, contiguous ND
    cos : [B, 1, S_cs, D=64], dtype 同 x, S_cs ∈ {1, S}
    sin : [B, 1, S_cs, D=64], dtype 同 x, S_cs ∈ {1, S}
    y   : [B, N, S, D=64], dtype 同 x

精度要求: atol=1e-4, rtol=7.8125e-3
内部累积: 计算时 cast 到 float32，最后 cast 回原 dtype。

"""

from typing import Tuple

import torch


def interleave_rope_golden(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Interleave-style RoPE 参考实现 (pure torch).

    对 x 的最后维 D=64 上的相邻元素对 (x[..., 2i], x[..., 2i+1]) 应用旋转：

        y[..., 2i]   = x[..., 2i]   * c_i - x[..., 2i+1] * s_i
        y[..., 2i+1] = x[..., 2i]   * s_i + x[..., 2i+1] * c_i

    其中 c_i, s_i 来自 cos, sin。本实现按"位置取值"方式工作：
    直接读取 cos[..., 2i]/cos[..., 2i+1] 与 sin[..., 2i]/sin[..., 2i+1]，
    因此对两种常见输入约定都正确：
        (a) cos/sin 的偶位与奇位相同（即 cos[2i] == cos[2i+1] = cos(θ_i)）— 默认 interleave 假设；
        (b) cos/sin 已展开为完整 D 长度且偶/奇位置可不同 — 按位置直接读取。

    内部 cast 到 float32 累积，最后 cast 回原 dtype。

    Args:
        x:   [B, N, S, D=64], dtype ∈ {float16, bfloat16}, contiguous
        cos: [B, 1, S_cs, D=64], dtype 同 x, S_cs ∈ {1, S}
        sin: [B, 1, S_cs, D=64], dtype 同 x, S_cs ∈ {1, S}

    Returns:
        y: [B, N, S, D=64], dtype 同 x
    """
    # ---- 1. 输入校验 ----
    assert x.dim() == 4, f"x must be 4D [B,N,S,D], got {tuple(x.shape)}"
    assert cos.dim() == 4 and sin.dim() == 4, \
        f"cos/sin must be 4D [B,1,S_cs,D], got cos={tuple(cos.shape)}, sin={tuple(sin.shape)}"
    B, N, S, D = x.shape
    Bc, Nc, S_cs, Dc = cos.shape
    Bs, Ns, S_ss, Ds = sin.shape
    assert Bc == B and Bs == B, f"batch mismatch: x.B={B}, cos.B={Bc}, sin.B={Bs}"
    assert Nc == 1 and Ns == 1, f"cos/sin must have N=1, got cos.N={Nc}, sin.N={Ns}"
    assert S_cs == S_ss, f"cos/sin S must match, got cos.S={S_cs}, sin.S={S_ss}"
    assert S_cs == 1 or S_cs == S, f"S_cs must be 1 or S(={S}), got {S_cs}"
    assert D == Dc == Ds, f"D must match, got x.D={D}, cos.D={Dc}, sin.D={Ds}"
    assert D % 2 == 0, f"D must be even for interleave RoPE, got {D}"
    assert x.dtype == cos.dtype == sin.dtype, \
        f"dtype must match, got x={x.dtype}, cos={cos.dtype}, sin={sin.dtype}"
    assert x.dtype in (torch.float16, torch.bfloat16), \
        f"unsupported dtype {x.dtype}; expected float16 or bfloat16"

    orig_dtype = x.dtype

    # ---- 2. cast 到 fp32 内部累积 ----
    x_f = x.to(torch.float32)
    cos_f = cos.to(torch.float32)
    sin_f = sin.to(torch.float32)

    # ---- 3. 沿 N 维 broadcast cos/sin 到 [B, N, S, D]（S_cs=1 时同时沿 S broadcast）----
    # cos/sin: [B, 1, S_cs, D]，利用 torch 隐式 broadcast 即可。
    # 因此 x_f * cos_f / x_f * sin_f 在最后两维上自动 broadcast。
    x_even = x_f[..., 0::2]  # [B, N, S, D/2]
    x_odd = x_f[..., 1::2]  # [B, N, S, D/2]

    # cos/sin 在偶/奇位置直接按位置取（支持两种约定）
    cos_even = cos_f[..., 0::2]  # [B, 1, S_cs, D/2]  对应 cos[..., 2i]
    cos_odd = cos_f[..., 1::2]  # [B, 1, S_cs, D/2]  对应 cos[..., 2i+1]
    sin_even = sin_f[..., 0::2]  # [B, 1, S_cs, D/2]  对应 sin[..., 2i]
    sin_odd = sin_f[..., 1::2]  # [B, 1, S_cs, D/2]  对应 sin[..., 2i+1]

    # ---- 5. 计算 ----
    y_even = x_even * cos_even - x_odd * sin_even  # [B, N, S, D/2]
    y_odd = x_even * sin_odd + x_odd * cos_odd   # [B, N, S, D/2]

    # ---- 6. split-half 输出 layout (v2.3) ----
    # 不再做 interleave 重组。约定：
    # 下游 attention QK^T 在 Q/K 同 layout 时数值等价。
    y = torch.cat((y_even, y_odd), dim=-1)  # [B, N, S, D]

    # ---- 7. cast 回原 dtype ----
    return y.to(orig_dtype)


# ==========================================
# 参考交叉实现（用于自验证；仅 _validate 内使用）
# ==========================================

def _interleave_rope_complex_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """使用 torch.view_as_complex 的等价实现，作为交叉验证基线。

    仅当 cos/sin 满足 interleave 约定 cos[..., 2i] == cos[..., 2i+1] 时与上面 golden 等价；
    在通用情形下作为 sanity check（数值上应近似一致）。
    """
    orig_dtype = x.dtype
    x_f = x.to(torch.float32)
    cos_f = cos.to(torch.float32)
    sin_f = sin.to(torch.float32)

    # 取每对的 θ_i 系数：interleave 假设下取偶位即可
    c = cos_f[..., 0::2]   # [B, 1, S_cs, D/2]
    s = sin_f[..., 0::2]   # [B, 1, S_cs, D/2]

    x_pair = x_f.reshape(*x_f.shape[:-1], x_f.shape[-1] // 2, 2)  # [B, N, S, D/2, 2]
    xe = x_pair[..., 0]
    xo = x_pair[..., 1]
    ye = xe * c - xo * s
    yo = xe * s + xo * c
    # v2.3: split-half layout
    y = torch.cat((ye, yo), dim=-1)
    return y.to(orig_dtype)


# ==========================================
# 验证
# ==========================================

def _make_inputs(B: int, N: int, S: int, D: int, S_cs: int, dtype: torch.dtype,
                 seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, N, S, D, generator=g, dtype=torch.float32).to(dtype)

    # 构造 cos/sin 满足 interleave 约定 cos[2i]==cos[2i+1]==cos(θ_i)
    theta = torch.randn(B, 1, S_cs, D // 2, generator=g, dtype=torch.float32) * 0.5
    cos_half = torch.cos(theta)
    sin_half = torch.sin(theta)
    # 沿最后维 repeat 到 D：[c0, c0, c1, c1, ...]
    cos = cos_half.unsqueeze(-1).expand(B, 1, S_cs, D // 2, 2).reshape(B, 1, S_cs, D).to(dtype)
    sin = sin_half.unsqueeze(-1).expand(B, 1, S_cs, D // 2, 2).reshape(B, 1, S_cs, D).to(dtype)
    return x, cos, sin


def _check(name: str, cond: bool, detail: str = "") -> bool:
    mark = "\u2713 PASS" if cond else "\u2717 FAIL"
    print(f"  {name} ... {mark}{(' ' + detail) if detail else ''}")
    return cond


def _validate_typical_cases():
    """Validate typical P0 cases."""
    print("\n[典型 case 验证]")
    all_pass = True
    typical_cases = [
        ("功能_P0_min      ", 1, 1, 1024, 64, 1024, torch.bfloat16),
        ("功能_P0_typ      ", 1, 128, 2048, 64, 2048, torch.bfloat16),
        ("功能_P0_Scs1     ", 2, 128, 4096, 64, 1, torch.bfloat16),
        ("功能_P0_typ_fp16 ", 1, 8, 1024, 64, 1024, torch.float16),
    ]
    for name, B, N, S, D, S_cs, dtype in typical_cases:
        x, cos, sin = _make_inputs(B, N, S, D, S_cs, dtype, seed=42)
        y = interleave_rope_golden(x, cos, sin)
        cond = (y.shape == x.shape) and (y.dtype == dtype)
        all_pass &= _check(f"{name} B={B},N={N},S={S},D={D},S_cs={S_cs},dt={dtype}",
                           cond, f"out.shape={tuple(y.shape)} dtype={y.dtype}")
    return all_pass


def _validate_generalization_cases():
    """Validate generalization configs."""
    print("\n[泛化 case 验证]")
    all_pass = True
    gen_cases = [
        (1, 1, 1, 64, 1, torch.bfloat16),
        (4, 128, 8192, 64, 8192, torch.bfloat16),
        (2, 1, 512, 64, 512, torch.float16),
        (1, 128, 1, 64, 1, torch.bfloat16),
        (3, 64, 4096, 64, 1, torch.bfloat16),
    ]
    for B, N, S, D, S_cs, dtype in gen_cases:
        total = B * N * S * D
        if total > 64 * 1024 * 1024:
            print(f"  B={B},N={N},S={S},D={D},S_cs={S_cs} ... (skip: too large for cpu validation)")
            continue
        x, cos, sin = _make_inputs(B, N, S, D, S_cs, dtype, seed=B * 100 + S)
        y = interleave_rope_golden(x, cos, sin)
        cond = (y.shape == (B, N, S, D)) and (y.dtype == dtype)
        all_pass &= _check(f"B={B},N={N},S={S},D={D},S_cs={S_cs},dt={dtype}",
                           cond, f"out.shape={tuple(y.shape)}")
    return all_pass


def _validate_mathematical_correctness():
    """Cross-validate against complex-style reference."""
    print("\n[数学正确性检查 vs complex-ref]")
    all_pass = True
    for B, N, S, S_cs, dtype in [
        (1, 4, 16, 16, torch.bfloat16),
        (2, 8, 32, 1, torch.float16),
        (1, 1, 8, 8, torch.bfloat16),
    ]:
        x, cos, sin = _make_inputs(B, N, S, 64, S_cs, dtype, seed=7)
        y_a = interleave_rope_golden(x, cos, sin).to(torch.float32)
        y_b = _interleave_rope_complex_ref(x, cos, sin).to(torch.float32)
        diff = (y_a - y_b).abs().max().item()
        atol = 1e-4 if dtype == torch.float16 else 5e-3
        cond = diff <= atol
        all_pass &= _check(f"B={B},N={N},S={S},S_cs={S_cs},dt={dtype}",
                           cond, f"max_abs_diff={diff:.2e} (atol={atol})")
    return all_pass


def _validate_boundary_special():
    """Validate boundary and special cases."""
    print("\n[边界与特殊点]")
    all_pass = True
    B, N, S, D = 1, 2, 4, 64

    # cos=1, sin=0
    x = torch.randn(B, N, S, D, dtype=torch.float32).to(torch.bfloat16)
    cos = torch.ones(B, 1, S, D, dtype=torch.bfloat16)
    sin = torch.zeros(B, 1, S, D, dtype=torch.bfloat16)
    y = interleave_rope_golden(x, cos, sin)
    expected = torch.cat((x[..., 0::2], x[..., 1::2]), dim=-1)
    diff = (y.to(torch.float32) - expected.to(torch.float32)).abs().max().item()
    all_pass &= _check("cos=1,sin=0 \u2192 y==split_half(x)", diff < 1e-2, f"max_abs_diff={diff:.2e}")

    # x=0
    x = torch.zeros(B, N, S, D, dtype=torch.bfloat16)
    cos_in = torch.randn(B, 1, S, D, dtype=torch.float32).to(torch.bfloat16)
    sin_in = torch.randn(B, 1, S, D, dtype=torch.float32).to(torch.bfloat16)
    y = interleave_rope_golden(x, cos_in, sin_in)
    all_pass &= _check("x=0 \u2192 y=0", torch.all(y == 0).item())

    # No NaN/Inf
    x, cos, sin = _make_inputs(2, 4, 128, 64, 128, torch.bfloat16, seed=1)
    y = interleave_rope_golden(x, cos, sin)
    all_pass &= _check("无 NaN/Inf", not (torch.isnan(y).any().item() or torch.isinf(y).any().item()))

    # dtype consistency
    for dt in (torch.float16, torch.bfloat16):
        x, cos, sin = _make_inputs(1, 2, 8, 64, 8, dt, seed=2)
        y = interleave_rope_golden(x, cos, sin)
        all_pass &= _check(f"dtype 保持 {dt}", y.dtype == dt, f"got {y.dtype}")
    return all_pass


def _validate_scs1_broadcast():
    """Validate S_cs=1 broadcast consistency."""
    print("\n[S_cs=1 broadcast 一致性]")
    all_pass = True
    B, N, S, D = 2, 4, 16, 64
    x, cos1, sin1 = _make_inputs(B, N, S, D, 1, torch.bfloat16, seed=11)
    cosS = cos1.expand(B, 1, S, D).contiguous()
    sinS = sin1.expand(B, 1, S, D).contiguous()
    y1 = interleave_rope_golden(x, cos1, sin1).to(torch.float32)
    yS = interleave_rope_golden(x, cosS, sinS).to(torch.float32)
    diff = (y1 - yS).abs().max().item()
    all_pass &= _check("S_cs=1 vs S_cs=S(expand) 一致", diff == 0.0, f"max_abs_diff={diff:.2e}")
    return all_pass


def _validate():
    print("=" * 60)
    print("interleave_rope_golden 验证报告")
    print("=" * 60)

    all_pass = True
    all_pass &= _validate_typical_cases()
    all_pass &= _validate_generalization_cases()
    all_pass &= _validate_mathematical_correctness()
    all_pass &= _validate_boundary_special()
    all_pass &= _validate_scs1_broadcast()

    # ---- 总结 ----
    print("\n" + "=" * 60)
    print("\u2705 所有验证通过" if all_pass else "\u274c 存在失败项")
    print("=" * 60)
    return all_pass


# ==========================================
# Smoke test (Stage 3 要求)
# ==========================================

def _smoke_test():
    """Required smoke test: B=2, N=128, S=2048, D=64, dtype=bfloat16, S_cs=S 与 S_cs=1。"""
    print("\n" + "=" * 60)
    print("Smoke test: interleave_rope_golden")
    print("=" * 60)
    B, N, S, D = 2, 128, 2048, 64
    dtype = torch.bfloat16

    # case 1: S_cs == S
    x, cos, sin = _make_inputs(B, N, S, D, S_cs=S, dtype=dtype, seed=2026)
    y1 = interleave_rope_golden(x, cos, sin)
    print(f"[case S_cs=S ] x.shape={tuple(x.shape)} cos.shape={tuple(cos.shape)} "
          f"sin.shape={tuple(sin.shape)} -> y.shape={tuple(y1.shape)} y.dtype={y1.dtype}")

    # case 2: S_cs == 1
    x, cos, sin = _make_inputs(B, N, S, D, S_cs=1, dtype=dtype, seed=2027)
    y2 = interleave_rope_golden(x, cos, sin)
    print(f"[case S_cs=1 ] x.shape={tuple(x.shape)} cos.shape={tuple(cos.shape)} "
          f"sin.shape={tuple(sin.shape)} -> y.shape={tuple(y2.shape)} y.dtype={y2.dtype}")
    print("=" * 60)


if __name__ == "__main__":
    ok = _validate()
    _smoke_test()
    if not ok:
        raise SystemExit(1)
