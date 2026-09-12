#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PyPTO interleave_rope golden reference implementation (ASC ops-transformer 兼容).

算子: interleave_rope
公式 (与 ASC ops-transformer posembedding/interleave_rope 一致, half-split cos/sin 配对):

    x 为 interleave 排布: x_even[k] = x[..., 2k], x_odd[k] = x[..., 2k+1]  (k = 0..31)
    cos/sin 半区配对 (任意逐位值, 不假设前后半区相等):
        c_lo = cos[..., 0:32],  c_hi = cos[..., 32:64]
        s_lo = sin[..., 0:32],  s_hi = sin[..., 32:64]

    y[..., 0:32 ] = x_even · c_lo − x_odd · s_lo
    y[..., 32:64] = x_even · s_hi + x_odd · c_hi

    等价于 ASC README 公式: q = reshape(x,[B,N,S,D//2,2]).transpose(-1,-2).reshape([B,N,S,D]);
    q_embed = q·cos + RotateHalf(q)·sin。

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
    """Interleave-style RoPE 参考实现 (pure torch, ASC 半区配对).

    对 x 的最后维 D=64 上的相邻元素对 (x[..., 2i], x[..., 2i+1]) 应用旋转，
    cos/sin 按 ASC 约定以前后半区配对（cos[..., i] 配 x 偶位，cos[..., 32+i] 配 x 奇位）：

        y[..., 0:32 ] = x_even · cos[..., 0:32 ] − x_odd · sin[..., 0:32 ]
        y[..., 32:64] = x_even · sin[..., 32:64] + x_odd · cos[..., 32:64 ]

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
    half = D // 2

    # ---- 2. cast 到 fp32 内部累积 ----
    x_f = x.to(torch.float32)
    cos_f = cos.to(torch.float32)
    sin_f = sin.to(torch.float32)

    # ---- 3. x 奇偶位拆分 (interleave → split-half) ----
    x_even = x_f[..., 0::2]  # [B, N, S, D/2]
    x_odd = x_f[..., 1::2]   # [B, N, S, D/2]

    # ---- 4. cos/sin 半区切分（ASC 半区配对，零奇偶假设）----
    c_lo = cos_f[..., :half]    # [B, 1, S_cs, D/2]  配 x 偶位
    c_hi = cos_f[..., half:]    # [B, 1, S_cs, D/2]  配 x 奇位
    s_lo = sin_f[..., :half]    # [B, 1, S_cs, D/2]  配 x 偶位
    s_hi = sin_f[..., half:]    # [B, 1, S_cs, D/2]  配 x 奇位

    # ---- 5. 计算 ----
    y_even = x_even * c_lo - x_odd * s_lo  # [B, N, S, D/2]
    y_odd = x_even * s_hi + x_odd * c_hi   # [B, N, S, D/2]

    # ---- 6. split-half 输出 layout ----
    y = torch.cat((y_even, y_odd), dim=-1)  # [B, N, S, D]

    # ---- 7. cast 回原 dtype ----
    return y.to(orig_dtype)


# ==========================================
# 参考交叉实现（用于自验证；仅 _validate 内使用）
# ==========================================

def _asc_rotate_half_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """ASC README 公式的直接翻译，作为交叉验证基线（对任意逐位 cos/sin 等价）。

    q = reshape(x, [B,N,S,D//2,2]).transpose(-1,-2).reshape([B,N,S,D])   # interleave → split-half
    RotateHalf(q)[..., :32] = -q[..., 32:],  RotateHalf(q)[..., 32:] = q[..., :32]
    q_embed = q·cos + RotateHalf(q)·sin
    """
    orig_dtype = x.dtype
    x_f = x.to(torch.float32)
    B, N, S, D = x_f.shape
    half = D // 2

    # interleave → split-half
    q = x_f.reshape(B, N, S, half, 2).transpose(-1, -2).reshape(B, N, S, D)
    # RotateHalf
    rot = torch.cat((-q[..., half:], q[..., :half]), dim=-1)
    y = q * cos.to(torch.float32) + rot * sin.to(torch.float32)
    return y.to(orig_dtype)


def _rope_rotation_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """真实 RoPE 旋转语义参考（要求 cos/sin 为 half-duplicated: cat(freqs, freqs)）。

    对每个相邻对 (x[2i], x[2i+1]) 以角度 θ_i（cosθ_i = cos[..., i] = cos[..., 32+i]）旋转：
        y_even[i] = x[2i]·cosθ_i − x[2i+1]·sinθ_i
        y_odd[i]  = x[2i]·sinθ_i + x[2i+1]·cosθ_i
    """
    orig_dtype = x.dtype
    x_f = x.to(torch.float32)
    c = cos.to(torch.float32)[..., :32]  # half-duplicated 时前后半区相同
    s = sin.to(torch.float32)[..., :32]
    xe = x_f[..., 0::2]
    xo = x_f[..., 1::2]
    ye = xe * c - xo * s
    yo = xe * s + xo * c
    return torch.cat((ye, yo), dim=-1).to(orig_dtype)


# ==========================================
# 验证
# ==========================================

def _make_inputs(B: int, N: int, S: int, D: int, S_cs: int, dtype: torch.dtype,
                 seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """构造 ASC 约定的输入：x 随机；cos/sin 为 half-duplicated（cat(freqs, freqs)）。"""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, N, S, D, generator=g, dtype=torch.float32).to(dtype)

    # cos/sin 满足 half-duplicated 约定：cos[..., i] == cos[..., 32+i] == cos(θ_i)
    theta = torch.randn(B, 1, S_cs, D // 2, generator=g, dtype=torch.float32) * 0.5
    cos_half = torch.cos(theta)
    sin_half = torch.sin(theta)
    # 沿最后维 cat 成完整 D：[c0..c31 | c0..c31]
    cos = torch.cat((cos_half, cos_half), dim=-1).to(dtype)
    sin = torch.cat((sin_half, sin_half), dim=-1).to(dtype)
    return x, cos, sin


def _make_inputs_positional(B: int, N: int, S: int, D: int, S_cs: int, dtype: torch.dtype,
                            seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """构造任意逐位 cos/sin（前后半区独立，ASC example 风格）——最严格的配对测试。"""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, N, S, D, generator=g, dtype=torch.float32).to(dtype)
    cos = torch.randn(B, 1, S_cs, D, generator=g, dtype=torch.float32).clamp_(-1, 1).to(dtype)
    sin = torch.randn(B, 1, S_cs, D, generator=g, dtype=torch.float32).clamp_(-1, 1).to(dtype)
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
    """Cross-validate against ASC README formula (rotate-half) on arbitrary positional data."""
    print("\n[数学正确性检查 vs ASC rotate_half-ref（任意逐位 cos/sin）]")
    all_pass = True
    for B, N, S, S_cs, dtype in [
        (1, 4, 16, 16, torch.bfloat16),
        (2, 8, 32, 1, torch.float16),
        (1, 1, 8, 8, torch.bfloat16),
        (2, 3, 24, 24, torch.bfloat16),
    ]:
        x, cos, sin = _make_inputs_positional(B, N, S, 64, S_cs, dtype, seed=7)
        y_a = interleave_rope_golden(x, cos, sin).to(torch.float32)
        y_b = _asc_rotate_half_ref(x, cos, sin).to(torch.float32)
        diff = (y_a - y_b).abs().max().item()
        cond = diff == 0.0
        all_pass &= _check(f"B={B},N={N},S={S},S_cs={S_cs},dt={dtype}",
                           cond, f"max_abs_diff={diff:.2e}")
    return all_pass


def _validate_rope_semantics():
    """With half-duplicated cos/sin, the op must equal true rotation by θ."""
    print("\n[RoPE 旋转语义检查（half-duplicated cos/sin）]")
    all_pass = True
    for B, N, S, S_cs, dtype in [
        (1, 4, 16, 16, torch.bfloat16),
        (2, 2, 32, 1, torch.float16),
    ]:
        x, cos, sin = _make_inputs(B, N, S, 64, S_cs, dtype, seed=13)
        y_a = interleave_rope_golden(x, cos, sin).to(torch.float32)
        y_b = _rope_rotation_ref(x, cos, sin).to(torch.float32)
        diff = (y_a - y_b).abs().max().item()
        cond = diff == 0.0
        all_pass &= _check(f"B={B},N={N},S={S},S_cs={S_cs},dt={dtype}",
                           cond, f"max_abs_diff={diff:.2e}")
    return all_pass


def _validate_boundary_special():
    """Validate boundary and special cases."""
    print("\n[边界与特殊点]")
    all_pass = True
    B, N, S, D = 1, 2, 4, 64

    # cos 全 1（前后半区同为 1）、sin 全 0 → 纯 interleave→split-half 换排
    x = torch.randn(B, N, S, D, dtype=torch.float32).to(torch.bfloat16)
    cos = torch.ones(B, 1, S, D, dtype=torch.bfloat16)
    sin = torch.zeros(B, 1, S, D, dtype=torch.bfloat16)
    y = interleave_rope_golden(x, cos, sin)
    expected = torch.cat((x[..., 0::2], x[..., 1::2]), dim=-1)
    diff = (y.to(torch.float32) - expected.to(torch.float32)).abs().max().item()
    all_pass &= _check("cos=1,sin=0 \u2192 y==split_half(x)", diff < 1e-2, f"max_abs_diff={diff:.2e}")

    # 仅前半区为 1：y_even==x_even, y_odd==x_odd（半区配对的直接验证）
    cos = torch.zeros(B, 1, S, D, dtype=torch.bfloat16)
    cos[..., :32] = 1.0
    sin = torch.zeros(B, 1, S, D, dtype=torch.bfloat16)
    y = interleave_rope_golden(x, cos, sin)
    expected = torch.cat((x[..., 0::2], torch.zeros_like(x[..., 1::2])), dim=-1)
    diff = (y.to(torch.float32) - expected.to(torch.float32)).abs().max().item()
    all_pass &= _check("仅 c_lo=1 \u2192 y_odd==0（半区配对）", diff < 1e-2, f"max_abs_diff={diff:.2e}")

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
    print("interleave_rope_golden 验证报告 (ASC 半区配对)")
    print("=" * 60)

    all_pass = True
    all_pass &= _validate_typical_cases()
    all_pass &= _validate_generalization_cases()
    all_pass &= _validate_mathematical_correctness()
    all_pass &= _validate_rope_semantics()
    all_pass &= _validate_boundary_special()
    all_pass &= _validate_scs1_broadcast()

    # ---- 总结 ----
    print("\n" + "=" * 60)
    print("\u2705 所有验证通过" if all_pass else "\u274c 存在失败项")
    print("=" * 60)
    return all_pass


# ==========================================
# Smoke test
# ==========================================

def _smoke_test():
    """Smoke test: B=2, N=128, S=2048, D=64, dtype=bfloat16, S_cs=S 与 S_cs=1。"""
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
