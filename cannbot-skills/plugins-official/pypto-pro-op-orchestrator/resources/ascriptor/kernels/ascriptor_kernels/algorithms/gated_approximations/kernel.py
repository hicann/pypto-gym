# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Eight complete gated activations for A2/A3: five GELU source modes (tanh and
Abramowitz-Stegun erf) and three SwiGLU precisions, each tiled over runtime vector owners."""

import importlib
import math

from ascriptor.a2 import *
from functools import lru_cache

# ----------------------------------------------------------------------------------------------------
# gelu_tanh_f32.py
# Original instruction order, DBuff lifetimes and explicit ties-even output cast.
# ----------------------------------------------------------------------------------------------------

TILE_GELU_TANH_F32 = 6656

COEFF_CUBIC = 0.044715

NEG_TWO_C = -2.0 * math.sqrt(2.0 / math.pi)

def gelu_float_kernel(x: GM[f32, (1, 'n')], y: GM[f32, (1, 'n')], n: i32, tile_len: i32):
    in_ub = DBuff(DT.float, [1, TILE_GELU_TANH_F32], Position.UB)
    out_ub = DBuff(DT.float, [1, TILE_GELU_TANH_F32], Position.UB)
    t1 = Tensor(DT.float, [1, TILE_GELU_TANH_F32], Position.UB)
    t2 = Tensor(DT.float, [1, TILE_GELU_TANH_F32], Position.UB)
    exp_ub = Tensor(DT.float, [1, TILE_GELU_TANH_F32], Position.UB)

    n_tiles = CeilDiv(n, tile_len)
    tiles_per_core = CeilDiv(n_tiles, GetVecNum())
    tile_begin = Var(tiles_per_core * GetVecIdx())
    tile_end = Min(tile_begin + tiles_per_core, n_tiles)

    with auto_sync():
        for tile_idx in range(tile_begin, tile_end):
            n0 = Var(tile_idx * tile_len)
            valid = Min(tile_len, n - n0)
            src = in_ub[tile_idx]
            dst = out_ub[tile_idx]

            src[0:1, 0:valid] <<= x[0:1, n0:n0 + valid]
            mul(t1[0:1, 0:valid], src[0:1, 0:valid], src[0:1, 0:valid])
            mul(t2[0:1, 0:valid], t1[0:1, 0:valid], src[0:1, 0:valid])
            muls(t1[0:1, 0:valid], t2[0:1, 0:valid], COEFF_CUBIC)
            add(t2[0:1, 0:valid], src[0:1, 0:valid], t1[0:1, 0:valid])
            muls(t1[0:1, 0:valid], t2[0:1, 0:valid], NEG_TWO_C)
            exp(exp_ub[0:1, 0:valid], t1[0:1, 0:valid])
            adds(t1[0:1, 0:valid], exp_ub[0:1, 0:valid], 1.0)
            div(dst[0:1, 0:valid], src[0:1, 0:valid], t1[0:1, 0:valid])
            y[0:1, n0:n0 + valid] <<= dst[0:1, 0:valid]
    return y

@lru_cache(maxsize=2)
def make_gelu_tanh_f32_kernel(device):
    if device not in ("a2", "a3"):
        raise ValueError("This tensor-vector recipe supports A2/A3")
    api = importlib.import_module("ascriptor." + device)
    return api.kernel(mode="vec")(gelu_float_kernel)

# ----------------------------------------------------------------------------------------------------
# gelu_tanh_f16.py
# Original instruction order, DBuff lifetimes and explicit ties-even output cast.
# ----------------------------------------------------------------------------------------------------

TILE_GELU_TANH_F16 = 6656



def gelu_half_kernel(x: GM[f16, (1, 'n')], y: GM[f16, (1, 'n')], n: i32, tile_len: i32):
    in_f16 = DBuff(DT.half, [1, TILE_GELU_TANH_F16], Position.UB)
    out_f16 = DBuff(DT.half, [1, TILE_GELU_TANH_F16], Position.UB)
    x_f32 = Tensor(DT.float, [1, TILE_GELU_TANH_F16], Position.UB)
    t1 = Tensor(DT.float, [1, TILE_GELU_TANH_F16], Position.UB)
    t2 = Tensor(DT.float, [1, TILE_GELU_TANH_F16], Position.UB)
    exp_f32 = Tensor(DT.float, [1, TILE_GELU_TANH_F16], Position.UB)
    out_f32 = Tensor(DT.float, [1, TILE_GELU_TANH_F16], Position.UB)

    n_tiles = CeilDiv(n, tile_len)
    tiles_per_core = CeilDiv(n_tiles, GetVecNum())
    tile_begin = Var(tiles_per_core * GetVecIdx())
    tile_end = Min(tile_begin + tiles_per_core, n_tiles)

    with auto_sync():
        for tile_idx in range(tile_begin, tile_end):
            n0 = Var(tile_idx * tile_len)
            valid = Min(tile_len, n - n0)
            src = in_f16[tile_idx]
            dst = out_f16[tile_idx]

            src[0:1, 0:valid] <<= x[0:1, n0:n0 + valid]
            cast(x_f32[0:1, 0:valid], src[0:1, 0:valid], round_mode=RoundMode.NONE)
            mul(t1[0:1, 0:valid], x_f32[0:1, 0:valid], x_f32[0:1, 0:valid])
            mul(t2[0:1, 0:valid], t1[0:1, 0:valid], x_f32[0:1, 0:valid])
            muls(t1[0:1, 0:valid], t2[0:1, 0:valid], COEFF_CUBIC)
            add(t2[0:1, 0:valid], x_f32[0:1, 0:valid], t1[0:1, 0:valid])
            muls(t1[0:1, 0:valid], t2[0:1, 0:valid], NEG_TWO_C)
            exp(exp_f32[0:1, 0:valid], t1[0:1, 0:valid])
            adds(t1[0:1, 0:valid], exp_f32[0:1, 0:valid], 1.0)
            div(out_f32[0:1, 0:valid], x_f32[0:1, 0:valid], t1[0:1, 0:valid])
            cast(dst[0:1, 0:valid], out_f32[0:1, 0:valid], round_mode=RoundMode.TO_EVEN)
            y[0:1, n0:n0 + valid] <<= dst[0:1, 0:valid]
    return y

@lru_cache(maxsize=2)
def make_gelu_tanh_f16_kernel(device):
    if device not in ("a2", "a3"):
        raise ValueError("This tensor-vector recipe supports A2/A3")
    api = importlib.import_module("ascriptor." + device)
    return api.kernel(mode="vec")(gelu_half_kernel)

# ----------------------------------------------------------------------------------------------------
# gelu_tanh_bf16.py
# Original instruction order, DBuff lifetimes and explicit ties-even output cast.
# ----------------------------------------------------------------------------------------------------

TILE_GELU_TANH_BF16 = 6656



def gelu_bf16_kernel(x: GM[bf16, (1, 'n')], y: GM[bf16, (1, 'n')], n: i32, tile_len: i32):
    in_bf16 = DBuff(DT.bfloat16, [1, TILE_GELU_TANH_BF16], Position.UB)
    out_bf16 = DBuff(DT.bfloat16, [1, TILE_GELU_TANH_BF16], Position.UB)
    x_f32 = Tensor(DT.float, [1, TILE_GELU_TANH_BF16], Position.UB)
    t1 = Tensor(DT.float, [1, TILE_GELU_TANH_BF16], Position.UB)
    t2 = Tensor(DT.float, [1, TILE_GELU_TANH_BF16], Position.UB)
    exp_f32 = Tensor(DT.float, [1, TILE_GELU_TANH_BF16], Position.UB)
    out_f32 = Tensor(DT.float, [1, TILE_GELU_TANH_BF16], Position.UB)

    n_tiles = CeilDiv(n, tile_len)
    tiles_per_core = CeilDiv(n_tiles, GetVecNum())
    tile_begin = Var(tiles_per_core * GetVecIdx())
    tile_end = Min(tile_begin + tiles_per_core, n_tiles)

    with auto_sync():
        for tile_idx in range(tile_begin, tile_end):
            n0 = Var(tile_idx * tile_len)
            valid = Min(tile_len, n - n0)
            src = in_bf16[tile_idx]
            dst = out_bf16[tile_idx]

            src[0:1, 0:valid] <<= x[0:1, n0:n0 + valid]
            cast(x_f32[0:1, 0:valid], src[0:1, 0:valid], round_mode=RoundMode.NONE)
            mul(t1[0:1, 0:valid], x_f32[0:1, 0:valid], x_f32[0:1, 0:valid])
            mul(t2[0:1, 0:valid], t1[0:1, 0:valid], x_f32[0:1, 0:valid])
            muls(t1[0:1, 0:valid], t2[0:1, 0:valid], COEFF_CUBIC)
            add(t2[0:1, 0:valid], x_f32[0:1, 0:valid], t1[0:1, 0:valid])
            muls(t1[0:1, 0:valid], t2[0:1, 0:valid], NEG_TWO_C)
            exp(exp_f32[0:1, 0:valid], t1[0:1, 0:valid])
            adds(t1[0:1, 0:valid], exp_f32[0:1, 0:valid], 1.0)
            div(out_f32[0:1, 0:valid], x_f32[0:1, 0:valid], t1[0:1, 0:valid])
            cast(dst[0:1, 0:valid], out_f32[0:1, 0:valid], round_mode=RoundMode.TO_EVEN)
            y[0:1, n0:n0 + valid] <<= dst[0:1, 0:valid]
    return y

@lru_cache(maxsize=2)
def make_gelu_tanh_bf16_kernel(device):
    if device not in ("a2", "a3"):
        raise ValueError("This tensor-vector recipe supports A2/A3")
    api = importlib.import_module("ascriptor." + device)
    return api.kernel(mode="vec")(gelu_bf16_kernel)

# ----------------------------------------------------------------------------------------------------
# gelu_erf_f32.py
# Original instruction order, DBuff lifetimes and explicit ties-even output cast.
# ----------------------------------------------------------------------------------------------------

TILE_GELU_ERF_F32 = 5120

ONE_OVER_SQRT2 = 1.0 / math.sqrt(2.0)

ERF_P = 0.3275911

ERF_A1 = 0.254829592

ERF_A2 = -0.284496736

ERF_A3 = 1.421413741

ERF_A4 = -1.453152027

ERF_A5 = 1.061405429

def gelu_erf_float_kernel(x: GM[f32, (1, 'n')], y: GM[f32, (1, 'n')], n: i32, tile_len: i32):
    in_ub = DBuff(DT.float, [1, TILE_GELU_ERF_F32], Position.UB)
    out_ub = DBuff(DT.float, [1, TILE_GELU_ERF_F32], Position.UB)
    t1 = Tensor(DT.float, [1, TILE_GELU_ERF_F32], Position.UB)
    t2 = Tensor(DT.float, [1, TILE_GELU_ERF_F32], Position.UB)
    t3 = Tensor(DT.float, [1, TILE_GELU_ERF_F32], Position.UB)
    exp_ub = Tensor(DT.float, [1, TILE_GELU_ERF_F32], Position.UB)
    one_ub = Tensor(DT.float, [1, TILE_GELU_ERF_F32], Position.UB)

    n_tiles = CeilDiv(n, tile_len)
    tiles_per_core = CeilDiv(n_tiles, GetVecNum())
    tile_begin = Var(tiles_per_core * GetVecIdx())
    tile_end = Min(tile_begin + tiles_per_core, n_tiles)

    dup(one_ub, 1.0)

    with auto_sync():
        for tile_idx in range(tile_begin, tile_end):
            n0 = Var(tile_idx * tile_len)
            valid = Min(tile_len, n - n0)
            src = in_ub[tile_idx]
            dst = out_ub[tile_idx]

            src[0:1, 0:valid] <<= x[0:1, n0:n0 + valid]
            muls(t1[0:1, 0:valid], src[0:1, 0:valid], ONE_OVER_SQRT2)
            mul(t2[0:1, 0:valid], t1[0:1, 0:valid], t1[0:1, 0:valid])
            muls(t3[0:1, 0:valid], t2[0:1, 0:valid], -1.0)
            exp(exp_ub[0:1, 0:valid], t3[0:1, 0:valid])

            abs(t2[0:1, 0:valid], t1[0:1, 0:valid])
            muls(t3[0:1, 0:valid], t2[0:1, 0:valid], ONE_OVER_SQRT2)
            muls(t1[0:1, 0:valid], t2[0:1, 0:valid], ERF_P)
            adds(t1[0:1, 0:valid], t1[0:1, 0:valid], 1.0)
            div(t1[0:1, 0:valid], one_ub[0:1, 0:valid], t1[0:1, 0:valid])

            muls(t2[0:1, 0:valid], t1[0:1, 0:valid], ERF_A5)
            adds(t2[0:1, 0:valid], t2[0:1, 0:valid], ERF_A4)
            mul(t2[0:1, 0:valid], t2[0:1, 0:valid], t1[0:1, 0:valid])
            adds(t2[0:1, 0:valid], t2[0:1, 0:valid], ERF_A3)
            mul(t2[0:1, 0:valid], t2[0:1, 0:valid], t1[0:1, 0:valid])
            adds(t2[0:1, 0:valid], t2[0:1, 0:valid], ERF_A2)
            mul(t2[0:1, 0:valid], t2[0:1, 0:valid], t1[0:1, 0:valid])
            adds(t2[0:1, 0:valid], t2[0:1, 0:valid], ERF_A1)
            mul(t2[0:1, 0:valid], t2[0:1, 0:valid], t1[0:1, 0:valid])

            mul(t1[0:1, 0:valid], t3[0:1, 0:valid], t2[0:1, 0:valid])
            mul(t1[0:1, 0:valid], t1[0:1, 0:valid], exp_ub[0:1, 0:valid])
            muls(t2[0:1, 0:valid], src[0:1, 0:valid], 0.5)
            add(exp_ub[0:1, 0:valid], t2[0:1, 0:valid], t3[0:1, 0:valid])
            sub(dst[0:1, 0:valid], exp_ub[0:1, 0:valid], t1[0:1, 0:valid])

            y[0:1, n0:n0 + valid] <<= dst[0:1, 0:valid]
    return y

@lru_cache(maxsize=2)
def make_gelu_erf_f32_kernel(device):
    if device not in ("a2", "a3"):
        raise ValueError("This tensor-vector recipe supports A2/A3")
    api = importlib.import_module("ascriptor." + device)
    return api.kernel(mode="vec")(gelu_erf_float_kernel)

# ----------------------------------------------------------------------------------------------------
# gelu_erf_bf16.py
# Original instruction order, DBuff lifetimes and explicit ties-even output cast.
# ----------------------------------------------------------------------------------------------------

TILE_GELU_ERF_BF16 = 5120








def gelu_erf_bf16_kernel(x: GM[bf16, (1, 'n')], y: GM[bf16, (1, 'n')], n: i32, tile_len: i32):
    in_bf16 = DBuff(DT.bfloat16, [1, TILE_GELU_ERF_BF16], Position.UB)
    out_bf16 = DBuff(DT.bfloat16, [1, TILE_GELU_ERF_BF16], Position.UB)
    x_f32 = Tensor(DT.float, [1, TILE_GELU_ERF_BF16], Position.UB)
    t1 = Tensor(DT.float, [1, TILE_GELU_ERF_BF16], Position.UB)
    t2 = Tensor(DT.float, [1, TILE_GELU_ERF_BF16], Position.UB)
    t3 = Tensor(DT.float, [1, TILE_GELU_ERF_BF16], Position.UB)
    exp_f32 = Tensor(DT.float, [1, TILE_GELU_ERF_BF16], Position.UB)
    one_f32 = Tensor(DT.float, [1, TILE_GELU_ERF_BF16], Position.UB)
    out_f32 = Tensor(DT.float, [1, TILE_GELU_ERF_BF16], Position.UB)

    n_tiles = CeilDiv(n, tile_len)
    tiles_per_core = CeilDiv(n_tiles, GetVecNum())
    tile_begin = Var(tiles_per_core * GetVecIdx())
    tile_end = Min(tile_begin + tiles_per_core, n_tiles)

    dup(one_f32, 1.0)

    with auto_sync():
        for tile_idx in range(tile_begin, tile_end):
            n0 = Var(tile_idx * tile_len)
            valid = Min(tile_len, n - n0)
            src = in_bf16[tile_idx]
            dst = out_bf16[tile_idx]

            src[0:1, 0:valid] <<= x[0:1, n0:n0 + valid]
            cast(x_f32[0:1, 0:valid], src[0:1, 0:valid], round_mode=RoundMode.NONE)
            muls(t1[0:1, 0:valid], x_f32[0:1, 0:valid], ONE_OVER_SQRT2)
            mul(t2[0:1, 0:valid], t1[0:1, 0:valid], t1[0:1, 0:valid])
            muls(t3[0:1, 0:valid], t2[0:1, 0:valid], -1.0)
            exp(exp_f32[0:1, 0:valid], t3[0:1, 0:valid])
            abs(t2[0:1, 0:valid], t1[0:1, 0:valid])
            muls(t3[0:1, 0:valid], t2[0:1, 0:valid], ONE_OVER_SQRT2)
            muls(t1[0:1, 0:valid], t2[0:1, 0:valid], ERF_P)
            adds(t1[0:1, 0:valid], t1[0:1, 0:valid], 1.0)
            div(t1[0:1, 0:valid], one_f32[0:1, 0:valid], t1[0:1, 0:valid])
            muls(t2[0:1, 0:valid], t1[0:1, 0:valid], ERF_A5)
            adds(t2[0:1, 0:valid], t2[0:1, 0:valid], ERF_A4)
            mul(t2[0:1, 0:valid], t2[0:1, 0:valid], t1[0:1, 0:valid])
            adds(t2[0:1, 0:valid], t2[0:1, 0:valid], ERF_A3)
            mul(t2[0:1, 0:valid], t2[0:1, 0:valid], t1[0:1, 0:valid])
            adds(t2[0:1, 0:valid], t2[0:1, 0:valid], ERF_A2)
            mul(t2[0:1, 0:valid], t2[0:1, 0:valid], t1[0:1, 0:valid])
            adds(t2[0:1, 0:valid], t2[0:1, 0:valid], ERF_A1)
            mul(t2[0:1, 0:valid], t2[0:1, 0:valid], t1[0:1, 0:valid])
            mul(t1[0:1, 0:valid], t3[0:1, 0:valid], t2[0:1, 0:valid])
            mul(t1[0:1, 0:valid], t1[0:1, 0:valid], exp_f32[0:1, 0:valid])
            muls(t2[0:1, 0:valid], x_f32[0:1, 0:valid], 0.5)
            add(exp_f32[0:1, 0:valid], t2[0:1, 0:valid], t3[0:1, 0:valid])
            sub(out_f32[0:1, 0:valid], exp_f32[0:1, 0:valid], t1[0:1, 0:valid])
            cast(dst[0:1, 0:valid], out_f32[0:1, 0:valid], round_mode=RoundMode.TO_EVEN)
            y[0:1, n0:n0 + valid] <<= dst[0:1, 0:valid]
    return y

@lru_cache(maxsize=2)
def make_gelu_erf_bf16_kernel(device):
    if device not in ("a2", "a3"):
        raise ValueError("This tensor-vector recipe supports A2/A3")
    api = importlib.import_module("ascriptor." + device)
    return api.kernel(mode="vec")(gelu_erf_bf16_kernel)

# ----------------------------------------------------------------------------------------------------
# swiglu_f32.py
# Original instruction order, DBuff lifetimes and explicit ties-even output cast.
# ----------------------------------------------------------------------------------------------------

TILE_SWIGLU_F32 = 8192

@func()
def _swi_glu_compute_f32(
    x0_f32: Tensor, x1_f32: Tensor, neg_f32: Tensor,
    exp_den_f32: Tensor, valid: Var, neg_beta: Var,
):
    """Fused V compute: 5 ops as one stage to avoid intermediate bar_v."""
    muls(neg_f32[0:1, 0:valid], x0_f32[0:1, 0:valid], neg_beta)
    exp(exp_den_f32[0:1, 0:valid], neg_f32[0:1, 0:valid])
    adds(exp_den_f32[0:1, 0:valid], exp_den_f32[0:1, 0:valid], 1.0)
    mul(x0_f32[0:1, 0:valid], x0_f32[0:1, 0:valid], x1_f32[0:1, 0:valid])
    div(x0_f32[0:1, 0:valid], x0_f32[0:1, 0:valid], exp_den_f32[0:1, 0:valid])

def swi_glu_float_kernel(
    x0: GM[f32, (1, 'n')], x1: GM[f32, (1, 'n')], y: GM[f32, (1, 'n')], n: i32, neg_beta: f32, tile_len: i32,
):
    x0_ub = DBuff(DT.float, [1, TILE_SWIGLU_F32], Position.UB)
    x1_ub = Tensor(DT.float, [1, TILE_SWIGLU_F32], Position.UB)
    neg_ub = Tensor(DT.float, [1, TILE_SWIGLU_F32], Position.UB)
    exp_den_ub = Tensor(DT.float, [1, TILE_SWIGLU_F32], Position.UB)
    n_tiles = CeilDiv(n, tile_len)
    tiles_per_core = CeilDiv(n_tiles, GetVecNum())
    tile_begin = Var(tiles_per_core * GetVecIdx())
    tile_end = Min(tile_begin + tiles_per_core, n_tiles)

    with auto_sync():
        for tile_idx in range(tile_begin, tile_end):
            n0 = Var(tile_idx * tile_len)
            valid = Min(tile_len, n - n0)
            x0_tile = x0_ub[tile_idx]

            x0_tile[0:1, 0:valid] <<= x0[0:1, n0:n0 + valid]
            x1_ub[0:1, 0:valid] <<= x1[0:1, n0:n0 + valid]

            _swi_glu_compute_f32(x0_tile, x1_ub, neg_ub, exp_den_ub, valid, neg_beta)

            y[0:1, n0:n0 + valid] <<= x0_tile[0:1, 0:valid]
    return y

@lru_cache(maxsize=2)
def make_swiglu_f32_kernel(device):
    if device not in ("a2", "a3"):
        raise ValueError("This tensor-vector recipe supports A2/A3")
    api = importlib.import_module("ascriptor." + device)
    return api.kernel(mode="vec")(swi_glu_float_kernel)

# ----------------------------------------------------------------------------------------------------
# swiglu_f16.py
# Original instruction order, DBuff lifetimes and explicit ties-even output cast.
# ----------------------------------------------------------------------------------------------------

TILE_SWIGLU_F16 = 6144


def swi_glu_half_kernel(
    x0: GM[f16, (1, 'n')], x1: GM[f16, (1, 'n')], y: GM[f16, (1, 'n')], n: i32, neg_beta: f32, tile_len: i32,
):
    x0_f16 = DBuff(DT.half, [1, TILE_SWIGLU_F16], Position.UB)
    x1_f16 = DBuff(DT.half, [1, TILE_SWIGLU_F16], Position.UB)
    out_f16 = DBuff(DT.half, [1, TILE_SWIGLU_F16], Position.UB)
    x0_f32 = Tensor(DT.float, [1, TILE_SWIGLU_F16], Position.UB)
    x1_f32 = Tensor(DT.float, [1, TILE_SWIGLU_F16], Position.UB)
    neg_f32 = Tensor(DT.float, [1, TILE_SWIGLU_F16], Position.UB)
    exp_den_f32 = Tensor(DT.float, [1, TILE_SWIGLU_F16], Position.UB)
    n_tiles = CeilDiv(n, tile_len)
    tiles_per_core = CeilDiv(n_tiles, GetVecNum())
    tile_begin = Var(tiles_per_core * GetVecIdx())
    tile_end = Min(tile_begin + tiles_per_core, n_tiles)

    with auto_sync():
        for tile_idx in range(tile_begin, tile_end):
            n0 = Var(tile_idx * tile_len)
            valid = Min(tile_len, n - n0)
            s0 = x0_f16[tile_idx]
            s1 = x1_f16[tile_idx]
            d = out_f16[tile_idx]

            s0[0:1, 0:valid] <<= x0[0:1, n0:n0 + valid]
            s1[0:1, 0:valid] <<= x1[0:1, n0:n0 + valid]

            cast(x0_f32[0:1, 0:valid], s0[0:1, 0:valid], round_mode=RoundMode.NONE)
            cast(x1_f32[0:1, 0:valid], s1[0:1, 0:valid], round_mode=RoundMode.NONE)

            _swi_glu_compute_f32(x0_f32, x1_f32, neg_f32, exp_den_f32, valid, neg_beta)

            cast(d[0:1, 0:valid], x0_f32[0:1, 0:valid], round_mode=RoundMode.TO_EVEN)
            y[0:1, n0:n0 + valid] <<= d[0:1, 0:valid]
    return y

@lru_cache(maxsize=2)
def make_swiglu_f16_kernel(device):
    if device not in ("a2", "a3"):
        raise ValueError("This tensor-vector recipe supports A2/A3")
    api = importlib.import_module("ascriptor." + device)
    return api.kernel(mode="vec")(swi_glu_half_kernel)

# ----------------------------------------------------------------------------------------------------
# swiglu_bf16.py
# Original instruction order, DBuff lifetimes and explicit ties-even output cast.
# ----------------------------------------------------------------------------------------------------

TILE_SWIGLU_BF16 = 6144


def swi_glu_bfloat16_kernel(
    x0: GM[bf16, (1, 'n')], x1: GM[bf16, (1, 'n')], y: GM[bf16, (1, 'n')], n: i32, neg_beta: f32, tile_len: i32,
):
    x0_bf16 = DBuff(DT.bfloat16, [1, TILE_SWIGLU_BF16], Position.UB)
    x1_bf16 = DBuff(DT.bfloat16, [1, TILE_SWIGLU_BF16], Position.UB)
    out_bf16 = DBuff(DT.bfloat16, [1, TILE_SWIGLU_BF16], Position.UB)
    x0_f32 = Tensor(DT.float, [1, TILE_SWIGLU_BF16], Position.UB)
    x1_f32 = Tensor(DT.float, [1, TILE_SWIGLU_BF16], Position.UB)
    neg_f32 = Tensor(DT.float, [1, TILE_SWIGLU_BF16], Position.UB)
    exp_den_f32 = Tensor(DT.float, [1, TILE_SWIGLU_BF16], Position.UB)
    n_tiles = CeilDiv(n, tile_len)
    tiles_per_core = CeilDiv(n_tiles, GetVecNum())
    tile_begin = Var(tiles_per_core * GetVecIdx())
    tile_end = Min(tile_begin + tiles_per_core, n_tiles)

    with auto_sync():
        for tile_idx in range(tile_begin, tile_end):
            n0 = Var(tile_idx * tile_len)
            valid = Min(tile_len, n - n0)
            s0 = x0_bf16[tile_idx]
            s1 = x1_bf16[tile_idx]
            d = out_bf16[tile_idx]

            s0[0:1, 0:valid] <<= x0[0:1, n0:n0 + valid]
            s1[0:1, 0:valid] <<= x1[0:1, n0:n0 + valid]

            cast(x0_f32[0:1, 0:valid], s0[0:1, 0:valid], round_mode=RoundMode.NONE)
            cast(x1_f32[0:1, 0:valid], s1[0:1, 0:valid], round_mode=RoundMode.NONE)

            _swi_glu_compute_f32(x0_f32, x1_f32, neg_f32, exp_den_f32, valid, neg_beta)

            cast(d[0:1, 0:valid], x0_f32[0:1, 0:valid], round_mode=RoundMode.TO_EVEN)
            y[0:1, n0:n0 + valid] <<= d[0:1, 0:valid]
    return y

@lru_cache(maxsize=2)
def make_swiglu_bf16_kernel(device):
    if device not in ("a2", "a3"):
        raise ValueError("This tensor-vector recipe supports A2/A3")
    api = importlib.import_module("ascriptor." + device)
    return api.kernel(mode="vec")(swi_glu_bfloat16_kernel)

# ----------------------------------------------------------------------------------------------------
# gated_dispatch.py
# Select one of the eight declared source modes. Each is a complete activation with its own
# storage dtype and UB tile capacity, and each was a separate source file whose kernel factory
# was called `kernel_for`; they are renamed here so one file can hold all eight. A case names a
# variant and changes nothing else, which is what makes the five GELU modes and the three SwiGLU
# precisions directly comparable.
# ----------------------------------------------------------------------------------------------------

VARIANTS = {"gelu_tanh_f32": "make_gelu_tanh_f32_kernel",
            "gelu_tanh_f16": "make_gelu_tanh_f16_kernel",
            "gelu_tanh_bf16": "make_gelu_tanh_bf16_kernel",
            "gelu_erf_f32": "make_gelu_erf_f32_kernel",
            "gelu_erf_bf16": "make_gelu_erf_bf16_kernel",
            "swiglu_f32": "make_swiglu_f32_kernel",
            "swiglu_f16": "make_swiglu_f16_kernel",
            "swiglu_bf16": "make_swiglu_bf16_kernel"}


def make_kernel(variant, device):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown activation variant: {variant}; declared: {', '.join(VARIANTS)}")
    return globals()[VARIANTS[variant]](device)
