# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A2/A3 vector-only BF16 HiFloat8 quantize-dequantize."""

from ascriptor.a2 import *
from functools import lru_cache
from importlib import import_module

TILE = 512
HIF8_OVERFLOW_THRESHOLD = 2.0 ** 15 * 1.25
HIF8_MIN_VALUE = 2.0 ** -23
HIF8_DML_SCALE = 2.0 ** -22
FLOAT32_FINITE_MAX = 3.4028234663852886e38
FP32_EXP_MASK = 0x7F800000
FP32_NEG_INF = 0xFF800000
EXP_ABS_BIAS = -0x00800000


@func()
def _init_constants(exp_mask, half_value, zero, one, neg_half, pos_inf, neg_inf):
    dup(exp_mask, FP32_EXP_MASK)
    dup(half_value, 0.5)
    dup(zero, 0.0)
    dup(one, 1.0)
    dup(neg_half, -0.5)
    pos_inf_bits = pos_inf.reinterpret(DT.uint32, name="hif8_pos_inf_bits")
    neg_inf_bits = neg_inf.reinterpret(DT.uint32, name="hif8_neg_inf_bits")
    dup(pos_inf_bits, FP32_EXP_MASK)
    dup(neg_inf_bits, FP32_NEG_INF)


@func()
def _quantize_tile(
    x_float, y_float, work, abs_buf, exp_buf, exp_abs_buf, scale,
    tmp1, tmp2, tmp3, round_int, exp_mask,
    finite_flag, keep_flag, overflow_flag, le15_flag, le7_flag, le3_flag,
    nonneg_flag, half_value, zero, one, neg_half, pos_inf, neg_inf,
):
    abs(abs_buf, x_float)
    compare_scalar(finite_flag, abs_buf, FLOAT32_FINITE_MAX, CompareMode.LE)
    select(work, finite_flag, x_float, zero, SelectMode.TENSOR_SCALAR)
    abs(abs_buf, work)

    x_u16 = work.reinterpret(DT.uint16, name="hif8_work_u16")
    exp_u16 = exp_buf.reinterpret(DT.uint16, name="hif8_exp_u16")
    exp_mask_u16 = exp_mask.reinterpret(DT.uint16, name="hif8_exp_mask_u16")
    vand(exp_u16, x_u16, exp_mask_u16)
    exp_abs_u16 = exp_abs_buf.reinterpret(DT.uint16, name="hif8_exp_abs_u16")
    vnot(exp_abs_u16, exp_u16)
    vand(exp_abs_u16, exp_abs_u16, exp_mask_u16)
    exp_abs_i32 = exp_abs_buf.reinterpret(DT.int, name="hif8_exp_abs_i32")
    adds(exp_abs_i32, exp_abs_i32, EXP_ABS_BIAS)
    vmax(exp_abs_buf, exp_abs_buf, exp_buf)

    compare_scalar(keep_flag, exp_buf, HIF8_MIN_VALUE, CompareMode.GE)
    compare_scalar(le15_flag, exp_abs_buf, 32768.0, CompareMode.LE)
    compare_scalar(le7_flag, exp_abs_buf, 128.0, CompareMode.LE)
    compare_scalar(le3_flag, exp_abs_buf, 8.0, CompareMode.LE)
    vmaxs(scale, exp_buf, HIF8_DML_SCALE)
    select(tmp1, le15_flag, half_value, one, SelectMode.TENSOR_SCALAR)
    select(tmp2, le7_flag, half_value, one, SelectMode.TENSOR_SCALAR)
    select(tmp3, le3_flag, half_value, one, SelectMode.TENSOR_SCALAR)
    mul(scale, scale, tmp1)
    mul(scale, scale, tmp2)
    mul(scale, scale, tmp3)

    div(y_float, work, scale)
    compare_scalar(nonneg_flag, y_float, 0.0, CompareMode.GE)
    select(tmp1, nonneg_flag, half_value, neg_half, SelectMode.TENSOR_SCALAR)
    add(y_float, y_float, tmp1)
    cast(round_int, y_float, round_mode=RoundMode.TRUNC)
    cast(y_float, round_int, round_mode=RoundMode.NONE)
    mul(y_float, y_float, scale)
    select(y_float, keep_flag, y_float, zero, SelectMode.TENSOR_SCALAR)

    compare_scalar(overflow_flag, abs_buf, HIF8_OVERFLOW_THRESHOLD, CompareMode.GE)
    compare_scalar(nonneg_flag, work, 0.0, CompareMode.GE)
    select(tmp2, nonneg_flag, pos_inf, neg_inf, SelectMode.TENSOR_SCALAR)
    select(tmp1, overflow_flag, tmp2, zero, SelectMode.TENSOR_SCALAR)
    compare_scalar(keep_flag, abs_buf, HIF8_OVERFLOW_THRESHOLD, CompareMode.LT)
    select(y_float, keep_flag, y_float, zero, SelectMode.TENSOR_SCALAR)
    add(y_float, y_float, tmp1)

    select(tmp1, finite_flag, y_float, zero, SelectMode.TENSOR_SCALAR)
    abs(abs_buf, x_float)
    compare_scalar(overflow_flag, abs_buf, FLOAT32_FINITE_MAX, CompareMode.GT)
    select(tmp2, overflow_flag, x_float, zero, SelectMode.TENSOR_SCALAR)
    finite_u16 = finite_flag.reinterpret(DT.uint16, name="hif8_finite_u16")
    inf_u16 = overflow_flag.reinterpret(DT.uint16, name="hif8_inf_u16")
    nan_u16 = le15_flag.reinterpret(DT.uint16, name="hif8_nan_u16")
    not_inf_u16 = le7_flag.reinterpret(DT.uint16, name="hif8_not_inf_u16")
    vnot(nan_u16, finite_u16)
    vnot(not_inf_u16, inf_u16)
    vand(nan_u16, nan_u16, not_inf_u16)
    select(tmp3, le15_flag, x_float, zero, SelectMode.TENSOR_SCALAR)
    add(y_float, tmp1, tmp2)
    add(y_float, y_float, tmp3)


def hif8_kernel_bf16(x: GM[bf16, (1, "TX")], y: GM[bf16, (1, "TY")], total: i32):
    x_bf16 = DBuff(DT.bfloat16, [1, TILE], Position.UB)
    y_bf16 = DBuff(DT.bfloat16, [1, TILE], Position.UB)
    x_float = Tensor(DT.float, [1, TILE], Position.UB)
    y_float = Tensor(DT.float, [1, TILE], Position.UB)
    work = Tensor(DT.float, [1, TILE], Position.UB)
    abs_buf = Tensor(DT.float, [1, TILE], Position.UB)
    exp_buf = Tensor(DT.float, [1, TILE], Position.UB)
    exp_abs_buf = Tensor(DT.float, [1, TILE], Position.UB)
    scale = Tensor(DT.float, [1, TILE], Position.UB)
    tmp1 = Tensor(DT.float, [1, TILE], Position.UB)
    tmp2 = Tensor(DT.float, [1, TILE], Position.UB)
    tmp3 = Tensor(DT.float, [1, TILE], Position.UB)
    round_int = Tensor(DT.int, [1, TILE], Position.UB)
    exp_mask = Tensor(DT.uint32, [1, TILE], Position.UB)
    finite_flag = Tensor(DT.uint8, [1, TILE], Position.UB)
    keep_flag = Tensor(DT.uint8, [1, TILE], Position.UB)
    overflow_flag = Tensor(DT.uint8, [1, TILE], Position.UB)
    le15_flag = Tensor(DT.uint8, [1, TILE], Position.UB)
    le7_flag = Tensor(DT.uint8, [1, TILE], Position.UB)
    le3_flag = Tensor(DT.uint8, [1, TILE], Position.UB)
    nonneg_flag = Tensor(DT.uint8, [1, TILE], Position.UB)
    half_value = Tensor(DT.float, [1, TILE], Position.UB)
    zero = Tensor(DT.float, [1, TILE], Position.UB)
    one = Tensor(DT.float, [1, TILE], Position.UB)
    neg_half = Tensor(DT.float, [1, TILE], Position.UB)
    pos_inf = Tensor(DT.float, [1, TILE], Position.UB)
    neg_inf = Tensor(DT.float, [1, TILE], Position.UB)

    n_tiles = CeilDiv(total, TILE)
    tiles_per_core = CeilDiv(n_tiles, GetVecNum())
    tile_begin = Var(tiles_per_core * GetVecIdx())
    tile_end = Min(tile_begin + tiles_per_core, n_tiles)
    _init_constants(exp_mask, half_value, zero, one, neg_half, pos_inf, neg_inf)

    with auto_sync():
        for tile_idx in range(tile_begin, tile_end):
            offset = Var(tile_idx * TILE)
            valid = Min(TILE, total - offset)
            src = x_bf16[tile_idx]
            dst = y_bf16[tile_idx]
            src[0:1, 0:valid] <<= x[0:1, offset:offset + valid]
            cast(x_float[0:1, 0:valid], src[0:1, 0:valid], round_mode=RoundMode.NONE)
            _quantize_tile(
                x_float, y_float, work, abs_buf, exp_buf, exp_abs_buf, scale,
                tmp1, tmp2, tmp3, round_int, exp_mask,
                finite_flag, keep_flag, overflow_flag, le15_flag, le7_flag, le3_flag,
                nonneg_flag, half_value, zero, one, neg_half, pos_inf, neg_inf,
            )
            cast(dst[0:1, 0:valid], y_float[0:1, 0:valid], round_mode=RoundMode.TO_EVEN)
            y[0:1, offset:offset + valid] <<= dst[0:1, 0:valid]
    return y


@lru_cache(maxsize=2)
def build_kernel(device):
    if device not in ("a2", "a3"):
        raise ValueError("Shared tensor-vector source supports A2/A3")
    return import_module(f"ascriptor.{device}").kernel(mode="vec")(hif8_kernel_bf16)
