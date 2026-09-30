# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Three reconstructed grouped FP32 formats on one A2/A3 vector body: per-32 MXFP4 E2M1, two-level MBS-MXFP4, and per-32 MXFP8 E5M2."""

from ascriptor.a2 import *
from functools import lru_cache
from importlib import import_module

# ----------------------------------------------------------------------------------------------------
# mbs_helpers.py
# Shared vector primitives for the reconstructed FP32 MX formats.
# ----------------------------------------------------------------------------------------------------

MACRO_GROUP = 128
INNER_GROUP = 32
INNER_PER_MACRO = 4
MACRO_REP_STRIDE = 16
UB_MACRO_COLS = 128
SCALE_SLOTS = 64
SCALE_BCAST_COLS = 8
GROUP_FLAG_COLS = 512
E8M0_MIN_VALUE = 2.0 ** -127
FP32_EXP_MASK = 0x7F800000
MXFP4_FLOOR_SCALE_INPUT_FACTOR = 0.25
MACRO_FACTOR_TARGET = 6.0


@func()
def _e2m1_step(abs_norm: Tensor, q_abs: Tensor, flag: Tensor, const_full: Tensor,
                groups: Var, rep_stride: Var, threshold: Var, value: Var):
    dup(const_full, value, repeat=groups, dst_rep_stride=rep_stride, count_per_rep=INNER_GROUP)
    compare_scalar(flag, abs_norm, threshold, CompareMode.LT, repeat=groups,
                   dst_rep_stride=rep_stride, src1_rep_stride=rep_stride)
    select(q_abs, flag, q_abs, const_full, SelectMode.TENSOR_SCALAR, repeat=groups,
           dst_rep_stride=rep_stride, src1_rep_stride=rep_stride, src2_rep_stride=rep_stride)


@func()
def quantize_abs_to_e2m1_g32(abs_norm: Tensor, q_abs: Tensor, flag: Tensor,
                              const_full: Tensor, groups: Var, rep_stride: Var):
    dup(q_abs, 0.0, repeat=groups, dst_rep_stride=rep_stride, count_per_rep=INNER_GROUP)
    _e2m1_step(abs_norm, q_abs, flag, const_full, groups, rep_stride, 0.25, 0.5)
    _e2m1_step(abs_norm, q_abs, flag, const_full, groups, rep_stride, 0.75, 1.0)
    _e2m1_step(abs_norm, q_abs, flag, const_full, groups, rep_stride, 1.25, 1.5)
    _e2m1_step(abs_norm, q_abs, flag, const_full, groups, rep_stride, 1.75, 2.0)
    _e2m1_step(abs_norm, q_abs, flag, const_full, groups, rep_stride, 2.5, 3.0)
    _e2m1_step(abs_norm, q_abs, flag, const_full, groups, rep_stride, 3.5, 4.0)
    _e2m1_step(abs_norm, q_abs, flag, const_full, groups, rep_stride, 5.0, 6.0)


@func()
def build_e8m0_scale_s(scale_input_s: Tensor, scale_s: Tensor, exp_s: Tensor,
                        exp_mask_i32: Tensor, min_scale_s: Tensor, count: Var):
    scale_input_i16 = scale_input_s.reinterpret(DT.int16, name="scale_input_i16")
    exp_i16 = exp_s.reinterpret(DT.int16, name="scale_exp_i16")
    exp_mask_i16 = exp_mask_i32.reinterpret(DT.int16, name="scale_exp_mask_i16")
    vand(exp_i16[0:1, 0:2 * count], scale_input_i16[0:1, 0:2 * count],
         exp_mask_i16[0:1, 0:2 * count])
    vmax(scale_s[0:1, 0:count], exp_s[0:1, 0:count], min_scale_s[0:1, 0:count])


@func()
def build_e0m8_macro_factor(amax_s: Tensor, factor_s: Tensor, recip_s: Tensor,
                             safe_s: Tensor, scalar_flag: Tensor, one_s: Tensor,
                             macro_tgt_s: Tensor, exp_mask_i32: Tensor, macros: Var):
    safe_s[0:1, 0:macros] <<= amax_s[0:1, 0:macros]
    compare_scalar(scalar_flag[0:1, 0:macros], safe_s[0:1, 0:macros], 0.0, CompareMode.NE)
    select(safe_s[0:1, 0:macros], scalar_flag[0:1, 0:macros], safe_s[0:1, 0:macros],
           one_s[0:1, 0:macros], SelectMode.TENSOR_SCALAR)
    div(recip_s[0:1, 0:macros], macro_tgt_s[0:1, 0:macros], safe_s[0:1, 0:macros])
    recip_i16 = recip_s.reinterpret(DT.int16, name="macro_recip_i16")
    factor_i16 = factor_s.reinterpret(DT.int16, name="macro_factor_i16")
    exp_mask_i16 = exp_mask_i32.reinterpret(DT.int16, name="macro_exp_mask_i16")
    recip_i32 = recip_s.reinterpret(DT.int, name="macro_recip_i32")
    vand(factor_i16[0:1, 0:2 * macros], recip_i16[0:1, 0:2 * macros],
         exp_mask_i16[0:1, 0:2 * macros])
    div(safe_s[0:1, 0:macros], recip_s[0:1, 0:macros], factor_s[0:1, 0:macros])
    adds(safe_s[0:1, 0:macros], safe_s[0:1, 0:macros], -1.0)
    muls(safe_s[0:1, 0:macros], safe_s[0:1, 0:macros], 256.0)
    cast(recip_i32[0:1, 0:macros], safe_s[0:1, 0:macros], round_mode=RoundMode.TRUNC)
    cast(safe_s[0:1, 0:macros], recip_i32[0:1, 0:macros])
    muls(safe_s[0:1, 0:macros], safe_s[0:1, 0:macros], 1.0 / 256.0)
    adds(factor_s[0:1, 0:macros], safe_s[0:1, 0:macros], 1.0)
    compare_scalar(scalar_flag[0:1, 0:macros], amax_s[0:1, 0:macros], 0.0, CompareMode.NE)
    select(factor_s[0:1, 0:macros], scalar_flag[0:1, 0:macros], factor_s[0:1, 0:macros],
           one_s[0:1, 0:macros], SelectMode.TENSOR_SCALAR)

# ----------------------------------------------------------------------------------------------------
# mxfp4.py
# Plain per-32 MXFP4 E2M1 FP32 vector kernel.
# ----------------------------------------------------------------------------------------------------

GROUP = INNER_GROUP
ROW_PAD = 64
REP_F = 8
FLAG_COLS = 256
TILE_GROUPS = 32
UB_GUARD = 33
MXFP4_SCALE_BCAST_ROWS = 40


@func()
def _mxfp4_group_core(xt: Tensor, yt: Tensor, valid: Var, gabs: Tensor, q_abs: Tensor,
                       neg_q_abs: Tensor, const_full: Tensor, flag: Tensor, amax_s: Tensor,
                       scale_input_s: Tensor, scale_s: Tensor, exp_s: Tensor,
                       scale_bcast: Tensor, min_scale_s: Tensor, exp_mask_i32: Tensor):
    abs(gabs[0:valid, 0:GROUP], xt[0:valid, 0:GROUP], repeat=valid,
        dst_rep_stride=REP_F, src_rep_stride=REP_F, count_per_rep=GROUP)
    cmax(amax_s, gabs[0:valid, 0:GROUP], repeat=valid, src_rep_stride=REP_F, count_per_rep=GROUP)
    muls(scale_input_s[0:1, 0:valid], amax_s[0:1, 0:valid], MXFP4_FLOOR_SCALE_INPUT_FACTOR)
    build_e8m0_scale_s(scale_input_s, scale_s, exp_s, exp_mask_i32, min_scale_s, valid)
    brcb(scale_bcast, scale_s, repeat=CeilDiv(valid, 8), dst_blk_stride=1, dst_rep_stride=8)
    div(xt[0:valid, 0:GROUP], xt[0:valid, 0:GROUP], scale_bcast[0:valid, 0:SCALE_BCAST_COLS],
        repeat=valid, dst_rep_stride=REP_F, src1_rep_stride=REP_F, src2_rep_stride=1,
        count_per_rep=GROUP)
    abs(gabs[0:valid, 0:GROUP], xt[0:valid, 0:GROUP], repeat=valid,
        dst_rep_stride=REP_F, src_rep_stride=REP_F, count_per_rep=GROUP)
    quantize_abs_to_e2m1_g32(gabs[0:valid, 0:GROUP], q_abs[0:valid, 0:GROUP],
                             flag[0:valid, 0:GROUP], const_full[0:valid, 0:GROUP], valid, REP_F)
    dup(const_full[0:valid, 0:GROUP], 0.0, repeat=valid, dst_rep_stride=REP_F, count_per_rep=GROUP)
    muls(neg_q_abs[0:valid, 0:GROUP], q_abs[0:valid, 0:GROUP], -1.0,
         repeat=valid, dst_rep_stride=REP_F, src_rep_stride=REP_F, count_per_rep=GROUP)
    compare_scalar(flag[0:valid, 0:GROUP], xt[0:valid, 0:GROUP], 0.0, CompareMode.GE,
                   repeat=valid, dst_rep_stride=REP_F, src1_rep_stride=REP_F)
    select(gabs[0:valid, 0:GROUP], flag[0:valid, 0:GROUP], q_abs[0:valid, 0:GROUP],
           const_full[0:valid, 0:GROUP], SelectMode.TENSOR_SCALAR, repeat=valid,
           dst_rep_stride=REP_F, src1_rep_stride=REP_F, src2_rep_stride=REP_F)
    compare_scalar(flag[0:valid, 0:GROUP], xt[0:valid, 0:GROUP], 0.0, CompareMode.LT,
                   repeat=valid, dst_rep_stride=REP_F, src1_rep_stride=REP_F)
    select(neg_q_abs[0:valid, 0:GROUP], flag[0:valid, 0:GROUP], neg_q_abs[0:valid, 0:GROUP],
           const_full[0:valid, 0:GROUP], SelectMode.TENSOR_SCALAR, repeat=valid,
           dst_rep_stride=REP_F, src1_rep_stride=REP_F, src2_rep_stride=REP_F)
    add(xt[0:valid, 0:GROUP], gabs[0:valid, 0:GROUP], neg_q_abs[0:valid, 0:GROUP],
        repeat=valid, dst_rep_stride=REP_F, src1_rep_stride=REP_F, src2_rep_stride=REP_F,
        count_per_rep=GROUP)
    mul(yt[0:valid, 0:GROUP], xt[0:valid, 0:GROUP], scale_bcast[0:valid, 0:SCALE_BCAST_COLS],
        repeat=valid, dst_rep_stride=REP_F, src1_rep_stride=REP_F, src2_rep_stride=1,
        count_per_rep=GROUP)


def mxfp4_kernel_fp32(x: GM[f32, ("R", "C")], y: GM[f32, ("RY", "CY")], rows: i32, cols: i32):
    groups_per_row = Var(cols // GROUP)
    total_groups = Var(rows * groups_per_row)
    x_g = x.reshape([total_groups, GROUP], name="x_g")
    y_g = y.reshape([total_groups, GROUP], name="y_g")
    x_work = DBuff(DT.float, [UB_GUARD, ROW_PAD], Position.UB)
    y_work = DBuff(DT.float, [UB_GUARD, ROW_PAD], Position.UB)
    gabs = Tensor(DT.float, [UB_GUARD, ROW_PAD], Position.UB)
    q_abs = Tensor(DT.float, [UB_GUARD, ROW_PAD], Position.UB)
    neg_q_abs = Tensor(DT.float, [UB_GUARD, ROW_PAD], Position.UB)
    const_full = Tensor(DT.float, [UB_GUARD, ROW_PAD], Position.UB)
    flag = Tensor(DT.uint8, [UB_GUARD, FLAG_COLS], Position.UB)
    amax_s = Tensor(DT.float, [1, SCALE_SLOTS], Position.UB)
    scale_input_s = Tensor(DT.float, [1, SCALE_SLOTS], Position.UB)
    scale_s = Tensor(DT.float, [1, SCALE_SLOTS], Position.UB)
    exp_s = Tensor(DT.float, [1, SCALE_SLOTS], Position.UB)
    scale_bcast = Tensor(DT.float, [MXFP4_SCALE_BCAST_ROWS, SCALE_BCAST_COLS], Position.UB)
    min_scale_s = Tensor(DT.float, [1, SCALE_SLOTS], Position.UB)
    exp_mask_i32 = Tensor(DT.int, [1, SCALE_SLOTS], Position.UB)
    n_tiles = CeilDiv(total_groups, TILE_GROUPS)
    tiles_per_core = CeilDiv(n_tiles, GetVecNum())
    tile_begin = Var(tiles_per_core * GetVecIdx())
    tile_end = Min(tile_begin + tiles_per_core, n_tiles)
    dup(min_scale_s, E8M0_MIN_VALUE)
    dup(exp_mask_i32, FP32_EXP_MASK)
    with auto_sync():
        set_mask_normal()
        reset_mask()
        for tile_idx in range(tile_begin, tile_end):
            g0 = Var(tile_idx * TILE_GROUPS)
            valid = Min(TILE_GROUPS, total_groups - g0)
            xt = x_work[tile_idx]
            yt = y_work[tile_idx]
            gm_to_ub_pad(xt[0:valid, 0:GROUP], x_g[g0:g0 + valid, 0:GROUP], n_burst=valid,
                         burst_len_element=GROUP, src_stride_element=0,
                         dst_stride=(ROW_PAD - GROUP) // xt.dtype.C0)
            _mxfp4_group_core(xt, yt, valid, gabs, q_abs, neg_q_abs, const_full, flag,
                              amax_s, scale_input_s, scale_s, exp_s, scale_bcast,
                              min_scale_s, exp_mask_i32)
            ub_to_gm_pad(y_g[g0:g0 + valid, 0:GROUP], yt[0:valid, 0:GROUP], n_burst=valid,
                         burst_len_element=GROUP, src_stride=(ROW_PAD - GROUP) // yt.dtype.C0,
                         dst_stride_element=0)
            bar_mte3()
    return y


@lru_cache(maxsize=2)
def build_mxfp4_kernel(device):
    if device not in ("a2", "a3"):
        raise ValueError("Shared tensor-vector source supports A2/A3")
    return import_module(f"ascriptor.{device}").kernel(mode="vec")(mxfp4_kernel_fp32)

# ----------------------------------------------------------------------------------------------------
# mbs.py
# Two-level MBS-MXFP4 FP32 vector kernel.
# ----------------------------------------------------------------------------------------------------

TILE_MACROS = 32
UB_GUARD_MACROS = 33
MBS_SCALE_BCAST_ROWS = 32


@func()
def _quantize_inner(x_float_t: Tensor, macro_abs: Tensor, q_abs: Tensor,
                     neg_q_abs: Tensor, const_full: Tensor, group_flag: Tensor,
                     group_amax_s: Tensor, group_scale_input_s: Tensor,
                     group_scale_s: Tensor, group_exp_s: Tensor,
                     group_scale_bcast: Tensor, exp_mask_i32: Tensor,
                     min_scale_s: Tensor, valid_macros: Var, group_col0: Var):
    abs(macro_abs[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
        x_float_t[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
        repeat=valid_macros, dst_rep_stride=MACRO_REP_STRIDE,
        src_rep_stride=MACRO_REP_STRIDE, count_per_rep=INNER_GROUP)
    cmax(group_amax_s, macro_abs[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
         repeat=valid_macros, src_rep_stride=MACRO_REP_STRIDE, count_per_rep=INNER_GROUP)
    muls(group_scale_input_s[0:1, 0:valid_macros], group_amax_s[0:1, 0:valid_macros],
         MXFP4_FLOOR_SCALE_INPUT_FACTOR)
    build_e8m0_scale_s(group_scale_input_s, group_scale_s, group_exp_s,
                        exp_mask_i32, min_scale_s, valid_macros)
    brcb(group_scale_bcast, group_scale_s, repeat=CeilDiv(valid_macros, 8),
         dst_blk_stride=1, dst_rep_stride=8)
    div(x_float_t[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
        x_float_t[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
        group_scale_bcast[0:valid_macros, 0:SCALE_BCAST_COLS], repeat=valid_macros,
        dst_rep_stride=MACRO_REP_STRIDE, src1_rep_stride=MACRO_REP_STRIDE,
        src2_rep_stride=1, count_per_rep=INNER_GROUP)
    abs(macro_abs[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
        x_float_t[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
        repeat=valid_macros, dst_rep_stride=MACRO_REP_STRIDE,
        src_rep_stride=MACRO_REP_STRIDE, count_per_rep=INNER_GROUP)
    quantize_abs_to_e2m1_g32(
        macro_abs[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
        q_abs[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
        group_flag[0:valid_macros, 0:INNER_GROUP],
        const_full[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
        valid_macros, MACRO_REP_STRIDE)
    dup(const_full[0:valid_macros, group_col0:group_col0 + INNER_GROUP], 0.0,
        repeat=valid_macros, dst_rep_stride=MACRO_REP_STRIDE, count_per_rep=INNER_GROUP)
    muls(neg_q_abs[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
         q_abs[0:valid_macros, group_col0:group_col0 + INNER_GROUP], -1.0,
         repeat=valid_macros, dst_rep_stride=MACRO_REP_STRIDE,
         src_rep_stride=MACRO_REP_STRIDE, count_per_rep=INNER_GROUP)
    compare_scalar(group_flag[0:valid_macros, 0:INNER_GROUP],
                   x_float_t[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
                   0.0, CompareMode.GE, repeat=valid_macros,
                   dst_rep_stride=MACRO_REP_STRIDE, src1_rep_stride=MACRO_REP_STRIDE)
    select(macro_abs[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
           group_flag[0:valid_macros, 0:INNER_GROUP],
           q_abs[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
           const_full[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
           SelectMode.TENSOR_SCALAR, repeat=valid_macros, dst_rep_stride=MACRO_REP_STRIDE,
           src1_rep_stride=MACRO_REP_STRIDE, src2_rep_stride=MACRO_REP_STRIDE)
    compare_scalar(group_flag[0:valid_macros, 0:INNER_GROUP],
                   x_float_t[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
                   0.0, CompareMode.LT, repeat=valid_macros,
                   dst_rep_stride=MACRO_REP_STRIDE, src1_rep_stride=MACRO_REP_STRIDE)
    select(neg_q_abs[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
           group_flag[0:valid_macros, 0:INNER_GROUP],
           neg_q_abs[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
           const_full[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
           SelectMode.TENSOR_SCALAR, repeat=valid_macros, dst_rep_stride=MACRO_REP_STRIDE,
           src1_rep_stride=MACRO_REP_STRIDE, src2_rep_stride=MACRO_REP_STRIDE)
    add(x_float_t[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
        macro_abs[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
        neg_q_abs[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
        repeat=valid_macros, dst_rep_stride=MACRO_REP_STRIDE,
        src1_rep_stride=MACRO_REP_STRIDE, src2_rep_stride=MACRO_REP_STRIDE,
        count_per_rep=INNER_GROUP)
    mul(x_float_t[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
        x_float_t[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
        group_scale_bcast[0:valid_macros, 0:SCALE_BCAST_COLS], repeat=valid_macros,
        dst_rep_stride=MACRO_REP_STRIDE, src1_rep_stride=MACRO_REP_STRIDE,
        src2_rep_stride=1, count_per_rep=INNER_GROUP)


def mbs_mxfp4_kernel_fp32(x: GM[f32, ("R", "C")], y: GM[f32, ("RY", "CY")], rows: i32, cols: i32):
    macros_per_row = Var(cols // MACRO_GROUP)
    total_macros = Var(rows * macros_per_row)
    x_macro = x.reshape([total_macros, MACRO_GROUP], name="x_macro")
    y_macro = y.reshape([total_macros, MACRO_GROUP], name="y_macro")
    x_float = DBuff(DT.float, [UB_GUARD_MACROS, UB_MACRO_COLS], Position.UB)
    y_float = DBuff(DT.float, [UB_GUARD_MACROS, UB_MACRO_COLS], Position.UB)
    macro_abs = Tensor(DT.float, [UB_GUARD_MACROS, UB_MACRO_COLS], Position.UB)
    q_abs = Tensor(DT.float, [UB_GUARD_MACROS, UB_MACRO_COLS], Position.UB)
    neg_q_abs = Tensor(DT.float, [UB_GUARD_MACROS, UB_MACRO_COLS], Position.UB)
    const_full = Tensor(DT.float, [UB_GUARD_MACROS, UB_MACRO_COLS], Position.UB)
    macro_left_s = Tensor(DT.float, [1, SCALE_SLOTS], Position.UB)
    macro_right_s = Tensor(DT.float, [1, SCALE_SLOTS], Position.UB)
    macro_amax_s = Tensor(DT.float, [1, SCALE_SLOTS], Position.UB)
    macro_safe_s = Tensor(DT.float, [1, SCALE_SLOTS], Position.UB)
    macro_recip_s = Tensor(DT.float, [1, SCALE_SLOTS], Position.UB)
    macro_factor_s = Tensor(DT.float, [1, SCALE_SLOTS], Position.UB)
    macro_factor_bcast = Tensor(DT.float, [MBS_SCALE_BCAST_ROWS, SCALE_BCAST_COLS], Position.UB)
    group_amax_s = Tensor(DT.float, [1, SCALE_SLOTS], Position.UB)
    group_scale_input_s = Tensor(DT.float, [1, SCALE_SLOTS], Position.UB)
    group_scale_s = Tensor(DT.float, [1, SCALE_SLOTS], Position.UB)
    group_exp_s = Tensor(DT.float, [1, SCALE_SLOTS], Position.UB)
    group_scale_bcast = Tensor(DT.float, [MBS_SCALE_BCAST_ROWS, SCALE_BCAST_COLS], Position.UB)
    scalar_flag = Tensor(DT.uint8, [1, SCALE_SLOTS], Position.UB)
    group_flag = Tensor(DT.uint8, [UB_GUARD_MACROS, GROUP_FLAG_COLS], Position.UB)
    one_s = Tensor(DT.float, [1, SCALE_SLOTS], Position.UB)
    macro_tgt_s = Tensor(DT.float, [1, SCALE_SLOTS], Position.UB)
    min_scale_s = Tensor(DT.float, [1, SCALE_SLOTS], Position.UB)
    exp_mask_i32 = Tensor(DT.int, [1, SCALE_SLOTS], Position.UB)
    n_tiles = CeilDiv(total_macros, TILE_MACROS)
    tiles_per_core = CeilDiv(n_tiles, GetVecNum())
    tile_begin = Var(tiles_per_core * GetVecIdx())
    tile_end = Min(tile_begin + tiles_per_core, n_tiles)
    dup(one_s, 1.0)
    dup(macro_tgt_s, MACRO_FACTOR_TARGET)
    dup(min_scale_s, E8M0_MIN_VALUE)
    dup(exp_mask_i32, FP32_EXP_MASK)
    with auto_sync():
        for tile_idx in range(tile_begin, tile_end):
            macro0 = Var(tile_idx * TILE_MACROS)
            valid_macros = Min(TILE_MACROS, total_macros - macro0)
            x_float_t = x_float[tile_idx]
            y_float_t = y_float[tile_idx]
            x_float_t[0:valid_macros, 0:MACRO_GROUP] <<= x_macro[macro0:macro0 + valid_macros, 0:MACRO_GROUP]
            abs(macro_abs[0:valid_macros, 0:MACRO_GROUP], x_float_t[0:valid_macros, 0:MACRO_GROUP],
                repeat=2 * valid_macros)
            cmax(macro_left_s, macro_abs[0:valid_macros, 0:64], repeat=valid_macros)
            cmax(macro_right_s, macro_abs[0:valid_macros, 64:128], repeat=valid_macros)
            vmax(macro_amax_s[0:1, 0:valid_macros], macro_left_s[0:1, 0:valid_macros],
                 macro_right_s[0:1, 0:valid_macros])
            build_e0m8_macro_factor(macro_amax_s, macro_factor_s, macro_recip_s,
                                     macro_safe_s, scalar_flag, one_s, macro_tgt_s,
                                     exp_mask_i32, valid_macros)
            brcb(macro_factor_bcast, macro_factor_s, repeat=CeilDiv(valid_macros, 8),
                 dst_blk_stride=1, dst_rep_stride=8)
            mul(x_float_t[0:valid_macros, 0:64], x_float_t[0:valid_macros, 0:64],
                macro_factor_bcast[0:valid_macros, 0:SCALE_BCAST_COLS], repeat=valid_macros)
            mul(x_float_t[0:valid_macros, 64:128], x_float_t[0:valid_macros, 64:128],
                macro_factor_bcast[0:valid_macros, 0:SCALE_BCAST_COLS], repeat=valid_macros)
            for inner_idx in range(0, INNER_PER_MACRO):
                group_col0 = Var(inner_idx * INNER_GROUP)
                _quantize_inner(x_float_t, macro_abs, q_abs, neg_q_abs, const_full,
                                group_flag, group_amax_s, group_scale_input_s, group_scale_s,
                                group_exp_s, group_scale_bcast, exp_mask_i32, min_scale_s,
                                valid_macros, group_col0)
            div(y_float_t[0:valid_macros, 0:64], x_float_t[0:valid_macros, 0:64],
                macro_factor_bcast[0:valid_macros, 0:SCALE_BCAST_COLS], repeat=valid_macros)
            div(y_float_t[0:valid_macros, 64:128], x_float_t[0:valid_macros, 64:128],
                macro_factor_bcast[0:valid_macros, 0:SCALE_BCAST_COLS], repeat=valid_macros)
            y_macro[macro0:macro0 + valid_macros, 0:MACRO_GROUP] <<= y_float_t[0:valid_macros, 0:MACRO_GROUP]
            bar_mte3()
    return y


@lru_cache(maxsize=2)
def build_mbs_mxfp4_kernel(device):
    if device not in ("a2", "a3"):
        raise ValueError("Shared tensor-vector source supports A2/A3")
    return import_module(f"ascriptor.{device}").kernel(mode="vec")(mbs_mxfp4_kernel_fp32)

# ----------------------------------------------------------------------------------------------------
# mxfp8e5m2.py
# Per-32 MXFP8 E5M2 FP32 vector kernel.
# ----------------------------------------------------------------------------------------------------

E5M2_SCALE_BCAST_ROWS = 40
E5M2_UPPER_COEF = 1.75
E5M2_PRIVEXP_FLOOR = 1.0 / 536870912.0
MAN_STEP_COEF = 0.25
E5M2_EPS = 5.421011e-20


@func()
def _mxfp8e5m2_group_core(xt: Tensor, yt: Tensor, valid: Var, pexp: Tensor,
                           privexp: Tensor, grpexp_s: Tensor, bound_s: Tensor,
                           scale_bcast: Tensor, exp_mask_i32: Tensor):
    abs(pexp[0:valid, 0:GROUP], xt[0:valid, 0:GROUP], repeat=valid,
        dst_rep_stride=REP_F, src_rep_stride=REP_F, count_per_rep=GROUP)
    pexp_i16 = pexp.reinterpret(DT.int16, name="pexp_i16")
    exp_mask_i16 = exp_mask_i32.reinterpret(DT.int16, name="e5m2_exp_mask_i16")
    vand(pexp_i16[0:valid, 0:2 * GROUP], pexp_i16[0:valid, 0:2 * GROUP],
         exp_mask_i16[0:1, 0:2 * GROUP], repeat=valid, dst_rep_stride=REP_F,
         src1_rep_stride=REP_F, src2_rep_stride=0, count_per_rep=2 * GROUP)
    cmax(grpexp_s, pexp[0:valid, 0:GROUP], repeat=valid,
         src_rep_stride=REP_F, count_per_rep=GROUP)
    muls(bound_s[0:1, 0:valid], grpexp_s[0:1, 0:valid], E5M2_PRIVEXP_FLOOR)
    brcb(scale_bcast, bound_s, repeat=CeilDiv(valid, 8), dst_blk_stride=1, dst_rep_stride=8)
    vmax(privexp[0:valid, 0:GROUP], pexp[0:valid, 0:GROUP],
         scale_bcast[0:valid, 0:SCALE_BCAST_COLS], repeat=valid,
         dst_rep_stride=REP_F, src1_rep_stride=REP_F, src2_rep_stride=1,
         count_per_rep=GROUP)
    muls(privexp[0:valid, 0:GROUP], privexp[0:valid, 0:GROUP], MAN_STEP_COEF,
         repeat=valid, dst_rep_stride=REP_F, src_rep_stride=REP_F, count_per_rep=GROUP)
    adds(privexp[0:valid, 0:GROUP], privexp[0:valid, 0:GROUP], E5M2_EPS,
         repeat=valid, dst_rep_stride=REP_F, src_rep_stride=REP_F, count_per_rep=GROUP)
    div(yt[0:valid, 0:GROUP], xt[0:valid, 0:GROUP], privexp[0:valid, 0:GROUP],
        repeat=valid, dst_rep_stride=REP_F, src1_rep_stride=REP_F,
        src2_rep_stride=REP_F, count_per_rep=GROUP)
    cast(yt[0:valid, 0:GROUP], yt[0:valid, 0:GROUP], repeat=valid,
         dst_rep_stride=REP_F, src_rep_stride=REP_F, count_per_rep=GROUP,
         round_mode=RoundMode.AWAY_FROM_ZERO)
    mul(yt[0:valid, 0:GROUP], yt[0:valid, 0:GROUP], privexp[0:valid, 0:GROUP],
        repeat=valid, dst_rep_stride=REP_F, src1_rep_stride=REP_F,
        src2_rep_stride=REP_F, count_per_rep=GROUP)
    muls(bound_s[0:1, 0:valid], grpexp_s[0:1, 0:valid], E5M2_UPPER_COEF)
    brcb(scale_bcast, bound_s, repeat=CeilDiv(valid, 8), dst_blk_stride=1, dst_rep_stride=8)
    vmin(yt[0:valid, 0:GROUP], yt[0:valid, 0:GROUP],
         scale_bcast[0:valid, 0:SCALE_BCAST_COLS], repeat=valid,
         dst_rep_stride=REP_F, src1_rep_stride=REP_F, src2_rep_stride=1,
         count_per_rep=GROUP)
    muls(bound_s[0:1, 0:valid], grpexp_s[0:1, 0:valid], -E5M2_UPPER_COEF)
    brcb(scale_bcast, bound_s, repeat=CeilDiv(valid, 8), dst_blk_stride=1, dst_rep_stride=8)
    vmax(yt[0:valid, 0:GROUP], yt[0:valid, 0:GROUP],
         scale_bcast[0:valid, 0:SCALE_BCAST_COLS], repeat=valid,
         dst_rep_stride=REP_F, src1_rep_stride=REP_F, src2_rep_stride=1,
         count_per_rep=GROUP)


def mxfp8e5m2_kernel_fp32(x: GM[f32, ("R", "C")], y: GM[f32, ("RY", "CY")], rows: i32, cols: i32):
    groups_per_row = Var(cols // GROUP)
    total_groups = Var(rows * groups_per_row)
    x_g = x.reshape([total_groups, GROUP], name="x_g")
    y_g = y.reshape([total_groups, GROUP], name="y_g")
    x_work = DBuff(DT.float, [UB_GUARD, ROW_PAD], Position.UB)
    y_work = DBuff(DT.float, [UB_GUARD, ROW_PAD], Position.UB)
    pexp = Tensor(DT.float, [UB_GUARD, ROW_PAD], Position.UB)
    privexp = Tensor(DT.float, [UB_GUARD, ROW_PAD], Position.UB)
    grpexp_s = Tensor(DT.float, [1, SCALE_SLOTS], Position.UB)
    bound_s = Tensor(DT.float, [1, SCALE_SLOTS], Position.UB)
    scale_bcast = Tensor(DT.float, [E5M2_SCALE_BCAST_ROWS, SCALE_BCAST_COLS], Position.UB)
    exp_mask_i32 = Tensor(DT.int, [1, SCALE_SLOTS], Position.UB)
    n_tiles = CeilDiv(total_groups, TILE_GROUPS)
    tiles_per_core = CeilDiv(n_tiles, GetVecNum())
    tile_begin = Var(tiles_per_core * GetVecIdx())
    tile_end = Min(tile_begin + tiles_per_core, n_tiles)
    dup(exp_mask_i32, FP32_EXP_MASK)
    with auto_sync():
        set_mask_normal()
        reset_mask()
        for tile_idx in range(tile_begin, tile_end):
            g0 = Var(tile_idx * TILE_GROUPS)
            valid = Min(TILE_GROUPS, total_groups - g0)
            xt = x_work[tile_idx]
            yt = y_work[tile_idx]
            gm_to_ub_pad(xt[0:valid, 0:GROUP], x_g[g0:g0 + valid, 0:GROUP],
                         n_burst=valid, burst_len_element=GROUP, src_stride_element=0,
                         dst_stride=(ROW_PAD - GROUP) // xt.dtype.C0)
            _mxfp8e5m2_group_core(xt, yt, valid, pexp, privexp, grpexp_s,
                                   bound_s, scale_bcast, exp_mask_i32)
            ub_to_gm_pad(y_g[g0:g0 + valid, 0:GROUP], yt[0:valid, 0:GROUP],
                         n_burst=valid, burst_len_element=GROUP,
                         src_stride=(ROW_PAD - GROUP) // yt.dtype.C0, dst_stride_element=0)
            bar_mte3()
    return y


@lru_cache(maxsize=2)
def build_mxfp8e5m2_kernel(device):
    if device not in ("a2", "a3"):
        raise ValueError("Shared tensor-vector source supports A2/A3")
    return import_module(f"ascriptor.{device}").kernel(mode="vec")(mxfp8e5m2_kernel_fp32)


# ----------------------------------------------------------------------------------------------------
# dispatcher
# The three formats share one signature -- (x, y, rows, cols) -- and one vector-only mode, so a
# case selects between them by name and changes nothing else about the launch.
# ----------------------------------------------------------------------------------------------------

BUILDERS = {"plain_mxfp4": build_mxfp4_kernel,
            "mbs_mxfp4": build_mbs_mxfp4_kernel,
            "mxfp8_e5m2": build_mxfp8e5m2_kernel}


def build_kernel(variant, device):
    """Bind the named format's body to the A2 or A3 facade. Each builder caches its own
    elaboration, so repeated cases of one variant compile the kernel once."""
    return BUILDERS[variant](device)
