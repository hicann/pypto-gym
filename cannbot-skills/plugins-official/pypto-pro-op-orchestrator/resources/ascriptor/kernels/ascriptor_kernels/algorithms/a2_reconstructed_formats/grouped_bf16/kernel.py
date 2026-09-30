# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Eleven reconstructed BF16 group-quantization entries on the A2/A3 vector unit: MBS-MXFP4, plain MXFP4, MXFP8 E5M2, three E2M1 group sizes, three signed-int4 group sizes, and the two HiFX levels."""

from ascriptor.a2 import *
from functools import lru_cache
from importlib import import_module

# ----------------------------------------------------------------------------------------------------
# mbs.py
# Preserved corrected source; local contracts distinguish every precision and scale mode.
# ----------------------------------------------------------------------------------------------------

MACRO_GROUP = 128
INNER_GROUP = 32
INNER_PER_MACRO = MACRO_GROUP // INNER_GROUP
E2M1_MAX = 6.0
E2M1_EXP_MAX = 2
MXFP4_FLOOR_SCALE_INPUT_FACTOR = 1.0 / (2.0 ** E2M1_EXP_MAX)
# Per-128 macro-factor target peak from the MBS-MXFP4 paper. Only the macro
# factor uses this; the inner per-32 MXFP4 path uses the HiFloat4 floor-scale rule.
MACRO_FACTOR_TARGET = E2M1_MAX

TILE_MACROS = 32  # Preserved source default; unchecked environment overrides are not a domain.
UB_GUARD_MACROS = TILE_MACROS + 1
MBS_SCALE_BCAST_ROWS = ((TILE_MACROS + 7) // 8) * 8
UB_MACRO_COLS = 128
SCALE_SLOTS = 64
SCALE_BCAST_COLS = 8
MACRO_REP_STRIDE = UB_MACRO_COLS // 8
GROUP_FLAG_COLS = UB_MACRO_COLS * 4

FP32_EXP_MASK = 0x7F800000
E8M0_MIN_VALUE = 2.0 ** -127
@func()
def quantize_abs_to_e2m1_g32(abs_norm: Tensor, q_abs: Tensor, flag: Tensor, const_full: Tensor, groups: Var, rep_stride: Var):
    dup(q_abs, 0.0, repeat=groups, dst_rep_stride=rep_stride, count_per_rep=INNER_GROUP)

    dup(const_full, 0.5, repeat=groups, dst_rep_stride=rep_stride, count_per_rep=INNER_GROUP)
    compare_scalar(flag, abs_norm, 0.25, CompareMode.LT, repeat=groups, dst_rep_stride=rep_stride, src1_rep_stride=rep_stride)
    select(
        q_abs,
        flag,
        q_abs,
        const_full,
        SelectMode.TENSOR_SCALAR,
        repeat=groups,
        dst_rep_stride=rep_stride,
        src1_rep_stride=rep_stride,
        src2_rep_stride=rep_stride,
    )

    dup(const_full, 1.0, repeat=groups, dst_rep_stride=rep_stride, count_per_rep=INNER_GROUP)
    compare_scalar(flag, abs_norm, 0.75, CompareMode.LT, repeat=groups, dst_rep_stride=rep_stride, src1_rep_stride=rep_stride)
    select(
        q_abs,
        flag,
        q_abs,
        const_full,
        SelectMode.TENSOR_SCALAR,
        repeat=groups,
        dst_rep_stride=rep_stride,
        src1_rep_stride=rep_stride,
        src2_rep_stride=rep_stride,
    )

    dup(const_full, 1.5, repeat=groups, dst_rep_stride=rep_stride, count_per_rep=INNER_GROUP)
    compare_scalar(flag, abs_norm, 1.25, CompareMode.LT, repeat=groups, dst_rep_stride=rep_stride, src1_rep_stride=rep_stride)
    select(
        q_abs,
        flag,
        q_abs,
        const_full,
        SelectMode.TENSOR_SCALAR,
        repeat=groups,
        dst_rep_stride=rep_stride,
        src1_rep_stride=rep_stride,
        src2_rep_stride=rep_stride,
    )

    dup(const_full, 2.0, repeat=groups, dst_rep_stride=rep_stride, count_per_rep=INNER_GROUP)
    compare_scalar(flag, abs_norm, 1.75, CompareMode.LT, repeat=groups, dst_rep_stride=rep_stride, src1_rep_stride=rep_stride)
    select(
        q_abs,
        flag,
        q_abs,
        const_full,
        SelectMode.TENSOR_SCALAR,
        repeat=groups,
        dst_rep_stride=rep_stride,
        src1_rep_stride=rep_stride,
        src2_rep_stride=rep_stride,
    )

    dup(const_full, 3.0, repeat=groups, dst_rep_stride=rep_stride, count_per_rep=INNER_GROUP)
    compare_scalar(flag, abs_norm, 2.5, CompareMode.LT, repeat=groups, dst_rep_stride=rep_stride, src1_rep_stride=rep_stride)
    select(
        q_abs,
        flag,
        q_abs,
        const_full,
        SelectMode.TENSOR_SCALAR,
        repeat=groups,
        dst_rep_stride=rep_stride,
        src1_rep_stride=rep_stride,
        src2_rep_stride=rep_stride,
    )

    dup(const_full, 4.0, repeat=groups, dst_rep_stride=rep_stride, count_per_rep=INNER_GROUP)
    compare_scalar(flag, abs_norm, 3.5, CompareMode.LT, repeat=groups, dst_rep_stride=rep_stride, src1_rep_stride=rep_stride)
    select(
        q_abs,
        flag,
        q_abs,
        const_full,
        SelectMode.TENSOR_SCALAR,
        repeat=groups,
        dst_rep_stride=rep_stride,
        src1_rep_stride=rep_stride,
        src2_rep_stride=rep_stride,
    )

    dup(const_full, 6.0, repeat=groups, dst_rep_stride=rep_stride, count_per_rep=INNER_GROUP)
    compare_scalar(flag, abs_norm, 5.0, CompareMode.LT, repeat=groups, dst_rep_stride=rep_stride, src1_rep_stride=rep_stride)
    select(
        q_abs,
        flag,
        q_abs,
        const_full,
        SelectMode.TENSOR_SCALAR,
        repeat=groups,
        dst_rep_stride=rep_stride,
        src1_rep_stride=rep_stride,
        src2_rep_stride=rep_stride,
    )


@func()
def build_e0m8_macro_factor(
    amax_s: Tensor,
    factor_s: Tensor,
    recip_s: Tensor,
    safe_s: Tensor,
    scalar_flag: Tensor,
    one_s: Tensor,
    macro_tgt_s: Tensor,
    exp_mask_i32: Tensor,
    macros: Var,
):
    safe_s[0:1, 0:macros] <<= amax_s[0:1, 0:macros]
    compare_scalar(scalar_flag[0:1, 0:macros], safe_s[0:1, 0:macros], 0.0, CompareMode.NE)
    select(
        safe_s[0:1, 0:macros],
        scalar_flag[0:1, 0:macros],
        safe_s[0:1, 0:macros],
        one_s[0:1, 0:macros],
        SelectMode.TENSOR_SCALAR,
    )

    div(recip_s[0:1, 0:macros], macro_tgt_s[0:1, 0:macros], safe_s[0:1, 0:macros])

    recip_i16 = recip_s.reinterpret(DT.int16, name="macro_recip_i16")
    factor_i16 = factor_s.reinterpret(DT.int16, name="macro_factor_i16")
    exp_mask_i16 = exp_mask_i32.reinterpret(DT.int16, name="macro_exp_mask_i16")
    recip_i32 = recip_s.reinterpret(DT.int, name="macro_recip_i32")

    vand(factor_i16[0:1, 0:2 * macros], recip_i16[0:1, 0:2 * macros], exp_mask_i16[0:1, 0:2 * macros])
    div(safe_s[0:1, 0:macros], recip_s[0:1, 0:macros], factor_s[0:1, 0:macros])
    adds(safe_s[0:1, 0:macros], safe_s[0:1, 0:macros], -1.0)
    muls(safe_s[0:1, 0:macros], safe_s[0:1, 0:macros], 256.0)
    cast(recip_i32[0:1, 0:macros], safe_s[0:1, 0:macros], round_mode=RoundMode.TRUNC)
    cast(safe_s[0:1, 0:macros], recip_i32[0:1, 0:macros])
    muls(safe_s[0:1, 0:macros], safe_s[0:1, 0:macros], 1.0 / 256.0)
    adds(factor_s[0:1, 0:macros], safe_s[0:1, 0:macros], 1.0)

    compare_scalar(scalar_flag[0:1, 0:macros], amax_s[0:1, 0:macros], 0.0, CompareMode.NE)
    select(
        factor_s[0:1, 0:macros],
        scalar_flag[0:1, 0:macros],
        factor_s[0:1, 0:macros],
        one_s[0:1, 0:macros],
        SelectMode.TENSOR_SCALAR,
    )


@func()
def build_e8m0_scale_s(
    scale_input_s: Tensor,
    scale_s: Tensor,
    exp_s: Tensor,
    exp_mask_i32: Tensor,
    min_scale_s: Tensor,
    count: Var,
):
    scale_input_i16 = scale_input_s.reinterpret(DT.int16, name="scale_input_i16")
    exp_i16 = exp_s.reinterpret(DT.int16, name="scale_exp_i16")
    exp_mask_i16 = exp_mask_i32.reinterpret(DT.int16, name="scale_exp_mask_i16")
    vand(exp_i16[0:1, 0:2 * count], scale_input_i16[0:1, 0:2 * count], exp_mask_i16[0:1, 0:2 * count])
    vmax(scale_s[0:1, 0:count], exp_s[0:1, 0:count], min_scale_s[0:1, 0:count])


def mbs_mxfp4_kernel(x: GM[bf16, ('rows', 'cols')], y: GM[bf16, ('rows', 'cols')], rows: i32, cols: i32):
    macros_per_row = Var(cols // MACRO_GROUP)
    total_macros = Var(rows * macros_per_row)

    x_macro = x.reshape([total_macros, MACRO_GROUP], name="x_macro")
    y_macro = y.reshape([total_macros, MACRO_GROUP], name="y_macro")

    x_bf16 = DBuff(DT.bfloat16, [TILE_MACROS, UB_MACRO_COLS], Position.UB)
    y_bf16 = DBuff(DT.bfloat16, [TILE_MACROS, UB_MACRO_COLS], Position.UB)

    x_float = Tensor(DT.float, [UB_GUARD_MACROS, UB_MACRO_COLS], Position.UB)
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
    # `select` reuses the numeric repeat stride of the float dst/src operands,
    # so the uint8 flag rows need the same byte stride (128 fp32 elems == 512 B).
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
            x_bf16_t = x_bf16[tile_idx]
            y_bf16_t = y_bf16[tile_idx]

            x_bf16_t[0:valid_macros, 0:MACRO_GROUP] <<= x_macro[macro0:macro0 + valid_macros, 0:MACRO_GROUP]
            cast(x_float[0:valid_macros, 0:MACRO_GROUP], x_bf16_t[0:valid_macros, 0:MACRO_GROUP], repeat=2 * valid_macros)

            abs(macro_abs[0:valid_macros, 0:MACRO_GROUP], x_float[0:valid_macros, 0:MACRO_GROUP], repeat=2 * valid_macros)
            cmax(macro_left_s, macro_abs[0:valid_macros, 0:64], repeat=valid_macros)
            cmax(macro_right_s, macro_abs[0:valid_macros, 64:128], repeat=valid_macros)
            vmax(
                macro_amax_s[0:1, 0:valid_macros],
                macro_left_s[0:1, 0:valid_macros],
                macro_right_s[0:1, 0:valid_macros],
            )

            build_e0m8_macro_factor(
                macro_amax_s,
                macro_factor_s,
                macro_recip_s,
                macro_safe_s,
                scalar_flag,
                one_s,
                macro_tgt_s,
                exp_mask_i32,
                valid_macros,
            )
            brcb(macro_factor_bcast, macro_factor_s, repeat=CeilDiv(valid_macros, 8), dst_blk_stride=1, dst_rep_stride=8)

            mul(
                x_float[0:valid_macros, 0:64],
                x_float[0:valid_macros, 0:64],
                macro_factor_bcast[0:valid_macros, 0:SCALE_BCAST_COLS],
                repeat=valid_macros,
            )
            mul(
                x_float[0:valid_macros, 64:128],
                x_float[0:valid_macros, 64:128],
                macro_factor_bcast[0:valid_macros, 0:SCALE_BCAST_COLS],
                repeat=valid_macros,
            )

            for inner_idx in range(0, INNER_PER_MACRO):
                group_col0 = Var(inner_idx * INNER_GROUP)

                abs(
                    macro_abs[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
                    x_float[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
                    repeat=valid_macros,
                    dst_rep_stride=MACRO_REP_STRIDE,
                    src_rep_stride=MACRO_REP_STRIDE,
                    count_per_rep=INNER_GROUP,
                )
                cmax(
                    group_amax_s,
                    macro_abs[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
                    repeat=valid_macros,
                    src_rep_stride=MACRO_REP_STRIDE,
                    count_per_rep=INNER_GROUP,
                )
                muls(
                    group_scale_input_s[0:1, 0:valid_macros],
                    group_amax_s[0:1, 0:valid_macros],
                    MXFP4_FLOOR_SCALE_INPUT_FACTOR,
                )

                build_e8m0_scale_s(
                    group_scale_input_s,
                    group_scale_s,
                    group_exp_s,
                    exp_mask_i32,
                    min_scale_s,
                    valid_macros,
                )
                brcb(group_scale_bcast, group_scale_s, repeat=CeilDiv(valid_macros, 8), dst_blk_stride=1, dst_rep_stride=8)

                div(
                    x_float[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
                    x_float[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
                    group_scale_bcast[0:valid_macros, 0:SCALE_BCAST_COLS],
                    repeat=valid_macros,
                    dst_rep_stride=MACRO_REP_STRIDE,
                    src1_rep_stride=MACRO_REP_STRIDE,
                    src2_rep_stride=1,
                    count_per_rep=INNER_GROUP,
                )
                abs(
                    macro_abs[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
                    x_float[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
                    repeat=valid_macros,
                    dst_rep_stride=MACRO_REP_STRIDE,
                    src_rep_stride=MACRO_REP_STRIDE,
                    count_per_rep=INNER_GROUP,
                )
                quantize_abs_to_e2m1_g32(
                    macro_abs[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
                    q_abs[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
                    group_flag[0:valid_macros, 0:INNER_GROUP],
                    const_full[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
                    valid_macros,
                    MACRO_REP_STRIDE,
                )

                dup(
                    const_full[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
                    0.0,
                    repeat=valid_macros,
                    dst_rep_stride=MACRO_REP_STRIDE,
                    count_per_rep=INNER_GROUP,
                )
                muls(
                    neg_q_abs[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
                    q_abs[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
                    -1.0,
                    repeat=valid_macros,
                    dst_rep_stride=MACRO_REP_STRIDE,
                    src_rep_stride=MACRO_REP_STRIDE,
                    count_per_rep=INNER_GROUP,
                )
                compare_scalar(
                    group_flag[0:valid_macros, 0:INNER_GROUP],
                    x_float[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
                    0.0,
                    CompareMode.GE,
                    repeat=valid_macros,
                    dst_rep_stride=MACRO_REP_STRIDE,
                    src1_rep_stride=MACRO_REP_STRIDE,
                )
                select(
                    macro_abs[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
                    group_flag[0:valid_macros, 0:INNER_GROUP],
                    q_abs[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
                    const_full[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
                    SelectMode.TENSOR_SCALAR,
                    repeat=valid_macros,
                    dst_rep_stride=MACRO_REP_STRIDE,
                    src1_rep_stride=MACRO_REP_STRIDE,
                    src2_rep_stride=MACRO_REP_STRIDE,
                )
                compare_scalar(
                    group_flag[0:valid_macros, 0:INNER_GROUP],
                    x_float[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
                    0.0,
                    CompareMode.LT,
                    repeat=valid_macros,
                    dst_rep_stride=MACRO_REP_STRIDE,
                    src1_rep_stride=MACRO_REP_STRIDE,
                )
                select(
                    neg_q_abs[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
                    group_flag[0:valid_macros, 0:INNER_GROUP],
                    neg_q_abs[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
                    const_full[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
                    SelectMode.TENSOR_SCALAR,
                    repeat=valid_macros,
                    dst_rep_stride=MACRO_REP_STRIDE,
                    src1_rep_stride=MACRO_REP_STRIDE,
                    src2_rep_stride=MACRO_REP_STRIDE,
                )
                add(
                    x_float[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
                    macro_abs[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
                    neg_q_abs[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
                    repeat=valid_macros,
                    dst_rep_stride=MACRO_REP_STRIDE,
                    src1_rep_stride=MACRO_REP_STRIDE,
                    src2_rep_stride=MACRO_REP_STRIDE,
                    count_per_rep=INNER_GROUP,
                )
                mul(
                    x_float[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
                    x_float[0:valid_macros, group_col0:group_col0 + INNER_GROUP],
                    group_scale_bcast[0:valid_macros, 0:SCALE_BCAST_COLS],
                    repeat=valid_macros,
                    dst_rep_stride=MACRO_REP_STRIDE,
                    src1_rep_stride=MACRO_REP_STRIDE,
                    src2_rep_stride=1,
                    count_per_rep=INNER_GROUP,
                )

            div(
                x_float[0:valid_macros, 0:64],
                x_float[0:valid_macros, 0:64],
                macro_factor_bcast[0:valid_macros, 0:SCALE_BCAST_COLS],
                repeat=valid_macros,
            )
            div(
                x_float[0:valid_macros, 64:128],
                x_float[0:valid_macros, 64:128],
                macro_factor_bcast[0:valid_macros, 0:SCALE_BCAST_COLS],
                repeat=valid_macros,
            )

            cast(
                y_bf16_t[0:valid_macros, 0:MACRO_GROUP],
                x_float[0:valid_macros, 0:MACRO_GROUP],
                repeat=2 * valid_macros,
                round_mode=RoundMode.AWAY_FROM_ZERO,
            )
            y_macro[macro0:macro0 + valid_macros, 0:MACRO_GROUP] <<= y_bf16_t[0:valid_macros, 0:MACRO_GROUP]
            bar_mte3()

    return y

# ----------------------------------------------------------------------------------------------------
# mxfp4.py
# Preserved corrected source; local contracts distinguish every precision and scale mode.
# ----------------------------------------------------------------------------------------------------

GROUP = INNER_GROUP                 # 32 elements per MXFP4 block
ROW_PAD = 64                        # each block-row padded to a full 64-lane vector row
REP_F = ROW_PAD // 8                # fp32 per-row repeat stride in 32B blocks (64 fp32 = 8)
REP_B = ROW_PAD // 16               # bf16 per-row repeat stride in 32B blocks (64 bf16 = 4)
FLAG_COLS = ROW_PAD * 4            # uint8 flag row byte-stride matches the fp32 data row (256 B)

MXFP4_TILE_GROUPS = 32                    # 32-blocks per tile (< SCALE_SLOTS for scalar headroom)
MXFP4_UB_GUARD = MXFP4_TILE_GROUPS + 1          # +1 guard row: repeat=valid ops may touch row index `valid`
MXFP4_SCALE_BCAST_ROWS = ((MXFP4_UB_GUARD + 7) // 8) * 8


@func()
def _mxfp4_group_core(
    xt: Tensor, yt: Tensor, valid: Var,
    gabs: Tensor, q_abs: Tensor, neg_q_abs: Tensor, const_full: Tensor, flag: Tensor,
    amax_s: Tensor, scale_input_s: Tensor, scale_s: Tensor, exp_s: Tensor,
    scale_bcast: Tensor, min_scale_s: Tensor, exp_mask_i32: Tensor,
):
    """MXFP4 E2M1 quant-dequant for one tile of 32-blocks (fp32 rows, data in cols 0..31)."""
    # per-block amax -> e8m0 floor scale = 2**(floor(log2(amax)) - 2)
    abs(gabs[0:valid, 0:GROUP], xt[0:valid, 0:GROUP], repeat=valid,
        dst_rep_stride=REP_F, src_rep_stride=REP_F, count_per_rep=GROUP)
    cmax(amax_s, gabs[0:valid, 0:GROUP], repeat=valid, src_rep_stride=REP_F, count_per_rep=GROUP)
    muls(scale_input_s[0:1, 0:valid], amax_s[0:1, 0:valid], MXFP4_FLOOR_SCALE_INPUT_FACTOR)
    build_e8m0_scale_s(scale_input_s, scale_s, exp_s, exp_mask_i32, min_scale_s, valid)
    brcb(scale_bcast, scale_s, repeat=CeilDiv(valid, 8), dst_blk_stride=1, dst_rep_stride=8)

    # normalize x' = x / scale
    div(xt[0:valid, 0:GROUP], xt[0:valid, 0:GROUP], scale_bcast[0:valid, 0:SCALE_BCAST_COLS],
        repeat=valid, dst_rep_stride=REP_F, src1_rep_stride=REP_F, src2_rep_stride=1, count_per_rep=GROUP)

    # q_abs = e2m1(|x'|)  (round-nearest, ties to larger magnitude)
    abs(gabs[0:valid, 0:GROUP], xt[0:valid, 0:GROUP], repeat=valid,
        dst_rep_stride=REP_F, src_rep_stride=REP_F, count_per_rep=GROUP)
    quantize_abs_to_e2m1_g32(
        gabs[0:valid, 0:GROUP], q_abs[0:valid, 0:GROUP],
        flag[0:valid, 0:GROUP], const_full[0:valid, 0:GROUP], valid, REP_F,
    )

    # re-apply sign: signed_q = (x'>=0 ? q_abs : 0) + (x'<0 ? -q_abs : 0)
    dup(const_full[0:valid, 0:GROUP], 0.0, repeat=valid, dst_rep_stride=REP_F, count_per_rep=GROUP)
    muls(neg_q_abs[0:valid, 0:GROUP], q_abs[0:valid, 0:GROUP], -1.0,
         repeat=valid, dst_rep_stride=REP_F, src_rep_stride=REP_F, count_per_rep=GROUP)
    compare_scalar(flag[0:valid, 0:GROUP], xt[0:valid, 0:GROUP], 0.0, CompareMode.GE,
                   repeat=valid, dst_rep_stride=REP_F, src1_rep_stride=REP_F)
    select(gabs[0:valid, 0:GROUP], flag[0:valid, 0:GROUP],
           q_abs[0:valid, 0:GROUP], const_full[0:valid, 0:GROUP], SelectMode.TENSOR_SCALAR,
           repeat=valid, dst_rep_stride=REP_F, src1_rep_stride=REP_F, src2_rep_stride=REP_F)
    compare_scalar(flag[0:valid, 0:GROUP], xt[0:valid, 0:GROUP], 0.0, CompareMode.LT,
                   repeat=valid, dst_rep_stride=REP_F, src1_rep_stride=REP_F)
    select(neg_q_abs[0:valid, 0:GROUP], flag[0:valid, 0:GROUP],
           neg_q_abs[0:valid, 0:GROUP], const_full[0:valid, 0:GROUP], SelectMode.TENSOR_SCALAR,
           repeat=valid, dst_rep_stride=REP_F, src1_rep_stride=REP_F, src2_rep_stride=REP_F)
    add(xt[0:valid, 0:GROUP], gabs[0:valid, 0:GROUP], neg_q_abs[0:valid, 0:GROUP],
        repeat=valid, dst_rep_stride=REP_F, src1_rep_stride=REP_F, src2_rep_stride=REP_F, count_per_rep=GROUP)

    # dequantize y = signed_q * scale
    mul(yt[0:valid, 0:GROUP], xt[0:valid, 0:GROUP], scale_bcast[0:valid, 0:SCALE_BCAST_COLS],
        repeat=valid, dst_rep_stride=REP_F, src1_rep_stride=REP_F, src2_rep_stride=1, count_per_rep=GROUP)


def _mxfp4_make_scratch():
    gabs = Tensor(DT.float, [MXFP4_UB_GUARD, ROW_PAD], Position.UB)
    q_abs = Tensor(DT.float, [MXFP4_UB_GUARD, ROW_PAD], Position.UB)
    neg_q_abs = Tensor(DT.float, [MXFP4_UB_GUARD, ROW_PAD], Position.UB)
    const_full = Tensor(DT.float, [MXFP4_UB_GUARD, ROW_PAD], Position.UB)
    flag = Tensor(DT.uint8, [MXFP4_UB_GUARD, FLAG_COLS], Position.UB)
    amax_s = Tensor(DT.float, [1, SCALE_SLOTS], Position.UB)
    scale_input_s = Tensor(DT.float, [1, SCALE_SLOTS], Position.UB)
    scale_s = Tensor(DT.float, [1, SCALE_SLOTS], Position.UB)
    exp_s = Tensor(DT.float, [1, SCALE_SLOTS], Position.UB)
    scale_bcast = Tensor(DT.float, [MXFP4_SCALE_BCAST_ROWS, SCALE_BCAST_COLS], Position.UB)
    min_scale_s = Tensor(DT.float, [1, SCALE_SLOTS], Position.UB)
    exp_mask_i32 = Tensor(DT.int, [1, SCALE_SLOTS], Position.UB)
    return (gabs, q_abs, neg_q_abs, const_full, flag, amax_s, scale_input_s,
            scale_s, exp_s, scale_bcast, min_scale_s, exp_mask_i32)




def mxfp4_kernel_bf16(x: GM[bf16, ('rows', 'cols')], y: GM[bf16, ('rows', 'cols')], rows: i32, cols: i32):
    groups_per_row = Var(cols // GROUP)
    total_groups = Var(rows * groups_per_row)
    x_g = x.reshape([total_groups, GROUP], name="x_g")
    y_g = y.reshape([total_groups, GROUP], name="y_g")

    x_bf = DBuff(DT.bfloat16, [MXFP4_UB_GUARD, ROW_PAD], Position.UB)
    y_bf = DBuff(DT.bfloat16, [MXFP4_UB_GUARD, ROW_PAD], Position.UB)
    xt = Tensor(DT.float, [MXFP4_UB_GUARD, ROW_PAD], Position.UB)
    yt = Tensor(DT.float, [MXFP4_UB_GUARD, ROW_PAD], Position.UB)
    scratch = _mxfp4_make_scratch()
    (gabs, q_abs, neg_q_abs, const_full, flag, amax_s, scale_input_s,
     scale_s, exp_s, scale_bcast, min_scale_s, exp_mask_i32) = scratch

    n_tiles = CeilDiv(total_groups, MXFP4_TILE_GROUPS)
    tiles_per_core = CeilDiv(n_tiles, GetVecNum())
    tile_begin = Var(tiles_per_core * GetVecIdx())
    tile_end = Min(tile_begin + tiles_per_core, n_tiles)

    dup(min_scale_s, E8M0_MIN_VALUE)
    dup(exp_mask_i32, FP32_EXP_MASK)

    with auto_sync():
        set_mask_normal()
        reset_mask()
        for tile_idx in range(tile_begin, tile_end):
            g0 = Var(tile_idx * MXFP4_TILE_GROUPS)
            valid = Min(MXFP4_TILE_GROUPS, total_groups - g0)
            xb = x_bf[tile_idx]
            yb = y_bf[tile_idx]

            gm_to_ub_pad(
                xb[0:valid, 0:GROUP], x_g[g0:g0 + valid, 0:GROUP],
                n_burst=valid, burst_len_element=GROUP,
                src_stride_element=0, dst_stride=(ROW_PAD - GROUP) // xb.dtype.C0,
            )
            cast(xt[0:valid, 0:GROUP], xb[0:valid, 0:GROUP], repeat=valid,
                 dst_rep_stride=REP_F, src_rep_stride=REP_B, count_per_rep=GROUP,
                 round_mode=RoundMode.NONE)
            _mxfp4_group_core(xt, yt, valid, *scratch)
            # E2M1 dequant values are bf16-exact (<=3 significant bits), so the
            # narrowing cast is lossless regardless of round mode.
            cast(yb[0:valid, 0:GROUP], yt[0:valid, 0:GROUP], repeat=valid,
                 dst_rep_stride=REP_B, src_rep_stride=REP_F, count_per_rep=GROUP,
                 round_mode=RoundMode.AWAY_FROM_ZERO)
            ub_to_gm_pad(
                y_g[g0:g0 + valid, 0:GROUP], yb[0:valid, 0:GROUP],
                n_burst=valid, burst_len_element=GROUP,
                src_stride=(ROW_PAD - GROUP) // yb.dtype.C0, dst_stride_element=0,
            )
            bar_mte3()
    return y

# ----------------------------------------------------------------------------------------------------
# mxfp8e5m2.py
# Preserved corrected source; local contracts distinguish every precision and scale mode.
# ----------------------------------------------------------------------------------------------------

E5M2_TILE_GROUPS = 32
E5M2_UB_GUARD = E5M2_TILE_GROUPS + 1          # +1 guard row: repeat=valid ops may touch row index `valid`
E5M2_SCALE_BCAST_ROWS = ((E5M2_UB_GUARD + 7) // 8) * 8

# E5M2 format constants (see module docstring).
E5M2_UPPER_COEF = 1.75                    # 57344 = 1.75 * 2**15
E5M2_PRIVEXP_FLOOR = 1.0 / 536870912.0    # 2**-29
MAN_STEP_COEF = 0.25                      # /4 == 2 mantissa bits
E5M2_EPS = 5.421011e-20                   # guards division on all-zero blocks


@func()
def _mxfp8e5m2_group_core(
    xt: Tensor, yt: Tensor, valid: Var,
    pexp: Tensor, privexp: Tensor,
    grpexp_s: Tensor, bound_s: Tensor, scale_bcast: Tensor, exp_mask_i32: Tensor,
):
    """MXFP8 E5M2 quant-dequant for one tile of 32-blocks."""
    # per-element 2**elem_exp via exponent-bit extraction: pexp = |x| & 0x7F800000
    abs(pexp[0:valid, 0:GROUP], xt[0:valid, 0:GROUP], repeat=valid,
        dst_rep_stride=REP_F, src_rep_stride=REP_F, count_per_rep=GROUP)
    pexp_i16 = pexp.reinterpret(DT.int16, name="pexp_i16")
    exp_mask_i16 = exp_mask_i32.reinterpret(DT.int16, name="e5m2_exp_mask_i16")
    vand(pexp_i16[0:valid, 0:2 * GROUP], pexp_i16[0:valid, 0:2 * GROUP], exp_mask_i16[0:1, 0:2 * GROUP],
         repeat=valid, dst_rep_stride=REP_F, src1_rep_stride=REP_F, src2_rep_stride=0, count_per_rep=2 * GROUP)

    # grpexp = 2**Emax = block max of pexp
    cmax(grpexp_s, pexp[0:valid, 0:GROUP], repeat=valid, src_rep_stride=REP_F, count_per_rep=GROUP)

    # privexp equals max(grpexp/2**29, pexp) * 1/4 + eps.
    muls(bound_s[0:1, 0:valid], grpexp_s[0:1, 0:valid], E5M2_PRIVEXP_FLOOR)
    brcb(scale_bcast, bound_s, repeat=CeilDiv(valid, 8), dst_blk_stride=1, dst_rep_stride=8)
    vmax(privexp[0:valid, 0:GROUP], pexp[0:valid, 0:GROUP], scale_bcast[0:valid, 0:SCALE_BCAST_COLS],
         repeat=valid, dst_rep_stride=REP_F, src1_rep_stride=REP_F, src2_rep_stride=1, count_per_rep=GROUP)
    muls(privexp[0:valid, 0:GROUP], privexp[0:valid, 0:GROUP], MAN_STEP_COEF,
         repeat=valid, dst_rep_stride=REP_F, src_rep_stride=REP_F, count_per_rep=GROUP)
    adds(privexp[0:valid, 0:GROUP], privexp[0:valid, 0:GROUP], E5M2_EPS,
         repeat=valid, dst_rep_stride=REP_F, src_rep_stride=REP_F, count_per_rep=GROUP)

    # mant = round_half_away(x / step) * step   (signed; CAST_ROUND == vconv_f322f32a)
    div(yt[0:valid, 0:GROUP], xt[0:valid, 0:GROUP], privexp[0:valid, 0:GROUP],
        repeat=valid, dst_rep_stride=REP_F, src1_rep_stride=REP_F, src2_rep_stride=REP_F, count_per_rep=GROUP)
    cast(yt[0:valid, 0:GROUP], yt[0:valid, 0:GROUP], repeat=valid,
         dst_rep_stride=REP_F, src_rep_stride=REP_F, count_per_rep=GROUP, round_mode=RoundMode.AWAY_FROM_ZERO)
    mul(yt[0:valid, 0:GROUP], yt[0:valid, 0:GROUP], privexp[0:valid, 0:GROUP],
        repeat=valid, dst_rep_stride=REP_F, src1_rep_stride=REP_F, src2_rep_stride=REP_F, count_per_rep=GROUP)

    # clip to +/- 1.75 * grpexp (57344 * 2**shared_exp)
    muls(bound_s[0:1, 0:valid], grpexp_s[0:1, 0:valid], E5M2_UPPER_COEF)
    brcb(scale_bcast, bound_s, repeat=CeilDiv(valid, 8), dst_blk_stride=1, dst_rep_stride=8)
    vmin(yt[0:valid, 0:GROUP], yt[0:valid, 0:GROUP], scale_bcast[0:valid, 0:SCALE_BCAST_COLS],
         repeat=valid, dst_rep_stride=REP_F, src1_rep_stride=REP_F, src2_rep_stride=1, count_per_rep=GROUP)
    muls(bound_s[0:1, 0:valid], grpexp_s[0:1, 0:valid], -E5M2_UPPER_COEF)
    brcb(scale_bcast, bound_s, repeat=CeilDiv(valid, 8), dst_blk_stride=1, dst_rep_stride=8)
    vmax(yt[0:valid, 0:GROUP], yt[0:valid, 0:GROUP], scale_bcast[0:valid, 0:SCALE_BCAST_COLS],
         repeat=valid, dst_rep_stride=REP_F, src1_rep_stride=REP_F, src2_rep_stride=1, count_per_rep=GROUP)


def _mxfp8e5m2_make_scratch():
    pexp = Tensor(DT.float, [E5M2_UB_GUARD, ROW_PAD], Position.UB)
    privexp = Tensor(DT.float, [E5M2_UB_GUARD, ROW_PAD], Position.UB)
    grpexp_s = Tensor(DT.float, [1, SCALE_SLOTS], Position.UB)
    bound_s = Tensor(DT.float, [1, SCALE_SLOTS], Position.UB)
    scale_bcast = Tensor(DT.float, [E5M2_SCALE_BCAST_ROWS, SCALE_BCAST_COLS], Position.UB)
    exp_mask_i32 = Tensor(DT.int, [1, SCALE_SLOTS], Position.UB)
    return (pexp, privexp, grpexp_s, bound_s, scale_bcast, exp_mask_i32)




def mxfp8e5m2_kernel_bf16(x: GM[bf16, ('rows', 'cols')], y: GM[bf16, ('rows', 'cols')], rows: i32, cols: i32):
    groups_per_row = Var(cols // GROUP)
    total_groups = Var(rows * groups_per_row)
    x_g = x.reshape([total_groups, GROUP], name="x_g")
    y_g = y.reshape([total_groups, GROUP], name="y_g")

    x_bf = DBuff(DT.bfloat16, [E5M2_UB_GUARD, ROW_PAD], Position.UB)
    y_bf = DBuff(DT.bfloat16, [E5M2_UB_GUARD, ROW_PAD], Position.UB)
    xt = Tensor(DT.float, [E5M2_UB_GUARD, ROW_PAD], Position.UB)
    yt = Tensor(DT.float, [E5M2_UB_GUARD, ROW_PAD], Position.UB)
    scratch = _mxfp8e5m2_make_scratch()
    exp_mask_i32 = scratch[5]

    n_tiles = CeilDiv(total_groups, E5M2_TILE_GROUPS)
    tiles_per_core = CeilDiv(n_tiles, GetVecNum())
    tile_begin = Var(tiles_per_core * GetVecIdx())
    tile_end = Min(tile_begin + tiles_per_core, n_tiles)

    dup(exp_mask_i32, FP32_EXP_MASK)

    with auto_sync():
        set_mask_normal()
        reset_mask()
        for tile_idx in range(tile_begin, tile_end):
            g0 = Var(tile_idx * E5M2_TILE_GROUPS)
            valid = Min(E5M2_TILE_GROUPS, total_groups - g0)
            xb = x_bf[tile_idx]
            yb = y_bf[tile_idx]

            gm_to_ub_pad(
                xb[0:valid, 0:GROUP], x_g[g0:g0 + valid, 0:GROUP],
                n_burst=valid, burst_len_element=GROUP,
                src_stride_element=0, dst_stride=(ROW_PAD - GROUP) // xb.dtype.C0,
            )
            cast(xt[0:valid, 0:GROUP], xb[0:valid, 0:GROUP], repeat=valid,
                 dst_rep_stride=REP_F, src_rep_stride=REP_B, count_per_rep=GROUP,
                 round_mode=RoundMode.NONE)
            _mxfp8e5m2_group_core(xt, yt, valid, *scratch)
            # E5M2 dequant values carry at most three significant bits.
            cast(yb[0:valid, 0:GROUP], yt[0:valid, 0:GROUP], repeat=valid,
                 dst_rep_stride=REP_B, src_rep_stride=REP_F, count_per_rep=GROUP,
                 round_mode=RoundMode.AWAY_FROM_ZERO)
            ub_to_gm_pad(
                y_g[g0:g0 + valid, 0:GROUP], yb[0:valid, 0:GROUP],
                n_burst=valid, burst_len_element=GROUP,
                src_stride=(ROW_PAD - GROUP) // yb.dtype.C0, dst_stride_element=0,
            )
            bar_mte3()
    return y

# ----------------------------------------------------------------------------------------------------
# group16_bf16_fp4_e2m1.py
# Preserved corrected source; local contracts distinguish every precision and scale mode.
# ----------------------------------------------------------------------------------------------------

G16_GROUP_SIZE = 16
G16_TILE_GROUPS = 16
UB_GROUP_COLS = 64


@func()
def quantize_abs_to_e2m1_g16(abs_norm: Tensor, q_abs: Tensor, flag: Tensor, const_full: Tensor, groups: Var):
    dup(q_abs, 0.0, repeat=groups, count_per_rep=G16_GROUP_SIZE)

    dup(const_full, 0.5, repeat=groups, count_per_rep=G16_GROUP_SIZE)
    compare_scalar(flag, abs_norm, 0.25, CompareMode.LT, repeat=groups)
    select(q_abs, flag, q_abs, const_full, SelectMode.TENSOR_SCALAR, repeat=groups)

    dup(const_full, 1.0, repeat=groups, count_per_rep=G16_GROUP_SIZE)
    compare_scalar(flag, abs_norm, 0.75, CompareMode.LT, repeat=groups)
    select(q_abs, flag, q_abs, const_full, SelectMode.TENSOR_SCALAR, repeat=groups)

    dup(const_full, 1.5, repeat=groups, count_per_rep=G16_GROUP_SIZE)
    compare_scalar(flag, abs_norm, 1.25, CompareMode.LT, repeat=groups)
    select(q_abs, flag, q_abs, const_full, SelectMode.TENSOR_SCALAR, repeat=groups)

    dup(const_full, 2.0, repeat=groups, count_per_rep=G16_GROUP_SIZE)
    compare_scalar(flag, abs_norm, 1.75, CompareMode.LT, repeat=groups)
    select(q_abs, flag, q_abs, const_full, SelectMode.TENSOR_SCALAR, repeat=groups)

    dup(const_full, 3.0, repeat=groups, count_per_rep=G16_GROUP_SIZE)
    compare_scalar(flag, abs_norm, 2.5, CompareMode.LT, repeat=groups)
    select(q_abs, flag, q_abs, const_full, SelectMode.TENSOR_SCALAR, repeat=groups)

    dup(const_full, 4.0, repeat=groups, count_per_rep=G16_GROUP_SIZE)
    compare_scalar(flag, abs_norm, 3.5, CompareMode.LT, repeat=groups)
    select(q_abs, flag, q_abs, const_full, SelectMode.TENSOR_SCALAR, repeat=groups)

    dup(const_full, 6.0, repeat=groups, count_per_rep=G16_GROUP_SIZE)
    compare_scalar(flag, abs_norm, 5.0, CompareMode.LT, repeat=groups)
    select(q_abs, flag, q_abs, const_full, SelectMode.TENSOR_SCALAR, repeat=groups)


def group16_bf16_fp4_e2m1_kernel(x: GM[bf16, ('rows', 'cols')], y: GM[bf16, ('rows', 'cols')], rows: i32, cols: i32):
    groups_per_row = Var(cols // G16_GROUP_SIZE)
    total_groups = Var(rows * groups_per_row)

    x_group = x.reshape([total_groups, G16_GROUP_SIZE], name="x_group")
    y_group = y.reshape([total_groups, G16_GROUP_SIZE], name="y_group")

    x_bf16 = DBuff(DT.bfloat16, [G16_TILE_GROUPS, UB_GROUP_COLS], Position.UB)
    y_bf16 = DBuff(DT.bfloat16, [G16_TILE_GROUPS, UB_GROUP_COLS], Position.UB)

    x_float = Tensor(DT.float, [G16_TILE_GROUPS, UB_GROUP_COLS], Position.UB)
    abs_float = Tensor(DT.float, [G16_TILE_GROUPS, UB_GROUP_COLS], Position.UB)
    q_abs = Tensor(DT.float, [G16_TILE_GROUPS, UB_GROUP_COLS], Position.UB)
    neg_q_abs = Tensor(DT.float, [G16_TILE_GROUPS, UB_GROUP_COLS], Position.UB)
    const_full = Tensor(DT.float, [G16_TILE_GROUPS, UB_GROUP_COLS], Position.UB)

    scale_s = Tensor(DT.float, [1, SCALE_SLOTS], Position.UB)
    scale_safe_s = Tensor(DT.float, [1, SCALE_SLOTS], Position.UB)
    one_s = Tensor(DT.float, [1, SCALE_SLOTS], Position.UB)
    scale_bcast = Tensor(DT.float, [G16_TILE_GROUPS, SCALE_BCAST_COLS], Position.UB)
    safe_scale_bcast = Tensor(DT.float, [G16_TILE_GROUPS, SCALE_BCAST_COLS], Position.UB)
    scale_bf16 = Tensor(DT.bfloat16, [1, SCALE_SLOTS], Position.UB)

    flag = Tensor(DT.uint8, [G16_TILE_GROUPS, SCALE_BCAST_COLS], Position.UB)

    n_tiles = CeilDiv(total_groups, G16_TILE_GROUPS)
    tiles_per_core = CeilDiv(n_tiles, GetVecNum())
    tile_begin = Var(tiles_per_core * GetVecIdx())
    tile_end = Min(tile_begin + tiles_per_core, n_tiles)

    dup(one_s, 1.0)

    with auto_sync():
        for tile_idx in range(tile_begin, tile_end):
            group0 = Var(tile_idx * G16_TILE_GROUPS)
            valid_groups = Min(G16_TILE_GROUPS, total_groups - group0)
            x_bf16_t = x_bf16[tile_idx]
            y_bf16_t = y_bf16[tile_idx]

            x_bf16_t[0:valid_groups, 0:G16_GROUP_SIZE] <<= x_group[group0:group0 + valid_groups, 0:G16_GROUP_SIZE]
            cast(
                x_float[0:valid_groups, 0:G16_GROUP_SIZE],
                x_bf16_t[0:valid_groups, 0:G16_GROUP_SIZE],
                repeat=valid_groups,
                count_per_rep=G16_GROUP_SIZE,
            )

            abs(
                abs_float[0:valid_groups, 0:G16_GROUP_SIZE],
                x_float[0:valid_groups, 0:G16_GROUP_SIZE],
                repeat=valid_groups,
                count_per_rep=G16_GROUP_SIZE,
            )
            cmax(scale_s, abs_float, repeat=valid_groups, count_per_rep=G16_GROUP_SIZE)
            muls(scale_s[0:1, 0:valid_groups], scale_s[0:1, 0:valid_groups], 1.0 / 6.0)
            cast(scale_bf16[0:1, 0:valid_groups], scale_s[0:1, 0:valid_groups], round_mode=RoundMode.AWAY_FROM_ZERO)
            cast(scale_s[0:1, 0:valid_groups], scale_bf16[0:1, 0:valid_groups])

            scale_safe_s[0:1, 0:valid_groups] <<= scale_s[0:1, 0:valid_groups]
            compare_scalar(flag, scale_safe_s[0:1, 0:valid_groups], 0.0, CompareMode.NE)
            select(scale_safe_s[0:1, 0:valid_groups], flag, scale_safe_s[0:1, 0:valid_groups], one_s, SelectMode.TENSOR_SCALAR)

            brcb(scale_bcast, scale_s, repeat=CeilDiv(valid_groups, 8), dst_blk_stride=1, dst_rep_stride=8)
            brcb(safe_scale_bcast, scale_safe_s, repeat=CeilDiv(valid_groups, 8), dst_blk_stride=1, dst_rep_stride=8)

            div(
                x_float[0:valid_groups, 0:G16_GROUP_SIZE],
                x_float[0:valid_groups, 0:G16_GROUP_SIZE],
                safe_scale_bcast[0:valid_groups, 0:SCALE_BCAST_COLS],
                repeat=valid_groups,
                count_per_rep=G16_GROUP_SIZE,
            )
            abs(
                abs_float[0:valid_groups, 0:G16_GROUP_SIZE],
                x_float[0:valid_groups, 0:G16_GROUP_SIZE],
                repeat=valid_groups,
                count_per_rep=G16_GROUP_SIZE,
            )
            quantize_abs_to_e2m1_g16(
                abs_float[0:valid_groups, 0:G16_GROUP_SIZE],
                q_abs[0:valid_groups, 0:G16_GROUP_SIZE],
                flag,
                const_full[0:valid_groups, 0:G16_GROUP_SIZE],
                valid_groups,
            )

            dup(const_full[0:valid_groups, 0:G16_GROUP_SIZE], 0.0, repeat=valid_groups, count_per_rep=G16_GROUP_SIZE)
            muls(
                neg_q_abs[0:valid_groups, 0:G16_GROUP_SIZE],
                q_abs[0:valid_groups, 0:G16_GROUP_SIZE],
                -1.0,
                repeat=valid_groups,
                count_per_rep=G16_GROUP_SIZE,
            )
            compare_scalar(flag, x_float[0:valid_groups, 0:G16_GROUP_SIZE], 0.0, CompareMode.GE, repeat=valid_groups)
            select(
                abs_float[0:valid_groups, 0:G16_GROUP_SIZE],
                flag,
                q_abs[0:valid_groups, 0:G16_GROUP_SIZE],
                const_full[0:valid_groups, 0:G16_GROUP_SIZE],
                SelectMode.TENSOR_SCALAR,
                repeat=valid_groups,
            )
            compare_scalar(flag, x_float[0:valid_groups, 0:G16_GROUP_SIZE], 0.0, CompareMode.LT, repeat=valid_groups)
            select(
                neg_q_abs[0:valid_groups, 0:G16_GROUP_SIZE],
                flag,
                neg_q_abs[0:valid_groups, 0:G16_GROUP_SIZE],
                const_full[0:valid_groups, 0:G16_GROUP_SIZE],
                SelectMode.TENSOR_SCALAR,
                repeat=valid_groups,
            )
            add(
                x_float[0:valid_groups, 0:G16_GROUP_SIZE],
                abs_float[0:valid_groups, 0:G16_GROUP_SIZE],
                neg_q_abs[0:valid_groups, 0:G16_GROUP_SIZE],
                repeat=valid_groups,
                count_per_rep=G16_GROUP_SIZE,
            )
            mul(
                x_float[0:valid_groups, 0:G16_GROUP_SIZE],
                x_float[0:valid_groups, 0:G16_GROUP_SIZE],
                scale_bcast[0:valid_groups, 0:SCALE_BCAST_COLS],
                repeat=valid_groups,
                count_per_rep=G16_GROUP_SIZE,
            )

            cast(
                y_bf16_t[0:valid_groups, 0:G16_GROUP_SIZE],
                x_float[0:valid_groups, 0:G16_GROUP_SIZE],
                repeat=valid_groups,
                round_mode=RoundMode.AWAY_FROM_ZERO,
                count_per_rep=G16_GROUP_SIZE,
            )
            y_group[group0:group0 + valid_groups, 0:G16_GROUP_SIZE] <<= y_bf16_t[0:valid_groups, 0:G16_GROUP_SIZE]

    return y

# ----------------------------------------------------------------------------------------------------
# group16_bf16_fp4_e1m2.py
# Preserved corrected source; local contracts distinguish every precision and scale mode.
# ----------------------------------------------------------------------------------------------------

def group16_bf16_fp4_e1m2_kernel(x: GM[bf16, ('rows', 'cols')], y: GM[bf16, ('rows', 'cols')], rows: i32, cols: i32):
    groups_per_row = Var(cols // G16_GROUP_SIZE)
    total_groups = Var(rows * groups_per_row)

    x_group = x.reshape([total_groups, G16_GROUP_SIZE], name="x_group")
    y_group = y.reshape([total_groups, G16_GROUP_SIZE], name="y_group")

    x_bf16 = DBuff(DT.bfloat16, [G16_TILE_GROUPS, UB_GROUP_COLS], Position.UB)
    y_bf16 = DBuff(DT.bfloat16, [G16_TILE_GROUPS, UB_GROUP_COLS], Position.UB)

    x_float = Tensor(DT.float, [G16_TILE_GROUPS, UB_GROUP_COLS], Position.UB)
    abs_float = Tensor(DT.float, [G16_TILE_GROUPS, UB_GROUP_COLS], Position.UB)
    q_int = Tensor(DT.int, [G16_TILE_GROUPS, UB_GROUP_COLS], Position.UB)

    scale_s = Tensor(DT.float, [1, SCALE_SLOTS], Position.UB)
    scale_safe_s = Tensor(DT.float, [1, SCALE_SLOTS], Position.UB)
    one_s = Tensor(DT.float, [1, SCALE_SLOTS], Position.UB)
    scale_bcast = Tensor(DT.float, [G16_TILE_GROUPS, SCALE_BCAST_COLS], Position.UB)
    safe_scale_bcast = Tensor(DT.float, [G16_TILE_GROUPS, SCALE_BCAST_COLS], Position.UB)
    scale_bf16 = Tensor(DT.bfloat16, [1, SCALE_SLOTS], Position.UB)

    flag = Tensor(DT.uint8, [G16_TILE_GROUPS, SCALE_BCAST_COLS], Position.UB)

    n_tiles = CeilDiv(total_groups, G16_TILE_GROUPS)
    tiles_per_core = CeilDiv(n_tiles, GetVecNum())
    tile_begin = Var(tiles_per_core * GetVecIdx())
    tile_end = Min(tile_begin + tiles_per_core, n_tiles)

    dup(one_s, 1.0)

    with auto_sync():
        for tile_idx in range(tile_begin, tile_end):
            group0 = Var(tile_idx * G16_TILE_GROUPS)
            valid_groups = Min(G16_TILE_GROUPS, total_groups - group0)
            x_bf16_t = x_bf16[tile_idx]
            y_bf16_t = y_bf16[tile_idx]

            x_bf16_t[0:valid_groups, 0:G16_GROUP_SIZE] <<= x_group[group0:group0 + valid_groups, 0:G16_GROUP_SIZE]
            cast(
                x_float[0:valid_groups, 0:G16_GROUP_SIZE],
                x_bf16_t[0:valid_groups, 0:G16_GROUP_SIZE],
                repeat=valid_groups,
                count_per_rep=G16_GROUP_SIZE,
            )

            abs(
                abs_float[0:valid_groups, 0:G16_GROUP_SIZE],
                x_float[0:valid_groups, 0:G16_GROUP_SIZE],
                repeat=valid_groups,
                count_per_rep=G16_GROUP_SIZE,
            )
            cmax(scale_s, abs_float, repeat=valid_groups, count_per_rep=G16_GROUP_SIZE)
            muls(scale_s[0:1, 0:valid_groups], scale_s[0:1, 0:valid_groups], 1.0 / 7.0)
            cast(scale_bf16[0:1, 0:valid_groups], scale_s[0:1, 0:valid_groups], round_mode=RoundMode.AWAY_FROM_ZERO)
            cast(scale_s[0:1, 0:valid_groups], scale_bf16[0:1, 0:valid_groups])

            scale_safe_s[0:1, 0:valid_groups] <<= scale_s[0:1, 0:valid_groups]
            compare_scalar(flag, scale_safe_s[0:1, 0:valid_groups], 0.0, CompareMode.NE)
            select(scale_safe_s[0:1, 0:valid_groups], flag, scale_safe_s[0:1, 0:valid_groups], one_s, SelectMode.TENSOR_SCALAR)

            brcb(scale_bcast, scale_s, repeat=CeilDiv(valid_groups, 8), dst_blk_stride=1, dst_rep_stride=8)
            brcb(safe_scale_bcast, scale_safe_s, repeat=CeilDiv(valid_groups, 8), dst_blk_stride=1, dst_rep_stride=8)

            div(
                x_float[0:valid_groups, 0:G16_GROUP_SIZE],
                x_float[0:valid_groups, 0:G16_GROUP_SIZE],
                safe_scale_bcast[0:valid_groups, 0:SCALE_BCAST_COLS],
                repeat=valid_groups,
                count_per_rep=G16_GROUP_SIZE,
            )
            cast(
                q_int[0:valid_groups, 0:G16_GROUP_SIZE],
                x_float[0:valid_groups, 0:G16_GROUP_SIZE],
                repeat=valid_groups,
                round_mode=RoundMode.AWAY_FROM_ZERO,
                count_per_rep=G16_GROUP_SIZE,
            )
            cast(
                x_float[0:valid_groups, 0:G16_GROUP_SIZE],
                q_int[0:valid_groups, 0:G16_GROUP_SIZE],
                repeat=valid_groups,
                round_mode=RoundMode.NONE,
                count_per_rep=G16_GROUP_SIZE,
            )
            vmaxs(
                x_float[0:valid_groups, 0:G16_GROUP_SIZE],
                x_float[0:valid_groups, 0:G16_GROUP_SIZE],
                -8.0,
                repeat=valid_groups,
                count_per_rep=G16_GROUP_SIZE,
            )
            vmins(
                x_float[0:valid_groups, 0:G16_GROUP_SIZE],
                x_float[0:valid_groups, 0:G16_GROUP_SIZE],
                7.0,
                repeat=valid_groups,
                count_per_rep=G16_GROUP_SIZE,
            )
            mul(
                x_float[0:valid_groups, 0:G16_GROUP_SIZE],
                x_float[0:valid_groups, 0:G16_GROUP_SIZE],
                scale_bcast[0:valid_groups, 0:SCALE_BCAST_COLS],
                repeat=valid_groups,
                count_per_rep=G16_GROUP_SIZE,
            )

            cast(
                y_bf16_t[0:valid_groups, 0:G16_GROUP_SIZE],
                x_float[0:valid_groups, 0:G16_GROUP_SIZE],
                repeat=valid_groups,
                round_mode=RoundMode.AWAY_FROM_ZERO,
                count_per_rep=G16_GROUP_SIZE,
            )
            y_group[group0:group0 + valid_groups, 0:G16_GROUP_SIZE] <<= y_bf16_t[0:valid_groups, 0:G16_GROUP_SIZE]

    return y

# ----------------------------------------------------------------------------------------------------
# group32_bf16_fp4_e2m1.py
# Preserved corrected source; local contracts distinguish every precision and scale mode.
# ----------------------------------------------------------------------------------------------------

G32_GROUP_SIZE = 32
G32_TILE_GROUPS = 32


@func()
def quantize_abs_to_e2m1_g32_packed(abs_norm: Tensor, q_abs: Tensor, flag: Tensor, const_full: Tensor, groups: Var):
    dup(q_abs, 0.0, repeat=groups, count_per_rep=G32_GROUP_SIZE)

    dup(const_full, 0.5, repeat=groups, count_per_rep=G32_GROUP_SIZE)
    compare_scalar(flag, abs_norm, 0.25, CompareMode.LT, repeat=groups)
    select(q_abs, flag, q_abs, const_full, SelectMode.TENSOR_SCALAR, repeat=groups)

    dup(const_full, 1.0, repeat=groups, count_per_rep=G32_GROUP_SIZE)
    compare_scalar(flag, abs_norm, 0.75, CompareMode.LT, repeat=groups)
    select(q_abs, flag, q_abs, const_full, SelectMode.TENSOR_SCALAR, repeat=groups)

    dup(const_full, 1.5, repeat=groups, count_per_rep=G32_GROUP_SIZE)
    compare_scalar(flag, abs_norm, 1.25, CompareMode.LT, repeat=groups)
    select(q_abs, flag, q_abs, const_full, SelectMode.TENSOR_SCALAR, repeat=groups)

    dup(const_full, 2.0, repeat=groups, count_per_rep=G32_GROUP_SIZE)
    compare_scalar(flag, abs_norm, 1.75, CompareMode.LT, repeat=groups)
    select(q_abs, flag, q_abs, const_full, SelectMode.TENSOR_SCALAR, repeat=groups)

    dup(const_full, 3.0, repeat=groups, count_per_rep=G32_GROUP_SIZE)
    compare_scalar(flag, abs_norm, 2.5, CompareMode.LT, repeat=groups)
    select(q_abs, flag, q_abs, const_full, SelectMode.TENSOR_SCALAR, repeat=groups)

    dup(const_full, 4.0, repeat=groups, count_per_rep=G32_GROUP_SIZE)
    compare_scalar(flag, abs_norm, 3.5, CompareMode.LT, repeat=groups)
    select(q_abs, flag, q_abs, const_full, SelectMode.TENSOR_SCALAR, repeat=groups)

    dup(const_full, 6.0, repeat=groups, count_per_rep=G32_GROUP_SIZE)
    compare_scalar(flag, abs_norm, 5.0, CompareMode.LT, repeat=groups)
    select(q_abs, flag, q_abs, const_full, SelectMode.TENSOR_SCALAR, repeat=groups)


def group32_bf16_fp4_e2m1_kernel(x: GM[bf16, ('rows', 'cols')], y: GM[bf16, ('rows', 'cols')], rows: i32, cols: i32):
    groups_per_row = Var(cols // G32_GROUP_SIZE)
    total_groups = Var(rows * groups_per_row)

    x_group = x.reshape([total_groups, G32_GROUP_SIZE], name="x_group")
    y_group = y.reshape([total_groups, G32_GROUP_SIZE], name="y_group")

    x_bf16 = DBuff(DT.bfloat16, [G32_TILE_GROUPS, UB_GROUP_COLS], Position.UB)
    y_bf16 = DBuff(DT.bfloat16, [G32_TILE_GROUPS, UB_GROUP_COLS], Position.UB)

    x_float = Tensor(DT.float, [G32_TILE_GROUPS, UB_GROUP_COLS], Position.UB)
    abs_float = Tensor(DT.float, [G32_TILE_GROUPS, UB_GROUP_COLS], Position.UB)
    q_abs = Tensor(DT.float, [G32_TILE_GROUPS, UB_GROUP_COLS], Position.UB)
    neg_q_abs = Tensor(DT.float, [G32_TILE_GROUPS, UB_GROUP_COLS], Position.UB)
    const_full = Tensor(DT.float, [G32_TILE_GROUPS, UB_GROUP_COLS], Position.UB)

    scale_s = Tensor(DT.float, [1, SCALE_SLOTS], Position.UB)
    scale_safe_s = Tensor(DT.float, [1, SCALE_SLOTS], Position.UB)
    one_s = Tensor(DT.float, [1, SCALE_SLOTS], Position.UB)
    scale_bcast = Tensor(DT.float, [G32_TILE_GROUPS, SCALE_BCAST_COLS], Position.UB)
    safe_scale_bcast = Tensor(DT.float, [G32_TILE_GROUPS, SCALE_BCAST_COLS], Position.UB)
    scale_bf16 = Tensor(DT.bfloat16, [1, SCALE_SLOTS], Position.UB)

    flag = Tensor(DT.uint8, [G32_TILE_GROUPS, SCALE_BCAST_COLS], Position.UB)

    n_tiles = CeilDiv(total_groups, G32_TILE_GROUPS)
    tiles_per_core = CeilDiv(n_tiles, GetVecNum())
    tile_begin = Var(tiles_per_core * GetVecIdx())
    tile_end = Min(tile_begin + tiles_per_core, n_tiles)

    dup(one_s, 1.0)

    with auto_sync():
        for tile_idx in range(tile_begin, tile_end):
            group0 = Var(tile_idx * G32_TILE_GROUPS)
            valid_groups = Min(G32_TILE_GROUPS, total_groups - group0)
            x_bf16_t = x_bf16[tile_idx]
            y_bf16_t = y_bf16[tile_idx]

            x_bf16_t[0:valid_groups, 0:G32_GROUP_SIZE] <<= x_group[group0:group0 + valid_groups, 0:G32_GROUP_SIZE]
            cast(
                x_float[0:valid_groups, 0:G32_GROUP_SIZE],
                x_bf16_t[0:valid_groups, 0:G32_GROUP_SIZE],
                repeat=valid_groups,
                count_per_rep=G32_GROUP_SIZE,
            )

            abs(
                abs_float[0:valid_groups, 0:G32_GROUP_SIZE],
                x_float[0:valid_groups, 0:G32_GROUP_SIZE],
                repeat=valid_groups,
                count_per_rep=G32_GROUP_SIZE,
            )
            cmax(scale_s, abs_float, repeat=valid_groups, count_per_rep=G32_GROUP_SIZE)
            muls(scale_s[0:1, 0:valid_groups], scale_s[0:1, 0:valid_groups], 1.0 / 6.0)
            cast(scale_bf16[0:1, 0:valid_groups], scale_s[0:1, 0:valid_groups], round_mode=RoundMode.AWAY_FROM_ZERO)
            cast(scale_s[0:1, 0:valid_groups], scale_bf16[0:1, 0:valid_groups])

            scale_safe_s[0:1, 0:valid_groups] <<= scale_s[0:1, 0:valid_groups]
            compare_scalar(flag, scale_safe_s[0:1, 0:valid_groups], 0.0, CompareMode.NE)
            select(scale_safe_s[0:1, 0:valid_groups], flag, scale_safe_s[0:1, 0:valid_groups], one_s, SelectMode.TENSOR_SCALAR)

            brcb(scale_bcast, scale_s, repeat=CeilDiv(valid_groups, 8), dst_blk_stride=1, dst_rep_stride=8)
            brcb(safe_scale_bcast, scale_safe_s, repeat=CeilDiv(valid_groups, 8), dst_blk_stride=1, dst_rep_stride=8)

            div(
                x_float[0:valid_groups, 0:G32_GROUP_SIZE],
                x_float[0:valid_groups, 0:G32_GROUP_SIZE],
                safe_scale_bcast[0:valid_groups, 0:SCALE_BCAST_COLS],
                repeat=valid_groups,
                count_per_rep=G32_GROUP_SIZE,
            )
            abs(
                abs_float[0:valid_groups, 0:G32_GROUP_SIZE],
                x_float[0:valid_groups, 0:G32_GROUP_SIZE],
                repeat=valid_groups,
                count_per_rep=G32_GROUP_SIZE,
            )
            quantize_abs_to_e2m1_g32_packed(
                abs_float[0:valid_groups, 0:G32_GROUP_SIZE],
                q_abs[0:valid_groups, 0:G32_GROUP_SIZE],
                flag,
                const_full[0:valid_groups, 0:G32_GROUP_SIZE],
                valid_groups,
            )

            dup(const_full[0:valid_groups, 0:G32_GROUP_SIZE], 0.0, repeat=valid_groups, count_per_rep=G32_GROUP_SIZE)
            muls(
                neg_q_abs[0:valid_groups, 0:G32_GROUP_SIZE],
                q_abs[0:valid_groups, 0:G32_GROUP_SIZE],
                -1.0,
                repeat=valid_groups,
                count_per_rep=G32_GROUP_SIZE,
            )
            compare_scalar(flag, x_float[0:valid_groups, 0:G32_GROUP_SIZE], 0.0, CompareMode.GE, repeat=valid_groups)
            select(
                abs_float[0:valid_groups, 0:G32_GROUP_SIZE],
                flag,
                q_abs[0:valid_groups, 0:G32_GROUP_SIZE],
                const_full[0:valid_groups, 0:G32_GROUP_SIZE],
                SelectMode.TENSOR_SCALAR,
                repeat=valid_groups,
            )
            compare_scalar(flag, x_float[0:valid_groups, 0:G32_GROUP_SIZE], 0.0, CompareMode.LT, repeat=valid_groups)
            select(
                neg_q_abs[0:valid_groups, 0:G32_GROUP_SIZE],
                flag,
                neg_q_abs[0:valid_groups, 0:G32_GROUP_SIZE],
                const_full[0:valid_groups, 0:G32_GROUP_SIZE],
                SelectMode.TENSOR_SCALAR,
                repeat=valid_groups,
            )
            add(
                x_float[0:valid_groups, 0:G32_GROUP_SIZE],
                abs_float[0:valid_groups, 0:G32_GROUP_SIZE],
                neg_q_abs[0:valid_groups, 0:G32_GROUP_SIZE],
                repeat=valid_groups,
                count_per_rep=G32_GROUP_SIZE,
            )
            mul(
                x_float[0:valid_groups, 0:G32_GROUP_SIZE],
                x_float[0:valid_groups, 0:G32_GROUP_SIZE],
                scale_bcast[0:valid_groups, 0:SCALE_BCAST_COLS],
                repeat=valid_groups,
                count_per_rep=G32_GROUP_SIZE,
            )

            cast(
                y_bf16_t[0:valid_groups, 0:G32_GROUP_SIZE],
                x_float[0:valid_groups, 0:G32_GROUP_SIZE],
                repeat=valid_groups,
                round_mode=RoundMode.AWAY_FROM_ZERO,
                count_per_rep=G32_GROUP_SIZE,
            )
            y_group[group0:group0 + valid_groups, 0:G32_GROUP_SIZE] <<= y_bf16_t[0:valid_groups, 0:G32_GROUP_SIZE]

    return y

# ----------------------------------------------------------------------------------------------------
# group32_bf16_fp4_e1m2.py
# Preserved corrected source; local contracts distinguish every precision and scale mode.
# ----------------------------------------------------------------------------------------------------

def group32_bf16_fp4_e1m2_kernel(x: GM[bf16, ('rows', 'cols')], y: GM[bf16, ('rows', 'cols')], rows: i32, cols: i32):
    groups_per_row = Var(cols // G32_GROUP_SIZE)
    total_groups = Var(rows * groups_per_row)

    x_group = x.reshape([total_groups, G32_GROUP_SIZE], name="x_group")
    y_group = y.reshape([total_groups, G32_GROUP_SIZE], name="y_group")

    x_bf16 = DBuff(DT.bfloat16, [G32_TILE_GROUPS, UB_GROUP_COLS], Position.UB)
    y_bf16 = DBuff(DT.bfloat16, [G32_TILE_GROUPS, UB_GROUP_COLS], Position.UB)

    x_float = Tensor(DT.float, [G32_TILE_GROUPS, UB_GROUP_COLS], Position.UB)
    abs_float = Tensor(DT.float, [G32_TILE_GROUPS, UB_GROUP_COLS], Position.UB)
    q_int = Tensor(DT.int, [G32_TILE_GROUPS, UB_GROUP_COLS], Position.UB)

    scale_s = Tensor(DT.float, [1, SCALE_SLOTS], Position.UB)
    scale_safe_s = Tensor(DT.float, [1, SCALE_SLOTS], Position.UB)
    one_s = Tensor(DT.float, [1, SCALE_SLOTS], Position.UB)
    scale_bcast = Tensor(DT.float, [G32_TILE_GROUPS, SCALE_BCAST_COLS], Position.UB)
    safe_scale_bcast = Tensor(DT.float, [G32_TILE_GROUPS, SCALE_BCAST_COLS], Position.UB)
    scale_bf16 = Tensor(DT.bfloat16, [1, SCALE_SLOTS], Position.UB)

    flag = Tensor(DT.uint8, [G32_TILE_GROUPS, SCALE_BCAST_COLS], Position.UB)

    n_tiles = CeilDiv(total_groups, G32_TILE_GROUPS)
    tiles_per_core = CeilDiv(n_tiles, GetVecNum())
    tile_begin = Var(tiles_per_core * GetVecIdx())
    tile_end = Min(tile_begin + tiles_per_core, n_tiles)

    dup(one_s, 1.0)

    with auto_sync():
        for tile_idx in range(tile_begin, tile_end):
            group0 = Var(tile_idx * G32_TILE_GROUPS)
            valid_groups = Min(G32_TILE_GROUPS, total_groups - group0)
            x_bf16_t = x_bf16[tile_idx]
            y_bf16_t = y_bf16[tile_idx]

            x_bf16_t[0:valid_groups, 0:G32_GROUP_SIZE] <<= x_group[group0:group0 + valid_groups, 0:G32_GROUP_SIZE]
            cast(
                x_float[0:valid_groups, 0:G32_GROUP_SIZE],
                x_bf16_t[0:valid_groups, 0:G32_GROUP_SIZE],
                repeat=valid_groups,
                count_per_rep=G32_GROUP_SIZE,
            )

            abs(
                abs_float[0:valid_groups, 0:G32_GROUP_SIZE],
                x_float[0:valid_groups, 0:G32_GROUP_SIZE],
                repeat=valid_groups,
                count_per_rep=G32_GROUP_SIZE,
            )
            cmax(scale_s, abs_float, repeat=valid_groups, count_per_rep=G32_GROUP_SIZE)
            muls(scale_s[0:1, 0:valid_groups], scale_s[0:1, 0:valid_groups], 1.0 / 7.0)
            cast(scale_bf16[0:1, 0:valid_groups], scale_s[0:1, 0:valid_groups], round_mode=RoundMode.AWAY_FROM_ZERO)
            cast(scale_s[0:1, 0:valid_groups], scale_bf16[0:1, 0:valid_groups])

            scale_safe_s[0:1, 0:valid_groups] <<= scale_s[0:1, 0:valid_groups]
            compare_scalar(flag, scale_safe_s[0:1, 0:valid_groups], 0.0, CompareMode.NE)
            select(scale_safe_s[0:1, 0:valid_groups], flag, scale_safe_s[0:1, 0:valid_groups], one_s, SelectMode.TENSOR_SCALAR)

            brcb(scale_bcast, scale_s, repeat=CeilDiv(valid_groups, 8), dst_blk_stride=1, dst_rep_stride=8)
            brcb(safe_scale_bcast, scale_safe_s, repeat=CeilDiv(valid_groups, 8), dst_blk_stride=1, dst_rep_stride=8)

            div(
                x_float[0:valid_groups, 0:G32_GROUP_SIZE],
                x_float[0:valid_groups, 0:G32_GROUP_SIZE],
                safe_scale_bcast[0:valid_groups, 0:SCALE_BCAST_COLS],
                repeat=valid_groups,
                count_per_rep=G32_GROUP_SIZE,
            )
            cast(
                q_int[0:valid_groups, 0:G32_GROUP_SIZE],
                x_float[0:valid_groups, 0:G32_GROUP_SIZE],
                repeat=valid_groups,
                round_mode=RoundMode.AWAY_FROM_ZERO,
                count_per_rep=G32_GROUP_SIZE,
            )
            cast(
                x_float[0:valid_groups, 0:G32_GROUP_SIZE],
                q_int[0:valid_groups, 0:G32_GROUP_SIZE],
                repeat=valid_groups,
                round_mode=RoundMode.NONE,
                count_per_rep=G32_GROUP_SIZE,
            )
            vmaxs(
                x_float[0:valid_groups, 0:G32_GROUP_SIZE],
                x_float[0:valid_groups, 0:G32_GROUP_SIZE],
                -8.0,
                repeat=valid_groups,
                count_per_rep=G32_GROUP_SIZE,
            )
            vmins(
                x_float[0:valid_groups, 0:G32_GROUP_SIZE],
                x_float[0:valid_groups, 0:G32_GROUP_SIZE],
                7.0,
                repeat=valid_groups,
                count_per_rep=G32_GROUP_SIZE,
            )
            mul(
                x_float[0:valid_groups, 0:G32_GROUP_SIZE],
                x_float[0:valid_groups, 0:G32_GROUP_SIZE],
                scale_bcast[0:valid_groups, 0:SCALE_BCAST_COLS],
                repeat=valid_groups,
                count_per_rep=G32_GROUP_SIZE,
            )

            cast(
                y_bf16_t[0:valid_groups, 0:G32_GROUP_SIZE],
                x_float[0:valid_groups, 0:G32_GROUP_SIZE],
                repeat=valid_groups,
                round_mode=RoundMode.AWAY_FROM_ZERO,
                count_per_rep=G32_GROUP_SIZE,
            )
            y_group[group0:group0 + valid_groups, 0:G32_GROUP_SIZE] <<= y_bf16_t[0:valid_groups, 0:G32_GROUP_SIZE]

    return y

# ----------------------------------------------------------------------------------------------------
# group64_bf16_fp4_e2m1.py
# Preserved corrected source; local contracts distinguish every precision and scale mode.
# ----------------------------------------------------------------------------------------------------

G64_GROUP_SIZE = 64
G64_TILE_GROUPS = 64


@func()
def quantize_abs_to_e2m1(abs_norm: Tensor, q_abs: Tensor, flag: Tensor, const_full: Tensor):
    dup(q_abs, 0.0)

    dup(const_full, 0.5)
    compare_scalar(flag, abs_norm, 0.25, CompareMode.LT)
    select(q_abs, flag, q_abs, const_full, SelectMode.TENSOR_SCALAR)

    dup(const_full, 1.0)
    compare_scalar(flag, abs_norm, 0.75, CompareMode.LT)
    select(q_abs, flag, q_abs, const_full, SelectMode.TENSOR_SCALAR)

    dup(const_full, 1.5)
    compare_scalar(flag, abs_norm, 1.25, CompareMode.LT)
    select(q_abs, flag, q_abs, const_full, SelectMode.TENSOR_SCALAR)

    dup(const_full, 2.0)
    compare_scalar(flag, abs_norm, 1.75, CompareMode.LT)
    select(q_abs, flag, q_abs, const_full, SelectMode.TENSOR_SCALAR)

    dup(const_full, 3.0)
    compare_scalar(flag, abs_norm, 2.5, CompareMode.LT)
    select(q_abs, flag, q_abs, const_full, SelectMode.TENSOR_SCALAR)

    dup(const_full, 4.0)
    compare_scalar(flag, abs_norm, 3.5, CompareMode.LT)
    select(q_abs, flag, q_abs, const_full, SelectMode.TENSOR_SCALAR)

    dup(const_full, 6.0)
    compare_scalar(flag, abs_norm, 5.0, CompareMode.LT)
    select(q_abs, flag, q_abs, const_full, SelectMode.TENSOR_SCALAR)


def group64_bf16_fp4_e2m1_kernel(x: GM[bf16, ('rows', 'cols')], y: GM[bf16, ('rows', 'cols')], rows: i32, cols: i32):
    groups_per_row = Var(cols // G64_GROUP_SIZE)
    total_groups = Var(rows * groups_per_row)

    x_group = x.reshape([total_groups, G64_GROUP_SIZE], name="x_group")
    y_group = y.reshape([total_groups, G64_GROUP_SIZE], name="y_group")

    x_bf16 = DBuff(DT.bfloat16, [G64_TILE_GROUPS, G64_GROUP_SIZE], Position.UB)
    y_bf16 = DBuff(DT.bfloat16, [G64_TILE_GROUPS, G64_GROUP_SIZE], Position.UB)

    x_float = Tensor(DT.float, [G64_TILE_GROUPS, G64_GROUP_SIZE], Position.UB)
    abs_float = Tensor(DT.float, [G64_TILE_GROUPS, G64_GROUP_SIZE], Position.UB)
    q_abs = Tensor(DT.float, [G64_TILE_GROUPS, G64_GROUP_SIZE], Position.UB)
    neg_q_abs = Tensor(DT.float, [G64_TILE_GROUPS, G64_GROUP_SIZE], Position.UB)
    const_full = Tensor(DT.float, [G64_TILE_GROUPS, G64_GROUP_SIZE], Position.UB)

    scale_s = Tensor(DT.float, [1, G64_TILE_GROUPS], Position.UB)
    scale_safe_s = Tensor(DT.float, [1, G64_TILE_GROUPS], Position.UB)
    one_s = Tensor(DT.float, [1, G64_TILE_GROUPS], Position.UB)
    scale_bcast = Tensor(DT.float, [G64_TILE_GROUPS, SCALE_BCAST_COLS], Position.UB)
    safe_scale_bcast = Tensor(DT.float, [G64_TILE_GROUPS, SCALE_BCAST_COLS], Position.UB)
    scale_bf16 = Tensor(DT.bfloat16, [1, G64_TILE_GROUPS], Position.UB)

    flag = Tensor(DT.uint8, [G64_TILE_GROUPS, SCALE_BCAST_COLS], Position.UB)

    n_tiles = CeilDiv(total_groups, G64_TILE_GROUPS)
    tiles_per_core = CeilDiv(n_tiles, GetVecNum())
    tile_begin = Var(tiles_per_core * GetVecIdx())
    tile_end = Min(tile_begin + tiles_per_core, n_tiles)

    dup(one_s, 1.0)

    with auto_sync():
        for tile_idx in range(tile_begin, tile_end):
            group0 = Var(tile_idx * G64_TILE_GROUPS)
            valid_groups = Min(G64_TILE_GROUPS, total_groups - group0)
            x_bf16_t = x_bf16[tile_idx]
            y_bf16_t = y_bf16[tile_idx]

            x_bf16_t[0:valid_groups, 0:G64_GROUP_SIZE] <<= x_group[group0:group0 + valid_groups, 0:G64_GROUP_SIZE]
            cast(x_float[0:valid_groups, 0:G64_GROUP_SIZE], x_bf16_t[0:valid_groups, 0:G64_GROUP_SIZE])

            abs(abs_float[0:valid_groups, 0:G64_GROUP_SIZE], x_float[0:valid_groups, 0:G64_GROUP_SIZE])
            cmax(scale_s, abs_float, repeat=valid_groups)
            muls(scale_s[0:1, 0:valid_groups], scale_s[0:1, 0:valid_groups], 1.0 / 6.0)
            cast(scale_bf16[0:1, 0:valid_groups], scale_s[0:1, 0:valid_groups], round_mode=RoundMode.AWAY_FROM_ZERO)
            cast(scale_s[0:1, 0:valid_groups], scale_bf16[0:1, 0:valid_groups])

            scale_safe_s[0:1, 0:valid_groups] <<= scale_s[0:1, 0:valid_groups]
            compare_scalar(flag, scale_safe_s[0:1, 0:valid_groups], 0.0, CompareMode.NE)
            select(scale_safe_s[0:1, 0:valid_groups], flag, scale_safe_s[0:1, 0:valid_groups], one_s, SelectMode.TENSOR_SCALAR)

            brcb(scale_bcast, scale_s, repeat=CeilDiv(valid_groups, 8), dst_blk_stride=1, dst_rep_stride=8)
            brcb(safe_scale_bcast, scale_safe_s, repeat=CeilDiv(valid_groups, 8), dst_blk_stride=1, dst_rep_stride=8)

            div(x_float[0:valid_groups, 0:G64_GROUP_SIZE], x_float[0:valid_groups, 0:G64_GROUP_SIZE], safe_scale_bcast[0:valid_groups, 0:SCALE_BCAST_COLS])
            abs(abs_float[0:valid_groups, 0:G64_GROUP_SIZE], x_float[0:valid_groups, 0:G64_GROUP_SIZE])
            quantize_abs_to_e2m1(
                abs_float[0:valid_groups, 0:G64_GROUP_SIZE],
                q_abs[0:valid_groups, 0:G64_GROUP_SIZE],
                flag,
                const_full[0:valid_groups, 0:G64_GROUP_SIZE],
            )

            dup(const_full[0:valid_groups, 0:G64_GROUP_SIZE], 0.0)
            muls(neg_q_abs[0:valid_groups, 0:G64_GROUP_SIZE], q_abs[0:valid_groups, 0:G64_GROUP_SIZE], -1.0)
            compare_scalar(flag, x_float[0:valid_groups, 0:G64_GROUP_SIZE], 0.0, CompareMode.GE)
            select(abs_float[0:valid_groups, 0:G64_GROUP_SIZE], flag, q_abs[0:valid_groups, 0:G64_GROUP_SIZE], const_full[0:valid_groups, 0:G64_GROUP_SIZE], SelectMode.TENSOR_SCALAR)
            compare_scalar(flag, x_float[0:valid_groups, 0:G64_GROUP_SIZE], 0.0, CompareMode.LT)
            select(neg_q_abs[0:valid_groups, 0:G64_GROUP_SIZE], flag, neg_q_abs[0:valid_groups, 0:G64_GROUP_SIZE], const_full[0:valid_groups, 0:G64_GROUP_SIZE], SelectMode.TENSOR_SCALAR)
            add(x_float[0:valid_groups, 0:G64_GROUP_SIZE], abs_float[0:valid_groups, 0:G64_GROUP_SIZE], neg_q_abs[0:valid_groups, 0:G64_GROUP_SIZE])
            mul(x_float[0:valid_groups, 0:G64_GROUP_SIZE], x_float[0:valid_groups, 0:G64_GROUP_SIZE], scale_bcast[0:valid_groups, 0:SCALE_BCAST_COLS])

            cast(y_bf16_t[0:valid_groups, 0:G64_GROUP_SIZE], x_float[0:valid_groups, 0:G64_GROUP_SIZE], round_mode=RoundMode.AWAY_FROM_ZERO)
            y_group[group0:group0 + valid_groups, 0:G64_GROUP_SIZE] <<= y_bf16_t[0:valid_groups, 0:G64_GROUP_SIZE]

    return y

# ----------------------------------------------------------------------------------------------------
# group64_bf16_fp4_e1m2.py
# Preserved corrected source; local contracts distinguish every precision and scale mode.
# ----------------------------------------------------------------------------------------------------

def group64_bf16_fp4_e1m2_kernel(x: GM[bf16, ('rows', 'cols')], y: GM[bf16, ('rows', 'cols')], rows: i32, cols: i32):
    groups_per_row = Var(cols // G64_GROUP_SIZE)
    total_groups = Var(rows * groups_per_row)

    x_group = x.reshape([total_groups, G64_GROUP_SIZE], name="x_group")
    y_group = y.reshape([total_groups, G64_GROUP_SIZE], name="y_group")

    x_bf16 = DBuff(DT.bfloat16, [G64_TILE_GROUPS, G64_GROUP_SIZE], Position.UB)
    y_bf16 = DBuff(DT.bfloat16, [G64_TILE_GROUPS, G64_GROUP_SIZE], Position.UB)

    x_float = Tensor(DT.float, [G64_TILE_GROUPS, G64_GROUP_SIZE], Position.UB)
    abs_float = Tensor(DT.float, [G64_TILE_GROUPS, G64_GROUP_SIZE], Position.UB)
    q_int = Tensor(DT.int, [G64_TILE_GROUPS, G64_GROUP_SIZE], Position.UB)

    scale_s = Tensor(DT.float, [1, G64_TILE_GROUPS], Position.UB)
    scale_safe_s = Tensor(DT.float, [1, G64_TILE_GROUPS], Position.UB)
    one_s = Tensor(DT.float, [1, G64_TILE_GROUPS], Position.UB)
    scale_bcast = Tensor(DT.float, [G64_TILE_GROUPS, SCALE_BCAST_COLS], Position.UB)
    safe_scale_bcast = Tensor(DT.float, [G64_TILE_GROUPS, SCALE_BCAST_COLS], Position.UB)
    scale_bf16 = Tensor(DT.bfloat16, [1, G64_TILE_GROUPS], Position.UB)

    flag = Tensor(DT.uint8, [G64_TILE_GROUPS, SCALE_BCAST_COLS], Position.UB)

    n_tiles = CeilDiv(total_groups, G64_TILE_GROUPS)
    tiles_per_core = CeilDiv(n_tiles, GetVecNum())
    tile_begin = Var(tiles_per_core * GetVecIdx())
    tile_end = Min(tile_begin + tiles_per_core, n_tiles)

    dup(one_s, 1.0)

    with auto_sync():
        for tile_idx in range(tile_begin, tile_end):
            group0 = Var(tile_idx * G64_TILE_GROUPS)
            valid_groups = Min(G64_TILE_GROUPS, total_groups - group0)
            x_bf16_t = x_bf16[tile_idx]
            y_bf16_t = y_bf16[tile_idx]

            x_bf16_t[0:valid_groups, 0:G64_GROUP_SIZE] <<= x_group[group0:group0 + valid_groups, 0:G64_GROUP_SIZE]
            cast(x_float[0:valid_groups, 0:G64_GROUP_SIZE], x_bf16_t[0:valid_groups, 0:G64_GROUP_SIZE])

            abs(abs_float[0:valid_groups, 0:G64_GROUP_SIZE], x_float[0:valid_groups, 0:G64_GROUP_SIZE])
            cmax(scale_s, abs_float, repeat=valid_groups)
            muls(scale_s[0:1, 0:valid_groups], scale_s[0:1, 0:valid_groups], 1.0 / 7.0)
            cast(scale_bf16[0:1, 0:valid_groups], scale_s[0:1, 0:valid_groups], round_mode=RoundMode.AWAY_FROM_ZERO)
            cast(scale_s[0:1, 0:valid_groups], scale_bf16[0:1, 0:valid_groups])

            scale_safe_s[0:1, 0:valid_groups] <<= scale_s[0:1, 0:valid_groups]
            compare_scalar(flag, scale_safe_s[0:1, 0:valid_groups], 0.0, CompareMode.NE)
            select(scale_safe_s[0:1, 0:valid_groups], flag, scale_safe_s[0:1, 0:valid_groups], one_s, SelectMode.TENSOR_SCALAR)

            brcb(scale_bcast, scale_s, repeat=CeilDiv(valid_groups, 8), dst_blk_stride=1, dst_rep_stride=8)
            brcb(safe_scale_bcast, scale_safe_s, repeat=CeilDiv(valid_groups, 8), dst_blk_stride=1, dst_rep_stride=8)

            div(x_float[0:valid_groups, 0:G64_GROUP_SIZE], x_float[0:valid_groups, 0:G64_GROUP_SIZE], safe_scale_bcast[0:valid_groups, 0:SCALE_BCAST_COLS])
            cast(q_int[0:valid_groups, 0:G64_GROUP_SIZE], x_float[0:valid_groups, 0:G64_GROUP_SIZE], round_mode=RoundMode.AWAY_FROM_ZERO)
            cast(x_float[0:valid_groups, 0:G64_GROUP_SIZE], q_int[0:valid_groups, 0:G64_GROUP_SIZE], round_mode=RoundMode.NONE)
            vmaxs(x_float[0:valid_groups, 0:G64_GROUP_SIZE], x_float[0:valid_groups, 0:G64_GROUP_SIZE], -8.0)
            vmins(x_float[0:valid_groups, 0:G64_GROUP_SIZE], x_float[0:valid_groups, 0:G64_GROUP_SIZE], 7.0)
            mul(x_float[0:valid_groups, 0:G64_GROUP_SIZE], x_float[0:valid_groups, 0:G64_GROUP_SIZE], scale_bcast[0:valid_groups, 0:SCALE_BCAST_COLS])

            cast(y_bf16_t[0:valid_groups, 0:G64_GROUP_SIZE], x_float[0:valid_groups, 0:G64_GROUP_SIZE], round_mode=RoundMode.AWAY_FROM_ZERO)
            y_group[group0:group0 + valid_groups, 0:G64_GROUP_SIZE] <<= y_bf16_t[0:valid_groups, 0:G64_GROUP_SIZE]

    return y

# ----------------------------------------------------------------------------------------------------
# hifx.py
# Preserved corrected source; local contracts distinguish every precision and scale mode.
# ----------------------------------------------------------------------------------------------------

HIFX_GROUP_SIZE = 64

# ---------------------------------------------------------------------------
# PyTorch reference (formerly to_hifx_torch.py). HiFX quantization runs along the
# last axis: each contiguous 64-element block forms one group sharing a level-1
# E6M2 scale, an 8-way level-2 E1 exponent, and a 16-way level-3 E1 exponent;
# every datum is then rounded half-away-from-zero into the in-group N-bit value
# (Ng = N-2 fractional bits; HiF4 -> 2, HiF5 -> 3). Equivalent to the numpy
# To_HiFX under transpose: to_hifx(x) == To_HiFX(x.T).T.
# ---------------------------------------------------------------------------
# E6M2: 2^[-48, 15] * 1.M2; 2^15 * 1.75 is NaN, so the max finite is 2^15 * 1.5.
E6M2_MAX = 2.0 ** 15 * 1.5
E6M2_MIN = 2.0 ** (-48)
# Max abs from level-2&3 * in-group 1.M2 scaling: 2^2 * 1.75 = 7.
LEVEL23_MAX = 7.0




# Groups processed per vec tile. The per-8 / per-4 reductions and the per-4 fold
# issue ops with repeat = HIFX_TILE_GROUPS * 8 (8 sub-blocks per 64-lane group). The
# AscendC vector repeatTimes is uint8 (max 255), so HIFX_TILE_GROUPS must satisfy
# HIFX_TILE_GROUPS * 8 <= 255 -> HIFX_TILE_GROUPS <= 31. At 32 the repeat is 256 and wraps
# on real HW (the simulator uses an int and does not, so it silently passes).
# 24 keeps repeat at 192 with margin.
HIFX_TILE_GROUPS = 24
# Per-group scalar buffers ([1, *]) must be a full fp32 vector wide. A scalar op
# with repeat=CeilDiv(valid, 64) runs under the full 64-lane mask, so it always
# touches 64 elements regardless of how many groups (<= HIFX_TILE_GROUPS) are live.
# Sizing these at HIFX_TILE_GROUPS=32 makes the op overrun by 32 elements into the
# next UB tensor: the simulator clips to the logical extent and hides it, but
# real HW corrupts the neighbour and the whole tile collapses to zeros. 64 wide
# keeps every implicit full-mask scalar op in bounds (matches mbs_mxfp4's 64).
SCALAR_COLS = 64

# N parametrization (HiF4 -> N=4 / NG=2, HiF5 -> N=5 / NG=3). Select with env
# HIFX_N; the kernel bakes the resulting scalar constants at trace time.
NG = 2  # Named wrappers below select the two preserved source modes.
NG_SCALE = float(2 ** NG)        # 2^Ng
NG_INV = float(2 ** (-NG))       # 2^-Ng
OVER = 2.0 - NG_INV              # in-group overflow clamp

# bf16(1/7): the level-1 reciprocal constant (To_BF16(1/7) in the reference).
CONST_REC = 0.142578125  # Exact source BF16-even(1/7), bits0x3e12.


@func()
def _hifx_body(x, y, rows, cols, NG_SCALE, NG_INV, OVER):
    groups_per_row = Var(cols // HIFX_GROUP_SIZE)
    total_groups = Var(rows * groups_per_row)

    x_group = x.reshape([total_groups, HIFX_GROUP_SIZE], name="x_group")
    y_group = y.reshape([total_groups, HIFX_GROUP_SIZE], name="y_group")

    x_bf16 = DBuff(DT.bfloat16, [HIFX_TILE_GROUPS, HIFX_GROUP_SIZE], Position.UB)
    y_bf16 = DBuff(DT.bfloat16, [HIFX_TILE_GROUPS, HIFX_GROUP_SIZE], Position.UB)

    x_float = Tensor(DT.float, [HIFX_TILE_GROUPS, HIFX_GROUP_SIZE], Position.UB)
    absf = Tensor(DT.float, [HIFX_TILE_GROUPS, HIFX_GROUP_SIZE], Position.UB)
    neg_mag = Tensor(DT.float, [HIFX_TILE_GROUPS, HIFX_GROUP_SIZE], Position.UB)
    pos_mag = Tensor(DT.float, [HIFX_TILE_GROUPS, HIFX_GROUP_SIZE], Position.UB)
    grp_int = Tensor(DT.int, [HIFX_TILE_GROUPS, HIFX_GROUP_SIZE], Position.UB)
    flag = Tensor(DT.uint8, [HIFX_TILE_GROUPS, HIFX_GROUP_SIZE], Position.UB)

    # level-2 (per-8 block) exponent state, kept full-width so every op is one
    # window per row (avoids the [TILE,8] narrow-row repeat hazard).
    v8_full = Tensor(DT.float, [HIFX_TILE_GROUPS, HIFX_GROUP_SIZE], Position.UB)
    e1flag = Tensor(DT.uint8, [HIFX_TILE_GROUPS, HIFX_GROUP_SIZE], Position.UB)
    pow_neg_full = Tensor(DT.float, [HIFX_TILE_GROUPS, HIFX_GROUP_SIZE], Position.UB)
    pow_pos_full = Tensor(DT.float, [HIFX_TILE_GROUPS, HIFX_GROUP_SIZE], Position.UB)
    one_full = Tensor(DT.float, [HIFX_TILE_GROUPS, HIFX_GROUP_SIZE], Position.UB)
    zero_full = Tensor(DT.float, [HIFX_TILE_GROUPS, HIFX_GROUP_SIZE], Position.UB)

    # level-3 (per-4 even/odd) exponent state
    v16e_full = Tensor(DT.float, [HIFX_TILE_GROUPS, HIFX_GROUP_SIZE], Position.UB)
    v16o_full = Tensor(DT.float, [HIFX_TILE_GROUPS, HIFX_GROUP_SIZE], Position.UB)
    e16e_flag = Tensor(DT.uint8, [HIFX_TILE_GROUPS, HIFX_GROUP_SIZE], Position.UB)
    e16o_flag = Tensor(DT.uint8, [HIFX_TILE_GROUPS, HIFX_GROUP_SIZE], Position.UB)
    half_full = Tensor(DT.float, [HIFX_TILE_GROUPS, HIFX_GROUP_SIZE], Position.UB)
    two_full = Tensor(DT.float, [HIFX_TILE_GROUPS, HIFX_GROUP_SIZE], Position.UB)
    v16e_s = Tensor(DT.float, [HIFX_TILE_GROUPS, SCALE_BCAST_COLS], Position.UB)
    v16o_s = Tensor(DT.float, [HIFX_TILE_GROUPS, SCALE_BCAST_COLS], Position.UB)

    vmax_s = Tensor(DT.float, [1, SCALAR_COLS], Position.UB)
    v8_s = Tensor(DT.float, [HIFX_TILE_GROUPS, SCALE_BCAST_COLS], Position.UB)
    sf_s = Tensor(DT.float, [1, SCALAR_COLS], Position.UB)
    e6m2_s = Tensor(DT.float, [1, SCALAR_COLS], Position.UB)
    exp_s = Tensor(DT.float, [1, SCALAR_COLS], Position.UB)
    t_s = Tensor(DT.float, [1, SCALAR_COLS], Position.UB)
    rec_s = Tensor(DT.float, [1, SCALAR_COLS], Position.UB)
    one_s = Tensor(DT.float, [1, SCALAR_COLS], Position.UB)
    t_int = Tensor(DT.int, [1, SCALAR_COLS], Position.UB)
    sf_bf16 = Tensor(DT.bfloat16, [1, SCALAR_COLS], Position.UB)
    rec_bf16 = Tensor(DT.bfloat16, [1, SCALAR_COLS], Position.UB)
    exp_mask_i32 = Tensor(DT.int, [1, SCALAR_COLS], Position.UB)

    e6m2_bcast = Tensor(DT.float, [HIFX_TILE_GROUPS, SCALE_BCAST_COLS], Position.UB)
    rec_bcast = Tensor(DT.float, [HIFX_TILE_GROUPS, SCALE_BCAST_COLS], Position.UB)

    n_tiles = CeilDiv(total_groups, HIFX_TILE_GROUPS)
    tiles_per_core = CeilDiv(n_tiles, GetVecNum())
    tile_begin = Var(tiles_per_core * GetVecIdx())
    tile_end = Min(tile_begin + tiles_per_core, n_tiles)

    dup(one_s, 1.0)
    dup(exp_mask_i32, FP32_EXP_MASK)
    dup(one_full, 1.0)
    dup(zero_full, 0.0)
    dup(half_full, 0.5)
    dup(two_full, 2.0)

    with auto_sync():
        for tile_idx in range(tile_begin, tile_end):
            group0 = Var(tile_idx * HIFX_TILE_GROUPS)
            valid = Min(HIFX_TILE_GROUPS, total_groups - group0)
            x_bf16_t = x_bf16[tile_idx]
            y_bf16_t = y_bf16[tile_idx]

            x_bf16_t[0:valid, 0:HIFX_GROUP_SIZE] <<= x_group[group0:group0 + valid, 0:HIFX_GROUP_SIZE]
            cast(x_float[0:valid, 0:HIFX_GROUP_SIZE], x_bf16_t[0:valid, 0:HIFX_GROUP_SIZE])
            abs(absf[0:valid, 0:HIFX_GROUP_SIZE], x_float[0:valid, 0:HIFX_GROUP_SIZE])

            # --- level-1 scale: Vmax (per-64) and V8 (per-8) ---
            cmax(vmax_s, absf, repeat=valid)
            cmax(v8_s, absf, repeat=valid * 8, src_rep_stride=1, count_per_rep=8)
            cmax(v16e_s, absf, repeat=valid * 8, src_rep_stride=1, count_per_rep=4)
            # Odd per-4 max WITHOUT a 16B-unaligned source read. Reading the odd
            # half-block via absf[:, 4:] is element-4 (16-byte) unaligned: the
            # simulator honors it, but real A2 ignores the offset and collapses
            # V16_odd onto V16_even, giving an off-by-one DE on every odd
            # half-block. Since abs values are >= 0, zero each block's even half
            # (lanes 0-3, 32B-aligned dst) in a scratch copy, then the per-8 max
            # recovers max(lanes 4-7). neg_mag is reused here as scratch (it is
            # not live until the level-3 fold below).
            adds(neg_mag[0:valid, 0:HIFX_GROUP_SIZE], absf[0:valid, 0:HIFX_GROUP_SIZE], 0.0)
            dup(neg_mag[0:valid, 0:HIFX_GROUP_SIZE], 0.0, repeat=valid * 8, dst_rep_stride=1, count_per_rep=4)
            cmax(v16o_s, neg_mag, repeat=valid * 8, src_rep_stride=1, count_per_rep=8)

            muls(sf_s[0:1, 0:valid], vmax_s[0:1, 0:valid], CONST_REC)
            cast(sf_bf16[0:1, 0:valid], sf_s[0:1, 0:valid], round_mode=RoundMode.TO_EVEN)
            cast(sf_s[0:1, 0:valid], sf_bf16[0:1, 0:valid])

            # bf16_to_e6m2: clamp, extract 2^E via exponent mask, round mantissa to 2 bits
            vmaxs(e6m2_s[0:1, 0:valid], sf_s[0:1, 0:valid], E6M2_MIN)
            vmins(e6m2_s[0:1, 0:valid], e6m2_s[0:1, 0:valid], E6M2_MAX)
            e6m2_i16 = e6m2_s.reinterpret(DT.int16)
            exp_i16 = exp_s.reinterpret(DT.int16)
            mask_i16 = exp_mask_i32.reinterpret(DT.int16)
            vand(exp_i16[0:1, 0:2 * valid], e6m2_i16[0:1, 0:2 * valid], mask_i16[0:1, 0:2 * valid])
            div(t_s[0:1, 0:valid], e6m2_s[0:1, 0:valid], exp_s[0:1, 0:valid])
            muls(t_s[0:1, 0:valid], t_s[0:1, 0:valid], 4.0)
            cast(t_int[0:1, 0:valid], t_s[0:1, 0:valid], round_mode=RoundMode.TO_EVEN)
            cast(t_s[0:1, 0:valid], t_int[0:1, 0:valid])
            mul(e6m2_s[0:1, 0:valid], t_s[0:1, 0:valid], exp_s[0:1, 0:valid])
            muls(e6m2_s[0:1, 0:valid], e6m2_s[0:1, 0:valid], 0.25)

            div(rec_s[0:1, 0:valid], one_s[0:1, 0:valid], e6m2_s[0:1, 0:valid])
            cast(rec_bf16[0:1, 0:valid], rec_s[0:1, 0:valid], round_mode=RoundMode.TO_EVEN)
            cast(rec_s[0:1, 0:valid], rec_bf16[0:1, 0:valid])

            brcb(rec_bcast, rec_s, repeat=CeilDiv(valid, 8), dst_blk_stride=1, dst_rep_stride=8)
            brcb(e6m2_bcast, e6m2_s, repeat=CeilDiv(valid, 8), dst_blk_stride=1, dst_rep_stride=8)

            # --- level-2: E1_8 = (V8*REC >= 4); per-block 2^-E1_8 and 2^E1_8 ---
            brcb(v8_full, v8_s, repeat=valid, dst_blk_stride=1, dst_rep_stride=8)
            mul(v8_full[0:valid, 0:HIFX_GROUP_SIZE], v8_full[0:valid, 0:HIFX_GROUP_SIZE], rec_bcast[0:valid, 0:SCALE_BCAST_COLS])
            compare_scalar(e1flag[0:valid, 0:HIFX_GROUP_SIZE], v8_full[0:valid, 0:HIFX_GROUP_SIZE], 4.0, CompareMode.GE)
            select(v8_full[0:valid, 0:HIFX_GROUP_SIZE], e1flag[0:valid, 0:HIFX_GROUP_SIZE],
                   one_full[0:valid, 0:HIFX_GROUP_SIZE], zero_full[0:valid, 0:HIFX_GROUP_SIZE], SelectMode.TENSOR_SCALAR)
            muls(pow_neg_full[0:valid, 0:HIFX_GROUP_SIZE], v8_full[0:valid, 0:HIFX_GROUP_SIZE], -0.5)
            adds(pow_neg_full[0:valid, 0:HIFX_GROUP_SIZE], pow_neg_full[0:valid, 0:HIFX_GROUP_SIZE], 1.0)
            adds(pow_pos_full[0:valid, 0:HIFX_GROUP_SIZE], v8_full[0:valid, 0:HIFX_GROUP_SIZE], 1.0)

            # --- level-3: E1_16 = (V16 * REC * 2^-E1_8 >= 2), per-4 even/odd ---
            brcb(v16e_full, v16e_s, repeat=valid, dst_blk_stride=1, dst_rep_stride=8)
            mul(v16e_full[0:valid, 0:HIFX_GROUP_SIZE], v16e_full[0:valid, 0:HIFX_GROUP_SIZE], rec_bcast[0:valid, 0:SCALE_BCAST_COLS])
            mul(v16e_full[0:valid, 0:HIFX_GROUP_SIZE], v16e_full[0:valid, 0:HIFX_GROUP_SIZE], pow_neg_full[0:valid, 0:HIFX_GROUP_SIZE])
            compare_scalar(e16e_flag[0:valid, 0:HIFX_GROUP_SIZE], v16e_full[0:valid, 0:HIFX_GROUP_SIZE], 2.0, CompareMode.GE)
            brcb(v16o_full, v16o_s, repeat=valid, dst_blk_stride=1, dst_rep_stride=8)
            mul(v16o_full[0:valid, 0:HIFX_GROUP_SIZE], v16o_full[0:valid, 0:HIFX_GROUP_SIZE], rec_bcast[0:valid, 0:SCALE_BCAST_COLS])
            mul(v16o_full[0:valid, 0:HIFX_GROUP_SIZE], v16o_full[0:valid, 0:HIFX_GROUP_SIZE], pow_neg_full[0:valid, 0:HIFX_GROUP_SIZE])
            compare_scalar(e16o_flag[0:valid, 0:HIFX_GROUP_SIZE], v16o_full[0:valid, 0:HIFX_GROUP_SIZE], 2.0, CompareMode.GE)

            # fold per-4 2^-E1_16 into pow_neg_full (even half = lanes 0..3 of each block)
            select(v16e_full[0:valid, 0:HIFX_GROUP_SIZE], e16e_flag[0:valid, 0:HIFX_GROUP_SIZE],
                   half_full[0:valid, 0:HIFX_GROUP_SIZE], one_full[0:valid, 0:HIFX_GROUP_SIZE], SelectMode.TENSOR_SCALAR)
            select(v16o_full[0:valid, 0:HIFX_GROUP_SIZE], e16o_flag[0:valid, 0:HIFX_GROUP_SIZE],
                   half_full[0:valid, 0:HIFX_GROUP_SIZE], one_full[0:valid, 0:HIFX_GROUP_SIZE], SelectMode.TENSOR_SCALAR)
            adds(neg_mag[0:valid, 0:HIFX_GROUP_SIZE], v16o_full[0:valid, 0:HIFX_GROUP_SIZE], 0.0)
            adds(neg_mag[0:valid, 0:HIFX_GROUP_SIZE], v16e_full[0:valid, 0:HIFX_GROUP_SIZE], 0.0,
                 repeat=valid * 8, dst_rep_stride=1, src_rep_stride=1, count_per_rep=4)
            mul(pow_neg_full[0:valid, 0:HIFX_GROUP_SIZE], pow_neg_full[0:valid, 0:HIFX_GROUP_SIZE], neg_mag[0:valid, 0:HIFX_GROUP_SIZE])

            # fold per-4 2^E1_16 into pow_pos_full
            select(v16e_full[0:valid, 0:HIFX_GROUP_SIZE], e16e_flag[0:valid, 0:HIFX_GROUP_SIZE],
                   two_full[0:valid, 0:HIFX_GROUP_SIZE], one_full[0:valid, 0:HIFX_GROUP_SIZE], SelectMode.TENSOR_SCALAR)
            select(v16o_full[0:valid, 0:HIFX_GROUP_SIZE], e16o_flag[0:valid, 0:HIFX_GROUP_SIZE],
                   two_full[0:valid, 0:HIFX_GROUP_SIZE], one_full[0:valid, 0:HIFX_GROUP_SIZE], SelectMode.TENSOR_SCALAR)
            adds(pos_mag[0:valid, 0:HIFX_GROUP_SIZE], v16o_full[0:valid, 0:HIFX_GROUP_SIZE], 0.0)
            adds(pos_mag[0:valid, 0:HIFX_GROUP_SIZE], v16e_full[0:valid, 0:HIFX_GROUP_SIZE], 0.0,
                 repeat=valid * 8, dst_rep_stride=1, src_rep_stride=1, count_per_rep=4)
            mul(pow_pos_full[0:valid, 0:HIFX_GROUP_SIZE], pow_pos_full[0:valid, 0:HIFX_GROUP_SIZE], pos_mag[0:valid, 0:HIFX_GROUP_SIZE])

            # --- in-group: igpv_bf16 = bf16(abs * REC * 2^-DE) ---
            mul(absf[0:valid, 0:HIFX_GROUP_SIZE], absf[0:valid, 0:HIFX_GROUP_SIZE], rec_bcast[0:valid, 0:SCALE_BCAST_COLS])
            mul(absf[0:valid, 0:HIFX_GROUP_SIZE], absf[0:valid, 0:HIFX_GROUP_SIZE], pow_neg_full[0:valid, 0:HIFX_GROUP_SIZE])
            cast(x_bf16_t[0:valid, 0:HIFX_GROUP_SIZE], absf[0:valid, 0:HIFX_GROUP_SIZE], round_mode=RoundMode.TO_EVEN)
            cast(absf[0:valid, 0:HIFX_GROUP_SIZE], x_bf16_t[0:valid, 0:HIFX_GROUP_SIZE])

            # RHA round: in_grp = floor(igpv*2^Ng + 0.5) * 2^-Ng, clamp to OVER
            muls(absf[0:valid, 0:HIFX_GROUP_SIZE], absf[0:valid, 0:HIFX_GROUP_SIZE], NG_SCALE)
            adds(absf[0:valid, 0:HIFX_GROUP_SIZE], absf[0:valid, 0:HIFX_GROUP_SIZE], 0.5)
            cast(grp_int[0:valid, 0:HIFX_GROUP_SIZE], absf[0:valid, 0:HIFX_GROUP_SIZE], round_mode=RoundMode.FLOOR)
            cast(absf[0:valid, 0:HIFX_GROUP_SIZE], grp_int[0:valid, 0:HIFX_GROUP_SIZE])
            muls(absf[0:valid, 0:HIFX_GROUP_SIZE], absf[0:valid, 0:HIFX_GROUP_SIZE], NG_INV)
            vmins(absf[0:valid, 0:HIFX_GROUP_SIZE], absf[0:valid, 0:HIFX_GROUP_SIZE], OVER)

            # --- reconstruct: grp = sign * E6M2 * 2^DE * in_grp ---
            mul(absf[0:valid, 0:HIFX_GROUP_SIZE], absf[0:valid, 0:HIFX_GROUP_SIZE], e6m2_bcast[0:valid, 0:SCALE_BCAST_COLS])
            mul(absf[0:valid, 0:HIFX_GROUP_SIZE], absf[0:valid, 0:HIFX_GROUP_SIZE], pow_pos_full[0:valid, 0:HIFX_GROUP_SIZE])
            muls(neg_mag[0:valid, 0:HIFX_GROUP_SIZE], absf[0:valid, 0:HIFX_GROUP_SIZE], -1.0)
            compare_scalar(flag[0:valid, 0:HIFX_GROUP_SIZE], x_float[0:valid, 0:HIFX_GROUP_SIZE], 0.0, CompareMode.GE)
            select(pos_mag[0:valid, 0:HIFX_GROUP_SIZE], flag[0:valid, 0:HIFX_GROUP_SIZE],
                   absf[0:valid, 0:HIFX_GROUP_SIZE], zero_full[0:valid, 0:HIFX_GROUP_SIZE], SelectMode.TENSOR_SCALAR)
            compare_scalar(flag[0:valid, 0:HIFX_GROUP_SIZE], x_float[0:valid, 0:HIFX_GROUP_SIZE], 0.0, CompareMode.LT)
            select(neg_mag[0:valid, 0:HIFX_GROUP_SIZE], flag[0:valid, 0:HIFX_GROUP_SIZE],
                   neg_mag[0:valid, 0:HIFX_GROUP_SIZE], zero_full[0:valid, 0:HIFX_GROUP_SIZE], SelectMode.TENSOR_SCALAR)
            add(absf[0:valid, 0:HIFX_GROUP_SIZE], pos_mag[0:valid, 0:HIFX_GROUP_SIZE], neg_mag[0:valid, 0:HIFX_GROUP_SIZE])

            cast(y_bf16_t[0:valid, 0:HIFX_GROUP_SIZE], absf[0:valid, 0:HIFX_GROUP_SIZE], round_mode=RoundMode.TO_EVEN)
            y_group[group0:group0 + valid, 0:HIFX_GROUP_SIZE] <<= y_bf16_t[0:valid, 0:HIFX_GROUP_SIZE]
            # Drain the UB->GM (MTE3) store before the tile's DBuff slot is reused
            # by the next tile's load. Defensive multi-tile synchronization that
            # mirrors mbs_mxfp4; not required for single-tile shapes.
            bar_mte3()

    return y


def hifx4(x: GM[bf16, ('rows', 'cols')], y: GM[bf16, ('rows', 'cols')], rows: i32, cols: i32):
    return _hifx_body(x, y, rows, cols, 4.0, 0.25, 1.75)


def hifx5(x: GM[bf16, ('rows', 'cols')], y: GM[bf16, ('rows', 'cols')], rows: i32, cols: i32):
    return _hifx_body(x, y, rows, cols, 8.0, 0.125, 1.875)


# ----------------------------------------------------------------------------------------------------
# dispatcher
# Every entry has the same signature -- (x, y, rows, cols) -- so a case selects one by name and
# changes nothing else. The mode is not uniform: the MX bodies are `vec`, while the group and
# HiFX bodies were written as `mix` launches, which is recorded per entry rather than assumed.
# ----------------------------------------------------------------------------------------------------

ENTRIES = {'mbs_mxfp4_kernel': 'mix',
           'mxfp4_kernel_bf16': 'vec',
           'mxfp8e5m2_kernel_bf16': 'vec',
           'group16_bf16_fp4_e2m1_kernel': 'mix',
           'group16_bf16_fp4_e1m2_kernel': 'mix',
           'group32_bf16_fp4_e2m1_kernel': 'mix',
           'group32_bf16_fp4_e1m2_kernel': 'mix',
           'group64_bf16_fp4_e2m1_kernel': 'mix',
           'group64_bf16_fp4_e1m2_kernel': 'mix',
           'hifx4': 'mix',
           'hifx5': 'mix'}


@lru_cache(maxsize=16)
def build_kernel(device, entry):
    """Bind one named entry to the A2 or A3 facade, in the launch mode that entry was written
    for. The cache means a case list that repeats an entry elaborates it once."""
    if device not in ("a2", "a3") or entry not in ENTRIES:
        raise ValueError("Select a preserved A2/A3 entry")
    return import_module(f"ascriptor.{device}").kernel(mode=ENTRIES[entry])(globals()[entry])
