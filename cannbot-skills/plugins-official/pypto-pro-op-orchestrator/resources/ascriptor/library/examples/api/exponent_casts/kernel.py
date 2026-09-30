# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F403, F405, F841
"""Reviewed production closures; host inputs/references are local and independent.

Pure-vector teaching units explicitly launch one vector participant. Original
mode=vec bodies, masks, byte footprints and overlapping-write barriers remain.
"""

# SHA256: 8f959b4ef2ae5f65416fd5f9eb4f2746003c8629500e752b56503d8b04318ea1

from ascriptor.a5 import *  # noqa: F401,F403

COLS = 128  # 128 bf16 lanes = 256 bytes = one full vector register

ROWS = 4


@vf()
def e8m0_from_bf16_vf(src: Tensor, dst: Tensor):
    m_u8 = MaskReg(DT.uint8, name="q_m_u8")
    m_low = MaskReg(DT.uint8, init_mode=MaskType.LOWEST128, name="q_m_low")
    narrow = CastConfig(saturate=True, reg_layout=RegLayout.ZERO, name="e8m0_narrow")

    x = Reg(DT.bfloat16, name="q_x")
    x <<= src[0]
    bits = x.reinterpret(DT.uint16, name="q_bits")

    shifted = Reg(DT.uint16, name="q_shifted")
    shiftls(shifted, bits, 1)  # the sign bit falls off the top
    shiftrs(shifted, shifted, 8)  # ... leaving the exponent field in the low byte

    sparse = Reg(DT.uint8, name="q_sparse")
    sparse_u16 = sparse.reinterpret(DT.uint16, name="q_sparse_u16")
    dense = Reg(DT.uint8, name="q_dense")
    cast(sparse, shifted, narrow, m_u8)  # u16 -> u8, one code per even byte
    pack(dense, sparse_u16, HighLowPart.LOWEST)  # ... compacted into the low 128 bytes
    # DIST_NORM_B8 writes a whole register even under a prefix mask, so `dst` is a full 256 bytes
    # and only its first COLS carry codes.
    reg_to_ub_normal(dst[0], dense, m_low)


@vf()
def bf16_from_e8m0_vf(src: Tensor, dst: Tensor):
    m_u16 = MaskReg(DT.uint16, name="d_m_u16")
    nan_m = MaskReg(DT.uint16, init_mode=MaskType.NONE, name="d_nan")
    widen = CastConfig(reg_layout=RegLayout.ZERO, name="e8m0_widen")

    codes = Reg(DT.uint8, name="d_codes")
    ub_to_reg_unpack(codes, src[0])  # 128 dense codes -> the even byte of every u16 lane

    wide = Reg(DT.uint16, name="d_wide")
    cast(wide, codes, widen, m_u16)
    out = Reg(DT.uint16, name="d_out")
    shiftls(out, wide, 7)  # exponent into bits 14..7, mantissa zero, sign zero

    # code 255 would decode to +inf; the instruction the doc describes yields NaN, so patch it.
    nan_v = Reg(DT.uint16, name="d_nanv")
    nan_v <<= 0x7FC0
    compare(nan_m, out, 0x7F80, CompareMode.EQ)
    fixed = Reg(DT.uint16, name="d_fixed")
    select(fixed, nan_v, out, mask=nan_m)
    dst[0] <<= fixed.reinterpret(DT.bfloat16, name="d_bf16")


@kernel(mode="vec", block_dim=1)
def e8m0_from_bf16(x: GM[bf16, (ROWS, COLS)], y: GM[u8, (ROWS, COLS)]):
    ub_src = Tensor(DT.bfloat16, [1, COLS], Position.UB, name="q_ub_src")
    ub_dst = Tensor(DT.uint8, [1, 2 * COLS], Position.UB, name="q_ub_dst")
    rows_per_vec = CeilDiv(ROWS, GetVecNum())
    row_begin = Var(rows_per_vec * GetVecIdx())
    row_end = Min(row_begin + rows_per_vec, ROWS)
    with auto_sync():
        for row in range(row_begin, row_end):
            ub_src[:, :] <<= x[row : row + 1, :]
            e8m0_from_bf16_vf(ub_src, ub_dst)
            y[row : row + 1, :] <<= ub_dst[0:1, 0:COLS]
    return y


@kernel(mode="vec", block_dim=1)
def bf16_from_e8m0(x: GM[u8, (ROWS, COLS)], y: GM[bf16, (ROWS, COLS)]):
    ub_src = Tensor(DT.uint8, [1, COLS], Position.UB, name="d_ub_src")
    ub_dst = Tensor(DT.bfloat16, [1, COLS], Position.UB, name="d_ub_dst")
    rows_per_vec = CeilDiv(ROWS, GetVecNum())
    row_begin = Var(rows_per_vec * GetVecIdx())
    row_end = Min(row_begin + rows_per_vec, ROWS)
    with auto_sync():
        for row in range(row_begin, row_end):
            ub_src[:, :] <<= x[row : row + 1, :]
            bf16_from_e8m0_vf(ub_src, ub_dst)
            y[row : row + 1, :] <<= ub_dst
    return y
