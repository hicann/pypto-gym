# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F403, F405
"""An FP16 row normalized in FP32: unpack, widen, reduce, broadcast, narrow, compact.

This is the shape every normalization starts with, and it is the only unit that exercises
the reduction's return path: `cadd` leaves the sum in lane 0, `reg_to_ub_single` parks it
and `ub_to_reg_single` brings it back across all lanes. `sqrt` and `div` stand in for the
`rsqrt` the A5 register vocabulary does not have.
"""

from ascriptor.a5 import *

WIDTH = 64  # one FP32 register: 64 lanes, and the even lanes of an FP16 register
EPS = 1.0 / 1024.0  # representable in FP16; an all-zero row is finite only because of it


@vf()
def row_norm_vf(x_ub: Tensor, w_ub: Tensor, scratch: Tensor, o_ub: Tensor, rows: Var):
    m32 = MaskReg(DT.float, name="m32")
    m16 = MaskReg(DT.half, name="m16")
    widen = CastConfig(reg_layout=RegLayout.ZERO, name="widen")
    narrow = CastConfig(round_mode=RoundMode.TO_EVEN, reg_layout=RegLayout.ZERO, name="narrow")

    w16 = Reg(DT.half, name="w16")
    ub_to_reg_unpack(w16, w_ub[0])  # 64 dense FP16 weights -> the even lanes
    w32 = Reg(DT.float, name="w32")
    cast(w32, w16, widen, m32)  # even lanes -> 64 FP32 lanes

    ones = Reg(DT.float, name="ones")
    dup(ones, 1.0)

    for row in range(rows):
        x16 = Reg(DT.half, name="x16")
        ub_to_reg_unpack(x16, x_ub[row * WIDTH])
        x32 = Reg(DT.float, name="x32")
        cast(x32, x16, widen, m32)

        square = Reg(DT.float, name="square")
        mul(square, x32, x32)
        total = Reg(DT.float, name="total")
        cadd(total, square)  # the whole-register sum lands in lane 0

        reg_to_ub_single(scratch[0], total)  # lane 0 -> one UB cell
        spread = Reg(DT.float, name="spread")
        ub_to_reg_single(spread, scratch[0])  # that cell -> every lane

        mean = Reg(DT.float, name="mean")
        muls(mean, spread, 1.0 / WIDTH)
        biased = Reg(DT.float, name="biased")
        adds(biased, mean, EPS)
        root = Reg(DT.float, name="root")
        sqrt(root, biased)  # no rsqrt in this vocabulary: sqrt then div
        scale = Reg(DT.float, name="scale")
        div(scale, ones, root)

        normalized = Reg(DT.float, name="normalized")
        mul(normalized, x32, scale)
        weighted = Reg(DT.float, name="weighted")
        mul(weighted, normalized, w32)

        out16 = Reg(DT.half, name="out16")
        cast(out16, weighted, narrow, m16)  # FP32 -> the even lanes of an FP16 register
        reg_to_ub_downsample(o_ub[row * WIDTH], out16)  # those lanes -> 64 dense elements


@kernel(mode="vec", block_dim=1)
def row_norm(x: GM[f16, (4, WIDTH)], w: GM[f16, (1, WIDTH)], o: GM[f16, (4, WIDTH)], rows: i32):
    ub_x = Tensor(DT.half, [4, WIDTH], Position.UB, name="ub_x")
    ub_w = Tensor(DT.half, [1, WIDTH], Position.UB, name="ub_w")
    ub_o = Tensor(DT.half, [4, WIDTH], Position.UB, name="ub_o")
    scratch = Tensor(DT.float, [1, WIDTH], Position.UB, name="scratch")
    with auto_sync():
        ub_x <<= x
        ub_w <<= w
        row_norm_vf(ub_x, ub_w, scratch, ub_o, rows)
        o <<= ub_o
    return o
