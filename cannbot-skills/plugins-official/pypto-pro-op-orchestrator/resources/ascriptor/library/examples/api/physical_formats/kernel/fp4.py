# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Physical cast body preserved from the individually reviewed vec_only/bf16_to_fp4_e1m2.py."""

from ascriptor.a5 import *  # noqa: F403

ROWS = 64
COLS = 128
PACKED_COLS = COLS // 2


@vf()
def cast_bf16_to_fp4_e1m2_vf(src: Tensor, dst: Tensor):
    src_reg = Reg(DT.bfloat16, name="src_reg")
    dst_reg = Reg(DT.fp4_e1m2, name="dst_reg")
    dst_carrier_reg = dst_reg.reinterpret(DT.uint8, name="dst_carrier_reg")
    mask = MaskReg(DT.bfloat16, name="bf16_mask")
    cfg = CastConfig(round_mode=RoundMode.TO_EVEN, reg_layout=RegLayout.ZERO, name="bf16_to_fp4_e1m2_rint")

    src_reg <<= src[0]
    cast(dst_reg, src_reg, cfg, mask)
    # BF16->FP4 writes packed carrier bytes into every fourth uint8 register slot selected by
    # RegLayout.ZERO; pack4 compacts those bytes for UB/GM.
    dst[0] <<= dst_carrier_reg.pack4()


@kernel(mode="vec", block_dim=1)
def bf16_to_fp4_e1m2_kernel(x: GM[bf16, ("rows", COLS)], y_carrier: GM[u8, ("rows", PACKED_COLS)], rows: i32):
    ub_src = Tensor(DT.bfloat16, [1, COLS], Position.UB, name="ub_src")
    ub_dst_u8 = Tensor(DT.uint8, [1, PACKED_COLS], Position.UB, name="ub_dst_u8")

    rows_per_vec = CeilDiv(rows, GetVecNum())
    row_begin = Var(rows_per_vec * GetVecIdx())
    row_end = Min(row_begin + rows_per_vec, rows)

    with auto_sync():
        for row in range(row_begin, row_end):
            ub_src[:, :] <<= x[row:row + 1, :]
            cast_bf16_to_fp4_e1m2_vf(ub_src, ub_dst_u8)
            y_carrier[row:row + 1, :] <<= ub_dst_u8

    return y_carrier
