# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Physical cast body preserved from the individually reviewed utility/micro_cast_fp8_pack4_dual.py."""

from ascriptor.a5 import *


BLK = 1
SRC_COLS_F32 = 64
DST_COLS_FP8_PACKED = 64


@vf()
def cast_float_to_fp8_pack4_micro(src: Tensor, out_e5m2: Tensor, out_e4m3: Tensor):
    src_reg = Reg(DT.float)
    dst_e5m2 = Reg(DT.e5m2)
    dst_e4m3 = Reg(DT.e4m3)
    mask_e5m2 = MaskReg(DT.e5m2)
    mask_e4m3 = MaskReg(DT.e4m3)
    cfg_zero = CastConfig(reg_layout=RegLayout.ZERO, name="cfg_zero")

    src_reg <<= src[0]
    cast(dst_e5m2, src_reg, cfg_zero, mask_e5m2)
    cast(dst_e4m3, src_reg, cfg_zero, mask_e4m3)
    # float(4B)->fp8(1B) with RegLayout.ZERO populates one lane per 4 slots.
    # pack4 is required to squeeze sparse register lanes into contiguous UB output.
    out_e5m2[0] <<= dst_e5m2.pack4()
    out_e4m3[0] <<= dst_e4m3.pack4()


@kernel(mode="vec", block_dim=1)
def micro_cast_fp8_pack4_dual_kernel(src: GM[f32, ('rows', 64)], out_e5m2: GM[DT.e5m2, ('rows', 64)], out_e4m3: GM[DT.e4m3, ('rows', 64)], rows: i32):
    ub_src = DBuff(src.dtype, [BLK, SRC_COLS_F32], Position.UB)
    ub_e5m2 = DBuff(out_e5m2.dtype, [BLK, DST_COLS_FP8_PACKED], Position.UB)
    ub_e4m3 = DBuff(out_e4m3.dtype, [BLK, DST_COLS_FP8_PACKED], Position.UB)
    idx = Var(0)

    rows_per_vec = CeilDiv(rows, GetVecNum())
    begin = Var(rows_per_vec * GetVecIdx())
    end = Min(begin + rows_per_vec, rows)
    with auto_sync():
        for row in range(begin, end, BLK):
            ub_src[idx] <<= src[row : row + BLK, :]
            cast_float_to_fp8_pack4_micro(ub_src[idx], ub_e5m2[idx], ub_e4m3[idx])
            out_e5m2[row : row + BLK, :] <<= ub_e5m2[idx]
            out_e4m3[row : row + BLK, :] <<= ub_e4m3[idx]
            idx += 1
    return out_e5m2, out_e4m3
