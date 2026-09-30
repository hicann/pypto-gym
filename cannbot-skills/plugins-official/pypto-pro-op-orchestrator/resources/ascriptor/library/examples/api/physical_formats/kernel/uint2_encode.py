# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Physical cast body preserved from the individually reviewed vec_only/bf16_to_uint2.py."""

from ascriptor.a5 import *


COLS = 128
PACKED_COLS = COLS // 4


@vf()
def cast_bf16_to_uint2_vf(src: Tensor, dst: Tensor):
    src_bf16 = Reg(DT.bfloat16, name="src_bf16")
    src_half = Reg(DT.half, name="src_half")
    sparse_u8 = Reg(DT.uint8, name="sparse_u8")
    sparse_u16 = sparse_u8.reinterpret(DT.uint16, name="sparse_u16")
    dense_u8 = Reg(DT.uint8, name="dense_u8")
    dense_u32 = dense_u8.reinterpret(DT.uint32, name="dense_u32")

    shifted6 = Reg(DT.uint32, name="shifted6")
    packed_pairs = Reg(DT.uint32, name="packed_pairs")
    shifted12 = Reg(DT.uint32, name="shifted12")
    packed_quads = Reg(DT.uint32, name="packed_quads")
    packed_quads_u8 = packed_quads.reinterpret(DT.uint8, name="packed_quads_u8")

    mask_b16 = MaskReg(DT.bfloat16, name="mask_b16")
    mask_u8_all = MaskReg(DT.uint8, name="mask_u8_all")
    mask_u32_low32 = MaskReg(DT.uint32, init_mode=MaskType.LOWEST32, name="mask_u32_low32")
    mask_pack4_low32 = MaskReg(DT.uint8, init_mode=MaskType.LOWEST128, name="mask_pack4_low32")

    bf16_to_half_cfg = CastConfig(round_mode=RoundMode.TO_EVEN, reg_layout=RegLayout.ZERO, name="bf16_to_half_rint")
    half_to_u8_cfg = CastConfig(round_mode=RoundMode.TO_EVEN, reg_layout=RegLayout.ZERO, name="half_to_u8_rint")

    src_bf16 <<= src[0]
    cast(src_half, src_bf16, bf16_to_half_cfg, mask_b16)
    cast(sparse_u8, src_half, half_to_u8_cfg, mask_u8_all)

    # half -> uint8 with RegLayout.ZERO places each value in the low byte of a
    # uint16 lane. Native Pack compacts those 128 bytes into the low half.
    pack(dense_u8, sparse_u16, HighLowPart.LOWEST)

    # Each uint32 lane now contains four byte values:
    #   w = a | (b << 8) | (c << 16) | (d << 24), a..d in [0, 3].
    # The two shift/or stages move them to bit positions 0, 2, 4, and 6.
    shiftrs(shifted6, dense_u32, 6, mask=mask_u32_low32)
    vor(packed_pairs, dense_u32, shifted6, mask=mask_u32_low32)
    shiftrs(shifted12, packed_pairs, 12, mask=mask_u32_low32)
    vor(packed_quads, packed_pairs, shifted12, mask=mask_u32_low32)

    # One packed uint2 carrier is now in the low byte of each active uint32
    # lane. PACK4_B32 writes those 32 bytes contiguously to UB.
    dst[0] <<= mask_pack4_low32 * packed_quads_u8.pack4()


@kernel(mode="vec", block_dim=1)
def bf16_to_uint2_kernel(x: GM[bf16, ('rows', 128)], y_carrier: GM[u8, ('rows', 32)], rows: i32):
    ub_src = Tensor(DT.bfloat16, [1, COLS], Position.UB, name="ub_src")
    ub_dst = Tensor(DT.uint8, [1, PACKED_COLS], Position.UB, name="ub_dst")

    rows_per_vec = CeilDiv(rows, GetVecNum())
    row_begin = Var(rows_per_vec * GetVecIdx())
    row_end = Min(row_begin + rows_per_vec, rows)

    with auto_sync():
        for row in range(row_begin, row_end):
            ub_src[:, :] <<= x[row:row + 1, :]
            cast_bf16_to_uint2_vf(ub_src, ub_dst)
            y_carrier[row:row + 1, :] <<= ub_dst

    return y_carrier
