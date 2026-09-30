# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Physical cast body preserved from the individually reviewed vec_only/uint2_to_bf16.py."""

from ascriptor.a5 import *


COLS = 128
PACKED_COLS = COLS // 4
UNPACK4_LOAD_BYTES = 64


@vf()
def cast_uint2_to_bf16_vf(src: Tensor, dense_scratch: Tensor, dst: Tensor):
    carrier_u8 = Reg(DT.uint8, name="carrier_u8")
    carrier_u32 = carrier_u8.reinterpret(DT.uint32, name="carrier_u32")

    shifted12 = Reg(DT.uint32, name="shifted12")
    spread_nibbles = Reg(DT.uint32, name="spread_nibbles")
    nibble_pairs = Reg(DT.uint32, name="nibble_pairs")
    shifted6 = Reg(DT.uint32, name="shifted6")
    spread_bytes = Reg(DT.uint32, name="spread_bytes")
    dense_u32 = Reg(DT.uint32, name="dense_u32")
    dense_u8 = dense_u32.reinterpret(DT.uint8, name="dense_u8")
    cast_src_u8 = Reg(DT.uint8, name="cast_src_u8")

    nibble_mask = Reg(DT.uint32, name="nibble_mask")
    byte_mask = Reg(DT.uint32, name="byte_mask")
    dense_half = Reg(DT.half, name="dense_half")
    dense_bf16 = Reg(DT.bfloat16, name="dense_bf16")

    mask_u32_low32 = MaskReg(DT.uint32, init_mode=MaskType.LOWEST32, name="mask_u32_low32")
    mask_u8_low128 = MaskReg(DT.uint8, init_mode=MaskType.LOWEST128, name="mask_u8_low128")
    mask_b16_all = MaskReg(DT.bfloat16, name="mask_b16_all")

    u8_to_half_cfg = CastConfig(round_mode=RoundMode.NONE, reg_layout=RegLayout.ZERO, name="u8_to_half")
    half_to_bf16_cfg = CastConfig(round_mode=RoundMode.TO_EVEN, reg_layout=RegLayout.ZERO, name="half_to_bf16_rint")

    # DIST_UNPACK4_B8 places each carrier byte in the low byte of one uint32
    # lane. Only the first 32 lanes belong to the fixed [64, 32] input row.
    ub_to_reg_unpack4(carrier_u8, src[0])

    # For p = a | b<<2 | c<<4 | d<<6, spread the four 2-bit fields into the
    # four bytes of one uint32 lane: a | b<<8 | c<<16 | d<<24.
    nibble_mask <<= 0x000F000F
    byte_mask <<= 0x03030303
    shiftls(shifted12, carrier_u32, 12, mask=mask_u32_low32)
    vor(spread_nibbles, carrier_u32, shifted12, mask=mask_u32_low32)
    vand(nibble_pairs, spread_nibbles, nibble_mask, mask=mask_u32_low32)
    shiftls(shifted6, nibble_pairs, 6, mask=mask_u32_low32)
    vor(spread_bytes, nibble_pairs, shifted6, mask=mask_u32_low32)
    vand(dense_u32, spread_bytes, byte_mask, mask=mask_u32_low32)

    # uint8 -> half with P0 consumes the even byte in every uint16 lane. Store
    # the 128 dense bytes and reload them with UNPACK_B8 to place each one in
    # that even-byte slot before the cast.
    reg_to_ub_normal(dense_scratch[0], dense_u8, mask_u8_low128)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)
    ub_to_reg_unpack(cast_src_u8, dense_scratch[0])
    cast(dense_half, cast_src_u8, u8_to_half_cfg, mask_b16_all)
    cast(dense_bf16, dense_half, half_to_bf16_cfg, mask_b16_all)
    dst[0] <<= dense_bf16


@kernel(mode="vec", block_dim=1)
def uint2_to_bf16_kernel(x_carrier: GM[u8, ('rows', 32)], padding: GM[u8, ('rows', 32)], y: GM[bf16, ('rows', 128)], rows: i32):
    # UNPACK4_B8 consumes 64 input bytes per register. The active low 32
    # uint32 lanes only depend on the first 32 bytes populated from GM.
    # Explicit padding initializes the other physical load bytes; output must ignore it.
    ub_carrier = Tensor(DT.uint8, [1, UNPACK4_LOAD_BYTES], Position.UB, name="ub_carrier")
    # DIST_NORM_B8 writes one full 256-byte register even with a prefix mask;
    # keep a full-register scratch allocation and reload only its first 128 B.
    ub_dense_scratch = Tensor(DT.uint8, [1, 2 * COLS], Position.UB, name="ub_dense_scratch")
    ub_dst = Tensor(DT.bfloat16, [1, COLS], Position.UB, name="ub_dst")

    rows_per_vec = CeilDiv(rows, GetVecNum())
    row_begin = Var(rows_per_vec * GetVecIdx())
    row_end = Min(row_begin + rows_per_vec, rows)

    with auto_sync():
        for row in range(row_begin, row_end):
            ub_carrier[:, PACKED_COLS:] <<= padding[row:row + 1, :]
            ub_carrier[:, :PACKED_COLS] <<= x_carrier[row:row + 1, :]
            cast_uint2_to_bf16_vf(ub_carrier, ub_dense_scratch, ub_dst)
            y[row:row + 1, :] <<= ub_dst

    return y
