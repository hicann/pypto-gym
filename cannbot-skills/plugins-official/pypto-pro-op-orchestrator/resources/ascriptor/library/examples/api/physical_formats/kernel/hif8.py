# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Physical cast body preserved from the individually reviewed utility/hif8_carrier_reinterpret_cast.py."""

from ascriptor.a5 import *


FLOAT_TILE = 64
HALF_TILE = 128
@vf()
def cast_hif8_to_float_vf(src_hif8: Tensor, dst_float: Tensor):
    src_reg = Reg(DT.hif8)
    dst_reg = Reg(DT.float)
    dst_mask = MaskReg(DT.float)
    cfg = CastConfig(round_mode=RoundMode.NONE, name="cfg_hif8_to_float")

    ub_to_reg_unpack4(src_reg, src_hif8[0])
    cast(dst_reg, src_reg, cfg, dst_mask)
    dst_float[0] <<= dst_reg


@vf()
def cast_float_to_hif8_vf(src_float: Tensor, dst_hif8: Tensor):
    src_reg = Reg(DT.float)
    dst_reg = Reg(DT.hif8)
    dst_mask = MaskReg(DT.hif8)
    cfg = CastConfig(round_mode=RoundMode.AWAY_FROM_ZERO, name="cfg_float_to_hif8")

    src_reg <<= src_float[0]
    cast(dst_reg, src_reg, cfg, dst_mask)
    dst_hif8[0] <<= dst_reg.pack4()


@vf()
def cast_float_to_hif8_hybrid_vf(src_float: Tensor, dst_hif8: Tensor):
    src_reg = Reg(DT.float)
    dst_reg = Reg(DT.hif8)
    dst_mask = MaskReg(DT.hif8)
    cfg = CastConfig(round_mode=RoundMode.HYBRID, name="cfg_float_to_hif8_hybrid")

    src_reg <<= src_float[0]
    cast(dst_reg, src_reg, cfg, dst_mask)
    dst_hif8[0] <<= dst_reg.pack4()


@vf()
def cast_hif8_to_half_vf(src_hif8: Tensor, dst_half: Tensor):
    src_reg = Reg(DT.hif8)
    dst_reg = Reg(DT.half)
    dst_mask = MaskReg(DT.half)
    cfg = CastConfig(round_mode=RoundMode.NONE, name="cfg_hif8_to_half")

    ub_to_reg_unpack(src_reg, src_hif8[0])
    cast(dst_reg, src_reg, cfg, dst_mask)
    dst_half[0] <<= dst_reg


@vf()
def cast_half_to_hif8_vf(src_half: Tensor, dst_hif8: Tensor):
    src_reg = Reg(DT.half)
    dst_reg = Reg(DT.hif8)
    dst_mask = MaskReg(DT.hif8)
    cfg = CastConfig(round_mode=RoundMode.AWAY_FROM_ZERO, name="cfg_half_to_hif8")

    src_reg <<= src_half[0]
    cast(dst_reg, src_reg, cfg, dst_mask)
    dst_hif8[0] <<= dst_reg.downsample()


@vf()
def cast_half_to_hif8_hybrid_vf(src_half: Tensor, dst_hif8: Tensor):
    src_reg = Reg(DT.half)
    dst_reg = Reg(DT.hif8)
    dst_mask = MaskReg(DT.hif8)
    cfg = CastConfig(round_mode=RoundMode.HYBRID, name="cfg_half_to_hif8_hybrid")

    src_reg <<= src_half[0]
    cast(dst_reg, src_reg, cfg, dst_mask)
    dst_hif8[0] <<= dst_reg.downsample()


@kernel(mode="vec", block_dim=1)
def hif8_carrier_to_float_kernel(x_carrier: GM[u8, ('total',)], y_float: GM[f32, ('total',)], total: i32):
    ub_carrier = Tensor(DT.uint8, [1, FLOAT_TILE], Position.UB)
    ub_hif8 = ub_carrier.reinterpret(DT.hif8, name="ub_hif8")
    ub_float = Tensor(DT.float, [1, FLOAT_TILE], Position.UB)

    n_tiles = CeilDiv(total, FLOAT_TILE)
    tiles_per_vec = CeilDiv(n_tiles, GetVecNum())
    tile_begin = Var(tiles_per_vec * GetVecIdx())
    tile_end = Min(tile_begin + tiles_per_vec, n_tiles)

    with auto_sync():
        for tile in range(tile_begin, tile_end):
            offset = Var(tile * FLOAT_TILE)
            ub_carrier[:, :] <<= x_carrier[offset:offset + FLOAT_TILE]
            cast_hif8_to_float_vf(ub_hif8, ub_float)
            y_float[offset:offset + FLOAT_TILE] <<= ub_float

    return y_float


@kernel(mode="vec", block_dim=1)
def float_to_hif8_carrier_kernel(x_float: GM[f32, ('total',)], y_carrier: GM[u8, ('total',)], total: i32):
    ub_float = Tensor(DT.float, [1, FLOAT_TILE], Position.UB)
    ub_carrier = Tensor(DT.uint8, [1, FLOAT_TILE], Position.UB)
    ub_hif8 = ub_carrier.reinterpret(DT.hif8, name="ub_hif8_out")

    n_tiles = CeilDiv(total, FLOAT_TILE)
    tiles_per_vec = CeilDiv(n_tiles, GetVecNum())
    tile_begin = Var(tiles_per_vec * GetVecIdx())
    tile_end = Min(tile_begin + tiles_per_vec, n_tiles)

    with auto_sync():
        for tile in range(tile_begin, tile_end):
            offset = Var(tile * FLOAT_TILE)
            ub_float[:, :] <<= x_float[offset:offset + FLOAT_TILE]
            cast_float_to_hif8_vf(ub_float, ub_hif8)
            y_carrier[offset:offset + FLOAT_TILE] <<= ub_carrier

    return y_carrier


@kernel(mode="vec", block_dim=1)
def float_to_hif8_hybrid_carrier_kernel(x_float: GM[f32, ('total',)], y_carrier: GM[u8, ('total',)], total: i32):
    ub_float = Tensor(DT.float, [1, FLOAT_TILE], Position.UB)
    ub_carrier = Tensor(DT.uint8, [1, FLOAT_TILE], Position.UB)
    ub_hif8 = ub_carrier.reinterpret(DT.hif8, name="ub_hif8_hybrid_out")

    n_tiles = CeilDiv(total, FLOAT_TILE)
    tiles_per_vec = CeilDiv(n_tiles, GetVecNum())
    tile_begin = Var(tiles_per_vec * GetVecIdx())
    tile_end = Min(tile_begin + tiles_per_vec, n_tiles)

    with auto_sync():
        for tile in range(tile_begin, tile_end):
            offset = Var(tile * FLOAT_TILE)
            ub_float[:, :] <<= x_float[offset:offset + FLOAT_TILE]
            cast_float_to_hif8_hybrid_vf(ub_float, ub_hif8)
            y_carrier[offset:offset + FLOAT_TILE] <<= ub_carrier

    return y_carrier


@kernel(mode="vec", block_dim=1)
def hif8_carrier_to_half_kernel(x_carrier: GM[u8, ('total',)], y_half: GM[f16, ('total',)], total: i32):
    ub_carrier = Tensor(DT.uint8, [1, HALF_TILE], Position.UB)
    ub_hif8 = ub_carrier.reinterpret(DT.hif8, name="ub_hif8_half")
    ub_half = Tensor(DT.half, [1, HALF_TILE], Position.UB)

    n_tiles = CeilDiv(total, HALF_TILE)
    tiles_per_vec = CeilDiv(n_tiles, GetVecNum())
    tile_begin = Var(tiles_per_vec * GetVecIdx())
    tile_end = Min(tile_begin + tiles_per_vec, n_tiles)

    with auto_sync():
        for tile in range(tile_begin, tile_end):
            offset = Var(tile * HALF_TILE)
            ub_carrier[:, :] <<= x_carrier[offset:offset + HALF_TILE]
            cast_hif8_to_half_vf(ub_hif8, ub_half)
            y_half[offset:offset + HALF_TILE] <<= ub_half

    return y_half


@kernel(mode="vec", block_dim=1)
def half_to_hif8_carrier_kernel(x_half: GM[f16, ('total',)], y_carrier: GM[u8, ('total',)], total: i32):
    ub_half = Tensor(DT.half, [1, HALF_TILE], Position.UB)
    ub_carrier = Tensor(DT.uint8, [1, HALF_TILE], Position.UB)
    ub_hif8 = ub_carrier.reinterpret(DT.hif8, name="ub_hif8_half_out")

    n_tiles = CeilDiv(total, HALF_TILE)
    tiles_per_vec = CeilDiv(n_tiles, GetVecNum())
    tile_begin = Var(tiles_per_vec * GetVecIdx())
    tile_end = Min(tile_begin + tiles_per_vec, n_tiles)

    with auto_sync():
        for tile in range(tile_begin, tile_end):
            offset = Var(tile * HALF_TILE)
            ub_half[:, :] <<= x_half[offset:offset + HALF_TILE]
            cast_half_to_hif8_vf(ub_half, ub_hif8)
            y_carrier[offset:offset + HALF_TILE] <<= ub_carrier

    return y_carrier


@kernel(mode="vec", block_dim=1)
def half_to_hif8_hybrid_carrier_kernel(x_half: GM[f16, ('total',)], y_carrier: GM[u8, ('total',)], total: i32):
    ub_half = Tensor(DT.half, [1, HALF_TILE], Position.UB)
    ub_carrier = Tensor(DT.uint8, [1, HALF_TILE], Position.UB)
    ub_hif8 = ub_carrier.reinterpret(DT.hif8, name="ub_hif8_half_hybrid_out")

    n_tiles = CeilDiv(total, HALF_TILE)
    tiles_per_vec = CeilDiv(n_tiles, GetVecNum())
    tile_begin = Var(tiles_per_vec * GetVecIdx())
    tile_end = Min(tile_begin + tiles_per_vec, n_tiles)

    with auto_sync():
        for tile in range(tile_begin, tile_end):
            offset = Var(tile * HALF_TILE)
            ub_half[:, :] <<= x_half[offset:offset + HALF_TILE]
            cast_half_to_hif8_hybrid_vf(ub_half, ub_hif8)
            y_carrier[offset:offset + HALF_TILE] <<= ub_carrier

    return y_carrier
