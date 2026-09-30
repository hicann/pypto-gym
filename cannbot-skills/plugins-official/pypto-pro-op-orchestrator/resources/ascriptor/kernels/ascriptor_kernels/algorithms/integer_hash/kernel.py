# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The 32-bit MurmurHash3 fmix32 finalizer, exact modulo 2^32 on INT32 carriers."""

import importlib

from ascriptor.a2 import *
from functools import lru_cache

TILE_U32 = 8192

C1_U32 = 0x85EBCA6B

C2_U32 = 0xC2B2AE35

SHIFT_1 = 16

SHIFT_2 = 13

SHIFT_3 = 16

def _as_int32(value: int) -> int:
    """Reinterpret a uint32 constant as the int32 with the same bit pattern."""
    return value - (1 << 32) if value >= (1 << 31) else value

C1_I32 = _as_int32(C1_U32)

C2_I32 = _as_int32(C2_U32)

@func()
def _xor_shift_right(h_i32: Tensor, t_i32: Tensor, o_i32: Tensor, shift: int):
    """Logical32-bit shift and exact halfword OR/AND/NOT XOR."""
    h_u32 = h_i32.reinterpret(DT.uint32)
    t_u32 = t_i32.reinterpret(DT.uint32)
    shiftrs(t_u32, h_u32, shift)

    h_u16 = h_i32.reinterpret(DT.uint16)
    t_u16 = t_i32.reinterpret(DT.uint16)
    o_u16 = o_i32.reinterpret(DT.uint16)
    vor(o_u16, h_u16, t_u16)
    vand(t_u16, h_u16, t_u16)
    vnot(t_u16, t_u16)
    vand(h_u16, o_u16, t_u16)

@func()
def _mul_const(h_i32: Tensor, c_i32: Tensor, const_i32: int):
    """Materialize the full INT32 constant payload before the modular tensor product."""
    dup(c_i32, const_i32)
    mul(h_i32, h_i32, c_i32)

def murmur3_fmix32_kernel(x: GM[i32, (1, 'n')], y: GM[i32, (1, 'n')], n: i32, tile_len: i32):
    h_ub = DBuff(DT.int, [1, TILE_U32], Position.UB)
    t_ub = Tensor(DT.int, [1, TILE_U32], Position.UB)
    o_ub = Tensor(DT.int, [1, TILE_U32], Position.UB)

    n_tiles = CeilDiv(n, tile_len)
    tiles_per_core = CeilDiv(n_tiles, GetVecNum())
    tile_begin = Var(tiles_per_core * GetVecIdx())
    tile_end = Min(tile_begin + tiles_per_core, n_tiles)

    with auto_sync():
        for tile_idx in range(tile_begin, tile_end):
            n0 = Var(tile_idx * tile_len)
            valid = Min(tile_len, n - n0)
            h_tile = h_ub[tile_idx]

            h = h_tile[0:1, 0:valid]
            t = t_ub[0:1, 0:valid]
            o = o_ub[0:1, 0:valid]

            h <<= x[0:1, n0:n0 + valid]

            _xor_shift_right(h, t, o, SHIFT_1)
            _mul_const(h, o, C1_I32)
            _xor_shift_right(h, t, o, SHIFT_2)
            _mul_const(h, o, C2_I32)
            _xor_shift_right(h, t, o, SHIFT_3)

            y[0:1, n0:n0 + valid] <<= h
    return y

@lru_cache(maxsize=2)
def kernel_for(device):
    if device not in ("a2", "a3"):
        raise ValueError("This tensor-vector finalizer supports A2/A3")
    return importlib.import_module("ascriptor."+device).kernel(mode="vec")(murmur3_fmix32_kernel)
