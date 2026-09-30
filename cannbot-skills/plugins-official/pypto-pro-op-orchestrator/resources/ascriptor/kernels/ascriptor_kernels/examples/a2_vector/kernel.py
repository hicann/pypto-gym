# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A2/A3 masked scale: y = (x * mask) * scale, one kernel body bound to either facade.

The A2 tensor-vector vocabulary is common to both devices. Only the decorator chooses the
device profile, so there is no duplicated algorithm source and no process-global switch."""

from ascriptor.a2 import *
from functools import lru_cache
from importlib import import_module

TILE_FLOAT = 8192


def masked_scale_float_kernel(x: GM[f32, (1, "n")], mask: GM[f32, (1, "n")], y: GM[f32, (1, "n")], n: i32, scale: f32, tile_len: i32):
    x_ub = DBuff(DT.float, [1, TILE_FLOAT], Position.UB)
    mask_ub = DBuff(DT.float, [1, TILE_FLOAT], Position.UB)
    work_ub = DBuff(DT.float, [1, TILE_FLOAT], Position.UB)
    n_tiles = CeilDiv(n, tile_len)
    tiles_per_core = CeilDiv(n_tiles, GetVecNum())
    tile_begin = Var(tiles_per_core * GetVecIdx())
    tile_end = Min(tile_begin + tiles_per_core, n_tiles)

    with auto_sync():
        for tile_idx in range(tile_begin, tile_end):
            n0 = Var(tile_idx * tile_len)
            valid = Min(tile_len, n - n0)
            x_tile = x_ub[tile_idx]
            m_tile = mask_ub[tile_idx]
            w_tile = work_ub[tile_idx]
            x_tile[0:1, 0:valid] <<= x[0:1, n0:n0 + valid]
            m_tile[0:1, 0:valid] <<= mask[0:1, n0:n0 + valid]
            mul(w_tile[0:1, 0:valid], x_tile[0:1, 0:valid], m_tile[0:1, 0:valid])
            muls(w_tile[0:1, 0:valid], w_tile[0:1, 0:valid], scale)
            y[0:1, n0:n0 + valid] <<= w_tile[0:1, 0:valid]
    return y


@lru_cache(maxsize=2)
def build_kernel(device: str):
    if device not in ("a2", "a3"):
        raise ValueError("masked_scale supports the a2 and a3 facades")
    facade = import_module(f"ascriptor.{device}")
    return facade.kernel(mode="vec")(masked_scale_float_kernel)
