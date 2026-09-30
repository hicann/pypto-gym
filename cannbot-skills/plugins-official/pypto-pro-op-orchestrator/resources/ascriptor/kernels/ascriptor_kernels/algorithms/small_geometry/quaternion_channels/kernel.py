# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Quaternion to 3x3 rotation over physical [4, N] channels, with the source MIX launch and its padded UB rows."""

import importlib

from ascriptor.a2 import *
from functools import lru_cache

TILE = 64

def quat_to_rotation_kernel(r: GM[f32, (4, 'N')], rot_out: GM[f32, (9, 'N')], N: i32):
    # One UB buffer per quaternion component (each [1, TILE] = one C0-aligned row).
    ub_w = Tensor(DT.float, [1, TILE], Position.UB)
    ub_x = Tensor(DT.float, [1, TILE], Position.UB)
    ub_y = Tensor(DT.float, [1, TILE], Position.UB)
    ub_z = Tensor(DT.float, [1, TILE], Position.UB)

    # Scratch tensors reused across the 9 output computations.
    ub_a = Tensor(DT.float, [1, TILE], Position.UB)
    ub_b = Tensor(DT.float, [1, TILE], Position.UB)
    ub_norm = Tensor(DT.float, [1, TILE], Position.UB)

    # 9 output channels (R00..R22 in row-major 3x3 flattening order).
    ub_R00 = Tensor(DT.float, [1, TILE], Position.UB)
    ub_R01 = Tensor(DT.float, [1, TILE], Position.UB)
    ub_R02 = Tensor(DT.float, [1, TILE], Position.UB)
    ub_R10 = Tensor(DT.float, [1, TILE], Position.UB)
    ub_R11 = Tensor(DT.float, [1, TILE], Position.UB)
    ub_R12 = Tensor(DT.float, [1, TILE], Position.UB)
    ub_R20 = Tensor(DT.float, [1, TILE], Position.UB)
    ub_R21 = Tensor(DT.float, [1, TILE], Position.UB)
    ub_R22 = Tensor(DT.float, [1, TILE], Position.UB)

    # Split tiles across vec sub-blocks (20 cube cores * 2 sub-blocks = 40 vec lanes on a2).
    n_tiles = CeilDiv(N, TILE)
    tiles_per_core = CeilDiv(n_tiles, GetVecNum())
    t_begin = Var(tiles_per_core * GetVecIdx())
    t_end = Min(t_begin + tiles_per_core, n_tiles)

    with auto_sync():
        for ti in range(t_begin, t_end):
            col0 = Var(ti * TILE)
            valid = Min(TILE, N - col0)

            # Contiguous loads: one row of r per component.
            ub_w <<= r[0:1, col0:col0 + valid]
            ub_x <<= r[1:2, col0:col0 + valid]
            ub_y <<= r[2:3, col0:col0 + valid]
            ub_z <<= r[3:4, col0:col0 + valid]

            # norm = sqrt(w² + x² + y² + z²)
            mul(ub_a, ub_w, ub_w)
            mul(ub_b, ub_x, ub_x)
            add(ub_a, ub_a, ub_b)
            mul(ub_b, ub_y, ub_y)
            add(ub_a, ub_a, ub_b)
            mul(ub_b, ub_z, ub_z)
            add(ub_a, ub_a, ub_b)
            sqrt(ub_norm, ub_a)

            # Normalize quaternion in place.
            div(ub_w, ub_w, ub_norm)
            div(ub_x, ub_x, ub_norm)
            div(ub_y, ub_y, ub_norm)
            div(ub_z, ub_z, ub_norm)

            # R00 = 1 - 2*(y² + z²)
            mul(ub_a, ub_y, ub_y)
            mul(ub_b, ub_z, ub_z)
            add(ub_a, ub_a, ub_b)
            muls(ub_a, ub_a, -2.0)
            adds(ub_R00, ub_a, 1.0)

            # R01 equals 2*(x*y - w*z).
            mul(ub_a, ub_x, ub_y)
            mul(ub_b, ub_w, ub_z)
            sub(ub_a, ub_a, ub_b)
            muls(ub_R01, ub_a, 2.0)

            # R02 equals 2*(x*z + w*y).
            mul(ub_a, ub_x, ub_z)
            mul(ub_b, ub_w, ub_y)
            add(ub_a, ub_a, ub_b)
            muls(ub_R02, ub_a, 2.0)

            # R10 equals 2*(x*y + w*z).
            mul(ub_a, ub_x, ub_y)
            mul(ub_b, ub_w, ub_z)
            add(ub_a, ub_a, ub_b)
            muls(ub_R10, ub_a, 2.0)

            # R11 = 1 - 2*(x² + z²)
            mul(ub_a, ub_x, ub_x)
            mul(ub_b, ub_z, ub_z)
            add(ub_a, ub_a, ub_b)
            muls(ub_a, ub_a, -2.0)
            adds(ub_R11, ub_a, 1.0)

            # R12 equals 2*(y*z - w*x).
            mul(ub_a, ub_y, ub_z)
            mul(ub_b, ub_w, ub_x)
            sub(ub_a, ub_a, ub_b)
            muls(ub_R12, ub_a, 2.0)

            # R20 equals 2*(x*z - w*y).
            mul(ub_a, ub_x, ub_z)
            mul(ub_b, ub_w, ub_y)
            sub(ub_a, ub_a, ub_b)
            muls(ub_R20, ub_a, 2.0)

            # R21 equals 2*(y*z + w*x).
            mul(ub_a, ub_y, ub_z)
            mul(ub_b, ub_w, ub_x)
            add(ub_a, ub_a, ub_b)
            muls(ub_R21, ub_a, 2.0)

            # R22 = 1 - 2*(x² + y²)
            mul(ub_a, ub_x, ub_x)
            mul(ub_b, ub_y, ub_y)
            add(ub_a, ub_a, ub_b)
            muls(ub_a, ub_a, -2.0)
            adds(ub_R22, ub_a, 1.0)

            # Contiguous stores: one row of the rotation output per channel.
            rot_out[0:1, col0:col0 + valid] <<= ub_R00
            rot_out[1:2, col0:col0 + valid] <<= ub_R01
            rot_out[2:3, col0:col0 + valid] <<= ub_R02
            rot_out[3:4, col0:col0 + valid] <<= ub_R10
            rot_out[4:5, col0:col0 + valid] <<= ub_R11
            rot_out[5:6, col0:col0 + valid] <<= ub_R12
            rot_out[6:7, col0:col0 + valid] <<= ub_R20
            rot_out[7:8, col0:col0 + valid] <<= ub_R21
            rot_out[8:9, col0:col0 + valid] <<= ub_R22

    return rot_out

@lru_cache(maxsize=4)
def kernel_for(device, mode="mix"):
    if device not in ('a2', 'a3') or mode not in ("mix", "vec"):
        raise ValueError("Unsupported geometry facade or launch mode")
    return importlib.import_module("ascriptor."+device).kernel(mode=mode)(quat_to_rotation_kernel)
