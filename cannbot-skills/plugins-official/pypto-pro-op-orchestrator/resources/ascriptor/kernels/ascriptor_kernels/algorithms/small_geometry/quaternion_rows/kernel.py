# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Quaternion to 3x3 rotation over physical [N, 4] rows: a scalar vector function over padded UB rows, double-buffered per chunk."""

import importlib

from ascriptor.a5 import *
from functools import lru_cache

CHUNK = 32

IN_COLS = 4

OUT_COLS = 9

IN_PAD = 8

OUT_PAD = 16

@vf()
def quat_to_rotation_vf(qbuf: Tensor, Rbuf: Tensor, rows: Var):
    w = Reg(DT.float)
    x = Reg(DT.float)
    y = Reg(DT.float)
    z = Reg(DT.float)
    norm = Reg(DT.float)
    t = Reg(DT.float)
    s = Reg(DT.float)
    one_r = Reg(DT.float)
    out_r = Reg(DT.float)

    one_r.fill(1.0)

    for i in range(rows):
        # Load one raw quaternion (w, x, y, z).
        w <<= qbuf[i:i + 1, 0:1].single()
        x <<= qbuf[i:i + 1, 1:2].single()
        y <<= qbuf[i:i + 1, 2:3].single()
        z <<= qbuf[i:i + 1, 3:4].single()

        # norm equals sqrt(w^2 + x^2 + y^2 + z^2).
        t <<= w * w
        s <<= x * x
        t <<= t + s
        s <<= y * y
        t <<= t + s
        s <<= z * z
        t <<= t + s
        norm <<= t.sqrt()

        # Normalize quaternion in place (scalar registers).
        w <<= w / norm
        x <<= x / norm
        y <<= y / norm
        z <<= z / norm

        # R[0,0] equals 1 - 2*(y^2 + z^2).
        t <<= y * y
        s <<= z * z
        t <<= t + s
        t <<= t * 2.0
        out_r <<= one_r - t
        Rbuf[i:i + 1, 0:1] <<= out_r.single_value()

        # R[0,1] equals 2*(x*y - w*z).
        t <<= x * y
        s <<= w * z
        t <<= t - s
        out_r <<= t * 2.0
        Rbuf[i:i + 1, 1:2] <<= out_r.single_value()

        # R[0,2] equals 2*(x*z + w*y).
        t <<= x * z
        s <<= w * y
        t <<= t + s
        out_r <<= t * 2.0
        Rbuf[i:i + 1, 2:3] <<= out_r.single_value()

        # R[1,0] equals 2*(x*y + w*z).
        t <<= x * y
        s <<= w * z
        t <<= t + s
        out_r <<= t * 2.0
        Rbuf[i:i + 1, 3:4] <<= out_r.single_value()

        # R[1,1] equals 1 - 2*(x^2 + z^2).
        t <<= x * x
        s <<= z * z
        t <<= t + s
        t <<= t * 2.0
        out_r <<= one_r - t
        Rbuf[i:i + 1, 4:5] <<= out_r.single_value()

        # R[1,2] equals 2*(y*z - w*x).
        t <<= y * z
        s <<= w * x
        t <<= t - s
        out_r <<= t * 2.0
        Rbuf[i:i + 1, 5:6] <<= out_r.single_value()

        # R[2,0] equals 2*(x*z - w*y).
        t <<= x * z
        s <<= w * y
        t <<= t - s
        out_r <<= t * 2.0
        Rbuf[i:i + 1, 6:7] <<= out_r.single_value()

        # R[2,1] equals 2*(y*z + w*x).
        t <<= y * z
        s <<= w * x
        t <<= t + s
        out_r <<= t * 2.0
        Rbuf[i:i + 1, 7:8] <<= out_r.single_value()

        # R[2,2] equals 1 - 2*(x^2 + y^2).
        t <<= x * x
        s <<= y * y
        t <<= t + s
        t <<= t * 2.0
        out_r <<= one_r - t
        Rbuf[i:i + 1, 8:9] <<= out_r.single_value()

def quat_to_rotation_kernel(r: GM[f32, ('N', 4)], rot_out: GM[f32, ('N', 9)], N: i32):
    qbuf = DBuff(DT.float, [CHUNK, IN_PAD], Position.UB)
    Rbuf = DBuff(DT.float, [CHUNK, OUT_PAD], Position.UB)

    buf_cnt = Var(0)

    total_chunks = CeilDiv(N, CHUNK)
    chunks_per_core = CeilDiv(total_chunks, GetVecNum())
    chunk_begin = Var(chunks_per_core * GetVecIdx())
    chunk_end = Min(chunk_begin + chunks_per_core, total_chunks)

    with auto_sync():
        for chunk_idx in range(chunk_begin, chunk_end):
            row0 = Var(chunk_idx * CHUNK)
            valid_rows = Min(CHUNK, N - row0)

            # GM [valid_rows, 4] -> UB [valid_rows, 8] (4 real + 4 junk per row).
            qbuf[buf_cnt] <<= r[row0:row0 + valid_rows, 0:IN_COLS]

            quat_to_rotation_vf(qbuf[buf_cnt], Rbuf[buf_cnt], valid_rows)

            # UB [valid_rows, 16] -> GM [valid_rows, 9] (drop the 7 junk cols per row).
            rot_out[row0:row0 + valid_rows, 0:OUT_COLS] <<= Rbuf[buf_cnt][0:valid_rows, 0:OUT_COLS]

            buf_cnt += 1

    return rot_out

@lru_cache(maxsize=4)
def kernel_for(device, mode="mix"):
    if device not in ('a5',) or mode not in ("mix", "vec"):
        raise ValueError("Unsupported geometry facade or launch mode")
    return importlib.import_module("ascriptor."+device).kernel(mode=mode)(quat_to_rotation_kernel)
