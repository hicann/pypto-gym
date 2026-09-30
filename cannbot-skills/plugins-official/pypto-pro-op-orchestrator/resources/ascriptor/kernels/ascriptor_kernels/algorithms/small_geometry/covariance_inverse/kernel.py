# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Closed-form inverse of a symmetric 2x2 covariance, one matrix per row, over padded UB rows."""

import importlib

from ascriptor.a5 import *
from functools import lru_cache

CHUNK = 32

IN_COLS = 1

PAD = 8

@vf()
def cov2x2_inverse_vf(c00b: Tensor, c01b: Tensor, c11b: Tensor,
                     i0b: Tensor, i1b: Tensor, i2b: Tensor, rows: Var):
    c00 = Reg(DT.float)
    c01 = Reg(DT.float)
    c11 = Reg(DT.float)
    det = Reg(DT.float)
    tmp = Reg(DT.float)
    out = Reg(DT.float)

    for i in range(rows):
        c00 <<= c00b[i:i + 1, 0:1].single()
        c01 <<= c01b[i:i + 1, 0:1].single()
        c11 <<= c11b[i:i + 1, 0:1].single()

        # det equals c00 * c11 - c01 * c01.
        det <<= c00 * c11
        tmp <<= c01 * c01
        det <<= det - tmp

        # inv_0 equals c11 / det.
        out <<= c11 / det
        i0b[i:i + 1, 0:1] <<= out.single_value()

        # inv_1 = -c01 / det  (compute c01/det then negate via *-1.0)
        tmp <<= c01 / det
        out <<= tmp * -1.0
        i1b[i:i + 1, 0:1] <<= out.single_value()

        # inv_2 equals c00 / det.
        out <<= c00 / det
        i2b[i:i + 1, 0:1] <<= out.single_value()

def cov2x2_inverse_kernel(
    cov2_00: GM[f32, ('N', 1)], cov2_01: GM[f32, ('N', 1)], cov2_11: GM[f32, ('N', 1)],
    inv_00: GM[f32, ('N', 1)], inv_01: GM[f32, ('N', 1)], inv_11: GM[f32, ('N', 1)],
    N: i32,
):
    c00_ub = DBuff(DT.float, [CHUNK, PAD], Position.UB)
    c01_ub = DBuff(DT.float, [CHUNK, PAD], Position.UB)
    c11_ub = DBuff(DT.float, [CHUNK, PAD], Position.UB)
    i0_ub = DBuff(DT.float, [CHUNK, PAD], Position.UB)
    i1_ub = DBuff(DT.float, [CHUNK, PAD], Position.UB)
    i2_ub = DBuff(DT.float, [CHUNK, PAD], Position.UB)

    buf_cnt = Var(0)

    total_chunks = CeilDiv(N, CHUNK)
    chunks_per_core = CeilDiv(total_chunks, GetVecNum())
    chunk_begin = Var(chunks_per_core * GetVecIdx())
    chunk_end = Min(chunk_begin + chunks_per_core, total_chunks)

    with auto_sync():
        for chunk_idx in range(chunk_begin, chunk_end):
            row0 = Var(chunk_idx * CHUNK)
            valid_rows = Min(CHUNK, N - row0)

            # GM [valid_rows, 1] -> UB [valid_rows, 8] (1 real + 7 junk per row).
            c00_ub[buf_cnt] <<= cov2_00[row0:row0 + valid_rows, 0:IN_COLS]
            c01_ub[buf_cnt] <<= cov2_01[row0:row0 + valid_rows, 0:IN_COLS]
            c11_ub[buf_cnt] <<= cov2_11[row0:row0 + valid_rows, 0:IN_COLS]

            cov2x2_inverse_vf(
                c00_ub[buf_cnt], c01_ub[buf_cnt], c11_ub[buf_cnt],
                i0_ub[buf_cnt], i1_ub[buf_cnt], i2_ub[buf_cnt],
                valid_rows,
            )

            # UB [valid_rows, 8] -> GM [valid_rows, 1] (drop the 7 junk cols).
            inv_00[row0:row0 + valid_rows, 0:IN_COLS] <<= i0_ub[buf_cnt][0:valid_rows, 0:IN_COLS]
            inv_01[row0:row0 + valid_rows, 0:IN_COLS] <<= i1_ub[buf_cnt][0:valid_rows, 0:IN_COLS]
            inv_11[row0:row0 + valid_rows, 0:IN_COLS] <<= i2_ub[buf_cnt][0:valid_rows, 0:IN_COLS]

            buf_cnt += 1

    return inv_00, inv_01, inv_11

@lru_cache(maxsize=4)
def kernel_for(device, mode="mix"):
    if device not in ('a5',) or mode not in ("mix", "vec"):
        raise ValueError("Unsupported geometry facade or launch mode")
    return importlib.import_module("ascriptor."+device).kernel(mode=mode)(cov2x2_inverse_kernel)
