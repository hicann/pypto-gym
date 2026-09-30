# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Sequential FP32 row prefix sums that reset at every chunk boundary, with both the
wide (row-pitch) and the narrow (64-column padded) storage branches."""

from ascriptor.a5 import *

VEC_WIDTH = 64

@vf()
def chunk_row_cumsum_vf(src: Tensor, dst: Tensor, rows: Var, row_stride: Var, cols64: Var, chunk_size: Var):
    prev_reg = Reg(DT.float)
    curr_reg = Reg(DT.float)

    for c in range(cols64):
        col_off = Var(c * VEC_WIDTH)
        prev_reg <<= src[col_off]
        dst[col_off] <<= prev_reg
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

    for r in range(1, rows):
        row_base = Var(r * row_stride)
        for c in range(cols64):
            curr_off = Var(row_base + c * VEC_WIDTH)
            prev_off = Var(curr_off - row_stride)
            curr_reg <<= src[curr_off]
            prev_reg <<= dst[prev_off]
            curr_reg <<= curr_reg + prev_reg
            dst[curr_off] <<= curr_reg
            vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@kernel(mode="mix", block_dim=1)
def chunk_row_cumsum_kernel(x: GM[f32, ('M', 'H')], y: GM[f32, ('M', 'H')], M: i32, H: i32, chunk_size: i32):
    xbuf = Tensor(DT.float, [chunk_size, H], Position.UB)
    ybuf = Tensor(DT.float, [chunk_size, H], Position.UB)
    xbuf_pad64 = Tensor(DT.float, [chunk_size, VEC_WIDTH], Position.UB)
    ybuf_pad64 = Tensor(DT.float, [chunk_size, VEC_WIDTH], Position.UB)

    cols64 = Var(H // VEC_WIDTH)
    total_chunks = CeilDiv(M, chunk_size)
    chunks_per_core = CeilDiv(total_chunks, GetVecNum())
    chunk_begin = Var(chunks_per_core * GetVecIdx())
    chunk_end = Min(chunk_begin + chunks_per_core, total_chunks)

    if H < VEC_WIDTH:
        # `narrow` is `H` wherever this arm runs -- the guard says so. It is written out because a
        # backend prints *both* arms and specialises each against the launch's actual H, and with
        # H = 128 the unclamped form asks for a 512-byte burst into a 256-byte row and a pad stride
        # of -8 so that the two sum back to the allocation's pitch. That is correct only as a pair:
        # neither half means anything alone, which is why the narrow row pitch stops being readable
        # as a row pitch. Clamping states what the guard already guarantees, and costs nothing.
        narrow = Var(Min(H, VEC_WIDTH))
        with auto_sync():
            for chunk_idx in range(chunk_begin, chunk_end):
                row0 = Var(chunk_idx * chunk_size)
                valid_rows = Min(chunk_size, M - row0)
                pad_stride = Var((VEC_WIDTH - narrow) // DT.float.C0)
                gm_to_ub_pad(
                    xbuf_pad64,
                    x[row0:row0 + valid_rows, 0:narrow],
                    valid_rows,
                    narrow,
                    0,
                    pad_stride,
                )
                chunk_row_cumsum_vf(xbuf_pad64, ybuf_pad64, valid_rows, VEC_WIDTH, 1, chunk_size)
                ub_to_gm_pad(
                    y[row0:row0 + valid_rows, 0:narrow],
                    ybuf_pad64[0:valid_rows, 0:VEC_WIDTH],
                    valid_rows,
                    narrow,
                    pad_stride,
                    0,
                )
    else:
        with auto_sync():
            for chunk_idx in range(chunk_begin, chunk_end):
                row0 = Var(chunk_idx * chunk_size)
                valid_rows = Min(chunk_size, M - row0)
                xbuf <<= x[row0:row0 + valid_rows, 0:H]
                chunk_row_cumsum_vf(xbuf, ybuf, valid_rows, H, cols64, chunk_size)
                y[row0:row0 + valid_rows, 0:H] <<= ybuf[0:valid_rows, 0:H]

    return y
