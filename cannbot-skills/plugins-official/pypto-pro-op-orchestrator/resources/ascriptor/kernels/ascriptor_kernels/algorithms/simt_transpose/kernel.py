# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Cube matrix product published to UB, then transposed into GM by a SIMT launch: two thread-index decompositions of the same transpose."""

from ascriptor.a5 import *

# ----------------------------------------------------------------------------------------------------
# contiguous_read.py
# Preserved contiguous-read SIMT transpose with one explicit GM publisher.
# ----------------------------------------------------------------------------------------------------

@simt()
def transpose_simt_contig_read(in_ub: Tensor, out_gm: GMTensor, M: Var, N: Var):
    """Decompose the flat thread index into the *input* layout.

    row = i // N, col = i % N, then out_gm[col * M + row] = in_ub[i].
    UB reads are contiguous; GM writes are strided by M.
    """
    total = M * N
    for i in range(simt_thread_id(), total, simt_thread_num()):
        row = i // N
        col = i % N
        out_gm[col * M + row] = in_ub[row * N + col]


@kernel()
def simt_matmul_transpose_contig_read_kernel(
    a: GM[f32, ('m', 'k')], b: GM[f32, ('n', 'k')], c_t: GM[f32, ('n', 'm')], m: i32, n: i32, k: i32
):
    l1a = Tensor(DT.float, [m, k], Position.L1)
    l1b = Tensor(DT.float, [n, k], Position.L1)
    l0c = Tensor(DT.float, [m, n], Position.L0C)
    out_ub = Tensor(DT.float, [m, n], Position.UB)

    cv = CvMutex(flag_id=0, depth=2, dst_end_pipe=Pipe.V)

    # One cube group owns this whole-matrix operation; other groups are idle.
    if GetCubeIdx() == 0:
        with auto_sync():
            l1a <<= a[:, :]
            l1b <<= b[:, :]
            matmul(l0c, l1a, l1b)
            cv.lock()
            l0c_to_ub(out_ub, l0c, dual_mode=DualMode.SINGLE, sub_block_id=0)
            l0c_to_ub(out_ub, l0c, dual_mode=DualMode.SINGLE, sub_block_id=1)
            cv.ready()
            cv.wait()
            if GetSubBlockIdx() == 0:
                transpose_simt_contig_read(out_ub, c_t, m, n)
            cv.free()
    return c_t

# ----------------------------------------------------------------------------------------------------
# contiguous_write.py
# Preserved contiguous-write SIMT transpose with one explicit GM publisher.
# ----------------------------------------------------------------------------------------------------

@simt()
def transpose_simt_contig_write(in_ub: Tensor, out_gm: GMTensor, M: Var, N: Var):
    """Decompose the flat thread index into the *output* layout.

    col = i // M, row = i % M, then out_gm[i] = in_ub[row * N + col].
    GM writes are contiguous; UB reads are strided by N.
    """
    total = M * N
    for i in range(simt_thread_id(), total, simt_thread_num()):
        col = i // M
        row = i % M
        out_gm[i] = in_ub[row * N + col]


@kernel()
def simt_matmul_transpose_contig_write_kernel(
    a: GM[f32, ('m', 'k')], b: GM[f32, ('n', 'k')], c_t: GM[f32, ('n', 'm')], m: i32, n: i32, k: i32
):
    l1a = Tensor(DT.float, [m, k], Position.L1)
    l1b = Tensor(DT.float, [n, k], Position.L1)
    l0c = Tensor(DT.float, [m, n], Position.L0C)
    out_ub = Tensor(DT.float, [m, n], Position.UB)

    # CvMutex coordinates the cube → vec handover for the UB tile.  Two
    # l0c_to_ub calls (sub_block_id=0 then 1) populate both vec0 and vec1
    # UB views with the full DualMode.SINGLE tile so either sub-block can
    # source the SIMT transpose.  dst_end_pipe=Pipe.V so the V-pipe SIMT
    # task must complete before vec_ready is signalled back to cube.
    cv = CvMutex(flag_id=0, depth=2, dst_end_pipe=Pipe.V)

    # One cube group owns this whole-matrix operation; other groups are idle.
    if GetCubeIdx() == 0:
        with auto_sync():
            l1a <<= a[:, :]
            l1b <<= b[:, :]
            matmul(l0c, l1a, l1b)
            cv.lock()
            l0c_to_ub(out_ub, l0c, dual_mode=DualMode.SINGLE, sub_block_id=0)
            l0c_to_ub(out_ub, l0c, dual_mode=DualMode.SINGLE, sub_block_id=1)
            cv.ready()
            cv.wait()
            if GetSubBlockIdx() == 0:
                transpose_simt_contig_write(out_ub, c_t, m, n)
            cv.free()
    return c_t
