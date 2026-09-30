# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Reviewed int64_ops_onboard device bodies; host cases and references are local and independent."""

from ascriptor.a5 import *
COLS = 32
ADDS_K = 1000000007
MULS_K = 3
MAXS_K = 500000000
MINS_K = -400000000
AXPY_K = 5

@vf()
def int64_elt_vf(ub_a: Tensor, ub_b: Tensor, ub_min: Tensor, ub_max: Tensor, ub_neg: Tensor, ub_abs: Tensor, ub_adds: Tensor, ub_muls: Tensor, ub_maxs: Tensor, ub_mins: Tensor, ub_copy: Tensor, ub_axpy: Tensor):
    ra = Reg(DT.int64, name='ra')
    rb = Reg(DT.int64, name='rb')
    ra <<= ub_a[0]
    rb <<= ub_b[0]
    r_min = Reg(DT.int64, name='r_min')
    r_max = Reg(DT.int64, name='r_max')
    r_neg = Reg(DT.int64, name='r_neg')
    r_abs = Reg(DT.int64, name='r_abs')
    r_adds = Reg(DT.int64, name='r_adds')
    r_muls = Reg(DT.int64, name='r_muls')
    r_maxs = Reg(DT.int64, name='r_maxs')
    r_mins = Reg(DT.int64, name='r_mins')
    r_copy = Reg(DT.int64, name='r_copy')
    r_axpy = Reg(DT.int64, name='r_axpy')
    vmin(r_min, ra, rb)
    vmax(r_max, ra, rb)
    r_neg <<= ra.neg()
    r_abs <<= ra.abs()
    adds(r_adds, ra, ADDS_K)
    muls(r_muls, ra, MULS_K)
    vmaxs(r_maxs, ra, MAXS_K)
    vmins(r_mins, ra, MINS_K)
    r_copy <<= ra
    r_axpy <<= rb
    axpy(r_axpy, ra, AXPY_K)
    ub_min[0] <<= r_min
    ub_max[0] <<= r_max
    ub_neg[0] <<= r_neg
    ub_abs[0] <<= r_abs
    ub_adds[0] <<= r_adds
    ub_muls[0] <<= r_muls
    ub_maxs[0] <<= r_maxs
    ub_mins[0] <<= r_mins
    ub_copy[0] <<= r_copy
    ub_axpy[0] <<= r_axpy

@kernel(mode='vec', block_dim=1)
def int64_elt_kernel(a: GM[i64, ('rows', 32)], b: GM[i64, ('rows', 32)], o_min: GM[i64, ('rows', 32)], o_max: GM[i64, ('rows', 32)], o_neg: GM[i64, ('rows', 32)], o_abs: GM[i64, ('rows', 32)], o_adds: GM[i64, ('rows', 32)], o_muls: GM[i64, ('rows', 32)], o_maxs: GM[i64, ('rows', 32)], o_mins: GM[i64, ('rows', 32)], o_copy: GM[i64, ('rows', 32)], o_axpy: GM[i64, ('rows', 32)], rows: i32):
    ub_a = Tensor(DT.int64, [1, COLS], Position.UB, name='ub_a')
    ub_b = Tensor(DT.int64, [1, COLS], Position.UB, name='ub_b')
    ubs = [Tensor(DT.int64, [1, COLS], Position.UB, name=f'ub_o{i}') for i in unroll(10)]
    rows_per_vec = CeilDiv(rows, GetVecNum())
    row_begin = Var(rows_per_vec * GetVecIdx())
    row_end = Min(row_begin + rows_per_vec, rows)
    outs = [o_min, o_max, o_neg, o_abs, o_adds, o_muls, o_maxs, o_mins, o_copy, o_axpy]
    with auto_sync():
        for r in range(row_begin, row_end):
            ub_a[:, :] <<= a[r:r + 1, :]
            ub_b[:, :] <<= b[r:r + 1, :]
            int64_elt_vf(ub_a, ub_b, *ubs)
            for (gm, ub) in zip(outs, ubs):
                gm[r:r + 1, :] <<= ub
    return (o_min, o_max, o_neg, o_abs, o_adds, o_muls, o_maxs, o_mins, o_copy, o_axpy)

@vf()
def int64_reduce_vf(ub_a: Tensor, ub_cadd: Tensor, ub_cmax: Tensor, ub_cmin: Tensor):
    ra = Reg(DT.int64, name='ra')
    ra <<= ub_a[0]
    r_cadd = Reg(DT.int64, name='r_cadd')
    r_cmax = Reg(DT.int64, name='r_cmax')
    r_cmin = Reg(DT.int64, name='r_cmin')
    cadd(r_cadd, ra)
    cmax(r_cmax, ra)
    cmin(r_cmin, ra)
    ub_cadd[0] <<= r_cadd
    ub_cmax[0] <<= r_cmax
    ub_cmin[0] <<= r_cmin

@kernel(mode='vec', block_dim=1)
def int64_reduce_kernel(a: GM[i64, ('rows', 32)], o_cadd: GM[i64, ('rows', 32)], o_cmax: GM[i64, ('rows', 32)], o_cmin: GM[i64, ('rows', 32)], rows: i32):
    ub_a = Tensor(DT.int64, [1, COLS], Position.UB, name='ub_a')
    ub_cadd = Tensor(DT.int64, [1, COLS], Position.UB, name='ub_cadd')
    ub_cmax = Tensor(DT.int64, [1, COLS], Position.UB, name='ub_cmax')
    ub_cmin = Tensor(DT.int64, [1, COLS], Position.UB, name='ub_cmin')
    rows_per_vec = CeilDiv(rows, GetVecNum())
    row_begin = Var(rows_per_vec * GetVecIdx())
    row_end = Min(row_begin + rows_per_vec, rows)
    with auto_sync():
        for r in range(row_begin, row_end):
            ub_a[:, :] <<= a[r:r + 1, :]
            int64_reduce_vf(ub_a, ub_cadd, ub_cmax, ub_cmin)
            o_cadd[r:r + 1, :] <<= ub_cadd
            o_cmax[r:r + 1, :] <<= ub_cmax
            o_cmin[r:r + 1, :] <<= ub_cmin
    return (o_cadd, o_cmax, o_cmin)
