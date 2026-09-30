# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""All eight complex operations at two widths, including corrected full-register abs staging."""

from ascriptor.a5 import *
C64_COLS = 32
C32_COLS = 64
ADDS_K = 1.5 - 0.5j
MULS_K = 2.0 + 1j
DUP_K = 3.0 - 2j

@vf()
def c64_arith_vf(ub_a: Tensor, ub_b: Tensor, ub_add: Tensor, ub_sub: Tensor, ub_mul: Tensor, ub_div: Tensor):
    ra = Reg(DT.complex64, name='c64_ar_a')
    rb = Reg(DT.complex64, name='c64_ar_b')
    ra <<= ub_a[0]
    rb <<= ub_b[0]
    r_add = Reg(DT.complex64, name='c64_ar_add')
    r_sub = Reg(DT.complex64, name='c64_ar_sub')
    r_mul = Reg(DT.complex64, name='c64_ar_mul')
    r_div = Reg(DT.complex64, name='c64_ar_div')
    add(r_add, ra, rb)
    sub(r_sub, ra, rb)
    mul(r_mul, ra, rb)
    div(r_div, ra, rb)
    ub_add[0] <<= r_add
    ub_sub[0] <<= r_sub
    ub_mul[0] <<= r_mul
    ub_div[0] <<= r_div

@kernel(mode='vec', block_dim=1)
def c64_arith_kernel(a: GM[DT.complex64, ('rows', 32)], b: GM[DT.complex64, ('rows', 32)], o_add: GM[DT.complex64, ('rows', 32)], o_sub: GM[DT.complex64, ('rows', 32)], o_mul: GM[DT.complex64, ('rows', 32)], o_div: GM[DT.complex64, ('rows', 32)], rows: i32):
    ub_a = Tensor(DT.complex64, [1, C64_COLS], Position.UB, name='c64_ub_aa')
    ub_b = Tensor(DT.complex64, [1, C64_COLS], Position.UB, name='c64_ub_ab')
    ubs = [Tensor(DT.complex64, [1, C64_COLS], Position.UB, name=f'c64_ub_ar{i}') for i in unroll(4)]
    rpv = CeilDiv(rows, GetVecNum())
    rb0 = Var(rpv * GetVecIdx())
    re0 = Min(rb0 + rpv, rows)
    outs = [o_add, o_sub, o_mul, o_div]
    with auto_sync():
        for r in range(rb0, re0):
            ub_a[:, :] <<= a[r:r + 1, :]
            ub_b[:, :] <<= b[r:r + 1, :]
            c64_arith_vf(ub_a, ub_b, *ubs)
            for (gm, ub) in zip(outs, ubs):
                gm[r:r + 1, :] <<= ub
    return (o_add, o_sub, o_mul, o_div)

@vf()
def c64_scalar_vf(ub_a: Tensor, ub_adds: Tensor, ub_muls: Tensor):
    ra = Reg(DT.complex64, name='c64_sc_a')
    ra <<= ub_a[0]
    r_adds = Reg(DT.complex64, name='c64_sc_adds')
    r_muls = Reg(DT.complex64, name='c64_sc_muls')
    adds(r_adds, ra, ADDS_K)
    muls(r_muls, ra, MULS_K)
    ub_adds[0] <<= r_adds
    ub_muls[0] <<= r_muls

@kernel(mode='vec', block_dim=1)
def c64_scalar_kernel(a: GM[DT.complex64, ('rows', 32)], o_adds: GM[DT.complex64, ('rows', 32)], o_muls: GM[DT.complex64, ('rows', 32)], rows: i32):
    ub_a = Tensor(DT.complex64, [1, C64_COLS], Position.UB, name='c64_ub_sca')
    ub_adds = Tensor(DT.complex64, [1, C64_COLS], Position.UB, name='c64_ub_adds')
    ub_muls = Tensor(DT.complex64, [1, C64_COLS], Position.UB, name='c64_ub_muls')
    rpv = CeilDiv(rows, GetVecNum())
    rb0 = Var(rpv * GetVecIdx())
    re0 = Min(rb0 + rpv, rows)
    with auto_sync():
        for r in range(rb0, re0):
            ub_a[:, :] <<= a[r:r + 1, :]
            c64_scalar_vf(ub_a, ub_adds, ub_muls)
            o_adds[r:r + 1, :] <<= ub_adds
            o_muls[r:r + 1, :] <<= ub_muls
    return (o_adds, o_muls)

@vf()
def c64_abs_vf(ub_a: Tensor, ub_abs: Tensor):
    ra = Reg(DT.complex64, name='c64_ab_a')
    ra <<= ub_a[0]
    r_abs = Reg(DT.float, name='c64_ab_r')
    abs(r_abs, ra)
    ub_abs[0] <<= r_abs

@kernel(mode='vec', block_dim=1)
def c64_abs_kernel(a: GM[DT.complex64, ('rows', 32)], o_abs: GM[f32, ('rows', 32)], rows: i32):
    ub_a = Tensor(DT.complex64, [1, C64_COLS], Position.UB, name='c64_ub_aba')
    # The physical store spans 256 bytes; only the defined first 32 real lanes are published.
    ub_abs = Tensor(DT.float, [1, 2 * C64_COLS], Position.UB, name='c64_ub_abs')
    rpv = CeilDiv(rows, GetVecNum())
    rb0 = Var(rpv * GetVecIdx())
    re0 = Min(rb0 + rpv, rows)
    with auto_sync():
        for r in range(rb0, re0):
            ub_a[:, :] <<= a[r:r + 1, :]
            c64_abs_vf(ub_a, ub_abs)
            o_abs[r:r + 1, :] <<= ub_abs[0:1, 0:C64_COLS]
    return o_abs

@vf()
def c64_dup_vf(ub_a: Tensor, ub_dup: Tensor):
    ra = Reg(DT.complex64, name='c64_dp_a')
    ra <<= ub_a[0]
    r_dup = Reg(DT.complex64, name='c64_dp_r')
    dup(r_dup, DUP_K)
    ub_dup[0] <<= r_dup

@kernel(mode='vec', block_dim=1)
def c64_dup_kernel(a: GM[DT.complex64, ('rows', 32)], o_dup: GM[DT.complex64, ('rows', 32)], rows: i32):
    ub_a = Tensor(DT.complex64, [1, C64_COLS], Position.UB, name='c64_ub_dpa')
    ub_dup = Tensor(DT.complex64, [1, C64_COLS], Position.UB, name='c64_ub_dup')
    rpv = CeilDiv(rows, GetVecNum())
    rb0 = Var(rpv * GetVecIdx())
    re0 = Min(rb0 + rpv, rows)
    with auto_sync():
        for r in range(rb0, re0):
            ub_a[:, :] <<= a[r:r + 1, :]
            c64_dup_vf(ub_a, ub_dup)
            o_dup[r:r + 1, :] <<= ub_dup
    return o_dup

@vf()
def c32_arith_vf(ub_a: Tensor, ub_b: Tensor, ub_add: Tensor, ub_sub: Tensor, ub_mul: Tensor, ub_div: Tensor):
    ra = Reg(DT.complex32, name='c32_ar_a')
    rb = Reg(DT.complex32, name='c32_ar_b')
    ra <<= ub_a[0]
    rb <<= ub_b[0]
    r_add = Reg(DT.complex32, name='c32_ar_add')
    r_sub = Reg(DT.complex32, name='c32_ar_sub')
    r_mul = Reg(DT.complex32, name='c32_ar_mul')
    r_div = Reg(DT.complex32, name='c32_ar_div')
    add(r_add, ra, rb)
    sub(r_sub, ra, rb)
    mul(r_mul, ra, rb)
    div(r_div, ra, rb)
    ub_add[0] <<= r_add
    ub_sub[0] <<= r_sub
    ub_mul[0] <<= r_mul
    ub_div[0] <<= r_div

@kernel(mode='vec', block_dim=1)
def c32_arith_kernel(a: GM[DT.complex32, ('rows', 64)], b: GM[DT.complex32, ('rows', 64)], o_add: GM[DT.complex32, ('rows', 64)], o_sub: GM[DT.complex32, ('rows', 64)], o_mul: GM[DT.complex32, ('rows', 64)], o_div: GM[DT.complex32, ('rows', 64)], rows: i32):
    ub_a = Tensor(DT.complex32, [1, C32_COLS], Position.UB, name='c32_ub_aa')
    ub_b = Tensor(DT.complex32, [1, C32_COLS], Position.UB, name='c32_ub_ab')
    ubs = [Tensor(DT.complex32, [1, C32_COLS], Position.UB, name=f'c32_ub_ar{i}') for i in unroll(4)]
    rpv = CeilDiv(rows, GetVecNum())
    rb0 = Var(rpv * GetVecIdx())
    re0 = Min(rb0 + rpv, rows)
    outs = [o_add, o_sub, o_mul, o_div]
    with auto_sync():
        for r in range(rb0, re0):
            ub_a[:, :] <<= a[r:r + 1, :]
            ub_b[:, :] <<= b[r:r + 1, :]
            c32_arith_vf(ub_a, ub_b, *ubs)
            for (gm, ub) in zip(outs, ubs):
                gm[r:r + 1, :] <<= ub
    return (o_add, o_sub, o_mul, o_div)

@vf()
def c32_scalar_vf(ub_a: Tensor, ub_adds: Tensor, ub_muls: Tensor):
    ra = Reg(DT.complex32, name='c32_sc_a')
    ra <<= ub_a[0]
    r_adds = Reg(DT.complex32, name='c32_sc_adds')
    r_muls = Reg(DT.complex32, name='c32_sc_muls')
    adds(r_adds, ra, ADDS_K)
    muls(r_muls, ra, MULS_K)
    ub_adds[0] <<= r_adds
    ub_muls[0] <<= r_muls

@kernel(mode='vec', block_dim=1)
def c32_scalar_kernel(a: GM[DT.complex32, ('rows', 64)], o_adds: GM[DT.complex32, ('rows', 64)], o_muls: GM[DT.complex32, ('rows', 64)], rows: i32):
    ub_a = Tensor(DT.complex32, [1, C32_COLS], Position.UB, name='c32_ub_sca')
    ub_adds = Tensor(DT.complex32, [1, C32_COLS], Position.UB, name='c32_ub_adds')
    ub_muls = Tensor(DT.complex32, [1, C32_COLS], Position.UB, name='c32_ub_muls')
    rpv = CeilDiv(rows, GetVecNum())
    rb0 = Var(rpv * GetVecIdx())
    re0 = Min(rb0 + rpv, rows)
    with auto_sync():
        for r in range(rb0, re0):
            ub_a[:, :] <<= a[r:r + 1, :]
            c32_scalar_vf(ub_a, ub_adds, ub_muls)
            o_adds[r:r + 1, :] <<= ub_adds
            o_muls[r:r + 1, :] <<= ub_muls
    return (o_adds, o_muls)

@vf()
def c32_abs_vf(ub_a: Tensor, ub_abs: Tensor):
    ra = Reg(DT.complex32, name='c32_ab_a')
    ra <<= ub_a[0]
    r_abs = Reg(DT.half, name='c32_ab_r')
    abs(r_abs, ra)
    ub_abs[0] <<= r_abs

@kernel(mode='vec', block_dim=1)
def c32_abs_kernel(a: GM[DT.complex32, ('rows', 64)], o_abs: GM[DT.half, ('rows', 64)], rows: i32):
    ub_a = Tensor(DT.complex32, [1, C32_COLS], Position.UB, name='c32_ub_aba')
    # The physical store spans 256 bytes; only the defined first 64 real lanes are published.
    ub_abs = Tensor(DT.half, [1, 2 * C32_COLS], Position.UB, name='c32_ub_abs')
    rpv = CeilDiv(rows, GetVecNum())
    rb0 = Var(rpv * GetVecIdx())
    re0 = Min(rb0 + rpv, rows)
    with auto_sync():
        for r in range(rb0, re0):
            ub_a[:, :] <<= a[r:r + 1, :]
            c32_abs_vf(ub_a, ub_abs)
            o_abs[r:r + 1, :] <<= ub_abs[0:1, 0:C32_COLS]
    return o_abs

@vf()
def c32_dup_vf(ub_a: Tensor, ub_dup: Tensor):
    ra = Reg(DT.complex32, name='c32_dp_a')
    ra <<= ub_a[0]
    r_dup = Reg(DT.complex32, name='c32_dp_r')
    dup(r_dup, DUP_K)
    ub_dup[0] <<= r_dup

@kernel(mode='vec', block_dim=1)
def c32_dup_kernel(a: GM[DT.complex32, ('rows', 64)], o_dup: GM[DT.complex32, ('rows', 64)], rows: i32):
    ub_a = Tensor(DT.complex32, [1, C32_COLS], Position.UB, name='c32_ub_dpa')
    ub_dup = Tensor(DT.complex32, [1, C32_COLS], Position.UB, name='c32_ub_dup')
    rpv = CeilDiv(rows, GetVecNum())
    rb0 = Var(rpv * GetVecIdx())
    re0 = Min(rb0 + rpv, rows)
    with auto_sync():
        for r in range(rb0, re0):
            ub_a[:, :] <<= a[r:r + 1, :]
            c32_dup_vf(ub_a, ub_dup)
            o_dup[r:r + 1, :] <<= ub_dup
    return o_dup
