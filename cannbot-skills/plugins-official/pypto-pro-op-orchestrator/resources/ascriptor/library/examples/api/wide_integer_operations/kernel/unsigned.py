# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Reviewed uint64_native_ops_onboard device bodies; host cases and references are local and independent."""

from ascriptor.a5 import *
COLS = 32
SHIFT = 5
DUP_K = (1 << 63) + (1 << 40) + 7
CMP_S = 1 << 63

@vf()
def u64_core_vf(ub_a: Tensor, ub_b: Tensor, ub_add: Tensor, ub_sub: Tensor, ub_mul: Tensor, ub_and: Tensor, ub_or: Tensor, ub_xor: Tensor, ub_not: Tensor, ub_ls: Tensor, ub_rs: Tensor, ub_dup: Tensor, ub_mad: Tensor):
    ra = Reg(DT.uint64, name='uc_a')
    rb = Reg(DT.uint64, name='uc_b')
    ra <<= ub_a[0]
    rb <<= ub_b[0]
    r_add = Reg(DT.uint64, name='uc_add')
    r_sub = Reg(DT.uint64, name='uc_sub')
    r_mul = Reg(DT.uint64, name='uc_mul')
    r_and = Reg(DT.uint64, name='uc_and')
    r_or = Reg(DT.uint64, name='uc_or')
    r_xor = Reg(DT.uint64, name='uc_xor')
    r_not = Reg(DT.uint64, name='uc_not')
    r_ls = Reg(DT.uint64, name='uc_ls')
    r_rs = Reg(DT.uint64, name='uc_rs')
    r_dup = Reg(DT.uint64, name='uc_dup')
    r_mad = Reg(DT.uint64, name='uc_mad')
    add(r_add, ra, rb)
    sub(r_sub, ra, rb)
    mul(r_mul, ra, rb)
    vand(r_and, ra, rb)
    vor(r_or, ra, rb)
    vxor(r_xor, ra, rb)
    vnot(r_not, ra)
    shiftls(r_ls, ra, SHIFT)
    shiftrs(r_rs, ra, SHIFT)
    dup(r_dup, DUP_K)
    r_mad <<= rb
    muladddst(r_mad, ra, ra)
    ub_add[0] <<= r_add
    ub_sub[0] <<= r_sub
    ub_mul[0] <<= r_mul
    ub_and[0] <<= r_and
    ub_or[0] <<= r_or
    ub_xor[0] <<= r_xor
    ub_not[0] <<= r_not
    ub_ls[0] <<= r_ls
    ub_rs[0] <<= r_rs
    ub_dup[0] <<= r_dup
    ub_mad[0] <<= r_mad

@kernel(mode='vec', block_dim=1)
def u64_core_kernel(a: GM[u64, ('rows', 32)], b: GM[u64, ('rows', 32)], o_add: GM[u64, ('rows', 32)], o_sub: GM[u64, ('rows', 32)], o_mul: GM[u64, ('rows', 32)], o_and: GM[u64, ('rows', 32)], o_or: GM[u64, ('rows', 32)], o_xor: GM[u64, ('rows', 32)], o_not: GM[u64, ('rows', 32)], o_ls: GM[u64, ('rows', 32)], o_rs: GM[u64, ('rows', 32)], o_dup: GM[u64, ('rows', 32)], o_mad: GM[u64, ('rows', 32)], rows: i32):
    ub_a = Tensor(DT.uint64, [1, COLS], Position.UB, name='ub_uca')
    ub_b = Tensor(DT.uint64, [1, COLS], Position.UB, name='ub_ucb')
    ubs = [Tensor(DT.uint64, [1, COLS], Position.UB, name=f'ub_uc{i}') for i in unroll(11)]
    rpv = CeilDiv(rows, GetVecNum())
    rb0 = Var(rpv * GetVecIdx())
    re0 = Min(rb0 + rpv, rows)
    outs = [o_add, o_sub, o_mul, o_and, o_or, o_xor, o_not, o_ls, o_rs, o_dup, o_mad]
    with auto_sync():
        for r in range(rb0, re0):
            ub_a[:, :] <<= a[r:r + 1, :]
            ub_b[:, :] <<= b[r:r + 1, :]
            u64_core_vf(ub_a, ub_b, *ubs)
            for (gm, ub) in zip(outs, ubs):
                gm[r:r + 1, :] <<= ub
    return (o_add, o_sub, o_mul, o_and, o_or, o_xor, o_not, o_ls, o_rs, o_dup, o_mad)

@vf()
def u64_cmpsel_vf(ub_a: Tensor, ub_b: Tensor, ub_rr: Tensor, ub_rs: Tensor):
    ra = Reg(DT.uint64, name='us_a')
    rb = Reg(DT.uint64, name='us_b')
    ra <<= ub_a[0]
    rb <<= ub_b[0]
    r_rr = Reg(DT.uint64, name='us_rr')
    r_rs = Reg(DT.uint64, name='us_rs')
    m_rr = MaskReg(DT.uint64, init_mode=MaskType.NONE, name='us_m_rr')
    m_rs = MaskReg(DT.uint64, init_mode=MaskType.NONE, name='us_m_rs')
    compare(m_rr, ra, rb, CompareMode.GT)
    select(r_rr, ra, rb, mask=m_rr)
    compare(m_rs, ra, CMP_S, CompareMode.GT)
    select(r_rs, ra, rb, mask=m_rs)
    ub_rr[0] <<= r_rr
    ub_rs[0] <<= r_rs

@kernel(mode='vec', block_dim=1)
def u64_cmpsel_kernel(a: GM[u64, ('rows', 32)], b: GM[u64, ('rows', 32)], o_rr: GM[u64, ('rows', 32)], o_rs: GM[u64, ('rows', 32)], rows: i32):
    ub_a = Tensor(DT.uint64, [1, COLS], Position.UB, name='ub_usa')
    ub_b = Tensor(DT.uint64, [1, COLS], Position.UB, name='ub_usb')
    ub_rr = Tensor(DT.uint64, [1, COLS], Position.UB, name='ub_usrr')
    ub_rs = Tensor(DT.uint64, [1, COLS], Position.UB, name='ub_usrs')
    rpv = CeilDiv(rows, GetVecNum())
    rb0 = Var(rpv * GetVecIdx())
    re0 = Min(rb0 + rpv, rows)
    with auto_sync():
        for r in range(rb0, re0):
            ub_a[:, :] <<= a[r:r + 1, :]
            ub_b[:, :] <<= b[r:r + 1, :]
            u64_cmpsel_vf(ub_a, ub_b, ub_rr, ub_rs)
            o_rr[r:r + 1, :] <<= ub_rr
            o_rs[r:r + 1, :] <<= ub_rs
    return (o_rr, o_rs)


@vf()
def u64_absolute_vf(left: Tensor, right: Tensor, output: Tensor):
    a = Reg(DT.uint64)
    b = Reg(DT.uint64)
    result = Reg(DT.uint64)
    a <<= left[0]
    b <<= right[0]
    abssub(result, a, b)
    output[0] <<= result


@kernel(mode='vec', block_dim=1)
def u64_absolute_kernel(a: GM[u64, ('rows', 32)], b: GM[u64, ('rows', 32)], output: GM[u64, ('rows', 32)], rows: i32):
    left = Tensor(DT.uint64, [1, 32], Position.UB)
    right = Tensor(DT.uint64, [1, 32], Position.UB)
    result = Tensor(DT.uint64, [1, 32], Position.UB)
    with auto_sync():
        for row in range(rows):
            left <<= a[row:row + 1, :]
            right <<= b[row:row + 1, :]
            u64_absolute_vf(left, right, result)
            output[row:row + 1, :] <<= result
    return output
