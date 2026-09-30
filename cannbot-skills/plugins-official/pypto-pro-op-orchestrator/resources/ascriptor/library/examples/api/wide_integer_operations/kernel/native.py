# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Reviewed int64_native_ops_onboard device bodies; host cases and references are local and independent."""

from ascriptor.a5 import *
COLS = 32
SHIFT = 5
DUP_K = 2 ** 40 + 12345
CMP_S = 12345678

@vf()
def int64_safe_vf(ub_a: Tensor, ub_b: Tensor, ub_add: Tensor, ub_sub: Tensor, ub_mul: Tensor, ub_and: Tensor, ub_or: Tensor, ub_xor: Tensor, ub_not: Tensor):
    ra = Reg(DT.int64, name='sf_a')
    rb = Reg(DT.int64, name='sf_b')
    ra <<= ub_a[0]
    rb <<= ub_b[0]
    r_add = Reg(DT.int64, name='sf_add')
    r_sub = Reg(DT.int64, name='sf_sub')
    r_mul = Reg(DT.int64, name='sf_mul')
    r_and = Reg(DT.int64, name='sf_and')
    r_or = Reg(DT.int64, name='sf_or')
    r_xor = Reg(DT.int64, name='sf_xor')
    r_not = Reg(DT.int64, name='sf_not')
    add(r_add, ra, rb)
    sub(r_sub, ra, rb)
    mul(r_mul, ra, rb)
    vand(r_and, ra, rb)
    vor(r_or, ra, rb)
    vxor(r_xor, ra, rb)
    vnot(r_not, ra)
    ub_add[0] <<= r_add
    ub_sub[0] <<= r_sub
    ub_mul[0] <<= r_mul
    ub_and[0] <<= r_and
    ub_or[0] <<= r_or
    ub_xor[0] <<= r_xor
    ub_not[0] <<= r_not

@kernel(mode='vec', block_dim=1)
def int64_safe_kernel(a: GM[i64, ('rows', 32)], b: GM[i64, ('rows', 32)], o_add: GM[i64, ('rows', 32)], o_sub: GM[i64, ('rows', 32)], o_mul: GM[i64, ('rows', 32)], o_and: GM[i64, ('rows', 32)], o_or: GM[i64, ('rows', 32)], o_xor: GM[i64, ('rows', 32)], o_not: GM[i64, ('rows', 32)], rows: i32):
    ub_a = Tensor(DT.int64, [1, COLS], Position.UB, name='ub_sa')
    ub_b = Tensor(DT.int64, [1, COLS], Position.UB, name='ub_sb')
    ubs = [Tensor(DT.int64, [1, COLS], Position.UB, name=f'ub_sf{i}') for i in unroll(7)]
    rpv = CeilDiv(rows, GetVecNum())
    rb0 = Var(rpv * GetVecIdx())
    re0 = Min(rb0 + rpv, rows)
    outs = [o_add, o_sub, o_mul, o_and, o_or, o_xor, o_not]
    with auto_sync():
        for r in range(rb0, re0):
            ub_a[:, :] <<= a[r:r + 1, :]
            ub_b[:, :] <<= b[r:r + 1, :]
            int64_safe_vf(ub_a, ub_b, *ubs)
            for (gm, ub) in zip(outs, ubs):
                gm[r:r + 1, :] <<= ub
    return (o_add, o_sub, o_mul, o_and, o_or, o_xor, o_not)

@vf()
def int64_shift_vf(ub_a: Tensor, ub_ls: Tensor, ub_rs: Tensor):
    ra = Reg(DT.int64, name='sh_a')
    ra <<= ub_a[0]
    r_ls = Reg(DT.int64, name='sh_ls')
    r_rs = Reg(DT.int64, name='sh_rs')
    shiftls(r_ls, ra, SHIFT)
    shiftrs(r_rs, ra, SHIFT)
    ub_ls[0] <<= r_ls
    ub_rs[0] <<= r_rs

@kernel(mode='vec', block_dim=1)
def int64_shift_kernel(a: GM[i64, ('rows', 32)], o_ls: GM[i64, ('rows', 32)], o_rs: GM[i64, ('rows', 32)], rows: i32):
    ub_a = Tensor(DT.int64, [1, COLS], Position.UB, name='ub_sha')
    ub_ls = Tensor(DT.int64, [1, COLS], Position.UB, name='ub_ls')
    ub_rs = Tensor(DT.int64, [1, COLS], Position.UB, name='ub_rs')
    rpv = CeilDiv(rows, GetVecNum())
    rb0 = Var(rpv * GetVecIdx())
    re0 = Min(rb0 + rpv, rows)
    with auto_sync():
        for r in range(rb0, re0):
            ub_a[:, :] <<= a[r:r + 1, :]
            int64_shift_vf(ub_a, ub_ls, ub_rs)
            o_ls[r:r + 1, :] <<= ub_ls
            o_rs[r:r + 1, :] <<= ub_rs
    return (o_ls, o_rs)

@vf()
def int64_dup_vf(ub_a: Tensor, ub_dup: Tensor):
    ra = Reg(DT.int64, name='dp_a')
    ra <<= ub_a[0]
    r_dup = Reg(DT.int64, name='dp_r')
    dup(r_dup, DUP_K)
    ub_dup[0] <<= r_dup

@kernel(mode='vec', block_dim=1)
def int64_dup_kernel(a: GM[i64, ('rows', 32)], o_dup: GM[i64, ('rows', 32)], rows: i32):
    ub_a = Tensor(DT.int64, [1, COLS], Position.UB, name='ub_dupa')
    ub_dup = Tensor(DT.int64, [1, COLS], Position.UB, name='ub_dup')
    rpv = CeilDiv(rows, GetVecNum())
    rb0 = Var(rpv * GetVecIdx())
    re0 = Min(rb0 + rpv, rows)
    with auto_sync():
        for r in range(rb0, re0):
            ub_a[:, :] <<= a[r:r + 1, :]
            int64_dup_vf(ub_a, ub_dup)
            o_dup[r:r + 1, :] <<= ub_dup
    return o_dup

@vf()
def int64_cmpsel_vf(ub_a: Tensor, ub_b: Tensor, ub_rr: Tensor, ub_rs: Tensor):
    ra = Reg(DT.int64, name='cs_a')
    rb = Reg(DT.int64, name='cs_b')
    ra <<= ub_a[0]
    rb <<= ub_b[0]
    r_rr = Reg(DT.int64, name='cs_rr')
    r_rs = Reg(DT.int64, name='cs_rs')
    m_rr = MaskReg(DT.int64, init_mode=MaskType.NONE, name='cs_m_rr')
    m_rs = MaskReg(DT.int64, init_mode=MaskType.NONE, name='cs_m_rs')
    compare(m_rr, ra, rb, CompareMode.GT)
    select(r_rr, ra, rb, mask=m_rr)
    compare(m_rs, ra, CMP_S, CompareMode.GT)
    select(r_rs, ra, rb, mask=m_rs)
    ub_rr[0] <<= r_rr
    ub_rs[0] <<= r_rs

@kernel(mode='vec', block_dim=1)
def int64_cmpsel_kernel(a: GM[i64, ('rows', 32)], b: GM[i64, ('rows', 32)], o_rr: GM[i64, ('rows', 32)], o_rs: GM[i64, ('rows', 32)], rows: i32):
    ub_a = Tensor(DT.int64, [1, COLS], Position.UB, name='ub_ca')
    ub_b = Tensor(DT.int64, [1, COLS], Position.UB, name='ub_cb')
    ub_rr = Tensor(DT.int64, [1, COLS], Position.UB, name='ub_rr')
    ub_rs = Tensor(DT.int64, [1, COLS], Position.UB, name='ub_rs2')
    rpv = CeilDiv(rows, GetVecNum())
    rb0 = Var(rpv * GetVecIdx())
    re0 = Min(rb0 + rpv, rows)
    with auto_sync():
        for r in range(rb0, re0):
            ub_a[:, :] <<= a[r:r + 1, :]
            ub_b[:, :] <<= b[r:r + 1, :]
            int64_cmpsel_vf(ub_a, ub_b, ub_rr, ub_rs)
            o_rr[r:r + 1, :] <<= ub_rr
            o_rs[r:r + 1, :] <<= ub_rs
    return (o_rr, o_rs)

@vf()
def int64_fused_vf(ub_a: Tensor, ub_b: Tensor, ub_abs: Tensor, ub_mad: Tensor):
    ra = Reg(DT.int64, name='fu_a')
    rb = Reg(DT.int64, name='fu_b')
    ra <<= ub_a[0]
    rb <<= ub_b[0]
    r_abs = Reg(DT.int64, name='fu_abs')
    r_mad = Reg(DT.int64, name='fu_mad')
    abssub(r_abs, ra, rb)
    r_mad <<= rb
    muladddst(r_mad, ra, ra)
    ub_abs[0] <<= r_abs
    ub_mad[0] <<= r_mad

@kernel(mode='vec', block_dim=1)
def int64_fused_kernel(a: GM[i64, ('rows', 32)], b: GM[i64, ('rows', 32)], o_abs: GM[i64, ('rows', 32)], o_mad: GM[i64, ('rows', 32)], rows: i32):
    ub_a = Tensor(DT.int64, [1, COLS], Position.UB, name='ub_fa')
    ub_b = Tensor(DT.int64, [1, COLS], Position.UB, name='ub_fb')
    ub_abs = Tensor(DT.int64, [1, COLS], Position.UB, name='ub_fabs')
    ub_mad = Tensor(DT.int64, [1, COLS], Position.UB, name='ub_fmad')
    rpv = CeilDiv(rows, GetVecNum())
    rb0 = Var(rpv * GetVecIdx())
    re0 = Min(rb0 + rpv, rows)
    with auto_sync():
        for r in range(rb0, re0):
            ub_a[:, :] <<= a[r:r + 1, :]
            ub_b[:, :] <<= b[r:r + 1, :]
            int64_fused_vf(ub_a, ub_b, ub_abs, ub_mad)
            o_abs[r:r + 1, :] <<= ub_abs
            o_mad[r:r + 1, :] <<= ub_mad
    return (o_abs, o_mad)
