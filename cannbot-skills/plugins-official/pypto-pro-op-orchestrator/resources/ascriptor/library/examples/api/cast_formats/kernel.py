# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F403, F405, F841
"""Reviewed production closures; host inputs/references are local and independent.

Pure-vector teaching units explicitly launch one vector participant. Original
mode=vec bodies, masks, byte footprints and overlapping-write barriers remain.
"""

# SHA256: 52922a59a362709c419ff92ea75650a79443cd54670dc74b411781ee15bbd5cf

from ascriptor.a5 import *  # noqa: F401,F403

ROWS = 4

W32 = 64  # 64 x 32-bit lanes = 256 bytes = one full vector register

W64 = 32  # 32 x 64-bit lanes = 256 bytes = one full vector register


@vf()
def b64_widen_vf(s32: Tensor, s64: Tensor, o_widen: Tensor, o_narrow: Tensor):
    m64 = MaskReg(DT.int64, name="b_m64")
    m32 = MaskReg(DT.int32, name="b_m32")
    low32 = MaskReg(DT.int32, init_mode=MaskType.LOWEST32, name="b_low32")
    plain = CastConfig(reg_layout=RegLayout.ZERO, name="b64_plain")

    # b64_widen: i32 -> i64. Source lanes 0..31 reach the first destination register.
    a32 = Reg(DT.int32, name="b_a32")
    a32 <<= s32[0]
    w64 = Reg(DT.int64, name="b_w64")
    cast(w64, a32, plain, m64)
    o_widen[0] <<= w64

    # Argument-free i64 -> i32 discards high bits independently of CTRL (D-233).
    a64 = Reg(DT.int64, name="b_a64")
    a64 <<= s64[0]
    n32 = Reg(DT.int32, name="b_n32")
    cast(n32, a64, plain, m32)
    reg_to_ub_normal(o_narrow[0], n32, low32)


@vf()
def b64_float_vf(s64: Tensor, sf32: Tensor, o_tof32: Tensor, o_fromf32: Tensor):
    m64 = MaskReg(DT.int64, name="bf_m64")
    m32 = MaskReg(DT.int32, name="bf_m32")
    low32 = MaskReg(DT.int32, init_mode=MaskType.LOWEST32, name="bf_low32")
    trunc = CastConfig(round_mode=RoundMode.TRUNC, reg_layout=RegLayout.ZERO, name="b64_trunc")

    # f32_from_b64: i64 -> f32.
    a64 = Reg(DT.int64, name="bf_a64")
    a64 <<= s64[0]
    tof = Reg(DT.float, name="bf_tof")
    cast(tof, a64, trunc, m32)
    reg_to_ub_normal(o_tof32[0], tof, low32)

    # b64_from_f32: f32 -> i64.
    g = Reg(DT.float, name="bf_g")
    g <<= sf32[0]
    q64 = Reg(DT.int64, name="bf_q64")
    cast(q64, g, trunc, m64)
    o_fromf32[0] <<= q64


@kernel(mode="vec", block_dim=1)
def cast_b64_widen(x32: GM[i32, (ROWS, W32)], x64: GM[i64, (ROWS, W64)], o_widen: GM[i64, (ROWS, W64)],
    o_narrow: GM[i32, (ROWS, W64)]):
    ub32 = Tensor(DT.int32, [1, W32], Position.UB, name="b_ub32")
    ub64 = Tensor(DT.int64, [1, W64], Position.UB, name="b_ub64")
    o_w = Tensor(DT.int64, [1, W64], Position.UB, name="b_o_w")
    o_n = Tensor(DT.int32, [1, W32], Position.UB, name="b_o_n")
    rows_per_vec = CeilDiv(ROWS, GetVecNum())
    row_begin = Var(rows_per_vec * GetVecIdx())
    row_end = Min(row_begin + rows_per_vec, ROWS)
    with auto_sync():
        for row in range(row_begin, row_end):
            ub32[:, :] <<= x32[row : row + 1, :]
            ub64[:, :] <<= x64[row : row + 1, :]
            b64_widen_vf(ub32, ub64, o_w, o_n)
            o_widen[row : row + 1, :] <<= o_w
            o_narrow[row : row + 1, :] <<= o_n[0:1, 0:W64]
    return o_widen, o_narrow


@kernel(mode="vec", block_dim=1)
def cast_b64_float(x64: GM[i64, (ROWS, W64)], xf: GM[f32, (ROWS, W32)], o_tof32: GM[f32, (ROWS, W64)],
    o_fromf32: GM[i64, (ROWS, W64)]):
    ub64 = Tensor(DT.int64, [1, W64], Position.UB, name="bf_ub64")
    ubf = Tensor(DT.float, [1, W32], Position.UB, name="bf_ubf")
    o_t = Tensor(DT.float, [1, W32], Position.UB, name="bf_o_t")
    o_q = Tensor(DT.int64, [1, W64], Position.UB, name="bf_o_q")
    rows_per_vec = CeilDiv(ROWS, GetVecNum())
    row_begin = Var(rows_per_vec * GetVecIdx())
    row_end = Min(row_begin + rows_per_vec, ROWS)
    with auto_sync():
        for row in range(row_begin, row_end):
            ub64[:, :] <<= x64[row : row + 1, :]
            ubf[:, :] <<= xf[row : row + 1, :]
            b64_float_vf(ub64, ubf, o_t, o_q)
            o_tof32[row : row + 1, :] <<= o_t[0:1, 0:W64]
            o_fromf32[row : row + 1, :] <<= o_q
    return o_tof32, o_fromf32


I4_LANES = 128  # 128 x 16-bit lanes against 64 packed carrier bytes

I4_BYTES = I4_LANES // 2


@vf()
def i4_narrow_vf(sh: Tensor, si: Tensor, o_f16: Tensor, o_i16: Tensor):
    m16 = MaskReg(DT.half, name="i4_m16")
    mi16 = MaskReg(DT.int16, name="i4_mi16")
    rnd = CastConfig(round_mode=RoundMode.TO_EVEN, saturate=True, reg_layout=RegLayout.ZERO, name="i4_from_f16")
    sat = CastConfig(saturate=True, reg_layout=RegLayout.ZERO, name="i4_from_i16")

    h = Reg(DT.half, name="i4_h")
    h <<= sh[0]
    q1 = Reg(DT.int4, name="i4_q1")
    q1_u8 = q1.reinterpret(DT.uint8, name="i4_q1_u8")
    cast(q1, h, rnd, m16)  # f16 -> i4, rnd_sat_part_t
    o_f16[0] <<= q1_u8.pack4()

    n = Reg(DT.int16, name="i4_n")
    n <<= si[0]
    q2 = Reg(DT.int4, name="i4_q2")
    q2_u8 = q2.reinterpret(DT.uint8, name="i4_q2_u8")
    cast(q2, n, sat, mi16)  # i16 -> i4, sat_part_t
    o_i16[0] <<= q2_u8.pack4()


@vf()
def i4_widen_vf(sq: Tensor, o_f16: Tensor, o_bf16: Tensor, o_i16: Tensor):
    m16 = MaskReg(DT.half, name="i4w_m16")
    mb16 = MaskReg(DT.bfloat16, name="i4w_mb16")
    mi16 = MaskReg(DT.int16, name="i4w_mi16")
    plain = CastConfig(reg_layout=RegLayout.ZERO, name="i4_widen")

    q = Reg(DT.int4, name="i4w_q")
    ub_to_reg_unpack4(q, sq[0])  # 64 dense carrier bytes -> one byte in every four

    h = Reg(DT.half, name="i4w_h")
    cast(h, q, plain, m16)  # i4 -> f16, part_t
    o_f16[0] <<= h
    b = Reg(DT.bfloat16, name="i4w_b")
    cast(b, q, plain, mb16)  # i4 -> bf16, part_t
    o_bf16[0] <<= b
    n = Reg(DT.int16, name="i4w_n")
    cast(n, q, plain, mi16)  # i4 -> i16, part_t
    o_i16[0] <<= n


@kernel(mode="vec", block_dim=1)
def cast_i4_narrow(xh: GM[f16, (ROWS, I4_LANES)], xi: GM[i16, (ROWS, I4_LANES)],
    o_f16: GM[u8, (ROWS, I4_BYTES)], o_i16: GM[u8, (ROWS, I4_BYTES)]):
    ubh = Tensor(DT.half, [1, I4_LANES], Position.UB, name="i4_ubh")
    ubi = Tensor(DT.int16, [1, I4_LANES], Position.UB, name="i4_ubi")
    oa = Tensor(DT.uint8, [1, I4_BYTES], Position.UB, name="i4_oa")
    ob = Tensor(DT.uint8, [1, I4_BYTES], Position.UB, name="i4_ob")
    rows_per_vec = CeilDiv(ROWS, GetVecNum())
    row_begin = Var(rows_per_vec * GetVecIdx())
    row_end = Min(row_begin + rows_per_vec, ROWS)
    with auto_sync():
        for row in range(row_begin, row_end):
            ubh[:, :] <<= xh[row : row + 1, :]
            ubi[:, :] <<= xi[row : row + 1, :]
            i4_narrow_vf(ubh, ubi, oa, ob)
            o_f16[row : row + 1, :] <<= oa
            o_i16[row : row + 1, :] <<= ob
    return o_f16, o_i16


@kernel(mode="vec", block_dim=1)
def cast_i4_widen(xq: GM[u8, (ROWS, I4_BYTES)], o_f16: GM[f16, (ROWS, I4_LANES)],
    o_bf16: GM[bf16, (ROWS, I4_LANES)], o_i16: GM[i16, (ROWS, I4_LANES)]):
    ubq = Tensor(DT.uint8, [1, I4_BYTES], Position.UB, name="i4w_ubq")
    oa = Tensor(DT.half, [1, I4_LANES], Position.UB, name="i4w_oa")
    ob = Tensor(DT.bfloat16, [1, I4_LANES], Position.UB, name="i4w_ob")
    oc = Tensor(DT.int16, [1, I4_LANES], Position.UB, name="i4w_oc")
    rows_per_vec = CeilDiv(ROWS, GetVecNum())
    row_begin = Var(rows_per_vec * GetVecIdx())
    row_end = Min(row_begin + rows_per_vec, ROWS)
    with auto_sync():
        for row in range(row_begin, row_end):
            ubq[:, :] <<= xq[row : row + 1, :]
            i4_widen_vf(ubq, oa, ob, oc)
            o_f16[row : row + 1, :] <<= oa
            o_bf16[row : row + 1, :] <<= ob
            o_i16[row : row + 1, :] <<= oc
    return o_f16, o_bf16, o_i16
