# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Four one-query FP16 attention bodies that differ in exactly two ways: the streamed key tile (128 or 256) and how the probability tile is published to L1 (ND copy or compact NZ).

The helpers a body needs at its own tile width carry that width in their name; the three that
depend only on D_HEAD are shared, except in mha_ifa_256, whose copies dropped an unused Dn
parameter and so are kept apart as *_nd256."""

import math

from ascriptor.a5 import *

# ----------------------------------------------------------------------------------------------------
# nd128.py
# ----------------------------------------------------------------------------------------------------

BASES_128 = 128
D_HEAD = 128
SCALE = math.sqrt(1.0 / D_HEAD)

# Row-specialized streamed MHA example for decode-like `L=1` use.


@vf()
def scale_qk_row_128(ub_qk: Tensor, ub_scaled_qk: Tensor):
    qk_regs = RegList(DT.float, BASES_128 // 64)
    qk_regs <<= ub_qk[0]
    qk_regs <<= qk_regs * SCALE
    ub_scaled_qk[0] <<= qk_regs


@vf()
def get_p_and_max_sum_row_128(ub_qk: Tensor, ub_max: Tensor, ub_sum: Tensor, ub_expdiff: Tensor, ub_p: Tensor):
    regs = RegList(DT.float, BASES_128 // 64)
    prev_max = Reg(DT.float)
    max_reg = Reg(DT.float)
    prev_sum = Reg(DT.float)
    sum_reg = Reg(DT.float)
    expdiff_reg = Reg(DT.float)

    regs <<= ub_qk[0]
    prev_max <<= ub_max[0].single()
    prev_sum <<= ub_sum[0].single()

    max_reg <<= regs.cmax()
    max_reg <<= max_reg.dup()
    max_reg <<= max_reg.vmax(prev_max)

    expdiff_reg <<= prev_max - max_reg
    expdiff_reg <<= expdiff_reg.exp()

    regs <<= regs - max_reg
    regs <<= regs.exp()

    sum_reg <<= regs.cadd()
    sum_reg <<= sum_reg.dup()
    sum_reg <<= sum_reg + expdiff_reg * prev_sum

    ub_p[0] <<= regs
    ub_max[0] <<= max_reg.single_value()
    ub_sum[0] <<= sum_reg.single_value()
    ub_expdiff[0] <<= expdiff_reg.single_value()


@vf()
def init_buffers(qk_sum: Tensor, qk_max: Tensor, expdiff0: Tensor, expdiff1: Tensor, ub_accum: Tensor, Dn: Var):
    reg = Reg(DT.float)
    reg <<= 0.0

    qk_sum <<= reg
    for i in range(8 * D_HEAD // 64):
        ub_accum[i * 64] <<= reg

    reg <<= -99999.0
    qk_max <<= reg

    reg <<= 1.0
    expdiff0 <<= reg
    expdiff1 <<= reg


@vf()
def accum_pv_row(accum: Tensor, curr_pv: Tensor, expdiff: Tensor, Dn: Var):
    accum_regs = RegList(DT.float, D_HEAD // 64)
    curr_regs = RegList(DT.float, D_HEAD // 64)
    expdiff_reg = Reg(DT.float)

    expdiff_reg <<= expdiff[0].single()
    accum_regs <<= accum[0]
    curr_regs <<= curr_pv[0]
    accum_regs <<= accum_regs * expdiff_reg
    accum_regs <<= accum_regs + curr_regs
    accum[0] <<= accum_regs


@vf()
def divide_sum_row(accum: Tensor, sum_ub: Tensor, Dn: Var):
    regs = RegList(DT.float, D_HEAD // 64)
    sum_reg = Reg(DT.float)

    sum_reg <<= sum_ub[0].single()
    regs <<= accum[0]
    regs <<= regs / sum_reg
    accum[0] <<= regs


@kernel()
def mha_ifa_v2(
    q: GM[f16, ('bh', 'L', 'Dn')], k: GM[f16, ('bh', 'S', 'Dn')], v: GM[f16, ('bh', 'S', 'Dn')], output: GM[f32, ('bh', 'L', 'Dn')],
    bh: i32, L: i32, S: i32, Dn: i32,
):
    qk_mutex = CvMutex(0, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    p_mutex = VcMutex(1, depth=2, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)
    pv_mutex = CvMutex(2, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)

    l1q = DBuff(DT.half, [16, Dn], Position.L1)
    l1k = TBuff(DT.half, [BASES_128, Dn], Position.L1)
    l1v = TBuff(DT.half, [BASES_128, Dn], Position.L1)
    l1p = DBuff(DT.half, [16, BASES_128], Position.L1)
    l0c_qk = DBuff(DT.float, [16, BASES_128], Position.L0C)
    l0c_pv = DBuff(DT.float, [16, Dn], Position.L0C)

    ub_qk = DBuff(DT.float, [8, BASES_128], Position.UB)
    ub_pv = DBuff(DT.float, [8, Dn], Position.UB)
    ub_scaled_qk = Tensor(DT.float, [8, BASES_128], Position.UB)
    ub_p_half = DBuff(DT.half, [8, BASES_128], Position.UB)
    ub_qk_sum = Tensor(DT.float, [1, 64], Position.UB)
    ub_qk_max = Tensor(DT.float, [1, 64], Position.UB)
    ub_expdiff = DBuff(DT.float, [1, 64], Position.UB)
    ub_accum = Tensor(DT.float, [8, Dn], Position.UB)

    bh_per_core = CeilDiv(bh, GetCubeNum())
    bh_begin = Var(bh_per_core * GetCubeIdx())
    bh_end = Min(bh_begin + bh_per_core, bh)

    rows = Var(1)
    q_cnt = Var(0)
    stage1_cnt = Var(0)
    stage2_cnt = Var(0)

    for bh_idx in range(bh_begin, bh_end):
        l1q[q_cnt][:rows, :] <<= q[bh_idx, :rows, :]
        bar_all()
        init_buffers(ub_qk_sum, ub_qk_max, ub_expdiff[0], ub_expdiff[1], ub_accum, Dn)

        with auto_sync():
            for s in range(0, S + BASES_128, BASES_128):
                if s < S:
                    l1k[stage1_cnt] <<= k[bh_idx, s:s + BASES_128, :]
                    matmul(l0c_qk[stage1_cnt], l1q[q_cnt], l1k[stage1_cnt], splitk=128)

                    qk_mutex.lock()
                    ub_qk[stage1_cnt] <<= l0c_qk[stage1_cnt]
                    qk_mutex.ready()

                    qk_mutex.wait()
                    scale_qk_row_128(ub_qk[stage1_cnt], ub_scaled_qk)
                    get_p_and_max_sum_row_128(
                        ub_scaled_qk,
                        ub_qk_max,
                        ub_qk_sum,
                        ub_expdiff[stage1_cnt],
                        ub_p_half[stage1_cnt],
                    )
                    qk_mutex.free()

                    p_mutex.lock()
                    if GetSubBlockIdx() == 0:
                        l1p[stage1_cnt][:rows, :] <<= ub_p_half[stage1_cnt][:rows, :]
                    p_mutex.ready()
                    stage1_cnt += 1

                if s > 0:
                    prev_s = Var(s - BASES_128)
                    l1v[stage2_cnt] <<= v[bh_idx, prev_s:prev_s + BASES_128, :]

                    p_mutex.wait()
                    matmul(l0c_pv[stage2_cnt], l1p[stage2_cnt], l1v[stage2_cnt].T, splitn=128)
                    p_mutex.free()

                    pv_mutex.lock()
                    ub_pv[stage2_cnt] <<= l0c_pv[stage2_cnt]
                    pv_mutex.ready()

                    pv_mutex.wait()
                    accum_pv_row(ub_accum, ub_pv[stage2_cnt], ub_expdiff[stage2_cnt], Dn)
                    pv_mutex.free()
                    stage2_cnt += 1

        divide_sum_row(ub_accum, ub_qk_sum, Dn)
        bar_all()
        if GetSubBlockIdx() == 0:
            output[bh_idx, :rows, :] <<= ub_accum[:rows, :]

        q_cnt += 1

    return output

# ----------------------------------------------------------------------------------------------------
# nz128.py
# ----------------------------------------------------------------------------------------------------

# Row-specialized streamed MHA example that publishes probability tiles to L1 in NZ layout.






@vf()
def pack_p_to_nz_row_128(src_nd: Tensor, dst_nz: Tensor, n_rows: Var):
    reg = Reg(DT.half)
    for i in range(n_rows):
        reg <<= src_nd[i * BASES_128]
        reg_to_ub(dst_nz[i * 16], reg, n_rows)








@kernel()
def mha_ifa_nz(
    q: GM[f16, ('bh', 'L', 'Dn')], k: GM[f16, ('bh', 'S', 'Dn')], v: GM[f16, ('bh', 'S', 'Dn')], output: GM[f32, ('bh', 'L', 'Dn')],
    bh: i32, L: i32, S: i32, Dn: i32,
):
    qk_mutex = CvMutex(0, depth=2, dst_end_pipe=Pipe.V)
    p_mutex = VcMutex(1, depth=2, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)
    pv_mutex = CvMutex(2, depth=2, dst_end_pipe=Pipe.V)

    l1q = DBuff(DT.half, [16, Dn], Position.L1)
    l1k = TBuff(DT.half, [BASES_128, Dn], Position.L1)
    l1v = TBuff(DT.half, [BASES_128, Dn], Position.L1)
    l1p = DBuff(DT.half, [16, BASES_128], Position.L1)
    l0c_qk = DBuff(DT.float, [16, BASES_128], Position.L0C)
    l0c_pv = DBuff(DT.float, [16, Dn], Position.L0C)

    ub_qk = DBuff(DT.float, [8, BASES_128], Position.UB)
    ub_pv = DBuff(DT.float, [8, Dn], Position.UB)
    ub_scaled_qk = Tensor(DT.float, [8, BASES_128], Position.UB)
    ub_p_nd = DBuff(DT.half, [8, BASES_128], Position.UB)
    # the compact-NZ staging buffer's HEIGHT is the fractal-row stride the publish reads with
    # (M_src = src.shape[0], the same rule the oracle's ub_to_l1_nz documents), and
    # pack_p_to_nz_row_128 packs it with a pitch of `rows` - so the two agree only when the
    # buffer is exactly `rows` tall. Declared [8, BASES_128] it was 8, and the publish read
    # seven fractal columns of never-written memory (an all-NaN golden, D-118).
    ub_p_nz = DBuff(DT.half, [1, BASES_128], Position.UB)
    ub_qk_sum = Tensor(DT.float, [1, 64], Position.UB)
    ub_qk_max = Tensor(DT.float, [1, 64], Position.UB)
    ub_expdiff = DBuff(DT.float, [1, 64], Position.UB)
    ub_accum = Tensor(DT.float, [8, Dn], Position.UB)

    bh_per_core = CeilDiv(bh, GetCubeNum())
    bh_begin = Var(bh_per_core * GetCubeIdx())
    bh_end = Min(bh_begin + bh_per_core, bh)

    rows = Var(1)
    q_cnt = Var(0)
    stage1_cnt = Var(0)
    stage2_cnt = Var(0)

    for bh_idx in range(bh_begin, bh_end):
        l1q[q_cnt][:rows, :] <<= q[bh_idx, :rows, :]
        bar_all()
        init_buffers(ub_qk_sum, ub_qk_max, ub_expdiff[0], ub_expdiff[1], ub_accum, Dn)

        with auto_sync():
            for s in range(0, S + BASES_128, BASES_128):
                if s < S:
                    l1k[stage1_cnt] <<= k[bh_idx, s:s + BASES_128, :]
                    matmul(l0c_qk[stage1_cnt], l1q[q_cnt], l1k[stage1_cnt], splitk=128)

                    qk_mutex.lock()
                    ub_qk[stage1_cnt] <<= l0c_qk[stage1_cnt]
                    qk_mutex.ready()

                    qk_mutex.wait()
                    scale_qk_row_128(ub_qk[stage1_cnt], ub_scaled_qk)
                    get_p_and_max_sum_row_128(
                        ub_scaled_qk,
                        ub_qk_max,
                        ub_qk_sum,
                        ub_expdiff[stage1_cnt],
                        ub_p_nd[stage1_cnt],
                    )
                    qk_mutex.free()

                    p_mutex.lock()
                    if GetSubBlockIdx() == 0:
                        pack_p_to_nz_row_128(ub_p_nd[stage1_cnt], ub_p_nz[stage1_cnt], rows)
                        l1p[stage1_cnt][:rows, :] <<= ub_p_nz[stage1_cnt][:rows, :].nz()
                    p_mutex.ready()
                    stage1_cnt += 1

                if s > 0:
                    prev_s = Var(s - BASES_128)
                    l1v[stage2_cnt] <<= v[bh_idx, prev_s:prev_s + BASES_128, :]

                    p_mutex.wait()
                    matmul(l0c_pv[stage2_cnt], l1p[stage2_cnt], l1v[stage2_cnt].T, splitn=128)
                    p_mutex.free()

                    pv_mutex.lock()
                    ub_pv[stage2_cnt] <<= l0c_pv[stage2_cnt]
                    pv_mutex.ready()

                    pv_mutex.wait()
                    accum_pv_row(ub_accum, ub_pv[stage2_cnt], ub_expdiff[stage2_cnt], Dn)
                    pv_mutex.free()
                    stage2_cnt += 1

        divide_sum_row(ub_accum, ub_qk_sum, Dn)
        bar_all()
        if GetSubBlockIdx() == 0:
            output[bh_idx, :rows, :] <<= ub_accum[:rows, :]

        q_cnt += 1

    return output

# ----------------------------------------------------------------------------------------------------
# nd256.py
# ----------------------------------------------------------------------------------------------------

BASES_256 = 256
SPLIT_K = 64
SPLIT_N = 64

# Row-specialized streamed MHA example for decode-like `L=1` use.
# This variant keeps `BASES_256=256` on-chip, uses `splitk=64` for qk, and `splitn=64` for pv.


@vf()
def scale_qk_row_256(ub_qk: Tensor, ub_scaled_qk: Tensor):
    qk_regs = RegList(DT.float, BASES_256 // 64)
    qk_regs <<= ub_qk[0]
    qk_regs <<= qk_regs * SCALE
    ub_scaled_qk[0] <<= qk_regs


@vf()
def get_p_and_max_sum_row_256(ub_qk: Tensor, ub_max: Tensor, ub_sum: Tensor, ub_expdiff: Tensor, ub_p: Tensor):
    regs = RegList(DT.float, BASES_256 // 64)
    prev_max = Reg(DT.float)
    max_reg = Reg(DT.float)
    prev_sum = Reg(DT.float)
    sum_reg = Reg(DT.float)
    expdiff_reg = Reg(DT.float)

    regs <<= ub_qk[0]
    prev_max <<= ub_max[0].single()
    prev_sum <<= ub_sum[0].single()

    max_reg <<= regs.cmax()
    max_reg <<= max_reg.dup()
    max_reg <<= max_reg.vmax(prev_max)

    expdiff_reg <<= prev_max - max_reg
    expdiff_reg <<= expdiff_reg.exp()

    regs <<= regs - max_reg
    regs <<= regs.exp()

    sum_reg <<= regs.cadd()
    sum_reg <<= sum_reg.dup()
    sum_reg <<= sum_reg + expdiff_reg * prev_sum

    ub_p[0] <<= regs
    ub_max[0] <<= max_reg.single_value()
    ub_sum[0] <<= sum_reg.single_value()
    ub_expdiff[0] <<= expdiff_reg.single_value()


@vf()
def init_buffers_nd256(qk_sum: Tensor, qk_max: Tensor, expdiff0: Tensor, expdiff1: Tensor, ub_accum: Tensor):
    reg = Reg(DT.float)
    reg <<= 0.0

    qk_sum <<= reg
    for i in range(8 * D_HEAD // 64):
        ub_accum[i * 64] <<= reg

    reg <<= -99999.0
    qk_max <<= reg

    reg <<= 1.0
    expdiff0 <<= reg
    expdiff1 <<= reg


@vf()
def accum_pv_row_nd256(accum: Tensor, curr_pv: Tensor, expdiff: Tensor):
    accum_regs = RegList(DT.float, D_HEAD // 64)
    curr_regs = RegList(DT.float, D_HEAD // 64)
    expdiff_reg = Reg(DT.float)

    expdiff_reg <<= expdiff[0].single()
    accum_regs <<= accum[0]
    curr_regs <<= curr_pv[0]
    accum_regs <<= accum_regs * expdiff_reg
    accum_regs <<= accum_regs + curr_regs
    accum[0] <<= accum_regs


@vf()
def divide_sum_row_nd256(accum: Tensor, sum_ub: Tensor):
    regs = RegList(DT.float, D_HEAD // 64)
    sum_reg = Reg(DT.float)

    sum_reg <<= sum_ub[0].single()
    regs <<= accum[0]
    regs <<= regs / sum_reg
    accum[0] <<= regs


@kernel()
def mha_ifa_256(
    q: GM[f16, ('bh', 'L', 'Dn')], k: GM[f16, ('bh', 'S', 'Dn')], v: GM[f16, ('bh', 'S', 'Dn')], output: GM[f32, ('bh', 'L', 'Dn')],
    bh: i32, L: i32, S: i32, Dn: i32,
):
    qk_mutex = CvMutex(0, depth=2, dst_end_pipe=Pipe.V)
    p_mutex = VcMutex(1, depth=2, dst_end_pipe=Pipe.MTE1)
    pv_mutex = CvMutex(2, depth=2, dst_end_pipe=Pipe.V)

    l1q = DBuff(DT.half, [16, D_HEAD], Position.L1)
    l1k = TBuff(DT.half, [BASES_256, D_HEAD], Position.L1)
    l1v = TBuff(DT.half, [BASES_256, D_HEAD], Position.L1)
    l1p = DBuff(DT.half, [16, BASES_256], Position.L1)
    l0c_qk = DBuff(DT.float, [16, BASES_256], Position.L0C)
    l0c_pv = DBuff(DT.float, [16, D_HEAD], Position.L0C)

    ub_qk = DBuff(DT.float, [8, BASES_256], Position.UB)
    ub_pv = DBuff(DT.float, [8, D_HEAD], Position.UB)
    ub_scaled_qk = Tensor(DT.float, [8, BASES_256], Position.UB)
    ub_p_half = DBuff(DT.half, [8, BASES_256], Position.UB)
    ub_qk_sum = Tensor(DT.float, [1, 64], Position.UB)
    ub_qk_max = Tensor(DT.float, [1, 64], Position.UB)
    ub_expdiff = DBuff(DT.float, [1, 64], Position.UB)
    ub_accum = Tensor(DT.float, [8, D_HEAD], Position.UB)

    bh_per_core = CeilDiv(bh, GetCubeNum())
    bh_begin = Var(bh_per_core * GetCubeIdx())
    bh_end = Min(bh_begin + bh_per_core, bh)

    rows = Var(1)
    q_cnt = Var(0)
    stage1_cnt = Var(0)
    stage2_cnt = Var(0)

    for bh_idx in range(bh_begin, bh_end):
        l1q[q_cnt][:rows, :] <<= q[bh_idx, :rows, :]
        bar_all()
        init_buffers_nd256(ub_qk_sum, ub_qk_max, ub_expdiff[0], ub_expdiff[1], ub_accum)

        with auto_sync():
            for s in range(0, S + BASES_256, BASES_256):
                if s < S:
                    l1k[stage1_cnt] <<= k[bh_idx, s:s + BASES_256, :]
                    matmul(l0c_qk[stage1_cnt], l1q[q_cnt], l1k[stage1_cnt], splitk=SPLIT_K)

                    qk_mutex.lock()
                    ub_qk[stage1_cnt] <<= l0c_qk[stage1_cnt]
                    qk_mutex.ready()

                    qk_mutex.wait()
                    scale_qk_row_256(ub_qk[stage1_cnt], ub_scaled_qk)
                    get_p_and_max_sum_row_256(
                        ub_scaled_qk,
                        ub_qk_max,
                        ub_qk_sum,
                        ub_expdiff[stage1_cnt],
                        ub_p_half[stage1_cnt],
                    )
                    qk_mutex.free()

                    p_mutex.lock()
                    if GetSubBlockIdx() == 0:
                        l1p[stage1_cnt][:rows, :] <<= ub_p_half[stage1_cnt][:rows, :]
                    p_mutex.ready()
                    stage1_cnt += 1

                if s > 0:
                    prev_s = Var(s - BASES_256)
                    l1v[stage2_cnt] <<= v[bh_idx, prev_s:prev_s + BASES_256, :]

                    p_mutex.wait()
                    matmul(l0c_pv[stage2_cnt], l1p[stage2_cnt], l1v[stage2_cnt].T, splitn=SPLIT_N)
                    p_mutex.free()

                    pv_mutex.lock()
                    ub_pv[stage2_cnt] <<= l0c_pv[stage2_cnt]
                    pv_mutex.ready()

                    pv_mutex.wait()
                    accum_pv_row_nd256(ub_accum, ub_pv[stage2_cnt], ub_expdiff[stage2_cnt])
                    pv_mutex.free()
                    stage2_cnt += 1

        divide_sum_row_nd256(ub_accum, ub_qk_sum)
        bar_all()
        if GetSubBlockIdx() == 0:
            output[bh_idx, :rows, :] <<= ub_accum[:rows, :]

        q_cnt += 1

    return output

# ----------------------------------------------------------------------------------------------------
# nz256.py
# ----------------------------------------------------------------------------------------------------

PACK_REG_LANES = 128

# Row-specialized streamed MHA example that publishes probability tiles to L1 in NZ layout.
# This variant keeps the NZ handoff path from `mha_ifa_nz.py` while widening the streamed S tile to 256.






@vf()
def pack_p_to_nz_row_256(src_nd: Tensor, dst_nz: Tensor, n_rows: Var):
    reg0 = Reg(DT.half)
    reg1 = Reg(DT.half)
    for i in range(n_rows):
        reg0 <<= src_nd[i * BASES_256]
        reg1 <<= src_nd[i * BASES_256 + PACK_REG_LANES]
        reg_to_ub(dst_nz[i * 16], reg0, n_rows)
        # The second half-row starts at block 8 in packed-NZ order, so it advances by 128 * n_rows.
        reg_to_ub(dst_nz[i * 16 + PACK_REG_LANES * n_rows], reg1, n_rows)








@kernel()
def mha_ifa_nz_256(
    q: GM[f16, ('bh', 'L', 'Dn')], k: GM[f16, ('bh', 'S', 'Dn')], v: GM[f16, ('bh', 'S', 'Dn')], output: GM[f32, ('bh', 'L', 'Dn')],
    bh: i32, L: i32, S: i32, Dn: i32,
):
    qk_mutex = CvMutex(0, depth=2, dst_end_pipe=Pipe.V)
    p_mutex = VcMutex(1, depth=2, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)
    pv_mutex = CvMutex(2, depth=2, dst_end_pipe=Pipe.V)

    l1q = DBuff(DT.half, [16, Dn], Position.L1)
    l1k = TBuff(DT.half, [BASES_256, Dn], Position.L1)
    l1v = TBuff(DT.half, [BASES_256, Dn], Position.L1)
    l1p = DBuff(DT.half, [16, BASES_256], Position.L1)
    l0c_qk = DBuff(DT.float, [16, BASES_256], Position.L0C)
    l0c_pv = DBuff(DT.float, [16, Dn], Position.L0C)

    ub_qk = DBuff(DT.float, [8, BASES_256], Position.UB)
    ub_pv = DBuff(DT.float, [8, Dn], Position.UB)
    ub_scaled_qk = Tensor(DT.float, [8, BASES_256], Position.UB)
    ub_p_nd = DBuff(DT.half, [8, BASES_256], Position.UB)
    # the compact-NZ staging buffer's HEIGHT is the fractal-row stride the publish reads with
    # (M_src = src.shape[0], the same rule the oracle's ub_to_l1_nz documents), and
    # pack_p_to_nz_row_256 packs it with a pitch of `rows` - so the two agree only when the
    # buffer is exactly `rows` tall. Declared [8, BASES_256] it was 8, and the publish read
    # seven fractal columns of never-written memory (an all-NaN golden, D-118).
    ub_p_nz = DBuff(DT.half, [1, BASES_256], Position.UB)
    ub_qk_sum = Tensor(DT.float, [1, 64], Position.UB)
    ub_qk_max = Tensor(DT.float, [1, 64], Position.UB)
    ub_expdiff = DBuff(DT.float, [1, 64], Position.UB)
    ub_accum = Tensor(DT.float, [8, Dn], Position.UB)

    bh_per_core = CeilDiv(bh, GetCubeNum())
    bh_begin = Var(bh_per_core * GetCubeIdx())
    bh_end = Min(bh_begin + bh_per_core, bh)

    rows = Var(1)
    q_cnt = Var(0)
    stage1_cnt = Var(0)
    stage2_cnt = Var(0)

    for bh_idx in range(bh_begin, bh_end):
        l1q[q_cnt][:rows, :] <<= q[bh_idx, :rows, :]
        bar_all()
        init_buffers(ub_qk_sum, ub_qk_max, ub_expdiff[0], ub_expdiff[1], ub_accum, Dn)

        with auto_sync():
            for s in range(0, S + BASES_256, BASES_256):
                if s < S:
                    l1k[stage1_cnt] <<= k[bh_idx, s:s + BASES_256, :]
                    matmul(l0c_qk[stage1_cnt], l1q[q_cnt], l1k[stage1_cnt], splitk=SPLIT_K)

                    qk_mutex.lock()
                    ub_qk[stage1_cnt] <<= l0c_qk[stage1_cnt]
                    qk_mutex.ready()

                    qk_mutex.wait()
                    scale_qk_row_256(ub_qk[stage1_cnt], ub_scaled_qk)
                    get_p_and_max_sum_row_256(
                        ub_scaled_qk,
                        ub_qk_max,
                        ub_qk_sum,
                        ub_expdiff[stage1_cnt],
                        ub_p_nd[stage1_cnt],
                    )
                    qk_mutex.free()

                    p_mutex.lock()
                    if GetSubBlockIdx() == 0:
                        pack_p_to_nz_row_256(ub_p_nd[stage1_cnt], ub_p_nz[stage1_cnt], rows)
                        l1p[stage1_cnt][:rows, :] <<= ub_p_nz[stage1_cnt][:rows, :].nz()
                    p_mutex.ready()
                    stage1_cnt += 1

                if s > 0:
                    prev_s = Var(s - BASES_256)
                    l1v[stage2_cnt] <<= v[bh_idx, prev_s:prev_s + BASES_256, :]

                    p_mutex.wait()
                    matmul(l0c_pv[stage2_cnt], l1p[stage2_cnt], l1v[stage2_cnt].T, splitn=SPLIT_N)
                    p_mutex.free()

                    pv_mutex.lock()
                    ub_pv[stage2_cnt] <<= l0c_pv[stage2_cnt]
                    pv_mutex.ready()

                    pv_mutex.wait()
                    accum_pv_row(ub_accum, ub_pv[stage2_cnt], ub_expdiff[stage2_cnt], Dn)
                    pv_mutex.free()
                    stage2_cnt += 1

        divide_sum_row(ub_accum, ub_qk_sum, Dn)
        bar_all()
        if GetSubBlockIdx() == 0:
            output[bh_idx, :rows, :] <<= ub_accum[:rows, :]

        q_cnt += 1

    return output
