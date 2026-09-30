# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One-query E4M3 attention over 256-key tiles, with independent Q/K/V scales and 16x-scaled E4M3 probability staging."""

import math

from ascriptor.a5 import *



BASES = 256
D_HEAD = 128
P_SCALE = 16.0
NEG_INF = -99999.0
SPLIT_K = 64
SPLIT_N = 64
SOFTMAX_SCALE = 1.0 / math.sqrt(D_HEAD)


@vf()
def init_row_state(qk_sum: Tensor, qk_max: Tensor, expdiff0: Tensor, expdiff1: Tensor, ub_accum: Tensor, Dn: Var):
    zero_reg = Reg(DT.float)
    neg_reg = Reg(DT.float)

    zero_reg <<= 0.0
    qk_sum <<= zero_reg
    for i in range(8 * D_HEAD // 64):
        ub_accum[i * 64] <<= zero_reg

    neg_reg <<= NEG_INF
    qk_max <<= neg_reg

    zero_reg <<= 1.0
    expdiff0 <<= zero_reg
    expdiff1 <<= zero_reg


@vf()
def scale_qk_row(ub_qk: Tensor, ub_scaled_qk: Tensor, scale: Var):
    qk_regs = RegList(DT.float, BASES // 64)
    qk_regs <<= ub_qk[0]
    qk_regs <<= qk_regs * scale
    ub_scaled_qk[0] <<= qk_regs


@vf()
def scale_qk_row_tail(ub_qk: Tensor, ub_scaled_qk: Tensor, scale: Var, valid_cols: Var):
    cols = Reg(DT.int)
    qk_reg = Reg(DT.float)
    neg_reg = Reg(DT.float)
    mask = MaskReg(DT.int, init_mode=MaskType.NONE)

    neg_reg <<= NEG_INF
    for chunk in range(BASES // 64):
        qk_reg <<= ub_qk[chunk * 64]
        qk_reg <<= qk_reg * scale
        cols.arange(chunk * 64)
        compare(mask, cols, valid_cols, CompareMode.LT)
        select(qk_reg, qk_reg, neg_reg, mask=mask)
        ub_scaled_qk[chunk * 64] <<= qk_reg


@vf()
def get_p_and_max_sum_row(ub_qk: Tensor, ub_max: Tensor, ub_sum: Tensor, ub_expdiff: Tensor, ub_p: Tensor):
    regs = RegList(DT.float, BASES // 64)
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

    regs <<= regs * P_SCALE
    ub_p[0] <<= regs
    ub_max[0] <<= max_reg.single_value()
    ub_sum[0] <<= sum_reg.single_value()
    ub_expdiff[0] <<= expdiff_reg.single_value()


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
def finalize_output_row(accum: Tensor, sum_ub: Tensor, scale_v: Var, Dn: Var):
    regs = RegList(DT.float, D_HEAD // 64)
    sum_reg = Reg(DT.float)

    sum_reg <<= sum_ub[0].single()
    regs <<= accum[0]
    regs <<= regs * scale_v
    regs <<= regs / sum_reg
    accum[0] <<= regs


@kernel()
def mha_ifa_fp8_scale_256(
    q: GM[DT.e4m3, ('bh', 'L', 'Dn')], k: GM[DT.e4m3, ('bh', 'S', 'Dn')], v: GM[DT.e4m3, ('bh', 'S', 'Dn')], output: GM[f32, ('bh', 'L', 'Dn')],
    bh: i32, L: i32, S: i32, Dn: i32, scale_q: f32, scale_k: f32, scale_v: f32,
):
    qk_mutex = CvMutex(0, depth=2, dst_end_pipe=Pipe.V)
    p_mutex = VcMutex(1, depth=2, dst_end_pipe=Pipe.MTE1)
    pv_mutex = CvMutex(2, depth=2, dst_end_pipe=Pipe.V)

    l1q = DBuff(DT.e4m3, [16, Dn], Position.L1)
    l1k = TBuff(DT.e4m3, [BASES, Dn], Position.L1)
    l1v = TBuff(DT.e4m3, [BASES, Dn], Position.L1)
    l1p = DBuff(DT.e4m3, [16, BASES], Position.L1)
    l0c_qk = DBuff(DT.float, [16, BASES], Position.L0C)
    l0c_pv = DBuff(DT.float, [16, Dn], Position.L0C)

    ub_qk = DBuff(DT.float, [8, BASES], Position.UB)
    ub_pv = DBuff(DT.float, [8, Dn], Position.UB)
    ub_scaled_qk = Tensor(DT.float, [8, BASES], Position.UB)
    ub_p = DBuff(DT.e4m3, [8, BASES], Position.UB)
    ub_qk_sum = Tensor(DT.float, [1, 64], Position.UB)
    ub_qk_max = Tensor(DT.float, [1, 64], Position.UB)
    ub_expdiff = DBuff(DT.float, [1, 64], Position.UB)
    ub_accum = Tensor(DT.float, [8, Dn], Position.UB)

    qk_scale = scale_q * SOFTMAX_SCALE
    qk_scale = qk_scale * scale_k
    out_scale = scale_v / P_SCALE

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
        init_row_state(ub_qk_sum, ub_qk_max, ub_expdiff[0], ub_expdiff[1], ub_accum, Dn)

        with auto_sync():
            for s in range(0, S + BASES, BASES):
                if s < S:
                    valid_cols = Min(BASES, S - s)
                    l1k[stage1_cnt] <<= k[bh_idx, s:s + valid_cols, :]
                    matmul(l0c_qk[stage1_cnt], l1q[q_cnt], l1k[stage1_cnt], splitk=SPLIT_K)

                    qk_mutex.lock()
                    ub_qk[stage1_cnt] <<= l0c_qk[stage1_cnt]
                    qk_mutex.ready()

                    qk_mutex.wait()
                    if valid_cols < BASES:
                        scale_qk_row_tail(ub_qk[stage1_cnt], ub_scaled_qk, qk_scale, valid_cols)
                    else:
                        scale_qk_row(ub_qk[stage1_cnt], ub_scaled_qk, qk_scale)
                    get_p_and_max_sum_row(
                        ub_scaled_qk,
                        ub_qk_max,
                        ub_qk_sum,
                        ub_expdiff[stage1_cnt],
                        ub_p[stage1_cnt],
                    )
                    qk_mutex.free()

                    p_mutex.lock()
                    if GetSubBlockIdx() == 0:
                        l1p[stage1_cnt][:rows, :] <<= ub_p[stage1_cnt][:rows, :]
                    p_mutex.ready()
                    stage1_cnt += 1

                if s > 0:
                    prev_s = Var(s - BASES)
                    prev_valid_cols = Min(BASES, S - prev_s)
                    # Contract only valid V rows; overlapping fill/load is unsafe (RFC0005).
                    l1v[stage2_cnt] <<= v[bh_idx, prev_s:prev_s + prev_valid_cols, :]

                    p_mutex.wait()
                    matmul(l0c_pv[stage2_cnt], l1p[stage2_cnt], l1v[stage2_cnt].T, k=prev_valid_cols, splitn=SPLIT_N)
                    p_mutex.free()

                    pv_mutex.lock()
                    ub_pv[stage2_cnt] <<= l0c_pv[stage2_cnt]
                    pv_mutex.ready()

                    pv_mutex.wait()
                    accum_pv_row(ub_accum, ub_pv[stage2_cnt], ub_expdiff[stage2_cnt], Dn)
                    pv_mutex.free()
                    stage2_cnt += 1

        finalize_output_row(ub_accum, ub_qk_sum, out_scale, Dn)
        bar_all()
        if GetSubBlockIdx() == 0:
            output[bh_idx, :rows, :] <<= ub_accum[:rows, :]

        q_cnt += 1

    return output
