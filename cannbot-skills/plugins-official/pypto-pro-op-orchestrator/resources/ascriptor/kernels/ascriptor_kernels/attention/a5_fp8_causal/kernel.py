# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Causal E5M2 flash attention: FP32 scores, maxima, sums and output, with only the probability matrix cast to E5M2 for PV."""

from ascriptor.a5 import *



TILE_M = 128
TILE_N = 128
D_HEAD = 128
ROWS_PER_SB = TILE_M // 2
CHUNKS_N = TILE_N // 64
CHUNKS_D = D_HEAD // 64
NEG_LARGE = -1.0e30


@vf()
def init_row_state_vf(ub_rmax: Tensor, ub_rsum: Tensor, ub_accum: Tensor):
    zero = Reg(DT.float)
    neg = Reg(DT.float)
    zero <<= 0.0
    neg <<= NEG_LARGE

    for r in range(ROWS_PER_SB):
        ub_rmax[0:1, r:r + 1] <<= neg.single_value()
        ub_rsum[0:1, r:r + 1] <<= zero.single_value()

    for i in range(ROWS_PER_SB * D_HEAD // 64):
        ub_accum[i * 64] <<= zero


@vf()
def scale_score_multi(ub_score: Tensor, scale: Var, rows: Var):
    row_regs = RegList(DT.float, CHUNKS_N)

    for r in range(rows):
        score_row = ub_score[r:r + 1, :]
        row_regs <<= score_row
        row_regs <<= row_regs * scale
        score_row <<= row_regs
        vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def apply_causal_diagonal_sb0(ub_score: Tensor, rows: Var):
    neg = Reg(DT.float)
    cols = Reg(DT.int)
    reg = Reg(DT.float)
    mask = MaskReg(DT.int, init_mode=MaskType.NONE)

    neg <<= NEG_LARGE
    for r in range(rows):
        row_off = Var(r * TILE_N)
        for c in range(CHUNKS_N):
            off = Var(row_off + c * 64)
            reg <<= ub_score[off]
            cols.arange(c * 64)
            compare(mask, cols, r + 1, CompareMode.LT)
            select(reg, reg, neg, mask=mask)
            ub_score[off] <<= reg
            vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def apply_causal_diagonal_sb1(ub_score: Tensor, rows: Var):
    neg = Reg(DT.float)
    cols = Reg(DT.int)
    reg = Reg(DT.float)
    mask = MaskReg(DT.int, init_mode=MaskType.NONE)

    neg <<= NEG_LARGE
    for r in range(rows):
        row_off = Var(r * TILE_N)
        for c in range(CHUNKS_N):
            off = Var(row_off + c * 64)
            reg <<= ub_score[off]
            cols.arange(c * 64)
            compare(mask, cols, r + ROWS_PER_SB + 1, CompareMode.LT)
            select(reg, reg, neg, mask=mask)
            ub_score[off] <<= reg
            vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def apply_n_tail_mask_multi(ub_score: Tensor, valid_n: Var, rows: Var):
    neg = Reg(DT.float)
    cols = Reg(DT.int)
    reg = Reg(DT.float)
    mask = MaskReg(DT.int, init_mode=MaskType.NONE)

    neg <<= NEG_LARGE
    for r in range(rows):
        row_off = Var(r * TILE_N)
        for c in range(CHUNKS_N):
            off = Var(row_off + c * 64)
            reg <<= ub_score[off]
            cols.arange(c * 64)
            compare(mask, cols, valid_n, CompareMode.LT)
            select(reg, reg, neg, mask=mask)
            ub_score[off] <<= reg
            vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def softmax_update_multi(
    ub_score: Tensor, ub_rmax: Tensor, ub_rsum: Tensor, ub_expdiff_slot: Tensor, rows: Var,
):
    row_regs = RegList(DT.float, CHUNKS_N)
    prev_max = Reg(DT.float)
    max_reg = Reg(DT.float)
    prev_sum = Reg(DT.float)
    sum_reg = Reg(DT.float)
    expdiff_reg = Reg(DT.float)

    for r in range(rows):
        score_row = ub_score[r:r + 1, :]

        row_regs <<= score_row
        prev_max <<= ub_rmax[0:1, r:r + 1].single()
        prev_sum <<= ub_rsum[0:1, r:r + 1].single()

        max_reg <<= row_regs.cmax()
        max_reg <<= max_reg.dup()
        max_reg <<= max_reg.vmax(prev_max)

        expdiff_reg <<= prev_max - max_reg
        expdiff_reg <<= expdiff_reg.exp()

        row_regs <<= row_regs - max_reg
        row_regs <<= row_regs.exp()

        sum_reg <<= row_regs.cadd()
        sum_reg <<= sum_reg.dup()
        sum_reg <<= sum_reg + expdiff_reg * prev_sum

        score_row <<= row_regs
        ub_rmax[0:1, r:r + 1] <<= max_reg.single_value()
        ub_rsum[0:1, r:r + 1] <<= sum_reg.single_value()
        ub_expdiff_slot[0:1, r:r + 1] <<= expdiff_reg.single_value()
        vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def quantize_p_fp8_multi(ub_p_float: Tensor, ub_p_fp8: Tensor, rows: Var):
    src_reg = Reg(DT.float)
    dst_reg = Reg(DT.e5m2)
    mask_e5m2 = MaskReg(DT.e5m2)
    cfg_zero = CastConfig(reg_layout=RegLayout.ZERO, name="cfg_zero")

    for r in range(rows):
        row_off = Var(r * TILE_N)
        for c in range(CHUNKS_N):
            off = Var(row_off + c * 64)
            src_reg <<= ub_p_float[off]
            cast(dst_reg, src_reg, cfg_zero, mask_e5m2)
            ub_p_fp8[off] <<= dst_reg.pack4()


@vf()
def accum_pv_multi(ub_accum: Tensor, ub_pv: Tensor, ub_expdiff_slot: Tensor, rows: Var):
    accum_regs = RegList(DT.float, CHUNKS_D)
    pv_regs = RegList(DT.float, CHUNKS_D)
    expdiff_reg = Reg(DT.float)

    for r in range(rows):
        accum_row = ub_accum[r:r + 1, :]
        pv_row = ub_pv[r:r + 1, :]

        expdiff_reg <<= ub_expdiff_slot[0:1, r:r + 1].single()
        accum_regs <<= accum_row
        pv_regs <<= pv_row
        accum_regs <<= accum_regs * expdiff_reg
        accum_regs <<= accum_regs + pv_regs
        accum_row <<= accum_regs
        vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def finalize_multi(ub_accum: Tensor, ub_rsum: Tensor, rows: Var):
    row_regs = RegList(DT.float, CHUNKS_D)
    rsum_reg = Reg(DT.float)

    for r in range(rows):
        accum_row = ub_accum[r:r + 1, :]

        rsum_reg <<= ub_rsum[0:1, r:r + 1].single()
        row_regs <<= accum_row
        row_regs <<= row_regs / rsum_reg
        accum_row <<= row_regs
        vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@kernel()
def flash_attn_full_fp8_causal_kernel(
    q: GM[DT.e5m2, ('TQ', 'D')], k: GM[DT.e5m2, ('TK', 'D')], v: GM[DT.e5m2, ('TK', 'D')],
    out: GM[f32, ('TQ', 'D')], rowmax: GM[f32, ('TST',)], rowsum: GM[f32, ('TST',)],
    BH: i32, S1: i32, S2: i32, D: i32, scale: f32,
):
    qk_mutex = CvMutex(0, depth=2, dst_end_pipe=Pipe.V)
    p_mutex = VcMutex(1, depth=2, dst_end_pipe=Pipe.MTE1)
    pv_mutex = CvMutex(2, depth=2, dst_end_pipe=Pipe.V)

    l1q = DBuff(DT.e5m2, [TILE_M, D_HEAD], Position.L1)
    l1k = TBuff(DT.e5m2, [TILE_N, D_HEAD], Position.L1)
    l1v = TBuff(DT.e5m2, [TILE_N, D_HEAD], Position.L1)
    l1p = TBuff(DT.e5m2, [TILE_M, TILE_N], Position.L1)
    l0c_qk = DBuff(DT.float, [TILE_M, TILE_N], Position.L0C)
    l0c_pv = DBuff(DT.float, [TILE_M, D_HEAD], Position.L0C)

    ub_score = DBuff(DT.float, [ROWS_PER_SB, TILE_N], Position.UB)
    ub_pv = DBuff(DT.float, [ROWS_PER_SB, D_HEAD], Position.UB)
    ub_p = DBuff(DT.e5m2, [ROWS_PER_SB, TILE_N], Position.UB)
    ub_rmax = Tensor(DT.float, [1, 64], Position.UB)
    ub_rsum = Tensor(DT.float, [1, 64], Position.UB)
    ub_accum = Tensor(DT.float, [ROWS_PER_SB, D_HEAD], Position.UB)
    ub_expdiff = TBuff(DT.float, [1, 64], Position.UB)

    sb_row = Var(GetSubBlockIdx() * ROWS_PER_SB)
    row_begin = Var(GetSubBlockIdx() * ROWS_PER_SB)

    tiles_m = CeilDiv(S1, TILE_M)
    tiles_n = CeilDiv(S2, TILE_N)
    total_m = Var(BH * tiles_m)
    per_core = CeilDiv(total_m, GetCubeNum())
    mt_begin = Var(per_core * GetCubeIdx())
    mt_end = Min(mt_begin + per_core, total_m)

    for gmt in range(mt_begin, mt_end):
        stage1_cnt = Var(0)
        stage2_cnt = Var(0)
        q_cnt = Var(0)

        bh_idx = Var(gmt // tiles_m)
        tile_m = Var(gmt % tiles_m)
        q_row = Var(bh_idx * S1 + tile_m * TILE_M)
        kv_base = Var(bh_idx * S2)
        valid_m = Min(TILE_M, S1 - tile_m * TILE_M)
        local_valid_m = Min(ROWS_PER_SB, Max(valid_m - sb_row, 0))
        active_tiles_n = Min(tiles_n, tile_m + 1)

        l1q[q_cnt][0:valid_m, 0:D] <<= q[q_row:q_row + valid_m, 0:D]
        bar_all()
        init_row_state_vf(ub_rmax, ub_rsum, ub_accum)

        with auto_sync():
            for ni in range(0, active_tiles_n + 1):
                if ni < active_tiles_n:
                    n_off = Var(ni * TILE_N)
                    kv_row = Var(kv_base + n_off)
                    valid_n = Min(TILE_N, S2 - n_off)

                    l1k[stage1_cnt][0:valid_n, 0:D] <<= k[kv_row:kv_row + valid_n, 0:D]
                    matmul(l0c_qk[stage1_cnt], l1q[q_cnt], l1k[stage1_cnt], splitk=D_HEAD)

                    qk_mutex.lock()
                    ub_score[stage1_cnt] <<= l0c_qk[stage1_cnt]
                    qk_mutex.ready()

                    qk_mutex.wait()
                    scale_score_multi(ub_score[stage1_cnt], scale, local_valid_m)
                    if ni == tile_m:
                        if GetSubBlockIdx() == 0:
                            apply_causal_diagonal_sb0(ub_score[stage1_cnt], local_valid_m)
                        else:
                            apply_causal_diagonal_sb1(ub_score[stage1_cnt], local_valid_m)
                    if valid_n < TILE_N:
                        apply_n_tail_mask_multi(ub_score[stage1_cnt], valid_n, local_valid_m)
                    softmax_update_multi(
                        ub_score[stage1_cnt], ub_rmax, ub_rsum, ub_expdiff[stage1_cnt], local_valid_m,
                    )
                    quantize_p_fp8_multi(ub_score[stage1_cnt], ub_p[stage1_cnt], local_valid_m)
                    qk_mutex.free()

                    p_mutex.lock()
                    if local_valid_m > 0:
                        l1p[stage1_cnt][row_begin:row_begin + local_valid_m, :] <<= ub_p[stage1_cnt][0:local_valid_m, :]
                    bar_all()
                    p_mutex.ready()

                    stage1_cnt += 1

                if ni > 0:
                    prev_n_off = Var((ni - 1) * TILE_N)
                    prev_row = Var(kv_base + prev_n_off)
                    prev_valid_n = Min(TILE_N, S2 - prev_n_off)

                    # Contract only valid V rows; overlapping fill/load is unsafe (RFC0005).
                    l1v[stage2_cnt][0:prev_valid_n, 0:D] <<= v[prev_row:prev_row + prev_valid_n, 0:D]

                    p_mutex.wait()
                    matmul(l0c_pv[stage2_cnt], l1p[stage2_cnt], l1v[stage2_cnt].T, k=prev_valid_n, splitn=D_HEAD)
                    p_mutex.free()

                    pv_mutex.lock()
                    ub_pv[stage2_cnt] <<= l0c_pv[stage2_cnt]
                    pv_mutex.ready()

                    pv_mutex.wait()
                    accum_pv_multi(ub_accum, ub_pv[stage2_cnt], ub_expdiff[stage2_cnt], local_valid_m)
                    pv_mutex.free()

                    stage2_cnt += 1

        finalize_multi(ub_accum, ub_rsum, local_valid_m)
        bar_all()

        out_row = Var(q_row + row_begin)
        if local_valid_m > 0:
            out[out_row:out_row + local_valid_m, 0:D] <<= ub_accum[0:local_valid_m, 0:D]
            stats_row = Var(bh_idx * CeilDiv(S1, 8) * 8 + tile_m * TILE_M + row_begin)
            rowmax[stats_row:stats_row + local_valid_m] <<= ub_rmax[0:1, 0:local_valid_m]
            rowsum[stats_row:stats_row + local_valid_m] <<= ub_rsum[0:1, 0:local_valid_m]
    return out, rowmax, rowsum
