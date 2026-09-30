# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Six A2/A3 schedules for the same 32-token block-causal attention, in one file.

The algorithm never changes: mask key//32 <= query//32, FP16 Q/K/V, FP32 row state and
output, probabilities rounded to FP16 before the cube PV. Only the schedule differs --
v1 a two-slot lag-1 partition, v2 a next-query prefetch, v3 a deeper lag-3 snake, v4
four-key groups, v5 the same groups with a V preload ring, v6 one continuous cross-M-tile
stream. They share every masking helper; putting them in one file is what makes that
visible, and what makes the constants they do NOT share -- V4_GROUP_LOOKAHEAD and
V4_GROUP_STAGE_SLOTS against v5/v6's GROUP_LOOKAHEAD and GROUP_STAGE_SLOTS -- impossible
to read past. build_kernel(variant, device) binds one of them to either facade."""

from ascriptor.a2 import *
from builtins import range as py_range
from functools import lru_cache
from importlib import import_module

# ----------------------------------------------------------------------------------------------------
# v1.py
# ----------------------------------------------------------------------------------------------------

# RFC-0009 batch-1 rewrite: the two hand-written credit events are gone (autosync's loop-carried
# machinery guards accum_ub / ub_rmax_s / ub_rsum_s across M tiles and the V->MTE3 flush), the
# stage1_cnt / stage2_cnt pair collapses into the loop beat `ni` (producer ni, consumer ni - 1 -
# the cross-tile hand-backs through p_ws/pv_ws bound the run-ahead, as on flash_attn_full), and
# the workspace rings are GMBuffs whose algebra the gmbuff pass checks.

TILE_M = 128
TILE_N = 128
TILE_K = 128
HALF_M = TILE_M // 2
HALF_N = TILE_N // 2
BLOCK_CAUSAL = 32
NEG_LARGE = -1.0e30


@func()
def build_suffix_invalid_mask(valid_cols: Var, out_mask: Var):
    signed_mask = Var(-1, DT.int64)
    two_i64 = Var(2, DT.int64)
    for _ in range(0, valid_cols):
        signed_mask <<= signed_mask * two_i64
    out_mask <<= signed_mask


@func()
def mask_score_half_suffix_invalid(score_half: Tensor, valid_cols: Var):
    if valid_cols == 0:
        dup(score_half, NEG_LARGE)
    elif valid_cols < HALF_N:
        suffix_mask = Var(0, DT.uint64)
        build_suffix_invalid_mask(valid_cols, suffix_mask)
        set_mask(0, suffix_mask)
        dup(score_half, NEG_LARGE)
        reset_mask()


@func()
def apply_score_tail_mask(ub_score: Tensor, valid_n: Var):
    left_valid = Min(valid_n, HALF_N)
    right_valid = Max(valid_n - HALF_N, 0)
    mask_score_half_suffix_invalid(ub_score[0:HALF_M, 0:HALF_N], left_valid)
    mask_score_half_suffix_invalid(ub_score[0:HALF_M, HALF_N:TILE_N], right_valid)


@func()
def apply_score_row_tail_mask_after_shift(ub_score: Tensor, valid_rows: Var):
    if valid_rows == 0:
        dup(ub_score, NEG_LARGE)
    elif valid_rows < HALF_M:
        dup(ub_score[valid_rows:HALF_M, 0:TILE_N], NEG_LARGE)


@func()
def apply_diagonal_tile_block32_causal_mask(ub_score: Tensor, sb_row: Var):
    if sb_row == 0:
        dup(
            ub_score[0:BLOCK_CAUSAL, BLOCK_CAUSAL:HALF_N], NEG_LARGE,
            repeat=BLOCK_CAUSAL, dst_rep_stride=16, count_per_rep=BLOCK_CAUSAL,
        )
        reset_mask()
        dup(ub_score[0:HALF_M, HALF_N:TILE_N], NEG_LARGE)
    else:
        dup(
            ub_score[0:BLOCK_CAUSAL, HALF_N + BLOCK_CAUSAL:TILE_N], NEG_LARGE,
            repeat=BLOCK_CAUSAL, dst_rep_stride=16, count_per_rep=BLOCK_CAUSAL,
        )
        reset_mask()


def flash_attn_full_pj_half_block32_causal_kernel(
    q: GM[f16, ('TQ', 'D')], k: GM[f16, ('TK', 'D')], v: GM[f16, ('TK', 'D')], out: GM[f32, ('TQ', 'D')], rowmax: GM[f32, ('TST',)], rowsum: GM[f32, ('TST',)],
    S1: i32, S2: i32, D: i32, BH: i32, scale: f32,
):
    score_ws = GMBuff(DT.float, [TILE_M, TILE_N], slots=2, name="score_ws")
    p_ws = GMBuff(DT.half, [TILE_M, TILE_N], slots=2, name="p_ws")
    pv_ws = GMBuff(DT.float, [TILE_M, TILE_K], slots=2, name="pv_ws")

    l1q = DBuff(DT.half, [TILE_M, TILE_K], Position.L1)
    l1k = DBuff(DT.half, [TILE_N, TILE_K], Position.L1)
    l1p = DBuff(DT.half, [TILE_M, TILE_N], Position.L1)
    l1v = DBuff(DT.half, [TILE_N, TILE_K], Position.L1)
    l0c = DBuff(DT.float, [TILE_M, TILE_N], Position.L0C)

    ub_score = Tensor(DT.float, [HALF_M, TILE_N], Position.UB)
    ub_pv = Tensor(DT.float, [HALF_M, TILE_K], Position.UB)
    ub_tmp = Tensor(DT.float, [HALF_M, HALF_N], Position.UB)
    ub_max_s = Tensor(DT.float, [1, HALF_M], Position.UB)
    ub_rmax_s = Tensor(DT.float, [1, HALF_M], Position.UB)
    ub_sum_s = Tensor(DT.float, [1, HALF_M], Position.UB)
    ub_rsum_s = Tensor(DT.float, [1, HALF_M], Position.UB)
    ub_zero_s = Tensor(DT.float, [1, HALF_M], Position.UB)
    ub_max = Tensor(DT.float, [HALF_M, 8], Position.UB)
    ub_rowsum = Tensor(DT.float, [HALF_M, 8], Position.UB)
    ub_p = Tensor(DT.half, [HALF_M, TILE_N], Position.UB)
    ub_expdiff = Tensor(DT.float, [HALF_M, 8], Position.UB)
    accum_ub = Tensor(DT.float, [HALF_M, TILE_K], Position.UB)
    expdiff_buf = DBuff(DT.float, [1, HALF_M], Position.UB)

    qk_mutex = CvMutex(0, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)
    p_mutex = VcMutex(1, depth=2, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)
    pv_mutex = CvMutex(2, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)

    l1qk_cnt = Var(0)
    l1pv_cnt = Var(0)
    l0c_cnt = Var(0)

    cube_idx = GetCubeIdx()
    sb = GetSubBlockIdx()
    sb_row = Var(sb * HALF_M)

    tiles_m = CeilDiv(S1, TILE_M)
    tiles_n = CeilDiv(S2, TILE_N)
    total_m = Var(BH * tiles_m)
    per_core = CeilDiv(total_m, GetCubeNum())
    mt_begin = Var(per_core * cube_idx)
    mt_end = Min(mt_begin + per_core, total_m)

    for gmt in range(mt_begin, mt_end):
        bh = Var(gmt // tiles_m)
        lmt = Var(gmt % tiles_m)
        q_row = Var(bh * S1 + lmt * TILE_M)
        kv_base = Var(bh * S2)
        valid_m = Min(TILE_M, S1 - lmt * TILE_M)
        local_valid_m = Min(HALF_M, Max(valid_m - sb_row, 0))
        active_tiles_n = Min(tiles_n, lmt + 1)

        # The previous MTE3 writeback still reads accum_ub, ub_rmax_s, and ub_rsum_s;
        # autosync's loop-carried WAR machinery guards the reinitialisation.
        with auto_sync():
            dup(ub_rmax_s, NEG_LARGE)
            dup(ub_rsum_s, 0.0)
            dup(ub_zero_s, 0.0)
            dup(accum_ub, 0.0)

        for ni in range(0, active_tiles_n + 1):
            if ni < active_tiles_n:
                with auto_sync():
                    n_off = Var(ni * TILE_N)
                    kv_row = Var(kv_base + n_off)
                    valid_n = Min(TILE_N, S2 - n_off)

                    l1q[l1qk_cnt] <<= q[q_row:q_row + valid_m, 0:D]
                    l1k[l1qk_cnt] <<= k[kv_row:kv_row + valid_n, 0:D]
                    matmul(l0c[l0c_cnt], l1q[l1qk_cnt], l1k[l1qk_cnt], is_init=True)

                    qk_mutex.lock()
                    score_ws[ni, 0:TILE_M, 0:TILE_N] <<= l0c[l0c_cnt]
                    qk_mutex.ready()

                    qk_mutex.wait()
                    ub_score <<= score_ws[ni, sb_row:sb_row + HALF_M, 0:TILE_N]

                    muls(ub_score, ub_score, scale)
                    if ni == lmt:
                        apply_diagonal_tile_block32_causal_mask(ub_score, sb_row)
                    if valid_n < TILE_N:
                        apply_score_tail_mask(ub_score, valid_n)
                    vmax(ub_tmp, ub_score[0:HALF_M, 0:HALF_N], ub_score[0:HALF_M, HALF_N:TILE_N])
                    cmax(ub_max_s, ub_tmp)

                    add(expdiff_buf[ni], ub_rmax_s, ub_zero_s)
                    vmax(ub_rmax_s, ub_rmax_s, ub_max_s)
                    sub(expdiff_buf[ni], expdiff_buf[ni], ub_rmax_s)
                    exp(expdiff_buf[ni], expdiff_buf[ni])

                    brcb(ub_max, ub_rmax_s, repeat=HALF_M // 8, dst_blk_stride=1, dst_rep_stride=8)
                    sub(ub_score[0:HALF_M, 0:HALF_N], ub_score[0:HALF_M, 0:HALF_N], ub_max)
                    sub(ub_score[0:HALF_M, HALF_N:TILE_N], ub_score[0:HALF_M, HALF_N:TILE_N], ub_max)
                    if local_valid_m < HALF_M:
                        apply_score_row_tail_mask_after_shift(ub_score, local_valid_m)

                    exp(ub_score, ub_score)
                    add(ub_tmp, ub_score[0:HALF_M, 0:HALF_N], ub_score[0:HALF_M, HALF_N:TILE_N])
                    cadd(ub_sum_s, ub_tmp)
                    mul(ub_rsum_s, ub_rsum_s, expdiff_buf[ni])
                    add(ub_rsum_s, ub_rsum_s, ub_sum_s)

                    cast(ub_p, ub_score)

                    p_mutex.lock()
                    p_ws[ni, sb_row:sb_row + HALF_M, 0:TILE_N] <<= ub_p
                    p_mutex.ready()
                    qk_mutex.free()

                    l1qk_cnt += 1
                    l0c_cnt += 1

            if ni > 0:
                with auto_sync():
                    prev_nt = Var(ni - 1)
                    prev_n_off = Var(prev_nt * TILE_N)
                    prev_valid_n = Min(TILE_N, S2 - prev_n_off)
                    v_row = Var(kv_base + prev_n_off)

                    p_mutex.wait()
                    l1p[l1pv_cnt] <<= p_ws[ni - 1, 0:TILE_M, 0:TILE_N]
                    l1v[l1pv_cnt] <<= v[v_row:v_row + prev_valid_n, 0:D]
                    matmul(l0c[l0c_cnt], l1p[l1pv_cnt], l1v[l1pv_cnt].T, k=prev_valid_n, is_init=True)
                    p_mutex.free()

                    pv_mutex.lock()
                    pv_ws[ni - 1, 0:TILE_M, 0:D] <<= l0c[l0c_cnt]
                    pv_mutex.ready()

                    pv_mutex.wait()
                    ub_pv <<= pv_ws[ni - 1, sb_row:sb_row + HALF_M, 0:D]

                    brcb(ub_expdiff, expdiff_buf[ni - 1], repeat=HALF_M // 8, dst_blk_stride=1, dst_rep_stride=8)
                    mul(accum_ub[0:HALF_M, 0:HALF_N], accum_ub[0:HALF_M, 0:HALF_N], ub_expdiff)
                    mul(accum_ub[0:HALF_M, HALF_N:TILE_K], accum_ub[0:HALF_M, HALF_N:TILE_K], ub_expdiff)
                    add(accum_ub, accum_ub, ub_pv)
                    pv_mutex.free()

                    l1pv_cnt += 1
                    l0c_cnt += 1

        with auto_sync():
            brcb(ub_rowsum, ub_rsum_s, repeat=HALF_M // 8, dst_blk_stride=1, dst_rep_stride=8)
            div(accum_ub[0:HALF_M, 0:HALF_N], accum_ub[0:HALF_M, 0:HALF_N], ub_rowsum)
            div(accum_ub[0:HALF_M, HALF_N:TILE_K], accum_ub[0:HALF_M, HALF_N:TILE_K], ub_rowsum)

            if local_valid_m > 0:
                out_row = Var(q_row + sb_row)
                stats_row = Var((out_row // S1) * CeilDiv(S1, 8) * 8 + out_row % S1)
                out[out_row:out_row + local_valid_m, 0:D] <<= accum_ub[0:local_valid_m, 0:D]
                rowmax[stats_row:stats_row + local_valid_m] <<= ub_rmax_s[0:1, 0:local_valid_m]
                rowsum[stats_row:stats_row + local_valid_m] <<= ub_rsum_s[0:1, 0:local_valid_m]

    return out, rowmax, rowsum




@lru_cache(maxsize=2)
def build_v1_kernel(device):
    if device not in ("a2", "a3"):
        raise ValueError("Shared tensor-vector attention supports A2/A3")
    return import_module(f"ascriptor.{device}").kernel()(flash_attn_full_pj_half_block32_causal_kernel)

# ----------------------------------------------------------------------------------------------------
# v2.py
# ----------------------------------------------------------------------------------------------------

# RFC-0009 batch-1 rewrite: the five hand-written events are gone. The row-state double buffer
# is a Buff beat (core_task), the next-tile rowmax/rowsum prefetch travels UB -> UB instead of
# a GM round-trip (the old MTE3 -> MTE2 same-side GM hand-off was guarded only by a hand event;
# autosync_gm is off, so the GM leg had no analysable guard at all), and the accum/writeback
# flushes are autosync's. The three workspace rings stay on split_workspace + the two global
# stage counters: the prefetch branch interleaves TWO tiles' production streams into one ring,
# so the slot algebra is not the linear one-beat form the gmbuff pass checks (RFC-0009 s5) -
# their safety remains the mutex-depth trust layer, as in the L2 kernels.













def flash_attn_full_pj_half_block32_causal_v2_kernel(
    q: GM[f16, ('TQ', 'D')], k: GM[f16, ('TK', 'D')], v: GM[f16, ('TK', 'D')], out: GM[f32, ('TQ', 'D')], rowmax: GM[f32, ('TST',)], rowsum: GM[f32, ('TST',)],
    S1: i32, S2: i32, D: i32, BH: i32, scale: f32,
):
    score_ws = split_workspace(DT.float, [GetCubeNum(), 3, TILE_M, TILE_N], name="score_ws")
    p_ws = split_workspace(DT.half, [GetCubeNum(), 4, TILE_M, TILE_N], name="p_ws")
    pv_ws = split_workspace(DT.float, [GetCubeNum(), 3, TILE_M, TILE_K], name="pv_ws")

    l1q = DBuff(DT.half, [TILE_M, TILE_K], Position.L1)
    l1k = DBuff(DT.half, [TILE_N, TILE_K], Position.L1)
    l1p = DBuff(DT.half, [TILE_M, TILE_N], Position.L1)
    l1v = DBuff(DT.half, [TILE_N, TILE_K], Position.L1)
    l0c = DBuff(DT.float, [TILE_M, TILE_N], Position.L0C)

    ub_score = DBuff(DT.float, [HALF_M, TILE_N], Position.UB)
    ub_pv = Tensor(DT.float, [HALF_M, TILE_K], Position.UB)
    ub_tmp = Tensor(DT.float, [HALF_M, HALF_N], Position.UB)
    ub_max_s = Tensor(DT.float, [1, HALF_M], Position.UB)
    ub_rmax_s = DBuff(DT.float, [1, HALF_M], Position.UB)
    ub_rmax_prefetch = Tensor(DT.float, [1, HALF_M], Position.UB)
    ub_sum_s = Tensor(DT.float, [1, HALF_M], Position.UB)
    ub_rsum_s = DBuff(DT.float, [1, HALF_M], Position.UB)
    ub_rsum_prefetch = Tensor(DT.float, [1, HALF_M], Position.UB)
    ub_zero_s = Tensor(DT.float, [1, HALF_M], Position.UB)
    ub_max = Tensor(DT.float, [HALF_M, 8], Position.UB)
    ub_rowsum = Tensor(DT.float, [HALF_M, 8], Position.UB)
    ub_p = DBuff(DT.half, [HALF_M, TILE_N], Position.UB)
    ub_expdiff = Tensor(DT.float, [HALF_M, 8], Position.UB)
    accum_ub = Tensor(DT.float, [HALF_M, TILE_K], Position.UB)
    expdiff_buf = QBuff(DT.float, [1, HALF_M], Position.UB)

    qk_mutex = CvMutex(0, depth=3, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)
    p_mutex = VcMutex(1, depth=4, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)
    pv_mutex = CvMutex(2, depth=3, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)

    l1qk_cnt = Var(0)
    l1pv_cnt = Var(0)
    l0c_cnt = Var(0)
    stage1_cnt = Var(0)
    stage2_cnt = Var(0)

    cube_idx = GetCubeIdx()
    sb = GetSubBlockIdx()
    sb_row = Var(sb * HALF_M)

    tiles_m = CeilDiv(S1, TILE_M)
    tiles_n = CeilDiv(S2, TILE_N)
    core_count = GetCubeNum()
    total_m = Var(BH * tiles_m)
    per_core = CeilDiv(total_m, core_count)

    for core_task in range(0, per_core):
        logical_m = Var(core_task * core_count + cube_idx)
        if logical_m < total_m:
            bh = Var(logical_m // tiles_m)
            lmt = Var(logical_m % tiles_m)
            if var_mod(bh, 2) == 1:
                lmt <<= tiles_m - 1 - lmt
            q_row = Var(bh * S1 + lmt * TILE_M)
            kv_base = Var(bh * S2)
            valid_m = Min(TILE_M, S1 - lmt * TILE_M)
            local_valid_m = Min(HALF_M, Max(valid_m - sb_row, 0))
            active_tiles_n = Min(tiles_n, lmt + 1)
            ub_rmax_tile = ub_rmax_s[core_task]
            ub_rsum_tile = ub_rsum_s[core_task]
            prefetched_p = Var(0)
            if core_task > 0:
                prefetched_p <<= Min(active_tiles_n, 2)

            next_logical_m = Var((core_task + 1) * core_count + cube_idx)
            next_lmt = Var(next_logical_m % tiles_m)
            if var_mod(next_logical_m // tiles_m, 2) == 1:
                next_lmt <<= tiles_m - 1 - next_lmt
            next_q_row = Var((next_logical_m // tiles_m) * S1 + next_lmt * TILE_M)
            next_kv_base = Var((next_logical_m // tiles_m) * S2)
            next_valid_m = Min(TILE_M, S1 - next_lmt * TILE_M)
            next_local_valid_m = Min(HALF_M, Max(next_valid_m - sb_row, 0))
            next_active_tiles_n = Min(tiles_n, next_lmt + 1)

            # Row state slots are reused every other M tile; autosync's loop-carried
            # machinery guards the reuse. The prefetched next-tile row state arrives
            # directly from ub_*_prefetch (same V pipe, program-ordered).
            with auto_sync():
                dup(ub_zero_s, 0.0)
                if prefetched_p > 0:
                    add(ub_rmax_tile, ub_rmax_prefetch, ub_zero_s)
                    add(ub_rsum_tile, ub_rsum_prefetch, ub_zero_s)
                else:
                    dup(ub_rmax_tile, NEG_LARGE)
                    dup(ub_rsum_tile, 0.0)

            for ni in range(0, active_tiles_n + 3):
                if ni < active_tiles_n:
                    if ni >= prefetched_p:
                        with auto_sync():
                            n_off = Var(ni * TILE_N)
                            kv_row = Var(kv_base + n_off)
                            valid_n = Min(TILE_N, S2 - n_off)
                            score_slot = var_mod(stage1_cnt, 3)
                            p_slot = var_mod(stage1_cnt, 4)
                            ub_score_tile = ub_score[stage1_cnt]
                            ub_p_tile = ub_p[stage1_cnt]

                            l1q[l1qk_cnt] <<= q[q_row:q_row + valid_m, 0:D]
                            l1k[l1qk_cnt] <<= k[kv_row:kv_row + valid_n, 0:D]
                            matmul(l0c[l0c_cnt], l1q[l1qk_cnt], l1k[l1qk_cnt], is_init=True)

                            qk_mutex.lock()
                            score_ws[cube_idx, score_slot, 0:TILE_M, 0:TILE_N] <<= l0c[l0c_cnt]
                            qk_mutex.ready()

                            qk_mutex.wait()
                            ub_score_tile <<= score_ws[cube_idx, score_slot, sb_row:sb_row + HALF_M, 0:TILE_N]

                            muls(ub_score_tile, ub_score_tile, scale)
                            if ni == lmt:
                                apply_diagonal_tile_block32_causal_mask(ub_score_tile, sb_row)
                            if valid_n < TILE_N:
                                apply_score_tail_mask(ub_score_tile, valid_n)
                            vmax(ub_tmp, ub_score_tile[0:HALF_M, 0:HALF_N], ub_score_tile[0:HALF_M, HALF_N:TILE_N])
                            cmax(ub_max_s, ub_tmp)

                            add(expdiff_buf[p_slot], ub_rmax_tile, ub_zero_s)
                            vmax(ub_rmax_tile, ub_rmax_tile, ub_max_s)
                            sub(expdiff_buf[p_slot], expdiff_buf[p_slot], ub_rmax_tile)
                            exp(expdiff_buf[p_slot], expdiff_buf[p_slot])

                            brcb(ub_max, ub_rmax_tile, repeat=HALF_M // 8, dst_blk_stride=1, dst_rep_stride=8)
                            sub(ub_score_tile[0:HALF_M, 0:HALF_N], ub_score_tile[0:HALF_M, 0:HALF_N], ub_max)
                            sub(ub_score_tile[0:HALF_M, HALF_N:TILE_N], ub_score_tile[0:HALF_M, HALF_N:TILE_N], ub_max)
                            if local_valid_m < HALF_M:
                                apply_score_row_tail_mask_after_shift(ub_score_tile, local_valid_m)

                            exp(ub_score_tile, ub_score_tile)
                            add(ub_tmp, ub_score_tile[0:HALF_M, 0:HALF_N], ub_score_tile[0:HALF_M, HALF_N:TILE_N])
                            cadd(ub_sum_s, ub_tmp)
                            mul(ub_rsum_tile, ub_rsum_tile, expdiff_buf[p_slot])
                            add(ub_rsum_tile, ub_rsum_tile, ub_sum_s)

                            cast(ub_p_tile, ub_score_tile)

                            p_mutex.lock()
                            p_ws[cube_idx, p_slot, sb_row:sb_row + HALF_M, 0:TILE_N] <<= ub_p_tile
                            p_mutex.ready()
                            qk_mutex.free()

                            l1qk_cnt += 1
                            l0c_cnt += 1
                            stage1_cnt += 1

                if ni > 2:
                    prev_nt = Var(ni - 3)
                    prev_n_off = Var(prev_nt * TILE_N)
                    prev_valid_n = Min(TILE_N, S2 - prev_n_off)
                    p_slot = var_mod(stage2_cnt, 4)
                    pv_slot = var_mod(stage2_cnt, 3)
                    v_row = Var(kv_base + prev_n_off)

                    if prev_nt == 0:
                        with auto_sync():
                            dup(accum_ub, 0.0)

                    with auto_sync():
                        p_mutex.wait()
                        l1p[l1pv_cnt] <<= p_ws[cube_idx, p_slot, 0:TILE_M, 0:TILE_N]
                        l1v[l1pv_cnt] <<= v[v_row:v_row + prev_valid_n, 0:D]
                        matmul(l0c[l0c_cnt], l1p[l1pv_cnt], l1v[l1pv_cnt].T, k=prev_valid_n, is_init=True)
                        p_mutex.free()

                        pv_mutex.lock()
                        pv_ws[cube_idx, pv_slot, 0:TILE_M, 0:D] <<= l0c[l0c_cnt]
                        pv_mutex.ready()

                        pv_mutex.wait()
                        ub_pv <<= pv_ws[cube_idx, pv_slot, sb_row:sb_row + HALF_M, 0:D]

                        brcb(ub_expdiff, expdiff_buf[p_slot], repeat=HALF_M // 8, dst_blk_stride=1, dst_rep_stride=8)
                        mul(accum_ub[0:HALF_M, 0:HALF_N], accum_ub[0:HALF_M, 0:HALF_N], ub_expdiff)
                        mul(accum_ub[0:HALF_M, HALF_N:TILE_K], accum_ub[0:HALF_M, HALF_N:TILE_K], ub_expdiff)
                        add(accum_ub, accum_ub, ub_pv)
                        pv_mutex.free()

                        l1pv_cnt += 1
                        l0c_cnt += 1
                        stage2_cnt += 1

                    if prev_nt == active_tiles_n - 1:
                        with auto_sync():
                            brcb(ub_rowsum, ub_rsum_tile, repeat=HALF_M // 8, dst_blk_stride=1, dst_rep_stride=8)
                            div(accum_ub[0:HALF_M, 0:HALF_N], accum_ub[0:HALF_M, 0:HALF_N], ub_rowsum)
                            div(accum_ub[0:HALF_M, HALF_N:TILE_K], accum_ub[0:HALF_M, HALF_N:TILE_K], ub_rowsum)

                            if local_valid_m > 0:
                                out_row = Var(q_row + sb_row)
                                stats_row = Var((out_row // S1) * CeilDiv(S1, 8) * 8 + out_row % S1)
                                out[out_row:out_row + local_valid_m, 0:D] <<= accum_ub[0:local_valid_m, 0:D]
                                rowmax[stats_row:stats_row + local_valid_m] <<= ub_rmax_tile[0:1, 0:local_valid_m]
                                rowsum[stats_row:stats_row + local_valid_m] <<= ub_rsum_tile[0:1, 0:local_valid_m]

                if ni >= active_tiles_n:
                    if next_logical_m < total_m:
                        next_nt = Var(ni - active_tiles_n)
                        next_prefetch_tiles = Min(next_active_tiles_n, 2)
                        if next_nt < next_prefetch_tiles:
                            if next_nt == 0:
                                with auto_sync():
                                    dup(ub_rmax_prefetch, NEG_LARGE)
                                    dup(ub_rsum_prefetch, 0.0)

                            with auto_sync():
                                next_n_off = Var(next_nt * TILE_N)
                                next_kv_row = Var(next_kv_base + next_n_off)
                                next_valid_n = Min(TILE_N, S2 - next_n_off)
                                next_score_slot = var_mod(stage1_cnt, 3)
                                next_p_slot = var_mod(stage1_cnt, 4)
                                next_ub_score_tile = ub_score[stage1_cnt]
                                next_ub_p_tile = ub_p[stage1_cnt]

                                l1q[l1qk_cnt] <<= q[next_q_row:next_q_row + next_valid_m, 0:D]
                                l1k[l1qk_cnt] <<= k[next_kv_row:next_kv_row + next_valid_n, 0:D]
                                matmul(l0c[l0c_cnt], l1q[l1qk_cnt], l1k[l1qk_cnt], is_init=True)

                                qk_mutex.lock()
                                score_ws[cube_idx, next_score_slot, 0:TILE_M, 0:TILE_N] <<= l0c[l0c_cnt]
                                qk_mutex.ready()

                                qk_mutex.wait()
                                next_ub_score_tile <<= score_ws[
                                    cube_idx, next_score_slot, sb_row:sb_row + HALF_M, 0:TILE_N
                                ]

                                muls(next_ub_score_tile, next_ub_score_tile, scale)
                                if next_nt == next_lmt:
                                    apply_diagonal_tile_block32_causal_mask(next_ub_score_tile, sb_row)
                                if next_valid_n < TILE_N:
                                    apply_score_tail_mask(next_ub_score_tile, next_valid_n)
                                vmax(
                                    ub_tmp,
                                    next_ub_score_tile[0:HALF_M, 0:HALF_N],
                                    next_ub_score_tile[0:HALF_M, HALF_N:TILE_N],
                                )
                                cmax(ub_max_s, ub_tmp)

                                add(expdiff_buf[next_p_slot], ub_rmax_prefetch, ub_zero_s)
                                vmax(ub_rmax_prefetch, ub_rmax_prefetch, ub_max_s)
                                sub(expdiff_buf[next_p_slot], expdiff_buf[next_p_slot], ub_rmax_prefetch)
                                exp(expdiff_buf[next_p_slot], expdiff_buf[next_p_slot])

                                brcb(
                                    ub_max, ub_rmax_prefetch,
                                    repeat=HALF_M // 8, dst_blk_stride=1, dst_rep_stride=8,
                                )
                                sub(
                                    next_ub_score_tile[0:HALF_M, 0:HALF_N],
                                    next_ub_score_tile[0:HALF_M, 0:HALF_N],
                                    ub_max,
                                )
                                sub(
                                    next_ub_score_tile[0:HALF_M, HALF_N:TILE_N],
                                    next_ub_score_tile[0:HALF_M, HALF_N:TILE_N],
                                    ub_max,
                                )
                                if next_local_valid_m < HALF_M:
                                    apply_score_row_tail_mask_after_shift(next_ub_score_tile, next_local_valid_m)

                                exp(next_ub_score_tile, next_ub_score_tile)
                                add(
                                    ub_tmp,
                                    next_ub_score_tile[0:HALF_M, 0:HALF_N],
                                    next_ub_score_tile[0:HALF_M, HALF_N:TILE_N],
                                )
                                cadd(ub_sum_s, ub_tmp)
                                mul(ub_rsum_prefetch, ub_rsum_prefetch, expdiff_buf[next_p_slot])
                                add(ub_rsum_prefetch, ub_rsum_prefetch, ub_sum_s)

                                cast(next_ub_p_tile, next_ub_score_tile)

                                p_mutex.lock()
                                p_ws[cube_idx, next_p_slot, sb_row:sb_row + HALF_M, 0:TILE_N] <<= next_ub_p_tile
                                p_mutex.ready()
                                qk_mutex.free()

                                l1qk_cnt += 1
                                l0c_cnt += 1
                                stage1_cnt += 1

    return out, rowmax, rowsum




@lru_cache(maxsize=2)
def build_v2_kernel(device):
    if device not in ("a2", "a3"):
        raise ValueError("Shared tensor-vector attention supports A2/A3")
    return import_module(f"ascriptor.{device}").kernel()(flash_attn_full_pj_half_block32_causal_v2_kernel)

# ----------------------------------------------------------------------------------------------------
# v3.py
# ----------------------------------------------------------------------------------------------------

# RFC-0009 batch-1 rewrite: the four hand-written events are gone (autosync's loop-carried
# machinery guards the l1q[0] preload across M tiles, the accum/rmax/rsum reinitialisation, and
# the V->MTE3 flush), stage1_cnt / stage2_cnt collapse into the loop beat `ni` (producer ni,
# consumer ni - 3), and the workspace rings are GMBuffs checked by the gmbuff pass.













def flash_attn_full_pj_half_block32_causal_v3_kernel(
    q: GM[f16, ('TQ', 'D')], k: GM[f16, ('TK', 'D')], v: GM[f16, ('TK', 'D')], out: GM[f32, ('TQ', 'D')], rowmax: GM[f32, ('TST',)], rowsum: GM[f32, ('TST',)],
    S1: i32, S2: i32, D: i32, BH: i32, scale: f32,
):
    score_ws = GMBuff(DT.float, [TILE_M, TILE_N], slots=4, name="score_ws")
    p_ws = GMBuff(DT.half, [TILE_M, TILE_N], slots=4, name="p_ws")
    pv_ws = GMBuff(DT.float, [TILE_M, TILE_K], slots=4, name="pv_ws")

    l1q = DBuff(DT.half, [TILE_M, TILE_K], Position.L1)
    l1k = DBuff(DT.half, [TILE_N, TILE_K], Position.L1)
    l1p = DBuff(DT.half, [TILE_M, TILE_N], Position.L1)
    l1v = DBuff(DT.half, [TILE_N, TILE_K], Position.L1)
    l0c = DBuff(DT.float, [TILE_M, TILE_N], Position.L0C)

    ub_score = DBuff(DT.float, [HALF_M, TILE_N], Position.UB)
    ub_pv = Tensor(DT.float, [HALF_M, TILE_K], Position.UB)
    ub_tmp = Tensor(DT.float, [HALF_M, HALF_N], Position.UB)
    ub_max_s = Tensor(DT.float, [1, HALF_M], Position.UB)
    ub_rmax_s = Tensor(DT.float, [1, HALF_M], Position.UB)
    ub_sum_s = Tensor(DT.float, [1, HALF_M], Position.UB)
    ub_rsum_s = Tensor(DT.float, [1, HALF_M], Position.UB)
    ub_zero_s = Tensor(DT.float, [1, HALF_M], Position.UB)
    ub_max = Tensor(DT.float, [HALF_M, 8], Position.UB)
    ub_rowsum = Tensor(DT.float, [HALF_M, 8], Position.UB)
    ub_p = DBuff(DT.half, [HALF_M, TILE_N], Position.UB)
    ub_expdiff = Tensor(DT.float, [HALF_M, 8], Position.UB)
    accum_ub = Tensor(DT.float, [HALF_M, TILE_K], Position.UB)
    expdiff_buf = QBuff(DT.float, [1, HALF_M], Position.UB)

    qk_mutex = CvMutex(0, depth=4, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)
    p_mutex = VcMutex(1, depth=4, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)
    pv_mutex = CvMutex(2, depth=4, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)

    l1k_cnt = Var(0)
    l1pv_cnt = Var(0)
    l0c_cnt = Var(0)

    cube_idx = GetCubeIdx()
    sb = GetSubBlockIdx()
    sb_row = Var(sb * HALF_M)

    tiles_m = CeilDiv(S1, TILE_M)
    tiles_n = CeilDiv(S2, TILE_N)
    core_count = GetCubeNum()
    core_step = Var(1, DT.int)
    core_step <<= core_count
    total_m = Var(BH * tiles_m)

    for gmt in range(cube_idx, total_m, core_step):
        bh = Var(gmt // tiles_m)
        lmt = Var(gmt % tiles_m)
        if var_mod(bh, 2) == 1:
            lmt <<= tiles_m - 1 - lmt
        q_row = Var(bh * S1 + lmt * TILE_M)
        kv_base = Var(bh * S2)
        valid_m = Min(TILE_M, S1 - lmt * TILE_M)
        local_valid_m = Min(HALF_M, Max(valid_m - sb_row, 0))
        active_tiles_n = Min(tiles_n, lmt + 1)

        # The previous MTE3 writeback still reads accum_ub, ub_rmax_s, and ub_rsum_s, and the
        # previous tile's QK matmuls still read l1q[0]; autosync's loop-carried machinery guards
        # both reinitialisations.
        with auto_sync():
            dup(ub_rmax_s, NEG_LARGE)
            dup(ub_rsum_s, 0.0)
            dup(ub_zero_s, 0.0)
            dup(accum_ub, 0.0)

        with auto_sync():
            l1q[0] <<= q[q_row:q_row + valid_m, 0:D]

        for ni in range(0, active_tiles_n + 3):
            if ni < active_tiles_n:
                with auto_sync():
                    n_off = Var(ni * TILE_N)
                    kv_row = Var(kv_base + n_off)
                    valid_n = Min(TILE_N, S2 - n_off)
                    ub_score_tile = ub_score[ni]
                    ub_p_tile = ub_p[ni]

                    l1k[l1k_cnt] <<= k[kv_row:kv_row + valid_n, 0:D]
                    matmul(l0c[l0c_cnt], l1q[0], l1k[l1k_cnt], is_init=True)

                    qk_mutex.lock()
                    score_ws[ni, 0:TILE_M, 0:TILE_N] <<= l0c[l0c_cnt]
                    qk_mutex.ready()

                    qk_mutex.wait()
                    ub_score_tile <<= score_ws[ni, sb_row:sb_row + HALF_M, 0:TILE_N]

                    muls(ub_score_tile, ub_score_tile, scale)
                    if ni == lmt:
                        apply_diagonal_tile_block32_causal_mask(ub_score_tile, sb_row)
                    if valid_n < TILE_N:
                        apply_score_tail_mask(ub_score_tile, valid_n)
                    vmax(ub_tmp, ub_score_tile[0:HALF_M, 0:HALF_N], ub_score_tile[0:HALF_M, HALF_N:TILE_N])
                    cmax(ub_max_s, ub_tmp)

                    add(expdiff_buf[ni], ub_rmax_s, ub_zero_s)
                    vmax(ub_rmax_s, ub_rmax_s, ub_max_s)
                    sub(expdiff_buf[ni], expdiff_buf[ni], ub_rmax_s)
                    exp(expdiff_buf[ni], expdiff_buf[ni])

                    brcb(ub_max, ub_rmax_s, repeat=HALF_M // 8, dst_blk_stride=1, dst_rep_stride=8)
                    sub(ub_score_tile[0:HALF_M, 0:HALF_N], ub_score_tile[0:HALF_M, 0:HALF_N], ub_max)
                    sub(ub_score_tile[0:HALF_M, HALF_N:TILE_N], ub_score_tile[0:HALF_M, HALF_N:TILE_N], ub_max)
                    if local_valid_m < HALF_M:
                        apply_score_row_tail_mask_after_shift(ub_score_tile, local_valid_m)

                    exp(ub_score_tile, ub_score_tile)
                    add(ub_tmp, ub_score_tile[0:HALF_M, 0:HALF_N], ub_score_tile[0:HALF_M, HALF_N:TILE_N])
                    cadd(ub_sum_s, ub_tmp)
                    mul(ub_rsum_s, ub_rsum_s, expdiff_buf[ni])
                    add(ub_rsum_s, ub_rsum_s, ub_sum_s)

                    cast(ub_p_tile, ub_score_tile)

                    p_mutex.lock()
                    p_ws[ni, sb_row:sb_row + HALF_M, 0:TILE_N] <<= ub_p_tile
                    p_mutex.ready()
                    qk_mutex.free()

                    l1k_cnt += 1
                    l0c_cnt += 1

            if ni > 2:
                with auto_sync():
                    prev_nt = Var(ni - 3)
                    prev_n_off = Var(prev_nt * TILE_N)
                    prev_valid_n = Min(TILE_N, S2 - prev_n_off)
                    v_row = Var(kv_base + prev_n_off)

                    p_mutex.wait()
                    l1p[l1pv_cnt] <<= p_ws[ni - 3, 0:TILE_M, 0:TILE_N]
                    l1v[l1pv_cnt] <<= v[v_row:v_row + prev_valid_n, 0:D]
                    matmul(l0c[l0c_cnt], l1p[l1pv_cnt], l1v[l1pv_cnt].T, k=prev_valid_n, is_init=True)
                    p_mutex.free()

                    pv_mutex.lock()
                    pv_ws[ni - 3, 0:TILE_M, 0:D] <<= l0c[l0c_cnt]
                    pv_mutex.ready()

                    pv_mutex.wait()
                    ub_pv <<= pv_ws[ni - 3, sb_row:sb_row + HALF_M, 0:D]

                    brcb(ub_expdiff, expdiff_buf[ni - 3], repeat=HALF_M // 8, dst_blk_stride=1, dst_rep_stride=8)
                    mul(accum_ub[0:HALF_M, 0:HALF_N], accum_ub[0:HALF_M, 0:HALF_N], ub_expdiff)
                    mul(accum_ub[0:HALF_M, HALF_N:TILE_K], accum_ub[0:HALF_M, HALF_N:TILE_K], ub_expdiff)
                    add(accum_ub, accum_ub, ub_pv)
                    pv_mutex.free()

                    l1pv_cnt += 1
                    l0c_cnt += 1

        with auto_sync():
            brcb(ub_rowsum, ub_rsum_s, repeat=HALF_M // 8, dst_blk_stride=1, dst_rep_stride=8)
            div(accum_ub[0:HALF_M, 0:HALF_N], accum_ub[0:HALF_M, 0:HALF_N], ub_rowsum)
            div(accum_ub[0:HALF_M, HALF_N:TILE_K], accum_ub[0:HALF_M, HALF_N:TILE_K], ub_rowsum)

            if local_valid_m > 0:
                out_row = Var(q_row + sb_row)
                stats_row = Var((out_row // S1) * CeilDiv(S1, 8) * 8 + out_row % S1)
                out[out_row:out_row + local_valid_m, 0:D] <<= accum_ub[0:local_valid_m, 0:D]
                rowmax[stats_row:stats_row + local_valid_m] <<= ub_rmax_s[0:1, 0:local_valid_m]
                rowsum[stats_row:stats_row + local_valid_m] <<= ub_rsum_s[0:1, 0:local_valid_m]

    return out, rowmax, rowsum




@lru_cache(maxsize=2)
def build_v3_kernel(device):
    if device not in ("a2", "a3"):
        raise ValueError("Shared tensor-vector attention supports A2/A3")
    return import_module(f"ascriptor.{device}").kernel()(flash_attn_full_pj_half_block32_causal_v3_kernel)

# ----------------------------------------------------------------------------------------------------
# v4.py
# ----------------------------------------------------------------------------------------------------

GROUP_N = 4
V4_GROUP_STAGE_SLOTS = 4
V4_GROUP_LOOKAHEAD = 2
ROW_CHUNK = 32








@func()
def apply_score_tail_mask_chunk(ub_score: Tensor, valid_n: Var):
    left_valid = Min(valid_n, HALF_N)
    right_valid = Max(valid_n - HALF_N, 0)
    mask_score_half_suffix_invalid(ub_score[0:ROW_CHUNK, 0:HALF_N], left_valid)
    mask_score_half_suffix_invalid(ub_score[0:ROW_CHUNK, HALF_N:TILE_N], right_valid)




@func()
def apply_score_row_tail_mask_chunk_after_shift(ub_score: Tensor, valid_rows: Var):
    if valid_rows == 0:
        dup(ub_score, NEG_LARGE)
    elif valid_rows < ROW_CHUNK:
        dup(ub_score[valid_rows:ROW_CHUNK, 0:TILE_N], NEG_LARGE)




def flash_attn_full_pj_half_block32_causal_v4_kernel(
    q: GM[f16, ('TQ', 'D')], k: GM[f16, ('TK', 'D')], v: GM[f16, ('TK', 'D')], out: GM[f32, ('TQ', 'D')], rowmax: GM[f32, ('TST',)], rowsum: GM[f32, ('TST',)],
    S1: i32, S2: i32, D: i32, BH: i32, scale: f32,
):
    score_ws = GMBuff(DT.float, [TILE_M, GROUP_N * TILE_N], slots=V4_GROUP_STAGE_SLOTS, name="score_ws")
    p_ws = GMBuff(DT.half, [TILE_M, GROUP_N * TILE_N], slots=V4_GROUP_STAGE_SLOTS, name="p_ws")
    pv_ws = GMBuff(DT.float, [TILE_M, TILE_K], slots=V4_GROUP_STAGE_SLOTS, name="pv_ws")

    l1q = DBuff(DT.half, [TILE_M, TILE_K], Position.L1)
    l1k = DBuff(DT.half, [TILE_N, TILE_K], Position.L1)
    l1p = DBuff(DT.half, [TILE_M, TILE_N], Position.L1)
    l1v = DBuff(DT.half, [TILE_N, TILE_K], Position.L1)
    l0c = DBuff(DT.float, [TILE_M, TILE_N], Position.L0C)

    ub_score_group = QBuff(DT.float, [ROW_CHUNK, TILE_N], Position.UB)
    ub_p_group = QBuff(DT.half, [ROW_CHUNK, TILE_N], Position.UB)
    ub_pv = Tensor(DT.float, [HALF_M, TILE_K], Position.UB)
    ub_tmp = Tensor(DT.float, [ROW_CHUNK, HALF_N], Position.UB)
    ub_max_s = Tensor(DT.float, [1, ROW_CHUNK], Position.UB)
    ub_group_max_s = Tensor(DT.float, [1, ROW_CHUNK], Position.UB)
    ub_old_max_s = Tensor(DT.float, [1, ROW_CHUNK], Position.UB)
    ub_rmax_s = Tensor(DT.float, [1, HALF_M], Position.UB)
    ub_sum_s = Tensor(DT.float, [1, ROW_CHUNK], Position.UB)
    ub_group_sum_s = Tensor(DT.float, [1, ROW_CHUNK], Position.UB)
    ub_rsum_s = Tensor(DT.float, [1, HALF_M], Position.UB)
    ub_zero_s = Tensor(DT.float, [1, HALF_M], Position.UB)
    ub_zero_chunk_s = Tensor(DT.float, [1, ROW_CHUNK], Position.UB)
    ub_max = Tensor(DT.float, [ROW_CHUNK, 8], Position.UB)
    ub_rowsum = Tensor(DT.float, [HALF_M, 8], Position.UB)
    ub_expdiff = Tensor(DT.float, [HALF_M, 8], Position.UB)
    ub_group_expdiff_chunk_s = Tensor(DT.float, [1, ROW_CHUNK], Position.UB)
    accum_ub = Tensor(DT.float, [HALF_M, TILE_K], Position.UB)
    expdiff_buf = QBuff(DT.float, [1, HALF_M], Position.UB)

    qk_mutex = CvMutex(0, depth=V4_GROUP_STAGE_SLOTS, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)
    p_mutex = VcMutex(1, depth=V4_GROUP_STAGE_SLOTS, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE2)
    pv_mutex = CvMutex(2, depth=V4_GROUP_STAGE_SLOTS, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)

    l1k_cnt = Var(0)
    l1pv_cnt = Var(0)
    l0c_cnt = Var(0)
    q_cur = Var(0)

    cube_idx = GetCubeIdx()
    sb = GetSubBlockIdx()
    sb_row = Var(sb * HALF_M)

    tiles_m = CeilDiv(S1, TILE_M)
    tiles_n = CeilDiv(S2, TILE_N)
    core_count = GetCubeNum()
    core_step = Var(1, DT.int)
    core_step <<= core_count
    total_m = Var(BH * tiles_m)

    for gmt in range(cube_idx, total_m, core_step):
        bh = Var(gmt // tiles_m)
        lmt = Var(gmt % tiles_m)
        if var_mod(bh, 2) == 1:
            lmt <<= tiles_m - 1 - lmt
        q_row = Var(bh * S1 + lmt * TILE_M)
        kv_base = Var(bh * S2)
        valid_m = Min(TILE_M, S1 - lmt * TILE_M)
        local_valid_m = Min(HALF_M, Max(valid_m - sb_row, 0))
        active_tiles_n = Min(tiles_n, lmt + 1)
        active_groups = CeilDiv(active_tiles_n, GROUP_N)

        # The previous MTE3 writeback still reads accum_ub, ub_rmax_s, and ub_rsum_s;
        # autosync's loop-carried machinery guards the reinitialisation.
        with auto_sync():
            dup(ub_rmax_s, NEG_LARGE)
            dup(ub_rsum_s, 0.0)
            dup(ub_zero_s, 0.0)
            dup(ub_zero_chunk_s, 0.0, count=ROW_CHUNK)
            dup(accum_ub, 0.0)

        with auto_sync():
            l1q[q_cur] <<= q[q_row:q_row + valid_m, 0:D]

        for group_id in range(0, active_groups + V4_GROUP_LOOKAHEAD):
            if group_id < active_groups:
                group_start = Var(group_id * GROUP_N)
                group_len = Min(GROUP_N, active_tiles_n - group_start)

                qk_mutex.lock()
                for gi in range(0, GROUP_N):
                    if gi < group_len:
                        with auto_sync():
                            ni = Var(group_start + gi)
                            n_off = Var(ni * TILE_N)
                            kv_row = Var(kv_base + n_off)
                            valid_n = Min(TILE_N, S2 - n_off)

                            l1k[l1k_cnt] <<= k[kv_row:kv_row + valid_n, 0:D]
                            matmul(l0c[l0c_cnt], l1q[q_cur], l1k[l1k_cnt], is_init=True)

                            group_col = gi * TILE_N
                            score_ws[group_id, 0:TILE_M, group_col:group_col + TILE_N] <<= l0c[l0c_cnt]
                            l1k_cnt += 1
                            l0c_cnt += 1
                qk_mutex.ready()
                qk_mutex.wait()
                p_mutex.lock()

                for rb in unroll(0, HALF_M, ROW_CHUNK):
                    chunk_row = Var(sb_row + rb)
                    chunk_valid_m = Min(ROW_CHUNK, Max(valid_m - chunk_row, 0))

                    dup(ub_group_max_s, NEG_LARGE, count=ROW_CHUNK)
                    for gi in range(0, GROUP_N):
                        if gi < group_len:
                            with auto_sync():
                                ni = Var(group_start + gi)
                                n_off = Var(ni * TILE_N)
                                valid_n = Min(TILE_N, S2 - n_off)
                                ub_score = ub_score_group[gi]

                                group_col = gi * TILE_N
                                ub_score <<= score_ws[group_id, chunk_row:chunk_row + ROW_CHUNK, group_col:group_col + TILE_N]

                                muls(ub_score, ub_score, scale)
                                if ni == lmt:
                                    diag_valid_n = Min(valid_n, (chunk_row // BLOCK_CAUSAL + 1) * BLOCK_CAUSAL)
                                    apply_score_tail_mask_chunk(ub_score, diag_valid_n)
                                elif valid_n < TILE_N:
                                    apply_score_tail_mask_chunk(ub_score, valid_n)
                                vmax(ub_tmp, ub_score[0:ROW_CHUNK, 0:HALF_N], ub_score[0:ROW_CHUNK, HALF_N:TILE_N])
                                cmax(ub_max_s, ub_tmp)
                                vmax(ub_group_max_s, ub_group_max_s, ub_max_s, count=ROW_CHUNK)

                    add(ub_old_max_s, ub_rmax_s[0:1, rb:rb + ROW_CHUNK], ub_zero_chunk_s, count=ROW_CHUNK)
                    vmax(
                        ub_rmax_s[0:1, rb:rb + ROW_CHUNK],
                        ub_rmax_s[0:1, rb:rb + ROW_CHUNK],
                        ub_group_max_s,
                        count=ROW_CHUNK,
                    )
                    sub(ub_group_expdiff_chunk_s, ub_old_max_s, ub_rmax_s[0:1, rb:rb + ROW_CHUNK], count=ROW_CHUNK)
                    exp(ub_group_expdiff_chunk_s, ub_group_expdiff_chunk_s, count=ROW_CHUNK)
                    add(
                        expdiff_buf[group_id][0:1, rb:rb + ROW_CHUNK],
                        ub_group_expdiff_chunk_s,
                        ub_zero_chunk_s,
                        count=ROW_CHUNK,
                    )
                    dup(ub_group_sum_s, 0.0, count=ROW_CHUNK)

                    for gi in range(0, GROUP_N):
                        if gi < group_len:
                            with auto_sync():
                                ub_p = ub_p_group[gi]
                                group_col = gi * TILE_N
                                ub_score = ub_score_group[gi]

                                brcb(ub_max, ub_rmax_s[0:1, rb:rb + ROW_CHUNK], repeat=ROW_CHUNK // 8, dst_blk_stride=1, dst_rep_stride=8)
                                sub(ub_score[0:ROW_CHUNK, 0:HALF_N], ub_score[0:ROW_CHUNK, 0:HALF_N], ub_max)
                                sub(ub_score[0:ROW_CHUNK, HALF_N:TILE_N], ub_score[0:ROW_CHUNK, HALF_N:TILE_N], ub_max)
                                if chunk_valid_m < ROW_CHUNK:
                                    apply_score_row_tail_mask_chunk_after_shift(ub_score, chunk_valid_m)

                                exp(ub_score, ub_score)
                                add(ub_tmp, ub_score[0:ROW_CHUNK, 0:HALF_N], ub_score[0:ROW_CHUNK, HALF_N:TILE_N])
                                cadd(ub_sum_s, ub_tmp)
                                add(ub_group_sum_s, ub_group_sum_s, ub_sum_s, count=ROW_CHUNK)

                                cast(ub_p, ub_score)
                                p_ws[group_id, chunk_row:chunk_row + ROW_CHUNK, group_col:group_col + TILE_N] <<= ub_p

                    mul(
                        ub_rsum_s[0:1, rb:rb + ROW_CHUNK],
                        ub_rsum_s[0:1, rb:rb + ROW_CHUNK],
                        ub_group_expdiff_chunk_s,
                        count=ROW_CHUNK,
                    )
                    add(
                        ub_rsum_s[0:1, rb:rb + ROW_CHUNK],
                        ub_rsum_s[0:1, rb:rb + ROW_CHUNK],
                        ub_group_sum_s,
                        count=ROW_CHUNK,
                    )

                qk_mutex.free()
                p_mutex.ready()

            if group_id >= V4_GROUP_LOOKAHEAD:
                with cube_scope():
                    with auto_sync():
                        p_mutex.wait()
                        for gi in range(0, GROUP_N):
                            if (group_id - V4_GROUP_LOOKAHEAD) * GROUP_N + gi < active_tiles_n:
                                ni = Var((group_id - V4_GROUP_LOOKAHEAD) * GROUP_N + gi)
                                n_off = Var(ni * TILE_N)
                                valid_n = Min(TILE_N, S2 - n_off)
                                v_row = Var(kv_base + n_off)

                                group_col = gi * TILE_N
                                l1p[l1pv_cnt] <<= p_ws[group_id - V4_GROUP_LOOKAHEAD, 0:TILE_M, group_col:group_col + TILE_N]
                                l1v[l1pv_cnt] <<= v[v_row:v_row + valid_n, 0:D]
                                if gi == 0:
                                    matmul(l0c[l0c_cnt], l1p[l1pv_cnt], l1v[l1pv_cnt].T, k=valid_n, is_init=True)
                                else:
                                    matmul(l0c[l0c_cnt], l1p[l1pv_cnt], l1v[l1pv_cnt].T, k=valid_n, is_init=False)
                                l1pv_cnt += 1
                        p_mutex.free()

                        pv_mutex.lock()
                        pv_ws[group_id - V4_GROUP_LOOKAHEAD, 0:TILE_M, 0:D] <<= l0c[l0c_cnt]
                        pv_mutex.ready()
                        l0c_cnt += 1

                with auto_sync():
                    pv_mutex.wait()
                    ub_pv <<= pv_ws[group_id - V4_GROUP_LOOKAHEAD, sb_row:sb_row + HALF_M, 0:D]
                    pv_mutex.free()

                    brcb(ub_expdiff, expdiff_buf[group_id - V4_GROUP_LOOKAHEAD], repeat=HALF_M // 8, dst_blk_stride=1, dst_rep_stride=8)
                    mul(accum_ub[0:HALF_M, 0:HALF_N], accum_ub[0:HALF_M, 0:HALF_N], ub_expdiff)
                    mul(accum_ub[0:HALF_M, HALF_N:TILE_K], accum_ub[0:HALF_M, HALF_N:TILE_K], ub_expdiff)
                    add(accum_ub, accum_ub, ub_pv)

        with auto_sync():
            brcb(ub_rowsum, ub_rsum_s, repeat=HALF_M // 8, dst_blk_stride=1, dst_rep_stride=8)
            div(accum_ub[0:HALF_M, 0:HALF_N], accum_ub[0:HALF_M, 0:HALF_N], ub_rowsum)
            div(accum_ub[0:HALF_M, HALF_N:TILE_K], accum_ub[0:HALF_M, HALF_N:TILE_K], ub_rowsum)

            if local_valid_m > 0:
                out_row = Var(q_row + sb_row)
                stats_row = Var((out_row // S1) * CeilDiv(S1, 8) * 8 + out_row % S1)
                out[out_row:out_row + local_valid_m, 0:D] <<= accum_ub[0:local_valid_m, 0:D]
                rowmax[stats_row:stats_row + local_valid_m] <<= ub_rmax_s[0:1, 0:local_valid_m]
                rowsum[stats_row:stats_row + local_valid_m] <<= ub_rsum_s[0:1, 0:local_valid_m]
        q_cur += 1

    return out, rowmax, rowsum




@lru_cache(maxsize=2)
def build_v4_kernel(device):
    if device not in ("a2", "a3"):
        raise ValueError("Shared tensor-vector attention supports A2/A3")
    return import_module(f"ascriptor.{device}").kernel()(flash_attn_full_pj_half_block32_causal_v4_kernel)

# ----------------------------------------------------------------------------------------------------
# v5.py
# ----------------------------------------------------------------------------------------------------

# RFC-0009 batch-2 rewrite: the five hand-written events are gone (autosync guards the l1q
# preload, the group QBuffs, the accum/rmax/rsum reinitialisation and the store flushes), the
# stage1_cnt/stage2_cnt pair collapses into the tile-local beat `group_id` (producer group_id,
# consumer group_id - GROUP_LOOKAHEAD; the expdiff_store slot rows keep the same arithmetic),
# and the workspace rings are GMBuffs (the GROUP_N slot dim row-flattened) checked by the
# gmbuff pass. The bare GMTensor signature is modernised - this kernel (whose header comment
# named the ring invariant GMBuff machine-checks) had never compiled in ascriptor before.

# Stage-1 (QK->P) runs GROUP_LOOKAHEAD groups ahead of stage-2 (P@V). The shared
# GM workspaces (score_ws / p_ws / pv_ws) and the cross-side mutex depths form a
# ring; the producer overwrites slot (g % GROUP_STAGE_SLOTS) while the consumer
# reads slot ((g - GROUP_LOOKAHEAD) % GROUP_STAGE_SLOTS). To guarantee they never
# alias on real hardware the ring MUST be strictly larger than the lookahead.
# Both current simulators model mutex depth and slot reuse; ring-wrapping
# cases are necessary to exercise this invariant.
# Lookahead 3 / 5 slots measured marginally faster than 4 / 6 on HW and uses
# less GM workspace, at the same safety margin (slots - lookahead = 2).
GROUP_LOOKAHEAD = 3
GROUP_STAGE_SLOTS = 5
# Number of TILE_N V tiles held in the L1 preload ring. The V GM->L1 loads are
# issued before p_mutex.wait() so they overlap the wait for P; the ring must be
# large enough that an early V load never aliases a tile the PV matmul has not
# consumed yet (the simulator models this ring and the mutex depth).
V_PRELOAD_SLOTS = 8
ROW_CHUNK_VEC = 64
















def flash_attn_full_pj_half_block32_causal_v5_kernel(
    q: GM[f16, ('TQ', 'D')], k: GM[f16, ('TK', 'D')], v: GM[f16, ('TK', 'D')], out: GM[f32, ('TQ', 'D')],
    rowmax: GM[f32, ('TST',)], rowsum: GM[f32, ('TST',)],
    S1: i32, S2: i32, D: i32, BH: i32, scale: f32,
):
    score_ws = GMBuff(DT.float, [GROUP_N * TILE_M, TILE_N], slots=GROUP_STAGE_SLOTS, name="score_ws")
    p_ws = GMBuff(DT.half, [GROUP_N * TILE_M, TILE_N], slots=GROUP_STAGE_SLOTS, name="p_ws")
    pv_ws = GMBuff(DT.float, [TILE_M, TILE_K], slots=GROUP_STAGE_SLOTS, name="pv_ws")

    l1q = DBuff(DT.half, [TILE_M, TILE_K], Position.L1)
    l1k = DBuff(DT.half, [TILE_N, TILE_K], Position.L1)
    l1p = DBuff(DT.half, [TILE_M, TILE_N], Position.L1)
    l1v = Tensor(DT.half, [V_PRELOAD_SLOTS * TILE_N, TILE_K], Position.L1)
    l0c = DBuff(DT.float, [TILE_M, TILE_N], Position.L0C)

    ub_score_group = QBuff(DT.float, [ROW_CHUNK, TILE_N], Position.UB)
    ub_p_group = QBuff(DT.half, [ROW_CHUNK, TILE_N], Position.UB)
    ub_pv = Tensor(DT.float, [HALF_M, TILE_K], Position.UB)
    ub_tmp = Tensor(DT.float, [ROW_CHUNK, HALF_N], Position.UB)
    ub_max_s = Tensor(DT.float, [2, ROW_CHUNK_VEC], Position.UB)
    ub_group_max_s = Tensor(DT.float, [2, ROW_CHUNK_VEC], Position.UB)
    ub_old_max_s = Tensor(DT.float, [2, ROW_CHUNK_VEC], Position.UB)
    ub_rmax_s = Tensor(DT.float, [2, ROW_CHUNK_VEC], Position.UB)
    ub_sum_s = Tensor(DT.float, [2, ROW_CHUNK_VEC], Position.UB)
    ub_group_sum_s = Tensor(DT.float, [2, ROW_CHUNK_VEC], Position.UB)
    ub_rsum_s = Tensor(DT.float, [2, ROW_CHUNK_VEC], Position.UB)
    ub_max = Tensor(DT.float, [ROW_CHUNK, 8], Position.UB)
    ub_brcb_max = Tensor(DT.float, [ROW_CHUNK, 8], Position.UB)
    ub_rowsum = Tensor(DT.float, [HALF_M, 8], Position.UB)
    ub_expdiff = Tensor(DT.float, [HALF_M, 8], Position.UB)
    accum_ub = Tensor(DT.float, [HALF_M, TILE_K], Position.UB)
    # expdiff bridges stage-1 -> stage-2 and must hold GROUP_LOOKAHEAD+1 live
    # values, one [2, ROW_CHUNK_VEC] tile per workspace slot. A QBuff is hard
    # capped at 4 slots, so it is replaced with an explicit slot-indexed Tensor:
    # row (slot * 2 + rb_slot) stores the two 32-row chunks of slot `slot`.
    expdiff_store = Tensor(DT.float, [GROUP_STAGE_SLOTS * 2, ROW_CHUNK_VEC], Position.UB)

    qk_mutex = CvMutex(0, depth=GROUP_STAGE_SLOTS, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)
    p_mutex = VcMutex(1, depth=GROUP_STAGE_SLOTS, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE2)
    pv_mutex = CvMutex(2, depth=GROUP_STAGE_SLOTS, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)

    l1k_cnt = Var(0)
    l1pv_cnt = Var(0)
    l0c_cnt = Var(0)
    q_cur = Var(0)

    cube_idx = GetCubeIdx()
    sb = GetSubBlockIdx()
    sb_row = Var(sb * HALF_M)

    tiles_m = CeilDiv(S1, TILE_M)
    tiles_n = CeilDiv(S2, TILE_N)
    core_count = GetCubeNum()
    core_step = Var(1, DT.int)
    core_step <<= core_count
    total_m = Var(BH * tiles_m)

    for gmt in range(cube_idx, total_m, core_step):
        bh = Var(gmt // tiles_m)
        lmt = Var(gmt % tiles_m)
        if var_mod(bh, 2) == 1:
            lmt <<= tiles_m - 1 - lmt
        q_row = Var(bh * S1 + lmt * TILE_M)
        kv_base = Var(bh * S2)
        valid_m = Min(TILE_M, S1 - lmt * TILE_M)
        local_valid_m = Min(HALF_M, Max(valid_m - sb_row, 0))
        active_tiles_n = Min(tiles_n, lmt + 1)
        active_groups = CeilDiv(active_tiles_n, GROUP_N)

        # The previous MTE3 writeback still reads accum_ub, ub_rmax_s, and ub_rsum_s;
        # autosync's loop-carried machinery guards the reinitialisation.
        with auto_sync():
            dup(ub_rmax_s, NEG_LARGE)
            dup(ub_rsum_s, 0.0)
            dup(accum_ub, 0.0)

        with auto_sync():
            l1q[q_cur] <<= q[q_row:q_row + valid_m, 0:D]

        for group_id in range(0, active_groups + GROUP_LOOKAHEAD):
            if group_id < active_groups:
                group_start = Var(group_id * GROUP_N)
                group_len = Min(GROUP_N, active_tiles_n - group_start)
                stage1_slot = var_mod(group_id, GROUP_STAGE_SLOTS)

                for pair_start in range(0, GROUP_N, 2):
                    if pair_start < group_len:
                        qk_mutex.lock()
                        for pair_i in range(0, 2):
                            gi = pair_start + pair_i
                            if gi < group_len:
                                with auto_sync():
                                    ni = Var(group_start + gi)
                                    n_off = Var(ni * TILE_N)
                                    kv_row = Var(kv_base + n_off)
                                    valid_n = Min(TILE_N, S2 - n_off)

                                    l1k[l1k_cnt] <<= k[kv_row:kv_row + valid_n, 0:D]
                                    matmul(l0c[l0c_cnt], l1q[q_cur], l1k[l1k_cnt], is_init=True)

                                    score_ws[group_id, gi * TILE_M:gi * TILE_M + TILE_M, 0:TILE_N] <<= l0c[l0c_cnt]
                                    l1k_cnt += 1
                                    l0c_cnt += 1
                        qk_mutex.ready()

                for rb in unroll(0, HALF_M, ROW_CHUNK):
                    rb_slot = rb // ROW_CHUNK
                    chunk_row = Var(sb_row + rb)
                    chunk_valid_m = Min(ROW_CHUNK, Max(valid_m - chunk_row, 0))

                    for gi in range(0, GROUP_N):
                        if gi < group_len:
                            with auto_sync():
                                ni = Var(group_start + gi)
                                n_off = Var(ni * TILE_N)
                                valid_n = Min(TILE_N, S2 - n_off)
                                ub_score = ub_score_group[gi]

                                if rb == 0:
                                    if gi == 0 or gi == 2:
                                        qk_mutex.wait()
                                ub_score <<= score_ws[group_id, gi * TILE_M + chunk_row:gi * TILE_M + chunk_row + ROW_CHUNK, 0:TILE_N]

                                muls(ub_score, ub_score, scale)
                                if ni == lmt:
                                    diag_valid_n = Min(valid_n, (chunk_row // BLOCK_CAUSAL + 1) * BLOCK_CAUSAL)
                                    apply_score_tail_mask_chunk(ub_score, diag_valid_n)
                                elif valid_n < TILE_N:
                                    apply_score_tail_mask_chunk(ub_score, valid_n)
                                vmax(ub_tmp, ub_score[0:ROW_CHUNK, 0:HALF_N], ub_score[0:ROW_CHUNK, HALF_N:TILE_N])
                                cgmax(ub_max, ub_tmp)
                                if gi == 0:
                                    cgmax(
                                        ub_group_max_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK],
                                        ub_max,
                                        repeat=ROW_CHUNK // 8,
                                        dst_rep_stride=1,
                                        src_blk_stride=1,
                                        src_rep_stride=8,
                                    )
                                else:
                                    cgmax(ub_max_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK], ub_max, repeat=ROW_CHUNK // 8, dst_rep_stride=1, src_blk_stride=1, src_rep_stride=8)
                                    vmax(
                                        ub_group_max_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK_VEC],
                                        ub_group_max_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK_VEC],
                                        ub_max_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK_VEC],
                                    )
                            if rb == HALF_M - ROW_CHUNK:
                                if gi == 1 or gi == 3 or gi == group_len - 1:
                                    qk_mutex.free()

                    exp_row = Var(stage1_slot * 2 + rb_slot)
                    if group_id == 0:
                        ub_rmax_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK_VEC] <<= ub_group_max_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK_VEC]
                    else:
                        ub_old_max_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK_VEC] <<= ub_rmax_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK_VEC]
                        vmax(
                            ub_rmax_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK_VEC],
                            ub_rmax_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK_VEC],
                            ub_group_max_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK_VEC],
                        )
                        sub(
                            expdiff_store[exp_row:exp_row + 1, 0:ROW_CHUNK_VEC],
                            ub_old_max_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK_VEC],
                            ub_rmax_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK_VEC],
                        )
                        exp(
                            expdiff_store[exp_row:exp_row + 1, 0:ROW_CHUNK_VEC],
                            expdiff_store[exp_row:exp_row + 1, 0:ROW_CHUNK_VEC],
                        )
                    if rb == 0:
                        p_mutex.lock()

                    brcb(ub_brcb_max, ub_rmax_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK], repeat=ROW_CHUNK // 8, dst_blk_stride=1, dst_rep_stride=8)
                    for gi in range(0, GROUP_N):
                        if gi < group_len:
                            with auto_sync():
                                ub_p = ub_p_group[gi]
                                ub_score = ub_score_group[gi]

                                sub(ub_score[0:ROW_CHUNK, 0:HALF_N], ub_score[0:ROW_CHUNK, 0:HALF_N], ub_brcb_max)
                                sub(ub_score[0:ROW_CHUNK, HALF_N:TILE_N], ub_score[0:ROW_CHUNK, HALF_N:TILE_N], ub_brcb_max)
                                if chunk_valid_m < ROW_CHUNK:
                                    apply_score_row_tail_mask_chunk_after_shift(ub_score, chunk_valid_m)

                                exp(ub_score, ub_score)
                                add(ub_tmp, ub_score[0:ROW_CHUNK, 0:HALF_N], ub_score[0:ROW_CHUNK, HALF_N:TILE_N])
                                cgadd(ub_max, ub_tmp)
                                if gi == 0:
                                    cgadd(
                                        ub_group_sum_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK],
                                        ub_max,
                                        repeat=ROW_CHUNK // 8,
                                        dst_rep_stride=1,
                                        src_blk_stride=1,
                                        src_rep_stride=8,
                                    )
                                else:
                                    cgadd(ub_sum_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK], ub_max, repeat=ROW_CHUNK // 8, dst_rep_stride=1, src_blk_stride=1, src_rep_stride=8)
                                    add(
                                        ub_group_sum_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK_VEC],
                                        ub_group_sum_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK_VEC],
                                        ub_sum_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK_VEC],
                                    )

                                cast(ub_p, ub_score)
                                p_ws[group_id, gi * TILE_M + chunk_row:gi * TILE_M + chunk_row + ROW_CHUNK, 0:TILE_N] <<= ub_p

                    if group_id == 0:
                        ub_rsum_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK_VEC] <<= ub_group_sum_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK_VEC]
                    else:
                        mul(
                            ub_rsum_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK_VEC],
                            ub_rsum_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK_VEC],
                            expdiff_store[exp_row:exp_row + 1, 0:ROW_CHUNK_VEC],
                        )
                        add(
                            ub_rsum_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK_VEC],
                            ub_rsum_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK_VEC],
                            ub_group_sum_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK_VEC],
                        )

                p_mutex.ready()

            if group_id >= GROUP_LOOKAHEAD:
                stage2_slot = var_mod(group_id - GROUP_LOOKAHEAD, GROUP_STAGE_SLOTS)

                with cube_scope():
                    with auto_sync():
                        # Preload this group's V tiles into the L1 ring before waiting on
                        # P, so the V GM->L1 loads overlap the p_mutex wait. preload uses
                        # a shadow counter; the matmul below catches up via l1pv_cnt.
                        preload_l1pv_cnt = Var(l1pv_cnt)
                        for gi in range(0, GROUP_N):
                            if (group_id - GROUP_LOOKAHEAD) * GROUP_N + gi < active_tiles_n:
                                pre_ni = Var((group_id - GROUP_LOOKAHEAD) * GROUP_N + gi)
                                pre_n_off = Var(pre_ni * TILE_N)
                                pre_valid_n = Min(TILE_N, S2 - pre_n_off)
                                pre_v_row = Var(kv_base + pre_n_off)
                                pre_v_slot = var_mod(preload_l1pv_cnt, V_PRELOAD_SLOTS)
                                pre_v_base = Var(pre_v_slot * TILE_N)
                                # A trailing K tile fills only `pre_valid_n` of the slot's TILE_N
                                # rows, but the PV matmul below contracts the WHOLE slot, and a
                                # first-use ring slot is uninitialized L1 (the simulator fills it
                                # with 0xFF = NaN). P's masked columns are exact zeros, but
                                # 0 * NaN = NaN. Write the slot as v6 does -- the load's destination
                                # extent matching its source, then a DISJOINT backfill of the
                                # unloaded rows with finite in-range V. Two overlapping writers on
                                # one L1 slot are not ordered against each other by issue order and
                                # a bar_mte2() between them does not fix it: the board leaves the
                                # slot's trailing columns holding poison, and 2.14e-4 * poison is
                                # the tail row's 4.7e-02 board diff (D-214).
                                l1v[pre_v_base:pre_v_base + pre_valid_n, 0:TILE_K] <<= v[pre_v_row:pre_v_row + pre_valid_n, 0:D]
                                if pre_valid_n < TILE_N:
                                    pad_rows = Var(TILE_N - pre_valid_n)
                                    l1v[pre_v_base + pre_valid_n:pre_v_base + TILE_N, 0:TILE_K] <<= v[kv_base:kv_base + pad_rows, 0:D]
                                preload_l1pv_cnt += 1

                        p_mutex.wait()
                        for gi in range(0, GROUP_N):
                            if (group_id - GROUP_LOOKAHEAD) * GROUP_N + gi < active_tiles_n:
                                v_slot = var_mod(l1pv_cnt, V_PRELOAD_SLOTS)
                                v_base = Var(v_slot * TILE_N)
                                l1p[l1pv_cnt] <<= p_ws[group_id - GROUP_LOOKAHEAD, gi * TILE_M:gi * TILE_M + TILE_M, 0:TILE_N]
                                if gi == 0:
                                    matmul(l0c[l0c_cnt], l1p[l1pv_cnt], l1v[v_base:v_base + TILE_N, 0:TILE_K].T, is_init=True)
                                else:
                                    matmul(l0c[l0c_cnt], l1p[l1pv_cnt], l1v[v_base:v_base + TILE_N, 0:TILE_K].T, is_init=False)
                                l1pv_cnt += 1
                        p_mutex.free()

                        pv_mutex.lock()
                        pv_ws[group_id - GROUP_LOOKAHEAD, 0:TILE_M, 0:D] <<= l0c[l0c_cnt]
                        pv_mutex.ready()
                        l0c_cnt += 1

                with auto_sync():
                    pv_mutex.wait()
                    ub_pv <<= pv_ws[group_id - GROUP_LOOKAHEAD, sb_row:sb_row + HALF_M, 0:D]
                    pv_mutex.free()

                    if group_id != GROUP_LOOKAHEAD:
                        exp_row2 = Var(stage2_slot * 2)
                        brcb(ub_expdiff[0:ROW_CHUNK, 0:8], expdiff_store[exp_row2:exp_row2 + 1, 0:ROW_CHUNK], repeat=ROW_CHUNK // 8, dst_blk_stride=1, dst_rep_stride=8)
                        brcb(ub_expdiff[ROW_CHUNK:HALF_M, 0:8], expdiff_store[exp_row2 + 1:exp_row2 + 2, 0:ROW_CHUNK], repeat=ROW_CHUNK // 8, dst_blk_stride=1, dst_rep_stride=8)
                        mul(accum_ub[0:HALF_M, 0:HALF_N], accum_ub[0:HALF_M, 0:HALF_N], ub_expdiff)
                        mul(accum_ub[0:HALF_M, HALF_N:TILE_K], accum_ub[0:HALF_M, HALF_N:TILE_K], ub_expdiff)
                    add(accum_ub, accum_ub, ub_pv)

        with auto_sync():
            brcb(ub_rowsum[0:ROW_CHUNK, 0:8], ub_rsum_s[0:1, 0:ROW_CHUNK], repeat=ROW_CHUNK // 8, dst_blk_stride=1, dst_rep_stride=8)
            brcb(ub_rowsum[ROW_CHUNK:HALF_M, 0:8], ub_rsum_s[1:2, 0:ROW_CHUNK], repeat=ROW_CHUNK // 8, dst_blk_stride=1, dst_rep_stride=8)
            div(accum_ub[0:HALF_M, 0:HALF_N], accum_ub[0:HALF_M, 0:HALF_N], ub_rowsum)
            div(accum_ub[0:HALF_M, HALF_N:TILE_K], accum_ub[0:HALF_M, HALF_N:TILE_K], ub_rowsum)

            if local_valid_m > 0:
                out_row = Var(q_row + sb_row)
                stats_row = Var((out_row // S1) * CeilDiv(S1, 8) * 8 + out_row % S1)
                out[out_row:out_row + local_valid_m, 0:D] <<= accum_ub[0:local_valid_m, 0:D]
                first_rows = Min(local_valid_m, ROW_CHUNK)
                rowmax[stats_row:stats_row + first_rows] <<= ub_rmax_s[0:1, 0:first_rows]
                rowsum[stats_row:stats_row + first_rows] <<= ub_rsum_s[0:1, 0:first_rows]
                if local_valid_m > ROW_CHUNK:
                    second_rows = Var(local_valid_m - ROW_CHUNK)
                    rowmax[stats_row + ROW_CHUNK:stats_row + ROW_CHUNK + second_rows] <<= ub_rmax_s[1:2, 0:second_rows]
                    rowsum[stats_row + ROW_CHUNK:stats_row + ROW_CHUNK + second_rows] <<= ub_rsum_s[1:2, 0:second_rows]
        q_cur += 1

    return out, rowmax, rowsum






@lru_cache(maxsize=2)
def build_v5_kernel(device):
    if device not in ("a2", "a3"):
        raise ValueError("Shared tensor-vector attention supports A2/A3")
    return import_module(f"ascriptor.{device}").kernel()(flash_attn_full_pj_half_block32_causal_v5_kernel)

# ----------------------------------------------------------------------------------------------------
# v6.py
# ----------------------------------------------------------------------------------------------------

# RFC-0009 batch-2 rewrite: the six hand-written events are gone (autosync guards the l1q ring,
# the group QBuffs, the accum reinitialisation and the store flushes), the stage1_cnt/stage2_cnt
# pair collapses into the single stream beat `g` (producer g, consumer g - GROUP_LOOKAHEAD; the
# slot row-indices of ub_rmax_s/expdiff_store keep the same arithmetic on g), and the workspace
# rings are GMBuffs (the GROUP_N slot dim row-flattened) checked by the gmbuff pass. The bare
# GMTensor signature is modernised - this kernel had never compiled in ascriptor before.

# rowmax/rowsum ring depth in M-tiles. stage-1 leads stage-2 by GROUP_LOOKAHEAD
# GROUPS, so at most GROUP_LOOKAHEAD+1 distinct M-tiles are live; MTILE_SLOTS must
# be strictly larger than that lead. 5 keeps the same (slots - lookahead == 2)
# margin used for the workspace rings.
MTILE_SLOTS = 5










def flash_attn_full_pj_half_block32_causal_v6_kernel(
    q: GM[f16, ('TQ', 'D')], k: GM[f16, ('TK', 'D')], v: GM[f16, ('TK', 'D')], out: GM[f32, ('TQ', 'D')],
    rowmax: GM[f32, ('TST',)], rowsum: GM[f32, ('TST',)],
    S1: i32, S2: i32, D: i32, BH: i32, scale: f32,
):
    score_ws = GMBuff(DT.float, [GROUP_N * TILE_M, TILE_N], slots=GROUP_STAGE_SLOTS, name="score_ws")
    p_ws = GMBuff(DT.half, [GROUP_N * TILE_M, TILE_N], slots=GROUP_STAGE_SLOTS, name="p_ws")
    pv_ws = GMBuff(DT.float, [TILE_M, TILE_K], slots=GROUP_STAGE_SLOTS, name="pv_ws")

    l1q = DBuff(DT.half, [TILE_M, TILE_K], Position.L1)
    l1k = DBuff(DT.half, [TILE_N, TILE_K], Position.L1)
    l1p = DBuff(DT.half, [TILE_M, TILE_N], Position.L1)
    l1v = Tensor(DT.half, [V_PRELOAD_SLOTS * TILE_N, TILE_K], Position.L1)
    l0c = DBuff(DT.float, [TILE_M, TILE_N], Position.L0C)

    ub_score_group = QBuff(DT.float, [ROW_CHUNK, TILE_N], Position.UB)
    ub_p_group = QBuff(DT.half, [ROW_CHUNK, TILE_N], Position.UB)
    ub_pv = Tensor(DT.float, [HALF_M, TILE_K], Position.UB)
    ub_tmp = Tensor(DT.float, [ROW_CHUNK, HALF_N], Position.UB)
    ub_max_s = Tensor(DT.float, [2, ROW_CHUNK_VEC], Position.UB)
    ub_group_max_s = Tensor(DT.float, [2, ROW_CHUNK_VEC], Position.UB)
    ub_old_max_s = Tensor(DT.float, [2, ROW_CHUNK_VEC], Position.UB)
    ub_sum_s = Tensor(DT.float, [2, ROW_CHUNK_VEC], Position.UB)
    ub_group_sum_s = Tensor(DT.float, [2, ROW_CHUNK_VEC], Position.UB)
    # Loop-carried running max/sum: ring of MTILE_SLOTS M-tiles, each holding the
    # two 32-row chunks -> row (mslot * 2 + rb_slot), mirroring expdiff_store.
    ub_rmax_s = Tensor(DT.float, [MTILE_SLOTS * 2, ROW_CHUNK_VEC], Position.UB)
    ub_rsum_s = Tensor(DT.float, [MTILE_SLOTS * 2, ROW_CHUNK_VEC], Position.UB)
    ub_max = Tensor(DT.float, [ROW_CHUNK, 8], Position.UB)
    ub_brcb_max = Tensor(DT.float, [ROW_CHUNK, 8], Position.UB)
    ub_rowsum = Tensor(DT.float, [HALF_M, 8], Position.UB)
    ub_expdiff = Tensor(DT.float, [HALF_M, 8], Position.UB)
    accum_ub = Tensor(DT.float, [HALF_M, TILE_K], Position.UB)
    expdiff_store = Tensor(DT.float, [GROUP_STAGE_SLOTS * 2, ROW_CHUNK_VEC], Position.UB)

    qk_mutex = CvMutex(0, depth=GROUP_STAGE_SLOTS, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)
    p_mutex = VcMutex(1, depth=GROUP_STAGE_SLOTS, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE2)
    pv_mutex = CvMutex(2, depth=GROUP_STAGE_SLOTS, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)

    l1k_cnt = Var(0)
    l1pv_cnt = Var(0)
    l0c_cnt = Var(0)
    q_cur = Var(0)

    # Stage-1/stage-2 cursors over this core's (m_tile, group) stream. The cube
    # and vec sides each keep a PRIVATE copy advanced only on their own side: a
    # single "both"-sided manually-advanced cursor confuses the side splitter
    # (its reverse side-routing pass does not capture loop-carried scalar deps,
    # and a conditional advance leaks active_groups across the cut). Both copies
    # start at 0 and advance with identical arithmetic, so they stay in lock-step.
    s1kc = Var(0)   # stage-1 M-tile index (cube)
    s1gc = Var(0)   # stage-1 group-in-tile (cube)
    s1kv = Var(0)   # stage-1 M-tile index (vec)
    s1gv = Var(0)   # stage-1 group-in-tile (vec)
    s2kc = Var(0)   # stage-2 M-tile index (cube)
    s2gc = Var(0)   # stage-2 group-in-tile (cube)
    s2kv = Var(0)   # stage-2 M-tile index (vec)
    s2gv = Var(0)   # stage-2 group-in-tile (vec)

    cube_idx = GetCubeIdx()
    sb = GetSubBlockIdx()
    sb_row = Var(sb * HALF_M)

    tiles_m = CeilDiv(S1, TILE_M)
    tiles_n = CeilDiv(S2, TILE_N)
    core_count = GetCubeNum()
    core_step = Var(1, DT.int)
    core_step <<= core_count
    total_m = Var(BH * tiles_m)
    n_mtiles = CeilDiv(Max(total_m - cube_idx, 0), core_step)

    # Pre-pass: total stage-1 groups for this core (snake-reorder aware).
    total_groups = Var(0)
    for kk in range(0, n_mtiles):
        gmt_p = Var(cube_idx + kk * core_step)
        bh_p = Var(gmt_p // tiles_m)
        raw_lmt_p = Var(gmt_p % tiles_m)
        odd_p = Var(var_mod(bh_p, 2))
        # branchless snake reorder: odd bh -> reverse lmt within the bh
        lmt_p = Var(raw_lmt_p + odd_p * (tiles_m - 1 - 2 * raw_lmt_p))
        atn_p = Min(tiles_n, lmt_p + 1)
        total_groups += CeilDiv(atn_p, GROUP_N)

    for g in range(0, total_groups + GROUP_LOOKAHEAD):
        # stage1_slot / stage2_slot ride "both"-sided counters incremented once
        # per group, exactly like v5 (cube writes the workspace slot, vec reads it).
        stage1_slot = var_mod(g, GROUP_STAGE_SLOTS)
        stage2_slot = var_mod(g - GROUP_LOOKAHEAD, GROUP_STAGE_SLOTS)

        # Reinitialise accum before this M-tile's first stage-2 group; the WAR
        # against the previous M-tile's writeback is autosync's loop-carried case.
        if g >= GROUP_LOOKAHEAD and s2gv == 0:
            with auto_sync():
                dup(accum_ub, 0.0)

        if g < total_groups:
            # ============ STAGE 1: QK -> softmax -> P for stream group g ============
            # --- cube-side metadata (private cube cursor s1kc/s1gc) ---
            gmt1c = Var(cube_idx + s1kc * core_step)
            bh1c = Var(gmt1c // tiles_m)
            raw1c = Var(gmt1c % tiles_m)
            lmt1c = Var(raw1c + var_mod(bh1c, 2) * (tiles_m - 1 - 2 * raw1c))
            q_row1 = Var(bh1c * S1 + lmt1c * TILE_M)
            kv_base1 = Var(bh1c * S2)
            valid_m1c = Min(TILE_M, S1 - lmt1c * TILE_M)
            atn1c = Min(tiles_n, lmt1c + 1)
            ag1c = CeilDiv(atn1c, GROUP_N)
            group_start_c = Var(s1gc * GROUP_N)
            group_len_c = Min(GROUP_N, atn1c - group_start_c)

            if s1gc == 0:
                with auto_sync():
                    l1q[q_cur] <<= q[q_row1:q_row1 + valid_m1c, 0:D]

            for pair_start in range(0, GROUP_N, 2):
                if pair_start < group_len_c:
                    qk_mutex.lock()
                    for pair_i in range(0, 2):
                        gi = pair_start + pair_i
                        if gi < group_len_c:
                            with auto_sync():
                                ni = Var(group_start_c + gi)
                                n_off = Var(ni * TILE_N)
                                kv_row = Var(kv_base1 + n_off)
                                valid_n = Min(TILE_N, S2 - n_off)

                                l1k[l1k_cnt] <<= k[kv_row:kv_row + valid_n, 0:D]
                                matmul(l0c[l0c_cnt], l1q[q_cur], l1k[l1k_cnt], is_init=True)

                                score_ws[g, gi * TILE_M:gi * TILE_M + TILE_M, 0:TILE_N] <<= l0c[l0c_cnt]
                                l1k_cnt += 1
                                l0c_cnt += 1
                    qk_mutex.ready()

            # cube cursor advance (cube-private)
            s1kc += (s1gc + 1) // ag1c
            q_cur += (s1gc + 1) // ag1c
            s1gc <<= var_mod(s1gc + 1, ag1c)

            # --- vec-side metadata (private vec cursor s1kv/s1gv) ---
            gmt1v = Var(cube_idx + s1kv * core_step)
            bh1v = Var(gmt1v // tiles_m)
            raw1v = Var(gmt1v % tiles_m)
            lmt1v = Var(raw1v + var_mod(bh1v, 2) * (tiles_m - 1 - 2 * raw1v))
            valid_m1v = Min(TILE_M, S1 - lmt1v * TILE_M)
            atn1v = Min(tiles_n, lmt1v + 1)
            ag1v = CeilDiv(atn1v, GROUP_N)
            group_start_v = Var(s1gv * GROUP_N)
            group_len_v = Min(GROUP_N, atn1v - group_start_v)
            mslot1 = var_mod(s1kv, MTILE_SLOTS)

            for rb in unroll(0, HALF_M, ROW_CHUNK):
                rb_slot = rb // ROW_CHUNK
                rmax_row1 = Var(mslot1 * 2 + rb_slot)
                chunk_row = Var(sb_row + rb)
                chunk_valid_m = Min(ROW_CHUNK, Max(valid_m1v - chunk_row, 0))

                for gi in range(0, GROUP_N):
                    if gi < group_len_v:
                        with auto_sync():
                            ni = Var(group_start_v + gi)
                            n_off = Var(ni * TILE_N)
                            valid_n = Min(TILE_N, S2 - n_off)
                            ub_score = ub_score_group[gi]

                            if rb == 0:
                                if gi == 0 or gi == 2:
                                    qk_mutex.wait()
                            ub_score <<= score_ws[g, gi * TILE_M + chunk_row:gi * TILE_M + chunk_row + ROW_CHUNK, 0:TILE_N]

                            muls(ub_score, ub_score, scale)
                            if ni == lmt1v:
                                diag_valid_n = Min(valid_n, (chunk_row // BLOCK_CAUSAL + 1) * BLOCK_CAUSAL)
                                apply_score_tail_mask_chunk(ub_score, diag_valid_n)
                            elif valid_n < TILE_N:
                                apply_score_tail_mask_chunk(ub_score, valid_n)
                            vmax(ub_tmp, ub_score[0:ROW_CHUNK, 0:HALF_N], ub_score[0:ROW_CHUNK, HALF_N:TILE_N])
                            cgmax(ub_max, ub_tmp)
                            if gi == 0:
                                cgmax(
                                    ub_group_max_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK],
                                    ub_max,
                                    repeat=ROW_CHUNK // 8,
                                    dst_rep_stride=1,
                                    src_blk_stride=1,
                                    src_rep_stride=8,
                                )
                            else:
                                cgmax(ub_max_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK], ub_max, repeat=ROW_CHUNK // 8, dst_rep_stride=1, src_blk_stride=1, src_rep_stride=8)
                                vmax(
                                    ub_group_max_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK_VEC],
                                    ub_group_max_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK_VEC],
                                    ub_max_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK_VEC],
                                )
                        if rb == HALF_M - ROW_CHUNK:
                            if gi == 1 or gi == 3 or gi == group_len_v - 1:
                                qk_mutex.free()

                exp_row = Var(stage1_slot * 2 + rb_slot)
                if s1gv == 0:
                    ub_rmax_s[rmax_row1:rmax_row1 + 1, 0:ROW_CHUNK_VEC] <<= ub_group_max_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK_VEC]
                else:
                    ub_old_max_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK_VEC] <<= ub_rmax_s[rmax_row1:rmax_row1 + 1, 0:ROW_CHUNK_VEC]
                    vmax(
                        ub_rmax_s[rmax_row1:rmax_row1 + 1, 0:ROW_CHUNK_VEC],
                        ub_rmax_s[rmax_row1:rmax_row1 + 1, 0:ROW_CHUNK_VEC],
                        ub_group_max_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK_VEC],
                    )
                    sub(
                        expdiff_store[exp_row:exp_row + 1, 0:ROW_CHUNK_VEC],
                        ub_old_max_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK_VEC],
                        ub_rmax_s[rmax_row1:rmax_row1 + 1, 0:ROW_CHUNK_VEC],
                    )
                    exp(
                        expdiff_store[exp_row:exp_row + 1, 0:ROW_CHUNK_VEC],
                        expdiff_store[exp_row:exp_row + 1, 0:ROW_CHUNK_VEC],
                    )
                if rb == 0:
                    p_mutex.lock()

                brcb(ub_brcb_max, ub_rmax_s[rmax_row1:rmax_row1 + 1, 0:ROW_CHUNK], repeat=ROW_CHUNK // 8, dst_blk_stride=1, dst_rep_stride=8)
                for gi in range(0, GROUP_N):
                    if gi < group_len_v:
                        with auto_sync():
                            ub_p = ub_p_group[gi]
                            ub_score = ub_score_group[gi]

                            sub(ub_score[0:ROW_CHUNK, 0:HALF_N], ub_score[0:ROW_CHUNK, 0:HALF_N], ub_brcb_max)
                            sub(ub_score[0:ROW_CHUNK, HALF_N:TILE_N], ub_score[0:ROW_CHUNK, HALF_N:TILE_N], ub_brcb_max)
                            if chunk_valid_m < ROW_CHUNK:
                                apply_score_row_tail_mask_chunk_after_shift(ub_score, chunk_valid_m)

                            exp(ub_score, ub_score)
                            add(ub_tmp, ub_score[0:ROW_CHUNK, 0:HALF_N], ub_score[0:ROW_CHUNK, HALF_N:TILE_N])
                            cgadd(ub_max, ub_tmp)
                            if gi == 0:
                                cgadd(
                                    ub_group_sum_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK],
                                    ub_max,
                                    repeat=ROW_CHUNK // 8,
                                    dst_rep_stride=1,
                                    src_blk_stride=1,
                                    src_rep_stride=8,
                                )
                            else:
                                cgadd(ub_sum_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK], ub_max, repeat=ROW_CHUNK // 8, dst_rep_stride=1, src_blk_stride=1, src_rep_stride=8)
                                add(
                                    ub_group_sum_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK_VEC],
                                    ub_group_sum_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK_VEC],
                                    ub_sum_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK_VEC],
                                )

                            cast(ub_p, ub_score)
                            p_ws[g, gi * TILE_M + chunk_row:gi * TILE_M + chunk_row + ROW_CHUNK, 0:TILE_N] <<= ub_p

                if s1gv == 0:
                    ub_rsum_s[rmax_row1:rmax_row1 + 1, 0:ROW_CHUNK_VEC] <<= ub_group_sum_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK_VEC]
                else:
                    mul(
                        ub_rsum_s[rmax_row1:rmax_row1 + 1, 0:ROW_CHUNK_VEC],
                        ub_rsum_s[rmax_row1:rmax_row1 + 1, 0:ROW_CHUNK_VEC],
                        expdiff_store[exp_row:exp_row + 1, 0:ROW_CHUNK_VEC],
                    )
                    add(
                        ub_rsum_s[rmax_row1:rmax_row1 + 1, 0:ROW_CHUNK_VEC],
                        ub_rsum_s[rmax_row1:rmax_row1 + 1, 0:ROW_CHUNK_VEC],
                        ub_group_sum_s[rb_slot:rb_slot + 1, 0:ROW_CHUNK_VEC],
                    )

            p_mutex.ready()
            # vec cursor advance (vec-private)
            s1kv += (s1gv + 1) // ag1v
            s1gv <<= var_mod(s1gv + 1, ag1v)

        if g >= GROUP_LOOKAHEAD:
            # ============ STAGE 2: P@V -> accum for stream group (g-LOOKAHEAD) ============
            # --- cube-side metadata (private cube cursor s2kc/s2gc) ---
            gmt2c = Var(cube_idx + s2kc * core_step)
            bh2c = Var(gmt2c // tiles_m)
            raw2c = Var(gmt2c % tiles_m)
            lmt2c = Var(raw2c + var_mod(bh2c, 2) * (tiles_m - 1 - 2 * raw2c))
            kv_base2 = Var(bh2c * S2)
            atn2c = Min(tiles_n, lmt2c + 1)
            ag2c = CeilDiv(atn2c, GROUP_N)
            group_start2 = Var(s2gc * GROUP_N)

            # --- vec-side metadata (private vec cursor s2kv/s2gv) ---
            gmt2v = Var(cube_idx + s2kv * core_step)
            bh2v = Var(gmt2v // tiles_m)
            raw2v = Var(gmt2v % tiles_m)
            lmt2v = Var(raw2v + var_mod(bh2v, 2) * (tiles_m - 1 - 2 * raw2v))
            q_row2 = Var(bh2v * S1 + lmt2v * TILE_M)
            valid_m2v = Min(TILE_M, S1 - lmt2v * TILE_M)
            atn2v = Min(tiles_n, lmt2v + 1)
            ag2v = CeilDiv(atn2v, GROUP_N)
            local_valid_m2 = Min(HALF_M, Max(valid_m2v - sb_row, 0))
            mslot2 = var_mod(s2kv, MTILE_SLOTS)

            with cube_scope():
                with auto_sync():
                    preload_l1pv_cnt = Var(l1pv_cnt)
                    for gi in range(0, GROUP_N):
                        if group_start2 + gi < atn2c:
                            pre_ni = Var(group_start2 + gi)
                            pre_n_off = Var(pre_ni * TILE_N)
                            pre_valid_n = Min(TILE_N, S2 - pre_n_off)
                            pre_v_row = Var(kv_base2 + pre_n_off)
                            pre_v_slot = var_mod(preload_l1pv_cnt, V_PRELOAD_SLOTS)
                            pre_v_base = Var(pre_v_slot * TILE_N)
                            l1v[pre_v_base:pre_v_base + pre_valid_n, 0:TILE_K] <<= v[pre_v_row:pre_v_row + pre_valid_n, 0:D]
                            # Tail N-tile: rows [pre_valid_n:TILE_N] of a first-use ring
                            # slot are uninitialized L1 (the simulator fills L1 with
                            # 0xFF = NaN). The P@V matmul contracts the full TILE_N; those
                            # P columns are masked to 0, but 0 * NaN = NaN. Backfill the
                            # unloaded rows with finite in-range V (their columns are
                            # masked, so any finite V yields a clean 0 contribution),
                            # keeping the product NaN-free on simulator and hardware.
                            if pre_valid_n < TILE_N:
                                pad_rows = Var(TILE_N - pre_valid_n)
                                l1v[pre_v_base + pre_valid_n:pre_v_base + TILE_N, 0:TILE_K] <<= v[kv_base2:kv_base2 + pad_rows, 0:D]
                            preload_l1pv_cnt += 1

                    p_mutex.wait()
                    for gi in range(0, GROUP_N):
                        if group_start2 + gi < atn2c:
                            v_slot = var_mod(l1pv_cnt, V_PRELOAD_SLOTS)
                            v_base = Var(v_slot * TILE_N)
                            l1p[l1pv_cnt] <<= p_ws[g - GROUP_LOOKAHEAD, gi * TILE_M:gi * TILE_M + TILE_M, 0:TILE_N]
                            if gi == 0:
                                matmul(l0c[l0c_cnt], l1p[l1pv_cnt], l1v[v_base:v_base + TILE_N, 0:TILE_K].T, is_init=True)
                            else:
                                matmul(l0c[l0c_cnt], l1p[l1pv_cnt], l1v[v_base:v_base + TILE_N, 0:TILE_K].T, is_init=False)
                            l1pv_cnt += 1
                    p_mutex.free()

                    pv_mutex.lock()
                    pv_ws[g - GROUP_LOOKAHEAD, 0:TILE_M, 0:D] <<= l0c[l0c_cnt]
                    pv_mutex.ready()
                    l0c_cnt += 1

            # cube cursor advance (cube-private)
            s2kc += (s2gc + 1) // ag2c
            s2gc <<= var_mod(s2gc + 1, ag2c)

            with auto_sync():
                pv_mutex.wait()
                ub_pv <<= pv_ws[g - GROUP_LOOKAHEAD, sb_row:sb_row + HALF_M, 0:D]
                pv_mutex.free()

                if s2gv != 0:
                    exp_row2 = Var(stage2_slot * 2)
                    brcb(ub_expdiff[0:ROW_CHUNK, 0:8], expdiff_store[exp_row2:exp_row2 + 1, 0:ROW_CHUNK], repeat=ROW_CHUNK // 8, dst_blk_stride=1, dst_rep_stride=8)
                    brcb(ub_expdiff[ROW_CHUNK:HALF_M, 0:8], expdiff_store[exp_row2 + 1:exp_row2 + 2, 0:ROW_CHUNK], repeat=ROW_CHUNK // 8, dst_blk_stride=1, dst_rep_stride=8)
                    mul(accum_ub[0:HALF_M, 0:HALF_N], accum_ub[0:HALF_M, 0:HALF_N], ub_expdiff)
                    mul(accum_ub[0:HALF_M, HALF_N:TILE_K], accum_ub[0:HALF_M, HALF_N:TILE_K], ub_expdiff)
                add(accum_ub, accum_ub, ub_pv)

            if (s2gv + 1) // ag2v == 1:
                # ----- FINALIZE M-tile s2kv (all vec-side) -----
                with auto_sync():
                    rsum_row0 = Var(mslot2 * 2)
                    rsum_row1 = Var(mslot2 * 2 + 1)
                    brcb(ub_rowsum[0:ROW_CHUNK, 0:8], ub_rsum_s[rsum_row0:rsum_row0 + 1, 0:ROW_CHUNK], repeat=ROW_CHUNK // 8, dst_blk_stride=1, dst_rep_stride=8)
                    brcb(ub_rowsum[ROW_CHUNK:HALF_M, 0:8], ub_rsum_s[rsum_row1:rsum_row1 + 1, 0:ROW_CHUNK], repeat=ROW_CHUNK // 8, dst_blk_stride=1, dst_rep_stride=8)
                    div(accum_ub[0:HALF_M, 0:HALF_N], accum_ub[0:HALF_M, 0:HALF_N], ub_rowsum)
                    div(accum_ub[0:HALF_M, HALF_N:TILE_K], accum_ub[0:HALF_M, HALF_N:TILE_K], ub_rowsum)

                    if local_valid_m2 > 0:
                        out_row = Var(q_row2 + sb_row)
                        stats_row = Var((out_row // S1) * CeilDiv(S1, 8) * 8 + out_row % S1)
                        out[out_row:out_row + local_valid_m2, 0:D] <<= accum_ub[0:local_valid_m2, 0:D]
                        first_rows = Min(local_valid_m2, ROW_CHUNK)
                        rowmax[stats_row:stats_row + first_rows] <<= ub_rmax_s[rsum_row0:rsum_row0 + 1, 0:first_rows]
                        rowsum[stats_row:stats_row + first_rows] <<= ub_rsum_s[rsum_row0:rsum_row0 + 1, 0:first_rows]
                        if local_valid_m2 > ROW_CHUNK:
                            second_rows = Var(local_valid_m2 - ROW_CHUNK)
                            rowmax[stats_row + ROW_CHUNK:stats_row + ROW_CHUNK + second_rows] <<= ub_rmax_s[rsum_row1:rsum_row1 + 1, 0:second_rows]
                            rowsum[stats_row + ROW_CHUNK:stats_row + ROW_CHUNK + second_rows] <<= ub_rsum_s[rsum_row1:rsum_row1 + 1, 0:second_rows]
            # vec cursor advance (vec-private)
            s2kv += (s2gv + 1) // ag2v
            s2gv <<= var_mod(s2gv + 1, ag2v)

    return out, rowmax, rowsum






@lru_cache(maxsize=2)
def build_v6_kernel(device):
    if device not in ("a2", "a3"):
        raise ValueError("Shared tensor-vector attention supports A2/A3")
    return import_module(f"ascriptor.{device}").kernel()(flash_attn_full_pj_half_block32_causal_v6_kernel)


# ----------------------------------------------------------------------------------------------------
# dispatcher
# ----------------------------------------------------------------------------------------------------

BUILDERS = {"v1": build_v1_kernel, "v2": build_v2_kernel, "v3": build_v3_kernel,
            "v4": build_v4_kernel, "v5": build_v5_kernel, "v6": build_v6_kernel}


def build_kernel(variant, device):
    """Bind one variant's body to the a2 or a3 facade. Each builder is separately
    lru_cached, so asking for the same variant and device twice returns one kernel."""
    if variant not in BUILDERS:
        raise ValueError(f"unknown variant {variant!r}; one of {sorted(BUILDERS)}")
    return BUILDERS[variant](device)
