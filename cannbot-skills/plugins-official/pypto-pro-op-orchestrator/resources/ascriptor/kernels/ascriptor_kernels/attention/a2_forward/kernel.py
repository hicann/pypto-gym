# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Nine A2/A3 forward-attention sources behind one dispatcher.

They are not nine schedules for one kernel: they differ in ABI as well as in schedule.
dense_fp16 is FP16 in and FP32 out with no row statistics; mha_bf16 adds BF16 probabilities
and publishes rowmax/rowsum; gqa_bf16 takes B/HQ/HKV and maps several query heads onto one
KV head; mha_d256 and mha_d256_bf16 widen D to 256 -- which halves the M tile to 64 rows and
is why their constants and masking helpers carry a D256_/d256_ prefix here -- and switch
block32 causality at run time; the three pj_* sources store the probabilities through
HiFloat8 at a 128-key group; pv_stage is not full attention at all but the unnormalized
partial PV of each 128-key tile. build_kernel(variant, device) binds one of them to the a2
or a3 facade.

Names that had to be separated when the nine modules became one file: the two different
kernels both called flash_attn_full_pj_hif8_kernel are now
flash_attn_full_pj_hif8_bf16_lag2_kernel and flash_attn_full_pj_hif8_fp16_lag1_kernel, and
the three different (lookahead, slots) rings are DENSE_ (2/4), RING_ (3/5) and D256_ (2/4)."""

from ascriptor.a2 import *
from builtins import range as py_range
from functools import lru_cache
from importlib import import_module

# ----------------------------------------------------------------------------------------------------
# dense_fp16.py
# ----------------------------------------------------------------------------------------------------

# RFC-0009 batch-1 rewrite: the five hand-written credit events are gone — every on-chip tile
# they guarded is a Buff (or a plain tensor whose loop-carried hazard autosync tracks), all
# compute sits inside auto_sync regions, and the workspace rings are GMBuffs indexed with one
# logical beat (producer `group_id`, consumer `group_id - DENSE_GROUP_LOOKAHEAD`). The ring algebra —
# one beat counter, reader lag < slots, every cross-side access inside a mutex window of depth
# <= slots — is machine-checked by the gmbuff pass, no longer a comment-only invariant.


TILE_M = 128
TILE_N = 128
TILE_K = 128
HALF_M = TILE_M // 2
HALF_N = TILE_N // 2
GROUP_N = 4
DENSE_GROUP_STAGE_SLOTS = 4
DENSE_GROUP_LOOKAHEAD = 2
ROW_CHUNK = 32
NEG_LARGE = -1.0e30


def flash_attn_full_kernel(
    q: GM[f16, ('TQ', 'D')], k: GM[f16, ('TK', 'D')], v: GM[f16, ('TK', 'D')], out: GM[f32, ('TQ', 'D')],
    S1: i32, S2: i32, D: i32, BH: i32, scale: f32,
):
    score_ws = GMBuff(DT.float, [TILE_M, GROUP_N * TILE_N], slots=DENSE_GROUP_STAGE_SLOTS, name="score_ws")
    p_ws = GMBuff(DT.half, [TILE_M, GROUP_N * TILE_N], slots=DENSE_GROUP_STAGE_SLOTS, name="p_ws")
    pv_ws = GMBuff(DT.float, [TILE_M, TILE_K], slots=DENSE_GROUP_STAGE_SLOTS, name="pv_ws")

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

    qk_mutex = CvMutex(0, depth=DENSE_GROUP_STAGE_SLOTS, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)
    p_mutex = VcMutex(1, depth=DENSE_GROUP_STAGE_SLOTS, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE2)
    pv_mutex = CvMutex(2, depth=DENSE_GROUP_STAGE_SLOTS, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)

    l1k_cnt = Var(0)
    l1pv_cnt = Var(0)
    l0c_cnt = Var(0)
    q_cur = Var(0)

    cube_idx = GetCubeIdx()
    sb = GetSubBlockIdx()
    sb_row = Var(sb * HALF_M)

    tiles_m = CeilDiv(S1, TILE_M)
    tiles_n = CeilDiv(S2, TILE_N)
    active_groups = CeilDiv(tiles_n, GROUP_N)
    core_count = GetCubeNum()
    core_step = Var(1, DT.int)
    core_step <<= core_count
    total_m = Var(BH * tiles_m)

    for gmt in range(cube_idx, total_m, core_step):
        bh = Var(gmt // tiles_m)
        lmt = Var(gmt % tiles_m)
        q_row = Var(bh * S1 + lmt * TILE_M)
        kv_base = Var(bh * S2)

        with auto_sync():
            dup(ub_rmax_s, NEG_LARGE)
            dup(ub_rsum_s, 0.0)
            dup(ub_zero_s, 0.0)
            dup(ub_zero_chunk_s, 0.0, count=ROW_CHUNK)
            dup(accum_ub, 0.0)

        with auto_sync():
            l1q[q_cur] <<= q[q_row:q_row + TILE_M, 0:D]

        for group_id in range(0, active_groups + DENSE_GROUP_LOOKAHEAD):
            if group_id < active_groups:
                group_start = Var(group_id * GROUP_N)
                group_len = Min(GROUP_N, tiles_n - group_start)

                qk_mutex.lock()
                for gi in range(0, GROUP_N):
                    if gi < group_len:
                        with auto_sync():
                            ni = Var(group_start + gi)
                            n_off = Var(ni * TILE_N)
                            kv_row = Var(kv_base + n_off)

                            l1k[l1k_cnt] <<= k[kv_row:kv_row + TILE_N, 0:D]
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

                    with auto_sync():
                        dup(ub_group_max_s, NEG_LARGE, count=ROW_CHUNK)
                    for gi in range(0, GROUP_N):
                        if gi < group_len:
                            with auto_sync():
                                ub_score = ub_score_group[gi]
                                group_col = gi * TILE_N
                                ub_score <<= score_ws[
                                    group_id, chunk_row:chunk_row + ROW_CHUNK, group_col:group_col + TILE_N
                                ]

                                muls(ub_score, ub_score, scale)
                                vmax(ub_tmp, ub_score[0:ROW_CHUNK, 0:HALF_N], ub_score[0:ROW_CHUNK, HALF_N:TILE_N])
                                cmax(ub_max_s, ub_tmp)
                                vmax(ub_group_max_s, ub_group_max_s, ub_max_s, count=ROW_CHUNK)

                    with auto_sync():
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

                        brcb(
                            ub_max, ub_rmax_s[0:1, rb:rb + ROW_CHUNK],
                            repeat=ROW_CHUNK // 8, dst_blk_stride=1, dst_rep_stride=8,
                        )
                    for gi in range(0, GROUP_N):
                        if gi < group_len:
                            with auto_sync():
                                ub_p = ub_p_group[gi]
                                ub_score = ub_score_group[gi]
                                group_col = gi * TILE_N

                                sub(ub_score[0:ROW_CHUNK, 0:HALF_N], ub_score[0:ROW_CHUNK, 0:HALF_N], ub_max)
                                sub(ub_score[0:ROW_CHUNK, HALF_N:TILE_N], ub_score[0:ROW_CHUNK, HALF_N:TILE_N], ub_max)

                                exp(ub_score, ub_score)
                                add(ub_tmp, ub_score[0:ROW_CHUNK, 0:HALF_N], ub_score[0:ROW_CHUNK, HALF_N:TILE_N])
                                cadd(ub_sum_s, ub_tmp)
                                add(ub_group_sum_s, ub_group_sum_s, ub_sum_s, count=ROW_CHUNK)

                                cast(ub_p, ub_score)
                                p_ws[
                                    group_id, chunk_row:chunk_row + ROW_CHUNK, group_col:group_col + TILE_N
                                ] <<= ub_p

                    with auto_sync():
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

            if group_id >= DENSE_GROUP_LOOKAHEAD:
                with cube_scope():
                    with auto_sync():
                        p_mutex.wait()
                        group_start2 = Var((group_id - DENSE_GROUP_LOOKAHEAD) * GROUP_N)
                        group_len2 = Min(GROUP_N, tiles_n - group_start2)
                        for gi in range(0, GROUP_N):
                            if gi < group_len2:
                                ni = Var(group_start2 + gi)
                                n_off = Var(ni * TILE_N)
                                v_row = Var(kv_base + n_off)
                                group_col = gi * TILE_N

                                l1p[l1pv_cnt] <<= p_ws[
                                    group_id - DENSE_GROUP_LOOKAHEAD, 0:TILE_M, group_col:group_col + TILE_N
                                ]
                                l1v[l1pv_cnt] <<= v[v_row:v_row + TILE_N, 0:D]
                                if gi == 0:
                                    matmul(l0c[l0c_cnt], l1p[l1pv_cnt], l1v[l1pv_cnt].T, is_init=True)
                                else:
                                    matmul(l0c[l0c_cnt], l1p[l1pv_cnt], l1v[l1pv_cnt].T, is_init=False)
                                l1pv_cnt += 1
                        p_mutex.free()

                        pv_mutex.lock()
                        pv_ws[group_id - DENSE_GROUP_LOOKAHEAD, 0:TILE_M, 0:D] <<= l0c[l0c_cnt]
                        pv_mutex.ready()
                        l0c_cnt += 1

                with auto_sync():
                    pv_mutex.wait()
                    ub_pv <<= pv_ws[group_id - DENSE_GROUP_LOOKAHEAD, sb_row:sb_row + HALF_M, 0:D]
                    pv_mutex.free()

                    brcb(ub_expdiff, expdiff_buf[group_id - DENSE_GROUP_LOOKAHEAD],
                         repeat=HALF_M // 8, dst_blk_stride=1, dst_rep_stride=8)
                    mul(accum_ub[0:HALF_M, 0:HALF_N], accum_ub[0:HALF_M, 0:HALF_N], ub_expdiff)
                    mul(accum_ub[0:HALF_M, HALF_N:TILE_K], accum_ub[0:HALF_M, HALF_N:TILE_K], ub_expdiff)
                    add(accum_ub, accum_ub, ub_pv)

        with auto_sync():
            brcb(ub_rowsum, ub_rsum_s, repeat=HALF_M // 8, dst_blk_stride=1, dst_rep_stride=8)
            div(accum_ub[0:HALF_M, 0:HALF_N], accum_ub[0:HALF_M, 0:HALF_N], ub_rowsum)
            div(accum_ub[0:HALF_M, HALF_N:TILE_K], accum_ub[0:HALF_M, HALF_N:TILE_K], ub_rowsum)

            out_row = Var(q_row + sb_row)
            out[out_row:out_row + HALF_M, 0:D] <<= accum_ub
        q_cur += 1

    return out



@lru_cache(maxsize=2)
def build_dense_fp16_kernel(device):
    if device not in ("a2", "a3"):
        raise ValueError("Shared tensor-vector source supports A2/A3")
    return import_module(f"ascriptor.{device}").kernel()(flash_attn_full_kernel)

# ----------------------------------------------------------------------------------------------------
# gqa_bf16.py
# ----------------------------------------------------------------------------------------------------

# RFC-0009 batch-2 rewrite: the six hand-written events are gone (autosync guards the l1q ring,
# the group QBuffs, the accum reinitialisation and the store flushes), the stage1_cnt/stage2_cnt
# pair collapses into the single stream beat `g` (producer g, consumer g - RING_GROUP_LOOKAHEAD; the
# slot row-indices of ub_rmax_s/expdiff_store keep the same arithmetic on g), and the workspace
# rings are GMBuffs (the GROUP_N slot dim row-flattened) checked by the gmbuff pass. The bare
# GMTensor signature is modernised - this kernel had never compiled in ascriptor before.

RING_GROUP_LOOKAHEAD = 3
RING_GROUP_STAGE_SLOTS = 5
# Running max/sum ring depth in M-tiles. stage-1 leads stage-2 by RING_GROUP_LOOKAHEAD
# GROUPS, so at most RING_GROUP_LOOKAHEAD+1 distinct M-tiles are live; MTILE_SLOTS must
# be strictly larger than that lead.
MTILE_SLOTS = 5
V_PRELOAD_SLOTS = 8
ROW_CHUNK_VEC = 64


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


def flash_attn_fullmask_gqa_bf16_kernel(
    q: GM[bf16, ('TQ', 'D')], k: GM[bf16, ('TK', 'D')], v: GM[bf16, ('TK', 'D')], out: GM[f32, ('TQ', 'D')],
    rowmax: GM[f32, ('TST',)], rowsum: GM[f32, ('TST',)],
    S1: i32, S2: i32, D: i32, BHQ: i32, HQ: i32, HKV: i32, scale: f32,
):
    score_ws = GMBuff(DT.float, [GROUP_N * TILE_M, TILE_N], slots=RING_GROUP_STAGE_SLOTS, name="score_ws")
    p_ws = GMBuff(DT.bfloat16, [GROUP_N * TILE_M, TILE_N], slots=RING_GROUP_STAGE_SLOTS, name="p_ws")
    pv_ws = GMBuff(DT.float, [TILE_M, TILE_K], slots=RING_GROUP_STAGE_SLOTS, name="pv_ws")

    l1q = DBuff(DT.bfloat16, [TILE_M, TILE_K], Position.L1)
    l1k = DBuff(DT.bfloat16, [TILE_N, TILE_K], Position.L1)
    l1p = DBuff(DT.bfloat16, [TILE_M, TILE_N], Position.L1)
    l1v = Tensor(DT.bfloat16, [V_PRELOAD_SLOTS * TILE_N, TILE_K], Position.L1)
    l0c = DBuff(DT.float, [TILE_M, TILE_N], Position.L0C)

    ub_score_group = QBuff(DT.float, [ROW_CHUNK, TILE_N], Position.UB)
    ub_p_group = QBuff(DT.bfloat16, [ROW_CHUNK, TILE_N], Position.UB)
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
    expdiff_store = Tensor(DT.float, [RING_GROUP_STAGE_SLOTS * 2, ROW_CHUNK_VEC], Position.UB)

    qk_mutex = CvMutex(0, depth=RING_GROUP_STAGE_SLOTS, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)
    p_mutex = VcMutex(1, depth=RING_GROUP_STAGE_SLOTS, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE2)
    pv_mutex = CvMutex(2, depth=RING_GROUP_STAGE_SLOTS, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)

    l1k_cnt = Var(0)
    l1pv_cnt = Var(0)
    l0c_cnt = Var(0)
    q_cur = Var(0)

    # Stage-1/stage-2 cursors over this core's (m_tile, group) stream. The cube
    # and vec sides each keep a PRIVATE copy advanced only on their own side, so
    # the side-splitter never has to track a loop-carried scalar across the cut.
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
    total_m = Var(BHQ * tiles_m)
    n_mtiles = CeilDiv(Max(total_m - cube_idx, 0), core_step)

    # GQA/MQA: HQ query heads share HKV kv heads in groups of group_size = HQ/HKV.
    # A query batch-head `bh` (0..BHQ-1) maps to the kv batch-head
    #   kv_bh = (bh // HQ) * HKV + (bh % HQ) // group_size.
    # Work distribution, Q/out/rowmax/rowsum are all indexed by the QUERY bh, so
    # only the cube-side K/V GM base addresses change vs the MHA kernel. MQA is the
    # HKV=1 special case: every query head maps to kv_bh = bh // HQ (= 0 for B=1).
    group_size = Var(HQ // HKV)

    # Full-mask: every M-tile visits all tiles_n key-tiles in ag_const groups, so
    # the snake-aware per-M-tile pre-pass of v6 collapses to one multiply.
    ag_const = CeilDiv(tiles_n, GROUP_N)
    total_groups = Var(n_mtiles * ag_const)

    for g in range(0, total_groups + RING_GROUP_LOOKAHEAD):
        stage1_slot = var_mod(g, RING_GROUP_STAGE_SLOTS)
        stage2_slot = var_mod(g - RING_GROUP_LOOKAHEAD, RING_GROUP_STAGE_SLOTS)

        # Reinitialise accum before this M-tile's first stage-2 group; the WAR
        # against the previous M-tile's writeback is autosync's loop-carried case.
        if g >= RING_GROUP_LOOKAHEAD and s2gv == 0:
            with auto_sync():
                dup(accum_ub, 0.0)

        if g < total_groups:
            # ============ STAGE 1: QK -> softmax -> P for stream group g ============
            # --- cube-side metadata (private cube cursor s1kc/s1gc) ---
            gmt1c = Var(cube_idx + s1kc * core_step)
            bh1c = Var(gmt1c // tiles_m)
            lmt1c = Var(gmt1c % tiles_m)
            q_row1 = Var(bh1c * S1 + lmt1c * TILE_M)
            kv_bh1c = Var((bh1c // HQ) * HKV + (var_mod(bh1c, HQ) // group_size))
            kv_base1 = Var(kv_bh1c * S2)
            valid_m1c = Min(TILE_M, S1 - lmt1c * TILE_M)
            group_start_c = Var(s1gc * GROUP_N)
            group_len_c = Min(GROUP_N, tiles_n - group_start_c)

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
            s1kc += (s1gc + 1) // ag_const
            q_cur += (s1gc + 1) // ag_const
            s1gc <<= var_mod(s1gc + 1, ag_const)

            # --- vec-side metadata (private vec cursor s1kv/s1gv) ---
            gmt1v = Var(cube_idx + s1kv * core_step)
            bh1v = Var(gmt1v // tiles_m)
            lmt1v = Var(gmt1v % tiles_m)
            valid_m1v = Min(TILE_M, S1 - lmt1v * TILE_M)
            group_start_v = Var(s1gv * GROUP_N)
            group_len_v = Min(GROUP_N, tiles_n - group_start_v)
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
                            if valid_n < TILE_N:
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
            s1kv += (s1gv + 1) // ag_const
            s1gv <<= var_mod(s1gv + 1, ag_const)

        if g >= RING_GROUP_LOOKAHEAD:
            # ============ STAGE 2: P@V -> accum for stream group (g-LOOKAHEAD) ============
            # --- cube-side metadata (private cube cursor s2kc/s2gc) ---
            gmt2c = Var(cube_idx + s2kc * core_step)
            bh2c = Var(gmt2c // tiles_m)
            kv_bh2c = Var((bh2c // HQ) * HKV + (var_mod(bh2c, HQ) // group_size))
            kv_base2 = Var(kv_bh2c * S2)
            group_start2 = Var(s2gc * GROUP_N)

            # --- vec-side metadata (private vec cursor s2kv/s2gv) ---
            gmt2v = Var(cube_idx + s2kv * core_step)
            bh2v = Var(gmt2v // tiles_m)
            lmt2v = Var(gmt2v % tiles_m)
            q_row2 = Var(bh2v * S1 + lmt2v * TILE_M)
            valid_m2v = Min(TILE_M, S1 - lmt2v * TILE_M)
            local_valid_m2 = Min(HALF_M, Max(valid_m2v - sb_row, 0))
            mslot2 = var_mod(s2kv, MTILE_SLOTS)

            with cube_scope():
                with auto_sync():
                    preload_l1pv_cnt = Var(l1pv_cnt)
                    for gi in range(0, GROUP_N):
                        if group_start2 + gi < tiles_n:
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
                            # unloaded rows with finite in-range V so the product stays
                            # NaN-free on simulator and hardware.
                            if pre_valid_n < TILE_N:
                                pad_rows = Var(TILE_N - pre_valid_n)
                                l1v[pre_v_base + pre_valid_n:pre_v_base + TILE_N, 0:TILE_K] <<= v[kv_base2:kv_base2 + pad_rows, 0:D]
                            preload_l1pv_cnt += 1

                    p_mutex.wait()
                    for gi in range(0, GROUP_N):
                        if group_start2 + gi < tiles_n:
                            v_slot = var_mod(l1pv_cnt, V_PRELOAD_SLOTS)
                            v_base = Var(v_slot * TILE_N)
                            l1p[l1pv_cnt] <<= p_ws[g - RING_GROUP_LOOKAHEAD, gi * TILE_M:gi * TILE_M + TILE_M, 0:TILE_N]
                            if gi == 0:
                                matmul(l0c[l0c_cnt], l1p[l1pv_cnt], l1v[v_base:v_base + TILE_N, 0:TILE_K].T, is_init=True)
                            else:
                                matmul(l0c[l0c_cnt], l1p[l1pv_cnt], l1v[v_base:v_base + TILE_N, 0:TILE_K].T, is_init=False)
                            l1pv_cnt += 1
                    p_mutex.free()

                    pv_mutex.lock()
                    pv_ws[g - RING_GROUP_LOOKAHEAD, 0:TILE_M, 0:D] <<= l0c[l0c_cnt]
                    pv_mutex.ready()
                    l0c_cnt += 1

            # cube cursor advance (cube-private)
            s2kc += (s2gc + 1) // ag_const
            s2gc <<= var_mod(s2gc + 1, ag_const)

            with auto_sync():
                pv_mutex.wait()
                ub_pv <<= pv_ws[g - RING_GROUP_LOOKAHEAD, sb_row:sb_row + HALF_M, 0:D]
                pv_mutex.free()

                if s2gv != 0:
                    exp_row2 = Var(stage2_slot * 2)
                    brcb(ub_expdiff[0:ROW_CHUNK, 0:8], expdiff_store[exp_row2:exp_row2 + 1, 0:ROW_CHUNK], repeat=ROW_CHUNK // 8, dst_blk_stride=1, dst_rep_stride=8)
                    brcb(ub_expdiff[ROW_CHUNK:HALF_M, 0:8], expdiff_store[exp_row2 + 1:exp_row2 + 2, 0:ROW_CHUNK], repeat=ROW_CHUNK // 8, dst_blk_stride=1, dst_rep_stride=8)
                    mul(accum_ub[0:HALF_M, 0:HALF_N], accum_ub[0:HALF_M, 0:HALF_N], ub_expdiff)
                    mul(accum_ub[0:HALF_M, HALF_N:TILE_K], accum_ub[0:HALF_M, HALF_N:TILE_K], ub_expdiff)
                add(accum_ub, accum_ub, ub_pv)

            if (s2gv + 1) // ag_const == 1:
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
                        stats_row = Var(bh2v * CeilDiv(S1, 8) * 8 + lmt2v * TILE_M + sb_row)
                        out[out_row:out_row + local_valid_m2, 0:D] <<= accum_ub[0:local_valid_m2, 0:D]
                        first_rows = Min(local_valid_m2, ROW_CHUNK)
                        rowmax[stats_row:stats_row + first_rows] <<= ub_rmax_s[rsum_row0:rsum_row0 + 1, 0:first_rows]
                        rowsum[stats_row:stats_row + first_rows] <<= ub_rsum_s[rsum_row0:rsum_row0 + 1, 0:first_rows]
                        if local_valid_m2 > ROW_CHUNK:
                            second_rows = Var(local_valid_m2 - ROW_CHUNK)
                            rowmax[stats_row + ROW_CHUNK:stats_row + ROW_CHUNK + second_rows] <<= ub_rmax_s[rsum_row1:rsum_row1 + 1, 0:second_rows]
                            rowsum[stats_row + ROW_CHUNK:stats_row + ROW_CHUNK + second_rows] <<= ub_rsum_s[rsum_row1:rsum_row1 + 1, 0:second_rows]
            # vec cursor advance (vec-private)
            s2kv += (s2gv + 1) // ag_const
            s2gv <<= var_mod(s2gv + 1, ag_const)

    return out, rowmax, rowsum





@lru_cache(maxsize=2)
def build_gqa_bf16_kernel(device):
    if device not in ("a2", "a3"):
        raise ValueError("Shared tensor-vector source supports A2/A3")
    return import_module(f"ascriptor.{device}").kernel()(flash_attn_fullmask_gqa_bf16_kernel)

# ----------------------------------------------------------------------------------------------------
# mha_bf16.py
# ----------------------------------------------------------------------------------------------------

# RFC-0009 batch-2 rewrite: the six hand-written events are gone (autosync guards the l1q ring,
# the group QBuffs, the accum reinitialisation and the store flushes), the stage1_cnt/stage2_cnt
# pair collapses into the single stream beat `g` (producer g, consumer g - RING_GROUP_LOOKAHEAD; the
# slot row-indices of ub_rmax_s/expdiff_store keep the same arithmetic on g), and the workspace
# rings are GMBuffs (the GROUP_N slot dim row-flattened) checked by the gmbuff pass. The bare
# GMTensor signature is modernised - this kernel had never compiled in ascriptor before.











def flash_attn_fullmask_bf16_kernel(
    q: GM[bf16, ('TQ', 'D')], k: GM[bf16, ('TK', 'D')], v: GM[bf16, ('TK', 'D')], out: GM[f32, ('TQ', 'D')],
    rowmax: GM[f32, ('TST',)], rowsum: GM[f32, ('TST',)],
    S1: i32, S2: i32, D: i32, BH: i32, scale: f32,
):
    score_ws = GMBuff(DT.float, [GROUP_N * TILE_M, TILE_N], slots=RING_GROUP_STAGE_SLOTS, name="score_ws")
    p_ws = GMBuff(DT.bfloat16, [GROUP_N * TILE_M, TILE_N], slots=RING_GROUP_STAGE_SLOTS, name="p_ws")
    pv_ws = GMBuff(DT.float, [TILE_M, TILE_K], slots=RING_GROUP_STAGE_SLOTS, name="pv_ws")

    l1q = DBuff(DT.bfloat16, [TILE_M, TILE_K], Position.L1)
    l1k = DBuff(DT.bfloat16, [TILE_N, TILE_K], Position.L1)
    l1p = DBuff(DT.bfloat16, [TILE_M, TILE_N], Position.L1)
    l1v = Tensor(DT.bfloat16, [V_PRELOAD_SLOTS * TILE_N, TILE_K], Position.L1)
    l0c = DBuff(DT.float, [TILE_M, TILE_N], Position.L0C)

    ub_score_group = QBuff(DT.float, [ROW_CHUNK, TILE_N], Position.UB)
    ub_p_group = QBuff(DT.bfloat16, [ROW_CHUNK, TILE_N], Position.UB)
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
    expdiff_store = Tensor(DT.float, [RING_GROUP_STAGE_SLOTS * 2, ROW_CHUNK_VEC], Position.UB)
    # 2-slot l1q ring.

    qk_mutex = CvMutex(0, depth=RING_GROUP_STAGE_SLOTS, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)
    p_mutex = VcMutex(1, depth=RING_GROUP_STAGE_SLOTS, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE2)
    pv_mutex = CvMutex(2, depth=RING_GROUP_STAGE_SLOTS, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)

    l1k_cnt = Var(0)
    l1pv_cnt = Var(0)
    l0c_cnt = Var(0)
    q_cur = Var(0)

    # Stage-1/stage-2 cursors over this core's (m_tile, group) stream. The cube
    # and vec sides each keep a PRIVATE copy advanced only on their own side, so
    # the side-splitter never has to track a loop-carried scalar across the cut.
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

    # Full-mask: every M-tile visits all tiles_n key-tiles in ag_const groups, so
    # the snake-aware per-M-tile pre-pass of v6 collapses to one multiply.
    ag_const = CeilDiv(tiles_n, GROUP_N)
    total_groups = Var(n_mtiles * ag_const)

    for g in range(0, total_groups + RING_GROUP_LOOKAHEAD):
        stage1_slot = var_mod(g, RING_GROUP_STAGE_SLOTS)
        stage2_slot = var_mod(g - RING_GROUP_LOOKAHEAD, RING_GROUP_STAGE_SLOTS)

        # Reinitialise accum before this M-tile's first stage-2 group; the WAR
        # against the previous M-tile's writeback is autosync's loop-carried case.
        if g >= RING_GROUP_LOOKAHEAD and s2gv == 0:
            with auto_sync():
                dup(accum_ub, 0.0)

        if g < total_groups:
            # ============ STAGE 1: QK -> softmax -> P for stream group g ============
            # --- cube-side metadata (private cube cursor s1kc/s1gc) ---
            gmt1c = Var(cube_idx + s1kc * core_step)
            bh1c = Var(gmt1c // tiles_m)
            lmt1c = Var(gmt1c % tiles_m)
            q_row1 = Var(bh1c * S1 + lmt1c * TILE_M)
            kv_base1 = Var(bh1c * S2)
            valid_m1c = Min(TILE_M, S1 - lmt1c * TILE_M)
            group_start_c = Var(s1gc * GROUP_N)
            group_len_c = Min(GROUP_N, tiles_n - group_start_c)

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
            s1kc += (s1gc + 1) // ag_const
            q_cur += (s1gc + 1) // ag_const
            s1gc <<= var_mod(s1gc + 1, ag_const)

            # --- vec-side metadata (private vec cursor s1kv/s1gv) ---
            gmt1v = Var(cube_idx + s1kv * core_step)
            bh1v = Var(gmt1v // tiles_m)
            lmt1v = Var(gmt1v % tiles_m)
            valid_m1v = Min(TILE_M, S1 - lmt1v * TILE_M)
            group_start_v = Var(s1gv * GROUP_N)
            group_len_v = Min(GROUP_N, tiles_n - group_start_v)
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
                            if valid_n < TILE_N:
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
            s1kv += (s1gv + 1) // ag_const
            s1gv <<= var_mod(s1gv + 1, ag_const)

        if g >= RING_GROUP_LOOKAHEAD:
            # ============ STAGE 2: P@V -> accum for stream group (g-LOOKAHEAD) ============
            # --- cube-side metadata (private cube cursor s2kc/s2gc) ---
            gmt2c = Var(cube_idx + s2kc * core_step)
            bh2c = Var(gmt2c // tiles_m)
            kv_base2 = Var(bh2c * S2)
            group_start2 = Var(s2gc * GROUP_N)

            # --- vec-side metadata (private vec cursor s2kv/s2gv) ---
            gmt2v = Var(cube_idx + s2kv * core_step)
            bh2v = Var(gmt2v // tiles_m)
            lmt2v = Var(gmt2v % tiles_m)
            q_row2 = Var(bh2v * S1 + lmt2v * TILE_M)
            valid_m2v = Min(TILE_M, S1 - lmt2v * TILE_M)
            local_valid_m2 = Min(HALF_M, Max(valid_m2v - sb_row, 0))
            mslot2 = var_mod(s2kv, MTILE_SLOTS)

            with cube_scope():
                with auto_sync():
                    preload_l1pv_cnt = Var(l1pv_cnt)
                    for gi in range(0, GROUP_N):
                        if group_start2 + gi < tiles_n:
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
                            # unloaded rows with finite in-range V so the product stays
                            # NaN-free on simulator and hardware.
                            if pre_valid_n < TILE_N:
                                pad_rows = Var(TILE_N - pre_valid_n)
                                l1v[pre_v_base + pre_valid_n:pre_v_base + TILE_N, 0:TILE_K] <<= v[kv_base2:kv_base2 + pad_rows, 0:D]
                            preload_l1pv_cnt += 1

                    p_mutex.wait()
                    for gi in range(0, GROUP_N):
                        if group_start2 + gi < tiles_n:
                            v_slot = var_mod(l1pv_cnt, V_PRELOAD_SLOTS)
                            v_base = Var(v_slot * TILE_N)
                            l1p[l1pv_cnt] <<= p_ws[g - RING_GROUP_LOOKAHEAD, gi * TILE_M:gi * TILE_M + TILE_M, 0:TILE_N]
                            if gi == 0:
                                matmul(l0c[l0c_cnt], l1p[l1pv_cnt], l1v[v_base:v_base + TILE_N, 0:TILE_K].T, is_init=True)
                            else:
                                matmul(l0c[l0c_cnt], l1p[l1pv_cnt], l1v[v_base:v_base + TILE_N, 0:TILE_K].T, is_init=False)
                            l1pv_cnt += 1
                    p_mutex.free()

                    pv_mutex.lock()
                    pv_ws[g - RING_GROUP_LOOKAHEAD, 0:TILE_M, 0:D] <<= l0c[l0c_cnt]
                    pv_mutex.ready()
                    l0c_cnt += 1

            # cube cursor advance (cube-private)
            s2kc += (s2gc + 1) // ag_const
            s2gc <<= var_mod(s2gc + 1, ag_const)

            with auto_sync():
                pv_mutex.wait()
                ub_pv <<= pv_ws[g - RING_GROUP_LOOKAHEAD, sb_row:sb_row + HALF_M, 0:D]
                pv_mutex.free()

                if s2gv != 0:
                    exp_row2 = Var(stage2_slot * 2)
                    brcb(ub_expdiff[0:ROW_CHUNK, 0:8], expdiff_store[exp_row2:exp_row2 + 1, 0:ROW_CHUNK], repeat=ROW_CHUNK // 8, dst_blk_stride=1, dst_rep_stride=8)
                    brcb(ub_expdiff[ROW_CHUNK:HALF_M, 0:8], expdiff_store[exp_row2 + 1:exp_row2 + 2, 0:ROW_CHUNK], repeat=ROW_CHUNK // 8, dst_blk_stride=1, dst_rep_stride=8)
                    mul(accum_ub[0:HALF_M, 0:HALF_N], accum_ub[0:HALF_M, 0:HALF_N], ub_expdiff)
                    mul(accum_ub[0:HALF_M, HALF_N:TILE_K], accum_ub[0:HALF_M, HALF_N:TILE_K], ub_expdiff)
                add(accum_ub, accum_ub, ub_pv)

            if (s2gv + 1) // ag_const == 1:
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
                        stats_row = Var(bh2v * CeilDiv(S1, 8) * 8 + lmt2v * TILE_M + sb_row)
                        out[out_row:out_row + local_valid_m2, 0:D] <<= accum_ub[0:local_valid_m2, 0:D]
                        first_rows = Min(local_valid_m2, ROW_CHUNK)
                        rowmax[stats_row:stats_row + first_rows] <<= ub_rmax_s[rsum_row0:rsum_row0 + 1, 0:first_rows]
                        rowsum[stats_row:stats_row + first_rows] <<= ub_rsum_s[rsum_row0:rsum_row0 + 1, 0:first_rows]
                        if local_valid_m2 > ROW_CHUNK:
                            second_rows = Var(local_valid_m2 - ROW_CHUNK)
                            rowmax[stats_row + ROW_CHUNK:stats_row + ROW_CHUNK + second_rows] <<= ub_rmax_s[rsum_row1:rsum_row1 + 1, 0:second_rows]
                            rowsum[stats_row + ROW_CHUNK:stats_row + ROW_CHUNK + second_rows] <<= ub_rsum_s[rsum_row1:rsum_row1 + 1, 0:second_rows]
            # vec cursor advance (vec-private)
            s2kv += (s2gv + 1) // ag_const
            s2gv <<= var_mod(s2gv + 1, ag_const)

    return out, rowmax, rowsum





@lru_cache(maxsize=2)
def build_mha_bf16_kernel(device):
    if device not in ("a2", "a3"):
        raise ValueError("Shared tensor-vector source supports A2/A3")
    return import_module(f"ascriptor.{device}").kernel()(flash_attn_fullmask_bf16_kernel)

# ----------------------------------------------------------------------------------------------------
# mha_d256.py
# MHA Flash Attention, D=256, A2/910B3, bf16 inputs.
#
# Target spec: B=1, H=8, S=2049, D=256 (S1=S2=2049, non-aligned on both axes).
#
# Design notes (see the UB budget below):
#   - D=256 makes the two [D256_HALF_M, 256] fp32 buffers (accum numerator + PV landing)
#     the UB bottleneck. With D256_TILE_M=128 that is 64KB each and the whole layout
#     overflows the 192KB/sub-block A2 UB. We therefore use D256_TILE_M=64 (D256_HALF_M=32):
#     each big buffer is only 32KB, the whole layout is ~100KB, and accum / ub_pv
#     stay single [32,256] blocks (no lo/hi split needed on the vec side).
#   - QK contracts over D=256, split into a 2x128 K-loop (accumulating matmul).
#   - L0C on A2 is 128KB; a [64,256] fp32 PV tile is 64KB. To keep QK and PV on one
#     shared [64,128] L0C DBuff (pattern a2-mixed-pipeline Section 2 "Shared L0C"),
#     PV is emitted as two [64,128] matmuls (lo/hi halves of D) published into the
#     two D-column regions of one pv_ws slot; the vec side then reads a single
#     [32,256] ub_pv.
#   - S=2049 is non-aligned on S1 (rows) and S2 (columns). Tail handling follows
#     flash_attn_full_pj_hif8.py: S2 column mask before cmax, S1 row mask after the
#     max-shift and before exp, GM writeback limited to local_valid_m rows.
#   - is_causal is carried in the signature; the causal path is added in a later
#     stage. This file validates the non-causal (full) path first.
#
# Kernel contract:
#   q:   [BH*S1, 256]  flat, DT.bfloat16
#   k:   [BH*S2, 256]  flat, DT.bfloat16
#   v:   [BH*S2, 256]  flat, DT.bfloat16
#   out: [BH*S1, 256]  flat, DT.float32
#   Scalar Vars: total_q, total_kv, D, S1, S2, BH, scale, is_causal
# ----------------------------------------------------------------------------------------------------

D256_TILE_M = 64
D256_HALF_M = D256_TILE_M // 2      # 32: rows owned by one vec sub-block
D_VAL = 256
D256_GROUP_STAGE_SLOTS = 4
D256_GROUP_LOOKAHEAD = 2
# A2 vec processes a full 256B vector = 64 fp32 lanes per op. D256_HALF_M=32 is only
# half a vector, so row-scalar [1, D256_HALF_M] buffers would overrun on dup/elementwise
# ops. Size them to a full vector width and use only the first D256_HALF_M lanes.
SCALAR_W = 64
# block32 causal: a query attends to keys in its own 32-block and all earlier
# blocks (k_block <= q_block). D256_HALF_M=32 == one 32-block, so each vec sub-block
# owns exactly one q_block and the mask is column-uniform per 32-col k-block.
BLOCK_CAUSAL = 32
KBLK_M = D256_TILE_M // BLOCK_CAUSAL   # 2 q-blocks per M tile
KBLK_N = TILE_N // BLOCK_CAUSAL   # 4 k-blocks per N tile


@func()
def d256_build_suffix_invalid_mask(valid_cols: Var, out_mask: Var):
    """uint64 mask whose lowest `valid_cols` bits are 0, the rest 1 (invalid)."""
    signed_mask = Var(-1, DT.int64)
    two_i64 = Var(2, DT.int64)
    for _ in range(0, valid_cols):
        signed_mask <<= signed_mask * two_i64
    out_mask <<= signed_mask


@func()
def d256_mask_score_half_suffix_invalid(score_half: Tensor, valid_cols: Var):
    """Force the invalid suffix of one 64-column score half to NEG_LARGE."""
    if valid_cols == 0:
        dup(score_half, NEG_LARGE)
    elif valid_cols < HALF_N:
        suffix_mask = Var(0, DT.uint64)
        d256_build_suffix_invalid_mask(valid_cols, suffix_mask)
        set_mask(0, suffix_mask)
        dup(score_half, NEG_LARGE)
        reset_mask()


@func()
def d256_apply_score_tail_mask(ub_score: Tensor, valid_n: Var):
    """S2 column tail: invalid columns behave like -inf before cmax."""
    left_valid = Min(valid_n, HALF_N)
    right_valid = Max(valid_n - HALF_N, 0)
    d256_mask_score_half_suffix_invalid(ub_score[0:D256_HALF_M, 0:HALF_N], left_valid)
    d256_mask_score_half_suffix_invalid(ub_score[0:D256_HALF_M, HALF_N:TILE_N], right_valid)


@func()
def d256_apply_score_tail_mask_chunk(ub_score: Tensor, valid_n: Var):
    """S2/causal suffix mask for one ROW_CHUNK score tile."""
    left_valid = Min(valid_n, HALF_N)
    right_valid = Max(valid_n - HALF_N, 0)
    d256_mask_score_half_suffix_invalid(ub_score[0:ROW_CHUNK, 0:HALF_N], left_valid)
    d256_mask_score_half_suffix_invalid(ub_score[0:ROW_CHUNK, HALF_N:TILE_N], right_valid)


@func()
def d256_apply_score_row_tail_mask_after_shift(ub_score: Tensor, valid_rows: Var):
    """S1 row tail: invalid local rows -> NEG_LARGE (become 0 after exp)."""
    if valid_rows == 0:
        dup(ub_score, NEG_LARGE)
    elif valid_rows < D256_HALF_M:
        dup(ub_score[valid_rows:D256_HALF_M, 0:TILE_N], NEG_LARGE)


@func()
def d256_apply_score_row_tail_mask_chunk_after_shift(ub_score: Tensor, valid_rows: Var):
    """S1 row tail for one ROW_CHUNK score tile."""
    if valid_rows == 0:
        dup(ub_score, NEG_LARGE)
    elif valid_rows < ROW_CHUNK:
        dup(ub_score[valid_rows:ROW_CHUNK, 0:TILE_N], NEG_LARGE)


def mha_flash_d256(
    q: GM[bf16, ('total_q', 'D')], k: GM[bf16, ('total_kv', 'D')], v: GM[bf16, ('total_kv', 'D')], out: GM[f32, ('total_q', 'D')],
    total_q: i32, total_kv: i32, D: i32, S1: i32, S2: i32,
    BH: i32, scale: f32, is_causal: i32,
):
    score_ws = GMBuff(DT.float, [D256_TILE_M, GROUP_N * TILE_N], slots=D256_GROUP_STAGE_SLOTS, name="score_ws")
    p_ws = GMBuff(DT.bfloat16, [D256_TILE_M, GROUP_N * TILE_N], slots=D256_GROUP_STAGE_SLOTS, name="p_ws")
    pv_ws = GMBuff(DT.float, [D256_TILE_M, D_VAL], slots=D256_GROUP_STAGE_SLOTS, name="pv_ws")

    l1q = DBuff(DT.bfloat16, [D256_TILE_M, TILE_K], Position.L1)
    l1k = DBuff(DT.bfloat16, [TILE_N, TILE_K], Position.L1)
    l1p = DBuff(DT.bfloat16, [D256_TILE_M, TILE_N], Position.L1)
    l1v = DBuff(DT.bfloat16, [TILE_N, TILE_K], Position.L1)
    l0c = DBuff(DT.float, [D256_TILE_M, TILE_N], Position.L0C)
    l0c_hi = DBuff(DT.float, [D256_TILE_M, TILE_N], Position.L0C)

    ub_score_group = QBuff(DT.float, [ROW_CHUNK, TILE_N], Position.UB)
    ub_p_group = QBuff(DT.bfloat16, [ROW_CHUNK, TILE_N], Position.UB)
    ub_pv = Tensor(DT.float, [D256_HALF_M, D_VAL], Position.UB)
    ub_tmp = Tensor(DT.float, [ROW_CHUNK, HALF_N], Position.UB)
    ub_max_s = Tensor(DT.float, [1, SCALAR_W], Position.UB)
    ub_group_max_s = Tensor(DT.float, [1, SCALAR_W], Position.UB)
    ub_old_max_s = Tensor(DT.float, [1, SCALAR_W], Position.UB)
    ub_rmax_s = Tensor(DT.float, [1, SCALAR_W], Position.UB)
    ub_sum_s = Tensor(DT.float, [1, SCALAR_W], Position.UB)
    ub_group_sum_s = Tensor(DT.float, [1, SCALAR_W], Position.UB)
    ub_rsum_s = Tensor(DT.float, [1, SCALAR_W], Position.UB)
    ub_zero_s = Tensor(DT.float, [1, SCALAR_W], Position.UB)
    ub_zero_chunk_s = Tensor(DT.float, [1, SCALAR_W], Position.UB)
    ub_max = Tensor(DT.float, [ROW_CHUNK, 8], Position.UB)
    ub_rowsum = Tensor(DT.float, [D256_HALF_M, 8], Position.UB)
    ub_expdiff = Tensor(DT.float, [D256_HALF_M, 8], Position.UB)
    ub_group_expdiff_chunk_s = Tensor(DT.float, [1, SCALAR_W], Position.UB)
    accum_ub = Tensor(DT.float, [D256_HALF_M, D_VAL], Position.UB)
    expdiff_buf = QBuff(DT.float, [1, SCALAR_W], Position.UB)

    qk_mutex = CvMutex(0, depth=D256_GROUP_STAGE_SLOTS, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)
    p_mutex = VcMutex(1, depth=D256_GROUP_STAGE_SLOTS, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE2)
    pv_mutex = CvMutex(2, depth=D256_GROUP_STAGE_SLOTS, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)

    l1k_cnt = Var(0)
    l1pv_cnt = Var(0)
    l1v_cnt = Var(0)
    l0c_cnt = Var(0)
    l0c_hi_cnt = Var(0)

    cube_idx = GetCubeIdx()
    sb = GetSubBlockIdx()
    sb_row = Var(sb * D256_HALF_M)

    tiles_m = CeilDiv(S1, D256_TILE_M)
    tiles_n = CeilDiv(S2, TILE_N)
    total_m = Var(BH * tiles_m)
    core_count = GetCubeNum()
    core_step = Var(1, DT.int)
    core_step <<= core_count

    for gmt in range(cube_idx, total_m, core_step):
        bh = Var(gmt // tiles_m)
        lmt = Var(gmt % tiles_m)
        q_row = Var(bh * S1 + lmt * D256_TILE_M)
        kv_base = Var(bh * S2)
        valid_m = Min(D256_TILE_M, S1 - lmt * D256_TILE_M)
        local_valid_m = Min(D256_HALF_M, Max(valid_m - sb_row, 0))
        q_block = Var(lmt * KBLK_M + sb)   # this sub-block's single 32-block index

        # All N tiles are processed. Causal correctness comes from the per-tile
        # column mask below (a fully-upper-triangle tile gets eff_valid_n=0 -> all
        # columns masked -> P=0 -> no contribution). We do NOT shrink the inner loop
        # bound for causal: a per-M-tile-reduced `range(0, active+1)` deadlocks the
        # cube/vec handoff in this D256_TILE_M=64 lo/hi-PV pipeline (validated: the reduced
        # bound hangs, the full bound is correct). Skipping upper-triangle tiles is a
        # future optimization that needs the reduced-bound sync issue rooted first.
        active_tiles_n = tiles_n
        active_groups = CeilDiv(active_tiles_n, GROUP_N)

        # Previous MTE3 writeback still reads accum_ub / ub_rmax_s / ub_rsum_s;
        # autosync's loop-carried machinery guards the reinitialisation.
        with auto_sync():
            dup(ub_rmax_s, NEG_LARGE)
            dup(ub_rsum_s, 0.0)
            dup(ub_zero_s, 0.0)
            dup(ub_zero_chunk_s, 0.0, count=ROW_CHUNK)
            dup(accum_ub, 0.0)

        with auto_sync():
            l1q[0] <<= q[q_row:q_row + valid_m, 0:TILE_K]
            l1q[1] <<= q[q_row:q_row + valid_m, TILE_K:D_VAL]

        for group_id in range(0, active_groups + D256_GROUP_LOOKAHEAD):
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

                            l1k[l1k_cnt] <<= k[kv_row:kv_row + valid_n, 0:TILE_K]
                            matmul(
                                l0c[l0c_cnt], l1q[0], l1k[l1k_cnt],
                                m=D256_TILE_M, n=TILE_N, k=TILE_K, is_init=True,
                            )
                            l1k_cnt += 1
                            l1k[l1k_cnt] <<= k[kv_row:kv_row + valid_n, TILE_K:D_VAL]
                            matmul(
                                l0c[l0c_cnt], l1q[1], l1k[l1k_cnt],
                                m=D256_TILE_M, n=TILE_N, k=TILE_K, is_init=False,
                            )
                            l1k_cnt += 1

                            group_col = gi * TILE_N
                            score_ws[
                                group_id, 0:D256_TILE_M, group_col:group_col + TILE_N
                            ] <<= l0c[l0c_cnt]
                            l0c_cnt += 1
                            bar_m()
                qk_mutex.ready()
                qk_mutex.wait()
                p_mutex.lock()

                for rb in unroll(0, D256_HALF_M, ROW_CHUNK):
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

                                ub_score <<= score_ws[
                                    group_id, chunk_row:chunk_row + ROW_CHUNK, group_col:group_col + TILE_N
                                ]
                                muls(ub_score, ub_score, scale)

                                causal_cols = Min(Max((q_block - ni * KBLK_N + 1) * BLOCK_CAUSAL, 0), TILE_N)
                                eff_valid_n = Min(valid_n, causal_cols + (1 - is_causal) * TILE_N)
                                if eff_valid_n < TILE_N:
                                    d256_apply_score_tail_mask_chunk(ub_score, eff_valid_n)

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

                    brcb(
                        ub_max, ub_rmax_s[0:1, rb:rb + ROW_CHUNK],
                        repeat=ROW_CHUNK // 8, dst_blk_stride=1, dst_rep_stride=8,
                    )
                    for gi in range(0, GROUP_N):
                        if gi < group_len:
                            with auto_sync():
                                ub_p = ub_p_group[gi]
                                ub_score = ub_score_group[gi]
                                group_col = gi * TILE_N

                                sub(ub_score[0:ROW_CHUNK, 0:HALF_N], ub_score[0:ROW_CHUNK, 0:HALF_N], ub_max)
                                sub(ub_score[0:ROW_CHUNK, HALF_N:TILE_N], ub_score[0:ROW_CHUNK, HALF_N:TILE_N], ub_max)
                                if chunk_valid_m < ROW_CHUNK:
                                    d256_apply_score_row_tail_mask_chunk_after_shift(ub_score, chunk_valid_m)

                                exp(ub_score, ub_score)
                                add(ub_tmp, ub_score[0:ROW_CHUNK, 0:HALF_N], ub_score[0:ROW_CHUNK, HALF_N:TILE_N])
                                cadd(ub_sum_s, ub_tmp)
                                add(ub_group_sum_s, ub_group_sum_s, ub_sum_s, count=ROW_CHUNK)

                                cast(ub_p, ub_score)
                                p_ws[
                                    group_id, chunk_row:chunk_row + ROW_CHUNK, group_col:group_col + TILE_N
                                ] <<= ub_p

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

            if group_id >= D256_GROUP_LOOKAHEAD:

                with cube_scope():
                    with auto_sync():
                        p_mutex.wait()
                        group_start2 = Var((group_id - D256_GROUP_LOOKAHEAD) * GROUP_N)
                        group_len2 = Min(GROUP_N, active_tiles_n - group_start2)
                        for gi in range(0, GROUP_N):
                            if gi < group_len2:
                                ni = Var(group_start2 + gi)
                                n_off = Var(ni * TILE_N)
                                prev_valid_n = Min(TILE_N, S2 - n_off)
                                v_row = Var(kv_base + n_off)
                                group_col = gi * TILE_N

                                l1p[l1pv_cnt] <<= p_ws[
                                    group_id - D256_GROUP_LOOKAHEAD, 0:D256_TILE_M, group_col:group_col + TILE_N
                                ]

                                l1v[l1v_cnt][0:prev_valid_n, 0:TILE_K] <<= v[v_row:v_row + prev_valid_n, 0:TILE_K]
                                matmul(
                                    l0c[l0c_cnt], l1p[l1pv_cnt], l1v[l1v_cnt].T,
                                    m=D256_TILE_M, n=TILE_K, k=prev_valid_n, is_init=(gi == 0),
                                )
                                l1v_cnt += 1

                                l1v[l1v_cnt][0:prev_valid_n, 0:TILE_K] <<= v[v_row:v_row + prev_valid_n, TILE_K:D_VAL]
                                matmul(
                                    l0c_hi[l0c_hi_cnt], l1p[l1pv_cnt], l1v[l1v_cnt].T,
                                    m=D256_TILE_M, n=TILE_K, k=prev_valid_n, is_init=(gi == 0),
                                )
                                l1v_cnt += 1
                                l1pv_cnt += 1
                                bar_m()
                        p_mutex.free()

                        pv_mutex.lock()
                        pv_ws[group_id - D256_GROUP_LOOKAHEAD, 0:D256_TILE_M, 0:TILE_K] <<= l0c[l0c_cnt]
                        pv_ws[group_id - D256_GROUP_LOOKAHEAD, 0:D256_TILE_M, TILE_K:D_VAL] <<= l0c_hi[l0c_hi_cnt]
                        pv_mutex.ready()
                        l0c_cnt += 1
                        l0c_hi_cnt += 1

                with auto_sync():
                    pv_mutex.wait()
                    ub_pv <<= pv_ws[group_id - D256_GROUP_LOOKAHEAD, sb_row:sb_row + D256_HALF_M, 0:D_VAL]

                    brcb(ub_expdiff, expdiff_buf[group_id - D256_GROUP_LOOKAHEAD], repeat=D256_HALF_M // 8, dst_blk_stride=1, dst_rep_stride=8)
                    # accum *= expdiff (broadcast [32,8] over 256 cols in 4 x 64-col segments)
                    mul(accum_ub[0:D256_HALF_M, 0:HALF_N], accum_ub[0:D256_HALF_M, 0:HALF_N], ub_expdiff)
                    mul(accum_ub[0:D256_HALF_M, HALF_N:TILE_N], accum_ub[0:D256_HALF_M, HALF_N:TILE_N], ub_expdiff)
                    mul(accum_ub[0:D256_HALF_M, TILE_N:TILE_N + HALF_N], accum_ub[0:D256_HALF_M, TILE_N:TILE_N + HALF_N], ub_expdiff)
                    mul(accum_ub[0:D256_HALF_M, TILE_N + HALF_N:D_VAL], accum_ub[0:D256_HALF_M, TILE_N + HALF_N:D_VAL], ub_expdiff)
                    add(accum_ub, accum_ub, ub_pv)
                    pv_mutex.free()

        with auto_sync():
            brcb(ub_rowsum, ub_rsum_s, repeat=D256_HALF_M // 8, dst_blk_stride=1, dst_rep_stride=8)
            div(accum_ub[0:D256_HALF_M, 0:HALF_N], accum_ub[0:D256_HALF_M, 0:HALF_N], ub_rowsum)
            div(accum_ub[0:D256_HALF_M, HALF_N:TILE_N], accum_ub[0:D256_HALF_M, HALF_N:TILE_N], ub_rowsum)
            div(accum_ub[0:D256_HALF_M, TILE_N:TILE_N + HALF_N], accum_ub[0:D256_HALF_M, TILE_N:TILE_N + HALF_N], ub_rowsum)
            div(accum_ub[0:D256_HALF_M, TILE_N + HALF_N:D_VAL], accum_ub[0:D256_HALF_M, TILE_N + HALF_N:D_VAL], ub_rowsum)

            if local_valid_m > 0:
                out_row = Var(q_row + sb_row)
                out[out_row:out_row + local_valid_m, 0:D_VAL] <<= accum_ub[0:local_valid_m, 0:D_VAL]

    return out


# ---------------------------------------------------------------------------
# Reference implementation (matches the kernel precision path)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@lru_cache(maxsize=2)
def build_mha_d256_kernel(device):
    if device not in ("a2", "a3"):
        raise ValueError("Shared tensor-vector source supports A2/A3")
    return import_module(f"ascriptor.{device}").kernel()(mha_flash_d256)

# ----------------------------------------------------------------------------------------------------
# mha_d256_bf16.py
# MHA Flash Attention, D=256, A2/910B3, bf16 inputs and bf16 output.
#
# Target spec: B=1, H=8, S=2049, D=256 (S1=S2=2049, non-aligned on both axes).
#
# Design notes (see the UB budget below):
#   - D=256 makes the two [D256_HALF_M, 256] fp32 buffers (accum numerator + PV landing)
#     the UB bottleneck. With D256_TILE_M=128 that is 64KB each and the whole layout
#     overflows the 192KB/sub-block A2 UB. We therefore use D256_TILE_M=64 (D256_HALF_M=32):
#     each big fp32 buffer is only 32KB. The grouped score/P scratch plus the
#     bf16 output-cast landing buffer now use ~190.5KB, still below the
#     192KB/sub-block A2 UB budget, and accum / ub_pv stay single [32,256] blocks
#     (no lo/hi split needed on the vec side).
#   - QK contracts over D=256, split into a 2x128 K-loop (accumulating matmul).
#   - The inner loop groups up to four N tiles. Q is loaded once per M tile, scores
#     for the group are staged together, row max/sum are updated once per
#     ROW_CHUNK per group, and grouped PV accumulates low/high D halves into two
#     dedicated [64,128] L0C DBuffs before publishing one [64,256] pv_ws tile.
#   - S=2049 is non-aligned on S1 (rows) and S2 (columns). Tail handling follows
#     flash_attn_full_pj_hif8.py: S2 column mask before cmax, S1 row mask after the
#     max-shift and before exp, GM writeback limited to local_valid_m rows.
#   - Output is finalized in fp32, then explicitly cast to bf16 in UB before GM
#     writeback. The fp32-output companion stays in mha_flash_d256.py.
#
# Kernel contract:
#   q:   [BH*S1, 256]  flat, DT.bfloat16
#   k:   [BH*S2, 256]  flat, DT.bfloat16
#   v:   [BH*S2, 256]  flat, DT.bfloat16
#   out: [BH*S1, 256]  flat, DT.bfloat16
#   Scalar Vars: total_q, total_kv, D, S1, S2, BH, scale, is_causal
# ----------------------------------------------------------------------------------------------------

def mha_flash_d256_bf16(
    q: GM[bf16, ('total_q', 'D')], k: GM[bf16, ('total_kv', 'D')], v: GM[bf16, ('total_kv', 'D')], out: GM[bf16, ('total_q', 'D')],
    total_q: i32, total_kv: i32, D: i32, S1: i32, S2: i32,
    BH: i32, scale: f32, is_causal: i32,
):
    score_ws = GMBuff(DT.float, [D256_TILE_M, GROUP_N * TILE_N], slots=D256_GROUP_STAGE_SLOTS, name="score_ws")
    p_ws = GMBuff(DT.bfloat16, [D256_TILE_M, GROUP_N * TILE_N], slots=D256_GROUP_STAGE_SLOTS, name="p_ws")
    pv_ws = GMBuff(DT.float, [D256_TILE_M, D_VAL], slots=D256_GROUP_STAGE_SLOTS, name="pv_ws")

    l1q = DBuff(DT.bfloat16, [D256_TILE_M, TILE_K], Position.L1)
    l1k = DBuff(DT.bfloat16, [TILE_N, TILE_K], Position.L1)
    l1p = DBuff(DT.bfloat16, [D256_TILE_M, TILE_N], Position.L1)
    l1v = DBuff(DT.bfloat16, [TILE_N, TILE_K], Position.L1)
    l0c = DBuff(DT.float, [D256_TILE_M, TILE_N], Position.L0C)
    l0c_hi = DBuff(DT.float, [D256_TILE_M, TILE_N], Position.L0C)

    ub_score_group = QBuff(DT.float, [ROW_CHUNK, TILE_N], Position.UB)
    ub_p_group = QBuff(DT.bfloat16, [ROW_CHUNK, TILE_N], Position.UB)
    ub_pv = Tensor(DT.float, [D256_HALF_M, D_VAL], Position.UB)
    ub_tmp = Tensor(DT.float, [ROW_CHUNK, HALF_N], Position.UB)
    ub_max_s = Tensor(DT.float, [1, SCALAR_W], Position.UB)
    ub_group_max_s = Tensor(DT.float, [1, SCALAR_W], Position.UB)
    ub_old_max_s = Tensor(DT.float, [1, SCALAR_W], Position.UB)
    ub_rmax_s = Tensor(DT.float, [1, SCALAR_W], Position.UB)
    ub_sum_s = Tensor(DT.float, [1, SCALAR_W], Position.UB)
    ub_group_sum_s = Tensor(DT.float, [1, SCALAR_W], Position.UB)
    ub_rsum_s = Tensor(DT.float, [1, SCALAR_W], Position.UB)
    ub_zero_s = Tensor(DT.float, [1, SCALAR_W], Position.UB)
    ub_zero_chunk_s = Tensor(DT.float, [1, SCALAR_W], Position.UB)
    ub_max = Tensor(DT.float, [ROW_CHUNK, 8], Position.UB)
    ub_rowsum = Tensor(DT.float, [D256_HALF_M, 8], Position.UB)
    ub_expdiff = Tensor(DT.float, [D256_HALF_M, 8], Position.UB)
    ub_group_expdiff_chunk_s = Tensor(DT.float, [1, SCALAR_W], Position.UB)
    accum_ub = Tensor(DT.float, [D256_HALF_M, D_VAL], Position.UB)
    ub_out = Tensor(DT.bfloat16, [D256_HALF_M, D_VAL], Position.UB)
    expdiff_buf = QBuff(DT.float, [1, SCALAR_W], Position.UB)

    qk_mutex = CvMutex(0, depth=D256_GROUP_STAGE_SLOTS, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)
    p_mutex = VcMutex(1, depth=D256_GROUP_STAGE_SLOTS, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE2)
    pv_mutex = CvMutex(2, depth=D256_GROUP_STAGE_SLOTS, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)

    l1k_cnt = Var(0)
    l1pv_cnt = Var(0)
    l1v_cnt = Var(0)
    l0c_cnt = Var(0)
    l0c_hi_cnt = Var(0)

    cube_idx = GetCubeIdx()
    sb = GetSubBlockIdx()
    sb_row = Var(sb * D256_HALF_M)

    tiles_m = CeilDiv(S1, D256_TILE_M)
    tiles_n = CeilDiv(S2, TILE_N)
    total_m = Var(BH * tiles_m)
    core_count = GetCubeNum()
    core_step = Var(1, DT.int)
    core_step <<= core_count

    for gmt in range(cube_idx, total_m, core_step):
        bh = Var(gmt // tiles_m)
        lmt = Var(gmt % tiles_m)
        q_row = Var(bh * S1 + lmt * D256_TILE_M)
        kv_base = Var(bh * S2)
        valid_m = Min(D256_TILE_M, S1 - lmt * D256_TILE_M)
        local_valid_m = Min(D256_HALF_M, Max(valid_m - sb_row, 0))
        q_block = Var(lmt * KBLK_M + sb)   # this sub-block's single 32-block index

        # All N tiles are processed. Causal correctness comes from the per-tile
        # column mask below (a fully-upper-triangle tile gets eff_valid_n=0 -> all
        # columns masked -> P=0 -> no contribution). We do NOT shrink the inner loop
        # bound for causal: a per-M-tile-reduced `range(0, active+1)` deadlocks the
        # cube/vec handoff in this D256_TILE_M=64 lo/hi-PV pipeline (validated: the reduced
        # bound hangs, the full bound is correct). Skipping upper-triangle tiles is a
        # future optimization that needs the reduced-bound sync issue rooted first.
        active_tiles_n = tiles_n
        active_groups = CeilDiv(active_tiles_n, GROUP_N)

        # Previous MTE3 writeback still reads accum_ub / ub_rmax_s / ub_rsum_s;
        # autosync's loop-carried machinery guards the reinitialisation.
        with auto_sync():
            dup(ub_rmax_s, NEG_LARGE)
            dup(ub_rsum_s, 0.0)
            dup(ub_zero_s, 0.0)
            dup(ub_zero_chunk_s, 0.0, count=ROW_CHUNK)
            dup(accum_ub, 0.0)

        with auto_sync():
            l1q[0] <<= q[q_row:q_row + valid_m, 0:TILE_K]
            l1q[1] <<= q[q_row:q_row + valid_m, TILE_K:D_VAL]

        for group_id in range(0, active_groups + D256_GROUP_LOOKAHEAD):
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

                            l1k[l1k_cnt] <<= k[kv_row:kv_row + valid_n, 0:TILE_K]
                            matmul(
                                l0c[l0c_cnt], l1q[0], l1k[l1k_cnt],
                                m=D256_TILE_M, n=TILE_N, k=TILE_K, is_init=True,
                            )
                            l1k_cnt += 1
                            l1k[l1k_cnt] <<= k[kv_row:kv_row + valid_n, TILE_K:D_VAL]
                            matmul(
                                l0c[l0c_cnt], l1q[1], l1k[l1k_cnt],
                                m=D256_TILE_M, n=TILE_N, k=TILE_K, is_init=False,
                            )
                            l1k_cnt += 1

                            group_col = gi * TILE_N
                            score_ws[
                                group_id, 0:D256_TILE_M, group_col:group_col + TILE_N
                            ] <<= l0c[l0c_cnt]
                            l0c_cnt += 1
                            bar_m()
                qk_mutex.ready()
                qk_mutex.wait()
                p_mutex.lock()

                for rb in unroll(0, D256_HALF_M, ROW_CHUNK):
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

                                ub_score <<= score_ws[
                                    group_id, chunk_row:chunk_row + ROW_CHUNK, group_col:group_col + TILE_N
                                ]
                                muls(ub_score, ub_score, scale)

                                causal_cols = Min(Max((q_block - ni * KBLK_N + 1) * BLOCK_CAUSAL, 0), TILE_N)
                                eff_valid_n = Min(valid_n, causal_cols + (1 - is_causal) * TILE_N)
                                if eff_valid_n < TILE_N:
                                    d256_apply_score_tail_mask_chunk(ub_score, eff_valid_n)

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

                    brcb(
                        ub_max, ub_rmax_s[0:1, rb:rb + ROW_CHUNK],
                        repeat=ROW_CHUNK // 8, dst_blk_stride=1, dst_rep_stride=8,
                    )
                    for gi in range(0, GROUP_N):
                        if gi < group_len:
                            with auto_sync():
                                ub_p = ub_p_group[gi]
                                ub_score = ub_score_group[gi]
                                group_col = gi * TILE_N

                                sub(ub_score[0:ROW_CHUNK, 0:HALF_N], ub_score[0:ROW_CHUNK, 0:HALF_N], ub_max)
                                sub(ub_score[0:ROW_CHUNK, HALF_N:TILE_N], ub_score[0:ROW_CHUNK, HALF_N:TILE_N], ub_max)
                                if chunk_valid_m < ROW_CHUNK:
                                    d256_apply_score_row_tail_mask_chunk_after_shift(ub_score, chunk_valid_m)

                                exp(ub_score, ub_score)
                                add(ub_tmp, ub_score[0:ROW_CHUNK, 0:HALF_N], ub_score[0:ROW_CHUNK, HALF_N:TILE_N])
                                cadd(ub_sum_s, ub_tmp)
                                add(ub_group_sum_s, ub_group_sum_s, ub_sum_s, count=ROW_CHUNK)

                                cast(ub_p, ub_score)
                                p_ws[
                                    group_id, chunk_row:chunk_row + ROW_CHUNK, group_col:group_col + TILE_N
                                ] <<= ub_p

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

            if group_id >= D256_GROUP_LOOKAHEAD:

                with cube_scope():
                    with auto_sync():
                        p_mutex.wait()
                        group_start2 = Var((group_id - D256_GROUP_LOOKAHEAD) * GROUP_N)
                        group_len2 = Min(GROUP_N, active_tiles_n - group_start2)
                        for gi in range(0, GROUP_N):
                            if gi < group_len2:
                                ni = Var(group_start2 + gi)
                                n_off = Var(ni * TILE_N)
                                prev_valid_n = Min(TILE_N, S2 - n_off)
                                v_row = Var(kv_base + n_off)
                                group_col = gi * TILE_N

                                l1p[l1pv_cnt] <<= p_ws[
                                    group_id - D256_GROUP_LOOKAHEAD, 0:D256_TILE_M, group_col:group_col + TILE_N
                                ]

                                l1v[l1v_cnt][0:prev_valid_n, 0:TILE_K] <<= v[v_row:v_row + prev_valid_n, 0:TILE_K]
                                matmul(
                                    l0c[l0c_cnt], l1p[l1pv_cnt], l1v[l1v_cnt].T,
                                    m=D256_TILE_M, n=TILE_K, k=prev_valid_n, is_init=(gi == 0),
                                )
                                l1v_cnt += 1

                                l1v[l1v_cnt][0:prev_valid_n, 0:TILE_K] <<= v[v_row:v_row + prev_valid_n, TILE_K:D_VAL]
                                matmul(
                                    l0c_hi[l0c_hi_cnt], l1p[l1pv_cnt], l1v[l1v_cnt].T,
                                    m=D256_TILE_M, n=TILE_K, k=prev_valid_n, is_init=(gi == 0),
                                )
                                l1v_cnt += 1
                                l1pv_cnt += 1
                                bar_m()
                        p_mutex.free()

                        pv_mutex.lock()
                        pv_ws[group_id - D256_GROUP_LOOKAHEAD, 0:D256_TILE_M, 0:TILE_K] <<= l0c[l0c_cnt]
                        pv_ws[group_id - D256_GROUP_LOOKAHEAD, 0:D256_TILE_M, TILE_K:D_VAL] <<= l0c_hi[l0c_hi_cnt]
                        pv_mutex.ready()
                        l0c_cnt += 1
                        l0c_hi_cnt += 1

                with auto_sync():
                    pv_mutex.wait()
                    ub_pv <<= pv_ws[group_id - D256_GROUP_LOOKAHEAD, sb_row:sb_row + D256_HALF_M, 0:D_VAL]

                    brcb(ub_expdiff, expdiff_buf[group_id - D256_GROUP_LOOKAHEAD], repeat=D256_HALF_M // 8, dst_blk_stride=1, dst_rep_stride=8)
                    # accum *= expdiff (broadcast [32,8] over 256 cols in 4 x 64-col segments)
                    mul(accum_ub[0:D256_HALF_M, 0:HALF_N], accum_ub[0:D256_HALF_M, 0:HALF_N], ub_expdiff)
                    mul(accum_ub[0:D256_HALF_M, HALF_N:TILE_N], accum_ub[0:D256_HALF_M, HALF_N:TILE_N], ub_expdiff)
                    mul(accum_ub[0:D256_HALF_M, TILE_N:TILE_N + HALF_N], accum_ub[0:D256_HALF_M, TILE_N:TILE_N + HALF_N], ub_expdiff)
                    mul(accum_ub[0:D256_HALF_M, TILE_N + HALF_N:D_VAL], accum_ub[0:D256_HALF_M, TILE_N + HALF_N:D_VAL], ub_expdiff)
                    add(accum_ub, accum_ub, ub_pv)
                    pv_mutex.free()

        with auto_sync():
            brcb(ub_rowsum, ub_rsum_s, repeat=D256_HALF_M // 8, dst_blk_stride=1, dst_rep_stride=8)
            div(accum_ub[0:D256_HALF_M, 0:HALF_N], accum_ub[0:D256_HALF_M, 0:HALF_N], ub_rowsum)
            div(accum_ub[0:D256_HALF_M, HALF_N:TILE_N], accum_ub[0:D256_HALF_M, HALF_N:TILE_N], ub_rowsum)
            div(accum_ub[0:D256_HALF_M, TILE_N:TILE_N + HALF_N], accum_ub[0:D256_HALF_M, TILE_N:TILE_N + HALF_N], ub_rowsum)
            div(accum_ub[0:D256_HALF_M, TILE_N + HALF_N:D_VAL], accum_ub[0:D256_HALF_M, TILE_N + HALF_N:D_VAL], ub_rowsum)
            # Keep this as one dense cast. Four 64-column strided casts produced NaNs
            # in simulator for this wide bf16 output path.
            cast(ub_out, accum_ub)

            if local_valid_m > 0:
                out_row = Var(q_row + sb_row)
                out[out_row:out_row + local_valid_m, 0:D_VAL] <<= ub_out[0:local_valid_m, 0:D_VAL]

    return out


# ---------------------------------------------------------------------------
# Reference implementation (matches the kernel precision path)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@lru_cache(maxsize=2)
def build_mha_d256_bf16_kernel(device):
    if device not in ("a2", "a3"):
        raise ValueError("Shared tensor-vector source supports A2/A3")
    return import_module(f"ascriptor.{device}").kernel()(mha_flash_d256_bf16)

# ----------------------------------------------------------------------------------------------------
# pj_bf16_causal.py
# ----------------------------------------------------------------------------------------------------

# RFC-0009 batch-2 rewrite: the two hand-written credit events are gone (autosync guards the
# accum/rmax/rsum reinitialisation and the V->MTE3 flush), the stage1_cnt/stage2_cnt pair
# collapses into the loop beat `ni` (producer ni, consumer ni - 1), and the workspace rings are
# GMBuffs checked by the gmbuff pass. The bare GMTensor signature is modernised - this kernel
# had never compiled in ascriptor before.

P_HIF8_SCALE = 128.0
HIF8_MIN_CLAMP = 2.0 ** -22
EXP_MASK = 0x7F800000
EXPABS_BIAS = -0x00800000
EXPABS_LE15 = 32768.0
EXPABS_LE7 = 128.0
EXPABS_LE3 = 8.0

@func()
def quantize_p_chunk_nonneg_scaled(
    p_chunk: Tensor,
    exp_chunk: Tensor,
    meta_chunk: Tensor,
    factor_chunk: Tensor,
    flag_chunk: Tensor,
    one_chunk: Tensor,
    expmask_u32: Tensor,
):
    meta_chunk <<= p_chunk

    x_u16 = meta_chunk.reinterpret(DT.uint16)
    exp_u16 = exp_chunk.reinterpret(DT.uint16)
    expmask_u16 = expmask_u32.reinterpret(DT.uint16)
    vand(exp_u16, x_u16, expmask_u16)

    expabs_u16 = meta_chunk.reinterpret(DT.uint16)
    vnot(expabs_u16, exp_u16)
    vand(expabs_u16, expabs_u16, expmask_u16)
    expabs_i32 = meta_chunk.reinterpret(DT.int)
    adds(expabs_i32, expabs_i32, EXPABS_BIAS)
    vmax(meta_chunk, meta_chunk, exp_chunk)

    vmaxs(exp_chunk, exp_chunk, HIF8_MIN_CLAMP)

    dup(factor_chunk, 0.5)
    compare_scalar(flag_chunk, meta_chunk, EXPABS_LE15, CompareMode.LE)
    select(factor_chunk, flag_chunk, factor_chunk, one_chunk, SelectMode.TENSOR_SCALAR)
    mul(exp_chunk, exp_chunk, factor_chunk)

    dup(factor_chunk, 0.5)
    compare_scalar(flag_chunk, meta_chunk, EXPABS_LE7, CompareMode.LE)
    select(factor_chunk, flag_chunk, factor_chunk, one_chunk, SelectMode.TENSOR_SCALAR)
    mul(exp_chunk, exp_chunk, factor_chunk)

    dup(factor_chunk, 0.5)
    compare_scalar(flag_chunk, meta_chunk, EXPABS_LE3, CompareMode.LE)
    select(factor_chunk, flag_chunk, factor_chunk, one_chunk, SelectMode.TENSOR_SCALAR)
    mul(exp_chunk, exp_chunk, factor_chunk)

    div(p_chunk, p_chunk, exp_chunk)
    adds(p_chunk, p_chunk, 0.5)
    roundint = meta_chunk.reinterpret(DT.int)
    cast(roundint, p_chunk, round_mode=RoundMode.TRUNC)
    cast(p_chunk, roundint)
    mul(p_chunk, p_chunk, exp_chunk)
    muls(p_chunk, p_chunk, 1.0 / P_HIF8_SCALE)






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


def _suffix_invalid_mask_const(valid_cols: int) -> int:
    return -(1 << valid_cols)


@func()
def apply_diagonal_tile_causal_mask(ub_score: Tensor, sb_row: Var):
    if sb_row == 0:
        dup(ub_score[0:HALF_M, HALF_N:TILE_N], NEG_LARGE)
        for row in unroll(HALF_M):
            valid_cols = row + 1
            if valid_cols < HALF_N:
                set_mask(0, _suffix_invalid_mask_const(valid_cols))
                dup(ub_score[row:row + 1, 0:HALF_N], NEG_LARGE)
                reset_mask()
    else:
        for row in unroll(HALF_M):
            valid_cols = row + 1
            if valid_cols < HALF_N:
                set_mask(0, _suffix_invalid_mask_const(valid_cols))
                dup(ub_score[row:row + 1, HALF_N:TILE_N], NEG_LARGE)
                reset_mask()


def flash_attn_full_pj_hif8_causal_kernel(
    q: GM[bf16, ('TQ', 'D')], k: GM[bf16, ('TK', 'D')], v: GM[bf16, ('TK', 'D')], out: GM[bf16, ('TQ', 'D')],
    rowmax: GM[f32, ('TST',)], rowsum: GM[f32, ('TST',)],
    S1: i32, S2: i32, D: i32, BH: i32, scale: f32,
):
    score_ws = GMBuff(DT.float, [TILE_M, TILE_N], slots=2, name="score_ws")
    p_ws = GMBuff(DT.bfloat16, [TILE_M, TILE_N], slots=2, name="p_ws")
    pv_ws = GMBuff(DT.float, [TILE_M, TILE_K], slots=2, name="pv_ws")

    l1q = DBuff(DT.bfloat16, [TILE_M, TILE_K], Position.L1)
    l1k = DBuff(DT.bfloat16, [TILE_N, TILE_K], Position.L1)
    l1p = DBuff(DT.bfloat16, [TILE_M, TILE_N], Position.L1)
    l1v = DBuff(DT.bfloat16, [TILE_N, TILE_K], Position.L1)
    l0c = DBuff(DT.float, [TILE_M, TILE_N], Position.L0C)

    ub_score = Tensor(DT.float, [HALF_M, TILE_N], Position.UB)
    ub_pv = Tensor(DT.float, [HALF_M, TILE_K], Position.UB)
    ub_chunk = Tensor(DT.float, [HALF_M, HALF_N], Position.UB)
    ub_tmp = Tensor(DT.float, [HALF_M, HALF_N], Position.UB)
    ub_meta = Tensor(DT.float, [HALF_M, HALF_N], Position.UB)
    ub_factor = Tensor(DT.float, [HALF_M, HALF_N], Position.UB)
    ub_one = Tensor(DT.float, [1, HALF_N], Position.UB)
    ub_max_s = Tensor(DT.float, [1, HALF_M], Position.UB)
    ub_rmax_s = Tensor(DT.float, [1, HALF_M], Position.UB)
    ub_sum_s = Tensor(DT.float, [1, HALF_M], Position.UB)
    ub_rsum_s = Tensor(DT.float, [1, HALF_M], Position.UB)
    ub_zero_s = Tensor(DT.float, [1, HALF_M], Position.UB)
    ub_max = Tensor(DT.float, [HALF_M, 8], Position.UB)
    ub_rowsum = Tensor(DT.float, [HALF_M, 8], Position.UB)
    ub_p = Tensor(DT.bfloat16, [HALF_M, TILE_N], Position.UB)
    ub_flag = Tensor(DT.uint8, [HALF_M, HALF_N], Position.UB)
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
    expmask_u32 = ub_p.reinterpret(DT.uint32)

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
        # autosync's loop-carried machinery guards the reinitialisation.
        with auto_sync():
            dup(ub_rmax_s, NEG_LARGE)
            dup(ub_rsum_s, 0.0)
            dup(ub_zero_s, 0.0)
            dup(ub_one, 1.0)
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
                        apply_diagonal_tile_causal_mask(ub_score, sb_row)
                    if valid_n < TILE_N:
                        apply_score_tail_mask(ub_score, valid_n)
                    reset_mask()
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

                    muls(ub_score, ub_score, P_HIF8_SCALE)
                    dup(expmask_u32, EXP_MASK)
                    ub_chunk <<= ub_score[0:HALF_M, 0:HALF_N]
                    quantize_p_chunk_nonneg_scaled(
                        ub_chunk, ub_tmp, ub_meta, ub_factor, ub_flag, ub_one, expmask_u32,
                    )
                    ub_score[0:HALF_M, 0:HALF_N] <<= ub_chunk
                    ub_chunk <<= ub_score[0:HALF_M, HALF_N:TILE_N]
                    quantize_p_chunk_nonneg_scaled(
                        ub_chunk, ub_tmp, ub_meta, ub_factor, ub_flag, ub_one, expmask_u32,
                    )
                    ub_score[0:HALF_M, HALF_N:TILE_N] <<= ub_chunk
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
                    # Do not write the pad rows at all -- contract only the valid ones. The
                    # overlapping form (clear the whole slot, then load a prefix of it) puts
                    # two writers on the same L1 bytes, which the board does not order by
                    # issue order and a bar_mte2() does not fix (D-214, board-proved on v5).
                    l1v[l1pv_cnt][0:prev_valid_n, 0:D] <<= v[v_row:v_row + prev_valid_n, 0:D]
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
            cast(ub_p, accum_ub)

            if local_valid_m > 0:
                out_row = Var(q_row + sb_row)
                stats_row = Var(bh * CeilDiv(S1, 8) * 8 + lmt * TILE_M + sb_row)
                rowmax[stats_row:stats_row + local_valid_m] <<= ub_rmax_s[0:1, 0:local_valid_m]
                rowsum[stats_row:stats_row + local_valid_m] <<= ub_rsum_s[0:1, 0:local_valid_m]
                out[out_row:out_row + local_valid_m, 0:D] <<= ub_p[0:local_valid_m, 0:D]

    return out, rowmax, rowsum






@lru_cache(maxsize=2)
def build_pj_bf16_causal_kernel(device):
    if device not in ("a2", "a3"):
        raise ValueError("Shared tensor-vector source supports A2/A3")
    return import_module(f"ascriptor.{device}").kernel()(flash_attn_full_pj_hif8_causal_kernel)

# ----------------------------------------------------------------------------------------------------
# pj_bf16_lag2.py
# ----------------------------------------------------------------------------------------------------

# RFC-0009 batch-1 rewrite: the two hand-written credit events are gone (autosync's loop-carried
# machinery guards accum_ub / ub_rmax_s / ub_rsum_s across M tiles and the V->MTE3 flush), the
# stage1_cnt / stage2_cnt pair collapses into the loop beat `ni` (producer ni, consumer ni - 2),
# and the workspace rings are GMBuffs whose algebra the gmbuff pass checks.












def flash_attn_full_pj_hif8_bf16_lag2_kernel(
    q: GM[bf16, ('TQ', 'D')], k: GM[bf16, ('TK', 'D')], v: GM[bf16, ('TK', 'D')], out: GM[bf16, ('TQ', 'D')], rowmax: GM[f32, ('TST',)], rowsum: GM[f32, ('TST',)],
    S1: i32, S2: i32, D: i32, BH: i32, scale: f32,
):
    score_ws = GMBuff(DT.float, [TILE_M, TILE_N], slots=3, name="score_ws")
    p_ws = GMBuff(DT.bfloat16, [TILE_M, TILE_N], slots=3, name="p_ws")
    pv_ws = GMBuff(DT.float, [TILE_M, TILE_K], slots=3, name="pv_ws")

    l1q = DBuff(DT.bfloat16, [TILE_M, TILE_K], Position.L1)
    l1k = DBuff(DT.bfloat16, [TILE_N, TILE_K], Position.L1)
    l1p = DBuff(DT.bfloat16, [TILE_M, TILE_N], Position.L1)
    l1v = DBuff(DT.bfloat16, [TILE_N, TILE_K], Position.L1)
    l0c = DBuff(DT.float, [TILE_M, TILE_N], Position.L0C)

    ub_score = Tensor(DT.float, [HALF_M, TILE_N], Position.UB)
    ub_pv = Tensor(DT.float, [HALF_M, TILE_K], Position.UB)
    ub_chunk = Tensor(DT.float, [HALF_M, HALF_N], Position.UB)
    ub_tmp = Tensor(DT.float, [HALF_M, HALF_N], Position.UB)
    ub_meta = Tensor(DT.float, [HALF_M, HALF_N], Position.UB)
    ub_factor = Tensor(DT.float, [HALF_M, HALF_N], Position.UB)
    ub_one = Tensor(DT.float, [1, HALF_N], Position.UB)
    ub_max_s = Tensor(DT.float, [1, HALF_M], Position.UB)
    ub_rmax_s = Tensor(DT.float, [1, HALF_M], Position.UB)
    ub_sum_s = Tensor(DT.float, [1, HALF_M], Position.UB)
    ub_rsum_s = Tensor(DT.float, [1, HALF_M], Position.UB)
    ub_zero_s = Tensor(DT.float, [1, HALF_M], Position.UB)
    ub_max = Tensor(DT.float, [HALF_M, 8], Position.UB)
    ub_rowsum = Tensor(DT.float, [HALF_M, 8], Position.UB)
    ub_p = Tensor(DT.bfloat16, [HALF_M, TILE_N], Position.UB)
    ub_flag = Tensor(DT.uint8, [HALF_M, HALF_N], Position.UB)
    ub_expdiff = Tensor(DT.float, [HALF_M, 8], Position.UB)
    accum_ub = Tensor(DT.float, [HALF_M, TILE_K], Position.UB)
    expdiff_buf = TBuff(DT.float, [1, HALF_M], Position.UB)

    qk_mutex = CvMutex(0, depth=3, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)
    p_mutex = VcMutex(1, depth=3, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)
    pv_mutex = CvMutex(2, depth=3, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)

    l1qk_cnt = Var(0)
    l1pv_cnt = Var(0)
    l0c_cnt = Var(0)

    cube_idx = GetCubeIdx()
    sb = GetSubBlockIdx()
    sb_row = Var(sb * HALF_M)
    expmask_u32 = ub_p.reinterpret(DT.uint32)

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

        # The previous MTE3 writeback still reads accum_ub, ub_rmax_s, and ub_rsum_s;
        # autosync's loop-carried machinery guards the reinitialisation.
        with auto_sync():
            dup(ub_rmax_s, NEG_LARGE)
            dup(ub_rsum_s, 0.0)
            dup(ub_zero_s, 0.0)
            dup(ub_one, 1.0)
            dup(accum_ub, 0.0)

        for ni in range(0, tiles_n + 2):
            if ni < tiles_n:
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

                    muls(ub_score, ub_score, P_HIF8_SCALE)
                    dup(expmask_u32, EXP_MASK)
                    ub_chunk <<= ub_score[0:HALF_M, 0:HALF_N]
                    quantize_p_chunk_nonneg_scaled(
                        ub_chunk, ub_tmp, ub_meta, ub_factor, ub_flag, ub_one, expmask_u32,
                    )
                    ub_score[0:HALF_M, 0:HALF_N] <<= ub_chunk
                    ub_chunk <<= ub_score[0:HALF_M, HALF_N:TILE_N]
                    quantize_p_chunk_nonneg_scaled(
                        ub_chunk, ub_tmp, ub_meta, ub_factor, ub_flag, ub_one, expmask_u32,
                    )
                    ub_score[0:HALF_M, HALF_N:TILE_N] <<= ub_chunk
                    cast(ub_p, ub_score)

                    p_mutex.lock()
                    p_ws[ni, sb_row:sb_row + HALF_M, 0:TILE_N] <<= ub_p
                    p_mutex.ready()
                    qk_mutex.free()

                    l1qk_cnt += 1
                    l0c_cnt += 1

            if ni > 1:
                with auto_sync():
                    prev_nt = Var(ni - 2)
                    prev_n_off = Var(prev_nt * TILE_N)
                    prev_valid_n = Min(TILE_N, S2 - prev_n_off)
                    v_row = Var(kv_base + prev_n_off)

                    p_mutex.wait()
                    l1p[l1pv_cnt] <<= p_ws[ni - 2, 0:TILE_M, 0:TILE_N]
                    l1v[l1pv_cnt][0:prev_valid_n, 0:D] <<= v[v_row:v_row + prev_valid_n, 0:D]
                    matmul(l0c[l0c_cnt], l1p[l1pv_cnt], l1v[l1pv_cnt].T, k=prev_valid_n, is_init=True)
                    p_mutex.free()

                    pv_mutex.lock()
                    pv_ws[ni - 2, 0:TILE_M, 0:D] <<= l0c[l0c_cnt]
                    pv_mutex.ready()

                    pv_mutex.wait()
                    ub_pv <<= pv_ws[ni - 2, sb_row:sb_row + HALF_M, 0:D]

                    brcb(ub_expdiff, expdiff_buf[ni - 2], repeat=HALF_M // 8, dst_blk_stride=1, dst_rep_stride=8)
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
            cast(ub_p, accum_ub)

            if local_valid_m > 0:
                out_row = Var(q_row + sb_row)
                stats_row = Var(bh * CeilDiv(S1, 8) * 8 + lmt * TILE_M + sb_row)
                out[out_row:out_row + local_valid_m, 0:D] <<= ub_p[0:local_valid_m, 0:D]
                rowmax[stats_row:stats_row + local_valid_m] <<= ub_rmax_s[0:1, 0:local_valid_m]
                rowsum[stats_row:stats_row + local_valid_m] <<= ub_rsum_s[0:1, 0:local_valid_m]

    return out, rowmax, rowsum



@lru_cache(maxsize=2)
def build_pj_bf16_lag2_kernel(device):
    if device not in ("a2", "a3"):
        raise ValueError("Shared tensor-vector source supports A2/A3")
    return import_module(f"ascriptor.{device}").kernel()(flash_attn_full_pj_hif8_bf16_lag2_kernel)

# ----------------------------------------------------------------------------------------------------
# pj_fp16_lag1.py
# ----------------------------------------------------------------------------------------------------

# RFC-0009 batch-1 rewrite: the two hand-written credit events are gone (autosync's loop-carried
# machinery guards accum_ub / ub_rmax_s / ub_rsum_s across M tiles and the V->MTE3 flush), the
# stage1_cnt / stage2_cnt pair collapses into the loop beat `ni` (producer ni, consumer ni - 1),
# and the workspace rings are GMBuffs whose algebra the gmbuff pass checks.












def flash_attn_full_pj_hif8_fp16_lag1_kernel(
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
    ub_chunk = Tensor(DT.float, [HALF_M, HALF_N], Position.UB)
    ub_pv = Tensor(DT.float, [HALF_M, TILE_K], Position.UB)
    ub_tmp = Tensor(DT.float, [HALF_M, HALF_N], Position.UB)
    ub_meta = Tensor(DT.float, [HALF_M, HALF_N], Position.UB)
    ub_factor = Tensor(DT.float, [HALF_M, HALF_N], Position.UB)
    ub_one = Tensor(DT.float, [1, HALF_N], Position.UB)
    ub_max_s = Tensor(DT.float, [1, HALF_M], Position.UB)
    ub_rmax_s = Tensor(DT.float, [1, HALF_M], Position.UB)
    ub_sum_s = Tensor(DT.float, [1, HALF_M], Position.UB)
    ub_rsum_s = Tensor(DT.float, [1, HALF_M], Position.UB)
    ub_zero_s = Tensor(DT.float, [1, HALF_M], Position.UB)
    ub_max = Tensor(DT.float, [HALF_M, 8], Position.UB)
    ub_rowsum = Tensor(DT.float, [HALF_M, 8], Position.UB)
    ub_p = Tensor(DT.half, [HALF_M, TILE_N], Position.UB)
    ub_flag = Tensor(DT.uint8, [HALF_M, HALF_N], Position.UB)
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
    expmask_u32 = ub_p.reinterpret(DT.uint32)

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

        # The previous MTE3 writeback still reads accum_ub, ub_rmax_s, and ub_rsum_s;
        # autosync's loop-carried machinery guards the reinitialisation.
        with auto_sync():
            dup(ub_rmax_s, NEG_LARGE)
            dup(ub_rsum_s, 0.0)
            dup(ub_zero_s, 0.0)
            dup(ub_one, 1.0)
            dup(accum_ub, 0.0)

        for ni in range(0, tiles_n + 1):
            if ni < tiles_n:
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

                    muls(ub_score, ub_score, P_HIF8_SCALE)
                    dup(expmask_u32, EXP_MASK)
                    ub_chunk <<= ub_score[0:HALF_M, 0:HALF_N]
                    quantize_p_chunk_nonneg_scaled(
                        ub_chunk, ub_tmp, ub_meta, ub_factor, ub_flag, ub_one, expmask_u32,
                    )
                    ub_score[0:HALF_M, 0:HALF_N] <<= ub_chunk
                    ub_chunk <<= ub_score[0:HALF_M, HALF_N:TILE_N]
                    quantize_p_chunk_nonneg_scaled(
                        ub_chunk, ub_tmp, ub_meta, ub_factor, ub_flag, ub_one, expmask_u32,
                    )
                    ub_score[0:HALF_M, HALF_N:TILE_N] <<= ub_chunk
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
                    l1v[l1pv_cnt][0:prev_valid_n, 0:D] <<= v[v_row:v_row + prev_valid_n, 0:D]
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
                stats_row = Var(bh * CeilDiv(S1, 8) * 8 + lmt * TILE_M + sb_row)
                out[out_row:out_row + local_valid_m, 0:D] <<= accum_ub[0:local_valid_m, 0:D]
                rowmax[stats_row:stats_row + local_valid_m] <<= ub_rmax_s[0:1, 0:local_valid_m]
                rowsum[stats_row:stats_row + local_valid_m] <<= ub_rsum_s[0:1, 0:local_valid_m]

    return out, rowmax, rowsum



@lru_cache(maxsize=2)
def build_pj_fp16_lag1_kernel(device):
    if device not in ("a2", "a3"):
        raise ValueError("Shared tensor-vector source supports A2/A3")
    return import_module(f"ascriptor.{device}").kernel()(flash_attn_full_pj_hif8_fp16_lag1_kernel)

# ----------------------------------------------------------------------------------------------------
# pv_stage.py
# ----------------------------------------------------------------------------------------------------

def flash_attn_score_pv_kernel(
    q: GM[f16, ('TQ', 'D')], k: GM[f16, ('TK', 'D')], v: GM[f16, ('TK', 'D')], pv: GM[f32, ('TPV', 'D')],
    S1: i32, S2: i32, D: i32, BH: i32, scale: f32,
):
    # RFC-0009 batch 3: the 2-slot rings become GMBuffs so the gmbuff pass machine-checks the
    # ring algebra (single beat, lag < slots, mutex depth <= slots). No protocol change.
    score_ws = GMBuff(DT.float, [TILE_M, TILE_N], slots=2, name="score_ws")
    p_ws = GMBuff(DT.half, [TILE_M, TILE_N], slots=2, name="p_ws")

    l1q = DBuff(DT.half, [TILE_M, TILE_K], Position.L1)
    l1k = DBuff(DT.half, [TILE_N, TILE_K], Position.L1)
    l1p = DBuff(DT.half, [TILE_M, TILE_N], Position.L1)
    l1v = DBuff(DT.half, [TILE_N, TILE_K], Position.L1)
    l0c = DBuff(DT.float, [TILE_M, TILE_N], Position.L0C)

    ub_score = Tensor(DT.float, [HALF_M, TILE_N], Position.UB)
    ub_tmp = Tensor(DT.float, [HALF_M, HALF_N], Position.UB)
    ub_max_s = Tensor(DT.float, [1, HALF_M], Position.UB)
    ub_rmax_s = Tensor(DT.float, [1, HALF_M], Position.UB)
    ub_max = Tensor(DT.float, [HALF_M, 8], Position.UB)
    ub_p = Tensor(DT.half, [HALF_M, TILE_N], Position.UB)

    qk_mutex = CvMutex(0, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)
    p_mutex = VcMutex(1, depth=2, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)

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

        dup(ub_rmax_s, NEG_LARGE)

        for ni in range(0, tiles_n + 1):
            if ni < tiles_n:
                with auto_sync():
                    n_off = Var(ni * TILE_N)
                    k_row = Var(kv_base + n_off)
                    l1q[l1qk_cnt] <<= q[q_row:q_row + TILE_M, 0:D]
                    l1k[l1qk_cnt] <<= k[k_row:k_row + TILE_N, 0:D]
                    matmul(l0c[l0c_cnt], l1q[l1qk_cnt], l1k[l1qk_cnt], is_init=True)

                    qk_mutex.lock()
                    score_ws[ni, 0:TILE_M, 0:TILE_N] <<= l0c[l0c_cnt]
                    qk_mutex.ready()

                    qk_mutex.wait()
                    ub_score <<= score_ws[ni, sb_row:sb_row + HALF_M, 0:TILE_N]

                    muls(ub_score, ub_score, scale)
                    vmax(ub_tmp, ub_score[0:HALF_M, 0:HALF_N], ub_score[0:HALF_M, HALF_N:TILE_N])
                    cmax(ub_max_s, ub_tmp)
                    vmax(ub_rmax_s, ub_rmax_s, ub_max_s)
                    brcb(ub_max, ub_rmax_s, repeat=HALF_M // 8, dst_blk_stride=1, dst_rep_stride=8)

                    sub(ub_score[0:HALF_M, 0:HALF_N], ub_score[0:HALF_M, 0:HALF_N], ub_max)
                    sub(ub_score[0:HALF_M, HALF_N:TILE_N], ub_score[0:HALF_M, HALF_N:TILE_N], ub_max)

                    exp(ub_score, ub_score)
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
                    v_row = Var(kv_base + prev_n_off)
                    out_row = Var((bh * tiles_n + prev_nt) * S1 + lmt * TILE_M)

                    p_mutex.wait()
                    l1p[l1pv_cnt] <<= p_ws[ni - 1, 0:TILE_M, 0:TILE_N]
                    l1v[l1pv_cnt] <<= v[v_row:v_row + TILE_N, 0:D]
                    matmul(l0c[l0c_cnt], l1p[l1pv_cnt], l1v[l1pv_cnt].T, is_init=True)
                    pv[out_row:out_row + TILE_M, 0:D] <<= l0c[l0c_cnt]
                    p_mutex.free()

                    l1pv_cnt += 1
                    l0c_cnt += 1

    return pv



@lru_cache(maxsize=2)
def build_pv_stage_kernel(device):
    if device not in ("a2", "a3"):
        raise ValueError("Shared tensor-vector source supports A2/A3")
    return import_module(f"ascriptor.{device}").kernel()(flash_attn_score_pv_kernel)


# ----------------------------------------------------------------------------------------------------
# dispatcher
# ----------------------------------------------------------------------------------------------------

BUILDERS = {"dense_fp16": build_dense_fp16_kernel, "gqa_bf16": build_gqa_bf16_kernel,
            "mha_bf16": build_mha_bf16_kernel, "mha_d256": build_mha_d256_kernel,
            "mha_d256_bf16": build_mha_d256_bf16_kernel,
            "pj_bf16_causal": build_pj_bf16_causal_kernel,
            "pj_bf16_lag2": build_pj_bf16_lag2_kernel,
            "pj_fp16_lag1": build_pj_fp16_lag1_kernel, "pv_stage": build_pv_stage_kernel}


def build_kernel(variant, device):
    """Bind one variant's body to the a2 or a3 facade. Each builder is separately
    lru_cached, so asking for the same variant and device twice returns one kernel."""
    if variant not in BUILDERS:
        raise ValueError(f"unknown variant {variant!r}; one of {sorted(BUILDERS)}")
    return BUILDERS[variant](device)
