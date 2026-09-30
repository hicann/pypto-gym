# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A2/A3 SAGE decode attention: signed-int4 Q/K carriers, int8 V, int8 probabilities.

One kernel body is bound to either facade by build_kernel(device); the tensor-vector
vocabulary is common to both, so the algorithm is written once."""

from ascriptor.a2 import *
from functools import lru_cache
from importlib import import_module

TILE_M = 16
TILE_N = 512
D_HEAD = 128
CARRIER_D = D_HEAD // 8
BRCB_BLOCKS = 8  # one brcb at repeat=1 writes 8 blocks, so its destination owns 8 rows whatever TILE_M is
VEC_ROWS = 1
CHUNK_N = 64
P_SCALE = 127.0
NEG_INF = -99999.0
SPLIT_K = 128
SPLIT_N = 128
SOFTMAX_SCALE = D_HEAD ** -0.5


@func()
def build_suffix_invalid_mask(valid_cols: Var, out_mask: Var):
    """The lanes from ``valid_cols`` up, set. Its one caller holds ``valid_cols`` in [1, CHUNK_N),
    so ``1 << valid_cols`` keeps every bit it started with and the complement is exact. Doubling a
    signed -1 that many times, as this did, reaches the same mask through up to 63 data-dependent
    scalar iterations in the vector instruction stream, one step from the signed overflow that
    CHUNK_N == 64 is all that prevents."""
    ones = Var(0xFFFFFFFFFFFFFFFF, DT.uint64)
    low = Var(1, DT.uint64) << valid_cols
    out_mask <<= ones ^ (low - 1)


@func()
def mask_score_chunk_suffix_invalid(score_chunk: Tensor, valid_cols: Var):
    if valid_cols == 0:
        dup(score_chunk, NEG_INF)
    elif valid_cols < CHUNK_N:
        suffix_mask = Var(0, DT.uint64)
        build_suffix_invalid_mask(valid_cols, suffix_mask)
        set_mask(0, suffix_mask)
        dup(score_chunk, NEG_INF)
        reset_mask()


@func()
def apply_score_tail_mask(ub_score: Tensor, valid_cols: Var):
    for col in range(0, TILE_N, CHUNK_N):
        chunk_valid = Min(CHUNK_N, Max(valid_cols - col, 0))
        mask_score_chunk_suffix_invalid(ub_score[0:VEC_ROWS, col:col + CHUNK_N], chunk_valid)


def sage2_vnomean_int4_kernel(
    q: GM[i32, ("BQ", 1, CARRIER_D)],
    k: GM[i32, ("BK", "S2K", CARRIER_D)],
    v: GM[i8, ("BV", "S2V", D_HEAD)],
    scale_q: GM[f32, ("BSQ", 1, 1)],
    scale_k: GM[f32, ("BSK", "TN")],
    scale_v: GM[f32, ("BSV", 1, D_HEAD)],
    qm: GM[f16, ("BQM", 1, D_HEAD)],
    k_smooth: GM[f16, ("BKS", "S2KS", D_HEAD)],
    out: GM[f32, ("BO", 1, D_HEAD)],
    rowmax: GM[f32, ("BRM", 1)],
    rowsum: GM[f32, ("BRS", 1)],
    BH: i32,
    S2: i32,
):
    score_ws = split_workspace(DT.int, [GetCubeNum(), 2, TILE_M, TILE_N], name="score_ws")
    qm_ksmooth_ws = split_workspace(DT.half, [GetCubeNum(), 2, TILE_M, TILE_N], name="qm_ksmooth_ws")
    p_ws = split_workspace(DT.int8, [GetCubeNum(), 2, TILE_M, TILE_N], name="p_ws")
    pv_ws = split_workspace(DT.int, [GetCubeNum(), 2, TILE_M, D_HEAD], name="pv_ws")

    l1q = DBuff(DT.int, [TILE_M, CARRIER_D], Position.L1)
    l1qm = DBuff(DT.half, [TILE_M, D_HEAD], Position.L1)
    l1k = DBuff(DT.int, [TILE_N, CARRIER_D], Position.L1)
    l1ks = DBuff(DT.half, [TILE_N, D_HEAD], Position.L1)
    l1p = DBuff(DT.int8, [TILE_M, TILE_N], Position.L1)
    l1v = DBuff(DT.int8, [TILE_N, D_HEAD], Position.L1)
    l0c = DBuff(DT.int, [TILE_M, Max(TILE_N, D_HEAD)], Position.L0C)
    l0qks = DBuff(DT.float, [TILE_M, Max(TILE_N, D_HEAD)], Position.L0C)

    ub_score_int = Tensor(DT.int, [VEC_ROWS, TILE_N], Position.UB)
    ub_score = Tensor(DT.float, [VEC_ROWS, TILE_N], Position.UB)
    ub_qks_half = Tensor(DT.half, [VEC_ROWS, TILE_N], Position.UB)
    ub_qks = Tensor(DT.float, [VEC_ROWS, TILE_N], Position.UB)
    ub_scalev = Tensor(DT.float, [VEC_ROWS, D_HEAD], Position.UB)
    ub_score_chunk = Tensor(DT.float, [VEC_ROWS, CHUNK_N], Position.UB)
    ub_pv_int = Tensor(DT.int, [VEC_ROWS, D_HEAD], Position.UB)
    ub_pv = Tensor(DT.float, [VEC_ROWS, D_HEAD], Position.UB)
    ub_p_half = Tensor(DT.half, [VEC_ROWS, TILE_N], Position.UB)
    ub_p = Tensor(DT.int8, [VEC_ROWS, TILE_N], Position.UB)
    ub_max_s = Tensor(DT.float, [1, CHUNK_N], Position.UB)
    ub_rmax_s = Tensor(DT.float, [1, CHUNK_N], Position.UB)
    ub_sum_s = Tensor(DT.float, [1, CHUNK_N], Position.UB)
    ub_rsum_s = Tensor(DT.float, [1, CHUNK_N], Position.UB)
    ub_zero_s = Tensor(DT.float, [1, CHUNK_N], Position.UB)
    ub_max = Tensor(DT.float, [BRCB_BLOCKS, 8], Position.UB)
    ub_rowsum = Tensor(DT.float, [BRCB_BLOCKS, 8], Position.UB)
    ub_expdiff = Tensor(DT.float, [BRCB_BLOCKS, 8], Position.UB)
    accum_ub = Tensor(DT.float, [VEC_ROWS, D_HEAD], Position.UB)
    expdiff_buf = DBuff(DT.float, [1, CHUNK_N], Position.UB)

    qk_mutex = CvMutex(0, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)
    p_mutex = VcMutex(1, depth=2, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)
    pv_mutex = CvMutex(2, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)
    accum_store_ready = SEvent(Pipe.V, Pipe.MTE3, name="accum_store_ready")
    accum_store_valid = SEvent(Pipe.MTE3, Pipe.V, preset=True, name="accum_store_valid")

    q_cnt = Var(0)
    l1qk_cnt = Var(0)
    l1pv_cnt = Var(0)
    l0c_cnt = Var(0)
    stage1_cnt = Var(0)
    stage2_cnt = Var(0)

    cube_idx = GetCubeIdx()
    rows = Var(1)
    tiles_n = CeilDiv(S2, TILE_N)
    per_core = CeilDiv(BH, GetCubeNum())
    bh_begin = Var(per_core * cube_idx)
    bh_end = Min(bh_begin + per_core, BH)
    scale_q_value = Var(0.0, DT.float)
    scale_k_value = Var(0.0, DT.float)
    out_scale = 1.0 / P_SCALE

    for bh_idx in range(bh_begin, bh_end):
        l1q[q_cnt][:rows, :] <<= q[bh_idx, :rows, :]
        l1qm[q_cnt][:rows, :] <<= qm[bh_idx, :rows, :]
        bar_all()
        dup(ub_rmax_s, NEG_INF)
        dup(ub_rsum_s, 0.0)
        dup(ub_zero_s, 0.0)
        accum_store_valid.wait()
        dup(accum_ub, 0.0)

        for ni in range(0, tiles_n + 1):
            if ni < tiles_n:
                with auto_sync():
                    n_off = Var(ni * TILE_N)
                    valid_n = Min(TILE_N, S2 - n_off)
                    stage1_slot = var_and(stage1_cnt, 1)

                    l1k[l1qk_cnt] <<= k[bh_idx, n_off:n_off + valid_n, :]
                    l1ks[l1qk_cnt] <<= k_smooth[bh_idx, n_off:n_off + valid_n, :]
                    matmul(
                        l0c[l0c_cnt],
                        l1q[q_cnt].reinterpret(DT.int4, name="q_int4"),
                        l1k[l1qk_cnt].reinterpret(DT.int4, name="k_int4"),
                        splitn=SPLIT_N,
                    )
                    matmul(l0qks[l0c_cnt], l1qm[q_cnt], l1ks[l1qk_cnt], splitn=SPLIT_N)

                    qk_mutex.lock()
                    score_ws[cube_idx, stage1_slot, 0:TILE_M, 0:TILE_N] <<= l0c[l0c_cnt]
                    qm_ksmooth_ws[cube_idx, stage1_slot, 0:TILE_M, 0:TILE_N] <<= l0qks[l0c_cnt]
                    qk_mutex.ready()

                    qk_mutex.wait()
                    ub_score_int <<= score_ws[cube_idx, stage1_slot, 0:VEC_ROWS, 0:TILE_N]
                    cast(ub_score, ub_score_int)
                    ub_qks_half <<= qm_ksmooth_ws[cube_idx, stage1_slot, 0:VEC_ROWS, 0:TILE_N]
                    cast(ub_qks, ub_qks_half)
                    scale_q_value.GetValueFrom(scale_q[bh_idx, :rows, 0:1])
                    scale_k_value.GetValueFrom(scale_k[bh_idx, ni:ni + 1])
                    qk_scale = scale_q_value * scale_k_value
                    muls(ub_score, ub_score, qk_scale)
                    add(ub_score, ub_score, ub_qks)
                    muls(ub_score, ub_score, SOFTMAX_SCALE)

                    if valid_n < TILE_N:
                        apply_score_tail_mask(ub_score, valid_n)

                    ub_score_chunk <<= ub_score[0:VEC_ROWS, 0:CHUNK_N]
                    for col in range(CHUNK_N, TILE_N, CHUNK_N):
                        vmax(ub_score_chunk, ub_score_chunk, ub_score[0:VEC_ROWS, col:col + CHUNK_N])
                    cmax(ub_max_s, ub_score_chunk)

                    add(expdiff_buf[stage1_slot], ub_rmax_s, ub_zero_s)
                    vmax(ub_rmax_s, ub_rmax_s, ub_max_s)
                    sub(expdiff_buf[stage1_slot], expdiff_buf[stage1_slot], ub_rmax_s)
                    exp(expdiff_buf[stage1_slot], expdiff_buf[stage1_slot])

                    brcb(ub_max, ub_rmax_s, repeat=1, dst_blk_stride=1, dst_rep_stride=8)
                    for col in range(0, TILE_N, CHUNK_N):
                        score_chunk = ub_score[0:VEC_ROWS, col:col + CHUNK_N]
                        sub(score_chunk, score_chunk, ub_max)
                        exp(score_chunk, score_chunk)

                    ub_score_chunk <<= ub_score[0:VEC_ROWS, 0:CHUNK_N]
                    for col in range(CHUNK_N, TILE_N, CHUNK_N):
                        add(ub_score_chunk, ub_score_chunk, ub_score[0:VEC_ROWS, col:col + CHUNK_N])
                    cadd(ub_sum_s, ub_score_chunk)
                    mul(ub_rsum_s, ub_rsum_s, expdiff_buf[stage1_slot])
                    add(ub_rsum_s, ub_rsum_s, ub_sum_s)

                    muls(ub_score, ub_score, P_SCALE)
                    cast(ub_p_half, ub_score, round_mode=RoundMode.TO_EVEN)
                    cast(ub_p, ub_p_half)

                    p_mutex.lock()
                    if GetSubBlockIdx() == 0:
                        p_ws[cube_idx, stage1_slot, 0:1, 0:TILE_N] <<= ub_p[0:1, 0:TILE_N]
                    p_mutex.ready()
                    qk_mutex.free()

                    l1qk_cnt += 1
                    l0c_cnt += 1
                    stage1_cnt += 1

            if ni > 0:
                with auto_sync():
                    prev_ni = Var(ni - 1)
                    prev_n_off = Var(prev_ni * TILE_N)
                    prev_valid_n = Min(TILE_N, S2 - prev_n_off)
                    stage2_slot = var_and(stage2_cnt, 1)

                    l1v[l1pv_cnt] <<= v[bh_idx, prev_n_off:prev_n_off + prev_valid_n, :]
                    p_mutex.wait()
                    l1p[l1pv_cnt][0:1, 0:TILE_N] <<= p_ws[cube_idx, stage2_slot, 0:1, 0:TILE_N]
                    # The L1 V tail contains only prev_valid_n rows. Functional
                    # simulation zero-initialises unread storage, but silicon
                    # does not; bind the physical contraction to the loaded K.
                    matmul(
                        l0c[l0c_cnt],
                        l1p[l1pv_cnt],
                        l1v[l1pv_cnt].T,
                        m=TILE_M,
                        n=D_HEAD,
                        k=prev_valid_n,
                        splitk=SPLIT_K,
                    )
                    p_mutex.free()

                    pv_mutex.lock()
                    pv_ws[cube_idx, stage2_slot, 0:TILE_M, 0:D_HEAD] <<= l0c[l0c_cnt][0:TILE_M, 0:D_HEAD]
                    pv_mutex.ready()

                    pv_mutex.wait()
                    ub_pv_int <<= pv_ws[cube_idx, stage2_slot, 0:VEC_ROWS, 0:D_HEAD]
                    cast(ub_pv, ub_pv_int)
                    ub_scalev <<= scale_v[bh_idx, 0:VEC_ROWS, 0:D_HEAD]
                    mul(ub_pv, ub_pv, ub_scalev)

                    brcb(ub_expdiff, expdiff_buf[stage2_slot], repeat=1, dst_blk_stride=1, dst_rep_stride=8)
                    for col in range(0, D_HEAD, CHUNK_N):
                        accum_chunk = accum_ub[0:VEC_ROWS, col:col + CHUNK_N]
                        mul(accum_chunk, accum_chunk, ub_expdiff)
                        add(accum_chunk, accum_chunk, ub_pv[0:VEC_ROWS, col:col + CHUNK_N])
                    pv_mutex.free()

                    l1pv_cnt += 1
                    l0c_cnt += 1
                    stage2_cnt += 1

        brcb(ub_rowsum, ub_rsum_s, repeat=1, dst_blk_stride=1, dst_rep_stride=8)
        for col in range(0, D_HEAD, CHUNK_N):
            accum_chunk = accum_ub[0:VEC_ROWS, col:col + CHUNK_N]
            muls(accum_chunk, accum_chunk, out_scale)
            div(accum_chunk, accum_chunk, ub_rowsum)

        if GetSubBlockIdx() == 0:
            accum_store_ready.set()
            accum_store_ready.wait()
            out[bh_idx, :rows, :] <<= accum_ub[:rows, :]
            rowmax[bh_idx:bh_idx + rows, 0:1] <<= ub_rmax_s[0:1, 0:1]
            rowsum[bh_idx:bh_idx + rows, 0:1] <<= ub_rsum_s[0:1, 0:1]
        accum_store_valid.set()
        q_cnt += 1

    return out, rowmax, rowsum


@lru_cache(maxsize=2)
def build_kernel(device):
    if device not in ("a2", "a3"):
        raise ValueError("Shared tensor-vector source supports A2/A3")
    return import_module(f"ascriptor.{device}").kernel()(sage2_vnomean_int4_kernel)
