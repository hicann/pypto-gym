# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A2/A3 MLA attention over 512 non-rotary plus 64 rotary score features, B1/HQ8/HKV4.

One kernel body is bound to either facade by build_kernel(device); the tensor-vector
vocabulary is common to both, so the algorithm is written once."""

from ascriptor.a2 import *
from functools import lru_cache
from importlib import import_module



# First MLA implementation for flattened head-major inputs:
# q_*: [HQ * S, D], k_*/v: [HKV * SKV, D], out: [HQ * S, D_NOPE].

TILE_M = 128
TILE_N = 128
TILE_D = 128
HALF_M = TILE_M // 2
HALF_N = TILE_N // 2
# D=512 is handled as four 128-column chunks; one vec sub-block's 64 rows fit in UB.
ACCUM_ROWS = HALF_M

HQ = 8
HKV = 4
Q_PER_KV = HQ // HKV
D_NOPE = 512
D_ROPE = 64
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


def mla_b1_hq8_hkv4_kernel(
    q_nope: GM[f16, ("TQ", "Dn")],
    q_rope: GM[f16, ("TQ", "Dr")],
    k_nope: GM[f16, ("TK", "Dn")],
    k_rope: GM[f16, ("TK", "Dr")],
    v: GM[f16, ("TK", "Dn")],
    out: GM[f32, ("TQ", "Dn")],
    S: i32,
    SKV: i32,
    scale: f32,
):
    score_ws = split_workspace(DT.float, [GetCubeNum(), 2, TILE_M, TILE_N], name="score_ws")
    p_ws = split_workspace(DT.half, [GetCubeNum(), 2, TILE_M, TILE_N], name="p_ws")
    pv_ws = split_workspace(DT.float, [GetCubeNum(), 2, TILE_M, TILE_D], name="pv_ws")
    accum_ws = split_workspace(DT.float, [GetCubeNum(), 2, D_NOPE // TILE_D, TILE_M, TILE_D], name="accum_ws")

    l1qn = QBuff(DT.half, [TILE_M, TILE_D], Position.L1, sync_depth=2)
    l1kn = QBuff(DT.half, [TILE_N, TILE_D], Position.L1, sync_depth=2)
    l1qr = DBuff(DT.half, [TILE_M, D_ROPE], Position.L1)
    l1kr = DBuff(DT.half, [TILE_N, D_ROPE], Position.L1)
    l1p = DBuff(DT.half, [TILE_M, TILE_N], Position.L1)
    l1v = QBuff(DT.half, [TILE_N, TILE_D], Position.L1, sync_depth=2)
    l0c = DBuff(DT.float, [TILE_M, TILE_N], Position.L0C)

    ub_score = DBuff(DT.float, [HALF_M, TILE_N], Position.UB)
    ub_pv = Tensor(DT.float, [ACCUM_ROWS, TILE_D], Position.UB)
    ub_accum = Tensor(DT.float, [ACCUM_ROWS, TILE_D], Position.UB)
    ub_tmp = Tensor(DT.float, [HALF_M, HALF_N], Position.UB)
    # The score-reduction scratch is dead before P is cast.  Reuse its 16 KiB
    # instead of placing a two-slot P buffer above the A3 AIV stack boundary.
    ub_p = ub_tmp.reinterpret(DT.half, name="ub_p")
    ub_max_s = Tensor(DT.float, [1, HALF_M], Position.UB)
    ub_rmax_s = Tensor(DT.float, [1, HALF_M], Position.UB)
    ub_sum_s = Tensor(DT.float, [1, HALF_M], Position.UB)
    ub_rsum_s = Tensor(DT.float, [1, HALF_M], Position.UB)
    ub_zero_s = Tensor(DT.float, [1, HALF_M], Position.UB)
    ub_max = Tensor(DT.float, [HALF_M, 8], Position.UB)
    ub_rowsum = Tensor(DT.float, [ACCUM_ROWS, 8], Position.UB)
    ub_expdiff = Tensor(DT.float, [ACCUM_ROWS, 8], Position.UB)
    expdiff_buf = DBuff(DT.float, [1, HALF_M], Position.UB)

    accum_store_ready = SEvent(Pipe.V, Pipe.MTE3, name="accum_store_ready")
    accum_store_valid = SEvent(Pipe.MTE3, Pipe.V, preset=True, name="accum_store_valid")
    accum_store_done = SEvent(Pipe.MTE3, Pipe.MTE2, name="accum_store_done")
    accum_ws_valid = DEvent(Pipe.MTE3, Pipe.MTE2, name="accum_ws_valid")

    qk_mutex = CvMutex(0, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)
    p_mutex = VcMutex(1, depth=2, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE2)
    pv_mutex = CvMutex(2, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)
    accum_ws_valid.set()

    l1qk_cnt = Var(0)
    l1pv_cnt = Var(0)
    l1v_cnt = Var(0)
    l0c_cnt = Var(0)
    stage1_cnt = Var(0)
    stage2_cnt = Var(0)

    cube_idx = GetCubeIdx()
    sb = GetSubBlockIdx()
    sb_row = Var(sb * HALF_M)

    # S/SKV are positive by the unit contract and both tile sizes are 128.
    # Spell the power-of-two division directly so C220 does not materialize
    # signed-division correction temporaries in the AIV stack.
    tiles_m = var_shr(S + TILE_M - 1, 7)
    tiles_n = var_shr(SKV + TILE_N - 1, 7)
    total_m = Var(HQ * tiles_m)

    # Round-robin assignment covers every logical M tile once and avoids the
    # generic signed CeilDiv(total_m, cube_num) expansion on the vector side.
    for gmt in range(cube_idx, total_m, GetCubeNum()):
        hq = Var(gmt // tiles_m)
        lmt = Var(gmt % tiles_m)
        kvh = Var(hq // Q_PER_KV)
        q_row = Var(hq * S + lmt * TILE_M)
        kv_base = Var(kvh * SKV)
        valid_m = Min(TILE_M, S - lmt * TILE_M)
        local_valid_m = Min(HALF_M, Max(valid_m - sb_row, 0))

        accum_store_valid.wait()
        dup(ub_rmax_s, NEG_LARGE)
        dup(ub_rsum_s, 0.0)
        dup(ub_zero_s, 0.0)
        accum_cnt = Var(0)

        for ni in range(0, tiles_n + 1):
            if ni < tiles_n:
                with auto_sync():
                    n_off = Var(ni * TILE_N)
                    kv_row = Var(kv_base + n_off)
                    valid_n = Min(TILE_N, SKV - n_off)
                    # These counters are monotonic from zero; bit selection is
                    # equivalent to Euclidean modulo and avoids the A3 scalar
                    # spill caused by the generic negative-remainder expansion.
                    stage1_slot = var_and(stage1_cnt, 1)
                    ub_score_tile = ub_score[stage1_cnt]
                    ub_p_tile = ub_p

                    l1qn[l1qk_cnt] <<= q_nope[q_row:q_row + valid_m, 0:TILE_D]
                    l1kn[l1qk_cnt] <<= k_nope[kv_row:kv_row + valid_n, 0:TILE_D]
                    matmul(l0c[l0c_cnt], l1qn[l1qk_cnt], l1kn[l1qk_cnt], is_init=True)
                    bar_m()
                    l1qk_cnt += 1

                    for doff in range(TILE_D, D_NOPE, TILE_D):
                        l1qn[l1qk_cnt] <<= q_nope[q_row:q_row + valid_m, doff:doff + TILE_D]
                        l1kn[l1qk_cnt] <<= k_nope[kv_row:kv_row + valid_n, doff:doff + TILE_D]
                        matmul(l0c[l0c_cnt], l1qn[l1qk_cnt], l1kn[l1qk_cnt], is_init=False)
                        bar_m()
                        l1qk_cnt += 1

                    l1qr[l1qk_cnt] <<= q_rope[q_row:q_row + valid_m, 0:D_ROPE]
                    l1kr[l1qk_cnt] <<= k_rope[kv_row:kv_row + valid_n, 0:D_ROPE]
                    matmul(l0c[l0c_cnt], l1qr[l1qk_cnt], l1kr[l1qk_cnt], is_init=False)
                    bar_m()
                    l1qk_cnt += 1

                    qk_mutex.lock()
                    score_ws[cube_idx, stage1_slot, 0:TILE_M, 0:TILE_N] <<= l0c[l0c_cnt]
                    qk_mutex.ready()

                    qk_mutex.wait()
                    ub_score_tile <<= score_ws[cube_idx, stage1_slot, sb_row:sb_row + HALF_M, 0:TILE_N]

                    muls(ub_score_tile, ub_score_tile, scale)
                    if valid_n < TILE_N:
                        apply_score_tail_mask(ub_score_tile, valid_n)
                    vmax(ub_tmp, ub_score_tile[0:HALF_M, 0:HALF_N], ub_score_tile[0:HALF_M, HALF_N:TILE_N])
                    cmax(ub_max_s, ub_tmp)

                    add(expdiff_buf[stage1_slot], ub_rmax_s, ub_zero_s)
                    vmax(ub_rmax_s, ub_rmax_s, ub_max_s)
                    sub(expdiff_buf[stage1_slot], expdiff_buf[stage1_slot], ub_rmax_s)
                    exp(expdiff_buf[stage1_slot], expdiff_buf[stage1_slot])

                    brcb(ub_max, ub_rmax_s, repeat=HALF_M // 8, dst_blk_stride=1, dst_rep_stride=8)
                    sub(ub_score_tile[0:HALF_M, 0:HALF_N], ub_score_tile[0:HALF_M, 0:HALF_N], ub_max)
                    sub(ub_score_tile[0:HALF_M, HALF_N:TILE_N], ub_score_tile[0:HALF_M, HALF_N:TILE_N], ub_max)
                    if local_valid_m < HALF_M:
                        apply_score_row_tail_mask_after_shift(ub_score_tile, local_valid_m)

                    exp(ub_score_tile, ub_score_tile)
                    add(ub_tmp, ub_score_tile[0:HALF_M, 0:HALF_N], ub_score_tile[0:HALF_M, HALF_N:TILE_N])
                    cadd(ub_sum_s, ub_tmp)
                    mul(ub_rsum_s, ub_rsum_s, expdiff_buf[stage1_slot])
                    add(ub_rsum_s, ub_rsum_s, ub_sum_s)

                    cast(ub_p_tile, ub_score_tile)

                    p_mutex.lock()
                    p_ws[cube_idx, stage1_slot, sb_row:sb_row + HALF_M, 0:TILE_N] <<= ub_p_tile
                    p_mutex.ready()
                    qk_mutex.free()

                    l0c_cnt += 1
                    stage1_cnt += 1

            if ni > 0:
                with auto_sync():
                    prev_nt = Var(ni - 1)
                    prev_n_off = Var(prev_nt * TILE_N)
                    prev_valid_n = Min(TILE_N, SKV - prev_n_off)
                    stage2_slot = var_and(stage2_cnt, 1)
                    accum_read_slot = var_and(accum_cnt, 1)
                    accum_write_slot = var_and(accum_cnt + 1, 1)
                    v_row = Var(kv_base + prev_n_off)

                    p_mutex.wait()
                    l1p[l1pv_cnt] <<= p_ws[cube_idx, stage2_slot, 0:TILE_M, 0:TILE_N]
                    p_mutex.free()
                    accum_ws_valid.wait()

                    for doff in range(0, D_NOPE, TILE_D):
                        dchunk = var_shr(doff, 7)
                        pv_slot = var_and(l1v_cnt, 1)
                        l1v[l1v_cnt] <<= v[v_row:v_row + prev_valid_n, doff:doff + TILE_D]
                        matmul(
                            l0c[l0c_cnt],
                            l1p[l1pv_cnt],
                            l1v[l1v_cnt].T,
                            m=TILE_M,
                            n=TILE_D,
                            k=prev_valid_n,
                            is_init=True,
                        )

                        pv_mutex.lock()
                        pv_ws[cube_idx, pv_slot, 0:TILE_M, 0:TILE_D] <<= l0c[l0c_cnt]
                        pv_mutex.ready()

                        pv_mutex.wait()
                        for row8 in range(0, HALF_M, ACCUM_ROWS):
                            acc_row = Var(sb_row + row8)
                            ub_pv <<= pv_ws[cube_idx, pv_slot, acc_row:acc_row + ACCUM_ROWS, 0:TILE_D]
                            if accum_cnt == 0:
                                dup(ub_accum, 0.0)
                            else:
                                ub_accum <<= accum_ws[
                                    cube_idx,
                                    accum_read_slot,
                                    dchunk,
                                    acc_row:acc_row + ACCUM_ROWS,
                                    0:TILE_D,
                                ]
                            bar_mte2()

                            brcb(
                                ub_expdiff,
                                expdiff_buf[stage2_slot][0:1, row8:row8 + ACCUM_ROWS],
                                repeat=ACCUM_ROWS // 8,
                                dst_blk_stride=1,
                                dst_rep_stride=8,
                            )
                            mul(ub_accum[0:ACCUM_ROWS, 0:HALF_N], ub_accum[0:ACCUM_ROWS, 0:HALF_N], ub_expdiff)
                            mul(ub_accum[0:ACCUM_ROWS, HALF_N:TILE_D], ub_accum[0:ACCUM_ROWS, HALF_N:TILE_D], ub_expdiff)
                            add(ub_accum, ub_accum, ub_pv)
                            accum_store_ready.set()
                            accum_store_ready.wait()
                            accum_ws[cube_idx, accum_write_slot, dchunk, acc_row:acc_row + ACCUM_ROWS, 0:TILE_D] <<= ub_accum
                            accum_store_done.set()
                            accum_store_done.wait()
                        pv_mutex.free()

                        l1v_cnt += 1
                        l0c_cnt += 1

                    accum_ws_valid.set()
                    l1pv_cnt += 1
                    accum_cnt += 1
                    stage2_cnt += 1

        with auto_sync():
            final_accum_slot = var_and(accum_cnt, 1)
            accum_ws_valid.wait()
            for doff in range(0, D_NOPE, TILE_D):
                dchunk = var_shr(doff, 7)
                for row8 in range(0, HALF_M, ACCUM_ROWS):
                    acc_row = Var(sb_row + row8)
                    row_valid = Min(ACCUM_ROWS, Max(local_valid_m - row8, 0))
                    ub_accum <<= accum_ws[cube_idx, final_accum_slot, dchunk, acc_row:acc_row + ACCUM_ROWS, 0:TILE_D]
                    bar_mte2()
                    brcb(
                        ub_rowsum,
                        ub_rsum_s[0:1, row8:row8 + ACCUM_ROWS],
                        repeat=ACCUM_ROWS // 8,
                        dst_blk_stride=1,
                        dst_rep_stride=8,
                    )
                    div(ub_accum[0:ACCUM_ROWS, 0:HALF_N], ub_accum[0:ACCUM_ROWS, 0:HALF_N], ub_rowsum)
                    div(ub_accum[0:ACCUM_ROWS, HALF_N:TILE_D], ub_accum[0:ACCUM_ROWS, HALF_N:TILE_D], ub_rowsum)
                    if row_valid > 0:
                        out_row = Var(q_row + acc_row)
                        accum_store_ready.set()
                        accum_store_ready.wait()
                        out[out_row:out_row + row_valid, doff:doff + TILE_D] <<= ub_accum[0:row_valid, 0:TILE_D]
                        accum_store_done.set()
                        accum_store_done.wait()
        accum_store_valid.set()
        accum_ws_valid.set()

    accum_ws_valid.wait()
    return out


@lru_cache(maxsize=2)
def build_kernel(device):
    if device not in ("a2", "a3"):
        raise ValueError("Shared tensor-vector source supports A2/A3")
    return import_module(f"ascriptor.{device}").kernel()(mla_b1_hq8_hkv4_kernel)
