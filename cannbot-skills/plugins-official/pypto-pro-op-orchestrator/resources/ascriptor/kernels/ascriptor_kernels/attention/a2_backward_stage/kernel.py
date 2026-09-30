# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A2/A3 dense attention backward stage: dQK in BF16, gV through a HiFloat8 probability store.

One kernel body is bound to either facade by build_kernel(device); the tensor-vector
vocabulary is common to both, so the algorithm is written once."""

from ascriptor.a2 import *
from functools import lru_cache
from importlib import import_module

TILE_M = 128
TILE_N = 128
TILE_K = 128
HALF_M = TILE_M // 2
HALF_N = TILE_N // 2
SUBS_M = 32
HIF8_CHUNK_M = 16
HIF8_MIN_NORMAL = 2.0 ** -23
HIF8_MIN_CLAMP = 2.0 ** -22
EXP_MASK = 0x7F800000
EXPABS_BIAS = -0x00800000
EXPABS_LE15 = 32768.0
EXPABS_LE7 = 128.0
EXPABS_LE3 = 8.0


@func()
def build_suffix_invalid_mask(valid_cols: Var, out_mask: Var):
    signed_mask = Var(-1, DT.int64)
    two_i64 = Var(2, DT.int64)
    for _ in range(0, valid_cols):
        signed_mask <<= signed_mask * two_i64
    out_mask <<= signed_mask


@func()
def zero_half_suffix_invalid(buf: Tensor, valid_cols: Var):
    if valid_cols == 0:
        dup(buf, 0.0)
    elif valid_cols < HALF_N:
        suffix_mask = Var(0, DT.uint64)
        build_suffix_invalid_mask(valid_cols, suffix_mask)
        set_mask(0, suffix_mask)
        dup(buf, 0.0)
        reset_mask()


@func()
def apply_prob_tail_mask(prob_buf: Tensor, valid_n: Var):
    left_valid = Min(valid_n, HALF_N)
    right_valid = Max(valid_n - HALF_N, 0)
    zero_half_suffix_invalid(prob_buf[:, 0:HALF_N], left_valid)
    zero_half_suffix_invalid(prob_buf[:, HALF_N:TILE_N], right_valid)


@func()
def quantize_prob_chunk_nonneg_simple(
    prob_chunk: Tensor,
    meta_chunk: Tensor,
    scale_chunk: Tensor,
    factor_chunk: Tensor,
    one_chunk: Tensor,
    keepflag_chunk: Tensor,
    flag_chunk: Tensor,
    expmask_u32: Tensor,
):
    meta_chunk <<= prob_chunk

    x_u16 = meta_chunk.reinterpret(DT.uint16)
    scale_u16 = scale_chunk.reinterpret(DT.uint16)
    expmask_u16 = expmask_u32.reinterpret(DT.uint16)
    vand(scale_u16, x_u16, expmask_u16)

    expabs_u16 = meta_chunk.reinterpret(DT.uint16)
    vnot(expabs_u16, scale_u16)
    vand(expabs_u16, expabs_u16, expmask_u16)
    expabs_i32 = meta_chunk.reinterpret(DT.int)
    adds(expabs_i32, expabs_i32, EXPABS_BIAS)
    vmax(meta_chunk, meta_chunk, scale_chunk)

    compare_scalar(keepflag_chunk, scale_chunk, HIF8_MIN_NORMAL, CompareMode.GE)
    vmaxs(scale_chunk, scale_chunk, HIF8_MIN_CLAMP)

    dup(factor_chunk, 0.5)
    compare_scalar(flag_chunk, meta_chunk, EXPABS_LE15, CompareMode.LE)
    select(factor_chunk, flag_chunk, factor_chunk, one_chunk, SelectMode.TENSOR_SCALAR)
    mul(scale_chunk, scale_chunk, factor_chunk)

    dup(factor_chunk, 0.5)
    compare_scalar(flag_chunk, meta_chunk, EXPABS_LE7, CompareMode.LE)
    select(factor_chunk, flag_chunk, factor_chunk, one_chunk, SelectMode.TENSOR_SCALAR)
    mul(scale_chunk, scale_chunk, factor_chunk)

    dup(factor_chunk, 0.5)
    compare_scalar(flag_chunk, meta_chunk, EXPABS_LE3, CompareMode.LE)
    select(factor_chunk, flag_chunk, factor_chunk, one_chunk, SelectMode.TENSOR_SCALAR)
    mul(scale_chunk, scale_chunk, factor_chunk)

    div(prob_chunk, prob_chunk, scale_chunk)
    adds(prob_chunk, prob_chunk, 0.5)
    roundint = meta_chunk.reinterpret(DT.int)
    cast(roundint, prob_chunk, round_mode=RoundMode.TRUNC)
    cast(prob_chunk, roundint)
    mul(prob_chunk, prob_chunk, scale_chunk)
    dup(factor_chunk, 0.0)
    select(prob_chunk, keepflag_chunk, prob_chunk, factor_chunk, SelectMode.TENSOR_SCALAR)


ZERO_ALLVEC_FLAG = 2
ZERO_VEC_TO_CUBE_FLAG = 3
FINAL_CUBE_TO_VEC_FLAG = 4
FINAL_ALLCUBE_FLAG = 5
A2_CUBE_NUM = 20
A2_VEC_LANE_NUM = A2_CUBE_NUM * 2


def attn_backward_dense_total_tail_stage1_prob_dqk_gq_gk_gv_hif8_output_cast_kernel(
    q: GM[bf16, ('B', 'H', 'S1', 'D')], k: GM[bf16, ('B', 'H', 'S2', 'D')], v: GM[bf16, ('B', 'H', 'S2', 'D')], o: GM[bf16, ('B', 'H', 'S1', 'D')], grad: GM[bf16, ('B', 'H', 'S1', 'D')], qkmax: GM[f32, ('B', 'H', 'S1')], qksum: GM[f32, ('B', 'H', 'S1')],
    gq_out: GM[bf16, ('B', 'H', 'S1', 'D')], gk_out: GM[bf16, ('B', 'H', 'S2', 'D')], gv_out: GM[bf16, ('B', 'H', 'S2', 'D')],
    B: i32, H: i32, S1: i32, S2: i32, D: i32, scale: f32,
):
    # Dense attention backward final a2 kernel with bf16 outputs:
    # qk equals q.float() @ k.float().t().
    # dp equals grad.float() @ v.float().t().
    # prob equals exp(qk * scale - qkmax) / qksum.
    # prob_hif8 equals hif8_quantize_positive_finite(prob).
    # dqk equals prob * (dp - sum(o * grad)) * scale.
    # Accumulate gq_fp32 with dqk.bfloat16().float() @ k.float().
    # Accumulate gk_fp32 with dqk.bfloat16().float().transpose(-1, -2) @ q.float().
    # Accumulate gv_fp32 with prob_hif8.bfloat16().float().transpose(-1, -2) @ grad.float().
    # gq/gk/gv = fp32_accum.cast(qkv_dtype)
    BH = Var(B * H)
    Q_ROWS = Var(BH * S1)
    KV_ROWS = Var(BH * S2)

    q_flat = q.reshape([Q_ROWS, D], name="q_flat")
    k_flat = k.reshape([KV_ROWS, D], name="k_flat")
    v_flat = v.reshape([KV_ROWS, D], name="v_flat")
    o_flat = o.reshape([Q_ROWS, D], name="o_flat")
    grad_flat = grad.reshape([Q_ROWS, D], name="grad_flat")
    qkmax_flat = qkmax.reshape([BH, S1], name="qkmax_flat")
    qksum_flat = qksum.reshape([BH, S1], name="qksum_flat")
    gq_flat = gq_out.reshape([Q_ROWS, D], name="gq_out_flat")
    gk_flat = gk_out.reshape([KV_ROWS, D], name="gk_out_flat")
    gv_flat = gv_out.reshape([KV_ROWS, D], name="gv_out_flat")

    # RFC-0009 batch 3: the four 2-slot rings become GMBuffs so the gmbuff pass machine-checks
    # the ring algebra (single beat, lag < slots, mutex depth <= slots). No protocol change.
    # The gq/gk/gv accumulators below stay split_workspace: they are core-shared flat surfaces
    # fed by atomic_add, not rings - there is no beat to check.
    qk_ws = GMBuff(DT.float, [TILE_M, TILE_N], slots=2, name="qk_ws")
    dp_ws = GMBuff(DT.float, [TILE_M, TILE_N], slots=2, name="dp_ws")
    p_ws = GMBuff(DT.bfloat16, [TILE_M, TILE_N], slots=2, name="p_ws")
    dqk_ws = GMBuff(DT.bfloat16, [TILE_M, TILE_N], slots=2, name="dqk_ws")
    gq_acc_ws = split_workspace(DT.float, [Q_ROWS, D], name="gq_acc_ws")
    gk_acc_ws = split_workspace(DT.float, [KV_ROWS, D], name="gk_acc_ws")
    gv_acc_ws = split_workspace(DT.float, [KV_ROWS, D], name="gv_acc_ws")

    l1q = DBuff(DT.bfloat16, [TILE_M, TILE_K], Position.L1)
    l1g = DBuff(DT.bfloat16, [TILE_M, TILE_K], Position.L1)
    l1k = TBuff(DT.bfloat16, [TILE_N, TILE_K], Position.L1)
    l1v = DBuff(DT.bfloat16, [TILE_N, TILE_K], Position.L1)
    l1p = DBuff(DT.bfloat16, [TILE_M, TILE_N], Position.L1)
    l1dqk = DBuff(DT.bfloat16, [TILE_M, TILE_N], Position.L1)
    l0c = DBuff(DT.float, [TILE_M, TILE_N], Position.L0C)

    qkbuf = DBuff(DT.float, [HIF8_CHUNK_M, TILE_N], Position.UB)
    dpbuf = DBuff(DT.float, [HIF8_CHUNK_M, TILE_N], Position.UB)
    cast_fp32buf = DBuff(DT.float, [HIF8_CHUNK_M, TILE_K], Position.UB)
    cast_outbuf = DBuff(gq_out.dtype, [HIF8_CHUNK_M, TILE_K], Position.UB)
    pbuf = DBuff(DT.bfloat16, [HIF8_CHUNK_M, TILE_N], Position.UB)
    dqkbuf = DBuff(DT.bfloat16, [HIF8_CHUNK_M, TILE_N], Position.UB)
    obuf = Tensor(DT.bfloat16, [HALF_M, TILE_K], Position.UB)
    gradbuf = Tensor(DT.bfloat16, [HALF_M, TILE_K], Position.UB)
    ofp32buf = Tensor(DT.float, [SUBS_M, TILE_K], Position.UB)
    gradfp32buf = Tensor(DT.float, [SUBS_M, TILE_K], Position.UB)
    qkmaxbuf = Tensor(DT.float, [1, HALF_M], Position.UB)
    qksumbuf = Tensor(DT.float, [1, HALF_M], Position.UB)
    qkmaxbrcb = Tensor(DT.float, [HALF_M, 8], Position.UB)
    qksumbrcb = Tensor(DT.float, [HALF_M, 8], Position.UB)
    odobuf = Tensor(DT.float, [1, SUBS_M], Position.UB)
    odobrcb = Tensor(DT.float, [HALF_M, 8], Position.UB)
    quant_meta = Tensor(DT.float, [HIF8_CHUNK_M, TILE_N], Position.UB)
    quant_scale = Tensor(DT.float, [HIF8_CHUNK_M, TILE_N], Position.UB)
    quant_factor = Tensor(DT.float, [HIF8_CHUNK_M, TILE_N], Position.UB)
    quant_one = Tensor(DT.float, [1, TILE_N], Position.UB)
    quant_keepflag = Tensor(DT.uint8, [HIF8_CHUNK_M, TILE_N], Position.UB)
    quant_flag = Tensor(DT.uint8, [HIF8_CHUNK_M, TILE_N], Position.UB)
    expmask_u32 = Tensor(DT.uint32, [HIF8_CHUNK_M, TILE_N], Position.UB)

    qkdp_mutex = CvMutex(0, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)
    pdqk_mutex = VcMutex(1, depth=2, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)

    l1q_cnt = Var(0)
    l1g_cnt = Var(0)
    l1k_cnt = Var(0)
    l1v_cnt = Var(0)
    l1p_cnt = Var(0)
    l1dqk_cnt = Var(0)
    l0c_cnt = Var(0)
    stage2_cnt = Var(0)
    vec_in_cnt = Var(0)
    vec_out_cnt = Var(0)
    zero_ws_cnt = Var(0)
    cast_in_cnt = Var(0)
    cast_out_cnt = Var(0)

    cube_idx = GetCubeIdx()
    sb = GetSubBlockIdx()
    vec_lane_id = Var(cube_idx * 2 + sb)
    vec_lane_count = Var(GetCubeNum() * 2)
    q_cast_chunks = CeilDiv(Q_ROWS, HIF8_CHUNK_M)
    kv_cast_chunks = CeilDiv(KV_ROWS, HIF8_CHUNK_M)

    tiles_m = CeilDiv(S1, TILE_M)
    tiles_n = CeilDiv(S2, TILE_N)
    total_m = Var(BH * tiles_m)
    per_core = CeilDiv(total_m, GetCubeNum())
    mt_begin = Var(per_core * cube_idx)
    mt_end = Min(mt_begin + per_core, total_m)

    with vec_scope():
        dup(quant_one, 1.0)
        dup(expmask_u32, EXP_MASK)

        with auto_sync():
            for chunk_idx in range(0, q_cast_chunks):
                if var_mod(chunk_idx, vec_lane_count) == vec_lane_id:
                    row0 = Var(chunk_idx * HIF8_CHUNK_M)
                    valid_rows = Min(HIF8_CHUNK_M, Q_ROWS - row0)
                    zero_tile = cast_fp32buf[zero_ws_cnt]
                    dup(zero_tile, 0.0)
                    gq_acc_ws[row0:row0 + valid_rows, 0:D] <<= zero_tile[0:valid_rows, 0:D]
                    zero_ws_cnt += 1

            for chunk_idx in range(0, kv_cast_chunks):
                if var_mod(chunk_idx, vec_lane_count) == vec_lane_id:
                    row0 = Var(chunk_idx * HIF8_CHUNK_M)
                    valid_rows = Min(HIF8_CHUNK_M, KV_ROWS - row0)
                    zero_tile = cast_fp32buf[zero_ws_cnt]
                    dup(zero_tile, 0.0)
                    gk_acc_ws[row0:row0 + valid_rows, 0:D] <<= zero_tile[0:valid_rows, 0:D]
                    zero_ws_cnt += 1

            for chunk_idx in range(0, kv_cast_chunks):
                if var_mod(chunk_idx, vec_lane_count) == vec_lane_id:
                    row0 = Var(chunk_idx * HIF8_CHUNK_M)
                    valid_rows = Min(HIF8_CHUNK_M, KV_ROWS - row0)
                    zero_tile = cast_fp32buf[zero_ws_cnt]
                    dup(zero_tile, 0.0)
                    gv_acc_ws[row0:row0 + valid_rows, 0:D] <<= zero_tile[0:valid_rows, 0:D]
                    zero_ws_cnt += 1

        allvec_ready(ZERO_ALLVEC_FLAG)
        allvec_wait(ZERO_ALLVEC_FLAG)
        vec_ready(ZERO_VEC_TO_CUBE_FLAG)

    wait_vec(ZERO_VEC_TO_CUBE_FLAG)

    for gmt in range(mt_begin, mt_end):
        bh = Var(gmt // tiles_m)
        lmt = Var(gmt % tiles_m)
        row_in_bh = Var(lmt * TILE_M)
        q_row = Var(bh * S1 + row_in_bh)
        kv_base = Var(bh * S2)
        valid_m = Min(TILE_M, S1 - row_in_bh)

        half_rows = CeilDiv(valid_m, 2)
        row_begin = Var(sb * half_rows)
        row_end = Min(row_begin + half_rows, valid_m)
        row_count = Var(row_end - row_begin)
        vec_row_in_bh = Var(row_in_bh + row_begin)
        vec_row = Var(q_row + row_begin)

        qbuf = l1q[l1q_cnt]
        gbuf = l1g[l1g_cnt]

        with auto_sync():
            # Invalid score rows are masked; later transposed products use valid_m.

            qbuf <<= q_flat[q_row:q_row + valid_m, 0:D]
            gbuf <<= grad_flat[q_row:q_row + valid_m, 0:D]

            if row_count > 0:
                bar_all()
                if row_count < HALF_M:
                    dup(obuf.reinterpret(DT.int), 0)
                    dup(gradbuf.reinterpret(DT.int), 0)
                    dup(qkmaxbuf, 0.0)
                    dup(qksumbuf, 0.0)
                    bar_all()

                obuf[0:row_count, 0:D] <<= o_flat[vec_row:vec_row + row_count, 0:D]
                gradbuf[0:row_count, 0:D] <<= grad_flat[vec_row:vec_row + row_count, 0:D]
                qkmaxbuf[0:1, 0:row_count] <<= qkmax_flat[bh:bh + 1, vec_row_in_bh:vec_row_in_bh + row_count]
                qksumbuf[0:1, 0:row_count] <<= qksum_flat[bh:bh + 1, vec_row_in_bh:vec_row_in_bh + row_count]

                brcb(qkmaxbrcb, qkmaxbuf, repeat=HALF_M // 8, dst_blk_stride=1, dst_rep_stride=8)
                brcb(qksumbrcb, qksumbuf, repeat=HALF_M // 8, dst_blk_stride=1, dst_rep_stride=8)

                for subs in range(0, HALF_M, SUBS_M):
                    cast(ofp32buf, obuf[subs:subs + SUBS_M, 0:D])
                    cast(gradfp32buf, gradbuf[subs:subs + SUBS_M, 0:D])

                    mul(ofp32buf[:, 0:64], ofp32buf[:, 0:64], gradfp32buf[:, 0:64])
                    mul(ofp32buf[:, 64:128], ofp32buf[:, 64:128], gradfp32buf[:, 64:128])
                    add(ofp32buf[:, 0:64], ofp32buf[:, 0:64], ofp32buf[:, 64:128])
                    cadd(odobuf, ofp32buf[:, 0:64])
                    brcb(
                        odobrcb[subs:subs + SUBS_M, :], odobuf,
                        repeat=SUBS_M // 8, dst_blk_stride=1, dst_rep_stride=8,
                    )

        for ni in range(0, tiles_n + 1):
            if ni < tiles_n:
                n_off = Var(ni * TILE_N)
                kv_row = Var(kv_base + n_off)
                valid_n = Min(TILE_N, S2 - n_off)
                kbuf = l1k[l1k_cnt]
                vbuf = l1v[l1v_cnt]

                with auto_sync():
                    # Invalid key columns are masked; gQ contracts only prev_valid_n.

                    kbuf <<= k_flat[kv_row:kv_row + valid_n, 0:D]
                    vbuf <<= v_flat[kv_row:kv_row + valid_n, 0:D]

                    matmul(l0c[l0c_cnt], qbuf, kbuf, is_init=True)
                    qkdp_mutex.lock()
                    qk_ws[ni, 0:TILE_M, 0:TILE_N] <<= l0c[l0c_cnt]
                    l0c_cnt += 1

                    matmul(l0c[l0c_cnt], gbuf, vbuf, is_init=True)
                    dp_ws[ni, 0:TILE_M, 0:TILE_N] <<= l0c[l0c_cnt]
                    qkdp_mutex.ready()

                    qkdp_mutex.wait()
                    pdqk_mutex.lock()
                    if row_count > 0:
                        quat_tiles = CeilDiv(row_count, HIF8_CHUNK_M)
                        for qmt in range(0, quat_tiles):
                            chunk_off = Var(qmt * HIF8_CHUNK_M)
                            chunk_rows = Min(HIF8_CHUNK_M, row_count - chunk_off)

                            qk_tile = qkbuf[vec_in_cnt]
                            dp_tile = dpbuf[vec_in_cnt]
                            p_tile = pbuf[vec_out_cnt]
                            dqk_tile = dqkbuf[vec_out_cnt]

                            qk_tile[0:chunk_rows, 0:TILE_N] <<= qk_ws[
                                ni, row_begin + chunk_off:row_begin + chunk_off + chunk_rows, 0:TILE_N
                            ]
                            dp_tile[0:chunk_rows, 0:TILE_N] <<= dp_ws[
                                ni, row_begin + chunk_off:row_begin + chunk_off + chunk_rows, 0:TILE_N
                            ]
                            if chunk_rows < HIF8_CHUNK_M:
                                dup(qk_tile[chunk_rows:HIF8_CHUNK_M, 0:TILE_N], 0.0)
                                dup(dp_tile[chunk_rows:HIF8_CHUNK_M, 0:TILE_N], 0.0)

                            muls(qk_tile, qk_tile, scale)
                            sub(qk_tile[:, 0:64], qk_tile[:, 0:64], qkmaxbrcb[chunk_off:chunk_off + HIF8_CHUNK_M, :])
                            sub(qk_tile[:, 64:128], qk_tile[:, 64:128], qkmaxbrcb[chunk_off:chunk_off + HIF8_CHUNK_M, :])
                            exp(qk_tile, qk_tile)
                            div(qk_tile[:, 0:64], qk_tile[:, 0:64], qksumbrcb[chunk_off:chunk_off + HIF8_CHUNK_M, :])
                            div(qk_tile[:, 64:128], qk_tile[:, 64:128], qksumbrcb[chunk_off:chunk_off + HIF8_CHUNK_M, :])
                            if valid_n < TILE_N:
                                apply_prob_tail_mask(qk_tile, valid_n)
                            if chunk_rows < HIF8_CHUNK_M:
                                dup(qk_tile[chunk_rows:HIF8_CHUNK_M, 0:TILE_N], 0.0)

                            sub(dp_tile[:, 0:64], dp_tile[:, 0:64], odobrcb[chunk_off:chunk_off + HIF8_CHUNK_M, :])
                            sub(dp_tile[:, 64:128], dp_tile[:, 64:128], odobrcb[chunk_off:chunk_off + HIF8_CHUNK_M, :])
                            mul(dp_tile, qk_tile, dp_tile)
                            muls(dp_tile, dp_tile, scale)

                            cast(dqk_tile, dp_tile)
                            quantize_prob_chunk_nonneg_simple(
                                qk_tile, quant_meta, quant_scale, quant_factor, quant_one,
                                quant_keepflag, quant_flag, expmask_u32,
                            )
                            cast(p_tile, qk_tile)

                            p_ws[
                                ni, row_begin + chunk_off:row_begin + chunk_off + chunk_rows, 0:TILE_N
                            ] <<= p_tile[0:chunk_rows, 0:TILE_N]
                            dqk_ws[
                                ni, row_begin + chunk_off:row_begin + chunk_off + chunk_rows, 0:TILE_N
                            ] <<= dqk_tile[0:chunk_rows, 0:TILE_N]

                            vec_in_cnt += 1
                            vec_out_cnt += 1
                    # Tail subblocks can own zero rows on the last M tile. They still
                    # must complete the vec->cube handoff so the cube side does not
                    # wait forever for a lane that had no producer writes this round.
                    pdqk_mutex.ready()
                    qkdp_mutex.free()

                    l1k_cnt += 1
                    l1v_cnt += 1
                    l0c_cnt += 1

            if ni > 0:
                prev_nt = Var(ni - 1)
                prev_n_off = Var(prev_nt * TILE_N)
                prev_valid_n = Min(TILE_N, S2 - prev_n_off)
                prev_k_row = Var(kv_base + prev_n_off)
                pslot = l1p[l1p_cnt]
                dqkslot = l1dqk[l1dqk_cnt]
                kgqbuf = l1k[stage2_cnt]

                with auto_sync():
                    pdqk_mutex.wait()

                    # Preserve only valid query rows in P/dQK, without overlapping fills.

                    pslot[0:valid_m, 0:TILE_N] <<= p_ws[ni - 1, 0:valid_m, 0:TILE_N]
                    dqkslot[0:valid_m, 0:TILE_N] <<= dqk_ws[ni - 1, 0:valid_m, 0:TILE_N]

                    matmul(l0c[l0c_cnt], pslot.T, gbuf.T, k=valid_m, is_init=True)
                    with atomic_add():
                        gv_acc_ws[prev_k_row:prev_k_row + prev_valid_n, 0:D] <<= l0c[l0c_cnt][0:prev_valid_n, 0:D]
                    l0c_cnt += 1

                    matmul(l0c[l0c_cnt], dqkslot.T, qbuf.T, k=valid_m, is_init=True)
                    with atomic_add():
                        gk_acc_ws[prev_k_row:prev_k_row + prev_valid_n, 0:D] <<= l0c[l0c_cnt][0:prev_valid_n, 0:D]
                    l0c_cnt += 1

                    matmul(l0c[l0c_cnt], dqkslot, kgqbuf.T, k=prev_valid_n, is_init=True)
                    with atomic_add():
                        gq_acc_ws[q_row:q_row + valid_m, 0:D] <<= l0c[l0c_cnt][0:valid_m, 0:D]
                    pdqk_mutex.free()

                    l1p_cnt += 1
                    l1dqk_cnt += 1
                    l0c_cnt += 1
                    stage2_cnt += 1

        l1q_cnt += 1
        l1g_cnt += 1

    allcube_ready(FINAL_ALLCUBE_FLAG)
    allcube_wait(FINAL_ALLCUBE_FLAG)
    cube_ready(FINAL_CUBE_TO_VEC_FLAG)

    with vec_scope():
        wait_cube(FINAL_CUBE_TO_VEC_FLAG)

        with auto_sync():
            for chunk_idx in range(0, q_cast_chunks):
                if var_mod(chunk_idx, vec_lane_count) == vec_lane_id:
                    row0 = Var(chunk_idx * HIF8_CHUNK_M)
                    valid_rows = Min(HIF8_CHUNK_M, Q_ROWS - row0)
                    fp32_tile = cast_fp32buf[cast_in_cnt]
                    out_tile = cast_outbuf[cast_out_cnt]

                    fp32_tile[0:valid_rows, 0:D] <<= gq_acc_ws[row0:row0 + valid_rows, 0:D]
                    if valid_rows < HIF8_CHUNK_M:
                        dup(fp32_tile[valid_rows:HIF8_CHUNK_M, 0:D], 0.0)
                    cast(out_tile, fp32_tile)
                    gq_flat[row0:row0 + valid_rows, 0:D] <<= out_tile[0:valid_rows, 0:D]

                    cast_in_cnt += 1
                    cast_out_cnt += 1

            for chunk_idx in range(0, kv_cast_chunks):
                if var_mod(chunk_idx, vec_lane_count) == vec_lane_id:
                    row0 = Var(chunk_idx * HIF8_CHUNK_M)
                    valid_rows = Min(HIF8_CHUNK_M, KV_ROWS - row0)
                    fp32_tile = cast_fp32buf[cast_in_cnt]
                    out_tile = cast_outbuf[cast_out_cnt]

                    fp32_tile[0:valid_rows, 0:D] <<= gk_acc_ws[row0:row0 + valid_rows, 0:D]
                    if valid_rows < HIF8_CHUNK_M:
                        dup(fp32_tile[valid_rows:HIF8_CHUNK_M, 0:D], 0.0)
                    cast(out_tile, fp32_tile)
                    gk_flat[row0:row0 + valid_rows, 0:D] <<= out_tile[0:valid_rows, 0:D]

                    cast_in_cnt += 1
                    cast_out_cnt += 1

            for chunk_idx in range(0, kv_cast_chunks):
                if var_mod(chunk_idx, vec_lane_count) == vec_lane_id:
                    row0 = Var(chunk_idx * HIF8_CHUNK_M)
                    valid_rows = Min(HIF8_CHUNK_M, KV_ROWS - row0)
                    fp32_tile = cast_fp32buf[cast_in_cnt]
                    out_tile = cast_outbuf[cast_out_cnt]

                    fp32_tile[0:valid_rows, 0:D] <<= gv_acc_ws[row0:row0 + valid_rows, 0:D]
                    if valid_rows < HIF8_CHUNK_M:
                        dup(fp32_tile[valid_rows:HIF8_CHUNK_M, 0:D], 0.0)
                    cast(out_tile, fp32_tile)
                    gv_flat[row0:row0 + valid_rows, 0:D] <<= out_tile[0:valid_rows, 0:D]

                    cast_in_cnt += 1
                    cast_out_cnt += 1

    return gq_out, gk_out, gv_out


@lru_cache(maxsize=2)
def build_kernel(device):
    if device not in ("a2", "a3"):
        raise ValueError("Shared tensor-vector backward stage supports A2/A3")
    return import_module(f"ascriptor.{device}").kernel()(attn_backward_dense_total_tail_stage1_prob_dqk_gq_gk_gv_hif8_output_cast_kernel)
