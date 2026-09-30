# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Two per-task GM metadata ABIs over one unscaled-HiFloat8 FMA attention body.

meta8 carries eight int32 fields per (query tile, key tile) task; meta11 adds the cache slot,
the q-buffer parity and the first/last-key flags so the loop reads them instead of computing
them. Scalar geometry that lives in GM is loaded explicitly through GetValueFrom on a
one-element GM view."""

import math

from ascriptor.a5 import *

# Ported from two source scripts, then hand edited:
# Hand edits: explicit A5 binding, typed carrier/metadata ABI, fixed CACHE3/PRELOAD2.
# Hand edits: scalar loads use explicit GetValueFrom(one-element GM view).
# Hand edits: D-214 tail PV contracts valid V rows; no overlapping full-L1 zero fill.
# Original per-task loads, unscaled P, FMA and publication protocols retained.
_pyrange = unroll  # the old script aliased Python's range for compile-time unrolling
TILE_M = 128
ROWS_PER_SB = 64
QLANES = 64
TILE_N = 128
D_HEAD = 128
SOFTMAX_SCALE = 1.0 / math.sqrt(128)
HALF = 64
CHUNKS_D = 2
KROW = 64
M_Q = 64
CD = 2
PRELOAD_N = 2
CACHE = 3
MAX_CORE_COUNT = 32
MAX_FD_WS = 64
UNROLL = 4
RB_MAX = 32
RB_EXP = 16
NEG_LARGE = -1.0e30
META_F = 8
META_F_EXT = 11

@vf()
def init_softmax_state_vf(ub_rmax: Tensor, ub_rsum: Tensor):
    neg = Reg(DT.float); zero = Reg(DT.float)
    neg <<= NEG_LARGE; zero <<= 0.0
    ub_rmax[0:1, 0:QLANES] <<= neg
    ub_rsum[0:1, 0:QLANES] <<= zero


@vf()
def softmax_t_scale_vf(ub_score: Tensor, ub_rmax: Tensor, ub_rsum: Tensor, ub_p: Tensor, ub_old_weight: Tensor):
    sreg = RegList(DT.float, UNROLL); acc = RegList(DT.float, UNROLL); preg = RegList(DT.float, UNROLL)
    p_hi = RegList(DT.hif8, UNROLL); p_hj = RegList(DT.hif8, UNROLL)
    p_mask4 = MaskReg(DT.hif8, init_mode=MaskType.NONE); p_mask4 <<= Var(4 * QLANES, dtype=DT.uint32)
    cfg_i = CastConfig(round_mode=RoundMode.AWAY_FROM_ZERO, reg_layout=RegLayout.ZERO, name="p_hif8_i")
    neg = Reg(DT.float); block_max = Reg(DT.float); block_sum = Reg(DT.float); t = Reg(DT.float)
    v_pmax = Reg(DT.float); v_nmax = Reg(DT.float); v_oldw = Reg(DT.float)
    v_psum = Reg(DT.float); v_nsum = Reg(DT.float)
    neg <<= NEG_LARGE
    for u in unroll(UNROLL):
        acc[u] <<= neg
    for rb in range(RB_MAX):
        for u in unroll(UNROLL):
            sreg[u] <<= ub_score[(rb * UNROLL + u):(rb * UNROLL + u) + 1, :]
            acc[u] <<= acc[u].vmax(sreg[u])
    block_max <<= acc[0].vmax(acc[1]); t <<= acc[2].vmax(acc[3]); block_max <<= block_max.vmax(t)
    muls(block_max, block_max, SOFTMAX_SCALE)
    v_pmax <<= ub_rmax[0:1, 0:QLANES]
    v_nmax <<= block_max.vmax(v_pmax)
    expsub(v_oldw, v_pmax, v_nmax)
    for u in unroll(UNROLL):
        acc[u] <<= 0.0
    for rb in range(RB_EXP):
        for u in unroll(UNROLL):
            sreg[u] <<= ub_score[(rb * UNROLL + u):(rb * UNROLL + u) + 1, :]
            muls(sreg[u], sreg[u], SOFTMAX_SCALE)
            expsub(preg[u], sreg[u], v_nmax)
            acc[u] <<= acc[u] + preg[u]
            cast(p_hi[u], preg[u], cfg_i)
        for u in unroll(UNROLL):
            sreg[u] <<= ub_score[(rb * UNROLL + u + HALF):(rb * UNROLL + u + HALF) + 1, :]
            muls(sreg[u], sreg[u], SOFTMAX_SCALE)
            expsub(preg[u], sreg[u], v_nmax)
            acc[u] <<= acc[u] + preg[u]
            cast(p_hj[u], preg[u], cfg_i)
        for u in unroll(UNROLL):
            reg_to_ub_pack4(ub_p[(rb * UNROLL + u) * KROW], p_hi[u], mask=p_mask4)
            reg_to_ub_pack4(ub_p[((rb * UNROLL + u) + HALF) * KROW], p_hj[u], mask=p_mask4)
    block_sum <<= acc[0] + acc[1]; t <<= acc[2] + acc[3]; block_sum <<= block_sum + t
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)
    v_psum <<= ub_rsum[0:1, 0:QLANES]
    v_nsum <<= v_psum * v_oldw
    v_nsum <<= v_nsum + block_sum
    ub_rmax[0:1, 0:QLANES] <<= v_nmax
    ub_rsum[0:1, 0:QLANES] <<= v_nsum
    ub_old_weight[0:1, 0:QLANES] <<= v_oldw
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def softmax_t_scale_tail_vf(ub_score: Tensor, ub_rmax: Tensor, ub_rsum: Tensor, ub_p: Tensor, ub_old_weight: Tensor, valid_n: Var):
    s = Reg(DT.float); e = Reg(DT.float); neg = Reg(DT.float)
    block_max = Reg(DT.float); block_sum = Reg(DT.float); p_hif8 = Reg(DT.hif8)
    rowmask = MaskReg(DT.float, init_mode=MaskType.ALL)
    p_mask4 = MaskReg(DT.hif8, init_mode=MaskType.NONE); p_mask4 <<= Var(4 * QLANES, dtype=DT.uint32)
    cfg_i = CastConfig(round_mode=RoundMode.AWAY_FROM_ZERO, reg_layout=RegLayout.ZERO, name="p_hif8_tail_i")
    v_pmax = Reg(DT.float); v_nmax = Reg(DT.float); v_oldw = Reg(DT.float)
    v_psum = Reg(DT.float); v_nsum = Reg(DT.float)
    block_max <<= NEG_LARGE
    neg <<= NEG_LARGE
    for n in range(TILE_N):
        cnt_n = Var(QLANES * Min(Max(valid_n - n, 0), 1), dtype=DT.uint32)
        s <<= ub_score[n:n + 1, :]
        s <<= s * SOFTMAX_SCALE
        update_mask(rowmask, cnt_n)
        select(s, s, neg, mask=rowmask)
        block_max <<= block_max.vmax(s)
    v_pmax <<= ub_rmax[0:1, 0:QLANES]
    v_nmax <<= block_max.vmax(v_pmax)
    expsub(v_oldw, v_pmax, v_nmax)
    block_sum <<= 0.0
    for n in range(TILE_N):
        cnt_n = Var(QLANES * Min(Max(valid_n - n, 0), 1), dtype=DT.uint32)
        s <<= ub_score[n:n + 1, :]
        s <<= s * SOFTMAX_SCALE
        update_mask(rowmask, cnt_n)
        select(s, s, neg, mask=rowmask)
        expsub(e, s, v_nmax)
        block_sum <<= block_sum + e
        cast(p_hif8, e, cfg_i)
        reg_to_ub_pack4(ub_p[n * KROW], p_hif8, mask=p_mask4)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)
    v_psum <<= ub_rsum[0:1, 0:QLANES]
    v_nsum <<= v_psum * v_oldw
    v_nsum <<= v_nsum + block_sum
    ub_rmax[0:1, 0:QLANES] <<= v_nmax
    ub_rsum[0:1, 0:QLANES] <<= v_nsum
    ub_old_weight[0:1, 0:QLANES] <<= v_oldw
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def accum_pv_vf(ub_accum: Tensor, ub_pv: Tensor, ub_old_weight: Tensor):
    # acc = acc*old_weight + pv, fused via muldstadd (dst = dst*src0 + src1). Rows unrolled by UNROLL
    # to overlap the per-row load->FMA->store chains; each row is CHUNKS_D 64-lane fp32 regs.
    acc = RegList(DT.float, UNROLL * CHUNKS_D); pv = RegList(DT.float, UNROLL * CHUNKS_D)
    ow = RegList(DT.float, UNROLL)
    for rb in range(ROWS_PER_SB // UNROLL):
        for u in unroll(UNROLL):
            r = rb * UNROLL + u
            ow[u] <<= ub_old_weight[0:1, r:r + 1].single()
            for c in unroll(CHUNKS_D):
                k = u * CHUNKS_D + c
                acc[k] <<= ub_accum[r:r + 1, c * 64:(c + 1) * 64]
                pv[k] <<= ub_pv[r:r + 1, c * 64:(c + 1) * 64]
                muldstadd(acc[k], ow[u], pv[k])
                ub_accum[r:r + 1, c * 64:(c + 1) * 64] <<= acc[k]
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def accum_pv_first_vf(ub_accum: Tensor, ub_pv: Tensor):
    pv = RegList(DT.float, CHUNKS_D)
    for r in range(ROWS_PER_SB):
        pv <<= ub_pv[r:r + 1, 0:D_HEAD]
        ub_accum[r:r + 1, 0:D_HEAD] <<= pv
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def init_accum_vf(ub_accum: Tensor):
    zero = Reg(DT.float)
    zero <<= 0.0
    for r in range(ROWS_PER_SB):
        ub_accum[r:r + 1, 0:D_HEAD] <<= zero
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def final_div_cast_bf16_vf(ub_accum: Tensor, ub_rsum: Tensor, ub_out: Tensor):
    acc = RegList(DT.float, CHUNKS_D); rsum = Reg(DT.float)
    for r in range(ROWS_PER_SB):
        rsum <<= ub_rsum[0:1, r:r + 1].single()
        acc <<= ub_accum[r:r + 1, 0:D_HEAD]
        acc <<= acc / rsum
        ub_out[r:r + 1, 0:D_HEAD] <<= acc
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def fd_merge_vf(ub_o0: Tensor, ub_o1: Tensor, ub_m0: Tensor, ub_m1: Tensor,
                ub_s0: Tensor, ub_s1: Tensor, ub_A: Tensor, ub_den: Tensor, ub_merged: Tensor):
    m0v = Reg(DT.float); m1v = Reg(DT.float); s0v = Reg(DT.float); s1v = Reg(DT.float)
    Av = Reg(DT.float); denv = Reg(DT.float)
    m0v <<= ub_m0[0:1, 0:M_Q]; m1v <<= ub_m1[0:1, 0:M_Q]
    s0v <<= ub_s0[0:1, 0:M_Q]; s1v <<= ub_s1[0:1, 0:M_Q]
    expsub(Av, m0v, m1v)
    denv <<= s0v * Av
    denv <<= denv + s1v
    ub_A[0:1, 0:M_Q] <<= Av
    ub_den[0:1, 0:M_Q] <<= denv
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)
    o0 = RegList(DT.float, CD); o1 = RegList(DT.float, CD); Ar = Reg(DT.float); dr = Reg(DT.float)
    for r in range(M_Q):
        Ar <<= ub_A[0:1, r:r + 1].single()
        dr <<= ub_den[0:1, r:r + 1].single()
        o0 <<= ub_o0[r:r + 1, :]
        o1 <<= ub_o1[r:r + 1, :]
        o0 <<= o0 * Ar
        o0 <<= o0 + o1
        o0 <<= o0 / dr
        ub_merged[r:r + 1, :] <<= o0
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def output_cast_vf(ub_merged: Tensor, ub_out: Tensor):
    regs = RegList(DT.float, CD); hregs = RegList(DT.bfloat16, CD)
    cfg = CastConfig(round_mode=RoundMode.AWAY_FROM_ZERO, reg_layout=RegLayout.ZERO, name="o_bf16")
    for r in range(M_Q):
        regs <<= ub_merged[r:r + 1, :]
        for c in unroll(CD):
            cast(hregs[c], regs[c], cfg)
            reg_to_ub_downsample(ub_out[r:r + 1, c * 64:(c + 1) * 64], hregs[c])
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@func()
def _qk_bridge_splitn(ub_dst, l0c_src, l0c_scale):
    # ONE SPLITN fixpipe copy; HW auto-splits N across the 2 vec sub-blocks. Cannot requant, so
    # scale must be 1.0 (plain copy). A/B control to isolate the dual-SINGLE bridge's extra cost.
    l0c_to_ub(ub_dst, l0c_src[0:TILE_N, 0:TILE_M],
              M=TILE_N, N=TILE_M, N_dst=ROWS_PER_SB, M_src=TILE_N,
              dual_mode=DualMode.SPLITN, sub_block_id=0, scale=l0c_scale)


@func()
def _meta_fd_body(q, k, v, out, meta, B, MQ, N, D):
    q_hif8 = q.reinterpret(DT.hif8, name="q_hif8")
    k_hif8 = k.reinterpret(DT.hif8, name="k_hif8")
    v_hif8 = v.reinterpret(DT.hif8, name="v_hif8")

    cv = CvMutex(0, depth=2, src_start_pipe=Pipe.FIX, dst_start_pipe=Pipe.V,
                 src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    pv_mutex = CvMutex(2, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    p_mutex = VcMutex(1, depth=CACHE, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE1)

    l1q = DBuff(DT.hif8, [TILE_M, D_HEAD], Position.L1)
    l1k = TBuff(DT.hif8, [TILE_N, D_HEAD], Position.L1)
    l1v = TBuff(DT.hif8, [TILE_N, D_HEAD], Position.L1)
    l1p = TBuff(DT.hif8, [TILE_N, TILE_M], Position.L1)
    l0c_qk = DBuff(DT.float, [TILE_M, TILE_M], Position.L0C)
    l0c_pv = DBuff(DT.float, [TILE_M, D_HEAD], Position.L0C)

    ub_score = DBuff(DT.float, [TILE_N, ROWS_PER_SB], Position.UB)
    ub_pv = DBuff(DT.float, [ROWS_PER_SB, D_HEAD], Position.UB)
    ub_p = Tensor(DT.hif8, [TILE_N, KROW], Position.UB)
    ub_old_weight = TBuff(DT.float, [1, QLANES], Position.UB)
    ub_rmax = TBuff(DT.float, [1, QLANES], Position.UB)
    ub_rsum = TBuff(DT.float, [1, QLANES], Position.UB)
    ub_merge_a = Tensor(DT.float, [1, TILE_M], Position.UB)
    ub_merge_den = Tensor(DT.float, [1, TILE_M], Position.UB)
    ub_accum = Tensor(DT.float, [ROWS_PER_SB, D_HEAD], Position.UB)
    ub_out = Tensor(DT.bfloat16, [ROWS_PER_SB, D_HEAD], Position.UB)

    fd_accum = split_workspace(DT.float, [MAX_FD_WS, TILE_M, D_HEAD], name="fd_accum")
    fd_max = split_workspace(DT.float, [MAX_FD_WS, 1, TILE_M], name="fd_max")
    fd_sum = split_workspace(DT.float, [MAX_FD_WS, 1, TILE_M], name="fd_sum")

    core = Var(GetCubeIdx())
    core_count = Var(GetCubeNum())
    vec = Var(GetVecIdx())
    row_begin = Var(GetSubBlockIdx() * ROWS_PER_SB)

    tiles_m_per_b = CeilDiv(MQ, TILE_M)
    k_tiles = CeilDiv(N, TILE_N)
    total_blocks = Var(B * tiles_m_per_b)
    total_tasks = Var(total_blocks * k_tiles)
    flat_start = Var(total_tasks * core // core_count)
    flat_end = Var(total_tasks * (core + 1) // core_count)
    my_tasks = Var(Max(flat_end - flat_start, 0))

    tail_ws = Var(-1, DT.int)
    head_ws = Var(-1, DT.int)
    split_count = Var(0, DT.int)
    for b in range(1, MAX_CORE_COUNT):
        if core_count > b:
            boundary = Var(total_tasks * b // core_count)
            if boundary % k_tiles != 0:
                if core == b:
                    tail_ws <<= split_count * 2 + 1
                if core + 1 == b:
                    head_ws <<= split_count * 2
                split_count += 1


    with auto_sync():
        prefix_mt = Var(-1, DT.int)
        suffix_mt = Var(-1, DT.int)
        if my_tasks > 0:
            start_k = Var(flat_start % k_tiles)
            if start_k != 0:
                prefix_mt <<= flat_start // k_tiles
            end_k = Var(flat_end % k_tiles)
            if end_k != 0:
                suffix_mt <<= (flat_end - 1) // k_tiles

        for step in range(0, my_tasks + PRELOAD_N):
            if step < my_tasks:
                t = Var(flat_start + step)
                mbase = Var(t * META_F)
                # ---- metadata lookups replace: mt//kt via k_tiles, q_base/kv_base muls,
                #      m0/n0 muls, valid_m/valid_n mins ----
                mt = Var(0, DT.int); mt.GetValueFrom(meta[0:1, (mbase + 0):(mbase + 0) + 1])
                kt = Var(0, DT.int); kt.GetValueFrom(meta[0:1, (mbase + 1):(mbase + 1) + 1])
                q_base = Var(0, DT.int); q_base.GetValueFrom(meta[0:1, (mbase + 2):(mbase + 2) + 1])
                kv_base = Var(0, DT.int); kv_base.GetValueFrom(meta[0:1, (mbase + 3):(mbase + 3) + 1])
                m0 = Var(0, DT.int); m0.GetValueFrom(meta[0:1, (mbase + 4):(mbase + 4) + 1])
                n0 = Var(0, DT.int); n0.GetValueFrom(meta[0:1, (mbase + 5):(mbase + 5) + 1])
                valid_m = Var(0, DT.int); valid_m.GetValueFrom(meta[0:1, (mbase + 6):(mbase + 6) + 1])
                valid_n = Var(0, DT.int); valid_n.GetValueFrom(meta[0:1, (mbase + 7):(mbase + 7) + 1])
                qm = Var(mt % 2)      # cheap bit-mod, keep in-core
                sm = Var(mt % CACHE)  # cheap small-mod, keep in-core

                need_q = Var(0, DT.int)
                if kt == 0:
                    need_q <<= 1
                if step == 0:
                    need_q <<= 1
                if need_q > 0:
                    l1q[qm][0:valid_m, 0:D] <<= q_hif8[q_base + m0:q_base + m0 + valid_m, 0:D]
                    init_softmax_state_vf(ub_rmax[sm], ub_rsum[sm])

                l1k[step][0:valid_n, 0:D] <<= k_hif8[kv_base + n0:kv_base + n0 + valid_n, 0:D]
                matmul(l0c_qk[step], l1k[step], l1q[qm],
                       m=TILE_N, n=TILE_M, k=D_HEAD, is_init=True)

                cv.lock()
                _qk_bridge_splitn(ub_score[step], l0c_qk[step], 1.0)
                cv.ready()
                cv.wait()
                if valid_n < TILE_N:
                    softmax_t_scale_tail_vf(ub_score[step], ub_rmax[sm], ub_rsum[sm], ub_p,
                                            ub_old_weight[step], valid_n)
                else:
                    softmax_t_scale_vf(ub_score[step], ub_rmax[sm], ub_rsum[sm], ub_p,
                                       ub_old_weight[step])
                cv.free()

                p_mutex.lock()
                ub_to_l1_nd2nz(l1p[step][0:TILE_N, row_begin:row_begin + ROWS_PER_SB], ub_p,
                               m_dst=TILE_N, n_dst=ROWS_PER_SB, m_src=TILE_N, n_src=ROWS_PER_SB, N_src=KROW)
                p_mutex.ready()

            if step >= PRELOAD_N:
                lag = Var(step - PRELOAD_N)
                lag_t = Var(flat_start + lag)
                lbase = Var(lag_t * META_F)
                lag_mt = Var(0, DT.int); lag_mt.GetValueFrom(meta[0:1, (lbase + 0):(lbase + 0) + 1])
                lag_kt = Var(0, DT.int); lag_kt.GetValueFrom(meta[0:1, (lbase + 1):(lbase + 1) + 1])
                lag_q_base = Var(0, DT.int); lag_q_base.GetValueFrom(meta[0:1, (lbase + 2):(lbase + 2) + 1])
                lag_kv_base = Var(0, DT.int); lag_kv_base.GetValueFrom(meta[0:1, (lbase + 3):(lbase + 3) + 1])
                lag_m0 = Var(0, DT.int); lag_m0.GetValueFrom(meta[0:1, (lbase + 4):(lbase + 4) + 1])
                lag_n0 = Var(0, DT.int); lag_n0.GetValueFrom(meta[0:1, (lbase + 5):(lbase + 5) + 1])
                lag_valid_m = Var(0, DT.int); lag_valid_m.GetValueFrom(meta[0:1, (lbase + 6):(lbase + 6) + 1])
                v_rows = Var(0, DT.int); v_rows.GetValueFrom(meta[0:1, (lbase + 7):(lbase + 7) + 1])  # = valid_n(lag_t)
                sm2 = Var(lag_mt % CACHE)
                local_rows = Min(ROWS_PER_SB, Max(lag_valid_m - row_begin, 0))

                l1v[lag][0:v_rows, 0:D] <<= v_hif8[lag_kv_base + lag_n0:lag_kv_base + lag_n0 + v_rows, 0:D]
                p_mutex.wait()
                matmul(l0c_pv[lag][0:TILE_M, 0:D_HEAD], l1p[lag].T, l1v[lag].T,
                       m=TILE_M, n=D_HEAD, k=v_rows, is_init=True)
                p_mutex.free()

                pv_mutex.lock()
                l0c_to_ub(ub_pv[lag], l0c_pv[lag][0:TILE_M, 0:D_HEAD],
                          M=TILE_M, N=D_HEAD, N_dst=D_HEAD, M_src=TILE_M,
                          dual_mode=DualMode.SPLITM, sub_block_id=0)
                pv_mutex.ready()
                pv_mutex.wait()

                if lag_kt == 0:
                    init_accum_vf(ub_accum)
                    accum_pv_first_vf(ub_accum, ub_pv[lag])
                elif lag == 0:
                    init_accum_vf(ub_accum)
                    accum_pv_first_vf(ub_accum, ub_pv[lag])
                else:
                    accum_pv_vf(ub_accum, ub_pv[lag], ub_old_weight[lag])
                pv_mutex.free()

                if lag_kt == k_tiles - 1:
                    if lag_mt == prefix_mt:
                        if local_rows > 0:
                            fd_accum[tail_ws, row_begin:row_begin + local_rows, 0:D_HEAD] <<= ub_accum[0:local_rows, 0:D_HEAD]
                            fd_max[tail_ws, 0:1, row_begin:row_begin + local_rows] <<= ub_rmax[sm2][0:1, 0:local_rows]
                            fd_sum[tail_ws, 0:1, row_begin:row_begin + local_rows] <<= ub_rsum[sm2][0:1, 0:local_rows]
                    else:
                        final_div_cast_bf16_vf(ub_accum, ub_rsum[sm2], ub_out)
                        if local_rows > 0:
                            out[lag_q_base + lag_m0 + row_begin:lag_q_base + lag_m0 + row_begin + local_rows, 0:D] <<= (
                                ub_out[0:local_rows, 0:D]
                            )

                if lag + 1 == my_tasks:
                    if lag_kt != k_tiles - 1:
                        if lag_mt == suffix_mt:
                            if local_rows > 0:
                                fd_accum[head_ws, row_begin:row_begin + local_rows, 0:D_HEAD] <<= ub_accum[0:local_rows, 0:D_HEAD]
                                fd_max[head_ws, 0:1, row_begin:row_begin + local_rows] <<= ub_rmax[sm2][0:1, 0:local_rows]
                                fd_sum[head_ws, 0:1, row_begin:row_begin + local_rows] <<= ub_rsum[sm2][0:1, 0:local_rows]

        allvec_ready(7, Pipe.MTE3)

    allvec_wait(7, Pipe.MTE2)

    fd_task = Var(vec // 2)
    fd_row0 = Var((vec % 2) * ROWS_PER_SB)
    fd_found = Var(0, DT.int)
    fd_mt = Var(0, DT.int)
    scan_rank = Var(0, DT.int)
    for b in range(1, MAX_CORE_COUNT):
        if core_count > b:
            boundary = Var(total_tasks * b // core_count)
            if boundary % k_tiles != 0:
                if scan_rank == fd_task:
                    fd_found <<= 1
                    fd_mt <<= boundary // k_tiles
                scan_rank += 1

    if fd_found > 0:
        fd_batch = Var(fd_mt // tiles_m_per_b)
        fd_local_mt = Var(fd_mt % tiles_m_per_b)
        fd_q_base = Var(fd_batch * MQ)
        fd_m0 = Var(fd_local_mt * TILE_M)
        fd_valid_m = Min(TILE_M, MQ - fd_m0)
        fd_rows = Min(ROWS_PER_SB, Max(fd_valid_m - fd_row0, 0))
        fd_lo = Var(fd_task * 2)
        fd_hi = Var(fd_lo + 1)
        if fd_rows > 0:
            with auto_sync():
                ub_rmax[0][0:1, 0:ROWS_PER_SB] <<= fd_max[fd_lo, 0:1, fd_row0:fd_row0 + ROWS_PER_SB]
                ub_rmax[1][0:1, 0:ROWS_PER_SB] <<= fd_max[fd_hi, 0:1, fd_row0:fd_row0 + ROWS_PER_SB]
                ub_rsum[0][0:1, 0:ROWS_PER_SB] <<= fd_sum[fd_lo, 0:1, fd_row0:fd_row0 + ROWS_PER_SB]
                ub_rsum[1][0:1, 0:ROWS_PER_SB] <<= fd_sum[fd_hi, 0:1, fd_row0:fd_row0 + ROWS_PER_SB]
                ub_pv[0][:, :] <<= fd_accum[fd_lo, fd_row0:fd_row0 + ROWS_PER_SB, 0:D_HEAD]
                ub_pv[1][:, :] <<= fd_accum[fd_hi, fd_row0:fd_row0 + ROWS_PER_SB, 0:D_HEAD]
                merge_fn = fd_merge_vf
                merge_fn(ub_pv[0], ub_pv[1], ub_rmax[0], ub_rmax[1],
                         ub_rsum[0], ub_rsum[1], ub_merge_a, ub_merge_den, ub_pv[0])
                output_cast_vf(ub_pv[0], ub_out)
                out[fd_q_base + fd_m0 + fd_row0:fd_q_base + fd_m0 + fd_row0 + fd_rows, 0:D] <<= (
                    ub_out[0:fd_rows, 0:D]
                )
    return out


@kernel()
def qk_softmax_pv_v6_meta_kernel(q: GM[u8, ('TQ', 'D')], k: GM[u8, ('TK', 'D')], v: GM[u8, ('TK', 'D')], meta: GM[i32, (1, 'MF')],
                                 out: GM[bf16, ('TQ', 'D')], B: Var, MQ: Var, N: Var, D: Var):
    return _meta_fd_body(q, k, v, out, meta, B, MQ, N, D)


@func()
def _meta_fd_body_ext(q, k, v, out, meta, B, MQ, N, D):
    # meta-EXT: also precompute the cheap per-row scalars into GM so the loop READS them instead of
    # computing: sm=mt%CACHE, qm=mt%2, first_k=(kt==0), last_k=(kt==k_tiles-1). kt itself is dropped
    # (flags replace every kt compare). Trades +3 GM scalar-loads/step-pair for -7 ALU ops (3 mod +
    # 4 cmp). Everything else is identical to _meta_fd_body. layout META_F_EXT=11:
    # Metadata fields in order: mt, q_base, kv_base, m0, n0, valid_m, valid_n, sm, qm, first_k, last_k.
    q_hif8 = q.reinterpret(DT.hif8, name="q_hif8")
    k_hif8 = k.reinterpret(DT.hif8, name="k_hif8")
    v_hif8 = v.reinterpret(DT.hif8, name="v_hif8")

    cv = CvMutex(0, depth=2, src_start_pipe=Pipe.FIX, dst_start_pipe=Pipe.V,
                 src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    pv_mutex = CvMutex(2, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    p_mutex = VcMutex(1, depth=CACHE, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE1)

    l1q = DBuff(DT.hif8, [TILE_M, D_HEAD], Position.L1)
    l1k = TBuff(DT.hif8, [TILE_N, D_HEAD], Position.L1)
    l1v = TBuff(DT.hif8, [TILE_N, D_HEAD], Position.L1)
    l1p = TBuff(DT.hif8, [TILE_N, TILE_M], Position.L1)
    l0c_qk = DBuff(DT.float, [TILE_M, TILE_M], Position.L0C)
    l0c_pv = DBuff(DT.float, [TILE_M, D_HEAD], Position.L0C)

    ub_score = DBuff(DT.float, [TILE_N, ROWS_PER_SB], Position.UB)
    ub_pv = DBuff(DT.float, [ROWS_PER_SB, D_HEAD], Position.UB)
    ub_p = Tensor(DT.hif8, [TILE_N, KROW], Position.UB)
    ub_old_weight = TBuff(DT.float, [1, QLANES], Position.UB)
    ub_rmax = TBuff(DT.float, [1, QLANES], Position.UB)
    ub_rsum = TBuff(DT.float, [1, QLANES], Position.UB)
    ub_merge_a = Tensor(DT.float, [1, TILE_M], Position.UB)
    ub_merge_den = Tensor(DT.float, [1, TILE_M], Position.UB)
    ub_accum = Tensor(DT.float, [ROWS_PER_SB, D_HEAD], Position.UB)
    ub_out = Tensor(DT.bfloat16, [ROWS_PER_SB, D_HEAD], Position.UB)

    fd_accum = split_workspace(DT.float, [MAX_FD_WS, TILE_M, D_HEAD], name="fd_accum")
    fd_max = split_workspace(DT.float, [MAX_FD_WS, 1, TILE_M], name="fd_max")
    fd_sum = split_workspace(DT.float, [MAX_FD_WS, 1, TILE_M], name="fd_sum")

    core = Var(GetCubeIdx())
    core_count = Var(GetCubeNum())
    vec = Var(GetVecIdx())
    row_begin = Var(GetSubBlockIdx() * ROWS_PER_SB)

    tiles_m_per_b = CeilDiv(MQ, TILE_M)
    k_tiles = CeilDiv(N, TILE_N)
    total_blocks = Var(B * tiles_m_per_b)
    total_tasks = Var(total_blocks * k_tiles)
    flat_start = Var(total_tasks * core // core_count)
    flat_end = Var(total_tasks * (core + 1) // core_count)
    my_tasks = Var(Max(flat_end - flat_start, 0))

    tail_ws = Var(-1, DT.int)
    head_ws = Var(-1, DT.int)
    split_count = Var(0, DT.int)
    for b in range(1, MAX_CORE_COUNT):
        if core_count > b:
            boundary = Var(total_tasks * b // core_count)
            if boundary % k_tiles != 0:
                if core == b:
                    tail_ws <<= split_count * 2 + 1
                if core + 1 == b:
                    head_ws <<= split_count * 2
                split_count += 1


    with auto_sync():
        prefix_mt = Var(-1, DT.int)
        suffix_mt = Var(-1, DT.int)
        if my_tasks > 0:
            start_k = Var(flat_start % k_tiles)
            if start_k != 0:
                prefix_mt <<= flat_start // k_tiles
            end_k = Var(flat_end % k_tiles)
            if end_k != 0:
                suffix_mt <<= (flat_end - 1) // k_tiles

        for step in range(0, my_tasks + PRELOAD_N):
            if step < my_tasks:
                t = Var(flat_start + step)
                mbase = Var(t * META_F_EXT)
                # step reads only what it uses: skip mt (step-unused) + kt/last_k; qm/sm/first_k are loads now
                q_base = Var(0, DT.int); q_base.GetValueFrom(meta[0:1, (mbase + 1):(mbase + 1) + 1])
                kv_base = Var(0, DT.int); kv_base.GetValueFrom(meta[0:1, (mbase + 2):(mbase + 2) + 1])
                m0 = Var(0, DT.int); m0.GetValueFrom(meta[0:1, (mbase + 3):(mbase + 3) + 1])
                n0 = Var(0, DT.int); n0.GetValueFrom(meta[0:1, (mbase + 4):(mbase + 4) + 1])
                valid_m = Var(0, DT.int); valid_m.GetValueFrom(meta[0:1, (mbase + 5):(mbase + 5) + 1])
                valid_n = Var(0, DT.int); valid_n.GetValueFrom(meta[0:1, (mbase + 6):(mbase + 6) + 1])
                sm = Var(0, DT.int); sm.GetValueFrom(meta[0:1, (mbase + 7):(mbase + 7) + 1])
                qm = Var(0, DT.int); qm.GetValueFrom(meta[0:1, (mbase + 8):(mbase + 8) + 1])
                first_k = Var(0, DT.int); first_k.GetValueFrom(meta[0:1, (mbase + 9):(mbase + 9) + 1])

                need_q = Var(0, DT.int)
                if first_k > 0:
                    need_q <<= 1
                if step == 0:
                    need_q <<= 1
                if need_q > 0:
                    l1q[qm][0:valid_m, 0:D] <<= q_hif8[q_base + m0:q_base + m0 + valid_m, 0:D]
                    init_softmax_state_vf(ub_rmax[sm], ub_rsum[sm])

                l1k[step][0:valid_n, 0:D] <<= k_hif8[kv_base + n0:kv_base + n0 + valid_n, 0:D]
                matmul(l0c_qk[step], l1k[step], l1q[qm],
                       m=TILE_N, n=TILE_M, k=D_HEAD, is_init=True)

                cv.lock()
                _qk_bridge_splitn(ub_score[step], l0c_qk[step], 1.0)
                cv.ready()
                cv.wait()
                if valid_n < TILE_N:
                    softmax_t_scale_tail_vf(ub_score[step], ub_rmax[sm], ub_rsum[sm], ub_p,
                                            ub_old_weight[step], valid_n)
                else:
                    softmax_t_scale_vf(ub_score[step], ub_rmax[sm], ub_rsum[sm], ub_p,
                                       ub_old_weight[step])
                cv.free()

                p_mutex.lock()
                ub_to_l1_nd2nz(l1p[step][0:TILE_N, row_begin:row_begin + ROWS_PER_SB], ub_p,
                               m_dst=TILE_N, n_dst=ROWS_PER_SB, m_src=TILE_N, n_src=ROWS_PER_SB, N_src=KROW)
                p_mutex.ready()

            if step >= PRELOAD_N:
                lag = Var(step - PRELOAD_N)
                lag_t = Var(flat_start + lag)
                lbase = Var(lag_t * META_F_EXT)
                # lag reads: skip qm (step-only); sm2 = sm(lag row); first_k/last_k replace kt compares
                lag_mt = Var(0, DT.int); lag_mt.GetValueFrom(meta[0:1, (lbase + 0):(lbase + 0) + 1])
                lag_q_base = Var(0, DT.int); lag_q_base.GetValueFrom(meta[0:1, (lbase + 1):(lbase + 1) + 1])
                lag_kv_base = Var(0, DT.int); lag_kv_base.GetValueFrom(meta[0:1, (lbase + 2):(lbase + 2) + 1])
                lag_m0 = Var(0, DT.int); lag_m0.GetValueFrom(meta[0:1, (lbase + 3):(lbase + 3) + 1])
                lag_n0 = Var(0, DT.int); lag_n0.GetValueFrom(meta[0:1, (lbase + 4):(lbase + 4) + 1])
                lag_valid_m = Var(0, DT.int); lag_valid_m.GetValueFrom(meta[0:1, (lbase + 5):(lbase + 5) + 1])
                v_rows = Var(0, DT.int); v_rows.GetValueFrom(meta[0:1, (lbase + 6):(lbase + 6) + 1])  # = valid_n(lag_t)
                sm2 = Var(0, DT.int); sm2.GetValueFrom(meta[0:1, (lbase + 7):(lbase + 7) + 1])
                lag_first_k = Var(0, DT.int); lag_first_k.GetValueFrom(meta[0:1, (lbase + 9):(lbase + 9) + 1])
                lag_last_k = Var(0, DT.int); lag_last_k.GetValueFrom(meta[0:1, (lbase + 10):(lbase + 10) + 1])
                local_rows = Min(ROWS_PER_SB, Max(lag_valid_m - row_begin, 0))

                l1v[lag][0:v_rows, 0:D] <<= v_hif8[lag_kv_base + lag_n0:lag_kv_base + lag_n0 + v_rows, 0:D]
                p_mutex.wait()
                matmul(l0c_pv[lag][0:TILE_M, 0:D_HEAD], l1p[lag].T, l1v[lag].T,
                       m=TILE_M, n=D_HEAD, k=v_rows, is_init=True)
                p_mutex.free()

                pv_mutex.lock()
                l0c_to_ub(ub_pv[lag], l0c_pv[lag][0:TILE_M, 0:D_HEAD],
                          M=TILE_M, N=D_HEAD, N_dst=D_HEAD, M_src=TILE_M,
                          dual_mode=DualMode.SPLITM, sub_block_id=0)
                pv_mutex.ready()
                pv_mutex.wait()

                if lag_first_k > 0:
                    init_accum_vf(ub_accum)
                    accum_pv_first_vf(ub_accum, ub_pv[lag])
                elif lag == 0:
                    init_accum_vf(ub_accum)
                    accum_pv_first_vf(ub_accum, ub_pv[lag])
                else:
                    accum_pv_vf(ub_accum, ub_pv[lag], ub_old_weight[lag])
                pv_mutex.free()

                if lag_last_k > 0:
                    if lag_mt == prefix_mt:
                        if local_rows > 0:
                            fd_accum[tail_ws, row_begin:row_begin + local_rows, 0:D_HEAD] <<= ub_accum[0:local_rows, 0:D_HEAD]
                            fd_max[tail_ws, 0:1, row_begin:row_begin + local_rows] <<= ub_rmax[sm2][0:1, 0:local_rows]
                            fd_sum[tail_ws, 0:1, row_begin:row_begin + local_rows] <<= ub_rsum[sm2][0:1, 0:local_rows]
                    else:
                        final_div_cast_bf16_vf(ub_accum, ub_rsum[sm2], ub_out)
                        if local_rows > 0:
                            out[lag_q_base + lag_m0 + row_begin:lag_q_base + lag_m0 + row_begin + local_rows, 0:D] <<= (
                                ub_out[0:local_rows, 0:D]
                            )

                if lag + 1 == my_tasks:
                    if lag_last_k == 0:
                        if lag_mt == suffix_mt:
                            if local_rows > 0:
                                fd_accum[head_ws, row_begin:row_begin + local_rows, 0:D_HEAD] <<= ub_accum[0:local_rows, 0:D_HEAD]
                                fd_max[head_ws, 0:1, row_begin:row_begin + local_rows] <<= ub_rmax[sm2][0:1, 0:local_rows]
                                fd_sum[head_ws, 0:1, row_begin:row_begin + local_rows] <<= ub_rsum[sm2][0:1, 0:local_rows]

        allvec_ready(7, Pipe.MTE3)

    allvec_wait(7, Pipe.MTE2)

    fd_task = Var(vec // 2)
    fd_row0 = Var((vec % 2) * ROWS_PER_SB)
    fd_found = Var(0, DT.int)
    fd_mt = Var(0, DT.int)
    scan_rank = Var(0, DT.int)
    for b in range(1, MAX_CORE_COUNT):
        if core_count > b:
            boundary = Var(total_tasks * b // core_count)
            if boundary % k_tiles != 0:
                if scan_rank == fd_task:
                    fd_found <<= 1
                    fd_mt <<= boundary // k_tiles
                scan_rank += 1

    if fd_found > 0:
        fd_batch = Var(fd_mt // tiles_m_per_b)
        fd_local_mt = Var(fd_mt % tiles_m_per_b)
        fd_q_base = Var(fd_batch * MQ)
        fd_m0 = Var(fd_local_mt * TILE_M)
        fd_valid_m = Min(TILE_M, MQ - fd_m0)
        fd_rows = Min(ROWS_PER_SB, Max(fd_valid_m - fd_row0, 0))
        fd_lo = Var(fd_task * 2)
        fd_hi = Var(fd_lo + 1)
        if fd_rows > 0:
            with auto_sync():
                ub_rmax[0][0:1, 0:ROWS_PER_SB] <<= fd_max[fd_lo, 0:1, fd_row0:fd_row0 + ROWS_PER_SB]
                ub_rmax[1][0:1, 0:ROWS_PER_SB] <<= fd_max[fd_hi, 0:1, fd_row0:fd_row0 + ROWS_PER_SB]
                ub_rsum[0][0:1, 0:ROWS_PER_SB] <<= fd_sum[fd_lo, 0:1, fd_row0:fd_row0 + ROWS_PER_SB]
                ub_rsum[1][0:1, 0:ROWS_PER_SB] <<= fd_sum[fd_hi, 0:1, fd_row0:fd_row0 + ROWS_PER_SB]
                ub_pv[0][:, :] <<= fd_accum[fd_lo, fd_row0:fd_row0 + ROWS_PER_SB, 0:D_HEAD]
                ub_pv[1][:, :] <<= fd_accum[fd_hi, fd_row0:fd_row0 + ROWS_PER_SB, 0:D_HEAD]
                merge_fn = fd_merge_vf
                merge_fn(ub_pv[0], ub_pv[1], ub_rmax[0], ub_rmax[1],
                         ub_rsum[0], ub_rsum[1], ub_merge_a, ub_merge_den, ub_pv[0])
                output_cast_vf(ub_pv[0], ub_out)
                out[fd_q_base + fd_m0 + fd_row0:fd_q_base + fd_m0 + fd_row0 + fd_rows, 0:D] <<= (
                    ub_out[0:fd_rows, 0:D]
                )
    return out


@kernel()
def qk_softmax_pv_v6_meta_ext_kernel(q: GM[u8, ('TQ', 'D')], k: GM[u8, ('TK', 'D')], v: GM[u8, ('TK', 'D')], meta: GM[i32, (1, 'MF')],
                                     out: GM[bf16, ('TQ', 'D')], B: Var, MQ: Var, N: Var, D: Var):
    return _meta_fd_body_ext(q, k, v, out, meta, B, MQ, N, D)
