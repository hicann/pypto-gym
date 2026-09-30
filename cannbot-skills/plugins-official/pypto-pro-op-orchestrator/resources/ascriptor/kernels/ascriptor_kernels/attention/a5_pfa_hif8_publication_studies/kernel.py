# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Seven complete unscaled-HiFloat8 attention studies over one shared body.

Four of them differ only in how a HiFloat8 P tile reaches L1 in NZ: nz1-nz4 are the softmax
`@vf` pair handed to `_full_nz_body` (or `_full_nz4_body` for the pitch-33 deinterleave).
The other three differ only in the synchronisation protocol over one identical ND schedule:
`cprobe_base` uses end-pipe-only C/V mutexes, `cprobe_mutexse` passes the same mutexes with
their start pipes declared equal to their end pipes, and `cprobe_splitevent` keeps the plain
mutexes and adds two `SEvent(Pipe.M, Pipe.FIX)` inside the cube side. Every body takes its
mutexes as arguments, so a protocol variant is a different call and not a forked body."""

from ascriptor.a5 import *

# ----------------------------------------------------------------------------------------------------
# common.py -- the pieces every variant shares: softmax/accumulate/merge VFs and the fixpipe bridge.
# ----------------------------------------------------------------------------------------------------

TILE_M = TILE_N = D_HEAD = 128
ROWS_PER_SB = QLANES = HALF = KROW = M_Q = 64
CHUNKS_D = CD = 2
SOFTMAX_SCALE = 128 ** -0.5
PRELOAD_N = 2
CACHE = 3
_CacheBuf = TBuff
MAX_CORE_COUNT = 32
MAX_FD_WS = 64
UNROLL = 4
RB_MAX = 32
RB_EXP = 16
NEG_LARGE = -1.0e30

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

# ----------------------------------------------------------------------------------------------------
# nd.py -- the ND publication schedule and its three synchronisation protocols.
# ----------------------------------------------------------------------------------------------------

def _alloc_v6(q, k, v):
    q_hif8 = q.reinterpret(DT.hif8, name="q_hif8")
    k_hif8 = k.reinterpret(DT.hif8, name="k_hif8")
    v_hif8 = v.reinterpret(DT.hif8, name="v_hif8")
    l1q = DBuff(DT.hif8, [TILE_M, D_HEAD], Position.L1)
    l1k = _CacheBuf(DT.hif8, [TILE_N, D_HEAD], Position.L1)
    l1v = _CacheBuf(DT.hif8, [TILE_N, D_HEAD], Position.L1)
    l1p = _CacheBuf(DT.hif8, [TILE_N, TILE_M], Position.L1)
    l0c_qk = DBuff(DT.float, [TILE_M, TILE_M], Position.L0C)
    l0c_pv = DBuff(DT.float, [TILE_M, D_HEAD], Position.L0C)
    ub_score = DBuff(DT.float, [TILE_N, ROWS_PER_SB], Position.UB)
    ub_pv = DBuff(DT.float, [ROWS_PER_SB, D_HEAD], Position.UB)
    ub_p = Tensor(DT.hif8, [TILE_N, KROW], Position.UB)
    ub_old_weight = _CacheBuf(DT.float, [1, QLANES], Position.UB)
    ub_rmax = _CacheBuf(DT.float, [1, QLANES], Position.UB)
    ub_rsum = _CacheBuf(DT.float, [1, QLANES], Position.UB)
    ub_merge_a = Tensor(DT.float, [1, TILE_M], Position.UB)
    ub_merge_den = Tensor(DT.float, [1, TILE_M], Position.UB)
    ub_accum = Tensor(DT.float, [ROWS_PER_SB, D_HEAD], Position.UB)
    ub_out = Tensor(DT.bfloat16, [ROWS_PER_SB, D_HEAD], Position.UB)
    fd_accum = split_workspace(DT.float, [MAX_FD_WS, TILE_M, D_HEAD], name="fd_accum")
    fd_max = split_workspace(DT.float, [MAX_FD_WS, 1, TILE_M], name="fd_max")
    fd_sum = split_workspace(DT.float, [MAX_FD_WS, 1, TILE_M], name="fd_sum")
    return (q_hif8, k_hif8, v_hif8, l1q, l1k, l1v, l1p, l0c_qk, l0c_pv, ub_score, ub_pv, ub_p,
            ub_old_weight, ub_rmax, ub_rsum, ub_merge_a, ub_merge_den, ub_accum, ub_out, fd_accum, fd_max, fd_sum)

@func()
def _v6_full_body(q, k, v, out, B, MQ, N, D, cv, pv_mutex, p_mutex):
    (q_hif8, k_hif8, v_hif8, l1q, l1k, l1v, l1p, l0c_qk, l0c_pv, ub_score, ub_pv, ub_p, ub_old_weight,
     ub_rmax, ub_rsum, ub_merge_a, ub_merge_den, ub_accum, ub_out, fd_accum, fd_max, fd_sum) = _alloc_v6(q, k, v)

    core = Var(GetCubeIdx())
    core_count = Var(GetCubeNum())
    vec = Var(GetVecIdx())
    row_begin = Var(GetSubBlockIdx() * ROWS_PER_SB)
    tiles_m_per_b = CeilDiv(MQ, TILE_M)
    k_tiles = CeilDiv(N, TILE_N)
    total_tasks = Var(B * tiles_m_per_b * k_tiles)
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

    # D-214: PV contracts only loaded V rows; no overlapping whole-L1 fill.

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
                mt = Var(t // k_tiles)
                kt = Var(t % k_tiles)
                qm = Var(mt % 2)
                sm = Var(mt % CACHE)
                batch = Var(mt // tiles_m_per_b)
                local_mt = Var(mt % tiles_m_per_b)
                q_base = Var(batch * MQ)
                kv_base = Var(batch * N)
                m0 = Var(local_mt * TILE_M)
                valid_m = Min(TILE_M, MQ - m0)
                n0 = Var(kt * TILE_N)
                valid_n = Min(TILE_N, N - n0)

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
                lag_mt = Var(lag_t // k_tiles)
                lag_kt = Var(lag_t % k_tiles)
                sm2 = Var(lag_mt % CACHE)
                lag_batch = Var(lag_mt // tiles_m_per_b)
                lag_local_mt = Var(lag_mt % tiles_m_per_b)
                lag_q_base = Var(lag_batch * MQ)
                lag_kv_base = Var(lag_batch * N)
                lag_m0 = Var(lag_local_mt * TILE_M)
                lag_valid_m = Min(TILE_M, MQ - lag_m0)
                local_rows = Min(ROWS_PER_SB, Max(lag_valid_m - row_begin, 0))
                lag_n0 = Var(lag_kt * TILE_N)
                v_rows = Min(TILE_N, N - lag_n0)

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
                fd_merge_vf(ub_pv[0], ub_pv[1], ub_rmax[0], ub_rmax[1],
                            ub_rsum[0], ub_rsum[1], ub_merge_a, ub_merge_den, ub_pv[0])
                output_cast_vf(ub_pv[0], ub_out)
                out[fd_q_base + fd_m0 + fd_row0:fd_q_base + fd_m0 + fd_row0 + fd_rows, 0:D] <<= (
                    ub_out[0:fd_rows, 0:D]
                )
    return out

@func()
def _v6_splitevent_body(q, k, v, out, B, MQ, N, D, cv, pv_mutex, p_mutex):
    (q_hif8, k_hif8, v_hif8, l1q, l1k, l1v, l1p, l0c_qk, l0c_pv, ub_score, ub_pv, ub_p, ub_old_weight,
     ub_rmax, ub_rsum, ub_merge_a, ub_merge_den, ub_accum, ub_out, fd_accum, fd_max, fd_sum) = _alloc_v6(q, k, v)

    ev_qk = SEvent(Pipe.M, Pipe.FIX, name="ev_qk")
    ev_pv = SEvent(Pipe.M, Pipe.FIX, name="ev_pv")

    core = Var(GetCubeIdx())
    core_count = Var(GetCubeNum())
    vec = Var(GetVecIdx())
    row_begin = Var(GetSubBlockIdx() * ROWS_PER_SB)
    tiles_m_per_b = CeilDiv(MQ, TILE_M)
    k_tiles = CeilDiv(N, TILE_N)
    total_tasks = Var(B * tiles_m_per_b * k_tiles)
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

    # D-214: PV contracts only loaded V rows; no overlapping whole-L1 fill.

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
                mt = Var(t // k_tiles)
                kt = Var(t % k_tiles)
                qm = Var(mt % 2)
                sm = Var(mt % CACHE)
                batch = Var(mt // tiles_m_per_b)
                local_mt = Var(mt % tiles_m_per_b)
                q_base = Var(batch * MQ)
                kv_base = Var(batch * N)
                m0 = Var(local_mt * TILE_M)
                valid_m = Min(TILE_M, MQ - m0)
                n0 = Var(kt * TILE_N)
                valid_n = Min(TILE_N, N - n0)

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
                ev_qk.set()
                ev_qk.wait()

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
                lag_mt = Var(lag_t // k_tiles)
                lag_kt = Var(lag_t % k_tiles)
                sm2 = Var(lag_mt % CACHE)
                lag_batch = Var(lag_mt // tiles_m_per_b)
                lag_local_mt = Var(lag_mt % tiles_m_per_b)
                lag_q_base = Var(lag_batch * MQ)
                lag_kv_base = Var(lag_batch * N)
                lag_m0 = Var(lag_local_mt * TILE_M)
                lag_valid_m = Min(TILE_M, MQ - lag_m0)
                local_rows = Min(ROWS_PER_SB, Max(lag_valid_m - row_begin, 0))
                lag_n0 = Var(lag_kt * TILE_N)
                v_rows = Min(TILE_N, N - lag_n0)

                l1v[lag][0:v_rows, 0:D] <<= v_hif8[lag_kv_base + lag_n0:lag_kv_base + lag_n0 + v_rows, 0:D]
                p_mutex.wait()
                matmul(l0c_pv[lag][0:TILE_M, 0:D_HEAD], l1p[lag].T, l1v[lag].T,
                       m=TILE_M, n=D_HEAD, k=v_rows, is_init=True)
                p_mutex.free()
                ev_pv.set()
                ev_pv.wait()

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
                fd_merge_vf(ub_pv[0], ub_pv[1], ub_rmax[0], ub_rmax[1],
                            ub_rsum[0], ub_rsum[1], ub_merge_a, ub_merge_den, ub_pv[0])
                output_cast_vf(ub_pv[0], ub_out)
                out[fd_q_base + fd_m0 + fd_row0:fd_q_base + fd_m0 + fd_row0 + fd_rows, 0:D] <<= (
                    ub_out[0:fd_rows, 0:D]
                )
    return out

def _cv_normal():
    return CvMutex(0, depth=2, src_start_pipe=Pipe.FIX, dst_start_pipe=Pipe.V,
                   src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)

def _pv_normal():
    return CvMutex(2, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)

def _p_normal():
    return VcMutex(1, depth=CACHE, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE1)

def _pv_se():
    return CvMutex(2, depth=2, src_start_pipe=Pipe.FIX, dst_start_pipe=Pipe.V,
                   src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)

def _p_se():
    return VcMutex(1, depth=CACHE, src_start_pipe=Pipe.MTE3, dst_start_pipe=Pipe.MTE1,
                   src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE1)

@kernel()
def cprobe_base_kernel(q: GM[u8, ('TQ', 'D')], k: GM[u8, ('TK', 'D')], v: GM[u8, ('TK', 'D')], out: GM[bf16, ('TQ', 'D')], B: i32, MQ: i32, N: i32, D: i32):
    return _v6_full_body(q, k, v, out, B, MQ, N, D, _cv_normal(), _pv_normal(), _p_normal())

@kernel()
def cprobe_mutexse_kernel(q: GM[u8, ('TQ', 'D')], k: GM[u8, ('TK', 'D')], v: GM[u8, ('TK', 'D')], out: GM[bf16, ('TQ', 'D')], B: i32, MQ: i32, N: i32, D: i32):
    return _v6_full_body(q, k, v, out, B, MQ, N, D, _cv_normal(), _pv_se(), _p_se())

@kernel()
def cprobe_splitevent_kernel(q: GM[u8, ('TQ', 'D')], k: GM[u8, ('TK', 'D')], v: GM[u8, ('TK', 'D')], out: GM[bf16, ('TQ', 'D')], B: i32, MQ: i32, N: i32, D: i32):
    return _v6_splitevent_body(q, k, v, out, B, MQ, N, D, _cv_normal(), _pv_normal(), _p_normal())

# ----------------------------------------------------------------------------------------------------
# nz.py -- four ways to publish a HiFloat8 P tile straight into L1 in NZ.
# ----------------------------------------------------------------------------------------------------

NZ_C0 = 32
NZ_M = 129
SLAB = 32
FRAC_STRIDE4 = 33
KEYS_PER_GRP = 4
RB4 = 16
UNROLL4 = 2
U_TAIL = 4
_DEINT_U8 = False

def _deint(d0, d1, s0, s1):
    if _DEINT_U8:
        deinterleave(d0.reinterpret(DT.uint8), d1.reinterpret(DT.uint8),
                     s0.reinterpret(DT.uint8), s1.reinterpret(DT.uint8))
    else:
        deinterleave(d0, d1, s0, s1)

@vf()
def softmax_nz1_vf(ub_score: Tensor, ub_rmax: Tensor, ub_rsum: Tensor, ub_p: Tensor, ub_old_weight: Tensor):
    sreg = RegList(DT.float, UNROLL); acc = RegList(DT.float, UNROLL); preg = RegList(DT.float, UNROLL)
    p_hi = RegList(DT.hif8, UNROLL); p_hj = RegList(DT.hif8, UNROLL)
    # NZ pack: squeeze the stride-4 sparse hif8 (RegLayout.ZERO) to dense lanes 0..63 (M4), then a strided
    # reg_to_ub (blk_stride=129, mask=Q) lands query j at ub_p byte (j//32)*129*32 + (j%32) [+ base r*32].
    m4 = MaskReg(DT.hif8, init_mode=MaskType.MULTI4)
    qmask = MaskReg(DT.hif8, init_mode=MaskType.LOWQUAT)
    dpi = Reg(DT.hif8); dpj = Reg(DT.hif8)
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
            squeeze(dpi, p_hi[u], mask=m4)
            reg_to_ub(ub_p[(rb * UNROLL + u) * NZ_C0], dpi, NZ_M, mask=qmask)
            squeeze(dpj, p_hj[u], mask=m4)
            reg_to_ub(ub_p[((rb * UNROLL + u) + HALF) * NZ_C0], dpj, NZ_M, mask=qmask)
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
def softmax_nz1_tail_vf(ub_score: Tensor, ub_rmax: Tensor, ub_rsum: Tensor, ub_p: Tensor, ub_old_weight: Tensor, valid_n: Var):
    s = Reg(DT.float); e = Reg(DT.float); neg = Reg(DT.float)
    block_max = Reg(DT.float); block_sum = Reg(DT.float); p_hif8 = Reg(DT.hif8)
    m4 = MaskReg(DT.hif8, init_mode=MaskType.MULTI4)
    qmask = MaskReg(DT.hif8, init_mode=MaskType.LOWQUAT)
    dp = Reg(DT.hif8)
    rowmask = MaskReg(DT.float, init_mode=MaskType.ALL)
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
        squeeze(dp, p_hif8, mask=m4)
        reg_to_ub(ub_p[n * NZ_C0], dp, NZ_M, mask=qmask)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)
    v_psum <<= ub_rsum[0:1, 0:QLANES]
    v_nsum <<= v_psum * v_oldw
    v_nsum <<= v_nsum + block_sum
    ub_rmax[0:1, 0:QLANES] <<= v_nmax
    ub_rsum[0:1, 0:QLANES] <<= v_nsum
    ub_old_weight[0:1, 0:QLANES] <<= v_oldw
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

@vf()
def softmax_nz2_vf(ub_score: Tensor, ub_rmax: Tensor, ub_rsum: Tensor, ub_p: Tensor, ub_old_weight: Tensor):
    sreg = RegList(DT.float, UNROLL); acc = RegList(DT.float, UNROLL); preg = RegList(DT.float, UNROLL)
    p_hi = RegList(DT.hif8, UNROLL); p_hj = RegList(DT.hif8, UNROLL)
    # NZ pack: reg->reg gather with uint8 index [0,4,8,..,252] compacts the stride-4 sparse hif8 to dense
    # lanes 0..63 (dpi[k]=src[4k]); same strided reg_to_ub(blk_stride=129, mask=Q) as nz1 places each query.
    qmask = MaskReg(DT.hif8, init_mode=MaskType.LOWQUAT)
    dpi = Reg(DT.hif8); dpj = Reg(DT.hif8)
    g8 = Reg(DT.int8); arange(g8, 0); shiftls(g8, g8, 2)   # [0,4,8,..] mod 256; lanes 0..63 = [0,4,..,252]
    gidx = g8.reinterpret(DT.uint8)
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
            gather(dpi, p_hi[u], gidx)
            reg_to_ub(ub_p[(rb * UNROLL + u) * NZ_C0], dpi, NZ_M, mask=qmask)
            gather(dpj, p_hj[u], gidx)
            reg_to_ub(ub_p[((rb * UNROLL + u) + HALF) * NZ_C0], dpj, NZ_M, mask=qmask)
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
def softmax_nz2_tail_vf(ub_score: Tensor, ub_rmax: Tensor, ub_rsum: Tensor, ub_p: Tensor, ub_old_weight: Tensor, valid_n: Var):
    s = Reg(DT.float); e = Reg(DT.float); neg = Reg(DT.float)
    block_max = Reg(DT.float); block_sum = Reg(DT.float); p_hif8 = Reg(DT.hif8)
    qmask = MaskReg(DT.hif8, init_mode=MaskType.LOWQUAT)
    dp = Reg(DT.hif8)
    g8 = Reg(DT.int8); arange(g8, 0); shiftls(g8, g8, 2)
    gidx = g8.reinterpret(DT.uint8)
    rowmask = MaskReg(DT.float, init_mode=MaskType.ALL)
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
        gather(dp, p_hif8, gidx)
        reg_to_ub(ub_p[n * NZ_C0], dp, NZ_M, mask=qmask)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)
    v_psum <<= ub_rsum[0:1, 0:QLANES]
    v_nsum <<= v_psum * v_oldw
    v_nsum <<= v_nsum + block_sum
    ub_rmax[0:1, 0:QLANES] <<= v_nmax
    ub_rsum[0:1, 0:QLANES] <<= v_nsum
    ub_old_weight[0:1, 0:QLANES] <<= v_oldw
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

@vf()
def softmax_nz3_vf(ub_score: Tensor, ub_rmax: Tensor, ub_rsum: Tensor, ub_p: Tensor, ub_old_weight: Tensor):
    sreg = RegList(DT.float, UNROLL); acc = RegList(DT.float, UNROLL); preg = RegList(DT.float, UNROLL)
    p_hi = RegList(DT.hif8, UNROLL); p_hj = RegList(DT.hif8, UNROLL)
    # NZ pack: squeeze the sparse hif8 to dense lanes 0..63, then reg_to_ub_scatter places query j directly
    # at ub_p byte sidx[j] = (j//32)*129*32 + (j%32) [+ base r*32].
    # Interleaving zero bytes makes payload j occupy lane2*j; LOWHALF enables64 indices.
    m4 = MaskReg(DT.hif8, init_mode=MaskType.MULTI4)
    qmask = MaskReg(DT.hif8, init_mode=MaskType.LOWHALF)
    # M10-031: byte scatter with u16 indices reads even data/predicate lanes.
    scatter_data = Reg(DT.hif8); unused = Reg(DT.hif8)
    # C310 has no HiFloat8 scalar-broadcast overload; initialize bytes explicitly.
    zero_bytes = Reg(DT.uint8)
    zero_bytes <<= 0
    zeros = zero_bytes.reinterpret(DT.hif8)
    dpi = Reg(DT.hif8); dpj = Reg(DT.hif8)
    sidx16 = Reg(DT.int16); _build_sidx16(sidx16)
    sidx = sidx16.reinterpret(DT.uint16)
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
            squeeze(dpi, p_hi[u], mask=m4)
            interleave(scatter_data, unused, dpi, zeros)
            reg_to_ub_scatter(ub_p[(rb * UNROLL + u) * NZ_C0], scatter_data, sidx, mask=qmask)
            squeeze(dpj, p_hj[u], mask=m4)
            interleave(scatter_data, unused, dpj, zeros)
            reg_to_ub_scatter(ub_p[((rb * UNROLL + u) + HALF) * NZ_C0], scatter_data, sidx, mask=qmask)
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
def softmax_nz3_tail_vf(ub_score: Tensor, ub_rmax: Tensor, ub_rsum: Tensor, ub_p: Tensor, ub_old_weight: Tensor, valid_n: Var):
    s = Reg(DT.float); e = Reg(DT.float); neg = Reg(DT.float)
    block_max = Reg(DT.float); block_sum = Reg(DT.float); p_hif8 = Reg(DT.hif8)
    m4 = MaskReg(DT.hif8, init_mode=MaskType.MULTI4)
    qmask = MaskReg(DT.hif8, init_mode=MaskType.LOWHALF)
    # M10-031: byte scatter with u16 indices reads even data/predicate lanes.
    scatter_data = Reg(DT.hif8); unused = Reg(DT.hif8)
    # C310 has no HiFloat8 scalar-broadcast overload; initialize bytes explicitly.
    zero_bytes = Reg(DT.uint8)
    zero_bytes <<= 0
    zeros = zero_bytes.reinterpret(DT.hif8)
    dp = Reg(DT.hif8)
    sidx16 = Reg(DT.int16); _build_sidx16(sidx16)
    sidx = sidx16.reinterpret(DT.uint16)
    rowmask = MaskReg(DT.float, init_mode=MaskType.ALL)
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
        squeeze(dp, p_hif8, mask=m4)
        interleave(scatter_data, unused, dp, zeros)
        reg_to_ub_scatter(ub_p[n * NZ_C0], scatter_data, sidx, mask=qmask)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)
    v_psum <<= ub_rsum[0:1, 0:QLANES]
    v_nsum <<= v_psum * v_oldw
    v_nsum <<= v_nsum + block_sum
    ub_rmax[0:1, 0:QLANES] <<= v_nmax
    ub_rsum[0:1, 0:QLANES] <<= v_nsum
    ub_old_weight[0:1, 0:QLANES] <<= v_oldw
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

@vf()
def softmax_nz4_vf(ub_score: Tensor, ub_rmax: Tensor, ub_rsum: Tensor, ub_p: Tensor, ub_old_weight: Tensor):
    sreg = RegList(DT.float, UNROLL); acc = RegList(DT.float, UNROLL); preg = RegList(DT.float, UNROLL)
    # NZ pack (nz4): pack 4 keys per register with a 3-deep deinterleave tree (fin = [k0|k1|k2|k3] dense,
    # 256 lanes = 8 fractals) + one strided reg_to_ub (blk_stride=33). Key kk lands in fractals 2kk,2kk+1
    # at M-row m; store is 4 bulk UB2L1_NZ transfers (one slab of 32 keys each).
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
    # exp + nz4 pack: 4 keys (m, m+32, m+64, m+96) -> deinterleave tree -> fin -> one reg_to_ub
    # reuse acc[] (free after block_max) as the 4 per-slab running sums — no separate sacc RegList (register
    # economy: the full-kernel aiv inlines this VF with accum_pv/fd_merge and the deinterleave chain is heavy).
    for kk in unroll(KEYS_PER_GRP):
        acc[kk] <<= 0.0
    # shared deint temps: per-iteration regs + higher unroll were board-tested and gave NO speedup — the
    # deint pack is throughput-bound (the deinterleaves themselves), not latency-bound, so shared/unroll2 is best.
    hh = RegList(DT.hif8, KEYS_PER_GRP)             # 4 cast'd keys (stride-4 sparse, RegLayout.ZERO)
    a = Reg(DT.hif8); b = Reg(DT.hif8); fin = Reg(DT.hif8); dmy = Reg(DT.hif8)
    for rb in range(RB4):
        for u in unroll(UNROLL4):
            for kk in unroll(KEYS_PER_GRP):
                row = (rb * UNROLL4 + u) + kk * SLAB          # key m + kk*32
                sreg[0] <<= ub_score[row:row + 1, :]
                muls(sreg[0], sreg[0], SOFTMAX_SCALE)
                expsub(preg[0], sreg[0], v_nmax)
                acc[kk] <<= acc[kk] + preg[0]
                cast(hh[kk], preg[0], cfg_i)
            _deint(a,   dmy, hh[0], hh[1])
            _deint(b,   dmy, hh[2], hh[3])
            _deint(fin, dmy, a,     b)
            reg_to_ub(ub_p[(rb * UNROLL4 + u) * NZ_C0], fin, FRAC_STRIDE4)
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
def softmax_nz4_tail_vf(ub_score: Tensor, ub_rmax: Tensor, ub_rsum: Tensor, ub_p: Tensor, ub_old_weight: Tensor, valid_n: Var):
    s = Reg(DT.float); e = Reg(DT.float); neg = Reg(DT.float)
    block_max = Reg(DT.float); block_sum = Reg(DT.float); p_hif8 = Reg(DT.hif8)
    qmask = MaskReg(DT.hif8, init_mode=MaskType.LOWQUAT)
    dp = Reg(DT.hif8)
    g8 = Reg(DT.int8); arange(g8, 0); shiftls(g8, g8, 2)
    gidx = g8.reinterpret(DT.uint8)
    rowmask = MaskReg(DT.float, init_mode=MaskType.ALL)
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
    # nested DSL loops (g=slab, m=row) keep this LOOPED, not 128x-unrolled, AND compute the slab base with
    # multiplies only (n = g*SLAB+m, base = g*2112 + m*32) so there is no var_div. A _pyrange here fully
    # unrolled the 128 keys -> 1334-line VF -> the full-kernel aiv spilled 102 vector slots (26KB > 6KB).
    _SLAB_STRIDE = 2 * FRAC_STRIDE4 * NZ_C0          # 2112 = 2 fractals * 33 * 32
    for g in range(KEYS_PER_GRP):
        for m in range(SLAB):
            n = g * SLAB + m
            cnt_n = Var(QLANES * Min(Max(valid_n - n, 0), 1), dtype=DT.uint32)
            s <<= ub_score[n:n + 1, :]
            s <<= s * SOFTMAX_SCALE
            update_mask(rowmask, cnt_n)
            select(s, s, neg, mask=rowmask)
            expsub(e, s, v_nmax)
            block_sum <<= block_sum + e
            cast(p_hif8, e, cfg_i)
            gather(dp, p_hif8, gidx)
            reg_to_ub(ub_p[g * _SLAB_STRIDE + m * NZ_C0], dp, FRAC_STRIDE4, mask=qmask)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)
    v_psum <<= ub_rsum[0:1, 0:QLANES]
    v_nsum <<= v_psum * v_oldw
    v_nsum <<= v_nsum + block_sum
    ub_rmax[0:1, 0:QLANES] <<= v_nmax
    ub_rsum[0:1, 0:QLANES] <<= v_nsum
    ub_old_weight[0:1, 0:QLANES] <<= v_oldw
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

@func()
def _build_sidx16(sidx16):
    # sidx[j] = (j//32)*129*32 + (j%32) = j + (j//32)*4096  (int16, dense lanes 0..63; 64..127 unused).
    q16 = Reg(DT.int16)
    arange(sidx16, 0)                       # j
    shiftrs(q16, sidx16, 5)                 # j//32
    muls(q16, q16, 4096)                    # (j//32)*4096
    sidx16 <<= sidx16 + q16                 # j + (j//32)*4096

def _alloc_nz(q, k, v):
    q_hif8 = q.reinterpret(DT.hif8, name="q_hif8")
    k_hif8 = k.reinterpret(DT.hif8, name="k_hif8")
    v_hif8 = v.reinterpret(DT.hif8, name="v_hif8")
    l1q = DBuff(DT.hif8, [TILE_M, D_HEAD], Position.L1)
    l1k = _CacheBuf(DT.hif8, [TILE_N, D_HEAD], Position.L1)
    l1v = _CacheBuf(DT.hif8, [TILE_N, D_HEAD], Position.L1)
    l1p = _CacheBuf(DT.hif8, [TILE_N, TILE_M], Position.L1)
    l0c_qk = DBuff(DT.float, [TILE_M, TILE_M], Position.L0C)
    l0c_pv = DBuff(DT.float, [TILE_M, D_HEAD], Position.L0C)
    ub_score = DBuff(DT.float, [TILE_N, ROWS_PER_SB], Position.UB)
    ub_pv = DBuff(DT.float, [ROWS_PER_SB, D_HEAD], Position.UB)
    ub_p = Tensor(DT.hif8, [TILE_N + 1, ROWS_PER_SB], Position.UB)   # [129,64] NZ, M_src=shape[0]=129
    ub_old_weight = _CacheBuf(DT.float, [1, QLANES], Position.UB)
    ub_rmax = _CacheBuf(DT.float, [1, QLANES], Position.UB)
    ub_rsum = _CacheBuf(DT.float, [1, QLANES], Position.UB)
    ub_merge_a = Tensor(DT.float, [1, TILE_M], Position.UB)
    ub_merge_den = Tensor(DT.float, [1, TILE_M], Position.UB)
    ub_accum = Tensor(DT.float, [ROWS_PER_SB, D_HEAD], Position.UB)
    ub_out = Tensor(DT.bfloat16, [ROWS_PER_SB, D_HEAD], Position.UB)
    fd_accum = split_workspace(DT.float, [MAX_FD_WS, TILE_M, D_HEAD], name="fd_accum")
    fd_max = split_workspace(DT.float, [MAX_FD_WS, 1, TILE_M], name="fd_max")
    fd_sum = split_workspace(DT.float, [MAX_FD_WS, 1, TILE_M], name="fd_sum")
    return (q_hif8, k_hif8, v_hif8, l1q, l1k, l1v, l1p, l0c_qk, l0c_pv, ub_score, ub_pv, ub_p,
            ub_old_weight, ub_rmax, ub_rsum, ub_merge_a, ub_merge_den, ub_accum, ub_out, fd_accum, fd_max, fd_sum)

@func()
def _full_nz_body(q, k, v, out, B, MQ, N, D, cv, pv_mutex, p_mutex, softmax_fn, softmax_tail_fn):
    (q_hif8, k_hif8, v_hif8, l1q, l1k, l1v, l1p, l0c_qk, l0c_pv, ub_score, ub_pv, ub_p, ub_old_weight,
     ub_rmax, ub_rsum, ub_merge_a, ub_merge_den, ub_accum, ub_out, fd_accum, fd_max, fd_sum) = _alloc_nz(q, k, v)

    core = Var(GetCubeIdx())
    core_count = Var(GetCubeNum())
    vec = Var(GetVecIdx())
    row_begin = Var(GetSubBlockIdx() * ROWS_PER_SB)
    tiles_m_per_b = CeilDiv(MQ, TILE_M)
    k_tiles = CeilDiv(N, TILE_N)
    total_tasks = Var(B * tiles_m_per_b * k_tiles)
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

    # D-214: PV contracts only loaded V rows; no overlapping whole-L1 fill.

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
                mt = Var(t // k_tiles)
                kt = Var(t % k_tiles)
                qm = Var(mt % 2)
                sm = Var(mt % CACHE)
                batch = Var(mt // tiles_m_per_b)
                local_mt = Var(mt % tiles_m_per_b)
                q_base = Var(batch * MQ)
                kv_base = Var(batch * N)
                m0 = Var(local_mt * TILE_M)
                valid_m = Min(TILE_M, MQ - m0)
                n0 = Var(kt * TILE_N)
                valid_n = Min(TILE_N, N - n0)

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
                    softmax_tail_fn(ub_score[step], ub_rmax[sm], ub_rsum[sm], ub_p,
                                    ub_old_weight[step], valid_n)
                else:
                    softmax_fn(ub_score[step], ub_rmax[sm], ub_rsum[sm], ub_p,
                               ub_old_weight[step])
                cv.free()

                p_mutex.lock()
                l1p[step][0:TILE_N, row_begin:row_begin + ROWS_PER_SB] <<= ub_p.nz()[0:TILE_N, 0:ROWS_PER_SB]
                p_mutex.ready()

            if step >= PRELOAD_N:
                lag = Var(step - PRELOAD_N)
                lag_t = Var(flat_start + lag)
                lag_mt = Var(lag_t // k_tiles)
                lag_kt = Var(lag_t % k_tiles)
                sm2 = Var(lag_mt % CACHE)
                lag_batch = Var(lag_mt // tiles_m_per_b)
                lag_local_mt = Var(lag_mt % tiles_m_per_b)
                lag_q_base = Var(lag_batch * MQ)
                lag_kv_base = Var(lag_batch * N)
                lag_m0 = Var(lag_local_mt * TILE_M)
                lag_valid_m = Min(TILE_M, MQ - lag_m0)
                local_rows = Min(ROWS_PER_SB, Max(lag_valid_m - row_begin, 0))
                lag_n0 = Var(lag_kt * TILE_N)
                v_rows = Min(TILE_N, N - lag_n0)

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
                fd_merge_vf(ub_pv[0], ub_pv[1], ub_rmax[0], ub_rmax[1],
                            ub_rsum[0], ub_rsum[1], ub_merge_a, ub_merge_den, ub_pv[0])
                output_cast_vf(ub_pv[0], ub_out)
                out[fd_q_base + fd_m0 + fd_row0:fd_q_base + fd_m0 + fd_row0 + fd_rows, 0:D] <<= (
                    ub_out[0:fd_rows, 0:D]
                )
    return out

def _alloc_nz4(q, k, v):
    q_hif8 = q.reinterpret(DT.hif8, name="q_hif8")
    k_hif8 = k.reinterpret(DT.hif8, name="k_hif8")
    v_hif8 = v.reinterpret(DT.hif8, name="v_hif8")
    l1q = DBuff(DT.hif8, [TILE_M, D_HEAD], Position.L1)
    l1k = _CacheBuf(DT.hif8, [TILE_N, D_HEAD], Position.L1)
    l1v = _CacheBuf(DT.hif8, [TILE_N, D_HEAD], Position.L1)
    l1p = _CacheBuf(DT.hif8, [TILE_N, TILE_M], Position.L1)
    l0c_qk = DBuff(DT.float, [TILE_M, TILE_M], Position.L0C)
    l0c_pv = DBuff(DT.float, [TILE_M, D_HEAD], Position.L0C)
    ub_score = DBuff(DT.float, [TILE_N, ROWS_PER_SB], Position.UB)
    ub_pv = DBuff(DT.float, [ROWS_PER_SB, D_HEAD], Position.UB)
    ub_p = Tensor(DT.hif8, [FRAC_STRIDE4, KEYS_PER_GRP * ROWS_PER_SB], Position.UB)   # [33,256] NZ, M_src=33
    ub_old_weight = _CacheBuf(DT.float, [1, QLANES], Position.UB)
    ub_rmax = _CacheBuf(DT.float, [1, QLANES], Position.UB)
    ub_rsum = _CacheBuf(DT.float, [1, QLANES], Position.UB)
    ub_merge_a = Tensor(DT.float, [1, TILE_M], Position.UB)
    ub_merge_den = Tensor(DT.float, [1, TILE_M], Position.UB)
    ub_accum = Tensor(DT.float, [ROWS_PER_SB, D_HEAD], Position.UB)
    ub_out = Tensor(DT.bfloat16, [ROWS_PER_SB, D_HEAD], Position.UB)
    fd_accum = split_workspace(DT.float, [MAX_FD_WS, TILE_M, D_HEAD], name="fd_accum")
    fd_max = split_workspace(DT.float, [MAX_FD_WS, 1, TILE_M], name="fd_max")
    fd_sum = split_workspace(DT.float, [MAX_FD_WS, 1, TILE_M], name="fd_sum")
    return (q_hif8, k_hif8, v_hif8, l1q, l1k, l1v, l1p, l0c_qk, l0c_pv, ub_score, ub_pv, ub_p,
            ub_old_weight, ub_rmax, ub_rsum, ub_merge_a, ub_merge_den, ub_accum, ub_out, fd_accum, fd_max, fd_sum)

@func()
def _full_nz4_body(q, k, v, out, B, MQ, N, D, cv, pv_mutex, p_mutex, softmax_fn, softmax_tail_fn):
    (q_hif8, k_hif8, v_hif8, l1q, l1k, l1v, l1p, l0c_qk, l0c_pv, ub_score, ub_pv, ub_p, ub_old_weight,
     ub_rmax, ub_rsum, ub_merge_a, ub_merge_den, ub_accum, ub_out, fd_accum, fd_max, fd_sum) = _alloc_nz4(q, k, v)

    core = Var(GetCubeIdx())
    core_count = Var(GetCubeNum())
    vec = Var(GetVecIdx())
    row_begin = Var(GetSubBlockIdx() * ROWS_PER_SB)
    tiles_m_per_b = CeilDiv(MQ, TILE_M)
    k_tiles = CeilDiv(N, TILE_N)
    total_tasks = Var(B * tiles_m_per_b * k_tiles)
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

    # D-214: PV contracts only loaded V rows; no overlapping whole-L1 fill.

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
                mt = Var(t // k_tiles)
                kt = Var(t % k_tiles)
                qm = Var(mt % 2)
                sm = Var(mt % CACHE)
                batch = Var(mt // tiles_m_per_b)
                local_mt = Var(mt % tiles_m_per_b)
                q_base = Var(batch * MQ)
                kv_base = Var(batch * N)
                m0 = Var(local_mt * TILE_M)
                valid_m = Min(TILE_M, MQ - m0)
                n0 = Var(kt * TILE_N)
                valid_n = Min(TILE_N, N - n0)

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
                    softmax_tail_fn(ub_score[step], ub_rmax[sm], ub_rsum[sm], ub_p,
                                    ub_old_weight[step], valid_n)
                else:
                    softmax_fn(ub_score[step], ub_rmax[sm], ub_rsum[sm], ub_p,
                               ub_old_weight[step])
                cv.free()

                p_mutex.lock()
                for g in unroll(KEYS_PER_GRP):
                    l1p[step][g * SLAB:(g + 1) * SLAB, row_begin:row_begin + ROWS_PER_SB] <<= (
                        ub_p.nz()[0:SLAB, g * ROWS_PER_SB:(g + 1) * ROWS_PER_SB]
                    )
                p_mutex.ready()

            if step >= PRELOAD_N:
                lag = Var(step - PRELOAD_N)
                lag_t = Var(flat_start + lag)
                lag_mt = Var(lag_t // k_tiles)
                lag_kt = Var(lag_t % k_tiles)
                sm2 = Var(lag_mt % CACHE)
                lag_batch = Var(lag_mt // tiles_m_per_b)
                lag_local_mt = Var(lag_mt % tiles_m_per_b)
                lag_q_base = Var(lag_batch * MQ)
                lag_kv_base = Var(lag_batch * N)
                lag_m0 = Var(lag_local_mt * TILE_M)
                lag_valid_m = Min(TILE_M, MQ - lag_m0)
                local_rows = Min(ROWS_PER_SB, Max(lag_valid_m - row_begin, 0))
                lag_n0 = Var(lag_kt * TILE_N)
                v_rows = Min(TILE_N, N - lag_n0)

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
                fd_merge_vf(ub_pv[0], ub_pv[1], ub_rmax[0], ub_rmax[1],
                            ub_rsum[0], ub_rsum[1], ub_merge_a, ub_merge_den, ub_pv[0])
                output_cast_vf(ub_pv[0], ub_out)
                out[fd_q_base + fd_m0 + fd_row0:fd_q_base + fd_m0 + fd_row0 + fd_rows, 0:D] <<= (
                    ub_out[0:fd_rows, 0:D]
                )
    return out




@kernel()
def nzp_full_nz1_kernel(q: GM[u8, ('TQ', 'D')], k: GM[u8, ('TK', 'D')], v: GM[u8, ('TK', 'D')], out: GM[bf16, ('TQ', 'D')], B: i32, MQ: i32, N: i32, D: i32):
    return _full_nz_body(q, k, v, out, B, MQ, N, D, _cv_normal(), _pv_normal(), _p_normal(),
                         softmax_nz1_vf, softmax_nz1_tail_vf)

@kernel()
def nzp_full_nz2_kernel(q: GM[u8, ('TQ', 'D')], k: GM[u8, ('TK', 'D')], v: GM[u8, ('TK', 'D')], out: GM[bf16, ('TQ', 'D')], B: i32, MQ: i32, N: i32, D: i32):
    return _full_nz_body(q, k, v, out, B, MQ, N, D, _cv_normal(), _pv_normal(), _p_normal(),
                         softmax_nz2_vf, softmax_nz2_tail_vf)

@kernel()
def nzp_full_nz3_kernel(q: GM[u8, ('TQ', 'D')], k: GM[u8, ('TK', 'D')], v: GM[u8, ('TK', 'D')], out: GM[bf16, ('TQ', 'D')], B: i32, MQ: i32, N: i32, D: i32):
    return _full_nz_body(q, k, v, out, B, MQ, N, D, _cv_normal(), _pv_normal(), _p_normal(),
                         softmax_nz3_vf, softmax_nz3_tail_vf)

@kernel()
def nzp_full_nz4_kernel(q: GM[u8, ('TQ', 'D')], k: GM[u8, ('TK', 'D')], v: GM[u8, ('TK', 'D')], out: GM[bf16, ('TQ', 'D')], B: i32, MQ: i32, N: i32, D: i32):
    return _full_nz4_body(q, k, v, out, B, MQ, N, D, _cv_normal(), _pv_normal(), _p_normal(),
                          softmax_nz4_vf, softmax_nz4_tail_vf)
