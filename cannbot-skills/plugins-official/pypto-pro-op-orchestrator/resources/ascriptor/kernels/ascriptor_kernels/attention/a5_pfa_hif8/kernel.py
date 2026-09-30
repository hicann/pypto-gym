# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Ten all-HiFloat8 PFA attention kernels: four publication/grouping bodies and six precision studies.

`nd` publishes P key-major in ND. `nz` (the promoted V6) builds NZ directly in UB with a
four-key deinterleave at physical stride 33 and biases the exponent by ln16 so P and its
denominator both carry a factor of 16. `v8` groups three physical 128-key score tiles under
one maximum update so three P beats accumulate into a single L0C PV result. `v9` exports the
score to FP16 through two SINGLE drains with a FixP19 scale and keeps its online state in FP16.

The five `study_*` bodies are an ablation over ONE schedule: V1/V2 apply the FixP19 scale at
the fixpipe and carry FP32/FP16 state; V3/V4 apply it in the vector function; v3_splitn is V3
with a single SPLITN fixpipe bridge instead of the dual-SINGLE one. Each has a `_nocv`
companion whose cube-to-vector mutexes are replaced by a no-op, which makes the missing handoff
show up as a wrong value instead of a hang; those are profiling probes and are deliberately
incorrect, so no case runs them."""

import math

from ascriptor.a5 import *

# ----------------------------------------------------------------------------------------------------
# nd.py
# All-HiFloat8 PFA with key-major ND probability publication.
#
# Q/K/V use finite HiFloat8 uint8 carriers, interpreted as HiFloat8 by the A5
# matmuls. The vector stage uses FP32 online state and casts unnormalized P
# with ties away from zero before PV; the FP32 denominator precedes that cast.
# Each batch shares one KV stream across flattened query rows. The local runner
# checks at most two FD owners and rejects duplicate interior split boundaries.
# Use run.py and contract.json for generated cases and current validation.
# ----------------------------------------------------------------------------------------------------

# ---- geometry (copied from qk_softmax_pv_flat; that module is a5pr so we cannot import it) ----
TILE_M = 128
ROWS_PER_SB = 64
QLANES = ROWS_PER_SB
TILE_N = 128
D_HEAD = 128
SOFTMAX_SCALE = 1.0 / math.sqrt(D_HEAD)
HALF = TILE_N // 2
CHUNKS_D = D_HEAD // 64
KROW = QLANES   # 64: pack4 key-major row stride (32-aligned)
# agg_vf's fd_merge_nd_vf / output_cast_nd_vf (inlined below) refer to M_Q / CD verbatim.
M_Q = QLANES
CD = CHUNKS_D

PRELOAD_N = 2
CACHE = PRELOAD_N + 1
MAX_CORE_COUNT = 32
MAX_FD_WS = MAX_CORE_COUNT * 2
UNROLL = 4
RB_MAX = TILE_N // UNROLL
RB_EXP = HALF // UNROLL
NEG_LARGE = -1.0e30


@vf()
def init_softmax_state_vf(ub_rmax: Tensor, ub_rsum: Tensor):
    neg = Reg(DT.float); zero = Reg(DT.float)
    neg <<= NEG_LARGE; zero <<= 0.0
    ub_rmax[0:1, 0:QLANES] <<= neg
    ub_rsum[0:1, 0:QLANES] <<= zero


@vf()
def accum_pv_nd_vf(ub_accum: Tensor, ub_pv: Tensor, ub_old_weight: Tensor):
    acc = RegList(DT.float, CHUNKS_D); pv = RegList(DT.float, CHUNKS_D)
    old_weight = Reg(DT.float)
    for r in range(ROWS_PER_SB):
        old_weight <<= ub_old_weight[0:1, r:r + 1].single()
        acc <<= ub_accum[r:r + 1, 0:D_HEAD]
        pv <<= ub_pv[r:r + 1, 0:D_HEAD]
        acc <<= acc * old_weight
        acc <<= acc + pv
        ub_accum[r:r + 1, 0:D_HEAD] <<= acc
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def accum_pv_first_vf(ub_accum: Tensor, ub_pv: Tensor):
    pv = RegList(DT.float, CHUNKS_D)
    for r in range(ROWS_PER_SB):
        pv <<= ub_pv[r:r + 1, 0:D_HEAD]
        ub_accum[r:r + 1, 0:D_HEAD] <<= pv
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


# ---- flash-decoding aggregation VFs (inlined from agg_vf.py) ------------------
@vf()
def fd_merge_nd_vf(ub_o0: Tensor, ub_o1: Tensor, ub_m0: Tensor, ub_m1: Tensor,
                ub_s0: Tensor, ub_s1: Tensor, ub_A: Tensor, ub_den: Tensor,
                ub_merged: Tensor):
    """AGG-1: 2-split merge + normalize. Prelude (vectorized 64-lane) precomputes A=exp(m0-m1)
    and den_eff=s0*A+s1; main loop per row does O=(o0*A+o1)/den_eff (6 VLD/2 VMUL/2 VADD/2 VDIV/2 VST)."""
    # --- prelude: vectorized over the 64 query lanes (m/s are [1, 64]) ---
    m0v = Reg(DT.float)
    m1v = Reg(DT.float)
    s0v = Reg(DT.float)
    s1v = Reg(DT.float)
    Av = Reg(DT.float)
    denv = Reg(DT.float)
    m0v <<= ub_m0[0:1, 0:M_Q]
    m1v <<= ub_m1[0:1, 0:M_Q]
    s0v <<= ub_s0[0:1, 0:M_Q]
    s1v <<= ub_s1[0:1, 0:M_Q]
    expsub(Av, m0v, m1v)                          # A = exp(m0 - m1)
    denv <<= s0v * Av                             # s0*A
    denv <<= denv + s1v                           # den_eff = s0*A + s1
    ub_A[0:1, 0:M_Q] <<= Av
    ub_den[0:1, 0:M_Q] <<= denv
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

    # --- main loop: O = (o0*A + o1) / den_eff, per query row ---
    o0 = RegList(DT.float, CD)
    o1 = RegList(DT.float, CD)
    Ar = Reg(DT.float)
    dr = Reg(DT.float)
    for r in range(M_Q):
        Ar <<= ub_A[0:1, r:r + 1].single()        # broadcast A[r]   (RV_VLD)
        dr <<= ub_den[0:1, r:r + 1].single()      # broadcast den[r] (RV_VLD)
        o0 <<= ub_o0[r:r + 1, :]                   # 2 VLD
        o1 <<= ub_o1[r:r + 1, :]                   # 2 VLD
        o0 <<= o0 * Ar                             # o0 *= A  (2 RV_VMUL)
        o0 <<= o0 + o1                             # o0 += o1 (2 RV_VADD)
        o0 <<= o0 / dr                             # o0 /= den_eff (2 RV_VDIV)
        ub_merged[r:r + 1, :] <<= o0               # store fp32 merged O (2 RV_VST)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def output_cast_nd_vf(ub_merged: Tensor, ub_out: Tensor):
    """AGG-2: fp32 -> bf16 cast of the merged O, per D-chunk, then downsample-pack store."""
    regs = RegList(DT.float, CD)
    hregs = RegList(DT.bfloat16, CD)
    cfg = CastConfig(round_mode=RoundMode.AWAY_FROM_ZERO, reg_layout=RegLayout.ZERO,
                     name="o_bf16")
    for r in range(M_Q):
        regs <<= ub_merged[r:r + 1, :]            # 2 VLD
        for c in unroll(CD):
            cast(hregs[c], regs[c], cfg)          # fp32 -> bf16 (RV_VCVT_F2F)
            reg_to_ub_downsample(ub_out[r:r + 1, c * 64:(c + 1) * 64], hregs[c])  # pack 64 bf16 (RV_VST)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def init_accum_vf(ub_accum: Tensor):
    zero = Reg(DT.float)
    zero <<= 0.0
    for r in range(ROWS_PER_SB):
        ub_accum[r:r + 1, 0:D_HEAD] <<= zero
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def softmax_t_vf(ub_score: Tensor, ub_rmax: Tensor, ub_rsum: Tensor, ub_p: Tensor,
                 ub_old_weight: Tensor):
    """Online softmax, hif8 P^T KEY-CONTIGUOUS ND via reg_to_ub_pack4 (CAST_ROUND encode)."""
    sreg = RegList(DT.float, UNROLL); acc = RegList(DT.float, UNROLL); preg = RegList(DT.float, UNROLL)
    p_hi = RegList(DT.hif8, UNROLL); p_hj = RegList(DT.hif8, UNROLL)
    p_mask4 = MaskReg(DT.hif8, init_mode=MaskType.NONE)
    p_mask4 <<= Var(4 * QLANES, dtype=DT.uint32)
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
def neg_fill_rows(ub_score: Tensor, valid_n: Var):
    """NEG the INVALID key rows of a partial K-tile's score^T, once, so the tail softmax below
    needs no per-row mask. A row that reads NEG loses the lane-wise max and exponentiates to
    exactly 0 - what the mask produced - so the arithmetic is unchanged. What goes away is one
    scalar clamp per key row INSIDE the vec scope, and pl has no narrower spelling than a 64-bit
    compare for it (D-139 / D-150): on the board that clamp was the whole of fd_modified's
    tail-shape gap (D-156)."""
    neg = Reg(DT.float)
    neg <<= NEG_LARGE
    for n in range(valid_n, TILE_N):
        ub_score[n:n + 1, 0:QLANES] <<= neg


@vf()
def softmax_t_tail_vf(ub_score: Tensor, ub_rmax: Tensor, ub_rsum: Tensor, ub_p: Tensor,
                      ub_old_weight: Tensor):
    """Tail softmax (partial key tile), pack4 key-major hif8 P; invalid keys -> zero P row."""
    s = Reg(DT.float); e = Reg(DT.float); neg = Reg(DT.float)
    block_max = Reg(DT.float); block_sum = Reg(DT.float); p_hif8 = Reg(DT.hif8)
    p_mask4 = MaskReg(DT.hif8, init_mode=MaskType.NONE)
    p_mask4 <<= Var(4 * QLANES, dtype=DT.uint32)
    cfg_i = CastConfig(round_mode=RoundMode.AWAY_FROM_ZERO, reg_layout=RegLayout.ZERO, name="p_hif8_tail_i")
    v_pmax = Reg(DT.float); v_nmax = Reg(DT.float); v_oldw = Reg(DT.float)
    v_psum = Reg(DT.float); v_nsum = Reg(DT.float)
    block_max <<= NEG_LARGE
    neg <<= NEG_LARGE
    for n in range(TILE_N):
        s <<= ub_score[n:n + 1, :]
        s <<= s * SOFTMAX_SCALE
        block_max <<= block_max.vmax(s)
    v_pmax <<= ub_rmax[0:1, 0:QLANES]
    v_nmax <<= block_max.vmax(v_pmax)
    expsub(v_oldw, v_pmax, v_nmax)
    block_sum <<= 0.0
    for n in range(TILE_N):
        s <<= ub_score[n:n + 1, :]
        s <<= s * SOFTMAX_SCALE
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


@kernel()
def qk_softmax_pv_hif8_kernel(q: GM[u8, ('TQ', 'D')], k: GM[u8, ('TK', 'D')], v: GM[u8, ('TK', 'D')], out: GM[bf16, ('TQ', 'D')],
                              B: i32, MQ: i32, N: i32, D: i32):
    # Public Q/K/V are uint8 carriers (torch has no hif8 dtype). Relabel them as hif8 (zero-copy,
    # same byte width) so the QK^T / PV matmuls consume hif8 L0A/L0B.
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
    ub_p = Tensor(DT.hif8, [TILE_N, KROW], Position.UB)   # pack4 key-major: row k = key k
    ub_old_weight = TBuff(DT.float, [1, QLANES], Position.UB)
    ub_rmax = TBuff(DT.float, [1, QLANES], Position.UB)
    ub_rsum = TBuff(DT.float, [1, QLANES], Position.UB)
    ub_accum = Tensor(DT.float, [ROWS_PER_SB, D_HEAD], Position.UB)
    ub_out = Tensor(DT.bfloat16, [ROWS_PER_SB, D_HEAD], Position.UB)
    # Dedicated single buffers for the prefix partial state. The full pipe cycles
    # ub_rmax/ub_rsum/ub_accum through slots {0,1,2}=mt%CACHE; since the prefix runs
    # BEFORE the full pipe on the same core, a full-pipe init_softmax_state/init_accum
    # whose slot aliases the prefix slot resets the row-max to -inf (and zeros accum)
    # before the prefix's workspace store has flushed -- a WAR the autosync tracker
    # misses across the literal/runtime TBuff index. Separate buffers remove the alias.
    ub_accum_pfx = Tensor(DT.float, [ROWS_PER_SB, D_HEAD], Position.UB)
    ub_rmax_pfx = Tensor(DT.float, [1, QLANES], Position.UB)
    ub_rsum_pfx = Tensor(DT.float, [1, QLANES], Position.UB)
    # Same story for the suffix partial: a core may run prefix -> suffix back to back
    # (no full pipe between), and the merge phase reuses ub_accum/ub_rmax[0] right after
    # the barrier. Give the suffix its own buffers too, fully decoupled from both the
    # full pipe's {0,1,2} slots and the prefix buffers.
    ub_rmax_sfx = Tensor(DT.float, [1, QLANES], Position.UB)
    ub_rsum_sfx = Tensor(DT.float, [1, QLANES], Position.UB)

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
    prefix_tasks = Var(0, DT.int)
    if my_tasks > 0:
        start_k = Var(flat_start % k_tiles)
        if start_k != 0:
            prefix_tasks <<= Min(k_tiles - start_k, my_tasks)
    pipe_start = Var(flat_start + prefix_tasks)
    pipe_tasks = Var(my_tasks - prefix_tasks)
    suffix_tasks = Var(0, DT.int)
    if pipe_tasks > 0:
        end_k = Var(flat_end % k_tiles)
        if end_k != 0:
            suffix_tasks <<= end_k
    full_tasks = Var(pipe_tasks - suffix_tasks)
    suffix_start = Var(flat_end - suffix_tasks)

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

    # l1p NOT pre-zeroed: every published P tile fully covers the [TILE_N, TILE_M] PV-matmul read
    # (both subblocks write their query columns; tail softmax writes 0 for invalid keys). Zeroing on
    # CUBE would race the VEC nd2nz publish (cross-core WAW) and corrupt l1p on cannsim.

    with auto_sync():
        # --- fused single-loop (replaces prefix/full/suffix) ---
        # Pre-compute which M-tile is the boundary prefix / suffix for this core.
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
                # === STAGE 1: QK^T + softmax + P^T publish ===
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

                # Q load on the first task of each M-tile for this core
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
                l0c_to_ub(ub_score[step], l0c_qk[step][0:TILE_N, 0:TILE_M],
                          M=TILE_N, N=TILE_M, N_dst=ROWS_PER_SB, M_src=TILE_N,
                          dual_mode=DualMode.SPLITN, sub_block_id=0)
                cv.ready()
                cv.wait()
                if valid_n < TILE_N:
                    neg_fill_rows(ub_score[step], valid_n)
                    softmax_t_tail_vf(ub_score[step], ub_rmax[sm], ub_rsum[sm], ub_p,
                                      ub_old_weight[step])
                else:
                    softmax_t_vf(ub_score[step], ub_rmax[sm], ub_rsum[sm], ub_p,
                                 ub_old_weight[step])
                cv.free()

                p_mutex.lock()
                ub_to_l1_nd2nz(l1p[step][0:TILE_N, row_begin:row_begin + ROWS_PER_SB], ub_p,
                               m_dst=TILE_N, n_dst=ROWS_PER_SB, m_src=TILE_N, n_src=ROWS_PER_SB, N_src=KROW)
                p_mutex.ready()

            if step >= PRELOAD_N:
                # === STAGE 2 (lag-2): PV + accum + finalize ===
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

                # A core can start in the middle of an M tile when FD splits that
                # tile by K. That first local partial has no prior local accum even
                # though its global K index is nonzero.
                if lag_kt == 0:
                    init_accum_vf(ub_accum)
                    accum_pv_first_vf(ub_accum, ub_pv[lag])
                elif lag == 0:
                    init_accum_vf(ub_accum)
                    accum_pv_first_vf(ub_accum, ub_pv[lag])
                else:
                    accum_pv_nd_vf(ub_accum, ub_pv[lag], ub_old_weight[lag])
                pv_mutex.free()

                # M-tile ended for this core?
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

    # Cross-core barrier: gate the merge's GM->UB loads (MTE2) on every core's workspace
    # stores (MTE3). allvec_wait is a true vec-wide barrier, so once it returns all peer
    # partials are visible.
    allvec_wait(7, Pipe.MTE2)

    # The per-vec merge target is data-dependent: not every vec owns a split block
    # (fd_found), and a partial M-tile makes a sub-block empty (fd_rows == 0). Keep these
    # runtime branches OUTSIDE auto_sync and wrap only the branch-free load->merge->store
    # body, so every vec that enters the region runs the identical synced sequence.
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
                ub_rmax[0][:, :] <<= fd_max[fd_lo, 0:1, fd_row0:fd_row0 + ROWS_PER_SB]
                ub_rmax[1][:, :] <<= fd_max[fd_hi, 0:1, fd_row0:fd_row0 + ROWS_PER_SB]
                ub_rsum[0][:, :] <<= fd_sum[fd_lo, 0:1, fd_row0:fd_row0 + ROWS_PER_SB]
                ub_rsum[1][:, :] <<= fd_sum[fd_hi, 0:1, fd_row0:fd_row0 + ROWS_PER_SB]
                ub_pv[0][:, :] <<= fd_accum[fd_lo, fd_row0:fd_row0 + ROWS_PER_SB, 0:D_HEAD]
                ub_pv[1][:, :] <<= fd_accum[fd_hi, fd_row0:fd_row0 + ROWS_PER_SB, 0:D_HEAD]
                # Merge in place into ub_pv[0] (already holds partial-0; fd_merge reads then
                # writes each row, so aliasing o0 and the output is safe).
                fd_merge_nd_vf(ub_pv[0], ub_pv[1], ub_rmax[0], ub_rmax[1],
                            ub_rsum[0], ub_rsum[1], ub_old_weight[0], ub_old_weight[1], ub_pv[0])
                output_cast_nd_vf(ub_pv[0], ub_out)
                out[fd_q_base + fd_m0 + fd_row0:fd_q_base + fd_m0 + fd_row0 + fd_rows, 0:D] <<= (
                    ub_out[0:fd_rows, 0:D]
                )
    return out

# ----------------------------------------------------------------------------------------------------
# nz.py
# Promoted V6 all-HiFloat8 PFA with direct NZ probability publication.
#
# Four-key deinterleave constructs NZ directly in UB with physical stride33.
# The FP32 exponential path scales both P and its denominator by16 before the
# HiFloat8 probability cast. The corrected tail uses one invalid-score NEG fill.
# A5 binding, partition validation and independent reference are local to this unit.
# ----------------------------------------------------------------------------------------------------

# fp32_to_hif8 / hif8_to_fp32 (easyasc.dtypehelper.hif8_codec) come in via the a5pr/a5 wildcard above.

_pyrange = unroll  # the old script aliased Python's range for compile-time unrolling

LN16 = math.log(16.0)   # exp-max bias: subtract ln16 so P (and rsum/PV) scale x16 -> hif8 quantizes 16*P (denser

_CacheBuf = QBuff if CACHE >= 4 else TBuff  # l1 K/V/P + softmax-state buffer depth tracks CACHE (TBuff@3, QBuff@4)

# ---- nz4 ("deinterleave 4-key") NZ P-store constants ----
NZ_C0 = 32                  # hif8 C0 (32-byte fractal inner dim)
SLAB = TILE_N // 4          # 32 keys per slab (4 slabs across TILE_N)
FRAC_STRIDE4 = SLAB + 1     # 33 -- M_src (NZ fractal row stride), +1 pad row/fractal (bank-conflict dodge)
KEYS_PER_GRP = 4            # 4 keys packed per register via the 3-deep deinterleave tree
RB4 = 16                    # rb count for the pack loop (RB4*UNROLL4 = 32 = SLAB M-rows)
UNROLL4 = 2                 # unroll(2), 4 keys/iter, per-iteration regs (real pipelining of the deint chains)
_DEINT_U8 = False           # hif8-native deinterleave; flip True if board rejects hif8 (uint8 reinterpret)


def _deint(d0, d1, s0, s1):
    if _DEINT_U8:
        deinterleave(d0.reinterpret(DT.uint8), d1.reinterpret(DT.uint8),
                     s0.reinterpret(DT.uint8), s1.reinterpret(DT.uint8))
    else:
        deinterleave(d0, d1, s0, s1)




# =========================== softmax: NZ-direct P-store via deinterleave 4-key pack ===========================
@vf()
def softmax_vf(ub_score: Tensor, ub_rmax: Tensor, ub_rsum: Tensor, ub_p: Tensor, ub_old_weight: Tensor):
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
    v_nmax_e = Reg(DT.float)
    adds(v_nmax_e, v_nmax, -LN16)   # exp-only x16 bias (kept out of the stored/compared running max, which stays true)
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
                expsub(preg[0], sreg[0], v_nmax_e)   # = 16 * exp(s - nmax): rsum (acc) and the hif8 P both carry x16
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
def softmax_tail_vf(ub_score: Tensor, ub_rmax: Tensor, ub_rsum: Tensor, ub_p: Tensor, ub_old_weight: Tensor):
    s = Reg(DT.float); e = Reg(DT.float); neg = Reg(DT.float)
    block_max = Reg(DT.float); block_sum = Reg(DT.float); p_hif8 = Reg(DT.hif8)
    qmask = MaskReg(DT.hif8, init_mode=MaskType.LOWQUAT)
    dp = Reg(DT.hif8)
    g8 = Reg(DT.int8); arange(g8, 0); shiftls(g8, g8, 2)
    gidx = g8.reinterpret(DT.uint8)
    cfg_i = CastConfig(round_mode=RoundMode.AWAY_FROM_ZERO, reg_layout=RegLayout.ZERO, name="p_hif8_tail_i")
    v_pmax = Reg(DT.float); v_nmax = Reg(DT.float); v_oldw = Reg(DT.float)
    v_psum = Reg(DT.float); v_nsum = Reg(DT.float)
    block_max <<= NEG_LARGE
    neg <<= NEG_LARGE
    for n in range(TILE_N):
        s <<= ub_score[n:n + 1, :]
        s <<= s * SOFTMAX_SCALE
        block_max <<= block_max.vmax(s)
    v_pmax <<= ub_rmax[0:1, 0:QLANES]
    v_nmax <<= block_max.vmax(v_pmax)
    expsub(v_oldw, v_pmax, v_nmax)
    v_nmax_e = Reg(DT.float)
    adds(v_nmax_e, v_nmax, -LN16)   # exp-only x16 bias (same as softmax_vf); stored/compared max stays true
    block_sum <<= 0.0
    # nested DSL loops (g=slab, m=row) keep this LOOPED, not 128x-unrolled, AND compute the slab base with
    # multiplies only (n = g*SLAB+m, base = g*2112 + m*32) so there is no var_div. A _pyrange here fully
    # unrolled the 128 keys -> 1334-line VF -> the full-kernel aiv spilled 102 vector slots (26KB > 6KB).
    _SLAB_STRIDE = 2 * FRAC_STRIDE4 * NZ_C0          # 2112 = 2 fractals * 33 * 32
    for g in range(KEYS_PER_GRP):
        for m in range(SLAB):
            n = g * SLAB + m
            s <<= ub_score[n:n + 1, :]
            s <<= s * SOFTMAX_SCALE
            expsub(e, s, v_nmax_e)   # = 16 * exp(s - nmax)
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


# =========================== PV accum / finalize ===========================
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








# =========================== FD aggregation ===========================
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


# =========================== QK L0C->UB bridge (single SPLITN fixpipe) ===========================
@func()
def _qk_bridge_splitn(ub_dst, l0c_src, l0c_scale):
    # ONE SPLITN fixpipe copy; HW auto-splits N across the 2 vec sub-blocks. Cannot requant, so
    # scale must be 1.0 (plain copy).
    l0c_to_ub(ub_dst, l0c_src[0:TILE_N, 0:TILE_M],
              M=TILE_N, N=TILE_M, N_dst=ROWS_PER_SB, M_src=TILE_N,
              dual_mode=DualMode.SPLITN, sub_block_id=0, scale=l0c_scale)


# =========================== shared allocation (NZ ub_p [33,256]; 4 bulk stores) ===========================
def _alloc(q, k, v):
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


# ---- full V6 with NZ P-store (softmax VFs passed as @func params: trace-time data, not a constant `if`) ----
@func()
def _v6_body(q, k, v, out, B, MQ, N, D, cv, pv_mutex, p_mutex, softmax_fn, softmax_tail_fn):
    (q_hif8, k_hif8, v_hif8, l1q, l1k, l1v, l1p, l0c_qk, l0c_pv, ub_score, ub_pv, ub_p, ub_old_weight,
     ub_rmax, ub_rsum, ub_merge_a, ub_merge_den, ub_accum, ub_out, fd_accum, fd_max, fd_sum) = _alloc(q, k, v)

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

    # PV contracts only valid loaded V rows; no overlapping L1 fill.

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
                    neg_fill_rows(ub_score[step], valid_n)
                    softmax_tail_fn(ub_score[step], ub_rmax[sm], ub_rsum[sm], ub_p,
                                    ub_old_weight[step])
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
def pfa_fd_v6_allhif8_kernel(q: GM[u8, ('TQ', 'D')], k: GM[u8, ('TK', 'D')], v: GM[u8, ('TK', 'D')], out: GM[bf16, ('TQ', 'D')], B: i32, MQ: i32, N: i32, D: i32):
    cv = CvMutex(0, depth=2, src_start_pipe=Pipe.FIX, dst_start_pipe=Pipe.V,
                 src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    pv_mutex = CvMutex(2, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    p_mutex = VcMutex(1, depth=CACHE, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE1)
    return _v6_body(q, k, v, out, B, MQ, N, D, cv, pv_mutex, p_mutex, softmax_vf, softmax_tail_vf)

# ----------------------------------------------------------------------------------------------------
# study.py
# Existing all-HiFloat8 PFA precision and synchronization studies.
#
# The four precision bodies use one schedule with explicit trace-time parameters:
# V1 applies FixP19 scale then FP32 state; V2 applies FixP19 scale then FP16
# score/state; V3 applies scale in the FP32 vector function; V4 rounds the raw
# score to FP16 before FP16 vector scaling/state. The split-N control retains V3
# arithmetic. Each has an intentionally unsynchronized nocv diagnostic companion.
# Select named cases through run.py; diagnose.py reproduces the failed ablations.
# Exact current evidence is independent of the source campaign's reported timings.
# ----------------------------------------------------------------------------------------------------

KREGS = TILE_N // 2            # 64
RB_MAX_H = KREGS // UNROLL     # 16
RB_EXP_H = KREGS // UNROLL     # 16
NEG_LARGE_H = -60000.0




@vf()
def init_softmax_half_vf(ub_rmax: Tensor, ub_rsum: Tensor):
    neg = Reg(DT.half); zero = Reg(DT.half)
    neg <<= NEG_LARGE_H; zero <<= 0.0
    ub_rmax[0:1, 0:TILE_N] <<= neg
    ub_rsum[0:1, 0:TILE_N] <<= zero


# =========================== fp32 softmax — 4 FULLY SEPARATE VFs (no shared `if`) ===========================
# A constant `if` inside a VF is codegen'd into a hardware conditional branch that tanks VF
# throughput, so scale / noscale are duplicated bodies (the only delta is the muls lines).
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
def softmax_t_noscale_vf(ub_score: Tensor, ub_rmax: Tensor, ub_rsum: Tensor, ub_p: Tensor, ub_old_weight: Tensor):
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
    v_pmax <<= ub_rmax[0:1, 0:QLANES]
    v_nmax <<= block_max.vmax(v_pmax)
    expsub(v_oldw, v_pmax, v_nmax)
    for u in unroll(UNROLL):
        acc[u] <<= 0.0
    for rb in range(RB_EXP):
        for u in unroll(UNROLL):
            sreg[u] <<= ub_score[(rb * UNROLL + u):(rb * UNROLL + u) + 1, :]
            expsub(preg[u], sreg[u], v_nmax)
            acc[u] <<= acc[u] + preg[u]
            cast(p_hi[u], preg[u], cfg_i)
        for u in unroll(UNROLL):
            sreg[u] <<= ub_score[(rb * UNROLL + u + HALF):(rb * UNROLL + u + HALF) + 1, :]
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
def softmax_t_noscale_tail_vf(ub_score: Tensor, ub_rmax: Tensor, ub_rsum: Tensor, ub_p: Tensor, ub_old_weight: Tensor, valid_n: Var):
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


# =========================== fp16 softmax — 4 FULLY SEPARATE VFs ===========================
@vf()
def softmax_half_scale_vf(ub_h: Tensor, ub_rmax: Tensor, ub_rsum: Tensor, ub_p: Tensor, ub_old_weight: Tensor):
    sreg = RegList(DT.half, UNROLL); acc = RegList(DT.half, UNROLL); preg = RegList(DT.half, UNROLL)
    p_h = RegList(DT.hif8, UNROLL)
    cfg_i = CastConfig(round_mode=RoundMode.AWAY_FROM_ZERO, reg_layout=RegLayout.ZERO, name="ph_hif8")
    neg = Reg(DT.half); block_max = Reg(DT.half); block_sum = Reg(DT.half); t = Reg(DT.half); swap = Reg(DT.half)
    v_pmax = Reg(DT.half); v_nmax = Reg(DT.half); v_oldw = Reg(DT.half)
    v_psum = Reg(DT.half); v_nsum = Reg(DT.half)
    # half-swap gather index [64..127, 0..63] = lane XOR 64, so gather(swap, x, idxu) exchanges the
    # two 64-lane halves (even-key reduce <-> odd-key reduce) for the cross-half vmax/add merge.
    idx16 = Reg(DT.int16); arange(idx16, 0)
    swp64 = Reg(DT.int16); swp64 <<= QLANES
    vxor(idx16, idx16, swp64)
    idxu = idx16.reinterpret(DT.uint16)
    neg <<= NEG_LARGE_H
    for u in unroll(UNROLL):
        acc[u] <<= neg
    for rb in range(RB_MAX_H):
        for u in unroll(UNROLL):
            sreg[u] <<= ub_h[(rb * UNROLL + u) * TILE_N]
            acc[u] <<= acc[u].vmax(sreg[u])
    block_max <<= acc[0].vmax(acc[1]); t <<= acc[2].vmax(acc[3]); block_max <<= block_max.vmax(t)
    gather(swap, block_max, idxu)
    block_max <<= block_max.vmax(swap)
    muls(block_max, block_max, SOFTMAX_SCALE)
    v_pmax <<= ub_rmax[0:1, 0:TILE_N]
    v_nmax <<= block_max.vmax(v_pmax)
    sub(v_oldw, v_pmax, v_nmax); exp(v_oldw, v_oldw)
    for u in unroll(UNROLL):
        acc[u] <<= 0.0
    for rb in range(RB_EXP_H):
        for u in unroll(UNROLL):
            sreg[u] <<= ub_h[(rb * UNROLL + u) * TILE_N]
            muls(sreg[u], sreg[u], SOFTMAX_SCALE)
            sub(preg[u], sreg[u], v_nmax)
            exp(preg[u], preg[u])
            acc[u] <<= acc[u] + preg[u]
            cast(p_h[u], preg[u], cfg_i)
        for u in unroll(UNROLL):
            reg_to_ub_downsample(ub_p[(rb * UNROLL + u) * TILE_N], p_h[u])
    block_sum <<= acc[0] + acc[1]; t <<= acc[2] + acc[3]; block_sum <<= block_sum + t
    gather(swap, block_sum, idxu)
    block_sum <<= block_sum + swap
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)
    v_psum <<= ub_rsum[0:1, 0:TILE_N]
    v_nsum <<= v_psum * v_oldw
    v_nsum <<= v_nsum + block_sum
    ub_rmax[0:1, 0:TILE_N] <<= v_nmax
    ub_rsum[0:1, 0:TILE_N] <<= v_nsum
    ub_old_weight[0:1, 0:TILE_N] <<= v_oldw
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def softmax_half_noscale_vf(ub_h: Tensor, ub_rmax: Tensor, ub_rsum: Tensor, ub_p: Tensor, ub_old_weight: Tensor):
    sreg = RegList(DT.half, UNROLL); acc = RegList(DT.half, UNROLL); preg = RegList(DT.half, UNROLL)
    p_h = RegList(DT.hif8, UNROLL)
    cfg_i = CastConfig(round_mode=RoundMode.AWAY_FROM_ZERO, reg_layout=RegLayout.ZERO, name="ph_hif8")
    neg = Reg(DT.half); block_max = Reg(DT.half); block_sum = Reg(DT.half); t = Reg(DT.half); swap = Reg(DT.half)
    v_pmax = Reg(DT.half); v_nmax = Reg(DT.half); v_oldw = Reg(DT.half)
    v_psum = Reg(DT.half); v_nsum = Reg(DT.half)
    # half-swap gather index [64..127, 0..63] = lane XOR 64, so gather(swap, x, idxu) exchanges the
    # two 64-lane halves (even-key reduce <-> odd-key reduce) for the cross-half vmax/add merge.
    idx16 = Reg(DT.int16); arange(idx16, 0)
    swp64 = Reg(DT.int16); swp64 <<= QLANES
    vxor(idx16, idx16, swp64)
    idxu = idx16.reinterpret(DT.uint16)
    neg <<= NEG_LARGE_H
    for u in unroll(UNROLL):
        acc[u] <<= neg
    for rb in range(RB_MAX_H):
        for u in unroll(UNROLL):
            sreg[u] <<= ub_h[(rb * UNROLL + u) * TILE_N]
            acc[u] <<= acc[u].vmax(sreg[u])
    block_max <<= acc[0].vmax(acc[1]); t <<= acc[2].vmax(acc[3]); block_max <<= block_max.vmax(t)
    gather(swap, block_max, idxu)
    block_max <<= block_max.vmax(swap)
    v_pmax <<= ub_rmax[0:1, 0:TILE_N]
    v_nmax <<= block_max.vmax(v_pmax)
    sub(v_oldw, v_pmax, v_nmax); exp(v_oldw, v_oldw)
    for u in unroll(UNROLL):
        acc[u] <<= 0.0
    for rb in range(RB_EXP_H):
        for u in unroll(UNROLL):
            sreg[u] <<= ub_h[(rb * UNROLL + u) * TILE_N]
            sub(preg[u], sreg[u], v_nmax)
            exp(preg[u], preg[u])
            acc[u] <<= acc[u] + preg[u]
            cast(p_h[u], preg[u], cfg_i)
        for u in unroll(UNROLL):
            reg_to_ub_downsample(ub_p[(rb * UNROLL + u) * TILE_N], p_h[u])
    block_sum <<= acc[0] + acc[1]; t <<= acc[2] + acc[3]; block_sum <<= block_sum + t
    gather(swap, block_sum, idxu)
    block_sum <<= block_sum + swap
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)
    v_psum <<= ub_rsum[0:1, 0:TILE_N]
    v_nsum <<= v_psum * v_oldw
    v_nsum <<= v_nsum + block_sum
    ub_rmax[0:1, 0:TILE_N] <<= v_nmax
    ub_rsum[0:1, 0:TILE_N] <<= v_nsum
    ub_old_weight[0:1, 0:TILE_N] <<= v_oldw
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def softmax_half_scale_tail_vf(ub_h: Tensor, ub_rmax: Tensor, ub_rsum: Tensor, ub_p: Tensor, ub_old_weight: Tensor, valid_n: Var):
    s = Reg(DT.half); e = Reg(DT.half); neg = Reg(DT.half); p_hif8 = Reg(DT.hif8)
    block_max = Reg(DT.half); block_sum = Reg(DT.half)
    read_half = Reg(DT.half); unused_half = Reg(DT.half)
    rowmask = MaskReg(DT.half, init_mode=MaskType.ALL)
    p_row_mask = MaskReg(DT.hif8, init_mode=MaskType.LOWHALF)
    cfg_i = CastConfig(round_mode=RoundMode.AWAY_FROM_ZERO, reg_layout=RegLayout.ZERO, name="ph_tail_i")
    v_pmax = Reg(DT.half); v_nmax = Reg(DT.half); v_oldw = Reg(DT.half)
    v_psum = Reg(DT.half); v_nsum = Reg(DT.half)
    block_max <<= NEG_LARGE_H
    neg <<= NEG_LARGE_H
    for n in range(TILE_N):
        cnt_n = Var(QLANES * Min(Max(valid_n - n, 0), 1), dtype=DT.uint32)
        update_mask(rowmask, cnt_n)
        # UNPK reads exactly 64 half elements; recover their contiguous lane order.
        ub_to_reg_unpack(read_half, ub_h[n:n + 1, 0:QLANES])
        deinterleave(s, unused_half, read_half, read_half)
        s <<= s * SOFTMAX_SCALE
        select(s, s, neg, mask=rowmask)
        block_max <<= block_max.vmax(s)
    v_pmax <<= ub_rmax[0:1, 0:QLANES]
    v_nmax <<= block_max.vmax(v_pmax)
    sub(v_oldw, v_pmax, v_nmax); exp(v_oldw, v_oldw)
    block_sum <<= 0.0
    for n in range(TILE_N):
        cnt_n = Var(QLANES * Min(Max(valid_n - n, 0), 1), dtype=DT.uint32)
        update_mask(rowmask, cnt_n)
        # UNPK reads exactly 64 half elements; recover their contiguous lane order.
        ub_to_reg_unpack(read_half, ub_h[n:n + 1, 0:QLANES])
        deinterleave(s, unused_half, read_half, read_half)
        s <<= s * SOFTMAX_SCALE
        select(s, s, neg, mask=rowmask)
        sub(e, s, v_nmax); exp(e, e)
        block_sum <<= block_sum + e
        cast(p_hif8, e, cfg_i)
        # PACK_B16 packs 128 carrier bytes into one 64-byte P row (M10-060).
        reg_to_ub_downsample(ub_p[n * KROW], p_hif8, mask=p_row_mask)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)
    v_psum <<= ub_rsum[0:1, 0:QLANES]
    v_nsum <<= v_psum * v_oldw
    v_nsum <<= v_nsum + block_sum
    ub_rmax[0:1, 0:QLANES] <<= v_nmax
    ub_rsum[0:1, 0:QLANES] <<= v_nsum
    ub_old_weight[0:1, 0:QLANES] <<= v_oldw
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def softmax_half_noscale_tail_vf(ub_h: Tensor, ub_rmax: Tensor, ub_rsum: Tensor, ub_p: Tensor, ub_old_weight: Tensor, valid_n: Var):
    s = Reg(DT.half); e = Reg(DT.half); neg = Reg(DT.half); p_hif8 = Reg(DT.hif8)
    block_max = Reg(DT.half); block_sum = Reg(DT.half)
    read_half = Reg(DT.half); unused_half = Reg(DT.half)
    rowmask = MaskReg(DT.half, init_mode=MaskType.ALL)
    p_row_mask = MaskReg(DT.hif8, init_mode=MaskType.LOWHALF)
    cfg_i = CastConfig(round_mode=RoundMode.AWAY_FROM_ZERO, reg_layout=RegLayout.ZERO, name="ph_tail_i")
    v_pmax = Reg(DT.half); v_nmax = Reg(DT.half); v_oldw = Reg(DT.half)
    v_psum = Reg(DT.half); v_nsum = Reg(DT.half)
    block_max <<= NEG_LARGE_H
    neg <<= NEG_LARGE_H
    for n in range(TILE_N):
        cnt_n = Var(QLANES * Min(Max(valid_n - n, 0), 1), dtype=DT.uint32)
        update_mask(rowmask, cnt_n)
        # UNPK reads exactly 64 half elements; recover their contiguous lane order.
        ub_to_reg_unpack(read_half, ub_h[n:n + 1, 0:QLANES])
        deinterleave(s, unused_half, read_half, read_half)
        select(s, s, neg, mask=rowmask)
        block_max <<= block_max.vmax(s)
    v_pmax <<= ub_rmax[0:1, 0:QLANES]
    v_nmax <<= block_max.vmax(v_pmax)
    sub(v_oldw, v_pmax, v_nmax); exp(v_oldw, v_oldw)
    block_sum <<= 0.0
    for n in range(TILE_N):
        cnt_n = Var(QLANES * Min(Max(valid_n - n, 0), 1), dtype=DT.uint32)
        update_mask(rowmask, cnt_n)
        # UNPK reads exactly 64 half elements; recover their contiguous lane order.
        ub_to_reg_unpack(read_half, ub_h[n:n + 1, 0:QLANES])
        deinterleave(s, unused_half, read_half, read_half)
        select(s, s, neg, mask=rowmask)
        sub(e, s, v_nmax); exp(e, e)
        block_sum <<= block_sum + e
        cast(p_hif8, e, cfg_i)
        # PACK_B16 packs 128 carrier bytes into one 64-byte P row (M10-060).
        reg_to_ub_downsample(ub_p[n * KROW], p_hif8, mask=p_row_mask)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)
    v_psum <<= ub_rsum[0:1, 0:QLANES]
    v_nsum <<= v_psum * v_oldw
    v_nsum <<= v_nsum + block_sum
    ub_rmax[0:1, 0:QLANES] <<= v_nmax
    ub_rsum[0:1, 0:QLANES] <<= v_nsum
    ub_old_weight[0:1, 0:QLANES] <<= v_oldw
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)




@vf()
def accum_pv_fp16ow_vf(ub_accum: Tensor, ub_pv: Tensor, ub_old_weight: Tensor):
    # fp16 old_weight read per-row as a single-value (.single()) broadcast + scalar cast to float
    # (a scalar cast round-trips exactly, unlike a vector half->float cast). acc = acc*ow + pv fused
    # via muldstadd; rows unrolled by UNROLL to overlap the load->cast->FMA->store chains.
    acc = RegList(DT.float, UNROLL * CHUNKS_D); pv = RegList(DT.float, UNROLL * CHUNKS_D)
    ow_h = RegList(DT.half, UNROLL); ow_f = RegList(DT.float, UNROLL)
    cfg_hf = CastConfig(round_mode=RoundMode.NONE, name="ow_h2f")
    for rb in range(ROWS_PER_SB // UNROLL):
        for u in unroll(UNROLL):
            r = rb * UNROLL + u
            ow_h[u] <<= ub_old_weight[0:1, r:r + 1].single()
            cast(ow_f[u], ow_h[u], cfg_hf)
            for c in unroll(CHUNKS_D):
                k = u * CHUNKS_D + c
                acc[k] <<= ub_accum[r:r + 1, c * 64:(c + 1) * 64]
                pv[k] <<= ub_pv[r:r + 1, c * 64:(c + 1) * 64]
                muldstadd(acc[k], ow_f[u], pv[k])
                ub_accum[r:r + 1, c * 64:(c + 1) * 64] <<= acc[k]
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)






@vf()
def final_div_fp16rsum_vf(ub_accum: Tensor, ub_rsum: Tensor, ub_out: Tensor):
    acc = RegList(DT.float, CHUNKS_D); rsum_h = Reg(DT.half); rsum_f = Reg(DT.float)
    cfg_hf = CastConfig(round_mode=RoundMode.NONE, name="rs_h2f")
    for r in range(ROWS_PER_SB):
        rsum_h <<= ub_rsum[0:1, r:r + 1].single()
        cast(rsum_f, rsum_h, cfg_hf)
        acc <<= ub_accum[r:r + 1, 0:D_HEAD]
        acc <<= acc / rsum_f
        ub_out[r:r + 1, 0:D_HEAD] <<= acc
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)






@vf()
def fd_merge_fp16_vf(ub_o0: Tensor, ub_o1: Tensor, ub_m0: Tensor, ub_m1: Tensor,
                     ub_s0: Tensor, ub_s1: Tensor, ub_A: Tensor, ub_den: Tensor, ub_merged: Tensor):
    # ub_m0/m1/s0/s1 are fp16. A half->float cast of a *vector* of distinct per-lane values mis-lays
    # the lanes (only single-value .single() broadcasts round-trip exactly, as in accum_pv_fp16ow_vf),
    # so compute the rescale weight A = exp(m0-m1) and denominator den = s0*A + s1 PER ROW from
    # single-value casts, all math in float (o0/o1 are float PV partials). ub_A/ub_den unused here.
    o0 = RegList(DT.float, CD); o1 = RegList(DT.float, CD)
    m0h = Reg(DT.half); m1h = Reg(DT.half); s0h = Reg(DT.half); s1h = Reg(DT.half)
    m0f = Reg(DT.float); m1f = Reg(DT.float); s0f = Reg(DT.float); s1f = Reg(DT.float)
    Ar = Reg(DT.float); dr = Reg(DT.float)
    cfg_hf = CastConfig(round_mode=RoundMode.NONE, name="merge_h2f")
    for r in range(M_Q):
        m0h <<= ub_m0[0:1, r:r + 1].single(); cast(m0f, m0h, cfg_hf)
        m1h <<= ub_m1[0:1, r:r + 1].single(); cast(m1f, m1h, cfg_hf)
        s0h <<= ub_s0[0:1, r:r + 1].single(); cast(s0f, s0h, cfg_hf)
        s1h <<= ub_s1[0:1, r:r + 1].single(); cast(s1f, s1h, cfg_hf)
        expsub(Ar, m0f, m1f)
        dr <<= s0f * Ar
        dr <<= dr + s1f
        o0 <<= ub_o0[r:r + 1, :]
        o1 <<= ub_o1[r:r + 1, :]
        o0 <<= o0 * Ar
        o0 <<= o0 + o1
        o0 <<= o0 / dr
        ub_merged[r:r + 1, :] <<= o0
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)




# =========================== QK L0C->UB bridge (dual-SINGLE vs single-SPLITN; @func param) ===========================
@func()
def _qk_bridge_dual_single(ub_dst, l0c_src, l0c_scale):
    # TWO SINGLE fixpipe copies: vec sub-block 0 reads L0C query-cols [0:64], sub-block 1 reads
    # [64:128]. Requant (scale!=1) and fp32->fp16 downcast ride the SINGLE deqScalar path.
    l0c_to_ub(ub_dst, l0c_src[0:TILE_N, 0:ROWS_PER_SB],
              M=TILE_N, N=ROWS_PER_SB, N_dst=ROWS_PER_SB, M_src=TILE_N,
              dual_mode=DualMode.SINGLE, sub_block_id=0, scale=l0c_scale)
    l0c_to_ub(ub_dst, l0c_src[0:TILE_N, ROWS_PER_SB:TILE_M],
              M=TILE_N, N=ROWS_PER_SB, N_dst=ROWS_PER_SB, M_src=TILE_N,
              dual_mode=DualMode.SINGLE, sub_block_id=1, scale=l0c_scale)




# =========================== shared FD schedule (@func; VFs/dtype passed as params) ===========================
@func()
def _variant_fd_body(q, k, v, out, B, MQ, N, D, sdt, state_w,
                     init_fn, softmax_fn, softmax_tail_fn, accum_fn, final_fn, merge_fn, l0c_scale,
                     qk_bridge_fn, cv, pv_mutex):
    q_hif8 = q.reinterpret(DT.hif8, name="q_hif8")
    k_hif8 = k.reinterpret(DT.hif8, name="k_hif8")
    v_hif8 = v.reinterpret(DT.hif8, name="v_hif8")

    # cv (QK-fix<->softmax) and pv_mutex (PV-fix<->accum) are passed in so a nocv probe can swap them for
    # no-op stubs (PV_NOCV=1) and measure each variant's cube/vec parallel ceiling. p_mutex (P-chain,
    # MTE3<->MTE1) stays internal -- it is not part of the fixpipe<->vec sync under study.
    p_mutex = VcMutex(1, depth=CACHE, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE1)

    l1q = DBuff(DT.hif8, [TILE_M, D_HEAD], Position.L1)
    l1k = _CacheBuf(DT.hif8, [TILE_N, D_HEAD], Position.L1)
    l1v = _CacheBuf(DT.hif8, [TILE_N, D_HEAD], Position.L1)
    l1p = _CacheBuf(DT.hif8, [TILE_N, TILE_M], Position.L1)
    l0c_qk = DBuff(DT.float, [TILE_M, TILE_M], Position.L0C)
    l0c_pv = DBuff(DT.float, [TILE_M, D_HEAD], Position.L0C)

    ub_score = DBuff(sdt, [TILE_N, ROWS_PER_SB], Position.UB)
    ub_pv = DBuff(DT.float, [ROWS_PER_SB, D_HEAD], Position.UB)
    ub_p = Tensor(DT.hif8, [TILE_N, KROW], Position.UB)
    ub_old_weight = _CacheBuf(sdt, [1, state_w], Position.UB)
    ub_rmax = _CacheBuf(sdt, [1, state_w], Position.UB)
    ub_rsum = _CacheBuf(sdt, [1, state_w], Position.UB)
    # fd-merge scratch for the per-query rescale weight A and denominator: the merge math is float
    # (o0/o1 are float PV accumulators), so A/den must be float. For fp16 variants ub_old_weight is
    # half, so it cannot double as this scratch -- use dedicated float buffers for all variants.
    ub_merge_a = Tensor(DT.float, [1, TILE_M], Position.UB)
    ub_merge_den = Tensor(DT.float, [1, TILE_M], Position.UB)
    ub_accum = Tensor(DT.float, [ROWS_PER_SB, D_HEAD], Position.UB)
    ub_out = Tensor(DT.bfloat16, [ROWS_PER_SB, D_HEAD], Position.UB)

    fd_accum = split_workspace(DT.float, [MAX_FD_WS, TILE_M, D_HEAD], name="fd_accum")
    fd_max = split_workspace(sdt, [MAX_FD_WS, 1, TILE_M], name="fd_max")
    fd_sum = split_workspace(sdt, [MAX_FD_WS, 1, TILE_M], name="fd_sum")

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

    # Tail PV reads only valid loaded V rows; no overlapping fill.

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
                    init_fn(ub_rmax[sm], ub_rsum[sm])

                l1k[step][0:valid_n, 0:D] <<= k_hif8[kv_base + n0:kv_base + n0 + valid_n, 0:D]
                matmul(l0c_qk[step], l1k[step], l1q[qm],
                       m=TILE_N, n=TILE_M, k=D_HEAD, is_init=True)

                cv.lock()
                # QK score L0C->UB. Bridge (dual-SINGLE vs single-SPLITN) is a @func param so the
                # fixpipe cost can be A/B'd without a constant `if` in the body.
                qk_bridge_fn(ub_score[step], l0c_qk[step], l0c_scale)
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
                    accum_fn(ub_accum, ub_pv[lag], ub_old_weight[lag])
                pv_mutex.free()

                if lag_kt == k_tiles - 1:
                    if lag_mt == prefix_mt:
                        if local_rows > 0:
                            fd_accum[tail_ws, row_begin:row_begin + local_rows, 0:D_HEAD] <<= ub_accum[0:local_rows, 0:D_HEAD]
                            fd_max[tail_ws, 0:1, row_begin:row_begin + local_rows] <<= ub_rmax[sm2][0:1, 0:local_rows]
                            fd_sum[tail_ws, 0:1, row_begin:row_begin + local_rows] <<= ub_rsum[sm2][0:1, 0:local_rows]
                    else:
                        final_fn(ub_accum, ub_rsum[sm2], ub_out)
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
                merge_fn(ub_pv[0], ub_pv[1], ub_rmax[0], ub_rmax[1],
                         ub_rsum[0], ub_rsum[1], ub_merge_a, ub_merge_den, ub_pv[0])
                output_cast_vf(ub_pv[0], ub_out)
                out[fd_q_base + fd_m0 + fd_row0:fd_q_base + fd_m0 + fd_row0 + fd_rows, 0:D] <<= (
                    ub_out[0:fd_rows, 0:D]
                )
    return out


# =========================== per-variant param tuples + mutex helpers ===========================
# Each variant differs ONLY by these args to the shared _variant_fd_body; the real and nocv kernels reuse
# the SAME tuple so they can never drift. Order matches the @func params after (q,k,v,out,B,MQ,N,D):
# Tuple fields: sdt, state_w, init_fn, softmax_fn, softmax_tail_fn, accum_fn,
# final_fn, merge_fn, l0c_scale, qk_bridge_fn.
_V1_ARGS = (DT.float, QLANES, init_softmax_state_vf, softmax_t_noscale_vf, softmax_t_noscale_tail_vf,
            accum_pv_vf, final_div_cast_bf16_vf, fd_merge_vf, SOFTMAX_SCALE, _qk_bridge_dual_single)
_V2_ARGS = (DT.half, TILE_N, init_softmax_half_vf, softmax_half_noscale_vf, softmax_half_noscale_tail_vf,
            accum_pv_fp16ow_vf, final_div_fp16rsum_vf, fd_merge_fp16_vf, SOFTMAX_SCALE, _qk_bridge_dual_single)
_V3_ARGS = (DT.float, QLANES, init_softmax_state_vf, softmax_t_scale_vf, softmax_t_scale_tail_vf,
            accum_pv_vf, final_div_cast_bf16_vf, fd_merge_vf, 1.0, _qk_bridge_dual_single)
_V4_ARGS = (DT.half, TILE_N, init_softmax_half_vf, softmax_half_scale_vf, softmax_half_scale_tail_vf,
            accum_pv_fp16ow_vf, final_div_fp16rsum_vf, fd_merge_fp16_vf, 1.0, _qk_bridge_dual_single)
# v6 = v3 but QK L0C->UB via ONE SPLITN instead of the dual-SINGLE bridge (scale=1.0 needs no requant, so
# SPLITN is valid; isolates the double-issue fixpipe cost).
_V6_ARGS = (DT.float, QLANES, init_softmax_state_vf, softmax_t_scale_vf, softmax_t_scale_tail_vf,
            accum_pv_vf, final_div_cast_bf16_vf, fd_merge_vf, 1.0, _qk_bridge_splitn)


def _mk_cv():
    return CvMutex(0, depth=2, src_start_pipe=Pipe.FIX, dst_start_pipe=Pipe.V,
                   src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)


def _mk_pv():
    return CvMutex(2, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)


class _NoOpMutex:
    """PV_NOCV stub: emits NO cube/vec (cv, pv_mutex) sync so the profiler shows the UNsynchronized
    parallel ceiling. Output is INCORRECT by design (fixpipe->vec races) -- read makespan + pipe ratios."""
    def lock(self): pass
    def ready(self): pass
    def wait(self): pass
    def free(self): pass


# =========================== 5 real per-variant kernels (no constant `if` in DSL bodies) ===========================
@kernel()
def qk_softmax_pv_v1_kernel(q: GM[u8, ('TQ', 'D')], k: GM[u8, ('TK', 'D')], v: GM[u8, ('TK', 'D')], out: GM[bf16, ('TQ', 'D')], B: i32, MQ: i32, N: i32, D: i32):
    cv = _mk_cv(); pv_mutex = _mk_pv()
    return _variant_fd_body(q, k, v, out, B, MQ, N, D, *_V1_ARGS, cv, pv_mutex)


@kernel()
def qk_softmax_pv_v2_kernel(q: GM[u8, ('TQ', 'D')], k: GM[u8, ('TK', 'D')], v: GM[u8, ('TK', 'D')], out: GM[bf16, ('TQ', 'D')], B: i32, MQ: i32, N: i32, D: i32):
    cv = _mk_cv(); pv_mutex = _mk_pv()
    return _variant_fd_body(q, k, v, out, B, MQ, N, D, *_V2_ARGS, cv, pv_mutex)


@kernel()
def qk_softmax_pv_v3_kernel(q: GM[u8, ('TQ', 'D')], k: GM[u8, ('TK', 'D')], v: GM[u8, ('TK', 'D')], out: GM[bf16, ('TQ', 'D')], B: i32, MQ: i32, N: i32, D: i32):
    cv = _mk_cv(); pv_mutex = _mk_pv()
    return _variant_fd_body(q, k, v, out, B, MQ, N, D, *_V3_ARGS, cv, pv_mutex)


@kernel()
def qk_softmax_pv_v4_kernel(q: GM[u8, ('TQ', 'D')], k: GM[u8, ('TK', 'D')], v: GM[u8, ('TK', 'D')], out: GM[bf16, ('TQ', 'D')], B: i32, MQ: i32, N: i32, D: i32):
    cv = _mk_cv(); pv_mutex = _mk_pv()
    return _variant_fd_body(q, k, v, out, B, MQ, N, D, *_V4_ARGS, cv, pv_mutex)


@kernel()
def qk_softmax_pv_v3_splitn_kernel(q: GM[u8, ('TQ', 'D')], k: GM[u8, ('TK', 'D')], v: GM[u8, ('TK', 'D')], out: GM[bf16, ('TQ', 'D')], B: i32, MQ: i32, N: i32, D: i32):
    cv = _mk_cv(); pv_mutex = _mk_pv()
    return _variant_fd_body(q, k, v, out, B, MQ, N, D, *_V6_ARGS, cv, pv_mutex)


# =========================== 5 nocv probe kernels (cv + pv_mutex -> _NoOpMutex; select with PV_NOCV=1) =====
# makespan(real) - makespan(nocv) = the cv/pv sync stall for that variant; the pipe-ratio shift shows its
# cube/vec parallel ceiling. INCORRECT output by design -- profiling only.
@kernel()
def qk_softmax_pv_v1_nocv_kernel(q: GM[u8, ('TQ', 'D')], k: GM[u8, ('TK', 'D')], v: GM[u8, ('TK', 'D')], out: GM[bf16, ('TQ', 'D')], B: i32, MQ: i32, N: i32, D: i32):
    cv = _NoOpMutex(); pv_mutex = _NoOpMutex()
    return _variant_fd_body(q, k, v, out, B, MQ, N, D, *_V1_ARGS, cv, pv_mutex)


@kernel()
def qk_softmax_pv_v2_nocv_kernel(q: GM[u8, ('TQ', 'D')], k: GM[u8, ('TK', 'D')], v: GM[u8, ('TK', 'D')], out: GM[bf16, ('TQ', 'D')], B: i32, MQ: i32, N: i32, D: i32):
    cv = _NoOpMutex(); pv_mutex = _NoOpMutex()
    return _variant_fd_body(q, k, v, out, B, MQ, N, D, *_V2_ARGS, cv, pv_mutex)


@kernel()
def qk_softmax_pv_v3_nocv_kernel(q: GM[u8, ('TQ', 'D')], k: GM[u8, ('TK', 'D')], v: GM[u8, ('TK', 'D')], out: GM[bf16, ('TQ', 'D')], B: i32, MQ: i32, N: i32, D: i32):
    cv = _NoOpMutex(); pv_mutex = _NoOpMutex()
    return _variant_fd_body(q, k, v, out, B, MQ, N, D, *_V3_ARGS, cv, pv_mutex)


@kernel()
def qk_softmax_pv_v4_nocv_kernel(q: GM[u8, ('TQ', 'D')], k: GM[u8, ('TK', 'D')], v: GM[u8, ('TK', 'D')], out: GM[bf16, ('TQ', 'D')], B: i32, MQ: i32, N: i32, D: i32):
    cv = _NoOpMutex(); pv_mutex = _NoOpMutex()
    return _variant_fd_body(q, k, v, out, B, MQ, N, D, *_V4_ARGS, cv, pv_mutex)


@kernel()
def qk_softmax_pv_v3_splitn_nocv_kernel(q: GM[u8, ('TQ', 'D')], k: GM[u8, ('TK', 'D')], v: GM[u8, ('TK', 'D')], out: GM[bf16, ('TQ', 'D')], B: i32, MQ: i32, N: i32, D: i32):
    cv = _NoOpMutex(); pv_mutex = _NoOpMutex()
    return _variant_fd_body(q, k, v, out, B, MQ, N, D, *_V6_ARGS, cv, pv_mutex)

# ----------------------------------------------------------------------------------------------------
# v8.py
# V8 all-HiFloat8 PFA with logical384-key groups and FP32 softmax state.
#
# Three physical128-key QK score tiles share one maximum update. Their scaled16
# HiFloat8 P products accumulate in one L0C PV result per group. Three fixed
# score slots and one PV slot preserve the source two-group producer lead.
# The corrected credit ring holds six ordered P0/P1/P2 beats. The source tail
# domain is129..384 keys in the final group. Use the local run.py cases; the
# archived HQ8/S4099 campaign and its timing are recorded only as source history.
# ----------------------------------------------------------------------------------------------------

GROUP_N = 3 * TILE_N
FAST_TAIL_N = 3
# Keep two complete producer groups ahead of the delayed PV consumer.  P and V
# therefore need three physical L1 slots; K remains double-buffered because only
# the immediately following producer group is preloaded. The extra lag hides
# score/PV hand-off bubbles without changing the arithmetic.
GROUP_PRELOAD = 2
GROUP_CACHE = GROUP_PRELOAD + 1
BUFFER_SLOTS = 2
# The producer publishes P0/P1/P2 as ordered beats and runs GROUP_PRELOAD
# groups ahead of the consumer: that many complete groups (three beats each)
# reside in L1 before the consumer frees the first credit, so the credit ring
# holds 3 * GROUP_PRELOAD tokens.  Three credits (one group) deadlocked: the
# producer's second group blocked on its P0 credit before freeing the score
# rings the consumer's QK of that group waits for.
P_EVENT_DEPTH = 3 * GROUP_PRELOAD





@vf()
def group_prepare_vf(ub_score0: Tensor, ub_score1: Tensor, ub_rmax: Tensor,
                     ub_rsum: Tensor, ub_old_weight: Tensor):
    """Update max once for two full score tiles and pre-scale the old denominator."""
    sreg = RegList(DT.float, UNROLL)
    acc = RegList(DT.float, UNROLL)
    neg = Reg(DT.float)
    block_max = Reg(DT.float)
    tmp = Reg(DT.float)
    prev_max = Reg(DT.float)
    next_max = Reg(DT.float)
    old_weight = Reg(DT.float)
    prev_sum = Reg(DT.float)
    next_sum = Reg(DT.float)
    neg <<= NEG_LARGE
    for u in unroll(UNROLL):
        acc[u] <<= neg
    for rb in range(RB_MAX):
        for u in unroll(UNROLL):
            row = rb * UNROLL + u
            sreg[u] <<= ub_score0[row:row + 1, :]
            acc[u] <<= acc[u].vmax(sreg[u])
            sreg[u] <<= ub_score1[row:row + 1, :]
            acc[u] <<= acc[u].vmax(sreg[u])
    block_max <<= acc[0].vmax(acc[1])
    tmp <<= acc[2].vmax(acc[3])
    block_max <<= block_max.vmax(tmp)
    muls(block_max, block_max, SOFTMAX_SCALE)
    prev_max <<= ub_rmax[0:1, 0:QLANES]
    next_max <<= block_max.vmax(prev_max)
    expsub(old_weight, prev_max, next_max)
    prev_sum <<= ub_rsum[0:1, 0:QLANES]
    next_sum <<= prev_sum * old_weight
    ub_rmax[0:1, 0:QLANES] <<= next_max
    ub_rsum[0:1, 0:QLANES] <<= next_sum
    ub_old_weight[0:1, 0:QLANES] <<= old_weight
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def group_prepare_one_vf(ub_score: Tensor, ub_rmax: Tensor, ub_rsum: Tensor,
                         ub_old_weight: Tensor):
    """Update the online state from one full score tile without emitting P."""
    sreg = RegList(DT.float, UNROLL)
    acc = RegList(DT.float, UNROLL)
    neg = Reg(DT.float)
    block_max = Reg(DT.float)
    tmp = Reg(DT.float)
    prev_max = Reg(DT.float)
    next_max = Reg(DT.float)
    old_weight = Reg(DT.float)
    prev_sum = Reg(DT.float)
    next_sum = Reg(DT.float)
    neg <<= NEG_LARGE
    for u in unroll(UNROLL):
        acc[u] <<= neg
    for rb in range(RB_MAX):
        for u in unroll(UNROLL):
            row = rb * UNROLL + u
            sreg[u] <<= ub_score[row:row + 1, :]
            acc[u] <<= acc[u].vmax(sreg[u])
    block_max <<= acc[0].vmax(acc[1])
    tmp <<= acc[2].vmax(acc[3])
    block_max <<= block_max.vmax(tmp)
    muls(block_max, block_max, SOFTMAX_SCALE)
    prev_max <<= ub_rmax[0:1, 0:QLANES]
    next_max <<= block_max.vmax(prev_max)
    expsub(old_weight, prev_max, next_max)
    prev_sum <<= ub_rsum[0:1, 0:QLANES]
    next_sum <<= prev_sum * old_weight
    ub_rmax[0:1, 0:QLANES] <<= next_max
    ub_rsum[0:1, 0:QLANES] <<= next_sum
    ub_old_weight[0:1, 0:QLANES] <<= old_weight
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def neg_fill_rows_v8(ub_score: Tensor, valid_n: Var):
    """NEG the INVALID key rows of a partial group tile, once, so the masked vfs below need no
    per-row mask. A row that reads NEG loses the lane-wise max and exponentiates to exactly 0 -
    what the mask produced - so the arithmetic is unchanged. What goes away is one scalar clamp
    per key row INSIDE the vec scope, for which pl has no narrower spelling than a 64-bit compare
    (D-139 / D-150); on the board that clamp was the whole of fd_modified's tail-shape gap
    (D-156). Idempotent, so a tile refined and then emitted is filled once per call safely."""
    neg = Reg(DT.float)
    neg <<= NEG_LARGE
    for n in range(valid_n, TILE_N):
        ub_score[n:n + 1, 0:QLANES] <<= neg


@vf()
def group_refine_tail_max_vf(ub_score: Tensor, ub_rmax: Tensor, ub_rsum: Tensor,
                             ub_old_weight: Tensor):
    """Refine a two-half group state with one third-half max, its invalid rows NEG-filled."""
    score = Reg(DT.float)
    neg = Reg(DT.float)
    block_max = Reg(DT.float)
    prev_max = Reg(DT.float)
    next_max = Reg(DT.float)
    tail_weight = Reg(DT.float)
    prev_sum = Reg(DT.float)
    next_sum = Reg(DT.float)
    prior_weight = Reg(DT.float)
    combined_weight = Reg(DT.float)
    neg <<= NEG_LARGE
    block_max <<= NEG_LARGE
    for ni in range(TILE_N):
        score <<= ub_score[ni:ni + 1, :]
        muls(score, score, SOFTMAX_SCALE)
        block_max <<= block_max.vmax(score)
    prev_max <<= ub_rmax[0:1, 0:QLANES]
    next_max <<= block_max.vmax(prev_max)
    expsub(tail_weight, prev_max, next_max)
    prev_sum <<= ub_rsum[0:1, 0:QLANES]
    next_sum <<= prev_sum * tail_weight
    prior_weight <<= ub_old_weight[0:1, 0:QLANES]
    combined_weight <<= prior_weight * tail_weight
    ub_rmax[0:1, 0:QLANES] <<= next_max
    ub_rsum[0:1, 0:QLANES] <<= next_sum
    ub_old_weight[0:1, 0:QLANES] <<= combined_weight
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def group_refine_tail3_max_vf(ub_score: Tensor, ub_rmax: Tensor, ub_rsum: Tensor,
                              ub_old_weight: Tensor):
    """Refine the group state from the fixed three-key production tail."""
    score = Reg(DT.float)
    block_max = Reg(DT.float)
    prev_max = Reg(DT.float)
    next_max = Reg(DT.float)
    tail_weight = Reg(DT.float)
    prev_sum = Reg(DT.float)
    next_sum = Reg(DT.float)
    prior_weight = Reg(DT.float)
    combined_weight = Reg(DT.float)
    block_max <<= NEG_LARGE
    for ni in range(FAST_TAIL_N):
        score <<= ub_score[ni:ni + 1, :]
        muls(score, score, SOFTMAX_SCALE)
        block_max <<= block_max.vmax(score)
    prev_max <<= ub_rmax[0:1, 0:QLANES]
    next_max <<= block_max.vmax(prev_max)
    expsub(tail_weight, prev_max, next_max)
    prev_sum <<= ub_rsum[0:1, 0:QLANES]
    next_sum <<= prev_sum * tail_weight
    prior_weight <<= ub_old_weight[0:1, 0:QLANES]
    combined_weight <<= prior_weight * tail_weight
    ub_rmax[0:1, 0:QLANES] <<= next_max
    ub_rsum[0:1, 0:QLANES] <<= next_sum
    ub_old_weight[0:1, 0:QLANES] <<= combined_weight
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def group_emit_vf(ub_score: Tensor, ub_rmax: Tensor, ub_rsum: Tensor, ub_p: Tensor):
    """Emit one full HIF8 P tile against a group-shared max and add its denominator."""
    sreg = RegList(DT.float, UNROLL)
    psum = RegList(DT.float, UNROLL)
    preg = RegList(DT.float, UNROLL)
    group_max = Reg(DT.float)
    exp_max = Reg(DT.float)
    block_sum = Reg(DT.float)
    tmp = Reg(DT.float)
    running_sum = Reg(DT.float)
    next_sum = Reg(DT.float)
    cfg = CastConfig(round_mode=RoundMode.AWAY_FROM_ZERO, reg_layout=RegLayout.ZERO,
                     name="group_p_hif8")
    group_max <<= ub_rmax[0:1, 0:QLANES]
    running_sum <<= ub_rsum[0:1, 0:QLANES]
    adds(exp_max, group_max, -LN16)
    for kk in unroll(KEYS_PER_GRP):
        psum[kk] <<= 0.0
    hh = RegList(DT.hif8, KEYS_PER_GRP)
    aa = Reg(DT.hif8)
    bb = Reg(DT.hif8)
    fin = Reg(DT.hif8)
    dummy = Reg(DT.hif8)
    for rb in range(RB4):
        for u in unroll(UNROLL4):
            for kk in unroll(KEYS_PER_GRP):
                row = (rb * UNROLL4 + u) + kk * SLAB
                sreg[0] <<= ub_score[row:row + 1, :]
                muls(sreg[0], sreg[0], SOFTMAX_SCALE)
                expsub(preg[0], sreg[0], exp_max)
                psum[kk] <<= psum[kk] + preg[0]
                cast(hh[kk], preg[0], cfg)
            deinterleave(aa, dummy, hh[0], hh[1])
            deinterleave(bb, dummy, hh[2], hh[3])
            deinterleave(fin, dummy, aa, bb)
            reg_to_ub(ub_p[(rb * UNROLL4 + u) * NZ_C0], fin, FRAC_STRIDE4)
    block_sum <<= psum[0] + psum[1]
    tmp <<= psum[2] + psum[3]
    block_sum <<= block_sum + tmp
    next_sum <<= running_sum + block_sum
    ub_rsum[0:1, 0:QLANES] <<= next_sum
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def group_emit_tail_vf(ub_score: Tensor, ub_rmax: Tensor, ub_rsum: Tensor,
                       ub_p: Tensor):
    """Emit a HIF8 P tile (invalid rows NEG-filled) against a group-shared max, and add its denominator."""
    score = Reg(DT.float)
    expv = Reg(DT.float)
    neg = Reg(DT.float)
    group_max = Reg(DT.float)
    exp_max = Reg(DT.float)
    block_sum = Reg(DT.float)
    running_sum = Reg(DT.float)
    next_sum = Reg(DT.float)
    p_hif8 = Reg(DT.hif8)
    packed = Reg(DT.hif8)
    gather_index_i8 = Reg(DT.int8)
    arange(gather_index_i8, 0)
    shiftls(gather_index_i8, gather_index_i8, 2)
    gather_index = gather_index_i8.reinterpret(DT.uint8)
    qmask = MaskReg(DT.hif8, init_mode=MaskType.LOWQUAT)
    cfg = CastConfig(round_mode=RoundMode.AWAY_FROM_ZERO, reg_layout=RegLayout.ZERO,
                     name="group_p_hif8_tail")
    neg <<= NEG_LARGE
    group_max <<= ub_rmax[0:1, 0:QLANES]
    running_sum <<= ub_rsum[0:1, 0:QLANES]
    adds(exp_max, group_max, -LN16)
    block_sum <<= 0.0
    slab_stride = 2 * FRAC_STRIDE4 * NZ_C0
    for g in range(KEYS_PER_GRP):
        for m in range(SLAB):
            ni = g * SLAB + m
            score <<= ub_score[ni:ni + 1, :]
            muls(score, score, SOFTMAX_SCALE)
            expsub(expv, score, exp_max)
            block_sum <<= block_sum + expv
            cast(p_hif8, expv, cfg)
            gather(packed, p_hif8, gather_index)
            reg_to_ub(ub_p[g * slab_stride + m * NZ_C0], packed,
                      FRAC_STRIDE4, mask=qmask)
    next_sum <<= running_sum + block_sum
    ub_rsum[0:1, 0:QLANES] <<= next_sum
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def group_emit_tail3_vf(ub_score: Tensor, ub_rmax: Tensor, ub_rsum: Tensor,
                        ub_p: Tensor):
    """Emit only the three P rows consumed by the exact-K production tail."""
    score = Reg(DT.float)
    expv = Reg(DT.float)
    group_max = Reg(DT.float)
    exp_max = Reg(DT.float)
    block_sum = Reg(DT.float)
    running_sum = Reg(DT.float)
    next_sum = Reg(DT.float)
    p_hif8 = Reg(DT.hif8)
    packed = Reg(DT.hif8)
    gather_index_i8 = Reg(DT.int8)
    arange(gather_index_i8, 0)
    shiftls(gather_index_i8, gather_index_i8, 2)
    gather_index = gather_index_i8.reinterpret(DT.uint8)
    qmask = MaskReg(DT.hif8, init_mode=MaskType.LOWQUAT)
    cfg = CastConfig(round_mode=RoundMode.AWAY_FROM_ZERO, reg_layout=RegLayout.ZERO,
                     name="group_p_hif8_tail3")
    group_max <<= ub_rmax[0:1, 0:QLANES]
    running_sum <<= ub_rsum[0:1, 0:QLANES]
    adds(exp_max, group_max, -LN16)
    block_sum <<= 0.0
    for ni in range(FAST_TAIL_N):
        score <<= ub_score[ni:ni + 1, :]
        muls(score, score, SOFTMAX_SCALE)
        expsub(expv, score, exp_max)
        block_sum <<= block_sum + expv
        cast(p_hif8, expv, cfg)
        gather(packed, p_hif8, gather_index)
        reg_to_ub(ub_p[ni * NZ_C0], packed, FRAC_STRIDE4, mask=qmask)
    next_sum <<= running_sum + block_sum
    ub_rsum[0:1, 0:QLANES] <<= next_sum
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def group_softmax_p0_vf(ub_score0: Tensor, ub_score1: Tensor, ub_score2: Tensor, ub_rmax: Tensor,
                        ub_rsum: Tensor, ub_old_weight: Tensor, ub_p0: Tensor):
    """Prepare one full 384-key group and expose its first P third."""
    sreg = RegList(DT.float, UNROLL)
    acc = RegList(DT.float, UNROLL)
    preg = Reg(DT.float)
    neg = Reg(DT.float)
    block_max = Reg(DT.float)
    tmp = Reg(DT.float)
    prev_max = Reg(DT.float)
    next_max = Reg(DT.float)
    old_weight = Reg(DT.float)
    prev_sum = Reg(DT.float)
    next_sum = Reg(DT.float)
    exp_max = Reg(DT.float)
    block_sum = Reg(DT.float)
    cfg = CastConfig(round_mode=RoundMode.AWAY_FROM_ZERO, reg_layout=RegLayout.ZERO,
                     name="group_pair_p_hif8")

    neg <<= NEG_LARGE
    for u in unroll(UNROLL):
        acc[u] <<= neg
    for rb in range(RB_MAX):
        for u in unroll(UNROLL):
            row = rb * UNROLL + u
            sreg[u] <<= ub_score0[row:row + 1, :]
            acc[u] <<= acc[u].vmax(sreg[u])
            sreg[u] <<= ub_score1[row:row + 1, :]
            acc[u] <<= acc[u].vmax(sreg[u])
            sreg[u] <<= ub_score2[row:row + 1, :]
            acc[u] <<= acc[u].vmax(sreg[u])
    block_max <<= acc[0].vmax(acc[1])
    tmp <<= acc[2].vmax(acc[3])
    block_max <<= block_max.vmax(tmp)
    muls(block_max, block_max, SOFTMAX_SCALE)
    prev_max <<= ub_rmax[0:1, 0:QLANES]
    next_max <<= block_max.vmax(prev_max)
    expsub(old_weight, prev_max, next_max)
    prev_sum <<= ub_rsum[0:1, 0:QLANES]
    next_sum <<= prev_sum * old_weight
    adds(exp_max, next_max, -LN16)

    hh = RegList(DT.hif8, KEYS_PER_GRP)
    aa = Reg(DT.hif8)
    bb = Reg(DT.hif8)
    fin = Reg(DT.hif8)
    dummy = Reg(DT.hif8)

    for kk in unroll(KEYS_PER_GRP):
        acc[kk] <<= 0.0
    for rb in range(RB4):
        for u in unroll(UNROLL4):
            for kk in unroll(KEYS_PER_GRP):
                row = (rb * UNROLL4 + u) + kk * SLAB
                sreg[0] <<= ub_score0[row:row + 1, :]
                muls(sreg[0], sreg[0], SOFTMAX_SCALE)
                expsub(preg, sreg[0], exp_max)
                acc[kk] <<= acc[kk] + preg
                cast(hh[kk], preg, cfg)
            deinterleave(aa, dummy, hh[0], hh[1])
            deinterleave(bb, dummy, hh[2], hh[3])
            deinterleave(fin, dummy, aa, bb)
            reg_to_ub(ub_p0[(rb * UNROLL4 + u) * NZ_C0], fin, FRAC_STRIDE4)
    block_sum <<= acc[0] + acc[1]
    tmp <<= acc[2] + acc[3]
    block_sum <<= block_sum + tmp
    next_sum <<= next_sum + block_sum

    ub_rmax[0:1, 0:QLANES] <<= next_max
    ub_rsum[0:1, 0:QLANES] <<= next_sum
    ub_old_weight[0:1, 0:QLANES] <<= old_weight
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def group_softmax_p0_first_vf(ub_score0: Tensor, ub_score1: Tensor, ub_score2: Tensor,
                              ub_rmax: Tensor, ub_rsum: Tensor, ub_p0: Tensor):
    """Prepare the first full group and expose P0 without initialized state."""
    sreg = RegList(DT.float, UNROLL)
    acc = RegList(DT.float, UNROLL)
    preg = Reg(DT.float)
    neg = Reg(DT.float)
    block_max = Reg(DT.float)
    tmp = Reg(DT.float)
    exp_max = Reg(DT.float)
    block_sum = Reg(DT.float)
    cfg = CastConfig(round_mode=RoundMode.AWAY_FROM_ZERO, reg_layout=RegLayout.ZERO,
                     name="group_first_pair_p_hif8")

    neg <<= NEG_LARGE
    for u in unroll(UNROLL):
        acc[u] <<= neg
    for rb in range(RB_MAX):
        for u in unroll(UNROLL):
            row = rb * UNROLL + u
            sreg[u] <<= ub_score0[row:row + 1, :]
            acc[u] <<= acc[u].vmax(sreg[u])
            sreg[u] <<= ub_score1[row:row + 1, :]
            acc[u] <<= acc[u].vmax(sreg[u])
            sreg[u] <<= ub_score2[row:row + 1, :]
            acc[u] <<= acc[u].vmax(sreg[u])
    block_max <<= acc[0].vmax(acc[1])
    tmp <<= acc[2].vmax(acc[3])
    block_max <<= block_max.vmax(tmp)
    muls(block_max, block_max, SOFTMAX_SCALE)
    adds(exp_max, block_max, -LN16)

    hh = RegList(DT.hif8, KEYS_PER_GRP)
    aa = Reg(DT.hif8)
    bb = Reg(DT.hif8)
    fin = Reg(DT.hif8)
    dummy = Reg(DT.hif8)

    for kk in unroll(KEYS_PER_GRP):
        acc[kk] <<= 0.0
    for rb in range(RB4):
        for u in unroll(UNROLL4):
            for kk in unroll(KEYS_PER_GRP):
                row = (rb * UNROLL4 + u) + kk * SLAB
                sreg[0] <<= ub_score0[row:row + 1, :]
                muls(sreg[0], sreg[0], SOFTMAX_SCALE)
                expsub(preg, sreg[0], exp_max)
                acc[kk] <<= acc[kk] + preg
                cast(hh[kk], preg, cfg)
            deinterleave(aa, dummy, hh[0], hh[1])
            deinterleave(bb, dummy, hh[2], hh[3])
            deinterleave(fin, dummy, aa, bb)
            reg_to_ub(ub_p0[(rb * UNROLL4 + u) * NZ_C0], fin, FRAC_STRIDE4)
    block_sum <<= acc[0] + acc[1]
    tmp <<= acc[2] + acc[3]
    block_sum <<= block_sum + tmp

    ub_rmax[0:1, 0:QLANES] <<= block_max
    ub_rsum[0:1, 0:QLANES] <<= block_sum
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@func()
def _qk_bridge_splitn_group(ub_dst, l0c_src):
    l0c_to_ub(ub_dst, l0c_src[0:TILE_N, 0:TILE_M],
              M=TILE_N, N=TILE_M, N_dst=ROWS_PER_SB, M_src=TILE_N,
              dual_mode=DualMode.SPLITN, sub_block_id=0, scale=1.0)


@func()
def _publish_group_p(l1p_dst, ub_p, row_begin, p_mutex):
    p_mutex.lock()
    for g in unroll(KEYS_PER_GRP):
        l1p_dst[g * SLAB:(g + 1) * SLAB, row_begin:row_begin + ROWS_PER_SB] <<= (
            ub_p.nz()[0:SLAB, g * ROWS_PER_SB:(g + 1) * ROWS_PER_SB]
        )
    p_mutex.ready()


@func()
def _publish_group_p3(l1p_dst, ub_p, row_begin, p_mutex):
    """Publish only the three NZ rows consumed by the exact-K tail PV."""
    p_mutex.lock()
    l1p_dst[0:FAST_TAIL_N, row_begin:row_begin + ROWS_PER_SB] <<= (
        ub_p.nz()[0:FAST_TAIL_N, 0:ROWS_PER_SB]
    )
    p_mutex.ready()


@func()
def _publish_group_pair_split_ready(l1p0_dst, ub_p0, l1p1_dst, ub_p1, row_begin, p_mutex):
    """Publish each physical P half as soon as its own MTE3 stores retire."""
    p_mutex.lock()
    for g in unroll(KEYS_PER_GRP):
        l1p0_dst[g * SLAB:(g + 1) * SLAB, row_begin:row_begin + ROWS_PER_SB] <<= (
            ub_p0.nz()[0:SLAB, g * ROWS_PER_SB:(g + 1) * ROWS_PER_SB]
        )
    p_mutex.ready()

    p_mutex.lock()
    for g in unroll(KEYS_PER_GRP):
        l1p1_dst[g * SLAB:(g + 1) * SLAB, row_begin:row_begin + ROWS_PER_SB] <<= (
            ub_p1.nz()[0:SLAB, g * ROWS_PER_SB:(g + 1) * ROWS_PER_SB]
        )
    p_mutex.ready()


@func()
def _alloc_group(q, k, v):
    q_hif8 = q.reinterpret(DT.hif8, name="q_hif8")
    k_hif8 = k.reinterpret(DT.hif8, name="k_hif8")
    v_hif8 = v.reinterpret(DT.hif8, name="v_hif8")
    l1q = DBuff(DT.hif8, [TILE_M, D_HEAD], Position.L1)
    # Keep the head K0 independently ready; only K1/K2 share a slot.
    # Aligned NZ column bands preserve each operand byte layout.
    l1k0 = DBuff(DT.hif8, [TILE_N, D_HEAD], Position.L1)
    l1k = DBuff(DT.hif8, [TILE_N, 2 * D_HEAD], Position.L1)
    # Same slot index across all three slabs; NZ column bands keep
    # each operand contiguous while sharing one physical slot identity.
    l1v = TBuff(DT.hif8, [TILE_N, 3 * D_HEAD], Position.L1)
    l1p0 = TBuff(DT.hif8, [TILE_N, TILE_M], Position.L1)
    l1p1 = TBuff(DT.hif8, [TILE_N, TILE_M], Position.L1)
    l1p2 = TBuff(DT.hif8, [TILE_N, TILE_M], Position.L1)
    l0c_qk = TBuff(DT.float, [TILE_M, TILE_M], Position.L0C)
    l0c_pv = Tensor(DT.float, [TILE_M, D_HEAD], Position.L0C)
    ub_score = TBuff(DT.float, [TILE_N, ROWS_PER_SB], Position.UB)
    ub_pv = DBuff(DT.float, [ROWS_PER_SB, D_HEAD], Position.UB)
    ub_p0 = DBuff(DT.hif8, [FRAC_STRIDE4, KEYS_PER_GRP * ROWS_PER_SB], Position.UB)
    ub_p1 = DBuff(DT.hif8, [FRAC_STRIDE4, KEYS_PER_GRP * ROWS_PER_SB], Position.UB)
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
    return (q_hif8, k_hif8, v_hif8, l1q, l1k0, l1k, l1v,
            l1p0, l1p1, l1p2, l0c_qk, l0c_pv, ub_score, ub_pv, ub_p0, ub_p1,
            ub_old_weight, ub_rmax, ub_rsum, ub_merge_a, ub_merge_den, ub_accum, ub_out,
            fd_accum, fd_max, fd_sum)


@func()
def _group384_body(q, k, v, out, B, MQ, N, D, cv0, cv1, pv_mutex, p_mutex):
    (q_hif8, k_hif8, v_hif8, l1q, l1k0, l1k, l1v,
     l1p0, l1p1, l1p2, l0c_qk, l0c_pv, ub_score, ub_pv, ub_p0, ub_p1,
     ub_old_weight, ub_rmax, ub_rsum, ub_merge_a, ub_merge_den, ub_accum, ub_out,
     fd_accum, fd_max, fd_sum) = _alloc_group(q, k, v)

    core = Var(GetCubeIdx())
    core_count = Var(GetCubeNum())
    vec = Var(GetVecIdx())
    row_begin = Var(GetSubBlockIdx() * ROWS_PER_SB)
    tiles_m_per_b = CeilDiv(MQ, TILE_M)
    k_groups = CeilDiv(N, GROUP_N)
    # Only the last group is partial. Specialize its K extents from N
    # instead of sizing L0 views from the loop-carried group index.
    tail_keys = (N - 1) % GROUP_N + 1
    tail_k0 = Min(TILE_N, tail_keys)
    tail_k1 = Max(1, Min(TILE_N, tail_keys - TILE_N))
    tail_k2 = Max(1, Min(TILE_N, tail_keys - 2 * TILE_N))
    total_tasks = Var(B * tiles_m_per_b * k_groups)
    tasks_floor = Var(total_tasks // core_count)
    # Avoid Var modulo here: the side-split trace exposes an inactive-side
    # GetCubeNum placeholder of zero, while the already-supported division is
    # normalized by lowering.  The algebraic remainder has identical runtime value.
    tasks_extra = Var(total_tasks - tasks_floor * core_count)
    flat_start = Var(core * tasks_floor + Min(core, tasks_extra))
    flat_end = Var((core + 1) * tasks_floor + Min(core + 1, tasks_extra))
    my_tasks = Var(Max(flat_end - flat_start, 0))

    tail_ws = Var(-1, DT.int)
    head_ws = Var(-1, DT.int)
    split_count = Var(0, DT.int)
    for boundary_core in range(1, MAX_CORE_COUNT):
        if core_count > boundary_core:
            boundary = Var(boundary_core * tasks_floor + Min(boundary_core, tasks_extra))
            if boundary % k_groups != 0:
                if core == boundary_core:
                    tail_ws <<= split_count * 2 + 1
                if core + 1 == boundary_core:
                    head_ws <<= split_count * 2
                split_count += 1

    last_group_valid = Var(N - (k_groups - 1) * GROUP_N)
    # Tail K score rows are explicitly masked before use; PV reads only valid V rows.
    # Do not overlap a whole-L1 fill with the valid-prefix GM load (D-214).
    # L0C reuse (mmad -> FixP result, FixP -> next mmad on the slot) and the UB
    # accumulator / output reuse across the boundary stores are autosync's: the
    # hand-written per-slot events of the old kernel set tokens on branches no
    # wait matched (its `preset=True` relied on the old framework's destructor).
    with auto_sync():
        prefix_mt = Var(-1, DT.int)
        suffix_mt = Var(-1, DT.int)
        if my_tasks > 0:
            start_group = Var(flat_start % k_groups)
            if start_group != 0:
                prefix_mt <<= flat_start // k_groups
            end_group = Var(flat_end % k_groups)
            if end_group != 0:
                suffix_mt <<= (flat_end - 1) // k_groups

        # Seed producer/consumer coordinates once.  The hot loop advances them
        # with cheap increments and wrap tests instead of repeating integer
        # division/modulo for both sides of every group beat.
        p_start_mt = Var(flat_start // k_groups)
        p_kg = Var(flat_start - p_start_mt * k_groups)
        p_qm = Var(p_start_mt % 2)
        p_sm = Var(p_start_mt % GROUP_CACHE)
        p_batch = Var(p_start_mt // tiles_m_per_b)
        p_local_mt = Var(p_start_mt - p_batch * tiles_m_per_b)
        p_q_base = Var(p_batch * MQ)
        p_kv_base = Var(p_batch * N)
        p_m0 = Var(p_local_mt * TILE_M)
        p_need_q = Var(1, DT.int)

        c_lag = Var(0, DT.int)
        c_mt = Var(p_start_mt)
        c_kg = Var(p_kg)
        c_sm = Var(p_sm)
        c_batch = Var(p_batch)
        c_local_mt = Var(p_local_mt)
        c_q_base = Var(p_q_base)
        c_kv_base = Var(p_kv_base)
        c_m0 = Var(p_m0)

        # V preload coordinates advance independently from p_*.  V[c_lag] is
        # consumed before its physical slot is refilled, then the next V group
        # is queued at the bottom of the same iteration.
        vl_lag = Var(0, DT.int)
        vl_kg = Var(p_kg)
        vl_local_mt = Var(p_local_mt)
        vl_kv_base = Var(p_kv_base)

        # Prime K for the first producer beat. Every later K group is loaded at
        # the end of its predecessor beat into the opposite DBuff slot.
        if my_tasks > 0:
            pk_n0 = Var(p_kg * GROUP_N)
            pk_valid_n0 = Min(TILE_N, N - pk_n0)
            pk_n1 = Var(pk_n0 + TILE_N)
            pk_valid_n1 = Min(TILE_N, Max(N - pk_n1, 0))
            pk_n2 = Var(pk_n1 + TILE_N)
            pk_valid_n2 = Min(TILE_N, Max(N - pk_n2, 0))
            l1k0[0][0:pk_valid_n0, 0:D] <<= (
                k_hif8[p_kv_base + pk_n0:p_kv_base + pk_n0 + pk_valid_n0, 0:D]
            )
            if pk_valid_n1 > 0:
                l1k[0][:, 0 * D_HEAD:1 * D_HEAD][0:pk_valid_n1, 0:D] <<= (
                    k_hif8[p_kv_base + pk_n1:p_kv_base + pk_n1 + pk_valid_n1, 0:D]
                )
            if pk_valid_n2 > 0:
                l1k[0][:, 1 * D_HEAD:2 * D_HEAD][0:pk_valid_n2, 0:D] <<= (
                    k_hif8[p_kv_base + pk_n2:p_kv_base + pk_n2 + pk_valid_n2, 0:D]
                )

        for step in range(0, my_tasks + GROUP_PRELOAD):
            if step < my_tasks:
                p_n0 = Var(p_kg * GROUP_N)
                valid_m = Min(TILE_M, MQ - p_m0)
                valid_n0 = Min(TILE_N, N - p_n0)
                n1 = Var(p_n0 + TILE_N)
                valid_n1 = Min(TILE_N, Max(N - n1, 0))
                n2 = Var(n1 + TILE_N)
                valid_n2 = Min(TILE_N, Max(N - n2, 0))

                if p_need_q > 0:
                    l1q[p_qm][0:valid_m, 0:D] <<= (
                        q_hif8[p_q_base + p_m0:p_q_base + p_m0 + valid_m, 0:D]
                    )
                    # Only a full three-half group has a dedicated first-group
                    # VF.  Every one/two-half tail starts from explicit state.
                    if valid_n2 != TILE_N:
                        init_softmax_state_vf(ub_rmax[p_sm], ub_rsum[p_sm])

                matmul(l0c_qk[0], l1k0[step], l1q[p_qm],
                       m=TILE_N, n=TILE_M, k=D_HEAD, is_init=True)
                cv0.lock()
                _qk_bridge_splitn_group(ub_score[0], l0c_qk[0])
                cv0.ready()

                matmul(l0c_qk[1], l1k[step][:, 0 * D_HEAD:1 * D_HEAD], l1q[p_qm],
                       m=TILE_N, n=TILE_M, k=D_HEAD, is_init=True)
                cv1.lock()
                _qk_bridge_splitn_group(ub_score[1], l0c_qk[1])
                cv1.ready()

                # QK2 owns a third fixed score slot. Even a 128/<128/0 tail
                # executes this zero-K tile: cv0 is a depth-two score0/score2
                # ring, so skipping the second beat would phase-shift the next
                # query tile onto the wrong physical event slot. The result is
                # ignored by softmax and PV when valid_n2 is zero.
                matmul(l0c_qk[2], l1k[step][:, 1 * D_HEAD:2 * D_HEAD], l1q[p_qm],
                       m=TILE_N, n=TILE_M, k=D_HEAD, is_init=True)
                cv0.lock()
                _qk_bridge_splitn_group(ub_score[2], l0c_qk[2])
                cv0.ready()

                cv0.wait()
                cv1.wait()
                # cv0 is a depth-2 FIFO: its first token owns score0 and its
                # second token owns score2.
                cv0.wait()
                if valid_n2 == TILE_N:
                    if p_need_q > 0:
                        group_softmax_p0_first_vf(
                            ub_score[0], ub_score[1], ub_score[2],
                            ub_rmax[p_sm], ub_rsum[p_sm], ub_p0[step]
                        )
                    else:
                        group_softmax_p0_vf(
                            ub_score[0], ub_score[1], ub_score[2],
                            ub_rmax[p_sm], ub_rsum[p_sm],
                            ub_old_weight[step], ub_p0[step]
                        )
                elif valid_n2 > 0:
                    group_prepare_vf(
                        ub_score[0], ub_score[1], ub_rmax[p_sm],
                        ub_rsum[p_sm], ub_old_weight[step]
                    )
                    if valid_n2 == FAST_TAIL_N:
                        group_refine_tail3_max_vf(
                            ub_score[2], ub_rmax[p_sm], ub_rsum[p_sm],
                            ub_old_weight[step]
                        )
                    else:
                        neg_fill_rows_v8(ub_score[2], valid_n2)
                        group_refine_tail_max_vf(
                            ub_score[2], ub_rmax[p_sm], ub_rsum[p_sm],
                            ub_old_weight[step]
                        )
                    group_emit_vf(
                        ub_score[0], ub_rmax[p_sm], ub_rsum[p_sm], ub_p0[step]
                    )
                else:
                    group_prepare_one_vf(
                        ub_score[0], ub_rmax[p_sm], ub_rsum[p_sm],
                        ub_old_weight[step]
                    )
                    neg_fill_rows_v8(ub_score[1], valid_n1)
                    group_refine_tail_max_vf(
                        ub_score[1], ub_rmax[p_sm], ub_rsum[p_sm],
                        ub_old_weight[step]
                    )
                    group_emit_vf(
                        ub_score[0], ub_rmax[p_sm], ub_rsum[p_sm], ub_p0[step]
                    )
                cv0.free()
                _publish_group_p(l1p0[step], ub_p0[step], row_begin, p_mutex)

                # Each published P beat can enter Cube while Vector emits the
                # next beat against the shared group max.
                if valid_n1 == TILE_N:
                    group_emit_vf(
                        ub_score[1], ub_rmax[p_sm], ub_rsum[p_sm], ub_p1[step]
                    )
                else:
                    neg_fill_rows_v8(ub_score[1], valid_n1)
                    group_emit_tail_vf(
                        ub_score[1], ub_rmax[p_sm], ub_rsum[p_sm],
                        ub_p1[step]
                    )
                cv1.free()
                _publish_group_p(l1p1[step], ub_p1[step], row_begin, p_mutex)
                # P0's UB tile is retired after its MTE3 publish; reuse it for
                # P2 while retaining a distinct L1 P2 operand for Cube.
                if valid_n2 > 0:
                    if valid_n2 == TILE_N:
                        group_emit_vf(
                            ub_score[2], ub_rmax[p_sm], ub_rsum[p_sm], ub_p0[step]
                        )
                    elif valid_n2 == FAST_TAIL_N:
                        group_emit_tail3_vf(
                            ub_score[2], ub_rmax[p_sm], ub_rsum[p_sm], ub_p0[step]
                        )
                    else:
                        neg_fill_rows_v8(ub_score[2], valid_n2)
                        group_emit_tail_vf(
                            ub_score[2], ub_rmax[p_sm], ub_rsum[p_sm],
                            ub_p0[step]
                        )
                    if valid_n2 == FAST_TAIL_N:
                        _publish_group_p3(l1p2[step], ub_p0[step], row_begin, p_mutex)
                    else:
                        _publish_group_p(l1p2[step], ub_p0[step], row_begin, p_mutex)
                else:
                    # p_mutex has six credits for ordered P0/P1/P2 beats. Publish a
                    # phase-only three-row beat so the next logical group keeps
                    # the same physical credit mapping; consumer PV skips it.
                    _publish_group_p3(l1p2[step], ub_p0[step], row_begin, p_mutex)
                cv0.free()

                p_need_q <<= 0
                p_kg += 1
                if p_kg == k_groups:
                    p_kg <<= 0
                    p_need_q <<= 1
                    p_qm <<= 1 - p_qm
                    p_sm += 1
                    if p_sm == GROUP_CACHE:
                        p_sm <<= 0
                    p_local_mt += 1
                    p_m0 += TILE_M
                    if p_local_mt == tiles_m_per_b:
                        p_local_mt <<= 0
                        p_m0 <<= 0
                        p_batch += 1
                        p_q_base += MQ
                        p_kv_base += N

                # Queue K[next] before V[current]. The next producer can start
                # from K0 as soon as its head transfer is ready, while the rest
                # of this MTE2 sequence overlaps the current consumer's PV work.
                if step + 1 < my_tasks:
                    nk_n0 = Var(p_kg * GROUP_N)
                    nk_valid_n0 = Min(TILE_N, N - nk_n0)
                    nk_n1 = Var(nk_n0 + TILE_N)
                    nk_valid_n1 = Min(TILE_N, Max(N - nk_n1, 0))
                    nk_n2 = Var(nk_n1 + TILE_N)
                    nk_valid_n2 = Min(TILE_N, Max(N - nk_n2, 0))
                    l1k0[step + 1][0:nk_valid_n0, 0:D] <<= (
                        k_hif8[p_kv_base + nk_n0:p_kv_base + nk_n0 + nk_valid_n0, 0:D]
                    )
                    if nk_valid_n1 > 0:
                        l1k[step + 1][:, 0 * D_HEAD:1 * D_HEAD][0:nk_valid_n1, 0:D] <<= (
                            k_hif8[p_kv_base + nk_n1:p_kv_base + nk_n1 + nk_valid_n1, 0:D]
                        )
                    if nk_valid_n2 > 0:
                        l1k[step + 1][:, 1 * D_HEAD:2 * D_HEAD][0:nk_valid_n2, 0:D] <<= (
                            k_hif8[p_kv_base + nk_n2:p_kv_base + nk_n2 + nk_valid_n2, 0:D]
                        )

            if step >= GROUP_PRELOAD:
                c_n0 = Var(c_kg * GROUP_N)
                lag_valid_m = Min(TILE_M, MQ - c_m0)
                local_rows = Min(ROWS_PER_SB, Max(lag_valid_m - row_begin, 0))
                v_rows0 = Min(TILE_N, N - c_n0)
                lag_n1 = Var(c_n0 + TILE_N)
                v_rows1 = Min(TILE_N, Max(N - lag_n1, 0))
                lag_n2 = Var(lag_n1 + TILE_N)
                v_rows2 = Min(TILE_N, Max(N - lag_n2, 0))

                # Only one PV result is emitted per logical group. Its previous
                # FixP has the intervening QK0/QK1/QK2 work in which to retire.
                p_mutex.wait()
                matmul(l0c_pv,
                       l1p0[c_lag].T, l1v[c_lag][:, 0 * D_HEAD:1 * D_HEAD].T,
                       m=TILE_M, n=D_HEAD, k=TILE_N, is_init=True)
                p_mutex.free()
                if v_rows1 > 0:
                    p_mutex.wait()
                    if v_rows1 == TILE_N:
                        matmul(l0c_pv,
                               l1p1[c_lag].T, l1v[c_lag][:, 1 * D_HEAD:2 * D_HEAD].T,
                               m=TILE_M, n=D_HEAD, k=TILE_N, is_init=False)
                    else:
                        matmul(l0c_pv,
                               l1p1[c_lag].T, l1v[c_lag][:, 1 * D_HEAD:2 * D_HEAD].T,
                               m=TILE_M, n=D_HEAD, k=tail_k1, is_init=False)
                    p_mutex.free()
                p_mutex.wait()
                if v_rows2 > 0:
                    if v_rows2 == TILE_N:
                        matmul(l0c_pv,
                               l1p2[c_lag].T, l1v[c_lag][:, 2 * D_HEAD:3 * D_HEAD].T,
                               m=TILE_M, n=D_HEAD, k=TILE_N, is_init=False)
                    else:
                        # Partial P/V accumulate paths must not read padded key
                        # rows: a full-K accumulate can introduce NaNs even when
                        # both L1 operands were explicitly zero-filled.
                        matmul(l0c_pv,
                               l1p2[c_lag].T, l1v[c_lag][:, 2 * D_HEAD:3 * D_HEAD].T,
                               m=TILE_M, n=D_HEAD, k=tail_k2, is_init=False)
                p_mutex.free()

                pv_mutex.lock()
                l0c_to_ub(ub_pv[c_lag], l0c_pv,
                          M=TILE_M, N=D_HEAD, N_dst=D_HEAD, M_src=TILE_M,
                          dual_mode=DualMode.SPLITM, sub_block_id=0)
                pv_mutex.ready()
                pv_mutex.wait()

                if c_kg == 0:
                    # The first PV overwrites every accumulator element.
                    accum_pv_first_vf(ub_accum, ub_pv[c_lag])
                elif c_lag == 0:
                    accum_pv_first_vf(ub_accum, ub_pv[c_lag])
                else:
                    accum_pv_vf(ub_accum, ub_pv[c_lag], ub_old_weight[c_lag])
                pv_mutex.free()

                if c_kg == k_groups - 1:
                    if c_mt == prefix_mt:
                        if local_rows > 0:
                            fd_accum[tail_ws, row_begin:row_begin + local_rows, 0:D_HEAD] <<= (
                                ub_accum[0:local_rows, 0:D_HEAD]
                            )
                            fd_max[tail_ws, 0:1, row_begin:row_begin + local_rows] <<= (
                                ub_rmax[c_sm][0:1, 0:local_rows]
                            )
                            fd_sum[tail_ws, 0:1, row_begin:row_begin + local_rows] <<= (
                                ub_rsum[c_sm][0:1, 0:local_rows]
                            )
                    else:
                        final_div_cast_bf16_vf(ub_accum, ub_rsum[c_sm], ub_out)
                        if local_rows > 0:
                            out[c_q_base + c_m0 + row_begin:
                                c_q_base + c_m0 + row_begin + local_rows, 0:D] <<= (
                                ub_out[0:local_rows, 0:D]
                            )

                if c_lag + 1 == my_tasks:
                    if c_kg != k_groups - 1:
                        if c_mt == suffix_mt:
                            if local_rows > 0:
                                fd_accum[head_ws, row_begin:row_begin + local_rows, 0:D_HEAD] <<= (
                                    ub_accum[0:local_rows, 0:D_HEAD]
                                )
                                fd_max[head_ws, 0:1, row_begin:row_begin + local_rows] <<= (
                                    ub_rmax[c_sm][0:1, 0:local_rows]
                                )
                                fd_sum[head_ws, 0:1, row_begin:row_begin + local_rows] <<= (
                                    ub_rsum[c_sm][0:1, 0:local_rows]
                                )

                c_lag += 1
                c_kg += 1
                if c_kg == k_groups:
                    c_kg <<= 0
                    c_mt += 1
                    c_sm += 1
                    if c_sm == GROUP_CACHE:
                        c_sm <<= 0
                    c_local_mt += 1
                    c_m0 += TILE_M
                    if c_local_mt == tiles_m_per_b:
                        c_local_mt <<= 0
                        c_m0 <<= 0
                        c_batch += 1
                        c_q_base += MQ
                        c_kv_base += N

            # Refill V only after the consumer has retired the same modulo-3
            # slot.  The first two iterations prime V0/V1; every steady
            # iteration consumes one old group and then queues one future group.
            if vl_lag < my_tasks:
                vl_n0 = Var(vl_kg * GROUP_N)
                vl_rows0 = Min(TILE_N, N - vl_n0)
                vl_n1 = Var(vl_n0 + TILE_N)
                vl_rows1 = Min(TILE_N, Max(N - vl_n1, 0))
                vl_n2 = Var(vl_n1 + TILE_N)
                vl_rows2 = Min(TILE_N, Max(N - vl_n2, 0))
                l1v[vl_lag][:, 0 * D_HEAD:1 * D_HEAD][0:vl_rows0, 0:D] <<= (
                    v_hif8[vl_kv_base + vl_n0:vl_kv_base + vl_n0 + vl_rows0, 0:D]
                )
                if vl_rows1 > 0:
                    l1v[vl_lag][:, 1 * D_HEAD:2 * D_HEAD][0:vl_rows1, 0:D] <<= (
                        v_hif8[vl_kv_base + vl_n1:vl_kv_base + vl_n1 + vl_rows1, 0:D]
                    )
                if vl_rows2 > 0:
                    l1v[vl_lag][:, 2 * D_HEAD:3 * D_HEAD][0:vl_rows2, 0:D] <<= (
                        v_hif8[vl_kv_base + vl_n2:vl_kv_base + vl_n2 + vl_rows2, 0:D]
                    )
                vl_lag += 1
                vl_kg += 1
                if vl_kg == k_groups:
                    vl_kg <<= 0
                    vl_local_mt += 1
                    if vl_local_mt == tiles_m_per_b:
                        vl_local_mt <<= 0
                        vl_kv_base += N

        allvec_ready(7, Pipe.MTE3)

    allvec_wait(7, Pipe.MTE2)

    fd_task = Var(vec // 2)
    fd_row0 = Var((vec % 2) * ROWS_PER_SB)
    fd_found = Var(0, DT.int)
    fd_mt = Var(0, DT.int)
    scan_rank = Var(0, DT.int)
    for boundary_core in range(1, MAX_CORE_COUNT):
        if core_count > boundary_core:
            boundary = Var(boundary_core * tasks_floor + Min(boundary_core, tasks_extra))
            if boundary % k_groups != 0:
                if scan_rank == fd_task:
                    fd_found <<= 1
                    fd_mt <<= boundary // k_groups
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
def v8_allhif8_kernel(q: GM[u8, ('TQ', 'D')], k: GM[u8, ('TK', 'D')], v: GM[u8, ('TK', 'D')],
                      out: GM[bf16, ('TQ', 'D')], B: i32, MQ: i32, N: i32, D: i32):
    cv0 = CvMutex(0, depth=2, src_start_pipe=Pipe.FIX, dst_start_pipe=Pipe.V,
                  src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    cv1 = CvMutex(1, depth=1, src_start_pipe=Pipe.FIX, dst_start_pipe=Pipe.V,
                  src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    pv_mutex = CvMutex(2, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    p_mutex = VcMutex(3, depth=P_EVENT_DEPTH, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE1)
    return _group384_body(q, k, v, out, B, MQ, N, D, cv0, cv1, pv_mutex, p_mutex)

# ----------------------------------------------------------------------------------------------------
# v9.py
# V9 all-HiFloat8 PFA with FP16 score exports and online state.
#
# Two FP32-to-FP16 SINGLE exports use the FixP19 scale and preserve adjacent-key
# pair reductions. Two score and two PV L0C slots support the source two-group
# producer lead; QK0/QK2 share the first score slot. Unscaled HiFloat8 P has a
# separate UB buffer for each beat, then direct NZ publication with stride33.
# Every positive final-group length1..384 retires three P0/P1/P2 beats. The
# phase-only P2 path uses the same existing explicit event pair as real P2.
# Use run.py and the local source ledger for current cases and precision evidence.
# The archived HQ40/S9360 campaign is historical and does not select an M10 winner.
# ----------------------------------------------------------------------------------------------------

PAIR_ROWS = TILE_N // 2
PAIR_RB = PAIR_ROWS // UNROLL

PAIR_SLAB_ROWS = SLAB // 2
PAIR_SLAB_RB = PAIR_SLAB_ROWS // UNROLL
SLAB_STRIDE4 = 2 * FRAC_STRIDE4 * NZ_C0




@vf()
def group_half_softmax_p0_first_vf(ub_score0: Tensor, ub_score1: Tensor,
                                   ub_score2: Tensor, ub_rmax: Tensor,
                                   ub_rsum: Tensor, ub_p0: Tensor):
    """Initialize a full logical group and emit NZ-ready P0 in FP16."""
    sreg = RegList(DT.half, UNROLL)
    acc = RegList(DT.half, UNROLL)
    preg = RegList(DT.half, UNROLL)
    p_h = RegList(DT.hif8, UNROLL)
    neg = Reg(DT.half)
    block_max = Reg(DT.half)
    block_sum = Reg(DT.half)
    tmp = Reg(DT.half)
    swap = Reg(DT.half)
    cfg = CastConfig(round_mode=RoundMode.AWAY_FROM_ZERO,
                     reg_layout=RegLayout.ZERO, name="gfp16_first_p")
    halfmask = MaskReg(DT.hif8, init_mode=MaskType.LOWHALF)
    pair01 = Reg(DT.hif8)
    pair23 = Reg(DT.hif8)
    deint_dummy = Reg(DT.hif8)
    gathered01 = Reg(DT.hif8)
    gathered23 = Reg(DT.hif8)
    packed_even = Reg(DT.hif8)
    lane_i8 = Reg(DT.int8)
    arange(lane_i8, 0)
    low6_mask = Reg(DT.int8)
    low6_mask <<= 63
    bit6_mask = Reg(DT.int8)
    bit6_mask <<= 64
    low6 = Reg(DT.int8)
    vand(low6, lane_i8, low6_mask)
    bit6 = Reg(DT.int8)
    vand(bit6, lane_i8, bit6_mask)
    shiftls(bit6, bit6, 1)
    even_i8 = Reg(DT.int8)
    vor(even_i8, low6, bit6)
    even_idx = even_i8.reinterpret(DT.uint8)
    odd_i8 = Reg(DT.int8)
    vor(odd_i8, even_i8, bit6_mask)
    odd_idx = odd_i8.reinterpret(DT.uint8)
    idx16 = Reg(DT.int16)
    arange(idx16, 0)
    swp64 = Reg(DT.int16)
    swp64 <<= QLANES
    vxor(idx16, idx16, swp64)
    idxu = idx16.reinterpret(DT.uint16)

    neg <<= NEG_LARGE_H
    for u in unroll(UNROLL):
        acc[u] <<= neg
    for rb in range(PAIR_RB):
        for u in unroll(UNROLL):
            row = rb * UNROLL + u
            sreg[u] <<= ub_score0[row * TILE_N]
            acc[u] <<= acc[u].vmax(sreg[u])
            sreg[u] <<= ub_score1[row * TILE_N]
            acc[u] <<= acc[u].vmax(sreg[u])
            sreg[u] <<= ub_score2[row * TILE_N]
            acc[u] <<= acc[u].vmax(sreg[u])
    block_max <<= acc[0].vmax(acc[1])
    tmp <<= acc[2].vmax(acc[3])
    block_max <<= block_max.vmax(tmp)
    gather(swap, block_max, idxu)
    block_max <<= block_max.vmax(swap)

    for u in unroll(UNROLL):
        acc[u] <<= 0.0
    for pair_row in range(PAIR_SLAB_ROWS):
        for u in unroll(UNROLL):
            row = pair_row + u * PAIR_SLAB_ROWS
            sreg[u] <<= ub_score0[row * TILE_N]
            sub(preg[u], sreg[u], block_max)
            exp(preg[u], preg[u])
            acc[u] <<= acc[u] + preg[u]
            cast(p_h[u], preg[u], cfg)
        deinterleave(pair01, deint_dummy, p_h[0], p_h[1])
        deinterleave(pair23, deint_dummy, p_h[2], p_h[3])
        nz_base = 2 * pair_row * NZ_C0
        gather(gathered01, pair01, even_idx)
        gather(gathered23, pair23, even_idx)
        select(packed_even, gathered01, gathered23, mask=halfmask)
        reg_to_ub(ub_p0[nz_base], packed_even, FRAC_STRIDE4)
        gather(gathered01, pair01, odd_idx)
        gather(gathered23, pair23, odd_idx)
        select(packed_even, gathered01, gathered23, mask=halfmask)
        reg_to_ub(ub_p0[nz_base + NZ_C0], packed_even, FRAC_STRIDE4)
    block_sum <<= acc[0] + acc[1]
    tmp <<= acc[2] + acc[3]
    block_sum <<= block_sum + tmp
    gather(swap, block_sum, idxu)
    block_sum <<= block_sum + swap
    ub_rmax[0:1, 0:TILE_N] <<= block_max
    ub_rsum[0:1, 0:TILE_N] <<= block_sum
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def group_half_softmax_p0_vf(ub_score0: Tensor, ub_score1: Tensor,
                             ub_score2: Tensor, ub_rmax: Tensor,
                             ub_rsum: Tensor, ub_old_weight: Tensor,
                             ub_p0: Tensor):
    """Update a full logical group and emit NZ-ready P0 in FP16."""
    sreg = RegList(DT.half, UNROLL)
    acc = RegList(DT.half, UNROLL)
    preg = RegList(DT.half, UNROLL)
    p_h = RegList(DT.hif8, UNROLL)
    neg = Reg(DT.half)
    block_max = Reg(DT.half)
    block_sum = Reg(DT.half)
    tmp = Reg(DT.half)
    swap = Reg(DT.half)
    prev_max = Reg(DT.half)
    next_max = Reg(DT.half)
    old_weight = Reg(DT.half)
    prev_sum = Reg(DT.half)
    next_sum = Reg(DT.half)
    cfg = CastConfig(round_mode=RoundMode.AWAY_FROM_ZERO,
                     reg_layout=RegLayout.ZERO, name="gfp16_p0")
    halfmask = MaskReg(DT.hif8, init_mode=MaskType.LOWHALF)
    pair01 = Reg(DT.hif8)
    pair23 = Reg(DT.hif8)
    deint_dummy = Reg(DT.hif8)
    gathered01 = Reg(DT.hif8)
    gathered23 = Reg(DT.hif8)
    packed_even = Reg(DT.hif8)
    lane_i8 = Reg(DT.int8)
    arange(lane_i8, 0)
    low6_mask = Reg(DT.int8)
    low6_mask <<= 63
    bit6_mask = Reg(DT.int8)
    bit6_mask <<= 64
    low6 = Reg(DT.int8)
    vand(low6, lane_i8, low6_mask)
    bit6 = Reg(DT.int8)
    vand(bit6, lane_i8, bit6_mask)
    shiftls(bit6, bit6, 1)
    even_i8 = Reg(DT.int8)
    vor(even_i8, low6, bit6)
    even_idx = even_i8.reinterpret(DT.uint8)
    odd_i8 = Reg(DT.int8)
    vor(odd_i8, even_i8, bit6_mask)
    odd_idx = odd_i8.reinterpret(DT.uint8)
    idx16 = Reg(DT.int16)
    arange(idx16, 0)
    swp64 = Reg(DT.int16)
    swp64 <<= QLANES
    vxor(idx16, idx16, swp64)
    idxu = idx16.reinterpret(DT.uint16)

    neg <<= NEG_LARGE_H
    for u in unroll(UNROLL):
        acc[u] <<= neg
    for rb in range(PAIR_RB):
        for u in unroll(UNROLL):
            row = rb * UNROLL + u
            sreg[u] <<= ub_score0[row * TILE_N]
            acc[u] <<= acc[u].vmax(sreg[u])
            sreg[u] <<= ub_score1[row * TILE_N]
            acc[u] <<= acc[u].vmax(sreg[u])
            sreg[u] <<= ub_score2[row * TILE_N]
            acc[u] <<= acc[u].vmax(sreg[u])
    block_max <<= acc[0].vmax(acc[1])
    tmp <<= acc[2].vmax(acc[3])
    block_max <<= block_max.vmax(tmp)
    gather(swap, block_max, idxu)
    block_max <<= block_max.vmax(swap)
    prev_max <<= ub_rmax[0:1, 0:TILE_N]
    next_max <<= block_max.vmax(prev_max)
    sub(old_weight, prev_max, next_max)
    exp(old_weight, old_weight)
    prev_sum <<= ub_rsum[0:1, 0:TILE_N]
    next_sum <<= prev_sum * old_weight

    for u in unroll(UNROLL):
        acc[u] <<= 0.0
    for pair_row in range(PAIR_SLAB_ROWS):
        for u in unroll(UNROLL):
            row = pair_row + u * PAIR_SLAB_ROWS
            sreg[u] <<= ub_score0[row * TILE_N]
            sub(preg[u], sreg[u], next_max)
            exp(preg[u], preg[u])
            acc[u] <<= acc[u] + preg[u]
            cast(p_h[u], preg[u], cfg)
        deinterleave(pair01, deint_dummy, p_h[0], p_h[1])
        deinterleave(pair23, deint_dummy, p_h[2], p_h[3])
        nz_base = 2 * pair_row * NZ_C0
        gather(gathered01, pair01, even_idx)
        gather(gathered23, pair23, even_idx)
        select(packed_even, gathered01, gathered23, mask=halfmask)
        reg_to_ub(ub_p0[nz_base], packed_even, FRAC_STRIDE4)
        gather(gathered01, pair01, odd_idx)
        gather(gathered23, pair23, odd_idx)
        select(packed_even, gathered01, gathered23, mask=halfmask)
        reg_to_ub(ub_p0[nz_base + NZ_C0], packed_even, FRAC_STRIDE4)
    block_sum <<= acc[0] + acc[1]
    tmp <<= acc[2] + acc[3]
    block_sum <<= block_sum + tmp
    gather(swap, block_sum, idxu)
    block_sum <<= block_sum + swap
    next_sum <<= next_sum + block_sum
    ub_rmax[0:1, 0:TILE_N] <<= next_max
    ub_rsum[0:1, 0:TILE_N] <<= next_sum
    ub_old_weight[0:1, 0:TILE_N] <<= old_weight
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def neg_fill_rows_h(ub_score: Tensor, valid_n: Var):
    """NEG the INVALID key rows of a partial FP16 score tile, once, so the masked half vfs below
    need no per-row mask. Those vfs load a KEY PAIR per register and select NEG straight after the
    load, with no scale in between, so a pre-filled row is bit-identical to what the select
    produced. What goes away is one scalar clamp per key pair INSIDE the vec scope, for which pl
    has no narrower spelling than a 64-bit compare (D-139 / D-150); on the board that clamp was
    the whole of fd_modified's tail-shape gap (D-156). Idempotent: a tile masked twice on one
    path is filled twice with the same bytes."""
    neg = Reg(DT.half)
    neg <<= NEG_LARGE_H
    row_mask = MaskReg(DT.half, init_mode=MaskType.LOWHALF)
    for n in range(valid_n, TILE_N):
        # One score row is 64 halves; an unmasked native store writes 128 halves.
        # Keep the following score/PV slot outside this fill (M10-057).
        reg_to_ub(ub_score[n:n + 1, 0:QLANES], neg, mask=row_mask)


@vf()
def group_half_short_p0_vf(ub_score0: Tensor, ub_score1: Tensor,
                           ub_rmax: Tensor, ub_rsum: Tensor,
                           ub_old_weight: Tensor, ub_p0: Tensor):
    """Update a 128-plus-partial group and emit its full NZ-ready P0 tile."""
    sreg = RegList(DT.half, UNROLL)
    tail = Reg(DT.half)
    acc = RegList(DT.half, UNROLL)
    preg = RegList(DT.half, UNROLL)
    p_h = RegList(DT.hif8, UNROLL)
    neg = Reg(DT.half)
    block_max = Reg(DT.half)
    block_sum = Reg(DT.half)
    tmp = Reg(DT.half)
    swap = Reg(DT.half)
    prev_max = Reg(DT.half)
    next_max = Reg(DT.half)
    old_weight = Reg(DT.half)
    prev_sum = Reg(DT.half)
    next_sum = Reg(DT.half)
    cfg = CastConfig(round_mode=RoundMode.AWAY_FROM_ZERO,
                     reg_layout=RegLayout.ZERO, name="gfp16_short_p0")
    qmask = MaskReg(DT.hif8, init_mode=MaskType.LOWQUAT)
    packed_even = Reg(DT.hif8)
    packed_odd = Reg(DT.hif8)
    even_i8 = Reg(DT.int8)
    arange(even_i8, 0)
    shiftls(even_i8, even_i8, 1)
    even_idx = even_i8.reinterpret(DT.uint8)
    odd_i8 = Reg(DT.int8)
    arange(odd_i8, 0)
    shiftls(odd_i8, odd_i8, 1)
    high_bit = Reg(DT.int8)
    high_bit <<= -128
    vxor(odd_i8, odd_i8, high_bit)
    odd_idx = odd_i8.reinterpret(DT.uint8)
    idx16 = Reg(DT.int16)
    arange(idx16, 0)
    swp64 = Reg(DT.int16)
    swp64 <<= QLANES
    vxor(idx16, idx16, swp64)
    idxu = idx16.reinterpret(DT.uint16)

    neg <<= NEG_LARGE_H
    for u in unroll(UNROLL):
        acc[u] <<= neg
    for rb in range(PAIR_RB):
        for u in unroll(UNROLL):
            row = rb * UNROLL + u
            sreg[u] <<= ub_score0[row * TILE_N]
            acc[u] <<= acc[u].vmax(sreg[u])
            tail <<= ub_score1[row * TILE_N]
            acc[u] <<= acc[u].vmax(tail)
    block_max <<= acc[0].vmax(acc[1])
    tmp <<= acc[2].vmax(acc[3])
    block_max <<= block_max.vmax(tmp)
    gather(swap, block_max, idxu)
    block_max <<= block_max.vmax(swap)
    prev_max <<= ub_rmax[0:1, 0:TILE_N]
    next_max <<= block_max.vmax(prev_max)
    sub(old_weight, prev_max, next_max)
    exp(old_weight, old_weight)
    prev_sum <<= ub_rsum[0:1, 0:TILE_N]
    next_sum <<= prev_sum * old_weight

    for u in unroll(UNROLL):
        acc[u] <<= 0.0
    for slab_id in range(KEYS_PER_GRP):
        for rb in range(PAIR_SLAB_RB):
            for u in unroll(UNROLL):
                pair_row = rb * UNROLL + u
                row = slab_id * PAIR_SLAB_ROWS + pair_row
                sreg[u] <<= ub_score0[row * TILE_N]
                sub(preg[u], sreg[u], next_max)
                exp(preg[u], preg[u])
                acc[u] <<= acc[u] + preg[u]
                cast(p_h[u], preg[u], cfg)
            for u in unroll(UNROLL):
                pair_row = rb * UNROLL + u
                nz_base = slab_id * SLAB_STRIDE4 + 2 * pair_row * NZ_C0
                gather(packed_even, p_h[u], even_idx)
                reg_to_ub(ub_p0[nz_base], packed_even, FRAC_STRIDE4, mask=qmask)
                gather(packed_odd, p_h[u], odd_idx)
                reg_to_ub(ub_p0[nz_base + NZ_C0], packed_odd,
                          FRAC_STRIDE4, mask=qmask)
    block_sum <<= acc[0] + acc[1]
    tmp <<= acc[2] + acc[3]
    block_sum <<= block_sum + tmp
    gather(swap, block_sum, idxu)
    block_sum <<= block_sum + swap
    next_sum <<= next_sum + block_sum
    ub_rmax[0:1, 0:TILE_N] <<= next_max
    ub_rsum[0:1, 0:TILE_N] <<= next_sum
    ub_old_weight[0:1, 0:TILE_N] <<= old_weight
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

@vf()
def group_half_one_tail_p0_vf(ub_score0: Tensor, ub_rmax: Tensor,
                              ub_rsum: Tensor, ub_old_weight: Tensor,
                              ub_p0: Tensor):
    """Update from one partial score tile (invalid rows NEG-filled) and emit its NZ-ready P0."""
    score = Reg(DT.half)
    expv = Reg(DT.half)
    neg = Reg(DT.half)
    block_max = Reg(DT.half)
    block_sum = Reg(DT.half)
    swap = Reg(DT.half)
    prev_max = Reg(DT.half)
    next_max = Reg(DT.half)
    old_weight = Reg(DT.half)
    prev_sum = Reg(DT.half)
    next_sum = Reg(DT.half)
    p_h = Reg(DT.hif8)
    cfg = CastConfig(round_mode=RoundMode.AWAY_FROM_ZERO,
                     reg_layout=RegLayout.ZERO, name="gfp16_one_tail_p0")
    qmask = MaskReg(DT.hif8, init_mode=MaskType.LOWQUAT)
    packed_even = Reg(DT.hif8)
    packed_odd = Reg(DT.hif8)
    even_i8 = Reg(DT.int8)
    arange(even_i8, 0)
    shiftls(even_i8, even_i8, 1)
    even_idx = even_i8.reinterpret(DT.uint8)
    odd_i8 = Reg(DT.int8)
    arange(odd_i8, 0)
    shiftls(odd_i8, odd_i8, 1)
    high_bit = Reg(DT.int8)
    high_bit <<= -128
    vxor(odd_i8, odd_i8, high_bit)
    odd_idx = odd_i8.reinterpret(DT.uint8)
    idx16 = Reg(DT.int16)
    arange(idx16, 0)
    swp64 = Reg(DT.int16)
    swp64 <<= QLANES
    vxor(idx16, idx16, swp64)
    idxu = idx16.reinterpret(DT.uint16)

    neg <<= NEG_LARGE_H
    block_max <<= neg
    for slab_id in range(KEYS_PER_GRP):
        for pair_row in range(PAIR_SLAB_ROWS):
            row = slab_id * PAIR_SLAB_ROWS + pair_row
            score <<= ub_score0[row * TILE_N]
            block_max <<= block_max.vmax(score)
    gather(swap, block_max, idxu)
    block_max <<= block_max.vmax(swap)
    prev_max <<= ub_rmax[0:1, 0:TILE_N]
    next_max <<= block_max.vmax(prev_max)
    sub(old_weight, prev_max, next_max)
    exp(old_weight, old_weight)
    prev_sum <<= ub_rsum[0:1, 0:TILE_N]
    next_sum <<= prev_sum * old_weight

    block_sum <<= 0.0
    for slab_id in range(KEYS_PER_GRP):
        for pair_row in range(PAIR_SLAB_ROWS):
            row = slab_id * PAIR_SLAB_ROWS + pair_row
            score <<= ub_score0[row * TILE_N]
            sub(expv, score, next_max)
            exp(expv, expv)
            block_sum <<= block_sum + expv
            cast(p_h, expv, cfg)
            nz_base = slab_id * SLAB_STRIDE4 + 2 * pair_row * NZ_C0
            gather(packed_even, p_h, even_idx)
            reg_to_ub(ub_p0[nz_base], packed_even, FRAC_STRIDE4, mask=qmask)
            gather(packed_odd, p_h, odd_idx)
            reg_to_ub(ub_p0[nz_base + NZ_C0], packed_odd,
                      FRAC_STRIDE4, mask=qmask)
    gather(swap, block_sum, idxu)
    block_sum <<= block_sum + swap
    next_sum <<= next_sum + block_sum
    ub_rmax[0:1, 0:TILE_N] <<= next_max
    ub_rsum[0:1, 0:TILE_N] <<= next_sum
    ub_old_weight[0:1, 0:TILE_N] <<= old_weight
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

@vf()
def group_half_long_tail_p0_vf(ub_score0: Tensor, ub_score1: Tensor,
                               ub_score2: Tensor, ub_rmax: Tensor,
                               ub_rsum: Tensor, ub_old_weight: Tensor,
                               ub_p0: Tensor):
    """Update a 256-plus-partial group and emit its full NZ-ready P0 tile."""
    sreg = RegList(DT.half, UNROLL)
    tail = Reg(DT.half)
    acc = RegList(DT.half, UNROLL)
    preg = RegList(DT.half, UNROLL)
    p_h = RegList(DT.hif8, UNROLL)
    neg = Reg(DT.half)
    block_max = Reg(DT.half)
    block_sum = Reg(DT.half)
    tmp = Reg(DT.half)
    swap = Reg(DT.half)
    prev_max = Reg(DT.half)
    next_max = Reg(DT.half)
    old_weight = Reg(DT.half)
    prev_sum = Reg(DT.half)
    next_sum = Reg(DT.half)
    cfg = CastConfig(round_mode=RoundMode.AWAY_FROM_ZERO,
                     reg_layout=RegLayout.ZERO, name="gfp16_long_tail_p0")
    halfmask = MaskReg(DT.hif8, init_mode=MaskType.LOWHALF)
    pair01 = Reg(DT.hif8)
    pair23 = Reg(DT.hif8)
    deint_dummy = Reg(DT.hif8)
    gathered01 = Reg(DT.hif8)
    gathered23 = Reg(DT.hif8)
    packed_even = Reg(DT.hif8)
    lane_i8 = Reg(DT.int8)
    arange(lane_i8, 0)
    low6_mask = Reg(DT.int8)
    low6_mask <<= 63
    bit6_mask = Reg(DT.int8)
    bit6_mask <<= 64
    low6 = Reg(DT.int8)
    vand(low6, lane_i8, low6_mask)
    bit6 = Reg(DT.int8)
    vand(bit6, lane_i8, bit6_mask)
    shiftls(bit6, bit6, 1)
    even_i8 = Reg(DT.int8)
    vor(even_i8, low6, bit6)
    even_idx = even_i8.reinterpret(DT.uint8)
    odd_i8 = Reg(DT.int8)
    vor(odd_i8, even_i8, bit6_mask)
    odd_idx = odd_i8.reinterpret(DT.uint8)
    idx16 = Reg(DT.int16)
    arange(idx16, 0)
    swp64 = Reg(DT.int16)
    swp64 <<= QLANES
    vxor(idx16, idx16, swp64)
    idxu = idx16.reinterpret(DT.uint16)

    neg <<= NEG_LARGE_H
    for u in unroll(UNROLL):
        acc[u] <<= neg
    for rb in range(PAIR_RB):
        for u in unroll(UNROLL):
            row = rb * UNROLL + u
            sreg[u] <<= ub_score0[row * TILE_N]
            acc[u] <<= acc[u].vmax(sreg[u])
            sreg[u] <<= ub_score1[row * TILE_N]
            acc[u] <<= acc[u].vmax(sreg[u])
            tail <<= ub_score2[row * TILE_N]
            acc[u] <<= acc[u].vmax(tail)
    block_max <<= acc[0].vmax(acc[1])
    tmp <<= acc[2].vmax(acc[3])
    block_max <<= block_max.vmax(tmp)
    gather(swap, block_max, idxu)
    block_max <<= block_max.vmax(swap)
    prev_max <<= ub_rmax[0:1, 0:TILE_N]
    next_max <<= block_max.vmax(prev_max)
    sub(old_weight, prev_max, next_max)
    exp(old_weight, old_weight)
    prev_sum <<= ub_rsum[0:1, 0:TILE_N]
    next_sum <<= prev_sum * old_weight

    for u in unroll(UNROLL):
        acc[u] <<= 0.0
    for pair_row in range(PAIR_SLAB_ROWS):
        for u in unroll(UNROLL):
            row = pair_row + u * PAIR_SLAB_ROWS
            sreg[u] <<= ub_score0[row * TILE_N]
            sub(preg[u], sreg[u], next_max)
            exp(preg[u], preg[u])
            acc[u] <<= acc[u] + preg[u]
            cast(p_h[u], preg[u], cfg)
        deinterleave(pair01, deint_dummy, p_h[0], p_h[1])
        deinterleave(pair23, deint_dummy, p_h[2], p_h[3])
        nz_base = 2 * pair_row * NZ_C0
        gather(gathered01, pair01, even_idx)
        gather(gathered23, pair23, even_idx)
        select(packed_even, gathered01, gathered23, mask=halfmask)
        reg_to_ub(ub_p0[nz_base], packed_even, FRAC_STRIDE4)
        gather(gathered01, pair01, odd_idx)
        gather(gathered23, pair23, odd_idx)
        select(packed_even, gathered01, gathered23, mask=halfmask)
        reg_to_ub(ub_p0[nz_base + NZ_C0], packed_even, FRAC_STRIDE4)
    block_sum <<= acc[0] + acc[1]
    tmp <<= acc[2] + acc[3]
    block_sum <<= block_sum + tmp
    gather(swap, block_sum, idxu)
    block_sum <<= block_sum + swap
    next_sum <<= next_sum + block_sum
    ub_rmax[0:1, 0:TILE_N] <<= next_max
    ub_rsum[0:1, 0:TILE_N] <<= next_sum
    ub_old_weight[0:1, 0:TILE_N] <<= old_weight
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

@vf()
def group_half_emit_vf(ub_score: Tensor, ub_rmax: Tensor,
                       ub_rsum: Tensor, ub_p: Tensor):
    """Emit one full NZ-ready P tile against the shared FP16 max."""
    sreg = RegList(DT.half, UNROLL)
    preg = RegList(DT.half, UNROLL)
    acc = RegList(DT.half, UNROLL)
    p_h = RegList(DT.hif8, UNROLL)
    group_max = Reg(DT.half)
    block_sum = Reg(DT.half)
    tmp = Reg(DT.half)
    swap = Reg(DT.half)
    running_sum = Reg(DT.half)
    next_sum = Reg(DT.half)
    cfg = CastConfig(round_mode=RoundMode.AWAY_FROM_ZERO,
                     reg_layout=RegLayout.ZERO, name="gfp16_emit")
    halfmask = MaskReg(DT.hif8, init_mode=MaskType.LOWHALF)
    pair01 = Reg(DT.hif8)
    pair23 = Reg(DT.hif8)
    deint_dummy = Reg(DT.hif8)
    gathered01 = Reg(DT.hif8)
    gathered23 = Reg(DT.hif8)
    packed_even = Reg(DT.hif8)
    lane_i8 = Reg(DT.int8)
    arange(lane_i8, 0)
    low6_mask = Reg(DT.int8)
    low6_mask <<= 63
    bit6_mask = Reg(DT.int8)
    bit6_mask <<= 64
    low6 = Reg(DT.int8)
    vand(low6, lane_i8, low6_mask)
    bit6 = Reg(DT.int8)
    vand(bit6, lane_i8, bit6_mask)
    shiftls(bit6, bit6, 1)
    even_i8 = Reg(DT.int8)
    vor(even_i8, low6, bit6)
    even_idx = even_i8.reinterpret(DT.uint8)
    odd_i8 = Reg(DT.int8)
    vor(odd_i8, even_i8, bit6_mask)
    odd_idx = odd_i8.reinterpret(DT.uint8)
    idx16 = Reg(DT.int16)
    arange(idx16, 0)
    swp64 = Reg(DT.int16)
    swp64 <<= QLANES
    vxor(idx16, idx16, swp64)
    idxu = idx16.reinterpret(DT.uint16)

    group_max <<= ub_rmax[0:1, 0:TILE_N]
    running_sum <<= ub_rsum[0:1, 0:TILE_N]
    for u in unroll(UNROLL):
        acc[u] <<= 0.0
    for pair_row in range(PAIR_SLAB_ROWS):
        for u in unroll(UNROLL):
            row = pair_row + u * PAIR_SLAB_ROWS
            sreg[u] <<= ub_score[row * TILE_N]
            sub(preg[u], sreg[u], group_max)
            exp(preg[u], preg[u])
            acc[u] <<= acc[u] + preg[u]
            cast(p_h[u], preg[u], cfg)
        deinterleave(pair01, deint_dummy, p_h[0], p_h[1])
        deinterleave(pair23, deint_dummy, p_h[2], p_h[3])
        nz_base = 2 * pair_row * NZ_C0
        gather(gathered01, pair01, even_idx)
        gather(gathered23, pair23, even_idx)
        select(packed_even, gathered01, gathered23, mask=halfmask)
        reg_to_ub(ub_p[nz_base], packed_even, FRAC_STRIDE4)
        gather(gathered01, pair01, odd_idx)
        gather(gathered23, pair23, odd_idx)
        select(packed_even, gathered01, gathered23, mask=halfmask)
        reg_to_ub(ub_p[nz_base + NZ_C0], packed_even, FRAC_STRIDE4)
    block_sum <<= acc[0] + acc[1]
    tmp <<= acc[2] + acc[3]
    block_sum <<= block_sum + tmp
    gather(swap, block_sum, idxu)
    block_sum <<= block_sum + swap
    next_sum <<= running_sum + block_sum
    ub_rsum[0:1, 0:TILE_N] <<= next_sum
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def group_half_emit_tail_vf(ub_score: Tensor, ub_rmax: Tensor,
                            ub_rsum: Tensor, ub_p: Tensor):
    """Emit one NZ-ready P tile (invalid rows NEG-filled) against the shared FP16 max."""
    score = Reg(DT.half)
    expv = Reg(DT.half)
    neg = Reg(DT.half)
    group_max = Reg(DT.half)
    block_sum = Reg(DT.half)
    swap = Reg(DT.half)
    running_sum = Reg(DT.half)
    next_sum = Reg(DT.half)
    p_h = Reg(DT.hif8)
    cfg = CastConfig(round_mode=RoundMode.AWAY_FROM_ZERO,
                     reg_layout=RegLayout.ZERO, name="gfp16_emit_tail")
    qmask = MaskReg(DT.hif8, init_mode=MaskType.LOWQUAT)
    packed_even = Reg(DT.hif8)
    packed_odd = Reg(DT.hif8)
    even_i8 = Reg(DT.int8)
    arange(even_i8, 0)
    shiftls(even_i8, even_i8, 1)
    even_idx = even_i8.reinterpret(DT.uint8)
    odd_i8 = Reg(DT.int8)
    arange(odd_i8, 0)
    shiftls(odd_i8, odd_i8, 1)
    high_bit = Reg(DT.int8)
    high_bit <<= -128
    vxor(odd_i8, odd_i8, high_bit)
    odd_idx = odd_i8.reinterpret(DT.uint8)
    idx16 = Reg(DT.int16)
    arange(idx16, 0)
    swp64 = Reg(DT.int16)
    swp64 <<= QLANES
    vxor(idx16, idx16, swp64)
    idxu = idx16.reinterpret(DT.uint16)

    neg <<= NEG_LARGE_H
    group_max <<= ub_rmax[0:1, 0:TILE_N]
    running_sum <<= ub_rsum[0:1, 0:TILE_N]
    block_sum <<= 0.0
    for slab_id in range(KEYS_PER_GRP):
        for pair_row in range(PAIR_SLAB_ROWS):
            row = slab_id * PAIR_SLAB_ROWS + pair_row
            score <<= ub_score[row * TILE_N]
            sub(expv, score, group_max)
            exp(expv, expv)
            block_sum <<= block_sum + expv
            cast(p_h, expv, cfg)
            nz_base = slab_id * SLAB_STRIDE4 + 2 * pair_row * NZ_C0
            gather(packed_even, p_h, even_idx)
            reg_to_ub(ub_p[nz_base], packed_even, FRAC_STRIDE4, mask=qmask)
            gather(packed_odd, p_h, odd_idx)
            reg_to_ub(ub_p[nz_base + NZ_C0], packed_odd,
                      FRAC_STRIDE4, mask=qmask)
    gather(swap, block_sum, idxu)
    block_sum <<= block_sum + swap
    next_sum <<= running_sum + block_sum
    ub_rsum[0:1, 0:TILE_N] <<= next_sum
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@func()
def _qk_bridge_two_single_group(ub_dst, l0c_src):
    """Downcast one complete 128x64 score tile into each AIV subblock."""
    l0c_to_ub(ub_dst, l0c_src[0:TILE_N, 0:ROWS_PER_SB],
              M=TILE_N, N=ROWS_PER_SB, N_dst=ROWS_PER_SB, M_src=TILE_N,
              dual_mode=DualMode.SINGLE, sub_block_id=0, scale=SOFTMAX_SCALE)
    l0c_to_ub(ub_dst, l0c_src[0:TILE_N, ROWS_PER_SB:TILE_M],
              M=TILE_N, N=ROWS_PER_SB, N_dst=ROWS_PER_SB, M_src=TILE_N,
              dual_mode=DualMode.SINGLE, sub_block_id=1, scale=SOFTMAX_SCALE)


@func()
def _publish_group_p_v9(l1p_dst, ub_p, row_begin, p_mutex, ready_evt):
    p_mutex.lock()
    ready_evt.wait()
    for slab_id in unroll(KEYS_PER_GRP):
        l1p_dst[slab_id * SLAB:(slab_id + 1) * SLAB,
                row_begin:row_begin + ROWS_PER_SB] <<= (
            ub_p.nz()[0:SLAB,
                      slab_id * ROWS_PER_SB:(slab_id + 1) * ROWS_PER_SB]
        )
    p_mutex.ready()


@func()
def _publish_group_p3_v9(l1p_dst, ub_p, row_begin, p_mutex, ready_evt):
    """Publish a minimal unused payload with the same explicit P2 readiness edge."""
    p_mutex.lock()
    ready_evt.wait()
    l1p_dst[0:FAST_TAIL_N, row_begin:row_begin + ROWS_PER_SB] <<= (
        ub_p.nz()[0:FAST_TAIL_N, 0:ROWS_PER_SB]
    )
    p_mutex.ready()


@func()
def _alloc_group_v9(q, k, v):
    q_hif8 = q.reinterpret(DT.hif8, name="q_hif8")
    k_hif8 = k.reinterpret(DT.hif8, name="k_hif8")
    v_hif8 = v.reinterpret(DT.hif8, name="v_hif8")
    l1q = DBuff(DT.hif8, [TILE_M, D_HEAD], Position.L1)
    # Keep the head K0 independently ready; only K1/K2 share a slot.
    # Aligned NZ column bands preserve each operand byte layout.
    l1k0 = DBuff(DT.hif8, [TILE_N, D_HEAD], Position.L1)
    l1k = DBuff(DT.hif8, [TILE_N, 2 * D_HEAD], Position.L1)
    # Same slot index across all three slabs; NZ column bands keep
    # each operand contiguous while sharing one physical slot identity.
    l1v = TBuff(DT.hif8, [TILE_N, 3 * D_HEAD], Position.L1)
    l1p0 = TBuff(DT.hif8, [TILE_N, TILE_M], Position.L1)
    l1p1 = TBuff(DT.hif8, [TILE_N, TILE_M], Position.L1)
    l1p2 = TBuff(DT.hif8, [TILE_N, TILE_M], Position.L1)
    l0c_qk = DBuff(DT.float, [TILE_M, TILE_M], Position.L0C)
    l0c_pv = DBuff(DT.float, [TILE_M, D_HEAD], Position.L0C)
    ub_score = TBuff(DT.half, [TILE_N, ROWS_PER_SB], Position.UB)
    ub_pv = DBuff(DT.float, [ROWS_PER_SB, D_HEAD], Position.UB)
    ub_p0 = DBuff(DT.hif8, [FRAC_STRIDE4, KEYS_PER_GRP * ROWS_PER_SB], Position.UB)
    ub_p1 = DBuff(DT.hif8, [FRAC_STRIDE4, KEYS_PER_GRP * ROWS_PER_SB], Position.UB)
    ub_p2 = DBuff(DT.hif8, [FRAC_STRIDE4, KEYS_PER_GRP * ROWS_PER_SB], Position.UB)
    ub_old_weight = TBuff(DT.half, [1, TILE_N], Position.UB)
    ub_rmax = TBuff(DT.half, [1, TILE_N], Position.UB)
    ub_rsum = TBuff(DT.half, [1, TILE_N], Position.UB)
    ub_merge_a = Tensor(DT.float, [1, TILE_M], Position.UB)
    ub_merge_den = Tensor(DT.float, [1, TILE_M], Position.UB)
    ub_accum = Tensor(DT.float, [ROWS_PER_SB, D_HEAD], Position.UB)
    ub_out = Tensor(DT.bfloat16, [ROWS_PER_SB, D_HEAD], Position.UB)
    fd_accum = split_workspace(DT.float, [MAX_FD_WS, TILE_M, D_HEAD], name="fd_accum")
    fd_max = split_workspace(DT.half, [MAX_FD_WS, 1, TILE_M], name="fd_max")
    fd_sum = split_workspace(DT.half, [MAX_FD_WS, 1, TILE_M], name="fd_sum")
    return (q_hif8, k_hif8, v_hif8, l1q, l1k0, l1k, l1v,
            l1p0, l1p1, l1p2, l0c_qk, l0c_pv, ub_score, ub_pv, ub_p0, ub_p1, ub_p2,
            ub_old_weight, ub_rmax, ub_rsum, ub_merge_a, ub_merge_den, ub_accum, ub_out,
            fd_accum, fd_max, fd_sum)


@func()
def _group384_body_v9(q, k, v, out, B, MQ, N, D, cv0, cv1, pv_mutex, p_mutex,
                   qk0_l0c_reusable, qk1_l0c_reusable, qk2_l0c_reusable,
                   pv_l0c_reusable, l0c_mmad_ready, p0_ready, p1_ready, p2_ready):
    (q_hif8, k_hif8, v_hif8, l1q, l1k0, l1k, l1v,
     l1p0, l1p1, l1p2, l0c_qk, l0c_pv, ub_score, ub_pv, ub_p0, ub_p1, ub_p2,
     ub_old_weight, ub_rmax, ub_rsum, ub_merge_a, ub_merge_den, ub_accum, ub_out,
     fd_accum, fd_max, fd_sum) = _alloc_group_v9(q, k, v)

    core = Var(GetCubeIdx())
    core_count = Var(GetCubeNum())
    vec = Var(GetVecIdx())
    row_begin = Var(GetSubBlockIdx() * ROWS_PER_SB)
    tiles_m_per_b = CeilDiv(MQ, TILE_M)
    k_groups = CeilDiv(N, GROUP_N)
    # Only the last group is partial. Specialize its K extents from N
    # instead of sizing L0 views from the loop-carried group index.
    tail_keys = (N - 1) % GROUP_N + 1
    tail_k0 = Min(TILE_N, tail_keys)
    tail_k1 = Max(1, Min(TILE_N, tail_keys - TILE_N))
    tail_k2 = Max(1, Min(TILE_N, tail_keys - 2 * TILE_N))
    total_tasks = Var(B * tiles_m_per_b * k_groups)
    tasks_floor = Var(total_tasks // core_count)
    # Avoid Var modulo here: the side-split trace exposes an inactive-side
    # GetCubeNum placeholder of zero, while the already-supported division is
    # normalized by lowering.  The algebraic remainder has identical runtime value.
    tasks_extra = Var(total_tasks - tasks_floor * core_count)
    flat_start = Var(core * tasks_floor + Min(core, tasks_extra))
    flat_end = Var((core + 1) * tasks_floor + Min(core + 1, tasks_extra))
    my_tasks = Var(Max(flat_end - flat_start, 0))

    tail_ws = Var(-1, DT.int)
    head_ws = Var(-1, DT.int)
    split_count = Var(0, DT.int)
    for boundary_core in range(1, MAX_CORE_COUNT):
        if core_count > boundary_core:
            boundary = Var(boundary_core * tasks_floor + Min(boundary_core, tasks_extra))
            if boundary % k_groups != 0:
                if core == boundary_core:
                    tail_ws <<= split_count * 2 + 1
                if core + 1 == boundary_core:
                    head_ws <<= split_count * 2
                split_count += 1

    last_group_valid = Var(N - (k_groups - 1) * GROUP_N)
    # Invalid score rows are masked; PV reads only valid loaded V rows.
    # No overlapping whole-L1 fill (D-214).
    # The explicit per-slot events are the sole Fix->M reuse authority. QK0 and
    # QK2 rotate through one shared score slot; PV rotates through two slots.
    # Lead-2 can expose three QK results plus one older PV result before Fix
    # consumes the FIFO head, so the result-ready edge needs four credits.
    with auto_sync(mode="manual_fix"):
        prefix_mt = Var(-1, DT.int)
        suffix_mt = Var(-1, DT.int)
        if my_tasks > 0:
            start_group = Var(flat_start % k_groups)
            if start_group != 0:
                prefix_mt <<= flat_start // k_groups
            end_group = Var(flat_end % k_groups)
            if end_group != 0:
                suffix_mt <<= (flat_end - 1) // k_groups

        # Seed producer/consumer coordinates once.  The hot loop advances them
        # with cheap increments and wrap tests instead of repeating integer
        # division/modulo for both sides of every group beat.
        p_start_mt = Var(flat_start // k_groups)
        p_kg = Var(flat_start - p_start_mt * k_groups)
        p_qm = Var(p_start_mt % 2)
        p_sm = Var(p_start_mt % GROUP_CACHE)
        p_batch = Var(p_start_mt // tiles_m_per_b)
        p_local_mt = Var(p_start_mt - p_batch * tiles_m_per_b)
        p_q_base = Var(p_batch * MQ)
        p_kv_base = Var(p_batch * N)
        p_m0 = Var(p_local_mt * TILE_M)
        p_need_q = Var(1, DT.int)

        c_lag = Var(0, DT.int)
        c_mt = Var(p_start_mt)
        c_kg = Var(p_kg)
        c_sm = Var(p_sm)
        c_batch = Var(p_batch)
        c_local_mt = Var(p_local_mt)
        c_q_base = Var(p_q_base)
        c_kv_base = Var(p_kv_base)
        c_m0 = Var(p_m0)

        # V preload coordinates advance independently from p_*.  V[c_lag] is
        # consumed before its physical slot is refilled, then the next V group
        # is queued at the bottom of the same iteration.
        vl_lag = Var(0, DT.int)
        vl_kg = Var(p_kg)
        vl_local_mt = Var(p_local_mt)
        vl_kv_base = Var(p_kv_base)

        # Prime K for the first producer beat. Every later K group is loaded at
        # the end of its predecessor beat into the opposite DBuff slot.
        if my_tasks > 0:
            pk_n0 = Var(p_kg * GROUP_N)
            pk_valid_n0 = Min(TILE_N, N - pk_n0)
            pk_n1 = Var(pk_n0 + TILE_N)
            pk_valid_n1 = Min(TILE_N, Max(N - pk_n1, 0))
            pk_n2 = Var(pk_n1 + TILE_N)
            pk_valid_n2 = Min(TILE_N, Max(N - pk_n2, 0))
            l1k0[0][0:pk_valid_n0, 0:D] <<= (
                k_hif8[p_kv_base + pk_n0:p_kv_base + pk_n0 + pk_valid_n0, 0:D]
            )
            if pk_valid_n1 > 0:
                l1k[0][:, 0 * D_HEAD:1 * D_HEAD][0:pk_valid_n1, 0:D] <<= (
                    k_hif8[p_kv_base + pk_n1:p_kv_base + pk_n1 + pk_valid_n1, 0:D]
                )
            if pk_valid_n2 > 0:
                l1k[0][:, 1 * D_HEAD:2 * D_HEAD][0:pk_valid_n2, 0:D] <<= (
                    k_hif8[p_kv_base + pk_n2:p_kv_base + pk_n2 + pk_valid_n2, 0:D]
                )

        for step in range(0, my_tasks + GROUP_PRELOAD):
            if step < my_tasks:
                p_n0 = Var(p_kg * GROUP_N)
                valid_m = Min(TILE_M, MQ - p_m0)
                valid_n0 = Min(TILE_N, N - p_n0)
                n1 = Var(p_n0 + TILE_N)
                valid_n1 = Min(TILE_N, Max(N - n1, 0))
                n2 = Var(n1 + TILE_N)
                valid_n2 = Min(TILE_N, Max(N - n2, 0))

                if p_need_q > 0:
                    l1q[p_qm][0:valid_m, 0:D] <<= (
                        q_hif8[p_q_base + p_m0:p_q_base + p_m0 + valid_m, 0:D]
                    )
                    # Only a full three-tile group has a dedicated first-group
                    # VF. Every partial group starts from explicit state.
                    if valid_n2 != TILE_N:
                        init_softmax_half_vf(ub_rmax[p_sm], ub_rsum[p_sm])

                qk0_l0c_reusable.wait()
                matmul(l0c_qk[0], l1k0[step], l1q[p_qm],
                       m=TILE_N, n=TILE_M, k=D_HEAD, is_init=True)
                l0c_mmad_ready.set()
                cv0.lock()
                l0c_mmad_ready.wait()
                _qk_bridge_two_single_group(ub_score[0], l0c_qk[0])
                qk0_l0c_reusable.set()
                cv0.ready()

                qk1_l0c_reusable.wait()
                matmul(l0c_qk[1], l1k[step][:, 0 * D_HEAD:1 * D_HEAD], l1q[p_qm],
                       m=TILE_N, n=TILE_M, k=D_HEAD, is_init=True)
                l0c_mmad_ready.set()
                cv1.lock()
                l0c_mmad_ready.wait()
                _qk_bridge_two_single_group(ub_score[1], l0c_qk[1])
                qk1_l0c_reusable.set()
                cv1.ready()

                # QK2 reuses QK0's score slot after its FixP export. Even a
                # 128/<128/0 tail executes this zero-K tile: cv0 is a depth-two
                # score0/score2 ring, so skipping the second beat would
                # phase-shift the next query tile onto the wrong event slot.
                qk2_l0c_reusable.wait()
                matmul(l0c_qk[0], l1k[step][:, 1 * D_HEAD:2 * D_HEAD], l1q[p_qm],
                       m=TILE_N, n=TILE_M, k=D_HEAD, is_init=True)
                l0c_mmad_ready.set()
                cv0.lock()
                l0c_mmad_ready.wait()
                _qk_bridge_two_single_group(ub_score[2], l0c_qk[0])
                qk2_l0c_reusable.set()
                cv0.ready()

                cv0.wait()
                cv1.wait()
                # cv0 is a depth-2 FIFO: its first token owns score0 and its
                # second token owns score2.
                cv0.wait()
                if valid_n2 == TILE_N:
                    if p_need_q > 0:
                        group_half_softmax_p0_first_vf(
                            ub_score[0], ub_score[1], ub_score[2],
                            ub_rmax[p_sm], ub_rsum[p_sm], ub_p0[step]
                        )
                    else:
                        group_half_softmax_p0_vf(
                            ub_score[0], ub_score[1], ub_score[2],
                            ub_rmax[p_sm], ub_rsum[p_sm],
                            ub_old_weight[step], ub_p0[step]
                        )
                else:
                    if valid_n1 == 0:
                        neg_fill_rows_h(ub_score[0], valid_n0)
                        group_half_one_tail_p0_vf(
                            ub_score[0], ub_rmax[p_sm], ub_rsum[p_sm],
                            ub_old_weight[step], ub_p0[step]
                        )
                    else:
                        if valid_n2 == 0:
                            neg_fill_rows_h(ub_score[1], valid_n1)
                            group_half_short_p0_vf(
                                ub_score[0], ub_score[1], ub_rmax[p_sm],
                                ub_rsum[p_sm], ub_old_weight[step], ub_p0[step]
                            )
                        else:
                            neg_fill_rows_h(ub_score[2], valid_n2)
                            group_half_long_tail_p0_vf(
                                ub_score[0], ub_score[1], ub_score[2],
                                ub_rmax[p_sm], ub_rsum[p_sm],
                                ub_old_weight[step], ub_p0[step]
                            )
                cv0.free()
                p0_ready.set()
                _publish_group_p_v9(l1p0[step], ub_p0[step], row_begin, p_mutex, p0_ready)

                # Each published P beat can enter Cube while Vector emits the
                # next beat against the shared group max.
                if valid_n1 == TILE_N:
                    group_half_emit_vf(
                        ub_score[1], ub_rmax[p_sm], ub_rsum[p_sm], ub_p1[step]
                    )
                else:
                    # valid_n1 can be zero. The masked emitter then produces a
                    # zero tile while retaining the original V->MTE3 ownership
                    # scope and the fixed P0/P1/P2 event sequence.
                    neg_fill_rows_h(ub_score[1], valid_n1)
                    group_half_emit_tail_vf(
                        ub_score[1], ub_rmax[p_sm], ub_rsum[p_sm],
                        ub_p1[step]
                    )
                cv1.free()
                p1_ready.set()
                _publish_group_p_v9(l1p1[step], ub_p1[step], row_begin, p_mutex, p1_ready)
                # P2 owns an independent UB tile. Reusing P0 here is a real-A5
                # MTE3->Vector race even though the simulator accepts it.
                if valid_n2 > 0:
                    if valid_n2 == TILE_N:
                        group_half_emit_vf(
                            ub_score[2], ub_rmax[p_sm], ub_rsum[p_sm], ub_p2[step]
                        )
                    else:
                        neg_fill_rows_h(ub_score[2], valid_n2)
                        group_half_emit_tail_vf(
                            ub_score[2], ub_rmax[p_sm], ub_rsum[p_sm],
                            ub_p2[step]
                        )
                    p2_ready.set()
                    _publish_group_p_v9(l1p2[step], ub_p2[step], row_begin, p_mutex, p2_ready)
                else:
                    # p_mutex has six credits for ordered P0/P1/P2 beats. Publish a
                    # phase-only three-row beat so the next logical group keeps
                    # the same physical credit mapping; consumer PV skips it.
                    p2_ready.set()
                    _publish_group_p3_v9(l1p2[step], ub_p2[step], row_begin, p_mutex, p2_ready)
                cv0.free()

                p_need_q <<= 0
                p_kg += 1
                if p_kg == k_groups:
                    p_kg <<= 0
                    p_need_q <<= 1
                    p_qm <<= 1 - p_qm
                    p_sm += 1
                    if p_sm == GROUP_CACHE:
                        p_sm <<= 0
                    p_local_mt += 1
                    p_m0 += TILE_M
                    if p_local_mt == tiles_m_per_b:
                        p_local_mt <<= 0
                        p_m0 <<= 0
                        p_batch += 1
                        p_q_base += MQ
                        p_kv_base += N

                # Queue K[next] before V[current]. The next producer can start
                # from K0 as soon as its head transfer is ready, while the rest
                # of this MTE2 sequence overlaps the current consumer's PV work.
                if step + 1 < my_tasks:
                    nk_n0 = Var(p_kg * GROUP_N)
                    nk_valid_n0 = Min(TILE_N, N - nk_n0)
                    nk_n1 = Var(nk_n0 + TILE_N)
                    nk_valid_n1 = Min(TILE_N, Max(N - nk_n1, 0))
                    nk_n2 = Var(nk_n1 + TILE_N)
                    nk_valid_n2 = Min(TILE_N, Max(N - nk_n2, 0))
                    l1k0[step + 1][0:nk_valid_n0, 0:D] <<= (
                        k_hif8[p_kv_base + nk_n0:p_kv_base + nk_n0 + nk_valid_n0, 0:D]
                    )
                    if nk_valid_n1 > 0:
                        l1k[step + 1][:, 0 * D_HEAD:1 * D_HEAD][0:nk_valid_n1, 0:D] <<= (
                            k_hif8[p_kv_base + nk_n1:p_kv_base + nk_n1 + nk_valid_n1, 0:D]
                        )
                    if nk_valid_n2 > 0:
                        l1k[step + 1][:, 1 * D_HEAD:2 * D_HEAD][0:nk_valid_n2, 0:D] <<= (
                            k_hif8[p_kv_base + nk_n2:p_kv_base + nk_n2 + nk_valid_n2, 0:D]
                        )

            if step >= GROUP_PRELOAD:
                c_n0 = Var(c_kg * GROUP_N)
                lag_valid_m = Min(TILE_M, MQ - c_m0)
                local_rows = Min(ROWS_PER_SB, Max(lag_valid_m - row_begin, 0))
                v_rows0 = Min(TILE_N, N - c_n0)
                lag_n1 = Var(c_n0 + TILE_N)
                v_rows1 = Min(TILE_N, Max(N - lag_n1, 0))
                lag_n2 = Var(lag_n1 + TILE_N)
                v_rows2 = Min(TILE_N, Max(N - lag_n2, 0))

                # Only one PV result is emitted per logical group. Two L0C
                # slots let the following PV MMAD overlap the previous FixP.
                pv_l0c_reusable.wait()
                p_mutex.wait()
                if v_rows0 == TILE_N:
                    matmul(l0c_pv[c_lag],
                           l1p0[c_lag].T, l1v[c_lag][:, 0 * D_HEAD:1 * D_HEAD].T,
                           m=TILE_M, n=D_HEAD, k=TILE_N, is_init=True)
                else:
                    matmul(l0c_pv[c_lag],
                           l1p0[c_lag].T, l1v[c_lag][:, 0 * D_HEAD:1 * D_HEAD].T,
                           m=TILE_M, n=D_HEAD, k=tail_k0, is_init=True)
                p_mutex.free()
                # P1 always owns the second ring token, including a dummy beat
                # for tails no longer than 128 keys.
                p_mutex.wait()
                if v_rows1 > 0:
                    if v_rows1 == TILE_N:
                        matmul(l0c_pv[c_lag],
                               l1p1[c_lag].T, l1v[c_lag][:, 1 * D_HEAD:2 * D_HEAD].T,
                               m=TILE_M, n=D_HEAD, k=TILE_N, is_init=False)
                    else:
                        matmul(l0c_pv[c_lag],
                               l1p1[c_lag].T, l1v[c_lag][:, 1 * D_HEAD:2 * D_HEAD].T,
                               m=TILE_M, n=D_HEAD, k=tail_k1, is_init=False)
                p_mutex.free()
                p_mutex.wait()
                if v_rows2 > 0:
                    if v_rows2 == TILE_N:
                        matmul(l0c_pv[c_lag],
                               l1p2[c_lag].T, l1v[c_lag][:, 2 * D_HEAD:3 * D_HEAD].T,
                               m=TILE_M, n=D_HEAD, k=TILE_N, is_init=False)
                    else:
                        # Partial P/V accumulate paths must not read padded key
                        # rows: a full-K accumulate can introduce NaNs even when
                        # both L1 operands were explicitly zero-filled.
                        matmul(l0c_pv[c_lag],
                               l1p2[c_lag].T, l1v[c_lag][:, 2 * D_HEAD:3 * D_HEAD].T,
                               m=TILE_M, n=D_HEAD, k=tail_k2, is_init=False)
                p_mutex.free()
                l0c_mmad_ready.set()

                pv_mutex.lock()
                l0c_mmad_ready.wait()
                l0c_to_ub(ub_pv[c_lag], l0c_pv[c_lag],
                          M=TILE_M, N=D_HEAD, N_dst=D_HEAD, M_src=TILE_M,
                          dual_mode=DualMode.SPLITM, sub_block_id=0)
                pv_l0c_reusable.set()
                pv_mutex.ready()
                pv_mutex.wait()

                if c_kg == 0:
                    # The first PV overwrites every accumulator element.
                    accum_pv_first_vf(ub_accum, ub_pv[c_lag])
                elif c_lag == 0:
                    accum_pv_first_vf(ub_accum, ub_pv[c_lag])
                else:
                    accum_pv_fp16ow_vf(ub_accum, ub_pv[c_lag], ub_old_weight[c_lag])
                pv_mutex.free()

                if c_kg == k_groups - 1:
                    if c_mt == prefix_mt:
                        if local_rows > 0:
                            fd_accum[tail_ws, row_begin:row_begin + local_rows, 0:D_HEAD] <<= (
                                ub_accum[0:local_rows, 0:D_HEAD]
                            )
                            fd_max[tail_ws, 0:1, row_begin:row_begin + local_rows] <<= (
                                ub_rmax[c_sm][0:1, 0:local_rows]
                            )
                            fd_sum[tail_ws, 0:1, row_begin:row_begin + local_rows] <<= (
                                ub_rsum[c_sm][0:1, 0:local_rows]
                            )
                    else:
                        final_div_fp16rsum_vf(ub_accum, ub_rsum[c_sm], ub_out)
                        if local_rows > 0:
                            out[c_q_base + c_m0 + row_begin:
                                c_q_base + c_m0 + row_begin + local_rows, 0:D] <<= (
                                ub_out[0:local_rows, 0:D]
                            )

                if c_lag + 1 == my_tasks:
                    if c_kg != k_groups - 1:
                        if c_mt == suffix_mt:
                            if local_rows > 0:
                                fd_accum[head_ws, row_begin:row_begin + local_rows, 0:D_HEAD] <<= (
                                    ub_accum[0:local_rows, 0:D_HEAD]
                                )
                                fd_max[head_ws, 0:1, row_begin:row_begin + local_rows] <<= (
                                    ub_rmax[c_sm][0:1, 0:local_rows]
                                )
                                fd_sum[head_ws, 0:1, row_begin:row_begin + local_rows] <<= (
                                    ub_rsum[c_sm][0:1, 0:local_rows]
                                )

                c_lag += 1
                c_kg += 1
                if c_kg == k_groups:
                    c_kg <<= 0
                    c_mt += 1
                    c_sm += 1
                    if c_sm == GROUP_CACHE:
                        c_sm <<= 0
                    c_local_mt += 1
                    c_m0 += TILE_M
                    if c_local_mt == tiles_m_per_b:
                        c_local_mt <<= 0
                        c_m0 <<= 0
                        c_batch += 1
                        c_q_base += MQ
                        c_kv_base += N

            # Refill V only after the consumer has retired the same modulo-3
            # slot. The first two iterations prime V0/V1; every steady
            # iteration consumes one old group and then queues one future group.
            if vl_lag < my_tasks:
                vl_n0 = Var(vl_kg * GROUP_N)
                vl_rows0 = Min(TILE_N, N - vl_n0)
                vl_n1 = Var(vl_n0 + TILE_N)
                vl_rows1 = Min(TILE_N, Max(N - vl_n1, 0))
                vl_n2 = Var(vl_n1 + TILE_N)
                vl_rows2 = Min(TILE_N, Max(N - vl_n2, 0))
                l1v[vl_lag][:, 0 * D_HEAD:1 * D_HEAD][0:vl_rows0, 0:D] <<= (
                    v_hif8[vl_kv_base + vl_n0:vl_kv_base + vl_n0 + vl_rows0, 0:D]
                )
                if vl_rows1 > 0:
                    l1v[vl_lag][:, 1 * D_HEAD:2 * D_HEAD][0:vl_rows1, 0:D] <<= (
                        v_hif8[vl_kv_base + vl_n1:vl_kv_base + vl_n1 + vl_rows1, 0:D]
                    )
                if vl_rows2 > 0:
                    l1v[vl_lag][:, 2 * D_HEAD:3 * D_HEAD][0:vl_rows2, 0:D] <<= (
                        v_hif8[vl_kv_base + vl_n2:vl_kv_base + vl_n2 + vl_rows2, 0:D]
                    )
                vl_lag += 1
                vl_kg += 1
                if vl_kg == k_groups:
                    vl_kg <<= 0
                    vl_local_mt += 1
                    if vl_local_mt == tiles_m_per_b:
                        vl_local_mt <<= 0
                        vl_kv_base += N

        allvec_ready(7, Pipe.MTE3)

    allvec_wait(7, Pipe.MTE2)

    fd_task = Var(vec // 2)
    fd_row0 = Var((vec % 2) * ROWS_PER_SB)
    fd_found = Var(0, DT.int)
    fd_mt = Var(0, DT.int)
    scan_rank = Var(0, DT.int)
    for boundary_core in range(1, MAX_CORE_COUNT):
        if core_count > boundary_core:
            boundary = Var(boundary_core * tasks_floor + Min(boundary_core, tasks_extra))
            if boundary % k_groups != 0:
                if scan_rank == fd_task:
                    fd_found <<= 1
                    fd_mt <<= boundary // k_groups
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
                fd_merge_fp16_vf(ub_pv[0], ub_pv[1], ub_rmax[0], ub_rmax[1],
                                 ub_rsum[0], ub_rsum[1], ub_merge_a, ub_merge_den, ub_pv[0])
                output_cast_vf(ub_pv[0], ub_out)
                out[fd_q_base + fd_m0 + fd_row0:fd_q_base + fd_m0 + fd_row0 + fd_rows, 0:D] <<= (
                    ub_out[0:fd_rows, 0:D]
                )
    return out


@kernel()
def pfa_fd_v9_allhif8_kernel(q: GM[u8, ('TQ', 'D')], k: GM[u8, ('TK', 'D')], v: GM[u8, ('TK', 'D')],
                            out: GM[bf16, ('TQ', 'D')], B: i32, MQ: i32, N: i32, D: i32):
    cv0 = CvMutex(0, depth=2, src_start_pipe=Pipe.FIX, dst_start_pipe=Pipe.V,
                  src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    cv1 = CvMutex(1, depth=1, src_start_pipe=Pipe.FIX, dst_start_pipe=Pipe.V,
                  src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    pv_mutex = CvMutex(2, depth=2,
                       src_start_pipe=Pipe.FIX, dst_start_pipe=Pipe.V,
                       src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    # One explicit V -> MTE3 ready per P tile: emit sets, the publish store waits. The
    # three hand-offs alternate strictly per tile, so depth one is exact; letting autosync
    # coalesce them into one flag undercounts the run-ahead (D-079's bounds gap).
    p0_ready = SEvent(Pipe.V, Pipe.MTE3, name="p0_ready")
    p1_ready = SEvent(Pipe.V, Pipe.MTE3, name="p1_ready")
    p2_ready = SEvent(Pipe.V, Pipe.MTE3, name="p2_ready")
    p_mutex = VcMutex(3, depth=P_EVENT_DEPTH,
                      src_start_pipe=Pipe.MTE3, dst_start_pipe=Pipe.MTE1,
                      src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE1)
    qk0_l0c_reusable = SEvent(Pipe.FIX, Pipe.M, preset=True, name="qk02_l0c_reusable")
    qk1_l0c_reusable = SEvent(Pipe.FIX, Pipe.M, preset=True, name="qk1_l0c_reusable")
    pv_l0c_reusable = DEvent(Pipe.FIX, Pipe.M, preset=True, name="pv_l0c_reusable")
    l0c_mmad_ready = QEvent(Pipe.M, Pipe.FIX, name="l0c_mmad_ready")
    return _group384_body_v9(q, k, v, out, B, MQ, N, D, cv0, cv1, pv_mutex, p_mutex,
                          qk0_l0c_reusable, qk1_l0c_reusable, qk0_l0c_reusable,
                          pv_l0c_reusable, l0c_mmad_ready, p0_ready, p1_ready, p2_ready)
