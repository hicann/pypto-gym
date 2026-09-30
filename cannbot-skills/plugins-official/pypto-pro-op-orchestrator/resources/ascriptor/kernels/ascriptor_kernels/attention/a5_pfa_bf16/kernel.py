# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The BF16 PFA forward baseline, in two shardings of the same mathematics.

Both bodies compute the same dense MQA attention -- KV-major QK (score^T = K @ Q^T), a lane-wise
transposed online softmax, a BF16 P published NZ-direct, and a PV matmul -- and differ only in how
the work is spread over cores.

`qk_softmax_pv_flat_kernel` partitions by query tile and FUSES the (M-tile, K-tile) loops into one
linear `step` loop, so the lag-2 software pipeline carries across M-tile boundaries: M-tile B's QK
ramp overlaps M-tile A's PV drain and the per-M-tile ramp/drain bubbles (2 QK-only and 2 PV-only
trips each) disappear. The price is k_tiles >= 2, and the per-M-tile state (rsum, accum, out, l1q)
has to be double-buffered by M-tile parity so A's drain and B's ramp do not share it; `rmax` stays
single, because only stage 1 touches it and stage 1 is strictly serial in `step`.

`qk_softmax_pv_fd_kernel` shards over the flat (m_tile, k_tile) stream-K grid instead, so a core
boundary can land inside an M tile: that tile writes an unnormalised partial (accumulator, row max,
row sum) to a GM workspace and, after a vec-wide barrier, two aggregation VFs merge the pair with an
explicit ties-away BF16 cast at the end. It assumes at most two cores own one M tile.

The shared geometry and the softmax/accumulate/finalise VFs below are one copy: the two modules
defined them identically."""

import math

from ascriptor.a5 import *

# ----------------------------------------------------------------------------------------------------
# flat.py
# FLATTENED variant of qk_softmax_pv_pipe: the (M-tile, K-tile) loops are FUSED into ONE linear
# `step` loop so the lag-2 software pipeline carries ACROSS M-tile boundaries -- M-tile B's QK ramp-up
# overlaps M-tile A's PV drain, removing the per-M-tile ramp/drain bubbles (2 QK-only + 2 PV-only trips
# each) that the nested version pays at every M-tile. Requires k_tiles >= 2 (lag-2 needs >=2 K-tiles per
# M-tile, and the m%2 state double-buffering needs adjacent M-tiles to differ in parity). To let A's
# drain overlap B's ramp, the per-M-tile state (rsum, accum, out, l1q) is double-buffered by M-tile
# parity; rmax stays single (only stage1 touches it, and stage1 is strictly serial in step).
#
# Full PFA forward (QK + transposed softmax + PV), KV-major, 2-AIV, LAG-2 lookahead.
#
# Builds on qk_softmax_pipe (KV-major QK = K@Q^T, lane-wise vmax softmax) + the validated PV recipe
# from pfa_full_tn128_transpose / pv_probe (P^T [129,64] pack, .nz() column-publish, PV = matmul(l1p.T,
# l1v.T)). LAG-2: stage1 (QK+softmax+publish P[n]) runs 2 tiles ahead of stage2 (PV[n-2]+accum), per
# the user's "pre-compute 2 QK tiles before PV". Buffers that span the 2-trip gap (l1p, oldw, blkw) are
# TBuff(3); p_mutex depth=3. After the K-loop: normalize O/rsum -> bf16 -> GM.
#
# Dense (non-causal). Handles K-tail (N % TILE_N != 0) via softmax_t_tail_vf (select invalid keys to
# NEG -> P=0) + valid-K PV contraction, and M-tail (M % TILE_M != 0) via valid_m/local_rows store-cropping.
#
# MQA + batch: q is [B*MQ, D] with MQ=HQ*SQ query rows per batch (HKV=1: heads flattened along rows,
# sharing the batch's KV); k/v are [B*N, D] with N=SKV keys per batch. KV is SHARED across heads but
# INDEPENDENT per batch -- so a query tile may straddle head boundaries (when SQ % TILE_M != 0) yet
# never batch boundaries; each tile's KV comes from its own batch via kv_base. The (batch, local M-tile)
# pairs are linearized to one index for core distribution. dense only (causal masking would break the
# cross-head-tile sharing).
# ----------------------------------------------------------------------------------------------------

TILE_M = 128                      # query rows per cube block (2 AIV sub-blocks)
ROWS_PER_SB = 64                  # query rows per AIV sub-block (= QLANES)
QLANES = ROWS_PER_SB
TILE_N = 128                      # keys per K-tile
D_HEAD = 128
CHUNKS_D = D_HEAD // 64           # 2 fp32 regs per O row
SOFTMAX_SCALE = 1.0 / math.sqrt(D_HEAD)
NEG_LARGE = -1.0e30
NZ_C0 = 16
HALF = TILE_N // 2                # 64: pair key i with key i+64 (official 2-key pack: 64 VDINTLV/VSSTB)
FRAC_STRIDE = HALF + 1            # 65: ub_p is [65, 128] (2 keys per row, query inner)
UNROLL = 4                        # 4-way unroll: 4 independent max/sum partial chains (recover ILP)
RB_MAX = TILE_N // UNROLL         # 32: max pass over 128 keys
RB_EXP = HALF // UNROLL           # 16: exp pass over 64 pairs (i, i+64)
PRELOAD_N = 2                     # lag-2 lookahead


# ---- softmax / accum / finalize VFs ------------------------------------------
@vf()
def init_softmax_state_vf(ub_rmax: Tensor, ub_rsum: Tensor):
    neg = Reg(DT.float); zero = Reg(DT.float)
    neg <<= NEG_LARGE; zero <<= 0.0
    ub_rmax[0:1, 0:QLANES] <<= neg
    ub_rsum[0:1, 0:QLANES] <<= zero


@vf()
def softmax_t_vf(ub_score: Tensor, ub_rmax: Tensor, ub_rsum: Tensor, ub_p: Tensor,
                 ub_old_weight: Tensor):
    """Online softmax over [key=128, query=64] score^T, OFFICIAL single-weight (new_max) scheme:
    P^T = exp(scale*s - new_m) (folds the block weight into P -> accum is just O*old_scale + PV, one
    weight). 2-key pack: pair key i with i+64, deinterleave into one 128-lane reg (64 VDINTLV/VSSTB),
    VSSTB to ub_p[65,128]; published via 2 MTE3 (key-halves). new_m computed BETWEEN the max and exp
    passes so pass 2 exps against the running max directly."""
    sreg = RegList(DT.float, UNROLL); acc = RegList(DT.float, UNROLL); preg = RegList(DT.float, UNROLL)
    h_i = RegList(DT.bfloat16, UNROLL); h_j = RegList(DT.bfloat16, UNROLL)
    pk = Reg(DT.bfloat16); pk_hi = Reg(DT.bfloat16)
    neg = Reg(DT.float); block_max = Reg(DT.float); block_sum = Reg(DT.float); t = Reg(DT.float)
    v_pmax = Reg(DT.float); v_nmax = Reg(DT.float); v_oldw = Reg(DT.float)
    v_psum = Reg(DT.float); v_nsum = Reg(DT.float)
    # pass 1: per-query max over the 128 keys -- 4 independent partial-max chains (on RAW scores), tree-combine
    neg <<= NEG_LARGE
    for u in unroll(UNROLL):
        acc[u] <<= neg
    for rb in range(RB_MAX):
        for u in unroll(UNROLL):
            sreg[u] <<= ub_score[(rb * UNROLL + u):(rb * UNROLL + u) + 1, :]
            acc[u] <<= acc[u].vmax(sreg[u])
    block_max <<= acc[0].vmax(acc[1]); t <<= acc[2].vmax(acc[3]); block_max <<= block_max.vmax(t)
    muls(block_max, block_max, SOFTMAX_SCALE)            # scale the max ONCE (max(scale*s)=scale*max(s))
    # online max (between passes): new_m = max(block_max, old_m); old_scale = exp(old_m - new_m)
    v_pmax <<= ub_rmax[0:1, 0:QLANES]
    v_nmax <<= block_max.vmax(v_pmax)
    expsub(v_oldw, v_pmax, v_nmax)
    # pass 2: exp(scale*s - new_m) + per-query sum (4 partial-sum chains) + 2-key pack (i paired with i+HALF)
    for u in unroll(UNROLL):
        acc[u] <<= 0.0
    for rb in range(RB_EXP):
        for u in unroll(UNROLL):                         # the "i" keys (0..63)
            sreg[u] <<= ub_score[(rb * UNROLL + u):(rb * UNROLL + u) + 1, :]
            muls(sreg[u], sreg[u], SOFTMAX_SCALE)
            expsub(preg[u], sreg[u], v_nmax)
            acc[u] <<= acc[u] + preg[u]
            h_i[u] <<= preg[u].astype(DT.bfloat16)
        for u in unroll(UNROLL):                         # the "i+64" keys
            sreg[u] <<= ub_score[(rb * UNROLL + u + HALF):(rb * UNROLL + u + HALF) + 1, :]
            muls(sreg[u], sreg[u], SOFTMAX_SCALE)
            expsub(preg[u], sreg[u], v_nmax)
            acc[u] <<= acc[u] + preg[u]
            h_j[u] <<= preg[u].astype(DT.bfloat16)
        for u in unroll(UNROLL):
            deinterleave(pk, pk_hi, h_i[u], h_j[u])       # pk = [key_i 64q, key_(i+64) 64q]
            reg_to_ub(ub_p[(rb * UNROLL + u) * NZ_C0], pk, FRAC_STRIDE)
    block_sum <<= acc[0] + acc[1]; t <<= acc[2] + acc[3]; block_sum <<= block_sum + t
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)
    # online sum: new_l = old_l*old_scale + block_sum (block_sum already exp'd against new_m -> no blkw)
    v_psum <<= ub_rsum[0:1, 0:QLANES]
    v_nsum <<= v_psum * v_oldw
    v_nsum <<= v_nsum + block_sum
    ub_rmax[0:1, 0:QLANES] <<= v_nmax
    ub_rsum[0:1, 0:QLANES] <<= v_nsum
    ub_old_weight[0:1, 0:QLANES] <<= v_oldw
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


# Measured: the per-row scalar clamp this replaced was pinning the pypto path at ratio 0.63 on
# any K axis that is not a multiple of 128.
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
    """Tail variant for a PARTIAL K-tile (valid_n < TILE_N): invalid key rows (n >= valid_n, whose
    QK score is stale/garbage) are NEG-filled by neg_fill_rows before this vf runs, so they are NEG at the lane-wise max/exp, so they drop out of
    the per-query max and give P=exp(NEG-new_m)=0 (no sum/PV contribution). Non-unrolled (only the last
    K-tile per query-tile uses it); same single-weight new_m scheme + 2-key (i,i+64) pack as the main VF."""
    s = Reg(DT.float); e = Reg(DT.float); neg = Reg(DT.float)
    block_max = Reg(DT.float); block_sum = Reg(DT.float)
    h_i = Reg(DT.bfloat16); h_j = Reg(DT.bfloat16); pk = Reg(DT.bfloat16); dummy = Reg(DT.bfloat16)
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
    for i in range(HALF):
        s <<= ub_score[i:i + 1, :]
        s <<= s * SOFTMAX_SCALE
        expsub(e, s, v_nmax)
        block_sum <<= block_sum + e
        h_i <<= e.astype(DT.bfloat16)
        s <<= ub_score[i + HALF:i + HALF + 1, :]
        s <<= s * SOFTMAX_SCALE
        expsub(e, s, v_nmax)
        block_sum <<= block_sum + e
        h_j <<= e.astype(DT.bfloat16)
        deinterleave(pk, dummy, h_i, h_j)
        reg_to_ub(ub_p[i * NZ_C0], pk, FRAC_STRIDE)
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
    """Single-weight online rescale: O[r] = O[r]*old_scale[r] + PV[r] (P already exp'd against the
    running max in softmax, so PV needs no block weight)."""
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
    """First PV block: O = PV (no prior O to rescale; P already exp'd against new_m)."""
    pv = RegList(DT.float, CHUNKS_D)
    for r in range(ROWS_PER_SB):
        pv <<= ub_pv[r:r + 1, 0:D_HEAD]
        ub_accum[r:r + 1, 0:D_HEAD] <<= pv
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def final_div_cast_bf16_vf(ub_accum: Tensor, ub_rsum: Tensor, ub_out: Tensor):
    """O = ub_accum / ub_rsum (per-query), cast fp32 -> bf16."""
    acc = RegList(DT.float, CHUNKS_D); rsum = Reg(DT.float)
    for r in range(ROWS_PER_SB):
        rsum <<= ub_rsum[0:1, r:r + 1].single()
        acc <<= ub_accum[r:r + 1, 0:D_HEAD]
        acc <<= acc / rsum
        ub_out[r:r + 1, 0:D_HEAD] <<= acc
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


# ---- kernel (LAG-2) ----------------------------------------------------------
@kernel()
def qk_softmax_pv_flat_kernel(q: GM[bf16, ('TQ', 'D')], k: GM[bf16, ('TK', 'D')], v: GM[bf16, ('TK', 'D')], out: GM[bf16, ('TQ', 'D')],
                         B: i32, MQ: i32, N: i32, D: i32):
    # MQA + batch (same tensor contract as qk_softmax_pv_pipe): q [B*MQ, D] (MQ=HQ*SQ, heads flattened,
    # KV shared within a batch), k/v [B*N, D] (KV independent per batch). FLATTENED scheduling: one
    # linear `step` loop covers all (M-tile, K-tile) work for this core; stage1 runs at `step`, stage2
    # (PV) lags by PRELOAD_N so the pipeline spans M-tile boundaries.
    cv = CvMutex(0, depth=2, src_start_pipe=Pipe.FIX, dst_start_pipe=Pipe.V,
                 src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    p_mutex = VcMutex(1, depth=3, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE1)   # vec P-publish -> cube PV (lag-2)
    pv_mutex = CvMutex(2, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)        # cube PV -> vec accum

    l1q = DBuff(DT.bfloat16, [TILE_M, D_HEAD], Position.L1)           # ping-pong by M-tile parity
    l1k = DBuff(DT.bfloat16, [TILE_N, D_HEAD], Position.L1)           # ping-pong by step
    l1v = DBuff(DT.bfloat16, [TILE_N, D_HEAD], Position.L1)
    l1p = TBuff(DT.bfloat16, [TILE_N, TILE_M], Position.L1)            # P^T [key,query], lag-2 ring (wi%3)
    l0c = DBuff(DT.float, [TILE_M, TILE_M], Position.L0C)             # QK -> slot0, PV -> slot1 (l0c_cnt)
    ub_score = DBuff(DT.float, [TILE_N, ROWS_PER_SB], Position.UB)    # per sub-block [key=128, query=64]
    ub_p = Tensor(DT.bfloat16, [FRAC_STRIDE, 2 * ROWS_PER_SB], Position.UB)  # [65,128] 2-key pack, single-buffer
    ub_pv = DBuff(DT.float, [ROWS_PER_SB, D_HEAD], Position.UB)
    ub_accum = DBuff(DT.float, [ROWS_PER_SB, D_HEAD], Position.UB)    # O accumulator, double-buffered by M-tile parity
    ub_out = DBuff(DT.bfloat16, [ROWS_PER_SB, D_HEAD], Position.UB)   # double-buffered by M-tile parity
    ub_rmax = Tensor(DT.float, [1, QLANES], Position.UB)             # single (only stage1, strictly serial in step)
    ub_rsum = DBuff(DT.float, [1, QLANES], Position.UB)              # double-buffered by M-tile parity (finalize lags)
    ub_old_weight = TBuff(DT.float, [1, QLANES], Position.UB)         # oldw ring: softmax[n] -> accum[n] (lag-2, wi%3)

    row_begin = Var(GetSubBlockIdx() * ROWS_PER_SB)
    tiles_m_per_b = CeilDiv(MQ, TILE_M)                  # query-tiles per batch (heads flattened)
    k_tiles = CeilDiv(N, TILE_N)
    total_tiles = Var(B * tiles_m_per_b)                 # linearize (batch, local M-tile) -> one index
    tiles_per_core = CeilDiv(total_tiles, GetCubeNum())
    gt_begin = Var(tiles_per_core * GetCubeIdx())
    gt_end = Min(gt_begin + tiles_per_core, total_tiles)
    my_tiles = Var(gt_end - gt_begin)                    # this core's M-tile count
    my_work = Var(my_tiles * k_tiles)                    # this core's stage1 work units (M-tile x K-tile)

    # PV contracts only valid V rows, avoiding an overlapping fill/load writer pair.


    with auto_sync():
        l0c_cnt = Var(0)                                 # global QK/PV L0C ring (QK then PV each step)
        for step in range(0, my_work + PRELOAD_N):
            if step < my_work:
                # === STAGE 1 @ wi1=step: QK^T -> SPLITN softmax -> P^T publish ===
                wi1 = Var(step)
                m1 = Var(wi1 // k_tiles)                          # local M-tile index (0..my_tiles-1)
                k1 = Var(wi1 % k_tiles)                           # K-tile within the M-tile
                gt1 = Var(gt_begin + m1)
                q_base1 = Var((gt1 // tiles_m_per_b) * MQ)
                kv_base1 = Var((gt1 // tiles_m_per_b) * N)
                m0_1 = Var((gt1 % tiles_m_per_b) * TILE_M)
                valid_m1 = Min(TILE_M, MQ - m0_1)
                n0 = Var(k1 * TILE_N)
                valid_n = Min(TILE_N, N - n0)
                if k1 == 0:                                       # new M-tile: load Q, reset softmax state
                    l1q[m1][0:valid_m1, 0:D] <<= q[q_base1 + m0_1:q_base1 + m0_1 + valid_m1, 0:D]
                    init_softmax_state_vf(ub_rmax, ub_rsum[m1])
                l1k[step][0:valid_n, 0:D] <<= k[kv_base1 + n0:kv_base1 + n0 + valid_n, 0:D]
                matmul(l0c[l0c_cnt], l1k[step], l1q[m1],
                       m=TILE_N, n=TILE_M, k=D_HEAD, is_init=True)
                cv.lock()
                l0c_to_ub(ub_score[step], l0c[l0c_cnt][0:TILE_N, 0:TILE_M],
                          M=TILE_N, N=TILE_M, N_dst=ROWS_PER_SB, M_src=TILE_N,
                          dual_mode=DualMode.SPLITN, sub_block_id=0)
                cv.ready()
                cv.wait()
                if valid_n < TILE_N:
                    neg_fill_rows(ub_score[step], valid_n)
                    softmax_t_tail_vf(ub_score[step], ub_rmax, ub_rsum[m1], ub_p,
                                      ub_old_weight[wi1])
                else:
                    softmax_t_vf(ub_score[step], ub_rmax, ub_rsum[m1], ub_p,
                                 ub_old_weight[wi1])
                cv.free()
                p_mutex.lock()
                l1p[wi1][0:HALF, row_begin:row_begin + ROWS_PER_SB] <<= (
                    ub_p.nz()[0:HALF, 0:ROWS_PER_SB])                          # keys 0-63
                l1p[wi1][HALF:TILE_N, row_begin:row_begin + ROWS_PER_SB] <<= (
                    ub_p.nz()[0:HALF, ROWS_PER_SB:2 * ROWS_PER_SB])            # keys 64-127
                p_mutex.ready()
                l0c_cnt += 1

            if step >= PRELOAD_N:
                # === STAGE 2 @ wi2=step-PRELOAD_N (lag-2): PV -> accum, finalize at the M-tile's last K ===
                wi2 = Var(step - PRELOAD_N)
                m2 = Var(wi2 // k_tiles)
                k2 = Var(wi2 % k_tiles)
                gt2 = Var(gt_begin + m2)
                q_base2 = Var((gt2 // tiles_m_per_b) * MQ)
                kv_base2 = Var((gt2 // tiles_m_per_b) * N)
                m0_2 = Var((gt2 % tiles_m_per_b) * TILE_M)
                valid_m2 = Min(TILE_M, MQ - m0_2)
                local_rows2 = Min(ROWS_PER_SB, Max(valid_m2 - row_begin, 0))
                n0_2 = Var(k2 * TILE_N)
                v_rows = Min(TILE_N, N - n0_2)
                l1v[step][0:v_rows, 0:D] <<= v[kv_base2 + n0_2:kv_base2 + n0_2 + v_rows, 0:D]
                p_mutex.wait()
                matmul(l0c[l0c_cnt][0:TILE_M, 0:D_HEAD], l1p[wi2].T, l1v[step].T,
                       m=TILE_M, n=D_HEAD, k=v_rows, is_init=True)   # no-split: k=128 fits L0
                p_mutex.free()
                pv_mutex.lock()
                l0c_to_ub(ub_pv[step], l0c[l0c_cnt][0:TILE_M, 0:D_HEAD],
                          M=TILE_M, N=D_HEAD, N_dst=D_HEAD, M_src=TILE_M,
                          dual_mode=DualMode.SPLITM, sub_block_id=0)
                pv_mutex.ready()
                pv_mutex.wait()
                if k2 == 0:
                    accum_pv_first_vf(ub_accum[m2], ub_pv[step])
                else:
                    accum_pv_vf(ub_accum[m2], ub_pv[step], ub_old_weight[wi2])
                pv_mutex.free()
                l0c_cnt += 1
                if k2 == k_tiles - 1:                             # M-tile complete: normalize + store
                    final_div_cast_bf16_vf(ub_accum[m2], ub_rsum[m2], ub_out[m2])
                    if local_rows2 > 0:
                        out[q_base2 + m0_2 + row_begin:q_base2 + m0_2 + row_begin + local_rows2, 0:D] <<= (
                            ub_out[m2][0:local_rows2, 0:D])
    return out


# ----------------------------------------------------------------------------------------------------
# fd.py
# Flash-decoding style stream-K variant of the replicated PFA forward.
#
# This keeps the same math contract as qk_softmax_pv_flat.py:
#   q   : [B*MQ, D] bf16   with MQ = HQ*SQ per batch
#   k/v : [B*N,  D] bf16   one KV stream per batch, shared by all heads in that batch
#   out : [B*MQ, D] bf16
#
# Change vs qk_softmax_pv_flat.py:
#   - work is sharded across the flat (global_m_tile, k_tile) grid instead of only across M tiles
#   - each core owns one contiguous flat interval [flat_start, flat_end)
#   - if a core boundary lands inside an M tile, that tile is split across two cores only
#   - each split tile writes an unnormalized partial (accum, row-max, row-sum) to GM workspace
#   - after a vec-wide barrier, two aggregation VFs merge the pair of partials
#
# This implementation assumes the chosen flat interval schedule splits each M tile across at most
# two cores; every block_dim in main.py's CASES satisfies that.
# ----------------------------------------------------------------------------------------------------

# agg_vf.py refers to M_Q / CD verbatim; here they equal QLANES / CHUNKS_D.
M_Q = QLANES                      # 64 query lanes per sub-block
CD = CHUNKS_D                     # 2 fp32 regs per O row

CACHE = PRELOAD_N + 1
MAX_CORE_COUNT = 32
MAX_FD_WS = MAX_CORE_COUNT * 2


# ---- flash-decoding aggregation VFs (inlined from agg_vf.py) ------------------
@vf()
def fd_merge_vf(ub_o0: Tensor, ub_o1: Tensor, ub_m0: Tensor, ub_m1: Tensor,
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
def output_cast_vf(ub_merged: Tensor, ub_out: Tensor):
    """AGG-2: fp32 -> bf16 cast of the merged O, per D-chunk, then downsample-pack store.

    Official d3b4 also carries RV_PLT 128 (B32) right after each load: a float-SOURCE
    active-lane mask (UpdateMask<float>) for the downsampling cast (NOT a saturation clamp
    -- bf16 has fp32's exponent range). easyasc's `cast(dst_bf16, src_float, mask)` resolves
    the mask against the bf16 DST lanes, so the B32 float-source mask cannot be passed through
    the cast here; this VF therefore matches RV_VLD/RV_VCVT_F2F/RV_VST but not the 128 RV_PLT."""
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


@kernel()
def qk_softmax_pv_fd_kernel(q: GM[bf16, ('TQ', 'D')], k: GM[bf16, ('TK', 'D')], v: GM[bf16, ('TK', 'D')], out: GM[bf16, ('TQ', 'D')],
                         B: i32, MQ: i32, N: i32, D: i32):
    cv = CvMutex(0, depth=2, src_start_pipe=Pipe.FIX, dst_start_pipe=Pipe.V,
                 src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    pv_mutex = CvMutex(2, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)

    l1q = DBuff(DT.bfloat16, [TILE_M, D_HEAD], Position.L1)
    l1k = TBuff(DT.bfloat16, [TILE_N, D_HEAD], Position.L1)
    l1v = TBuff(DT.bfloat16, [TILE_N, D_HEAD], Position.L1)
    l1p = TBuff(DT.bfloat16, [TILE_N, TILE_M], Position.L1)
    l0c_qk = DBuff(DT.float, [TILE_M, TILE_M], Position.L0C)
    l0c_pv = DBuff(DT.float, [TILE_M, D_HEAD], Position.L0C)

    ub_score = DBuff(DT.float, [TILE_N, ROWS_PER_SB], Position.UB)
    ub_pv = DBuff(DT.float, [ROWS_PER_SB, D_HEAD], Position.UB)
    ub_p = Tensor(DT.bfloat16, [FRAC_STRIDE, 2 * ROWS_PER_SB], Position.UB)
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

    # Valid P cells are fully published; PV reads only the loaded V rows.

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
                    l1q[qm][0:valid_m, 0:D] <<= q[q_base + m0:q_base + m0 + valid_m, 0:D]
                    init_softmax_state_vf(ub_rmax[sm], ub_rsum[sm])

                l1k[step][0:valid_n, 0:D] <<= k[kv_base + n0:kv_base + n0 + valid_n, 0:D]
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

                l1p[step][0:HALF, row_begin:row_begin + ROWS_PER_SB] <<= ub_p.nz()[0:HALF, 0:ROWS_PER_SB]
                l1p[step][HALF:TILE_N, row_begin:row_begin + ROWS_PER_SB] <<= ub_p.nz()[0:HALF, ROWS_PER_SB:2 * ROWS_PER_SB]
                vec_ready(1, Pipe.MTE3)

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

                l1v[lag][0:v_rows, 0:D] <<= v[lag_kv_base + lag_n0:lag_kv_base + lag_n0 + v_rows, 0:D]
                wait_vec(1, Pipe.S)
                matmul(l0c_pv[lag][0:TILE_M, 0:D_HEAD], l1p[lag].T, l1v[lag].T,
                       m=TILE_M, n=D_HEAD, k=v_rows, is_init=True)

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
                    accum_pv_vf(ub_accum, ub_pv[lag], ub_old_weight[lag])
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
    # body, so every vec that enters the region runs the identical synced sequence (the same
    # shape auto_sync handles cleanly in agg_probe_kernel) instead of a data-dependent one.
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
                fd_merge_vf(ub_pv[0], ub_pv[1], ub_rmax[0], ub_rmax[1],
                            ub_rsum[0], ub_rsum[1], ub_old_weight[0], ub_old_weight[1], ub_pv[0])
                output_cast_vf(ub_pv[0], ub_out)
                out[fd_q_base + fd_m0 + fd_row0:fd_q_base + fd_m0 + fd_row0 + fd_rows, 0:D] <<= (
                    ub_out[0:fd_rows, 0:D]
                )
    return out
