# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""GQA prefill with HiFloat8 Q/K on a stream-K plus Flash-Decode schedule the host hands it.

Q and K arrive as uint8 carriers holding HiFloat8 values and are reinterpreted to DT.hif8 inside
the kernel for the QK^T matmul; V, P, the softmax state, the Flash-Decode workspace and the
output all stay on the bf16/fp32 path. The score tile is computed transposed (score^T = K @ Q^T),
so the query rows are its N axis -- which is why its L0C drain is SPLITN while the PV drain is
the default SPLITM.

One contiguous M axis: Hk=1, so every query head shares the single KV head and M = Hq*S1 is one
row axis with no per-row head decode. Each AIC core owns a contiguous interval [flat_start,
flat_end) of the flat (m_block-major, s2_tile-inner) grid, so a single m-block's key range may be
split across two cores; those two write fp32 partials (unnormalised O, row max, row sum) to a GM
workspace instead of finalising, and after an AIV-wide barrier two AIV merge jobs per split block
recombine them with a max-aware log-sum-exp.

Nothing about the grid is decided on the device: the schedule is precomputed on the host and
passed in as the int32 fa_meta/fd_meta tensors, which the kernel reads per core and per vector
lane with Var.GetValueFrom. The generator that builds them is in reference.py."""

import math

from ascriptor.a5 import *

TILE_M = 128
TILE_N = 128
D_HEAD = 128
ROWS_SB = TILE_M // 2     # 64 vec sub-block rows
CD = D_HEAD // 64         # 2 fp32 regs cover a 128-col D row
C0 = 16                   # bf16 NZ C0 block
UNROLL_F = 4             # P1 const-trip softmax unroll (ROWS_SB//UNROLL_F runtime iters x unroll(UNROLL_F))

PRELOAD_N = 2             # FIA lag-2
CACHE = PRELOAD_N + 1     # 3-deep task cache

IO_DT = DT.bfloat16
NEG_LARGE = -1.0e30

# --- transposed-softmax constants (score^T = K@Q^T -> [key, query]; lane-wise vmax) ---
QLANES = ROWS_SB              # 64 query lanes per fp32 reg (query is the LANE axis after transpose)
HALF = TILE_N // 2            # 64: 2-key pack (key i paired with i+64 -> one 128-lane bf16 reg)
FRAC_STRIDE = HALF + 1        # 65: ub_p [65,128] padded NZ stride (odd -> dodges UB 2^k bank conflict)
RB_MAX = TILE_N // UNROLL_F   # 32: max-pass over 128 keys (4-way unrolled partial-max chains)
RB_EXP = HALF // UNROLL_F     # 16: exp-pass over 64 key-pairs (i, i+64)
SOFTMAX_SCALE = 1.0 / math.sqrt(D_HEAD)

# Metadata field layouts (1-D int32 GM tensors).
FA_NF = 6     # [enable, flat_start, flat_end, tail_ws, head_ws, pad]
FD_NF = 6     # [enable, m_idx, ws_lo, ws_hi, row0, nrows]

# Workspace slot count (= 2 * num_fd_tasks). Set by run_case before tracing.
NWS_GLOBAL = 64  # Bounded by two workspace slots per active split; at most32 launch cores.


# The host metadata generator that fills fa_meta/fd_meta lives in reference.py; nothing below
# decides a grid, it only reads the one it was handed.
# ============================ vec stages ============================
@vf()
def init_state(ub_rmax: Tensor, ub_rsum: Tensor):
    """rowmax=-inf, rowsum=0 for one m-block's running softmax state (64-wide)."""
    neg = Reg(DT.float)
    zero = Reg(DT.float)
    neg <<= NEG_LARGE
    zero <<= 0.0
    ub_rmax[0:1, 0:ROWS_SB] <<= neg
    ub_rsum[0:1, 0:ROWS_SB] <<= zero


@vf()
def softmax_t_vf(ub_score: Tensor, ub_rmax: Tensor, ub_rsum: Tensor, ub_p: Tensor,
                 ub_old_weight: Tensor):
    """TRANSPOSED online softmax over [key=128, query=64] score^T, single-weight (new-max) scheme:
    P^T = exp(scale*s - new_m) (block weight folded into P -> accum is just O*old_scale + PV). Lane-wise
    vmax over the 128 key ROWS gives the per-query max directly in lanes (no cmax/EQ-scatter). 2-key pack
    (key i with i+64 -> one 128-lane bf16 reg via deinterleave) -> ub_p[65,128]; new_m between the passes."""
    sreg = RegList(DT.float, UNROLL_F); acc = RegList(DT.float, UNROLL_F); preg = RegList(DT.float, UNROLL_F)
    h_i = RegList(DT.bfloat16, UNROLL_F); h_j = RegList(DT.bfloat16, UNROLL_F)
    pk = Reg(DT.bfloat16); pk_hi = Reg(DT.bfloat16)
    neg = Reg(DT.float); block_max = Reg(DT.float); block_sum = Reg(DT.float); t = Reg(DT.float)
    v_pmax = Reg(DT.float); v_nmax = Reg(DT.float); v_oldw = Reg(DT.float)
    v_psum = Reg(DT.float); v_nsum = Reg(DT.float)
    # pass 1: per-query max over 128 keys (4 independent partial-max chains on RAW scores), tree-combine
    neg <<= NEG_LARGE
    for u in unroll(UNROLL_F):
        acc[u] <<= neg
    for rb in range(RB_MAX):
        for u in unroll(UNROLL_F):
            sreg[u] <<= ub_score[(rb * UNROLL_F + u):(rb * UNROLL_F + u) + 1, :]
            acc[u] <<= acc[u].vmax(sreg[u])
    block_max <<= acc[0].vmax(acc[1]); t <<= acc[2].vmax(acc[3]); block_max <<= block_max.vmax(t)
    muls(block_max, block_max, SOFTMAX_SCALE)        # scale the max ONCE (max(scale*s)=scale*max(s))
    v_pmax <<= ub_rmax[0:1, 0:QLANES]
    v_nmax <<= block_max.vmax(v_pmax)
    expsub(v_oldw, v_pmax, v_nmax)                   # old_scale = exp(old_m - new_m)
    # pass 2: exp(scale*s - new_m) + per-query sum (4 partial-sum chains) + 2-key (i, i+64) pack
    for u in unroll(UNROLL_F):
        acc[u] <<= 0.0
    for rb in range(RB_EXP):
        for u in unroll(UNROLL_F):                   # the "i" keys (0..63)
            sreg[u] <<= ub_score[(rb * UNROLL_F + u):(rb * UNROLL_F + u) + 1, :]
            muls(sreg[u], sreg[u], SOFTMAX_SCALE)
            expsub(preg[u], sreg[u], v_nmax)
            acc[u] <<= acc[u] + preg[u]
            h_i[u] <<= preg[u].astype(DT.bfloat16)
        for u in unroll(UNROLL_F):                   # the "i+64" keys
            sreg[u] <<= ub_score[(rb * UNROLL_F + u + HALF):(rb * UNROLL_F + u + HALF) + 1, :]
            muls(sreg[u], sreg[u], SOFTMAX_SCALE)
            expsub(preg[u], sreg[u], v_nmax)
            acc[u] <<= acc[u] + preg[u]
            h_j[u] <<= preg[u].astype(DT.bfloat16)
        for u in unroll(UNROLL_F):
            deinterleave(pk, pk_hi, h_i[u], h_j[u])  # pk = [key_i 64q | key_(i+64) 64q]
            reg_to_ub(ub_p[(rb * UNROLL_F + u) * C0], pk, FRAC_STRIDE)
    block_sum <<= acc[0] + acc[1]; t <<= acc[2] + acc[3]; block_sum <<= block_sum + t
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)
    v_psum <<= ub_rsum[0:1, 0:QLANES]
    v_nsum <<= v_psum * v_oldw
    v_nsum <<= v_nsum + block_sum                    # block_sum already exp'd vs new_m -> no blkw
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
    """TRANSPOSED softmax for a PARTIAL K-tile (valid_n < TILE_N): invalid key ROWS (n >= valid_n,
    stale QK) are NEG-filled by neg_fill_rows before this vf runs, so they are NEG at the lane-wise max/exp, so they drop out of the per-query
    max and give P=exp(NEG-new_m)=0. Non-unrolled (only the last K-tile per query-tile uses it);
    same single-weight new_m scheme + 2-key (i, i+64) pack as softmax_t_vf."""
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
        reg_to_ub(ub_p[i * C0], pk, FRAC_STRIDE)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)
    v_psum <<= ub_rsum[0:1, 0:QLANES]
    v_nsum <<= v_psum * v_oldw
    v_nsum <<= v_nsum + block_sum
    ub_rmax[0:1, 0:QLANES] <<= v_nmax
    ub_rsum[0:1, 0:QLANES] <<= v_nsum
    ub_old_weight[0:1, 0:QLANES] <<= v_oldw
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def accum_pv_t(ub_accum: Tensor, ub_pv: Tensor, ub_old_w: Tensor, rows: Var):
    """TRANSPOSED single-weight online rescale: O[r] = O[r]*old_scale[r] + PV[r]. P is already exp'd
    against the running max in softmax_t, so PV needs NO block weight (one weight, simpler than the
    two-weight champion). rows = local query rows (const-trip processes all 64; tail rows contained)."""
    acc = RegList(DT.float, CD)
    pv_regs = RegList(DT.float, CD)
    old_w = Reg(DT.float)
    for rb in range(ROWS_SB // UNROLL_F):
        for ru in unroll(UNROLL_F):
            r = rb * UNROLL_F + ru
            old_w <<= ub_old_w[0:1, r:r + 1].single()
            acc <<= ub_accum[r:r + 1, :]
            pv_regs <<= ub_pv[r:r + 1, :]
            acc <<= acc * old_w
            acc <<= acc + pv_regs
            ub_accum[r:r + 1, :] <<= acc


@vf()
def accum_pv_first_t(ub_accum: Tensor, ub_pv: Tensor, rows: Var):
    """First owned K-tile: O = PV (no prior O to rescale; P already exp'd against new_m)."""
    pv_regs = RegList(DT.float, CD)
    for rb in range(ROWS_SB // UNROLL_F):
        for ru in unroll(UNROLL_F):
            r = rb * UNROLL_F + ru
            pv_regs <<= ub_pv[r:r + 1, :]
            ub_accum[r:r + 1, :] <<= pv_regs


@vf()
def finalize(ub_accum: Tensor, ub_rsum: Tensor, ub_out: Tensor, rows: Var):
    """out = accum / rowsum; bf16 cast on store. For non-split (fully owned) m-blocks."""
    regs = RegList(DT.float, CD)
    sum_reg = Reg(DT.float)
    for rb in range(ROWS_SB // UNROLL_F):    # P1 const-unroll
        for ru in unroll(UNROLL_F):
            r = rb * UNROLL_F + ru
            sum_reg <<= ub_rsum[0:1, r:r + 1].single()
            regs <<= ub_accum[r:r + 1, :]
            regs <<= regs / sum_reg
            ub_out[r:r + 1, :] <<= regs


@vf()
def fd_merge_t(ub_o0: Tensor, ub_o1: Tensor, ub_m0: Tensor, ub_m1: Tensor,
               ub_s0: Tensor, ub_s1: Tensor, ub_A: Tensor, ub_den: Tensor, ub_merged: Tensor):
    """2-split merge + normalize, LEAN form (== max-aware): A=exp(m0-m1), den_eff=s0*A+s1 precomputed
    VECTORIZED over the 64 query lanes; main loop O=(o0*A+o1)/den_eff per row."""
    m0v = Reg(DT.float); m1v = Reg(DT.float); s0v = Reg(DT.float); s1v = Reg(DT.float)
    Av = Reg(DT.float); denv = Reg(DT.float)
    m0v <<= ub_m0[0:1, 0:ROWS_SB]
    m1v <<= ub_m1[0:1, 0:ROWS_SB]
    s0v <<= ub_s0[0:1, 0:ROWS_SB]
    s1v <<= ub_s1[0:1, 0:ROWS_SB]
    expsub(Av, m0v, m1v)                          # A = exp(m0 - m1)
    denv <<= s0v * Av
    denv <<= denv + s1v                           # den_eff = s0*A + s1
    ub_A[0:1, 0:ROWS_SB] <<= Av
    ub_den[0:1, 0:ROWS_SB] <<= denv
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)
    o0 = RegList(DT.float, CD); o1 = RegList(DT.float, CD)
    Ar = Reg(DT.float); dr = Reg(DT.float)
    for r in range(ROWS_SB):
        Ar <<= ub_A[0:1, r:r + 1].single()
        dr <<= ub_den[0:1, r:r + 1].single()
        o0 <<= ub_o0[r:r + 1, :]
        o1 <<= ub_o1[r:r + 1, :]
        o0 <<= o0 * Ar
        o0 <<= o0 + o1
        o0 <<= o0 / dr
        ub_merged[r:r + 1, :] <<= o0                # O = (o0*A + o1)/den_eff (fp32)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def output_cast_t(ub_merged: Tensor, ub_out: Tensor):
    """fp32 merged O -> bf16, per D-chunk, downsample-pack store."""
    regs = RegList(DT.float, CD); hregs = RegList(DT.bfloat16, CD)
    cfg = CastConfig(round_mode=RoundMode.AWAY_FROM_ZERO, reg_layout=RegLayout.ZERO, name="fd_o_bf16")
    for r in range(ROWS_SB):
        regs <<= ub_merged[r:r + 1, :]
        for c in unroll(CD):
            cast(hregs[c], regs[c], cfg)
            reg_to_ub_downsample(ub_out[r:r + 1, c * 64:(c + 1) * 64], hregs[c])
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def copy_partial_t(dst_a: Tensor, dst_m: Tensor, dst_s: Tensor,
                   src_a: Tensor, src_m: Tensor, src_s: Tensor):
    """Capture the FD partial state (accum / row-max / row-sum) into DEDICATED fixed buffers via VEC
    reads, BEFORE the GM workspace store. The next m-block's init_state/accum_pv clobber the ring/single
    source buffers (ub_rmax[mt%CACHE] / single ub_accum); routing through a vec read makes that a
    vec-vec WAR on a single/fixed buffer (auto_sync tracks it -> the clobber is ordered after this read),
    and the MTE3 store then reads the stable copy. Root fix for the FD partial-clobber 0.808/NaN
    (the runtime-ring-index WAR the autosync tracker missed)."""
    mv = Reg(DT.float)
    sv = Reg(DT.float)
    mv <<= src_m[0:1, 0:ROWS_SB]
    dst_m[0:1, 0:ROWS_SB] <<= mv
    sv <<= src_s[0:1, 0:ROWS_SB]
    dst_s[0:1, 0:ROWS_SB] <<= sv
    a = RegList(DT.float, CD)
    for r in range(ROWS_SB):
        a <<= src_a[r:r + 1, :]
        dst_a[r:r + 1, :] <<= a
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


# ============================ kernel ============================
@kernel()
def fia_gqa_step3_transpose_kernel(q: GM[u8, ('M', 'D')], k: GM[u8, ('S2', 'D')], v: GM[bf16, ('S2', 'D')],
                         fa_meta: GM[i32, ('FA_META',)], fd_meta: GM[i32, ('FD_META',)], out: GM[bf16, ('M', 'D')],
                         M: i32, S2: i32, D: i32, scale: f32, tiles_n: i32):
    q_hif8 = q.reinterpret(DT.hif8, name="q_hif8")
    k_hif8 = k.reinterpret(DT.hif8, name="k_hif8")

    # Compact sync: start_pipe == end_pipe so each side's wait sits on the SAME pipe as its set
    # (FIA's intra-block handoff). Default start=Pipe.S routed the wait through scalar -> camodel
    # WAIT_FLAG_CUBE/VEC stalls; matching to FIX/V keeps it WAIT_INTRA_BLOCK (vec runs dense).
    qk_mutex = CvMutex(0, depth=2, src_start_pipe=Pipe.FIX, dst_start_pipe=Pipe.V,
                       src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)   # AIC->AIV BMM1 (BOTH)
    pv_mutex = CvMutex(2, depth=2, src_start_pipe=Pipe.FIX, dst_start_pipe=Pipe.V,
                       src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)   # AIC->AIV BMM2 (BOTH)
    # p (flag 1): FORWARD-only L1-P ring -- raw vec_ready/wait_vec, no backward.
    # flag 6: AIV-wide Flash-Decode barrier (allvec, == official SyncAll).

    l1q = DBuff(DT.hif8, [TILE_M, D_HEAD], Position.L1)
    l1k = TBuff(DT.hif8, [TILE_N, D_HEAD], Position.L1)
    l1v = TBuff(IO_DT, [TILE_N, D_HEAD], Position.L1)
    l1p = TBuff(IO_DT, [TILE_N, TILE_M], Position.L1)          # P^T [key=128, query=128], column-published per sub-block
    l0c_qk = DBuff(DT.float, [TILE_M, TILE_M], Position.L0C)   # QK^T [key, query] = [128,128]
    l0c_pv = DBuff(DT.float, [TILE_M, D_HEAD], Position.L0C)

    ub_score = DBuff(DT.float, [TILE_N, ROWS_SB], Position.UB)    # score^T per sub-block [key=128, query=64] fp32 (SPLITN)
    ub_pv = DBuff(DT.float, [ROWS_SB, D_HEAD], Position.UB)
    ub_p = Tensor(IO_DT, [FRAC_STRIDE, 2 * ROWS_SB], Position.UB)  # 2-key NZ P^T pack [65,128], single-buffer
    ub_old_w = TBuff(DT.float, [1, 64], Position.UB)
    ub_rmax = TBuff(DT.float, [1, 64], Position.UB)
    ub_rsum = TBuff(DT.float, [1, 64], Position.UB)
    ub_accum = Tensor(DT.float, [ROWS_SB, D_HEAD], Position.UB)
    ub_out = Tensor(IO_DT, [ROWS_SB, D_HEAD], Position.UB)

    # Dedicated FD-partial capture buffers (fixed, NOT ring-indexed): copy the partial state here
    # before the GM store so a later m-block's ring/accum clobber can't corrupt the in-flight store.
    ub_accum_pfx = Tensor(DT.float, [ROWS_SB, D_HEAD], Position.UB)
    ub_rmax_pfx = Tensor(DT.float, [1, 64], Position.UB)
    ub_rsum_pfx = Tensor(DT.float, [1, 64], Position.UB)

    # Flash-Decode partial workspace (fp32). nws baked per config.
    fd_accum = split_workspace(DT.float, [NWS_GLOBAL, TILE_M, D_HEAD], name="fd_accum")
    fd_max = split_workspace(DT.float, [NWS_GLOBAL, 1, TILE_M], name="fd_max")
    fd_sum = split_workspace(DT.float, [NWS_GLOBAL, 1, TILE_M], name="fd_sum")

    core = Var(GetCubeIdx())
    vec = Var(GetVecIdx())
    row_begin = Var(GetSubBlockIdx() * ROWS_SB)

    # --- per-core stream-K metadata (GetValueFrom, ace_v13 pattern) ---
    fa_base = Var(core * FA_NF)
    flat_start = Var(0, DT.int)
    flat_start.GetValueFrom(fa_meta[fa_base + 1:fa_base + 2])
    flat_end = Var(0, DT.int)
    flat_end.GetValueFrom(fa_meta[fa_base + 2:fa_base + 3])
    tail_ws = Var(0, DT.int)
    tail_ws.GetValueFrom(fa_meta[fa_base + 3:fa_base + 4])
    head_ws = Var(0, DT.int)
    head_ws.GetValueFrom(fa_meta[fa_base + 4:fa_base + 5])
    num_tasks = Var(Max(flat_end - flat_start, 0))
    first_block_m = Var(flat_start // tiles_n)
    last_block_m = Var((flat_end - 1) // tiles_n)

    with auto_sync():
        for loop in range(0, num_tasks + PRELOAD_N):
            # ===== current task: BMM1 -> two-weight softmax -> P-NZ -> L1 =====
            if loop < num_tasks:
                c1 = Var(loop % CACHE)
                d1 = Var(loop % 2)
                t1 = Var(flat_start + loop)
                mt1 = Var(t1 // tiles_n)
                s2_1 = Var(t1 % tiles_n)
                cm1 = Var(mt1 % CACHE)
                cq1 = Var(mt1 % 2)

                q_row1 = Var(mt1 * TILE_M)
                valid_m1 = Min(TILE_M, M - q_row1)
                local_m1 = Min(ROWS_SB, Max(valid_m1 - row_begin, 0))
                n_off1 = Var(s2_1 * TILE_N)
                valid_n1 = Min(TILE_N, S2 - n_off1)

                if s2_1 == 0 or loop == 0:                 # m-block start (may begin mid-s2)
                    l1q[cq1][0:valid_m1, 0:D] <<= q_hif8[q_row1:q_row1 + valid_m1, 0:D]
                    init_state(ub_rmax[cm1], ub_rsum[cm1])

                l1k[c1][0:valid_n1, 0:D] <<= k_hif8[n_off1:n_off1 + valid_n1, 0:D]
                # transposed QK: score^T = K @ Q^T -> l0c [key=TILE_N, query=TILE_M]
                matmul(l0c_qk[d1], l1k[c1], l1q[cq1], m=TILE_N, n=TILE_M, k=D_HEAD, is_init=True)

                qk_mutex.lock()
                # SPLITN: split the query (free) axis 64+64 across the two AIV sub-blocks; fp32->fp32 dual-dest
                # (no downcast -> cheap fixpipe, unlike the fp16 SINGLE-mode +82us path).
                l0c_to_ub(ub_score[d1], l0c_qk[d1][0:TILE_N, 0:TILE_M],
                          M=TILE_N, N=TILE_M, N_dst=ROWS_SB, M_src=TILE_N,
                          dual_mode=DualMode.SPLITN, sub_block_id=0)
                qk_mutex.ready()
                qk_mutex.wait()
                if valid_n1 < TILE_N:                          # tail K-tile: NEG the invalid key ROWS -> P=0
                    neg_fill_rows(ub_score[d1], valid_n1)
                    softmax_t_tail_vf(ub_score[d1], ub_rmax[cm1], ub_rsum[cm1], ub_p, ub_old_w[c1])
                else:                                          # full K-tile: lane-wise vmax, no mask
                    softmax_t_vf(ub_score[d1], ub_rmax[cm1], ub_rsum[cm1], ub_p, ub_old_w[c1])
                qk_mutex.free()

                # publish P^T into l1p: 2-key pack demux -> keys 0-63 and 64-127, this sub-block's query cols
                l1p[c1][0:HALF, row_begin:row_begin + ROWS_SB] <<= ub_p.nz()[0:HALF, 0:ROWS_SB]
                l1p[c1][HALF:TILE_N, row_begin:row_begin + ROWS_SB] <<= ub_p.nz()[0:HALF, ROWS_SB:2 * ROWS_SB]
                vec_ready(1, Pipe.MTE3)

            # ===== lagged task: BMM2 -> accumulate -> finalize / FD-partial =====
            if loop >= PRELOAD_N:
                lag = Var(loop - PRELOAD_N)
                c2 = Var(lag % CACHE)
                d2 = Var(lag % 2)
                t2 = Var(flat_start + lag)
                mt2 = Var(t2 // tiles_n)
                s2_2 = Var(t2 % tiles_n)
                cm2 = Var(mt2 % CACHE)

                q_row2 = Var(mt2 * TILE_M)
                valid_m2 = Min(TILE_M, M - q_row2)
                local_m2 = Min(ROWS_SB, Max(valid_m2 - row_begin, 0))
                n_off2 = Var(s2_2 * TILE_N)
                valid_n2 = Min(TILE_N, S2 - n_off2)

                l1v[c2][0:valid_n2, 0:D] <<= v[n_off2:n_off2 + valid_n2, 0:D]
                wait_vec(1, Pipe.MTE1)   # L1-P ring: cube waits on its consume pipe (MTE1) -> intra-block, not S/FFTS
                # transposed PV: O[query, D] = P @ V via BOTH operands .T (l1p holds P^T); k=valid_n2 (K-tail)
                matmul(l0c_pv[d2], l1p[c2].T, l1v[c2].T, m=TILE_M, n=D_HEAD, k=valid_n2, is_init=True)

                pv_mutex.lock()
                ub_pv[d2] <<= l0c_pv[d2]                   # SPLITM dual-lane fp32 (each AIV its 64 query rows)
                pv_mutex.ready()
                pv_mutex.wait()
                if s2_2 == 0 or lag == 0:                  # m-block first owned s2: O = PV (single-weight, no rescale)
                    accum_pv_first_t(ub_accum, ub_pv[d2], local_m2)
                else:                                      # O = O*old_scale + PV
                    accum_pv_t(ub_accum, ub_pv[d2], ub_old_w[c2], local_m2)

                if s2_2 == tiles_n - 1 or lag == num_tasks - 1:   # m-block last owned s2
                    if mt2 == first_block_m and tail_ws >= 0:     # tail (upper-s2) partial: no divide
                        if local_m2 > 0:
                            copy_partial_t(ub_accum_pfx, ub_rmax_pfx, ub_rsum_pfx,
                                           ub_accum, ub_rmax[cm2], ub_rsum[cm2])
                            fd_accum[tail_ws, row_begin:row_begin + local_m2, 0:D_HEAD] <<= ub_accum_pfx[0:local_m2, 0:D_HEAD]
                            fd_max[tail_ws, 0:1, row_begin:row_begin + local_m2] <<= ub_rmax_pfx[0:1, 0:local_m2]
                            fd_sum[tail_ws, 0:1, row_begin:row_begin + local_m2] <<= ub_rsum_pfx[0:1, 0:local_m2]
                    elif mt2 == last_block_m and head_ws >= 0:    # head (lower-s2) partial: no divide
                        if local_m2 > 0:
                            copy_partial_t(ub_accum_pfx, ub_rmax_pfx, ub_rsum_pfx,
                                           ub_accum, ub_rmax[cm2], ub_rsum[cm2])
                            fd_accum[head_ws, row_begin:row_begin + local_m2, 0:D_HEAD] <<= ub_accum_pfx[0:local_m2, 0:D_HEAD]
                            fd_max[head_ws, 0:1, row_begin:row_begin + local_m2] <<= ub_rmax_pfx[0:1, 0:local_m2]
                            fd_sum[head_ws, 0:1, row_begin:row_begin + local_m2] <<= ub_rsum_pfx[0:1, 0:local_m2]
                    else:                                         # fully owned: finalize (separate divide) + store
                        finalize(ub_accum, ub_rsum[cm2], ub_out, local_m2)
                        out_row2 = Var(q_row2 + row_begin)
                        if local_m2 > 0:
                            out[out_row2:out_row2 + local_m2, 0:D] <<= ub_out[0:local_m2, 0:D]
                pv_mutex.free()

        # ===== Flash-Decode merge phase (AIV-only) =====
        # Every AIV lane must hit the barrier so the partials are globally visible.
        allvec_ready(6, Pipe.MTE3)
        allvec_wait(6, Pipe.MTE2)   # gate the merge's GM->UB partial loads (MTE2), not scalar (S)

        fd_b = Var(vec * FD_NF)
        fd_en = Var(0, DT.int)
        fd_en.GetValueFrom(fd_meta[fd_b + 0:fd_b + 1])
        if fd_en > 0:
            fd_m = Var(0, DT.int)
            fd_m.GetValueFrom(fd_meta[fd_b + 1:fd_b + 2])
            fd_lo = Var(0, DT.int)
            fd_lo.GetValueFrom(fd_meta[fd_b + 2:fd_b + 3])
            fd_hi = Var(0, DT.int)
            fd_hi.GetValueFrom(fd_meta[fd_b + 3:fd_b + 4])
            fd_row0 = Var(0, DT.int)
            fd_row0.GetValueFrom(fd_meta[fd_b + 4:fd_b + 5])
            fd_nrows = Var(0, DT.int)
            fd_nrows.GetValueFrom(fd_meta[fd_b + 5:fd_b + 6])

            out_base = Var(fd_m * TILE_M)
            # transpose FD: load this AIV's 64-query partial halves; merge full ROWS_SB rows
            # (single-weight lean form via fd_merge_t); cast bf16; store. Reuses ub_pv/ub_rmax/
            # ub_rsum/ub_old_w (free post-loop) -- no extra UB.
            ub_rmax[0][0:1, 0:ROWS_SB] <<= fd_max[fd_lo, 0:1, fd_row0:fd_row0 + ROWS_SB]
            ub_rmax[1][0:1, 0:ROWS_SB] <<= fd_max[fd_hi, 0:1, fd_row0:fd_row0 + ROWS_SB]
            ub_rsum[0][0:1, 0:ROWS_SB] <<= fd_sum[fd_lo, 0:1, fd_row0:fd_row0 + ROWS_SB]
            ub_rsum[1][0:1, 0:ROWS_SB] <<= fd_sum[fd_hi, 0:1, fd_row0:fd_row0 + ROWS_SB]
            ub_pv[0][0:ROWS_SB, 0:D_HEAD] <<= fd_accum[fd_lo, fd_row0:fd_row0 + ROWS_SB, 0:D_HEAD]
            ub_pv[1][0:ROWS_SB, 0:D_HEAD] <<= fd_accum[fd_hi, fd_row0:fd_row0 + ROWS_SB, 0:D_HEAD]
            fd_merge_t(ub_pv[0], ub_pv[1], ub_rmax[0], ub_rmax[1], ub_rsum[0], ub_rsum[1],
                       ub_old_w[0], ub_old_w[1], ub_pv[0])
            output_cast_t(ub_pv[0], ub_out)
            if fd_nrows > 0:
                out[out_base + fd_row0:out_base + fd_row0 + fd_nrows, 0:D] <<= ub_out[0:fd_nrows, 0:D]
    return out


# ============================ reference / runner ============================
            # Shape dimensions: Hq by S1 by D.
