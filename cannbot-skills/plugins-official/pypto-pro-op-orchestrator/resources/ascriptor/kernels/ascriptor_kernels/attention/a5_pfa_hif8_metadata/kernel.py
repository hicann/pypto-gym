# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""All-HiFloat8 GQA prefill on a stream-K plus Flash-Decode schedule the host hands it.

Q, K and V are all HiFloat8 in uint8 carriers, and so are the published probabilities: the QK
matmul computes the score transposed (score^T = K @ Q^T), the online softmax biases the running
max by -ln16 so that `expsub` returns 16*exp(s - m) for free, and that x16 threads through the
row sum and the accumulator and cancels in O = acc/rsum. The stored running max stays the true
one. P is published NZ-direct -- a 3-deep deinterleave 4-key pack, a strided reg_to_ub and four
bulk ub_to_l1_nz stores -- and feeds a HiFloat8 PV matmul; only the output is BF16.

Nothing about the grid is decided on the device. Each AIC core owns a contiguous interval
[flat_start, flat_end) of the flat (m_block-major, s2_tile-inner) grid, read out of the int32
fa_meta/fd_meta tensors with Var.GetValueFrom; the core count itself comes from GetCubeNum(), so
the same body runs on a 32-core and a 28-core part and only the host generator in reference.py is
target-aware. The two cores owning a split m-block write fp32 partials to a GM workspace through
`copy_partial_vf`, which is what keeps the store WAR-safe against the ring-index clobber that
hits the 28-core shape; an AIV-wide barrier and two merge lanes per split block recombine them.

Single-batch MQA/GQA with Hk=1: M = Hq*Sq is one contiguous query row axis and K/V are [S2, D]
shared by every head."""

import math

from ascriptor.a5 import *

# fp32_to_hif8 / hif8_to_fp32 (easyasc.dtypehelper.hif8_codec) come in via the a5pr/a5 wildcard above.

_pyrange = unroll  # the old script aliased Python's range for compile-time unrolling

# ---- shape / compute constants ----
TILE_M = 128
TILE_N = 128
D_HEAD = 128
ROWS_PER_SB = TILE_M // 2       # 64 vec sub-block query rows (query is the LANE axis after transpose)
QLANES = ROWS_PER_SB
CHUNKS_D = D_HEAD // 64         # 2 fp32 regs cover a 128-col D row
CD = CHUNKS_D
M_Q = QLANES
UNROLL = 4
RB_MAX = TILE_N // UNROLL       # 32: max-pass over 128 keys, 4-way unrolled partial-max chains
SOFTMAX_SCALE = 1.0 / math.sqrt(D_HEAD)
LN16 = math.log(16.0)          # exp-max bias: P=exp(s-(m-ln16))=16*exp(s-m); x16 threads rsum+acc, cancels in O
NEG_LARGE = -1.0e30

PRELOAD_N = 2  # Explicit default source pipeline depth.
CACHE = PRELOAD_N + 1
_CacheBuf = QBuff if CACHE >= 4 else TBuff             # l1 K/V/P + softmax-state depth tracks CACHE (TBuff@3, QBuff@4)
# ---- nz4 ("deinterleave 4-key") hif8 NZ P-store constants ----
NZ_C0 = 32                  # hif8 C0 (32-byte fractal inner dim)
SLAB = TILE_N // 4          # 32 keys per slab (4 slabs across TILE_N)
FRAC_STRIDE4 = SLAB + 1     # 33 -- M_src (NZ fractal row stride), +1 pad row/fractal (bank-conflict dodge)
KEYS_PER_GRP = 4            # 4 keys packed per register via the 3-deep deinterleave tree
RB4 = 16                    # rb count for the pack loop (RB4*UNROLL4 = 32 = SLAB M-rows)
UNROLL4 = 2                 # unroll(2), 4 keys/iter (throughput-bound; higher unroll gave no speedup)
_DEINT_U8 = False           # hif8-native deinterleave; flip True if board rejects hif8 (uint8 reinterpret)

# ---- host metadata field layouts (1-D int32 GM tensors) ----
FA_NF = 6     # per-core:      [enable, flat_start, flat_end, tail_ws, head_ws, pad]
FD_NF = 6     # per-AIV-lane:  [enable, m_idx, ws_lo, ws_hi, row0, nrows]
NWS_GLOBAL = 64  # Bounded by two workspace slots per active split; at most32 launch cores.

# Physical AIC core count of the selected device (globvars.core_num): a5=32, a5pr=28. The kernel is
# metadata-driven (GetCubeNum() == DEVICE_CORES at runtime), so ONLY the host metadata generator is
# target-aware -- fa_meta/fd_meta must have exactly DEVICE_CORES / 2*DEVICE_CORES rows. The `fia` shape
# uses the official section-17 cost-aware 28-core table on a5pr; on a5 (32 cores, no official table) it
# falls back to staggered near-even stream-K (== fd_modified fia32; the cheap 24-row M-tail block is
# spread task-wise, ~negligible skew).
# The host metadata generator that fills fa_meta/fd_meta -- including the section-17 exact 28-core
# FIA split for tiles_n = 33 -- lives in reference.py; nothing below decides a grid.
# ============================ vec stages (all-hif8) ============================
def _deint(d0, d1, s0, s1):
    if _DEINT_U8:
        deinterleave(d0.reinterpret(DT.uint8), d1.reinterpret(DT.uint8),
                     s0.reinterpret(DT.uint8), s1.reinterpret(DT.uint8))
    else:
        deinterleave(d0, d1, s0, s1)


@vf()
def init_softmax_state_vf(ub_rmax: Tensor, ub_rsum: Tensor):
    """rowmax=-inf, rowsum=0 for one m-block's running softmax state (64-wide)."""
    neg = Reg(DT.float); zero = Reg(DT.float)
    neg <<= NEG_LARGE; zero <<= 0.0
    ub_rmax[0:1, 0:QLANES] <<= neg
    ub_rsum[0:1, 0:QLANES] <<= zero


@vf()
def softmax_vf(ub_score: Tensor, ub_rmax: Tensor, ub_rsum: Tensor, ub_p: Tensor, ub_old_weight: Tensor):
    """TRANSPOSED online softmax over [key=128, query=64] score^T, single-weight (new-max) with ln16
    exp-bias -> hif8 P scaled x16. NZ-direct P-store: 4 keys/reg via a 3-deep deinterleave tree
    (fin=[k0|k1|k2|k3] dense, 256 lanes) + one strided reg_to_ub(blk_stride=33); key kk lands in
    fractals 2kk,2kk+1 at M-row m; store is 4 bulk UB2L1_NZ (one 32-key slab each)."""
    sreg = RegList(DT.float, UNROLL); acc = RegList(DT.float, UNROLL); preg = RegList(DT.float, UNROLL)
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
    muls(block_max, block_max, SOFTMAX_SCALE)        # scale the max ONCE (max(scale*s)=scale*max(s))
    v_pmax <<= ub_rmax[0:1, 0:QLANES]
    v_nmax <<= block_max.vmax(v_pmax)
    expsub(v_oldw, v_pmax, v_nmax)                   # old_scale = exp(old_m - new_m)
    v_nmax_e = Reg(DT.float)
    adds(v_nmax_e, v_nmax, -LN16)                    # exp-only x16 bias; stored/compared running max stays true
    # exp + nz4 pack: reuse acc[] (free after block_max) as the 4 per-slab running sums.
    for kk in unroll(KEYS_PER_GRP):
        acc[kk] <<= 0.0
    hh = RegList(DT.hif8, KEYS_PER_GRP)              # 4 cast'd keys (stride-4 sparse, RegLayout.ZERO)
    a = Reg(DT.hif8); b = Reg(DT.hif8); fin = Reg(DT.hif8); dmy = Reg(DT.hif8)
    for rb in range(RB4):
        for u in unroll(UNROLL4):
            for kk in unroll(KEYS_PER_GRP):
                row = (rb * UNROLL4 + u) + kk * SLAB          # key m + kk*32
                sreg[0] <<= ub_score[row:row + 1, :]
                muls(sreg[0], sreg[0], SOFTMAX_SCALE)
                expsub(preg[0], sreg[0], v_nmax_e)   # = 16 * exp(s - nmax): rsum (acc) and hif8 P both carry x16
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
def softmax_tail_vf(ub_score: Tensor, ub_rmax: Tensor, ub_rsum: Tensor, ub_p: Tensor,
                    ub_old_weight: Tensor):
    """TRANSPOSED softmax for a PARTIAL K-tile (valid_n < TILE_N): invalid key ROWS (n >= valid_n) are
    NEG-filled by neg_fill_rows before this vf runs, so they drop out of the max and give P=0. Same ln16 x16 bias. nz4 tail pack uses
    gather (hif8-native, unlike sim-only Squeeze). Nested DSL loops (g=slab, m=row) keep it LOOPED (not
    128x-unrolled) and compute the slab base with multiplies only (no var_div) -> no AIV VF stack spill."""
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
    adds(v_nmax_e, v_nmax, -LN16)                        # exp-only x16 bias (same as softmax_vf)
    block_sum <<= 0.0
    _SLAB_STRIDE = 2 * FRAC_STRIDE4 * NZ_C0              # 2112 = 2 fractals * 33 * 32
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


@vf()
def accum_pv_vf(ub_accum: Tensor, ub_pv: Tensor, ub_old_weight: Tensor):
    """Single-weight online rescale O = O*old_scale + PV, fused via muldstadd (dst = dst*src0 + src1).
    P is already exp'd against the running max in softmax -> PV needs NO block weight. Rows unrolled by
    UNROLL to overlap per-row load->FMA->store; each row is CHUNKS_D 64-lane fp32 regs."""
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
    """First owned K-tile of an m-block: O = PV (no prior O to rescale; P already exp'd vs new_m).
    Writes all ROWS_PER_SB rows -> ub_accum is fully initialized (no separate zero pass needed)."""
    pv = RegList(DT.float, CHUNKS_D)
    for r in range(ROWS_PER_SB):
        pv <<= ub_pv[r:r + 1, 0:D_HEAD]
        ub_accum[r:r + 1, 0:D_HEAD] <<= pv
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def final_div_cast_bf16_vf(ub_accum: Tensor, ub_rsum: Tensor, ub_out: Tensor):
    """out = accum / rowsum; bf16 cast on store. For non-split (fully owned) m-blocks. The x16 from the
    ln16 bias is carried by BOTH accum and rsum, so it cancels here (no explicit /16)."""
    acc = RegList(DT.float, CHUNKS_D); rsum = Reg(DT.float)
    for r in range(ROWS_PER_SB):
        rsum <<= ub_rsum[0:1, r:r + 1].single()
        acc <<= ub_accum[r:r + 1, 0:D_HEAD]
        acc <<= acc / rsum
        ub_out[r:r + 1, 0:D_HEAD] <<= acc
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def copy_partial_vf(dst_a: Tensor, dst_m: Tensor, dst_s: Tensor,
                    src_a: Tensor, src_m: Tensor, src_s: Tensor):
    """Capture the FD partial state (accum / row-max / row-sum) into DEDICATED fixed buffers via VEC
    reads, BEFORE the GM workspace store. A later m-block's init_softmax_state / accum_pv clobbers the
    ring source buffers (ub_rmax[mt%CACHE]) / single ub_accum; routing through a vec read makes that a
    vec-vec WAR on a fixed buffer (auto_sync tracks it -> the clobber is ordered after this read), and
    the MTE3 store then reads the stable copy. Root fix for the 28-core FIA FD partial-clobber (the
    runtime-ring-index WAR the autosync tracker misses)."""
    mv = Reg(DT.float); sv = Reg(DT.float)
    mv <<= src_m[0:1, 0:ROWS_PER_SB]
    dst_m[0:1, 0:ROWS_PER_SB] <<= mv
    sv <<= src_s[0:1, 0:ROWS_PER_SB]
    dst_s[0:1, 0:ROWS_PER_SB] <<= sv
    a = RegList(DT.float, CD)
    for r in range(ROWS_PER_SB):
        a <<= src_a[r:r + 1, :]
        dst_a[r:r + 1, :] <<= a
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def fd_merge_vf(ub_o0: Tensor, ub_o1: Tensor, ub_m0: Tensor, ub_m1: Tensor,
                ub_s0: Tensor, ub_s1: Tensor, ub_A: Tensor, ub_den: Tensor, ub_merged: Tensor):
    """2-split merge + normalize, LEAN (== max-aware): A=exp(m0-m1), den_eff=s0*A+s1 precomputed
    VECTORIZED over the 64 query lanes; main loop O=(o0*A+o1)/den_eff per row. The x16 bias cancels
    (den_eff carries it, o0/o1 carry it)."""
    m0v = Reg(DT.float); m1v = Reg(DT.float); s0v = Reg(DT.float); s1v = Reg(DT.float)
    Av = Reg(DT.float); denv = Reg(DT.float)
    m0v <<= ub_m0[0:1, 0:M_Q]; m1v <<= ub_m1[0:1, 0:M_Q]
    s0v <<= ub_s0[0:1, 0:M_Q]; s1v <<= ub_s1[0:1, 0:M_Q]
    expsub(Av, m0v, m1v)                          # A = exp(m0 - m1)
    denv <<= s0v * Av
    denv <<= denv + s1v                           # den_eff = s0*A + s1
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
        ub_merged[r:r + 1, :] <<= o0                # O = (o0*A + o1)/den_eff (fp32)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def output_cast_vf(ub_merged: Tensor, ub_out: Tensor):
    """fp32 merged O -> bf16, per D-chunk, downsample-pack store."""
    regs = RegList(DT.float, CD); hregs = RegList(DT.bfloat16, CD)
    cfg = CastConfig(round_mode=RoundMode.AWAY_FROM_ZERO, reg_layout=RegLayout.ZERO, name="fd_o_bf16")
    for r in range(M_Q):
        regs <<= ub_merged[r:r + 1, :]
        for c in unroll(CD):
            cast(hregs[c], regs[c], cfg)
            reg_to_ub_downsample(ub_out[r:r + 1, c * 64:(c + 1) * 64], hregs[c])
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


# ============================ kernel ============================
@kernel()
def v7_allhif8_kernel(q: GM[u8, ('M', 'D')], k: GM[u8, ('S2', 'D')], v: GM[u8, ('S2', 'D')],
                      fa_meta: GM[i32, ('FA_META',)], fd_meta: GM[i32, ('FD_META',)], out: GM[bf16, ('M', 'D')],
                      M: i32, S2: i32, D: i32, scale: f32, tiles_n: i32):
    # Compact sync: start_pipe == end_pipe so each side's wait sits on the SAME pipe as its set (FIA's
    # intra-block handoff). QK/PV C<->V mutexes + a FORWARD-only L1-P ring (VcMutex); flag 7 = AIV-wide
    # Flash-Decode barrier (allvec == SyncAll). `scale` arg is unused (VFs use the SOFTMAX_SCALE const).
    cv = CvMutex(0, depth=2, src_start_pipe=Pipe.FIX, dst_start_pipe=Pipe.V,
                 src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)          # AIC->AIV BMM1
    pv_mutex = CvMutex(2, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)   # AIC->AIV BMM2
    p_mutex = VcMutex(1, depth=CACHE, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE1)  # L1-P ring (vec->cube)

    q_hif8 = q.reinterpret(DT.hif8, name="q_hif8")
    k_hif8 = k.reinterpret(DT.hif8, name="k_hif8")
    v_hif8 = v.reinterpret(DT.hif8, name="v_hif8")
    l1q = DBuff(DT.hif8, [TILE_M, D_HEAD], Position.L1)
    l1k = _CacheBuf(DT.hif8, [TILE_N, D_HEAD], Position.L1)
    l1v = _CacheBuf(DT.hif8, [TILE_N, D_HEAD], Position.L1)
    l1p = _CacheBuf(DT.hif8, [TILE_N, TILE_M], Position.L1)          # P^T [key=128, query=128], per sub-block
    l0c_qk = DBuff(DT.float, [TILE_M, TILE_M], Position.L0C)         # QK^T [key, query]
    l0c_pv = DBuff(DT.float, [TILE_M, D_HEAD], Position.L0C)
    ub_score = DBuff(DT.float, [TILE_N, ROWS_PER_SB], Position.UB)   # score^T per sub-block [key=128, query=64]
    ub_pv = DBuff(DT.float, [ROWS_PER_SB, D_HEAD], Position.UB)
    ub_p = Tensor(DT.hif8, [FRAC_STRIDE4, KEYS_PER_GRP * ROWS_PER_SB], Position.UB)   # [33,256] NZ, M_src=33
    ub_old_weight = _CacheBuf(DT.float, [1, QLANES], Position.UB)
    ub_rmax = _CacheBuf(DT.float, [1, QLANES], Position.UB)
    ub_rsum = _CacheBuf(DT.float, [1, QLANES], Position.UB)
    ub_merge_a = Tensor(DT.float, [1, TILE_M], Position.UB)
    ub_merge_den = Tensor(DT.float, [1, TILE_M], Position.UB)
    ub_accum = Tensor(DT.float, [ROWS_PER_SB, D_HEAD], Position.UB)
    ub_out = Tensor(DT.bfloat16, [ROWS_PER_SB, D_HEAD], Position.UB)
    # dedicated FD-partial capture buffers (fixed, NOT ring-indexed): WAR-safe partial store
    ub_accum_pfx = Tensor(DT.float, [ROWS_PER_SB, D_HEAD], Position.UB)
    ub_rmax_pfx = Tensor(DT.float, [1, QLANES], Position.UB)
    ub_rsum_pfx = Tensor(DT.float, [1, QLANES], Position.UB)
    # Flash-Decode partial workspace (fp32). NWS_GLOBAL baked per config (= 2 * num_fd_tasks).
    fd_accum = split_workspace(DT.float, [NWS_GLOBAL, TILE_M, D_HEAD], name="fd_accum")
    fd_max = split_workspace(DT.float, [NWS_GLOBAL, 1, TILE_M], name="fd_max")
    fd_sum = split_workspace(DT.float, [NWS_GLOBAL, 1, TILE_M], name="fd_sum")

    core = Var(GetCubeIdx())
    vec = Var(GetVecIdx())
    row_begin = Var(GetSubBlockIdx() * ROWS_PER_SB)

    # --- per-core stream-K metadata (GetValueFrom, ace_v13 pattern) ---
    fa_base = Var(core * FA_NF)
    flat_start = Var(0, DT.int); flat_start.GetValueFrom(fa_meta[fa_base + 1:fa_base + 2])
    flat_end = Var(0, DT.int); flat_end.GetValueFrom(fa_meta[fa_base + 2:fa_base + 3])
    tail_ws = Var(0, DT.int); tail_ws.GetValueFrom(fa_meta[fa_base + 3:fa_base + 4])
    head_ws = Var(0, DT.int); head_ws.GetValueFrom(fa_meta[fa_base + 4:fa_base + 5])
    num_tasks = Var(Max(flat_end - flat_start, 0))
    first_block_m = Var(flat_start // tiles_n)
    last_block_m = Var((flat_end - 1) // tiles_n)

    # PV reads only valid loaded V rows; no overlapping full-tile fill.

    with auto_sync():
        for step in range(0, num_tasks + PRELOAD_N):
            # ===== current task: BMM1 -> ln16 softmax -> nz4 P-NZ -> L1 =====
            if step < num_tasks:
                t = Var(flat_start + step)
                mt = Var(t // tiles_n)
                kt = Var(t % tiles_n)
                qm = Var(mt % 2)
                sm = Var(mt % CACHE)
                q_row = Var(mt * TILE_M)
                valid_m = Min(TILE_M, M - q_row)
                n_off = Var(kt * TILE_N)
                valid_n = Min(TILE_N, S2 - n_off)

                if kt == 0 or step == 0:                    # m-block start (may begin mid-s2)
                    l1q[qm][0:valid_m, 0:D] <<= q_hif8[q_row:q_row + valid_m, 0:D]
                    init_softmax_state_vf(ub_rmax[sm], ub_rsum[sm])

                l1k[step][0:valid_n, 0:D] <<= k_hif8[n_off:n_off + valid_n, 0:D]
                # transposed QK: score^T = K @ Q^T -> l0c [key=TILE_N, query=TILE_M]
                matmul(l0c_qk[step], l1k[step], l1q[qm], m=TILE_N, n=TILE_M, k=D_HEAD, is_init=True)

                cv.lock()
                # SPLITN: split the query (free) axis 64+64 across the 2 AIV sub-blocks; fp32->fp32 (cheap fixpipe)
                l0c_to_ub(ub_score[step], l0c_qk[step][0:TILE_N, 0:TILE_M],
                          M=TILE_N, N=TILE_M, N_dst=ROWS_PER_SB, M_src=TILE_N,
                          dual_mode=DualMode.SPLITN, sub_block_id=0)
                cv.ready()
                cv.wait()
                if valid_n < TILE_N:                        # tail K-tile: NEG the invalid key ROWS -> P=0
                    neg_fill_rows(ub_score[step], valid_n)
                    softmax_tail_vf(ub_score[step], ub_rmax[sm], ub_rsum[sm], ub_p, ub_old_weight[step])
                else:
                    softmax_vf(ub_score[step], ub_rmax[sm], ub_rsum[sm], ub_p, ub_old_weight[step])
                cv.free()

                # publish P^T into l1p: 4 nz4 slabs (32 keys each) -> keys [g*32,(g+1)*32) x this sub-block's cols
                p_mutex.lock()
                for g in unroll(KEYS_PER_GRP):
                    l1p[step][g * SLAB:(g + 1) * SLAB, row_begin:row_begin + ROWS_PER_SB] <<= (
                        ub_p.nz()[0:SLAB, g * ROWS_PER_SB:(g + 1) * ROWS_PER_SB])
                p_mutex.ready()

            # ===== lagged task: BMM2 -> accumulate -> finalize / FD-partial =====
            if step >= PRELOAD_N:
                lag = Var(step - PRELOAD_N)
                lag_t = Var(flat_start + lag)
                lag_mt = Var(lag_t // tiles_n)
                lag_kt = Var(lag_t % tiles_n)
                sm2 = Var(lag_mt % CACHE)
                lag_q_row = Var(lag_mt * TILE_M)
                lag_valid_m = Min(TILE_M, M - lag_q_row)
                local_rows = Min(ROWS_PER_SB, Max(lag_valid_m - row_begin, 0))
                lag_n0 = Var(lag_kt * TILE_N)
                v_rows = Min(TILE_N, S2 - lag_n0)

                l1v[lag][0:v_rows, 0:D] <<= v_hif8[lag_n0:lag_n0 + v_rows, 0:D]
                p_mutex.wait()
                # transposed PV: O[query, D] = P @ V via BOTH .T (l1p holds P^T); full k=TILE_N (tail keys P=0)
                matmul(l0c_pv[lag][0:TILE_M, 0:D_HEAD], l1p[lag].T, l1v[lag].T,
                       m=TILE_M, n=D_HEAD, k=v_rows, is_init=True)
                p_mutex.free()

                pv_mutex.lock()
                l0c_to_ub(ub_pv[lag], l0c_pv[lag][0:TILE_M, 0:D_HEAD],
                          M=TILE_M, N=D_HEAD, N_dst=D_HEAD, M_src=TILE_M,
                          dual_mode=DualMode.SPLITM, sub_block_id=0)   # each AIV its 64 query rows
                pv_mutex.ready()
                pv_mutex.wait()
                if lag_kt == 0 or lag == 0:                 # m-block first owned s2: O = PV (no rescale)
                    accum_pv_first_vf(ub_accum, ub_pv[lag])
                else:                                       # O = O*old_scale + PV
                    accum_pv_vf(ub_accum, ub_pv[lag], ub_old_weight[lag])
                pv_mutex.free()

                if lag_kt == tiles_n - 1 or lag == num_tasks - 1:   # m-block last owned s2
                    if lag_mt == first_block_m and tail_ws >= 0:    # tail (upper-s2) partial: WAR-safe, no divide
                        if local_rows > 0:
                            copy_partial_vf(ub_accum_pfx, ub_rmax_pfx, ub_rsum_pfx,
                                            ub_accum, ub_rmax[sm2], ub_rsum[sm2])
                            fd_accum[tail_ws, row_begin:row_begin + local_rows, 0:D_HEAD] <<= ub_accum_pfx[0:local_rows, 0:D_HEAD]
                            fd_max[tail_ws, 0:1, row_begin:row_begin + local_rows] <<= ub_rmax_pfx[0:1, 0:local_rows]
                            fd_sum[tail_ws, 0:1, row_begin:row_begin + local_rows] <<= ub_rsum_pfx[0:1, 0:local_rows]
                    elif lag_mt == last_block_m and head_ws >= 0:   # head (lower-s2) partial: WAR-safe, no divide
                        if local_rows > 0:
                            copy_partial_vf(ub_accum_pfx, ub_rmax_pfx, ub_rsum_pfx,
                                            ub_accum, ub_rmax[sm2], ub_rsum[sm2])
                            fd_accum[head_ws, row_begin:row_begin + local_rows, 0:D_HEAD] <<= ub_accum_pfx[0:local_rows, 0:D_HEAD]
                            fd_max[head_ws, 0:1, row_begin:row_begin + local_rows] <<= ub_rmax_pfx[0:1, 0:local_rows]
                            fd_sum[head_ws, 0:1, row_begin:row_begin + local_rows] <<= ub_rsum_pfx[0:1, 0:local_rows]
                    else:                                           # fully owned: finalize (divide) + store
                        final_div_cast_bf16_vf(ub_accum, ub_rsum[sm2], ub_out)
                        out_row = Var(lag_q_row + row_begin)
                        if local_rows > 0:
                            out[out_row:out_row + local_rows, 0:D] <<= ub_out[0:local_rows, 0:D]

        # every AIV lane hits the barrier so the partials are globally visible
        allvec_ready(7, Pipe.MTE3)
    allvec_wait(7, Pipe.MTE2)   # gate the merge's GM->UB partial loads (MTE2), not scalar

    # ===== Flash-Decode merge phase (AIV-only), metadata-driven per lane =====
    fd_b = Var(vec * FD_NF)
    fd_en = Var(0, DT.int); fd_en.GetValueFrom(fd_meta[fd_b + 0:fd_b + 1])
    if fd_en > 0:
        fd_m = Var(0, DT.int); fd_m.GetValueFrom(fd_meta[fd_b + 1:fd_b + 2])
        fd_lo = Var(0, DT.int); fd_lo.GetValueFrom(fd_meta[fd_b + 2:fd_b + 3])
        fd_hi = Var(0, DT.int); fd_hi.GetValueFrom(fd_meta[fd_b + 3:fd_b + 4])
        fd_row0 = Var(0, DT.int); fd_row0.GetValueFrom(fd_meta[fd_b + 4:fd_b + 5])
        fd_nrows = Var(0, DT.int); fd_nrows.GetValueFrom(fd_meta[fd_b + 5:fd_b + 6])
        out_base = Var(fd_m * TILE_M)
        with auto_sync():
            # load this lane's 64-query partial halves; merge full ROWS_PER_SB rows; cast bf16; store.
            ub_rmax[0][0:1, 0:ROWS_PER_SB] <<= fd_max[fd_lo, 0:1, fd_row0:fd_row0 + ROWS_PER_SB]
            ub_rmax[1][0:1, 0:ROWS_PER_SB] <<= fd_max[fd_hi, 0:1, fd_row0:fd_row0 + ROWS_PER_SB]
            ub_rsum[0][0:1, 0:ROWS_PER_SB] <<= fd_sum[fd_lo, 0:1, fd_row0:fd_row0 + ROWS_PER_SB]
            ub_rsum[1][0:1, 0:ROWS_PER_SB] <<= fd_sum[fd_hi, 0:1, fd_row0:fd_row0 + ROWS_PER_SB]
            ub_pv[0][:, :] <<= fd_accum[fd_lo, fd_row0:fd_row0 + ROWS_PER_SB, 0:D_HEAD]
            ub_pv[1][:, :] <<= fd_accum[fd_hi, fd_row0:fd_row0 + ROWS_PER_SB, 0:D_HEAD]
            fd_merge_vf(ub_pv[0], ub_pv[1], ub_rmax[0], ub_rmax[1],
                        ub_rsum[0], ub_rsum[1], ub_merge_a, ub_merge_den, ub_pv[0])
            output_cast_vf(ub_pv[0], ub_out)
            if fd_nrows > 0:
                out[out_base + fd_row0:out_base + fd_row0 + fd_nrows, 0:D] <<= ub_out[0:fd_nrows, 0:D]
    return out


# ============================ reference / runner ============================
              # Shape dimensions: M by D.
