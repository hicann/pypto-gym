# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Preserved V8 P stage: exp, the fused accumulator rescale, and the strided P publication.

v8_helpers holds the exact local helper closure from the corrected V8 and V6 sources; stage.py
is the kernel that uses it."""

import math

from ascriptor.a5 import *

# ----------------------------------------------------------------------------------------------------
# v8_helpers.py
# Exact local helper closure from corrected V8 and V6 sources.
# ----------------------------------------------------------------------------------------------------

ROWS_PER_SB = 64
QLANES = ROWS_PER_SB
TILE_N = 128
FAST_TAIL_N = 3
D_HEAD = 128
SOFTMAX_SCALE = 1.0 / math.sqrt(D_HEAD)
LN16 = math.log(16.0)
UNROLL = 4
RB_MAX = TILE_N // UNROLL
NEG_LARGE = -1.0e30
NZ_C0 = 32
SLAB = TILE_N // 4
FRAC_STRIDE4 = SLAB + 1
KEYS_PER_GRP = 4
RB4 = 16
UNROLL4 = 2
CHUNKS_D = D_HEAD // 64

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
def final_div_cast_bf16_vf(ub_accum: Tensor, ub_rsum: Tensor, ub_out: Tensor):
    acc = RegList(DT.float, CHUNKS_D); rsum = Reg(DT.float)
    for r in range(ROWS_PER_SB):
        rsum <<= ub_rsum[0:1, r:r + 1].single()
        acc <<= ub_accum[r:r + 1, 0:D_HEAD]
        acc <<= acc / rsum
        ub_out[r:r + 1, 0:D_HEAD] <<= acc
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

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

# ----------------------------------------------------------------------------------------------------
# stage.py
# Preserved V8 stage sample; independent generated reference replaces golden recording.
# ----------------------------------------------------------------------------------------------------

TILE = 128
ROWS = 64

@kernel()
def v8_p_path(s0: GM[f32, (128, 64)], s1: GM[f32, (128, 64)], s2: GM[f32, (128, 64)],
              acc_in: GM[f32, (64, 128)], pv_in: GM[f32, (64, 128)], w_in: GM[f32, (1, 64)], zp: GM[u8, (33, 256)],
              p0: GM[u8, (33, 256)], p1: GM[u8, (33, 256)], p2: GM[u8, (33, 256)],
              st1: GM[f32, (2, 64)], st2: GM[f32, (2, 64)], st3: GM[f32, (2, 64)], st4: GM[f32, (3, 64)],
              acc_out: GM[f32, (64, 128)], fin: GM[bf16, (64, 128)]):
    ub_s0 = Tensor(DT.float, [128, 64], Position.UB)
    ub_s1 = Tensor(DT.float, [128, 64], Position.UB)
    ub_s2 = Tensor(DT.float, [128, 64], Position.UB)
    ub_rmax = Tensor(DT.float, [1, 64], Position.UB)
    ub_rsum = Tensor(DT.float, [1, 64], Position.UB)
    ub_ow = Tensor(DT.float, [1, 64], Position.UB)
    ub_p0 = Tensor(DT.hif8, [33, 256], Position.UB)
    ub_p1 = Tensor(DT.hif8, [33, 256], Position.UB)
    ub_p2 = Tensor(DT.hif8, [33, 256], Position.UB)
    ub_acc = Tensor(DT.float, [64, 128], Position.UB)
    ub_pv = Tensor(DT.float, [64, 128], Position.UB)
    ub_w = Tensor(DT.float, [1, 64], Position.UB)
    ub_out = Tensor(DT.bfloat16, [64, 128], Position.UB)
    # This stage has one public output owner; the peer sub-block stays idle.
    vector = Var(GetSubBlockIdx())
    if vector == 0:
        with auto_sync():
            ub_s0 <<= s0
            ub_s1 <<= s1
            ub_s2 <<= s2
            # the strided P stores leave one 32-byte block per fractal column untouched (block 32 + 33 i) and the tail
            # writes three rows only: zero the tiles first so every byte of the copied-out tiles is deterministic
            ub_p0_u8 = ub_p0.reinterpret(DT.uint8)
            ub_p0_u8 <<= zp
            ub_p1_u8 = ub_p1.reinterpret(DT.uint8)
            ub_p1_u8 <<= zp
            ub_p2_u8 = ub_p2.reinterpret(DT.uint8)
            ub_p2_u8 <<= zp
            group_softmax_p0_first_vf(ub_s0, ub_s1, ub_s2, ub_rmax, ub_rsum, ub_p0)
            st1[0:1, :] <<= ub_rmax
            st1[1:2, :] <<= ub_rsum
            p0 <<= ub_p0.reinterpret(DT.uint8)
            group_emit_vf(ub_s1, ub_rmax, ub_rsum, ub_p1)
            st2[0:1, :] <<= ub_rmax
            st2[1:2, :] <<= ub_rsum
            p1 <<= ub_p1.reinterpret(DT.uint8)
            group_emit_tail3_vf(ub_s2, ub_rmax, ub_rsum, ub_p2)
            st3[0:1, :] <<= ub_rmax
            st3[1:2, :] <<= ub_rsum
            p2 <<= ub_p2.reinterpret(DT.uint8)
            ub_acc <<= acc_in
            ub_pv <<= pv_in
            ub_w <<= w_in
            accum_pv_vf(ub_acc, ub_pv, ub_w)
            acc_out <<= ub_acc
            final_div_cast_bf16_vf(ub_acc, ub_rsum, ub_out)
            fin <<= ub_out
            group_prepare_vf(ub_s0, ub_s1, ub_rmax, ub_rsum, ub_ow)
            st4[0:1, :] <<= ub_rmax
            st4[1:2, :] <<= ub_rsum
            st4[2:3, :] <<= ub_ow
    return p0, p1, p2, st1, st2, st3, st4, acc_out, fin
