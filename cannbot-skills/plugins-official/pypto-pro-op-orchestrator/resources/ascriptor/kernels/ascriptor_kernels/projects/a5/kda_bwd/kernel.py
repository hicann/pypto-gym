# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Nine DSL stages of the KDA (Kimi Delta Attention) backward."""

import math

from ascriptor.a5 import *

# ----------------------------------------------------------------------------------------------------
# scan_fused.py
# KDA kernel port; the launch ABI for this stage is main.py's `execute`.
#
# The arithmetic and work partition come from the reviewed source. Migration
# uses a shared L0 operand pair with cube-phase barriers to fit the eight-flag
# event budget.
# ----------------------------------------------------------------------------------------------------

L = 64

D = 128

HALF_L = L // 2

HALF_D = D // 2

BF16_C0 = 16

QG_SCALE = 1.0 / 11.313708498984761

DAQK_SCALE = 1.0 / (D ** 0.5)

LN2 = math.log(2.0)

@vf()
def cast_dht_to_f32_vf(dht_h_ub: Tensor, dstate_f_ub: Tensor, row_begin_d: Var):
    reg_f32 = Reg(DT.float)
    dst_base = Var(row_begin_d * D)
    n_loops = Var(HALF_D * D // 64)  # M10: the static VF loop bound must be integral.
    for i in range(n_loops):
        reg_f32 <<= dht_h_ub[i * 64]
        dstate_f_ub[dst_base + i * 64] <<= reg_f32
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

@vf()
def snapshot_and_cast_state_vf(dstate_f_ub: Tensor, state_h_ub: Tensor, state_h_nz_ub: Tensor, row_begin_d: Var):
    row_lo = Reg(DT.float)
    row_hi = Reg(DT.float)
    lo_bf16 = Reg(DT.bfloat16)
    hi_bf16 = Reg(DT.bfloat16)
    row_bf16 = Reg(DT.bfloat16)
    dummy_bf16 = Reg(DT.bfloat16)
    for r in range(HALF_D):
        row_lo <<= dstate_f_ub[row_begin_d + r:row_begin_d + r + 1, 0:64]
        row_hi <<= dstate_f_ub[row_begin_d + r:row_begin_d + r + 1, 64:D]
        lo_bf16 <<= row_lo.astype(DT.bfloat16)
        hi_bf16 <<= row_hi.astype(DT.bfloat16)
        deinterleave(row_bf16, dummy_bf16, lo_bf16, hi_bf16)
        reg_to_ub(state_h_ub[r:r + 1, 0:D], row_bf16)
        reg_to_ub(state_h_nz_ub[r * BF16_C0], row_bf16, HALF_D)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

@vf()
def add_dv_and_cast_bf16_vf(dv_delta_ub: Tensor, dv_sum_ub: Tensor, dv_bf16_ub: Tensor, dv_bf16_nz_ub: Tensor):
    delta_lo = Reg(DT.float)
    delta_hi = Reg(DT.float)
    sum_lo = Reg(DT.float)
    sum_hi = Reg(DT.float)
    lo_bf16 = Reg(DT.bfloat16)
    hi_bf16 = Reg(DT.bfloat16)
    row_bf16 = Reg(DT.bfloat16)
    dummy_bf16 = Reg(DT.bfloat16)
    for r in range(HALF_L):
        delta_lo <<= dv_delta_ub[r:r + 1, 0:64]
        delta_hi <<= dv_delta_ub[r:r + 1, 64:D]
        sum_lo <<= dv_sum_ub[r:r + 1, 0:64]
        sum_hi <<= dv_sum_ub[r:r + 1, 64:D]
        sum_lo <<= sum_lo + delta_lo
        sum_hi <<= sum_hi + delta_hi
        lo_bf16 <<= sum_lo.astype(DT.bfloat16)
        hi_bf16 <<= sum_hi.astype(DT.bfloat16)
        deinterleave(row_bf16, dummy_bf16, lo_bf16, hi_bf16)
        reg_to_ub(dv_bf16_ub[r:r + 1, 0:D], row_bf16)
        reg_to_ub(dv_bf16_nz_ub[r * BF16_C0], row_bf16, HALF_L)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

@vf()
def exp2_glast_vf(g_last_half_ub: Tensor, exp_g_last_half_ub: Tensor):
    # Per-chunk decay 2^g_last = exp(g_last * ln2): g_last (= g_cumsum at the chunk
    # boundary) already carries the 1/ln2 factor. Replaces the old host
    # torch.exp(g_last*ln2). One HALF_D-lane shot per sub-block.
    r = Reg(DT.float)
    r <<= g_last_half_ub[0:1, 0:HALF_D]   # bf16 -> fp32 upcast
    r <<= r * LN2
    r <<= r.exp()
    exp_g_last_half_ub[0:1, 0:HALF_D] <<= r
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

@vf()
def update_dstate_vf(
    seed_ub: Tensor,
    corr_ub: Tensor,
    dstate_f_ub: Tensor,
    state_h_ub: Tensor,
    state_h_nz_ub: Tensor,
    exp_g_last_half_ub: Tensor,
    row_begin_d: Var,
):
    seed_lo = Reg(DT.float)
    seed_hi = Reg(DT.float)
    corr_lo = Reg(DT.float)
    corr_hi = Reg(DT.float)
    state_lo = Reg(DT.float)
    state_hi = Reg(DT.float)
    scale = Reg(DT.float)
    lo_bf16 = Reg(DT.bfloat16)
    hi_bf16 = Reg(DT.bfloat16)
    row_bf16 = Reg(DT.bfloat16)
    dummy_bf16 = Reg(DT.bfloat16)
    for r in range(HALF_D):
        scale <<= exp_g_last_half_ub[0:1, r:r + 1].single()
        seed_lo <<= seed_ub[r:r + 1, 0:64]
        seed_hi <<= seed_ub[r:r + 1, 64:D]
        seed_lo <<= seed_lo * QG_SCALE   # 1/sqrt(D): was host qg pre-scale
        seed_hi <<= seed_hi * QG_SCALE
        corr_lo <<= corr_ub[r:r + 1, 0:64]
        corr_hi <<= corr_ub[r:r + 1, 64:D]
        state_lo <<= dstate_f_ub[row_begin_d + r:row_begin_d + r + 1, 0:64]
        state_hi <<= dstate_f_ub[row_begin_d + r:row_begin_d + r + 1, 64:D]
        state_lo <<= state_lo * scale
        state_hi <<= state_hi * scale
        seed_lo <<= seed_lo + state_lo
        seed_hi <<= seed_hi + state_hi
        seed_lo <<= seed_lo - corr_lo
        seed_hi <<= seed_hi - corr_hi
        dstate_f_ub[row_begin_d + r:row_begin_d + r + 1, 0:64] <<= seed_lo
        dstate_f_ub[row_begin_d + r:row_begin_d + r + 1, 64:D] <<= seed_hi
        lo_bf16 <<= seed_lo.astype(DT.bfloat16)
        hi_bf16 <<= seed_hi.astype(DT.bfloat16)
        deinterleave(row_bf16, dummy_bf16, lo_bf16, hi_bf16)
        reg_to_ub(state_h_ub[r:r + 1, 0:D], row_bf16)
        reg_to_ub(state_h_nz_ub[r * BF16_C0], row_bf16, HALF_D)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

@vf()
def update_dstate_final_vf(
    seed_ub: Tensor,
    corr_ub: Tensor,
    dstate_f_ub: Tensor,
    state_h_ub: Tensor,
    exp_g_last_half_ub: Tensor,
    row_begin_d: Var,
):
    seed_lo = Reg(DT.float)
    seed_hi = Reg(DT.float)
    corr_lo = Reg(DT.float)
    corr_hi = Reg(DT.float)
    state_lo = Reg(DT.float)
    state_hi = Reg(DT.float)
    scale = Reg(DT.float)
    lo_bf16 = Reg(DT.bfloat16)
    hi_bf16 = Reg(DT.bfloat16)
    row_bf16 = Reg(DT.bfloat16)
    dummy_bf16 = Reg(DT.bfloat16)
    for r in range(HALF_D):
        scale <<= exp_g_last_half_ub[0:1, r:r + 1].single()
        seed_lo <<= seed_ub[r:r + 1, 0:64]
        seed_hi <<= seed_ub[r:r + 1, 64:D]
        seed_lo <<= seed_lo * QG_SCALE   # 1/sqrt(D): was host qg pre-scale
        seed_hi <<= seed_hi * QG_SCALE
        corr_lo <<= corr_ub[r:r + 1, 0:64]
        corr_hi <<= corr_ub[r:r + 1, 64:D]
        state_lo <<= dstate_f_ub[row_begin_d + r:row_begin_d + r + 1, 0:64]
        state_hi <<= dstate_f_ub[row_begin_d + r:row_begin_d + r + 1, 64:D]
        state_lo <<= state_lo * scale
        state_hi <<= state_hi * scale
        seed_lo <<= seed_lo + state_lo
        seed_hi <<= seed_hi + state_hi
        seed_lo <<= seed_lo - corr_lo
        seed_hi <<= seed_hi - corr_hi
        lo_bf16 <<= seed_lo.astype(DT.bfloat16)
        hi_bf16 <<= seed_hi.astype(DT.bfloat16)
        deinterleave(row_bf16, dummy_bf16, lo_bf16, hi_bf16)
        reg_to_ub(state_h_ub[r:r + 1, 0:D], row_bf16)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

@func()
def _phase_matmul(dst: Tensor, lhs: Tensor, rhs: Tensor, l0a: Tensor, l0b: Tensor, m: Var, n: Var, k: Var):
    # Shared operands trade overlap for an explicit finite event footprint.
    bar_all()
    l1_to_l0(l0a, lhs)
    l1_to_l0(l0b, rhs)
    bar_all()
    mmad(dst, l0a, l0b, M=m, N=n, K=k, is_init=True)
    bar_all()


@kernel()
def scan_fused_kernel(kg: GM[bf16, ('B', 'T', 'HV', 128)], qg: GM[bf16, ('B', 'T', 'HV', 128)], w: GM[bf16, ('B', 'T', 'HV', 128)], g_last: GM[bf16, ('B', 'C', 'HV', 128)], grad_out: GM[bf16, ('B', 'T', 'HV', 128)], Aqk: GM[bf16, ('B', 'T', 'HV', 64)], v_new: GM[bf16, ('B', 'T', 'HV', 128)], dht: GM[bf16, ('B', 'HV', 64, 256)], dAqk: GM[bf16, ('B', 'T', 'HV', 64)], dh: GM[bf16, ('B', 'C', 'HV', 128, 128)], dv: GM[bf16, ('B', 'T', 'HV', 128)], dh0: GM[bf16, ('B', 'HV', 128, 128)], B: i32, HV: i32, C: i32):
    phase_l0a = Tensor(DT.bfloat16, [D, D], Position.L0A)
    phase_l0b = Tensor(DT.bfloat16, [D, D], Position.L0B)
    state_bridge = VcMutex(
        0,
        depth=1,
        src_start_pipe=Pipe.MTE3,
        src_end_pipe=Pipe.MTE3,
        dst_start_pipe=Pipe.MTE1,
        dst_end_pipe=Pipe.MTE1,
    )
    dv_bridge = VcMutex(
        1,
        depth=1,
        src_start_pipe=Pipe.MTE3,
        src_end_pipe=Pipe.MTE3,
        dst_start_pipe=Pipe.MTE1,
        dst_end_pipe=Pipe.MTE1,
    )
    dvdelta_mutex = CvMutex(2, depth=1, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    seed_mutex = CvMutex(3, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    corr_mutex = CvMutex(4, depth=1, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    dv0_mutex = CvMutex(5, depth=1, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)

    l1_kg = Tensor(DT.bfloat16, [L, D], Position.L1)
    l1_qg = DBuff(DT.bfloat16, [L, D], Position.L1)
    l1_w = DBuff(DT.bfloat16, [L, D], Position.L1)
    l1_do = DBuff(DT.bfloat16, [L, D], Position.L1)
    l1_Aqk = DBuff(DT.bfloat16, [L, L], Position.L1)
    l1_vnew = DBuff(DT.bfloat16, [L, D], Position.L1)
    l1_dv = Tensor(DT.bfloat16, [L, D], Position.L1)
    l1_dstate = Tensor(DT.bfloat16, [D, D], Position.L1)

    l0c_dvdelta = Tensor(DT.float, [L, D], Position.L0C)
    l0c_seed = DBuff(DT.float, [D, D], Position.L0C)
    l0c_corr = Tensor(DT.float, [D, D], Position.L0C)
    l0c_dv0 = Tensor(DT.float, [L, D], Position.L0C)
    # dAqk shares the dvdelta accumulator: its matmul runs only after the
    # dvdelta fixpipe of the current chunk has drained.
    l0c_daqk = l0c_dvdelta

    dstate_f_ub = Tensor(DT.float, [D, D], Position.UB)
    state_h_ub = Tensor(DT.bfloat16, [HALF_D, D], Position.UB)
    state_h_nz_ub = Tensor(DT.bfloat16, [HALF_D, D], Position.UB)
    dv_delta_ub = Tensor(DT.float, [HALF_L, D], Position.UB)
    dht_h_ub = dv_delta_ub.reinterpret(DT.bfloat16)
    dv_sum_ub = Tensor(DT.float, [HALF_L, D], Position.UB)
    dv_bf16_ub = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    dv_bf16_nz_ub = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    seed_ub = DBuff(DT.float, [HALF_D, D], Position.UB)
    corr_ub = Tensor(DT.float, [HALF_D, D], Position.UB)
    g_last_half_ub = Tensor(DT.bfloat16, [1, HALF_D], Position.UB)
    exp_g_last_half_ub = Tensor(DT.float, [1, HALF_D], Position.UB)

    bhv_count = B * HV
    bhv_per_core = CeilDiv(bhv_count, GetCubeNum())
    bhv_begin = Var(bhv_per_core * GetCubeIdx())
    bhv_end = Min(bhv_begin + bhv_per_core, bhv_count)
    row_begin_l = Var(GetSubBlockIdx() * HALF_L)
    row_end_l = Var(row_begin_l + HALF_L)
    row_begin_d = Var(GetSubBlockIdx() * HALF_D)
    row_end_d = Var(row_begin_d + HALF_D)
    # dht GM rides as [B, HV, D/2, 2*D]: each sub-block owns HALF_L bf16 rows.
    dht_row_begin = Var(GetSubBlockIdx() * HALF_L)
    dht_row_end = Var(dht_row_begin + HALF_L)

    with auto_sync():
        for bhv in range(bhv_begin, bhv_end):
            b_idx = Var(bhv / HV)
            hv_idx = Var(bhv % HV)
            dht_h_ub[0:HALF_L, 0:2 * D] <<= dht[b_idx, hv_idx, dht_row_begin:dht_row_end, 0:2 * D]
            cast_dht_to_f32_vf(dht_h_ub, dstate_f_ub, row_begin_d)
            snapshot_and_cast_state_vf(dstate_f_ub, state_h_ub, state_h_nz_ub, row_begin_d)

            # Initial (last) chunk C-1: strided BTHVD/BTHVL loads from the frozen
            # public seams (explicit pitch HV*D / HV*L).
            tok0 = Var((C - 1) * L)
            tok1 = Var(tok0 + L)
            gm_to_l1_nd2nz(l1_qg[0][0:L, 0:D], qg[b_idx, tok0:tok1, hv_idx, 0:D], L, D, HV * D, L)
            gm_to_l1_nd2nz(l1_w[0][0:L, 0:D], w[b_idx, tok0:tok1, hv_idx, 0:D], L, D, HV * D, L)
            gm_to_l1_nd2nz(l1_do[0][0:L, 0:D], grad_out[b_idx, tok0:tok1, hv_idx, 0:D], L, D, HV * D, L)
            gm_to_l1_nd2nz(l1_Aqk[0][0:L, 0:L], Aqk[b_idx, tok0:tok1, hv_idx, 0:L], L, L, HV * L, L)
            gm_to_l1_nd2nz(l1_vnew[0][0:L, 0:D], v_new[b_idx, tok0:tok1, hv_idx, 0:D], L, D, HV * D, L)
            _phase_matmul(l0c_seed[0], l1_qg[0].T, l1_do[0].T, phase_l0a, phase_l0b, D, D, L)
            seed_mutex.lock()
            l0c_to_ub(seed_ub[0], l0c_seed[0], M=D, N=D, N_dst=D, M_src=D, dual_mode=DualMode.SPLITM, sub_block_id=0)
            seed_mutex.ready()
            _phase_matmul(l0c_daqk, l1_do[0], l1_vnew[0], phase_l0a, phase_l0b, L, L, D)
            # dAqk -> public BTHVL, scaled by 1/sqrt(D) in the fixpipe (finalize_pre
            # re-trils, so the below-diagonal values here are masked downstream).
            l0c_to_gm_nz2nd(dAqk[b_idx, tok0:tok1, hv_idx, 0:L], l0c_daqk, L, L, HV * L, L, scale=DAQK_SCALE)
            _phase_matmul(l0c_dv0, l1_Aqk[0].T, l1_do[0].T, phase_l0a, phase_l0b, L, D, L)
            dv0_mutex.lock()
            l0c_to_ub(dv_sum_ub, l0c_dv0, M=L, N=D, N_dst=D, M_src=L, dual_mode=DualMode.SPLITM, sub_block_id=0)
            dv0_mutex.ready()

            for rev_c in range(C):
                c_idx = Var(C - 1 - rev_c)
                next_c_idx = Var(c_idx - 1)
                ctok0 = Var(c_idx * L)
                # Shared by both the next-chunk preload and the later dAqk writeback.
                ntok0 = Var(next_c_idx * L)
                # g_last read raw strided from [B,C,HV,D]; exp2 computed in-kernel.
                g_last_half_ub[0:1, 0:HALF_D] <<= g_last[b_idx, c_idx, hv_idx, row_begin_d:row_end_d]
                exp2_glast_vf(g_last_half_ub, exp_g_last_half_ub)

                state_bridge.lock()
                l1_dstate[row_begin_d:row_end_d, 0:D] <<= state_h_nz_ub[0:HALF_D, 0:D].nz()
                state_bridge.ready()

                gm_to_l1_nd2nz(l1_kg[0:L, 0:D], kg[b_idx, ctok0:ctok0 + L, hv_idx, 0:D], L, D, HV * D, L)
                state_bridge.wait()
                _phase_matmul(l0c_dvdelta, l1_kg, l1_dstate.T, phase_l0a, phase_l0b, L, D, D)
                state_bridge.free()
                # dh -> inverse seam BCHVDD [B,C,HV,D,D] (contiguous [D,D] sub-block).
                dh[b_idx, c_idx, hv_idx, row_begin_d:row_end_d, 0:D] <<= state_h_ub[0:HALF_D, 0:D]

                dvdelta_mutex.lock()
                l0c_to_ub(dv_delta_ub, l0c_dvdelta, M=L, N=D, N_dst=D, M_src=L, dual_mode=DualMode.SPLITM, sub_block_id=0)
                dvdelta_mutex.ready()

                dvdelta_mutex.wait()
                dv0_mutex.wait()
                add_dv_and_cast_bf16_vf(dv_delta_ub, dv_sum_ub, dv_bf16_ub, dv_bf16_nz_ub)
                dvdelta_mutex.free()
                # dv_sum_ub is consumed; the next chunk's dv0 fixpipe may
                # overwrite it as soon as the prefetch runs.
                dv0_mutex.free()

                dv_bridge.lock()
                l1_dv[row_begin_l:row_end_l, 0:D] <<= dv_bf16_nz_ub[0:HALF_L, 0:D].nz()
                dv_bridge.ready()
                # dv -> public BTHVD [B,T,HV,D], strided (gap (HV-1)*D).
                ub_to_gm_pad(dv[b_idx, ctok0 + row_begin_l:ctok0 + row_end_l, hv_idx, 0:D], dv_bf16_ub[0:HALF_L, 0:D], HALF_L, D, 0, (HV - 1) * D)

                if rev_c + 1 < C:
                    gm_to_l1_nd2nz(l1_qg[rev_c + 1][0:L, 0:D], qg[b_idx, ntok0:ntok0 + L, hv_idx, 0:D], L, D, HV * D, L)
                    gm_to_l1_nd2nz(l1_w[rev_c + 1][0:L, 0:D], w[b_idx, ntok0:ntok0 + L, hv_idx, 0:D], L, D, HV * D, L)
                    gm_to_l1_nd2nz(l1_do[rev_c + 1][0:L, 0:D], grad_out[b_idx, ntok0:ntok0 + L, hv_idx, 0:D], L, D, HV * D, L)
                    gm_to_l1_nd2nz(l1_Aqk[rev_c + 1][0:L, 0:L], Aqk[b_idx, ntok0:ntok0 + L, hv_idx, 0:L], L, L, HV * L, L)
                    gm_to_l1_nd2nz(l1_vnew[rev_c + 1][0:L, 0:D], v_new[b_idx, ntok0:ntok0 + L, hv_idx, 0:D], L, D, HV * D, L)
                    _phase_matmul(l0c_seed[rev_c + 1], l1_qg[rev_c + 1].T, l1_do[rev_c + 1].T, phase_l0a, phase_l0b, D, D, L)
                    seed_mutex.lock()
                    l0c_to_ub(seed_ub[rev_c + 1], l0c_seed[rev_c + 1], M=D, N=D, N_dst=D, M_src=D, dual_mode=DualMode.SPLITM, sub_block_id=0)
                    seed_mutex.ready()

                dv_bridge.wait()
                _phase_matmul(l0c_corr, l1_w[rev_c].T, l1_dv.T, phase_l0a, phase_l0b, D, D, L)
                dv_bridge.free()

                corr_mutex.lock()
                l0c_to_ub(corr_ub, l0c_corr, M=D, N=D, N_dst=D, M_src=D, dual_mode=DualMode.SPLITM, sub_block_id=0)
                corr_mutex.ready()

                if rev_c + 1 < C:
                    _phase_matmul(l0c_daqk, l1_do[rev_c + 1], l1_vnew[rev_c + 1], phase_l0a, phase_l0b, L, L, D)
                    l0c_to_gm_nz2nd(dAqk[b_idx, ntok0:ntok0 + L, hv_idx, 0:L], l0c_daqk, L, L, HV * L, L, scale=DAQK_SCALE)
                    _phase_matmul(l0c_dv0, l1_Aqk[rev_c + 1].T, l1_do[rev_c + 1].T, phase_l0a, phase_l0b, L, D, L)
                    dv0_mutex.lock()
                    l0c_to_ub(dv_sum_ub, l0c_dv0, M=L, N=D, N_dst=D, M_src=L, dual_mode=DualMode.SPLITM, sub_block_id=0)
                    dv0_mutex.ready()
                seed_mutex.wait()
                corr_mutex.wait()
                if rev_c + 1 < C:
                    update_dstate_vf(seed_ub[rev_c], corr_ub, dstate_f_ub, state_h_ub, state_h_nz_ub, exp_g_last_half_ub, row_begin_d)
                else:
                    update_dstate_final_vf(seed_ub[rev_c], corr_ub, dstate_f_ub, state_h_ub, exp_g_last_half_ub, row_begin_d)
                corr_mutex.free()
                seed_mutex.free()

            dh0[b_idx, hv_idx, row_begin_d:row_end_d, 0:D] <<= state_h_ub[0:HALF_D, 0:D]
            bar_all()

    return dAqk, dh, dv, dh0

# ----------------------------------------------------------------------------------------------------
# inverse_mm.py
# KDA kernel port; the launch ABI for this stage is main.py's `execute`.
#
# The arithmetic and schedule come from the reviewed source; only imports and
# typed kernel signatures are migrated here.
# M10 hardware repair: use integer division for the two static VF trip counts.
# Float bounds made the vendor compiler assert before producing a vector object.
# ----------------------------------------------------------------------------------------------------

@vf()
def negate_cast_dw_vf(dvh_f_ub: Tensor, dw_h_ub: Tensor):
    reg_f32 = Reg(DT.float)
    n_loops = Var(HALF_L * D // 64)
    for i in range(n_loops):
        reg_f32 <<= dvh_f_ub[i * 64]
        reg_f32 <<= reg_f32 * (-1.0)
        dw_h_ub[i * 64] <<= reg_f32
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

@vf()
def cast_out_vf(out_f_ub: Tensor, out_h_ub: Tensor):
    reg_f32 = Reg(DT.float)
    n_loops = Var(HALF_L * D // 64)
    for i in range(n_loops):
        reg_f32 <<= out_f_ub[i * 64]
        out_h_ub[i * 64] <<= reg_f32
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

@kernel()
def inverse_mm_kernel(do_bf16: GM[bf16, ('B', 'T', 'HV', 128)], vnew_bf16: GM[bf16, ('B', 'T', 'HV', 128)], dv_bf16: GM[bf16, ('B', 'T', 'HV', 128)], h_bf16: GM[bf16, ('B', 'C', 'HV', 128, 128)], dh_bf16: GM[bf16, ('B', 'C', 'HV', 128, 128)], Akk_bf16: GM[bf16, ('B', 'T', 'HV', 64)], d_qg: GM[bf16, ('B', 'HV', 'C', 64, 128)], d_kg: GM[bf16, ('B', 'HV', 'C', 64, 128)], d_vh: GM[bf16, ('B', 'HV', 'C', 64, 128)], d_v_beta: GM[bf16, ('B', 'HV', 'C', 64, 128)], d_k_beta_g: GM[bf16, ('B', 'HV', 'C', 64, 128)], B: i32, HV: i32, C: i32):
    # Depth 2 on the two CvMutex bridges, 3 on the VcMutex, and that split is forced rather
    # than tuned. `local_mutex` hands out one ID per slot from a flat counter and the cube side
    # has 32; at depth 3 everywhere this kernel declared 34 and was refused before anything was
    # built, on every lowered path. `l1_Akk` and `l1_dw` are the two that cannot come down:
    # item p writes them and item p-2 reads them, so three are live at once. The FIX -> UB
    # handoffs are consumed inside the item that starts them, so two rotate cleanly there.
    dvh_mutex = CvMutex(0, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    dw_bridge = VcMutex(
        1,
        depth=3,
        src_start_pipe=Pipe.MTE3,
        src_end_pipe=Pipe.MTE3,
        dst_start_pipe=Pipe.MTE1,
        dst_end_pipe=Pipe.MTE1,
    )
    out_mutex = CvMutex(2, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE3)

    l1_do = DBuff(DT.bfloat16, [L, D], Position.L1)
    l1_vnew = DBuff(DT.bfloat16, [L, D], Position.L1)
    l1_dv = DBuff(DT.bfloat16, [L, D], Position.L1)
    l1_h = DBuff(DT.bfloat16, [D, D], Position.L1)
    l1_dh = DBuff(DT.bfloat16, [D, D], Position.L1)
    l1_Akk = TBuff(DT.bfloat16, [L, L], Position.L1)
    l1_dw = TBuff(DT.bfloat16, [L, D], Position.L1)

    l0c_dqg = Tensor(DT.float, [L, D], Position.L0C)
    l0c_dkg = Tensor(DT.float, [L, D], Position.L0C)
    l0c_dvh = DBuff(DT.float, [L, D], Position.L0C)
    l0c_dvbeta = DBuff(DT.float, [L, D], Position.L0C)
    l0c_dkbetag = DBuff(DT.float, [L, D], Position.L0C)

    dvh_f_ub = DBuff(DT.float, [HALF_L, D], Position.UB)
    dw_h_ub = TBuff(DT.bfloat16, [HALF_L, D], Position.UB)
    out_f_ub = DBuff(DT.float, [HALF_L, D], Position.UB)
    out_h_ub = DBuff(DT.bfloat16, [HALF_L, D], Position.UB)

    work_count = Var(B * HV * C)
    work_per_core = CeilDiv(work_count, GetCubeNum())
    work_begin = Var(work_per_core * GetCubeIdx())
    work_end = Min(work_begin + work_per_core, work_count)
    row_begin_l = Var(GetSubBlockIdx() * HALF_L)
    row_end_l = Var(row_begin_l + HALF_L)

    with auto_sync():
        for pipe_work in range(work_begin, work_end + 2):
            if pipe_work < work_end:
                c_idx = Var(pipe_work % C)
                bhv_idx = Var(pipe_work / C)
                hv_idx = Var(bhv_idx % HV)
                b_idx = Var(bhv_idx / HV)
                row0 = Var(c_idx * L)
                row1 = Var(row0 + L)

                # Strided BTHVD/BTHVL reads from the frozen public layout: explicit
                # N_src = HV*D / HV*L pitch (the <<= auto-infer would use D/L and
                # silently read compacted adjacent tokens -- wrong data, no crash).
                # h/dh stay contiguous [D, D] blocks (frozen [B, C, HV, D, D]); only
                # their index order changes, so the <<= auto-infer (N_src=D) is right.
                gm_to_l1_nd2nz(l1_dv[pipe_work][0:L, 0:D], dv_bf16[b_idx, row0:row1, hv_idx, 0:D], L, D, HV * D, L)
                l1_h[pipe_work][0:D, 0:D] <<= h_bf16[b_idx, c_idx, hv_idx, 0:D, 0:D]
                gm_to_l1_nd2nz(l1_do[pipe_work][0:L, 0:D], do_bf16[b_idx, row0:row1, hv_idx, 0:D], L, D, HV * D, L)
                gm_to_l1_nd2nz(l1_vnew[pipe_work][0:L, 0:D], vnew_bf16[b_idx, row0:row1, hv_idx, 0:D], L, D, HV * D, L)
                l1_dh[pipe_work][0:D, 0:D] <<= dh_bf16[b_idx, c_idx, hv_idx, 0:D, 0:D]
                gm_to_l1_nd2nz(l1_Akk[pipe_work][0:L, 0:L], Akk_bf16[b_idx, row0:row1, hv_idx, 0:L], L, L, HV * L, L)

                # The dvh matmul leads: it gates the FIX -> UB -> L1 d_w chain.
                matmul(l0c_dvh[pipe_work], l1_dv[pipe_work], l1_h[pipe_work], m=L, n=D, k=D, splitn=D)
                matmul(l0c_dqg, l1_do[pipe_work], l1_h[pipe_work], m=L, n=D, k=D, splitn=D)
                matmul(l0c_dvbeta[pipe_work], l1_Akk[pipe_work].T, l1_dv[pipe_work].T, m=L, n=D, k=L, splitn=D)
                matmul(l0c_dkg, l1_vnew[pipe_work], l1_dh[pipe_work], m=L, n=D, k=D, splitn=D)

                d_vh[b_idx, hv_idx, c_idx, 0:L, 0:D] <<= l0c_dvh[pipe_work]

                dvh_mutex.lock()
                l0c_to_ub(dvh_f_ub[pipe_work], l0c_dvh[pipe_work], M=L, N=D, N_dst=D, M_src=L, dual_mode=DualMode.SPLITM, sub_block_id=0)
                dvh_mutex.ready()
                dvh_mutex.wait()
                negate_cast_dw_vf(dvh_f_ub[pipe_work], dw_h_ub[pipe_work])
                dvh_mutex.free()

                dw_bridge.lock()
                l1_dw[pipe_work][row_begin_l:row_end_l, 0:D] <<= dw_h_ub[pipe_work][0:HALF_L, 0:D]
                dw_bridge.ready()

                # dqg / dvbeta / dkg leave through the vector side: FIX is L0C
                # read bound, so l0c_to_ub (138cy) + MTE3 store beats five
                # l0c_to_gm stores (1324cy each).
                out_mutex.lock()
                l0c_to_ub(out_f_ub[3 * pipe_work], l0c_dqg, M=L, N=D, N_dst=D, M_src=L, dual_mode=DualMode.SPLITM, sub_block_id=0)
                out_mutex.ready()
                out_mutex.wait()
                cast_out_vf(out_f_ub[3 * pipe_work], out_h_ub[3 * pipe_work])
                d_qg[b_idx, hv_idx, c_idx, row_begin_l:row_end_l, 0:D] <<= out_h_ub[3 * pipe_work][0:HALF_L, 0:D]
                out_mutex.free()

                out_mutex.lock()
                l0c_to_ub(out_f_ub[3 * pipe_work + 1], l0c_dvbeta[pipe_work], M=L, N=D, N_dst=D, M_src=L, dual_mode=DualMode.SPLITM, sub_block_id=0)
                out_mutex.ready()
                out_mutex.wait()
                cast_out_vf(out_f_ub[3 * pipe_work + 1], out_h_ub[3 * pipe_work + 1])
                d_v_beta[b_idx, hv_idx, c_idx, row_begin_l:row_end_l, 0:D] <<= out_h_ub[3 * pipe_work + 1][0:HALF_L, 0:D]
                out_mutex.free()

                out_mutex.lock()
                l0c_to_ub(out_f_ub[3 * pipe_work + 2], l0c_dkg, M=L, N=D, N_dst=D, M_src=L, dual_mode=DualMode.SPLITM, sub_block_id=0)
                out_mutex.ready()
                out_mutex.wait()
                cast_out_vf(out_f_ub[3 * pipe_work + 2], out_h_ub[3 * pipe_work + 2])
                d_kg[b_idx, hv_idx, c_idx, row_begin_l:row_end_l, 0:D] <<= out_h_ub[3 * pipe_work + 2][0:HALF_L, 0:D]
                out_mutex.free()

            if pipe_work >= work_begin + 2:
                prev_work = Var(pipe_work - 2)
                prev_c = Var(prev_work % C)
                prev_bhv = Var(prev_work / C)
                prev_hv = Var(prev_bhv % HV)
                prev_b = Var(prev_bhv / HV)

                dw_bridge.wait()
                matmul(l0c_dkbetag[prev_work], l1_Akk[prev_work].T, l1_dw[prev_work].T, m=L, n=D, k=L, splitn=D)
                dw_bridge.free()

                d_k_beta_g[prev_b, prev_hv, prev_c, 0:L, 0:D] <<= l0c_dkbetag[prev_work]

    return d_qg, d_kg, d_vh, d_v_beta, d_k_beta_g

# ----------------------------------------------------------------------------------------------------
# inverse_epilogue.py
# KDA kernel port; the launch ABI for this stage is main.py's `execute`.
#
# The arithmetic and cube-group partition come from the reviewed source.
# Whole-chunk vector work is owned only by subblock zero to eliminate duplicate
# GM writes; that is a deliberate correction to the source, not an omission.
# ----------------------------------------------------------------------------------------------------

REGS_D = D // 64


SCALE = 1.0 / (D ** 0.5)

BETA_SCALAR_PACK = DT.bfloat16.C0

DBETA_SCALAR_PACK = DT.float.C0

@vf()
def compute_exp_glast_vf(g_ub: Tensor, exp_glast_ub: Tensor):
    glast = RegList(DT.float, REGS_D)
    glast <<= g_ub[L - 1:L, 0:D]
    glast <<= glast * LN2
    glast <<= glast.exp()
    exp_glast_ub[0:1, 0:D] <<= glast
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

@vf()
def dglast_term1_vf(h_ub: Tensor, dh_ub: Tensor, exp_glast_ub: Tensor, dglast_ub: Tensor, row_begin: Var, rows: Var):
    h_regs = RegList(DT.float, REGS_D)
    dh_regs = RegList(DT.float, REGS_D)
    prod = RegList(DT.float, REGS_D)
    dot = Reg(DT.float)
    scale = Reg(DT.float)
    for r in range(rows):
        gk = Var(row_begin + r)
        h_regs <<= h_ub[r:r + 1, 0:D]
        dh_regs <<= dh_ub[r:r + 1, 0:D]
        prod <<= h_regs * dh_regs
        dot <<= prod.cadd()
        scale <<= exp_glast_ub[0:1, gk:gk + 1].single()
        dot <<= dot * scale
        dglast_ub[0:1, gk:gk + 1] <<= dot.single_value()
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

@vf()
def inverse_epilogue_vf(
    q_ub: Tensor,
    k_ub: Tensor,
    v_ub: Tensor,
    g_ub: Tensor,
    dqg_ub: Tensor,
    dkg_ub: Tensor,
    dvbeta_ub: Tensor,
    dkbetag_ub: Tensor,
    beta_ub: Tensor,
    dglast_ub: Tensor,
    dq_ub: Tensor,
    dk_ub: Tensor,
    dv_ub: Tensor,
    dbeta_ub: Tensor,
    dg_ub: Tensor,
    kexp_ub: Tensor,
    rows: Var,
):
    glast = RegList(DT.float, REGS_D)
    g = RegList(DT.float, REGS_D)
    expg = RegList(DT.float, REGS_D)
    explmg = RegList(DT.float, REGS_D)
    q = RegList(DT.float, REGS_D)
    k = RegList(DT.float, REGS_D)
    v = RegList(DT.float, REGS_D)
    dqg = RegList(DT.float, REGS_D)
    dkg = RegList(DT.float, REGS_D)
    dvbeta = RegList(DT.float, REGS_D)
    dkbetag = RegList(DT.float, REGS_D)
    dq = RegList(DT.float, REGS_D)
    dkkg = RegList(DT.float, REGS_D)
    kexp = RegList(DT.float, REGS_D)
    dkw = RegList(DT.float, REGS_D)
    dgw = RegList(DT.float, REGS_D)
    dg = RegList(DT.float, REGS_D)
    tmp = RegList(DT.float, REGS_D)
    prod = RegList(DT.float, REGS_D)
    acc = RegList(DT.float, REGS_D)
    beta_r = Reg(DT.float)
    db1 = Reg(DT.float)
    db2 = Reg(DT.float)

    glast <<= g_ub[L - 1:L, 0:D]
    acc.fill(0.0)

    for r in range(rows):
        g <<= g_ub[r:r + 1, 0:D]
        tmp <<= g * LN2
        expg <<= tmp.exp()
        tmp <<= glast - g
        tmp <<= tmp * LN2
        explmg <<= tmp.exp()

        # dq = d_qg * exp_g * scale  (fp32 kept for dg)
        dqg <<= dqg_ub[r:r + 1, 0:D]
        dq <<= dqg * expg
        dq <<= dq * SCALE
        dq_ub[r * D] <<= dq[0]
        dq_ub[r * D + 64] <<= dq[1]

        # dk_from_kg equals d_kg * exp2(g_last - g).
        dkg <<= dkg_ub[r:r + 1, 0:D]
        dkkg <<= dkg * explmg

        # k_exp equals k * exp_g.
        k <<= k_ub[r:r + 1, 0:D]
        kexp <<= k * expg
        kexp_ub[r * D] <<= kexp[0]
        kexp_ub[r * D + 64] <<= kexp[1]

        # dv equals d_v_beta * beta.
        beta_r <<= beta_ub[r:r + 1, 0:1].single()
        dvbeta <<= dvbeta_ub[r:r + 1, 0:D]
        tmp <<= dvbeta * beta_r
        dv_ub[r * D] <<= tmp[0]
        dv_ub[r * D + 64] <<= tmp[1]

        # dbeta equals (d_v_beta * v).sum + (d_k_beta_g * k_exp).sum.
        v <<= v_ub[r:r + 1, 0:D]
        prod <<= dvbeta * v
        db1 <<= prod.cadd()
        dkbetag <<= dkbetag_ub[r:r + 1, 0:D]
        prod <<= dkbetag * kexp
        db2 <<= prod.cadd()
        db1 <<= db1 + db2
        dbeta_ub[r:r + 1, 0:1] <<= db1.single_value()

        # dk_from_w equals d_k_beta_g * beta * exp_g ; dk_hv equals dk_from_kg + dk_from_w.
        dkw <<= dkbetag * beta_r
        dkw <<= dkw * expg
        tmp <<= dkkg + dkw
        dk_ub[r * D] <<= tmp[0]
        dk_ub[r * D + 64] <<= tmp[1]

        # dg_from_w equals d_k_beta_g * k_exp * beta.
        dgw <<= dkbetag * kexp
        dgw <<= dgw * beta_r

        # dg equals q * dq - k * dk_from_kg + dg_from_w.
        q <<= q_ub[r:r + 1, 0:D]
        dg <<= q * dq
        prod <<= k * dkkg
        dg <<= dg - prod
        dg <<= dg + dgw
        dg_ub[r * D] <<= dg[0]
        dg_ub[r * D + 64] <<= dg[1]

        # term2 accumulator: sum_token (k * dk_from_kg)
        acc <<= acc + prod

    # d_g_last = term1 (dglast_ub) + term2 (acc); add into dg row L-1 only.
    # The loop already wrote dg_ub[L-1]; barrier so that store is visible to the
    # read-modify-write below (store -> load on the same range).
    tmp <<= dglast_ub[0:1, 0:D]
    acc <<= acc + tmp
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)
    dg <<= dg_ub[L - 1:L, 0:D]
    dg <<= dg + acc
    vf_barrier(VfPipe.STORE, VfPipe.STORE)
    dg_ub[(L - 1) * D] <<= dg[0]
    dg_ub[(L - 1) * D + 64] <<= dg[1]
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

@kernel()
def inverse_epilogue_kernel(d_qg: GM[bf16, ('B', 'HV', 'C', 64, 128)], d_kg: GM[bf16, ('B', 'HV', 'C', 64, 128)], d_v_beta: GM[bf16, ('B', 'HV', 'C', 64, 128)], d_k_beta_g: GM[bf16, ('B', 'HV', 'C', 64, 128)], q: GM[bf16, ('B', 'T', 'Hq', 128)], k: GM[bf16, ('B', 'T', 'Hq', 128)], v: GM[bf16, ('B', 'T', 'HV', 128)], g_cumsum: GM[bf16, ('B', 'T', 'HV', 128)], beta: GM[bf16, ('B', 'T', 'HV')], h: GM[bf16, ('B', 'C', 'HV', 128, 128)], dh: GM[bf16, ('B', 'C', 'HV', 128, 128)], dq_hv: GM[bf16, ('B', 'T', 'HV', 128)], dk_hv: GM[bf16, ('B', 'T', 'HV', 128)], dv: GM[bf16, ('B', 'T', 'HV', 128)], dbeta: GM[f32, ('B', 'T', 'HV')], dg_core: GM[bf16, ('B', 'T', 'HV', 128)], k_exp: GM[bf16, ('B', 'HV', 'C', 64, 128)], B: i32, HV: i32, C: i32, Hq: i32, G: i32):
    dqg_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    dkg_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    dvbeta_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    dkbetag_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    q_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    k_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    v_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    g_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    # Scalar burst copies need a 32B physical row; packing beta/dbeta as [L, C0]
    # keeps each token's scalar at a row start instead of a sparse [1, L] layout.
    beta_ub = Tensor(DT.bfloat16, [L, BETA_SCALAR_PACK], Position.UB)
    h_ub = Tensor(DT.bfloat16, [HALF_D, D], Position.UB)
    dh_ub = Tensor(DT.bfloat16, [HALF_D, D], Position.UB)

    exp_glast_ub = Tensor(DT.float, [1, D], Position.UB)
    dglast_ub = Tensor(DT.float, [1, D], Position.UB)

    dq_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    dk_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    dv_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    dg_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    kexp_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    dbeta_ub = Tensor(DT.float, [L, DBETA_SCALAR_PACK], Position.UB)

    work_count = Var(B * C)
    work_per_core = CeilDiv(work_count, GetCubeNum())
    work_begin = Var(work_per_core * GetCubeIdx())
    work_end = Min(work_begin + work_per_core, work_count)

    with auto_sync():
        # One vector subblock owns the whole chunk; the peer must not duplicate GM writes.
        if GetSubBlockIdx() == 0:
            for work in range(work_begin, work_end):
                c_idx = Var(work % C)
                b_idx = Var(work / C)
                # All heads share scalar GM transaction blocks and stay on this core.
                for hv_idx in range(HV):
                    tok0 = Var(c_idx * L)
                    tok1 = Var(tok0 + L)
                    h_idx = Var(hv_idx / G)   # H->HV gather: q/k own a contiguous G-block

                    # g_cumsum strided from frozen BTHVD (explicit (HV-1)*D gap).
                    gm_to_ub_pad(g_ub[0:L, 0:D], g_cumsum[b_idx, tok0:tok1, hv_idx, 0:D], L, D, (HV - 1) * D, 0)
                    compute_exp_glast_vf(g_ub, exp_glast_ub)

                    for half in range(2):
                        row0 = Var(half * HALF_D)
                        row1 = Var(row0 + HALF_D)
                        # h/dh contiguous [D,D] blocks from frozen [B,C,HV,D,D]: index
                        # reorder only (the half-row slice keeps source row pitch D).
                        h_ub[0:HALF_D, 0:D] <<= h[b_idx, c_idx, hv_idx, row0:row1, 0:D]
                        dh_ub[0:HALF_D, 0:D] <<= dh[b_idx, c_idx, hv_idx, row0:row1, 0:D]
                        dglast_term1_vf(h_ub, dh_ub, exp_glast_ub, dglast_ub, row0, Var(HALF_D))

                    dqg_ub[0:L, 0:D] <<= d_qg[b_idx, hv_idx, c_idx, 0:L, 0:D]
                    dkg_ub[0:L, 0:D] <<= d_kg[b_idx, hv_idx, c_idx, 0:L, 0:D]
                    dvbeta_ub[0:L, 0:D] <<= d_v_beta[b_idx, hv_idx, c_idx, 0:L, 0:D]
                    dkbetag_ub[0:L, 0:D] <<= d_k_beta_g[b_idx, hv_idx, c_idx, 0:L, 0:D]
                    # q/k strided + H->HV gather from frozen [B,T,H,D] (gap (Hq-1)*D);
                    # v strided from frozen BTHVD; beta strided burst_len=1 from BTHV.
                    gm_to_ub_pad(q_ub[0:L, 0:D], q[b_idx, tok0:tok1, h_idx, 0:D], L, D, (Hq - 1) * D, 0)
                    gm_to_ub_pad(k_ub[0:L, 0:D], k[b_idx, tok0:tok1, h_idx, 0:D], L, D, (Hq - 1) * D, 0)
                    gm_to_ub_pad(v_ub[0:L, 0:D], v[b_idx, tok0:tok1, hv_idx, 0:D], L, D, (HV - 1) * D, 0)
                    gm_to_ub_pad(beta_ub[0:L, 0:1], beta[b_idx, tok0:tok1, hv_idx], L, 1, HV - 1, 0)

                    inverse_epilogue_vf(
                        q_ub, k_ub, v_ub, g_ub, dqg_ub, dkg_ub, dvbeta_ub, dkbetag_ub,
                        beta_ub, dglast_ub,
                        dq_ub, dk_ub, dv_ub, dbeta_ub, dg_ub, kexp_ub, Var(L),
                    )

                    # Inter-stage grads written strided to the frozen public seams:
                    # dq_hv/dk_hv/dv/dg_core BTHVD ((HV-1)*D gap), dbeta BTHV fp32
                    # ((HV-1) gap). k_exp stays GM-native (internal -> inverse_dainv).
                    ub_to_gm_pad(dq_hv[b_idx, tok0:tok1, hv_idx, 0:D], dq_ub[0:L, 0:D], L, D, 0, (HV - 1) * D)
                    ub_to_gm_pad(dk_hv[b_idx, tok0:tok1, hv_idx, 0:D], dk_ub[0:L, 0:D], L, D, 0, (HV - 1) * D)
                    ub_to_gm_pad(dv[b_idx, tok0:tok1, hv_idx, 0:D], dv_ub[0:L, 0:D], L, D, 0, (HV - 1) * D)
                    ub_to_gm_pad(dbeta[b_idx, tok0:tok1, hv_idx], dbeta_ub[0:L, 0:1], L, 1, 0, HV - 1)
                    ub_to_gm_pad(dg_core[b_idx, tok0:tok1, hv_idx, 0:D], dg_ub[0:L, 0:D], L, D, 0, (HV - 1) * D)
                    k_exp[b_idx, hv_idx, c_idx, 0:L, 0:D] <<= kexp_ub[0:L, 0:D]

    return dq_hv, dk_hv, dv, dbeta, dg_core, k_exp

# ----------------------------------------------------------------------------------------------------
# inverse_dainv.py
# KDA kernel port; the launch ABI for this stage is main.py's `execute`.
#
# The arithmetic and schedule come from the reviewed source. Migration corrects
# the two rectangular matmul RHS views to B[N,K], preserving the documented
# dv@v.T + dw@kexp.T formula.
# ----------------------------------------------------------------------------------------------------

SCALAR_PACK_BF16 = DT.bfloat16.C0

SCALAR_PACK_F32 = DT.float.C0

@vf()
def dtri_mask_vf(
    dainv_a_ub: Tensor,
    dainv_b_ub: Tensor,
    beta_h_ub: Tensor,
    beta_f_ub: Tensor,
    dtri_ub: Tensor,
    row_begin: Var,
    rows: Var,
):
    cols = Reg(DT.int)
    a = Reg(DT.float)
    b = Reg(DT.float)
    beta_h = Reg(DT.bfloat16)
    beta_idx_i32 = Reg(DT.int)
    beta_idx_u32 = beta_idx_i32.reinterpret(DT.uint32)
    betav = Reg(DT.float)
    out = Reg(DT.float)
    zero = Reg(DT.float)
    mask = MaskReg(DT.int, init_mode=MaskType.NONE)

    zero <<= 0.0
    for c in range(L):
        beta_h <<= beta_h_ub[c:c + 1, 0:1].single()
        betav <<= beta_h.cast()
        beta_f_ub[c:c + 1, 0:1] <<= betav.single_value()
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)
    beta_idx_i32.arange(0)
    beta_idx_i32 <<= beta_idx_i32 * SCALAR_PACK_F32
    betav.ub_gather(beta_f_ub, beta_idx_u32)
    for r in range(rows):
        rg = Var(row_begin + r)
        cols.arange(0)
        a <<= dainv_a_ub[r:r + 1, 0:L]
        b <<= dainv_b_ub[r:r + 1, 0:L]
        a <<= a + b
        a <<= a * betav
        compare(mask, cols, rg, CompareMode.LT)
        select(out, a, zero, mask=mask)
        dtri_ub[r:r + 1, 0:L] <<= out
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

@kernel()
def inverse_dainv_kernel(dv: GM[bf16, ('B', 'T', 'HV', 128)], v: GM[bf16, ('B', 'T', 'HV', 128)], d_w: GM[bf16, ('B', 'HV', 'C', 64, 128)], k_exp: GM[bf16, ('B', 'HV', 'C', 64, 128)], beta: GM[bf16, ('B', 'T', 'HV')], dtri: GM[bf16, ('B', 'HV', 'C', 64, 64)], B: i32, HV: i32, C: i32):
    dainv_mutex = CvMutex(0, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)

    l1_dv = DBuff(DT.bfloat16, [L, D], Position.L1)
    l1_v = DBuff(DT.bfloat16, [L, D], Position.L1)
    l1_dw = DBuff(DT.bfloat16, [L, D], Position.L1)
    l1_kexp = DBuff(DT.bfloat16, [L, D], Position.L1)
    l0c_a = DBuff(DT.float, [L, L], Position.L0C)
    l0c_b = DBuff(DT.float, [L, L], Position.L0C)
    dainv_a_ub = DBuff(DT.float, [HALF_L, L], Position.UB)
    dainv_b_ub = DBuff(DT.float, [HALF_L, L], Position.UB)
    beta_h_ub = DBuff(DT.bfloat16, [L, SCALAR_PACK_BF16], Position.UB)
    beta_f_ub = DBuff(DT.float, [L, SCALAR_PACK_F32], Position.UB)
    dtri_ub = DBuff(DT.bfloat16, [HALF_L, L], Position.UB)

    work_count = Var(B * HV * C)
    work_per_core = CeilDiv(work_count, GetCubeNum())
    work_begin = Var(work_per_core * GetCubeIdx())
    work_end = Min(work_begin + work_per_core, work_count)
    row_begin_l = Var(GetSubBlockIdx() * HALF_L)
    row_end_l = Var(row_begin_l + HALF_L)
    slot = Var(0)

    with auto_sync():
        for work in range(work_begin, work_end):
            c_idx = Var(work % C)
            bhv = Var(work / C)
            hv_idx = Var(bhv % HV)
            b_idx = Var(bhv / HV)
            row0 = Var(c_idx * L)
            row1 = Var(row0 + L)

            # dv/v strided from frozen BTHVD (explicit HV*D pitch); d_w/k_exp stay
            # GM-native (internal); beta strided bf16 from frozen BTHV, upcast to
            # fp32 implicitly on the Reg load inside dtri_mask_vf.
            gm_to_l1_nd2nz(l1_dv[slot][0:L, 0:D], dv[b_idx, row0:row1, hv_idx, 0:D], L, D, HV * D, L)
            gm_to_l1_nd2nz(l1_v[slot][0:L, 0:D], v[b_idx, row0:row1, hv_idx, 0:D], L, D, HV * D, L)
            l1_dw[slot][0:L, 0:D] <<= d_w[b_idx, hv_idx, c_idx, 0:L, 0:D]
            l1_kexp[slot][0:L, 0:D] <<= k_exp[b_idx, hv_idx, c_idx, 0:L, 0:D]
            gm_to_ub_pad(beta_h_ub[slot][0:L, 0:1], beta[b_idx, row0:row1, hv_idx], L, 1, HV - 1, 0)

            matmul(l0c_a[slot], l1_dv[slot], l1_v[slot], m=L, n=L, k=D, splitn=L)
            matmul(l0c_b[slot], l1_dw[slot], l1_kexp[slot], m=L, n=L, k=D, splitn=L)

            dainv_mutex.lock()
            l0c_to_ub(dainv_a_ub[slot], l0c_a[slot], M=L, N=L, N_dst=L, M_src=L, dual_mode=DualMode.SPLITM, sub_block_id=0)
            l0c_to_ub(dainv_b_ub[slot], l0c_b[slot], M=L, N=L, N_dst=L, M_src=L, dual_mode=DualMode.SPLITM, sub_block_id=0)
            dainv_mutex.ready()
            dainv_mutex.wait()
            dtri_mask_vf(dainv_a_ub[slot], dainv_b_ub[slot], beta_h_ub[slot], beta_f_ub[slot], dtri_ub[slot], row_begin_l, Var(HALF_L))
            dainv_mutex.free()
            dtri[b_idx, hv_idx, c_idx, row_begin_l:row_end_l, 0:L] <<= dtri_ub[slot][0:HALF_L, 0:L]
            slot += 1

    return dtri

# ----------------------------------------------------------------------------------------------------
# inverse_dakk_fused.py
# KDA kernel port; the launch ABI for this stage is main.py's `execute`.
#
# The arithmetic and schedule come from the reviewed source; only imports and
# typed kernel signatures are migrated here.
# ----------------------------------------------------------------------------------------------------

@vf()
def cast_tmp_to_bf16_vf(tmp_f_ub: Tensor, tmp_h_ub: Tensor):
    reg_f32 = Reg(DT.float)
    n_loops = Var(HALF_L * L // 64)  # M10: the static VF loop bound must be integral.
    for i in range(n_loops):
        reg_f32 <<= tmp_f_ub[i * 64]
        tmp_h_ub[i * 64] <<= reg_f32
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

@vf()
def neg_tril_vf(raw_ub: Tensor, dakk_ub: Tensor, row_begin: Var, rows: Var):
    # dAkk = -tril(dL_raw, -1): negate the strict-lower triangle, zero the rest.
    cols = Reg(DT.int)
    row = Reg(DT.float)
    out = Reg(DT.float)
    zero = Reg(DT.float)
    mask = MaskReg(DT.int, init_mode=MaskType.NONE)
    zero <<= 0.0
    for r in range(rows):
        rg = Var(row_begin + r)
        cols.arange(0)
        row <<= raw_ub[r:r + 1, 0:L]
        row <<= row * (-1.0)
        compare(mask, cols, rg, CompareMode.LT)
        select(out, row, zero, mask=mask)
        dakk_ub[r:r + 1, 0:L] <<= out
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

@kernel()
def inverse_dakk_fused_kernel(Akk_bf16: GM[bf16, ('B', 'T', 'HV', 64)], Dtri_bf16: GM[bf16, ('B', 'HV', 'T', 64)], dAkk: GM[bf16, ('B', 'T', 'HV', 64)], B: i32, HV: i32, T: i32):
    tmp_mutex = CvMutex(0, depth=3, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    tmp_bridge = VcMutex(
        1,
        depth=3,
        src_start_pipe=Pipe.MTE3,
        src_end_pipe=Pipe.MTE3,
        dst_start_pipe=Pipe.MTE1,
        dst_end_pipe=Pipe.MTE1,
    )
    out_mutex = CvMutex(2, depth=3, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)

    l1_Akk = TBuff(DT.bfloat16, [L, L], Position.L1)
    l1_Dtri = TBuff(DT.bfloat16, [L, L], Position.L1)
    l1_tmp = TBuff(DT.bfloat16, [L, L], Position.L1)
    l0c_tmp = TBuff(DT.float, [L, L], Position.L0C)
    l0c_out = TBuff(DT.float, [L, L], Position.L0C)
    tmp_f_ub = TBuff(DT.float, [HALF_L, L], Position.UB)
    tmp_h_ub = TBuff(DT.bfloat16, [HALF_L, L], Position.UB)
    dakk_f_ub = TBuff(DT.float, [HALF_L, L], Position.UB)
    dakk_h_ub = TBuff(DT.bfloat16, [HALF_L, L], Position.UB)

    chunk_count = Var(T / L)
    work_count = Var(B * HV * chunk_count)
    work_per_core = CeilDiv(work_count, GetCubeNum())
    work_begin = Var(work_per_core * GetCubeIdx())
    work_end = Min(work_begin + work_per_core, work_count)
    row_begin_l = Var(GetSubBlockIdx() * HALF_L)
    row_end_l = Var(row_begin_l + HALF_L)

    # Two-stage software pipeline: the mm2 of work item i runs two iterations
    # behind its producer chain, so MTE2/M/FIX/V/MTE3 of consecutive chunks
    # overlap instead of serializing on the FIX -> UB -> L1 round trip.
    with auto_sync():
        for pipe_work in range(work_begin, work_end + 2):
            if pipe_work < work_end:
                c_idx = Var(pipe_work % chunk_count)
                bhv_idx = Var(pipe_work / chunk_count)
                hv_idx = Var(bhv_idx % HV)
                b_idx = Var(bhv_idx / HV)
                row0 = Var(c_idx * L)
                row1 = Var(row0 + L)

                # Akk strided from frozen BTHVL (explicit HV*L pitch); Dtri stays
                # GM-native [B, HV, T, L] (internal, from inverse_dainv).
                gm_to_l1_nd2nz(l1_Akk[pipe_work][0:L, 0:L], Akk_bf16[b_idx, row0:row1, hv_idx, 0:L], L, L, HV * L, L)
                l1_Dtri[pipe_work][0:L, 0:L] <<= Dtri_bf16[b_idx, hv_idx, row0:row1, 0:L]
                matmul(l0c_tmp[pipe_work], l1_Dtri[pipe_work], l1_Akk[pipe_work], m=L, n=L, k=L, splitn=L)

                tmp_mutex.lock()
                l0c_to_ub(tmp_f_ub[pipe_work], l0c_tmp[pipe_work], M=L, N=L, N_dst=L, M_src=L, dual_mode=DualMode.SPLITM, sub_block_id=0)
                tmp_mutex.ready()
                tmp_mutex.wait()
                cast_tmp_to_bf16_vf(tmp_f_ub[pipe_work], tmp_h_ub[pipe_work])
                tmp_mutex.free()

                tmp_bridge.lock()
                l1_tmp[pipe_work][row_begin_l:row_end_l, 0:L] <<= tmp_h_ub[pipe_work][0:HALF_L, 0:L]
                tmp_bridge.ready()

            if pipe_work >= work_begin + 2:
                prev_work = Var(pipe_work - 2)
                prev_c = Var(prev_work % chunk_count)
                prev_bhv = Var(prev_work / chunk_count)
                prev_hv = Var(prev_bhv % HV)
                prev_b = Var(prev_bhv / HV)
                prev_row0 = Var(prev_c * L)
                prev_row1 = Var(prev_row0 + L)

                tmp_bridge.wait()
                matmul(l0c_out[prev_work], l1_Akk[prev_work].T, l1_tmp[prev_work].T, m=L, n=L, k=L, splitn=L)
                tmp_bridge.free()

                out_mutex.lock()
                l0c_to_ub(dakk_f_ub[prev_work], l0c_out[prev_work], M=L, N=L, N_dst=L, M_src=L, dual_mode=DualMode.SPLITM, sub_block_id=0)
                out_mutex.ready()
                out_mutex.wait()
                neg_tril_vf(dakk_f_ub[prev_work], dakk_h_ub[prev_work], row_begin_l, Var(HALF_L))
                out_mutex.free()
                # dAkk written strided to the frozen BTHVL seam (-> finalize_pre):
                # SPLITM half-row store, explicit (HV-1)*L token gap.
                ub_to_gm_pad(dAkk[prev_b, prev_row0 + row_begin_l:prev_row0 + row_end_l, prev_hv, 0:L], dakk_h_ub[prev_work][0:HALF_L, 0:L], HALF_L, L, 0, (HV - 1) * L)

    return dAkk

# ----------------------------------------------------------------------------------------------------
# finalize_pre.py
# KDA kernel port; the launch ABI for this stage is main.py's `execute`.
#
# The arithmetic and cube-group partition come from the reviewed source.
# Whole-chunk vector work is owned only by subblock zero to eliminate duplicate
# GM writes; that is a deliberate correction to the source, not an omission.
# ----------------------------------------------------------------------------------------------------

@vf()
def finalize_pre_vf(
    q_ub: Tensor,
    k_ub: Tensor,
    g_ub: Tensor,
    beta_ub: Tensor,
    dAqk_ub: Tensor,
    dAkk_ub: Tensor,
    q_scaled_ub: Tensor,
    k_scaled_ub: Tensor,
    kg_ub: Tensor,
    mqk_ub: Tensor,
    mbase_ub: Tensor,
    mbeta_ub: Tensor,
    rows: Var,
):
    glast = RegList(DT.float, REGS_D)
    g = RegList(DT.float, REGS_D)
    rscale = RegList(DT.float, REGS_D)
    cscale = RegList(DT.float, REGS_D)
    qr = RegList(DT.float, REGS_D)
    kr = RegList(DT.float, REGS_D)
    prod = RegList(DT.float, REGS_D)
    tmp = RegList(DT.float, REGS_D)
    cols = Reg(DT.int)
    daqk = Reg(DT.float)
    dakk = Reg(DT.float)
    base = Reg(DT.float)
    out = Reg(DT.float)
    zero = Reg(DT.float)
    beta_r = Reg(DT.float)
    mask_le = MaskReg(DT.int, init_mode=MaskType.NONE)
    mask_lt = MaskReg(DT.int, init_mode=MaskType.NONE)

    glast <<= g_ub[L - 1:L, 0:D]
    zero <<= 0.0

    for r in range(rows):
        g <<= g_ub[r:r + 1, 0:D]
        tmp <<= g - glast
        tmp <<= tmp * LN2
        rscale <<= tmp.exp()
        tmp <<= glast - g
        tmp <<= tmp * LN2
        cscale <<= tmp.exp()

        qr <<= q_ub[r:r + 1, 0:D]
        prod <<= qr * rscale
        q_scaled_ub[r * D] <<= prod[0]
        q_scaled_ub[r * D + 64] <<= prod[1]

        kr <<= k_ub[r:r + 1, 0:D]
        prod <<= kr * rscale
        k_scaled_ub[r * D] <<= prod[0]
        k_scaled_ub[r * D + 64] <<= prod[1]

        prod <<= kr * cscale
        kg_ub[r * D] <<= prod[0]
        kg_ub[r * D + 64] <<= prod[1]

        cols.arange(0)
        daqk <<= dAqk_ub[r:r + 1, 0:L]
        compare(mask_le, cols, Var(r + 1), CompareMode.LT)  # j < r+1  == j <= r
        select(out, daqk, zero, mask=mask_le)
        mqk_ub[r:r + 1, 0:L] <<= out

        dakk <<= dAkk_ub[r:r + 1, 0:L]
        compare(mask_lt, cols, r, CompareMode.LT)           # j < r
        select(base, dakk, zero, mask=mask_lt)
        mbase_ub[r:r + 1, 0:L] <<= base
        beta_r <<= beta_ub[r:r + 1, 0:1].single()
        base <<= base * beta_r
        mbeta_ub[r:r + 1, 0:L] <<= base
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

@kernel()
def finalize_pre_kernel(q: GM[bf16, ('B', 'T', 'H', 128)], k: GM[bf16, ('B', 'T', 'H', 128)], g_cumsum: GM[bf16, ('B', 'T', 'HV', 128)], beta: GM[bf16, ('B', 'T', 'HV')], dAqk: GM[bf16, ('B', 'T', 'HV', 64)], dAkk: GM[bf16, ('B', 'T', 'HV', 64)], q_scaled: GM[bf16, ('B', 'HV', 'C', 64, 128)], k_scaled: GM[bf16, ('B', 'HV', 'C', 64, 128)], kg: GM[bf16, ('B', 'HV', 'C', 64, 128)], m_qk: GM[bf16, ('B', 'HV', 'C', 64, 64)], m_base: GM[bf16, ('B', 'HV', 'C', 64, 64)], m_beta: GM[bf16, ('B', 'HV', 'C', 64, 64)], B: i32, HV: i32, C: i32, H: i32, G: i32):
    q_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    k_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    g_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    beta_ub = Tensor(DT.bfloat16, [L, SCALAR_PACK_BF16], Position.UB)
    dAqk_ub = Tensor(DT.bfloat16, [L, L], Position.UB)
    dAkk_ub = Tensor(DT.bfloat16, [L, L], Position.UB)

    q_scaled_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    k_scaled_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    kg_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    mqk_ub = Tensor(DT.bfloat16, [L, L], Position.UB)
    mbase_ub = Tensor(DT.bfloat16, [L, L], Position.UB)
    mbeta_ub = Tensor(DT.bfloat16, [L, L], Position.UB)

    work_count = Var(B * HV * C)
    work_per_core = CeilDiv(work_count, GetCubeNum())
    work_begin = Var(work_per_core * GetCubeIdx())
    work_end = Min(work_begin + work_per_core, work_count)

    with auto_sync():
        # One vector subblock owns the whole chunk; the peer must not duplicate GM writes.
        if GetSubBlockIdx() == 0:
            for work in range(work_begin, work_end):
                c_idx = Var(work % C)
                bhv = Var(work / C)
                hv_idx = Var(bhv % HV)
                b_idx = Var(bhv / HV)

                # Strided on-device reads from token-major BTHVD primaries/caches; the
                # H->HV gather (h_idx = hv_idx / G) happens here. q/k carry H heads
                # (pitch H*D); g/beta/dAqk/dAkk carry HV (pitch HV*D / HV / HV*L).
                row0 = Var(c_idx * L)
                h_idx = Var(hv_idx / G)
                gm_to_ub_pad(q_ub[0:L, 0:D], q[b_idx, row0:row0 + L, h_idx, 0:D], L, D, (H - 1) * D, 0)
                gm_to_ub_pad(k_ub[0:L, 0:D], k[b_idx, row0:row0 + L, h_idx, 0:D], L, D, (H - 1) * D, 0)
                gm_to_ub_pad(g_ub[0:L, 0:D], g_cumsum[b_idx, row0:row0 + L, hv_idx, 0:D], L, D, (HV - 1) * D, 0)
                gm_to_ub_pad(beta_ub[0:L, 0:1], beta[b_idx, row0:row0 + L, hv_idx], L, 1, HV - 1, 0)
                gm_to_ub_pad(dAqk_ub[0:L, 0:L], dAqk[b_idx, row0:row0 + L, hv_idx, 0:L], L, L, (HV - 1) * L, 0)
                gm_to_ub_pad(dAkk_ub[0:L, 0:L], dAkk[b_idx, row0:row0 + L, hv_idx, 0:L], L, L, (HV - 1) * L, 0)

                finalize_pre_vf(
                    q_ub, k_ub, g_ub, beta_ub, dAqk_ub, dAkk_ub,
                    q_scaled_ub, k_scaled_ub, kg_ub,
                    mqk_ub, mbase_ub, mbeta_ub, Var(L),
                )

                q_scaled[b_idx, hv_idx, c_idx, 0:L, 0:D] <<= q_scaled_ub[0:L, 0:D]
                k_scaled[b_idx, hv_idx, c_idx, 0:L, 0:D] <<= k_scaled_ub[0:L, 0:D]
                kg[b_idx, hv_idx, c_idx, 0:L, 0:D] <<= kg_ub[0:L, 0:D]
                m_qk[b_idx, hv_idx, c_idx, 0:L, 0:L] <<= mqk_ub[0:L, 0:L]
                m_base[b_idx, hv_idx, c_idx, 0:L, 0:L] <<= mbase_ub[0:L, 0:L]
                m_beta[b_idx, hv_idx, c_idx, 0:L, 0:L] <<= mbeta_ub[0:L, 0:L]

    return q_scaled, k_scaled, kg, m_qk, m_base, m_beta

# ----------------------------------------------------------------------------------------------------
# finalize_pair.py
# KDA kernel port; the launch ABI for this stage is main.py's `execute`.
#
# The arithmetic and schedule come from the reviewed source; only imports and
# typed kernel signatures are migrated here.
# ----------------------------------------------------------------------------------------------------

@kernel()
def finalize_pair_kernel(Mqk_bf16: GM[bf16, ('B', 'HV', 'T', 64)], Mbase_bf16: GM[bf16, ('B', 'HV', 'T', 64)], Mbeta_bf16: GM[bf16, ('B', 'HV', 'T', 64)], q_bf16: GM[bf16, ('B', 'HV', 'T', 128)], k_bf16: GM[bf16, ('B', 'HV', 'T', 128)], kg_bf16: GM[bf16, ('B', 'HV', 'T', 128)], dq_pair: GM[bf16, ('B', 'HV', 'T', 128)], dk_pair: GM[bf16, ('B', 'HV', 'T', 128)], s_base: GM[bf16, ('B', 'HV', 'T', 128)], t_beta: GM[bf16, ('B', 'HV', 'T', 128)], B: i32, HV: i32, T: i32):
    l1_Mqk = DBuff(DT.bfloat16, [L, L], Position.L1)
    l1_Mbase = DBuff(DT.bfloat16, [L, L], Position.L1)
    l1_Mbeta = DBuff(DT.bfloat16, [L, L], Position.L1)
    l1_q = DBuff(DT.bfloat16, [L, D], Position.L1)
    l1_k = DBuff(DT.bfloat16, [L, D], Position.L1)
    l1_kg = DBuff(DT.bfloat16, [L, D], Position.L1)
    l0c_dq = DBuff(DT.float, [L, D], Position.L0C)
    l0c_dk = DBuff(DT.float, [L, D], Position.L0C)
    l0c_s = DBuff(DT.float, [L, D], Position.L0C)
    l0c_t = DBuff(DT.float, [L, D], Position.L0C)

    chunk_count = Var(T / L)
    work_count = Var(B * HV * chunk_count)
    work_per_core = CeilDiv(work_count, GetCubeNum())
    work_begin = Var(work_per_core * GetCubeIdx())
    work_end = Min(work_begin + work_per_core, work_count)
    slot = Var(0)

    with auto_sync():
        for work_idx in range(work_begin, work_end):
            c_idx = Var(work_idx % chunk_count)
            bhv_idx = Var(work_idx / chunk_count)
            hv_idx = Var(bhv_idx % HV)
            b_idx = Var(bhv_idx / HV)
            row0 = Var(c_idx * L)
            row1 = Var(row0 + L)

            l1_Mqk[slot][0:L, 0:L] <<= Mqk_bf16[b_idx, hv_idx, row0:row1, 0:L]
            l1_q[slot][0:L, 0:D] <<= q_bf16[b_idx, hv_idx, row0:row1, 0:D]
            l1_kg[slot][0:L, 0:D] <<= kg_bf16[b_idx, hv_idx, row0:row1, 0:D]
            matmul(l0c_dq[slot], l1_Mqk[slot], l1_kg[slot].T, m=L, n=D, k=L, splitn=D)
            matmul(l0c_dk[slot], l1_Mqk[slot].T, l1_q[slot].T, m=L, n=D, k=L, splitn=D)
            dq_pair[b_idx, hv_idx, row0:row1, 0:D] <<= l0c_dq[slot]
            dk_pair[b_idx, hv_idx, row0:row1, 0:D] <<= l0c_dk[slot]

            l1_Mbase[slot][0:L, 0:L] <<= Mbase_bf16[b_idx, hv_idx, row0:row1, 0:L]
            l1_Mbeta[slot][0:L, 0:L] <<= Mbeta_bf16[b_idx, hv_idx, row0:row1, 0:L]
            l1_k[slot][0:L, 0:D] <<= k_bf16[b_idx, hv_idx, row0:row1, 0:D]
            matmul(l0c_s[slot], l1_Mbase[slot], l1_kg[slot].T, m=L, n=D, k=L, splitn=D)
            matmul(l0c_t[slot], l1_Mbeta[slot].T, l1_k[slot].T, m=L, n=D, k=L, splitn=D)
            s_base[b_idx, hv_idx, row0:row1, 0:D] <<= l0c_s[slot]
            t_beta[b_idx, hv_idx, row0:row1, 0:D] <<= l0c_t[slot]
            slot += 1

    return dq_pair, dk_pair, s_base, t_beta

# ----------------------------------------------------------------------------------------------------
# finalize_post.py
# KDA kernel port; the launch ABI for this stage is main.py's `execute`.
#
# The arithmetic and cube-group partition come from the reviewed source.
# Whole-chunk vector work is owned only by subblock zero to eliminate duplicate
# GM writes; that is a deliberate correction to the source, not an omission.
# ----------------------------------------------------------------------------------------------------

@vf()
def finalize_post_vf(
    g_ub: Tensor,
    qkl_ub: Tensor,
    qkr_ub: Tensor,
    sbase_ub: Tensor,
    tbeta_ub: Tensor,
    qhv_ub: Tensor,
    khv_ub: Tensor,
    beta_ub: Tensor,
    dqhv_in_ub: Tensor,
    dkhv_in_ub: Tensor,
    dbeta_in_ub: Tensor,
    dgcore_in_ub: Tensor,
    dqhv_out_ub: Tensor,
    dkhv_out_ub: Tensor,
    dbeta_out_ub: Tensor,
    dg_out_ub: Tensor,
    rows: Var,
):
    glast = RegList(DT.float, REGS_D)
    g = RegList(DT.float, REGS_D)
    rscale = RegList(DT.float, REGS_D)
    cscale = RegList(DT.float, REGS_D)
    qkl = RegList(DT.float, REGS_D)
    qkr = RegList(DT.float, REGS_D)
    sbase = RegList(DT.float, REGS_D)
    tbeta = RegList(DT.float, REGS_D)
    dqpair = RegList(DT.float, REGS_D)
    dkpair = RegList(DT.float, REGS_D)
    rowc = RegList(DT.float, REGS_D)
    colc = RegList(DT.float, REGS_D)
    qh = RegList(DT.float, REGS_D)
    kh = RegList(DT.float, REGS_D)
    dgqk = RegList(DT.float, REGS_D)
    dkkk = RegList(DT.float, REGS_D)
    dgkk = RegList(DT.float, REGS_D)
    dqhv = RegList(DT.float, REGS_D)
    dkhv = RegList(DT.float, REGS_D)
    dgp = RegList(DT.float, REGS_D)
    running = RegList(DT.float, REGS_D)
    tmp = RegList(DT.float, REGS_D)
    prod = RegList(DT.float, REGS_D)
    beta_r = Reg(DT.float)
    dbadd = Reg(DT.float)
    dbin = Reg(DT.float)
    dbout = Reg(DT.float)
    dbout_bf16 = Reg(DT.bfloat16)

    glast <<= g_ub[L - 1:L, 0:D]
    running.fill(0.0)

    # Walk rows L-1 -> 0 so `running` is the per-chunk reverse cumsum of dg'.
    for rt in range(rows):
        r = rows - 1 - rt

        g <<= g_ub[r:r + 1, 0:D]
        tmp <<= g - glast
        tmp <<= tmp * LN2
        rscale <<= tmp.exp()
        tmp <<= glast - g
        tmp <<= tmp * LN2
        cscale <<= tmp.exp()

        qkl <<= qkl_ub[r:r + 1, 0:D]
        dqpair <<= rscale * qkl
        qkr <<= qkr_ub[r:r + 1, 0:D]
        dkpair <<= cscale * qkr
        sbase <<= sbase_ub[r:r + 1, 0:D]
        rowc <<= rscale * sbase
        tbeta <<= tbeta_ub[r:r + 1, 0:D]
        colc <<= cscale * tbeta

        qh <<= qhv_ub[r:r + 1, 0:D]
        kh <<= khv_ub[r:r + 1, 0:D]
        beta_r <<= beta_ub[r:r + 1, 0:1].single()

        # dg_qk equals q_hv*dq_pair - k_hv*dk_pair.
        dgqk <<= qh * dqpair
        prod <<= kh * dkpair
        dgqk <<= dgqk - prod

        # dk_kk equals beta*row_contrib + col_contrib.
        dkkk <<= rowc * beta_r
        dkkk <<= dkkk + colc

        # dbeta_add equals (k_hv * row_contrib).sum(D).
        prod <<= kh * rowc
        dbadd <<= prod.cadd()

        # dg_kk equals beta*k_hv*row_contrib - k_hv*col_contrib.
        dgkk <<= kh * rowc
        dgkk <<= dgkk * beta_r
        prod <<= kh * colc
        dgkk <<= dgkk - prod

        # dq_hv' = dq_hv + dq_pair   (fp32 out)
        dqhv <<= dqhv_in_ub[r:r + 1, 0:D]
        dqhv <<= dqhv + dqpair
        dqhv_out_ub[r:r + 1, 0:D] <<= dqhv

        # dk_hv' = dk_hv + dk_pair + dk_kk   (fp32 out)
        dkhv <<= dkhv_in_ub[r:r + 1, 0:D]
        dkhv <<= dkhv + dkpair
        dkhv <<= dkhv + dkkk
        dkhv_out_ub[r:r + 1, 0:D] <<= dkhv

        # dbeta' = dbeta + dbeta_add ; cast fp32->bf16 for the public bf16 output
        dbin <<= dbeta_in_ub[r:r + 1, 0:1].single()
        dbout <<= dbin + dbadd
        dbout_bf16 <<= dbout.cast()       # fp32 -> bf16 (round-to-nearest-even)
        dbeta_out_ub[r:r + 1, 0:1] <<= dbout_bf16.single_value()

        # dg' = dg_core + dg_qk + dg_kk ; running += dg' ; dg[r] = running
        dgp <<= dgcore_in_ub[r:r + 1, 0:D]
        dgp <<= dgp + dgqk
        dgp <<= dgp + dgkk
        running <<= running + dgp
        dg_out_ub[r * D] <<= running[0]
        dg_out_ub[r * D + 64] <<= running[1]
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

@kernel()
def finalize_post_kernel(g_cumsum: GM[bf16, ('B', 'T', 'HV', 128)], qk_left: GM[bf16, ('B', 'HV', 'C', 64, 128)], qk_right: GM[bf16, ('B', 'HV', 'C', 64, 128)], s_base: GM[bf16, ('B', 'HV', 'C', 64, 128)], t_beta: GM[bf16, ('B', 'HV', 'C', 64, 128)], q: GM[bf16, ('B', 'T', 'H', 128)], k: GM[bf16, ('B', 'T', 'H', 128)], beta: GM[bf16, ('B', 'T', 'HV')], dq_hv_in: GM[bf16, ('B', 'T', 'HV', 128)], dk_hv_in: GM[bf16, ('B', 'T', 'HV', 128)], dbeta_in: GM[f32, ('B', 'T', 'HV')], dg_core_in: GM[bf16, ('B', 'T', 'HV', 128)], dq_hv_out: GM[f32, ('B', 'HV', 'C', 64, 128)], dk_hv_out: GM[f32, ('B', 'HV', 'C', 64, 128)], dbeta_out: GM[bf16, ('B', 'T', 'HV')], dg_out: GM[bf16, ('B', 'T', 'HV', 128)], B: i32, HV: i32, C: i32, H: i32, G: i32):
    g_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    qkl_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    qkr_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    sbase_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    tbeta_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    qhv_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    khv_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    beta_ub = Tensor(DT.bfloat16, [L, SCALAR_PACK_BF16], Position.UB)
    dqhv_in_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    dkhv_in_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    dbeta_in_ub = Tensor(DT.float, [L, SCALAR_PACK_F32], Position.UB)  # fp32 per-token scalar, 32B-row pack
    dgcore_in_ub = Tensor(DT.bfloat16, [L, D], Position.UB)

    dqhv_out_ub = Tensor(DT.float, [L, D], Position.UB)
    dkhv_out_ub = Tensor(DT.float, [L, D], Position.UB)
    dbeta_out_ub = Tensor(DT.bfloat16, [L, SCALAR_PACK_BF16], Position.UB)  # bf16 per-token scalar, 32B-row pack
    dg_out_ub = Tensor(DT.bfloat16, [L, D], Position.UB)

    work_count = Var(B * C)
    work_per_core = CeilDiv(work_count, GetCubeNum())
    work_begin = Var(work_per_core * GetCubeIdx())
    work_end = Min(work_begin + work_per_core, work_count)

    with auto_sync():
        # One vector subblock owns the whole chunk; the peer must not duplicate GM writes.
        if GetSubBlockIdx() == 0:
            for work in range(work_begin, work_end):
                c_idx = Var(work % C)
                b_idx = Var(work / C)
                # All heads share scalar GM transaction blocks and stay on this core.
                for hv_idx in range(HV):

                    row0 = Var(c_idx * L)
                    h_idx = Var(hv_idx / G)
                    # g_cumsum: cache, strided (pitch HV*D)
                    gm_to_ub_pad(g_ub[0:L, 0:D], g_cumsum[b_idx, row0:row0 + L, hv_idx, 0:D], L, D, (HV - 1) * D, 0)
                    # qk_*: intra-stage from finalize_pair, contiguous GM-native (keep <<=)
                    qkl_ub[0:L, 0:D] <<= qk_left[b_idx, hv_idx, c_idx, 0:L, 0:D]
                    qkr_ub[0:L, 0:D] <<= qk_right[b_idx, hv_idx, c_idx, 0:L, 0:D]
                    sbase_ub[0:L, 0:D] <<= s_base[b_idx, hv_idx, c_idx, 0:L, 0:D]
                    tbeta_ub[0:L, 0:D] <<= t_beta[b_idx, hv_idx, c_idx, 0:L, 0:D]
                    # q/k: primaries, strided (pitch H*D) with on-device H->HV gather
                    gm_to_ub_pad(qhv_ub[0:L, 0:D], q[b_idx, row0:row0 + L, h_idx, 0:D], L, D, (H - 1) * D, 0)
                    gm_to_ub_pad(khv_ub[0:L, 0:D], k[b_idx, row0:row0 + L, h_idx, 0:D], L, D, (H - 1) * D, 0)
                    # beta: primary, strided (pitch HV, burst_len=1)
                    gm_to_ub_pad(beta_ub[0:L, 0:1], beta[b_idx, row0:row0 + L, hv_idx], L, 1, HV - 1, 0)
                    # inverse-stage outputs, strided BTHVD/BTHV (dbeta_in is fp32 -> fp32 UB)
                    gm_to_ub_pad(dqhv_in_ub[0:L, 0:D], dq_hv_in[b_idx, row0:row0 + L, hv_idx, 0:D], L, D, (HV - 1) * D, 0)
                    gm_to_ub_pad(dkhv_in_ub[0:L, 0:D], dk_hv_in[b_idx, row0:row0 + L, hv_idx, 0:D], L, D, (HV - 1) * D, 0)
                    gm_to_ub_pad(dbeta_in_ub[0:L, 0:1], dbeta_in[b_idx, row0:row0 + L, hv_idx], L, 1, HV - 1, 0)
                    gm_to_ub_pad(dgcore_in_ub[0:L, 0:D], dg_core_in[b_idx, row0:row0 + L, hv_idx, 0:D], L, D, (HV - 1) * D, 0)

                    finalize_post_vf(
                        g_ub, qkl_ub, qkr_ub, sbase_ub, tbeta_ub, qhv_ub, khv_ub, beta_ub,
                        dqhv_in_ub, dkhv_in_ub, dbeta_in_ub, dgcore_in_ub,
                        dqhv_out_ub, dkhv_out_ub, dbeta_out_ub, dg_out_ub, Var(L),
                    )

                    # dq_hv'/dk_hv': intra-stage to finalize_reduce, contiguous GM-native fp32 (keep)
                    dq_hv_out[b_idx, hv_idx, c_idx, 0:L, 0:D] <<= dqhv_out_ub[0:L, 0:D]
                    dk_hv_out[b_idx, hv_idx, c_idx, 0:L, 0:D] <<= dkhv_out_ub[0:L, 0:D]
                    # dbeta/dg: public outputs, strided into token-major BTHV / BTHVD bf16
                    ub_to_gm_pad(dbeta_out[b_idx, row0:row0 + L, hv_idx], dbeta_out_ub[0:L, 0:1], L, 1, 0, HV - 1)
                    ub_to_gm_pad(dg_out[b_idx, row0:row0 + L, hv_idx, 0:D], dg_out_ub[0:L, 0:D], L, D, 0, (HV - 1) * D)

    return dq_hv_out, dk_hv_out, dbeta_out, dg_out

# ----------------------------------------------------------------------------------------------------
# finalize_reduce.py
# KDA kernel port; the launch ABI for this stage is main.py's `execute`.
#
# The arithmetic and cube-group partition come from the reviewed source.
# Whole-chunk vector work is owned only by subblock zero to eliminate duplicate
# GM writes; that is a deliberate correction to the source, not an omission.
# ----------------------------------------------------------------------------------------------------

@vf()
def reduce_add_vf(acc_ub: Tensor, add_ub: Tensor, rows: Var):
    a = RegList(DT.float, REGS_D)
    b = RegList(DT.float, REGS_D)
    for r in range(rows):
        a <<= acc_ub[r:r + 1, 0:D]
        b <<= add_ub[r:r + 1, 0:D]
        a <<= a + b
        acc_ub[r:r + 1, 0:D] <<= a
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

@vf()
def reduce_cast_vf(acc_ub: Tensor, out_ub: Tensor, rows: Var):
    a = RegList(DT.float, REGS_D)
    for r in range(rows):
        a <<= acc_ub[r:r + 1, 0:D]
        out_ub[r * D] <<= a[0]
        out_ub[r * D + 64] <<= a[1]
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

@kernel()
def finalize_reduce_kernel(dq_hv: GM[f32, ('B', 'HV', 'C', 64, 128)], dk_hv: GM[f32, ('B', 'HV', 'C', 64, 128)], dq_out: GM[bf16, ('B', 'T', 'H', 128)], dk_out: GM[bf16, ('B', 'T', 'H', 128)], B: i32, HV: i32, H: i32, C: i32, G: i32):
    dqacc_ub = Tensor(DT.float, [L, D], Position.UB)
    dkacc_ub = Tensor(DT.float, [L, D], Position.UB)
    dq_g_ub = Tensor(DT.float, [L, D], Position.UB)
    dk_g_ub = Tensor(DT.float, [L, D], Position.UB)
    dq_out_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    dk_out_ub = Tensor(DT.bfloat16, [L, D], Position.UB)

    work_count = Var(B * H * C)
    work_per_core = CeilDiv(work_count, GetCubeNum())
    work_begin = Var(work_per_core * GetCubeIdx())
    work_end = Min(work_begin + work_per_core, work_count)

    with auto_sync():
        # One vector subblock owns the whole chunk; the peer must not duplicate GM writes.
        if GetSubBlockIdx() == 0:
            for work in range(work_begin, work_end):
                c_idx = Var(work % C)
                bh = Var(work / C)
                h_idx = Var(bh % H)
                b_idx = Var(bh / H)
                hv0 = Var(h_idx * G)

                # g = 0 initialises the fp32 accumulator (no cast on the fp32 load).
                dqacc_ub[0:L, 0:D] <<= dq_hv[b_idx, hv0, c_idx, 0:L, 0:D]
                dkacc_ub[0:L, 0:D] <<= dk_hv[b_idx, hv0, c_idx, 0:L, 0:D]
                for g in range(1, G):
                    hv = Var(hv0 + g)
                    dq_g_ub[0:L, 0:D] <<= dq_hv[b_idx, hv, c_idx, 0:L, 0:D]
                    dk_g_ub[0:L, 0:D] <<= dk_hv[b_idx, hv, c_idx, 0:L, 0:D]
                    reduce_add_vf(dqacc_ub, dq_g_ub, Var(L))
                    reduce_add_vf(dkacc_ub, dk_g_ub, Var(L))
                reduce_cast_vf(dqacc_ub, dq_out_ub, Var(L))
                reduce_cast_vf(dkacc_ub, dk_out_ub, Var(L))
                # public dq/dk written strided into token-major BTHVD [B, T, H, D]
                # on-device (row pitch H*D); no host transpose.
                row0 = Var(c_idx * L)
                ub_to_gm_pad(dq_out[b_idx, row0:row0 + L, h_idx, 0:D], dq_out_ub[0:L, 0:D], L, D, 0, (H - 1) * D)
                ub_to_gm_pad(dk_out[b_idx, row0:row0 + L, h_idx, 0:D], dk_out_ub[0:L, 0:D], L, D, 0, (H - 1) * D)

    return dq_out, dk_out
