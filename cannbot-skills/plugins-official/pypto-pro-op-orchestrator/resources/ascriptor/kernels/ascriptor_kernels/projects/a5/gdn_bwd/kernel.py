# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Five DSL stages of the chunked gated delta-net (GDN) backward."""

from ascriptor.a5 import *  # noqa: F403 - public DSL facade  # noqa: F401, F403

# ----------------------------------------------------------------------------------------------------
# scan_local.py
# gdn_bwd production stage migrated from kernels/scan_local_bwd.py.
#
# M10 edits: preserve the selected kernel and its source dependency closure;
# replace the EasyASC facade, synthesize typed GM/scalar signatures from the
# reviewed stage contract, and preserve explicit Python unrolling with unroll.
# Host drivers live in main.py's `execute`; the independent reference is reference.py.
# The attention staging tile uses its compact [32,64] NZ footprint, and its
# strided stores mask the unused upper half of the BF16 register.
# Source SHA256: 5db5e115d74cdf9f8cfaa90bef1eff6242bbb8ac30b6ceac735ad97beaf5db9e
# ----------------------------------------------------------------------------------------------------

L = 64
D = 128


HALF_L = L // 2
BF16_C0 = 16




@vf()
def cast_l_rows_float_to_bf16_nz_vf(src_ub: Tensor, dst_nz_ub: Tensor, rows: Var):
    row_float = Reg(DT.float)
    lo_bf16 = Reg(DT.bfloat16)
    row_bf16 = Reg(DT.bfloat16)
    dummy_bf16 = Reg(DT.bfloat16)
    valid_lanes = MaskReg(DT.bfloat16, init_mode=MaskType.LOWHALF)

    for r in range(rows):
        row_float <<= src_ub[r:r + 1, 0:L]
        lo_bf16 <<= row_float.astype(DT.bfloat16)
        deinterleave(row_bf16, dummy_bf16, lo_bf16, dummy_bf16)
        reg_to_ub(dst_nz_ub[r * BF16_C0], row_bf16, rows, mask=valid_lanes)












@vf()
def mask_attn_decay_vf(
    score_to_attn_ub: Tensor,
    d_attn_to_d_score_ub: Tensor,
    decay_to_d_decay_ub: Tensor,
    attn_out_ub: Tensor,
    d_score_bf16_out_ub: Tensor,
    d_decay_out_ub: Tensor,
    d_decay_masked_out_ub: Tensor,
    rows: Var,
):
    score_regs = Reg(DT.float)
    d_attn_regs = Reg(DT.float)
    decay_regs = Reg(DT.float)
    tmp_regs = Reg(DT.float)
    score_bf16 = Reg(DT.bfloat16)
    dummy_bf16 = Reg(DT.bfloat16)
    score_row = Reg(DT.bfloat16)
    bf16_lowhalf = MaskReg(DT.bfloat16, init_mode=MaskType.LOWHALF)
    for r in range(rows):
        score_regs <<= score_to_attn_ub[r:r + 1, 0:L]
        d_attn_regs <<= d_attn_to_d_score_ub[r:r + 1, 0:L]
        decay_regs <<= decay_to_d_decay_ub[r:r + 1, 0:L]

        tmp_regs <<= score_regs * decay_regs
        attn_out_ub[r:r + 1, 0:L] <<= tmp_regs

        tmp_regs <<= d_attn_regs * decay_regs
        score_bf16 <<= tmp_regs.astype(DT.bfloat16)
        deinterleave(score_row, dummy_bf16, score_bf16, dummy_bf16)
        reg_to_ub(d_score_bf16_out_ub[r:r + 1, 0:L], score_row, mask=bf16_lowhalf)

        tmp_regs <<= d_attn_regs * score_regs
        d_decay_out_ub[r:r + 1, 0:L] <<= tmp_regs
        tmp_regs <<= tmp_regs * decay_regs
        d_decay_masked_out_ub[r:r + 1, 0:L] <<= tmp_regs
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)




























@kernel()
def scan_local_bwd_kernel(
    query: GM[bf16, ('B', 'H', 'C', 64, 128)],
    key: GM[bf16, ('B', 'H', 'C', 64, 128)],
    grad_output: GM[bf16, ('B', 'H', 'C', 64, 128)],
    decay_mask: GM[f32, ('B', 'H', 'C', 64, 64)],
    v_new_history: GM[bf16, ('B', 'H', 'C', 64, 128)],
    d_score_tmp: GM[bf16, ('B', 'H', 'C', 64, 64)],
    d_v_attn_tmp: GM[f32, ('B', 'H', 'C', 64, 128)],
    d_decay_core: GM[f32, ('B', 'H', 'C', 64, 64)],
    d_decay_core_masked: GM[f32, ('B', 'H', 'C', 64, 64)],
    B: i32,
    H: i32,
    C: i32,
):
    cvmutex = CvMutex(0, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    attn_mutex = VcMutex(1, depth=2, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE1)

    l1_q = DBuff(DT.bfloat16, [L, D], Position.L1)
    l1_k = DBuff(DT.bfloat16, [L, D], Position.L1)
    l1_d_out = TBuff(DT.bfloat16, [L, D], Position.L1)
    l1_v_new = DBuff(DT.bfloat16, [L, D], Position.L1)
    l1_attn = DBuff(DT.bfloat16, [L, L], Position.L1)

    l0c_ll0 = DBuff(DT.float, [L, L], Position.L0C)
    l0c_ll1 = DBuff(DT.float, [L, L], Position.L0C)
    l0c_dv = DBuff(DT.float, [L, D], Position.L0C)

    bf16_full_ub = DBuff(DT.bfloat16, [HALF_L, L], Position.UB)
    float_half0_ub = DBuff(DT.float, [HALF_L, D], Position.UB)
    d_score_bf16_ub = DBuff(DT.bfloat16, [HALF_L, L], Position.UB)
    ll0_ub = DBuff(DT.float, [HALF_L, L], Position.UB)
    ll1_ub = DBuff(DT.float, [HALF_L, L], Position.UB)
    ll2_ub = DBuff(DT.float, [HALF_L, L], Position.UB)
    d_decay_out_ub = DBuff(DT.float, [HALF_L, L], Position.UB)
    d_decay_masked_out_ub = DBuff(DT.float, [HALF_L, L], Position.UB)

    bhc_count = B * H * C
    bhc_per_core = CeilDiv(bhc_count, GetCubeNum())
    bhc_begin = Var(bhc_per_core * GetCubeIdx())
    bhc_end = Min(bhc_begin + bhc_per_core, bhc_count)

    for pipe_bhc in range(bhc_begin, bhc_end + 1):
        if pipe_bhc < bhc_end:
            with auto_sync():
                pipe_slot = var_mod(pipe_bhc - bhc_begin, 2)
                dout_slot = var_mod(pipe_bhc - bhc_begin, 3)
                c_idx = Var(pipe_bhc % C)
                bh = Var(pipe_bhc // C)
                b_idx = Var(bh // H)
                h_idx = Var(bh % H)
                row_begin_l = Var(GetSubBlockIdx() * HALF_L)
                row_end_l = Var(row_begin_l + HALF_L)
                rows_l = Var(HALF_L)

                l1_q[pipe_slot][0:L, 0:D] <<= query[b_idx, h_idx, c_idx, 0:L, 0:D]
                l1_k[pipe_slot][0:L, 0:D] <<= key[b_idx, h_idx, c_idx, 0:L, 0:D]

                l1_d_out[dout_slot][0:L, 0:D] <<= grad_output[b_idx, h_idx, c_idx, 0:L, 0:D]
                l1_v_new[pipe_slot][0:L, 0:D] <<= v_new_history[b_idx, h_idx, c_idx, 0:L, 0:D]

                matmul(l0c_ll0[pipe_slot], l1_q[pipe_slot], l1_k[pipe_slot], m=L, n=L, k=D, splitn=L)
                matmul(l0c_ll1[pipe_slot], l1_d_out[dout_slot], l1_v_new[pipe_slot], m=L, n=L, k=D, splitn=L)

                cvmutex.lock()
                ll0_ub[pipe_slot] <<= l0c_ll0[pipe_slot]
                ll1_ub[pipe_slot] <<= l0c_ll1[pipe_slot]
                cvmutex.ready()
                ll2_ub[pipe_slot][0:rows_l, 0:L] <<= decay_mask[b_idx, h_idx, c_idx, row_begin_l:row_end_l, 0:L]
                cvmutex.wait()
                mask_attn_decay_vf(
                    ll0_ub[pipe_slot],
                    ll1_ub[pipe_slot],
                    ll2_ub[pipe_slot],
                    float_half0_ub[pipe_slot],
                    d_score_bf16_ub[pipe_slot],
                    d_decay_out_ub[pipe_slot],
                    d_decay_masked_out_ub[pipe_slot],
                    rows_l,
                )
                cvmutex.free()

                attn_mutex.lock()
                cast_l_rows_float_to_bf16_nz_vf(float_half0_ub[pipe_slot], bf16_full_ub[pipe_slot], rows_l)
                l1_attn[pipe_slot][row_begin_l:row_end_l, 0:L] <<= bf16_full_ub[pipe_slot][0:rows_l, 0:L].nz()
                attn_mutex.ready()

                d_score_tmp[b_idx, h_idx, c_idx, row_begin_l:row_end_l, 0:L] <<= d_score_bf16_ub[pipe_slot][0:rows_l, 0:L]
                d_decay_core[b_idx, h_idx, c_idx, row_begin_l:row_end_l, 0:L] <<= d_decay_out_ub[pipe_slot][0:rows_l, 0:L]
                d_decay_core_masked[b_idx, h_idx, c_idx, row_begin_l:row_end_l, 0:L] <<= d_decay_masked_out_ub[pipe_slot][0:rows_l, 0:L]

        if pipe_bhc > bhc_begin:
            with auto_sync():
                bhc = Var(pipe_bhc - 1)
                bhc_slot = var_mod(bhc - bhc_begin, 2)
                dout_slot = var_mod(bhc - bhc_begin, 3)
                c_idx = Var(bhc % C)
                bh = Var(bhc // C)
                b_idx = Var(bh // H)
                h_idx = Var(bh % H)

                attn_mutex.wait()

                matmul(l0c_dv[bhc_slot], l1_attn[bhc_slot].T, l1_d_out[dout_slot].T, m=L, n=D, k=L, splitn=D)
                attn_mutex.free()

                d_v_attn_tmp[b_idx, h_idx, c_idx, 0:L, 0:D] <<= l0c_dv[bhc_slot]

    return d_score_tmp, d_v_attn_tmp, d_decay_core, d_decay_core_masked

# ----------------------------------------------------------------------------------------------------
# scan_state.py
# gdn_bwd production stage migrated from kernels/scan_state_bwd.py.
#
# M10 edits: preserve the selected kernel and its source dependency closure;
# replace the EasyASC facade, synthesize typed GM/scalar signatures from the
# reviewed stage contract, and preserve explicit Python unrolling with unroll.
# Host drivers live in main.py's `execute`; the independent reference is reference.py.
# Rebind the read-slot view at each reverse-iteration entry instead of mutating
# an outer tensor alias; the slot counter and numerical schedule are unchanged.
# Keep 32-row Q/dV NZ staging distinct from the 64-row state staging so the
# physical fractal pitch matches the producer's strided register stores.
# Use one explicit shared L0A/L0B operand pair with barriers around each cube
# step to stay within eight flags per pipe channel; retain M/N/K/init math.
# A5 local vector/DMA dependencies use per-buffer auto_sync mutexes.
# Explicit Cube/Vec and peer exchange handshakes remain unchanged.
# Source SHA256: 0747946e9814be93a50a72cab7769cd8114294fa1c998bd1a97f2d9558c3ef19
# ----------------------------------------------------------------------------------------------------

HALF_D = D // 2
REGS_D = D // 64


@vf()
def cast_d_rows_float_to_bf16_nz_vf(src_ub: Tensor, dst_nz_ub: Tensor, rows: Var):
    row_lo = Reg(DT.float)
    row_hi = Reg(DT.float)
    lo_bf16 = Reg(DT.bfloat16)
    hi_bf16 = Reg(DT.bfloat16)
    row_bf16 = Reg(DT.bfloat16)
    dummy_bf16 = Reg(DT.bfloat16)

    for r in range(rows):
        row_lo <<= src_ub[r:r + 1, 0:64]
        row_hi <<= src_ub[r:r + 1, 64:D]
        lo_bf16 <<= row_lo.astype(DT.bfloat16)
        hi_bf16 <<= row_hi.astype(DT.bfloat16)
        deinterleave(row_bf16, dummy_bf16, lo_bf16, hi_bf16)
        reg_to_ub(dst_nz_ub[r * BF16_C0], row_bf16, rows)


@vf()
def make_q_exp_nz_vf(q_bf16_ub: Tensor, g_ub: Tensor, q_exp_nz_ub: Tensor, exp_g_ub: Tensor, rows: Var):
    row_lo = Reg(DT.float)
    row_hi = Reg(DT.float)
    g_reg = Reg(DT.float)
    exp_g = Reg(DT.float)
    lo_bf16 = Reg(DT.bfloat16)
    hi_bf16 = Reg(DT.bfloat16)
    row_bf16 = Reg(DT.bfloat16)
    dummy_bf16 = Reg(DT.bfloat16)
    for r in range(rows):
        g_reg <<= g_ub[0:1, r:r + 1].single()
        exp_g <<= g_reg.exp()
        exp_g_ub[0:1, r:r + 1] <<= exp_g.single_value()
        row_lo <<= q_bf16_ub[r:r + 1, 0:64]
        row_lo <<= row_lo * exp_g
        row_hi <<= q_bf16_ub[r:r + 1, 64:D]
        row_hi <<= row_hi * exp_g
        lo_bf16 <<= row_lo.astype(DT.bfloat16)
        hi_bf16 <<= row_hi.astype(DT.bfloat16)
        deinterleave(row_bf16, dummy_bf16, lo_bf16, hi_bf16)
        reg_to_ub(q_exp_nz_ub[r * BF16_C0], row_bf16, rows)


@vf()
def add_scaled_d_rows_vf(score_ub: Tensor, exp_ub: Tensor, exp_g_ub: Tensor, out_ub: Tensor, rows: Var):
    score_regs = RegList(DT.float, REGS_D)
    exp_regs = RegList(DT.float, REGS_D)
    exp_g = Reg(DT.float)
    for r in range(rows):
        exp_g <<= exp_g_ub[0:1, r:r + 1].single()
        score_regs <<= score_ub[r:r + 1, 0:D]
        exp_regs <<= exp_ub[r:r + 1, 0:D]
        exp_regs <<= exp_regs * exp_g
        score_regs <<= score_regs + exp_regs
        out_ub[r:r + 1, 0:D] <<= score_regs
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def postprocess_dk_vf(d_k_score_ub: Tensor, d_k_weighted_ub: Tensor, exp_delta_ub: Tensor, out_ub: Tensor, rows: Var):
    score_regs = RegList(DT.float, REGS_D)
    weighted_regs = RegList(DT.float, REGS_D)
    exp_delta = Reg(DT.float)
    for r in range(rows):
        exp_delta <<= exp_delta_ub[0:1, r:r + 1].single()
        score_regs <<= d_k_score_ub[r:r + 1, 0:D]
        weighted_regs <<= d_k_weighted_ub[r:r + 1, 0:D]
        weighted_regs <<= weighted_regs * exp_delta
        score_regs <<= score_regs + weighted_regs
        out_ub[r:r + 1, 0:D] <<= score_regs
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)




@vf()
def add_cast_d_rows_bf16_vf(dep_ub: Tensor, base_ub: Tensor, out_bf16_ub: Tensor, rows: Var):
    dep_lo = Reg(DT.float)
    dep_hi = Reg(DT.float)
    base_lo = Reg(DT.float)
    base_hi = Reg(DT.float)
    lo_bf16 = Reg(DT.bfloat16)
    hi_bf16 = Reg(DT.bfloat16)
    row_bf16 = Reg(DT.bfloat16)
    dummy_bf16 = Reg(DT.bfloat16)
    for r in range(rows):
        dep_lo <<= dep_ub[r:r + 1, 0:64]
        dep_hi <<= dep_ub[r:r + 1, 64:D]
        base_lo <<= base_ub[r:r + 1, 0:64]
        base_hi <<= base_ub[r:r + 1, 64:D]
        base_lo <<= base_lo + dep_lo
        base_hi <<= base_hi + dep_hi
        lo_bf16 <<= base_lo.astype(DT.bfloat16)
        hi_bf16 <<= base_hi.astype(DT.bfloat16)
        deinterleave(row_bf16, dummy_bf16, lo_bf16, hi_bf16)
        reg_to_ub(out_bf16_ub[r:r + 1, 0:D], row_bf16)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def add_negate_cast_d_rows_bf16_vf(dep_ub: Tensor, base_ub: Tensor, sum_bf16_ub: Tensor, neg_ub: Tensor, neg_nz_ub: Tensor, rows: Var):
    dep_lo = Reg(DT.float)
    dep_hi = Reg(DT.float)
    base_lo = Reg(DT.float)
    base_hi = Reg(DT.float)
    lo_bf16 = Reg(DT.bfloat16)
    hi_bf16 = Reg(DT.bfloat16)
    row_bf16 = Reg(DT.bfloat16)
    dummy_bf16 = Reg(DT.bfloat16)
    for r in range(rows):
        dep_lo <<= dep_ub[r:r + 1, 0:64]
        dep_hi <<= dep_ub[r:r + 1, 64:D]
        base_lo <<= base_ub[r:r + 1, 0:64]
        base_hi <<= base_ub[r:r + 1, 64:D]
        base_lo <<= base_lo + dep_lo
        base_hi <<= base_hi + dep_hi
        lo_bf16 <<= base_lo.astype(DT.bfloat16)
        hi_bf16 <<= base_hi.astype(DT.bfloat16)
        deinterleave(row_bf16, dummy_bf16, lo_bf16, hi_bf16)
        reg_to_ub(sum_bf16_ub[r:r + 1, 0:D], row_bf16)
        base_lo <<= base_lo * -1.0
        base_hi <<= base_hi * -1.0
        neg_ub[r:r + 1, 0:64] <<= base_lo
        neg_ub[r:r + 1, 64:D] <<= base_hi
        lo_bf16 <<= base_lo.astype(DT.bfloat16)
        hi_bf16 <<= base_hi.astype(DT.bfloat16)
        deinterleave(row_bf16, dummy_bf16, lo_bf16, hi_bf16)
        reg_to_ub(neg_nz_ub[r * BF16_C0], row_bf16, rows)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def zero_d_rows_bf16_vf(dst_ub: Tensor, rows: Var):
    zero_lo = Reg(DT.bfloat16)
    zero_hi = Reg(DT.bfloat16)
    row_bf16 = Reg(DT.bfloat16)
    dummy_bf16 = Reg(DT.bfloat16)
    zero_lo <<= 0.0
    zero_hi <<= 0.0
    for r in range(rows):
        deinterleave(row_bf16, dummy_bf16, zero_lo, zero_hi)
        reg_to_ub(dst_ub[r:r + 1, 0:D], row_bf16)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)




@vf()
def update_d_state_half_cast_nz_vf(seed_ub: Tensor, old_d_state_ub: Tensor, out_ub: Tensor, dst_nz_ub: Tensor, g_last: Var):
    seed_lo = Reg(DT.float)
    seed_hi = Reg(DT.float)
    old_lo = Reg(DT.float)
    old_hi = Reg(DT.float)
    scale = Reg(DT.float)
    lo_bf16 = Reg(DT.bfloat16)
    hi_bf16 = Reg(DT.bfloat16)
    row_bf16 = Reg(DT.bfloat16)
    dummy_bf16 = Reg(DT.bfloat16)
    scale <<= g_last
    scale <<= scale.exp()
    for r in range(HALF_D):
        seed_lo <<= seed_ub[r:r + 1, 0:64]
        seed_hi <<= seed_ub[r:r + 1, 64:D]
        old_lo <<= old_d_state_ub[r:r + 1, 0:64]
        old_hi <<= old_d_state_ub[r:r + 1, 64:D]
        old_lo <<= old_lo * scale
        old_hi <<= old_hi * scale
        seed_lo <<= seed_lo + old_lo
        seed_hi <<= seed_hi + old_hi
        out_ub[r:r + 1, 0:64] <<= seed_lo
        out_ub[r:r + 1, 64:D] <<= seed_hi
        lo_bf16 <<= seed_lo.astype(DT.bfloat16)
        hi_bf16 <<= seed_hi.astype(DT.bfloat16)
        deinterleave(row_bf16, dummy_bf16, lo_bf16, hi_bf16)
        reg_to_ub(dst_nz_ub[r * BF16_C0], row_bf16, L)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def accum_state_dot_half_vf(d_state_rows_ub: Tensor, state_bf16_rows_ub: Tensor, accum_ub: Tensor, partial_vec_ub: Tensor):
    d_state_regs = RegList(DT.float, REGS_D)
    state_regs = RegList(DT.float, REGS_D)
    prod_regs = RegList(DT.float, REGS_D)
    low8_mask = MaskReg(DT.float, init_mode=MaskType.LOWEST8)
    row_sum = Reg(DT.float)
    acc = Reg(DT.float)
    acc_vec = Reg(DT.float)
    acc <<= 0.0
    for r in range(L):
        d_state_regs <<= d_state_rows_ub[r:r + 1, 0:D]
        state_regs <<= state_bf16_rows_ub[r:r + 1, 0:D]
        prod_regs <<= d_state_regs * state_regs
        row_sum <<= prod_regs.cadd()
        acc <<= acc + row_sum
    accum_ub[0:1, 0:1] <<= acc.single_value()
    acc_vec <<= acc
    reg_to_ub(partial_vec_ub[0:1, 0:8], acc_vec, mask=low8_mask)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def add_peer_partial_scalar_vf(accum_ub: Tensor, peer_partial_ub: Tensor):
    acc = Reg(DT.float)
    peer = Reg(DT.float)
    acc <<= accum_ub[0:1, 0:1].single()
    peer <<= peer_partial_ub[0:1, 0:1].single()
    acc <<= acc + peer
    accum_ub[0:1, 0:1] <<= acc.single_value()


@vf()
def broadcast_scalar_vf(src_ub: Tensor, dst_vec_ub: Tensor):
    low8_mask = MaskReg(DT.float, init_mode=MaskType.LOWEST8)
    val = Reg(DT.float)
    val <<= src_ub[0:1, 0:1].single()
    reg_to_ub(dst_vec_ub[0:1, 0:8], val, mask=low8_mask)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def pack_second_scalar_vf(dst_vec_ub: Tensor, second_vec_ub: Tensor):
    second = Reg(DT.float)
    second <<= second_vec_ub[0:1, 0:1].single()
    dst_vec_ub[0:1, 1:2] <<= second.single_value()
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def accum_dq_g_half_vf(d_q_exp_ub: Tensor, q_bf16_ub: Tensor, exp_g_ub: Tensor, d_g_ub: Tensor):
    dq_regs = RegList(DT.float, REGS_D)
    q_regs = RegList(DT.float, REGS_D)
    prod_regs = RegList(DT.float, REGS_D)
    exp_g = Reg(DT.float)
    row_sum = Reg(DT.float)
    for r in range(HALF_L):
        exp_g <<= exp_g_ub[0:1, r:r + 1].single()
        dq_regs <<= d_q_exp_ub[r:r + 1, 0:D]
        q_regs <<= q_bf16_ub[r:r + 1, 0:D]
        prod_regs <<= dq_regs * q_regs
        row_sum <<= prod_regs.cadd()
        row_sum <<= row_sum * exp_g
        d_g_ub[0:1, r:r + 1] <<= row_sum.single_value()
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def finish_g_half_vf(d_k_weighted_ub: Tensor, k_bf16_ub: Tensor, exp_delta_ub: Tensor, d_g_base_ub: Tensor, d_g_ub: Tensor, partial_scalar_ub: Tensor, partial_vec_ub: Tensor):
    dk_regs = RegList(DT.float, REGS_D)
    k_regs = RegList(DT.float, REGS_D)
    prod_regs = RegList(DT.float, REGS_D)
    low8_mask = MaskReg(DT.float, init_mode=MaskType.LOWEST8)
    exp_delta = Reg(DT.float)
    d_delta = Reg(DT.float)
    delta_sum = Reg(DT.float)
    g_val = Reg(DT.float)
    partial_vec = Reg(DT.float)
    delta_sum <<= 0.0
    for r in range(HALF_L):
        exp_delta <<= exp_delta_ub[0:1, r:r + 1].single()
        dk_regs <<= d_k_weighted_ub[r:r + 1, 0:D]
        k_regs <<= k_bf16_ub[r:r + 1, 0:D]
        prod_regs <<= dk_regs * k_regs
        d_delta <<= prod_regs.cadd()
        d_delta <<= d_delta * exp_delta
        delta_sum <<= delta_sum + d_delta
        g_val <<= d_g_base_ub[0:1, r:r + 1].single()
        g_val <<= g_val - d_delta
        d_g_ub[0:1, r:r + 1] <<= g_val.single_value()

    partial_scalar_ub[0:1, 0:1] <<= delta_sum.single_value()
    partial_vec <<= delta_sum
    reg_to_ub(partial_vec_ub[0:1, 0:8], partial_vec, mask=low8_mask)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def finish_g_half_tail_vf(d_g_ub: Tensor, local_delta_ub: Tensor, peer_delta_and_exp_ub: Tensor, g_last: Var):
    g_val = Reg(DT.float)
    local_sum = Reg(DT.float)
    peer_sum = Reg(DT.float)
    d_exp = Reg(DT.float)
    exp_last = Reg(DT.float)
    exp_last <<= g_last
    exp_last <<= exp_last.exp()
    g_val <<= d_g_ub[0:1, HALF_L - 1:HALF_L].single()
    local_sum <<= local_delta_ub[0:1, 0:1].single()
    peer_sum <<= peer_delta_and_exp_ub[0:1, 0:1].single()
    d_exp <<= peer_delta_and_exp_ub[0:1, 1:2].single()
    d_exp <<= d_exp * exp_last
    g_val <<= g_val + local_sum
    g_val <<= g_val + peer_sum
    g_val <<= g_val + d_exp
    d_g_ub[0:1, HALF_L - 1:HALF_L] <<= g_val.single_value()
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def finish_g_zero_base_half_vf(d_k_weighted_ub: Tensor, k_bf16_ub: Tensor, exp_delta_ub: Tensor, d_g_ub: Tensor, partial_scalar_ub: Tensor, partial_vec_ub: Tensor):
    dk_regs = RegList(DT.float, REGS_D)
    k_regs = RegList(DT.float, REGS_D)
    prod_regs = RegList(DT.float, REGS_D)
    low8_mask = MaskReg(DT.float, init_mode=MaskType.LOWEST8)
    exp_delta = Reg(DT.float)
    d_delta = Reg(DT.float)
    delta_sum = Reg(DT.float)
    g_val = Reg(DT.float)
    partial_vec = Reg(DT.float)
    delta_sum <<= 0.0
    for r in range(HALF_L):
        exp_delta <<= exp_delta_ub[0:1, r:r + 1].single()
        dk_regs <<= d_k_weighted_ub[r:r + 1, 0:D]
        k_regs <<= k_bf16_ub[r:r + 1, 0:D]
        prod_regs <<= dk_regs * k_regs
        d_delta <<= prod_regs.cadd()
        d_delta <<= d_delta * exp_delta
        delta_sum <<= delta_sum + d_delta
        g_val <<= d_delta * -1.0
        d_g_ub[0:1, r:r + 1] <<= g_val.single_value()

    partial_scalar_ub[0:1, 0:1] <<= delta_sum.single_value()
    partial_vec <<= delta_sum
    reg_to_ub(partial_vec_ub[0:1, 0:8], partial_vec, mask=low8_mask)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def finish_g_zero_base_half_tail_vf(d_g_ub: Tensor, local_partial_ub: Tensor, peer_partial_ub: Tensor):
    g_val = Reg(DT.float)
    local_sum = Reg(DT.float)
    peer_sum = Reg(DT.float)
    g_val <<= d_g_ub[0:1, HALF_L - 1:HALF_L].single()
    local_sum <<= local_partial_ub[0:1, 0:1].single()
    peer_sum <<= peer_partial_ub[0:1, 0:1].single()
    g_val <<= g_val + local_sum
    g_val <<= g_val + peer_sum
    d_g_ub[0:1, HALF_L - 1:HALF_L] <<= g_val.single_value()
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@func()
def _state_mmad(dst: Tensor, l0a: Tensor, l0b: Tensor, left: Tensor, right: Tensor, m, n, k, is_init=True):
    # A shared operand pair bounds MTE1->M flags across the reverse-C loop.
    bar_all()
    l1_to_l0(l0a, left)
    l1_to_l0(l0b, right)
    bar_all()
    mmad(dst, l0a, l0b, M=m, N=n, K=k, is_init=is_init)
    bar_all()


@kernel()
def scan_state_bwd_kernel(
    query: GM[bf16, ('B', 'H', 'C', 64, 128)],
    key: GM[bf16, ('B', 'H', 'C', 64, 128)],
    grad_output: GM[bf16, ('B', 'H', 'C', 64, 128)],
    g_cumsum: GM[f32, ('B', 'H', 'C', 64)],
    k_cumdecay: GM[bf16, ('B', 'H', 'C', 64, 128)],
    state_after_history: GM[bf16, ('B', 'H', 'C', 128, 128)],
    grad_final_state: GM[f32, ('B', 'H', 128, 128)],
    d_score_tmp: GM[bf16, ('B', 'H', 'C', 64, 64)],
    v_new_history: GM[bf16, ('B', 'H', 'C', 64, 128)],
    k_weighted_history: GM[bf16, ('B', 'H', 'C', 64, 128)],
    d_v_attn_tmp: GM[f32, ('B', 'H', 'C', 64, 128)],
    exp_delta_history: GM[f32, ('B', 'H', 'C', 64)],
    d_q_core: GM[f32, ('B', 'H', 'C', 64, 128)],
    d_k_core: GM[f32, ('B', 'H', 'C', 64, 128)],
    d_g_core: GM[f32, ('B', 'H', 'C', 64)],
    d_value_wu: GM[bf16, ('B', 'H', 'C', 64, 128)],
    d_k_cumdecay: GM[bf16, ('B', 'H', 'C', 64, 128)],
    B: i32,
    H: i32,
    C: i32,
):
    cvmutex = CvMutex(0, depth=1, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    cv_seed_mutex = CvMutex(1, depth=1, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    dvprime_mutex = VcMutex(2, depth=1, src_start_pipe=Pipe.MTE3, src_end_pipe=Pipe.MTE3, dst_start_pipe=Pipe.MTE1, dst_end_pipe=Pipe.MTE1)
    dstate_mutex = VcMutex(3, depth=1, src_start_pipe=Pipe.MTE3, src_end_pipe=Pipe.MTE3, dst_start_pipe=Pipe.MTE1, dst_end_pipe=Pipe.MTE1)

    l1_q = Tensor(DT.bfloat16, [L, D], Position.L1)
    l1_k = Tensor(DT.bfloat16, [L, D], Position.L1)
    l1_d_out = Tensor(DT.bfloat16, [L, D], Position.L1)
    l1_kcd = Tensor(DT.bfloat16, [L, D], Position.L1)
    l1_state = Tensor(DT.bfloat16, [D, D], Position.L1)
    l1_d_state = Tensor(DT.bfloat16, [D, D], Position.L1)
    l1_q_exp = Tensor(DT.bfloat16, [L, D], Position.L1)
    l1_k_weighted = Tensor(DT.bfloat16, [L, D], Position.L1)
    l1_v_new = Tensor(DT.bfloat16, [L, D], Position.L1)
    l1_d_score = Tensor(DT.bfloat16, [L, L], Position.L1)
    l1_d_v_prime = Tensor(DT.bfloat16, [L, D], Position.L1)

    shared_l0a = Tensor(DT.bfloat16, [D, D], Position.L0A)
    shared_l0b = Tensor(DT.bfloat16, [D, D], Position.L0B)
    l0c_ld0 = Tensor(DT.float, [L, D], Position.L0C)
    l0c_ld1 = Tensor(DT.float, [L, D], Position.L0C)
    l0c_dd = Tensor(DT.float, [D, D], Position.L0C)

    bf16_full_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    bf16_alt_ub = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    bf16_vprime_ub = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    state_half_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    q_raw_ub = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    k_raw_ub = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    d_q_exp_ub = Tensor(DT.float, [HALF_L, D], Position.UB)
    score_ub = Tensor(DT.float, [HALF_L, D], Position.UB)
    term_ub = Tensor(DT.float, [HALF_L, D], Position.UB)
    out_ub = Tensor(DT.float, [HALF_L, D], Position.UB)
    tmp_ub = Tensor(DT.float, [HALF_L, D], Position.UB)
    g_ub = Tensor(DT.float, [1, HALF_L], Position.UB)
    exp_g_half_ub = Tensor(DT.float, [1, HALF_L], Position.UB)
    exp_delta_ub = Tensor(DT.float, [1, HALF_L], Position.UB)
    d_exp_ub = Tensor(DT.float, [1, 8], Position.UB)
    d_g_base_ub = Tensor(DT.float, [1, HALF_L], Position.UB)
    d_g_full_ub = Tensor(DT.float, [1, HALF_L], Position.UB)
    d_g_partial_ub = Tensor(DT.float, [1, 8], Position.UB)
    d_g_peer_ub = Tensor(DT.float, [1, 8], Position.UB)
    ub_d_state = DBuff(DT.float, [HALF_D, D], Position.UB)
    ub_d_state_seed = Tensor(DT.float, [HALF_D, D], Position.UB)

    bh_count = B * H
    bh_per_core = CeilDiv(bh_count, GetCubeNum())
    bh_begin = Var(bh_per_core * GetCubeIdx())
    bh_end = Min(bh_begin + bh_per_core, bh_count)
    dstate_read_cnt = Var(0)
    dstate_write_cnt = Var(1)

    with auto_sync():
        for bh in range(bh_begin, bh_end):
            b_idx = Var(bh // H)
            h_idx = Var(bh % H)
            row_begin_l = Var(GetSubBlockIdx() * HALF_L)
            row_end_l = Var(row_begin_l + HALF_L)
            rows_l = Var(HALF_L)
            row_begin_d = Var(GetSubBlockIdx() * HALF_D)
            row_end_d = Var(row_begin_d + HALF_D)

            dstate_initial = ub_d_state[dstate_read_cnt]
            dstate_initial[0:HALF_D, 0:D] <<= grad_final_state[b_idx, h_idx, row_begin_d:row_end_d, 0:D]
            dstate_mutex.lock()
            cast_d_rows_float_to_bf16_nz_vf(dstate_initial[0:HALF_D, 0:D], bf16_full_ub, L)
            l1_d_state[row_begin_d:row_end_d, 0:D] <<= bf16_full_ub[0:L, 0:D].nz()
            bar_all()
            dstate_mutex.ready()
            dstate_mutex.wait()

            for rev_c in range(C):
                dstate_cur = ub_d_state[dstate_read_cnt]
                c_idx = Var(C - 1 - rev_c)
                c_prev = Var(c_idx - 1)
                g_last = Var(0.0, DT.float)

                if c_idx != 0:
                    g_last.GetValueFrom(g_cumsum[b_idx, h_idx, c_idx, L - 1:L])
                    q_raw_ub[0:rows_l, 0:D] <<= query[b_idx, h_idx, c_idx, row_begin_l:row_end_l, 0:D]
                    g_ub[0:1, 0:rows_l] <<= g_cumsum[b_idx, h_idx, c_idx, row_begin_l:row_end_l]
                    state_half_ub[0:L, 0:D] <<= state_after_history[b_idx, h_idx, c_prev, row_begin_d:row_end_d, 0:D]
                    make_q_exp_nz_vf(q_raw_ub, g_ub, bf16_alt_ub, exp_g_half_ub, rows_l)
                    l1_q_exp[row_begin_l:row_end_l, 0:D] <<= bf16_alt_ub[0:rows_l, 0:D].nz()
                    accum_state_dot_half_vf(dstate_cur[0:HALF_D, 0:D], state_half_ub, d_exp_ub, d_g_partial_ub)
                    # Both vector halves finish local partials before the peer exchange can overwrite them.
                    if GetSubBlockIdx() == 1:
                        d_g_core[b_idx, h_idx, c_idx, L - 16:L - 8] <<= d_g_partial_ub[0:1, 0:8]
                    intracore_allvec_ready(7, pipe=Pipe.MTE3)
                    intracore_allvec_wait(7, pipe=Pipe.MTE2)
                    if GetSubBlockIdx() == 0:
                        d_g_partial_ub[0:1, 0:8] <<= d_g_core[b_idx, h_idx, c_idx, L - 16:L - 8]
                        add_peer_partial_scalar_vf(d_exp_ub, d_g_partial_ub)
                        broadcast_scalar_vf(d_exp_ub, d_g_peer_ub)

                exp_delta_ub[0:1, 0:rows_l] <<= exp_delta_history[b_idx, h_idx, c_idx, row_begin_l:row_end_l]
                k_raw_ub[0:rows_l, 0:D] <<= key[b_idx, h_idx, c_idx, row_begin_l:row_end_l, 0:D]

                l1_q[0:L, 0:D] <<= query[b_idx, h_idx, c_idx, 0:L, 0:D]
                l1_k[0:L, 0:D] <<= key[b_idx, h_idx, c_idx, 0:L, 0:D]
                l1_v_new[0:L, 0:D] <<= v_new_history[b_idx, h_idx, c_idx, 0:L, 0:D]
                l1_d_score[0:L, 0:L] <<= d_score_tmp[b_idx, h_idx, c_idx, 0:L, 0:L]
                _state_mmad(l0c_ld1, shared_l0a, shared_l0b, l1_v_new, l1_d_state, m=L, n=D, k=D)
                cvmutex.lock()
                term_ub <<= l0c_ld1
                cvmutex.ready()
                cvmutex.wait()
                cvmutex.free()

                if c_idx == 0:
                    finish_g_zero_base_half_vf(term_ub, k_raw_ub, exp_delta_ub, d_g_full_ub, d_exp_ub, d_g_partial_ub)
                    if GetSubBlockIdx() == 0:
                        d_g_core[b_idx, h_idx, c_idx, L - 8:L] <<= d_g_partial_ub[0:1, 0:8]
                    intracore_allvec_ready(6, pipe=Pipe.MTE3)
                    intracore_allvec_wait(6, pipe=Pipe.MTE2)
                    if GetSubBlockIdx() == 1:
                        d_g_partial_ub[0:1, 0:8] <<= d_g_core[b_idx, h_idx, c_idx, L - 8:L]
                        finish_g_zero_base_half_tail_vf(d_g_full_ub, d_exp_ub, d_g_partial_ub)
                    d_g_core[b_idx, h_idx, c_idx, row_begin_l:row_end_l] <<= d_g_full_ub[0:1, 0:rows_l]

                    zero_d_rows_bf16_vf(bf16_alt_ub, rows_l)
                    d_k_cumdecay[b_idx, h_idx, c_idx, row_begin_l:row_end_l, 0:D] <<= bf16_alt_ub[0:rows_l, 0:D]

                    _state_mmad(l0c_ld0, shared_l0a, shared_l0b, l1_d_score, l1_k.T, m=L, n=D, k=L)
                    d_q_core[b_idx, h_idx, c_idx, 0:L, 0:D] <<= l0c_ld0

                    _state_mmad(l0c_ld0, shared_l0a, shared_l0b, l1_d_score.T, l1_q.T, m=L, n=D, k=L)
                    cvmutex.lock()
                    score_ub <<= l0c_ld0
                    cvmutex.ready()
                    cvmutex.wait()
                    postprocess_dk_vf(score_ub, term_ub, exp_delta_ub, d_q_exp_ub, rows_l)
                    d_k_core[b_idx, h_idx, c_idx, row_begin_l:row_end_l, 0:D] <<= d_q_exp_ub[0:rows_l, 0:D]
                    cvmutex.free()

                    l1_k_weighted[0:L, 0:D] <<= k_weighted_history[b_idx, h_idx, c_idx, 0:L, 0:D]
                    _state_mmad(l0c_ld0, shared_l0a, shared_l0b, l1_k_weighted, l1_d_state.T, m=L, n=D, k=D)
                    cvmutex.lock()
                    score_ub <<= l0c_ld0
                    cvmutex.ready()
                    tmp_ub[0:rows_l, 0:D] <<= d_v_attn_tmp[b_idx, h_idx, c_idx, row_begin_l:row_end_l, 0:D]
                    cvmutex.wait()
                    cvmutex.free()
                    add_cast_d_rows_bf16_vf(score_ub, tmp_ub, bf16_alt_ub, rows_l)
                    d_value_wu[b_idx, h_idx, c_idx, row_begin_l:row_end_l, 0:D] <<= bf16_alt_ub[0:rows_l, 0:D]
                else:
                    l1_d_out[0:L, 0:D] <<= grad_output[b_idx, h_idx, c_idx, 0:L, 0:D]
                    l1_kcd[0:L, 0:D] <<= k_cumdecay[b_idx, h_idx, c_idx, 0:L, 0:D]
                    l1_state[0:D, 0:D] <<= state_after_history[b_idx, h_idx, c_prev, 0:D, 0:D]
                    l1_k_weighted[0:L, 0:D] <<= k_weighted_history[b_idx, h_idx, c_idx, 0:L, 0:D]
                    _state_mmad(l0c_ld0, shared_l0a, shared_l0b, l1_d_out, l1_state, m=L, n=D, k=D)
                    cvmutex.lock()
                    score_ub <<= l0c_ld0
                    d_q_exp_ub <<= l0c_ld0
                    cvmutex.ready()
                    _state_mmad(l0c_dd, shared_l0a, shared_l0b, l1_q_exp.T, l1_d_out.T, m=D, n=D, k=L)
                    cvmutex.wait()
                    cvmutex.free()

                    _state_mmad(l0c_ld0, shared_l0a, shared_l0b, l1_k_weighted, l1_d_state.T, m=L, n=D, k=D)
                    cvmutex.lock()
                    out_ub <<= l0c_ld0
                    cvmutex.ready()
                    tmp_ub[0:rows_l, 0:D] <<= d_v_attn_tmp[b_idx, h_idx, c_idx, row_begin_l:row_end_l, 0:D]
                    cvmutex.wait()
                    cvmutex.free()
                    add_negate_cast_d_rows_bf16_vf(out_ub, tmp_ub, bf16_alt_ub, ub_d_state_seed, bf16_vprime_ub, rows_l)
                    d_value_wu[b_idx, h_idx, c_idx, row_begin_l:row_end_l, 0:D] <<= bf16_alt_ub[0:rows_l, 0:D]

                    dvprime_mutex.lock()
                    l1_d_v_prime[row_begin_l:row_end_l, 0:D] <<= bf16_vprime_ub[0:rows_l, 0:D].nz()
                    dvprime_mutex.ready()
                    dvprime_mutex.wait()
                    bar_all()

                    _state_mmad(l0c_ld0, shared_l0a, shared_l0b, l1_d_v_prime, l1_state, m=L, n=D, k=D)
                    d_k_cumdecay[b_idx, h_idx, c_idx, 0:L, 0:D] <<= l0c_ld0
                    _state_mmad(l0c_dd, shared_l0a, shared_l0b, l1_kcd.T, l1_d_v_prime.T, m=D, n=D, k=L, is_init=False)
                    cv_seed_mutex.lock()
                    l0c_to_ub(ub_d_state_seed, l0c_dd, M=D, N=D, N_dst=D, M_src=D, dual_mode=DualMode.SPLITM, sub_block_id=0)
                    cv_seed_mutex.ready()
                    dvprime_mutex.free()

                    accum_dq_g_half_vf(d_q_exp_ub, q_raw_ub, exp_g_half_ub, d_g_base_ub)
                    finish_g_half_vf(term_ub, k_raw_ub, exp_delta_ub, d_g_base_ub, d_g_full_ub, d_exp_ub, d_g_partial_ub)
                    if GetSubBlockIdx() == 0:
                        pack_second_scalar_vf(d_g_partial_ub, d_g_peer_ub)
                        d_g_core[b_idx, h_idx, c_idx, L - 8:L] <<= d_g_partial_ub[0:1, 0:8]
                    intracore_allvec_ready(6, pipe=Pipe.MTE3)
                    intracore_allvec_wait(6, pipe=Pipe.MTE2)
                    if GetSubBlockIdx() == 1:
                        d_g_partial_ub[0:1, 0:8] <<= d_g_core[b_idx, h_idx, c_idx, L - 8:L]
                        finish_g_half_tail_vf(d_g_full_ub, d_exp_ub, d_g_partial_ub, g_last)
                    d_g_core[b_idx, h_idx, c_idx, row_begin_l:row_end_l] <<= d_g_full_ub[0:1, 0:rows_l]

                    _state_mmad(l0c_ld0, shared_l0a, shared_l0b, l1_d_score, l1_k.T, m=L, n=D, k=L)
                    cvmutex.lock()
                    out_ub <<= l0c_ld0
                    cvmutex.ready()
                    cvmutex.wait()
                    add_scaled_d_rows_vf(out_ub, score_ub, exp_g_half_ub, out_ub, rows_l)
                    d_q_core[b_idx, h_idx, c_idx, row_begin_l:row_end_l, 0:D] <<= out_ub[0:rows_l, 0:D]
                    cvmutex.free()

                    _state_mmad(l0c_ld0, shared_l0a, shared_l0b, l1_d_score.T, l1_q.T, m=L, n=D, k=L)
                    cvmutex.lock()
                    score_ub <<= l0c_ld0
                    cvmutex.ready()
                    cvmutex.wait()
                    postprocess_dk_vf(score_ub, term_ub, exp_delta_ub, d_q_exp_ub, rows_l)
                    d_k_core[b_idx, h_idx, c_idx, row_begin_l:row_end_l, 0:D] <<= d_q_exp_ub[0:rows_l, 0:D]
                    cvmutex.free()

                if c_idx != 0:
                    dstate_mutex.free()
                    cv_seed_mutex.wait()
                    dstate_next = ub_d_state[dstate_write_cnt]
                    update_d_state_half_cast_nz_vf(ub_d_state_seed, dstate_cur, dstate_next, bf16_full_ub, g_last)
                    cv_seed_mutex.free()
                    dstate_mutex.lock()
                    l1_d_state[row_begin_d:row_end_d, 0:D] <<= bf16_full_ub[0:L, 0:D].nz()
                    bar_all()
                    dstate_mutex.ready()
                    dstate_mutex.wait()
                    dstate_read_cnt += 1
                    dstate_write_cnt += 1

            dstate_mutex.free()

    bar_all()
    return d_q_core, d_k_core, d_g_core, d_value_wu, d_k_cumdecay

# ----------------------------------------------------------------------------------------------------
# wu.py
# gdn_bwd production stage migrated from kernels/wu_bwd.py.
#
# M10 edits: preserve the selected kernel and its source dependency closure;
# replace the EasyASC facade, synthesize typed GM/scalar signatures from the
# reviewed stage contract, and preserve explicit Python unrolling with unroll.
# Host drivers live in main.py's `execute`; the independent reference is reference.py.
# Source SHA256: bc658c344ca35813b6d20f2384e81704e524ee9eff38d16dea0a265a814b329a
# ----------------------------------------------------------------------------------------------------

@vf()
def make_wu_inputs_vf(
    key_ub: Tensor,
    value_ub: Tensor,
    beta_ub: Tensor,
    g_ub: Tensor,
    v_beta_nz_ub: Tensor,
    k_beta_g_nz_ub: Tensor,
    k_beta_ub: Tensor,
    exp_g_ub: Tensor,
    rows: Var,
):
    beta_reg = Reg(DT.float)
    g_reg = Reg(DT.float)
    row_lo = Reg(DT.float)
    row_hi = Reg(DT.float)
    lo_bf16 = Reg(DT.bfloat16)
    hi_bf16 = Reg(DT.bfloat16)
    row_bf16 = Reg(DT.bfloat16)
    dummy_bf16 = Reg(DT.bfloat16)

    for r in range(rows):
        beta_reg <<= beta_ub[0:1, r:r + 1].single()
        g_reg <<= g_ub[0:1, r:r + 1].single()
        g_reg <<= g_reg.exp()
        exp_g_ub[0:1, r:r + 1] <<= g_reg.single_value()

        row_lo <<= value_ub[r:r + 1, 0:64]
        row_lo <<= row_lo * beta_reg
        row_hi <<= value_ub[r:r + 1, 64:D]
        row_hi <<= row_hi * beta_reg
        lo_bf16 <<= row_lo.astype(DT.bfloat16)
        hi_bf16 <<= row_hi.astype(DT.bfloat16)
        deinterleave(row_bf16, dummy_bf16, lo_bf16, hi_bf16)
        reg_to_ub(v_beta_nz_ub[r * BF16_C0], row_bf16, rows)

        row_lo <<= key_ub[r:r + 1, 0:64]
        row_lo <<= row_lo * beta_reg
        k_beta_ub[r:r + 1, 0:64] <<= row_lo
        row_lo <<= row_lo * g_reg
        row_hi <<= key_ub[r:r + 1, 64:D]
        row_hi <<= row_hi * beta_reg
        k_beta_ub[r:r + 1, 64:D] <<= row_hi
        row_hi <<= row_hi * g_reg
        lo_bf16 <<= row_lo.astype(DT.bfloat16)
        hi_bf16 <<= row_hi.astype(DT.bfloat16)
        deinterleave(row_bf16, dummy_bf16, lo_bf16, hi_bf16)
        reg_to_ub(k_beta_g_nz_ub[r * BF16_C0], row_bf16, rows)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def finalize_wu_partials_vf(
    d_k_beta_g_ub: Tensor,
    k_beta_ub: Tensor,
    exp_g_ub: Tensor,
    d_k_beta_ub: Tensor,
    d_g_ub: Tensor,
    rows: Var,
):
    d_k_beta_regs = RegList(DT.float, REGS_D)
    k_beta_regs = RegList(DT.float, REGS_D)
    tmp_regs = RegList(DT.float, REGS_D)
    exp_g = Reg(DT.float)
    d_g = Reg(DT.float)

    for r in range(rows):
        exp_g <<= exp_g_ub[0:1, r:r + 1].single()
        d_k_beta_regs <<= d_k_beta_g_ub[r:r + 1, 0:D]
        k_beta_regs <<= k_beta_ub[r:r + 1, 0:D]

        d_k_beta_regs <<= d_k_beta_regs * exp_g
        d_k_beta_ub[r:r + 1, 0:D] <<= d_k_beta_regs

        tmp_regs <<= d_k_beta_regs * k_beta_regs
        d_g <<= tmp_regs.cadd()
        d_g_ub[0:1, r:r + 1] <<= d_g.single_value()
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@kernel()
def wu_bwd_kernel(
    key: GM[bf16, ('B', 'H', 'C', 64, 128)],
    value: GM[bf16, ('B', 'H', 'C', 64, 128)],
    beta: GM[f32, ('B', 'H', 'C', 64)],
    g_cumsum: GM[f32, ('B', 'H', 'C', 64)],
    wu_attn_bf16: GM[bf16, ('B', 'H', 'C', 64, 64)],
    d_value_wu: GM[bf16, ('B', 'H', 'C', 64, 128)],
    d_k_cumdecay: GM[bf16, ('B', 'H', 'C', 64, 128)],
    d_wu: GM[bf16, ('B', 'H', 'C', 64, 64)],
    d_v_beta: GM[f32, ('B', 'H', 'C', 64, 128)],
    d_k_beta: GM[f32, ('B', 'H', 'C', 64, 128)],
    d_g_wu: GM[f32, ('B', 'H', 'C', 64)],
    B: i32,
    H: i32,
    C: i32,
):
    vcmutex = VcMutex(
        0,
        depth=2,
        src_start_pipe=Pipe.MTE3,
        src_end_pipe=Pipe.MTE3,
        dst_start_pipe=Pipe.MTE1,
        dst_end_pipe=Pipe.MTE1,
    )
    cvmutex = CvMutex(1, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)

    l1_v_beta = DBuff(DT.bfloat16, [L, D], Position.L1)
    l1_k_beta_g = DBuff(DT.bfloat16, [L, D], Position.L1)
    l1_w = DBuff(DT.bfloat16, [L, L], Position.L1)
    l1_d_value_wu = DBuff(DT.bfloat16, [L, D], Position.L1)
    l1_d_k_cumdecay = DBuff(DT.bfloat16, [L, D], Position.L1)

    l0c_d_wu = DBuff(DT.float, [L, L], Position.L0C)
    l0c_d_v_beta = DBuff(DT.float, [L, D], Position.L0C)
    l0c_d_k_beta_g = DBuff(DT.float, [L, D], Position.L0C)

    key_ub = DBuff(DT.bfloat16, [HALF_L, D], Position.UB)
    value_ub = DBuff(DT.bfloat16, [HALF_L, D], Position.UB)
    beta_ub = DBuff(DT.float, [1, HALF_L], Position.UB)
    g_ub = DBuff(DT.float, [1, HALF_L], Position.UB)
    v_beta_nz_ub = DBuff(DT.bfloat16, [HALF_L, D], Position.UB)
    k_beta_g_nz_ub = DBuff(DT.bfloat16, [HALF_L, D], Position.UB)
    k_beta_ub = DBuff(DT.float, [HALF_L, D], Position.UB)
    exp_g_ub = DBuff(DT.float, [1, HALF_L], Position.UB)
    d_k_beta_g_ub = DBuff(DT.float, [HALF_L, D], Position.UB)
    d_k_beta_ub = Tensor(DT.float, [HALF_L, D], Position.UB)
    d_g_ub = Tensor(DT.float, [1, HALF_L], Position.UB)

    bhc_count = B * H * C
    bhc_per_core = CeilDiv(bhc_count, GetCubeNum())
    bhc_begin = Var(bhc_per_core * GetCubeIdx())
    bhc_end = Min(bhc_begin + bhc_per_core, bhc_count)

    for pipe_bhc in range(bhc_begin, bhc_end + 1):
        if pipe_bhc < bhc_end:
            with auto_sync():
                pipe_slot = var_mod(pipe_bhc - bhc_begin, 2)
                c_idx = Var(pipe_bhc // (B * H))
                bh_remainder = Var(pipe_bhc % (B * H))
                b_idx = Var(bh_remainder // H)
                h_idx = Var(bh_remainder % H)

                row_begin = Var(GetSubBlockIdx() * HALF_L)
                row_end = Var(row_begin + HALF_L)
                rows_this = Var(HALF_L)

                vcmutex.lock()
                key_ub[pipe_slot][0:rows_this, 0:D] <<= key[b_idx, h_idx, c_idx, row_begin:row_end, 0:D]
                value_ub[pipe_slot][0:rows_this, 0:D] <<= value[b_idx, h_idx, c_idx, row_begin:row_end, 0:D]
                beta_ub[pipe_slot][0:1, 0:rows_this] <<= beta[b_idx, h_idx, c_idx, row_begin:row_end]
                g_ub[pipe_slot][0:1, 0:rows_this] <<= g_cumsum[b_idx, h_idx, c_idx, row_begin:row_end]
                make_wu_inputs_vf(
                    key_ub[pipe_slot],
                    value_ub[pipe_slot],
                    beta_ub[pipe_slot],
                    g_ub[pipe_slot],
                    v_beta_nz_ub[pipe_slot],
                    k_beta_g_nz_ub[pipe_slot],
                    k_beta_ub[pipe_slot],
                    exp_g_ub[pipe_slot],
                    rows_this,
                )
                l1_v_beta[pipe_slot][row_begin:row_end, 0:D] <<= v_beta_nz_ub[pipe_slot][0:rows_this, 0:D].nz()
                l1_k_beta_g[pipe_slot][row_begin:row_end, 0:D] <<= k_beta_g_nz_ub[pipe_slot][0:rows_this, 0:D].nz()
                vcmutex.ready()

        if pipe_bhc > bhc_begin:
            with auto_sync():
                bhc = Var(pipe_bhc - 1)
                bhc_slot = var_mod(bhc - bhc_begin, 2)
                c_idx = Var(bhc // (B * H))
                bh_remainder = Var(bhc % (B * H))
                b_idx = Var(bh_remainder // H)
                h_idx = Var(bh_remainder % H)

                row_begin = Var(GetSubBlockIdx() * HALF_L)
                row_end = Var(row_begin + HALF_L)
                rows_this = Var(HALF_L)
                d_k_beta_g_slot = d_k_beta_g_ub[bhc_slot]

                l1_w[bhc_slot][0:L, 0:L] <<= wu_attn_bf16[b_idx, h_idx, c_idx, 0:L, 0:L]
                l1_d_value_wu[bhc_slot][0:L, 0:D] <<= d_value_wu[b_idx, h_idx, c_idx, 0:L, 0:D]
                l1_d_k_cumdecay[bhc_slot][0:L, 0:D] <<= d_k_cumdecay[b_idx, h_idx, c_idx, 0:L, 0:D]
                vcmutex.wait()
                matmul(l0c_d_wu[bhc_slot], l1_d_value_wu[bhc_slot], l1_v_beta[bhc_slot], m=L, n=L, k=D, splitn=L)
                matmul(l0c_d_wu[bhc_slot], l1_d_k_cumdecay[bhc_slot], l1_k_beta_g[bhc_slot], m=L, n=L, k=D, splitn=L, is_init=False)
                matmul(l0c_d_v_beta[bhc_slot], l1_w[bhc_slot].T, l1_d_value_wu[bhc_slot].T, m=L, n=D, k=L, splitn=D)
                matmul(l0c_d_k_beta_g[bhc_slot], l1_w[bhc_slot].T, l1_d_k_cumdecay[bhc_slot].T, m=L, n=D, k=L, splitn=D)

                vcmutex.free()

                cvmutex.lock()
                d_k_beta_g_slot <<= l0c_d_k_beta_g[bhc_slot]
                cvmutex.ready()

                d_wu[b_idx, h_idx, c_idx, 0:L, 0:L] <<= l0c_d_wu[bhc_slot]
                d_v_beta[b_idx, h_idx, c_idx, 0:L, 0:D] <<= l0c_d_v_beta[bhc_slot]

                cvmutex.wait()
                finalize_wu_partials_vf(
                    d_k_beta_g_slot,
                    k_beta_ub[bhc_slot],
                    exp_g_ub[bhc_slot],
                    d_k_beta_ub,
                    d_g_ub,
                    rows_this,
                )
                cvmutex.free()
                d_k_beta[b_idx, h_idx, c_idx, row_begin:row_end, 0:D] <<= d_k_beta_ub[0:rows_this, 0:D]
                d_g_wu[b_idx, h_idx, c_idx, row_begin:row_end] <<= d_g_ub[0:1, 0:rows_this]
    return d_wu, d_v_beta, d_k_beta, d_g_wu

# ----------------------------------------------------------------------------------------------------
# inverse_preprocess.py
# gdn_bwd production stage migrated from kernels/inverse_preprocess_bwd.py.
#
# M10 edits: preserve the selected kernel and its source dependency closure;
# replace the EasyASC facade, synthesize typed GM/scalar signatures from the
# reviewed stage contract, and preserve explicit Python unrolling with unroll.
# Host drivers live in main.py's `execute`; the independent reference is reference.py.
# The delayed score publication has an explicit V-to-MTE3 event so it cannot
# reuse the next key producer's outstanding depth-one token during pipelining.
# Source SHA256: a29d45958ae30b4ca03b81bc52d449b5edbcc3b8e7b84d74069f9af704aaedc1
# ----------------------------------------------------------------------------------------------------

@vf()
def cast_key_beta_and_w_vf(
    key_bf16_ub: Tensor,
    beta_ub: Tensor,
    key_nz_ub: Tensor,
    k_beta_nz_ub: Tensor,
    rows: Var,
):
    beta_reg = Reg(DT.float)
    key_row = Reg(DT.bfloat16)
    row_lo = Reg(DT.float)
    row_hi = Reg(DT.float)
    lo_bf16 = Reg(DT.bfloat16)
    hi_bf16 = Reg(DT.bfloat16)
    row_bf16 = Reg(DT.bfloat16)
    dummy_bf16 = Reg(DT.bfloat16)

    for r in range(rows):
        beta_reg <<= beta_ub[0:1, r:r + 1].single()

        key_row <<= key_bf16_ub[r:r + 1, 0:D]
        reg_to_ub(key_nz_ub[r * BF16_C0], key_row, rows)

        row_lo <<= key_bf16_ub[r:r + 1, 0:64]
        row_lo <<= row_lo * beta_reg
        row_hi <<= key_bf16_ub[r:r + 1, 64:D]
        row_hi <<= row_hi * beta_reg
        lo_bf16 <<= row_lo.astype(DT.bfloat16)
        hi_bf16 <<= row_hi.astype(DT.bfloat16)
        deinterleave(row_bf16, dummy_bf16, lo_bf16, hi_bf16)
        reg_to_ub(k_beta_nz_ub[r * BF16_C0], row_bf16, rows)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)




@vf()
def cast_l_rows_float_to_bf16_nz_barrier_vf(src_ub: Tensor, dst_nz_ub: Tensor, rows: Var):
    row_float = Reg(DT.float)
    lo_bf16 = Reg(DT.bfloat16)
    row_bf16 = Reg(DT.bfloat16)
    dummy_bf16 = Reg(DT.bfloat16)
    bf16_lowhalf = MaskReg(DT.bfloat16, init_mode=MaskType.LOWHALF)

    for r in range(rows):
        row_float <<= src_ub[r:r + 1, 0:L]
        lo_bf16 <<= row_float.astype(DT.bfloat16)
        deinterleave(row_bf16, dummy_bf16, lo_bf16, dummy_bf16)
        reg_to_ub(dst_nz_ub[r * BF16_C0], row_bf16, rows, mask=bf16_lowhalf)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def mask_score_and_decay_vf(
    d_attn_ub: Tensor,
    preprocess_score_ub: Tensor,
    decay_mask_ub: Tensor,
    d_score_ub: Tensor,
    d_decay_ub: Tensor,
    d_decay_masked_ub: Tensor,
    row_begin: Var,
    rows: Var,
):
    cols = Reg(DT.int)
    d_attn_row = Reg(DT.float)
    score_row = Reg(DT.float)
    decay_row = Reg(DT.float)
    out_score = Reg(DT.float)
    out_decay = Reg(DT.float)
    selected_score = Reg(DT.float)
    selected_decay = Reg(DT.float)
    zero = Reg(DT.float)
    strict_lower_mask = MaskReg(DT.int, init_mode=MaskType.NONE)

    zero <<= 0.0

    for r in range(rows):
        abs_r = Var(row_begin + r)
        cols.arange(0)
        d_attn_row <<= d_attn_ub[r:r + 1, 0:L]
        score_row <<= preprocess_score_ub[r:r + 1, 0:L]
        decay_row <<= decay_mask_ub[r:r + 1, 0:L]

        out_score <<= d_attn_row * decay_row
        out_score <<= out_score * -1.0
        out_decay <<= d_attn_row * score_row
        out_decay <<= out_decay * -1.0

        compare(strict_lower_mask, cols, abs_r, CompareMode.LT)
        select(selected_score, out_score, zero, mask=strict_lower_mask)
        select(selected_decay, out_decay, zero, mask=strict_lower_mask)
        d_score_ub[r:r + 1, 0:L] <<= selected_score
        d_decay_ub[r:r + 1, 0:L] <<= selected_decay
        selected_decay <<= selected_decay * decay_row
        d_decay_masked_ub[r:r + 1, 0:L] <<= selected_decay
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@kernel()
def inverse_preprocess_bwd_kernel(
    key: GM[bf16, ('B', 'H', 'C', 64, 128)],
    beta: GM[f32, ('B', 'H', 'C', 64)],
    decay_mask: GM[f32, ('B', 'H', 'C', 64, 64)],
    wu_attn_bf16: GM[bf16, ('B', 'H', 'C', 64, 64)],
    d_wu: GM[bf16, ('B', 'H', 'C', 64, 64)],
    d_k_pre: GM[f32, ('B', 'H', 'C', 64, 128)],
    d_k_beta_pre: GM[f32, ('B', 'H', 'C', 64, 128)],
    d_decay_pre: GM[f32, ('B', 'H', 'C', 64, 64)],
    d_decay_pre_masked: GM[f32, ('B', 'H', 'C', 64, 64)],
    B: i32,
    H: i32,
    C: i32,
):
    key_mutex = VcMutex(
        0,
        depth=2,
        src_start_pipe=Pipe.MTE3,
        src_end_pipe=Pipe.MTE3,
        dst_start_pipe=Pipe.MTE1,
        dst_end_pipe=Pipe.MTE1,
    )
    cvmutex = CvMutex(1, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE3)
    score_mutex = VcMutex(
        2,
        depth=1,
        src_start_pipe=Pipe.MTE3,
        src_end_pipe=Pipe.MTE3,
        dst_start_pipe=Pipe.MTE1,
        dst_end_pipe=Pipe.MTE1,
    )
    fix_to_mte1 = SEvent(Pipe.FIX, Pipe.MTE1)
    fix_to_m = SEvent(Pipe.FIX, Pipe.M, name="inverse_l0c_reuse_ready")
    score_copy_ready = SEvent(Pipe.V, Pipe.MTE3, name="inverse_score_copy_ready")

    l1_key_buf = TBuff(DT.bfloat16, [L, D], Position.L1)
    l1_k_beta_buf = TBuff(DT.bfloat16, [L, D], Position.L1)
    l1_w_buf = DBuff(DT.bfloat16, [L, L], Position.L1)
    l1_d_wu_buf = DBuff(DT.bfloat16, [L, L], Position.L1)
    l1_tmp_buf = DBuff(DT.bfloat16, [L, L], Position.L1)
    l1_d_score_buf = DBuff(DT.bfloat16, [L, L], Position.L1)

    l0c_preprocess_score_buf = DBuff(DT.float, [L, L], Position.L0C)
    l0c_tmp_buf = DBuff(DT.float, [L, L], Position.L0C)
    l0c_d_k_buf = DBuff(DT.float, [L, D], Position.L0C)
    l0c_d_k_beta_buf = DBuff(DT.float, [L, D], Position.L0C)

    key_bf16_ub_buf = TBuff(DT.bfloat16, [HALF_L, D], Position.UB)
    beta_ub_buf = TBuff(DT.float, [1, HALF_L], Position.UB)
    key_nz_ub_buf = TBuff(DT.bfloat16, [HALF_L, D], Position.UB)
    k_beta_nz_ub_buf = TBuff(DT.bfloat16, [HALF_L, D], Position.UB)
    d_attn_ub_buf = TBuff(DT.float, [HALF_L, L], Position.UB)
    preprocess_score_ub_buf = TBuff(DT.float, [HALF_L, L], Position.UB)
    decay_mask_ub_buf = TBuff(DT.float, [HALF_L, L], Position.UB)
    d_score_nz_ub_buf = DBuff(DT.bfloat16, [HALF_L, L], Position.UB)

    bhc_count = B * H * C
    bhc_per_core = CeilDiv(bhc_count, GetCubeNum())
    bhc_begin = Var(bhc_per_core * GetCubeIdx())
    bhc_end = Min(bhc_begin + bhc_per_core, bhc_count)

    if bhc_begin < bhc_end:
        with auto_sync():
            for pipe_bhc in range(bhc_begin, bhc_end + 1):
                row_begin = Var(GetSubBlockIdx() * HALF_L)
                row_end = Min(row_begin + HALF_L, L)
                rows_this = Var(row_end - row_begin)

                if pipe_bhc < bhc_end:
                    c_idx_next = Var(pipe_bhc // (B * H))
                    bh_remainder_next = Var(pipe_bhc % (B * H))
                    b_idx_next = Var(bh_remainder_next // H)
                    h_idx_next = Var(bh_remainder_next % H)

                    l1_key_next = l1_key_buf[pipe_bhc]
                    l1_k_beta_next = l1_k_beta_buf[pipe_bhc]
                    l1_w_next = l1_w_buf[pipe_bhc]
                    l1_d_wu_next = l1_d_wu_buf[pipe_bhc]
                    l1_tmp_next = l1_tmp_buf[pipe_bhc]
                    key_bf16_ub_next = key_bf16_ub_buf[pipe_bhc]
                    beta_ub_next = beta_ub_buf[pipe_bhc]
                    key_nz_ub_next = key_nz_ub_buf[pipe_bhc]
                    k_beta_nz_ub_next = k_beta_nz_ub_buf[pipe_bhc]
                    decay_mask_ub_next = decay_mask_ub_buf[pipe_bhc]
                    d_attn_ub_next = d_attn_ub_buf[pipe_bhc]
                    preprocess_score_ub_next = preprocess_score_ub_buf[pipe_bhc]
                    l0c_preprocess_score_next = l0c_preprocess_score_buf[pipe_bhc]
                    l0c_tmp_next = l0c_tmp_buf[pipe_bhc]
                    key_mutex.lock()
                    key_bf16_ub_next[0:rows_this, 0:D] <<= key[
                        b_idx_next, h_idx_next, c_idx_next, row_begin:row_end, 0:D
                    ]
                    beta_ub_next[0:1, 0:rows_this] <<= beta[
                        b_idx_next, h_idx_next, c_idx_next, row_begin:row_end
                    ]
                    cast_key_beta_and_w_vf(
                        key_bf16_ub_next,
                        beta_ub_next,
                        key_nz_ub_next,
                        k_beta_nz_ub_next,
                        rows_this,
                    )
                    l1_key_next[row_begin:row_end, 0:D] <<= key_nz_ub_next[0:rows_this, 0:D].nz()
                    l1_k_beta_next[row_begin:row_end, 0:D] <<= k_beta_nz_ub_next[0:rows_this, 0:D].nz()
                    key_mutex.ready()
                    decay_mask_ub_next[0:rows_this, 0:L] <<= decay_mask[
                        b_idx_next, h_idx_next, c_idx_next, row_begin:row_end, 0:L
                    ]

                    l1_w_next[0:L, 0:L] <<= wu_attn_bf16[b_idx_next, h_idx_next, c_idx_next, 0:L, 0:L]
                    l1_d_wu_next[0:L, 0:L] <<= d_wu[b_idx_next, h_idx_next, c_idx_next, 0:L, 0:L]
                    key_mutex.wait()
                    matmul(l0c_preprocess_score_next, l1_k_beta_next, l1_key_next, m=L, n=L, k=D, splitn=L)
                    matmul(l0c_tmp_next, l1_w_next.T, l1_d_wu_next.T, m=L, n=L, k=L, splitn=L)
                    l1_tmp_next <<= l0c_tmp_next
                    fix_to_mte1.set()
                    fix_to_mte1.wait()
                    # FIX has retired tmp; its L0C slot now holds d_attn.
                    matmul(l0c_tmp_next, l1_tmp_next, l1_w_next, m=L, n=L, k=L, splitn=L)

                    cvmutex.lock()
                    d_attn_ub_next <<= l0c_tmp_next
                    preprocess_score_ub_next <<= l0c_preprocess_score_next
                    fix_to_m.set()
                    fix_to_m.wait()
                    cvmutex.ready()

                if pipe_bhc > bhc_begin:
                    bhc = Var(pipe_bhc - 1)
                    c_idx = Var(bhc // (B * H))
                    bh_remainder = Var(bhc % (B * H))
                    b_idx = Var(bh_remainder // H)
                    h_idx = Var(bh_remainder % H)

                    l1_key = l1_key_buf[bhc]
                    l1_k_beta = l1_k_beta_buf[bhc]
                    l1_d_score = l1_d_score_buf[bhc]
                    d_attn_ub = d_attn_ub_buf[bhc]
                    preprocess_score_ub = preprocess_score_ub_buf[bhc]
                    decay_mask_ub = decay_mask_ub_buf[bhc]
                    d_score_nz_ub = d_score_nz_ub_buf[bhc]
                    l0c_d_k = l0c_d_k_buf[bhc]
                    l0c_d_k_beta = l0c_d_k_beta_buf[bhc]

                    score_mutex.lock()
                    cvmutex.wait()
                    # All three source rows are in registers before the in-place stores.
                    mask_score_and_decay_vf(
                        d_attn_ub,
                        preprocess_score_ub,
                        decay_mask_ub,
                        d_attn_ub,
                        decay_mask_ub,
                        preprocess_score_ub,
                        row_begin,
                        rows_this,
                    )
                    cast_l_rows_float_to_bf16_nz_barrier_vf(d_attn_ub, d_score_nz_ub, rows_this)
                    # The delayed score consumer and next key producer need independent tokens.
                    score_copy_ready.set()
                    score_copy_ready.wait()
                    l1_d_score[row_begin:row_end, 0:L] <<= d_score_nz_ub[0:rows_this, 0:L].nz()
                    score_mutex.ready()
                    d_decay_pre[b_idx, h_idx, c_idx, row_begin:row_end, 0:L] <<= decay_mask_ub[0:rows_this, 0:L]
                    d_decay_pre_masked[b_idx, h_idx, c_idx, row_begin:row_end, 0:L] <<= preprocess_score_ub[0:rows_this, 0:L]
                    # These in-place results remain leased through their final MTE3 reads.
                    cvmutex.free()

                    score_mutex.wait()
                    matmul(l0c_d_k_beta, l1_d_score, l1_key.T, m=L, n=D, k=L, splitn=D)
                    matmul(l0c_d_k, l1_d_score.T, l1_k_beta.T, m=L, n=D, k=L, splitn=D)
                    score_mutex.free()
                    key_mutex.free()
                    d_k_beta_pre[b_idx, h_idx, c_idx, 0:L, 0:D] <<= l0c_d_k_beta
                    d_k_pre[b_idx, h_idx, c_idx, 0:L, 0:D] <<= l0c_d_k
    return d_k_pre, d_k_beta_pre, d_decay_pre, d_decay_pre_masked

# ----------------------------------------------------------------------------------------------------
# finalize.py
# gdn_bwd production stage migrated from kernels/finalize_bwd.py.
#
# M10 edits: preserve the selected kernel and its source dependency closure;
# replace the EasyASC facade, synthesize typed GM/scalar signatures from the
# reviewed stage contract, and preserve explicit Python unrolling with unroll.
# Host drivers live in main.py's `execute`; the independent reference is reference.py.
# Source SHA256: 60a787af01e7cbba1274071e19f86954405fe82921c6cdca03f651c097927c55
# ----------------------------------------------------------------------------------------------------

UPPER_TILE_L = L // 4
BHC_TILE = 4
ACC_TILE_M = 16


@vf()
def finalize_upper_vf(
    key_ub: Tensor,
    value_ub: Tensor,
    beta_ub: Tensor,
    d_v_beta_ub: Tensor,
    d_k_beta_wu_ub: Tensor,
    d_k_beta_pre_ub: Tensor,
    d_k_core_ub: Tensor,
    d_k_pre_ub: Tensor,
    d_value_out_ub: Tensor,
    d_beta_ub: Tensor,
    d_key_out_ub: Tensor,
    rows: Var,
):
    key_regs = RegList(DT.float, REGS_D)
    value_regs = RegList(DT.float, REGS_D)
    d_v_regs = RegList(DT.float, REGS_D)
    d_k_beta_regs = RegList(DT.float, REGS_D)
    d_k_beta_pre_regs = RegList(DT.float, REGS_D)
    d_k_regs = RegList(DT.float, REGS_D)
    d_k_pre_regs = RegList(DT.float, REGS_D)
    tmp_regs = RegList(DT.float, REGS_D)
    beta_reg = Reg(DT.float)
    beta_sum = Reg(DT.float)

    for r in range(rows):
        beta_reg <<= beta_ub[0:1, r:r + 1].single()
        key_regs <<= key_ub[r:r + 1, 0:D]
        value_regs <<= value_ub[r:r + 1, 0:D]
        d_v_regs <<= d_v_beta_ub[r:r + 1, 0:D]
        d_k_beta_regs <<= d_k_beta_wu_ub[r:r + 1, 0:D]
        d_k_beta_pre_regs <<= d_k_beta_pre_ub[r:r + 1, 0:D]
        d_k_regs <<= d_k_core_ub[r:r + 1, 0:D]
        d_k_pre_regs <<= d_k_pre_ub[r:r + 1, 0:D]

        d_k_beta_regs <<= d_k_beta_regs + d_k_beta_pre_regs

        tmp_regs <<= d_v_regs * value_regs
        key_regs <<= d_k_beta_regs * key_regs
        tmp_regs <<= tmp_regs + key_regs
        beta_sum <<= tmp_regs.cadd()
        d_beta_ub[0:1, r:r + 1] <<= beta_sum.single_value()

        d_v_regs <<= d_v_regs * beta_reg
        d_value_out_ub[r:r + 1, 0:D] <<= d_v_regs

        d_k_beta_regs <<= d_k_beta_regs * beta_reg
        d_k_regs <<= d_k_regs + d_k_pre_regs
        d_k_regs <<= d_k_regs + d_k_beta_regs
        d_key_out_ub[r:r + 1, 0:D] <<= d_k_regs


@kernel()
def finalize_bwd_kernel(
    key: GM[bf16, ('B', 'H', 'C', 64, 128)],
    value: GM[bf16, ('B', 'H', 'C', 64, 128)],
    beta: GM[f32, ('B', 'H', 'C', 64)],
    d_k_core: GM[f32, ('B', 'H', 'C', 64, 128)],
    d_g_core: GM[f32, ('BHC', 64)],
    d_decay_core_masked: GM[f32, ('BHC', 64, 64)],
    d_v_beta: GM[f32, ('B', 'H', 'C', 64, 128)],
    d_k_beta_wu: GM[f32, ('B', 'H', 'C', 64, 128)],
    d_g_wu: GM[f32, ('BHC', 64)],
    d_k_pre: GM[f32, ('B', 'H', 'C', 64, 128)],
    d_k_beta_pre: GM[f32, ('B', 'H', 'C', 64, 128)],
    d_decay_pre_masked: GM[f32, ('BHC', 64, 64)],
    selector_rows: GM[f32, (16, 64)],
    neg_selector_rows: GM[f32, (16, 64)],
    eye: GM[f32, (64, 64)],
    reverse_tril: GM[f32, (64, 64)],
    d_key: GM[f32, ('B', 'H', 'C', 64, 128)],
    d_value: GM[f32, ('B', 'H', 'C', 64, 128)],
    d_beta: GM[f32, ('B', 'H', 'C', 64)],
    d_g: GM[f32, ('BHC', 64)],
    B: i32,
    H: i32,
    C: i32,
    BHC: i32,
):
    key_ub = DBuff(DT.bfloat16, [HALF_L, D], Position.UB)
    value_ub = DBuff(DT.bfloat16, [HALF_L, D], Position.UB)
    beta_ub = DBuff(DT.float, [1, HALF_L], Position.UB)
    d_k_pre_ub = DBuff(DT.float, [HALF_L, D], Position.UB)
    d_k_beta_wu_ub = DBuff(DT.float, [HALF_L, D], Position.UB)
    d_k_beta_pre_ub = DBuff(DT.float, [HALF_L, D], Position.UB)
    d_v_beta_ub = DBuff(DT.float, [HALF_L, D], Position.UB)
    d_k_core_ub = DBuff(DT.float, [HALF_L, D], Position.UB)
    d_beta_ub = DBuff(DT.float, [1, UPPER_TILE_L], Position.UB)
    d_value_out_ub = DBuff(DT.float, [UPPER_TILE_L, D], Position.UB)
    d_key_out_ub = DBuff(DT.float, [UPPER_TILE_L, D], Position.UB)

    bhc_count = B * H * C
    bhc_per_vec = CeilDiv(bhc_count, GetVecNum())
    bhc_begin = Var(bhc_per_vec * GetVecIdx())
    bhc_end = Min(bhc_begin + bhc_per_vec, bhc_count)
    rows_this = Var(UPPER_TILE_L)
    in_buf_cnt = Var(0)
    out_buf_cnt = Var(0)

    with auto_sync():
        for bhc in range(bhc_begin, bhc_end):
            c_idx = Var(bhc // (B * H))
            bh_remainder = Var(bhc % (B * H))
            b_idx = Var(bh_remainder // H)
            h_idx = Var(bh_remainder % H)

            for half_idx in range(2):
                half_begin = Var(half_idx * HALF_L)
                half_end = Var(half_begin + HALF_L)

                key_ub[in_buf_cnt][0:HALF_L, 0:D] <<= key[b_idx, h_idx, c_idx, half_begin:half_end, 0:D]
                value_ub[in_buf_cnt][0:HALF_L, 0:D] <<= value[b_idx, h_idx, c_idx, half_begin:half_end, 0:D]
                beta_ub[in_buf_cnt][0:1, 0:HALF_L] <<= beta[b_idx, h_idx, c_idx, half_begin:half_end]
                d_v_beta_ub[in_buf_cnt][0:HALF_L, 0:D] <<= d_v_beta[b_idx, h_idx, c_idx, half_begin:half_end, 0:D]
                d_k_beta_wu_ub[in_buf_cnt][0:HALF_L, 0:D] <<= d_k_beta_wu[b_idx, h_idx, c_idx, half_begin:half_end, 0:D]
                d_k_beta_pre_ub[in_buf_cnt][0:HALF_L, 0:D] <<= d_k_beta_pre[b_idx, h_idx, c_idx, half_begin:half_end, 0:D]
                d_k_core_ub[in_buf_cnt][0:HALF_L, 0:D] <<= d_k_core[b_idx, h_idx, c_idx, half_begin:half_end, 0:D]
                d_k_pre_ub[in_buf_cnt][0:HALF_L, 0:D] <<= d_k_pre[b_idx, h_idx, c_idx, half_begin:half_end, 0:D]

                for tile_idx in range(2):
                    local_begin = Var(tile_idx * UPPER_TILE_L)
                    local_end = Var(local_begin + UPPER_TILE_L)
                    row_begin = Var(half_begin + local_begin)
                    row_end = Var(row_begin + UPPER_TILE_L)

                    finalize_upper_vf(
                        key_ub[in_buf_cnt][local_begin:local_end, 0:D],
                        value_ub[in_buf_cnt][local_begin:local_end, 0:D],
                        beta_ub[in_buf_cnt][0:1, local_begin:local_end],
                        d_v_beta_ub[in_buf_cnt][local_begin:local_end, 0:D],
                        d_k_beta_wu_ub[in_buf_cnt][local_begin:local_end, 0:D],
                        d_k_beta_pre_ub[in_buf_cnt][local_begin:local_end, 0:D],
                        d_k_core_ub[in_buf_cnt][local_begin:local_end, 0:D],
                        d_k_pre_ub[in_buf_cnt][local_begin:local_end, 0:D],
                        d_value_out_ub[out_buf_cnt],
                        d_beta_ub[out_buf_cnt],
                        d_key_out_ub[out_buf_cnt],
                        rows_this,
                    )

                    d_value[b_idx, h_idx, c_idx, row_begin:row_end, 0:D] <<= d_value_out_ub[out_buf_cnt][0:rows_this, 0:D]
                    d_beta[b_idx, h_idx, c_idx, row_begin:row_end] <<= d_beta_ub[out_buf_cnt][0:1, 0:rows_this]
                    d_key[b_idx, h_idx, c_idx, row_begin:row_end, 0:D] <<= d_key_out_ub[out_buf_cnt][0:rows_this, 0:D]
                    out_buf_cnt += 1
                in_buf_cnt += 1

    reset_cache()

    l1_g_core = Tensor(DT.float, [ACC_TILE_M, L], Position.L1)
    l1_g_wu = Tensor(DT.float, [ACC_TILE_M, L], Position.L1)
    l1_decay_core = DBuff(DT.float, [L, L], Position.L1)
    l1_decay_pre = DBuff(DT.float, [L, L], Position.L1)
    l1_selectors = Tensor(DT.float, [BHC_TILE * BHC_TILE, L], Position.L1)
    l1_neg_selectors = Tensor(DT.float, [BHC_TILE * BHC_TILE, L], Position.L1)
    l1_eye = Tensor(DT.float, [L, L], Position.L1)
    l1_reverse = Tensor(DT.float, [L, L], Position.L1)
    l1_x = Tensor(DT.float, [ACC_TILE_M, L], Position.L1)

    l0c_x = Tensor(DT.float, [ACC_TILE_M, L], Position.L0C)
    l0c_out = Tensor(DT.float, [ACC_TILE_M, L], Position.L0C)

    x_ready = SEvent(Pipe.FIX, Pipe.MTE1, name="finalize_d_g_cube_x_ready")
    tile_count = CeilDiv(BHC, BHC_TILE)
    tile_per_core = CeilDiv(tile_count, GetCubeNum())
    tile_begin = Var(tile_per_core * GetCubeIdx())
    tile_end = Min(tile_begin + tile_per_core, tile_count)
    decay_buf_cnt = Var(0)

    with auto_sync():
        l1_selectors[0:BHC_TILE * BHC_TILE, 0:L] <<= selector_rows[0:BHC_TILE * BHC_TILE, 0:L]
        l1_neg_selectors[0:BHC_TILE * BHC_TILE, 0:L] <<= neg_selector_rows[0:BHC_TILE * BHC_TILE, 0:L]
        l1_eye[0:L, 0:L] <<= eye[0:L, 0:L]
        l1_reverse[0:L, 0:L] <<= reverse_tril[0:L, 0:L]

        for tile_idx in range(tile_begin, tile_end):
            tile_bhc_begin = Var(tile_idx * BHC_TILE)
            rows_cube = Min(BHC_TILE, BHC - tile_bhc_begin)

            l1_g_core[0:rows_cube, 0:L] <<= d_g_core[tile_bhc_begin:tile_bhc_begin + rows_cube, 0:L]
            l1_g_wu[0:rows_cube, 0:L] <<= d_g_wu[tile_bhc_begin:tile_bhc_begin + rows_cube, 0:L]

            matmul(l0c_x, l1_g_core, l1_eye.T, m=rows_cube, n=L, k=L)
            matmul(l0c_x, l1_g_wu, l1_eye.T, m=rows_cube, n=L, k=L, is_init=False)

            for local_t in range(rows_cube):
                bhc_flat = Var(tile_bhc_begin + local_t)
                selector_begin = Var(local_t * BHC_TILE)
                selector_end = Var(selector_begin + rows_cube)
                current_decay_core = l1_decay_core[decay_buf_cnt]
                current_decay_pre = l1_decay_pre[decay_buf_cnt]

                current_decay_core[0:L, 0:L] <<= d_decay_core_masked[bhc_flat, 0:L, 0:L]
                current_decay_pre[0:L, 0:L] <<= d_decay_pre_masked[bhc_flat, 0:L, 0:L]

                matmul(
                    l0c_x,
                    l1_selectors[selector_begin:selector_end, 0:L],
                    current_decay_core,
                    m=rows_cube,
                    n=L,
                    k=L,
                    is_init=False,
                )
                matmul(
                    l0c_x,
                    l1_selectors[selector_begin:selector_end, 0:L],
                    current_decay_pre,
                    m=rows_cube,
                    n=L,
                    k=L,
                    is_init=False,
                )
                matmul(
                    l0c_x,
                    l1_neg_selectors[selector_begin:selector_end, 0:L],
                    current_decay_core.T,
                    m=rows_cube,
                    n=L,
                    k=L,
                    is_init=False,
                )
                matmul(
                    l0c_x,
                    l1_neg_selectors[selector_begin:selector_end, 0:L],
                    current_decay_pre.T,
                    m=rows_cube,
                    n=L,
                    k=L,
                    is_init=False,
                )
                decay_buf_cnt += 1

            l1_x <<= l0c_x
            x_ready.set()
            x_ready.wait()
            matmul(l0c_out, l1_x, l1_reverse.T, m=rows_cube, n=L, k=L)
            d_g[tile_bhc_begin:tile_bhc_begin + rows_cube, 0:L] <<= l0c_out[0:rows_cube, 0:L]
    return d_key, d_value, d_beta, d_g
