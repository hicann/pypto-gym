# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Five DSL stages of the chunked ungated Delta Rule backward."""

from ascriptor.a5 import *  # noqa: F401,F403  # noqa: F401, F403

# ----------------------------------------------------------------------------------------------------
# scan_local_bwd.py
# Delta Rule migrated kernel; typed ABI derived from its input/allocation contract.
#
# Source: projects/a5/delta_rule_bwd/kernels/scan_local_bwd.py.
# M10 edits: public facade, explicit annotations, native range/unroll and local closure.
# Host references and launch logic are maintained separately inside this unit.
# ----------------------------------------------------------------------------------------------------

L = 64
D = 128
HALF_L = L // 2


@vf()
def cast_l_rows_float_to_bf16_nz_vf(src_ub: Tensor, dst_nz_ub: Tensor, rows: Var):
    """Cast [rows, L] float attn rows to the bf16 NZ layout L1 expects."""
    row_float = Reg(DT.float)
    lo_bf16 = Reg(DT.bfloat16)
    row_bf16 = Reg(DT.bfloat16)
    dummy_bf16 = Reg(DT.bfloat16)
    live_bf16 = MaskReg(DT.bfloat16, init_mode=MaskType.LOWHALF)
    BF16_C0 = 16
    for r in range(rows):
        row_float <<= src_ub[r:r + 1, 0:L]
        lo_bf16 <<= row_float.astype(DT.bfloat16)
        deinterleave(row_bf16, dummy_bf16, lo_bf16, dummy_bf16)
        reg_to_ub(dst_nz_ub[r * BF16_C0], row_bf16, rows, mask=live_bf16)


@vf()
def mask_attn_causal_vf(
    score_to_attn_ub: Tensor,
    d_attn_to_d_score_ub: Tensor,
    attn_out_ub: Tensor,
    d_score_bf16_out_ub: Tensor,
    row_begin: Var,
    rows: Var,
):
    """attn = score * causal_keep; d_score = (d_attn * causal_keep) -> bf16.

    The causal keep is GDN's decay_mask at g == 0: for absolute row `abs_r`, keep
    columns 0..abs_r (lower triangle incl. diagonal), zero the rest.
    """
    cols = Reg(DT.int)
    zero = Reg(DT.float)
    score_regs = Reg(DT.float)
    d_attn_regs = Reg(DT.float)
    tmp_regs = Reg(DT.float)
    score_bf16 = Reg(DT.bfloat16)
    dummy_bf16 = Reg(DT.bfloat16)
    score_row = Reg(DT.bfloat16)
    keep_mask = MaskReg(DT.int, init_mode=MaskType.NONE)
    bf16_lowhalf = MaskReg(DT.bfloat16, init_mode=MaskType.LOWHALF)

    zero <<= 0.0
    cols.arange(0)
    for r in range(rows):
        abs_r = Var(row_begin + r)
        score_regs <<= score_to_attn_ub[r:r + 1, 0:L]
        d_attn_regs <<= d_attn_to_d_score_ub[r:r + 1, 0:L]
        compare(keep_mask, cols, abs_r + 1, CompareMode.LT)

        select(tmp_regs, score_regs, zero, mask=keep_mask)
        attn_out_ub[r:r + 1, 0:L] <<= tmp_regs

        select(tmp_regs, d_attn_regs, zero, mask=keep_mask)
        score_bf16 <<= tmp_regs.astype(DT.bfloat16)
        deinterleave(score_row, dummy_bf16, score_bf16, dummy_bf16)
        reg_to_ub(d_score_bf16_out_ub[r:r + 1, 0:L], score_row, mask=bf16_lowhalf)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@kernel()
def scan_local_bwd_kernel(
    query: GM[bf16, ('B', 'H', 'C', 64, 128)],
    key: GM[bf16, ('B', 'H', 'C', 64, 128)],
    grad_output: GM[bf16, ('B', 'H', 'C', 64, 128)],
    v_new_history: GM[bf16, ('B', 'H', 'C', 64, 128)],
    d_score_tmp: GM[bf16, ('B', 'H', 'C', 64, 64)],
    d_v_attn_tmp: GM[f32, ('B', 'H', 'C', 64, 128)],
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

    # M10: the compact NZ payload has HALF_L*L elements; preserve that pitch
    # at the view boundary instead of slicing it out of an unrelated D-wide tile.
    bf16_full_ub = DBuff(DT.bfloat16, [HALF_L, L], Position.UB)
    float_half0_ub = DBuff(DT.float, [HALF_L, D], Position.UB)
    d_score_bf16_ub = DBuff(DT.bfloat16, [HALF_L, L], Position.UB)
    ll0_ub = DBuff(DT.float, [HALF_L, L], Position.UB)
    ll1_ub = DBuff(DT.float, [HALF_L, L], Position.UB)

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
                cvmutex.wait()
                mask_attn_causal_vf(
                    ll0_ub[pipe_slot],
                    ll1_ub[pipe_slot],
                    float_half0_ub[pipe_slot],
                    d_score_bf16_ub[pipe_slot],
                    row_begin_l,
                    rows_l,
                )
                cvmutex.free()

                attn_mutex.lock()
                cast_l_rows_float_to_bf16_nz_vf(float_half0_ub[pipe_slot], bf16_full_ub[pipe_slot], rows_l)
                l1_attn[pipe_slot][row_begin_l:row_end_l, 0:L] <<= bf16_full_ub[pipe_slot][0:rows_l, 0:L].nz()
                attn_mutex.ready()

                d_score_tmp[b_idx, h_idx, c_idx, row_begin_l:row_end_l, 0:L] <<= d_score_bf16_ub[pipe_slot][0:rows_l, 0:L]

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

    return d_score_tmp, d_v_attn_tmp

# ----------------------------------------------------------------------------------------------------
# scan_state_bwd.py
# Delta Rule migrated kernel; typed ABI derived from its input/allocation contract.
#
# Source: projects/a5/delta_rule_bwd/kernels/scan_state_bwd.py.
# M10 edits: public facade, explicit annotations, native range/unroll and local closure.
# Host references and launch logic are maintained separately inside this unit.
# ----------------------------------------------------------------------------------------------------

HALF_D = D // 2
REGS_D = D // 64
BF16_C0 = 16


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
def add_d_rows_vf(a_ub: Tensor, b_ub: Tensor, out_ub: Tensor, rows: Var):
    """out = a + b over [rows, D] (ungated collapse of add_scaled / postprocess_dk)."""
    a_regs = RegList(DT.float, REGS_D)
    b_regs = RegList(DT.float, REGS_D)
    for r in range(rows):
        a_regs <<= a_ub[r:r + 1, 0:D]
        b_regs <<= b_ub[r:r + 1, 0:D]
        a_regs <<= a_regs + b_regs
        out_ub[r:r + 1, 0:D] <<= a_regs
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
def update_d_state_add_cast_nz_vf(seed_ub: Tensor, old_d_state_ub: Tensor, out_ub: Tensor, dst_nz_ub: Tensor):
    """d_state_next = seed + old (no exp(g_last) decay), staged fp32 + bf16 NZ."""
    seed_lo = Reg(DT.float)
    seed_hi = Reg(DT.float)
    old_lo = Reg(DT.float)
    old_hi = Reg(DT.float)
    lo_bf16 = Reg(DT.bfloat16)
    hi_bf16 = Reg(DT.bfloat16)
    row_bf16 = Reg(DT.bfloat16)
    dummy_bf16 = Reg(DT.bfloat16)
    for r in range(HALF_D):
        seed_lo <<= seed_ub[r:r + 1, 0:64]
        seed_hi <<= seed_ub[r:r + 1, 64:D]
        old_lo <<= old_d_state_ub[r:r + 1, 0:64]
        old_hi <<= old_d_state_ub[r:r + 1, 64:D]
        seed_lo <<= seed_lo + old_lo
        seed_hi <<= seed_hi + old_hi
        out_ub[r:r + 1, 0:64] <<= seed_lo
        out_ub[r:r + 1, 64:D] <<= seed_hi
        lo_bf16 <<= seed_lo.astype(DT.bfloat16)
        hi_bf16 <<= seed_hi.astype(DT.bfloat16)
        deinterleave(row_bf16, dummy_bf16, lo_bf16, hi_bf16)
        reg_to_ub(dst_nz_ub[r * BF16_C0], row_bf16, L)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@kernel()
def scan_state_bwd_kernel(
    query: GM[bf16, ('B', 'H', 'C', 64, 128)],
    key: GM[bf16, ('B', 'H', 'C', 64, 128)],
    grad_output: GM[bf16, ('B', 'H', 'C', 64, 128)],
    k_cumdecay: GM[bf16, ('B', 'H', 'C', 64, 128)],
    state_after_history: GM[bf16, ('B', 'H', 'C', 128, 128)],
    grad_final_state: GM[f32, ('B', 'H', 128, 128)],
    d_score_tmp: GM[bf16, ('B', 'H', 'C', 64, 64)],
    v_new_history: GM[bf16, ('B', 'H', 'C', 64, 128)],
    d_v_attn_tmp: GM[f32, ('B', 'H', 'C', 64, 128)],
    d_q_core: GM[f32, ('B', 'H', 'C', 64, 128)],
    d_k_core: GM[f32, ('B', 'H', 'C', 64, 128)],
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
    dkcum_zero_out_ready = SEvent(Pipe.V, Pipe.MTE3, name="dst_dkcum_zero_out_ready")
    dvalue_out_ready = SEvent(Pipe.V, Pipe.MTE3, name="dst_dvalue_out_ready")
    dq_out_ready = SEvent(Pipe.V, Pipe.MTE3, name="dst_dq_out_ready")
    dk_out_ready = SEvent(Pipe.V, Pipe.MTE3, name="dst_dk_out_ready")
    dvat_in_ready = SEvent(Pipe.MTE2, Pipe.V, name="dst_dvat_in_ready")
    dvprime_out_ready = SEvent(Pipe.V, Pipe.MTE3, name="dst_dvprime_out_ready")
    dstate_in_ready = SEvent(Pipe.MTE2, Pipe.V, name="dst_dstate_in_ready")
    dstate_out_ready = SEvent(Pipe.V, Pipe.MTE3, name="dst_dstate_out_ready")

    # M10: explicit single-slot L0 operands keep sequential phases within the
    # eight-flag hardware budget; barriers prove reuse without pre-set families.
    phase_l0a = Tensor(DT.bfloat16, [D, D], Position.L0A)
    phase_l0b = Tensor(DT.bfloat16, [D, D], Position.L0B)
    l1_q = Tensor(DT.bfloat16, [L, D], Position.L1)
    l1_k = Tensor(DT.bfloat16, [L, D], Position.L1)
    l1_d_out = Tensor(DT.bfloat16, [L, D], Position.L1)
    l1_kcd = Tensor(DT.bfloat16, [L, D], Position.L1)
    l1_state = Tensor(DT.bfloat16, [D, D], Position.L1)
    l1_d_state = Tensor(DT.bfloat16, [D, D], Position.L1)
    l1_v_new = Tensor(DT.bfloat16, [L, D], Position.L1)
    l1_d_score = Tensor(DT.bfloat16, [L, L], Position.L1)
    l1_d_v_prime = Tensor(DT.bfloat16, [L, D], Position.L1)

    l0c_ld0 = Tensor(DT.float, [L, D], Position.L0C)
    l0c_ld1 = Tensor(DT.float, [L, D], Position.L0C)
    l0c_dd = Tensor(DT.float, [D, D], Position.L0C)

    bf16_full_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    # M10: d_v_prime has HALF_L rows and a distinct compact-NZ pitch from the
    # HALF_D-row d_state payload. Sharing the full tile misdecodes its panels.
    dvprime_nz_ub = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    bf16_alt_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    score_ub = Tensor(DT.float, [HALF_L, D], Position.UB)
    term_ub = Tensor(DT.float, [HALF_L, D], Position.UB)
    out_ub = Tensor(DT.float, [HALF_L, D], Position.UB)
    tmp_ub = Tensor(DT.float, [HALF_L, D], Position.UB)
    dk_scratch_ub = Tensor(DT.float, [HALF_L, D], Position.UB)
    # Single in-place fp32 d_state carry. The recurrence is serialized by the
    # per-chunk bar_all(), so a 2-slot ping-pong buys no overlap; a DBuff whose
    # slot is selected by a loop-incremented Var view created *before* the
    # rev_c loop drifts after the first update (wrong d_state for C>=3). Plain
    # in-place add is correct (update reads row r before writing row r).
    ub_d_state = Tensor(DT.float, [HALF_D, D], Position.UB)
    ub_d_state_seed = Tensor(DT.float, [HALF_D, D], Position.UB)

    bh_count = B * H
    bh_per_core = CeilDiv(bh_count, GetCubeNum())
    bh_begin = Var(bh_per_core * GetCubeIdx())
    bh_end = Min(bh_begin + bh_per_core, bh_count)

    for bh in range(bh_begin, bh_end):
        b_idx = Var(bh // H)
        h_idx = Var(bh % H)
        row_begin_l = Var(GetSubBlockIdx() * HALF_L)
        row_end_l = Var(row_begin_l + HALF_L)
        rows_l = Var(HALF_L)
        row_begin_d = Var(GetSubBlockIdx() * HALF_D)
        row_end_d = Var(row_begin_d + HALF_D)

        dstate_cur = ub_d_state
        dstate_cur[0:HALF_D, 0:D] <<= grad_final_state[b_idx, h_idx, row_begin_d:row_end_d, 0:D]
        dstate_in_ready.set()
        dstate_in_ready.wait()
        dstate_mutex.lock()
        cast_d_rows_float_to_bf16_nz_vf(dstate_cur[0:HALF_D, 0:D], bf16_full_ub, L)
        dstate_out_ready.set()
        dstate_out_ready.wait()
        l1_d_state[row_begin_d:row_end_d, 0:D] <<= bf16_full_ub[0:L, 0:D].nz()
        bar_all()
        dstate_mutex.ready()
        dstate_mutex.wait()

        for rev_c in range(C):
            c_idx = Var(C - 1 - rev_c)
            c_prev = Var(c_idx - 1)

            with auto_sync():
                l1_q[0:L, 0:D] <<= query[b_idx, h_idx, c_idx, 0:L, 0:D]
                l1_k[0:L, 0:D] <<= key[b_idx, h_idx, c_idx, 0:L, 0:D]
                l1_v_new[0:L, 0:D] <<= v_new_history[b_idx, h_idx, c_idx, 0:L, 0:D]
                l1_d_score[0:L, 0:L] <<= d_score_tmp[b_idx, h_idx, c_idx, 0:L, 0:L]
                bar_all()
                l1_to_l0(phase_l0a, l1_v_new)
                l1_to_l0(phase_l0b, l1_d_state)
                bar_all()
                mmad(l0c_ld1, phase_l0a, phase_l0b, M=L, N=D, K=D, is_init=True)
                bar_all()
                cvmutex.lock()
                term_ub <<= l0c_ld1
                cvmutex.ready()
            cvmutex.wait()
            cvmutex.free()

            if c_idx == 0:
                zero_d_rows_bf16_vf(bf16_alt_ub, rows_l)
                dkcum_zero_out_ready.set()
                dkcum_zero_out_ready.wait()
                d_k_cumdecay[b_idx, h_idx, c_idx, row_begin_l:row_end_l, 0:D] <<= bf16_alt_ub[0:rows_l, 0:D]

                with auto_sync():
                    bar_all()
                    l1_to_l0(phase_l0a, l1_d_score)
                    l1_to_l0(phase_l0b, l1_k.T)
                    bar_all()
                    mmad(l0c_ld0, phase_l0a, phase_l0b, M=L, N=D, K=L, is_init=True)
                    bar_all()
                    d_q_core[b_idx, h_idx, c_idx, 0:L, 0:D] <<= l0c_ld0

                with auto_sync():
                    bar_all()
                    l1_to_l0(phase_l0a, l1_d_score.T)
                    l1_to_l0(phase_l0b, l1_q.T)
                    bar_all()
                    mmad(l0c_ld0, phase_l0a, phase_l0b, M=L, N=D, K=L, is_init=True)
                    bar_all()
                    cvmutex.lock()
                    score_ub <<= l0c_ld0
                    cvmutex.ready()
                cvmutex.wait()
                add_d_rows_vf(score_ub, term_ub, dk_scratch_ub, rows_l)
                dk_out_ready.set()
                dk_out_ready.wait()
                d_k_core[b_idx, h_idx, c_idx, row_begin_l:row_end_l, 0:D] <<= dk_scratch_ub[0:rows_l, 0:D]
                cvmutex.free()

                with auto_sync():
                    bar_all()
                    l1_to_l0(phase_l0a, l1_k)
                    l1_to_l0(phase_l0b, l1_d_state.T)
                    bar_all()
                    mmad(l0c_ld0, phase_l0a, phase_l0b, M=L, N=D, K=D, is_init=True)
                    bar_all()
                    cvmutex.lock()
                    score_ub <<= l0c_ld0
                    cvmutex.ready()
                tmp_ub[0:rows_l, 0:D] <<= d_v_attn_tmp[b_idx, h_idx, c_idx, row_begin_l:row_end_l, 0:D]
                dvat_in_ready.set()
                cvmutex.wait()
                cvmutex.free()
                dvat_in_ready.wait()
                add_cast_d_rows_bf16_vf(score_ub, tmp_ub, bf16_alt_ub, rows_l)
                dvalue_out_ready.set()
                dvalue_out_ready.wait()
                d_value_wu[b_idx, h_idx, c_idx, row_begin_l:row_end_l, 0:D] <<= bf16_alt_ub[0:rows_l, 0:D]
            else:
                with auto_sync():
                    l1_d_out[0:L, 0:D] <<= grad_output[b_idx, h_idx, c_idx, 0:L, 0:D]
                    l1_kcd[0:L, 0:D] <<= k_cumdecay[b_idx, h_idx, c_idx, 0:L, 0:D]
                    l1_state[0:D, 0:D] <<= state_after_history[b_idx, h_idx, c_prev, 0:D, 0:D]
                    bar_all()
                    l1_to_l0(phase_l0a, l1_d_out)
                    l1_to_l0(phase_l0b, l1_state)
                    bar_all()
                    mmad(l0c_ld0, phase_l0a, phase_l0b, M=L, N=D, K=D, is_init=True)
                    bar_all()
                    cvmutex.lock()
                    score_ub <<= l0c_ld0
                    cvmutex.ready()
                with auto_sync():
                    bar_all()
                    l1_to_l0(phase_l0a, l1_q.T)
                    l1_to_l0(phase_l0b, l1_d_out.T)
                    bar_all()
                    mmad(l0c_dd, phase_l0a, phase_l0b, M=D, N=D, K=L, is_init=True)
                    bar_all()
                cvmutex.wait()
                cvmutex.free()

                with auto_sync():
                    bar_all()
                    l1_to_l0(phase_l0a, l1_k)
                    l1_to_l0(phase_l0b, l1_d_state.T)
                    bar_all()
                    mmad(l0c_ld0, phase_l0a, phase_l0b, M=L, N=D, K=D, is_init=True)
                    bar_all()
                    cvmutex.lock()
                    out_ub <<= l0c_ld0
                    cvmutex.ready()
                tmp_ub[0:rows_l, 0:D] <<= d_v_attn_tmp[b_idx, h_idx, c_idx, row_begin_l:row_end_l, 0:D]
                dvat_in_ready.set()
                cvmutex.wait()
                cvmutex.free()
                dvat_in_ready.wait()
                add_negate_cast_d_rows_bf16_vf(out_ub, tmp_ub, bf16_alt_ub, ub_d_state_seed, dvprime_nz_ub, rows_l)
                dvalue_out_ready.set()
                dvalue_out_ready.wait()
                d_value_wu[b_idx, h_idx, c_idx, row_begin_l:row_end_l, 0:D] <<= bf16_alt_ub[0:rows_l, 0:D]

                dvprime_mutex.lock()
                dvprime_out_ready.set()
                dvprime_out_ready.wait()
                l1_d_v_prime[row_begin_l:row_end_l, 0:D] <<= dvprime_nz_ub[0:rows_l, 0:D].nz()
                dvprime_mutex.ready()
                dvprime_mutex.wait()
                bar_all()

                with auto_sync():
                    bar_all()
                    l1_to_l0(phase_l0a, l1_d_v_prime)
                    l1_to_l0(phase_l0b, l1_state)
                    bar_all()
                    mmad(l0c_ld0, phase_l0a, phase_l0b, M=L, N=D, K=D, is_init=True)
                    bar_all()
                    d_k_cumdecay[b_idx, h_idx, c_idx, 0:L, 0:D] <<= l0c_ld0
                    bar_all()
                    l1_to_l0(phase_l0a, l1_kcd.T)
                    l1_to_l0(phase_l0b, l1_d_v_prime.T)
                    bar_all()
                    mmad(l0c_dd, phase_l0a, phase_l0b, M=D, N=D, K=L, is_init=False)
                    bar_all()
                    cv_seed_mutex.lock()
                    l0c_to_ub(ub_d_state_seed, l0c_dd, M=D, N=D, N_dst=D, M_src=D, dual_mode=DualMode.SPLITM, sub_block_id=0)
                    cv_seed_mutex.ready()
                dvprime_mutex.free()

                with auto_sync():
                    bar_all()
                    l1_to_l0(phase_l0a, l1_d_score)
                    l1_to_l0(phase_l0b, l1_k.T)
                    bar_all()
                    mmad(l0c_ld0, phase_l0a, phase_l0b, M=L, N=D, K=L, is_init=True)
                    bar_all()
                    cvmutex.lock()
                    out_ub <<= l0c_ld0
                    cvmutex.ready()
                cvmutex.wait()
                add_d_rows_vf(out_ub, score_ub, out_ub, rows_l)
                dq_out_ready.set()
                dq_out_ready.wait()
                d_q_core[b_idx, h_idx, c_idx, row_begin_l:row_end_l, 0:D] <<= out_ub[0:rows_l, 0:D]
                cvmutex.free()

                with auto_sync():
                    bar_all()
                    l1_to_l0(phase_l0a, l1_d_score.T)
                    l1_to_l0(phase_l0b, l1_q.T)
                    bar_all()
                    mmad(l0c_ld0, phase_l0a, phase_l0b, M=L, N=D, K=L, is_init=True)
                    bar_all()
                    cvmutex.lock()
                    score_ub <<= l0c_ld0
                    cvmutex.ready()
                cvmutex.wait()
                add_d_rows_vf(score_ub, term_ub, dk_scratch_ub, rows_l)
                dk_out_ready.set()
                dk_out_ready.wait()
                d_k_core[b_idx, h_idx, c_idx, row_begin_l:row_end_l, 0:D] <<= dk_scratch_ub[0:rows_l, 0:D]
                cvmutex.free()

            if c_idx != 0:
                dstate_mutex.free()
                cv_seed_mutex.wait()
                dstate_next = ub_d_state
                update_d_state_add_cast_nz_vf(ub_d_state_seed, dstate_cur, dstate_next, bf16_full_ub)
                cv_seed_mutex.free()
                dstate_mutex.lock()
                dstate_out_ready.set()
                dstate_out_ready.wait()
                l1_d_state[row_begin_d:row_end_d, 0:D] <<= bf16_full_ub[0:L, 0:D].nz()
                bar_all()
                dstate_mutex.ready()
                dstate_mutex.wait()

        dstate_mutex.free()

    bar_all()
    return d_q_core, d_k_core, d_value_wu, d_k_cumdecay

# ----------------------------------------------------------------------------------------------------
# wu_bwd.py
# Delta Rule migrated kernel; typed ABI derived from its input/allocation contract.
#
# Source: projects/a5/delta_rule_bwd/kernels/wu_bwd.py.
# M10 edits: public facade, explicit annotations, native range/unroll and local closure.
# Host references and launch logic are maintained separately inside this unit.
# ----------------------------------------------------------------------------------------------------

@vf()
def make_wu_inputs_vf(
    key_ub: Tensor,
    value_ub: Tensor,
    beta_ub: Tensor,
    v_beta_nz_ub: Tensor,
    k_beta_nz_ub: Tensor,
    rows: Var,
):
    """Compute v_beta and k_beta (bf16 NZ) for the matmul pipeline.

    No gate: k_beta = key * beta only (no exp(g_cumsum) factor).
    """
    beta_reg = Reg(DT.float)
    row_lo = Reg(DT.float)
    row_hi = Reg(DT.float)
    lo_bf16 = Reg(DT.bfloat16)
    hi_bf16 = Reg(DT.bfloat16)
    row_bf16 = Reg(DT.bfloat16)
    dummy_bf16 = Reg(DT.bfloat16)

    for r in range(rows):
        beta_reg <<= beta_ub[0:1, r:r + 1].single()

        # v_beta = value * beta -> bf16 NZ
        row_lo <<= value_ub[r:r + 1, 0:64]
        row_lo <<= row_lo * beta_reg
        row_hi <<= value_ub[r:r + 1, 64:D]
        row_hi <<= row_hi * beta_reg
        lo_bf16 <<= row_lo.astype(DT.bfloat16)
        hi_bf16 <<= row_hi.astype(DT.bfloat16)
        deinterleave(row_bf16, dummy_bf16, lo_bf16, hi_bf16)
        reg_to_ub(v_beta_nz_ub[r * BF16_C0], row_bf16, rows)

        # k_beta = key * beta -> bf16 NZ  (no exp(g_cumsum) weight)
        row_lo <<= key_ub[r:r + 1, 0:64]
        row_lo <<= row_lo * beta_reg
        row_hi <<= key_ub[r:r + 1, 64:D]
        row_hi <<= row_hi * beta_reg
        lo_bf16 <<= row_lo.astype(DT.bfloat16)
        hi_bf16 <<= row_hi.astype(DT.bfloat16)
        deinterleave(row_bf16, dummy_bf16, lo_bf16, hi_bf16)
        reg_to_ub(k_beta_nz_ub[r * BF16_C0], row_bf16, rows)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@kernel()
def wu_bwd_kernel(
    key: GM[bf16, ('B', 'H', 'C', 64, 128)],
    value: GM[bf16, ('B', 'H', 'C', 64, 128)],
    beta: GM[f32, ('B', 'H', 'C', 64)],
    wu_attn_bf16: GM[bf16, ('B', 'H', 'C', 64, 64)],
    d_value_wu: GM[bf16, ('B', 'H', 'C', 64, 128)],
    d_k_cumdecay: GM[bf16, ('B', 'H', 'C', 64, 128)],
    d_wu: GM[bf16, ('B', 'H', 'C', 64, 64)],
    d_v_beta: GM[f32, ('B', 'H', 'C', 64, 128)],
    d_k_beta: GM[f32, ('B', 'H', 'C', 64, 128)],
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
    l1_k_beta = DBuff(DT.bfloat16, [L, D], Position.L1)
    l1_w = DBuff(DT.bfloat16, [L, L], Position.L1)
    l1_d_value_wu = DBuff(DT.bfloat16, [L, D], Position.L1)
    l1_d_k_cumdecay = DBuff(DT.bfloat16, [L, D], Position.L1)

    l0c_d_wu = DBuff(DT.float, [L, L], Position.L0C)
    l0c_d_v_beta = DBuff(DT.float, [L, D], Position.L0C)
    l0c_d_k_beta = DBuff(DT.float, [L, D], Position.L0C)

    key_ub = DBuff(DT.bfloat16, [HALF_L, D], Position.UB)
    value_ub = DBuff(DT.bfloat16, [HALF_L, D], Position.UB)
    beta_ub = DBuff(DT.float, [1, HALF_L], Position.UB)
    v_beta_nz_ub = DBuff(DT.bfloat16, [HALF_L, D], Position.UB)
    k_beta_nz_ub = DBuff(DT.bfloat16, [HALF_L, D], Position.UB)
    d_k_beta_ub = DBuff(DT.float, [HALF_L, D], Position.UB)

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
                make_wu_inputs_vf(
                    key_ub[pipe_slot],
                    value_ub[pipe_slot],
                    beta_ub[pipe_slot],
                    v_beta_nz_ub[pipe_slot],
                    k_beta_nz_ub[pipe_slot],
                    rows_this,
                )
                l1_v_beta[pipe_slot][row_begin:row_end, 0:D] <<= v_beta_nz_ub[pipe_slot][0:rows_this, 0:D].nz()
                l1_k_beta[pipe_slot][row_begin:row_end, 0:D] <<= k_beta_nz_ub[pipe_slot][0:rows_this, 0:D].nz()
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

                l1_w[bhc_slot][0:L, 0:L] <<= wu_attn_bf16[b_idx, h_idx, c_idx, 0:L, 0:L]
                l1_d_value_wu[bhc_slot][0:L, 0:D] <<= d_value_wu[b_idx, h_idx, c_idx, 0:L, 0:D]
                l1_d_k_cumdecay[bhc_slot][0:L, 0:D] <<= d_k_cumdecay[b_idx, h_idx, c_idx, 0:L, 0:D]
                vcmutex.wait()
                matmul(l0c_d_wu[bhc_slot], l1_d_value_wu[bhc_slot], l1_v_beta[bhc_slot], m=L, n=L, k=D, splitn=L)
                matmul(l0c_d_wu[bhc_slot], l1_d_k_cumdecay[bhc_slot], l1_k_beta[bhc_slot], m=L, n=L, k=D, splitn=L, is_init=False)
                matmul(l0c_d_v_beta[bhc_slot], l1_w[bhc_slot].T, l1_d_value_wu[bhc_slot].T, m=L, n=D, k=L, splitn=D)
                matmul(l0c_d_k_beta[bhc_slot], l1_w[bhc_slot].T, l1_d_k_cumdecay[bhc_slot].T, m=L, n=D, k=L, splitn=D)
                vcmutex.free()

                cvmutex.lock()
                d_k_beta_ub[bhc_slot] <<= l0c_d_k_beta[bhc_slot]
                cvmutex.ready()

                d_wu[b_idx, h_idx, c_idx, 0:L, 0:L] <<= l0c_d_wu[bhc_slot]
                d_v_beta[b_idx, h_idx, c_idx, 0:L, 0:D] <<= l0c_d_v_beta[bhc_slot]

                cvmutex.wait()
                cvmutex.free()
                d_k_beta[b_idx, h_idx, c_idx, row_begin:row_end, 0:D] <<= d_k_beta_ub[bhc_slot][0:rows_this, 0:D]

    return d_wu, d_v_beta, d_k_beta

# ----------------------------------------------------------------------------------------------------
# inverse_preprocess_bwd.py
# Delta Rule migrated kernel; typed ABI derived from its input/allocation contract.
#
# Source: projects/a5/delta_rule_bwd/kernels/inverse_preprocess_bwd.py.
# M10 edits: public facade, explicit annotations, native range/unroll and local closure.
# Host references and launch logic are maintained separately inside this unit.
# ----------------------------------------------------------------------------------------------------

@vf()
def cast_key_beta_and_w_vf(
    key_bf16_ub: Tensor,
    beta_ub: Tensor,
    key_nz_ub: Tensor,
    k_beta_nz_ub: Tensor,
    rows: Var,
):
    """Pack key rows into NZ for L1, and compute k_beta rows into NZ for L1."""
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
    """Cast [rows, L] float rows to bf16 NZ layout for L1."""
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
def mask_strict_lower_vf(
    d_attn_ub: Tensor,
    d_score_ub: Tensor,
    row_begin: Var,
    rows: Var,
):
    """d_score = -d_attn * strict_lower (constant mask built from abs row index).

    For absolute row `abs_r`, keep columns 0..abs_r-1 (strictly below diagonal),
    zero the diagonal and above.  `compare(mask, cols, abs_r, LT)` achieves this.
    """
    cols = Reg(DT.int)
    zero = Reg(DT.float)
    d_attn_row = Reg(DT.float)
    selected = Reg(DT.float)
    strict_lower_mask = MaskReg(DT.int, init_mode=MaskType.NONE)

    zero <<= 0.0
    cols.arange(0)

    for r in range(rows):
        abs_r = Var(row_begin + r)
        d_attn_row <<= d_attn_ub[r:r + 1, 0:L]
        compare(strict_lower_mask, cols, abs_r, CompareMode.LT)
        select(selected, d_attn_row, zero, mask=strict_lower_mask)
        selected <<= selected * -1.0
        d_score_ub[r:r + 1, 0:L] <<= selected
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@kernel()
def inverse_preprocess_bwd_kernel(
    key: GM[bf16, ('B', 'H', 'C', 64, 128)],
    beta: GM[f32, ('B', 'H', 'C', 64)],
    wu_attn_bf16: GM[bf16, ('B', 'H', 'C', 64, 64)],
    d_wu: GM[bf16, ('B', 'H', 'C', 64, 64)],
    d_k_pre: GM[bf16, ('B', 'H', 'C', 64, 128)],
    d_k_beta_pre: GM[bf16, ('B', 'H', 'C', 64, 128)],
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
    cvmutex = CvMutex(1, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
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
    # M10: key staging and the delayed score staging can publish on V before
    # the first MTE3 wait retires; their ready credits must be distinct.
    score_copy_ready = SEvent(Pipe.V, Pipe.MTE3, name="inverse_score_copy_ready")

    l1_key_buf = TBuff(DT.bfloat16, [L, D], Position.L1)
    l1_k_beta_buf = TBuff(DT.bfloat16, [L, D], Position.L1)
    l1_w_buf = DBuff(DT.bfloat16, [L, L], Position.L1)
    l1_d_wu_buf = DBuff(DT.bfloat16, [L, L], Position.L1)
    l1_tmp_buf = DBuff(DT.bfloat16, [L, L], Position.L1)
    l1_d_score_buf = DBuff(DT.bfloat16, [L, L], Position.L1)

    l0c_tmp_buf = DBuff(DT.float, [L, L], Position.L0C)
    l0c_d_attn_buf = DBuff(DT.float, [L, L], Position.L0C)
    l0c_d_k_buf = DBuff(DT.float, [L, D], Position.L0C)
    l0c_d_k_beta_buf = DBuff(DT.float, [L, D], Position.L0C)

    key_bf16_ub_buf = TBuff(DT.bfloat16, [HALF_L, D], Position.UB)
    beta_ub_buf = TBuff(DT.float, [1, HALF_L], Position.UB)
    key_nz_ub_buf = TBuff(DT.bfloat16, [HALF_L, D], Position.UB)
    k_beta_nz_ub_buf = TBuff(DT.bfloat16, [HALF_L, D], Position.UB)
    d_attn_ub_buf = TBuff(DT.float, [HALF_L, L], Position.UB)
    d_score_ub = Tensor(DT.float, [HALF_L, L], Position.UB)
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
                    d_attn_ub_next = d_attn_ub_buf[pipe_bhc]
                    l0c_tmp_next = l0c_tmp_buf[pipe_bhc]
                    l0c_d_attn_next = l0c_d_attn_buf[pipe_bhc]

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

                    l1_w_next[0:L, 0:L] <<= wu_attn_bf16[b_idx_next, h_idx_next, c_idx_next, 0:L, 0:L]
                    l1_d_wu_next[0:L, 0:L] <<= d_wu[b_idx_next, h_idx_next, c_idx_next, 0:L, 0:L]
                    key_mutex.wait()
                    matmul(l0c_tmp_next, l1_w_next.T, l1_d_wu_next.T, m=L, n=L, k=L, splitn=L)
                    l1_tmp_next <<= l0c_tmp_next
                    fix_to_mte1.set()
                    fix_to_mte1.wait()
                    matmul(l0c_d_attn_next, l1_tmp_next, l1_w_next, m=L, n=L, k=L, splitn=L)

                    cvmutex.lock()
                    d_attn_ub_next <<= l0c_d_attn_next
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
                    d_score_nz_ub = d_score_nz_ub_buf[bhc]
                    l0c_d_k = l0c_d_k_buf[bhc]
                    l0c_d_k_beta = l0c_d_k_beta_buf[bhc]

                    score_mutex.lock()
                    cvmutex.wait()
                    mask_strict_lower_vf(
                        d_attn_ub,
                        d_score_ub,
                        row_begin,
                        rows_this,
                    )
                    cvmutex.free()
                    cast_l_rows_float_to_bf16_nz_barrier_vf(d_score_ub, d_score_nz_ub, rows_this)
                    score_copy_ready.set()
                    score_copy_ready.wait()
                    l1_d_score[row_begin:row_end, 0:L] <<= d_score_nz_ub[0:rows_this, 0:L].nz()
                    score_mutex.ready()

                    score_mutex.wait()
                    matmul(l0c_d_k_beta, l1_d_score, l1_key.T, m=L, n=D, k=L, splitn=D)
                    matmul(l0c_d_k, l1_d_score.T, l1_k_beta.T, m=L, n=D, k=L, splitn=D)
                    score_mutex.free()
                    key_mutex.free()
                    d_k_beta_pre[b_idx, h_idx, c_idx, 0:L, 0:D] <<= l0c_d_k_beta
                    d_k_pre[b_idx, h_idx, c_idx, 0:L, 0:D] <<= l0c_d_k

    return d_k_pre, d_k_beta_pre

# ----------------------------------------------------------------------------------------------------
# finalize_bwd.py
# Delta Rule migrated kernel; typed ABI derived from its input/allocation contract.
#
# Source: projects/a5/delta_rule_bwd/kernels/finalize_bwd.py.
# M10 edits: public facade, explicit annotations, native range/unroll and local closure.
# Host references and launch logic are maintained separately inside this unit.
# ----------------------------------------------------------------------------------------------------

UPPER_TILE_L = L // 4

# Tensor argument indices -> [B, H, C, seq, (D)] axis symbols.
# Symbol 0=B, 1=H, 2=C; None means independent axis.


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

        # d_k_beta equals d_k_beta_wu + d_k_beta_pre.
        d_k_beta_regs <<= d_k_beta_regs + d_k_beta_pre_regs

        # d_beta equals sum(d_v_beta * value + d_k_beta * key, dim=-1).
        tmp_regs <<= d_v_regs * value_regs
        key_regs <<= d_k_beta_regs * key_regs
        tmp_regs <<= tmp_regs + key_regs
        beta_sum <<= tmp_regs.cadd()
        d_beta_ub[0:1, r:r + 1] <<= beta_sum.single_value()

        # d_v equals d_v_beta * beta.
        d_v_regs <<= d_v_regs * beta_reg
        d_value_out_ub[r:r + 1, 0:D] <<= d_v_regs

        # d_k equals d_k_core + d_k_pre + d_k_beta * beta.
        d_k_beta_regs <<= d_k_beta_regs * beta_reg
        d_k_regs <<= d_k_regs + d_k_pre_regs
        d_k_regs <<= d_k_regs + d_k_beta_regs
        d_key_out_ub[r:r + 1, 0:D] <<= d_k_regs


@kernel(mode="vec")
def finalize_bwd_kernel(
    key: GM[bf16, ('B', 'H', 'C', 64, 128)],
    value: GM[bf16, ('B', 'H', 'C', 64, 128)],
    beta: GM[f32, ('B', 'H', 'C', 64)],
    d_k_core: GM[f32, ('B', 'H', 'C', 64, 128)],
    d_v_beta: GM[f32, ('B', 'H', 'C', 64, 128)],
    d_k_beta_wu: GM[f32, ('B', 'H', 'C', 64, 128)],
    d_k_pre: GM[bf16, ('B', 'H', 'C', 64, 128)],
    d_k_beta_pre: GM[bf16, ('B', 'H', 'C', 64, 128)],
    d_key: GM[f32, ('B', 'H', 'C', 64, 128)],
    d_value: GM[f32, ('B', 'H', 'C', 64, 128)],
    d_beta: GM[f32, ('B', 'H', 'C', 64)],
    B: i32,
    H: i32,
    C: i32,
):
    key_ub = DBuff(DT.bfloat16, [HALF_L, D], Position.UB)
    value_ub = DBuff(DT.bfloat16, [HALF_L, D], Position.UB)
    beta_ub = DBuff(DT.float, [1, HALF_L], Position.UB)
    d_k_pre_ub = DBuff(DT.bfloat16, [HALF_L, D], Position.UB)  # bf16 bridge from inverse (half GM->UB traffic)
    d_k_beta_wu_ub = DBuff(DT.float, [HALF_L, D], Position.UB)
    d_k_beta_pre_ub = DBuff(DT.bfloat16, [HALF_L, D], Position.UB)  # bf16 bridge from inverse
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

    return d_key, d_value, d_beta
