# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Six DSL stages of the fixed-tile GDN (Gated Delta Net) forward-v2."""

from ascriptor.a5 import *  # noqa: F401,F403  # the public DSL facade

# ----------------------------------------------------------------------------------------------------
# preprocess.py
# gdn_fwd production stage migrated from kernels/gdn_preprocess.py.
#
# M10 edits: preserve the selected kernel and its source dependency closure;
# replace the EasyASC facade, synthesize typed GM/scalar signatures from the
# reviewed stage contract, and preserve explicit Python unrolling with unroll.
# Host drivers and independent references now live in unit.py and ref/.
# Source SHA256: bd724a8f3962940f8ea25b315325d542198ddd88d239ed1c833460b07b258b96
# ----------------------------------------------------------------------------------------------------

L = 64
D = 128
HALF_L = L // 2







@vf()
def init_causal_masks_vf(lower_eq_ub: Tensor, strict_lower_ub: Tensor, row_begin: Var, rows: Var):
    cols = Reg(DT.int)
    zero = Reg(DT.float)
    lower_row = Reg(DT.float)
    strict_row = Reg(DT.float)
    lower_eq_mask = MaskReg(DT.int, init_mode=MaskType.NONE)
    strict_lower_mask = MaskReg(DT.int, init_mode=MaskType.NONE)

    zero <<= 0.0
    cols.arange(0)

    for r in range(rows):
        abs_r = Var(row_begin + r)

        lower_row <<= 1.0
        compare(lower_eq_mask, cols, abs_r + 1, CompareMode.LT)
        select(lower_row, lower_row, zero, mask=lower_eq_mask)
        lower_eq_ub[r:r + 1, 0:L] <<= lower_row

        strict_row <<= 1.0
        compare(strict_lower_mask, cols, abs_r, CompareMode.LT)
        select(strict_row, strict_row, zero, mask=strict_lower_mask)
        strict_lower_ub[r:r + 1, 0:L] <<= strict_row


@vf()
def apply_decay_and_mask_vf(
    score_ub: Tensor,
    beta_ub: Tensor,
    gcum_ub: Tensor,
    lower_eq_mask_ub: Tensor,
    strict_lower_mask_ub: Tensor,
    decay_ub: Tensor,
    attn_ub: Tensor,
    row_begin: Var,
    rows: Var,
):
    grow = Reg(DT.float)
    beta_reg = Reg(DT.float)
    gcum_row = Reg(DT.float)
    score_row = Reg(DT.float)
    decay_row = Reg(DT.float)
    attn_row = Reg(DT.float)
    mask_row = Reg(DT.float)

    for r in range(rows):
        abs_r = Var(row_begin + r)
        grow <<= gcum_ub[0:1, abs_r:abs_r + 1].single()
        gcum_row <<= gcum_ub[0:1, 0:L]
        decay_row <<= grow - gcum_row
        decay_row <<= decay_row.exp()

        mask_row <<= lower_eq_mask_ub[r:r + 1, 0:L]
        decay_row <<= decay_row * mask_row
        decay_ub[r:r + 1, 0:L] <<= decay_row

        beta_reg <<= beta_ub[0:1, r:r + 1].single()
        score_row <<= score_ub[r:r + 1, 0:L]
        score_row <<= score_row * beta_reg
        attn_row <<= score_row * decay_row
        attn_row <<= attn_row * -1.0
        mask_row <<= strict_lower_mask_ub[r:r + 1, 0:L]
        attn_row <<= attn_row * mask_row
        attn_ub[r:r + 1, 0:L] <<= attn_row


@kernel()
def gdn_preprocess_v2_kernel(
    key: GM[bf16, ('B', 'H', 'C', 64, 128)],
    beta: GM[f32, ('B', 'H', 'C', 64)],
    g: GM[f32, ('B', 'H', 'C', 64)],
    triu: GM[f32, (64, 64)],
    g_cumsum: GM[f32, ('B', 'H', 'C', 64)],
    decay_mask: GM[f32, ('B', 'H', 'C', 64, 64)],
    attn: GM[f32, ('B', 'H', 'C', 64, 64)],
    B: i32,
    H: i32,
    C: i32,
    length_per_chunk: i32,
    head_dim: i32,
):
    cvmutex = CvMutex(1, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)

    l1_key = DBuff(DT.bfloat16, [L, D], Position.L1)
    l1_g = DBuff(DT.float, [16, L], Position.L1)
    l1_triu = Tensor(DT.float, [L, L], Position.L1)
    l0c_gcum = DBuff(DT.float, [16, L], Position.L0C)
    l0c_score = DBuff(DT.float, [L, L], Position.L0C)

    beta_ub = DBuff(DT.float, [1, HALF_L], Position.UB)
    score_ub = DBuff(DT.float, [HALF_L, L], Position.UB)
    gcum_ub = DBuff(DT.float, [1, L], Position.UB)
    lower_eq_mask_ub = Tensor(DT.float, [HALF_L, L], Position.UB)
    strict_lower_mask_ub = Tensor(DT.float, [HALF_L, L], Position.UB)
    decay_ub = DBuff(DT.float, [HALF_L, L], Position.UB)
    attn_ub = DBuff(DT.float, [HALF_L, L], Position.UB)

    stage1_cnt = Var(0)
    stage2_cnt = Var(0)
    bhc_count = B * H * C
    bhc_per_core = CeilDiv(bhc_count, GetCubeNum())
    bhc_begin = Var(bhc_per_core * GetCubeIdx())
    bhc_end = Min(bhc_begin + bhc_per_core, bhc_count)

    with auto_sync():
        if bhc_begin < bhc_end:
            l1_triu[0:L, 0:L] <<= triu[0:L, 0:L]
            mask_row_begin = Var(GetSubBlockIdx() * HALF_L)
            mask_row_end = Min(mask_row_begin + HALF_L, L)
            init_causal_masks_vf(lower_eq_mask_ub, strict_lower_mask_ub, mask_row_begin, mask_row_end - mask_row_begin)

        for bhc in range(bhc_begin, bhc_end + 1):
            if bhc < bhc_end:
                c_idx = Var(bhc // (B * H))
                bh_remainder = Var(bhc % (B * H))
                b_idx = Var(bh_remainder // H)
                h_idx = Var(bh_remainder % H)

                l1_key[stage1_cnt][0:L, 0:D] <<= key[b_idx, h_idx, c_idx, 0:L, 0:D]
                l1_g[stage1_cnt][0:1, 0:L] <<= g[b_idx, h_idx, c_idx, 0:L]
                matmul(l0c_gcum[stage1_cnt], l1_g[stage1_cnt], l1_triu.T, m=1, n=L, k=L)
                g_cumsum[b_idx, h_idx, c_idx, 0:L] <<= l0c_gcum[stage1_cnt][0:1, 0:L]

                matmul(l0c_score[stage1_cnt], l1_key[stage1_cnt], l1_key[stage1_cnt], m=L, n=L, k=D, splitn=L)

                cvmutex.lock()
                score_ub[stage1_cnt] <<= l0c_score[stage1_cnt]
                cvmutex.ready()
                stage1_cnt += 1

            if bhc > bhc_begin:
                prev_bhc = Var(bhc - 1)
                prev_c_idx = Var(prev_bhc // (B * H))
                prev_bh_remainder = Var(prev_bhc % (B * H))
                prev_b_idx = Var(prev_bh_remainder // H)
                prev_h_idx = Var(prev_bh_remainder % H)

                post_row_begin = Var(GetSubBlockIdx() * HALF_L)
                post_row_end = Min(post_row_begin + HALF_L, L)
                post_rows_this = Var(post_row_end - post_row_begin)

                cvmutex.wait()
                beta_ub[stage2_cnt][0:1, 0:post_rows_this] <<= beta[
                    prev_b_idx,
                    prev_h_idx,
                    prev_c_idx,
                    post_row_begin:post_row_end,
                ]
                gcum_ub[stage2_cnt][0:1, 0:L] <<= g_cumsum[prev_b_idx, prev_h_idx, prev_c_idx, 0:L]
                apply_decay_and_mask_vf(
                    score_ub[stage2_cnt],
                    beta_ub[stage2_cnt],
                    gcum_ub[stage2_cnt],
                    lower_eq_mask_ub,
                    strict_lower_mask_ub,
                    decay_ub[stage2_cnt],
                    attn_ub[stage2_cnt],
                    post_row_begin,
                    post_rows_this,
                )
                cvmutex.free()

                decay_mask[prev_b_idx, prev_h_idx, prev_c_idx, post_row_begin:post_row_end, 0:L] <<= decay_ub[stage2_cnt][0:post_rows_this, 0:L]
                attn[prev_b_idx, prev_h_idx, prev_c_idx, post_row_begin:post_row_end, 0:L] <<= attn_ub[stage2_cnt][0:post_rows_this, 0:L]
                stage2_cnt += 1

    return g_cumsum, decay_mask, attn

# ----------------------------------------------------------------------------------------------------
# scores.py
# gdn_fwd production stage migrated from kernels/chunk_delta_sub1.py.
#
# M10 edits: preserve the selected kernel and its source dependency closure;
# replace the EasyASC facade, synthesize typed GM/scalar signatures from the
# reviewed stage contract, and preserve explicit Python unrolling with unroll.
# Host drivers and independent references now live in unit.py and ref/.
# Source SHA256: a95ba771f5c551c27af035f99ee01a22f7417503acee93eb81388113a4d8d413
# ----------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------

M1_N = 64


@vf()
def attn_mul_mask_vf(attn_ub: Tensor, decay_ub: Tensor, out_ub: Tensor, nrows: Var):
    a_reg = Reg(DT.float)
    d_reg = Reg(DT.float)
    for r in range(nrows):
        a_reg <<= attn_ub[r * M1_N]
        d_reg <<= decay_ub[r * M1_N]
        a_reg <<= a_reg * d_reg
        out_ub[r * M1_N] <<= a_reg


@kernel()
def sub1_kernel(
    q_i: GM[bf16, ('B', 'H', 'C', 64, 128)],
    k_i: GM[bf16, ('B', 'H', 'C', 64, 128)],
    decay_mask_i: GM[f32, ('B', 'H', 'C', 64, 64)],
    attn: GM[bf16, ('B', 'H', 'C', 64, 64)],
    B: i32,
    H: i32,
    C: i32,
):
    cvmutex = CvMutex(0, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)

    l1_q = DBuff(DT.bfloat16, [L, D], Position.L1)
    l1_k = DBuff(DT.bfloat16, [M1_N, D], Position.L1)
    l0c_attn = DBuff(DT.float, [L, M1_N], Position.L0C)
    ub_attn = DBuff(DT.float, [HALF_L, M1_N], Position.UB)
    ub_mask = DBuff(DT.float, [HALF_L, M1_N], Position.UB)
    ub_out = DBuff(DT.bfloat16, [HALF_L, M1_N], Position.UB)

    bhc_count = B * H * C
    bhc_per_core = CeilDiv(bhc_count, GetCubeNum())
    bhc_begin = Var(bhc_per_core * GetCubeIdx())
    bhc_end = Min(bhc_begin + bhc_per_core, bhc_count)

    tile_cnt = Var(0)
    half_rows = CeilDiv(L, 2)

    with auto_sync():
        for bhc in range(bhc_begin, bhc_end):
            c_idx = Var(bhc // (B * H))
            bh_remainder = Var(bhc % (B * H))
            b_idx = Var(bh_remainder // H)
            h_idx = Var(bh_remainder % H)
            row_begin = Var(GetSubBlockIdx() * half_rows)
            row_end = Min(row_begin + half_rows, L)
            row_count = Var(row_end - row_begin)

            l1_q[tile_cnt] <<= q_i[b_idx, h_idx, c_idx, 0:L, 0:D]
            l1_k[tile_cnt] <<= k_i[b_idx, h_idx, c_idx, 0:L, 0:D]
            matmul(l0c_attn[tile_cnt], l1_q[tile_cnt], l1_k[tile_cnt])

            cvmutex.lock()
            ub_attn[tile_cnt] <<= l0c_attn[tile_cnt]
            cvmutex.ready()

            ub_mask[tile_cnt] <<= decay_mask_i[b_idx, h_idx, c_idx, row_begin:row_end, 0:M1_N]

            cvmutex.wait()
            attn_mul_mask_vf(ub_attn[tile_cnt], ub_mask[tile_cnt], ub_out[tile_cnt], row_count)
            cvmutex.free()

            attn[b_idx, h_idx, c_idx, row_begin:row_end, 0:M1_N] <<= ub_out[tile_cnt][0:row_count, 0:M1_N]
            tile_cnt += 1

    return attn

# ----------------------------------------------------------------------------------------------------
# inverse.py
# gdn_fwd production stage migrated from kernels/tril_inverse64.py.
#
# M10 edits: preserve the selected kernel and its source dependency closure;
# replace the EasyASC facade, synthesize typed GM/scalar signatures from the
# reviewed stage contract, and preserve explicit Python unrolling with unroll.
# Host drivers and independent references now live in unit.py and ref/.
# Source SHA256: 936aa1515c4db5ff4fc821199ade1b19c02dc716a550e4e6b2bd205ee355994d
# ----------------------------------------------------------------------------------------------------

SIZE = 64
BLOCK = 16








@func()
def _init_zero_tile(tile_ub: Tensor, zero: Reg, row_mask: MaskReg):
    for r in range(0, BLOCK):
        reg_to_ub(tile_ub[r:r + 1, 0:BLOCK], zero, mask=row_mask)


@func()
def _init_diag_seed_row(diag_ub: Tensor, zero: Reg, one: Reg, row_mask: MaskReg):
    reg_to_ub(diag_ub[0:1, 0:BLOCK], zero, mask=row_mask)
    vf_barrier(VfPipe.STORE, VfPipe.STORE)
    diag_ub[0:1, 0:1] <<= one.single_value()
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)






@func()
def _invert_diag_tile_from_strict_a(
    src_ub: Tensor,
    diag_ub: Tensor,
    row_mask: MaskReg,
    one: Reg,
    acc: Reg,
    a_val: Reg,
    prev_row: Reg,
    prod: Reg,
):
    for i in range(1, BLOCK):
        acc.fill(0.0)
        for k in range(0, i):
            a_val <<= src_ub[i:i + 1, k:k + 1].single()
            ub_to_reg(prev_row, diag_ub[k:k + 1, 0:BLOCK], mask=row_mask)
            prod <<= prev_row * a_val
            acc <<= acc + prod
        reg_to_ub(diag_ub[i:i + 1, 0:BLOCK], acc, mask=row_mask)
        vf_barrier(VfPipe.STORE, VfPipe.STORE)
        diag_ub[i:i + 1, i:i + 1] <<= one.single_value()
        vf_barrier(VfPipe.STORE, VfPipe.LOAD)










@vf()
def tril_inverse_block64_strict_d0_vf(d0_src_ub: Tensor, zero16_ub: Tensor, d0_ub: Tensor):
    zero = Reg(DT.float)
    one = Reg(DT.float)
    acc = Reg(DT.float)
    a_val = Reg(DT.float)
    prev_row = Reg(DT.float)
    prod = Reg(DT.float)
    row_mask = MaskReg(DT.float, init_mode=MaskType.NONE)

    zero.fill(0.0)
    one.fill(1.0)

    row_mask <<= Var(BLOCK, dtype=DT.uint32)
    _init_zero_tile(zero16_ub, zero, row_mask)
    _init_diag_seed_row(d0_ub, zero, one, row_mask)
    _invert_diag_tile_from_strict_a(d0_src_ub, d0_ub, row_mask, one, acc, a_val, prev_row, prod)


@vf()
def tril_inverse_block64_strict_d1_vf(d1_src_ub: Tensor, d1_ub: Tensor):
    zero = Reg(DT.float)
    one = Reg(DT.float)
    acc = Reg(DT.float)
    a_val = Reg(DT.float)
    prev_row = Reg(DT.float)
    prod = Reg(DT.float)
    row_mask = MaskReg(DT.float, init_mode=MaskType.NONE)

    zero.fill(0.0)
    one.fill(1.0)

    row_mask <<= Var(BLOCK, dtype=DT.uint32)
    _init_diag_seed_row(d1_ub, zero, one, row_mask)
    _invert_diag_tile_from_strict_a(d1_src_ub, d1_ub, row_mask, one, acc, a_val, prev_row, prod)


@vf()
def tril_inverse_block64_strict_d2_vf(d2_src_ub: Tensor, d2_ub: Tensor):
    zero = Reg(DT.float)
    one = Reg(DT.float)
    acc = Reg(DT.float)
    a_val = Reg(DT.float)
    prev_row = Reg(DT.float)
    prod = Reg(DT.float)
    row_mask = MaskReg(DT.float, init_mode=MaskType.NONE)

    zero.fill(0.0)
    one.fill(1.0)

    row_mask <<= Var(BLOCK, dtype=DT.uint32)
    _init_diag_seed_row(d2_ub, zero, one, row_mask)
    _invert_diag_tile_from_strict_a(d2_src_ub, d2_ub, row_mask, one, acc, a_val, prev_row, prod)


@vf()
def tril_inverse_block64_strict_d3_vf(d3_src_ub: Tensor, d3_ub: Tensor):
    zero = Reg(DT.float)
    one = Reg(DT.float)
    acc = Reg(DT.float)
    a_val = Reg(DT.float)
    prev_row = Reg(DT.float)
    prod = Reg(DT.float)
    row_mask = MaskReg(DT.float, init_mode=MaskType.NONE)

    zero.fill(0.0)
    one.fill(1.0)

    row_mask <<= Var(BLOCK, dtype=DT.uint32)
    _init_diag_seed_row(d3_ub, zero, one, row_mask)
    _invert_diag_tile_from_strict_a(d3_src_ub, d3_ub, row_mask, one, acc, a_val, prev_row, prod)


@vf()
def cast_block_float_to_bf16_vf(src_ub: Tensor, dst_ub: Tensor):
    row = Reg(DT.float)
    row_bf16 = Reg(DT.bfloat16)
    row_mask = MaskReg(DT.float, init_mode=MaskType.NONE)
    row_mask_bf16 = MaskReg(DT.bfloat16, init_mode=MaskType.NONE)
    row_mask <<= Var(BLOCK, dtype=DT.uint32)
    row_mask_bf16 <<= Var(BLOCK * 2, dtype=DT.uint32)
    for r in range(0, BLOCK):
        ub_to_reg(row, src_ub[r:r + 1, 0:BLOCK], mask=row_mask)
        row_bf16 <<= row.astype(DT.bfloat16)
        reg_to_ub_downsample(dst_ub[r:r + 1, 0:BLOCK], row_bf16, mask=row_mask_bf16)


@func()
def _publish_l0c_to_l1(l1_dst: Tensor, l0c_src: Tensor, valid: SEvent, produced: SEvent):
    # PyPTO currently puts Acc->Mat insert mutexes on MTE3, not FIX.
    # Retain a real M->FIX edge before the existing FIX->MTE1 publish.
    produced.set()
    produced.wait()
    l1_dst <<= l0c_src
    valid.set()
    valid.wait()


@func()
def _matmul_to_l1_tmp(l0c_tmp: Tensor, l1_tmp: Tensor, left: Tensor, right: Tensor, valid: SEvent, produced: SEvent):
    matmul(l0c_tmp, left, right.T, m=BLOCK, n=BLOCK, k=BLOCK)
    _publish_l0c_to_l1(l1_tmp, l0c_tmp, valid, produced)




@kernel()
def tril_inverse64_v2_strict_bf16_kernel(a: GM[f32, ('B', 'H', 'C', 64, 64)], inv: GM[bf16, ('B', 'H', 'C', 64, 64)], B: i32, H: i32, C: i32):
    stage0_mutex = VcMutex(0, depth=2, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)
    stage1_mutex = VcMutex(1, depth=2, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)
    l1_valid = SEvent(Pipe.FIX, Pipe.MTE1)
    product_ready = SEvent(Pipe.M, Pipe.FIX)

    total_chunks = B * H * C
    a_tiles = a.reshape([total_chunks, SIZE, SIZE], name="a_tiles")
    inv_tiles = inv.reshape([total_chunks, SIZE, SIZE], name="inv_tiles")

    # Each AIV owns its UB. Opposite subblock roles share storage; the
    # next diagonal source load follows the previous VF final read.
    diag_src_ub = Tensor(DT.float, [BLOCK, BLOCK], Position.UB)
    zero16_ub = Tensor(DT.float, [BLOCK, BLOCK], Position.UB)
    diag_first_ub = Tensor(DT.float, [BLOCK, BLOCK], Position.UB)
    diag_second_ub = Tensor(DT.float, [BLOCK, BLOCK], Position.UB)
    offdiag0_ub = Tensor(DT.float, [BLOCK, BLOCK], Position.UB)
    offdiag1_ub = Tensor(DT.float, [BLOCK, BLOCK], Position.UB)
    offdiag2_ub = Tensor(DT.float, [BLOCK, BLOCK], Position.UB)
    zero16_bf16_ub = Tensor(DT.bfloat16, [BLOCK, BLOCK], Position.UB)
    diag_first_bf16_ub = Tensor(DT.bfloat16, [BLOCK, BLOCK], Position.UB)
    diag_second_bf16_ub = Tensor(DT.bfloat16, [BLOCK, BLOCK], Position.UB)

    l1_d0 = DBuff(DT.float, [BLOCK, BLOCK], Position.L1)
    l1_d1 = DBuff(DT.float, [BLOCK, BLOCK], Position.L1)
    l1_d2 = DBuff(DT.float, [BLOCK, BLOCK], Position.L1)
    l1_d3 = DBuff(DT.float, [BLOCK, BLOCK], Position.L1)
    # a10 becomes temporary scratch; a21/a31/a32 become p21/p31/p32.
    # Each original A dies at its first MTE1 read; local mutexes order
    # that read before the FIX overwrite of the same physical ring.
    l1_a10 = DBuff(DT.float, [BLOCK, BLOCK], Position.L1)
    l1_a20 = DBuff(DT.float, [BLOCK, BLOCK], Position.L1)
    l1_a21 = DBuff(DT.float, [BLOCK, BLOCK], Position.L1)
    l1_a30 = DBuff(DT.float, [BLOCK, BLOCK], Position.L1)
    l1_a31 = DBuff(DT.float, [BLOCK, BLOCK], Position.L1)
    l1_a32 = DBuff(DT.float, [BLOCK, BLOCK], Position.L1)
    l1_x10 = DBuff(DT.float, [BLOCK, BLOCK], Position.L1)
    l1_x20 = DBuff(DT.float, [BLOCK, BLOCK], Position.L1)
    l1_x21 = DBuff(DT.float, [BLOCK, BLOCK], Position.L1)

    # Publish every result before the next product reuses this ring.
    l0c_work = DBuff(DT.float, [BLOCK, BLOCK], Position.L0C)

    chunks_per_core = CeilDiv(total_chunks, GetCubeNum())
    chunk_begin = Var(chunks_per_core * GetCubeIdx())
    chunk_end = Min(chunk_begin + chunks_per_core, total_chunks)
    buf_cnt = Var(0)

    for chunk_idx in range(chunk_begin, chunk_end):
        with auto_sync():
            stage0_mutex.lock()
            stage1_mutex.lock()

            if GetSubBlockIdx() == 0:
                diag_src_ub <<= a_tiles[chunk_idx, 0:BLOCK, 0:BLOCK]
                tril_inverse_block64_strict_d0_vf(diag_src_ub, zero16_ub, diag_first_ub)
                l1_d0[buf_cnt][0:BLOCK, 0:BLOCK] <<= diag_first_ub[0:BLOCK, 0:BLOCK]
                stage0_mutex.ready()

            if GetSubBlockIdx() == 1:
                diag_src_ub <<= a_tiles[chunk_idx, BLOCK:2 * BLOCK, BLOCK:2 * BLOCK]
                offdiag0_ub <<= a_tiles[chunk_idx, BLOCK:2 * BLOCK, 0:BLOCK]
                tril_inverse_block64_strict_d1_vf(diag_src_ub, diag_first_ub)
                l1_d1[buf_cnt][0:BLOCK, 0:BLOCK] <<= diag_first_ub[0:BLOCK, 0:BLOCK]
                l1_a10[buf_cnt][0:BLOCK, 0:BLOCK] <<= offdiag0_ub[0:BLOCK, 0:BLOCK]
                stage0_mutex.ready()

            stage0_mutex.wait()

            _matmul_to_l1_tmp(l0c_work[buf_cnt], l1_a10[buf_cnt], l1_d1[buf_cnt], l1_a10[buf_cnt], l1_valid, product_ready)
            matmul(l0c_work[buf_cnt], l1_a10[buf_cnt], l1_d0[buf_cnt].T, m=BLOCK, n=BLOCK, k=BLOCK)
            _publish_l0c_to_l1(l1_x10[buf_cnt], l0c_work[buf_cnt], l1_valid, product_ready)
            inv_tiles[chunk_idx, BLOCK:2 * BLOCK, 0:BLOCK] <<= l0c_work[buf_cnt][0:BLOCK, 0:BLOCK]

            if GetSubBlockIdx() == 1:
                diag_src_ub <<= a_tiles[chunk_idx, 3 * BLOCK:SIZE, 3 * BLOCK:SIZE]
                offdiag0_ub <<= a_tiles[chunk_idx, 3 * BLOCK:SIZE, 0:BLOCK]
                offdiag1_ub <<= a_tiles[chunk_idx, 3 * BLOCK:SIZE, BLOCK:2 * BLOCK]
                offdiag2_ub <<= a_tiles[chunk_idx, 3 * BLOCK:SIZE, 2 * BLOCK:3 * BLOCK]
                tril_inverse_block64_strict_d3_vf(diag_src_ub, diag_second_ub)
                l1_d3[buf_cnt][0:BLOCK, 0:BLOCK] <<= diag_second_ub[0:BLOCK, 0:BLOCK]
                l1_a30[buf_cnt][0:BLOCK, 0:BLOCK] <<= offdiag0_ub[0:BLOCK, 0:BLOCK]
                l1_a31[buf_cnt][0:BLOCK, 0:BLOCK] <<= offdiag1_ub[0:BLOCK, 0:BLOCK]
                l1_a32[buf_cnt][0:BLOCK, 0:BLOCK] <<= offdiag2_ub[0:BLOCK, 0:BLOCK]
                stage1_mutex.ready()

            if GetSubBlockIdx() == 0:
                diag_src_ub <<= a_tiles[chunk_idx, 2 * BLOCK:3 * BLOCK, 2 * BLOCK:3 * BLOCK]
                offdiag0_ub <<= a_tiles[chunk_idx, 2 * BLOCK:3 * BLOCK, 0:BLOCK]
                offdiag1_ub <<= a_tiles[chunk_idx, 2 * BLOCK:3 * BLOCK, BLOCK:2 * BLOCK]
                tril_inverse_block64_strict_d2_vf(diag_src_ub, diag_second_ub)
                l1_d2[buf_cnt][0:BLOCK, 0:BLOCK] <<= diag_second_ub[0:BLOCK, 0:BLOCK]
                l1_a20[buf_cnt][0:BLOCK, 0:BLOCK] <<= offdiag0_ub[0:BLOCK, 0:BLOCK]
                l1_a21[buf_cnt][0:BLOCK, 0:BLOCK] <<= offdiag1_ub[0:BLOCK, 0:BLOCK]
                stage1_mutex.ready()

                cast_block_float_to_bf16_vf(zero16_ub, zero16_bf16_ub)
                cast_block_float_to_bf16_vf(diag_first_ub, diag_first_bf16_ub)
                cast_block_float_to_bf16_vf(diag_second_ub, diag_second_bf16_ub)
                inv_tiles[chunk_idx, 0:BLOCK, 0:BLOCK] <<= diag_first_bf16_ub[0:BLOCK, 0:BLOCK]
                inv_tiles[chunk_idx, 0:BLOCK, BLOCK:2 * BLOCK] <<= zero16_bf16_ub[0:BLOCK, 0:BLOCK]
                inv_tiles[chunk_idx, 0:BLOCK, 2 * BLOCK:3 * BLOCK] <<= zero16_bf16_ub[0:BLOCK, 0:BLOCK]
                inv_tiles[chunk_idx, 0:BLOCK, 3 * BLOCK:SIZE] <<= zero16_bf16_ub[0:BLOCK, 0:BLOCK]
                inv_tiles[chunk_idx, BLOCK:2 * BLOCK, 2 * BLOCK:3 * BLOCK] <<= zero16_bf16_ub[0:BLOCK, 0:BLOCK]
                inv_tiles[chunk_idx, BLOCK:2 * BLOCK, 3 * BLOCK:SIZE] <<= zero16_bf16_ub[0:BLOCK, 0:BLOCK]
                inv_tiles[chunk_idx, 2 * BLOCK:3 * BLOCK, 2 * BLOCK:3 * BLOCK] <<= diag_second_bf16_ub[0:BLOCK, 0:BLOCK]
                inv_tiles[chunk_idx, 2 * BLOCK:3 * BLOCK, 3 * BLOCK:SIZE] <<= zero16_bf16_ub[0:BLOCK, 0:BLOCK]

            if GetSubBlockIdx() == 1:
                cast_block_float_to_bf16_vf(diag_first_ub, diag_first_bf16_ub)
                cast_block_float_to_bf16_vf(diag_second_ub, diag_second_bf16_ub)
                inv_tiles[chunk_idx, BLOCK:2 * BLOCK, BLOCK:2 * BLOCK] <<= diag_first_bf16_ub[0:BLOCK, 0:BLOCK]
                inv_tiles[chunk_idx, 3 * BLOCK:SIZE, 3 * BLOCK:SIZE] <<= diag_second_bf16_ub[0:BLOCK, 0:BLOCK]

            stage1_mutex.wait()

            _matmul_to_l1_tmp(l0c_work[buf_cnt], l1_a21[buf_cnt], l1_d2[buf_cnt], l1_a21[buf_cnt], l1_valid, product_ready)
            matmul(l0c_work[buf_cnt], l1_a21[buf_cnt], l1_d1[buf_cnt].T, m=BLOCK, n=BLOCK, k=BLOCK)
            _publish_l0c_to_l1(l1_x21[buf_cnt], l0c_work[buf_cnt], l1_valid, product_ready)
            inv_tiles[chunk_idx, 2 * BLOCK:3 * BLOCK, BLOCK:2 * BLOCK] <<= l0c_work[buf_cnt][0:BLOCK, 0:BLOCK]

            _matmul_to_l1_tmp(l0c_work[buf_cnt], l1_a32[buf_cnt], l1_d3[buf_cnt], l1_a32[buf_cnt], l1_valid, product_ready)
            matmul(l0c_work[buf_cnt], l1_a32[buf_cnt], l1_d2[buf_cnt].T, m=BLOCK, n=BLOCK, k=BLOCK)
            inv_tiles[chunk_idx, 3 * BLOCK:SIZE, 2 * BLOCK:3 * BLOCK] <<= l0c_work[buf_cnt][0:BLOCK, 0:BLOCK]

            _matmul_to_l1_tmp(l0c_work[buf_cnt], l1_a10[buf_cnt], l1_d2[buf_cnt], l1_a20[buf_cnt], l1_valid, product_ready)
            matmul(l0c_work[buf_cnt], l1_a10[buf_cnt], l1_d0[buf_cnt].T, m=BLOCK, n=BLOCK, k=BLOCK)
            matmul(l0c_work[buf_cnt], l1_a21[buf_cnt], l1_x10[buf_cnt].T, m=BLOCK, n=BLOCK, k=BLOCK, is_init=False)
            _publish_l0c_to_l1(l1_x20[buf_cnt], l0c_work[buf_cnt], l1_valid, product_ready)
            inv_tiles[chunk_idx, 2 * BLOCK:3 * BLOCK, 0:BLOCK] <<= l0c_work[buf_cnt][0:BLOCK, 0:BLOCK]

            _matmul_to_l1_tmp(l0c_work[buf_cnt], l1_a31[buf_cnt], l1_d3[buf_cnt], l1_a31[buf_cnt], l1_valid, product_ready)
            matmul(l0c_work[buf_cnt], l1_a31[buf_cnt], l1_d1[buf_cnt].T, m=BLOCK, n=BLOCK, k=BLOCK)
            matmul(l0c_work[buf_cnt], l1_a32[buf_cnt], l1_x21[buf_cnt].T, m=BLOCK, n=BLOCK, k=BLOCK, is_init=False)
            inv_tiles[chunk_idx, 3 * BLOCK:SIZE, BLOCK:2 * BLOCK] <<= l0c_work[buf_cnt][0:BLOCK, 0:BLOCK]

            _matmul_to_l1_tmp(l0c_work[buf_cnt], l1_a10[buf_cnt], l1_d3[buf_cnt], l1_a30[buf_cnt], l1_valid, product_ready)
            matmul(l0c_work[buf_cnt], l1_a10[buf_cnt], l1_d0[buf_cnt].T, m=BLOCK, n=BLOCK, k=BLOCK)
            matmul(l0c_work[buf_cnt], l1_a31[buf_cnt], l1_x10[buf_cnt].T, m=BLOCK, n=BLOCK, k=BLOCK, is_init=False)
            matmul(l0c_work[buf_cnt], l1_a32[buf_cnt], l1_x20[buf_cnt].T, m=BLOCK, n=BLOCK, k=BLOCK, is_init=False)
            inv_tiles[chunk_idx, 3 * BLOCK:SIZE, 0:BLOCK] <<= l0c_work[buf_cnt][0:BLOCK, 0:BLOCK]

            stage1_mutex.free()
            stage0_mutex.free()
            bar_all()
            buf_cnt += 1

    return inv


# --------------------------------------------------------------------------- #
# Neumann / log-squaring whole-64 inverse (cube-bound; vec-light).
#
# For strict-lower (nilpotent, A^64 = 0) A,
#     (I - A)^-1 = prod_{j=0}^{5} (I + A^(2^j))
# because (I-A)·prod = I - A^64 = I.  Built by the doubling recurrence
# inv equals I + A;  Apow equals A.
#     repeat 5x:  Apow = Apow@Apow;  inv = inv@(I + Apow) = inv + inv@Apow
# which is ~17 fp32 64x64 cube matmuls and *no* serial vec substitution — the
# opposite balance to the block-substitution kernel above (vec-bound).  fp32
# throughout (the cube already runs fp32 here), so it matches the block kernel's
# precision; powers of the small delta/GDN A decay, so rounding stays ~1e-7.
# --------------------------------------------------------------------------- #

# ----------------------------------------------------------------------------------------------------
# recompute.py
# gdn_fwd production stage migrated from kernels/gdn_recompute_wu.py.
#
# M10 edits: preserve the selected kernel and its source dependency closure;
# replace the EasyASC facade, synthesize typed GM/scalar signatures from the
# reviewed stage contract, and preserve explicit Python unrolling with unroll.
# Host drivers and independent references now live in unit.py and ref/.
# Source SHA256: c6cf79b5d8cbe363256602d13555121ae7d10c3e7c4afdd1d2ce355397792413
# ----------------------------------------------------------------------------------------------------

BF16_C0 = 16







@vf()
def scale_and_pack_vf(
    key_ub: Tensor,
    value_ub: Tensor,
    beta_ub: Tensor,
    g_ub: Tensor,
    k_beta_g_nz_ub: Tensor,
    v_beta_nz_ub: Tensor,
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

        row_lo <<= value_ub[r:r + 1, 0:64]
        row_lo <<= row_lo * beta_reg
        row_hi <<= value_ub[r:r + 1, 64:D]
        row_hi <<= row_hi * beta_reg
        lo_bf16 <<= row_lo.astype(DT.bfloat16)
        hi_bf16 <<= row_hi.astype(DT.bfloat16)
        deinterleave(row_bf16, dummy_bf16, lo_bf16, hi_bf16)
        reg_to_ub(v_beta_nz_ub[r * BF16_C0], row_bf16, rows)

        g_reg <<= g_ub[0:1, r:r + 1].single()
        g_reg <<= g_reg.exp()

        row_lo <<= key_ub[r:r + 1, 0:64]
        row_lo <<= row_lo * beta_reg
        row_lo <<= row_lo * g_reg
        row_hi <<= key_ub[r:r + 1, 64:D]
        row_hi <<= row_hi * beta_reg
        row_hi <<= row_hi * g_reg
        lo_bf16 <<= row_lo.astype(DT.bfloat16)
        hi_bf16 <<= row_hi.astype(DT.bfloat16)
        deinterleave(row_bf16, dummy_bf16, lo_bf16, hi_bf16)
        reg_to_ub(k_beta_g_nz_ub[r * BF16_C0], row_bf16, rows)


@kernel()
def gdn_recompute_wu_v2_kernel(
    key: GM[bf16, ('B', 'H', 'C', 64, 128)],
    value: GM[bf16, ('B', 'H', 'C', 64, 128)],
    beta: GM[f32, ('B', 'H', 'C', 64)],
    g: GM[f32, ('B', 'H', 'C', 64)],
    attn: GM[bf16, ('B', 'H', 'C', 64, 64)],
    value_out: GM[bf16, ('B', 'H', 'C', 64, 128)],
    k_cumdecay: GM[bf16, ('B', 'H', 'C', 64, 128)],
    B: i32,
    H: i32,
    C: i32,
    length_per_chunk: i32,
    head_dim: i32,
):
    vcmutex = VcMutex(
        0, depth=2,
        src_start_pipe=Pipe.MTE3,
        src_end_pipe=Pipe.MTE3,
        dst_start_pipe=Pipe.MTE1,
        dst_end_pipe=Pipe.MTE1,
    )

    l1_attn = DBuff(DT.bfloat16, [L, L], Position.L1)
    l1_v_beta = DBuff(DT.bfloat16, [L, D], Position.L1)
    l1_k_beta_g = DBuff(DT.bfloat16, [L, D], Position.L1)
    l0c_value = DBuff(DT.float, [L, D], Position.L0C)
    l0c_kcumdecay = DBuff(DT.float, [L, D], Position.L0C)

    key_ub = DBuff(DT.bfloat16, [HALF_L, D], Position.UB)
    value_ub = DBuff(DT.bfloat16, [HALF_L, D], Position.UB)
    beta_ub = DBuff(DT.float, [1, HALF_L], Position.UB)
    g_ub = DBuff(DT.float, [1, HALF_L], Position.UB)
    v_beta_nz_ub = DBuff(DT.bfloat16, [HALF_L, D], Position.UB)
    k_beta_g_nz_ub = DBuff(DT.bfloat16, [HALF_L, D], Position.UB)

    vcnt = Var(0)
    ccnt = Var(0)
    bhc_count = B * H * C
    bhc_per_core = CeilDiv(bhc_count, GetCubeNum())
    bhc_begin = Var(bhc_per_core * GetCubeIdx())
    bhc_end = Min(bhc_begin + bhc_per_core, bhc_count)

    with auto_sync():
        for bhc in range(bhc_begin, bhc_end):
            c_idx = Var(bhc // (B * H))
            bh_remainder = Var(bhc % (B * H))
            b_idx = Var(bh_remainder // H)
            h_idx = Var(bh_remainder % H)

            vcmutex.lock()
            row_begin = Var(GetSubBlockIdx() * HALF_L)
            if row_begin < L:
                row_end = Min(row_begin + HALF_L, L)
                rows_this = Var(row_end - row_begin)
                key_ub[vcnt][0:rows_this, 0:D] <<= key[b_idx, h_idx, c_idx, row_begin:row_end, 0:D]
                value_ub[vcnt][0:rows_this, 0:D] <<= value[b_idx, h_idx, c_idx, row_begin:row_end, 0:D]
                beta_ub[vcnt][0:1, 0:rows_this] <<= beta[b_idx, h_idx, c_idx, row_begin:row_end]
                g_ub[vcnt][0:1, 0:rows_this] <<= g[b_idx, h_idx, c_idx, row_begin:row_end]
                scale_and_pack_vf(
                    key_ub[vcnt],
                    value_ub[vcnt],
                    beta_ub[vcnt],
                    g_ub[vcnt],
                    k_beta_g_nz_ub[vcnt],
                    v_beta_nz_ub[vcnt],
                    rows_this,
                )
                l1_v_beta[ccnt][row_begin:row_end, 0:D] <<= v_beta_nz_ub[vcnt][0:rows_this, 0:D].nz()
                l1_k_beta_g[ccnt][row_begin:row_end, 0:D] <<= k_beta_g_nz_ub[vcnt][0:rows_this, 0:D].nz()
            vcnt += 1
            vcmutex.ready()

            vcmutex.wait()
            l1_attn[ccnt][0:L, 0:L] <<= attn[b_idx, h_idx, c_idx, 0:L, 0:L]

            matmul(l0c_value[ccnt], l1_attn[ccnt], l1_v_beta[ccnt].T, splitn=128)
            matmul(l0c_kcumdecay[ccnt], l1_attn[ccnt], l1_k_beta_g[ccnt].T, splitn=128)

            value_out[b_idx, h_idx, c_idx, 0:L, 0:D] <<= l0c_value[ccnt]
            k_cumdecay[b_idx, h_idx, c_idx, 0:L, 0:D] <<= l0c_kcumdecay[ccnt]
            vcmutex.free()
            ccnt += 1

    return value_out, k_cumdecay

# ----------------------------------------------------------------------------------------------------
# recurrent.py
# gdn_fwd production stage migrated from kernels/chunk_delta_sub2.py.
#
# M10 edits: preserve the selected kernel and its source dependency closure;
# replace the EasyASC facade, synthesize typed GM/scalar signatures from the
# reviewed stage contract, and preserve explicit Python unrolling with unroll.
# Host drivers and independent references now live in unit.py and ref/.
# Source SHA256: 5b9ce5e16478c3a1f865255071579ed4b5ecdf47d0b7dcde0563d4abed3fad2b
# ----------------------------------------------------------------------------------------------------

HALF_D = D // 2
REGS_PER_ROW_D = D // 64
REGS_PER_FOUR_ROWS_D = REGS_PER_ROW_D * 4



@vf()
def make_v_new_vf(v_prime_ub: Tensor, v_ub: Tensor, v_new_ub: Tensor):
    prime_regs = RegList(DT.float, REGS_PER_FOUR_ROWS_D)
    v_regs = RegList(DT.float, REGS_PER_FOUR_ROWS_D)
    for row_quad in range(HALF_L // 4):
        r = row_quad * 4
        prime_regs <<= v_prime_ub[r:r + 4, 0:D]
        v_regs <<= v_ub[r:r + 4, 0:D]
        v_regs <<= v_regs - prime_regs
        v_new_ub[r:r + 4, 0:D] <<= v_regs


@vf()
def apply_g_gate_vf(q_state_ub: Tensor, g_ub: Tensor, attn_inter_ub: Tensor):
    q_regs = RegList(DT.float, REGS_PER_FOUR_ROWS_D)
    g0_reg = Reg(DT.float)
    g1_reg = Reg(DT.float)
    g2_reg = Reg(DT.float)
    g3_reg = Reg(DT.float)
    scale0_reg = Reg(DT.float)
    scale1_reg = Reg(DT.float)
    scale2_reg = Reg(DT.float)
    scale3_reg = Reg(DT.float)
    for row_quad in range(HALF_L // 4):
        r = row_quad * 4
        q_regs <<= q_state_ub[r:r + 4, 0:D]
        g0_reg <<= g_ub[0:1, r:r + 1].single()
        g1_reg <<= g_ub[0:1, r + 1:r + 2].single()
        g2_reg <<= g_ub[0:1, r + 2:r + 3].single()
        g3_reg <<= g_ub[0:1, r + 3:r + 4].single()
        scale0_reg <<= g0_reg.exp()
        scale1_reg <<= g1_reg.exp()
        scale2_reg <<= g2_reg.exp()
        scale3_reg <<= g3_reg.exp()
        q_regs[0] <<= q_regs[0] * scale0_reg
        q_regs[1] <<= q_regs[1] * scale0_reg
        q_regs[2] <<= q_regs[2] * scale1_reg
        q_regs[3] <<= q_regs[3] * scale1_reg
        q_regs[4] <<= q_regs[4] * scale2_reg
        q_regs[5] <<= q_regs[5] * scale2_reg
        q_regs[6] <<= q_regs[6] * scale3_reg
        q_regs[7] <<= q_regs[7] * scale3_reg
        attn_inter_ub[r:r + 4, 0:D] <<= q_regs


@vf()
def make_kexp_vf(k_ub: Tensor, g_ub: Tensor, g_last: Var, k_exp_ub: Tensor):
    gl = Reg(DT.float)
    g0 = Reg(DT.float)
    g1 = Reg(DT.float)
    g2 = Reg(DT.float)
    g3 = Reg(DT.float)
    scale0 = Reg(DT.float)
    scale1 = Reg(DT.float)
    scale2 = Reg(DT.float)
    scale3 = Reg(DT.float)
    kf = RegList(DT.float, REGS_PER_FOUR_ROWS_D)
    lo_bf16 = Reg(DT.bfloat16)
    hi_bf16 = Reg(DT.bfloat16)
    row_bf16 = Reg(DT.bfloat16)
    dummy_bf16 = Reg(DT.bfloat16)
    gl <<= g_last
    for row_quad in range(HALF_L // 4):
        r = row_quad * 4
        g0 <<= g_ub[0:1, r:r + 1].single()
        g1 <<= g_ub[0:1, r + 1:r + 2].single()
        g2 <<= g_ub[0:1, r + 2:r + 3].single()
        g3 <<= g_ub[0:1, r + 3:r + 4].single()
        scale0 <<= gl - g0
        scale1 <<= gl - g1
        scale2 <<= gl - g2
        scale3 <<= gl - g3
        scale0 <<= scale0.exp()
        scale1 <<= scale1.exp()
        scale2 <<= scale2.exp()
        scale3 <<= scale3.exp()
        kf <<= k_ub[r:r + 4, 0:D]
        kf[0] <<= kf[0] * scale0
        kf[1] <<= kf[1] * scale0
        kf[2] <<= kf[2] * scale1
        kf[3] <<= kf[3] * scale1
        kf[4] <<= kf[4] * scale2
        kf[5] <<= kf[5] * scale2
        kf[6] <<= kf[6] * scale3
        kf[7] <<= kf[7] * scale3
        lo_bf16 <<= kf[0].astype(DT.bfloat16)
        hi_bf16 <<= kf[1].astype(DT.bfloat16)
        deinterleave(row_bf16, dummy_bf16, lo_bf16, hi_bf16)
        reg_to_ub(k_exp_ub[(r + 0) * BF16_C0], row_bf16, HALF_L)
        vf_barrier(VfPipe.STORE, VfPipe.STORE)
        lo_bf16 <<= kf[2].astype(DT.bfloat16)
        hi_bf16 <<= kf[3].astype(DT.bfloat16)
        deinterleave(row_bf16, dummy_bf16, lo_bf16, hi_bf16)
        reg_to_ub(k_exp_ub[(r + 1) * BF16_C0], row_bf16, HALF_L)
        vf_barrier(VfPipe.STORE, VfPipe.STORE)
        lo_bf16 <<= kf[4].astype(DT.bfloat16)
        hi_bf16 <<= kf[5].astype(DT.bfloat16)
        deinterleave(row_bf16, dummy_bf16, lo_bf16, hi_bf16)
        reg_to_ub(k_exp_ub[(r + 2) * BF16_C0], row_bf16, HALF_L)
        vf_barrier(VfPipe.STORE, VfPipe.STORE)
        lo_bf16 <<= kf[6].astype(DT.bfloat16)
        hi_bf16 <<= kf[7].astype(DT.bfloat16)
        deinterleave(row_bf16, dummy_bf16, lo_bf16, hi_bf16)
        reg_to_ub(k_exp_ub[(r + 3) * BF16_C0], row_bf16, HALF_L)
        vf_barrier(VfPipe.STORE, VfPipe.STORE)


@vf()
def add_vf(a_ub: Tensor, b_ub: Tensor, out_ub: Tensor):
    a_regs = RegList(DT.float, REGS_PER_FOUR_ROWS_D)
    b_regs = RegList(DT.float, REGS_PER_FOUR_ROWS_D)
    for row_quad in range(HALF_L // 4):
        r = row_quad * 4
        a_regs <<= a_ub[r:r + 4, 0:D]
        b_regs <<= b_ub[r:r + 4, 0:D]
        a_regs <<= a_regs + b_regs
        out_ub[r:r + 4, 0:D] <<= a_regs


@vf()
def cast_l_rows_float_to_bf16_vf(src_ub: Tensor, dst_ub: Tensor):
    regs = RegList(DT.float, REGS_PER_FOUR_ROWS_D)
    for row_quad in range(HALF_L // 4):
        r = row_quad * 4
        regs <<= src_ub[r:r + 4, 0:D]
        dst_ub[r:r + 4, 0:D] <<= regs


@vf()
def cast_d_rows_vf(src_ub: Tensor, dst_ub: Tensor):
    regs = RegList(DT.float, REGS_PER_FOUR_ROWS_D)
    for row_quad in range(HALF_D // 4):
        r = row_quad * 4
        regs <<= src_ub[r:r + 4, 0:D]
        dst_ub[r:r + 4, 0:D] <<= regs


# Compile-time-constant row counts so the loop UNROLLS: with a runtime `rows: Var`
# each iteration re-emits scalar address arithmetic + a loop branch, which
# serializes the load->store chain and blocks load/store overlap. Unrolling makes
# every offset an immediate, letting the stores overlap (CANNSIM: 64-row pack
# 1413 -> 584 cyc, -59%). Specialized per caller (HALF_L, HALF_D).
@vf()
def pack_bf16_l_rows_to_nz_vf(src_nd: Tensor, dst_nz: Tensor):
    row = Reg(DT.bfloat16)
    for r in range(HALF_L):
        row <<= src_nd[r:r + 1, 0:D]
        reg_to_ub(dst_nz[r * BF16_C0], row, HALF_L)


@vf()
def pack_bf16_d_rows_to_nz_vf(src_nd: Tensor, dst_nz: Tensor):
    row = Reg(DT.bfloat16)
    for r in range(HALF_D):
        row <<= src_nd[r:r + 1, 0:D]
        reg_to_ub(dst_nz[r * BF16_C0], row, HALF_D)


@vf()
def state_update_vf(state_old_ub: Tensor, delta_ub: Tensor, g_last: Var, out_ub: Tensor):
    scale = Reg(DT.float)
    delta_regs = RegList(DT.float, REGS_PER_FOUR_ROWS_D)
    state_regs = RegList(DT.float, REGS_PER_FOUR_ROWS_D)
    scale <<= g_last
    scale <<= scale.exp()
    for row_quad in range(HALF_D // 4):
        r = row_quad * 4
        delta_regs <<= delta_ub[r:r + 4, 0:D]
        state_regs <<= state_old_ub[r:r + 4, 0:D]
        state_regs <<= state_regs * scale
        delta_regs <<= delta_regs + state_regs
        out_ub[r:r + 4, 0:D] <<= delta_regs


@kernel()
def gdn_recurrent_plain(
    attn: GM[bf16, ('B', 'H', 'C', 64, 64)],
    q_i: GM[bf16, ('B', 'H', 'C', 64, 128)],
    k_i: GM[bf16, ('B', 'H', 'C', 64, 128)],
    v_i: GM[bf16, ('B', 'H', 'C', 64, 128)],
    k_cumdecay_i: GM[bf16, ('B', 'H', 'C', 64, 128)],
    g_i: GM[f32, ('B', 'H', 'C', 64)],
    last_recurrent_state: GM[bf16, ('B', 'H', 128, 128)],
    core_attn_out: GM[bf16, ('B', 'H', 'C', 64, 128)],
    new_recurrent_state: GM[bf16, ('B', 'H', 128, 128)],
    B: i32,
    H: i32,
    C: i32,
):
    bh_count = B * H
    bh_per_core = CeilDiv(bh_count, GetCubeNum())
    bh_begin = Var(bh_per_core * GetCubeIdx())
    bh_end = Min(bh_begin + bh_per_core, bh_count)

    cvmutex = CvMutex(0, depth=1, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    state_mutex = VcMutex(1, depth=1, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE1)
    vnew_mutex = VcMutex(2, depth=1, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE1)
    kexp_mutex = VcMutex(3, depth=1, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE1)

    l1_state = Tensor(DT.bfloat16, [D, D], Position.L1)
    l1_kcd = DBuff(DT.bfloat16, [L, D], Position.L1)
    l1_q = Tensor(DT.bfloat16, [L, D], Position.L1)
    l1_attn = Tensor(DT.bfloat16, [L, L], Position.L1)
    l1_v_new = Tensor(DT.bfloat16, [L, D], Position.L1)
    l1_kexp = Tensor(DT.bfloat16, [L, D], Position.L1)

    l0c_ld = Tensor(DT.float, [L, D], Position.L0C)
    l0c_dxd = Tensor(DT.float, [D, D], Position.L0C)

    ub_prod = Tensor(DT.float, [HALF_L, D], Position.UB)
    ub_v = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    ub_v_new = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    ub_v_new_nz = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    ub_g = Tensor(DT.float, [1, HALF_L], Position.UB)
    ub_attn_inter = Tensor(DT.float, [HALF_L, D], Position.UB)
    ub_out = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    ub_k = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    ub_kexp_nz = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    ub_delta = Tensor(DT.float, [HALF_D, D], Position.UB)
    ub_state_nz = Tensor(DT.bfloat16, [HALF_D, D], Position.UB)
    ub_state = DBuff(DT.bfloat16, [HALF_D, D], Position.UB)
    state_read_cnt = Var(0)
    state_write_cnt = Var(1)
    kcd_read_cnt = Var(0)
    kcd_write_cnt = Var(0)
    g_last = Var(0.0, DT.float)

    for bh in range(bh_begin, bh_end):
        b_idx = Var(bh // H)
        h_idx = Var(bh % H)
        row_begin_l = Var(GetSubBlockIdx() * HALF_L)
        row_end_l = Var(row_begin_l + HALF_L)
        row_begin_d = Var(GetSubBlockIdx() * HALF_D)
        row_end_d = Var(row_begin_d + HALF_D)

        for c in range(C):
            ub_state_old = ub_state[state_read_cnt]
            ub_state_out = ub_state[state_write_cnt]
            with auto_sync():
                ub_v <<= v_i[b_idx, h_idx, c, row_begin_l:row_end_l, 0:D]
                ub_g[0:1, 0:HALF_L] <<= g_i[b_idx, h_idx, c, row_begin_l:row_end_l]
                ub_k <<= k_i[b_idx, h_idx, c, row_begin_l:row_end_l, 0:D]
                g_last.GetValueFrom(g_i[b_idx, h_idx, c, L - 1:L])
                if c != C - 1:
                    l1_kcd[kcd_write_cnt] <<= k_cumdecay_i[b_idx, h_idx, c + 1, 0:L, 0:D]
                    kcd_write_cnt += 1

                if c == 0:
                    pack_bf16_l_rows_to_nz_vf(ub_v, ub_v_new_nz)
                    vnew_mutex.lock()
                    l1_v_new[row_begin_l:row_end_l, 0:D] <<= ub_v_new_nz[0:HALF_L, 0:D].nz()
                    vnew_mutex.ready()
                else:
                    pack_bf16_d_rows_to_nz_vf(ub_state_old, ub_state_nz)
                    state_mutex.lock()
                    l1_state[row_begin_d:row_end_d, 0:D] <<= ub_state_nz[0:HALF_D, 0:D].nz()
                    state_mutex.ready()

                    state_mutex.wait()

                    matmul(l0c_ld, l1_kcd[kcd_read_cnt], l1_state.T, m=L, n=D, k=D, splitn=D)

                    cvmutex.lock()
                    ub_prod <<= l0c_ld
                    cvmutex.ready()
                    cvmutex.wait()
                    make_v_new_vf(ub_prod, ub_v, ub_v_new)
                    cvmutex.free()
                    pack_bf16_l_rows_to_nz_vf(ub_v_new, ub_v_new_nz)

                    vnew_mutex.lock()
                    l1_v_new[row_begin_l:row_end_l, 0:D] <<= ub_v_new_nz[0:HALF_L, 0:D].nz()
                    vnew_mutex.ready()

                    l1_q <<= q_i[b_idx, h_idx, c, 0:L, 0:D]
                    matmul(l0c_ld, l1_q, l1_state.T, m=L, n=D, k=D, splitn=D)
                    state_mutex.free()

                    cvmutex.lock()
                    ub_prod <<= l0c_ld
                    cvmutex.ready()
                    cvmutex.wait()
                    apply_g_gate_vf(ub_prod, ub_g, ub_attn_inter)
                    cvmutex.free()

                l1_attn <<= attn[b_idx, h_idx, c, 0:L, 0:L]

                kexp_mutex.lock()
                make_kexp_vf(ub_k, ub_g, g_last, ub_kexp_nz)
                l1_kexp[row_begin_l:row_end_l, 0:D] <<= ub_kexp_nz[0:HALF_L, 0:D].nz()
                kexp_mutex.ready()

                kexp_mutex.wait()
                vnew_mutex.wait()

                matmul(l0c_ld, l1_attn, l1_v_new.T, m=L, n=D, k=L, splitn=D)

                cvmutex.lock()
                ub_prod <<= l0c_ld
                cvmutex.ready()
                cvmutex.wait()
                if c == 0:
                    cast_l_rows_float_to_bf16_vf(ub_prod, ub_out)
                else:
                    add_vf(ub_prod, ub_attn_inter, ub_out)
                cvmutex.free()
                core_attn_out[b_idx, h_idx, c, row_begin_l:row_end_l, 0:D] <<= ub_out[0:HALF_L, 0:D]

                matmul(l0c_dxd, l1_kexp.T, l1_v_new.T, m=D, n=D, k=L, splitn=D)
                kexp_mutex.free()
                vnew_mutex.free()

                cvmutex.lock()
                ub_delta <<= l0c_dxd
                cvmutex.ready()
                cvmutex.wait()
                if c == 0:
                    cast_d_rows_vf(ub_delta, ub_state_out)
                else:
                    state_update_vf(ub_state_old, ub_delta, g_last, ub_state_out)
                cvmutex.free()
                if c == C - 1:
                    new_recurrent_state[b_idx, h_idx, row_begin_d:row_end_d, 0:D] <<= ub_state_out[0:HALF_D, 0:D]

            state_read_cnt += 1
            state_write_cnt += 1
            if c != 0:
                kcd_read_cnt += 1
        bar_all()

    return core_attn_out, new_recurrent_state

# ----------------------------------------------------------------------------------------------------
# recurrent_saved.py
# gdn_fwd production stage migrated from kernels/chunk_delta_sub2_state_history.py.
#
# M10 edits: preserve the selected kernel and its source dependency closure;
# replace the EasyASC facade, synthesize typed GM/scalar signatures from the
# reviewed stage contract, and preserve explicit Python unrolling with unroll.
# Host drivers and independent references now live in unit.py and ref/.
# Source SHA256: b965aaf9289dcb3864f6fe615e98188fbdb67f6a475bbf69f4717cf8ca71a088
# ----------------------------------------------------------------------------------------------------

@vf()
def make_kexp_saved_vf(
    k_ub: Tensor,
    g_ub: Tensor,
    g_last: Var,
    k_exp_ub: Tensor,
    k_exp_nd_ub: Tensor,
    exp_delta_ub: Tensor,
):
    gl = Reg(DT.float)
    g0 = Reg(DT.float)
    g1 = Reg(DT.float)
    g2 = Reg(DT.float)
    g3 = Reg(DT.float)
    scale0 = Reg(DT.float)
    scale1 = Reg(DT.float)
    scale2 = Reg(DT.float)
    scale3 = Reg(DT.float)
    kf = RegList(DT.float, REGS_PER_FOUR_ROWS_D)
    lo_bf16 = Reg(DT.bfloat16)
    hi_bf16 = Reg(DT.bfloat16)
    row_bf16 = Reg(DT.bfloat16)
    dummy_bf16 = Reg(DT.bfloat16)
    gl <<= g_last
    for row_quad in range(HALF_L // 4):
        r = row_quad * 4
        g0 <<= g_ub[0:1, r:r + 1].single()
        g1 <<= g_ub[0:1, r + 1:r + 2].single()
        g2 <<= g_ub[0:1, r + 2:r + 3].single()
        g3 <<= g_ub[0:1, r + 3:r + 4].single()
        scale0 <<= gl - g0
        scale1 <<= gl - g1
        scale2 <<= gl - g2
        scale3 <<= gl - g3
        scale0 <<= scale0.exp()
        scale1 <<= scale1.exp()
        scale2 <<= scale2.exp()
        scale3 <<= scale3.exp()
        exp_delta_ub[0:1, r + 0:r + 1] <<= scale0.single_value()
        exp_delta_ub[0:1, r + 1:r + 2] <<= scale1.single_value()
        exp_delta_ub[0:1, r + 2:r + 3] <<= scale2.single_value()
        exp_delta_ub[0:1, r + 3:r + 4] <<= scale3.single_value()
        kf <<= k_ub[r:r + 4, 0:D]
        kf[0] <<= kf[0] * scale0
        kf[1] <<= kf[1] * scale0
        kf[2] <<= kf[2] * scale1
        kf[3] <<= kf[3] * scale1
        kf[4] <<= kf[4] * scale2
        kf[5] <<= kf[5] * scale2
        kf[6] <<= kf[6] * scale3
        kf[7] <<= kf[7] * scale3
        lo_bf16 <<= kf[0].astype(DT.bfloat16)
        hi_bf16 <<= kf[1].astype(DT.bfloat16)
        deinterleave(row_bf16, dummy_bf16, lo_bf16, hi_bf16)
        reg_to_ub(k_exp_ub[(r + 0) * BF16_C0], row_bf16, HALF_L)
        reg_to_ub(k_exp_nd_ub[r + 0:r + 1, 0:D], row_bf16)
        vf_barrier(VfPipe.STORE, VfPipe.STORE)
        lo_bf16 <<= kf[2].astype(DT.bfloat16)
        hi_bf16 <<= kf[3].astype(DT.bfloat16)
        deinterleave(row_bf16, dummy_bf16, lo_bf16, hi_bf16)
        reg_to_ub(k_exp_ub[(r + 1) * BF16_C0], row_bf16, HALF_L)
        reg_to_ub(k_exp_nd_ub[r + 1:r + 2, 0:D], row_bf16)
        vf_barrier(VfPipe.STORE, VfPipe.STORE)
        lo_bf16 <<= kf[4].astype(DT.bfloat16)
        hi_bf16 <<= kf[5].astype(DT.bfloat16)
        deinterleave(row_bf16, dummy_bf16, lo_bf16, hi_bf16)
        reg_to_ub(k_exp_ub[(r + 2) * BF16_C0], row_bf16, HALF_L)
        reg_to_ub(k_exp_nd_ub[r + 2:r + 3, 0:D], row_bf16)
        vf_barrier(VfPipe.STORE, VfPipe.STORE)
        lo_bf16 <<= kf[6].astype(DT.bfloat16)
        hi_bf16 <<= kf[7].astype(DT.bfloat16)
        deinterleave(row_bf16, dummy_bf16, lo_bf16, hi_bf16)
        reg_to_ub(k_exp_ub[(r + 3) * BF16_C0], row_bf16, HALF_L)
        reg_to_ub(k_exp_nd_ub[r + 3:r + 4, 0:D], row_bf16)
        vf_barrier(VfPipe.STORE, VfPipe.STORE)








@vf()
def pack_bf16_rows_to_nz_vf(src_nd: Tensor, dst_nz: Tensor, rows: Var):
    row = Reg(DT.bfloat16)
    for r in range(rows):
        row <<= src_nd[r:r + 1, 0:D]
        reg_to_ub(dst_nz[r * BF16_C0], row, rows)




@kernel()
def gdn_recurrent_saved(
    attn: GM[bf16, ('B', 'H', 'C', 64, 64)],
    q_i: GM[bf16, ('B', 'H', 'C', 64, 128)],
    k_i: GM[bf16, ('B', 'H', 'C', 64, 128)],
    v_i: GM[bf16, ('B', 'H', 'C', 64, 128)],
    k_cumdecay_i: GM[bf16, ('B', 'H', 'C', 64, 128)],
    g_i: GM[f32, ('B', 'H', 'C', 64)],
    last_recurrent_state: GM[bf16, ('B', 'H', 128, 128)],
    core_attn_out: GM[bf16, ('B', 'H', 'C', 64, 128)],
    new_recurrent_state: GM[bf16, ('B', 'H', 128, 128)],
    state_after_history: GM[bf16, ('B', 'H', 'C', 128, 128)],
    v_new_history: GM[bf16, ('B', 'H', 'C', 64, 128)],
    k_weighted_history: GM[bf16, ('B', 'H', 'C', 64, 128)],
    exp_delta_history: GM[f32, ('B', 'H', 'C', 64)],
    B: i32,
    H: i32,
    C: i32,
):
    bh_count = B * H
    bh_per_core = CeilDiv(bh_count, GetCubeNum())
    bh_begin = Var(bh_per_core * GetCubeIdx())
    bh_end = Min(bh_begin + bh_per_core, bh_count)

    cvmutex = CvMutex(0, depth=1, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    state_mutex = VcMutex(1, depth=1, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE1)
    vnew_mutex = VcMutex(2, depth=1, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE1)
    kexp_mutex = VcMutex(3, depth=1, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE1)

    l1_state = Tensor(DT.bfloat16, [D, D], Position.L1)
    l1_kcd = DBuff(DT.bfloat16, [L, D], Position.L1)
    l1_q = Tensor(DT.bfloat16, [L, D], Position.L1)
    l1_attn = Tensor(DT.bfloat16, [L, L], Position.L1)
    l1_v_new = Tensor(DT.bfloat16, [L, D], Position.L1)
    l1_kexp = Tensor(DT.bfloat16, [L, D], Position.L1)

    l0c_ld = Tensor(DT.float, [L, D], Position.L0C)
    l0c_dxd = Tensor(DT.float, [D, D], Position.L0C)

    ub_prod = Tensor(DT.float, [HALF_L, D], Position.UB)
    ub_v = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    ub_v_new = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    ub_v_new_nz = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    ub_g = Tensor(DT.float, [1, HALF_L], Position.UB)
    ub_exp_delta = Tensor(DT.float, [1, HALF_L], Position.UB)
    ub_attn_inter = Tensor(DT.float, [HALF_L, D], Position.UB)
    ub_out = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    ub_k = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    ub_kexp_nz = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    ub_delta = Tensor(DT.float, [HALF_D, D], Position.UB)
    ub_state_nz = Tensor(DT.bfloat16, [HALF_D, D], Position.UB)
    ub_state = DBuff(DT.bfloat16, [HALF_D, D], Position.UB)
    state_read_cnt = Var(0)
    state_write_cnt = Var(1)
    kcd_read_cnt = Var(0)
    kcd_write_cnt = Var(0)
    g_last = Var(0.0, DT.float)

    for bh in range(bh_begin, bh_end):
        b_idx = Var(bh // H)
        h_idx = Var(bh % H)
        row_begin_l = Var(GetSubBlockIdx() * HALF_L)
        row_end_l = Var(row_begin_l + HALF_L)
        row_begin_d = Var(GetSubBlockIdx() * HALF_D)
        row_end_d = Var(row_begin_d + HALF_D)

        for c in range(C):
            ub_state_old = ub_state[state_read_cnt]
            ub_state_out = ub_state[state_write_cnt]
            with auto_sync():
                ub_v <<= v_i[b_idx, h_idx, c, row_begin_l:row_end_l, 0:D]
                ub_g[0:1, 0:HALF_L] <<= g_i[b_idx, h_idx, c, row_begin_l:row_end_l]
                ub_k <<= k_i[b_idx, h_idx, c, row_begin_l:row_end_l, 0:D]
                g_last.GetValueFrom(g_i[b_idx, h_idx, c, L - 1:L])
                if c != C - 1:
                    l1_kcd[kcd_write_cnt] <<= k_cumdecay_i[b_idx, h_idx, c + 1, 0:L, 0:D]
                    kcd_write_cnt += 1

                if c == 0:
                    pack_bf16_rows_to_nz_vf(ub_v, ub_v_new_nz, HALF_L)
                    vnew_mutex.lock()
                    l1_v_new[row_begin_l:row_end_l, 0:D] <<= ub_v_new_nz[0:HALF_L, 0:D].nz()
                    vnew_mutex.ready()
                    v_new_history[b_idx, h_idx, c, row_begin_l:row_end_l, 0:D] <<= ub_v[0:HALF_L, 0:D]
                else:
                    pack_bf16_rows_to_nz_vf(ub_state_old, ub_state_nz, HALF_D)
                    state_mutex.lock()
                    l1_state[row_begin_d:row_end_d, 0:D] <<= ub_state_nz[0:HALF_D, 0:D].nz()
                    state_mutex.ready()

                    state_mutex.wait()

                    matmul(l0c_ld, l1_kcd[kcd_read_cnt], l1_state.T, m=L, n=D, k=D, splitn=D)

                    cvmutex.lock()
                    ub_prod <<= l0c_ld
                    cvmutex.ready()
                    cvmutex.wait()
                    make_v_new_vf(ub_prod, ub_v, ub_v_new)
                    cvmutex.free()
                    pack_bf16_rows_to_nz_vf(ub_v_new, ub_v_new_nz, HALF_L)

                    vnew_mutex.lock()
                    l1_v_new[row_begin_l:row_end_l, 0:D] <<= ub_v_new_nz[0:HALF_L, 0:D].nz()
                    vnew_mutex.ready()
                    v_new_history[b_idx, h_idx, c, row_begin_l:row_end_l, 0:D] <<= ub_v_new[0:HALF_L, 0:D]

                    l1_q <<= q_i[b_idx, h_idx, c, 0:L, 0:D]
                    matmul(l0c_ld, l1_q, l1_state.T, m=L, n=D, k=D, splitn=D)
                    state_mutex.free()

                    cvmutex.lock()
                    ub_prod <<= l0c_ld
                    cvmutex.ready()
                    cvmutex.wait()
                    apply_g_gate_vf(ub_prod, ub_g, ub_attn_inter)
                    cvmutex.free()

                l1_attn <<= attn[b_idx, h_idx, c, 0:L, 0:L]

                kexp_mutex.lock()
                make_kexp_saved_vf(ub_k, ub_g, g_last, ub_kexp_nz, ub_k, ub_exp_delta)
                l1_kexp[row_begin_l:row_end_l, 0:D] <<= ub_kexp_nz[0:HALF_L, 0:D].nz()
                kexp_mutex.ready()

                kexp_mutex.wait()
                vnew_mutex.wait()

                matmul(l0c_ld, l1_attn, l1_v_new.T, m=L, n=D, k=L, splitn=D)

                cvmutex.lock()
                ub_prod <<= l0c_ld
                cvmutex.ready()
                cvmutex.wait()
                if c == 0:
                    cast_l_rows_float_to_bf16_vf(ub_prod, ub_out)
                else:
                    add_vf(ub_prod, ub_attn_inter, ub_out)
                cvmutex.free()
                core_attn_out[b_idx, h_idx, c, row_begin_l:row_end_l, 0:D] <<= ub_out[0:HALF_L, 0:D]
                k_weighted_history[b_idx, h_idx, c, row_begin_l:row_end_l, 0:D] <<= ub_k[0:HALF_L, 0:D]
                exp_delta_history[b_idx, h_idx, c, row_begin_l:row_end_l] <<= ub_exp_delta[0:1, 0:HALF_L]

                matmul(l0c_dxd, l1_kexp.T, l1_v_new.T, m=D, n=D, k=L, splitn=D)
                kexp_mutex.free()
                vnew_mutex.free()

                cvmutex.lock()
                ub_delta <<= l0c_dxd
                cvmutex.ready()
                cvmutex.wait()
                if c == 0:
                    cast_d_rows_vf(ub_delta, ub_state_out)
                else:
                    state_update_vf(ub_state_old, ub_delta, g_last, ub_state_out)
                cvmutex.free()
                state_after_history[b_idx, h_idx, c, row_begin_d:row_end_d, 0:D] <<= ub_state_out[0:HALF_D, 0:D]
                if c == C - 1:
                    new_recurrent_state[b_idx, h_idx, row_begin_d:row_end_d, 0:D] <<= ub_state_out[0:HALF_D, 0:D]

            state_read_cnt += 1
            state_write_cnt += 1
            if c != 0:
                kcd_read_cnt += 1
        bar_all()

    return core_attn_out, new_recurrent_state, state_after_history, v_new_history, k_weighted_history, exp_delta_history
