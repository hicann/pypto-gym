# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Five DSL stages of the chunked ungated Delta Rule forward: four production launches plus
the standalone intra-chunk score leaf the fused recurrence is checked against."""

from ascriptor.a5 import *  # noqa: F401,F403  # the public DSL facade

# ----------------------------------------------------------------------------------------------------
# preprocess.py
# Delta Rule migrated kernel; typed ABI derived from its input/allocation contract.
#
# Source: projects/a5/delta_rule_fwd/kernels/delta_preprocess.py.
# M10 edits: public facade, explicit annotations, native range/unroll and local closure.
# Host references and launch logic are maintained separately inside this unit.
# ----------------------------------------------------------------------------------------------------

L = 64
D = 128
HALF_L = L // 2


@vf()
def init_strict_lower_vf(strict_lower_ub: Tensor, row_begin: Var, rows: Var):
    cols = Reg(DT.int)
    zero = Reg(DT.float)
    strict_row = Reg(DT.float)
    strict_lower_mask = MaskReg(DT.int, init_mode=MaskType.NONE)

    zero <<= 0.0
    cols.arange(0)

    for r in range(rows):
        abs_r = Var(row_begin + r)
        strict_row <<= 1.0
        compare(strict_lower_mask, cols, abs_r, CompareMode.LT)  # cols < abs_r (strict lower)
        select(strict_row, strict_row, zero, mask=strict_lower_mask)
        strict_lower_ub[r:r + 1, 0:L] <<= strict_row


@vf()
def apply_beta_and_mask_vf(
    score_ub: Tensor,
    beta_ub: Tensor,
    strict_lower_mask_ub: Tensor,
    attn_ub: Tensor,
    row_begin: Var,
    rows: Var,
):
    beta_reg = Reg(DT.float)
    score_row = Reg(DT.float)
    attn_row = Reg(DT.float)
    mask_row = Reg(DT.float)

    for r in range(rows):
        beta_reg <<= beta_ub[0:1, r:r + 1].single()
        score_row <<= score_ub[r:r + 1, 0:L]
        score_row <<= score_row * beta_reg
        attn_row <<= score_row * -1.0
        mask_row <<= strict_lower_mask_ub[r:r + 1, 0:L]
        attn_row <<= attn_row * mask_row
        attn_ub[r:r + 1, 0:L] <<= attn_row


@kernel()
def delta_preprocess_kernel(
    key: GM[bf16, ('B', 'H', 'C', 64, 128)],
    beta: GM[f32, ('B', 'H', 'C', 64)],
    attn: GM[f32, ('B', 'H', 'C', 64, 64)],
    B: i32,
    H: i32,
    C: i32,
    length_per_chunk: i32,
    head_dim: i32,
):
    cvmutex = CvMutex(1, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)

    l1_key = DBuff(DT.bfloat16, [L, D], Position.L1)
    l0c_score = DBuff(DT.float, [L, L], Position.L0C)

    beta_ub = DBuff(DT.float, [1, HALF_L], Position.UB)
    score_ub = DBuff(DT.float, [HALF_L, L], Position.UB)
    strict_lower_mask_ub = Tensor(DT.float, [HALF_L, L], Position.UB)
    attn_ub = DBuff(DT.float, [HALF_L, L], Position.UB)

    stage1_cnt = Var(0)
    stage2_cnt = Var(0)
    bhc_count = B * H * C
    bhc_per_core = CeilDiv(bhc_count, GetCubeNum())
    bhc_begin = Var(bhc_per_core * GetCubeIdx())
    bhc_end = Min(bhc_begin + bhc_per_core, bhc_count)

    with auto_sync():
        if bhc_begin < bhc_end:
            mask_row_begin = Var(GetSubBlockIdx() * HALF_L)
            mask_row_end = Min(mask_row_begin + HALF_L, L)
            init_strict_lower_vf(strict_lower_mask_ub, mask_row_begin, mask_row_end - mask_row_begin)

        for bhc in range(bhc_begin, bhc_end + 1):
            if bhc < bhc_end:
                c_idx = Var(bhc // (B * H))
                bh_remainder = Var(bhc % (B * H))
                b_idx = Var(bh_remainder // H)
                h_idx = Var(bh_remainder % H)

                l1_key[stage1_cnt][0:L, 0:D] <<= key[b_idx, h_idx, c_idx, 0:L, 0:D]
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
                apply_beta_and_mask_vf(
                    score_ub[stage2_cnt],
                    beta_ub[stage2_cnt],
                    strict_lower_mask_ub,
                    attn_ub[stage2_cnt],
                    post_row_begin,
                    post_rows_this,
                )
                cvmutex.free()

                attn[prev_b_idx, prev_h_idx, prev_c_idx, post_row_begin:post_row_end, 0:L] <<= attn_ub[stage2_cnt][0:post_rows_this, 0:L]
                stage2_cnt += 1

    return attn

# ----------------------------------------------------------------------------------------------------
# inverse.py
# Delta Rule migrated kernel; typed ABI derived from its input/allocation contract.
#
# Source: projects/a5/gdn_fwd/kernels/tril_inverse64.py.
# M10 edits: public facade, explicit annotations, native range/unroll and local closure.
# Host references and launch logic are maintained separately inside this unit.
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


@vf()
def _build_eye64_vf(eye_ub: Tensor):
    """Write a SIZE x SIZE fp32 identity into UB (zero each row, set the diagonal)."""
    zero = Reg(DT.float)
    one = Reg(DT.float)
    row_mask = MaskReg(DT.float, init_mode=MaskType.NONE)
    zero.fill(0.0)
    one.fill(1.0)
    row_mask <<= Var(SIZE, dtype=DT.uint32)
    for r in range(0, SIZE):
        reg_to_ub(eye_ub[r:r + 1, 0:SIZE], zero, mask=row_mask)
        vf_barrier(VfPipe.STORE, VfPipe.STORE)
        eye_ub[r:r + 1, r:r + 1] <<= one.single_value()
        vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def _cast_half_float_to_bf16_vf(src_ub: Tensor, dst_ub: Tensor):
    """Cast a (SIZE/2) x SIZE fp32 UB tile to bf16 (one sub-block's row half)."""
    half = SIZE // 2
    row = Reg(DT.float)
    row_bf16 = Reg(DT.bfloat16)
    row_mask = MaskReg(DT.float, init_mode=MaskType.NONE)
    row_mask_bf16 = MaskReg(DT.bfloat16, init_mode=MaskType.NONE)
    row_mask <<= Var(SIZE, dtype=DT.uint32)
    row_mask_bf16 <<= Var(SIZE * 2, dtype=DT.uint32)
    for r in range(0, half):
        ub_to_reg(row, src_ub[r:r + 1, 0:SIZE], mask=row_mask)
        row_bf16 <<= row.astype(DT.bfloat16)
        reg_to_ub_downsample(dst_ub[r:r + 1, 0:SIZE], row_bf16, mask=row_mask_bf16)


@vf()
def _cast_full_float_to_bf16_vf(src_ub: Tensor, dst_ub: Tensor):
    """Cast a SIZE x SIZE fp32 UB tile to bf16 (seed I / A as bf16 cube operands)."""
    row = Reg(DT.float)
    row_bf16 = Reg(DT.bfloat16)
    row_mask = MaskReg(DT.float, init_mode=MaskType.NONE)
    row_mask_bf16 = MaskReg(DT.bfloat16, init_mode=MaskType.NONE)
    row_mask <<= Var(SIZE, dtype=DT.uint32)
    row_mask_bf16 <<= Var(SIZE * 2, dtype=DT.uint32)
    for r in range(0, SIZE):
        ub_to_reg(row, src_ub[r:r + 1, 0:SIZE], mask=row_mask)
        row_bf16 <<= row.astype(DT.bfloat16)
        reg_to_ub_downsample(dst_ub[r:r + 1, 0:SIZE], row_bf16, mask=row_mask_bf16)


@kernel()
def tril_inverse64_neumann_strict_bf16_kernel(a: GM[f32, ('B', 'H', 'C', 64, 64)], inv: GM[bf16, ('B', 'H', 'C', 64, 64)], B: i32, H: i32, C: i32):
    # 2-chunk interleave: process chunks in PAIRS with compile-time parity A/B.
    # Each parity owns an independent Apow ping-pong / l1_inv / l0c / event set, so
    # chunk B's squares+invs (no data dep on A) fill chunk A's serial-chain bubbles
    # and vice versa. Cores that draw an odd chunk count finish with one solo chunk
    # (the validated single-chunk pipeline). DBuff (parity A=2 SEvents -> no runtime
    # index needed) sidesteps the SEvent-has-no-depth / DBuff-is-depth-2 limits that
    # blocked a runtime-parity cross-chunk pipeline.
    eye_setup_vc = VcMutex(2, depth=2, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE1)
    a_vc = VcMutex(1, depth=2, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)
    cast_cv = CvMutex(0, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    # Acc->Mat insert is currently classified as MTE3 by PyPTO auto_mutex.
    # Order its real FIX reader after every M producer before publishing L1.
    product_ready = SEvent(Pipe.M, Pipe.FIX)
    apow_a = SEvent(Pipe.FIX, Pipe.MTE1)
    apow_b = SEvent(Pipe.FIX, Pipe.MTE1)
    inva = SEvent(Pipe.FIX, Pipe.MTE1)
    invb = SEvent(Pipe.FIX, Pipe.MTE1)

    half = SIZE // 2
    total_chunks = B * H * C
    a_tiles = a.reshape([total_chunks, SIZE, SIZE], name="a_tiles")
    inv_tiles = inv.reshape([total_chunks, SIZE, SIZE], name="inv_tiles")

    eye_f32_ub = Tensor(DT.float, [SIZE, SIZE], Position.UB)       # built once
    eye_bf16_ub = Tensor(DT.bfloat16, [SIZE, SIZE], Position.UB)   # built once
    # DBuff slot0 = parity A, slot1 = parity B (both live within one pair).
    a_f32_ub = DBuff(DT.float, [SIZE, SIZE], Position.UB)
    a_bf16_ub = DBuff(DT.bfloat16, [SIZE, SIZE], Position.UB)
    inv_f32_ub = DBuff(DT.float, [half, SIZE], Position.UB)
    inv_bf16_ub = DBuff(DT.bfloat16, [half, SIZE], Position.UB)

    # bf16 L1 operands -> the cube's fast matmul path; fp32 L0C accumulate, FIX
    # casts fp32 L0C -> bf16 L1 on every publish (de-risked to 2.4e-4 vs exact).
    l1_eye = Tensor(DT.bfloat16, [SIZE, SIZE], Position.L1)        # constant, once
    # per-parity 2-slot Apow ping-pong (square writes Apow_{j+1} to the free slot
    # while inv_j still reads Apow_j) + per-parity running inverse.
    l1_apow_a = DBuff(DT.bfloat16, [SIZE, SIZE], Position.L1)
    l1_apow_b = DBuff(DT.bfloat16, [SIZE, SIZE], Position.L1)
    l1_inv_a = Tensor(DT.bfloat16, [SIZE, SIZE], Position.L1)
    l1_inv_b = Tensor(DT.bfloat16, [SIZE, SIZE], Position.L1)

    # l0c_inv DBuff rotates per pair so pair N's cast overlaps pair N+1's init.
    l0c_inv_a = DBuff(DT.float, [SIZE, SIZE], Position.L0C)
    l0c_inv_b = DBuff(DT.float, [SIZE, SIZE], Position.L0C)
    l0c_sq_a = Tensor(DT.float, [SIZE, SIZE], Position.L0C)
    l0c_sq_b = Tensor(DT.float, [SIZE, SIZE], Position.L0C)

    chunks_per_core = CeilDiv(total_chunks, GetCubeNum())
    chunk_begin = Var(chunks_per_core * GetCubeIdx())
    chunk_end = Min(chunk_begin + chunks_per_core, total_chunks)
    row_begin = Var(GetSubBlockIdx() * half)
    row_end = Var(row_begin + half)
    num_chunks = Var(chunk_end - chunk_begin)
    num_pairs = Var(num_chunks // 2)

    with auto_sync():
        # --- build the bf16 identity ONCE; VcMutex(MTE3->MTE1) syncs it to the cube
        #     (bar_all does NOT cover MTE3->MTE1, so a one-time publish needs this) ---
        eye_setup_vc.lock()
        if GetSubBlockIdx() == 0:
            _build_eye64_vf(eye_f32_ub)
            _cast_full_float_to_bf16_vf(eye_f32_ub, eye_bf16_ub)
            l1_eye[0:SIZE, 0:SIZE] <<= eye_bf16_ub[0:SIZE, 0:SIZE]
        eye_setup_vc.ready()
        eye_setup_vc.wait()
        eye_setup_vc.free()

        pair_buf = Var(0)
        for p in range(0, num_pairs):
            ca = Var(chunk_begin + 2 * p)
            cb = Var(ca + 1)

            # --- load A -> l1_apow_a[0], B -> l1_apow_b[0] (= Apow_0 each) ---
            a_vc.lock()
            if GetSubBlockIdx() == 0:
                a_f32_ub[0][0:SIZE, 0:SIZE] <<= a_tiles[ca, 0:SIZE, 0:SIZE]
                _cast_full_float_to_bf16_vf(a_f32_ub[0], a_bf16_ub[0])
                l1_apow_a[0][0:SIZE, 0:SIZE] <<= a_bf16_ub[0][0:SIZE, 0:SIZE]
                a_f32_ub[1][0:SIZE, 0:SIZE] <<= a_tiles[cb, 0:SIZE, 0:SIZE]
                _cast_full_float_to_bf16_vf(a_f32_ub[1], a_bf16_ub[1])
                l1_apow_b[0][0:SIZE, 0:SIZE] <<= a_bf16_ub[1][0:SIZE, 0:SIZE]
            a_vc.ready()
            a_vc.wait()

            # --- init: inv_0 = I + A = A@I + I@I  (A, then B) ---
            matmul(l0c_inv_a[pair_buf], l1_apow_a[0], l1_eye.T, m=SIZE, n=SIZE, k=SIZE)
            matmul(l0c_inv_a[pair_buf], l1_eye, l1_eye.T, m=SIZE, n=SIZE, k=SIZE, is_init=False)
            product_ready.set()
            product_ready.wait()
            l1_inv_a[0:SIZE, 0:SIZE] <<= l0c_inv_a[pair_buf][0:SIZE, 0:SIZE]
            inva.set()
            matmul(l0c_inv_b[pair_buf], l1_apow_b[0], l1_eye.T, m=SIZE, n=SIZE, k=SIZE)
            matmul(l0c_inv_b[pair_buf], l1_eye, l1_eye.T, m=SIZE, n=SIZE, k=SIZE, is_init=False)
            product_ready.set()
            product_ready.wait()
            l1_inv_b[0:SIZE, 0:SIZE] <<= l0c_inv_b[pair_buf][0:SIZE, 0:SIZE]
            invb.set()

            # --- sq_1: Apow_1 = Apow_0^2 -> slot1  (A, then B) ---
            matmul(l0c_sq_a, l1_apow_a[0], l1_apow_a[0].T, m=SIZE, n=SIZE, k=SIZE)
            product_ready.set()
            product_ready.wait()
            l1_apow_a[1][0:SIZE, 0:SIZE] <<= l0c_sq_a[0:SIZE, 0:SIZE]
            apow_a.set()
            matmul(l0c_sq_b, l1_apow_b[0], l1_apow_b[0].T, m=SIZE, n=SIZE, k=SIZE)
            product_ready.set()
            product_ready.wait()
            l1_apow_b[1][0:SIZE, 0:SIZE] <<= l0c_sq_b[0:SIZE, 0:SIZE]
            apow_b.set()

            # --- doublings j=1..4: square A,B one step ahead, then inv A,B.
            #     (rd, wr) = (Apow_j slot read, Apow_{j+1} slot written); inv_j also
            #     reads Apow_j in `rd`. Interleaving the two parities fills each
            #     square's FIX (publish) gap with the other parity's matmuls. ---
            for (rd, wr) in [(1, 0), (0, 1), (1, 0), (0, 1)]:
                apow_a.wait()                                                        # Apow_a_j
                matmul(l0c_sq_a, l1_apow_a[rd], l1_apow_a[rd].T, m=SIZE, n=SIZE, k=SIZE)
                product_ready.set()
                product_ready.wait()
                l1_apow_a[wr][0:SIZE, 0:SIZE] <<= l0c_sq_a[0:SIZE, 0:SIZE]            # Apow_a_{j+1}
                apow_a.set()
                apow_b.wait()                                                        # Apow_b_j
                matmul(l0c_sq_b, l1_apow_b[rd], l1_apow_b[rd].T, m=SIZE, n=SIZE, k=SIZE)
                product_ready.set()
                product_ready.wait()
                l1_apow_b[wr][0:SIZE, 0:SIZE] <<= l0c_sq_b[0:SIZE, 0:SIZE]            # Apow_b_{j+1}
                apow_b.set()
                inva.wait()                                                          # inv_a_{j-1}
                matmul(l0c_inv_a[pair_buf], l1_inv_a, l1_eye.T, m=SIZE, n=SIZE, k=SIZE)
                matmul(l0c_inv_a[pair_buf], l1_inv_a, l1_apow_a[rd].T, m=SIZE, n=SIZE, k=SIZE, is_init=False)
                product_ready.set()
                product_ready.wait()
                l1_inv_a[0:SIZE, 0:SIZE] <<= l0c_inv_a[pair_buf][0:SIZE, 0:SIZE]
                inva.set()
                invb.wait()                                                          # inv_b_{j-1}
                matmul(l0c_inv_b[pair_buf], l1_inv_b, l1_eye.T, m=SIZE, n=SIZE, k=SIZE)
                matmul(l0c_inv_b[pair_buf], l1_inv_b, l1_apow_b[rd].T, m=SIZE, n=SIZE, k=SIZE, is_init=False)
                product_ready.set()
                product_ready.wait()
                l1_inv_b[0:SIZE, 0:SIZE] <<= l0c_inv_b[pair_buf][0:SIZE, 0:SIZE]
                invb.set()

            # --- epilogue: inv_5 = inv_4 @ (I + Apow_5)  (Apow_5 in slot1), no publish ---
            apow_a.wait()                                                            # Apow_a_5
            inva.wait()                                                              # inv_a_4
            matmul(l0c_inv_a[pair_buf], l1_inv_a, l1_eye.T, m=SIZE, n=SIZE, k=SIZE)
            matmul(l0c_inv_a[pair_buf], l1_inv_a, l1_apow_a[1].T, m=SIZE, n=SIZE, k=SIZE, is_init=False)
            apow_b.wait()                                                            # Apow_b_5
            invb.wait()                                                              # inv_b_4
            matmul(l0c_inv_b[pair_buf], l1_inv_b, l1_eye.T, m=SIZE, n=SIZE, k=SIZE)
            matmul(l0c_inv_b[pair_buf], l1_inv_b, l1_apow_b[1].T, m=SIZE, n=SIZE, k=SIZE, is_init=False)
            a_vc.free()

            # --- cast A,B fp32 inv -> bf16 -> GM; SPLITM gives each sub-block a row half ---
            cast_cv.lock()
            l0c_to_ub(inv_f32_ub[0], l0c_inv_a[pair_buf], M=SIZE, N=SIZE, N_dst=SIZE, M_src=SIZE,
                      dual_mode=DualMode.SPLITM, sub_block_id=0)
            cast_cv.ready()
            cast_cv.wait()
            _cast_half_float_to_bf16_vf(inv_f32_ub[0], inv_bf16_ub[0])
            cast_cv.free()
            inv_tiles[ca, row_begin:row_end, 0:SIZE] <<= inv_bf16_ub[0][0:half, 0:SIZE]

            cast_cv.lock()
            l0c_to_ub(inv_f32_ub[1], l0c_inv_b[pair_buf], M=SIZE, N=SIZE, N_dst=SIZE, M_src=SIZE,
                      dual_mode=DualMode.SPLITM, sub_block_id=0)
            cast_cv.ready()
            cast_cv.wait()
            _cast_half_float_to_bf16_vf(inv_f32_ub[1], inv_bf16_ub[1])
            cast_cv.free()
            inv_tiles[cb, row_begin:row_end, 0:SIZE] <<= inv_bf16_ub[1][0:half, 0:SIZE]
            pair_buf += 1

        # --- odd tail: one solo chunk (the validated single-chunk pipeline, parity A) ---
        if num_chunks % 2 != 0:
            ct = Var(chunk_end - 1)
            a_vc.lock()
            if GetSubBlockIdx() == 0:
                a_f32_ub[0][0:SIZE, 0:SIZE] <<= a_tiles[ct, 0:SIZE, 0:SIZE]
                _cast_full_float_to_bf16_vf(a_f32_ub[0], a_bf16_ub[0])
                l1_apow_a[0][0:SIZE, 0:SIZE] <<= a_bf16_ub[0][0:SIZE, 0:SIZE]
            a_vc.ready()
            a_vc.wait()

            matmul(l0c_inv_a[pair_buf], l1_apow_a[0], l1_eye.T, m=SIZE, n=SIZE, k=SIZE)
            matmul(l0c_inv_a[pair_buf], l1_eye, l1_eye.T, m=SIZE, n=SIZE, k=SIZE, is_init=False)
            product_ready.set()
            product_ready.wait()
            l1_inv_a[0:SIZE, 0:SIZE] <<= l0c_inv_a[pair_buf][0:SIZE, 0:SIZE]
            inva.set()
            matmul(l0c_sq_a, l1_apow_a[0], l1_apow_a[0].T, m=SIZE, n=SIZE, k=SIZE)
            product_ready.set()
            product_ready.wait()
            l1_apow_a[1][0:SIZE, 0:SIZE] <<= l0c_sq_a[0:SIZE, 0:SIZE]
            apow_a.set()
            for (rd, wr) in [(1, 0), (0, 1), (1, 0), (0, 1)]:
                apow_a.wait()
                matmul(l0c_sq_a, l1_apow_a[rd], l1_apow_a[rd].T, m=SIZE, n=SIZE, k=SIZE)
                product_ready.set()
                product_ready.wait()
                l1_apow_a[wr][0:SIZE, 0:SIZE] <<= l0c_sq_a[0:SIZE, 0:SIZE]
                apow_a.set()
                inva.wait()
                matmul(l0c_inv_a[pair_buf], l1_inv_a, l1_eye.T, m=SIZE, n=SIZE, k=SIZE)
                matmul(l0c_inv_a[pair_buf], l1_inv_a, l1_apow_a[rd].T, m=SIZE, n=SIZE, k=SIZE, is_init=False)
                product_ready.set()
                product_ready.wait()
                l1_inv_a[0:SIZE, 0:SIZE] <<= l0c_inv_a[pair_buf][0:SIZE, 0:SIZE]
                inva.set()
            apow_a.wait()
            inva.wait()
            matmul(l0c_inv_a[pair_buf], l1_inv_a, l1_eye.T, m=SIZE, n=SIZE, k=SIZE)
            matmul(l0c_inv_a[pair_buf], l1_inv_a, l1_apow_a[1].T, m=SIZE, n=SIZE, k=SIZE, is_init=False)
            a_vc.free()

            cast_cv.lock()
            l0c_to_ub(inv_f32_ub[0], l0c_inv_a[pair_buf], M=SIZE, N=SIZE, N_dst=SIZE, M_src=SIZE,
                      dual_mode=DualMode.SPLITM, sub_block_id=0)
            cast_cv.ready()
            cast_cv.wait()
            _cast_half_float_to_bf16_vf(inv_f32_ub[0], inv_bf16_ub[0])
            cast_cv.free()
            inv_tiles[ct, row_begin:row_end, 0:SIZE] <<= inv_bf16_ub[0][0:half, 0:SIZE]

    return inv


@kernel()
def tril_inverse64_neumann_strict_bf16_serial_kernel(a: GM[f32, ('B', 'H', 'C', 64, 64)], inv: GM[bf16, ('B', 'H', 'C', 64, 64)], B: i32, H: i32, C: i32):
    # Single-parity Neumann retains the original operation order. Both variants
    # explicitly order M->FIX publication as well as FIX->MTE1 consumption.
    eye_setup_vc = VcMutex(2, depth=2, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE1)
    a_vc = VcMutex(1, depth=2, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)
    cast_cv = CvMutex(0, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    product_ready = SEvent(Pipe.M, Pipe.FIX)
    apow_a = SEvent(Pipe.FIX, Pipe.MTE1)
    inva = SEvent(Pipe.FIX, Pipe.MTE1)

    half = SIZE // 2
    total_chunks = B * H * C
    a_tiles = a.reshape([total_chunks, SIZE, SIZE], name="a_tiles")
    inv_tiles = inv.reshape([total_chunks, SIZE, SIZE], name="inv_tiles")

    eye_f32_ub = Tensor(DT.float, [SIZE, SIZE], Position.UB)
    eye_bf16_ub = Tensor(DT.bfloat16, [SIZE, SIZE], Position.UB)
    a_f32_ub = DBuff(DT.float, [SIZE, SIZE], Position.UB)
    a_bf16_ub = DBuff(DT.bfloat16, [SIZE, SIZE], Position.UB)
    inv_f32_ub = DBuff(DT.float, [half, SIZE], Position.UB)
    inv_bf16_ub = DBuff(DT.bfloat16, [half, SIZE], Position.UB)

    l1_eye = Tensor(DT.bfloat16, [SIZE, SIZE], Position.L1)
    l1_apow_a = DBuff(DT.bfloat16, [SIZE, SIZE], Position.L1)
    l1_inv_a = Tensor(DT.bfloat16, [SIZE, SIZE], Position.L1)
    l0c_inv_a = DBuff(DT.float, [SIZE, SIZE], Position.L0C)
    l0c_sq_a = Tensor(DT.float, [SIZE, SIZE], Position.L0C)

    chunks_per_core = CeilDiv(total_chunks, GetCubeNum())
    chunk_begin = Var(chunks_per_core * GetCubeIdx())
    chunk_end = Min(chunk_begin + chunks_per_core, total_chunks)
    num_chunks = Var(chunk_end - chunk_begin)
    row_begin = Var(GetSubBlockIdx() * half)
    row_end = Var(row_begin + half)

    with auto_sync():
        eye_setup_vc.lock()
        if GetSubBlockIdx() == 0:
            _build_eye64_vf(eye_f32_ub)
            _cast_full_float_to_bf16_vf(eye_f32_ub, eye_bf16_ub)
            l1_eye[0:SIZE, 0:SIZE] <<= eye_bf16_ub[0:SIZE, 0:SIZE]
        eye_setup_vc.ready()
        eye_setup_vc.wait()
        eye_setup_vc.free()

        for c in range(0, num_chunks):
            ct = Var(chunk_begin + c)

            # --- load A -> l1_apow_a[0] (= Apow_0) ---
            a_vc.lock()
            if GetSubBlockIdx() == 0:
                a_f32_ub[0][0:SIZE, 0:SIZE] <<= a_tiles[ct, 0:SIZE, 0:SIZE]
                _cast_full_float_to_bf16_vf(a_f32_ub[0], a_bf16_ub[0])
                l1_apow_a[0][0:SIZE, 0:SIZE] <<= a_bf16_ub[0][0:SIZE, 0:SIZE]
            a_vc.ready()
            a_vc.wait()

            # --- init: inv_0 = I + A ;  sq_1: Apow_1 = Apow_0^2 -> slot1 ---
            matmul(l0c_inv_a[0], l1_apow_a[0], l1_eye.T, m=SIZE, n=SIZE, k=SIZE)
            matmul(l0c_inv_a[0], l1_eye, l1_eye.T, m=SIZE, n=SIZE, k=SIZE, is_init=False)
            product_ready.set()
            product_ready.wait()
            l1_inv_a[0:SIZE, 0:SIZE] <<= l0c_inv_a[0][0:SIZE, 0:SIZE]
            inva.set()
            matmul(l0c_sq_a, l1_apow_a[0], l1_apow_a[0].T, m=SIZE, n=SIZE, k=SIZE)
            product_ready.set()
            product_ready.wait()
            l1_apow_a[1][0:SIZE, 0:SIZE] <<= l0c_sq_a[0:SIZE, 0:SIZE]
            apow_a.set()

            # --- doublings j=1..4: square one step ahead, then inv ---
            for (rd, wr) in [(1, 0), (0, 1), (1, 0), (0, 1)]:
                apow_a.wait()
                matmul(l0c_sq_a, l1_apow_a[rd], l1_apow_a[rd].T, m=SIZE, n=SIZE, k=SIZE)
                product_ready.set()
                product_ready.wait()
                l1_apow_a[wr][0:SIZE, 0:SIZE] <<= l0c_sq_a[0:SIZE, 0:SIZE]
                apow_a.set()
                inva.wait()
                matmul(l0c_inv_a[0], l1_inv_a, l1_eye.T, m=SIZE, n=SIZE, k=SIZE)
                matmul(l0c_inv_a[0], l1_inv_a, l1_apow_a[rd].T, m=SIZE, n=SIZE, k=SIZE, is_init=False)
                product_ready.set()
                product_ready.wait()
                l1_inv_a[0:SIZE, 0:SIZE] <<= l0c_inv_a[0][0:SIZE, 0:SIZE]
                inva.set()

            # --- epilogue: inv_5 = inv_4 @ (I + Apow_5), no publish ---
            apow_a.wait()
            inva.wait()
            matmul(l0c_inv_a[0], l1_inv_a, l1_eye.T, m=SIZE, n=SIZE, k=SIZE)
            matmul(l0c_inv_a[0], l1_inv_a, l1_apow_a[1].T, m=SIZE, n=SIZE, k=SIZE, is_init=False)
            a_vc.free()

            # --- cast fp32 inv -> bf16 -> GM (SPLITM row half per sub-block) ---
            cast_cv.lock()
            l0c_to_ub(inv_f32_ub[0], l0c_inv_a[0], M=SIZE, N=SIZE, N_dst=SIZE, M_src=SIZE,
                      dual_mode=DualMode.SPLITM, sub_block_id=0)
            cast_cv.ready()
            cast_cv.wait()
            _cast_half_float_to_bf16_vf(inv_f32_ub[0], inv_bf16_ub[0])
            cast_cv.free()
            inv_tiles[ct, row_begin:row_end, 0:SIZE] <<= inv_bf16_ub[0][0:half, 0:SIZE]
    return inv

# ----------------------------------------------------------------------------------------------------
# recompute.py
# Delta Rule migrated kernel; typed ABI derived from its input/allocation contract.
#
# Source: projects/a5/delta_rule_fwd/kernels/delta_recompute_wu.py.
# M10 edits: public facade, explicit annotations, native range/unroll and local closure.
# Host references and launch logic are maintained separately inside this unit.
# ----------------------------------------------------------------------------------------------------

BF16_C0 = 16


@vf()
def scale_and_pack_vf(
    key_ub: Tensor,
    value_ub: Tensor,
    beta_ub: Tensor,
    k_beta_nz_ub: Tensor,
    v_beta_nz_ub: Tensor,
    rows: Var,
):
    beta_reg = Reg(DT.float)
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

        row_lo <<= key_ub[r:r + 1, 0:64]
        row_lo <<= row_lo * beta_reg
        row_hi <<= key_ub[r:r + 1, 64:D]
        row_hi <<= row_hi * beta_reg
        lo_bf16 <<= row_lo.astype(DT.bfloat16)
        hi_bf16 <<= row_hi.astype(DT.bfloat16)
        deinterleave(row_bf16, dummy_bf16, lo_bf16, hi_bf16)
        reg_to_ub(k_beta_nz_ub[r * BF16_C0], row_bf16, rows)


@kernel()
def delta_recompute_wu_kernel(
    key: GM[bf16, ('B', 'H', 'C', 64, 128)],
    value: GM[bf16, ('B', 'H', 'C', 64, 128)],
    beta: GM[f32, ('B', 'H', 'C', 64)],
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
    l1_k_beta = DBuff(DT.bfloat16, [L, D], Position.L1)
    l0c_value = DBuff(DT.float, [L, D], Position.L0C)
    l0c_kcumdecay = DBuff(DT.float, [L, D], Position.L0C)

    key_ub = DBuff(DT.bfloat16, [HALF_L, D], Position.UB)
    value_ub = DBuff(DT.bfloat16, [HALF_L, D], Position.UB)
    beta_ub = DBuff(DT.float, [1, HALF_L], Position.UB)
    v_beta_nz_ub = DBuff(DT.bfloat16, [HALF_L, D], Position.UB)
    k_beta_nz_ub = DBuff(DT.bfloat16, [HALF_L, D], Position.UB)

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
                scale_and_pack_vf(
                    key_ub[vcnt],
                    value_ub[vcnt],
                    beta_ub[vcnt],
                    k_beta_nz_ub[vcnt],
                    v_beta_nz_ub[vcnt],
                    rows_this,
                )
                l1_v_beta[ccnt][row_begin:row_end, 0:D] <<= v_beta_nz_ub[vcnt][0:rows_this, 0:D].nz()
                l1_k_beta[ccnt][row_begin:row_end, 0:D] <<= k_beta_nz_ub[vcnt][0:rows_this, 0:D].nz()
            vcnt += 1
            vcmutex.ready()

            vcmutex.wait()
            l1_attn[ccnt][0:L, 0:L] <<= attn[b_idx, h_idx, c_idx, 0:L, 0:L]

            matmul(l0c_value[ccnt], l1_attn[ccnt], l1_v_beta[ccnt].T, splitn=128)
            matmul(l0c_kcumdecay[ccnt], l1_attn[ccnt], l1_k_beta[ccnt].T, splitn=128)

            value_out[b_idx, h_idx, c_idx, 0:L, 0:D] <<= l0c_value[ccnt]
            k_cumdecay[b_idx, h_idx, c_idx, 0:L, 0:D] <<= l0c_kcumdecay[ccnt]
            vcmutex.free()
            ccnt += 1

    return value_out, k_cumdecay

# ----------------------------------------------------------------------------------------------------
# scores.py
# Delta Rule migrated kernel; typed ABI derived from its input/allocation contract.
#
# Source: projects/a5/delta_rule_fwd/kernels/chunk_delta_sub1.py.
# M10 edits: public facade, explicit annotations, native range/unroll and local closure.
# Host references and launch logic are maintained separately inside this unit.
# ----------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------

M1_N = 64


@vf()
def attn_causal_keep_vf(attn_ub: Tensor, out_ub: Tensor, row_begin: Var, nrows: Var):
    cols = Reg(DT.int)
    zero = Reg(DT.float)
    a_reg = Reg(DT.float)
    keep_mask = MaskReg(DT.int, init_mode=MaskType.NONE)

    zero <<= 0.0
    cols.arange(0)

    for r in range(nrows):
        abs_r = Var(row_begin + r)
        a_reg <<= attn_ub[r * M1_N]
        # keep columns <= abs_r (lower triangle incl. diagonal), zero the rest.
        compare(keep_mask, cols, abs_r + 1, CompareMode.LT)
        select(a_reg, a_reg, zero, mask=keep_mask)
        out_ub[r * M1_N] <<= a_reg


@kernel()
def sub1_kernel(
    q_i: GM[bf16, ('B', 'H', 'C', 64, 128)],
    k_i: GM[bf16, ('B', 'H', 'C', 64, 128)],
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

            cvmutex.wait()
            attn_causal_keep_vf(ub_attn[tile_cnt], ub_out[tile_cnt], row_begin, row_count)
            cvmutex.free()

            attn[b_idx, h_idx, c_idx, row_begin:row_end, 0:M1_N] <<= ub_out[tile_cnt][0:row_count, 0:M1_N]
            tile_cnt += 1

    return attn

# ----------------------------------------------------------------------------------------------------
# recurrence.py
# Delta Rule migrated kernel; typed ABI derived from its input/allocation contract.
#
# Source: projects/a5/delta_rule_fwd/kernels/chunk_delta_sub2.py.
# M10 edits: public facade, explicit annotations, native range/unroll and local closure.
# Host references and launch logic are maintained separately inside this unit.
# ----------------------------------------------------------------------------------------------------

HALF_D = D // 2
REGS_PER_ROW_D = D // 64
REGS_PER_FOUR_ROWS_D = REGS_PER_ROW_D * 4
SCORE_N = L  # intra-chunk attn is [L, L]; one fp32 reg holds a 64-wide row


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


@vf()
def pack_bf16_rows_to_nz_vf(src_nd: Tensor, dst_nz: Tensor, rows: Var):
    row = Reg(DT.bfloat16)
    for r in range(rows):
        row <<= src_nd[r:r + 1, 0:D]
        reg_to_ub(dst_nz[r * BF16_C0], row, rows)


@vf()
def score_causal_keep_vf(score_ub: Tensor, out_ub: Tensor, row_begin: Var, nrows: Var):
    """Fused sub1 epilogue: keep the lower triangle (incl. diagonal) of q@k^T.

    Ported verbatim from chunk_delta_sub1: reads the fp32 score row, zeros the
    columns strictly above the diagonal, and casts to bf16 on store. The same body
    is `attn_causal_keep_vf` above, inside the standalone `sub1_kernel` leaf; they
    are kept apart so that changing the fused path cannot silently change the leaf
    the fused path is checked against.
    """
    cols = Reg(DT.int)
    zero = Reg(DT.float)
    a_reg = Reg(DT.float)
    keep_mask = MaskReg(DT.int, init_mode=MaskType.NONE)

    zero <<= 0.0
    cols.arange(0)

    for r in range(nrows):
        abs_r = Var(row_begin + r)
        a_reg <<= score_ub[r * SCORE_N]
        # keep columns <= abs_r (lower triangle incl. diagonal), zero the rest.
        compare(keep_mask, cols, abs_r + 1, CompareMode.LT)
        select(a_reg, a_reg, zero, mask=keep_mask)
        out_ub[r * SCORE_N] <<= a_reg


@vf()
def state_add_vf(state_old_ub: Tensor, delta_ub: Tensor, out_ub: Tensor):
    """S = S_old + S_delta with no gate decay (replaces state_update_vf)."""
    delta_regs = RegList(DT.float, REGS_PER_FOUR_ROWS_D)
    state_regs = RegList(DT.float, REGS_PER_FOUR_ROWS_D)
    for row_quad in range(HALF_D // 4):
        r = row_quad * 4
        delta_regs <<= delta_ub[r:r + 4, 0:D]
        state_regs <<= state_old_ub[r:r + 4, 0:D]
        delta_regs <<= delta_regs + state_regs
        out_ub[r:r + 4, 0:D] <<= delta_regs


@kernel()
def sub2_kernel(
    q_i: GM[bf16, ('B', 'H', 'C', 64, 128)],
    k_i: GM[bf16, ('B', 'H', 'C', 64, 128)],
    v_i: GM[bf16, ('B', 'H', 'C', 64, 128)],
    k_cumdecay_i: GM[bf16, ('B', 'H', 'C', 64, 128)],
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

    # Lever A: the cube->vec bridge (l0c_ld -> ub_prod) is double-buffered and the
    # cvmutex is depth-2, so the cube side can issue the next matmul/publish while
    # the vec side still consumes the previous result instead of strict ping-pong.
    cvmutex = CvMutex(0, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    state_mutex = VcMutex(1, depth=1, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE1)
    vnew_mutex = VcMutex(2, depth=1, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE1)
    knz_mutex = VcMutex(3, depth=1, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE1)
    # Fused sub1: q@k^T (cube) -> causal-keep mask (vec) -> l1_attn (vec -> cube).
    attn_cv = CvMutex(4, depth=1, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    attn_vc = VcMutex(5, depth=1, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE1)

    l1_state = Tensor(DT.bfloat16, [D, D], Position.L1)
    l1_kcd = DBuff(DT.bfloat16, [L, D], Position.L1)
    l1_q = Tensor(DT.bfloat16, [L, D], Position.L1)
    l1_k = Tensor(DT.bfloat16, [L, D], Position.L1)  # full key for the fused q@k^T
    l1_attn = Tensor(DT.bfloat16, [L, L], Position.L1)
    l1_v_new = Tensor(DT.bfloat16, [L, D], Position.L1)
    l1_k_nz = Tensor(DT.bfloat16, [L, D], Position.L1)

    l0c_ld = DBuff(DT.float, [L, D], Position.L0C)
    l0c_dxd = Tensor(DT.float, [D, D], Position.L0C)
    l0c_attn = Tensor(DT.float, [L, L], Position.L0C)  # fused q@k^T (L0C 256KB has room)

    ub_prod = DBuff(DT.float, [HALF_L, D], Position.UB)
    ub_attn_score = Tensor(DT.float, [HALF_L, L], Position.UB)  # fused q@k^T rows
    ub_attn_keep = Tensor(DT.bfloat16, [HALF_L, L], Position.UB)  # masked attn (bf16)
    ub_v = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    ub_v_new = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    ub_v_new_nz = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    ub_out = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    ub_k = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    ub_k_nz = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    ub_delta = Tensor(DT.float, [HALF_D, D], Position.UB)
    ub_state_nz = Tensor(DT.bfloat16, [HALF_D, D], Position.UB)
    ub_state = DBuff(DT.bfloat16, [HALF_D, D], Position.UB)
    state_read_cnt = Var(0)
    state_write_cnt = Var(1)
    kcd_read_cnt = Var(0)
    kcd_write_cnt = Var(0)
    prod_cnt = Var(0)  # rotates the double-buffered l0c_ld/ub_prod bridge slot

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
                ub_k <<= k_i[b_idx, h_idx, c, row_begin_l:row_end_l, 0:D]
                # Full q/k in L1 for the fused intra-chunk attn (q is reused by q@state).
                l1_q <<= q_i[b_idx, h_idx, c, 0:L, 0:D]
                l1_k <<= k_i[b_idx, h_idx, c, 0:L, 0:D]
                if c != C - 1:
                    l1_kcd[kcd_write_cnt] <<= k_cumdecay_i[b_idx, h_idx, c + 1, 0:L, 0:D]
                    kcd_write_cnt += 1

                # --- fused sub1: attn = causal_keep(q @ k^T) ---
                # State-independent, computed here (early) so the masked attn lands in
                # L1 well before the deferred output matmul; this removes the sub1
                # kernel and the attn GM round-trip and overlaps sub2's cube/vec slack.
                matmul(l0c_attn, l1_q, l1_k, m=L, n=L, k=D, splitn=L)
                attn_cv.lock()
                # bare l0c->ub: the half-height ub signals the per-sub-block row split
                # (dynamic l0c[row_begin_l:...] slices are rejected by l0c_to_ub).
                ub_attn_score <<= l0c_attn
                attn_cv.ready()
                attn_cv.wait()
                score_causal_keep_vf(ub_attn_score, ub_attn_keep, row_begin_l, HALF_L)
                attn_cv.free()
                attn_vc.lock()
                l1_attn[row_begin_l:row_end_l, 0:L] <<= ub_attn_keep[0:HALF_L, 0:L]
                attn_vc.ready()

                # Pack the key to NZ early (state-independent) so the critical
                # state-delta matmul is not gated by waiting for it.
                knz_mutex.lock()
                pack_bf16_rows_to_nz_vf(ub_k, ub_k_nz, HALF_L)
                l1_k_nz[row_begin_l:row_end_l, 0:D] <<= ub_k_nz[0:HALF_L, 0:D].nz()
                knz_mutex.ready()

                # --- produce v_new ---
                if c == 0:
                    pack_bf16_rows_to_nz_vf(ub_v, ub_v_new_nz, HALF_L)
                    vnew_mutex.lock()
                    l1_v_new[row_begin_l:row_end_l, 0:D] <<= ub_v_new_nz[0:HALF_L, 0:D].nz()
                    vnew_mutex.ready()
                else:
                    pack_bf16_rows_to_nz_vf(ub_state_old, ub_state_nz, HALF_D)
                    state_mutex.lock()
                    l1_state[row_begin_d:row_end_d, 0:D] <<= ub_state_nz[0:HALF_D, 0:D].nz()
                    state_mutex.ready()

                    state_mutex.wait()
                    matmul(l0c_ld[prod_cnt], l1_kcd[kcd_read_cnt], l1_state.T, m=L, n=D, k=D, splitn=D)

                    cvmutex.lock()
                    ub_prod[prod_cnt] <<= l0c_ld[prod_cnt]
                    cvmutex.ready()
                    cvmutex.wait()
                    make_v_new_vf(ub_prod[prod_cnt], ub_v, ub_v_new)
                    cvmutex.free()
                    prod_cnt += 1
                    pack_bf16_rows_to_nz_vf(ub_v_new, ub_v_new_nz, HALF_L)

                    vnew_mutex.lock()
                    l1_v_new[row_begin_l:row_end_l, 0:D] <<= ub_v_new_nz[0:HALF_L, 0:D].nz()
                    vnew_mutex.ready()

                    # attn_inter = q @ state[c] is folded into the OUTPUT L0C tile:
                    # matmul3 (attn @ v_new) accumulates on top of it below
                    # (is_init=False), so this needs no cube->vec bridge and no
                    # separate ub_attn_inter vec add.  prod_cnt is NOT advanced here,
                    # so the output matmul reuses this same l0c_ld slot.  state[c] is
                    # released right after this last read.  l1_q is already loaded at
                    # the top of the iteration (shared with the fused q@k^T).
                    matmul(l0c_ld[prod_cnt], l1_q, l1_state.T, m=L, n=D, k=D, splitn=D)
                    state_mutex.free()

                # --- state delta (critical path): advance state[c+1] BEFORE the output,
                # so the next chunk's recurrence can start while this chunk's output runs.
                knz_mutex.wait()
                vnew_mutex.wait()
                matmul(l0c_dxd, l1_k_nz.T, l1_v_new.T, m=D, n=D, k=L, splitn=D)
                knz_mutex.free()

                cvmutex.lock()
                ub_delta <<= l0c_dxd
                cvmutex.ready()
                cvmutex.wait()
                if c == 0:
                    cast_d_rows_vf(ub_delta, ub_state_out)
                else:
                    state_add_vf(ub_state_old, ub_delta, ub_state_out)
                cvmutex.free()
                if c == C - 1:
                    new_recurrent_state[b_idx, h_idx, row_begin_d:row_end_d, 0:D] <<= ub_state_out[0:HALF_D, 0:D]

                # --- output (deferred, off the state critical path):
                # out = attn @ v_new + q @ state.  `l1_attn` is the fused intra-chunk
                # attention produced at the top of the iteration; wait for that
                # vec->cube publish here.  For c>0 the q@state term is already in this
                # L0C tile, so matmul3 accumulates (is_init=False); for c==0 there is
                # no inter-chunk term, so matmul3 initializes.
                attn_vc.wait()
                if c == 0:
                    matmul(l0c_ld[prod_cnt], l1_attn, l1_v_new.T, m=L, n=D, k=L, splitn=D)
                else:
                    matmul(l0c_ld[prod_cnt], l1_attn, l1_v_new.T, m=L, n=D, k=L, splitn=D, is_init=False)
                attn_vc.free()
                vnew_mutex.free()

                cvmutex.lock()
                ub_prod[prod_cnt] <<= l0c_ld[prod_cnt]
                cvmutex.ready()
                cvmutex.wait()
                cast_l_rows_float_to_bf16_vf(ub_prod[prod_cnt], ub_out)
                cvmutex.free()
                prod_cnt += 1
                core_attn_out[b_idx, h_idx, c, row_begin_l:row_end_l, 0:D] <<= ub_out[0:HALF_L, 0:D]

            state_read_cnt += 1
            state_write_cnt += 1
            if c != 0:
                kcd_read_cnt += 1
        bar_all()

    return core_attn_out, new_recurrent_state


# Named saved_state variant: post-chunk BF16 state and BF16 v_new before state update.
@kernel()
def recurrence_saved_kernel(
    q_i: GM[bf16, ('B', 'H', 'C', 64, 128)],
    k_i: GM[bf16, ('B', 'H', 'C', 64, 128)],
    v_i: GM[bf16, ('B', 'H', 'C', 64, 128)],
    k_cumdecay_i: GM[bf16, ('B', 'H', 'C', 64, 128)],
    last_recurrent_state: GM[bf16, ('B', 'H', 128, 128)],
    core_attn_out: GM[bf16, ('B', 'H', 'C', 64, 128)],
    new_recurrent_state: GM[bf16, ('B', 'H', 128, 128)],
    state_after_history: GM[bf16, ('B', 'H', 'C', 128, 128)],
    v_new_history: GM[bf16, ('B', 'H', 'C', 64, 128)],
    B: i32,
    H: i32,
    C: i32,
):
    bh_count = B * H
    bh_per_core = CeilDiv(bh_count, GetCubeNum())
    bh_begin = Var(bh_per_core * GetCubeIdx())
    bh_end = Min(bh_begin + bh_per_core, bh_count)

    # Lever A: the cube->vec bridge (l0c_ld -> ub_prod) is double-buffered and the
    # cvmutex is depth-2, so the cube side can issue the next matmul/publish while
    # the vec side still consumes the previous result instead of strict ping-pong.
    cvmutex = CvMutex(0, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    state_mutex = VcMutex(1, depth=1, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE1)
    vnew_mutex = VcMutex(2, depth=1, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE1)
    knz_mutex = VcMutex(3, depth=1, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE1)
    # Fused sub1: q@k^T (cube) -> causal-keep mask (vec) -> l1_attn (vec -> cube).
    attn_cv = CvMutex(4, depth=1, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    attn_vc = VcMutex(5, depth=1, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE1)

    l1_state = Tensor(DT.bfloat16, [D, D], Position.L1)
    l1_kcd = DBuff(DT.bfloat16, [L, D], Position.L1)
    l1_q = Tensor(DT.bfloat16, [L, D], Position.L1)
    l1_k = Tensor(DT.bfloat16, [L, D], Position.L1)  # full key for the fused q@k^T
    l1_attn = Tensor(DT.bfloat16, [L, L], Position.L1)
    l1_v_new = Tensor(DT.bfloat16, [L, D], Position.L1)
    l1_k_nz = Tensor(DT.bfloat16, [L, D], Position.L1)

    l0c_ld = DBuff(DT.float, [L, D], Position.L0C)
    l0c_dxd = Tensor(DT.float, [D, D], Position.L0C)
    l0c_attn = Tensor(DT.float, [L, L], Position.L0C)  # fused q@k^T (L0C 256KB has room)

    ub_prod = DBuff(DT.float, [HALF_L, D], Position.UB)
    ub_attn_score = Tensor(DT.float, [HALF_L, L], Position.UB)  # fused q@k^T rows
    ub_attn_keep = Tensor(DT.bfloat16, [HALF_L, L], Position.UB)  # masked attn (bf16)
    ub_v = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    ub_v_new = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    ub_v_new_nz = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    ub_out = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    ub_k = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    ub_k_nz = Tensor(DT.bfloat16, [HALF_L, D], Position.UB)
    ub_delta = Tensor(DT.float, [HALF_D, D], Position.UB)
    ub_state_nz = Tensor(DT.bfloat16, [HALF_D, D], Position.UB)
    ub_state = DBuff(DT.bfloat16, [HALF_D, D], Position.UB)
    state_read_cnt = Var(0)
    state_write_cnt = Var(1)
    kcd_read_cnt = Var(0)
    kcd_write_cnt = Var(0)
    prod_cnt = Var(0)  # rotates the double-buffered l0c_ld/ub_prod bridge slot

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
                ub_k <<= k_i[b_idx, h_idx, c, row_begin_l:row_end_l, 0:D]
                # Full q/k in L1 for the fused intra-chunk attn (q is reused by q@state).
                l1_q <<= q_i[b_idx, h_idx, c, 0:L, 0:D]
                l1_k <<= k_i[b_idx, h_idx, c, 0:L, 0:D]
                if c != C - 1:
                    l1_kcd[kcd_write_cnt] <<= k_cumdecay_i[b_idx, h_idx, c + 1, 0:L, 0:D]
                    kcd_write_cnt += 1

                # --- fused sub1: attn = causal_keep(q @ k^T) ---
                # State-independent, computed here (early) so the masked attn lands in
                # L1 well before the deferred output matmul; this removes the sub1
                # kernel and the attn GM round-trip and overlaps sub2's cube/vec slack.
                matmul(l0c_attn, l1_q, l1_k, m=L, n=L, k=D, splitn=L)
                attn_cv.lock()
                # bare l0c->ub: the half-height ub signals the per-sub-block row split
                # (dynamic l0c[row_begin_l:...] slices are rejected by l0c_to_ub).
                ub_attn_score <<= l0c_attn
                attn_cv.ready()
                attn_cv.wait()
                score_causal_keep_vf(ub_attn_score, ub_attn_keep, row_begin_l, HALF_L)
                attn_cv.free()
                attn_vc.lock()
                l1_attn[row_begin_l:row_end_l, 0:L] <<= ub_attn_keep[0:HALF_L, 0:L]
                attn_vc.ready()

                # Pack the key to NZ early (state-independent) so the critical
                # state-delta matmul is not gated by waiting for it.
                knz_mutex.lock()
                pack_bf16_rows_to_nz_vf(ub_k, ub_k_nz, HALF_L)
                l1_k_nz[row_begin_l:row_end_l, 0:D] <<= ub_k_nz[0:HALF_L, 0:D].nz()
                knz_mutex.ready()

                # --- produce v_new ---
                if c == 0:
                    v_new_history[b_idx, h_idx, c, row_begin_l:row_end_l, 0:D] <<= ub_v
                    pack_bf16_rows_to_nz_vf(ub_v, ub_v_new_nz, HALF_L)
                    vnew_mutex.lock()
                    l1_v_new[row_begin_l:row_end_l, 0:D] <<= ub_v_new_nz[0:HALF_L, 0:D].nz()
                    vnew_mutex.ready()
                else:
                    pack_bf16_rows_to_nz_vf(ub_state_old, ub_state_nz, HALF_D)
                    state_mutex.lock()
                    l1_state[row_begin_d:row_end_d, 0:D] <<= ub_state_nz[0:HALF_D, 0:D].nz()
                    state_mutex.ready()

                    state_mutex.wait()
                    matmul(l0c_ld[prod_cnt], l1_kcd[kcd_read_cnt], l1_state.T, m=L, n=D, k=D, splitn=D)

                    cvmutex.lock()
                    ub_prod[prod_cnt] <<= l0c_ld[prod_cnt]
                    cvmutex.ready()
                    cvmutex.wait()
                    make_v_new_vf(ub_prod[prod_cnt], ub_v, ub_v_new)
                    v_new_history[b_idx, h_idx, c, row_begin_l:row_end_l, 0:D] <<= ub_v_new
                    cvmutex.free()
                    prod_cnt += 1
                    pack_bf16_rows_to_nz_vf(ub_v_new, ub_v_new_nz, HALF_L)

                    vnew_mutex.lock()
                    l1_v_new[row_begin_l:row_end_l, 0:D] <<= ub_v_new_nz[0:HALF_L, 0:D].nz()
                    vnew_mutex.ready()

                    # attn_inter = q @ state[c] is folded into the OUTPUT L0C tile:
                    # matmul3 (attn @ v_new) accumulates on top of it below
                    # (is_init=False), so this needs no cube->vec bridge and no
                    # separate ub_attn_inter vec add.  prod_cnt is NOT advanced here,
                    # so the output matmul reuses this same l0c_ld slot.  state[c] is
                    # released right after this last read.  l1_q is already loaded at
                    # the top of the iteration (shared with the fused q@k^T).
                    matmul(l0c_ld[prod_cnt], l1_q, l1_state.T, m=L, n=D, k=D, splitn=D)
                    state_mutex.free()

                # --- state delta (critical path): advance state[c+1] BEFORE the output,
                # so the next chunk's recurrence can start while this chunk's output runs.
                knz_mutex.wait()
                vnew_mutex.wait()
                matmul(l0c_dxd, l1_k_nz.T, l1_v_new.T, m=D, n=D, k=L, splitn=D)
                knz_mutex.free()

                cvmutex.lock()
                ub_delta <<= l0c_dxd
                cvmutex.ready()
                cvmutex.wait()
                if c == 0:
                    cast_d_rows_vf(ub_delta, ub_state_out)
                else:
                    state_add_vf(ub_state_old, ub_delta, ub_state_out)
                cvmutex.free()
                state_after_history[b_idx, h_idx, c, row_begin_d:row_end_d, 0:D] <<= ub_state_out
                if c == C - 1:
                    new_recurrent_state[b_idx, h_idx, row_begin_d:row_end_d, 0:D] <<= ub_state_out[0:HALF_D, 0:D]

                # --- output (deferred, off the state critical path):
                # out = attn @ v_new + q @ state.  `l1_attn` is the fused intra-chunk
                # attention produced at the top of the iteration; wait for that
                # vec->cube publish here.  For c>0 the q@state term is already in this
                # L0C tile, so matmul3 accumulates (is_init=False); for c==0 there is
                # no inter-chunk term, so matmul3 initializes.
                attn_vc.wait()
                if c == 0:
                    matmul(l0c_ld[prod_cnt], l1_attn, l1_v_new.T, m=L, n=D, k=L, splitn=D)
                else:
                    matmul(l0c_ld[prod_cnt], l1_attn, l1_v_new.T, m=L, n=D, k=L, splitn=D, is_init=False)
                attn_vc.free()
                vnew_mutex.free()

                cvmutex.lock()
                ub_prod[prod_cnt] <<= l0c_ld[prod_cnt]
                cvmutex.ready()
                cvmutex.wait()
                cast_l_rows_float_to_bf16_vf(ub_prod[prod_cnt], ub_out)
                cvmutex.free()
                prod_cnt += 1
                core_attn_out[b_idx, h_idx, c, row_begin_l:row_end_l, 0:D] <<= ub_out[0:HALF_L, 0:D]

            state_read_cnt += 1
            state_write_cnt += 1
            if c != 0:
                kcd_read_cnt += 1
        bar_all()

    return core_attn_out, new_recurrent_state, state_after_history, v_new_history
