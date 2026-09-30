# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Five DSL stages of the fixed-length KDA (Kimi Delta Attention) forward."""

from ascriptor.a5 import *

# ----------------------------------------------------------------------------------------------------
# gate.py
# KDA kernel port; ABI and source provenance are owned by contract.json.
#
# The arithmetic and schedule come from the reviewed source; only imports and
# typed kernel signatures are migrated here.
# ----------------------------------------------------------------------------------------------------

L = 64

K_DIM = 128

K_BLOCK = 64

K_TILES = K_DIM // K_BLOCK

@vf()
def gate_cumsum_vf(src_ub: Tensor, dst_ub: Tensor):
    prev = Reg(DT.float)
    curr = Reg(DT.float)
    prev_exp = Reg(DT.float)

    for tile in range(K_TILES):
        k_begin = tile * K_BLOCK
        k_end = k_begin + K_BLOCK
        prev <<= src_ub[0:1, k_begin:k_end]
        prev_exp <<= prev.exp()
        dst_ub[0:1, k_begin:k_end] <<= prev_exp

        for r in range(1, L):
            curr <<= src_ub[r:r + 1, k_begin:k_end]
            prev <<= prev + curr
            prev_exp <<= prev.exp()
            dst_ub[r:r + 1, k_begin:k_end] <<= prev_exp

@kernel()
def kda_sub1_gate_kernel(g_raw: GM[f32, ('B', 'HV', 'C', 64, 128)], eg: GM[f32, ('B', 'HV', 'C', 64, 128)], B: i32, HV: i32, C: i32, length_per_chunk: i32, head_dim: i32):
    src_ub = DBuff(DT.float, [L, K_DIM], Position.UB)
    dst_ub = DBuff(DT.float, [L, K_DIM], Position.UB)

    work_count = B * HV * C
    work_per_vec = CeilDiv(work_count, GetVecNum())
    work_begin = Var(work_per_vec * GetVecIdx())
    work_end = Min(work_begin + work_per_vec, work_count)

    with auto_sync():
        for work in range(work_begin, work_end):
            c_idx = Var(work % C)
            tmp = Var(work // C)
            hv_idx = Var(tmp % HV)
            b_idx = Var(tmp // HV)

            src_ub[work][0:L, 0:K_DIM] <<= g_raw[b_idx, hv_idx, c_idx, 0:L, 0:K_DIM]
            gate_cumsum_vf(src_ub[work], dst_ub[work])
            eg[b_idx, hv_idx, c_idx, 0:L, 0:K_DIM] <<= dst_ub[work][0:L, 0:K_DIM]

    return eg

# ----------------------------------------------------------------------------------------------------
# intra.py
# KDA kernel port; ABI and source provenance are owned by contract.json.
#
# The arithmetic and schedule come from the reviewed source; only imports and
# typed kernel signatures are migrated here.
# ----------------------------------------------------------------------------------------------------

HALF_L = L // 2

@vf()
def score_preprocess_vf(
    q_ub: Tensor,
    k_ub: Tensor,
    eg_ub: Tensor,
    beta_ub: Tensor,
    qg_ub: Tensor,
    kbg_ub: Tensor,
    kgneg_ub: Tensor,
    rows: Var,
    scale: Var,
):
    q_row = Reg(DT.float)
    k_row = Reg(DT.float)
    eg_row = Reg(DT.float)
    beta_val = Reg(DT.float)
    out_row = Reg(DT.float)

    for r in range(rows):
        beta_val <<= beta_ub[0:1, r:r + 1].single()
        for tile in range(K_TILES):
            k_begin = tile * K_BLOCK
            k_end = k_begin + K_BLOCK
            q_row <<= q_ub[r:r + 1, k_begin:k_end]
            k_row <<= k_ub[r:r + 1, k_begin:k_end]
            eg_row <<= eg_ub[r:r + 1, k_begin:k_end]

            out_row <<= q_row * eg_row
            out_row <<= out_row * scale
            qg_ub[r:r + 1, k_begin:k_end] <<= out_row

            out_row <<= k_row * eg_row
            out_row <<= out_row * beta_val
            out_row <<= out_row.neg()
            kbg_ub[r:r + 1, k_begin:k_end] <<= out_row

            out_row <<= k_row / eg_row
            kgneg_ub[r:r + 1, k_begin:k_end] <<= out_row

@vf()
def apply_score_masks_vf(
    aqk_full_ub: Tensor,
    strict_full_ub: Tensor,
    aqk_out_ub: Tensor,
    strict_out_ub: Tensor,
    row_begin: Var,
    rows: Var,
):
    cols = Reg(DT.int)
    zero = Reg(DT.float)
    aqk_row = Reg(DT.float)
    strict_row = Reg(DT.float)
    aqk_masked = Reg(DT.float)
    strict_masked = Reg(DT.float)
    aqk_bf16 = Reg(DT.bfloat16)
    lower_eq_mask = MaskReg(DT.int, init_mode=MaskType.NONE)
    strict_lower_mask = MaskReg(DT.int, init_mode=MaskType.NONE)
    row_mask_bf16 = MaskReg(DT.bfloat16, init_mode=MaskType.NONE)

    zero <<= 0.0
    cols.arange(0)
    row_mask_bf16 <<= Var(L * 2, dtype=DT.uint32)

    for r in range(rows):
        abs_r = Var(row_begin + r)

        aqk_row <<= aqk_full_ub[r:r + 1, 0:L]
        compare(lower_eq_mask, cols, abs_r + 1, CompareMode.LT)
        select(aqk_masked, aqk_row, zero, mask=lower_eq_mask)
        aqk_bf16 <<= aqk_masked.astype(DT.bfloat16)
        reg_to_ub_downsample(aqk_out_ub[r:r + 1, 0:L], aqk_bf16, mask=row_mask_bf16)

        strict_row <<= strict_full_ub[r:r + 1, 0:L]
        compare(strict_lower_mask, cols, abs_r, CompareMode.LT)
        select(strict_masked, strict_row, zero, mask=strict_lower_mask)
        strict_out_ub[r:r + 1, 0:L] <<= strict_masked

@kernel()
def kda_sub2_score_kernel(q: GM[bf16, ('B', 'H', 'C', 64, 128)], k: GM[bf16, ('B', 'H', 'C', 64, 128)], eg: GM[f32, ('B', 'HV', 'C', 64, 128)], beta: GM[f32, ('B', 'HV', 'C', 64)], Aqk: GM[bf16, ('B', 'HV', 'C', 64, 64)], strict: GM[f32, ('B', 'HV', 'C', 64, 64)], B: i32, H: i32, HV: i32, C: i32, length_per_chunk: i32, head_dim: i32, scale: f32):
    vcmutex = VcMutex(
        0,
        depth=2,
        src_start_pipe=Pipe.MTE3,
        src_end_pipe=Pipe.MTE3,
        dst_start_pipe=Pipe.MTE1,
        dst_end_pipe=Pipe.MTE1,
    )
    score_cvmutex = CvMutex(
        1,
        depth=2,
        src_start_pipe=Pipe.FIX,
        dst_start_pipe=Pipe.V,
        src_end_pipe=Pipe.FIX,
        dst_end_pipe=Pipe.V,
    )
    l1_qg = DBuff(DT.float, [L, K_DIM], Position.L1)
    l1_kbg = DBuff(DT.float, [L, K_DIM], Position.L1)
    l1_kgneg = DBuff(DT.float, [L, K_DIM], Position.L1)
    l0c_aqk = DBuff(DT.float, [L, L], Position.L0C)
    l0c_strict = DBuff(DT.float, [L, L], Position.L0C)

    q_ub = DBuff(DT.bfloat16, [HALF_L, K_DIM], Position.UB)
    k_ub = DBuff(DT.bfloat16, [HALF_L, K_DIM], Position.UB)
    eg_ub = DBuff(DT.float, [HALF_L, K_DIM], Position.UB)
    beta_ub = DBuff(DT.float, [1, HALF_L], Position.UB)
    qg_ub = DBuff(DT.float, [HALF_L, K_DIM], Position.UB)
    kbg_ub = DBuff(DT.float, [HALF_L, K_DIM], Position.UB)
    kgneg_ub = DBuff(DT.float, [HALF_L, K_DIM], Position.UB)
    aqk_full_ub = DBuff(DT.float, [HALF_L, L], Position.UB)
    strict_full_ub = DBuff(DT.float, [HALF_L, L], Position.UB)
    aqk_out_ub = DBuff(DT.bfloat16, [HALF_L, L], Position.UB)
    strict_out_ub = DBuff(DT.float, [HALF_L, L], Position.UB)

    work_count = B * HV * C
    work_per_cube = CeilDiv(work_count, GetCubeNum())
    work_begin = Var(work_per_cube * GetCubeIdx())
    work_end = Min(work_begin + work_per_cube, work_count)
    group = Var(HV // H)
    pre_cnt = Var(0)
    cube_cnt = Var(0)
    post_cnt = Var(0)

    with auto_sync():
        for pipe_work in range(work_begin, work_end + 2):
            row_begin = Var(GetSubBlockIdx() * HALF_L)
            row_end = Min(row_begin + HALF_L, L)
            rows_this = Var(row_end - row_begin)

            if pipe_work < work_end:
                c_idx = Var(pipe_work % C)
                tmp = Var(pipe_work // C)
                hv_idx = Var(tmp % HV)
                b_idx = Var(tmp // HV)
                h_idx = Var(hv_idx // group)

                q_ub[pre_cnt][0:rows_this, 0:K_DIM] <<= q[b_idx, h_idx, c_idx, row_begin:row_end, 0:K_DIM]
                k_ub[pre_cnt][0:rows_this, 0:K_DIM] <<= k[b_idx, h_idx, c_idx, row_begin:row_end, 0:K_DIM]
                eg_ub[pre_cnt][0:rows_this, 0:K_DIM] <<= eg[b_idx, hv_idx, c_idx, row_begin:row_end, 0:K_DIM]
                beta_ub[pre_cnt][0:1, 0:rows_this] <<= beta[b_idx, hv_idx, c_idx, row_begin:row_end]
                score_preprocess_vf(
                    q_ub[pre_cnt],
                    k_ub[pre_cnt],
                    eg_ub[pre_cnt],
                    beta_ub[pre_cnt],
                    qg_ub[pre_cnt],
                    kbg_ub[pre_cnt],
                    kgneg_ub[pre_cnt],
                    rows_this,
                    scale,
                )

                vcmutex.lock()
                l1_qg[pre_cnt][row_begin:row_end, 0:K_DIM] <<= qg_ub[pre_cnt][0:rows_this, 0:K_DIM]
                l1_kbg[pre_cnt][row_begin:row_end, 0:K_DIM] <<= kbg_ub[pre_cnt][0:rows_this, 0:K_DIM]
                l1_kgneg[pre_cnt][row_begin:row_end, 0:K_DIM] <<= kgneg_ub[pre_cnt][0:rows_this, 0:K_DIM]
                vcmutex.ready()
                pre_cnt += 1

            if (pipe_work > work_begin) and (pipe_work < work_end + 1):
                vcmutex.wait()
                matmul(l0c_aqk[cube_cnt], l1_qg[cube_cnt], l1_kgneg[cube_cnt], splitk=K_BLOCK, m=L, n=L, k=K_DIM)
                matmul(l0c_strict[cube_cnt], l1_kbg[cube_cnt], l1_kgneg[cube_cnt], splitk=K_BLOCK, m=L, n=L, k=K_DIM)
                vcmutex.free()
                score_cvmutex.lock()
                aqk_full_ub[cube_cnt] <<= l0c_aqk[cube_cnt]
                strict_full_ub[cube_cnt] <<= l0c_strict[cube_cnt]
                score_cvmutex.ready()
                cube_cnt += 1

            if pipe_work > work_begin + 1:
                work = Var(pipe_work - 2)
                c_idx = Var(work % C)
                tmp = Var(work // C)
                hv_idx = Var(tmp % HV)
                b_idx = Var(tmp // HV)

                score_cvmutex.wait()
                apply_score_masks_vf(
                    aqk_full_ub[post_cnt],
                    strict_full_ub[post_cnt],
                    aqk_out_ub[post_cnt],
                    strict_out_ub[post_cnt],
                    row_begin,
                    rows_this,
                )
                score_cvmutex.free()
                Aqk[b_idx, hv_idx, c_idx, row_begin:row_end, 0:L] <<= aqk_out_ub[post_cnt][0:rows_this, 0:L]
                strict[b_idx, hv_idx, c_idx, row_begin:row_end, 0:L] <<= strict_out_ub[post_cnt][0:rows_this, 0:L]
                post_cnt += 1

    return Aqk, strict

# ----------------------------------------------------------------------------------------------------
# triangular_inverse.py
# KDA kernel port; ABI and source provenance are owned by contract.json.
#
# The arithmetic and schedule come from the reviewed source; only imports and
# typed kernel signatures are migrated here.
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

# ----------------------------------------------------------------------------------------------------
# wy.py
# KDA kernel port; ABI and source provenance are owned by contract.json.
#
# The arithmetic and schedule come from the reviewed source; only imports and
# typed kernel signatures are migrated here.
# ----------------------------------------------------------------------------------------------------

V_DIM = 128




SPLIT_N = 64

@vf()
def wy_preprocess_vf(
    q_ub: Tensor,
    k_ub: Tensor,
    beta_full_ub: Tensor,
    eg_ub: Tensor,
    eg_last_ub: Tensor,
    Akk_ub: Tensor,
    kexp_ub: Tensor,
    abeta_ub: Tensor,
    qg_ub: Tensor,
    kg_ub: Tensor,
    rows: Var,
):
    q_row = Reg(DT.float)
    k_row = Reg(DT.float)
    eg_row = Reg(DT.float)
    eg_last = Reg(DT.float)
    ratio_row = Reg(DT.float)
    a_row = Reg(DT.float)
    beta_row = Reg(DT.float)
    tmp_k = Reg(DT.float)
    tmp_a = Reg(DT.float)
    row_bf16 = Reg(DT.bfloat16)
    row_mask_bf16 = MaskReg(DT.bfloat16, init_mode=MaskType.NONE)

    row_mask_bf16 <<= Var(K_BLOCK * 2, dtype=DT.uint32)

    for r in range(rows):
        for tile in range(K_TILES):
            k_begin = tile * K_BLOCK
            k_end = k_begin + K_BLOCK
            q_row <<= q_ub[r:r + 1, k_begin:k_end]
            k_row <<= k_ub[r:r + 1, k_begin:k_end]
            eg_row <<= eg_ub[r:r + 1, k_begin:k_end]
            eg_last <<= eg_last_ub[0:1, k_begin:k_end]

            tmp_k <<= q_row * eg_row
            row_bf16 <<= tmp_k.astype(DT.bfloat16)
            reg_to_ub_downsample(qg_ub[r:r + 1, k_begin:k_end], row_bf16, mask=row_mask_bf16)

            tmp_k <<= k_row * eg_row
            row_bf16 <<= tmp_k.astype(DT.bfloat16)
            reg_to_ub_downsample(kexp_ub[r:r + 1, k_begin:k_end], row_bf16, mask=row_mask_bf16)

            ratio_row <<= eg_last / eg_row
            tmp_k <<= k_row * ratio_row
            row_bf16 <<= tmp_k.astype(DT.bfloat16)
            reg_to_ub_downsample(kg_ub[r:r + 1, k_begin:k_end], row_bf16, mask=row_mask_bf16)

        a_row <<= Akk_ub[r:r + 1, 0:L]
        beta_row <<= beta_full_ub[0:1, 0:L]
        tmp_a <<= a_row * beta_row
        row_bf16 <<= tmp_a.astype(DT.bfloat16)
        reg_to_ub_downsample(abeta_ub[r:r + 1, 0:L], row_bf16, mask=row_mask_bf16)

@kernel()
def kda_sub3_wy_kernel(q: GM[bf16, ('B', 'H', 'C', 64, 128)], k: GM[bf16, ('B', 'H', 'C', 64, 128)], v: GM[bf16, ('B', 'HV', 'C', 64, 128)], beta: GM[f32, ('B', 'HV', 'C', 64)], Akk: GM[bf16, ('B', 'HV', 'C', 64, 64)], eg: GM[f32, ('B', 'HV', 'C', 64, 128)], w: GM[bf16, ('B', 'HV', 'C', 64, 128)], u: GM[bf16, ('B', 'HV', 'C', 64, 128)], qg: GM[bf16, ('B', 'HV', 'C', 64, 128)], kg: GM[bf16, ('B', 'HV', 'C', 64, 128)], B: i32, H: i32, HV: i32, C: i32, length_per_chunk: i32, head_dim: i32, value_dim: i32):
    vcmutex = VcMutex(
        0, depth=2,
        src_start_pipe=Pipe.MTE3,
        src_end_pipe=Pipe.MTE3,
        dst_start_pipe=Pipe.MTE1,
        dst_end_pipe=Pipe.MTE1,
    )
    l1_abeta = DBuff(DT.bfloat16, [L, L], Position.L1)
    l1_kexp = DBuff(DT.bfloat16, [L, K_DIM], Position.L1)
    l1_v = DBuff(DT.bfloat16, [L, V_DIM], Position.L1)
    l0c_w = DBuff(DT.float, [L, K_DIM], Position.L0C)
    l0c_u = DBuff(DT.float, [L, V_DIM], Position.L0C)

    q_ub = DBuff(DT.bfloat16, [HALF_L, K_DIM], Position.UB)
    k_ub = DBuff(DT.bfloat16, [HALF_L, K_DIM], Position.UB)
    beta_full_ub = DBuff(DT.float, [1, L], Position.UB)
    eg_ub = DBuff(DT.float, [HALF_L, K_DIM], Position.UB)
    eg_last_ub = DBuff(DT.float, [1, K_DIM], Position.UB)
    Akk_ub = DBuff(DT.bfloat16, [HALF_L, L], Position.UB)
    kexp_ub = DBuff(DT.bfloat16, [HALF_L, K_DIM], Position.UB)
    abeta_ub = DBuff(DT.bfloat16, [HALF_L, L], Position.UB)
    qg_ub = DBuff(DT.bfloat16, [HALF_L, K_DIM], Position.UB)
    kg_ub = DBuff(DT.bfloat16, [HALF_L, K_DIM], Position.UB)

    work_count = B * HV * C
    work_per_cube = CeilDiv(work_count, GetCubeNum())
    work_begin = Var(work_per_cube * GetCubeIdx())
    work_end = Min(work_begin + work_per_cube, work_count)
    group = Var(HV // H)
    vcnt = Var(0)
    ccnt = Var(0)

    with auto_sync():
        for work in range(work_begin, work_end):
            c_idx = Var(work % C)
            tmp = Var(work // C)
            hv_idx = Var(tmp % HV)
            b_idx = Var(tmp // HV)
            h_idx = Var(hv_idx // group)

            row_begin = Var(GetSubBlockIdx() * HALF_L)
            row_end = Min(row_begin + HALF_L, L)
            rows_this = Var(row_end - row_begin)
            q_ub[vcnt][0:rows_this, 0:K_DIM] <<= q[b_idx, h_idx, c_idx, row_begin:row_end, 0:K_DIM]
            k_ub[vcnt][0:rows_this, 0:K_DIM] <<= k[b_idx, h_idx, c_idx, row_begin:row_end, 0:K_DIM]
            beta_full_ub[vcnt][0:1, 0:L] <<= beta[b_idx, hv_idx, c_idx, 0:L]
            eg_ub[vcnt][0:rows_this, 0:K_DIM] <<= eg[b_idx, hv_idx, c_idx, row_begin:row_end, 0:K_DIM]
            eg_last_ub[vcnt][0:1, 0:K_DIM] <<= eg[b_idx, hv_idx, c_idx, L - 1:L, 0:K_DIM]
            Akk_ub[vcnt][0:rows_this, 0:L] <<= Akk[b_idx, hv_idx, c_idx, row_begin:row_end, 0:L]
            wy_preprocess_vf(
                q_ub[vcnt],
                k_ub[vcnt],
                beta_full_ub[vcnt],
                eg_ub[vcnt],
                eg_last_ub[vcnt],
                Akk_ub[vcnt],
                kexp_ub[vcnt],
                abeta_ub[vcnt],
                qg_ub[vcnt],
                kg_ub[vcnt],
                rows_this,
            )

            qg[b_idx, hv_idx, c_idx, row_begin:row_end, 0:K_DIM] <<= qg_ub[vcnt][0:rows_this, 0:K_DIM]
            kg[b_idx, hv_idx, c_idx, row_begin:row_end, 0:K_DIM] <<= kg_ub[vcnt][0:rows_this, 0:K_DIM]

            vcmutex.lock()
            l1_abeta[ccnt][row_begin:row_end, 0:L] <<= abeta_ub[vcnt][0:rows_this, 0:L]
            l1_kexp[ccnt][row_begin:row_end, 0:K_DIM] <<= kexp_ub[vcnt][0:rows_this, 0:K_DIM]
            vcmutex.ready()

            l1_v[ccnt][0:L, 0:V_DIM] <<= v[b_idx, hv_idx, c_idx, 0:L, 0:V_DIM]

            vcnt += 1

            vcmutex.wait()
            matmul(l0c_w[ccnt], l1_abeta[ccnt], l1_kexp[ccnt].T, m=L, n=K_DIM, k=L, splitn=SPLIT_N)
            matmul(l0c_u[ccnt], l1_abeta[ccnt], l1_v[ccnt].T, m=L, n=V_DIM, k=L, splitn=SPLIT_N)
            vcmutex.free()
            w[b_idx, hv_idx, c_idx, 0:L, 0:K_DIM] <<= l0c_w[ccnt][0:L, 0:K_DIM]
            u[b_idx, hv_idx, c_idx, 0:L, 0:V_DIM] <<= l0c_u[ccnt][0:L, 0:V_DIM]
            ccnt += 1

    return w, u, qg, kg

# ----------------------------------------------------------------------------------------------------
# recurrent.py
# KDA kernel port; ABI and source provenance are owned by contract.json.
#
# The arithmetic and pair partition come from the reviewed source. At fixed
# V=128, both pair tiles are always valid; redundant guards are removed and
# state boundary events are explicit outside the chunk loop (contract.json).
# ----------------------------------------------------------------------------------------------------

V_BLOCK = 64

V_TILES = V_DIM // V_BLOCK


HALF_K = K_DIM // 2



@vf()
def cast_state_to_h_vf(state_ub: Tensor, h_ub: Tensor, rows: Var):
    state_row = Reg(DT.float)
    row_bf16 = Reg(DT.bfloat16)
    row_mask_bf16 = MaskReg(DT.bfloat16, init_mode=MaskType.NONE)

    row_mask_bf16 <<= Var(V_BLOCK * 2, dtype=DT.uint32)

    for r in range(rows):
        state_row <<= state_ub[r:r + 1, 0:V_BLOCK]
        row_bf16 <<= state_row.astype(DT.bfloat16)
        reg_to_ub_downsample(h_ub[r:r + 1, 0:V_BLOCK], row_bf16, mask=row_mask_bf16)

@vf()
def make_vnew_vf(prod_ub: Tensor, u_ub: Tensor, vnew_ub: Tensor, rows: Var):
    prod_row = Reg(DT.float)
    u_row = Reg(DT.float)
    row_bf16 = Reg(DT.bfloat16)
    row_mask_bf16 = MaskReg(DT.bfloat16, init_mode=MaskType.NONE)

    row_mask_bf16 <<= Var(V_BLOCK * 2, dtype=DT.uint32)

    for r in range(rows):
        prod_row <<= prod_ub[r:r + 1, 0:V_BLOCK]
        u_row <<= u_ub[r:r + 1, 0:V_BLOCK]
        prod_row <<= u_row - prod_row
        row_bf16 <<= prod_row.astype(DT.bfloat16)
        reg_to_ub_downsample(vnew_ub[r:r + 1, 0:V_BLOCK], row_bf16, mask=row_mask_bf16)

@vf()
def decay_state_vf(state_ub: Tensor, g_last_ub: Tensor, state_decayed_ub: Tensor, rows: Var):
    state_row = Reg(DT.float)
    decay = Reg(DT.float)

    for r in range(rows):
        state_row <<= state_ub[r:r + 1, 0:V_BLOCK]
        decay <<= g_last_ub[0:1, r:r + 1].single()
        state_row <<= state_row * decay
        state_decayed_ub[r:r + 1, 0:V_BLOCK] <<= state_row

@vf()
def add_delta_to_state_vf(state_decayed_ub: Tensor, delta_ub: Tensor, state_ub: Tensor, rows: Var):
    state_row = Reg(DT.float)
    delta_row = Reg(DT.float)

    for r in range(rows):
        state_row <<= state_decayed_ub[r:r + 1, 0:V_BLOCK]
        delta_row <<= delta_ub[r:r + 1, 0:V_BLOCK]
        state_row <<= state_row + delta_row
        state_ub[r:r + 1, 0:V_BLOCK] <<= state_row

@vf()
def qg_scale_vf(q_ub: Tensor, eg_ub: Tensor, qg_ub: Tensor, rows: Var, scale: Var):
    q_row = Reg(DT.float)
    eg_row = Reg(DT.float)
    tmp = Reg(DT.float)
    row_bf16 = Reg(DT.bfloat16)
    row_mask_bf16 = MaskReg(DT.bfloat16, init_mode=MaskType.NONE)

    row_mask_bf16 <<= Var(K_BLOCK * 2, dtype=DT.uint32)

    for r in range(rows):
        for tile in range(K_TILES):
            k_begin = tile * K_BLOCK
            k_end = k_begin + K_BLOCK
            q_row <<= q_ub[r:r + 1, k_begin:k_end]
            eg_row <<= eg_ub[r:r + 1, k_begin:k_end]
            tmp <<= q_row * eg_row
            tmp <<= tmp * scale
            row_bf16 <<= tmp.astype(DT.bfloat16)
            reg_to_ub_downsample(qg_ub[r:r + 1, k_begin:k_end], row_bf16, mask=row_mask_bf16)

@kernel()
def kda_sub45_fused_kernel(q: GM[bf16, ('B', 'H', 'C', 64, 128)], Aqk: GM[bf16, ('B', 'HV', 'C', 64, 64)], kg: GM[bf16, ('B', 'HV', 'C', 64, 128)], w: GM[bf16, ('B', 'HV', 'C', 64, 128)], u: GM[bf16, ('B', 'HV', 'C', 64, 128)], eg: GM[f32, ('B', 'HV', 'C', 64, 128)], initial_state: GM[f32, ('B', 'HV', 128, 128)], o: GM[bf16, ('B', 'HV', 'C', 64, 128)], final_state: GM[f32, ('B', 'HV', 128, 128)], B: i32, H: i32, HV: i32, C: i32, length_per_chunk: i32, head_dim: i32, value_dim: i32, scale: f32):
    prod_mutex = CvMutex(0, depth=2, src_start_pipe=Pipe.FIX, src_end_pipe=Pipe.FIX, dst_start_pipe=Pipe.V, dst_end_pipe=Pipe.V)
    vnew_mutex = VcMutex(1, depth=2, src_start_pipe=Pipe.MTE3, src_end_pipe=Pipe.MTE3, dst_start_pipe=Pipe.MTE1, dst_end_pipe=Pipe.MTE1)
    state_mutex = VcMutex(2, depth=2, src_start_pipe=Pipe.MTE3, src_end_pipe=Pipe.MTE3, dst_start_pipe=Pipe.MTE1, dst_end_pipe=Pipe.MTE1)
    delta_mutex = CvMutex(3, depth=2, src_start_pipe=Pipe.FIX, src_end_pipe=Pipe.FIX, dst_start_pipe=Pipe.V, dst_end_pipe=Pipe.V)
    qg_mutex = VcMutex(4, depth=2, src_start_pipe=Pipe.MTE3, src_end_pipe=Pipe.MTE3, dst_start_pipe=Pipe.MTE1, dst_end_pipe=Pipe.MTE1)

    state_ubout_valid = DEvent(Pipe.MTE3, Pipe.MTE2, preset=True)
    state_ubin_ready = DEvent(Pipe.MTE2, Pipe.V)
    state_ubout_ready = DEvent(Pipe.V, Pipe.MTE3)

    h_ubout_valid = DEvent(Pipe.MTE3, Pipe.V, preset=True)
    h_ubout_ready = DEvent(Pipe.V, Pipe.MTE3)
    vnew_ubout_valid = DEvent(Pipe.MTE3, Pipe.V, preset=True)
    vnew_ubout_ready = DEvent(Pipe.V, Pipe.MTE3)

    u_ubin_valid = DEvent(Pipe.V, Pipe.MTE2, preset=True)
    u_ubin_ready = DEvent(Pipe.MTE2, Pipe.V)
    g_ubin_valid = DEvent(Pipe.V, Pipe.MTE2, preset=True)
    g_ubin_ready = DEvent(Pipe.MTE2, Pipe.V)
    q_ubin_valid = DEvent(Pipe.V, Pipe.MTE2, preset=True)
    q_ubin_ready = DEvent(Pipe.MTE2, Pipe.V)
    qg_ubout_valid = DEvent(Pipe.MTE3, Pipe.V, preset=True)
    qg_ubout_ready = DEvent(Pipe.V, Pipe.MTE3)

    w_l1_valid = DEvent(Pipe.MTE1, Pipe.MTE2, preset=True)
    w_l1_ready = DEvent(Pipe.MTE2, Pipe.MTE1)
    kg_l1_valid = DEvent(Pipe.MTE1, Pipe.MTE2, preset=True)
    kg_l1_ready = DEvent(Pipe.MTE2, Pipe.MTE1)
    aqk_l1_valid = DEvent(Pipe.MTE1, Pipe.MTE2, preset=True)
    aqk_l1_ready = DEvent(Pipe.MTE2, Pipe.MTE1)

    prod_l0_valid = DEvent(Pipe.M, Pipe.MTE1, preset=True)
    prod_l0_ready = DEvent(Pipe.MTE1, Pipe.M)
    prod_l0c_valid = DEvent(Pipe.FIX, Pipe.M, preset=True)
    prod_l0c_ready = DEvent(Pipe.M, Pipe.FIX)
    delta_l0_valid = DEvent(Pipe.M, Pipe.MTE1, preset=True)
    delta_l0_ready = DEvent(Pipe.MTE1, Pipe.M)
    delta_l0c_valid = DEvent(Pipe.FIX, Pipe.M, preset=True)
    delta_l0c_ready = DEvent(Pipe.M, Pipe.FIX)
    out_l0c_valid = DEvent(Pipe.FIX, Pipe.M, preset=True)
    out_l0c_ready = DEvent(Pipe.M, Pipe.FIX)

    l1_state = DBuff(DT.bfloat16, [K_DIM, V_BLOCK], Position.L1)
    l1_qg = DBuff(DT.bfloat16, [L, K_DIM], Position.L1)
    l1_Aqk = DBuff(DT.bfloat16, [L, L], Position.L1)
    l1_w = DBuff(DT.bfloat16, [L, K_DIM], Position.L1)
    l1_kg = DBuff(DT.bfloat16, [L, K_DIM], Position.L1)
    l1_vnew = DBuff(DT.bfloat16, [L, V_BLOCK], Position.L1)
    l0a_prod = DBuff(DT.bfloat16, [L, K_DIM], Position.L0A)
    l0b_prod = DBuff(DT.bfloat16, [V_BLOCK, K_DIM], Position.L0B)
    l0a_delta = DBuff(DT.bfloat16, [K_DIM, L], Position.L0A)
    l0b_delta = DBuff(DT.bfloat16, [V_BLOCK, L], Position.L0B)
    l0c_prod = DBuff(DT.float, [L, V_BLOCK], Position.L0C)
    l0c_delta = DBuff(DT.float, [K_DIM, V_BLOCK], Position.L0C)
    l0c_out = DBuff(DT.float, [L, V_BLOCK], Position.L0C)

    state_ub = DBuff(DT.float, [HALF_K, V_BLOCK], Position.UB)
    h_ub = DBuff(DT.bfloat16, [HALF_K, V_BLOCK], Position.UB)
    prod_ub = DBuff(DT.float, [HALF_L, V_BLOCK], Position.UB)
    u_ub = DBuff(DT.bfloat16, [HALF_L, V_BLOCK], Position.UB)
    vnew_ub = DBuff(DT.bfloat16, [HALF_L, V_BLOCK], Position.UB)
    delta_ub = DBuff(DT.float, [HALF_K, V_BLOCK], Position.UB)
    state_decayed_ub = DBuff(DT.float, [HALF_K, V_BLOCK], Position.UB)
    g_last_ub = DBuff(DT.float, [1, HALF_K], Position.UB)
    q_ub = DBuff(DT.bfloat16, [HALF_L, K_DIM], Position.UB)
    eg_q_ub = DBuff(DT.float, [HALF_L, K_DIM], Position.UB)
    qg_ub = DBuff(DT.bfloat16, [HALF_L, K_DIM], Position.UB)

    pair_count = B * HV
    work_count = pair_count * V_TILES
    pair_begin = Var((pair_count * GetCubeIdx()) // GetCubeNum())
    pair_end = Var((pair_count * (GetCubeIdx() + 1)) // GetCubeNum())
    group = Var(HV // H)

    # auto sync is not used here because the nested for loops interfere with it,
    # causing auto sync to not output the correct event pairs
    for pair_idx in range(pair_begin, pair_end):
        pair = Var(pair_idx * V_TILES)
        row_begin_l = Var(GetSubBlockIdx() * HALF_L)
        row_end_l = Var(row_begin_l + HALF_L)
        row_begin_k = Var(GetSubBlockIdx() * HALF_K)
        row_end_k = Var(row_begin_k + HALF_K)

        for slot in range(2):
            work = Var(pair + slot)
            v_tile = Var(work % V_TILES)
            head_work = Var(work // V_TILES)
            hv_idx = Var(head_work % HV)
            b_idx = Var(head_work // HV)
            v_begin = Var(v_tile * V_BLOCK)
            v_end = Var(v_begin + V_BLOCK)
            state_ubout_valid.wait()
            state_ub[slot][0:HALF_K, 0:V_BLOCK] <<= initial_state[
                b_idx, hv_idx, row_begin_k:row_end_k, v_begin:v_end
            ]
            state_ubin_ready.set()
            state_ubin_ready.wait()

        for c_idx in range(C):
            # Vec stage A: publish current state as h and as bf16 L1.
            for slot in range(2):
                work = Var(pair + slot)
                head_work = Var(work // V_TILES)
                h_ubout_valid.wait()
                cast_state_to_h_vf(state_ub[slot], h_ub[slot], HALF_K)
                h_ubout_ready.set()
                h_ubout_ready.wait()
                state_mutex.lock()
                l1_state[slot][row_begin_k:row_end_k, 0:V_BLOCK] <<= h_ub[slot][
                    0:HALF_K, 0:V_BLOCK
                ]
                state_mutex.ready()
                h_ubout_valid.set()

            # Vec stage A1: build q * eg * scale once per head/chunk for both value tiles.
            work_q = Var(pair)
            head_work_q = Var(work_q // V_TILES)
            hv_q = Var(head_work_q % HV)
            b_q = Var(head_work_q // HV)
            h_q = Var(hv_q // group)
            q_slot = Var(c_idx % 2)
            q_ubin_valid.wait()
            q_ub[q_slot][0:HALF_L, 0:K_DIM] <<= q[b_q, h_q, c_idx, row_begin_l:row_end_l, 0:K_DIM]
            eg_q_ub[q_slot][0:HALF_L, 0:K_DIM] <<= eg[b_q, hv_q, c_idx, row_begin_l:row_end_l, 0:K_DIM]
            q_ubin_ready.set()
            q_ubin_ready.wait()
            qg_ubout_valid.wait()
            qg_scale_vf(q_ub[q_slot], eg_q_ub[q_slot], qg_ub[q_slot], HALF_L, scale)
            q_ubin_valid.set()
            qg_ubout_ready.set()
            qg_ubout_ready.wait()
            qg_mutex.lock()
            l1_qg[q_slot][row_begin_l:row_end_l, 0:K_DIM] <<= qg_ub[q_slot][0:HALF_L, 0:K_DIM]
            qg_mutex.ready()
            qg_ubout_valid.set()

            # Cube-side input for the local output product, shared by both value tiles.
            work_aqk = Var(pair)
            head_work_aqk = Var(work_aqk // V_TILES)
            hv_aqk = Var(head_work_aqk % HV)
            b_aqk = Var(head_work_aqk // HV)
            aqk_slot = Var(c_idx % 2)
            aqk_l1_valid.wait()
            l1_Aqk[aqk_slot][0:L, 0:L] <<= Aqk[b_aqk, hv_aqk, c_idx, 0:L, 0:L]
            aqk_l1_ready.set()

            # Vec stage A2: use the first w @ state wait gap for slot 0 decay.
            for slot in range(1):
                work = Var(pair + slot)
                head_work = Var(work // V_TILES)
                hv_idx = Var(head_work % HV)
                b_idx = Var(head_work // HV)
                g_ubin_valid.wait()
                g_last_ub[slot][0:1, 0:HALF_K] <<= eg[
                    b_idx, hv_idx, c_idx, L - 1:L, row_begin_k:row_end_k
                ]
                g_ubin_ready.set()
                g_ubin_ready.wait()
                decay_state_vf(state_ub[slot], g_last_ub[slot], state_decayed_ub[slot], HALF_K)
                g_ubin_valid.set()

            # Cube stage A: w @ state.
            for slot in range(2):
                work = Var(pair + slot)
                head_work = Var(work // V_TILES)
                hv_idx = Var(head_work % HV)
                b_idx = Var(head_work // HV)
                w_l1_valid.wait()
                l1_w[slot][0:L, 0:K_DIM] <<= w[b_idx, hv_idx, c_idx, 0:L, 0:K_DIM]
                w_l1_ready.set()
                state_mutex.wait()
                w_l1_ready.wait()
                prod_l0_valid.wait()
                l0a_prod[slot][0:L, 0:K_DIM] <<= l1_w[slot][0:L, 0:K_DIM]
                l0b_prod[slot][0:V_BLOCK, 0:K_DIM] <<= l1_state[slot].T
                w_l1_valid.set()
                prod_l0_ready.set()
                prod_l0_ready.wait()
                prod_l0c_valid.wait()
                # Keep the explicit cube pipe sequence visible because sync is manual here.
                mmad(l0c_prod[slot], l0a_prod[slot], l0b_prod[slot], M=L, N=V_BLOCK, K=K_DIM, is_init=True)
                prod_l0_valid.set()
                prod_l0c_ready.set()
                prod_mutex.lock()
                prod_l0c_ready.wait()
                prod_ub[slot] <<= l0c_prod[slot]
                prod_l0c_valid.set()
                prod_mutex.ready()

            # Vec stage B: u - w @ state, then publish v_new to L1.
            for slot in range(2):
                work = Var(pair + slot)
                v_tile = Var(work % V_TILES)
                head_work = Var(work // V_TILES)
                hv_idx = Var(head_work % HV)
                b_idx = Var(head_work // HV)
                v_begin = Var(v_tile * V_BLOCK)
                v_end = Var(v_begin + V_BLOCK)
                u_ubin_valid.wait()
                u_ub[slot][0:HALF_L, 0:V_BLOCK] <<= u[
                    b_idx, hv_idx, c_idx, row_begin_l:row_end_l, v_begin:v_end
                ]
                u_ubin_ready.set()
                prod_mutex.wait()
                u_ubin_ready.wait()
                vnew_ubout_valid.wait()
                make_vnew_vf(prod_ub[slot], u_ub[slot], vnew_ub[slot], HALF_L)
                u_ubin_valid.set()
                prod_mutex.free()
                vnew_ubout_ready.set()
                vnew_ubout_ready.wait()
                vnew_mutex.lock()
                l1_vnew[slot][row_begin_l:row_end_l, 0:V_BLOCK] <<= vnew_ub[slot][
                    0:HALF_L, 0:V_BLOCK
                ]
                vnew_mutex.ready()
                vnew_ubout_valid.set()

            # Vec stage B2: use the delta wait gap for slot 1 decay.
            for slot in range(1, 2):
                work = Var(pair + slot)
                head_work = Var(work // V_TILES)
                hv_idx = Var(head_work % HV)
                b_idx = Var(head_work // HV)
                g_ubin_valid.wait()
                g_last_ub[slot][0:1, 0:HALF_K] <<= eg[
                    b_idx, hv_idx, c_idx, L - 1:L, row_begin_k:row_end_k
                ]
                g_ubin_ready.set()
                g_ubin_ready.wait()
                decay_state_vf(state_ub[slot], g_last_ub[slot], state_decayed_ub[slot], HALF_K)
                g_ubin_valid.set()

            # Cube stage B: kg.T @ v_new.
            for slot in range(2):
                work = Var(pair + slot)
                head_work = Var(work // V_TILES)
                hv_idx = Var(head_work % HV)
                b_idx = Var(head_work // HV)
                kg_l1_valid.wait()
                l1_kg[slot][0:L, 0:K_DIM] <<= kg[b_idx, hv_idx, c_idx, 0:L, 0:K_DIM]
                kg_l1_ready.set()
                vnew_mutex.wait()
                kg_l1_ready.wait()
                delta_l0_valid.wait()
                l0a_delta[slot][0:K_DIM, 0:L] <<= l1_kg[slot].T
                l0b_delta[slot][0:V_BLOCK, 0:L] <<= l1_vnew[slot].T
                kg_l1_valid.set()
                vnew_mutex.free()
                delta_l0_ready.set()
                delta_l0_ready.wait()
                delta_l0c_valid.wait()
                mmad(
                    l0c_delta[slot],
                    l0a_delta[slot],
                    l0b_delta[slot],
                    M=K_DIM,
                    N=V_BLOCK,
                    K=L,
                    is_init=True,
                )
                delta_l0_valid.set()
                delta_l0c_ready.set()
                delta_mutex.lock()
                delta_l0c_ready.wait()
                delta_ub[slot] <<= l0c_delta[slot]
                delta_l0c_valid.set()
                delta_mutex.ready()

            # Cube stage C: assemble the sub5 output while h and v_new are still on chip.
            work_out = Var(pair)
            aqk_slot = Var(c_idx % 2)
            qg_slot = Var(c_idx % 2)
            aqk_l1_ready.wait()
            qg_mutex.wait()

            for slot in range(2):
                work = Var(pair + slot)
                v_tile = Var(work % V_TILES)
                head_work = Var(work // V_TILES)
                hv_idx = Var(head_work % HV)
                b_idx = Var(head_work // HV)
                v_begin = Var(v_tile * V_BLOCK)
                v_end = Var(v_begin + V_BLOCK)

                prod_l0_valid.wait()
                l0a_prod[slot][0:L, 0:K_DIM] <<= l1_qg[qg_slot][0:L, 0:K_DIM]
                l0b_prod[slot][0:V_BLOCK, 0:K_DIM] <<= l1_state[slot].T
                state_mutex.free()
                prod_l0_ready.set()
                prod_l0_ready.wait()
                out_l0c_valid.wait()
                # Reuse the prod L0A/L0B buffers for qg @ h.
                mmad(l0c_out[slot], l0a_prod[slot], l0b_prod[slot], M=L, N=V_BLOCK, K=K_DIM, is_init=True)
                prod_l0_valid.set()
                delta_l0_valid.wait()
                l0a_delta[slot][0:L, 0:L] <<= l1_Aqk[aqk_slot][0:L, 0:L]
                delta_l0_ready.set()
                delta_l0_ready.wait()
                mmad(l0c_out[slot], l0a_delta[slot][0:L, 0:L], l0b_delta[slot], M=L, N=V_BLOCK, K=L, is_init=False)
                delta_l0_valid.set()
                out_l0c_ready.set()
                out_l0c_ready.wait()
                o[b_idx, hv_idx, c_idx, 0:L, v_begin:v_end] <<= l0c_out[slot][0:L, 0:V_BLOCK]
                out_l0c_valid.set()

            qg_mutex.free()
            aqk_l1_valid.set()

            # Vec stage C: apply recurrent state update.
            for slot in range(2):
                work = Var(pair + slot)
                delta_mutex.wait()
                add_delta_to_state_vf(state_decayed_ub[slot], delta_ub[slot], state_ub[slot], HALF_K)
                delta_mutex.free()

        for slot in range(2):
            work = Var(pair + slot)
            v_tile = Var(work % V_TILES)
            head_work = Var(work // V_TILES)
            hv_idx = Var(head_work % HV)
            b_idx = Var(head_work // HV)
            v_begin = Var(v_tile * V_BLOCK)
            v_end = Var(v_begin + V_BLOCK)
            state_ubout_ready.set()
            state_ubout_ready.wait()
            final_state[b_idx, hv_idx, row_begin_k:row_end_k, v_begin:v_end] <<= state_ub[
                slot
            ][0:HALF_K, 0:V_BLOCK]
            state_ubout_valid.set()

    return o, final_state
