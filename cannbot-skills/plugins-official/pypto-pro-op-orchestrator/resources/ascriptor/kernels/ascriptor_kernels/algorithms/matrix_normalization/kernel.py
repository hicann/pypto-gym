# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Four preserved FP16 matrix-product normalization variants: small and large row-sum, row L2, and per-128-column absmax."""

from ascriptor.a5 import *

# ----------------------------------------------------------------------------------------------------
# row_sum_small.py
# Preserved row_sum_small matrix normalization; see the local shape/domain contract.
# ----------------------------------------------------------------------------------------------------

SMALL_TILE_M = 128
TILE_N = 256
TILE_K = 128
SPLIT_N = 128
VEC_WIDTH = 64
REGS_PER_ROW = TILE_N // VEC_WIDTH


@vf()
def normalize_rows_vf(src: Tensor, dst: Tensor, rows: Var):
    row_regs = RegList(DT.float, REGS_PER_ROW)
    sum_reg = Reg(DT.float)
    denom_reg = Reg(DT.float)

    for r in range(rows):
        src_row = src[r:r + 1, :]
        dst_row = dst[r:r + 1, :]

        row_regs <<= src_row
        sum_reg <<= row_regs.cadd()
        denom_reg <<= sum_reg.dup()
        row_regs <<= row_regs / denom_reg
        dst_row <<= row_regs


@kernel()
def matmul_rowwise_norm_kernel(x: GM[f16, ('M', 'K')], y: GM[f16, ('N', 'K')], z: GM[f32, ('M', 'N')], M: i32, N: i32, K: i32):
    cvmutex = CvMutex(0, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)

    l1x = DBuff(DT.half, [SMALL_TILE_M, TILE_K], Position.L1)
    l1y = DBuff(DT.half, [TILE_N, TILE_K], Position.L1)
    l0c = DBuff(DT.float, [SMALL_TILE_M, TILE_N], Position.L0C)
    xbuf = DBuff(DT.float, [SMALL_TILE_M // 2, TILE_N], Position.UB)
    outbuf = DBuff(DT.float, [SMALL_TILE_M // 2, TILE_N], Position.UB)

    tile_cnt = Var(0)
    vf_rows = Var(SMALL_TILE_M // 2)

    tile_m = CeilDiv(M, SMALL_TILE_M)
    tile_m_per_core = CeilDiv(tile_m, GetCubeNum())
    tile_m_begin = Var(tile_m_per_core * GetCubeIdx())
    tile_m_end = Min(tile_m_begin + tile_m_per_core, tile_m)

    with auto_sync():
        for mt in range(tile_m_begin, tile_m_end):
            m0 = Var(mt * SMALL_TILE_M)
            valid_m = Min(SMALL_TILE_M, M - m0)

            l1x[tile_cnt] <<= x[m0:m0 + valid_m, 0:K]
            l1y[tile_cnt] <<= y[0:N, 0:K]
            matmul(l0c[tile_cnt], l1x[tile_cnt], l1y[tile_cnt], splitn=SPLIT_N)

            cvmutex.lock()
            xbuf[tile_cnt] <<= l0c[tile_cnt]
            cvmutex.ready()

            cvmutex.wait()
            normalize_rows_vf(xbuf[tile_cnt], outbuf[tile_cnt], vf_rows)
            half_rows = CeilDiv(valid_m, 2)
            sb_idx = GetSubBlockIdx()
            row_begin = sb_idx * half_rows
            row_end = Min(row_begin + half_rows, valid_m)
            row_count = row_end - row_begin
            z[m0 + row_begin:m0 + row_end, 0:N] <<= outbuf[tile_cnt][0:row_count, :]
            cvmutex.free()
            tile_cnt += 1
    return z

# ----------------------------------------------------------------------------------------------------
# row_sum_large.py
# Preserved row_sum_large matrix normalization; see the local shape/domain contract.
# ----------------------------------------------------------------------------------------------------

LARGE_TILE_M = 64


@vf()
def accumulate_row_sum_vf(src_tile: Tensor, row_sum_dup: Tensor, dst_tile: Tensor, rows: Var):
    row_regs = RegList(DT.float, REGS_PER_ROW)
    sum_reg = Reg(DT.float)
    sum_dup_reg = Reg(DT.float)
    acc_reg = Reg(DT.float)

    for r in range(rows):
        src_row = src_tile[r:r + 1, :]
        dst_row = dst_tile[r:r + 1, :]
        sum_row = row_sum_dup[r:r + 1, :]

        row_regs <<= src_row
        sum_reg <<= row_regs.cadd()
        sum_dup_reg <<= sum_reg.dup()
        acc_reg <<= sum_row
        acc_reg <<= acc_reg + sum_dup_reg
        sum_row <<= acc_reg
        dst_row <<= row_regs
        vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def normalize_rows_with_sum_vf(src_tile: Tensor, row_sum_dup: Tensor, dst_tile: Tensor, rows: Var):
    row_regs = RegList(DT.float, REGS_PER_ROW)
    denom_reg = Reg(DT.float)

    for r in range(rows):
        src_row = src_tile[r:r + 1, :]
        sum_row = row_sum_dup[r:r + 1, :]
        dst_row = dst_tile[r:r + 1, :]

        row_regs <<= src_row
        denom_reg <<= sum_row
        row_regs <<= row_regs / denom_reg
        dst_row <<= row_regs


@vf()
def zero_row_sum_vf(row_sum_dup: Tensor, rows: Var):
    zero_reg = Reg(DT.float)
    zero_reg <<= 0.0
    for r in range(rows):
        row_sum_dup[r:r + 1, :] <<= zero_reg


@kernel()
def matmul_rowwise_norm_large_nk_kernel(x: GM[f16, ('M', 'K')], y: GM[f16, ('N', 'K')], z: GM[f32, ('M', 'N')], M: i32, N: i32, K: i32):
    cvmutex = CvMutex(0, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)

    l1x = DBuff(DT.half, [LARGE_TILE_M, TILE_K], Position.L1)
    l1y = DBuff(DT.half, [TILE_N, TILE_K], Position.L1)
    l0c = DBuff(DT.float, [LARGE_TILE_M, TILE_N], Position.L0C)
    xbuf = DBuff(DT.float, [LARGE_TILE_M // 2, TILE_N], Position.UB)
    outbuf = DBuff(DT.float, [LARGE_TILE_M // 2, TILE_N], Position.UB)
    # Pass2 reloads must not alias the next tile's cube-owned FIX publication.
    reloadbuf = DBuff(DT.float, [LARGE_TILE_M // 2, TILE_N], Position.UB)
    row_sum_ub = Tensor(DT.float, [LARGE_TILE_M // 2, VEC_WIDTH], Position.UB)

    pass1_cnt = Var(0)
    pass2_cnt = Var(0)
    l1_cnt = Var(0)

    tile_m = CeilDiv(M, LARGE_TILE_M)
    tile_m_per_core = CeilDiv(tile_m, GetCubeNum())
    tile_m_begin = Var(tile_m_per_core * GetCubeIdx())
    tile_m_end = Min(tile_m_begin + tile_m_per_core, tile_m)

    for mt in range(tile_m_begin, tile_m_end):
        m0 = Var(mt * LARGE_TILE_M)
        valid_m = Min(LARGE_TILE_M, M - m0)
        half_rows = CeilDiv(valid_m, 2)
        sb_idx = Var(GetSubBlockIdx())
        row_start = Var(m0 + sb_idx * half_rows)
        rows = Min(half_rows, valid_m - sb_idx * half_rows)
        # row_sum_ub is a persistent UB accumulator and must start from 0 per tile-M.
        zero_row_sum_vf(row_sum_ub[0:rows, :], rows)

        with auto_sync():
            # Pass-1: matmul tiles + row-sum accumulation + temporary store to z.
            for n0 in range(0, N, TILE_N):
                valid_n = Min(TILE_N, N - n0)
                for k0 in range(0, K, TILE_K):
                    valid_k = Min(TILE_K, K - k0)
                    l1x[l1_cnt] <<= x[m0:m0 + valid_m, k0:k0 + valid_k]
                    l1y[l1_cnt] <<= y[n0:n0 + valid_n, k0:k0 + valid_k]
                    # Bound the contraction to the rows actually loaded. A trailing K tile fills
                    # only `valid_k` of the slot's TILE_K, and an unbounded matmul contracts the
                    # whole tile -- reading a first-use slot's poison, where 0 * NaN is NaN. At
                    # K = 130 (tile 0 full into slot 0, tile 1 partial into a never-written slot 1)
                    # every one of the 16384 outputs came back NaN; K = 260 hid it because two full
                    # tiles fill both slots first. The other 2dgrid matmuls already pass k= (D-213).
                    matmul(l0c[pass1_cnt], l1x[l1_cnt], l1y[l1_cnt], splitn=SPLIT_N, k=valid_k,
                           is_init=(k0 == 0))
                    l1_cnt += 1

                cvmutex.lock()
                xbuf[pass1_cnt] <<= l0c[pass1_cnt]
                cvmutex.ready()

                cvmutex.wait()
                accumulate_row_sum_vf(xbuf[pass1_cnt][0:rows, :], row_sum_ub[0:rows, :], outbuf[pass1_cnt][0:rows, :], rows)
                z[row_start:row_start + rows, n0:n0 + valid_n] <<= outbuf[pass1_cnt][0:rows, 0:valid_n]
                cvmutex.free()
                pass1_cnt += 1

        # Pass-2: normalize temporary z tiles with accumulated row sums.
        bar_all()
        pass2_cnt <<= pass1_cnt
        with auto_sync():
            for n0 in range(0, N, TILE_N):
                valid_n = Min(TILE_N, N - n0)
                reloadbuf[pass2_cnt][0:rows, 0:valid_n] <<= z[row_start:row_start + rows, n0:n0 + valid_n]
                normalize_rows_with_sum_vf(reloadbuf[pass2_cnt][0:rows, :], row_sum_ub[0:rows, :], outbuf[pass2_cnt][0:rows, :], rows)
                z[row_start:row_start + rows, n0:n0 + valid_n] <<= outbuf[pass2_cnt][0:rows, 0:valid_n]
                pass2_cnt += 1
    return z

# ----------------------------------------------------------------------------------------------------
# row_l2.py
# Preserved row_l2 matrix normalization; see the local shape/domain contract.
# ----------------------------------------------------------------------------------------------------

L2_TILE_M = 64
@vf()
def accumulate_row_l2_vf(src_tile: Tensor, row_sqsum_dup: Tensor, dst_tile: Tensor, rows: Var):
    row_regs = RegList(DT.float, REGS_PER_ROW)
    sq_regs = RegList(DT.float, REGS_PER_ROW)
    sum_reg = Reg(DT.float)
    sum_dup_reg = Reg(DT.float)
    acc_reg = Reg(DT.float)

    for r in range(rows):
        src_row = src_tile[r:r + 1, :]
        dst_row = dst_tile[r:r + 1, :]
        sqsum_row = row_sqsum_dup[r:r + 1, :]

        row_regs <<= src_row
        sq_regs <<= row_regs * row_regs
        sum_reg <<= sq_regs.cadd()
        sum_dup_reg <<= sum_reg.dup()
        acc_reg <<= sqsum_row
        acc_reg <<= acc_reg + sum_dup_reg
        sqsum_row <<= acc_reg
        dst_row <<= row_regs
        vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def normalize_rows_l2_vf(src_tile: Tensor, row_sqsum_dup: Tensor, dst_tile: Tensor, rows: Var):
    row_regs = RegList(DT.float, REGS_PER_ROW)
    denom_reg = Reg(DT.float)

    for r in range(rows):
        src_row = src_tile[r:r + 1, :]
        sqsum_row = row_sqsum_dup[r:r + 1, :]
        dst_row = dst_tile[r:r + 1, :]

        row_regs <<= src_row
        denom_reg <<= sqsum_row
        denom_reg <<= denom_reg.sqrt()
        row_regs <<= row_regs / denom_reg
        dst_row <<= row_regs


@vf()
def zero_row_sqsum_vf(row_sqsum_dup: Tensor, rows: Var):
    zero_reg = Reg(DT.float)
    zero_reg <<= 0.0
    for r in range(rows):
        row_sqsum_dup[r:r + 1, :] <<= zero_reg


@kernel()
def matmul_rowwise_l2_norm_kernel(x: GM[f16, ('M', 'K')], w: GM[f16, ('N', 'K')], out: GM[f32, ('M', 'N')], M: i32, N: i32, K: i32):
    cvmutex = CvMutex(0, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)

    l1x = DBuff(DT.half, [L2_TILE_M, TILE_K], Position.L1)
    l1w = DBuff(DT.half, [TILE_N, TILE_K], Position.L1)
    l0c = DBuff(DT.float, [L2_TILE_M, TILE_N], Position.L0C)
    tile_ub = DBuff(DT.float, [L2_TILE_M // 2, TILE_N], Position.UB)
    out_ub = DBuff(DT.float, [L2_TILE_M // 2, TILE_N], Position.UB)
    # Pass2 reloads must not alias the next tile's cube-owned FIX publication.
    reload_ub = DBuff(DT.float, [L2_TILE_M // 2, TILE_N], Position.UB)
    row_sqsum_ub = Tensor(DT.float, [L2_TILE_M // 2, VEC_WIDTH], Position.UB)

    pass1_cnt = Var(0)
    pass2_cnt = Var(0)
    l1_cnt = Var(0)

    tile_m = CeilDiv(M, L2_TILE_M)
    tile_m_per_core = CeilDiv(tile_m, GetCubeNum())
    tile_m_begin = Var(tile_m_per_core * GetCubeIdx())
    tile_m_end = Min(tile_m_begin + tile_m_per_core, tile_m)

    for mt in range(tile_m_begin, tile_m_end):
        m0 = Var(mt * L2_TILE_M)
        valid_m = Min(L2_TILE_M, M - m0)
        half_rows = CeilDiv(valid_m, 2)
        sb_idx = Var(GetSubBlockIdx())
        row_start = Var(m0 + sb_idx * half_rows)
        rows = Min(half_rows, valid_m - sb_idx * half_rows)
        zero_row_sqsum_vf(row_sqsum_ub[0:rows, :], rows)

        with auto_sync():
            # Pass-1 stores the raw matmul result and accumulates the per-row squared sum.
            for n0 in range(0, N, TILE_N):
                valid_n = Min(TILE_N, N - n0)
                for k0 in range(0, K, TILE_K):
                    valid_k = Min(TILE_K, K - k0)
                    l1x[l1_cnt] <<= x[m0:m0 + valid_m, k0:k0 + valid_k]
                    l1w[l1_cnt] <<= w[n0:n0 + valid_n, k0:k0 + valid_k]
                    matmul(
                        l0c[pass1_cnt], l1x[l1_cnt], l1w[l1_cnt],
                        m=valid_m, n=valid_n, k=valid_k,
                        splitn=SPLIT_N, is_init=(k0 == 0)
                    )
                    l1_cnt += 1

                cvmutex.lock()
                tile_ub[pass1_cnt] <<= l0c[pass1_cnt]
                cvmutex.ready()

                cvmutex.wait()
                accumulate_row_l2_vf(tile_ub[pass1_cnt][0:rows, :], row_sqsum_ub[0:rows, :], out_ub[pass1_cnt][0:rows, :], rows)
                out[row_start:row_start + rows, n0:n0 + valid_n] <<= out_ub[pass1_cnt][0:rows, 0:valid_n]
                cvmutex.free()
                pass1_cnt += 1

        bar_all()
        pass2_cnt <<= pass1_cnt
        with auto_sync():
            # Pass-2 reloads the temporary matmul result and divides by sqrt(sum(z^2)).
            for n0 in range(0, N, TILE_N):
                valid_n = Min(TILE_N, N - n0)
                reload_ub[pass2_cnt][0:rows, 0:valid_n] <<= out[row_start:row_start + rows, n0:n0 + valid_n]
                normalize_rows_l2_vf(reload_ub[pass2_cnt][0:rows, :], row_sqsum_ub[0:rows, :], out_ub[pass2_cnt][0:rows, :], rows)
                out[row_start:row_start + rows, n0:n0 + valid_n] <<= out_ub[pass2_cnt][0:rows, 0:valid_n]
                pass2_cnt += 1
    return out

# ----------------------------------------------------------------------------------------------------
# block_absmax.py
# Preserved block_absmax matrix normalization; see the local shape/domain contract.
# ----------------------------------------------------------------------------------------------------

ABSMAX_TILE_M = 128
ABSMAX_TILE_N = 128
CHUNK_N = 128
REGS_PER_CHUNK = CHUNK_N // 64


@vf()
def normalize_absmax_chunk_vf(src_f32_ub: Tensor, dst_f32_ub: Tensor, rows: Var):
    row_regs = RegList(DT.float, REGS_PER_CHUNK)
    abs_regs = RegList(DT.float, REGS_PER_CHUNK)
    max_reg = Reg(DT.float)
    max_dup_reg = Reg(DT.float)

    for r in range(rows):
        src_row = src_f32_ub[r:r + 1, :]
        dst_row = dst_f32_ub[r:r + 1, :]
        row_regs <<= src_row
        abs_regs <<= row_regs.abs()
        max_reg <<= abs_regs.cmax()
        max_dup_reg <<= max_reg.dup()
        row_regs <<= row_regs / max_dup_reg
        dst_row <<= row_regs


@kernel()
def matmul_chunk_absmax_norm128_kernel(x: GM[f16, ('M', 'K')], y: GM[f16, ('N', 'K')], z: GM[f32, ('M', 'N')], M: i32, N: i32, K: i32):
    cvmutex = CvMutex(0, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)

    l1x = DBuff(DT.half, [ABSMAX_TILE_M, TILE_K], Position.L1)
    l1y = DBuff(DT.half, [ABSMAX_TILE_N, TILE_K], Position.L1)
    l0c = DBuff(DT.float, [ABSMAX_TILE_M, ABSMAX_TILE_N], Position.L0C)
    xbuf = DBuff(DT.float, [ABSMAX_TILE_M // 2, ABSMAX_TILE_N], Position.UB)
    outbuf = DBuff(DT.float, [ABSMAX_TILE_M // 2, ABSMAX_TILE_N], Position.UB)

    l0c_ub_cnt = Var(0)
    l1cnt = Var(0)
    half_rows_per_sb = Var(ABSMAX_TILE_M // 2)

    tile_m = CeilDiv(M, ABSMAX_TILE_M)
    tile_m_per_core = CeilDiv(tile_m, GetCubeNum())
    tile_m_begin = Var(tile_m_per_core * GetCubeIdx())
    tile_m_end = Min(tile_m_begin + tile_m_per_core, tile_m)

    with auto_sync():
        for mt in range(tile_m_begin, tile_m_end):
            m0 = Var(mt * ABSMAX_TILE_M)
            valid_m = Min(ABSMAX_TILE_M, M - m0)
            sb_idx = Var(GetSubBlockIdx())
            row_begin = Var(sb_idx * half_rows_per_sb)
            rows = Var(Min(half_rows_per_sb, Max(valid_m - row_begin, 0)))
            row_end = Var(row_begin + rows)

            for n0 in range(0, N, ABSMAX_TILE_N):
                valid_n = Min(ABSMAX_TILE_N, N - n0)

                for k0 in range(0, K, TILE_K):
                    valid_k = Min(TILE_K, K - k0)
                    # M10-042: the bridge requires a physical128-row MMAD result.
                    # D-214: the rows the GM transfer does not land on must still be finite.
                    # The whole slot is cleared, not just the padding: a whole-tile fill is the
                    # only L1 fill pl can spell (pl.expands, A5-UP-004), and it writes the same
                    # values where it overlaps - the load overwrites every live row next, and the
                    # rest was unobserved either way. The extra MTE2 bytes are D-152's cost.
                    set_constant_to_l1(l1x[l1cnt], 0.0)
                    l1x[l1cnt] <<= x[m0:m0 + valid_m, k0:k0 + valid_k]
                    l1y[l1cnt] <<= y[n0:n0 + valid_n, k0:k0 + valid_k]
                    matmul(l0c[l0c_ub_cnt], l1x[l1cnt], l1y[l1cnt], m=ABSMAX_TILE_M, n=valid_n, k=valid_k, splitn=SPLIT_N, is_init=(k0 == 0))
                    l1cnt += 1

                cvmutex.lock()
                xbuf[l0c_ub_cnt] <<= l0c[l0c_ub_cnt]
                cvmutex.ready()

                cvmutex.wait()
                if rows > 0:
                    normalize_absmax_chunk_vf(xbuf[l0c_ub_cnt][0:rows, :], outbuf[l0c_ub_cnt][0:rows, :], rows)
                    z[m0 + row_begin:m0 + row_end, n0:n0 + valid_n] <<= outbuf[l0c_ub_cnt][0:rows, 0:valid_n]
                cvmutex.free()
                l0c_ub_cnt += 1
    return z
