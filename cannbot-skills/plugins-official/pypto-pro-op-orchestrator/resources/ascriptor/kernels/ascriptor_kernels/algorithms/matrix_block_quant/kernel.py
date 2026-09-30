# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Two E5M2 output-store paths over one shared FP16 matrix-product and block-scale body."""

from ascriptor.a5 import *
from ascriptor.a5 import CastConfig, RegLayout

TILE_M = 128
BLOCK_N = 128
TILE_K = 128
SCALE_DENOM = 224.0
REGS_PER_BLOCK = BLOCK_N // 64

@vf()
def blockwise_quant_e5m2_vf(src_f32_ub: Tensor, dst_e5m2_ub: Tensor, dst_scale_ub: Tensor, rows: Var):
    row_regs = RegList(DT.float, REGS_PER_BLOCK)
    abs_regs = RegList(DT.float, REGS_PER_BLOCK)
    max_reg = Reg(DT.float)
    scale_reg = Reg(DT.float)
    scale_dup_reg = Reg(DT.float)

    for r in range(rows):
        row_regs <<= src_f32_ub[r:r + 1, 0:BLOCK_N]
        abs_regs <<= row_regs.abs()
        max_reg <<= abs_regs.cmax()
        scale_reg <<= max_reg / SCALE_DENOM
        scale_dup_reg <<= scale_reg.dup()
        row_regs <<= row_regs / scale_dup_reg
        dst_e5m2_ub[r:r + 1, 0:BLOCK_N] <<= row_regs

        # Store one scalar per row/block by filling a 64-lane row then slicing 0:1 later.
        dst_scale_ub[r:r + 1, 0:64] <<= scale_dup_reg


@vf()
def blockwise_quant_e5m2_pack4_vf(src_f32_ub: Tensor, dst_e5m2_ub: Tensor, dst_scale_ub: Tensor, rows: Var):
    row_regs = RegList(DT.float, REGS_PER_BLOCK)
    abs_regs = RegList(DT.float, REGS_PER_BLOCK)
    max_reg = Reg(DT.float)
    scale_reg = Reg(DT.float)
    scale_dup_reg = Reg(DT.float)
    cfg_zero = CastConfig(reg_layout=RegLayout.ZERO, name="cfg_blockwise_quant_e5m2")
    fp8_reg0 = Reg(DT.e5m2)
    fp8_reg1 = Reg(DT.e5m2)

    for r in range(rows):
        row_regs <<= src_f32_ub[r:r + 1, 0:BLOCK_N]
        abs_regs <<= row_regs.abs()
        max_reg <<= abs_regs.cmax()
        scale_reg <<= max_reg / SCALE_DENOM
        scale_dup_reg <<= scale_reg.dup()
        row_regs <<= row_regs / scale_dup_reg

        fp8_reg0 <<= row_regs[0].astype(DT.e5m2, cfg_zero)
        fp8_reg1 <<= row_regs[1].astype(DT.e5m2, cfg_zero)
        dst_e5m2_ub[r:r + 1, 0:64] <<= fp8_reg0.pack4()
        dst_e5m2_ub[r:r + 1, 64:128] <<= fp8_reg1.pack4()

        # Store one scalar per row/block by filling a 64-lane row then slicing 0:1 later.
        dst_scale_ub[r:r + 1, 0:64] <<= scale_dup_reg


@func()
def _blockwise_body(x, y, z, scale, M, N, K, quantize):
    cvmutex = CvMutex(0, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)

    l1x = DBuff(DT.half, [TILE_K, TILE_M], Position.L1)
    l1y = DBuff(DT.half, [TILE_K, BLOCK_N], Position.L1)
    l0c = DBuff(DT.float, [TILE_M, BLOCK_N], Position.L0C)

    xbuf = DBuff(DT.float, [TILE_M // 2, BLOCK_N], Position.UB)
    zbuf = DBuff(DT.e5m2, [TILE_M // 2, BLOCK_N], Position.UB)
    scalebuf = DBuff(DT.float, [TILE_M // 2, 64], Position.UB)

    l1_cnt = Var(0)
    l0c_ub_cnt = Var(0)
    half_rows_per_sb = Var(TILE_M // 2)

    tile_m = CeilDiv(M, TILE_M)
    tile_m_per_core = CeilDiv(tile_m, GetCubeNum())
    tile_m_begin = Var(tile_m_per_core * GetCubeIdx())
    tile_m_end = Min(tile_m_begin + tile_m_per_core, tile_m)

    with auto_sync():
        for mt in range(tile_m_begin, tile_m_end):
            m0 = Var(mt * TILE_M)
            valid_m = Min(TILE_M, M - m0)
            sb_idx = Var(GetSubBlockIdx())
            row_begin = Var(sb_idx * half_rows_per_sb)
            rows = Var(Min(half_rows_per_sb, Max(valid_m - row_begin, 0)))
            row_end = Var(row_begin + rows)

            n_blk = Var(0)
            for n0 in range(0, N, BLOCK_N):
                valid_n = Min(BLOCK_N, N - n0)

                for k0 in range(0, K, TILE_K):
                    valid_k = Min(TILE_K, K - k0)
                    # M10-042: MMAD must produce the bridge's physical128-row pitch.
                    # D-214: the M-padding columns MMAD reads must still be zero.
                    # The whole slot is cleared, not just those columns: a whole-tile fill is the
                    # only L1 fill pl can spell (pl.expands, A5-UP-004), and it writes the same
                    # values where it overlaps - the load overwrites every live column next, and
                    # the rest was zero either way. The extra MTE2 bytes are D-152's cost.
                    set_constant_to_l1(l1x[l1_cnt], 0.0)
                    l1x[l1_cnt] <<= x[k0:k0 + valid_k, m0:m0 + valid_m]
                    l1y[l1_cnt] <<= y[k0:k0 + valid_k, n0:n0 + valid_n]
                    matmul(l0c[l0c_ub_cnt], l1x[l1_cnt].T, l1y[l1_cnt].T, m=TILE_M, n=valid_n, k=valid_k, is_init=(k0 == 0))
                    l1_cnt += 1

                cvmutex.lock()
                xbuf[l0c_ub_cnt] <<= l0c[l0c_ub_cnt]
                cvmutex.ready()

                cvmutex.wait()
                if rows > 0:
                    quantize(xbuf[l0c_ub_cnt][0:rows, 0:BLOCK_N], zbuf[l0c_ub_cnt][0:rows, 0:BLOCK_N], scalebuf[l0c_ub_cnt][0:rows, 0:64], rows)
                    z[m0 + row_begin:m0 + row_end, n0:n0 + valid_n] <<= zbuf[l0c_ub_cnt][0:rows, 0:valid_n]
                    scale[m0 + row_begin:m0 + row_end, n_blk:n_blk + 1] <<= scalebuf[l0c_ub_cnt][0:rows, 0:1]
                cvmutex.free()

                l0c_ub_cnt += 1
                n_blk += 1

    return z, scale


@kernel()
def block_quant_default(x: GM[f16, ('K', 'M')], y: GM[f16, ('K', 'N')], z: GM[DT.e5m2, ('M', 'N')], scale: GM[f32, ('M', 1)], M: i32, N: i32, K: i32):
    return _blockwise_body(x, y, z, scale, M, N, K, blockwise_quant_e5m2_vf)


@kernel()
def block_quant_pack4(x: GM[f16, ('K', 'M')], y: GM[f16, ('K', 'N')], z: GM[DT.e5m2, ('M', 'N')], scale: GM[f32, ('M', 1)], M: i32, N: i32, K: i32):
    return _blockwise_body(x, y, z, scale, M, N, K, blockwise_quant_e5m2_pack4_vf)
