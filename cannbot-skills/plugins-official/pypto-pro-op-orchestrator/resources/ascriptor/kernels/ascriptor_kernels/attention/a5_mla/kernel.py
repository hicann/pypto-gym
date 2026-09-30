# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""E4M3 MLA decode: a 512-wide non-positional product and a 64-wide positional one combined in the vector stage, with K_nope reused as the value matrix."""

import math

from ascriptor.a5 import *



BASES = 256  # the old script set these in its __main__ block; the kernels read them as module constants
Dn = 512
Dr = 64
SCALE = math.sqrt(1 / (Dn + Dr))

PV_CHUNK = 256


@vf()
def scale_add_qk_vf(ubin_rope: Tensor, ubin_nope: Tensor, ubout_qk: Tensor, scale_rope_var: Var, n_loops: Var):
    rope_reg = Reg(DT.float)
    nope_reg = Reg(DT.float)
    for i in range(n_loops):
        rope_reg <<= ubin_rope[i * 64]
        nope_reg <<= ubin_nope[i * 64]
        rope_reg <<= (rope_reg * scale_rope_var + nope_reg) * SCALE
        ubout_qk[i * 64] <<= rope_reg

@vf()
def get_p_and_max_sum(ub_qk: Tensor, ub_max: Tensor, ub_sum: Tensor, ub_expdiff: Tensor, ub_p: Tensor, n_rows: Var):
    regs = RegList(DT.float, BASES // 64)
    prev_max = Reg(DT.float)
    max_reg = Reg(DT.float)
    prev_sum = Reg(DT.float)
    sum_reg = Reg(DT.float)
    expdiffreg = Reg(DT.float)

    for i in range(n_rows):
        regs <<= ub_qk[i * BASES]
        prev_max <<= ub_max[i].single()
        prev_sum <<= ub_sum[i].single()
        # compute m = max(m, mprev)
        max_reg <<= regs.cmax()
        max_reg <<= max_reg.dup()
        max_reg <<= max_reg.vmax(prev_max)
        # compute exp(mprev - m)
        expdiffreg <<= prev_max - max_reg
        expdiffreg <<= expdiffreg.exp()
        # compute exp(x - m)
        regs <<= regs - max_reg
        regs <<= regs.exp()
        # compute sum(e)
        sum_reg <<= regs.cadd()
        sum_reg <<= sum_reg.dup()
        # s equals s + expdiff * sprev.
        sum_reg <<= sum_reg + expdiffreg * prev_sum
        # scale p matrix
        regs <<= regs * 16.0
        # output everything
        ub_p[i * BASES] <<= regs
        ub_max[i] <<= max_reg.single_value()
        ub_sum[i] <<= sum_reg.single_value()
        ub_expdiff[i] <<= expdiffreg.single_value()

@vf()
def init_buffers(qksum: Tensor, qkmax: Tensor, expdiff: Tensor, expdiff1: Tensor, ub_accum: Tensor, Dn: Var):
    reg = Reg(DT.float)
    reg <<= 0.0
    qksum <<= reg
    for i in range(8 * 512 // 64):
        ub_accum[i * 64] <<= reg
    reg <<= -99999.0
    qkmax <<= reg
    reg <<= 1.0
    expdiff <<= reg
    expdiff1 <<= reg

@vf()
def nd_to_nz(src: Tensor, dst: Tensor, n_rows: Var):
    # Fractal column height = the DECLARED rows of the NZ buffer (8): `.nz()` reads the packing
    # stride from the base tensor, so packing with the window height (the old form) mismatched
    # every publish narrower than the buffer - the old simulator returned all-NaN on this kernel.
    reg = Reg(DT.e4m3)
    for i in range(n_rows):
        reg <<= src[i * BASES]
        reg_to_ub(dst[i * 32], reg, 8)

@vf()
def accum_pv(accum: Tensor, curr_pv: Tensor, expdiff: Tensor, n_rows: Var, Dn: Var):
    regs1 = RegList(DT.float, PV_CHUNK // 64)
    regs2 = RegList(DT.float, PV_CHUNK // 64)
    expdiff_reg = Reg(DT.float)
    for i in range(n_rows):
        expdiff_reg <<= expdiff[i].single()
        for col in unroll(0, 512, PV_CHUNK):
            regs1 <<= accum[i * 512 + col]
            regs2 <<= curr_pv[i * 512 + col]
            regs1 <<= regs1 * expdiff_reg
            regs1 <<= regs1 + regs2
            accum[i * 512 + col] <<= regs1
        vf_barrier(VfPipe.STORE, VfPipe.LOAD)

@vf()
def devide_sum(accum: Tensor, sum_ub: Tensor, n_rows: Var, Dn: Var):
    regs = RegList(DT.float, PV_CHUNK // 64)
    sum_reg = Reg(DT.float)
    for i in range(n_rows):
        sum_reg <<= sum_ub[i].single()
        for col in unroll(0, 512, PV_CHUNK):
            regs <<= accum[i * 512 + col]
            regs <<= regs / sum_reg
            accum[i * 512 + col] <<= regs
        vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@kernel()
def mla_hif8(
    q_nope: GM[DT.e4m3, ('B', 'H', 'Dn')], q_rope: GM[DT.e4m3, ('B', 'H', 'Dr')], k_nope: GM[DT.e4m3, ('B', 'S', 'Dn')], k_rope: GM[DT.e4m3, ('B', 'S', 'Dr')], scale_rope: GM[f32, ('B',)], output: GM[f32, ('B', 'H', 'Dn')],
    B: i32, H: i32, S: i32, Dn: i32, Dr: i32,
):
    qk_mutex = CvMutex(0, depth=2, dst_end_pipe=Pipe.V)
    p_mutex = VcMutex(1, depth=2, dst_end_pipe=Pipe.MTE1)
    pv_mutex = CvMutex(2, depth=2, dst_end_pipe=Pipe.V)

    l1qn = DBuff(DT.e4m3, [16, Dn], Position.L1)
    l1qr = DBuff(DT.e4m3, [16, Dr], Position.L1)
    l1kn = TBuff(DT.e4m3, [BASES, Dn], Position.L1)
    l1kr = DBuff(DT.e4m3, [BASES, Dr], Position.L1)
    l1p = DBuff(DT.e4m3, [16, BASES], Position.L1)
    l0cr = DBuff(DT.float, [16, BASES], Position.L0C)
    l0cn = DBuff(DT.float, [16, BASES], Position.L0C)
    l0cpv = DBuff(DT.float, [16, Dn], Position.L0C)

    ubin_rope = DBuff(DT.float, [8, BASES], Position.UB)
    ubin_nope = DBuff(DT.float, [8, BASES], Position.UB)
    ubin_pv = DBuff(DT.float, [8, Dn], Position.UB)
    ubout_pv = DBuff(DT.float, [8, Dn], Position.UB)
    ub_rope_nope_sum = Tensor(DT.float, [8, BASES], Position.UB)
    ubout_qk = DBuff(DT.e4m3, [8, BASES], Position.UB)
    ubout_qk_nz = DBuff(DT.e4m3, [8, BASES], Position.UB)
    ub_qksum = Tensor(DT.float, [1, 64], Position.UB)
    ub_qkmax = Tensor(DT.float, [1, 64], Position.UB)
    ub_expdiff = DBuff(DT.float, [1, 64], Position.UB)
    ub_accum = Tensor(DT.float, [8, Dn], Position.UB)
    accum_store_ready = SEvent(Pipe.V, Pipe.MTE3, name="accum_store_ready")
    accum_store_valid = SEvent(Pipe.MTE3, Pipe.V, preset=True, name="accum_store_valid")

    stage1_cnt = Var(0)
    stage2_cnt = Var(0)
    h_pairs = CeilDiv(H, 2)
    pairs_per_core = CeilDiv(h_pairs, GetCubeNum())
    pair_begin = Var(pairs_per_core * GetCubeIdx())
    pair_end = Min(pair_begin + pairs_per_core, h_pairs)

    for hp in range(pair_begin, pair_end):
        h1 = Var(hp * 2)
        rows = Min(2, H - h1)
        with auto_sync():  # the QK A tiles: consumed by the l1_to_l0 loads of the s loop below (autosync owns the pair)
            l1qn[0] <<= q_nope[0, h1:h1 + rows, :]
            l1qr[0] <<= q_rope[0, h1:h1 + rows, :]
        accum_store_valid.wait()
        init_buffers(ub_qksum, ub_qkmax, ub_expdiff[0], ub_expdiff[1], ub_accum, Dn)

        for s in range(0, S + BASES, BASES):
            if s < S:
                with auto_sync():
                    l1kn[stage1_cnt] <<= k_nope[0, s:s + BASES, :]
                    l1kr[stage1_cnt] <<= k_rope[0, s:s + BASES, :]
                    matmul(l0cn[stage1_cnt], l1qn[0], l1kn[stage1_cnt], splitk=128)
                    matmul(l0cr[stage1_cnt], l1qr[0], l1kr[stage1_cnt])
                    qk_mutex.lock()
                    ubin_rope[stage1_cnt] <<= l0cr[stage1_cnt]
                    ubin_nope[stage1_cnt] <<= l0cn[stage1_cnt]
                    qk_mutex.ready()

                    qk_mutex.wait()
                    scale_rope_var = Var(0.0)
                    scale_rope_var <<= scale_rope
                    scale_add_qk_vf(ubin_rope[stage1_cnt], ubin_nope[stage1_cnt], ub_rope_nope_sum, scale_rope_var, rows * BASES // 64)
                    get_p_and_max_sum(ub_rope_nope_sum, ub_qkmax, ub_qksum, ub_expdiff[stage1_cnt], ubout_qk[stage1_cnt], rows)
                    nd_to_nz(ubout_qk[stage1_cnt], ubout_qk_nz[stage1_cnt], rows)
                    qk_mutex.free()

                    p_mutex.lock()
                    if GetSubBlockIdx() == 0:
                        l1p[stage1_cnt][:rows, :] <<= ubout_qk_nz[stage1_cnt][:rows, :].nz()
                    p_mutex.ready()
                    stage1_cnt += 1
            if s > 0:
                with auto_sync():
                    p_mutex.wait()
                    matmul(l0cpv[stage2_cnt], l1p[stage2_cnt], l1kn[stage2_cnt].T, splitn=128)
                    p_mutex.free()

                    pv_mutex.lock()
                    ubin_pv[stage2_cnt] <<= l0cpv[stage2_cnt]
                    pv_mutex.ready()

                    pv_mutex.wait()
                    accum_pv(ub_accum, ubin_pv[stage2_cnt], ub_expdiff[stage2_cnt], rows, Dn)
                    pv_mutex.free()
                    stage2_cnt += 1

        devide_sum(ub_accum, ub_qksum, rows, Dn)
        bar_all()
        accum_store_ready.set()
        accum_store_ready.wait()
        if GetSubBlockIdx() == 0:
            output[0, h1:h1 + rows, :] <<= ub_accum[0:rows, :]
        accum_store_valid.set()

    return output
