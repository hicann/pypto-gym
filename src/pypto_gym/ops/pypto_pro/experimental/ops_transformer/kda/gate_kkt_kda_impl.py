# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""gate_kkt_kda — fused gate_cumsum + kkt_kda with Phase 3 M-split.

Phase 3 (the most Vec-intensive phase with 2x exp + 2x mul) is split across
two Vec sub-blocks by row range: sub_id=0 → rows [0,HC), sub_id=1 → [HC,C).
Each sub-block processes HC=64 rows instead of C=128, halving the per-sub-block
Vec workload for exp/mul.

Phases 1, 2, 4, 5 run on sub_id=0 only with full C tiles.
Phase 2/4 Cube stores restricted to sub_id=0 to avoid GM write conflicts.

Pipeline:
    Phase 1 (Vec):  g fp16→fp32 → ws_g           (sub_id=0, full C)
    Phase 2 (Cube): L @ g → ws_gcs               (sub_id=0, full [C,K])
    Phase 3 (Vec):  per sub-block [HC,K]:
                    A = k*exp(g_cs) → ws_a, B^T → ws_b  (SPLIT)
    Phase 4 (Cube): L_full = A @ B^T → ws_l      (sub_id=0, full [C,C])
    Phase 5 (Vec):  beta scale + mask + cast → L_out  (sub_id=0, full C)
"""

import torch
import pypto_pro.language as pl

from .kda_common import C, K, HC, DEVICE as _DEVICE, build_chunk_tables


# ---- UB (Vec) addresses ----
# Phase 1 (full C, sub_id=0 only)
UB_GH  = 0x00000   # [C, K] fp16  32 KB
UB_GF  = 0x08000   # [C, K] fp32  64 KB

# Phase 3 (HalfC per sub-block)
UB_KH  = 0x00000   # [HC, K] fp16  16 KB
UB_KF  = 0x04000   # [HC, K] fp32  32 KB
UB_TV3 = 0x10000   # [K, HC] fp32  32 KB
UB_G   = 0x18000   # [HC, K] fp32  32 KB
UB_E   = 0x20000   # [HC, K] fp32  32 KB

# Phase 5 (full C, sub_id=0 only)
UB_L5    = 0x00000   # [C, C] fp32  64 KB
UB_T5    = 0x10000   # [C, C] fp32  64 KB
UB_BETA5 = 0x20000   # [1, C] fp16   256 B
UB_BF5   = 0x20200   # [1, C] fp32   512 B
UB_L16_5 = 0x20400   # [C, C] fp16  32 KB

# ---- L1 (Mat) addresses ----
L1_L    = 0x00000   # [C, C] fp32 NZ  64 KB
L1_G    = 0x10000   # [C, K] fp32 NZ  64 KB
L1_A    = 0x00000   # [C, K] fp32 NZ  64 KB  (= L1_L)
L1_B    = 0x10000   # [K, C] fp32 ZN  64 KB  (= L1_G)


@pl.jit(auto_mutex=True)
def gate_kkt_kda_kernel(
    g: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    beta: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    L_mat: pl.Tensor[[C, C], pl.DT_FP32],
    mask: pl.Tensor[[C, C], pl.DT_FP32],
    g_cs: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    L_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    ws_g: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    ws_gcs: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    ws_a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    ws_b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    ws_l: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    cu_seqlens: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
    chunk_tbase: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
    chunk_valid: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
):
    """Fused gate_cumsum + kkt_kda with Phase 3 M-split."""
    T = g.shape[1]
    HV_dim = g.shape[2]
    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    sub_id = pl.get_subblock_idx()
    ro = sub_id * HC

    num_seqs = cu_seqlens.shape[0] - 1
    if num_seqs == 1:
        num_chunks = (T + C - 1) // C
    else:
        num_chunks = chunk_tbase.shape[0]
    total_work = num_chunks * HV_dim

    # ── Phase 1 tiles (full C, sub_id=0 only) ──
    t_gh = pl.make_tile_group(
        type=pl.TileType(shape=[C, K], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Vec,
                         pad=pl.TilePad.zero, valid_shape=[-1, -1]),
        addrs=UB_GH, mutex_ids=[0])
    t_gf = pl.make_tile_group(
        type=pl.TileType(shape=[C, K], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=UB_GF, mutex_ids=[1])

    # ── Phase 2 tiles (Cube, full C) ──
    L_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=L1_L, mutex_ids=[2])
    g_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[C, K], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=L1_G, mutex_ids=[3])
    L_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0, mutex_ids=[4])
    g_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[C, K], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0, mutex_ids=[5])
    acc_g = pl.make_tile_group(
        type=pl.TileType(shape=[C, K], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                         fractal=1024),
        addrs=0x0, mutex_ids=[6])

    # ── Phase 3 tiles (HalfC per sub-block) ──
    t_kh = pl.make_tile_group(
        type=pl.TileType(shape=[HC, K], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Vec,
                         pad=pl.TilePad.zero, valid_shape=[-1, -1]),
        addrs=UB_KH, mutex_ids=[7])
    t_kf = pl.make_tile_group(
        type=pl.TileType(shape=[HC, K], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=UB_KF, mutex_ids=[8])
    t_gk = pl.make_tile_group(
        type=pl.TileType(shape=[HC, K], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec,
                         pad=pl.TilePad.zero, valid_shape=[-1, -1]),
        addrs=UB_G, mutex_ids=[9])
    t_e = pl.make_tile_group(
        type=pl.TileType(shape=[HC, K], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=UB_E, mutex_ids=[10])
    t_tv3 = pl.make_tile_group(
        type=pl.TileType(shape=[K, HC], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=UB_TV3, mutex_ids=[22])

    # ── Phase 4 tiles (Cube, full C) ──
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[C, K], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=L1_A, mutex_ids=[11])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[K, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=L1_B, mutex_ids=[12])
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[C, K], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0, mutex_ids=[13])
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[K, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0, mutex_ids=[14])
    acc_k = pl.make_tile_group(
        type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                         fractal=1024),
        addrs=0x0, mutex_ids=[15])

    # ── Phase 5 tiles (full C, sub_id=0 only) ──
    t_l5 = pl.make_tile_group(
        type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=UB_L5, mutex_ids=[16])
    t_t5 = pl.make_tile_group(
        type=pl.TileType(shape=[C, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=UB_T5, mutex_ids=[17])
    t_beta5 = pl.make_tile_group(
        type=pl.TileType(shape=[1, C], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Vec,
                         pad=pl.TilePad.zero, valid_shape=[-1, -1]),
        addrs=UB_BETA5, mutex_ids=[18])
    t_bf5 = pl.make_tile_group(
        type=pl.TileType(shape=[1, C], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=UB_BF5, mutex_ids=[19])
    t_l16_5 = pl.make_tile_group(
        type=pl.TileType(shape=[C, C], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Vec,
                         pad=pl.TilePad.zero, valid_shape=[-1, -1]),
        addrs=UB_L16_5, mutex_ids=[20])
    t_bf5_col = pl.make_tile_group(
        type=pl.TileType(shape=[C, 1], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec, layout=pl.DN),
        addrs=UB_BF5, mutex_ids=[21])

    # ── Tile references ──
    gh = t_gh.current()
    gf = t_gf.current()
    cur_L = L_l1.current()
    cur_g = g_l1.current()
    al_g = L_l0a.current()
    br_g = g_l0b.current()
    ac_g = acc_g.current()

    kh = t_kh.current()
    kf = t_kf.current()
    gk = t_gk.current()
    e = t_e.current()
    tv3 = t_tv3.current()
    cur_a = a_l1.current()
    cur_b = b_l1.current()
    al_k = a_l0a.current()
    br_k = b_l0b.current()
    ac_k = acc_k.current()

    lv5 = t_l5.current()
    tv5 = t_t5.current()
    bv5 = t_beta5.current()
    bf5 = t_bf5.current()
    bf5_col = t_bf5_col.current()
    l16_5 = t_l16_5.current()

    # ── Work loop ──
    for work_id in pl.range(core_id, total_work, num_cores):
        chunk_id = work_id // HV_dim
        head_id = work_id % HV_dim

        if num_seqs == 1:
            t_base = chunk_id * C
            valid_size = pl.min(T - t_base, C)
        else:
            t_base = pl.getval(chunk_tbase, chunk_id)
            valid_size = pl.getval(chunk_valid, chunk_id)

        # ================================================================
        #  Phase 1 (Vec): g fp16→fp32 → ws_g  (sub_id=0 only, full C)
        # ================================================================
        if sub_id == 0:
            with pl.section_vector():
                if valid_size < C:
                    pl.expands(gf, 0.0)
                    pl.store(ws_g, gf, [core_id * C, 0])
                    pl.store(ws_a, gf, [core_id * C, 0])
                    pl.store(ws_b, gf, [core_id * C, 0])
                pl.set_validshape(gh, [valid_size, K])
                pl.load(gh, g, [0, t_base, head_id, 0], order=[1, 3])
                pl.set_validshape(gh, [C, K])
                pl.cast(gf, gh, mode=pl.RoundMode.CAST_NONE)
                pl.set_validshape(gf, [valid_size, K])
                pl.store(ws_g, gf, [core_id * C, 0])
                pl.set_validshape(gf, [C, K])

                pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
        else:
            with pl.section_vector():
                pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=0)

        # ================================================================
        #  Phase 2 (Cube): L @ g → ws_gcs (full [C,K], sub_id=0 only)
        # ================================================================
        if sub_id == 0:
            with pl.section_cube():
                pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=0)

                pl.load(cur_L, L_mat, [0, 0])
                pl.set_validshape(cur_g, [C, K])
                pl.load(cur_g, ws_g, [core_id * C, 0])
                pl.move(al_g, cur_L)
                pl.move(br_g, cur_g)
                pl.matmul(ac_g, al_g, br_g)
                pl.set_validshape(ac_g, [C, K])
                pl.store(ws_gcs, ac_g, [core_id * C, 0])
                pl.set_validshape(ac_g, [valid_size, K])
                pl.store(g_cs, ac_g, [0, head_id, t_base, 0], order=[2, 3])

                pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=1)
        else:
            with pl.section_cube():
                pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=0)
                pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=1)

        # ================================================================
        #  Phase 3 (Vec): per sub-block [HC,K]
        # ================================================================
        valid_rows = pl.max(0, pl.min(HC, valid_size - ro))
        with pl.section_vector():
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=1)

            pl.set_validshape(kh, [valid_rows, K])
            pl.load(kh, k, [0, t_base + ro, head_id, 0], order=[1, 3])
            pl.set_validshape(kh, [HC, K])
            pl.cast(kf, kh, mode=pl.RoundMode.CAST_NONE)

            pl.set_validshape(gk, [valid_rows, K])
            pl.load(gk, ws_gcs, [core_id * C + ro, 0])
            pl.set_validshape(gk, [HC, K])

            pl.exp(e, gk)
            pl.mul(e, kf, e)
            pl.set_validshape(e, [valid_rows, K])
            pl.store(ws_a, e, [core_id * C + ro, 0])
            pl.set_validshape(e, [HC, K])

            pl.neg(gk, gk)
            pl.exp(e, gk)
            pl.mul(e, kf, e)
            pl.set_validshape(e, [valid_rows, K])
            pl.store(ws_b, e, [core_id * C + ro, 0])
            pl.set_validshape(e, [HC, K])

            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=2)

        # ================================================================
        #  Phase 4 (Cube): L_full = A @ B^T → ws_l (sub_id=0 only)
        # ================================================================
        if sub_id == 0:
            with pl.section_cube():
                pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=2)

                pl.load(cur_a, ws_a, [core_id * C, 0])
                pl.load(cur_b, ws_b, [core_id * C, 0], order=[1, 0])
                pl.move(al_k, cur_a)
                pl.move(br_k, cur_b)
                pl.matmul(ac_k, al_k, br_k)
                pl.store(ws_l, ac_k, [core_id * C, 0])

                pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=3)
        else:
            with pl.section_cube():
                pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=2)
                pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=3)

        pl.system.bar_all()

        # ================================================================
        #  Phase 5 (Vec): sub_id=0 only, full C
        # ================================================================
        if sub_id == 0:
            with pl.section_vector():
                pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=3)

                pl.load(lv5, ws_l, [core_id * C, 0])

                if valid_size < C:
                    pl.expands(bv5, 0.0)
                pl.set_validshape(bv5, [1, valid_size])
                pl.load(bv5, beta, [0, head_id, t_base])
                pl.set_validshape(bv5, [1, C])
                pl.cast(bf5, bv5, mode=pl.RoundMode.CAST_NONE)

                pl.expand_mul(tv5, lv5, bf5_col, dim=0)

                pl.load(lv5, mask, [0, 0])
                pl.mul(tv5, tv5, lv5)

                pl.set_validshape(l16_5, [C, C])
                pl.cast(l16_5, tv5, mode=pl.RoundMode.CAST_ROUND)
                pl.set_validshape(l16_5, [valid_size, C])
                pl.store(L_out, l16_5, [0, head_id, t_base, 0], order=[2, 3])

        pl.system.bar_all()


def run_gate_kkt_kda(g, k, beta, L_mat, mask, g_cs, L_out, num_cores, cu_seqlens=None):
    """Launch the fused gate_cumsum + kkt_kda kernel.

    Args:
        g:       [B, T, HV, K]  fp16   raw gate values (BSND)
        k:       [B, T, HV, K]  fp16   keys (BSND)
        beta:    [B, HV, T]     fp16   beta head-major (BNTD)
        L_mat:   [C, C]         fp32   lower-triangular ones (constant)
        mask:    [C, C]         fp32   strict lower-tri mask
        g_cs:    [B, HV, T, K]  fp32   cumulative gate sum output (BNTD)
        L_out:   [B, HV, T, C]  fp16   output (zeroed, BNTD)
        num_cores: int          number of AI cores to launch
        cu_seqlens: list[int] | None   cumulative sequence lengths
    """
    B, T, HV_dim, Kd = g.shape

    g = g.to(_DEVICE)
    k = k.to(_DEVICE)
    beta = beta.to(_DEVICE)
    L_mat = L_mat.to(_DEVICE)
    mask = mask.to(_DEVICE)
    g_cs = g_cs.to(_DEVICE)
    L_out = L_out.to(_DEVICE)

    cu_seqlens_t, chunk_tbase_t, chunk_valid_t, _ = build_chunk_tables(T, cu_seqlens, C, _DEVICE)

    nc = num_cores
    ws_g   = torch.empty(nc * C, Kd, device=_DEVICE, dtype=torch.float32)
    ws_gcs = torch.empty(nc * C, Kd, device=_DEVICE, dtype=torch.float32)
    ws_a   = torch.empty(nc * C, Kd, device=_DEVICE, dtype=torch.float32)
    ws_b   = torch.empty(nc * C, Kd, device=_DEVICE, dtype=torch.float32)
    ws_l   = torch.empty(nc * C, C, device=_DEVICE, dtype=torch.float32)

    gate_kkt_kda_kernel[None, nc](
        g, k, beta, L_mat, mask, g_cs, L_out,
        ws_g, ws_gcs, ws_a, ws_b, ws_l,
        cu_seqlens_t, chunk_tbase_t, chunk_valid_t,
    )
    torch.npu.synchronize()
