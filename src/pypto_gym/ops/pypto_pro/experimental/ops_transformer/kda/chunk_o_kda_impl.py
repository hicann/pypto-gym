# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""chunk_o_kda operator — fused single-kernel output pass for KDA.

Implements Stage 6 of the KDA (Kernelized Decay Attention) pipeline.

Math (per chunk of size C, per head):
    q_eff = q * exp(g_cs)                 # [C, K]
    k_eff = k * exp(-g_cs)                # [C, K]
    inter = q_eff @ S                     # [C, V]   S = s_snapshots[ci, head]
    Aqk   = tril(q_eff @ k_eff^T, k=0)    # [C, C]   inclusive lower triangle
    o     = inter + Aqk @ v_corr          # [C, V]

Fused-kernel data flow (5 phases, alternating Vec/Cube via cross-core sync):
    Phase 1 (Vec):  q_eff, k_eff, vcorr_f, s_f → per-core GM workspace
    Phase 2 (Cube): inter = q_eff @ S;  aqk_raw = q_eff @ k_eff^T → workspace
    Phase 3 (Vec):  aqk = aqk_raw * mask → workspace
    Phase 4 (Cube): tmp = aqk @ v_corr → workspace
    Phase 5 (Vec):  o = inter + tmp; cast fp16; store to GM output
"""

import torch
import pypto_pro.language as pl

from .kda_common import C, K, V, HC, DEVICE as _DEVICE, build_chunk_tables


# ---- L1 (Mat) addresses ----
L1_Q = 0x00000            # q_eff [C, K] fp32 NZ
L1_S = 0x10000            # S     [K, V] fp32 NZ
L1_K = 0x20000            # k_eff [C, K] fp32 ZN (transposed)
L1_AQK = 0x00000          # aqk    [C, C] fp32 NZ
L1_VC  = 0x10000          # v_corr [C, V] fp32 NZ

# ---- UB (Vec) addresses ----
UB_H = 0x00000            # fp16 load buffer       32 KB
UB_G = 0x08000            # g_cs fp32              64 KB
UB_F = 0x18000            # fp32 cast / product    64 KB
UB_E = 0x28000            # fp32 exp scratch       64 KB
UB_MASK = 0x00000         # mask [HC, C] fp32      32 KB
UB_AQK  = 0x08000         # aqk  [HC, C] fp32      32 KB
UB_INTER = 0x00000        # [HC, V] fp32           32 KB
UB_TMP   = 0x08000        # [HC, V] fp32           32 KB
UB_OH    = 0x10000        # [HC, V] fp16           16 KB


def _mat_nz(shape, dtype):
    return pl.TileType(shape=shape, dtype=dtype,
                       target_memory=pl.MemorySpace.Mat, layout=pl.NZ)


def _mat_zn(shape, dtype):
    return pl.TileType(shape=shape, dtype=dtype,
                       target_memory=pl.MemorySpace.Mat, layout=pl.ZN)


def _left(shape, dtype):
    return pl.TileType(shape=shape, dtype=dtype,
                       target_memory=pl.MemorySpace.Left, layout=pl.NZ)


def _right(shape, dtype):
    return pl.TileType(shape=shape, dtype=dtype,
                       target_memory=pl.MemorySpace.Right, layout=pl.ZN)


def _acc(shape):
    return pl.TileType(shape=shape, dtype=pl.DT_FP32,
                       target_memory=pl.MemorySpace.Acc,
                       layout=pl.NZ, fractal=1024)


def _vec(shape, dtype):
    return pl.TileType(shape=shape, dtype=dtype,
                       target_memory=pl.MemorySpace.Vec)


@pl.jit(auto_mutex=True)
def chunk_o_kda_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    g: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    vcorr: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    s_snap: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    mask: pl.Tensor[[C, C], pl.DT_FP32],
    o: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    ws_q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    ws_k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    ws_v: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    ws_s: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    ws_inter: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    ws_aqk: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    num_heads: pl.DT_INT32,
    seq_len: pl.DT_INT64,
    cu_seqlens: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
    num_seqs: pl.DT_INT32,
    chunk_tbase: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
    chunk_valid: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
):
    """Fused chunk_o_kda: 5 phases in one kernel via Cube<->Vec cross-core sync."""

    # -- Cube L1 tile groups (Phase 2: gemm qk) --
    q_l1 = pl.make_tile_group(type=_mat_nz([C, K], pl.DT_FP32), addrs=L1_Q, mutex_ids=[0])
    s_l1 = pl.make_tile_group(type=_mat_nz([K, V], pl.DT_FP32), addrs=L1_S, mutex_ids=[1])
    k_l1 = pl.make_tile_group(type=_mat_zn([K, C], pl.DT_FP32), addrs=L1_K, mutex_ids=[2])
    q_l0a = pl.make_tile_group(type=_left([C, K], pl.DT_FP32), addrs=0x0, mutex_ids=[3])
    b_l0b = pl.make_tile_group(type=_right([K, V], pl.DT_FP32), addrs=0x0, mutex_ids=[4])
    inter_acc = pl.make_tile_group(type=_acc([C, V]), addrs=0x0, mutex_ids=[5])
    aqk_acc = pl.make_tile_group(type=_acc([C, C]), addrs=0x10000, mutex_ids=[6])

    # -- Cube L1 tile groups (Phase 4: gemm av; L1 reused after Phase 2) --
    aqk_l1 = pl.make_tile_group(type=_mat_nz([C, C], pl.DT_FP32), addrs=L1_AQK, mutex_ids=[7])
    v_l1 = pl.make_tile_group(type=_mat_nz([C, V], pl.DT_FP32), addrs=L1_VC, mutex_ids=[8])
    a_l0a = pl.make_tile_group(type=_left([C, C], pl.DT_FP32), addrs=0x0, mutex_ids=[9])
    b_l0b2 = pl.make_tile_group(type=_right([C, V], pl.DT_FP32), addrs=0x0, mutex_ids=[10])
    tmp_acc = pl.make_tile_group(type=_acc([C, V]), addrs=0x0, mutex_ids=[11])

    # -- Vec UB tile groups (Phase 1: gate, HalfC per sub-block) --
    ub_h = pl.make_tile_group(
        type=pl.TileType(shape=[HC, K], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Vec,
                         pad=pl.TilePad.zero, valid_shape=[-1, -1]),
        addrs=UB_H, mutex_ids=[12])
    ub_g = pl.make_tile_group(
        type=pl.TileType(shape=[HC, K], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec,
                         pad=pl.TilePad.zero, valid_shape=[-1, -1]),
        addrs=UB_G, mutex_ids=[13])
    ub_f = pl.make_tile_group(type=_vec([HC, K], pl.DT_FP32), addrs=UB_F, mutex_ids=[14])
    ub_e = pl.make_tile_group(type=_vec([HC, K], pl.DT_FP32), addrs=UB_E, mutex_ids=[15])
    ub_h_snap = pl.make_tile_group(
        type=pl.TileType(shape=[K, V], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Vec),
        addrs=0x20000, mutex_ids=[21])
    ub_f_snap = pl.make_tile_group(type=_vec([K, V], pl.DT_FP32), addrs=0x30000, mutex_ids=[22])

    # -- Vec UB tile groups (Phase 3: mask; HalfC per sub-block, UB reused) --
    ub_mask = pl.make_tile_group(type=_vec([HC, C], pl.DT_FP32), addrs=UB_MASK, mutex_ids=[16])
    ub_aqk = pl.make_tile_group(type=_vec([HC, C], pl.DT_FP32), addrs=UB_AQK, mutex_ids=[17])

    # -- Vec UB tile groups (Phase 5: combine; HalfC per sub-block, UB reused) --
    ub_inter = pl.make_tile_group(type=_vec([HC, V], pl.DT_FP32), addrs=UB_INTER, mutex_ids=[18])
    ub_tmp = pl.make_tile_group(type=_vec([HC, V], pl.DT_FP32), addrs=UB_TMP, mutex_ids=[19])
    ub_oh = pl.make_tile_group(
        type=pl.TileType(shape=[HC, V], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Vec,
                         pad=pl.TilePad.zero, valid_shape=[-1, -1]),
        addrs=UB_OH, mutex_ids=[20])

    core_id = pl.get_block_idx() // pl.get_subblock_num()
    num_cores = pl.get_block_num()
    sub_id = pl.get_subblock_idx()

    if num_seqs == 1:
        total_chunks = (seq_len + C - 1) // C
    else:
        total_chunks = chunk_tbase.shape[0]
    total_work = total_chunks * num_heads

    for work_id in pl.range(core_id, total_work, num_cores):
        chunk_id = work_id // num_heads
        head_id = work_id % num_heads

        global_chunk_id = chunk_id
        if num_seqs == 1:
            t_base = chunk_id * C
            valid_size = pl.min(seq_len - t_base, C)
        else:
            t_base = pl.getval(chunk_tbase, chunk_id)
            valid_size = pl.getval(chunk_valid, chunk_id)

        # ================================================================
        #  Phase 1 (Vec): gate q/k, cast v/s (both sub-blocks, HalfC each)
        # ================================================================
        valid_rows = pl.max(0, pl.min(HC, valid_size - sub_id * HC))
        t_offset = t_base + sub_id * HC

        with pl.section_vector():
            bh = ub_h.current()
            bg = ub_g.current()
            bf = ub_f.current()
            be = ub_e.current()

            pl.set_validshape(bg, [valid_rows, K])
            pl.load(bg, g, [0, head_id, t_offset, 0], order=[2, 3])
            pl.set_validshape(bg, [HC, K])
            pl.exp(be, bg)

            pl.set_validshape(bh, [valid_rows, K])
            pl.load(bh, q, [0, t_offset, head_id, 0], order=[1, 3])
            pl.set_validshape(bh, [HC, K])
            pl.cast(bf, bh, mode=pl.RoundMode.CAST_NONE)
            pl.mul(bf, bf, be)
            pl.set_validshape(bf, [valid_rows, K])
            pl.store(ws_q, bf, [core_id * C + sub_id * HC, 0])
            pl.set_validshape(bf, [HC, K])

            pl.neg(be, bg)
            pl.exp(be, be)
            pl.set_validshape(bh, [valid_rows, K])
            pl.load(bh, k, [0, t_offset, head_id, 0], order=[1, 3])
            pl.set_validshape(bh, [HC, K])
            pl.cast(bf, bh, mode=pl.RoundMode.CAST_NONE)
            pl.mul(bf, bf, be)
            pl.set_validshape(bf, [valid_rows, K])
            pl.store(ws_k, bf, [core_id * C + sub_id * HC, 0])
            pl.set_validshape(bf, [HC, K])

            pl.set_validshape(bh, [valid_rows, V])
            pl.load(bh, vcorr, [0, head_id, t_offset, 0], order=[2, 3])
            pl.set_validshape(bh, [HC, V])
            pl.cast(bf, bh, mode=pl.RoundMode.CAST_NONE)
            pl.set_validshape(bf, [valid_rows, V])
            pl.store(ws_v, bf, [core_id * C + sub_id * HC, 0])
            pl.set_validshape(bf, [HC, V])

            if sub_id == 0:
                bh_s = ub_h_snap.current()
                bf_s = ub_f_snap.current()
                pl.load(bh_s, s_snap, [head_id, global_chunk_id, 0, 0], order=[2, 3])
                pl.cast(bf_s, bh_s, mode=pl.RoundMode.CAST_NONE)
                pl.store(ws_s, bf_s, [core_id * K, 0])

            pl.system.bar_all()
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=0)

        # ================================================================
        #  Phase 2 (Cube): inter = q_eff @ S;  aqk_raw = q_eff @ k_eff^T
        # ================================================================
        with pl.section_cube():
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=0)

            cur_q_l1 = q_l1.current()
            cur_s_l1 = s_l1.current()
            cur_k_l1 = k_l1.current()
            cur_q_l0a = q_l0a.current()
            cur_b_l0b = b_l0b.current()
            cur_inter_acc = inter_acc.current()
            cur_aqk_acc = aqk_acc.current()

            pl.load(cur_q_l1, ws_q, [core_id * C, 0])

            pl.load(cur_s_l1, ws_s, [core_id * K, 0])
            pl.move(cur_q_l0a, cur_q_l1)
            pl.move(cur_b_l0b, cur_s_l1)
            pl.matmul(cur_inter_acc, cur_q_l0a, cur_b_l0b)
            pl.store(ws_inter, cur_inter_acc, [core_id * C, 0])

            pl.load(cur_k_l1, ws_k, [core_id * C, 0], order=[1, 0])
            pl.move(cur_q_l0a, cur_q_l1)
            pl.move(cur_b_l0b, cur_k_l1)
            pl.matmul(cur_aqk_acc, cur_q_l0a, cur_b_l0b)
            pl.move(ub_aqk.current(), cur_aqk_acc,
                    acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)

            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=1)

        # ================================================================
        #  Phase 3 (Vec): aqk = aqk_raw * mask  (both sub-blocks, HalfC each)
        # ================================================================
        with pl.section_vector():
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=1)

            bm = ub_mask.current()
            ba = ub_aqk.current()

            pl.load(bm, mask, [sub_id * HC, 0])
            pl.mul(ba, ba, bm)
            pl.store(ws_aqk, ba, [core_id * C + sub_id * HC, 0])

            pl.system.bar_all()
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=2)

        # ================================================================
        #  Phase 4 (Cube): tmp = aqk @ v_corr
        # ================================================================
        with pl.section_cube():
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=2)

            cur_aqk_l1 = aqk_l1.current()
            cur_v_l1 = v_l1.current()
            cur_a_l0a = a_l0a.current()
            cur_b_l0b2 = b_l0b2.current()
            cur_tmp_acc = tmp_acc.current()

            pl.load(cur_aqk_l1, ws_aqk, [core_id * C, 0])
            pl.load(cur_v_l1, ws_v, [core_id * C, 0])
            pl.move(cur_a_l0a, cur_aqk_l1)
            pl.move(cur_b_l0b2, cur_v_l1)
            pl.matmul(cur_tmp_acc, cur_a_l0a, cur_b_l0b2)
            pl.move(ub_tmp.current(), cur_tmp_acc,
                    acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)

            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=3)

        # ================================================================
        #  Phase 5 (Vec): o = inter + tmp; cast fp16; store (both sub-blocks)
        # ================================================================
        with pl.section_vector():
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=3)

            bi = ub_inter.current()
            bt = ub_tmp.current()
            bo = ub_oh.current()

            valid_rows = pl.max(0, pl.min(HC, valid_size - sub_id * HC))
            pl.load(bi, ws_inter, [core_id * C + sub_id * HC, 0])
            pl.add(bi, bi, bt)
            pl.set_validshape(bo, [HC, V])
            pl.cast(bo, bi, mode=pl.RoundMode.CAST_ROUND)
            pl.set_validshape(bo, [valid_rows, V])
            pl.store(o, bo, [0, t_base + sub_id * HC, head_id, 0], order=[1, 3])

            pl.system.bar_all()


def run_chunk_o_kda(q, k, v_corr, s_snapshots, g_cs, mask, o,
                    num_chunks, num_cores, cu_seqlens_list=None):
    """Run the fused chunk_o_kda kernel on NPU.

    Args:
        q:             [1, T, HV, K]  fp16   queries (BSND)
        k:             [1, T, HV, K]  fp16   keys (BSND)
        v_corr:        [1, HV, T, V]  fp16   corrected values (BNTD)
        s_snapshots:   [HV, n_chunks, K, V]  fp16   state snapshots (HNCV)
        g_cs:          [1, HV, T, K]  fp32   cumulative gate sum (BNTD)
        mask:          [C, C]         fp32   inclusive lower-triangular mask
        o:             [1, T, HV, V]  fp16   output (BSND)
        num_chunks:    int
        num_cores:     int
        cu_seqlens_list: list[int] | None
    """
    q = q.to(_DEVICE)
    k = k.to(_DEVICE)
    v_corr = v_corr.to(_DEVICE)
    s_snapshots = s_snapshots.to(_DEVICE)
    g_cs = g_cs.to(_DEVICE)
    mask = mask.to(_DEVICE)
    o = o.to(_DEVICE)

    n_cores = num_cores
    T = q.shape[1]
    HV = q.shape[2]

    cu_seqlens_tensor, chunk_tbase_t, chunk_valid_t, _ = build_chunk_tables(T, cu_seqlens_list, C, _DEVICE)
    num_seqs = 1 if cu_seqlens_list is None else len(cu_seqlens_list) - 1

    ws_q = torch.empty(n_cores * C, K, device=_DEVICE, dtype=torch.float32)
    ws_k = torch.empty(n_cores * C, K, device=_DEVICE, dtype=torch.float32)
    ws_v = torch.empty(n_cores * C, V, device=_DEVICE, dtype=torch.float32)
    ws_s = torch.empty(n_cores * K, V, device=_DEVICE, dtype=torch.float32)
    ws_inter = torch.empty(n_cores * C, V, device=_DEVICE, dtype=torch.float32)
    ws_aqk = torch.empty(n_cores * C, C, device=_DEVICE, dtype=torch.float32)

    chunk_o_kda_kernel[None, n_cores](
        q, k, g_cs, v_corr, s_snapshots, mask, o,
        ws_q, ws_k, ws_v, ws_s, ws_inter, ws_aqk,
        HV, T, cu_seqlens_tensor, num_seqs,
        chunk_tbase_t, chunk_valid_t,
    )
    torch.npu.synchronize()
