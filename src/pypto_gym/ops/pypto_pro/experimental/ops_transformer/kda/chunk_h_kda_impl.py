# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""chunk_h_kda operator — sequential recurrent state pass for KDA.

Implements Stage 5 of the KDA (Kernelized Decay Attention) pipeline:
the hidden-state recurrence that snapshots S entering each chunk and
computes the corrected values v_corr.

Math (per chunk, per head):
    v_corr  = u - w @ S                              # [C, V]
    k_rest  = k * exp(g_total - g_cs)                # [C, K]
    S_new   = exp(g_total) * S + k_rest^T @ v_corr   # [K, V]

where g_total = g_cs[last-of-chunk, :] and S is the [K, V] recurrent
state that propagates sequentially across chunks within each sequence.
"""

import torch
import pypto_pro.language as pl

from .kda_common import (C, K, V, HC, K_DIM, V_DIM, DEVICE as _DEVICE,
                         make_cu_seqlens_tensor, alloc_chunk_h_workspaces)


# ---- L1 (Mat) addresses ----
L1_W   = 0x00000
L1_S   = 0x10000
L1_K   = 0x20000
L1_V   = 0x28000

# ---- UB (Vec) addresses ----
UB_A  = 0x00000           # [HC, V] fp32   32 KB
UB_B  = 0x08000           # [HC, V] fp16   16 KB
UB_C  = 0x0C000           # [HC, V] fp32   32 KB
UB_D  = 0x14000           # [1, K]  fp32   512 B
UB_WS = 0x14200           # [HC, V] fp32   32 KB  (WS from Acc via DualModeSplitM)
UB_KV = 0x1C200           # [HC, V] fp32   32 KB  (KV from Acc via DualModeSplitM)
UB_S_RES = 0x24200        # [HC, V] fp32   32 KB  (S state resident)
UB_K_PF = 0x2C200         # [HC, K] fp16   16 KB  (k prefetch)


@pl.jit(auto_mutex=True)
def chunk_h_kda_kernel(
    k_t: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    w_t: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    u_t: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    g_t: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    s_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    vcorr_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    ws_s: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    ws_k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    ws_v: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    ws_s_f16: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    ws_w_f16: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    num_heads: pl.DT_INT32,
    num_seqs: pl.DT_INT32,
    seq_len: pl.DT_INT64,
    cu_seqlens: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
    chunk_offsets: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
):
    """Sequential recurrent state pass for KDA."""
    # -- L1 tile groups --
    w_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[C, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=L1_W, mutex_ids=[0])
    s_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[K, V], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=L1_S, mutex_ids=[1])
    k_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[K, C], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=L1_K, mutex_ids=[5])
    v_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[C, V], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=L1_V, mutex_ids=[6])

    w_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[C, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0, mutex_ids=[2])
    s_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[K, V], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0, mutex_ids=[3])
    ws_acc = pl.make_tile_group(
        type=pl.TileType(shape=[C, V], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024),
        addrs=0x0, mutex_ids=[4])

    k_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[K, C], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0, mutex_ids=[7])
    v_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[C, V], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0, mutex_ids=[8])
    kv_acc = pl.make_tile_group(
        type=pl.TileType(shape=[K, V], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024),
        addrs=0x0, mutex_ids=[9])

    # -- UB tile groups (HalfC per sub-block) --
    ub_a = pl.make_tile_group(
        type=pl.TileType(shape=[HC, V], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=UB_A, mutex_ids=[10])
    ub_b = pl.make_tile_group(
        type=pl.TileType(shape=[HC, V], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec,
                         pad=pl.TilePad.zero, valid_shape=[-1, -1]),
        addrs=UB_B, mutex_ids=[11])
    ub_c = pl.make_tile_group(
        type=pl.TileType(shape=[HC, V], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec,
                         pad=pl.TilePad.zero, valid_shape=[-1, -1]),
        addrs=UB_C, mutex_ids=[12])
    ub_d = pl.make_tile_group(
        type=pl.TileType(shape=[1, K], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=UB_D, mutex_ids=[13])
    ub_ws = pl.make_tile_group(
        type=pl.TileType(shape=[HC, V], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=UB_WS, mutex_ids=[14])
    ub_kv = pl.make_tile_group(
        type=pl.TileType(shape=[HC, V], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=UB_KV, mutex_ids=[16])
    ub_d_col = pl.make_tile_group(
        type=pl.TileType(shape=[K, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN),
        addrs=UB_D, mutex_ids=[15])
    ub_s_res = pl.make_tile_group(
        type=pl.TileType(shape=[HC, V], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=UB_S_RES, mutex_ids=[17])
    ub_k_pf = pl.make_tile_group(
        type=pl.TileType(shape=[HC, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec,
                         pad=pl.TilePad.zero, valid_shape=[-1, -1]),
        addrs=UB_K_PF, mutex_ids=[18])

    core_id = pl.get_block_idx() // pl.get_subblock_num()
    num_cores = pl.get_block_num()
    sub_id = pl.get_subblock_idx()
    ro = sub_id * HC

    total_work = num_seqs * num_heads

    for work_id in pl.range(core_id, total_work, num_cores):
        seq_idx = work_id // num_heads
        head_id = work_id % num_heads
        bos = pl.getval(cu_seqlens, seq_idx)
        eos = pl.getval(cu_seqlens, seq_idx + 1)
        seq_len_seq = eos - bos
        num_chunks_seq = (seq_len_seq + C - 1) // C
        chunk_offset = pl.getval(chunk_offsets, seq_idx)

        with pl.section_vector():
            s_res = ub_s_res.current()
            pl.expands(s_res, 0.0)
            zero_b = ub_b.current()
            pl.cast(zero_b, s_res)
            pl.store(ws_s_f16, zero_b, [core_id * K + ro, 0])
            zero_a = ub_a.current()
            pl.load(zero_a, w_t, [0, head_id, bos + ro, 0], order=[2, 3])
            pl.cast(zero_b, zero_a)
            pl.store(ws_w_f16, zero_b, [core_id * C + ro, 0])
            first_valid = pl.min(seq_len_seq, C)
            first_rows = pl.max(0, pl.min(HC, first_valid - ro))
            k_pf = ub_k_pf.current()
            pl.set_validshape(k_pf, [first_rows, K])
            pl.load(k_pf, k_t, [0, bos + ro, head_id, 0], order=[1, 3])
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=3)

        for ci in pl.range(num_chunks_seq):
            t_base = bos + ci * C
            valid_size = pl.min(eos - t_base, C)
            valid_rows = pl.max(0, pl.min(HC, valid_size - ro))
            next_valid = pl.min(eos - (t_base + C), C)
            next_rows = pl.max(0, pl.min(HC, next_valid - ro))

            # ================================================================
            #  Phase 1 (Cube): WS = W @ S  ->  ub_ws (via AccToVecMode)
            # ================================================================
            with pl.section_cube():
                pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=3)

                cur_w_l1 = w_l1.current()
                cur_s_l1 = s_l1.current()
                cur_w_l0a = w_l0a.current()
                cur_s_l0b = s_l0b.current()
                cur_ws_acc = ws_acc.current()
                cur_ub_ws = ub_ws.current()

                pl.load(cur_w_l1, ws_w_f16, [core_id * C, 0])
                pl.load(cur_s_l1, ws_s_f16, [core_id * K, 0])
                pl.move(cur_w_l0a, cur_w_l1)
                pl.move(cur_s_l0b, cur_s_l1)
                pl.matmul(cur_ws_acc, cur_w_l0a, cur_s_l0b)
                pl.move(cur_ub_ws, cur_ws_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)

                pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

            # ================================================================
            #  Phase 2 (Vec): snapshot, v_corr, k_rest  (per sub-block)
            # ================================================================
            with pl.section_vector():
                pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=0)

                a = ub_a.current()
                b = ub_b.current()
                c = ub_c.current()
                d = ub_d.current()
                ws = ub_ws.current()
                s_res = ub_s_res.current()

                pl.cast(b, s_res)
                pl.store(s_out, b, [head_id, chunk_offset + ci, ro, 0], order=[2, 3])

                pl.set_validshape(c, [valid_rows, V])
                pl.load(c, u_t, [0, head_id, t_base + ro, 0], order=[2, 3])
                pl.set_validshape(c, [HC, V])
                pl.sub(c, c, ws)
                pl.cast(b, c)
                pl.set_validshape(b, [valid_rows, V])
                pl.store(vcorr_out, b, [0, head_id, t_base + ro, 0], order=[2, 3])
                pl.set_validshape(b, [HC, V])

                k_pf = ub_k_pf.current()
                pl.cast(a, k_pf)
                pl.set_validshape(c, [valid_rows, K])
                pl.load(ws, g_t, [0, head_id, t_base + ro, 0], order=[2, 3])
                pl.set_validshape(ws, [HC, V])
                pl.load(d, g_t, [0, head_id, t_base + valid_size - 1, 0], order=[2, 3])

                pl.expand_sub(c, ws, d, dim=1)
                pl.neg(c, c)
                pl.exp(c, c)
                pl.mul(a, a, c)
                pl.cast(b, a)
                pl.store(ws_k, b, [core_id * C + ro, 0])
                pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=1)

                if ci + 1 < num_chunks_seq:
                    pl.set_validshape(a, [next_rows, K])
                    pl.load(a, w_t, [0, head_id, t_base + C + ro, 0], order=[2, 3])
                    pl.set_validshape(a, [HC, K])
                    pl.cast(b, a)
                    pl.store(ws_w_f16, b, [core_id * C + ro, 0])

            # ================================================================
            #  Phase 3 (Cube): KV = k_rest^T @ v_corr  ->  ub_kv (via AccToVecMode)
            # ================================================================
            with pl.section_cube():
                pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=1)

                cur_k_l1 = k_l1.current()
                cur_v_l1 = v_l1.current()
                cur_k_l0a = k_l0a.current()
                cur_v_l0b = v_l0b.current()
                cur_kv_acc = kv_acc.current()
                cur_ub_kv = ub_kv.current()

                pl.load(cur_k_l1, ws_k, [core_id * C, 0], order=[1, 0])
                pl.load(cur_v_l1, vcorr_out, [0, head_id, t_base, 0], order=[2, 3])
                pl.move(cur_k_l0a, cur_k_l1)
                pl.move(cur_v_l0b, cur_v_l1)
                pl.matmul(cur_kv_acc, cur_k_l0a, cur_v_l0b)
                pl.move(cur_ub_kv, cur_kv_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)

                pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=2)

            # ================================================================
            #  Phase 4 (Vec): S = exp(g_total) * S + KV  (per sub-block)
            # ================================================================
            with pl.section_vector():
                pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=2)

                b = ub_b.current()
                d = ub_d.current()
                kv = ub_kv.current()
                s_res = ub_s_res.current()

                pl.load(d, g_t, [0, head_id, t_base + valid_size - 1, 0], order=[2, 3])
                pl.exp(d, d)
                d_col = ub_d_col.current()
                pl.expand_mul(s_res, s_res, d_col, dim=0)
                pl.add(s_res, s_res, kv)
                pl.cast(b, s_res)
                pl.store(ws_s_f16, b, [core_id * K + ro, 0])

                if ci + 1 < num_chunks_seq:
                    k_pf = ub_k_pf.current()
                    pl.set_validshape(k_pf, [next_rows, K])
                    pl.load(k_pf, k_t, [0, t_base + C + ro, head_id, 0], order=[1, 3])
                    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=3)


def run_chunk_h_kda(k, w, u, g_cs, s_out, vcorr_out,
                    cu_seqlens=None, num_cores=None):
    """Launch the chunk_h_kda kernel.

    Args:
        k:           [B, T, HV, K]  fp16   keys (BSND)
        w:           [B, HV, T, K]  fp32   w from wy_kda (BNTD)
        u:           [B, HV, T, V]  fp32   u from wy_kda (BNTD)
        g_cs:        [B, HV, T, K]  fp32   cumulative gate sum (BNTD)
        s_out:       [HV, n_chunks, K, V]  fp16  state snapshots output
        vcorr_out:   [B, HV, T, V]  fp16   corrected values output
        cu_seqlens:  list[int] | None
        num_cores:   int | None
    """
    k = k.to(_DEVICE)
    w = w.to(_DEVICE)
    u = u.to(_DEVICE)
    g_cs = g_cs.to(_DEVICE)
    s_out = s_out.to(_DEVICE)
    vcorr_out = vcorr_out.to(_DEVICE)

    HVd = k.shape[2]
    T = k.shape[1]

    if cu_seqlens is None:
        _ranges = [(0, T)]
    else:
        _ranges = [(cu_seqlens[i], cu_seqlens[i + 1]) for i in range(len(cu_seqlens) - 1)]
    num_seqs = len(_ranges)
    num_chunks = sum((eos - bos + C - 1) // C for bos, eos in _ranges)

    chunk_offsets_list = [0]
    for bos, eos in _ranges:
        chunk_offsets_list.append(chunk_offsets_list[-1] + (eos - bos + C - 1) // C)

    cu_seqlens_tensor = make_cu_seqlens_tensor(T, cu_seqlens, _DEVICE)
    chunk_offsets_tensor = torch.tensor(chunk_offsets_list, dtype=torch.int32).to(_DEVICE)

    ws_s, ws_k, ws_v, ws_s_f16, ws_w_f16 = alloc_chunk_h_workspaces(HVd, C, K_DIM, V_DIM, _DEVICE)

    if num_cores is None:
        num_cores = torch.npu.get_device_properties(0).cube_core_num
    nc = min(num_cores, HVd)

    chunk_h_kda_kernel[None, nc](
        k, w, u, g_cs,
        s_out, vcorr_out,
        ws_s, ws_k, ws_v, ws_s_f16, ws_w_f16,
        HVd, num_seqs, T,
        cu_seqlens_tensor, chunk_offsets_tensor,
    )
    torch.npu.synchronize()
