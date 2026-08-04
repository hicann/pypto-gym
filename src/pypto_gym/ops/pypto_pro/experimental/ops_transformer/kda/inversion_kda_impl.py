# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root directory of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""inversion_kda — cube-based Neumann series inversion.

Computes (I + L)^{-1} for each [128, 128] strictly lower triangular chunk
using a full-matrix Neumann series with repeated doubling, accelerated by
128x128 cube matmul:

    M = -L
    X = I,  Y = M
    for round in range(7):          # log2(128) = 7
        Y_sq = Y @ Y                # cube matmul
        X    = X + X @ Y            # cube matmul + vec add
        Y    = Y_sq

After 7 rounds: X = sum_{i=0}^{127} (-L)^i = (I + L)^{-1}.

This works because L is strictly lower triangular (nilpotent: L^128 = 0),
so the Neumann series terminates exactly.

All computation is inside a single fused kernel with alternating Vec and
Cube phases per Neumann round.  The entire 7-round loop and all
intermediate data movement happen on-chip.

Performance: 14 cube matmuls (7 rounds x 2) in a single kernel launch.
"""

import torch
import pypto_pro.language as pl

from .kda_common import C, DEVICE as _DEVICE

CS = C


# ---- L1 (Mat) addresses ----
L1_A = 0x00000   # [CS, CS] fp32 NZ  64 KB  — left operand
L1_B = 0x10000   # [CS, CS] fp32 NZ  64 KB  — right operand

# ---- UB (Vec) addresses ----
UB_A = 0x00000   # [CS, CS] fp32  64 KB
UB_B = 0x10000   # [CS, CS] fp32  64 KB
UB_C = 0x20000   # [CS, CS] fp32  64 KB


@pl.jit(auto_mutex=True)
def inversion_kda_cube_kernel(
    A: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    minus_identity: pl.Tensor[[CS, CS], pl.DT_FP32],
    A_inv: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    ws_x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    ws_y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    ws_ysq: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    ws_xy: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    cu_seqlens: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
):
    """Fused Neumann series inversion in a single kernel.

    Inputs (all on GM):
        A:              [B, H, T, CS]  fp16   strictly lower triangular L (BNTD)
        minus_identity: [CS, CS]       fp32   -I constant
        ws_x:           [nc*CS, CS]    fp32   X workspace
        ws_y:           [nc*CS, CS]    fp32   Y workspace
        ws_ysq:         [nc*CS, CS]    fp32   Y^2 workspace
        ws_xy:          [nc*CS, CS]    fp32   X@Y workspace
        cu_seqlens:     [num_seqs+1]   int32  cumulative sequence lengths

    Output:
        A_inv:  [B, H, T, CS]  fp32   (I + L)^{-1} (BNTD)
    """
    H = A.shape[1]
    num_seqs = cu_seqlens.shape[0] - 1
    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx() // pl.get_subblock_num()

    # ── Cube tile groups ──
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[CS, CS], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=L1_A, mutex_ids=[0])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[CS, CS], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=L1_B, mutex_ids=[1])
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[CS, CS], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0, mutex_ids=[2])
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[CS, CS], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0, mutex_ids=[3])
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[CS, CS], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                         fractal=1024),
        addrs=0x0, mutex_ids=[4])

    # ── Vec tile groups ──
    ub_ah = pl.make_tile_group(
        type=pl.TileType(shape=[CS, CS], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Vec,
                         pad=pl.TilePad.zero, valid_shape=[-1, -1]),
        addrs=0x30000, mutex_ids=[8])
    ub_a = pl.make_tile_group(
        type=pl.TileType(shape=[CS, CS], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec,
                         pad=pl.TilePad.zero, valid_shape=[-1, -1]),
        addrs=UB_A, mutex_ids=[5])
    ub_b = pl.make_tile_group(
        type=pl.TileType(shape=[CS, CS], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=UB_B, mutex_ids=[6])
    ub_c = pl.make_tile_group(
        type=pl.TileType(shape=[CS, CS], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=UB_C, mutex_ids=[7])

    # ── Compute num_chunks by scanning cu_seqlens ──
    num_chunks = 0
    for seq_idx in pl.range(0, num_seqs):
        seq_start = pl.getval(cu_seqlens, seq_idx)
        seq_end = pl.getval(cu_seqlens, seq_idx + 1)
        seq_len = seq_end - seq_start
        seq_num_chunks = (seq_len + CS - 1) // CS
        num_chunks = num_chunks + seq_num_chunks

    total_work = num_chunks * H

    # ── Work loop ──
    for work_id in pl.range(core_id, total_work, num_cores):
        chunk_id = work_id // H
        head_id = work_id % H

        off = core_id * CS

        # ── Scan cu_seqlens for row_start and valid_size ──
        accumulated_chunks = 0
        row_start = 0
        valid_size = CS
        found = 0
        for seq_idx in pl.range(0, num_seqs):
            if found == 0:
                seq_start = pl.getval(cu_seqlens, seq_idx)
                seq_end = pl.getval(cu_seqlens, seq_idx + 1)
                seq_len = seq_end - seq_start
                seq_num_chunks = (seq_len + CS - 1) // CS
                if chunk_id < accumulated_chunks + seq_num_chunks:
                    local_chunk = chunk_id - accumulated_chunks
                    row_start = seq_start + local_chunk * CS
                    remaining = seq_end - row_start
                    valid_size = pl.min(remaining, CS)
                    found = 1
                accumulated_chunks = accumulated_chunks + seq_num_chunks

        # ================================================================
        #  Init (Vec): X = I, Y = -L → GM workspaces
        # ================================================================
        with pl.section_vector():
            ah = ub_ah.current()
            a = ub_a.current()

            pl.set_validshape(a, [CS, CS])
            pl.load(a, minus_identity, [0, 0])
            pl.neg(a, a)
            pl.store(ws_x, a, [off, 0])

            pl.expands(a, 0.0)
            pl.store(ws_y, a, [off, 0])
            pl.set_validshape(ah, [valid_size, valid_size])
            pl.load(ah, A, [0, head_id, row_start, 0], order=[2, 3])
            pl.set_validshape(ah, [CS, CS])
            pl.cast(a, ah, mode=pl.RoundMode.CAST_NONE)
            pl.neg(a, a)
            pl.set_validshape(a, [valid_size, valid_size])
            pl.store(ws_y, a, [off, 0])
            pl.set_validshape(a, [CS, CS])

            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=0)

        # ================================================================
        #  Neumann rounds: Cube (Y_sq, XY) → Vec (X+XY, Y=Y_sq)
        # ================================================================
        for round in pl.range(7):
            with pl.section_cube():
                pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=0)

                cur_a = a_l1.current()
                cur_b = b_l1.current()
                al = a_l0a.current()
                br = b_l0b.current()
                ac = acc.current()

                pl.load(cur_b, ws_y, [off, 0])
                pl.move(br, cur_b)

                pl.load(cur_a, ws_y, [off, 0])
                pl.move(al, cur_a)
                pl.matmul(ac, al, br)
                pl.store(ws_ysq, ac, [off, 0])

                pl.load(cur_a, ws_x, [off, 0])
                pl.move(al, cur_a)
                pl.matmul(ac, al, br)
                pl.store(ws_xy, ac, [off, 0])

                pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=1)

            with pl.section_vector():
                pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=1)

                a = ub_a.current()
                b = ub_b.current()

                pl.load(a, ws_x, [off, 0])
                pl.load(b, ws_xy, [off, 0])
                pl.add(a, a, b)
                pl.store(ws_x, a, [off, 0])

                pl.load(a, ws_ysq, [off, 0])
                pl.store(ws_y, a, [off, 0])

                if round + 1 < 7:
                    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=0)

        # ================================================================
        #  Final (Vec): store X → A_inv
        # ================================================================
        with pl.section_vector():
            a = ub_a.current()
            pl.load(a, ws_x, [off, 0])
            pl.set_validshape(a, [valid_size, valid_size])
            pl.store(A_inv, a, [0, head_id, row_start, 0], order=[2, 3])


def _compute_num_chunks(cu_seqlens, T, cs):
    if cu_seqlens is None:
        return (T + cs - 1) // cs
    cu = cu_seqlens.tolist() if hasattr(cu_seqlens, "tolist") else cu_seqlens
    return sum((cu[i + 1] - cu[i] + cs - 1) // cs for i in range(len(cu) - 1))


def inversion_kda_cube(
    A: torch.Tensor,
    cu_seqlens=None,
    num_cores: int | None = None,
) -> torch.Tensor:
    """Compute (I + L)^{-1} using fused Neumann series kernel.

    Args:
        A:           [B, H, T, CS]  fp16   strictly lower triangular L (BNTD)
        cu_seqlens:  list[int] | None   cumulative sequence lengths
        num_cores:   int | None  None → auto-detect from device

    Returns:
        A_inv:  [B, H, T, CS]  fp32   (I + L)^{-1} (BNTD)
    """
    orig_device = A.device
    A = A.to(_DEVICE)
    if num_cores is None:
        num_cores = torch.npu.get_device_properties(0).cube_core_num
    B, H, T, _ = A.shape

    num_chunks = _compute_num_chunks(cu_seqlens, T, CS)
    total_work = num_chunks * H
    nc = min(num_cores, total_work)

    minus_identity = torch.zeros(CS, CS, device=_DEVICE, dtype=torch.float32)
    minus_identity.fill_diagonal_(-1.0)

    A_inv = torch.zeros(B, H, T, CS, device=_DEVICE, dtype=torch.float32)

    ws_x   = torch.empty(nc * CS, CS, device=_DEVICE, dtype=torch.float32)
    ws_y   = torch.empty(nc * CS, CS, device=_DEVICE, dtype=torch.float32)
    ws_ysq = torch.empty(nc * CS, CS, device=_DEVICE, dtype=torch.float32)
    ws_xy  = torch.empty(nc * CS, CS, device=_DEVICE, dtype=torch.float32)

    if cu_seqlens is None:
        cu_seqlens_t = torch.tensor([0, T], dtype=torch.int32, device=_DEVICE)
    else:
        cu_seqlens_t = torch.tensor(cu_seqlens, dtype=torch.int32, device=_DEVICE)

    inversion_kda_cube_kernel[None, nc](
        A, minus_identity, A_inv, ws_x, ws_y, ws_ysq, ws_xy, cu_seqlens_t
    )
    torch.npu.synchronize()

    return A_inv.to(orig_device)
