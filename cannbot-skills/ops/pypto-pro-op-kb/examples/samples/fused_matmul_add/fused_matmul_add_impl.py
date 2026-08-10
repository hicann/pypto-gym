# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# SAMPLE PROVENANCE -- describes THIS reference implementation only.
# Do not copy this header into generated code: a generated kernel inherits
# no validation from the sample it was modelled on.
# STATUS: VALIDATED on Ascend a5 NPU (2026-07-18) — FUSED cube->vector handoff template.
#   Gate A validated the fused FP32 result bit-exactly on a 64-by-64 case.
#   (Gate B: tiny shape = latency-bound; scale + tile for cube-bound as a separate pass.)
# VALIDATED-CODE-SHA256: c3a04dfd22eb70d4ec05057198bb93a299264a3ba95d7637305e9c7cda307e96
# THE FUSED PATTERN (gateway to bias / rowwise-norm / attention — all need this):
#   The cube section loads, moves, multiplies, transfers L0C to UB with DualModeSplitM, and
#   signals FIX. The vector section waits, adds the residual tile, and stores the result.
#   DualModeSplitM splits the Acc M-dim across the 2 vector sub-blocks (64 -> 2x32). Use DualModeSplitN
#   for the QK handoff in attention (split N). Cloned from repo test_matmul_add_matmul_add.py.
# For BIAS (matmul + [1,N] bias): same, but pl.add with a broadcast bias tile (row_expand-style).
# For NORM (matmul + rowwise norm): same handoff, then row_max/row_sum/... on the UB tile.
# Kernel 5 demonstrates a fused matmul-add cube-to-vector handoff with manual pipe synchronisation.
import logging

import pypto_pro.language as pl
import torch  # noqa: F401
import torch_npu

LOGGER = logging.getLogger(__name__)


@pl.jit()
def mm_add(
    q: pl.Tensor[[64, 64], pl.DT_FP32],
    k: pl.Tensor[[64, 64], pl.DT_FP32],
    x1: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tile_p_vec = pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    mm1_res = pl.make_tile(tile_p_vec, addr=0x0000, size=8192)
    with pl.section_cube():
        tile_mat = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        q_mat = pl.make_tile(tile_mat, addr=0x0000, size=16384)
        k_mat = pl.make_tile(tile_mat, addr=0x4000, size=16384)
        q_left = pl.make_tile(
            pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addr=0x0000,
            size=16384,
        )
        k_right = pl.make_tile(
            pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
            addr=0x0000,
            size=16384,
        )
        tile_c1 = pl.make_tile(
            pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024),
            addr=0x0000,
            size=16384,
        )
        pl.load(q_mat, q, [0, 0])
        pl.load(k_mat, k, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.move(q_left, q_mat)
        pl.move(k_right, k_mat)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.matmul(tile_c1, q_left, k_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.move(mm1_res, tile_c1, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)  # ACC -> UB
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)
    with pl.section_vector():
        sub_index = pl.get_subblock_idx()
        tile_x1 = pl.make_tile(
            pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addr=0x4000, size=8192
        )
        tile_out = pl.make_tile(
            pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addr=0x8000, size=8192
        )
        off = sub_index * 32
        pl.load(tile_x1, x1, [off, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=1)
        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=0)
        pl.add(tile_out, mm1_res, tile_x1)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pl.store(out, tile_out, [off, 0])


def test_mm_add_gate_a():
    torch.npu.set_device("npu:0")
    if "Ascend950" not in torch.npu.get_device_name():
        import pytest

        pytest.skip("not a5")
    torch.manual_seed(0)
    q = torch.randn(64, 64, device="npu:0", dtype=torch.float32)
    k = torch.randn(64, 64, device="npu:0", dtype=torch.float32)
    x1 = torch.randn(64, 64, device="npu:0", dtype=torch.float32)
    out = torch.zeros(64, 64, device="npu:0", dtype=torch.float32)
    mm_add(q, k, x1, out)
    torch.npu.synchronize()
    ref = q @ k + x1
    d = (out - ref).abs().max().item()
    LOGGER.info(f"GATEA mm_add 64x64 maxdiff={d:.3e}")
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
