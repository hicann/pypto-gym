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
# STATUS: VALIDATED on Ascend a5 NPU (2026-07-21) — GATE A PASS, first try.
# VALIDATED-CODE-SHA256: 829d4dd347cc21319d6a6a97575999971f8042e74b7d77f656de3b398e5d1ff8
# Both outputs were bit-exact. Diagnostics also confirmed that the accumulated output differed
# from either half alone, proving that the two sub-blocks were genuinely summed.
# Kernel P1: cube_vec_atomic_add_two_outputs — pipeline_patterns class.
# The first output stores the squared matmul result per sub-block; the second atomically adds
# the two 32-row halves into one output.
# Structure = the proven k5/k11 cube->vector handoff (handoff1, DualModeSplitM) + a second
# output written with pl.AtomicType.AtomicAdd. The atomic is the new capability under test:
# both vector sub-blocks accumulate their own 32-row slab into the SAME [0,0] region, so a
# correct result requires the atomic accumulate to serialize the two concurrent writes.
# Convention note: siblings compute q@k (not q@k.t()); easyasc golden is x@y.t() — the
# transpose convention is already proven separately by k4 (TN). Kept q@k to stay a pure
# delta on the proven template.
import logging

import pypto_pro.language as pl
import torch  # noqa: F401
import torch_npu

LOGGER = logging.getLogger(__name__)


@pl.jit()
def mm_atomic_two_out(
    q: pl.Tensor[[64, 64], pl.DT_FP32],
    k: pl.Tensor[[64, 64], pl.DT_FP32],
    out_sq: pl.Tensor[[64, 64], pl.DT_FP32],
    out_acc: pl.Tensor[[32, 64], pl.DT_FP32],
):
    mm1_res = pl.make_tile(
        pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addr=0x0000, size=8192
    )
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
        pl.move(mm1_res, tile_c1, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)
    with pl.section_vector():
        sub_index = pl.get_subblock_idx()
        off = sub_index * 32
        tile_o = pl.make_tile(
            pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addr=0x8000, size=8192
        )
        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=0)
        pl.mul(tile_o, mm1_res, mm1_res)  # z^2 on this sub-block's 32 rows
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pl.store(out_sq, tile_o, [off, 0])  # output 1: normal
        pl.store(out_acc, tile_o, [0, 0], atomic=pl.AtomicType.AtomicAdd)  # output 2: atomic accumulate


def test_cube_vec_atomic_two_out_gate_a():
    torch.npu.set_device("npu:0")
    if "Ascend950" not in torch.npu.get_device_name():
        import pytest

        pytest.skip("not a5")
    torch.manual_seed(0)
    q = torch.randn(64, 64, device="npu:0", dtype=torch.float32)
    k = torch.randn(64, 64, device="npu:0", dtype=torch.float32)
    out_sq = torch.zeros(64, 64, device="npu:0", dtype=torch.float32)
    out_acc = torch.zeros(32, 64, device="npu:0", dtype=torch.float32)  # atomic accumulates INTO this
    mm_atomic_two_out(q, k, out_sq, out_acc)
    torch.npu.synchronize()

    mm = q @ k
    ref_sq = mm * mm
    ref_acc = ref_sq[0:32] + ref_sq[32:64]
    d_sq = (out_sq - ref_sq).abs().max().item()
    d_acc = (out_acc - ref_acc).abs().max().item()
    rel_acc = d_acc / ref_acc.abs().max().item()
    LOGGER.info(f"GATEA cube_vec_atomic out_sq maxdiff={d_sq:.3e}")
    LOGGER.info(f"GATEA cube_vec_atomic out_acc maxdiff={d_acc:.3e} rel={rel_acc:.3e}")
    # Diagnostic: if the atomic silently overwrote instead of accumulating, out_acc equals
    # ONE of the halves rather than their sum. Print both so a failure is self-diagnosing.
    LOGGER.info(
        f"DIAG vs half0={(out_acc - ref_sq[0:32]).abs().max().item():.3e} "
        f"vs half1={(out_acc - ref_sq[32:64]).abs().max().item():.3e}"
    )
    torch.testing.assert_close(out_sq, ref_sq, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(out_acc, ref_acc, rtol=1e-3, atol=1e-2)
