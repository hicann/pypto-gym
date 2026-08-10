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
#   maxdiff = 3.815e-06 vs sqrt(abs(x)) @ rhs (tol 1e-3). 1 passed in 17.90s.
#   DIAG vs (x@rhs)=3.889e+01, vs (|x|@rhs)=9.981e+00  <- the result matches NEITHER the
#        raw nor the abs-only operand, proving BOTH vector ops (abs AND sqrt) reached the
#        cube through the UB->NZ->L1->L0A handoff.
#   ⇒ The vector->cube direction WORKS for a GM-sourced computed operand with the
#     NZ-L1 + row-offset recipe. (This is NOT an attention diagnostic — see note below.)
# VALIDATED-CODE-SHA256: b605d7fbe9a407ec48980232c161a48776e337b75e3343c2706d53db7bd38ce8
# Kernel P2: vec_cube_abs_sqrt_matmul — pipeline_patterns class.
#   out = sqrt(abs(x)) @ rhs      (VECTOR computes the matmul LEFT operand, then CUBE consumes it)
#
# This is the vector->cube direction (the "handoff2" shape): a tile COMPUTED in section_vector
# is reformatted ND->NZ, inserted into an L1 NZ buffer, moved to L0A and used as the matmul
# LEFT operand.
#
# Authored as a PURE DELTA on the repo's proven a5/docs/test_doc_memory_movement.py
# :insert_matmul_kernel (golden torch.matmul((x+y), rhs), rtol/atol 1e-2), which per its own
# header strictly mirrors the passing a5/matmul/test_single_dst.py. Only the vector arithmetic
# changed from addition to absolute-value followed by square-root. Every tile address, layout,
# sync and cross-core event is copied verbatim, INCLUDING:
#   - tile_nz size=8448 (deliberately PADDED above the 8192 the shape implies; the FA kernel
#     carries the same padded-row idiom and warns the move is sensitive to this descriptor)
#   - insert at ROW offset [off, 0] into an NZ L1 buffer (the GM-sourced recipe)
#   - cross-core event 2 is signalled by vector MTE3 and awaited by cube MTE1
#
import logging

import pypto_pro.language as pl
import torch  # noqa: F401
import torch_npu

LOGGER = logging.getLogger(__name__)


@pl.jit()
def vec_cube_abs_sqrt(
    x: pl.Tensor[[64, 64], pl.DT_FP32],
    rhs: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    v1_mat = pl.make_tile(
        pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addr=0x10000,
        size=16384,
    )

    with pl.section_vector():
        sub_index = pl.get_subblock_idx()
        off = sub_index * 32

        tile_x = pl.make_tile(
            pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addr=0x0000, size=8192
        )
        tile_abs = pl.make_tile(
            pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addr=0x2000, size=8192
        )
        tile_sum = pl.make_tile(
            pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addr=0x4000, size=8192
        )
        tile_nz = pl.make_tile(
            pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
            addr=0x6000,
            size=8448,
        )  # padded, per the template

        pl.load(tile_x, x, [off, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)

        pl.abs(tile_abs, tile_x)  # |x|
        pl.sqrt(tile_sum, tile_abs)  # sqrt(|x|)   <- the only delta vs the template
        pl.move(tile_nz, tile_sum)  # ND -> NZ

        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pl.insert(v1_mat, tile_nz, [off, 0])  # UB -> L1, NZ2NZ, ROW offset
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=2)

    with pl.section_cube():
        rhs_mat = pl.make_tile(
            pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addr=0x0000,
            size=16384,
        )
        v1_left = pl.make_tile(
            pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addr=0x0000,
            size=16384,
        )
        rhs_right = pl.make_tile(
            pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
            addr=0x0000,
            size=16384,
        )
        c_l0c = pl.make_tile(
            pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024),
            addr=0x0000,
            size=16384,
        )

        pl.load(rhs_mat, rhs, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.move(rhs_right, rhs_mat)

        pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.move(v1_left, v1_mat)

        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.matmul(c_l0c, v1_left, rhs_right)

        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.store(out, c_l0c, [0, 0])


def test_vec_cube_abs_sqrt_gate_a():
    torch.npu.set_device("npu:0")
    if "Ascend950" not in torch.npu.get_device_name():
        import pytest

        pytest.skip("not a5")
    torch.manual_seed(0)
    x = torch.randn([64, 64], device="npu:0", dtype=torch.float32)
    rhs = torch.randn([64, 64], device="npu:0", dtype=torch.float32)
    out = torch.zeros([64, 64], device="npu:0", dtype=torch.float32)

    vec_cube_abs_sqrt(x, rhs, out)
    torch.npu.synchronize()

    lhs_ref = torch.sqrt(torch.abs(x))
    ref = torch.matmul(lhs_ref.float(), rhs.float())
    d = (out - ref).abs().max().item()
    LOGGER.info(f"GATEA vec_cube_abs_sqrt 64x64 maxdiff={d:.3e}")
    # Diagnostic: if the L1 operand were scrambled we would still get SOME matmul; compare
    # against the un-transformed operand to prove the vector arithmetic actually reached cube.
    LOGGER.info(
        f"DIAG vs (x@rhs)={(out - torch.matmul(x, rhs)).abs().max().item():.3e} "
        f"vs (|x|@rhs)={(out - torch.matmul(torch.abs(x), rhs)).abs().max().item():.3e}"
    )
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
