# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Kernel 12 fuses a 64-by-64 matmul with scaled, rounded, and saturated int8 output quantisation.
# Quant happens on the L0C->UB fixpipe move: pl.move(vec, acc, pre_quant_scalar=<f32 bits of scale>,
# acc_to_vec_mode=SingleModeVec0). Cloned from datacopy/test_deq_scalar_move_fp32_to_int8.py but with a
# REAL (random, non-identity) q@k. twin_of matmul quant/cast family (mxfp8/int8 output).
import logging
import struct

import pypto_pro.language as pl
import torch  # noqa: F401
import torch_npu

LOGGER = logging.getLogger(__name__)

SCALE = 16.0
SCALE_BITS = struct.unpack("!I", struct.pack("!f", SCALE))[0]


@pl.jit()
def mm_quant_int8(
    q: pl.Tensor[[64, 64], pl.DT_FP32],
    k: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_INT8],
):
    vec_tile = pl.make_tile(
        pl.TileType(shape=[64, 64], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec), addr=0x0000, size=4096
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
        acc = pl.make_tile(
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
        pl.matmul(acc, q_left, k_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.move(
            vec_tile, acc, pre_quant_scalar=SCALE_BITS, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0
        )  # L0C fp32 -> UB int8, scalar quant
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)
    with pl.section_vector():
        sub_id = pl.get_subblock_idx()
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
        if sub_id == 0:
            pl.store(out, vec_tile, [0, 0])
        pl.system.bar_all()


def test_mm_quant_int8_gate_a():
    torch.npu.set_device("npu:0")
    if "Ascend950" not in torch.npu.get_device_name():
        import pytest

        pytest.skip("not a5")
    torch.manual_seed(0)
    q = torch.randn(64, 64, device="npu:0", dtype=torch.float32) * 0.25  # keep (q@k)*scale within int8 range
    k = torch.randn(64, 64, device="npu:0", dtype=torch.float32) * 0.25
    out = torch.zeros(64, 64, device="npu:0", dtype=torch.int8)
    mm_quant_int8(q, k, out)
    torch.npu.synchronize()
    raw = torch.matmul(q, k)
    ref = torch.clamp(torch.round(raw * SCALE), -128, 127).to(torch.int8)
    mism = (out.to(torch.int32) != ref.to(torch.int32)).sum().item()
    LOGGER.info(f"GATEA mm_quant_int8 64x64 mismatched={mism}/4096")
    torch.testing.assert_close(out.to(torch.int32), ref.to(torch.int32), rtol=0, atol=0)
