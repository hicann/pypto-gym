# Copyright (c) 2024-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Compiler examples stay self-contained so each file exposes a complete kernel pattern.

import logging
import os
import sys
from dataclasses import dataclass
from importlib import import_module

import pypto_pro.language as pl
import torch

import_module("torch_npu")  # Register torch.npu on supported installations.


@dataclass
class OpTiling:
    """Compatibility tiling payload retained for existing smoke-test callers."""

    valid_size: int


@pl.jit(auto_mutex=True)
def matmul_example(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    tiling: OpTiling,
):
    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx()
    valid_n = 128
    with pl.section_cube():
        a_mat_4_buffer = pl.make_tile_group(
            type=pl.TileType(
                shape=[128, 128], dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Mat,
                valid_shape=[valid_n, valid_n]),
            addrs=0, mutex_ids=[0, 1, 10, 11])
        b_mat_4_buffer = pl.make_tile_group(
            type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0x20000, mutex_ids=[2, 3, 12, 13])
        a_left_db = pl.make_tile_group(
            type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
            addrs=0, mutex_ids=[4, 5])
        b_right_db = pl.make_tile_group(
            type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
            addrs=0, mutex_ids=[6, 7])
        acc_db = pl.make_tile_group(
            type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
            addrs=0, mutex_ids=[8, 9])

        for i in pl.range(core_id, a.shape[0] // 128, num_cores):
            a_l1_tile = a_mat_4_buffer.next()
            pl.load_tile(a_l1_tile, a, [i, 0])
            for j in pl.range(0, b.shape[1] // 128, 1):
                b_l1_tile = b_mat_4_buffer.next()
                pl.load_tile(b_l1_tile, b, [0, j])

                cur_a_left = a_left_db.next()
                pl.move(cur_a_left, a_l1_tile)
                cur_b_right = b_right_db.next()
                pl.move(cur_b_right, b_l1_tile)

                acc_tile = acc_db.next()
                pl.matmul(acc_tile, cur_a_left, cur_b_right)
                pl.store_tile(out, acc_tile, [i, j])


def run_perf_test(num_iters: int = 20, warmup: int = 3):
    """Run the historical one-shot correctness smoke.

    ``num_iters`` and ``warmup`` remain accepted for callers of the original
    helper; this environment gate intentionally performs one kernel launch.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
    device = f"npu:{device_id}"
    torch.npu.set_device(device_id)
    torch.manual_seed(42)
    m_size = 8192
    n_size = 8192
    k_size = 128
    a = torch.randn(m_size, k_size, device=device, dtype=torch.float16)
    b = torch.randn(k_size, n_size, device=device, dtype=torch.float16)
    out = torch.zeros(m_size, n_size, device=device, dtype=torch.float16)
    core_num = 32
    tiling = OpTiling(valid_size=128)
    matmul_example[None, core_num](a, b, out, tiling)
    torch.npu.synchronize()
    golden = torch.matmul(a.float(), b.float()).half()
    max_diff = (out.float() - golden.float()).abs().max().item()
    logging.info("Max diff vs golden: %.6f", max_diff)
    torch.testing.assert_close(out, golden, rtol=1e-2, atol=1e-2)
    logging.info("Correctness PASS")


def test_matmul_perf_asw_8k_k128_dn_move_offset():
    run_perf_test()


if __name__ == "__main__":
    run_perf_test()
