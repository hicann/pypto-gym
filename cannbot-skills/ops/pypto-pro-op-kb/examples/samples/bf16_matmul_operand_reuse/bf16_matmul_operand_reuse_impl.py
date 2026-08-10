# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""BF16 matmul that reuses each A tile across the output-column loop.

Validation evidence is the embedded NPU test below. The kernel is a study
reference for operand residency; performance must be re-measured on the target
device and shape.
"""

import pypto_pro.language as pl
import torch
import torch_npu  # noqa: F401

TILE = 128
K = 128


@pl.jit(auto_mutex=True)  # PyPTO requires the complete tile-memory plan in one compiled kernel body
def bf16_matmul_operand_reuse(
    a: pl.Tensor[[pl.DYNAMIC, K], pl.DT_BF16],
    b: pl.Tensor[[K, pl.DYNAMIC], pl.DT_BF16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
):
    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx()

    with pl.section_cube():
        a_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, K],
                dtype=pl.DT_BF16,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NZ,
            ),
            addrs=0x00000,
            mutex_ids=[0, 1, 10, 11],
        )
        a_l0a = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, K],
                dtype=pl.DT_BF16,
                target_memory=pl.MemorySpace.Left,
                layout=pl.NZ,
            ),
            addrs=0x0,
            mutex_ids=[2, 5],
        )
        b_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[K, TILE],
                dtype=pl.DT_BF16,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NZ,
            ),
            addrs=0x20000,
            mutex_ids=[3, 4, 12, 13],
        )
        b_l0b = pl.make_tile_group(
            type=pl.TileType(
                shape=[K, TILE],
                dtype=pl.DT_BF16,
                target_memory=pl.MemorySpace.Right,
                layout=pl.ZN,
            ),
            addrs=0x0,
            mutex_ids=[6, 7],
        )
        acc = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE],
                dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Acc,
                layout=pl.NZ,
                fractal=1024,
            ),
            addrs=0x0,
            mutex_ids=[8, 9, 14, 15],
        )

        column_tiles = b.shape[1] // TILE
        row_tiles = a.shape[0] // TILE
        for row_tile in pl.range(core_id, row_tiles, num_cores):
            a_mat = a_l1.next()
            pl.load_tile(a_mat, a, [row_tile, 0])
            a_left = a_l0a.next()
            pl.move(a_left, a_mat)

            for column_tile in pl.range(0, column_tiles):
                b_mat = b_l1.next()
                pl.load_tile(b_mat, b, [0, column_tile])
                b_right = b_l0b.next()
                pl.move(b_right, b_mat)
                out_acc = acc.next()
                pl.matmul(out_acc, a_left, b_right)
                pl.store_tile(out, out_acc, [row_tile, column_tile])


def test_bf16_matmul_operand_reuse():
    torch.npu.set_device("npu:0")
    if "Ascend950" not in torch.npu.get_device_name():
        import pytest

        pytest.skip("requires an Ascend 950-family target")

    torch.manual_seed(0)
    rows = 128 * TILE
    columns = 8 * TILE
    a = torch.randn(rows, K, device="npu:0").to(torch.bfloat16)
    b = torch.randn(K, columns, device="npu:0").to(torch.bfloat16)
    out = torch.empty(rows, columns, device="npu:0", dtype=torch.bfloat16)

    bf16_matmul_operand_reuse[None, 32](a, b, out)
    torch.npu.synchronize()

    expected = a.float() @ b.float()
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)
