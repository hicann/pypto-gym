# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""负例：新一次 K 维累加的首块误用 matmul_acc，带入旧 L0C 结果。

单独运行 pytest -sv 本文件。PASS 表示观察到预期的错误；需要 Ascend 950。
先在同一 Kernel 内写入旧结果，使错误可确定复现，避免依赖未初始化内存。
"""

import os

import pypto_pro.language as pl
import pytest
import torch
import torch_npu  # noqa: F401


TILE = 128


@pl.jit(auto_mutex=True)
def bad_matmul_acc_reuses_stale_accumulator(
    a: pl.Tensor[[TILE, TILE], pl.DT_FP16],
    b: pl.Tensor[[TILE, TILE], pl.DT_FP16],
    out: pl.Tensor[[TILE, TILE], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x00000, mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x20000, mutex_ids=[1],
    )
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0000, mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0000, mutex_ids=[3],
    )
    accumulator = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024),
        addrs=0x0000, mutex_ids=[4],
    )
    with pl.section_cube():
        left_mat = a_l1.current()
        right_mat = b_l1.current()
        left = a_l0a.current()
        right = b_l0b.current()
        acc = accumulator.current()
        pl.load(left_mat, a, [0, 0])
        pl.load(right_mat, b, [0, 0])
        pl.move(left, left_mat)
        pl.move(right, right_mat)
        pl.matmul(acc, left, right)  # prior logical result in this L0C
        # BAD: first block of the new logical result should overwrite with matmul.
        pl.matmul_acc(acc, acc, left, right)
        pl.store(out, acc, [0, 0])


def test_matmul_acc_reuses_stale_accumulator() -> None:
    device = f"npu:{os.environ.get('TILE_FWK_DEVICE_ID', '0')}"
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    if "Ascend950" not in torch.npu.get_device_name():
        pytest.skip("Ascend 950 required")

    a = torch.ones((TILE, TILE), device=device, dtype=torch.float16)
    b = torch.ones((TILE, TILE), device=device, dtype=torch.float16)
    out = torch.zeros((TILE, TILE), device=device, dtype=torch.float32)
    bad_matmul_acc_reuses_stale_accumulator(a, b, out)
    torch.npu.synchronize()
    expected = torch.matmul(a.float(), b.float())
    torch.testing.assert_close(out, expected * 2, rtol=0, atol=0)
    assert torch.count_nonzero(out != expected).item() > 0, "stale accumulator did not reproduce"


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-sv", __file__]))
