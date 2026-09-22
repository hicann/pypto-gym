# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""负例：声明 pad=zero 却未执行 fillpad，归约读到尾部旧值。

单独运行 pytest -sv 本文件。PASS 表示观察到预期的错误；需要 Ascend 950。
先用已分配的输入初始化物理 Tile，再在下一次尾块 load 前设置 valid_shape。
"""

import os

import pypto_pro.language as pl
import pytest
import torch
import torch_npu  # noqa: F401


TILE = 64
VALID = 32


@pl.jit(auto_mutex=True)
def bad_pad_declared_without_fillpad(
    x: pl.Tensor[[TILE, TILE], pl.DT_FP32],
    poison: pl.Tensor[[TILE, TILE], pl.DT_FP32],
    out: pl.Tensor[[TILE, 1], pl.DT_FP32],
):
    source = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
        addrs=0x0000, mutex_ids=[0],
    )
    padded = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec, pad=pl.TilePad.zero),
        addrs=0x4000, mutex_ids=[1],
    )
    scratch = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=0x8000, mutex_ids=[2],
    )
    result = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, 1], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=0xC000, mutex_ids=[3],
    )
    with pl.section_vector():
        src = source.current()
        dst = padded.current()
        output_tile = result.current()
        pl.load(src, poison, [0, 0])
        pl.system.bar_mte2()
        pl.set_validshape(src, [VALID, VALID])
        pl.load(src, x, [0, 0])
        pl.load(dst, poison, [0, 0])
        # BAD: move does not replace fillpad; the invalid region stays nonzero.
        pl.move(dst, src)
        pl.sum(output_tile, dst, scratch.current(), dim=0)
        pl.store(out, output_tile, [0, 0])


def test_pad_declared_without_fillpad() -> None:
    device = f"npu:{os.environ.get('TILE_FWK_DEVICE_ID', '0')}"
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    if "Ascend950" not in torch.npu.get_device_name():
        pytest.skip("Ascend 950 required")

    x = torch.ones((TILE, TILE), device=device, dtype=torch.float32)
    poison = torch.full_like(x, 10.0)
    out = torch.zeros((TILE, 1), device=device, dtype=torch.float32)
    bad_pad_declared_without_fillpad(x, poison, out)
    torch.npu.synchronize()
    expected = torch.zeros_like(out)
    expected[:VALID, :] = VALID
    assert torch.count_nonzero(out != expected).item() > 0, "missing fillpad did not reproduce"


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-sv", __file__]))
