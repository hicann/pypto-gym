# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""负例：FP16 Tile 归约结果被当成 FP32 精度的 Golden 比较。

单独运行 pytest -sv 本文件。PASS 表示观察到预期的精度差；需要 Ascend 950。
这是精度路径选择错误，不表示 FP16 归约 API 本身计算错误。
"""

import os

import pypto_pro.language as pl
import pytest
import torch
import torch_npu  # noqa: F401


ROWS = 64
COLS = 128


@pl.jit(auto_mutex=True)
def fp16_reduction_with_fp16_output(
    x: pl.Tensor[[ROWS, COLS], pl.DT_FP16],
    out: pl.Tensor[[ROWS, 1], pl.DT_FP16],
):
    source = pl.make_tile_group(
        type=pl.TileType(shape=[ROWS, COLS], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Vec),
        addrs=0x0000, mutex_ids=[0],
    )
    scratch = pl.make_tile_group(
        type=pl.TileType(shape=[ROWS, COLS], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Vec),
        addrs=0x4000, mutex_ids=[1],
    )
    result = pl.make_tile_group(
        type=pl.TileType(shape=[ROWS, 1], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Vec),
        addrs=0x8000, mutex_ids=[2],
    )
    with pl.section_vector():
        input_tile = source.current()
        output_tile = result.current()
        pl.load(input_tile, x, [0, 0])
        pl.sum(output_tile, input_tile, scratch.current(), dim=0)
        pl.store(out, output_tile, [0, 0])


def test_fp16_reduction_expected_fp32_precision() -> None:
    device = f"npu:{os.environ.get('TILE_FWK_DEVICE_ID', '0')}"
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    if "Ascend950" not in torch.npu.get_device_name():
        pytest.skip("Ascend 950 required")

    x = torch.full((ROWS, COLS), 0.1, device=device, dtype=torch.float16)
    x[:, -1] = 0.2
    out = torch.zeros((ROWS, 1), device=device, dtype=torch.float16)
    fp16_reduction_with_fp16_output(x, out)
    torch.npu.synchronize()
    expected_fp32 = x.float().sum(dim=1, keepdim=True)
    assert torch.max((out.float() - expected_fp32).abs()).item() > 1e-4, "FP16 precision gap did not reproduce"


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-sv", __file__]))
