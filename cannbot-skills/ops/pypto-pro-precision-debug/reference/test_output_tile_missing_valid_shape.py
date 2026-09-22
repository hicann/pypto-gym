# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""负例：输出 Tile 未设 valid_shape，把逻辑输出之外的数据写回。

单独运行 pytest -sv 本文件。PASS 表示观察到预期的错误；需要 Ascend 950。
"""

import os

import pypto_pro.language as pl
import pytest
import torch
import torch_npu  # noqa: F401


TILE = 64
VALID_ROWS = 32


@pl.jit(auto_mutex=True)
def bad_output_tile_missing_valid_shape(
    x: pl.Tensor[[VALID_ROWS, TILE], pl.DT_FP16],
    poison: pl.Tensor[[TILE, TILE], pl.DT_FP16],
    out: pl.Tensor[[TILE, TILE], pl.DT_FP16],
):
    source = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
        addrs=0x0000, mutex_ids=[0],
    )
    result = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE], dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1],
        ),
        addrs=0x2000, mutex_ids=[1],
    )
    with pl.section_vector():
        src = source.current()
        dst = result.current()
        pl.load(src, poison, [0, 0])
        pl.system.bar_mte2()
        pl.set_validshape(src, [VALID_ROWS, TILE])
        pl.load(src, x, [0, 0])
        # BAD: same-size move uses dst's full valid_shape and copies the poisoned tail.
        pl.move(dst, src)
        pl.store(out, dst, [0, 0])


def test_output_tile_missing_valid_shape() -> None:
    device = f"npu:{os.environ.get('TILE_FWK_DEVICE_ID', '0')}"
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    if "Ascend950" not in torch.npu.get_device_name():
        pytest.skip("Ascend 950 required")

    x = torch.ones((VALID_ROWS, TILE), device=device, dtype=torch.float16)
    poison = torch.full((TILE, TILE), 7.0, device=device, dtype=torch.float16)
    out = torch.zeros((TILE, TILE), device=device, dtype=torch.float16)
    bad_output_tile_missing_valid_shape(x, poison, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out[:VALID_ROWS, :], x, rtol=0, atol=0)
    assert torch.count_nonzero(out[VALID_ROWS:, :]).item() > 0, "tail overwrite did not reproduce"


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-sv", __file__]))
