# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""负例：reinterpret 后未重新设置 valid_shape，尾部被写回。"""

# 单独运行 pytest -sv 本文件。PASS 表示观察到预期的错误；需要 Ascend 950。

import os

import pypto_pro.language as pl
import pytest
import torch
import torch_npu  # noqa: F401


TILE = 64
VALID_ROWS = 32


@pl.jit(auto_mutex=True)
def bad_reinterpret_view_missing_valid_shape(
    x: pl.Tensor[[VALID_ROWS, TILE], pl.DT_FP16],
    poison: pl.Tensor[[TILE, TILE], pl.DT_FP16],
    out: pl.Tensor[[TILE, TILE], pl.DT_FP16],
):
    source = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE], dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1],
        ),
        addrs=0x0000, mutex_ids=[0],
    )
    with pl.section_vector():
        tile = source.current()
        pl.load(tile, poison, [0, 0])  # initialize the full physical buffer
        pl.system.bar_mte2()
        pl.set_validshape(tile, [VALID_ROWS, TILE])
        pl.load(tile, x, [0, 0])  # this load is restricted to the valid region
        view = pl.reinterpret(tile, shape=[TILE, TILE])
        # BAD: view does not inherit tile's valid_shape; store writes the tail.
        pl.store(out, view, [0, 0])


def test_reinterpret_view_missing_valid_shape() -> None:
    device = f"npu:{os.environ.get('TILE_FWK_DEVICE_ID', '0')}"
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        pytest.skip(f"Ascend 950 required, current device is {device_name}")

    x = torch.ones((VALID_ROWS, TILE), device=device, dtype=torch.float16)
    poison = torch.full((TILE, TILE), 7.0, device=device, dtype=torch.float16)
    out = torch.zeros((TILE, TILE), device=device, dtype=torch.float16)
    bad_reinterpret_view_missing_valid_shape(x, poison, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out[:VALID_ROWS, :], x, rtol=0, atol=0)
    assert torch.count_nonzero(out[VALID_ROWS:, :]).item() > 0, "tail overwrite did not reproduce"


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-sv", __file__]))
