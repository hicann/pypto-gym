# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""负例：前一次 load 尚未被读取就复用同一 UB 地址。

单独运行 pytest -sv 本文件。PASS 表示观察到预期的错误；需要 Ascend 950。
"""

import os

import pypto_pro.language as pl
import pytest
import torch
import torch_npu  # noqa: F401


WIDTH = 64


@pl.jit()
def bad_load_reuses_address_before_read(
    a: pl.Tensor[[1, WIDTH], pl.DT_FP32],
    b: pl.Tensor[[1, WIDTH], pl.DT_FP32],
    out: pl.Tensor[[1, WIDTH], pl.DT_FP32],
):
    tile_type = pl.TileType(shape=[1, WIDTH], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    a_tile = pl.make_tile(tile_type, addr=0x000)
    b_tile = pl.make_tile(tile_type, addr=0x000)
    result = pl.make_tile(tile_type, addr=0x100)
    with pl.section_vector():
        pl.load(a_tile, a, [0, 0])
        pl.system.bar_mte2()
        # BAD: b_tile aliases a_tile, whose contents are still needed.
        pl.load(b_tile, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.add(result, a_tile, b_tile)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, result, [0, 0])


def test_load_reuses_address_before_read() -> None:
    device = f"npu:{os.environ.get('TILE_FWK_DEVICE_ID', '0')}"
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    if "Ascend950" not in torch.npu.get_device_name():
        pytest.skip("Ascend 950 required")

    a = torch.full((1, WIDTH), 2.0, device=device)
    b = torch.full((1, WIDTH), 3.0, device=device)
    out = torch.zeros_like(a)
    bad_load_reuses_address_before_read(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, b + b, rtol=0, atol=0)
    assert torch.count_nonzero(out != a + b).item() > 0, "premature overwrite did not reproduce"


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-sv", __file__]))
