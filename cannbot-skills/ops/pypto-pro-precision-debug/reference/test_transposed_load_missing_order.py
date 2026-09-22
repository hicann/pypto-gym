# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""A5 Cube 精度错误负例：转置存放的右操作数搬入时漏设 load order。"""

# 单独执行：pytest -sv test_transposed_load_missing_order.py，或 python test_transposed_load_missing_order.py。
# 每次只启动一次 Kernel；PASS 表示观察到精度不一致，FAIL 表示未复现或运行报错。
# SKIP 表示当前设备不是 A5。

import os

import pytest
import torch
import torch_npu  # noqa: F401 - registers the NPU backend
import pypto_pro.language as pl


TILE = 128


@pl.jit(auto_mutex=True)
def bad_transposed_load_missing_order(
    a: pl.Tensor[[TILE, TILE], pl.DT_FP16],
    b_t: pl.Tensor[[TILE, TILE], pl.DT_FP16],
    out: pl.Tensor[[TILE, TILE], pl.DT_FP32],
):
    mat_type = pl.TileType(
        shape=[TILE, TILE], dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
    )
    left_type = pl.TileType(
        shape=[TILE, TILE], dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Left, layout=pl.NZ,
    )
    right_type = pl.TileType(
        shape=[TILE, TILE], dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Right, layout=pl.ZN,
    )
    acc_type = pl.TileType(
        shape=[TILE, TILE], dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024,
    )
    a_l1 = pl.make_tile_group(type=mat_type, addrs=0x00000, mutex_ids=[0])
    b_l1 = pl.make_tile_group(type=mat_type, addrs=0x20000, mutex_ids=[1])
    a_l0a = pl.make_tile_group(type=left_type, addrs=0x0000, mutex_ids=[2])
    b_l0b = pl.make_tile_group(type=right_type, addrs=0x0000, mutex_ids=[3])
    c_l0c = pl.make_tile_group(type=acc_type, addrs=0x0000, mutex_ids=[4])
    with pl.section_cube():
        cur_a = a_l1.current()
        cur_b = b_l1.current()
        al = a_l0a.current()
        br = b_l0b.current()
        ac = c_l0c.current()
        pl.load(cur_a, a, [0, 0])
        # BAD: b_t stores B transposed; omitting order=[1, 0] loads B.T as B.
        pl.load(cur_b, b_t, [0, 0])
        pl.move(al, cur_a)
        pl.move(br, cur_b)
        pl.matmul(ac, al, br)
        pl.store(out, ac, [0, 0])


def _random(shape, device):
    return torch.randn(shape, device=device, dtype=torch.float16)


def _layout_inputs(device):
    a = _random([TILE, TILE], device)
    logical_b = _random([TILE, TILE], device)
    b_t = logical_b.transpose(0, 1).contiguous()
    out = torch.zeros([TILE, TILE], device=device, dtype=torch.float32)
    return [a, b_t, out], torch.matmul(a.float(), logical_b.float())


def test_transposed_load_missing_order() -> None:
    """One launch; PASS confirms the missing load order changed the result."""
    device = f"npu:{os.environ.get('TILE_FWK_DEVICE_ID', '0')}"
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        pytest.skip(f"A5/Ascend950 required, current device is {device_name}")

    torch.manual_seed(20260827)
    args, expected = _layout_inputs(device)
    bad_transposed_load_missing_order(*args)
    torch.npu.synchronize()
    actual = args[-1]
    a, b_t, _ = args
    torch.testing.assert_close(actual, torch.matmul(a.float(), b_t.float()), rtol=2e-2, atol=2e-2)
    bad_count = int((~torch.isclose(actual, expected, rtol=2e-2, atol=2e-2)).sum().item())
    assert bad_count > 0, "transposed_load_missing_order: precision mismatch did not reproduce in this run"


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-sv", __file__]))
