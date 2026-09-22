# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""负例：mul_add_dst 的目标寄存器未初始化为要累加的加数。

单独运行 pytest -sv 本文件。PASS 表示观察到预期的错误；需要 Ascend 950。
"""

import os

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
import pytest
import torch
import torch_npu  # noqa: F401


WIDTH = 64


@pl.vector_function
def _bad_fma(a_tile, b_tile, zero_tile, out_tile):
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    a_reg = vf.load_align(a_tile, 0)
    b_reg = vf.load_align(b_tile, 0)
    # BAD: this should load the bias/addend, not a zero register.
    dst_reg = vf.load_align(zero_tile, 0)
    dst_reg = vf.mul_add_dst(a_reg, b_reg, mask)
    vf.store_align(out_tile, dst_reg, mask)


@pl.jit(auto_mutex=True)
def bad_vf_mul_add_dst_wrong_addend(
    a: pl.Tensor[[1, WIDTH], pl.DT_FP32],
    b: pl.Tensor[[1, WIDTH], pl.DT_FP32],
    bias: pl.Tensor[[1, WIDTH], pl.DT_FP32],
    zero: pl.Tensor[[1, WIDTH], pl.DT_FP32],
    out: pl.Tensor[[1, WIDTH], pl.DT_FP32],
):
    tile_type = pl.TileType(shape=[1, WIDTH], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    a_group = pl.make_tile_group(type=tile_type, addrs=0x000, mutex_ids=[0])
    b_group = pl.make_tile_group(type=tile_type, addrs=0x100, mutex_ids=[1])
    bias_group = pl.make_tile_group(type=tile_type, addrs=0x200, mutex_ids=[2])
    zero_group = pl.make_tile_group(type=tile_type, addrs=0x300, mutex_ids=[3])
    out_group = pl.make_tile_group(type=tile_type, addrs=0x400, mutex_ids=[4])
    with pl.section_vector():
        a_tile = a_group.current()
        b_tile = b_group.current()
        bias_tile = bias_group.current()
        zero_tile = zero_group.current()
        out_tile = out_group.current()
        pl.load(a_tile, a, [0, 0])
        pl.load(b_tile, b, [0, 0])
        pl.load(bias_tile, bias, [0, 0])
        pl.load(zero_tile, zero, [0, 0])
        _bad_fma(a_tile, b_tile, zero_tile, out_tile)
        pl.store(out, out_tile, [0, 0])


def test_vf_mul_add_dst_wrong_addend() -> None:
    device = f"npu:{os.environ.get('TILE_FWK_DEVICE_ID', '0')}"
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    if "Ascend950" not in torch.npu.get_device_name():
        pytest.skip("Ascend 950 required")

    a = torch.full((1, WIDTH), 2.0, device=device)
    b = torch.full((1, WIDTH), 3.0, device=device)
    bias = torch.full((1, WIDTH), 5.0, device=device)
    zero = torch.zeros_like(a)
    out = torch.zeros_like(a)
    bad_vf_mul_add_dst_wrong_addend(a, b, bias, zero, out)
    torch.npu.synchronize()
    expected = a * b + bias
    torch.testing.assert_close(out, a * b, rtol=0, atol=0)
    assert torch.count_nonzero(out != expected).item() > 0, "wrong addend did not reproduce"


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-sv", __file__]))
