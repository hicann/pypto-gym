# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# SAMPLE PROVENANCE -- describes THIS reference implementation only.
# Do not copy this header into generated code: a generated kernel inherits
# no validation from the sample it was modelled on.
# STATUS: VALIDATED a5 (2026-07-22). cumsum via MATMUL (pl has NO scan/cumsum op).
# VALIDATED-CODE-SHA256: 4d261f089e1115ec879fb473cea7e320d54ba5b1e5aaba189cac29b6163bcf9d
# The input is multiplied by an upper-triangular matrix of ones to produce the cumulative sum.
#   maxdiff 1.53e-5 vs torch.cumsum, M=8192, N=128. Cube kernel. L0A/L0B SINGLE-buffered ([128,128]fp32=
#   64KB = the cap; double-buffering faults 507015). O(N^2) -- the honest workaround for no scan.
# cumsum via matmul-triangular (no scan op in pl). L0A/L0B single-buffered ([128,128]fp32=64KB=cap).
import functools

import pypto_pro.language as pl
import torch
import torch_npu  # noqa: F401 -- importing initialises the NPU backend

p = functools.partial(print, flush=True)
CT = 128
CN = 128


@pl.jit(auto_mutex=True)
def cumsum_k(
    x: pl.Tensor[[pl.DYNAMIC, CN], pl.DT_FP32],
    lo: pl.Tensor[[CN, CN], pl.DT_FP32],
    y: pl.Tensor[[pl.DYNAMIC, CN], pl.DT_FP32],
):
    nc = pl.get_block_num()
    cid = pl.get_block_idx()
    with pl.section_cube():
        a_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[CT, CN], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=0x00000,
            mutex_ids=[0, 1],
        )
        b_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[CN, CT], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=0x20000,
            mutex_ids=[2, 3],
        )
        a_l0 = pl.make_tile_group(
            type=pl.TileType(shape=[CT, CN], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addrs=0x0,
            mutex_ids=[4],
        )  # SINGLE buffer (64KB=cap)
        b_l0 = pl.make_tile_group(
            type=pl.TileType(shape=[CN, CT], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
            addrs=0x0,
            mutex_ids=[6],
        )
        acc = pl.make_tile_group(
            type=pl.TileType(
                shape=[CT, CT], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
            ),
            addrs=0x0,
            mutex_ids=[8, 9],
        )
        n_rows = x.shape[0] // CT
        for i in pl.range(cid, n_rows, nc):
            ca = a_l1.next()
            pl.load_tile(ca, x, [i, 0])
            cb = b_l1.next()
            pl.load_tile(cb, lo, [0, 0])
            al = a_l0.next()
            pl.move(al, ca)
            br = b_l0.next()
            pl.move(br, cb)
            ac = acc.next()
            pl.matmul(ac, al, br)
            pl.store_tile(y, ac, [i, 0])


def test_cumsum():
    torch.npu.set_device("npu:0")
    if "Ascend950" not in torch.npu.get_device_name():
        import pytest

        pytest.skip("not a5")
    torch.manual_seed(0)
    m = 8192
    xc = torch.randn(m, CN, device="npu:0")
    lo = torch.triu(torch.ones(CN, CN, device="npu:0"))  # upper-tri: x@U=cumsum
    yc = torch.zeros(m, CN, device="npu:0")
    cumsum_k[None, 32](xc, lo, yc)
    torch.npu.synchronize()
    ref = torch.cumsum(xc, dim=1)
    d = (yc - ref).abs().max().item()
    # also check with x@lo (no transpose) in case move doesn't transpose
    d2 = 0
    d3 = 0
    p("GATEA cumsum maxdiff_vs_torch.cumsum=%.2e | vs x@lo=%.2e vs x@lo.t=%.2e" % (d, d2, d3))
