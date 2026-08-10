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
# STATUS: VALIDATED a5 (2026-07-22). reduce_sum (row-wise sum over N -> [M,1]) MEMORY-bound
#   (mte2 0.94, read 2.36 TB/s = the vector GM one-way ceiling). Load-heavy, tiny output.
#   Also contains cumsum_k (see cumsum_matmul_impl.py for the standalone).
# VALIDATED-CODE-SHA256: fefdcc3b5f1570b9549b130c6892a6ad3478600f7bb96d46fbc5cef016bd66b0
# reduce_sum (vector, row-wise -> [M,1], memory-bound) + cumsum (matmul with triangular, no scan op).
import csv
import functools
import glob
import os
from pathlib import Path

import pypto_pro.language as pl
import torch  # noqa
import torch_npu

p = functools.partial(print, flush=True)

# ---------- reduce_sum: row-wise sum over N -> y[M,1] ----------
TR = 8
NN = 1024


@pl.jit(auto_mutex=True)
def reduce_sum_k(x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32], y: pl.Tensor[[pl.DYNAMIC, 1], pl.DT_FP32]):
    tt = pl.TileType(shape=[TR, NN], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    rt = pl.TileType(
        shape=[TR, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN, valid_shape=[-1, -1]
    )
    ing = pl.make_tile_group(type=tt, addrs=[0x00000, 0x08000], mutex_ids=[0, 1])
    tmg = pl.make_tile_group(type=tt, addrs=[0x10000, 0x18000], mutex_ids=[2, 3])
    rdg = pl.make_tile_group(type=rt, addrs=[0x20000, 0x20400], mutex_ids=[4, 5])
    with pl.section_vector():
        rows = x.shape[0]
        cols = x.shape[1]
        nc = pl.get_block_num()
        cid = pl.get_block_idx()
        nt = (rows + TR - 1) // TR
        for t in pl.range(cid, nt, nc):
            ro = t * TR
            vr = pl.min(TR, rows - ro)
            xi = ing.next()
            pl.set_validshape(xi, [vr, cols])
            pl.load(xi, x, [ro, 0])
            tm = tmg.next()
            rd = rdg.next()
            pl.set_validshape(tm, [vr, cols])
            pl.set_validshape(rd, [vr, 1])
            pl.row_sum(rd, xi, tm)
            pl.store(y, rd, [ro, 0])


# ---------- cumsum: y = x @ U (U = upper-tri ones [N,N]); via x @ L.t() with L lower-tri ----------
# NO scan op in pl -> cumsum-as-matmul. N=128 single K-block. TILE=128. tiled multicore.
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
            mutex_ids=[4, 5],
        )
        b_l0 = pl.make_tile_group(
            type=pl.TileType(shape=[CN, CT], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
            addrs=0x0,
            mutex_ids=[6, 7],
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
            pl.load_tile(ca, x, [i, 0])  # x row-block [128,128]
            cb = b_l1.next()
            pl.load_tile(cb, lo, [0, 0])  # lower-tri ones [128,128]
            al = a_l0.next()
            pl.move(al, ca)
            br = b_l0.next()
            pl.move(br, cb)
            ac = acc.next()
            pl.matmul(ac, al, br)  # x @ lo.t() = x @ upper-tri = cumsum
            pl.store_tile(y, ac, [i, 0])


def _cfg():
    return vars(torch_npu.profiler)["_ExperimentalConfig"](
        export_type=[torch_npu.profiler.ExportType.Text],
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    )


def test_rc():
    torch.npu.set_device("npu:0")
    if "Ascend950" not in torch.npu.get_device_name():
        import pytest

        pytest.skip("not a5")
    p("=== reduce_sum + cumsum ===")
    torch.manual_seed(0)
    m = 8192
    # reduce_sum
    x = torch.randn(m, NN, device="npu:0")
    y = torch.zeros(m, 1, device="npu:0")
    reduce_sum_k[None, 32](x, y)
    torch.npu.synchronize()
    d = (y - x.sum(1, keepdim=True)).abs().max().item()
    p("GATEA reduce_sum maxdiff=%.2e" % d)
    if d < 1e-2:
        odir = os.path.join(".", "prof_rs%d" % os.getpid())
        os.makedirs(odir, exist_ok=True)
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            experimental_config=_cfg(),
            schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=3),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(odir, analyse_flag=True),
        ) as prof:
            for _ in range(6):
                reduce_sum_k[None, 32](x, y)
                torch.npu.synchronize()
                prof.step()
        for cf in glob.glob(os.path.join(odir, "**", "kernel_details.csv"), recursive=True):
            for row in csv.DictReader(Path(cf).read_text(encoding="utf-8").splitlines()):
                if "_k" not in row.get("Name", ""):
                    continue
                av = float(row.get("aiv_time(us)") or 0)
                vec = float(row.get("aiv_vec_ratio") or 0)
                m2 = float(row.get("aiv_mte2_ratio") or 0)
                m3 = float(row.get("aiv_mte3_ratio") or 0)
                rb = (m * NN * 4) / (av * 1e-6) / 1e12 if av > 0 else 0
                p(
                    "RESULT reduce_sum | aiv=%.1fus vec=%.3f mte2=%.3f mte3=%.3f | read=%.2fTB/s | %s"
                    % (av, vec, m2, m3, rb, "MEM" if m2 > vec else "COMPUTE")
                )
                break
            break
    # cumsum
    xc = torch.randn(m, CN, device="npu:0")
    lo = torch.tril(torch.ones(CN, CN, device="npu:0"))  # lower-tri ones; x@lo.t()=x@upper-tri=cumsum
    yc = torch.zeros(m, CN, device="npu:0")
    cumsum_k[None, 32](xc, lo, yc)
    torch.npu.synchronize()
    ref = torch.cumsum(xc, dim=1)
    d = (yc - ref).abs().max().item()
    p("GATEA cumsum maxdiff=%.2e (N=%d, via matmul-triangular)" % (d, CN))
