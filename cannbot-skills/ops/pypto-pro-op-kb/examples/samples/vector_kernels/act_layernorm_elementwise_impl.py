# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Compiler examples stay self-contained so each file exposes a complete kernel pattern.
# Batch A: layernorm, silu, gelu (compute-bound) + memory-bound elementwise. fp32.
# Spans the compute<->memory spectrum. No activation primitives in pl -> compose from exp/recip.
import csv
import functools
import glob
import os
from dataclasses import dataclass
from pathlib import Path

import pypto_pro.language as pl
import torch  # noqa
import torch.nn.functional as F
import torch_npu

p = functools.partial(print, flush=True)
EPS = 1e-5

# ---------- layernorm (N=512, TR=16 : the proven config) ----------
LN_N = 512
LN_TR = 16


@pl.jit(auto_mutex=True)
def layernorm_k(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    g: pl.Tensor[[1, LN_N], pl.DT_FP32],
    b: pl.Tensor[[1, LN_N], pl.DT_FP32],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tt = pl.TileType(shape=[LN_TR, LN_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    rt = pl.TileType(
        shape=[LN_TR, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN, valid_shape=[-1, -1]
    )
    gt = pl.TileType(shape=[1, LN_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    xin = pl.make_tile(tt, addr=0x00000, size=32768)
    xc = pl.make_tile(tt, addr=0x0C000, size=32768)
    tmp = pl.make_tile(tt, addr=0x18000, size=32768)
    out = pl.make_tile(tt, addr=0x24000, size=32768)
    gam = pl.make_tile(gt, addr=0x30000, size=2048)
    nbe = pl.make_tile(gt, addr=0x31000, size=2048)
    r0 = pl.make_tile(rt, addr=0x32000, size=64)
    r0m = pl.make_tile(
        pl.TileType(shape=[1, LN_TR], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addr=0x32000, size=64
    )
    r1 = pl.make_tile(rt, addr=0x32100, size=64)
    r1m = pl.make_tile(
        pl.TileType(shape=[1, LN_TR], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addr=0x32100, size=64
    )
    with pl.section_vector():
        rows = x.shape[0]
        cols = x.shape[1]
        nc = pl.get_block_num()
        cid = pl.get_block_idx()
        pl.load(gam, g, [0, 0])
        pl.load(nbe, b, [0, 0])
        pl.mul(nbe, nbe, -1.0)  # negbeta
        nt = (rows + LN_TR - 1) // LN_TR
        for t in pl.range(cid, nt, nc):
            ro = t * LN_TR
            vr = pl.min(LN_TR, rows - ro)
            pl.set_validshape(xin, [vr, cols])
            pl.load(xin, x, [ro, 0])
            pl.set_validshape(xc, [vr, cols])
            pl.set_validshape(tmp, [vr, cols])
            pl.set_validshape(out, [vr, cols])
            pl.set_validshape(r0, [vr, 1])
            pl.set_validshape(r1, [vr, 1])
            pl.row_sum(r0, xin, tmp)
            pl.set_validshape(r0m, [1, vr])
            pl.div(r0m, r0m, cols)  # mean
            pl.row_expand_sub(xc, xin, r0)  # xc=x-mean
            pl.mul(tmp, xc, xc)
            pl.row_sum(r1, tmp, out)
            pl.set_validshape(r1m, [1, vr])
            pl.div(r1m, r1m, cols)
            pl.add(r1m, r1m, EPS)
            pl.rsqrt(r1m, r1m)  # 1/std
            pl.row_expand_mul(xc, xc, r1)
            pl.col_expand_mul(xc, xc, gam)
            pl.col_expand_sub(out, xc, nbe)
            pl.store(y, out, [ro, 0])


# ---------- silu / gelu / memory-bound elementwise (N=1024, TR=8) ----------
TR = 8
NN = 1024


def _eg():
    tt = pl.TileType(shape=[TR, NN], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    a = pl.make_tile_group(type=tt, addrs=[0x00000, 0x08000], mutex_ids=[0, 1])
    o = pl.make_tile_group(type=tt, addrs=[0x10000, 0x18000], mutex_ids=[2, 3])
    t = pl.make_tile_group(type=tt, addrs=[0x20000, 0x28000], mutex_ids=[4, 5])
    return a, o, t


@pl.jit(auto_mutex=True)
def silu_k(x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32], y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32]):
    tt = pl.TileType(shape=[TR, NN], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    ag = pl.make_tile_group(type=tt, addrs=[0x00000, 0x08000], mutex_ids=[0, 1])
    og = pl.make_tile_group(type=tt, addrs=[0x10000, 0x18000], mutex_ids=[2, 3])
    tg = pl.make_tile_group(type=tt, addrs=[0x20000, 0x28000], mutex_ids=[4, 5])
    with pl.section_vector():
        rows = x.shape[0]
        cols = x.shape[1]
        nc = pl.get_block_num()
        cid = pl.get_block_idx()
        nt = (rows + TR - 1) // TR
        for i in pl.range(cid, nt, nc):
            ro = i * TR
            vr = pl.min(TR, rows - ro)
            xi = ag.next()
            pl.set_validshape(xi, [vr, cols])
            pl.load(xi, x, [ro, 0])
            o = og.next()
            tm = tg.next()
            pl.set_validshape(o, [vr, cols])
            pl.set_validshape(tm, [vr, cols])
            pl.mul(tm, xi, -1.0)
            pl.exp(tm, tm)
            pl.add(tm, tm, 1.0)
            pl.recip(tm, tm)  # sigmoid(x)
            pl.mul(o, xi, tm)  # x*sigmoid(x)
            pl.store(y, o, [ro, 0])


@pl.jit(auto_mutex=True)
def gelu_k(x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32], y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32]):
    tt = pl.TileType(shape=[TR, NN], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    ag = pl.make_tile_group(type=tt, addrs=[0x00000, 0x08000], mutex_ids=[0, 1])
    og = pl.make_tile_group(type=tt, addrs=[0x10000, 0x18000], mutex_ids=[2, 3])
    tg = pl.make_tile_group(type=tt, addrs=[0x20000, 0x28000], mutex_ids=[4, 5])
    with pl.section_vector():
        rows = x.shape[0]
        cols = x.shape[1]
        nc = pl.get_block_num()
        cid = pl.get_block_idx()
        nt = (rows + TR - 1) // TR
        for i in pl.range(cid, nt, nc):
            ro = i * TR
            vr = pl.min(TR, rows - ro)
            xi = ag.next()
            pl.set_validshape(xi, [vr, cols])
            pl.load(xi, x, [ro, 0])
            o = og.next()
            tm = tg.next()
            pl.set_validshape(o, [vr, cols])
            pl.set_validshape(tm, [vr, cols])
            pl.mul(tm, xi, -1.702)
            pl.exp(tm, tm)
            pl.add(tm, tm, 1.0)
            pl.recip(tm, tm)  # sigmoid(1.702x)
            pl.mul(o, xi, tm)  # x*sigmoid(1.702x) ~ gelu
            pl.store(y, o, [ro, 0])


@pl.jit(auto_mutex=True)
def elw_k(x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32], y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32]):
    tt = pl.TileType(shape=[TR, NN], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    ag = pl.make_tile_group(type=tt, addrs=[0x00000, 0x08000], mutex_ids=[0, 1])
    og = pl.make_tile_group(type=tt, addrs=[0x10000, 0x18000], mutex_ids=[2, 3])
    with pl.section_vector():
        rows = x.shape[0]
        cols = x.shape[1]
        nc = pl.get_block_num()
        cid = pl.get_block_idx()
        nt = (rows + TR - 1) // TR
        for i in pl.range(cid, nt, nc):
            ro = i * TR
            vr = pl.min(TR, rows - ro)
            xi = ag.next()
            pl.set_validshape(xi, [vr, cols])
            pl.load(xi, x, [ro, 0])
            o = og.next()
            pl.set_validshape(o, [vr, cols])
            pl.mul(o, xi, 2.0)
            pl.add(o, o, 1.0)  # y = 2x+1 : 1 read + 1 write => memory-bound
            pl.store(y, o, [ro, 0])


def _cfg():
    return vars(torch_npu.profiler)["_ExperimentalConfig"](
        export_type=[torch_npu.profiler.ExportType.Text],
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    )


VS = ["aiv_vec_ratio", "aiv_scalar_ratio", "aiv_mte2_ratio", "aiv_mte3_ratio"]


@dataclass(frozen=True)
class ProfileCase:
    name: str
    tensors: tuple[torch.Tensor, ...]
    rows: int
    cols: int
    element_bytes: int = 4


def _prof(kern, case: ProfileCase):
    name = case.name
    args = case.tensors
    m, n, eb = case.rows, case.cols, case.element_bytes
    odir = os.path.join(".", "prof_bA%d" % os.getpid(), name)
    os.makedirs(odir, exist_ok=True)
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        experimental_config=_cfg(),
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=3),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(odir, analyse_flag=True),
    ) as prof:
        for _ in range(6):
            kern[None, 32](*args)
            torch.npu.synchronize()
            prof.step()
    rw = 2 * m * n * eb
    for cf in glob.glob(os.path.join(odir, "**", "kernel_details.csv"), recursive=True):
        for row in csv.DictReader(Path(cf).read_text(encoding="utf-8").splitlines()):
            if "_k" not in row.get("Name", ""):
                continue
            s = {k: float(row.get(k, 0) or 0) for k in VS}
            av = float(row.get("aiv_time(us)") or 0)
            sec = av * 1e-6
            agg = rw / sec / 1e12 if sec > 0 else 0
            top = max(s, key=s.get)
            bd = (
                "COMPUTE(vec)"
                if top == "aiv_vec_ratio"
                else (
                    "MEM(%s)" % top.replace("aiv_", "").replace("_ratio", "")
                    if "mte" in top
                    else top.replace("aiv_", "").replace("_ratio", "")
                )
            )
            p(
                "RESULT %-10s | aiv=%6.1fus vec=%.3f mte2=%.3f mte3=%.3f scal=%.3f TOP=%-4s | GM(r+w)=%.2fTB/s | %s"
                % (
                    name,
                    av,
                    s["aiv_vec_ratio"],
                    s["aiv_mte2_ratio"],
                    s["aiv_mte3_ratio"],
                    s["aiv_scalar_ratio"],
                    top.replace("aiv_", "").replace("_ratio", ""),
                    agg,
                    bd,
                )
            )
            return
    p("RESULT %s NO_CSV" % name)


def test_batch_a():
    torch.npu.set_device("npu:0")
    if "Ascend950" not in torch.npu.get_device_name():
        import pytest

        pytest.skip("not a5")
    p("=== Batch A: layernorm / silu / gelu / elementwise (compute<->memory spectrum) ===")
    torch.manual_seed(0)
    # layernorm
    m = 8192
    xl = torch.randn(m, LN_N, device="npu:0", dtype=torch.float32)
    g = torch.randn(1, LN_N, device="npu:0", dtype=torch.float32)
    b = torch.randn(1, LN_N, device="npu:0", dtype=torch.float32)
    yl = torch.empty_like(xl)
    layernorm_k[None, 32](xl, g, b, yl)
    torch.npu.synchronize()
    ref = F.layer_norm(xl, (LN_N,), g.squeeze(0), b.squeeze(0), EPS)
    d = (yl - ref).abs().max().item()
    p("GATEA layernorm maxdiff=%.2e" % d)
    if d < 1e-2:
        _prof(layernorm_k, ProfileCase("layernorm", (xl, g, b, yl), m, LN_N))
    # silu/gelu/elw at N=1024
    x = torch.randn(m, NN, device="npu:0", dtype=torch.float32)
    ys = torch.empty_like(x)
    silu_k[None, 32](x, ys)
    torch.npu.synchronize()
    d = (ys - F.silu(x)).abs().max().item()
    p("GATEA silu      maxdiff=%.2e" % d)
    if d < 1e-3:
        _prof(silu_k, ProfileCase("silu", (x, ys), m, NN))
    yg = torch.empty_like(x)
    gelu_k[None, 32](x, yg)
    torch.npu.synchronize()
    d = (yg - x * torch.sigmoid(1.702 * x)).abs().max().item()
    p("GATEA gelu(sig-approx) maxdiff=%.2e vs-F.gelu=%.2e" % (d, (yg - F.gelu(x)).abs().max().item()))
    if d < 1e-3:
        _prof(gelu_k, ProfileCase("gelu", (x, yg), m, NN))
    ye = torch.empty_like(x)
    elw_k[None, 32](x, ye)
    torch.npu.synchronize()
    d = (ye - (x * 2.0 + 1.0)).abs().max().item()
    p("GATEA elw(2x+1) maxdiff=%.2e" % d)
    if d < 1e-4:
        _prof(elw_k, ProfileCase("elw", (x, ye), m, NN))
