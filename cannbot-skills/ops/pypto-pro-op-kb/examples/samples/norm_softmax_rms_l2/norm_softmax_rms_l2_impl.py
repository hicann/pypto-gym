# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Compiler examples stay self-contained so each file exposes a complete kernel pattern.
# SAMPLE PROVENANCE -- describes THIS reference implementation only.
# Do not copy this header into generated code: a generated kernel inherits
# no validation from the sample it was modelled on.
# STATUS: VALIDATED on Ascend a5 NPU (2026-07-22). 3 pure-vector row-normalisation kernels.
#   softmax  Gate A maxdiff=1.49e-8 | vec util 0.957, GM(r+w) 2.11 TB/s -> COMPUTE-bound (exp)
#   l2norm   Gate A maxdiff=2.98e-8 | vec util 0.919, GM(r+w) 3.77 TB/s -> near BOTH limits
#   rmsnorm  Gate A maxdiff=9.54e-7 | vec util 0.910, GM(r+w) 3.74 TB/s -> near BOTH limits
# VALIDATED-CODE-SHA256: 1a942aa81def9877b3300c617e80fa93ef2cfe303f15511740d4902da8095330
# FINDING: these ops are inherently COMPUTE-bound on the a5 vector engine -- each needs ~4-5
# vector ALU passes over the data (softmax: max,sub,exp,sum,div; l2/rms: sq,sum,rsqrt,mul) but
# only 2 memory passes (1 read + 1 write), so the vector ALU (aiv_vec) saturates first (0.91-0.96).
# softmax is the most compute-bound (exp is the heaviest op). The lighter l2norm/rmsnorm also push
# GM to ~3.77 TB/s -- near the vector memory aggregate ceiling -- so they sit close to BOTH bounds.
# Structure: row-tiles [TR=8, N=1024] fp32 across 32 vector cores, double-buffered in/out/tmp for
# GM overlap, set_validshape for the tail row-tile. rsqrt + scalar ops on the ROW-MAJOR alias of
# the [TR,1] DN reduction (DN view faults on scalar ops -- the layernorm pitfall). Cloned from the
# proven softmax sample + matmul_rowwise_l2_norm reduction idiom.
# The scaled softmax, RMSNorm, and L2Norm vector kernels are characterised against both compute
# utilisation and aggregate read/write memory bandwidth ceilings.
import csv
import functools
import glob
import os
from dataclasses import dataclass
from pathlib import Path

import pypto_pro.language as pl
import torch  # noqa
import torch_npu

p = functools.partial(print, flush=True)
TR = 8
NN = 1024
EPS = 1e-6


@pl.jit(auto_mutex=True)
def softmax_k(x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32], y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32]):
    tt = pl.TileType(shape=[TR, NN], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    rt = pl.TileType(
        shape=[TR, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN, valid_shape=[-1, -1]
    )
    ing = pl.make_tile_group(type=tt, addrs=[0x00000, 0x08000], mutex_ids=[0, 1])
    outg = pl.make_tile_group(type=tt, addrs=[0x10000, 0x18000], mutex_ids=[2, 3])
    tmpg = pl.make_tile_group(type=tt, addrs=[0x20000, 0x28000], mutex_ids=[4, 5])
    redg = pl.make_tile_group(type=rt, addrs=[0x30000, 0x30400], mutex_ids=[6, 7])
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
            o = outg.next()
            tm = tmpg.next()
            rd = redg.next()
            pl.set_validshape(o, [vr, cols])
            pl.set_validshape(tm, [vr, cols])
            pl.set_validshape(rd, [vr, 1])
            pl.row_max(rd, xi, tm)
            pl.row_expand_sub(o, xi, rd)
            pl.exp(o, o)
            pl.row_sum(rd, o, tm)
            pl.row_expand_div(o, o, rd)
            pl.store(y, o, [ro, 0])


@pl.jit(auto_mutex=True)
def l2norm_k(x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32], y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32]):
    tt = pl.TileType(shape=[TR, NN], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    rt = pl.TileType(
        shape=[TR, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN, valid_shape=[-1, -1]
    )
    ing = pl.make_tile_group(type=tt, addrs=[0x00000, 0x08000], mutex_ids=[0, 1])
    outg = pl.make_tile_group(type=tt, addrs=[0x10000, 0x18000], mutex_ids=[2, 3])
    tmpg = pl.make_tile_group(type=tt, addrs=[0x20000, 0x28000], mutex_ids=[4, 5])
    redg = pl.make_tile_group(type=rt, addrs=[0x30000, 0x30400], mutex_ids=[6, 7])
    rd = pl.make_tile(
        pl.TileType(shape=[TR, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN),
        addr=0x30800,
        size=512,
    )
    rdm = pl.make_tile(
        pl.TileType(shape=[1, TR], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addr=0x30800, size=512
    )
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
            o = outg.next()
            sq = tmpg.next()
            pl.set_validshape(o, [vr, cols])
            pl.set_validshape(sq, [vr, cols])
            pl.mul(sq, xi, xi)
            pl.row_sum(rd, sq, o)
            pl.add(rdm, rdm, EPS)
            pl.rsqrt(rdm, rdm)
            pl.row_expand_mul(o, xi, rd)
            pl.store(y, o, [ro, 0])


@pl.jit(auto_mutex=True)
def rmsnorm_k(x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32], y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32]):
    tt = pl.TileType(shape=[TR, NN], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    rt = pl.TileType(
        shape=[TR, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN, valid_shape=[-1, -1]
    )
    ing = pl.make_tile_group(type=tt, addrs=[0x00000, 0x08000], mutex_ids=[0, 1])
    outg = pl.make_tile_group(type=tt, addrs=[0x10000, 0x18000], mutex_ids=[2, 3])
    tmpg = pl.make_tile_group(type=tt, addrs=[0x20000, 0x28000], mutex_ids=[4, 5])
    redg = pl.make_tile_group(type=rt, addrs=[0x30000, 0x30400], mutex_ids=[6, 7])
    rd = pl.make_tile(
        pl.TileType(shape=[TR, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN),
        addr=0x30800,
        size=512,
    )
    rdm = pl.make_tile(
        pl.TileType(shape=[1, TR], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addr=0x30800, size=512
    )
    inv_n = 1.0 / NN
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
            o = outg.next()
            sq = tmpg.next()
            pl.set_validshape(o, [vr, cols])
            pl.set_validshape(sq, [vr, cols])
            pl.mul(sq, xi, xi)
            pl.row_sum(rd, sq, o)
            pl.mul(rdm, rdm, inv_n)
            pl.add(rdm, rdm, EPS)
            pl.rsqrt(rdm, rdm)
            pl.row_expand_mul(o, xi, rd)
            pl.store(y, o, [ro, 0])


def _cfg():
    return vars(torch_npu.profiler)["_ExperimentalConfig"](
        export_type=[torch_npu.profiler.ExportType.Text],
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    )


VS = ["aiv_vec_ratio", "aiv_scalar_ratio", "aiv_mte2_ratio", "aiv_mte3_ratio"]
VEC_CEIL = 2.3


@dataclass(frozen=True)
class ProfileCase:
    name: str
    tensors: tuple[torch.Tensor, torch.Tensor]
    rows: int
    cols: int


def _prof(kern, case: ProfileCase):
    name = case.name
    x, y = case.tensors
    m, n = case.rows, case.cols
    odir = os.path.join(".", "prof_norm%d" % os.getpid(), name)
    os.makedirs(odir, exist_ok=True)
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        experimental_config=_cfg(),
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=3),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(odir, analyse_flag=True),
    ) as prof:
        for _ in range(6):
            kern[None, 32](x, y)
            torch.npu.synchronize()
            prof.step()
    rw = 2 * m * n * 4
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
                "RESULT %-9s | aiv=%6.1fus vec=%.3f mte2=%.3f mte3=%.3f scal=%.3f "
                "TOP=%-4s | GM(r+w)=%.2fTB/s | BOUND=%s"
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


def test_norm():
    torch.npu.set_device("npu:0")
    if "Ascend950" not in torch.npu.get_device_name():
        import pytest

        pytest.skip("not a5")
    p("=== softmax / rmsnorm / l2norm : compute vs memory bound ===")
    torch.manual_seed(0)
    m, n = 8192, NN
    x = torch.randn(m, n, device="npu:0", dtype=torch.float32)
    y = torch.empty_like(x)
    softmax_k[None, 32](x, y)
    torch.npu.synchronize()
    d = (y - torch.softmax(x, dim=1)).abs().max().item()
    p("GATEA softmax  maxdiff=%.2e" % d)
    if d < 1e-4:
        _prof(softmax_k, ProfileCase("softmax", (x, y), m, n))
    y2 = torch.empty_like(x)
    l2norm_k[None, 32](x, y2)
    torch.npu.synchronize()
    d2 = (y2 - x * torch.rsqrt((x * x).sum(1, keepdim=True) + EPS)).abs().max().item()
    p("GATEA l2norm   maxdiff=%.2e" % d2)
    if d2 < 1e-4:
        _prof(l2norm_k, ProfileCase("l2norm", (x, y2), m, n))
    y3 = torch.empty_like(x)
    rmsnorm_k[None, 32](x, y3)
    torch.npu.synchronize()
    d3 = (y3 - x * torch.rsqrt((x * x).mean(1, keepdim=True) + EPS)).abs().max().item()
    p("GATEA rmsnorm  maxdiff=%.2e" % d3)
    if d3 < 1e-4:
        _prof(rmsnorm_k, ProfileCase("rmsnorm", (x, y3), m, n))
