# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# vf l2norm (sqrt/div, NO rsqrt) + vf silu (x/(1+exp(-x))). tile-op: l2norm 17.8us, silu 22.2us.
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
LANES = 64
MAX_N = 1024
TR = 8
EPS = 1e-6
SB = TR * MAX_N * 4


@pl.vector_function
def l2_vf(in_tile, out_tile, n_rows: pl.DT_INT64, n_cols: pl.DT_INT64):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    n_regs = (n_cols + LANES - 1) // LANES
    for m in pl.range(0, n_rows):
        base = m * MAX_N
        row_ss = vf.full(0.0, preg, dtype=pl.DT_FP32)
        for r in pl.range(0, n_regs):
            valid = pl.min(LANES, n_cols - r * LANES)
            mreg = vf.update_mask(valid, dtype=pl.DT_FP32)
            reg = vf.load_align(in_tile, base + r * LANES)
            sq = vf.mul(reg, reg, mreg)
            part = vf.reduce_sum(sq, mreg)
            row_ss = vf.add(row_ss, part, preg)
        row_ss = vf.adds(row_ss, EPS, preg)
        norm = vf.sqrt(row_ss, preg)
        norm_b = vf.full(norm, preg)  # sqrt(ss+eps)
        for r in pl.range(0, n_regs):
            valid = pl.min(LANES, n_cols - r * LANES)
            mreg = vf.update_mask(valid, dtype=pl.DT_FP32)
            reg = vf.load_align(in_tile, base + r * LANES)
            out = vf.div(reg, norm_b, mreg)
            vf.store_align(out_tile + base + r * LANES, out, mreg)


@pl.vector_function
def silu_vf(in_tile, out_tile, n_rows: pl.DT_INT64, n_cols: pl.DT_INT64):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    n_regs = (n_cols + LANES - 1) // LANES
    negone = vf.full(-1.0, preg, dtype=pl.DT_FP32)
    one = vf.full(1.0, preg, dtype=pl.DT_FP32)
    for m in pl.range(0, n_rows):
        base = m * MAX_N
        for r in pl.range(0, n_regs):
            valid = pl.min(LANES, n_cols - r * LANES)
            mreg = vf.update_mask(valid, dtype=pl.DT_FP32)
            reg = vf.load_align(in_tile, base + r * LANES)
            t = vf.mul(reg, negone, mreg)
            t = vf.exp(t, mreg)
            t = vf.add(t, one, mreg)
            out = vf.div(reg, t, mreg)  # x/(1+exp(-x))
            vf.store_align(out_tile + base + r * LANES, out, mreg)


def _mk(fn):
    @pl.jit(auto_mutex=True)
    def k(x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32], y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32]):
        tt = pl.TileType(shape=[TR, MAX_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
        ing = pl.make_tile_group(type=tt, addrs=[0, SB], mutex_ids=[0, 1])
        outg = pl.make_tile_group(type=tt, addrs=[2 * SB, 3 * SB], mutex_ids=[2, 3])
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
                pl.set_validshape(o, [vr, cols])
                fn(xi, o, vr, cols)
                pl.store(y, o, [ro, 0])

    return k


l2_vf_k = _mk(l2_vf)
silu_vf_k = _mk(silu_vf)


@dataclass(frozen=True)
class ProfileCase:
    name: str
    tensors: tuple[torch.Tensor, torch.Tensor]
    rows: int
    cols: int
    baseline: str


def _prof(k, case: ProfileCase):
    name = case.name
    x, y = case.tensors
    base = case.baseline
    odir = os.path.join(".", "prof_vf_%d" % os.getpid(), name)
    os.makedirs(odir, exist_ok=True)
    cfg = vars(torch_npu.profiler)["_ExperimentalConfig"](
        export_type=[torch_npu.profiler.ExportType.Text],
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    )
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        experimental_config=cfg,
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=3),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(odir, analyse_flag=True),
    ) as prof:
        for _ in range(6):
            k[None, 32](x, y)
            torch.npu.synchronize()
            prof.step()
    for cf in glob.glob(os.path.join(odir, "**", "kernel_details.csv"), recursive=True):
        for row in csv.DictReader(Path(cf).read_text(encoding="utf-8").splitlines()):
            if "k" not in row.get("Name", ""):
                continue
            av = float(row.get("aiv_time(us)") or 0)
            vec = float(row.get("aiv_vec_ratio") or 0)
            m2 = float(row.get("aiv_mte2_ratio") or 0)
            p("RESULT vf_%-8s | aiv=%.1fus vec=%.3f mte2=%.3f (tile-op %s us)" % (name, av, vec, m2, base))
            return


def test_vf():
    torch.npu.set_device("npu:0")
    if "Ascend950" not in torch.npu.get_device_name():
        import pytest

        pytest.skip("not a5")
    torch.manual_seed(0)
    m = 8192
    n = 1024
    x = torch.randn(m, n, device="npu:0")
    y = torch.empty(m, n, device="npu:0")
    l2_vf_k[None, 32](x, y)
    torch.npu.synchronize()
    d = (y - x * torch.rsqrt((x * x).sum(1, keepdim=True) + EPS)).abs().max().item()
    p("GATEA vf_l2norm maxdiff=%.2e" % d)
    if d < 1e-4:
        _prof(l2_vf_k, ProfileCase("l2norm", (x, y), m, n, "17.8"))
    y2 = torch.empty(m, n, device="npu:0")
    silu_vf_k[None, 32](x, y2)
    torch.npu.synchronize()
    d2 = (y2 - F.silu(x)).abs().max().item()
    p("GATEA vf_silu maxdiff=%.2e" % d2)
    if d2 < 1e-3:
        _prof(silu_vf_k, ProfileCase("silu", (x, y2), m, n, "22.2"))
