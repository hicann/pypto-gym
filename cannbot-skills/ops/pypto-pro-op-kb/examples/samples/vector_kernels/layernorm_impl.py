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
# STATUS: VALIDATED on Ascend a5 (2026-07-22). layernorm (mean+var+gamma+beta), fp32.
#   Gate A maxdiff=2.86e-6 vs F.layer_norm | vec util 0.954, GM(r+w) 1.42 TB/s -> COMPUTE-bound.
# VALIDATED-CODE-SHA256: d7334ce556cbb14175616f268dcde51c4153300208e21afee70b13a293139429
# KEY FIX: gamma/beta/negbeta MUST be make_tile_group (not bare make_tile) -- a bare COMPUTED tile
# (negbeta=mul(beta,-1)) is not auto_mutex-synced, so col_expand_sub read stale data (maxdiff~5).
# Two reductions (mean, then var of centered); rsqrt/div on the row-major alias; -beta via
# col_expand_sub (no col_expand_add exists). Cloned from tile_vector/test_layernorm.py.
# layernorm FIX: big tiles as make_tile_group (auto_mutex-tracked); reductions single aliased make_tile.
import csv
import functools
import glob
import os
from pathlib import Path

import pypto_pro.language as pl
import torch  # noqa
import torch.nn.functional as F
import torch_npu

p = functools.partial(print, flush=True)
LN_N = 512
LN_TR = 8
EPS = 1e-5


@pl.jit(auto_mutex=True)
def layernorm_k(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    g: pl.Tensor[[1, LN_N], pl.DT_FP32],
    b: pl.Tensor[[1, LN_N], pl.DT_FP32],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tt = pl.TileType(shape=[LN_TR, LN_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    gt = pl.TileType(shape=[1, LN_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    rt = pl.TileType(
        shape=[LN_TR, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN, valid_shape=[-1, -1]
    )
    rmt = pl.TileType(shape=[1, LN_TR], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    ing = pl.make_tile_group(type=tt, addrs=[0x00000, 0x04000], mutex_ids=[0, 1])
    xcg = pl.make_tile_group(type=tt, addrs=[0x08000, 0x0C000], mutex_ids=[2, 3])
    tmg = pl.make_tile_group(type=tt, addrs=[0x10000, 0x14000], mutex_ids=[4, 5])
    # gamma/beta loaded once, single tiles
    gamg = pl.make_tile_group(type=tt, addrs=[0x18000], mutex_ids=[10])
    betg = pl.make_tile_group(type=tt, addrs=[0x1C000], mutex_ids=[11])
    nbeg = pl.make_tile_group(type=tt, addrs=[0x24000], mutex_ids=[12])
    r0 = pl.make_tile(rt, addr=0x20000, size=64)
    r0m = pl.make_tile(rmt, addr=0x20000, size=64)
    r1 = pl.make_tile(rt, addr=0x20100, size=64)
    r1m = pl.make_tile(rmt, addr=0x20100, size=64)
    with pl.section_vector():
        rows = x.shape[0]
        cols = x.shape[1]
        nc = pl.get_block_num()
        cid = pl.get_block_idx()
        gam = gamg.next()
        pl.set_validshape(gam, [1, cols])
        pl.load(gam, g, [0, 0])
        bet = betg.next()
        pl.set_validshape(bet, [1, cols])
        pl.load(bet, b, [0, 0])
        nbe = nbeg.next()
        pl.set_validshape(nbe, [1, cols])
        pl.mul(nbe, bet, -1.0)
        nt = (rows + LN_TR - 1) // LN_TR
        for t in pl.range(cid, nt, nc):
            ro = t * LN_TR
            vr = pl.min(LN_TR, rows - ro)
            xin = ing.next()
            pl.set_validshape(xin, [vr, cols])
            pl.load(xin, x, [ro, 0])
            xc = xcg.next()
            tmp = tmg.next()
            pl.set_validshape(xc, [vr, cols])
            pl.set_validshape(tmp, [vr, cols])
            pl.set_validshape(r0, [vr, 1])
            pl.set_validshape(r1, [vr, 1])
            pl.row_sum(r0, xin, tmp)
            pl.set_validshape(r0m, [1, vr])
            pl.div(r0m, r0m, 512.0)  # mean
            pl.row_expand_sub(xc, xin, r0)  # x-mean
            pl.mul(tmp, xc, xc)
            pl.row_sum(r1, tmp, xin)
            pl.set_validshape(r1m, [1, vr])  # var num (xin as ws)
            pl.div(r1m, r1m, 512.0)
            pl.add(r1m, r1m, EPS)
            pl.rsqrt(r1m, r1m)  # 1/std
            pl.row_expand_mul(xc, xc, r1)
            pl.col_expand_mul(xc, xc, gam)
            pl.col_expand_sub(tmp, xc, nbe)  # *invstd*gamma+beta
            pl.store(y, tmp, [ro, 0])


def _cfg():
    return vars(torch_npu.profiler)["_ExperimentalConfig"](
        export_type=[torch_npu.profiler.ExportType.Text],
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    )


VS = ["aiv_vec_ratio", "aiv_scalar_ratio", "aiv_mte2_ratio", "aiv_mte3_ratio"]


def test_ln():
    torch.npu.set_device("npu:0")
    if "Ascend950" not in torch.npu.get_device_name():
        import pytest

        pytest.skip("not a5")
    torch.manual_seed(0)
    m = 8192
    x = torch.randn(m, LN_N, device="npu:0")
    g = torch.randn(1, LN_N, device="npu:0")
    b = torch.randn(1, LN_N, device="npu:0")
    y = torch.empty_like(x)
    layernorm_k[None, 32](x, g, b, y)
    torch.npu.synchronize()
    ref = F.layer_norm(x, (LN_N,), g.squeeze(0), b.squeeze(0), EPS)
    d = (y - ref).abs().max().item()
    p("GATEA layernorm maxdiff=%.2e" % d)
    if d < 1e-2:
        odir = os.path.join(".", "prof_ln%d" % os.getpid())
        os.makedirs(odir, exist_ok=True)
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            experimental_config=_cfg(),
            schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=3),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(odir, analyse_flag=True),
        ) as prof:
            for _ in range(6):
                layernorm_k[None, 32](x, g, b, y)
                torch.npu.synchronize()
                prof.step()
        for cf in glob.glob(os.path.join(odir, "**", "kernel_details.csv"), recursive=True):
            for row in csv.DictReader(Path(cf).read_text(encoding="utf-8").splitlines()):
                if "_k" not in row.get("Name", ""):
                    continue
                s = {k: float(row.get(k, 0) or 0) for k in VS}
                av = float(row.get("aiv_time(us)") or 0)
                sec = av * 1e-6
                agg = 2 * m * LN_N * 4 / sec / 1e12 if sec > 0 else 0
                top = max(s, key=s.get)
                p(
                    "RESULT layernorm | aiv=%.1fus vec=%.3f mte2=%.3f mte3=%.3f | GM(r+w)=%.2fTB/s | %s"
                    % (
                        av,
                        s["aiv_vec_ratio"],
                        s["aiv_mte2_ratio"],
                        s["aiv_mte3_ratio"],
                        agg,
                        "COMPUTE(vec)" if top == "aiv_vec_ratio" else "MEM",
                    )
                )
                break
            break
