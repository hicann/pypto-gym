# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# vf Batch 2: layernorm (clone vf template) + rope (interleaved, multi-op elementwise).
# RoPE packs cos/sin and even/odd outputs as contiguous H-wide halves. This keeps
# both the kernel and vector-function ABI within five parameters without suppression.
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
LANES = 64
EPS = 1e-5
# --- layernorm: N=512, TR=8 ---
LN = 512
LTR = 8
LSB = LTR * LN * 4


@pl.vector_function
def ln_vf(in_tile, out_tile, gamma_tile, beta_tile, n_rows: pl.DT_INT64):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    n_regs = (LN + LANES - 1) // LANES
    n_reg_f = vf.full(LN, preg, dtype=pl.DT_FP32)
    for m in pl.range(0, n_rows):
        base = m * LN
        row_sum = vf.full(0.0, preg, dtype=pl.DT_FP32)
        for r in pl.range(0, n_regs):
            valid = pl.min(LANES, LN - r * LANES)
            mreg = vf.update_mask(valid, dtype=pl.DT_FP32)
            reg = vf.load_align(in_tile, base + r * LANES)
            part = vf.reduce_sum(reg, mreg)
            row_sum = vf.add(row_sum, part, preg)
        mean_b = vf.full(row_sum, preg)
        mean_b = vf.div(mean_b, n_reg_f, preg)
        var_sum = vf.full(0.0, preg, dtype=pl.DT_FP32)
        for r in pl.range(0, n_regs):
            valid = pl.min(LANES, LN - r * LANES)
            mreg = vf.update_mask(valid, dtype=pl.DT_FP32)
            reg = vf.load_align(in_tile, base + r * LANES)
            xc = vf.sub(reg, mean_b, mreg)
            sq = vf.mul(xc, xc, mreg)
            part = vf.reduce_sum(sq, mreg)
            var_sum = vf.add(var_sum, part, preg)
        var_b = vf.full(var_sum, preg)
        var_b = vf.div(var_b, n_reg_f, preg)
        var_b = vf.adds(var_b, EPS, preg)
        std_b = vf.sqrt(var_b, preg)
        for r in pl.range(0, n_regs):
            valid = pl.min(LANES, LN - r * LANES)
            mreg = vf.update_mask(valid, dtype=pl.DT_FP32)
            reg = vf.load_align(in_tile, base + r * LANES)
            gm = vf.load_align(gamma_tile, r * LANES)
            bt = vf.load_align(beta_tile, r * LANES)
            xc = vf.sub(reg, mean_b, mreg)
            norm = vf.div(xc, std_b, mreg)
            out = vf.mul(norm, gm, mreg)
            out = vf.add(out, bt, mreg)
            vf.store_align(out_tile + base + r * LANES, out, mreg)


@pl.jit(auto_mutex=True)
def ln_k(
    x: pl.Tensor[[pl.DYNAMIC, LN], pl.DT_FP32],
    g: pl.Tensor[[1, LN], pl.DT_FP32],
    b: pl.Tensor[[1, LN], pl.DT_FP32],
    y: pl.Tensor[[pl.DYNAMIC, LN], pl.DT_FP32],
):
    tt = pl.TileType(shape=[LTR, LN], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    gt = pl.TileType(shape=[1, LN], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    ing = pl.make_tile_group(type=tt, addrs=[0, LSB], mutex_ids=[0, 1])
    outg = pl.make_tile_group(type=tt, addrs=[2 * LSB, 3 * LSB], mutex_ids=[2, 3])
    gg = pl.make_tile_group(type=gt, addrs=[4 * LSB], mutex_ids=[4])
    bg = pl.make_tile_group(type=gt, addrs=[4 * LSB + 2048], mutex_ids=[5])
    with pl.section_vector():
        rows = x.shape[0]
        nc = pl.get_block_num()
        cid = pl.get_block_idx()
        nt = (rows + LTR - 1) // LTR
        gs = gg.next()
        pl.load(gs, g, [0, 0])
        bs = bg.next()
        pl.load(bs, b, [0, 0])
        for t in pl.range(cid, nt, nc):
            ro = t * LTR
            vr = pl.min(LTR, rows - ro)
            xi = ing.next()
            pl.set_validshape(xi, [vr, LN])
            pl.load(xi, x, [ro, 0])
            o = outg.next()
            pl.set_validshape(o, [vr, LN])
            ln_vf(xi, o, gs, bs, vr)
            pl.store(y, o, [ro, 0])


# --- rope: D=128, H=64, TR=16. even/odd host-split. ye=xe*c-xo*s; yo=xe*s+xo*c. register-resident. ---
D = 128
H = 64
RTR = 16
RSB = RTR * H * 4


@pl.vector_function
def rope_vf(xe_t, xo_t, trig_t, y_t, n_rows: pl.DT_INT64):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    n_regs = (H + LANES - 1) // LANES
    for m in pl.range(0, n_rows):
        base = m * H
        pair_base = m * D
        for r in pl.range(0, n_regs):
            valid = pl.min(LANES, H - r * LANES)
            mreg = vf.update_mask(valid, dtype=pl.DT_FP32)
            xe = vf.load_align(xe_t, base + r * LANES)
            xo = vf.load_align(xo_t, base + r * LANES)
            c = vf.load_align(trig_t, pair_base + r * LANES)
            s = vf.load_align(trig_t, pair_base + H + r * LANES)
            a = vf.mul(xe, c, mreg)
            b = vf.mul(xo, s, mreg)
            ye = vf.sub(a, b, mreg)  # xe*c - xo*s
            a2 = vf.mul(xe, s, mreg)
            b2 = vf.mul(xo, c, mreg)
            yo = vf.add(a2, b2, mreg)  # xe*s + xo*c
            vf.store_align(y_t + pair_base + r * LANES, ye, mreg)
            vf.store_align(y_t + pair_base + H + r * LANES, yo, mreg)


@pl.jit(auto_mutex=True)
def rope_k(
    xe: pl.Tensor[[pl.DYNAMIC, H], pl.DT_FP32],
    xo: pl.Tensor[[pl.DYNAMIC, H], pl.DT_FP32],
    trig: pl.Tensor[[pl.DYNAMIC, D], pl.DT_FP32],
    y_parts: pl.Tensor[[pl.DYNAMIC, D], pl.DT_FP32],
):
    tt = pl.TileType(shape=[RTR, H], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    pair_tt = pl.TileType(shape=[RTR, D], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    xeg = pl.make_tile_group(type=tt, addrs=[0, RSB], mutex_ids=[0, 1])
    xog = pl.make_tile_group(type=tt, addrs=[2 * RSB, 3 * RSB], mutex_ids=[2, 3])
    trigg = pl.make_tile_group(type=pair_tt, addrs=[4 * RSB, 6 * RSB], mutex_ids=[4, 5])
    outg = pl.make_tile_group(type=pair_tt, addrs=[8 * RSB, 10 * RSB], mutex_ids=[6, 7])
    with pl.section_vector():
        rows = xe.shape[0]
        nc = pl.get_block_num()
        cid = pl.get_block_idx()
        nt = (rows + RTR - 1) // RTR
        for t in pl.range(cid, nt, nc):
            ro = t * RTR
            vr = pl.min(RTR, rows - ro)
            e = xeg.next()
            pl.set_validshape(e, [vr, H])
            pl.load(e, xe, [ro, 0])
            o = xog.next()
            pl.set_validshape(o, [vr, H])
            pl.load(o, xo, [ro, 0])
            trig_tile = trigg.next()
            pl.set_validshape(trig_tile, [vr, D])
            pl.load(trig_tile, trig, [ro, 0])
            y_tile = outg.next()
            pl.set_validshape(y_tile, [vr, D])
            rope_vf(e, o, trig_tile, y_tile, vr)
            pl.store(y_parts, y_tile, [ro, 0])


def _prof(k, name, args, base):
    odir = os.path.join(".", "prof_vfb2_%d" % os.getpid(), name)
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
            k[None, 32](*args)
            torch.npu.synchronize()
            prof.step()
    for cf in glob.glob(os.path.join(odir, "**", "kernel_details.csv"), recursive=True):
        for row in csv.DictReader(Path(cf).read_text(encoding="utf-8").splitlines()):
            if "_k" not in row.get("Name", ""):
                continue
            av = float(row.get("aiv_time(us)") or 0)
            vec = float(row.get("aiv_vec_ratio") or 0)
            m2 = float(row.get("aiv_mte2_ratio") or 0)
            p("RESULT vf_%-9s | aiv=%.1fus vec=%.3f mte2=%.3f (tile-op %s us)" % (name, av, vec, m2, base))
            return


def test_b2():
    torch.npu.set_device("npu:0")
    if "Ascend950" not in torch.npu.get_device_name():
        import pytest

        pytest.skip("not a5")
    torch.manual_seed(0)
    m = 8192
    xl = torch.randn(m, LN, device="npu:0")
    g = torch.randn(1, LN, device="npu:0")
    b = torch.randn(1, LN, device="npu:0")
    yl = torch.empty(m, LN, device="npu:0")
    ln_k[None, 32](xl, g, b, yl)
    torch.npu.synchronize()
    d = (yl - F.layer_norm(xl, (LN,), g.squeeze(0), b.squeeze(0), EPS)).abs().max().item()
    p("GATEA vf_layernorm maxdiff=%.2e" % d)
    if d < 1e-2:
        _prof(ln_k, "layernorm", (xl, g, b, yl), "23.6")
    x = torch.randn(m, D, device="npu:0")
    pos = torch.arange(m, device="npu:0").float().unsqueeze(1)
    freq = (10000.0 ** (-torch.arange(0, H, device="npu:0").float() / H)).unsqueeze(0)
    ang = pos * freq
    cs = torch.cos(ang).contiguous()
    sn = torch.sin(ang).contiguous()
    trig = torch.cat((cs, sn), dim=1)
    xe = x[:, 0::2].contiguous()
    xo = x[:, 1::2].contiguous()
    y_parts = torch.zeros(m, D, device="npu:0")
    rope_k[None, 32](xe, xo, trig, y_parts)
    torch.npu.synchronize()
    ye = y_parts[:, :H]
    yo = y_parts[:, H:]
    y = torch.stack([ye, yo], dim=2).reshape(m, D)
    ref = torch.stack([xe * cs - xo * sn, xe * sn + xo * cs], dim=2).reshape(m, D)
    d2 = (y - ref).abs().max().item()
    p("GATEA vf_rope maxdiff=%.2e" % d2)
    if d2 < 1e-3:
        _prof(rope_k, "rope", (xe, xo, trig, y_parts), "8.1")
