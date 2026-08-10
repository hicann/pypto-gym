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
# STATUS: VALIDATED a5 (2026-07-22). Interleaved RoPE, bit-exact (maxdiff 0.0), COMPUTE-bound
#   (vec 0.881). pl has NO trig op -> cos/sin precomputed per (position,freq) on host & passed in.
#   pl.load has NO stride -> even/odd pairs are HOST-SPLIT into 2D tensors (tile_dims 3D load did
#   NOT extract strided pairs, maxdiff 8.4); the rotation (ye=xe*cos-xo*sin, yo=xe*sin+xo*cos) is
#   on-device; host re-interleaves. The interleave is a host view; the compute is the kernel.
# VALIDATED-CODE-SHA256: fd9f60d3f927f68922096ba762c7ef65e350bfd469299086809db456c072e854
# Interleaved RoPE, host-split even/odd (load has no stride; tile_dims 3D didn't extract pairs).
# Kernel rotates clean 2D tiles. Cos/sin and even/odd outputs are packed as two
# contiguous H-wide halves so the fixed PyPTO ABI stays within five parameters.
import csv
import functools
import glob
import os
from pathlib import Path

import pypto_pro.language as pl
import torch  # noqa
import torch_npu

p = functools.partial(print, flush=True)
TR = 16
D = 128
H = D // 2


@pl.jit(auto_mutex=True)
def rope_k(
    xe: pl.Tensor[[pl.DYNAMIC, H], pl.DT_FP32],
    xo: pl.Tensor[[pl.DYNAMIC, H], pl.DT_FP32],
    trig: pl.Tensor[[pl.DYNAMIC, D], pl.DT_FP32],
    y_parts: pl.Tensor[[pl.DYNAMIC, D], pl.DT_FP32],
):
    tt = pl.TileType(shape=[TR, H], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    xeg = pl.make_tile_group(type=tt, addrs=[0x00000, 0x02000], mutex_ids=[0, 1])
    xog = pl.make_tile_group(type=tt, addrs=[0x04000, 0x06000], mutex_ids=[2, 3])
    csg = pl.make_tile_group(type=tt, addrs=[0x08000, 0x0A000], mutex_ids=[4, 5])
    sng = pl.make_tile_group(type=tt, addrs=[0x0C000, 0x0E000], mutex_ids=[6, 7])
    yeg = pl.make_tile_group(type=tt, addrs=[0x10000, 0x12000], mutex_ids=[8, 9])
    yog = pl.make_tile_group(type=tt, addrs=[0x14000, 0x16000], mutex_ids=[10, 11])
    t1 = pl.make_tile(tt, addr=0x18000, size=8192)
    t2 = pl.make_tile(tt, addr=0x1A000, size=8192)
    with pl.section_vector():
        rows = xe.shape[0]
        nc = pl.get_block_num()
        cid = pl.get_block_idx()
        nt = (rows + TR - 1) // TR
        for t in pl.range(cid, nt, nc):
            ro = t * TR
            vr = pl.min(TR, rows - ro)
            e = xeg.next()
            pl.set_validshape(e, [vr, H])
            pl.load(e, xe, [ro, 0])
            o = xog.next()
            pl.set_validshape(o, [vr, H])
            pl.load(o, xo, [ro, 0])
            c = csg.next()
            pl.set_validshape(c, [vr, H])
            pl.load(c, trig, [ro, 0])
            s = sng.next()
            pl.set_validshape(s, [vr, H])
            pl.load(s, trig, [ro, H])
            oe = yeg.next()
            oo = yog.next()
            pl.set_validshape(oe, [vr, H])
            pl.set_validshape(oo, [vr, H])
            pl.set_validshape(t1, [vr, H])
            pl.set_validshape(t2, [vr, H])
            pl.mul(t1, e, c)
            pl.mul(t2, o, s)
            pl.sub(oe, t1, t2)
            pl.mul(t1, e, s)
            pl.mul(t2, o, c)
            pl.add(oo, t1, t2)
            pl.store(y_parts, oe, [ro, 0])
            pl.store(y_parts, oo, [ro, H])


def test_rope():
    torch.npu.set_device("npu:0")
    if "Ascend950" not in torch.npu.get_device_name():
        import pytest

        pytest.skip("not a5")
    torch.manual_seed(0)
    m = 8192
    x = torch.randn(m, D, device="npu:0")
    pos = torch.arange(m, device="npu:0").float().unsqueeze(1)
    freq = (10000.0 ** (-torch.arange(0, H, device="npu:0").float() / H)).unsqueeze(0)
    ang = pos * freq
    cs = torch.cos(ang).contiguous()
    sn = torch.sin(ang).contiguous()
    trig = torch.cat((cs, sn), dim=1)
    xe = x[:, 0::2].contiguous()
    xo = x[:, 1::2].contiguous()  # host interleave-split
    y_parts = torch.zeros(m, D, device="npu:0")
    rope_k[None, 32](xe, xo, trig, y_parts)
    torch.npu.synchronize()
    ye = y_parts[:, :H]
    yo = y_parts[:, H:]
    y = torch.stack([ye, yo], dim=2).reshape(m, D)  # host re-interleave
    ref = torch.stack([xe * cs - xo * sn, xe * sn + xo * cs], dim=2).reshape(m, D)
    d = (y - ref).abs().max().item()
    p("GATEA rope_interleave maxdiff=%.2e" % d)
    if d < 1e-3:
        odir = os.path.join(".", "prof_rope2%d" % os.getpid())
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
                rope_k[None, 32](xe, xo, trig, y_parts)
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
                gm = (4 * m * H * 4) / (av * 1e-6) / 1e12 if av > 0 else 0
                p(
                    "RESULT rope | aiv=%.1fus vec=%.3f mte2=%.3f mte3=%.3f | GM=%.2fTB/s | %s"
                    % (av, vec, m2, m3, gm, "MEM" if max(m2, m3) > vec else "COMPUTE")
                )
                break
            break
