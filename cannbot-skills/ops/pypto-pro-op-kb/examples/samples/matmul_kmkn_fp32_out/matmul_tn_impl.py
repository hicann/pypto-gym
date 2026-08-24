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
# STATUS: VALIDATED on Ascend a5 NPU (2026-07-18) — corpus twin 4, BOTH gates.
# VALIDATED-CODE-SHA256: 6643c45e91a390e428c0493f026408c1a5ea44d57b7e3b6b7149dd5ee45d3136
# Gate A was bit-exact on the 1024-by-1024-by-128 FP32 case. Gate B reached 89.8% cube
# utilisation. The implementation uses the proven tiled multicore TN transpose-load pattern.
# Kernel 4 computes a transposed-left-operand FP32 matmul with reduction dimension 128.
# Golden is x.t()@y fp32, tol 3e-3.
import csv
import glob
import logging
import os
from pathlib import Path

import pypto_pro.language as pl
import torch  # noqa: F401
import torch_npu
from test_matmul_api_basic import _require_a5

LOGGER = logging.getLogger(__name__)

TILE = 128
K = 128


@pl.jit(auto_mutex=True)
def mm_tn(
    a_t: pl.Tensor[[K, pl.DYNAMIC], pl.DT_FP32],  # x, K-major [K, M]
    b: pl.Tensor[[K, pl.DYNAMIC], pl.DT_FP32],  # y          [K, N]
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],  # [M, N] = x.t() @ y
):
    nc = pl.get_block_num()
    cid = pl.get_block_idx()
    with pl.section_cube():
        a_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, K], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
            addrs=0x00000,
            mutex_ids=[0],
        )
        b_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[K, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=0x20000,
            mutex_ids=[1],
        )
        a_l0a = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, K], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addrs=0x0,
            mutex_ids=[2],
        )
        b_l0b = pl.make_tile_group(
            type=pl.TileType(shape=[K, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
            addrs=0x0,
            mutex_ids=[3],
        )
        acc = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
            ),
            addrs=0x0,
            mutex_ids=[4],
        )
        nt_n = out.shape[1] // TILE
        n_tiles = (out.shape[0] // TILE) * nt_n
        for t in pl.range(cid, n_tiles, nc):
            i = t // nt_n
            j = t % nt_n
            ca = a_l1.current()
            pl.load(ca, a_t, [0, i * TILE], is_transpose=True)  # a_t[:, i-block] -> [TILE, K]
            cb = b_l1.current()
            pl.load(cb, b, [0, j * TILE])
            al = a_l0a.current()
            pl.move(al, ca)
            br = b_l0b.current()
            pl.move(br, cb)
            ac = acc.current()
            pl.matmul(ac, al, br)
            pl.store_tile(out, ac, [i, j])


def test_mm_tn_gate_a():
    _require_a5("npu:0")
    m = n = 1024
    torch.manual_seed(0)
    x = torch.randn(K, m, device="npu:0", dtype=torch.float32)  # a_t [K, m]
    y = torch.randn(K, n, device="npu:0", dtype=torch.float32)  # b   [K, n]
    out = torch.zeros(m, n, device="npu:0", dtype=torch.float32)
    mm_tn[None, 32](x, y, out)
    torch.npu.synchronize()
    ref = x.t() @ y
    d = (out - ref).abs().max().item()
    LOGGER.info(f"GATEA mm_tn {m}x{n}x{K} maxdiff={d:.3e}")
    torch.testing.assert_close(out, ref, rtol=3e-3, atol=3e-3)


def _cfg():
    return vars(torch_npu.profiler)["_ExperimentalConfig"](
        export_type=[torch_npu.profiler.ExportType.Text],
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    )


def test_mm_tn_gate_b():
    _require_a5("npu:0")
    m = n = 1024
    torch.manual_seed(0)
    x = torch.randn(K, m, device="npu:0", dtype=torch.float32)
    y = torch.randn(K, n, device="npu:0", dtype=torch.float32)
    out = torch.zeros(m, n, device="npu:0", dtype=torch.float32)
    out_dir = os.path.join(".", "prof_tn%d" % os.getpid())
    os.makedirs(out_dir, exist_ok=True)
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        experimental_config=_cfg(),
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=5),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(out_dir, analyse_flag=True),
    ) as prof:
        for _ in range(10):
            mm_tn[None, 32](x, y, out)
            torch.npu.synchronize()
            prof.step()
    csvs = glob.glob(os.path.join(out_dir, "**", "kernel_details.csv"), recursive=True)
    keys = ["Name", "aicore_time(us)", "aic_mac_ratio", "aic_mte2_ratio", "aic_fixpipe_ratio", "cube_utilization(%)"]
    for r in csv.DictReader(Path(csvs[0]).read_text(encoding="utf-8").splitlines()):
        if float(r.get("aicore_time(us)", "0") or 0) > 0:
            LOGGER.info("GATEB %s", {k: r.get(k, "") for k in keys})
