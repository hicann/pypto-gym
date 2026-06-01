# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0
# -----------------------------------------------------------------------------------------------------------
"""Reproducible benchmark: LLaDA2 MoE Grouped GEMM (eager vs PyPTO).

Measures per-expert Python-loop eager versus single-kernel PyPTO grouped GEMM.
Both paths use pre-allocated output buffers for a fair comparison.

Usage:
    PTO_TILE_LIB_CODE_PATH=/path/to/pto-isa \
    TILE_FWK_DEVICE_ID=14 \
    python src/pypto_gym/transformers/llada2_moe/bench_grouped_gemm.py
"""

import os
import sys
import time

_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, "src")):
    parent = os.path.dirname(_p)
    if parent == _p:
        raise RuntimeError("could not locate src/")
    _p = parent
sys.path.insert(0, os.path.join(_p, "src"))

import torch
import torch.nn.functional as F

try:
    import torch_npu  # noqa: F401
except ImportError:
    print("torch_npu not available; this benchmark only runs on Ascend NPU.")
    sys.exit(0)

import pypto

pypto.set_host_options(
    compile_monitor_enable=True,
    compile_timeout=10,
    compile_timeout_stage=5,
    compile_monitor_print_interval=2,
)
torch_npu.npu.config.allow_internal_format = True

device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 14))
torch.npu.set_device(device_id)
DEV = f"npu:{device_id}"

from pypto_gym.ops.pypto_tile.llada2_moe.llada2_moe_grouped_gemm_impl import (
    llada2_moe_grouped_gemm,
)

H, I = 2048, 512
ITERS = 200
WARMUP = 20


def make_data(E, N_total):
    tpe = [N_total // E] * E
    for i in range(N_total - sum(tpe)):
        tpe[i] += 1
    cumsum = [0]
    for c in tpe:
        cumsum.append(cumsum[-1] + c)
    st = torch.randn(N_total, H, dtype=torch.bfloat16, device=DEV) * 0.02
    w13f = torch.randn(E * H, 2 * I, dtype=torch.bfloat16, device=DEV) * 0.02
    w2f = torch.randn(E * I, H, dtype=torch.bfloat16, device=DEV) * 0.02
    cs = torch.tensor(cumsum, dtype=torch.int32, device=DEV)
    res = torch.zeros(N_total, H, dtype=torch.bfloat16, device=DEV)
    return st, w13f, w2f, cs, res, tpe


def eager_grouped_prealloc(st, w13f, w2f, tpe, E, result_buf):
    """Eager with pre-allocated output buffer (fair comparison)."""
    offset = 0
    for e in range(E):
        n_e = tpe[e]
        if n_e == 0:
            continue
        x_e = st[offset : offset + n_e]
        w13_e = w13f[e * H : (e + 1) * H, :]
        w2_e = w2f[e * I : (e + 1) * I, :]
        gu = x_e @ w13_e
        sw = F.silu(gu[..., :I]) * gu[..., I:]
        result_buf[offset : offset + n_e] = (sw @ w2_e).to(torch.bfloat16)
        offset += n_e


configs = [
    (4, 16),
    (8, 32),
    (16, 64),
    (32, 64),
    (64, 128),
    (8, 64),
    (4, 64),
]

print(f"=== LLaDA2 MoE Grouped GEMM Benchmark (H={H}, I={I}) ===")
print(f"Device: NPU {device_id}  |  Iters: {ITERS}  |  Warmup: {WARMUP}")
print(f"{'E':>4} {'N':>5} | {'eager_prealloc(us)':>18} {'pypto(us)':>12} {'speedup':>8}")
print("-" * 60)

torch.manual_seed(42)
for E, N_total in configs:
    st, w13f, w2f, cs, res, tpe = make_data(E, N_total)
    res_eager = torch.zeros(N_total, H, dtype=torch.bfloat16, device=DEV)

    # Warmup
    for _ in range(WARMUP):
        eager_grouped_prealloc(st, w13f, w2f, tpe, E, res_eager)
    torch.npu.synchronize()
    for _ in range(WARMUP):
        llada2_moe_grouped_gemm(st, w13f, w2f, cs, res, E, H, I)
    torch.npu.synchronize()

    # Measure — 3 rounds, take min
    eager_times, pypto_times = [], []
    for _ in range(3):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            eager_grouped_prealloc(st, w13f, w2f, tpe, E, res_eager)
            torch.npu.synchronize()
        eager_times.append((time.perf_counter() - t0) / ITERS)

        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            llada2_moe_grouped_gemm(st, w13f, w2f, cs, res, E, H, I)
            torch.npu.synchronize()
        pypto_times.append((time.perf_counter() - t0) / ITERS)

    te = min(eager_times)
    tp = min(pypto_times)
    speedup = te / tp if tp > 0 else 0
    print(f"{E:>4} {N_total:>5} | {te * 1e6:>18.0f} {tp * 1e6:>12.0f} {speedup:>7.2f}x")

print("\nDone.")
