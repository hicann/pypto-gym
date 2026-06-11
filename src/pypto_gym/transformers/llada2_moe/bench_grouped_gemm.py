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

import collections
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
    raise RuntimeError("torch_npu not available; this benchmark only runs on Ascend NPU.") from None

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


GroupedGemmData = collections.namedtuple(
    "GroupedGemmData",
    ["states_data", "w13_data", "w2_data", "cs_data", "res_data",
     "tokens_per_exp"],
)


def make_data(n_experts, n_tokens):
    tokens_per_exp = [n_tokens // n_experts] * n_experts
    for i in range(n_tokens - sum(tokens_per_exp)):
        tokens_per_exp[i] += 1
    cumsum = [0]
    for c in tokens_per_exp:
        cumsum.append(cumsum[-1] + c)
    states_data = torch.randn(n_tokens, H, dtype=torch.bfloat16, device=DEV) * 0.02
    w13_data = torch.randn(n_experts * H, 2 * I, dtype=torch.bfloat16, device=DEV) * 0.02
    w2_data = torch.randn(n_experts * I, H, dtype=torch.bfloat16, device=DEV) * 0.02
    cs_data = torch.tensor(cumsum, dtype=torch.int32, device=DEV)
    res_data = torch.zeros(n_tokens, H, dtype=torch.bfloat16, device=DEV)
    return GroupedGemmData(
        states_data, w13_data, w2_data, cs_data, res_data, tokens_per_exp
    )


def eager_grouped_prealloc(states_data, w13_data, w2_data, tokens_per_exp, n_experts, result_buf):
    """Eager with pre-allocated output buffer (fair comparison)."""
    offset = 0
    for e in range(n_experts):
        n_e = tokens_per_exp[e]
        if n_e == 0:
            continue
        x_e = states_data[offset : offset + n_e]
        w13_e = w13_data[e * H : (e + 1) * H, :]
        w2_e = w2_data[e * I : (e + 1) * I, :]
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
    data = make_data(E, N_total)
    st = data.states_data
    w13f = data.w13_data
    w2f = data.w2_data
    cs = data.cs_data
    res = data.res_data
    tpe = data.tokens_per_exp
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
