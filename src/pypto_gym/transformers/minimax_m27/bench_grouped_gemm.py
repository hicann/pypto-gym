# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Reproducible benchmark: MiniMax M2.7 MoE Grouped GEMM (eager vs PyPTO).

Measures per-expert Python-loop eager (matching HF MiniMaxM2Experts.forward)
versus single-kernel PyPTO grouped GEMM with weight conversion.

Usage:
    PYPTO_VEC_TILE=128 PYPTO_VEC_NBUFFER=1 TILE_FWK_DEVICE_ID=0 \
    python src/pypto_gym/transformers/minimax_m27/bench_grouped_gemm.py
"""

import logging
import os
import sys
import time
from typing import NamedTuple

import torch
import torch.nn.functional as functional

try:
    import torch_npu  # noqa: F401
except ImportError as exc:
    raise ImportError("torch_npu not available; this benchmark only runs on Ascend NPU.") from exc

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("minimax_m27.bench_grouped_gemm")

# Default tile config for MiniMax M2.7 (H=3072); vector tile must fit the 192 KB UB on 910B.
os.environ.setdefault("PYPTO_VEC_NBUFFER", "1")
os.environ.setdefault("PYPTO_VEC_TILE", "128")

_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, "src")):
    parent = os.path.dirname(_p)
    if parent == _p:
        raise RuntimeError("could not locate src/")
    _p = parent
sys.path.insert(0, os.path.join(_p, "src"))

# MiniMax M2.7 dimensions.
HIDDEN_SIZE = 3072
INTERMEDIATE_SIZE = 1536
ITERS = 200
WARMUP = 20
CONFIGS = [
    (4, 16),
    (8, 32),
    (8, 64),
    (8, 256),
    (16, 128),
    (16, 512),
    (32, 256),
    (32, 1024),
    (64, 512),
    (64, 2048),
]


class Batch(NamedTuple):
    sorted_tokens: "torch.Tensor"
    gate_up_proj: "torch.Tensor"
    down_proj: "torch.Tensor"
    w13_flat: "torch.Tensor"
    w2_flat: "torch.Tensor"
    cumsum: "torch.Tensor"
    result: "torch.Tensor"
    tokens_per_expert: list


def make_data(num_experts, num_tokens, device):
    from pypto_gym.ops.pypto_tile.minimax_m27.minimax_m27_grouped_gemm_impl import (
        convert_minimax_weights,
    )

    tokens_per_expert = [num_tokens // num_experts] * num_experts
    for i in range(num_tokens - sum(tokens_per_expert)):
        tokens_per_expert[i] += 1
    cumsum_list = [0]
    for count in tokens_per_expert:
        cumsum_list.append(cumsum_list[-1] + count)
    sorted_tokens = torch.randn(num_tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device=device) * 0.02

    # MiniMax-format weights.
    gate_up_proj = torch.randn(num_experts, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE,
                               dtype=torch.bfloat16, device=device) * 0.02
    down_proj = torch.randn(num_experts, HIDDEN_SIZE, INTERMEDIATE_SIZE,
                            dtype=torch.bfloat16, device=device) * 0.02

    w13_flat, w2_flat = convert_minimax_weights(gate_up_proj, down_proj)
    cumsum = torch.tensor(cumsum_list, dtype=torch.int32, device=device)
    result = torch.zeros(num_tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    return Batch(sorted_tokens, gate_up_proj, down_proj, w13_flat, w2_flat,
                 cumsum, result, tokens_per_expert)


def eager_minimax(batch, result_buf):
    """Eager per-expert loop matching HF MiniMaxM2Experts.forward."""
    tokens_per_expert = batch.tokens_per_expert
    offset = 0
    for expert, count in enumerate(tokens_per_expert):
        if count == 0:
            continue
        x_e = batch.sorted_tokens[offset:offset + count]
        gate_up = functional.linear(x_e, batch.gate_up_proj[expert])
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = functional.silu(gate) * up
        result_buf[offset:offset + count] = functional.linear(hidden, batch.down_proj[expert]).to(torch.bfloat16)
        offset += count


def _setup(device_id):
    """Configure the NPU device and pypto host options; return the device str."""
    import pypto

    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"
    torch_npu.npu.config.allow_internal_format = True
    pypto.set_host_options(
        compile_monitor_enable=True,
        compile_timeout=10,
        compile_timeout_stage=5,
        compile_monitor_print_interval=2,
    )
    return device


def _bench_one_config(num_experts, num_tokens, device):
    """Warm up and time eager vs kernel for one config; return (eager_best, kernel_best)."""
    from pypto_gym.ops.pypto_tile.minimax_m27.minimax_m27_grouped_gemm_impl import (
        minimax_m27_moe_grouped_gemm,
        MoeDims,
    )

    batch = make_data(num_experts, num_tokens, device)
    result_eager = torch.zeros(num_tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    weights = (batch.w13_flat, batch.w2_flat)
    dims = MoeDims(num_experts, HIDDEN_SIZE, INTERMEDIATE_SIZE)

    for _ in range(WARMUP):
        eager_minimax(batch, result_eager)
    torch.npu.synchronize()
    for _ in range(WARMUP):
        minimax_m27_moe_grouped_gemm(batch.sorted_tokens, weights, batch.cumsum, batch.result, dims)
    torch.npu.synchronize()

    eager_times, kernel_times = [], []
    for _ in range(3):
        torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(ITERS):
            eager_minimax(batch, result_eager)
            torch.npu.synchronize()
        eager_times.append((time.perf_counter() - start) / ITERS)

        torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(ITERS):
            minimax_m27_moe_grouped_gemm(batch.sorted_tokens, weights, batch.cumsum, batch.result, dims)
            torch.npu.synchronize()
        kernel_times.append((time.perf_counter() - start) / ITERS)

    return min(eager_times), min(kernel_times)


def _log_header(device_id):
    """Print the benchmark banner / table header."""
    logger.info("=== MiniMax M2.7 MoE Grouped GEMM Benchmark (H=%d, I=%d) ===", HIDDEN_SIZE, INTERMEDIATE_SIZE)
    logger.info("Device: NPU %d  |  Iters: %d  |  Warmup: %d", device_id, ITERS, WARMUP)
    logger.info("VEC_TILE=%s  VEC_NBUFFER=%s",
                os.environ.get("PYPTO_VEC_TILE"), os.environ.get("PYPTO_VEC_NBUFFER"))
    logger.info("%4s %5s | %12s %12s %8s", "E", "N", "Eager(us)", "Kernel(us)", "Speedup")
    logger.info("-" * 52)


def main():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = _setup(device_id)

    _log_header(device_id)

    torch.manual_seed(42)
    for num_experts, num_tokens in CONFIGS:
        eager_best, kernel_best = _bench_one_config(num_experts, num_tokens, device)
        speedup = eager_best / kernel_best if kernel_best > 0 else 0
        logger.info("%4d %5d | %12.0f %12.0f %7.2fx",
                    num_experts, num_tokens, eager_best * 1e6, kernel_best * 1e6, speedup)

    logger.info("Done.")


if __name__ == "__main__":
    main()
