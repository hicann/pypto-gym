#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
FlashAttentionScoreGrad 性能测试

测量 NPU 上的 kernel 执行时间，包含预热和多次迭代取平均。
"""

import argparse
import os
import sys
import time
import logging

import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flash_attention_score_grad_golden import (
    ForwardDataConfig, generate_forward_data,
)
from flash_attention_score_grad_impl import flash_attention_score_grad_wrapper


class BenchConfig:
    """Container for benchmark configuration."""

    def __init__(self, name, batch_size, num_heads, seq_len, head_dim):
        self.name = name
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim


def bench(cfg: BenchConfig, device_id, warmup=5, repeat=20):
    device = f"npu:{device_id}"
    fwd_cfg = ForwardDataConfig(
        cfg.batch_size, cfg.num_heads, cfg.seq_len, cfg.head_dim,
        device=device)
    fwd = generate_forward_data(fwd_cfg)
    q, k, v, dy = fwd.q, fwd.k, fwd.v, fwd.dy
    sm, ss, ao, scale = fwd.softmax_max, fwd.softmax_sum, fwd.attention_out, fwd.scale

    # 预热
    for _ in range(warmup):
        flash_attention_score_grad_wrapper(
            q, k, v, dy, sm, ss, ao, scale,
            cfg.num_heads, cfg.head_dim)
    torch.npu.synchronize()

    # 计时
    times = []
    for _ in range(repeat):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        flash_attention_score_grad_wrapper(
            q, k, v, dy, sm, ss, ao, scale,
            cfg.num_heads, cfg.head_dim)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    avg = sum(times) / len(times)
    mn = min(times)
    mx = max(times)

    total_matmul_flops = (
        cfg.batch_size * cfg.num_heads * 7 * 2
        * cfg.seq_len * cfg.seq_len * cfg.head_dim)
    tflops = total_matmul_flops / (mn / 1000) / 1e12

    logger.info(
        "  %-30s  B=%2d N=%2d S=%4d D=%3d  |  "
        "avg=%8.3fms  min=%8.3fms  max=%8.3fms  |  "
        "~%.2f TFLOPS",
        cfg.name, cfg.batch_size, cfg.num_heads,
        cfg.seq_len, cfg.head_dim, avg, mn, mx, tflops)
    return avg, mn


def main():
    parser = argparse.ArgumentParser(
        description="FlashAttentionScoreGrad Perf Bench")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()

    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    import torch_npu
    torch.npu.set_device(device_id)

    logger.info("=" * 100)
    logger.info("FlashAttentionScoreGrad Performance Benchmark")
    logger.info(
        "Device: NPU:%d  Warmup: %d  Repeat: %d",
        device_id, args.warmup, args.repeat)
    logger.info("=" * 100)

    configs = [
        BenchConfig("S=128", 2, 8, 128, 64),
        BenchConfig("S=256", 2, 8, 256, 64),
        BenchConfig("S=512", 2, 8, 512, 64),
        BenchConfig("S=1024", 2, 8, 1024, 64),
        BenchConfig("S=2048", 2, 8, 2048, 64),
        BenchConfig("S=4096", 1, 8, 4096, 64),
        BenchConfig("S=8192", 1, 8, 8192, 64),
    ]

    for cfg in configs:
        try:
            bench(cfg, device_id, args.warmup, args.repeat)
        except Exception as e:
            logger.info("  %-30s  FAILED: %s", cfg.name, e)

    logger.info("=" * 100)


if __name__ == "__main__":
    main()
