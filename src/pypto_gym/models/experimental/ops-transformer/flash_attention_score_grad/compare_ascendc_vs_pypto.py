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
AscendC vs PyPTO FlashAttentionScoreGrad 性能对比

使用相同 shape 跑 torch_npu 内置算子 (AscendC) 和 PyPTO 实现，直接对比。
"""

from dataclasses import dataclass

import os
import sys
import time
import math
import logging
import torch
import torch_npu

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flash_attention_score_grad_golden import (
    generate_forward_data, ForwardDataConfig,
)
from flash_attention_score_grad_impl import (
    flash_attention_score_grad_wrapper, NUM_HEADS, HEAD_DIM,
)


@dataclass
class AscendcGradCallArgs:
    """Arguments for calling AscendC gradient operator."""

    q: object
    k: object
    v: object
    dy: object
    num_heads: int
    softmax_max: object
    softmax_sum: object
    attention_in: object
    scale: float
    seq_len: int
    batch_size: int


def _call_ascendc_grad(args: AscendcGradCallArgs):
    """Call AscendC attention gradient operator."""
    return torch_npu.npu_fusion_attention_grad_v2(
        args.q, args.k, args.v, args.dy, args.num_heads,
        pse=None,
        padding_mask=None,
        atten_mask=None,
        softmax_max=args.softmax_max,
        softmax_sum=args.softmax_sum,
        softmax_in=None,
        attention_in=args.attention_in,
        scale_value=args.scale,
        keep_prob=1.0,
        input_layout="BNSD",
        pre_tokens=args.seq_len,
        next_tokens=args.seq_len,
        seed=0,
        offset=0,
        numels=args.batch_size * args.num_heads * args.seq_len * args.seq_len,
        inner_precise=0,
        sparse_mode=0,
    )


def run_ascendc(cfg: AscendcConfig):
    """用 torch_npu.npu_fusion_attention_grad_v2 跑 AscendC 版本"""
    batch_size, _, seq_len, head_dim = cfg.q.shape
    fwd_result = torch_npu.npu_fusion_attention(
        cfg.q, cfg.k, cfg.v, cfg.num_heads,
        pse=None,
        padding_mask=None,
        atten_mask=None,
        scale=cfg.scale,
        keep_prob=1.0,
        input_layout="BNSD",
        pre_tockens=seq_len,
        next_tockens=seq_len,
        inner_precise=0,
        sparse_mode=0,
    )
    out_fwd = fwd_result[0]
    softmax_max_fwd = fwd_result[1]
    softmax_sum_fwd = fwd_result[2]

    for _ in range(cfg.warmup):
        ac_args = AscendcGradCallArgs(
            cfg.q, cfg.k, cfg.v, cfg.dy, cfg.num_heads,
            softmax_max_fwd, softmax_sum_fwd, out_fwd,
            cfg.scale, seq_len, batch_size)
        _call_ascendc_grad(ac_args)
    torch.npu.synchronize()

    times = []
    for _ in range(cfg.repeat):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        ac_args = AscendcGradCallArgs(
            cfg.q, cfg.k, cfg.v, cfg.dy, cfg.num_heads,
            softmax_max_fwd, softmax_sum_fwd, out_fwd,
            cfg.scale, seq_len, batch_size)
        _call_ascendc_grad(ac_args)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return times


class PyptoConfig:
    """Container for PyPTO run parameters."""

    def __init__(self, q, k, v, dy, softmax_max, softmax_sum,
                 attention_out, scale, num_heads, head_dim,
                 warmup=5, repeat=20):
        self.q = q
        self.k = k
        self.v = v
        self.dy = dy
        self.softmax_max = softmax_max
        self.softmax_sum = softmax_sum
        self.attention_out = attention_out
        self.scale = scale
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.warmup = warmup
        self.repeat = repeat


def run_pypto(cfg: PyptoConfig):
    """跑 PyPTO 版本"""
    for _ in range(cfg.warmup):
        flash_attention_score_grad_wrapper(
            cfg.q, cfg.k, cfg.v, cfg.dy,
            cfg.softmax_max, cfg.softmax_sum,
            cfg.attention_out, cfg.scale,
            cfg.num_heads, cfg.head_dim)
    torch.npu.synchronize()

    times = []
    for _ in range(cfg.repeat):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        flash_attention_score_grad_wrapper(
            cfg.q, cfg.k, cfg.v, cfg.dy,
            cfg.softmax_max, cfg.softmax_sum,
            cfg.attention_out, cfg.scale,
            cfg.num_heads, cfg.head_dim)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return times


def main():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"

    n_heads, head_dim = NUM_HEADS, HEAD_DIM

    configs = [
        ("B=2,S=128", 2, 128),
        ("B=2,S=256", 2, 256),
        ("B=2,S=512", 2, 512),
        ("B=2,S=1024", 2, 1024),
        ("B=2,S=2048", 2, 2048),
        ("B=1,S=4096", 1, 4096),
        ("B=1,S=8192", 1, 8192),
    ]

    logger.info("=" * 120)
    logger.info(
        "FlashAttentionScoreGrad: AscendC vs PyPTO  |  "
        "N=%d, D=%d, BF16, BNSD layout", n_heads, head_dim)
    logger.info("=" * 120)
    header = (
        f"{'Config':<16s} | {'AscendC min(ms)':>15s} {'avg(ms)':>10s} | "
        f"{'PyPTO min(ms)':>15s} {'avg(ms)':>10s} | "
        f"{'Ratio(AC/PT)':>12s} | "
        f"{'AscendC TFLOPS':>14s} {'PyPTO TFLOPS':>14s}")
    logger.info(header)
    logger.info("-" * 120)

    for name, batch_size, seq_len in configs:
        total_flops = (
            batch_size * n_heads * 7 * 2 * seq_len * seq_len * head_dim)

        try:
            fwd_cfg = ForwardDataConfig(
                batch_size, n_heads, seq_len, head_dim, device=device)
            fwd = generate_forward_data(fwd_cfg)
            q, k, v, dy = fwd.q, fwd.k, fwd.v, fwd.dy
            sm, ss, ao, scale = fwd.softmax_max, fwd.softmax_sum, fwd.attention_out, fwd.scale

            # AscendC
            try:
                ac_cfg = AscendcConfig(
                    q, k, v, dy, sm, ss, ao, scale, n_heads,
                    warmup=3, repeat=10)
                ac_times = run_ascendc(ac_cfg)
                ac_min = min(ac_times)
                ac_avg = sum(ac_times) / len(ac_times)
                ac_tflops = total_flops / (ac_min / 1000) / 1e12
            except Exception as e:
                ac_min = ac_avg = float('inf')
                ac_tflops = 0
                logger.info("  AscendC failed for %s: %s", name, e)

            # PyPTO
            try:
                pt_cfg = PyptoConfig(
                    q, k, v, dy, sm, ss, ao, scale, n_heads, head_dim,
                    warmup=3, repeat=10)
                pt_times = run_pypto(pt_cfg)
                pt_min = min(pt_times)
                pt_avg = sum(pt_times) / len(pt_times)
                pt_tflops = total_flops / (pt_min / 1000) / 1e12
            except Exception as e:
                pt_min = pt_avg = float('inf')
                pt_tflops = 0
                logger.info("  PyPTO failed for %s: %s", name, e)

            if ac_min > 0 and ac_min != float('inf'):
                ratio = pt_min / ac_min
            else:
                ratio = float('inf')
            row = (
                f"{name:<16s} | {ac_min:>14.3f}ms {ac_avg:>9.3f}ms | "
                f"{pt_min:>14.3f}ms {pt_avg:>9.3f}ms | "
                f"{ratio:>11.2f}x | "
                f"{ac_tflops:>13.2f}T {pt_tflops:>13.2f}T")
            logger.info(row)

        except Exception as e:
            logger.info("%-16s | FAILED: %s", name, e)

    logger.info("=" * 120)
    logger.info(
        "Ratio = PyPTO_time / AscendC_time "
        "(越小越好，<1 表示 PyPTO 更快)")


if __name__ == "__main__":
    main()
