#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
BSA Forward Unified Test & Benchmark

Modes:
  --mode precision     Precision test (baseline only, default)
  --mode concurrent    Precision test (baseline + concurrent)
  --mode perf          Swimlane perf benchmark (baseline only)
  --mode perf-compare  Perf benchmark (baseline vs concurrent timing)

Usage:
  cd models/experimental/attention/BSA
  python TEST/test_bsa_fwd.py                     # precision (baseline)
  python TEST/test_bsa_fwd.py --mode concurrent   # precision (baseline+concurrent)
  python TEST/test_bsa_fwd.py --mode perf         # swimlane perf table
  python TEST/test_bsa_fwd.py --mode perf-compare # perf compare baseline vs concurrent
  python TEST/test_bsa_fwd.py --cases quick       # only S256+S512+B2S256 (fast)
"""

import sys
import os
import time

import torch


def _resolve_bsa_root():
    """Locate BSA root directory (containing common/, FWD/, BWD/)."""
    _BSA_MARKER = ("BSA_README.md", os.path.join("common", "bsa_common.py"))

    # 1. Environment variable
    env_root = os.environ.get("BSA_ROOT")
    if env_root and all(os.path.isfile(os.path.join(env_root, m)) for m in _BSA_MARKER):
        return os.path.abspath(env_root)

    # 2. Auto-detect: walk upward from this script
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(20):
        if all(os.path.isfile(os.path.join(d, m)) for m in _BSA_MARKER):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent

    raise RuntimeError(
        "Cannot locate BSA root directory.\n"
        "  Set BSA_ROOT to the path containing common/, FWD/, BWD/.")


_bsa_root = _resolve_bsa_root()
for _sub in ('common', 'FWD', 'BWD'):
    _p = os.path.join(_bsa_root, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bsa_test_utils import (
    _check_env, logger, cfg, get_device,
    gen_inputs, _compare_tensors,
    _take_perf_timestamp, _collect_updated_dirs, _record_perf_from_dirs,
    _print_perf_summary, _perf_records,
)
_check_env()

from bsa_fwd_golden import bsa_forward_golden
from bsa_fwd_impl import (
    block_sparse_attention_forward,
    block_sparse_attention_forward_concurrent,
)


# ═══════════════════════════════════════════════════════════════════════
# Test Cases
# ═══════════════════════════════════════════════════════════════════════

ALL_CASES = [
    ("S256 Sparse50", 1, 4, 2, 256, 256, 0.5),
    ("S512 Sparse70", 1, 4, 4, 512, 512, 0.7),
    ("S1024 Sparse30", 1, 8, 1, 1024, 1024, 0.3),
    ("S2048 Sparse30", 1, 4, 2, 2048, 2048, 0.3),
    ("GQA group4", 1, 8, 2, 256, 512, 0.4),
    ("GQA Hq32", 1, 32, 8, 256, 256, 0.5),
    ("Dense 100%", 1, 4, 4, 256, 256, 1.0),
    ("B2 MHA S256", 2, 4, 4, 256, 256, 0.5),
    ("B2 GQA S256", 2, 8, 2, 256, 512, 0.5),
    ("B2 Dense S256", 2, 4, 4, 256, 256, 1.0),
    ("B2 Sparse S512", 2, 4, 4, 512, 512, 0.7),
    ("B2 MHA S1024", 2, 4, 4, 1024, 1024, 0.7),
    ("B4 MHA S256", 4, 4, 4, 256, 256, 0.5),
]

QUICK_CASES = [
    ("S256 Sparse50", 1, 4, 2, 256, 256, 0.5),
    ("S512 Sparse70", 1, 4, 4, 512, 512, 0.7),
    ("B2 MHA S256", 2, 4, 4, 256, 256, 0.5),
]

# Cases for S1024/S2048 bottleneck analysis + 方案A (larger BH)
LONG_CASES = [
    ("S1024 Sparse30", 1, 8, 1, 1024, 1024, 0.3),
    ("S2048 Sparse30", 1, 4, 2, 2048, 2048, 0.3),
    ("B2 MHA S1024", 2, 4, 4, 1024, 1024, 0.7),
    # 方案A: larger BH to test AICore utilization improvement
    ("B4 MHA S1024", 4, 4, 4, 1024, 1024, 0.7),
    ("Hq16 S1024", 1, 16, 2, 1024, 1024, 0.3),
]


# ═══════════════════════════════════════════════════════════════════════
# Mode: precision (baseline only / baseline+concurrent)
# ═══════════════════════════════════════════════════════════════════════

def do_precision_test(name, B, Hq, Hkv, Sq, Skv, sparsity, test_concurrent=False):
    """Run forward precision test for one configuration."""
    device = get_device()
    logger.info(f"  [FWD] {name}: B={B} Hq={Hq} Hkv={Hkv} Sq={Sq} Skv={Skv} sp={sparsity}")
    inputs = gen_inputs(B, Hq, Hkv, Sq, Skv, sparsity, device)
    Q, K, V, mask, asq, askv = inputs.Q, inputs.K, inputs.V, inputs.mask, inputs.asq, inputs.askv

    # Baseline
    _take_perf_timestamp()
    O_p, lse_p = block_sparse_attention_forward(Q, K, V, mask, asq, askv)
    updated_dirs = _collect_updated_dirs()
    _compare_tensors("O", O_g, O_p, cfg.fwd_atol, cfg.fwd_rtol)
    _compare_tensors("LSE", lse_g, lse_p, cfg.fwd_atol, cfg.fwd_rtol)
    _record_perf_from_dirs(f"{name} [B={B},Hq={Hq}]", updated_dirs, kernel_filter=("FWD",))
    if test_concurrent:
        O_c, lse_c = block_sparse_attention_forward_concurrent(Q, K, V, mask, asq, askv)
        _compare_tensors("O_conc", O_g, O_c, cfg.fwd_atol, cfg.fwd_rtol)
        _compare_tensors("LSE_conc", lse_g, lse_c, cfg.fwd_atol, cfg.fwd_rtol)
        logger.info(f"  [FWD] {name} precision: PASS (baseline & concurrent)")
    else:
        logger.info(f"  [FWD] {name} precision: PASS (baseline)")


# ═══════════════════════════════════════════════════════════════════════
# Mode: perf (swimlane benchmark, baseline only)
# ═══════════════════════════════════════════════════════════════════════

def do_perf_bench(name, B, Hq, Hkv, Sq, Skv, sparsity):
    """Run one FWD kernel call and collect swimlane perf data."""
    device = get_device()
    logger.info(f"  [FWD-perf] {name}: B={B} Hq={Hq} Hkv={Hkv} Sq={Sq} Skv={Skv}")
    inputs = gen_inputs(B, Hq, Hkv, Sq, Skv, sparsity, device)
    Q, K, V, mask, asq, askv = inputs.Q, inputs.K, inputs.V, inputs.mask, inputs.asq, inputs.askv

    _take_perf_timestamp()
    O, lse = block_sparse_attention_forward(Q, K, V, mask, asq, askv)
    torch.npu.synchronize()

    updated_dirs = _collect_updated_dirs()
    _record_perf_from_dirs(f"{name} [B={B},Hq={Hq},Hkv={Hkv},Sq={Sq},Skv={Skv}]",
                           updated_dirs, kernel_filter=("FWD",))


# ═══════════════════════════════════════════════════════════════════════
# Mode: perf-compare (baseline vs concurrent wall-clock timing)
# ═══════════════════════════════════════════════════════════════════════

def bench_fn(fn, args, warmup=3, repeats=5):
    """Benchmark a function by repeated invocation with warmup."""
    for _ in range(warmup):
        fn(*args)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(*args)
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / repeats * 1e6


def do_perf_compare(name, B, Hq, Hkv, Sq, Skv, sparsity):
    """Compare baseline vs concurrent wall-clock timing."""
    device = get_device()
    inputs = gen_inputs(B, Hq, Hkv, Sq, Skv, sparsity, device)
    Q, K, V, mask, asq, askv = inputs.Q, inputs.K, inputs.V, inputs.mask, inputs.asq, inputs.askv

    # Warmup + precision check
    O_g, lse_g = bsa_forward_golden(Q.cpu(), K.cpu(), V.cpu(), mask.cpu(),
                                     actual_seq_lengths=asq.cpu(),
                                     actual_seq_lengths_kv=askv.cpu())
    O_g, lse_g = O_g.to(device), lse_g.to(device)

    O_b, lse_b = block_sparse_attention_forward(Q, K, V, mask, asq, askv)
    _compare_tensors("O_base", O_g, O_b, cfg.fwd_atol, cfg.fwd_rtol)
    torch.npu.synchronize()

    O_c, lse_c = block_sparse_attention_forward_concurrent(Q, K, V, mask, asq, askv)
    _compare_tensors("O_conc", O_g, O_c, cfg.fwd_atol, cfg.fwd_rtol)
    torch.npu.synchronize()

    base_us = bench_fn(block_sparse_attention_forward, (Q, K, V, mask, asq, askv))
    conc_us = bench_fn(block_sparse_attention_forward_concurrent, (Q, K, V, mask, asq, askv))
    delta_pct = (conc_us - base_us) / base_us * 100
    logger.info(f"  FWD {name}: base={base_us:.0f}us conc={conc_us:.0f}us delta={delta_pct:+.1f}%")
    return base_us, conc_us


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def _run_cases(args, cases):
    """Run all test cases and collect results."""
    passed = 0
    failed = 0
    perf_results = {}

    for name, B, Hq, Hkv, Sq, Skv, sp in cases:
        logger.info(f"--- {name} ---")
        try:
            if args.mode == "precision":
                do_precision_test(name, B, Hq, Hkv, Sq, Skv, sp, test_concurrent=False)
                passed += 1
            elif args.mode == "concurrent":
                do_precision_test(name, B, Hq, Hkv, Sq, Skv, sp, test_concurrent=True)
                passed += 1
            elif args.mode == "perf":
                do_perf_bench(name, B, Hq, Hkv, Sq, Skv, sp)
            elif args.mode == "perf-compare":
                b, c = do_perf_compare(name, B, Hq, Hkv, Sq, Skv, sp)
                perf_results[name] = (b, c)
                passed += 1
            logger.info(f"  >> PASSED")
        except Exception as e:
            logger.error(f"  >> FAILED: {e}")
            failed += 1
    return passed, failed, perf_results


def _print_summary(args, passed, failed, perf_results):
    """Print test result summary."""
    if args.mode in ("precision", "concurrent"):
        logger.info("=" * 70)
        logger.info(f"Results: {passed}/{passed + failed} PASSED")
        logger.info("=" * 70)

    if args.mode == "perf-compare" and perf_results:
        logger.info("\n" + "=" * 70)
        logger.info(f"{'Case':<20} {'Base(us)':>10} {'Conc(us)':>10} {'Delta':>8}")
        logger.info("-" * 50)
        for name, (b, c) in perf_results.items():
            logger.info(f"  {name:<18} {b:>10.0f} {c:>10.0f} {(c-b)/b*100:>7.1f}%")
        logger.info("=" * 70)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="BSA Forward Unified Test & Benchmark")
    parser.add_argument("--mode", choices=["precision", "concurrent", "perf", "perf-compare"],
                        default="precision",
                        help="precision=baseline only, concurrent=baseline+concurrent, "
                             "perf=swimlane benchmark, perf-compare=wall-clock timing")
    parser.add_argument("--cases", choices=["all", "quick", "long"], default="all",
                        help="all=full suite, quick=S256+S512+B2S256, long=S1024+S2048+large BH")
    args = parser.parse_args()

    cases_map = {"all": ALL_CASES, "quick": QUICK_CASES, "long": LONG_CASES}
    cases = cases_map[args.cases]

    # Clear perf records for clean output
    import bsa_test_utils
    bsa_test_utils._perf_records.clear()

    logger.info("\n" + "=" * 70)
    logger.info(f"BSA Forward — mode={args.mode}, cases={args.cases}")
    logger.info(f"  dtype: {cfg.torch_dtype}  head_dim: {cfg.head_dim}  "
                f"block_shape: ({cfg.block_shape_x}, {cfg.block_shape_y})")
    logger.info(f"  fwd atol={cfg.fwd_atol}, rtol={cfg.fwd_rtol}")
    logger.info("=" * 70 + "\n")

    passed, failed, perf_results = _run_cases(args, cases)
    _print_summary(args, passed, failed, perf_results)
    _print_perf_summary()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())