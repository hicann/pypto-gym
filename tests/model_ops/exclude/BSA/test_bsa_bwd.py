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
BSA Backward Unified Test & Benchmark

Modes:
  --mode precision     Precision test (baseline only, default)
  --mode concurrent    Precision test (baseline + concurrent)
  --mode perf          Swimlane perf benchmark (baseline only)
  --mode perf-compare  Perf benchmark (baseline vs concurrent timing)

Usage:
  cd models/experimental/attention/BSA
  python TEST/test_bsa_bwd.py                     # precision (baseline)
  python TEST/test_bsa_bwd.py --mode concurrent   # precision (baseline+concurrent)
  python TEST/test_bsa_bwd.py --mode perf         # swimlane perf table
  python TEST/test_bsa_bwd.py --mode perf-compare # perf compare baseline vs concurrent
  python TEST/test_bsa_bwd.py --cases quick       # only S256+S512+B2S256 (fast)
"""

import sys
import os
import time


def _resolve_bsa_root():
    """Locate BSA root directory (containing common/, FWD/, BWD/)."""
    _BSA_MARKER = ("BSA_README.md", os.path.join("common", "bsa_common.py"))

    env_root = os.environ.get("BSA_ROOT")
    if env_root and all(os.path.isfile(os.path.join(env_root, m)) for m in _BSA_MARKER):
        return os.path.abspath(env_root)

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
    gen_inputs, _compare_grads,
    _take_perf_timestamp, _collect_updated_dirs, _record_perf_from_dirs,
    _print_perf_summary,
)
_check_env()

from bsa_fwd_golden import bsa_forward_golden
from bsa_fwd_impl import block_sparse_attention_forward
from bsa_bwd_golden import bsa_backward_golden
from bsa_bwd_impl import (
    block_sparse_attention_backward,
    block_sparse_attention_backward_concurrent,
)


# ═══════════════════════════════════════════════════════════════════════
# Test Cases
# ═══════════════════════════════════════════════════════════════════════

ALL_CASES = [
    ("S256 Sparse50", 1, 4, 2, 256, 256, 0.5),
    ("S512 Sparse70", 1, 4, 4, 512, 512, 0.7),
    ("S1024 Sparse30", 1, 8, 1, 1024, 1024, 0.3),
    ("MHA Dense", 1, 4, 4, 256, 256, 1.0),
    ("GQA", 1, 8, 2, 256, 512, 0.5),
    ("MHA sparse0.3", 1, 4, 4, 256, 256, 0.3),
    ("MHA sparse0.7", 1, 4, 4, 256, 256, 0.7),
    ("MHA S512", 1, 4, 4, 512, 512, 0.7),
    ("MHA S1024", 1, 4, 4, 1024, 1024, 0.7),
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


# ═══════════════════════════════════════════════════════════════════════
def do_precision_test(name, B, Hq, Hkv, Sq, Skv, sparsity, test_concurrent=False):
    """Run forward+backward precision test for one configuration."""
    device = get_device()
    logger.info(f"  [BWD] {name}: B={B} Hq={Hq} Hkv={Hkv} Sq={Sq} Skv={Skv} sp={sparsity}")
    inputs = gen_inputs(B, Hq, Hkv, Sq, Skv, sparsity, device)
    Q, K, V, dO, mask, asq, askv = inputs.Q, inputs.K, inputs.V, inputs.dO, inputs.mask, inputs.asq, inputs.askv

    # Forward pass (golden for backward reference, pypto for backward input)
    O_p, lse_p = block_sparse_attention_forward(Q, K, V, mask, asq, askv)
    O_g, lse_g = bsa_forward_golden(Q.cpu(), K.cpu(), V.cpu(), mask.cpu(),
                                     actual_seq_lengths=asq.cpu(),
                                     actual_seq_lengths_kv=askv.cpu())
    O_g, lse_g = O_g.to(device), lse_g.to(device)

    # Golden backward
    dQ_g, dK_g, dV_g = bsa_backward_golden(
        dO.cpu(), Q.cpu(), K.cpu(), V.cpu(),
        O_g.cpu(), lse_g.cpu(), mask.cpu(),
        actual_seq_lengths=asq.cpu(), actual_seq_lengths_kv=askv.cpu())

    # Baseline backward
    _take_perf_timestamp()
    dQ_p, dK_p, dV_p = block_sparse_attention_backward(
        dO, Q, K, V, O_p, lse_p, mask, asq, askv)

    from bsa_bwd_impl import last_backward_perf_dirs
    tracked_dirs = [d for d in last_backward_perf_dirs.values() if d is not None]
    _record_perf_from_dirs(f"{name} [B={B},Hq={Hq}]", tracked_dirs, kernel_filter=("dQ", "dK/dV"))

    _compare_grads(
        [("dQ", dQ_g, dQ_p), ("dK", dK_g, dK_p), ("dV", dV_g, dV_p)],
        cfg.bwd_atol, cfg.bwd_rtol)

    # Concurrent backward (optional)
    if test_concurrent:
        dQ_c, dK_c, dV_c = block_sparse_attention_backward_concurrent(
            dO, Q, K, V, O_p, lse_p, mask, asq, askv)
        _compare_grads(
            [("dQ_conc", dQ_g, dQ_c), ("dK_conc", dK_g, dK_c), ("dV_conc", dV_g, dV_c)],
            cfg.bwd_atol, cfg.bwd_rtol)
        logger.info(f"  [BWD] {name} precision: PASS (baseline & concurrent)")
    else:
        logger.info(f"  [BWD] {name} precision: PASS (baseline)")


# ═══════════════════════════════════════════════════════════════════════
# Mode: perf (swimlane benchmark)
# ═══════════════════════════════════════════════════════════════════════

def do_perf_bench(name, B, Hq, Hkv, Sq, Skv, sparsity):
    """Run one BWD kernel call and collect swimlane perf data."""
    device = get_device()
    logger.info(f"  [BWD-perf] {name}: B={B} Hq={Hq} Hkv={Hkv} Sq={Sq} Skv={Skv}")
    inputs = gen_inputs(B, Hq, Hkv, Sq, Skv, sparsity, device)
    Q, K, V, dO, mask, asq, askv = inputs.Q, inputs.K, inputs.V, inputs.dO, inputs.mask, inputs.asq, inputs.askv

    # FWD first (needed for BWD input)
    O, lse = block_sparse_attention_forward(Q, K, V, mask, asq, askv)
    torch.npu.synchronize()

    _take_perf_timestamp()
    dQ, dK, dV = block_sparse_attention_backward(dO, Q, K, V, O, lse, mask, asq, askv)
    torch.npu.synchronize()

    updated_dirs = _collect_updated_dirs()
    from bsa_bwd_impl import last_backward_perf_dirs
    tracked_dirs = [d for d in last_backward_perf_dirs.values() if d is not None]
    all_dirs = updated_dirs + tracked_dirs
    _record_perf_from_dirs(f"{name} [B={B},Hq={Hq},Hkv={Hkv},Sq={Sq},Skv={Skv}]",
                           all_dirs, kernel_filter=("dQ", "dK/dV"))


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
    """Compare baseline vs concurrent wall-clock timing for backward."""
    device = get_device()
    inputs = gen_inputs(B, Hq, Hkv, Sq, Skv, sparsity, device)
    Q, K, V, dO, mask, asq, askv = inputs.Q, inputs.K, inputs.V, inputs.dO, inputs.mask, inputs.asq, inputs.askv

    # FWD
    O_p, lse_p = block_sparse_attention_forward(Q, K, V, mask, asq, askv)
    O_g, lse_g = bsa_forward_golden(Q.cpu(), K.cpu(), V.cpu(), mask.cpu(),
                                     actual_seq_lengths=asq.cpu(),
                                     actual_seq_lengths_kv=askv.cpu())
    O_g, lse_g = O_g.to(device), lse_g.to(device)

    dQ_g, dK_g, dV_g = bsa_backward_golden(
        dO.cpu(), Q.cpu(), K.cpu(), V.cpu(), O_g.cpu(), lse_g.cpu(), mask.cpu(),
        actual_seq_lengths=asq.cpu(), actual_seq_lengths_kv=askv.cpu())

    # Baseline backward
    dQ_b, dK_b, dV_b = block_sparse_attention_backward(dO, Q, K, V, O_p, lse_p, mask, asq, askv)
    _compare_grads([("dQ_base", dQ_g, dQ_b), ("dK_base", dK_g, dK_b), ("dV_base", dV_g, dV_b)],
                   cfg.bwd_atol, cfg.bwd_rtol)
    torch.npu.synchronize()

    # Concurrent backward
    dQ_c, dK_c, dV_c = block_sparse_attention_backward_concurrent(dO, Q, K, V, O_p, lse_p, mask, asq, askv)
    _compare_grads([("dQ_conc", dQ_g, dQ_c), ("dK_conc", dK_g, dK_c), ("dV_conc", dV_g, dV_c)],
                   cfg.bwd_atol, cfg.bwd_rtol)
    torch.npu.synchronize()

    base_us = bench_fn(block_sparse_attention_backward, (dO, Q, K, V, O_p, lse_p, mask, asq, askv))
    conc_us = bench_fn(block_sparse_attention_backward_concurrent, (dO, Q, K, V, O_p, lse_p, mask, asq, askv))
    delta_pct = (conc_us - base_us) / base_us * 100
    logger.info(f"  BWD {name}: base={base_us:.0f}us conc={conc_us:.0f}us delta={delta_pct:+.1f}%")
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
    parser = argparse.ArgumentParser(description="BSA Backward Unified Test & Benchmark")
    parser.add_argument("--mode", choices=["precision", "concurrent", "perf", "perf-compare"],
                        default="precision",
                        help="precision=baseline only, concurrent=baseline+concurrent, "
                             "perf=swimlane benchmark, perf-compare=wall-clock timing")
    parser.add_argument("--cases", choices=["all", "quick"], default="all",
                        help="all=full suite, quick=S256+S512+B2S256 only")
    args = parser.parse_args()

    cases = ALL_CASES if args.cases == "all" else QUICK_CASES

    # Clear perf records for clean output
    import bsa_test_utils
    bsa_test_utils._perf_records.clear()

    logger.info("\n" + "=" * 70)
    logger.info(f"BSA Backward — mode={args.mode}, cases={args.cases}")
    logger.info(f"  dtype: {cfg.torch_dtype}  head_dim: {cfg.head_dim}  "
                f"block_shape: ({cfg.block_shape_x}, {cfg.block_shape_y})")
    logger.info(f"  bwd atol={cfg.bwd_atol}, rtol={cfg.bwd_rtol}")
    logger.info("=" * 70 + "\n")

    passed, failed, perf_results = _run_cases(args, cases)
    _print_summary(args, passed, failed, perf_results)
    _print_perf_summary()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())