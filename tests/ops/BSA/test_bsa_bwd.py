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
BSA Backward Test Suite

Usage:
  # From BSA root (output/ goes to BSA/output/)
  cd models/experimental/attention/BSA
  python TEST/test_bsa_bwd.py

  # Or from anywhere — set BSA_ROOT to locate the operator code:
  BSA_ROOT=/path/to/BSA python /any/where/test_bsa_bwd.py
"""


import sys, os; _p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')): _p = os.path.dirname(_p)
sys.path.insert(0, os.path.join(_p, 'src')); sys.path.insert(0, os.path.join(_p, 'src', 'pypto_gym', 'ops', 'pypto_tile'))

import sys
import os


def _resolve_bsa_root():
    """Locate BSA root directory (containing common/, FWD/, BWD/) from any location.

    Priority:
      1. BSA_ROOT environment variable (explicit user override)
      2. Auto-detect: search upward from this script for BSA_README.md + common/bsa_common.py
      3. Hardcoded default path (fallback)

    Returns:
        Absolute path to BSA root directory.

    Raises:
        RuntimeError: If BSA root cannot be located by any method.
    """
    _BSA_MARKER = ("BSA_README.md", os.path.join("common", "bsa_common.py"))

    # 1. Environment variable
    env_root = os.environ.get("BSA_ROOT")
    if env_root and all(os.path.isfile(os.path.join(env_root, m)) for m in _BSA_MARKER):
        return os.path.abspath(env_root)

    # 2. Auto-detect: walk upward from this script's directory
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(20):
        if all(os.path.isfile(os.path.join(d, m)) for m in _BSA_MARKER):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent

    # 3. Repo-relative fallback: search upward from CWD for the stable relative path
    _REL_BSA = os.path.join(
        "pypto_6304", "models", "experimental", "attention", "BSA")
    d = os.getcwd()
    for _ in range(30):
        candidate = os.path.join(d, _REL_BSA)
        if all(os.path.isfile(os.path.join(candidate, m)) for m in _BSA_MARKER):
            return os.path.abspath(candidate)
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent

    raise RuntimeError(
        "Cannot locate BSA root directory.\n"
        "  Please set BSA_ROOT to the path containing common/, FWD/, BWD/.\n"
        "  Example: BSA_ROOT=/path/to/BSA python test_bsa_bwd.py"
    )


# Add BSA subdirectories (common/, FWD/, BWD/) to sys.path for direct module imports.
# Golden files (bsa_fwd_golden.py, bsa_bwd_golden.py) live in TEST/ alongside this
# script, so Python automatically finds them via sys.path[0].
_bsa_root = _resolve_bsa_root()
for _sub in ('common', 'FWD', 'BWD'):
    _p = os.path.join(_bsa_root, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bsa_test_utils import (
    _check_env, logger, cfg, get_device,
    gen_inputs, _compare_grads,
    _take_perf_timestamp, _record_perf_from_dirs,
    _print_perf_summary,
)

# Run environment check immediately
_check_env()

from bsa_fwd_golden import bsa_forward_golden
from bsa_fwd_impl import block_sparse_attention_forward
from bsa_bwd_golden import bsa_backward_golden
from bsa_bwd_impl import block_sparse_attention_backward


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------
def do_backward_test(name, B, Hq, Hkv, Sq, Skv, sparsity):
    """Run forward+backward precision test for one configuration."""
    device = get_device()
    perf_label = f"{name} [{B}x{Hq}x{Sq}x{Skv}]"
    logger.info(f"  [BWD] {name}:")
    logger.info(f"    Q: [{B}, {Hq}, {Sq}, {cfg.head_dim}] (BNSD, {cfg.torch_dtype})  "
                f"K/V: [{B}, {Hkv}, {Skv}, {cfg.head_dim}]  sparsity={sparsity}")
    Q, K, V, dO, mask, asq, askv = gen_inputs(B, Hq, Hkv, Sq, Skv, sparsity, device)

    # Forward pass (golden for backward reference, pypto for backward input)
    O_g, lse_g = bsa_forward_golden(Q.cpu(), K.cpu(), V.cpu(), mask.cpu(),
                                     actual_seq_lengths=asq.cpu(),
                                     actual_seq_lengths_kv=askv.cpu())
    O_p, lse_p = block_sparse_attention_forward(Q, K, V, mask, asq, askv)

    # Golden backward
    dQ_g, dK_g, dV_g = bsa_backward_golden(
        dO.cpu(), Q.cpu(), K.cpu(), V.cpu(),
        O_g.cpu(), lse_g.cpu(), mask.cpu(),
        actual_seq_lengths=asq.cpu(), actual_seq_lengths_kv=askv.cpu())

    # PyPTO backward
    _take_perf_timestamp()
    dQ_p, dK_p, dV_p = block_sparse_attention_backward(
        dO, Q, K, V, O_p, lse_p, mask, asq, askv)

    # Collect perf data: use tracked output dirs from the backward module
    # (handles both newly-compiled and cached kernels)
    from bsa_bwd_impl import last_backward_perf_dirs
    tracked_dirs = [d for d in last_backward_perf_dirs.values() if d is not None]
    _record_perf_from_dirs(perf_label, tracked_dirs, kernel_filter=("dQ", "dK/dV"))

    # Compare gradients
    _compare_grads(
        [("dQ", dQ_g, dQ_p), ("dK", dK_g, dK_p), ("dV", dV_g, dV_p)],
        cfg.bwd_atol, cfg.bwd_rtol)


# ===========================================================================
# Backward Test Cases
# ===========================================================================

def test_11_bwd_basic():
    do_backward_test("BWD Basic", B=1, Hq=4, Hkv=2, Sq=256, Skv=256, sparsity=0.5)

def test_12_bwd_mha_dense():
    do_backward_test("BWD MHA Dense", B=1, Hq=4, Hkv=4, Sq=256, Skv=256, sparsity=1.0)

def test_13_bwd_gqa():
    do_backward_test("BWD GQA", B=1, Hq=8, Hkv=2, Sq=256, Skv=512, sparsity=0.5)

def test_14_bwd_long_seq():
    do_backward_test("BWD Long Seq", B=1, Hq=8, Hkv=1, Sq=1024, Skv=1024, sparsity=0.3)

def test_15_bwd_non_aligned():
    logger.info("  [SKIP] BWD NonAligned: boundary handling not implemented for non-aligned sequences")

def test_16_bwd_mha_03():
    do_backward_test("BWD MHA sparse0.3", B=1, Hq=4, Hkv=4, Sq=256, Skv=256, sparsity=0.3)

def test_17_bwd_mha_07():
    do_backward_test("BWD MHA sparse0.7", B=1, Hq=4, Hkv=4, Sq=256, Skv=256, sparsity=0.7)

def test_18_bwd_mha_medium():
    do_backward_test("BWD MHA medium", B=1, Hq=4, Hkv=4, Sq=512, Skv=512, sparsity=0.7)

def test_19_bwd_mha_medium():
    do_backward_test("BWD MHA long", B=1, Hq=4, Hkv=4, Sq=1024, Skv=1024, sparsity=0.7)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger.info("\n" + "=" * 70)
    logger.info("BSA Backward Test Suite")
    logger.info(f"  dtype: {cfg.torch_dtype}")
    logger.info(f"  head_dim: {cfg.head_dim}")
    logger.info(f"  block_shape: ({cfg.block_shape_x}, {cfg.block_shape_y})")
    logger.info(f"  backward atol={cfg.bwd_atol}, rtol={cfg.bwd_rtol}")
    logger.info("=" * 70 + "\n")

    all_tests = [
        ("BWD: Basic",                 test_11_bwd_basic),
        ("BWD: MHA Dense",             test_12_bwd_mha_dense),
        ("BWD: GQA",                   test_13_bwd_gqa),
        ("BWD: Long Seq",              test_14_bwd_long_seq),
        ("BWD: NonAligned",            test_15_bwd_non_aligned),
        ("BWD: MHA sparse0.3",         test_16_bwd_mha_03),
        ("BWD: MHA sparse0.7",         test_17_bwd_mha_07),
        ("BWD: MHA medium",            test_18_bwd_mha_medium),
        ("BWD: MHA long",              test_19_bwd_mha_medium),
    ]

    passed = 0
    failed = 0
    for name, test_fn in all_tests:
        logger.info("=" * 70)
        logger.info(f"Test: {name}")
        logger.info("=" * 70)
        try:
            test_fn()
            passed += 1
            logger.info(f"  >> PASSED")
        except Exception as e:
            logger.error(f"  >> FAILED: {e}")
            failed += 1
        logger.info("")

    logger.info("=" * 70)
    logger.info(f"Results: {passed}/{passed + failed} tests PASSED")
    if failed == 0:
        logger.info("All tests PASSED!")
    else:
        logger.info(f"{failed} tests FAILED")
    logger.info("=" * 70)

    _print_perf_summary()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
