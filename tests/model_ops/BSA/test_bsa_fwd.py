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
BSA Forward Test Suite

Usage:
  # From BSA root (output/ goes to BSA/output/)
  cd models/experimental/attention/BSA
  python TEST/test_bsa_fwd.py

  # Or from anywhere — set BSA_ROOT to locate the operator code:
  BSA_ROOT=/path/to/BSA python /any/where/test_bsa_fwd.py
"""


import sys
import os
_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')):
    _p = os.path.dirname(_p)
sys.path.insert(0, os.path.join(_p, 'src'))
sys.path.insert(0, os.path.join(_p, 'src', 'pypto_gym', 'ops', 'pypto_tile'))


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
        "  Example: BSA_ROOT=/path/to/BSA python test_bsa_fwd.py"
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
    gen_inputs, _compare_tensors,
    _take_perf_timestamp, _collect_updated_dirs, _record_perf_from_dirs,
    _print_perf_summary, _perf_records,
)

# Run environment check immediately
_check_env()

from bsa_fwd_golden import bsa_forward_golden
from bsa_fwd_impl import block_sparse_attention_forward


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------
def do_forward_test(name, B, Hq, Hkv, Sq, Skv, sparsity):
    """Run forward precision test for one configuration."""
    device = get_device()
    perf_label = f"{name} [{B}x{Hq}x{Sq}x{Skv}]"
    logger.info(f"  [FWD] {name}:")
    logger.info(f"    Q: [{B}, {Hq}, {Sq}, {cfg.head_dim}] (BNSD, {cfg.torch_dtype})  "
                f"K/V: [{B}, {Hkv}, {Skv}, {cfg.head_dim}]  sparsity={sparsity}")
    Q, K, V, _, mask, asq, askv = gen_inputs(B, Hq, Hkv, Sq, Skv, sparsity, device)

    # Golden (CPU)
    O_g, lse_g = bsa_forward_golden(Q.cpu(), K.cpu(), V.cpu(), mask.cpu(),
                                     actual_seq_lengths=asq.cpu(),
                                     actual_seq_lengths_kv=askv.cpu())
    O_g, lse_g = O_g.to(device), lse_g.to(device)

    # PyPTO kernel
    _take_perf_timestamp()
    O_p, lse_p = block_sparse_attention_forward(Q, K, V, mask, asq, askv)
    updated_dirs = _collect_updated_dirs()

    # Compare O and LSE
    _compare_tensors("O", O_g, O_p, cfg.fwd_atol, cfg.fwd_rtol)
    _compare_tensors("LSE", lse_g, lse_p, cfg.fwd_atol, cfg.fwd_rtol)

    _record_perf_from_dirs(perf_label, updated_dirs, kernel_filter=("FWD",))
    return O_p, lse_p


# ===========================================================================
# Forward Test Cases
# ===========================================================================

def test_01_basic_sparse():
    do_forward_test("Basic Sparse50", B=1, Hq=4, Hkv=2, Sq=256, Skv=256, sparsity=0.5)


def test_02_gqa_group4():
    do_forward_test("GQA group4", B=1, Hq=8, Hkv=2, Sq=256, Skv=512, sparsity=0.4)


def test_03_gqa_large():
    do_forward_test("GQA Hq32_Hkv8", B=1, Hq=32, Hkv=8, Sq=256, Skv=256, sparsity=0.5)


def test_04_long_seq():
    do_forward_test("Long Seq S1024", B=1, Hq=8, Hkv=1, Sq=1024, Skv=1024, sparsity=0.3)


def test_05_sparse30():
    do_forward_test("Sparse30%", B=1, Hq=4, Hkv=4, Sq=512, Skv=512, sparsity=0.3)


def test_06_sparse70():
    do_forward_test("Sparse70%", B=1, Hq=4, Hkv=4, Sq=512, Skv=512, sparsity=0.7)


def test_07_dense():
    do_forward_test("Dense 100%", B=1, Hq=4, Hkv=4, Sq=256, Skv=256, sparsity=1.0)


def test_08_batch2():
    do_forward_test("Batch2", B=2, Hq=4, Hkv=2, Sq=256, Skv=512, sparsity=0.5)


def test_09_non_aligned():
    logger.info("  [SKIP] NonAligned: boundary handling not implemented for non-aligned sequences")


def test_10_long_seq_2048():
    do_forward_test("Long Seq S2048", B=1, Hq=4, Hkv=2, Sq=2048, Skv=2048, sparsity=0.3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger.info("\n" + "=" * 70)
    logger.info("BSA Forward Test Suite")
    logger.info(f"  dtype: {cfg.torch_dtype}")
    logger.info(f"  head_dim: {cfg.head_dim}")
    logger.info(f"  block_shape: ({cfg.block_shape_x}, {cfg.block_shape_y})")
    logger.info(f"  forward atol={cfg.fwd_atol}, rtol={cfg.fwd_rtol}")
    logger.info("=" * 70 + "\n")

    all_tests = [
        ("FWD: Basic Sparse50", test_01_basic_sparse),
        ("FWD: GQA group4", test_02_gqa_group4),
        ("FWD: GQA Hq32 Hkv8", test_03_gqa_large),
        ("FWD: Long Seq S1024", test_04_long_seq),
        ("FWD: Sparse30%", test_05_sparse30),
        ("FWD: Sparse70%", test_06_sparse70),
        ("FWD: Dense 100%", test_07_dense),
        ("FWD: Batch2", test_08_batch2),
        ("FWD: NonAligned", test_09_non_aligned),
        ("FWD: Long Seq S2048", test_10_long_seq_2048),
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
