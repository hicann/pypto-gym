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
BSA Shared Test Utilities

Shared test infrastructure used by both test_bsa_fwd.py and test_bsa_bwd.py:
  - Environment check
  - Performance tracking
  - Input generation
  - Tensor comparison helpers
"""

import collections
import os
import re
import sys
import math
import json
import logging
import subprocess
from collections import defaultdict

import torch
import numpy as np
from numpy.testing import assert_allclose

from bsa_common import DEFAULT_CONFIG, generate_block_sparse_mask


BsaInputs = collections.namedtuple("BsaInputs", [
    "Q", "K", "V", "dO", "mask", "asq", "askv",
])

# shorthand
cfg = DEFAULT_CONFIG


# ===========================================================================
# Auto environment configuration
# ===========================================================================
_DEFAULT_ASCEND_HOME = "/home/developer/Ascend/cann-9.0.0"
_DEFAULT_PYPTO_PATH = "/mnt/workspace/gitCode/cann/pypto/python"
_FORK_PYPTO_PATH = "/mnt/workspace/gitCode/cann/mce/pypto_fork/pypto_6304/python"
_DEFAULT_PTO_ISA_PATH = "/mnt/workspace/gitCode/cann/mce/pto-isa"


def _check_ascend_env(errors, warnings):
    ascend = os.environ.get("ASCEND_HOME_PATH", _DEFAULT_ASCEND_HOME)
    if not os.path.isdir(ascend):
        errors.append(f"ASCEND_HOME_PATH not found: {ascend}")
    else:
        os.environ["ASCEND_HOME_PATH"] = ascend
        print(f"[ENV] ASCEND_HOME_PATH = {ascend}")
    devid = os.environ.get("TILE_FWK_DEVICE_ID", "0")
    os.environ["TILE_FWK_DEVICE_ID"] = devid
    print(f"[ENV] TILE_FWK_DEVICE_ID = {devid}")
    pto_isa_path = os.environ.get("PTO_TILE_LIB_CODE_PATH", _DEFAULT_PTO_ISA_PATH)
    if not os.path.isdir(pto_isa_path):
        warnings.append(f"PTO_TILE_LIB_CODE_PATH not found: {pto_isa_path}")
    else:
        os.environ["PTO_TILE_LIB_CODE_PATH"] = pto_isa_path
        print(f"[ENV] PTO_TILE_LIB_CODE_PATH = {pto_isa_path}")


def _check_pypto_path(errors):
    pypto_path = os.environ.get("PYPTO_PATH", _DEFAULT_PYPTO_PATH)
    if not os.path.isdir(pypto_path):
        errors.append(f"PyPTO path not found: {pypto_path}")
    else:
        existing = os.environ.get("PYTHONPATH", "")
        if pypto_path not in existing:
            os.environ["PYTHONPATH"] = pypto_path + ((":" + existing) if existing else "")
        if pypto_path not in sys.path:
            sys.path.insert(0, pypto_path)
        print(f"[ENV] PyPTO path = {pypto_path}")


def _check_npu_tools(errors, warnings):
    try:
        result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            warnings.append("npu-smi info returned non-zero")
        else:
            for line in result.stdout.strip().split("\n")[:4]:
                print(f"[NPU] {line}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        warnings.append("npu-smi not available")
    try:
        import torch_npu  # noqa: F401
        print(f"[ENV] torch_npu OK")
    except ImportError:
        errors.append("torch_npu import failed")
    try:
        import pypto  # noqa: F401
        print(f"[ENV] pypto OK  (path: {pypto.__file__})")
        has_dynamic = hasattr(pypto, 'DYNAMIC')
        print(f"[ENV] pypto DYNAMIC support: {has_dynamic}")
    except ImportError:
        errors.append("pypto import failed")


def _check_env():
    """Auto-configure environment variables and verify the runtime is ready."""
    errors = []
    warnings = []
    _check_ascend_env(errors, warnings)
    _check_pypto_path(errors)
    _check_npu_tools(errors, warnings)
    print()
    if warnings:
        for w in warnings:
            print(f"[WARN] {w}")
        print()
    if errors:
        print("=" * 60)
        print("Environment check FAILED:")
        for i, e in enumerate(errors, 1):
            print(f"\n  [{i}] {e}")
        print("\n" + "=" * 60)
        raise RuntimeError("Environment check FAILED") from None
    print("[ENV] All checks passed. Starting tests...\n")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
_fmt = logging.Formatter(fmt='%(asctime)s [%(levelname)s] %(message)s', datefmt='[%H:%M:%S]')
_handler = logging.StreamHandler()
_handler.setFormatter(_fmt)
logger.handlers.clear()
logger.addHandler(_handler)


def get_device():
    """Get NPU device string."""
    did = os.environ.get("TILE_FWK_DEVICE_ID", "0")
    return f"npu:{did}"


# ---------------------------------------------------------------------------
# Performance tracking
# ---------------------------------------------------------------------------
import time

_PERF_OUTPUT_BASE = os.path.abspath(os.path.join(os.getcwd(), "output"))
_perf_records = []
_perf_timestamp = None


def _take_perf_timestamp():
    global _perf_timestamp
    _perf_timestamp = time.time()


def _collect_updated_dirs():
    """Collect output dirs whose ``merged_swimlane.json`` was modified after
    :pyfunc:`_take_perf_timestamp` (minus a 0.5 s buffer).

    Callers that also pass *extra_dirs* to :pyfunc:`_record_perf_from_dirs`
    can supplement this list with cached-kernel output dirs tracked by the
    implementation module.
    """
    if _perf_timestamp is None or not os.path.isdir(_PERF_OUTPUT_BASE):
        return []
    updated = []
    for name in os.listdir(_PERF_OUTPUT_BASE):
        d = os.path.join(_PERF_OUTPUT_BASE, name)
        if not os.path.isdir(d):
            continue
        swim = os.path.join(d, "merged_swimlane.json")
        if os.path.isfile(swim):
            mtime = os.path.getmtime(swim)
            if mtime >= _perf_timestamp - 0.5:
                updated.append(d)
    updated.sort()
    return updated


def _identify_kernel(output_dir):
    """Identify kernel types from kernel_aicore directory.

    Returns a list of identified kernel type labels. For merged kernels that
    contain both dQ and dK/dV phases, the list will contain both labels.
    """
    kernel_aicore = os.path.join(output_dir, "kernel_aicore")
    if not os.path.isdir(kernel_aicore):
        return ["Unknown"]
    loop_prefixes = set()
    for f in os.listdir(kernel_aicore):
        m = re.match(r"TENSOR_LOOP_([a-z_]*?)(?:qblk|kblk|sub)", f)
        if m:
            prefix = m.group(1).rstrip("_")
            if prefix:
                loop_prefixes.add(prefix)
    prefix_map = {
        "fwd_s": "FWD", "fwd_d": "FWD", "fwd": "FWD",
        "dq": "dQ", "dqd": "dQ",
        "dkdv": "dK/dV", "dkdvd": "dK/dV",
    }
    identified = set()
    for prefix in loop_prefixes:
        for key, label in prefix_map.items():
            if prefix.startswith(key):
                identified.add(label)
    if not identified:
        return ["Unknown"]
    return sorted(identified)


def _parse_swimlane(output_dir):
    """Parse merged_swimlane.json, computing per-execution metrics.

    When the swimlane file accumulates events from multiple kernel
    executions (cached kernel reused across test cases), we isolate
    each execution by finding contiguous clusters of real-task events
    separated by large gaps (idle periods between test cases).

    Each cluster is treated as a single execution, and we return the
    metrics for the LAST cluster (the one most recently executed).
    """
    swim_path = os.path.join(output_dir, "merged_swimlane.json")
    if not os.path.exists(swim_path):
        return None
    with open(swim_path) as f:
        data = json.load(f)
    evts = data.get("traceEvents", [])
    if not evts:
        return None
    tid2core = {}
    for ev in evts:
        if ev.get("ph") == "M" and ev.get("name") == "thread_name":
            tid2core[ev["tid"]] = ev["args"]["name"]
    all_tasks = [ev for ev in evts if ev.get("cat") == "event" and ev.get("ph") == "X"]
    real_tasks = [t for t in all_tasks if "(fake)" not in t.get("name", "")]
    if not real_tasks:
        return None

    # Sort real tasks by start time
    real_tasks.sort(key=lambda t: t["ts"])

    # Split into clusters: a gap > 10ms between consecutive real tasks
    # indicates a new execution (idle period between test cases)
    CLUSTER_GAP_US = 10000  # 10ms in microseconds
    clusters = []
    current_cluster = [real_tasks[0]]
    for i in range(1, len(real_tasks)):
        prev_end = current_cluster[-1]["ts"] + current_cluster[-1]["dur"]
        curr_start = real_tasks[i]["ts"]
        if curr_start - prev_end > CLUSTER_GAP_US:
            clusters.append(current_cluster)
            current_cluster = [real_tasks[i]]
        else:
            current_cluster.append(real_tasks[i])
    clusters.append(current_cluster)

    # Use the LAST cluster (most recent execution)
    latest_cluster = clusters[-1] if clusters else real_tasks

    global_first = min(t["ts"] for t in latest_cluster)
    global_last = max(t["ts"] + t["dur"] for t in latest_cluster)
    task_time = global_last - global_first
    aicore_time = sum(t["dur"] for t in latest_cluster)
    active_core_ids = set()
    for t in latest_cluster:
        core_name = tid2core.get(t["tid"], "")
        m = re.search(r"(\d+)$", core_name)
        if m:
            active_core_ids.add(int(m.group(1)))
    num_active_cores = len(active_core_ids) if active_core_ids else 1
    util = (aicore_time / (task_time * num_active_cores) * 100) if task_time > 0 else 0
    return task_time, aicore_time, util


def _record_perf_from_dirs(test_name, updated_dirs, kernel_filter=None):
    """Record perf data from output dirs, supporting merged kernels that
    produce multiple kernel types (e.g. both dQ and dK/dV) from a single dir.

    When _identify_kernel returns a list with multiple types, a separate perf
    record is created for each type, all using the same combined swimlane data.
    Dirs are deduplicated to avoid processing the same merged output twice.
    """
    seen_dirs = set()
    best_per_type = {}
    for d in updated_dirs:
        # Deduplicate dirs (important for merged kernels where both dQ and
        # dK/dV perf entries point to the same output directory)
        if d in seen_dirs:
            continue
        seen_dirs.add(d)

        ktypes = _identify_kernel(d)
        swim = os.path.join(d, "merged_swimlane.json")
        if not os.path.isfile(swim):
            continue
        mtime = os.path.getmtime(swim)

        for ktype in ktypes:
            if kernel_filter and ktype not in kernel_filter:
                continue
            if ktype == "Unknown":
                continue
            if ktype not in best_per_type or mtime > best_per_type[ktype][1]:
                best_per_type[ktype] = (d, mtime)

    for ktype, (d, _) in best_per_type.items():
        result = _parse_swimlane(d)
        if result is not None:
            task_time, aicore_time, util = result
            _perf_records.append({
                "test_name": test_name, "kernel": ktype,
                "task_time_us": task_time, "aicore_time_us": aicore_time,
                "utilization": util,
            })
        else:
            _perf_records.append({
                "test_name": test_name, "kernel": ktype,
                "task_time_us": None, "aicore_time_us": None, "utilization": None,
            })


def _print_perf_summary():
    if not _perf_records:
        logger.info("\n[PERF] No performance data collected.")
        return
    max_test_len = max(len(r["test_name"]) for r in _perf_records)
    W_TEST = max(max_test_len + 2, 20)
    W_KERNEL = 8
    W_TASK = 14
    W_AICORE = 14
    W_UTIL = 10

    def _fmt_us(val):
        if val is None:
            return "N/A".rjust(W_TASK)
        if val >= 1000:
            return f"{val / 1000:.2f} ms".rjust(W_TASK)
        return f"{val:.1f} us".rjust(W_TASK)

    def _fmt_pct(val):
        if val is None:
            return "N/A".rjust(W_UTIL)
        return f"{val:.1f}%".rjust(W_UTIL)

    sep = "+" + "-" * (W_TEST + 2) + "+" + "-" * (W_KERNEL + 2) + \
          "+" + "-" * (W_TASK + 2) + "+" + "-" * (W_AICORE + 2) + \
          "+" + "-" * (W_UTIL + 2) + "+"
    header = "|" + " Test Case".ljust(W_TEST + 1) + \
             "|" + " Kernel".ljust(W_KERNEL + 1) + \
             "|" + " Task Time".ljust(W_TASK + 1) + \
             "|" + " AICore Time".ljust(W_AICORE + 1) + \
             "|" + " AICore Util".ljust(W_UTIL + 1) + "|"

    logger.info("")
    logger.info("=" * len(sep))
    logger.info("Performance Summary (from merged_swimlane.json)")
    logger.info("=" * len(sep))
    logger.info(sep)
    logger.info(header)
    logger.info(sep)

    current_test = None
    for rec in _perf_records:
        if rec["test_name"] != current_test:
            test_str = " " + rec["test_name"]
            current_test = rec["test_name"]
        else:
            test_str = ""
        row = "|" + test_str.ljust(W_TEST + 1) + \
              "|" + (" " + rec["kernel"]).ljust(W_KERNEL + 1) + \
              "|" + _fmt_us(rec["task_time_us"]) + " " + \
              "|" + _fmt_us(rec["aicore_time_us"]) + " " + \
              "|" + _fmt_pct(rec["utilization"]) + " |"
        logger.info(row)

    logger.info(sep)
    logger.info("")
    logger.info("指标说明 (与 PyPTO toolkit 定义一致):")
    logger.info("  Task Time   = 首 task 到末 task 的 wall-clock 耗时")
    logger.info("  AICore Time = 所有 task 的 dur 之和（总计算量）")
    logger.info("  AICore Util = AICoreTime / (TaskTime × 活跃物理核数) × 100%")
    logger.info("=" * len(sep))


# ---------------------------------------------------------------------------
# Input generation
# ---------------------------------------------------------------------------
def gen_inputs(B, Hq, Hkv, Sq, Skv, sparsity, device, seed=42):
    """Generate random BSA inputs in FP16, BNSD layout."""
    torch.manual_seed(seed)
    numQB = math.ceil(Sq / cfg.block_shape_x)
    numKB = math.ceil(Skv / cfg.block_shape_y)

    Q = torch.empty(B, Hq, Sq, cfg.head_dim, dtype=cfg.torch_dtype, device=device).uniform_(-1, 1)
    K = torch.empty(B, Hkv, Skv, cfg.head_dim, dtype=cfg.torch_dtype, device=device).uniform_(-1, 1)
    V = torch.empty(B, Hkv, Skv, cfg.head_dim, dtype=cfg.torch_dtype, device=device).uniform_(-1, 1)
    dO = torch.empty(B, Hq, Sq, cfg.head_dim, dtype=cfg.torch_dtype, device=device).uniform_(-1, 1)

    mask = generate_block_sparse_mask(
        B, Hq, numQB, numKB, sparsity=sparsity, device=device, seed=seed)

    asq = torch.full([B], Sq, dtype=torch.int64, device=device)
    askv = torch.full([B], Skv, dtype=torch.int64, device=device)

    return BsaInputs(Q=Q, K=K, V=V, dO=dO, mask=mask, asq=asq, askv=askv)


# ---------------------------------------------------------------------------
# Tensor comparison helpers
# ---------------------------------------------------------------------------
def _compare_tensors(label, golden, actual, atol, rtol):
    """Compare two tensors, log result, and assert closeness."""
    g_np = golden.cpu().float().numpy().flatten()
    p_np = actual.cpu().float().numpy().flatten()
    diff = np.max(np.abs(p_np - g_np))
    assert_allclose(p_np, g_np, atol=atol, rtol=rtol)
    logger.info(f"    {label}: {list(actual.shape)} PASS (max_diff={diff:.6f})")
    return diff


def _compare_grads(grad_pairs, atol, rtol):
    """Compare a list of (name, golden, actual) gradient pairs."""
    for gname, gg, gp in grad_pairs:
        g_np = gg.cpu().float().numpy().flatten()
        p_np = gp.cpu().float().numpy().flatten()
        diff = np.max(np.abs(p_np - g_np))
        assert_allclose(p_np, g_np, atol=atol, rtol=rtol)
        logger.info(f"    {gname}: PASS (max_diff={diff:.6f})")
