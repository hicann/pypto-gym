#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

"""Pytest harness for engram_backward kernel.

Precision Standard 2.1 three-way comparison for all 6 gradient outputs:
  * NPU PyPTO-Pro kernel output;
  * NPU benchmark from the small torch implementation at kernel dtype;
  * FP64 golden output from the same small implementation on CPU.

Run on NPU:
    pytest test_engram_backward.py -v
or direct:
    python test_engram_backward.py
"""

import logging
import math
import os
import sys

import torch
import torch_npu  # noqa: F401
import pytest
from numpy.testing import assert_allclose

# Ensure we can import the golden reference that ships next to this test.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Locate the repo root (the directory containing 'src') and add it to sys.path
# so the kernel impl under src/pypto_gym/.../engram/ is importable as a package.
_REPO_ROOT = _THIS_DIR
while not os.path.isdir(os.path.join(_REPO_ROOT, "src")):
    _REPO_ROOT = os.path.dirname(_REPO_ROOT)
    if _REPO_ROOT == os.path.dirname(_REPO_ROOT):
        break
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from engram_backward_golden import (
    engram_backward_golden,
    engram_forward_with_cache,
    _get_device,
)
from pypto_gym.ops.pypto_pro.experimental.ops_transformer.engram.engram_backward_impl import (
    engram_backward_wrapper,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Precision Standard 2.1 — triple comparison thresholds
# ═══════════════════════════════════════════════════════════════════
TRIPLE_THRESHOLDS = (2.0, 1.2, 1.2)  # MARE, MERE, RMSE ratio thresholds

SMALL_VALUE_THRES = {
    torch.float16: 2 ** -11,
    torch.bfloat16: 2 ** -8,
    torch.float32: 2 ** -14,
}

SMALL_VALUE_ERROR_THRES = {
    torch.float16: 2 ** -16,
    torch.bfloat16: 2 ** -16,
    torch.float32: 2 ** -30,
}

GRAD_NAMES = [
    "grad_hidden_states",
    "grad_embeddings",
    "grad_key_proj_weights",
    "grad_value_proj_weights",
    "grad_key_gamma",
    "grad_query_gamma",
]


# ═══════════════════════════════════════════════════════════════════
# Input construction
# ═══════════════════════════════════════════════════════════════════

def _make_case(device, b, s, m_h=4, h=1280, de=512, dtype=torch.bfloat16, seed=42):
    """Construct backward inputs: forward inputs → forward_with_cache → grad_output.

    gamma initialized to 1; weights ×0.5 to prevent BF16 matmul overflow.
    """
    torch.manual_seed(seed)
    hidden_states = torch.randn(b, s, m_h, h, dtype=dtype, device=device)
    embeddings = torch.randn(b, s, de, dtype=dtype, device=device)
    key_proj_weights = torch.randn(m_h, de, h, dtype=dtype, device=device) * 0.5
    value_proj_weights = torch.randn(de, h, dtype=dtype, device=device) * 0.5
    key_gamma = torch.ones(m_h, h, dtype=dtype, device=device)
    query_gamma = torch.ones(m_h, h, dtype=dtype, device=device)

    clamp_value, eps = 1e-6, 1e-6
    with torch.no_grad():
        _, cache = engram_forward_with_cache(
            hidden_states, embeddings, key_proj_weights, value_proj_weights,
            key_gamma, query_gamma, clamp_value, eps,
        )

    grad_output = torch.randn(b, s, m_h, h, dtype=dtype, device=device)

    # Args in engram_backward_wrapper signature order
    kernel_args = (
        grad_output,
        hidden_states, embeddings, key_proj_weights, value_proj_weights,
        key_gamma, query_gamma,
        cache["scores"].float(), cache["gates"].float(),
        cache["keys"], cache["value"],
    )
    kwargs = {"clamp_value": clamp_value, "eps": eps}
    return kernel_args, kwargs


# ═══════════════════════════════════════════════════════════════════
# Precision Standard 2.1 comparison helpers
# ═══════════════════════════════════════════════════════════════════

def _get_split_index(golden_data, dtype):
    """Split FP64 golden values into large and small-value regions."""
    thres = SMALL_VALUE_THRES[dtype]
    large_mask = torch.abs(golden_data) >= thres
    small_mask = torch.abs(golden_data) < thres
    return large_mask, small_mask, thres


def _compute_small_value(input_data, golden_data, dtype, small_mask):
    """Count small golden values whose absolute error exceeds the dtype bound."""
    if not torch.any(small_mask):
        return 0
    thres = SMALL_VALUE_ERROR_THRES[dtype]
    return torch.sum(torch.abs(input_data[small_mask] - golden_data[small_mask]) > thres).item()


def _compute_large_value(input_data, golden_data, large_mask):
    """Compute MARE, MERE and RMSE over the large-value region."""
    if not torch.any(large_mask):
        return 0.0, 0.0, 0.0
    input_large = input_data[large_mask]
    golden_large = golden_data[large_mask]
    abs_diff = torch.abs(input_large - golden_large)
    relative_error = abs_diff / (torch.abs(golden_large) + 1e-7)
    mare = torch.max(relative_error).item()
    mere = torch.mean(relative_error).item()
    rmse = torch.sqrt(torch.mean((input_large - golden_large) ** 2)).item()
    return mare, mere, rmse


def _compute_re(input_value, benchmark_value, small_value_thres):
    """Normalize kernel error by benchmark error, matching Precision 2.1."""
    if math.isinf(benchmark_value) or math.isnan(benchmark_value):
        return 1.0
    if math.isinf(input_value) or math.isnan(input_value):
        return 1000.0
    return input_value / max(benchmark_value, small_value_thres)


def precision_compare_triple(npu_data, benchmark_data, golden_data,
                             thresholds=TRIPLE_THRESHOLDS):
    """Precision Standard 2.1 three-way comparison."""
    if npu_data.shape != benchmark_data.shape or npu_data.shape != golden_data.shape:
        return "FAILED", float("inf"), float("inf"), float("inf"), float("inf")

    dtype = npu_data.dtype
    if dtype not in SMALL_VALUE_THRES:
        raise TypeError(f"Unsupported Precision Standard 2.1 dtype: {dtype}")

    npu_fp32 = npu_data.to(torch.float32).cpu()
    benchmark_fp32 = benchmark_data.to(torch.float32).cpu()
    golden_fp32 = golden_data.to(torch.float32).cpu()

    large_idx, small_idx, small_thres = _get_split_index(golden_fp32, dtype)
    npu_small_errors = _compute_small_value(npu_fp32, golden_fp32, dtype, small_idx)
    benchmark_small_errors = _compute_small_value(
        benchmark_fp32, golden_fp32, dtype, small_idx
    )
    small_value_ratio = npu_small_errors / max(benchmark_small_errors, 1)

    npu_mare, npu_mere, npu_rmse = _compute_large_value(npu_fp32, golden_fp32, large_idx)
    benchmark_mare, benchmark_mere, benchmark_rmse = _compute_large_value(
        benchmark_fp32, golden_fp32, large_idx
    )
    mare_ratio = _compute_re(npu_mare, benchmark_mare, small_thres)
    mere_ratio = _compute_re(npu_mere, benchmark_mere, small_thres)
    rmse_ratio = _compute_re(npu_rmse, benchmark_rmse, small_thres)

    passed = (
        small_value_ratio <= 2.0
        and mare_ratio <= thresholds[0]
        and mere_ratio <= thresholds[1]
        and rmse_ratio <= thresholds[2]
    )
    result = "PASS" if passed else "FAILED"
    return result, mare_ratio, mere_ratio, rmse_ratio, small_value_ratio


def _compare(name, npu_t, benchmark_t, golden_t):
    """Compare one gradient using Precision Standard 2.1."""
    result, mare, mere, rmse, small_value = precision_compare_triple(
        npu_t, benchmark_t, golden_t
    )
    log.info(
        f"  {name:30s} MARE={mare:.4f} MERE={mere:.4f} "
        f"RMSE={rmse:.4f} SmallVal={small_value:.4f} [{result}]"
    )
    return result == "PASS"


# ═══════════════════════════════════════════════════════════════════
# Core case runner: kernel vs benchmark vs FP64 golden
# ═══════════════════════════════════════════════════════════════════

def run_engram_backward_case(b, s, m_h=4, h=1280, de=512):
    """Run one case. Returns dict of per-gradient PASS/FAIL booleans."""
    device = _get_device()

    kernel_args, kwargs = _make_case(device, b, s, m_h, h, de)

    # Run kernel wrapper (scores / gates widened to FP32)
    npu_grads = engram_backward_wrapper(*kernel_args, **kwargs)

    # Benchmark: same inputs at low-precision dtype
    with torch.no_grad():
        benchmark_grads = engram_backward_golden(*kernel_args, **kwargs)

    # CPU FP64 golden: all tensor inputs widened
    golden_args = tuple(arg.detach().cpu().to(torch.float64) for arg in kernel_args)
    with torch.no_grad():
        golden_grads = engram_backward_golden(*golden_args, **kwargs)

    results = {}
    for name, nt, bt, gt in zip(GRAD_NAMES, npu_grads, benchmark_grads, golden_grads):
        results[name] = _compare(name, nt, bt, gt)
    return results


# ═══════════════════════════════════════════════════════════════════
# Test case matrix: (id, b, s, m_h, h, de)
#   smoke + typical shapes + boundary cases (小 m 补齐, m_h 范围, h=2560)
# ═══════════════════════════════════════════════════════════════════

CASES = [
    pytest.param(1, 128, 4, 1280, 512, id="b1_s128_mhc4_hiddensize1280_dime512"),
    pytest.param(1, 512, 4, 1280, 512, id="b1_s512_mhc4_hiddensize1280_dime512"),
    pytest.param(1, 1024, 4, 1280, 1024, id="b1_s1024_mhc4_hiddensize1280_dime1024"),
    pytest.param(16, 512, 4, 2560, 512, id="b16_s512_mhc4_hiddensize2560_dime512"),
    pytest.param(16, 1024, 4, 2560, 1024, id="b16_s1024_mhc4_hiddensize2560_dime1024"),
    pytest.param(1, 63, 4, 1280, 1024, id="b1_s63_mhc4_hiddensize1280_dime1024"),
    pytest.param(1, 127, 4, 1280, 512, id="b1_s127_mhc4_hiddensize1280_dime512"),
    pytest.param(1, 1023, 4, 2560, 1024, id="b1_s1023_mhc4_hiddensize2560_dime1024"),
    pytest.param(3, 255, 8, 1280, 1024, id="b3_s255_mhc8_hiddensize1280_dime1024"),
    pytest.param(5, 513, 16, 2560, 512, id="b5_s513_mhc16_hiddensize2560_dime512"),
]


@pytest.mark.soc("950")
@pytest.mark.parametrize("b,s,m_h,h,de", CASES)
def test_engram_backward_pytest(b, s, m_h, h, de):
    """Precision Standard 2.1 three-way comparison for all 6 gradients."""
    results = run_engram_backward_case(b, s, m_h, h, de)
    failed = [n for n, ok in results.items() if not ok]
    assert not failed, f"Precision check failed for: {failed}"


# ═══════════════════════════════════════════════════════════════════
# Direct execution entry point (no pytest needed)
# ═══════════════════════════════════════════════════════════════════

def main():
    all_pass = True
    for case in CASES:
        b, s, m_h, h, de = case.values
        name = case.id
        try:
            results = run_engram_backward_case(b, s, m_h, h, de)
            ok = all(results.values())
        except Exception as exc:
            log.info("  EXCEPTION in %s: %s", name, exc)
            ok = False
        log.info("  %-30s %s", name, "PASS" if ok else "FAIL")
        all_pass = all_pass and ok

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
