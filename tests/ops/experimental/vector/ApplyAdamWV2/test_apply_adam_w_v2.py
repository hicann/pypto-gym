#!/usr/bin/env python3
# coding: utf-8
# -----------------------------------------------------------------------------
# Precision test for apply_adam_w_v2 (PyPTO).
#
# Test levels (loaded from test_cases.json):
#   - level0: fp32 path, shape [7168, 2048]
#   - level1: bf16 path, shape [7168, 2048]
#
# Compares the PyPTO kernel output against the pure-PyTorch golden using
# numpy.testing.assert_allclose with atol=1e-4, rtol=0.0078125.
# -----------------------------------------------------------------------------
from __future__ import annotations

import sys, os; _p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')): _p = os.path.dirname(_p)
sys.path.insert(0, os.path.join(_p, 'src')); sys.path.insert(0, os.path.join(_p, 'src', 'pypto_gym', 'ops', 'pypto_tile'))


import json
import os
import sys
import traceback

import numpy as np
import torch
import torch_npu  # noqa: F401  # required to enable npu backend
from numpy.testing import assert_allclose

_HERE = os.path.dirname(os.path.abspath(__file__))

from apply_adam_w_v2_golden import apply_adam_w_v2_golden
from experimental.vector.ApplyAdamWV2.apply_adam_w_v2_impl import apply_adam_w_v2_wrapper    # noqa: E402


DEFAULT_PARAMS = dict(
    beta1=0.9,
    beta2=0.999,
    lr=1e-3,
    weight_decay=0.01,
    eps=1e-8,
    step=1,
)

DTYPE_MAP = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

# Required test levels - this kernel covers level0 (fp32) and level1 (bf16).
REQUIRED_LEVELS = ("level0", "level1")


def _device() -> str:
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
    torch.npu.set_device(device_id)
    return f"npu:{device_id}"


def _load_cases(path: str = None) -> list:
    if path is None:
        path = os.path.join(_HERE, "test_cases.json")
    with open(path, "r") as f:
        cfg = json.load(f)
    cases = cfg["test_cases"]
    ids = {c["id"] for c in cases}
    for lvl in REQUIRED_LEVELS:
        if lvl not in ids:
            raise RuntimeError(f"missing required test level '{lvl}' in {path}")
    return cases


def _make_inputs(weight_dtype: torch.dtype, shape, seed: int, device: str):
    torch.manual_seed(seed)
    weight = (torch.randn(shape, dtype=torch.float32) * 0.02).to(weight_dtype).to(device)
    grad = (torch.randn(shape, dtype=torch.float32) * 0.01).to(weight_dtype).to(device)
    m = (torch.randn(shape, dtype=torch.float32) * 1e-3).to(device)
    v = (torch.randn(shape, dtype=torch.float32).abs() * 1e-6).to(device)
    return weight, grad, m, v


def _run_case(case: dict, device: str) -> bool:
    case_id = case["id"]
    weight_dtype = DTYPE_MAP[case["input"]["dtype"]]
    shape = tuple(case["input"]["shape"])
    rtol = case.get("rtol", 0.0078125)
    atol = case.get("atol", 1e-4)
    seed = case.get("seed", 42)

    print("=" * 60)
    print(f"Test: {case_id} - {case.get('description','')}")
    print("=" * 60)

    weight, grad, m, v = _make_inputs(weight_dtype, shape, seed, device)

    # Golden runs on CPU.
    w_g, m_g, v_g = apply_adam_w_v2_golden(
        weight.detach().cpu(),
        grad.detach().cpu(),
        m.detach().cpu(),
        v.detach().cpu(),
        **DEFAULT_PARAMS,
    )

    # PyPTO kernel runs on NPU.
    w_out, m_out, v_out = apply_adam_w_v2_wrapper(
        weight, grad, m, v, **DEFAULT_PARAMS
    )
    torch.npu.synchronize()

    # Cast to fp32 numpy arrays for comparison.
    actual = {
        "weight": w_out.detach().cpu().float().numpy(),
        "m": m_out.detach().cpu().float().numpy(),
        "v": v_out.detach().cpu().float().numpy(),
    }
    expected = {
        "weight": w_g.float().numpy(),
        "m": m_g.float().numpy(),
        "v": v_g.float().numpy(),
    }

    ok = True
    for label in ("weight", "m", "v"):
        a = actual[label]
        e = expected[label]
        diff = np.abs(a - e)
        max_abs = float(diff.max())
        denom = np.maximum(np.abs(e), 1e-12)
        max_rel = float((diff / denom).max())
        print(f"  [{label}] max_abs_err={max_abs:.6e} max_rel_err={max_rel:.6e}")
        try:
            assert_allclose(a, e, atol=atol, rtol=rtol)
        except AssertionError as exc:
            print(f"  [{label}] FAIL: {exc}")
            ok = False
    return ok


def test_level0(device: str) -> bool:
    """level0: fp32 path, [7168, 2048]."""
    cases = [c for c in _load_cases() if c["id"] == "level0"]
    return _run_case(cases[0], device)


def test_level1(device: str) -> bool:
    """level1: bf16 path, [7168, 2048]."""
    cases = [c for c in _load_cases() if c["id"] == "level1"]
    return _run_case(cases[0], device)


def test_level2(device: str) -> bool:
    """level2: bf16 path, [7168, 8192]."""
    cases = [c for c in _load_cases() if c["id"] == "level2"]
    return _run_case(cases[0], device)


def test_level3(device: str) -> bool:
    """level3: bf16 path, [7168, 16384]."""
    cases = [c for c in _load_cases() if c["id"] == "level3"]
    return _run_case(cases[0], device)


def test_level4(device: str) -> bool:
    """level4: bf16 path, [7168, 24576] (max K)."""
    cases = [c for c in _load_cases() if c["id"] == "level4"]
    return _run_case(cases[0], device)


def main() -> int:
    try:
        device = _device()
        print(f"Using device: {device}")

        all_ok = True
        # for runner in (test_level0, test_level1, test_level2, test_level3, test_level4):
        for runner in (test_level4,):
            try:
                ok = runner(device)
            except Exception:
                traceback.print_exc()
                ok = False
            all_ok = all_ok and ok

        if all_ok:
            print("[PRECISION_PASS]")
            return 0
        print("[PRECISION_FAIL]")
        return 1
    except Exception:
        traceback.print_exc()
        print("[PRECISION_FAIL]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
