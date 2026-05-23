#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PyPTO RMSNorm operator test."""

import json
import os
import sys
import argparse

import torch
import numpy as np
import torch_npu  # noqa: F401
from numpy.testing import assert_allclose

_P = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_P, 'src')):
    _P = os.path.dirname(_P)
sys.path.insert(0, os.path.join(_P, 'src'))
sys.path.insert(0, os.path.join(_P, 'src', 'pypto_gym', 'ops', 'pypto_tile'))

_HERE = os.path.dirname(os.path.abspath(__file__))

from rms_norm_golden import RMSNorm_golden  # noqa: E402
from experimental.vector.rms_norm.rms_norm_impl import RMSNorm_wrapper  # noqa: E402


# ─────────────────────────────────────────────
# 1. environment utilities
# ─────────────────────────────────────────────

def get_device_id():
    """Get TILE_FWK_DEVICE_ID from environment."""
    if "TILE_FWK_DEVICE_ID" not in os.environ:
        print("Please set: export TILE_FWK_DEVICE_ID=<device_id>")
        return 0
    try:
        return int(os.environ["TILE_FWK_DEVICE_ID"])
    except ValueError:
        print(f"ERROR: TILE_FWK_DEVICE_ID must be int, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return 0


def load_test_cases(json_path=None):
    """Load test cases."""
    if json_path is None:
        json_path = os.path.join(_HERE, "test_cases.json")
    if not os.path.exists(json_path):
        print(f"ERROR: {json_path} not found")
        sys.exit(1)
    with open(json_path, "r") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# 2. test execution
# ─────────────────────────────────────────────

def run_single_case(case_data, device_id=None, run_mode="npu"):
    """Run a single test case."""
    case_id = case_data["id"]
    description = case_data.get("description", "")

    print("=" * 60)
    print(f"Test: {case_id} — {description}")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(case_data.get("seed", 42))

    input_shape = tuple(case_data["input"]["shape"])
    input_dtype = case_data["input"]["dtype"]
    eps = case_data.get("eps", 1e-5)
    rtol = case_data.get("rtol", 1e-3)
    atol = case_data.get("atol", 1e-3)

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(input_dtype, torch.float32)
    x = torch.randn(input_shape, dtype=dtype, device=device)

    result = RMSNorm_wrapper(x, eps=eps)

    golden = RMSNorm_golden(x, eps=eps)

    if run_mode == "npu" and device_id is not None:
        torch.npu.synchronize()

    print(f"  Input shape : {x.shape}")
    print(f"  Output shape: {result.shape}")
    print(f"  eps         : {eps}")

    result_np = result.cpu().float().numpy()
    golden_np = golden.cpu().float().numpy()
    max_diff = np.abs(result_np - golden_np).max()
    print(f"  Max diff    : {max_diff:.6e}")

    if run_mode == "npu":
        try:
            assert_allclose(result_np, golden_np, rtol=rtol, atol=atol)
            print(f"  ✓ Case {case_id} passed (rtol={rtol}, atol={atol})\n")
            return True
        except AssertionError as e:
            print(f"[PRECISION_FAIL] Case {case_id}: {e}", file=sys.stderr)
            return False
        except Exception as e:
            print(f"Runtime error in case {case_id}: {e}", file=sys.stderr)
            raise

    print(f"  ✓ Case {case_id} passed (sim mode, no NPU comparison)\n")
    return True


# ─────────────────────────────────────────────
# 3. CLI entry
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PyPTO RMSNorm operator test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                     Run all cases
  %(prog)s case_001            Run specific case
  %(prog)s --list              List all cases
        """,
    )
    parser.add_argument("case_id", type=str, nargs="?", help="Case ID to run")
    parser.add_argument("--list", action="store_true", help="List available cases")
    parser.add_argument(
        "--run_mode", "--run-mode",
        type=str, default="npu", choices=["npu", "sim"],
        help="Run mode (default: npu)",
    )
    parser.add_argument(
        "--json", type=str, default="test_cases.json",
        help="Test cases JSON file (default: test_cases.json)",
    )
    args = parser.parse_args()

    test_cases = load_test_cases(args.json)
    cases = test_cases.get("test_cases", [])

    if not cases:
        print("ERROR: No test cases found in JSON")
        sys.exit(1)

    if args.list:
        print(f"\nTest cases from {args.json}:\n")
        for case in cases:
            case_id = case["id"]
            desc = case.get("description", "")
            shape = case["input"]["shape"]
            dtype = case["input"]["dtype"]
            eps = case.get("eps", 1e-5)
            print(f"  {case_id}  — {desc}  [{dtype} {shape}] eps={eps}")
        return

    if args.case_id:
        case_data = None
        for case in cases:
            if case["id"] == args.case_id:
                case_data = case
                break
        if case_data is None:
            print(f"ERROR: unknown case '{args.case_id}'")
            print(f"Valid: {', '.join([c['id'] for c in cases])}")
            sys.exit(1)
        to_run = [case_data]
    else:
        to_run = cases

    device_id = None
    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            sys.exit(1)
        torch.npu.set_device(device_id)

    all_passed = True
    try:
        passed = 0
        for case_data in to_run:
            print(f"\n▸ Running {case_data['id']}")
            case_ok = run_single_case(case_data, device_id, args.run_mode)
            if case_ok:
                passed += 1
            else:
                all_passed = False

        print("\n" + "=" * 60)
        if all_passed:
            print(f"All tests passed! ({passed}/{len(to_run)} cases)")
            print("[PRECISION_PASS]")
        else:
            print(f"Some tests FAILED ({passed}/{len(to_run)} passed)")
        print("=" * 60)

        if not all_passed:
            sys.exit(1)

    except Exception as e:
        print(f"\nRuntime error: {e}", file=sys.stderr)
        import traceback as _tb
        _tb.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
