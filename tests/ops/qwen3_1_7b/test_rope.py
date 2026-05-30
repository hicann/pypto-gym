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
RoPE 精度测试脚本 - 部分融合算子
融合范围: Q/K per-head RMSNorm + RoPE

测试内容:
1. qwen3_qk_rope_q (N_q=16)
2. qwen3_qk_rope_k (N_kv=8)
"""

import os
import sys
import json
import argparse
import torch
import torch_npu  # noqa: F401
from pathlib import Path

_CUR = Path(__file__).resolve().parent
sys.path.insert(0, str(_CUR))
_IMPL = Path(__file__).resolve().parents[3] / "src/pypto_gym/ops/pypto_tile/qwen3_1_7b"
sys.path.insert(0, str(_IMPL))

from numpy.testing import assert_allclose
from rope_golden import rope_golden_3d, generate_cos_sin
from rope.rope_impl import qwen3_qk_rope_q, qwen3_qk_rope_k


def get_device():
    if "TILE_FWK_DEVICE_ID" in os.environ:
        device_id = int(os.environ["TILE_FWK_DEVICE_ID"])
        return f"npu:{device_id}"
    return "cpu"


def load_test_cases():
    json_path = _CUR / "test_rope.json"
    if not json_path.exists():
        print(f"ERROR: {json_path} not found")
        sys.exit(1)
    with open(json_path, "r") as f:
        return json.load(f)


def run_single_case(case_data, device):
    case_id = case_data["id"]
    description = case_data.get("description", "")
    kernel_name = case_data.get("kernel", "qwen3_qk_rope_q")

    print("=" * 60)
    print(f"Test: {case_id} — {description}")
    print(f"Kernel: {kernel_name}")
    print("=" * 60)

    torch.manual_seed(case_data.get("seed", 42))

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    inputs = case_data["input"]
    dtype = dtype_map[inputs["x"]["dtype"]]

    x_shape = inputs["x"]["shape"]
    S = x_shape[0]
    D = x_shape[2]
    rope_theta = inputs.get("rope_theta", {}).get("value", 1000000.0)
    eps = inputs["eps"]["value"]

    # Generate input
    x = torch.randn(x_shape, dtype=dtype, device="cpu")
    norm_weight = torch.randn(inputs["norm_weight"]["shape"], dtype=dtype, device="cpu")

    # Generate cos/sin
    cos, sin = generate_cos_sin(S, D, rope_theta, "cpu")

    # Golden output
    output_golden = rope_golden_3d(x, cos, sin, norm_weight, eps)

    # PyPTO kernel output
    output_impl = torch.empty(x_shape, dtype=dtype, device="cpu")

    try:
        if kernel_name == "qwen3_qk_rope_q":
            qwen3_qk_rope_q(x, cos, sin, norm_weight, output_impl)
        elif kernel_name == "qwen3_qk_rope_k":
            qwen3_qk_rope_k(x, cos, sin, norm_weight, output_impl)
        else:
            raise ValueError(f"Unknown kernel: {kernel_name}")
    except Exception as e:
        print(f"  ❌ Kernel execution failed: {e}")
        raise

    max_diff = torch.abs(output_golden - output_impl).max().item()
    print(f"  Max diff: {max_diff:.6e}")

    rtol = case_data.get("rtol", 1e-2)
    atol = case_data.get("atol", 1e-2)

    try:
        assert_allclose(output_impl.numpy(), output_golden.numpy(), rtol=rtol, atol=atol)
        print(f"  ✓ [PRECISION_PASS] diff < {rtol}")
    except AssertionError as e:
        print(f"  ❌ [PRECISION_FAIL] {e}", file=sys.stderr)
        raise

    expected_shape = case_data["output"]["shape"]
    expected_dtype = dtype_map[case_data["output"]["dtype"]]
    assert output_impl.shape == torch.Size(expected_shape), f"Shape mismatch: {output_impl.shape} vs {expected_shape}"
    assert output_impl.dtype == expected_dtype, f"Dtype mismatch: {output_impl.dtype} vs {expected_dtype}"
    print(f"  ✓ Shape and dtype verified")


def main():
    parser = argparse.ArgumentParser(description="RoPE 精度测试")
    parser.add_argument("case_id", nargs="?", help="运行单个用例")
    parser.add_argument("--list", action="store_true", help="列出所有用例")
    parser.add_argument("--kernel", choices=["q", "k"], help="只测试 Q 或 K kernel")
    args = parser.parse_args()

    test_cases = load_test_cases()
    cases = test_cases.get("test_cases", [])

    if args.list:
        print(f"\nTest cases from test_rope.json:\n")
        for case in cases:
            kernel = case.get("kernel", "")
            print(f"  {case['id']} — {case.get('description', '')} [{kernel}]")
        return

    device = get_device()
    if device.startswith("npu"):
        torch.npu.set_device(int(device.split(":")[1]))

    # Filter by kernel type
    if args.kernel:
        target_kernel = f"qwen3_qk_rope_{args.kernel}"
        cases = [c for c in cases if c.get("kernel") == target_kernel]

    to_run = cases if not args.case_id else [c for c in cases if c["id"] == args.case_id]

    if not to_run:
        print(f"No matching test cases found")
        return

    try:
        for case_data in to_run:
            run_single_case(case_data, device)
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()