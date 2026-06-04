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
SwiGLU MLP 度测试脚本
遍历 test_cases.json 执行精度对比
"""

import os
import sys
import json
import argparse
import torch
import torch_npu

import sys
import os
_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')):
    _p = os.path.dirname(_p)
sys.path.insert(0, os.path.join(_p, 'src'))
sys.path.insert(0, os.path.join(_p, 'src', 'pypto_gym', 'ops', 'pypto_tile'))

import numpy as np
from numpy.testing import assert_allclose

from swiglu_mlp_golden import swiglu_mlp_golden
from gutenocr_3b.swiglu_mlp.swiglu_mlp_impl import swiglu_mlp_fused_static


def get_device():
    if "TILE_FWK_DEVICE_ID" in os.environ:
        device_id = int(os.environ["TILE_FWK_DEVICE_ID"])
        return f"npu:{device_id}"
    return "cpu"


def load_test_cases():
    json_path = os.path.join(os.path.dirname(__file__), "test_cases.json")
    if not os.path.exists(json_path):
        print(f"ERROR: {json_path} not found")
        sys.exit(1)
    with open(json_path, "r") as f:
        return json.load(f)


def run_single_case(case_data, device):
    case_id = case_data["id"]
    description = case_data.get("description", "")
    
    print("=" * 60)
    print(f"Test: {case_id} — {description}")
    print("=" * 60)
    
    torch.manual_seed(case_data.get("seed", 42))
    
    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    
    inputs = case_data["input"]
    dtype = dtype_map[inputs["x"]["dtype"]]
    
    x = torch.randn(inputs["x"]["shape"], dtype=dtype, device=device)
    batch_size = x.shape[0]
    
    gate_weight = torch.randn(inputs["gate_weight"]["shape"], dtype=dtype, device=device)
    gate_bias = torch.randn(inputs["gate_bias"]["shape"], dtype=dtype, device=device) if inputs["gate_bias"]["shape"] else None
    up_weight = torch.randn(inputs["up_weight"]["shape"], dtype=dtype, device=device)
    up_bias = torch.randn(inputs["up_bias"]["shape"], dtype=dtype, device=device) if inputs["up_bias"]["shape"] else None
    down_weight = torch.randn(inputs["down_weight"]["shape"], dtype=dtype, device=device)
    down_bias = torch.randn(inputs["down_bias"]["shape"], dtype=dtype, device=device) if inputs["down_bias"]["shape"] else None
    
    output_golden = swiglu_mlp_golden(x.cpu(), gate_weight.cpu(), gate_bias.cpu() if gate_bias else None,
                                       up_weight.cpu(), up_bias.cpu() if up_bias else None,
                                       down_weight.cpu(), down_bias.cpu() if down_bias else None).to(device)
    
    # PyPTO kernel expects transposed weights
    gate_weight_t = gate_weight.T.contiguous()
    up_weight_t = up_weight.T.contiguous()
    down_weight_t = down_weight.T.contiguous()
    
    # Create zero bias if None
    if gate_bias is None:
        gate_bias = torch.zeros(inputs["gate_weight"]["shape"][0], dtype=dtype, device=device)
    if up_bias is None:
        up_bias = torch.zeros(inputs["up_weight"]["shape"][0], dtype=dtype, device=device)
    if down_bias is None:
        down_bias = torch.zeros(inputs["down_weight"]["shape"][0], dtype=dtype, device=device)
    
    output_impl = swiglu_mlp_fused_static(x, gate_weight_t, gate_bias, up_weight_t, up_bias,
                                          down_weight_t, down_bias, batch_size_static=batch_size)
    
    max_diff = torch.abs(output_golden - output_impl).max().item()
    print(f"  Max diff: {max_diff:.6e}")
    
    rtol = case_data.get("rtol", 1e-2)
    atol = case_data.get("atol", 1e-2)
    
    try:
        assert_allclose(output_impl.cpu().numpy(), output_golden.cpu().numpy(), rtol=rtol, atol=atol)
        print(f"[PRECISION_PASS] diff < {rtol}")
    except AssertionError as e:
        print(f"[PRECISION_FAIL] {e}", file=sys.stderr)
        raise
    
    expected_shape = case_data["output"]["shape"]
    expected_dtype = dtype_map[case_data["output"]["dtype"]]
    assert output_impl.shape == torch.Size(expected_shape), f"Shape mismatch: {output_impl.shape} vs {expected_shape}"
    assert output_impl.dtype == expected_dtype, f"Dtype mismatch: {output_impl.dtype} vs {expected_dtype}"


def main():
    parser = argparse.ArgumentParser(description="SwiGLU MLP 度测试")
    parser.add_argument("case_id", nargs="?", help="运行单个用例")
    parser.add_argument("--list", action="store_true", help="列出所有用例")
    args = parser.parse_args()
    
    test_cases = load_test_cases()
    cases = test_cases.get("test_cases", [])
    
    if args.list:
        print(f"\nTest cases from test_cases.json:\n")
        for case in cases:
            print(f"  {case['id']} — {case.get('description', '')}")
        return
    
    device = get_device()
    if device.startswith("npu"):
        torch.npu.set_device(int(device.split(":")[1]))
    
    to_run = cases if not args.case_id else [c for c in cases if c["id"] == args.case_id]
    
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