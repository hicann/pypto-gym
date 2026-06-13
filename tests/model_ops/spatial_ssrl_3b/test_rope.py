#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance of the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
RoPE 精度测试脚本
遍历 test_cases.json 执行精度对比
"""

import os
import sys
from pathlib import Path

import json
import argparse
import torch
import torch_npu  # noqa: F401

import sys
import os
_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')):
    _p = os.path.dirname(_p)
sys.path.insert(0, os.path.join(_p, 'src'))
sys.path.insert(0, os.path.join(_p, 'src', 'pypto_gym', 'ops', 'pypto_tile'))

import numpy as np
from numpy.testing import assert_allclose


from rope_golden import (
    apply_rotary_pos_emb_vision_golden,
    apply_multimodal_rotary_pos_emb_golden
)
from spatial_ssrl_3b.rope.rope_impl import (
    apply_rotary_pos_emb_vision_impl,
    apply_multimodal_rotary_pos_emb_impl
)


def get_device():
    if "TILE_FWK_DEVICE_ID" in os.environ:
        device_id = int(os.environ["TILE_FWK_DEVICE_ID"])
        return f"npu:{device_id}"
    return "cpu"


def load_test_cases():
    json_path = os.path.join(os.path.dirname(__file__), "test_cases_rope.json")
    if not os.path.exists(json_path):
        raise RuntimeError(f"Test cases file not found: {json_path}")
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
    rope_type = inputs.get("rope_type", "multimodal")
    
    dtype = dtype_map[inputs["q"]["dtype"]]
    
    q = torch.randn(inputs["q"]["shape"], dtype=dtype, device=device)
    k = torch.randn(inputs["k"]["shape"], dtype=dtype, device=device)
    cos = torch.randn(inputs["cos"]["shape"], dtype=dtype, device=device)
    sin = torch.randn(inputs["sin"]["shape"], dtype=dtype, device=device)
    
    if rope_type == "multimodal":
        mrope_section = inputs.get("mrope_section", [16, 24, 24])
        unsqueeze_dim = inputs.get("unsqueeze_dim", 1)
        
        q_golden, k_golden = apply_multimodal_rotary_pos_emb_golden(
            q.cpu(), k.cpu(), cos.cpu(), sin.cpu(), mrope_section, unsqueeze_dim
        ).to(device)
        
        q_impl, k_impl = apply_multimodal_rotary_pos_emb_impl(
            q, k, cos, sin, mrope_section, unsqueeze_dim
        )
    else:  # vision
        q_golden, k_golden = apply_rotary_pos_emb_vision_golden(
            q.cpu(), k.cpu(), cos.cpu(), sin.cpu()
        ).to(device)
        
        q_impl, k_impl = apply_rotary_pos_emb_vision_impl(q, k, cos, sin)
    
    q_diff = torch.abs(q_golden - q_impl).max().item()
    k_diff = torch.abs(k_golden - k_impl).max().item()
    
    print(f"  q max diff: {q_diff:.6e}")
    print(f"  k max diff: {k_diff:.6e}")
    
    rtol = case_data.get("rtol", 1e-3)
    atol = case_data.get("atol", 1e-3)
    
    try:
        assert_allclose(q_impl.cpu().numpy(), q_golden.cpu().numpy(), rtol=rtol, atol=atol)
        assert_allclose(k_impl.cpu().numpy(), k_golden.cpu().numpy(), rtol=rtol, atol=atol)
        print(f"[PRECISION_PASS] diff < {rtol}")
    except AssertionError as e:
        print(f"[PRECISION_FAIL] {e}", file=sys.stderr)
        raise
    
    expected_q_shape = case_data["output"]["q_shape"]
    expected_k_shape = case_data["output"]["k_shape"]
    expected_dtype = dtype_map[case_data["output"]["dtype"]]
    
    assert q_impl.shape == torch.Size(expected_q_shape), \
        f"q shape mismatch: {q_impl.shape} vs {expected_q_shape}"
    assert k_impl.shape == torch.Size(expected_k_shape), \
        f"k shape mismatch: {k_impl.shape} vs {expected_k_shape}"
    assert q_impl.dtype == expected_dtype, f"q dtype mismatch: {q_impl.dtype} vs {expected_dtype}"
    assert k_impl.dtype == expected_dtype, f"k dtype mismatch: {k_impl.dtype} vs {expected_dtype}"


def main():
    parser = argparse.ArgumentParser(description="RoPE 精度测试")
    parser.add_argument("case_id", nargs="?", help="运行单个用例")
    parser.add_argument("--list", action="store_true", help="列出所有用例")
    args = parser.parse_args()
    
    test_cases = load_test_cases()
    cases = test_cases.get("test_cases", [])
    
    if args.list:
        print(f"\nTest cases from test_cases_rope.json:\n")
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