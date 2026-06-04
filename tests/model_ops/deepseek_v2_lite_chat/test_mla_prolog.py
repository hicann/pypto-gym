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
MLA KV Prolog 精度测试脚本
遍历 test_cases.json 执行精度对比
"""

import os
import sys
from pathlib import Path

import json
import argparse
import torch
import torch_npu  # noqa: F401

_CUR = Path(__file__).resolve().parent
sys.path.insert(0, str(_CUR))
_IMPL = Path(__file__).resolve().parents[3] / "src/pypto_gym/ops/pypto_tile/deepseek_v2_lite_chat/mla_prolog"
sys.path.insert(0, str(_IMPL))

import numpy as np
from numpy.testing import assert_allclose

from mla_prolog_golden import mla_prolog_golden
from mla_prolog import mla_prolog_hybrid_optimized


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
    int_dtype_map = {"int64": torch.long, "int32": torch.int32}

    inputs = case_data["input"]
    
    hidden_dtype = dtype_map[inputs["hidden_states"]["dtype"]]
    hidden_states = torch.randn(inputs["hidden_states"]["shape"], dtype=hidden_dtype, device="cpu")
    
    kv_a_weight_dtype = dtype_map[inputs["kv_a_weight"]["dtype"]]
    kv_a_weight = torch.randn(inputs["kv_a_weight"]["shape"], dtype=kv_a_weight_dtype, device="cpu")
    
    kv_b_weight_dtype = dtype_map[inputs["kv_b_weight"]["dtype"]]
    kv_b_weight = torch.randn(inputs["kv_b_weight"]["shape"], dtype=kv_b_weight_dtype, device="cpu")
    
    ln_weight_dtype = dtype_map[inputs["ln_weight"]["dtype"]]
    ln_weight = torch.randn(inputs["ln_weight"]["shape"], dtype=ln_weight_dtype, device="cpu")
    
    eps = inputs["eps"]["value"]
    
    cos_dtype = dtype_map[inputs["cos"]["dtype"]]
    cos = torch.randn(inputs["cos"]["shape"], dtype=cos_dtype, device="cpu")
    
    sin_dtype = dtype_map[inputs["sin"]["dtype"]]
    sin = torch.randn(inputs["sin"]["shape"], dtype=sin_dtype, device="cpu")
    
    pos_ids_dtype = int_dtype_map[inputs["pos_ids"]["dtype"]]
    bsz = inputs["hidden_states"]["shape"][0]
    seq_len = inputs["hidden_states"]["shape"][1]
    pos_ids = torch.arange(seq_len, dtype=pos_ids_dtype, device="cpu").unsqueeze(0).expand(bsz, -1)

    k_nope_golden, value_golden, k_pe_golden = mla_prolog_golden(
        hidden_states, kv_a_weight, kv_b_weight, ln_weight, eps, cos, sin, pos_ids
    )

    k_nope_impl, value_impl, k_pe_impl = mla_prolog_hybrid_optimized(
        hidden_states, kv_a_weight, kv_b_weight, ln_weight, eps, cos, sin, pos_ids
    )

    print("\n[精度验证 - k_nope]")
    k_nope_diff = torch.abs(k_nope_golden - k_nope_impl).max().item()
    print(f"  Max diff: {k_nope_diff:.6e}")

    print("\n[精度验证 - value]")
    value_diff = torch.abs(value_golden - value_impl).max().item()
    print(f"  Max diff: {value_diff:.6e}")

    print("\n[精度验证 - k_pe]")
    k_pe_diff = torch.abs(k_pe_golden - k_pe_impl).max().item()
    print(f"  Max diff: {k_pe_diff:.6e}")

    rtol = case_data.get("rtol", 1e-2)
    atol = case_data.get("atol", 1e-2)

    try:
        assert_allclose(k_nope_impl.cpu().numpy(), k_nope_golden.cpu().numpy(), rtol=rtol, atol=atol)
        print(f"[PRECISION_PASS] k_nope diff < {rtol}")
    except AssertionError as e:
        print(f"[PRECISION_FAIL] k_nope: {e}", file=sys.stderr)
        raise

    try:
        assert_allclose(value_impl.cpu().numpy(), value_golden.cpu().numpy(), rtol=rtol, atol=atol)
        print(f"[PRECISION_PASS] value diff < {rtol}")
    except AssertionError as e:
        print(f"[PRECISION_FAIL] value: {e}", file=sys.stderr)
        raise

    try:
        assert_allclose(k_pe_impl.cpu().numpy(), k_pe_golden.cpu().numpy(), rtol=rtol, atol=atol)
        print(f"[PRECISION_PASS] k_pe diff < {rtol}")
    except AssertionError as e:
        print(f"[PRECISION_FAIL] k_pe: {e}", file=sys.stderr)
        raise

    outputs = case_data["output"]
    
    expected_k_nope_shape = torch.Size(outputs["k_nope"]["shape"])
    expected_k_nope_dtype = dtype_map[outputs["k_nope"]["dtype"]]
    assert k_nope_impl.shape == expected_k_nope_shape, f"k_nope shape mismatch: {k_nope_impl.shape} vs {expected_k_nope_shape}"
    assert k_nope_impl.dtype == expected_k_nope_dtype, f"k_nope dtype mismatch: {k_nope_impl.dtype} vs {expected_k_nope_dtype}"

    expected_value_shape = torch.Size(outputs["value"]["shape"])
    expected_value_dtype = dtype_map[outputs["value"]["dtype"]]
    assert value_impl.shape == expected_value_shape, f"value shape mismatch: {value_impl.shape} vs {expected_value_shape}"
    assert value_impl.dtype == expected_value_dtype, f"value dtype mismatch: {value_impl.dtype} vs {expected_value_dtype}"

    expected_k_pe_shape = torch.Size(outputs["k_pe"]["shape"])
    expected_k_pe_dtype = dtype_map[outputs["k_pe"]["dtype"]]
    assert k_pe_impl.shape == expected_k_pe_shape, f"k_pe shape mismatch: {k_pe_impl.shape} vs {expected_k_pe_shape}"
    assert k_pe_impl.dtype == expected_k_pe_dtype, f"k_pe dtype mismatch: {k_pe_impl.dtype} vs {expected_k_pe_dtype}"


def main():
    parser = argparse.ArgumentParser(description="MLA KV Prolog 精度测试")
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