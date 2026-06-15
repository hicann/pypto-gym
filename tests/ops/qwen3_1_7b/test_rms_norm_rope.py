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
Qwen3-1.7B RMSNorm + RoPE 精度测试脚本
遍历 test_cases.json 执行精度对比
"""

import os
import sys
from pathlib import Path

import json
import argparse
import logging
import torch
import torch_npu

_CUR = Path(__file__).resolve().parent
sys.path.insert(0, str(_CUR))
_IMPL = Path(__file__).resolve().parents[3] / "src/pypto_gym/ops/pypto_tile/qwen3_1_7b/rms_norm_rope"
sys.path.insert(0, str(_IMPL))

import numpy as np
from numpy.testing import assert_allclose

from rms_norm_rope_golden import rms_norm_rope_golden
from rrms_norm_rope_impl import qwen3_qk_rope_q, qwen3_qk_rope_k


def get_device():
    if "TILE_FWK_DEVICE_ID" in os.environ:
        device_id = int(os.environ["TILE_FWK_DEVICE_ID"])
        return f"npu:{device_id}"
    return "cpu"


def load_test_cases():
    json_path = os.path.join(os.path.dirname(__file__), "test_cases.json")
    if not os.path.exists(json_path):
        raise RuntimeError(f"Test cases file not found: {json_path}")
    with open(json_path, "r") as f:
        return json.load(f)


def run_single_case(case_data, device):
    case_id = case_data["id"]
    description = case_data.get("description", "")
    kernel_name = case_data.get("kernel", "qwen3_qk_rope_q")

    logging.info("=" * 60)
    logging.info(f"Test: {case_id} — {description}")
    logging.info(f"Kernel: {kernel_name}")
    logging.info("=" * 60)

    if torch.npu.is_available():
        dev_id = os.environ.get('TILE_FWK_DEVICE_ID', '0')
        torch.npu.set_device(f'npu:{dev_id}')

    torch.manual_seed(case_data.get("seed", 42))

    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}

    inputs = case_data["input"]

    x_dtype = dtype_map[inputs["x"]["dtype"]]
    x = torch.randn(inputs["x"]["shape"], dtype=x_dtype, device="cpu")

    cos_dtype = dtype_map[inputs["cos"]["dtype"]]
    cos = torch.randn(inputs["cos"]["shape"], dtype=cos_dtype, device="cpu")

    sin_dtype = dtype_map[inputs["sin"]["dtype"]]
    sin = torch.randn(inputs["sin"]["shape"], dtype=sin_dtype, device="cpu")

    w_norm_dtype = dtype_map[inputs["w_norm"]["dtype"]]
    w_norm = torch.randn(inputs["w_norm"]["shape"], dtype=w_norm_dtype, device="cpu")

    eps = inputs["eps"]["value"]

    x_npu = x.npu()
    cos_npu = cos.npu()
    sin_npu = sin.npu()
    w_norm_npu = w_norm.npu()

    out_golden = rms_norm_rope_golden(x_npu, cos_npu, sin_npu, w_norm_npu, eps)

    out_impl = torch.empty_like(x_npu)

    kernel = qwen3_qk_rope_q if kernel_name == "qwen3_qk_rope_q" else qwen3_qk_rope_k
    kernel(x_npu, cos_npu, sin_npu, w_norm_npu, out_impl)

    logging.info("\n[精度验证 - out]")
    out_diff = torch.abs(out_golden - out_impl).max().item()
    logging.info(f"  Max diff: {out_diff:.6e}")

    rtol = case_data.get("rtol", 0.1)
    atol = case_data.get("atol", 0.1)

    try:
        assert_allclose(out_impl.float().cpu().numpy(), out_golden.float().cpu().numpy(), rtol=rtol, atol=atol)
        logging.info(f"[PRECISION_PASS] out diff < {rtol}")
    except AssertionError as e:
        logging.info(f"[PRECISION_FAIL] out: {e}", file=sys.stderr)
        raise

    outputs = case_data["output"]

    expected_out_shape = torch.Size(outputs["out"]["shape"])
    expected_out_dtype = dtype_map[outputs["out"]["dtype"]]
    assert out_impl.shape == expected_out_shape, f"out shape mismatch: {out_impl.shape} vs {expected_out_shape}"
    assert out_impl.dtype == expected_out_dtype, f"out dtype mismatch: {out_impl.dtype} vs {expected_out_dtype}"


def test_qwen3_1_7b_rms_norm_rope():
    parser = argparse.ArgumentParser(description="Qwen3-1.7B RMSNorm + RoPE 精度测试")
    parser.add_argument("case_id", nargs="?", help="运行单个用例")
    parser.add_argument("--list", action="store_true", help="列出所有用例")
    args = parser.parse_args()

    test_cases = load_test_cases()
    cases = test_cases.get("test_cases", [])

    if args.list:
        logging.info(f"\nTest cases from test_cases.json:\n")
        for case in cases:
            logging.info(f"  {case['id']} — {case.get('description', '')} [{case.get('kernel', 'qwen3_qk_rope_q')}]")
        return

    device = get_device()
    if device.startswith("npu"):
        torch.npu.set_device(int(device.split(":")[1]))

    to_run = cases if not args.case_id else [c for c in cases if c["id"] == args.case_id]

    try:
        for case_data in to_run:
            run_single_case(case_data, device)
        logging.info("\n" + "=" * 60)
        logging.info("All tests passed!")
        logging.info("=" * 60)
    except Exception as e:
        logging.info(f"\nError: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_qwen3_1_7b_rms_norm_rope()