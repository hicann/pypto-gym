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
Qwen3-1.7B RMSNorm + RoPE Golden 自洽性测试
验证 golden 参考实现在多个 shape/dtype 下输出合法。
"""

import os
import sys
from pathlib import Path

import json
import logging
import torch

_CUR = Path(__file__).resolve().parent
sys.path.insert(0, str(_CUR))

from rms_norm_rope_golden import rms_norm_rope_golden


def load_test_cases():
    json_path = os.path.join(os.path.dirname(__file__), "test_cases.json")
    if not os.path.exists(json_path):
        raise RuntimeError(f"Test cases file not found: {json_path}")
    with open(json_path, "r") as f:
        return json.load(f)


def run_single_case(case_data):
    case_id = case_data["id"]
    description = case_data.get("description", "")

    logging.info("=" * 60)
    logging.info(f"Test: {case_id} — {description}")
    logging.info("=" * 60)

    torch.manual_seed(case_data.get("seed", 42))
    device = "cpu"

    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}

    inputs = case_data["input"]
    x_dtype = dtype_map[inputs["x"]["dtype"]]
    x = torch.randn(inputs["x"]["shape"], dtype=x_dtype, device=device)
    cos = torch.randn(inputs["cos"]["shape"], dtype=x_dtype, device=device)
    sin = torch.randn(inputs["sin"]["shape"], dtype=x_dtype, device=device)
    w_norm = torch.randn(inputs["w_norm"]["shape"], dtype=x_dtype, device=device)
    eps = inputs["eps"]["value"]

    out = rms_norm_rope_golden(x, cos, sin, w_norm, eps)

    # shape/dtype check
    expected_shape = torch.Size(case_data["output"]["out"]["shape"])
    expected_dtype = dtype_map[case_data["output"]["out"]["dtype"]]
    assert out.shape == expected_shape, f"shape mismatch: {out.shape} vs {expected_shape}"
    assert out.dtype == expected_dtype, f"dtype mismatch: {out.dtype} vs {expected_dtype}"

    # sanity: no NaN / Inf
    assert not torch.isnan(out).any(), "NaN detected in golden output"
    assert not torch.isinf(out).any(), "Inf detected in golden output"

    # sanity: reasonable value range
    out_max = out.abs().max().item()
    assert out_max < 100.0, f"golden output max too large: {out_max}"

    logging.info(f"[GOLDEN_PASS] shape={list(out.shape)}, dtype={out.dtype}, max_abs={out_max:.4f}")


def test_qwen3_1_7b_rms_norm_rope():
    test_cases = load_test_cases()
    cases = test_cases.get("test_cases", [])

    logging.info("\nTest cases from test_cases.json:")
    for case in cases:
        logging.info(f"  {case['id']} — {case.get('description', '')}")

    try:
        for case_data in cases:
            run_single_case(case_data)
        logging.info("\n" + "=" * 60)
        logging.info("All golden tests passed!")
        logging.info("=" * 60)
    except Exception as e:
        logging.info(f"\nError: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_qwen3_1_7b_rms_norm_rope()
