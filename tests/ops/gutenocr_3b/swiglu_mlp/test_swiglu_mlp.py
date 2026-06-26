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
SwiGLU MLP Golden 自洽性测试
验证 golden 参考实现在多个 shape/dtype 下输出合法。
"""

import os
import sys
from pathlib import Path

import json
import logging
import pytest
import torch

_CUR = Path(__file__).resolve().parent
sys.path.insert(0, str(_CUR))

from swiglu_mlp_golden import swiglu_mlp_golden


def load_test_cases():
    json_path = os.path.join(os.path.dirname(__file__), "swiglu_mlp_test_cases.json")
    if not os.path.exists(json_path):
        raise RuntimeError(f"Test cases file not found: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
        return data.get("test_cases", [])


def run_single_case(case_data):
    case_id = case_data["id"]
    description = case_data.get("description", "")

    logging.info("\n" + "=" * 60)
    logging.info(f"Test: {case_id} — {description}")
    logging.info("=" * 60)

    torch.manual_seed(case_data.get("seed", 42))
    device = "cpu"

    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}

    inputs = case_data["input"]
    dtype = dtype_map[inputs["x"]["dtype"]]

    x = torch.randn(inputs["x"]["shape"], dtype=dtype, device=device)
    gate_weight = torch.randn(inputs["gate_weight"]["shape"], dtype=dtype, device=device)
    gate_bias = torch.randn(inputs["gate_bias"]["shape"], dtype=dtype, device=device)
    up_weight = torch.randn(inputs["up_weight"]["shape"], dtype=dtype, device=device)
    up_bias = torch.randn(inputs["up_bias"]["shape"], dtype=dtype, device=device)
    down_weight = torch.randn(inputs["down_weight"]["shape"], dtype=dtype, device=device)
    down_bias = torch.randn(inputs["down_bias"]["shape"], dtype=dtype, device=device)

    output = swiglu_mlp_golden(x, gate_weight, gate_bias, up_weight, up_bias,
                               down_weight, down_bias)

    # shape/dtype check
    expected_shape = torch.Size(case_data["output"]["output"]["shape"])
    expected_dtype = dtype_map[case_data["output"]["output"]["dtype"]]
    assert output.shape == expected_shape, f"shape mismatch: {output.shape} vs {expected_shape}"
    assert output.dtype == expected_dtype, f"dtype mismatch: {output.dtype} vs {expected_dtype}"

    # sanity: no NaN / Inf
    assert not torch.isnan(output).any(), "NaN detected in golden output"
    assert not torch.isinf(output).any(), "Inf detected in golden output"

    # sanity: reasonable value range
    out_max = output.abs().max().item()
    assert 0.01 < out_max < 1e6, f"golden output max out of range: {out_max}"

    logging.info(f"[GOLDEN_PASS] shape={list(output.shape)}, dtype={output.dtype}, max_abs={out_max:.4f}")


@pytest.mark.parametrize("case_data", load_test_cases(), ids=lambda c: c["id"])
def test_swiglu_mlp(case_data):
    run_single_case(case_data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__, "-v", "-s"])
