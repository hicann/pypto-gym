#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared Precision Standard 2.1 comparison helpers for Engram tests."""

import math

import torch


TRIPLE_THRESHOLDS = (2.0, 1.2, 1.2)  # MARE, MERE, RMSE ratio thresholds

SMALL_VALUE_THRES = {
    torch.float16: 2 ** -11,
    torch.bfloat16: 2 ** -8,
    torch.float32: 2 ** -14,
    torch.uint8: 2 ** -4,
    torch.float8_e4m3fn: 2 ** -4,
}

SMALL_VALUE_ERROR_THRES = {
    torch.float16: 2 ** -16,
    torch.bfloat16: 2 ** -16,
    torch.float32: 2 ** -30,
    torch.uint8: 2 ** -6,
    torch.float8_e4m3fn: 2 ** -6,
}


def _get_split_index(golden_data, dtype):
    thres = SMALL_VALUE_THRES[dtype]
    large_mask = torch.abs(golden_data) >= thres
    small_mask = torch.abs(golden_data) < thres
    return large_mask, small_mask, thres


def _compute_small_value(input_data, golden_data, dtype, small_mask):
    if not torch.any(small_mask):
        return 0
    thres = SMALL_VALUE_ERROR_THRES[dtype]
    return torch.sum(torch.abs(input_data[small_mask] - golden_data[small_mask]) > thres).item()


def _compute_large_value(input_data, golden_data, large_mask):
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
    if math.isinf(benchmark_value) or math.isnan(benchmark_value):
        return 1.0
    if math.isinf(input_value) or math.isnan(input_value):
        return 1000.0
    return input_value / max(benchmark_value, small_value_thres)


def _compare(npu_data, benchmark_data, golden_data, thresholds=TRIPLE_THRESHOLDS):
    """Compare one tensor using the Precision Standard 2.1 three-way test."""
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
