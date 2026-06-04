
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
"""
import os
import math
import logging
import torch
import torch_npu
import numpy as np
from numpy.testing import assert_allclose
import pypto


# 小值域阈值
small_value_thres_dict = {
        torch.float16: 2**-11,
        torch.bfloat16: 2**-8,
        torch.float32: 2**-14,
        torch.uint8: 2**-4, torch.float8_e4m3fn: 2**-4
    }


# 小值域error指标
small_value_error_thres_dict = {
        torch.float16: 2**-16,
        torch.bfloat16: 2**-16,
        torch.float32: 2**-30,
        torch.uint8: 2**-6, torch.float8_e4m3fn: 2**-6
    }


def get_split_index(golden_data, dtype):
    thres = small_value_thres_dict[dtype]
    large_mask = torch.abs(golden_data) >= thres
    small_mask = torch.abs(golden_data) < thres
    return large_mask, small_mask, thres


def compute_matrix_small_value(input_data, golden_data, dtype, small_mask):
    if not torch.any(small_mask):
        return 0
    thres = small_value_error_thres_dict[dtype]

    error_count = torch.sum(torch.abs(input_data[small_mask] - golden_data[small_mask]) > thres).item()
    return error_count


def compute_matrix_large_value(input_data, golden_data, large_mask):
    if not torch.any(large_mask):
        return 0, 0, 0

    input_large = input_data[large_mask]
    golden_large = golden_data[large_mask]

    abs_diff = torch.abs(input_large - golden_large)
    relative_error = abs_diff / (torch.abs(golden_large) + 1e-7)

    mare = torch.max(relative_error).item()
    mere = torch.mean(relative_error).item()
    rmse = torch.sqrt(torch.mean((input_large - golden_large) ** 2)).item()

    return mare, mere, rmse


def compute_re_matrix(input_value, bm_value, small_value_thres):
    if math.isinf(bm_value) or math.isnan(bm_value):
        return 1
    if math.isinf(input_value) or math.isnan(input_value):
        return 1000
    return input_value / max(bm_value, small_value_thres)


def compute_re_triplet_matrix(npu_matrix, golden_matrix, small_value_thres):
    mare_npu, mere_npu, rmse_npu = npu_matrix
    mare_bm, mere_bm, rmse_bm = golden_matrix
    mare_matrix = compute_re_matrix(mare_npu, mare_bm, small_value_thres)
    mere_matrix = compute_re_matrix(mere_npu, mere_bm, small_value_thres)
    rmse_matrix = compute_re_matrix(rmse_npu, rmse_bm, small_value_thres)
    return mare_matrix, mere_matrix, rmse_matrix


# 昇腾算子精度标准2.1
# 精度等级 L0 thres: mare <= 10, mere <= 2, rmse <= 2 常规算子
# 精度等级 L1 thres: mare <= 5, mere <= 1.5, rmse <= 1.5 重要算子
# 精度等级 L2 thres: mare <= 2, mere <= 1.2, rmse <= 1.2 关键算子
def precision_compare_triple(npu_data, bm_data, golden_data, thres=(2, 1.2, 1.2)):
    dtype = npu_data.dtype
    if dtype in ["int8", "int32"]:
        raise NotImplementedError("precision compare triplet only support float")

    if dtype == torch.uint8:
        npu_data = torch_npu.npu_dtype_cast(npu_data, torch.float32, input_dtype=torch_npu.hifloat8)
        bm_data = torch_npu.npu_dtype_cast(bm_data, torch.float32, input_dtype=torch_npu.hifloat8)
        golden_data = torch_npu.npu_dtype_cast(golden_data, torch.float32, input_dtype=torch_npu.hifloat8)
    else:
        npu_data = npu_data.to(torch.float32)
        bm_data = bm_data.to(torch.float32)
        golden_data = golden_data.to(torch.float32)

    npu_data = npu_data.cpu()
    bm_data = bm_data.cpu()
    golden_data = golden_data.cpu()

    large_value_idx, small_value_idx, small_value_thres = get_split_index(golden_data, dtype)

    # 小值域场景
    npu_error_count = compute_matrix_small_value(npu_data, golden_data, dtype, small_value_idx)
    bm_error_count = compute_matrix_small_value(bm_data, golden_data, dtype, small_value_idx)
    small_value_matrix = npu_error_count / max(bm_error_count, 1)

    # 大值域场景
    mare_npu, mere_npu, rmse_npu = compute_matrix_large_value(npu_data, golden_data, large_value_idx)
    mare_bm, mere_bm, rmse_bm = compute_matrix_large_value(bm_data, golden_data, large_value_idx)
    mare_matrix, mere_matrix, rmse_matrix = compute_re_triplet_matrix(
        [mare_npu, mere_npu, rmse_npu], [mare_bm, mere_bm, rmse_bm], small_value_thres)

    is_mare_acceptable = mare_matrix <= thres[0]
    is_mere_acceptable = mere_matrix <= thres[1]
    is_rmse_acceptable = rmse_matrix <= thres[2]

    if small_value_matrix <= 2 and is_mare_acceptable and is_mere_acceptable and is_rmse_acceptable:
        result = "PASS"
    else:
        result = "FAILED"

    return result, mare_matrix, mere_matrix, rmse_matrix, small_value_matrix
