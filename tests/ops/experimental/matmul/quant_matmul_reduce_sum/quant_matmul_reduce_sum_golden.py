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
PyPTO quant_matmul_reduce_sum golden reference implementation.

Pure PyTorch implementation for accuracy validation.
No pypto imports — runs on CPU or NPU via standard torch ops.
"""

import torch


def quant_matmul_reduce_sum_golden(
    x1: torch.Tensor,
    x2: torch.Tensor,
    x1_scale: torch.Tensor,
    x2_scale: torch.Tensor,
) -> torch.Tensor:
    """Golden reference for quantized matmul + scale dequant + batch reduce sum.

    Computes:
        for each batch i:
            matmul_i = x1[i] @ x2[i]          # [m, k] @ [k, n] -> [m, n]
            scale_i  = x1_scale[i] * x2_scale  # [m, 1] * [1, n] -> [m, n]
            result  += matmul_i * scale_i
        output = result.to(bfloat16)

    Args:
        x1: INT8 input tensor of shape [batch, m, k]
        x2: INT8 input tensor of shape [batch, k, n]
        x1_scale: FP32 scale tensor of shape [batch, m]
        x2_scale: BF16/FP32 scale tensor of shape [n]

    Returns:
        torch.Tensor: BF16 output tensor of shape [m, n]
    """
    batch, m, k = x1.shape
    _, _, n = x2.shape

    result = torch.zeros((m, n), dtype=torch.float32)

    for i in range(batch):
        matmul_result = torch.matmul(x1[i].float(), x2[i].float())
        scale_broadcast = x1_scale[i].unsqueeze(1).float() * x2_scale.unsqueeze(0).float()
        result += matmul_result * scale_broadcast

    return result.to(torch.bfloat16)
