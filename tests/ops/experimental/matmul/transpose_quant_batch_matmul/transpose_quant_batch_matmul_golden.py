#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from dataclasses import dataclass
from typing import List
import torch


@dataclass
class TqbmmGoldenInputs:
    """
    Input parameters for generating golden result in transpose quantized batch matrix multiplication.

    Attributes:
        x1: Input tensor of shape [M, B, K] in FP8 format
        x2: Input tensor of shape [B, K, N] or [B, N, K] in FP8 format
        x1Scale: Scale factors for x1 in MXFP8 format, shape [M, B, K//64, 2] in E8M0
        x2Scale: Scale factors for x2 in MXFP8 format, shape varies by permX2
        permX1: Permutation for x1 (e.g. [1, 0, 2])
        permX2: Permutation for x2 (e.g. [0, 1, 2] or [0, 2, 1])
        permY: Permutation for output (e.g. [1, 0, 2])
        dtype: Output dtype flag (1=FP16, 27=BF16)
    """
    x1: torch.Tensor
    x2: torch.Tensor
    x1Scale: torch.Tensor
    x2Scale: torch.Tensor
    permX1: List[int]
    permX2: List[int]
    permY: List[int]
    dtype: int


def compute_golden_result(
    x1: torch.Tensor,
    x2: torch.Tensor,
    x1Scale: torch.Tensor,
    x2Scale: torch.Tensor,
    permX1: List[int],
    permX2: List[int],
    permY: List[int],
    dtype: int
) -> torch.Tensor:
    """
    Compute golden (reference) result for transpose quantized batch matrix multiplication.

    This is a pure PyTorch implementation that:
    1. Converts FP8 inputs to FP32
    2. Converts E8M0 scales to FP32
    3. Applies permutation (permX1, permX2)
    4. Broadcasts MX scales (repeat_interleave with 32x factor)
    5. Dequantizes: data * scale
    6. Batch matmul
    7. Applies output permutation (permY) and dtype cast

    Args:
        x1: Input tensor [M, B, K] in FP8
        x2: Input tensor [B, K, N] or [B, N, K] in FP8
        x1Scale: Scale tensor [M, B, K//64, 2] in E8M0
        x2Scale: Scale tensor — shape depends on permX2
        permX1: Permutation for x1
        permX2: Permutation for x2
        permY: Permutation for output
        dtype: Output dtype flag (1=FP16, 27=BF16)

    Returns:
        torch.Tensor: Golden output after permutation and dtype cast
    """
    # Step 1: FP8 -> FP32
    x1_fp32 = x1.float()
    x2_fp32 = x2.float()

    # Step 2: E8M0 Scale -> FP32
    x1_scale_fp32 = x1Scale.float()
    x2_scale_fp32 = x2Scale.float()

    # Step 3: x1 permute [M,B,K] -> [B,M,K]
    x1_perm = x1_fp32.permute(permX1)

    # Step 4: x2 permute
    x2_perm = x2_fp32.permute(permX2)

    # Step 5: Scale reshape + broadcast
    M, B, K = x1.shape
    N = x2.shape[-1] if permX2 == [0, 1, 2] else x2.shape[1]

    x1_scale_list = []
    for b_idx in range(B):
        x1_scale_batch = x1_scale_fp32[:, b_idx, :, :]
        x1_scale_flat = x1_scale_batch.reshape(M, K // 64 * 2)
        x1_scale_broadcast_batch = torch.repeat_interleave(x1_scale_flat, repeats=32, dim=-1)
        x1_scale_list.append(x1_scale_broadcast_batch)
    x1_scale_broadcast = torch.stack(x1_scale_list, dim=0)

    x2_scale_list = []
    for b_idx in range(B):
        if permX2 == [0, 1, 2]:
            x2_scale_batch = x2_scale_fp32[b_idx, :, :, :]
            x2_scale_swap = x2_scale_batch.swapaxes(-1, -2)
            x2_scale_flat = x2_scale_swap.reshape(K // 64 * 2, N)
            x2_scale_broadcast_batch = torch.repeat_interleave(x2_scale_flat, repeats=32, dim=-2)
        else:
            x2_scale_batch = x2_scale_fp32[b_idx, :, :, :]
            x2_scale_flat = x2_scale_batch.reshape(N, K // 64 * 2)
            x2_scale_swap = x2_scale_flat.swapaxes(-1, -2)
            x2_scale_broadcast_batch = torch.repeat_interleave(x2_scale_swap, repeats=32, dim=-2)
        x2_scale_list.append(x2_scale_broadcast_batch)
    x2_scale_broadcast = torch.stack(x2_scale_list, dim=0)

    # Step 6-7: Dequantize + matmul
    x1_dequant = x1_perm * x1_scale_broadcast
    x2_dequant = x2_perm * x2_scale_broadcast
    result = torch.matmul(x1_dequant, x2_dequant)

    # Step 8-9: Output permute + dtype cast
    output_perm = result.permute(permY)
    if dtype == 1:
        output = output_perm.half()
    elif dtype == 27:
        output = output_perm.bfloat16()
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return output


def gen_golden(inputs: TqbmmGoldenInputs) -> torch.Tensor:
    """
    Generate golden (reference) output for transpose quantized batch matrix multiplication.

    Args:
        inputs: Input parameters including tensors, scales, permutations and dtype flag

    Returns:
        torch.Tensor: Golden output tensor after permutation and dtype cast
    """
    return compute_golden_result(
        inputs.x1, inputs.x2, inputs.x1Scale, inputs.x2Scale,
        inputs.permX1, inputs.permX2, inputs.permY, inputs.dtype
    )