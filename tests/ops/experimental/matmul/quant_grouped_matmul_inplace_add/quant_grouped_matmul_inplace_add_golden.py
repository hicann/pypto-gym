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

from dataclasses import dataclass
import math
import torch


@dataclass
class GmmGoldenInputs:
    """
    Input parameters for generating golden result in grouped matrix multiplication.

    Attributes:
        a: Input tensor of shape [K, M] or [M, K] depending on a_trans flag
        b: Weight tensor of shape [K, N] or [N, K] depending on b_trans flag
        scaled_a: Scale factors for input tensor in MXFP8 format
        scaled_b: Scale factors for weight tensor in MXFP8 format
        y: Output tensor of shape [num_groups, M, N] (also serves as initial value for inplace add)
        num_groups: Number of groups to split K-axis
        a_trans: Whether input tensor is transposed (default: True)
        b_trans: Whether weight tensor is transposed (default: False)
    """
    a: torch.Tensor
    b: torch.Tensor
    scaled_a: torch.Tensor
    scaled_b: torch.Tensor
    y: torch.Tensor
    num_groups: int
    a_trans: bool = True
    b_trans: bool = False


@dataclass
class GoldenComputeInputs:
    """
    Input parameters for computing golden result in matrix multiplication.

    Attributes:
        x: Input tensor of shape [K, M] or [M, K] depending on a_trans flag
        weight: Weight tensor of shape [K, N] or [N, K] depending on b_trans flag
        scaled_x: Scale factors for input tensor
        scaled_weight: Scale factors for weight tensor
        a_trans: Whether input tensor is transposed
        b_trans: Whether weight tensor is transposed
    """
    x: torch.Tensor
    weight: torch.Tensor
    scaled_x: torch.Tensor
    scaled_weight: torch.Tensor
    a_trans: bool
    b_trans: bool


def compute_golden_result(inputs: GoldenComputeInputs) -> torch.Tensor:
    """
    Compute golden (reference) result for a single group's matrix multiplication.

    Args:
        inputs: Input parameters including tensors, scales, and transposition flags

    Returns:
        torch.Tensor: Golden output tensor after applying scaling and matrix multiplication
    """
    x = inputs.x
    weight = inputs.weight
    scaled_x_golden = inputs.scaled_x
    scaled_weight_golden = inputs.scaled_weight
    a_trans = inputs.a_trans
    b_trans = inputs.b_trans

    # Handle input tensor transposition
    if a_trans:
        x = torch.swapaxes(x, -1, -2)
        scaled_x_golden = torch.swapaxes(scaled_x_golden, -1, -2)
        if len(scaled_x_golden.shape) == 3:
            scaled_x_golden = scaled_x_golden.reshape(
                scaled_x_golden.shape[0] * scaled_x_golden.shape[1], scaled_x_golden.shape[2]
            )
        scaled_x_golden = torch.swapaxes(scaled_x_golden, -1, -2)
    else:
        if len(scaled_x_golden.shape) == 3:
            scaled_x_golden = scaled_x_golden.reshape(
                scaled_x_golden.shape[0], scaled_x_golden.shape[1] * scaled_x_golden.shape[2]
            )

    # Handle weight tensor transposition
    if b_trans:
        weight = torch.swapaxes(weight, -1, -2)
        if len(scaled_weight_golden.shape) == 3:
            scaled_weight_golden = scaled_weight_golden.reshape(
                scaled_weight_golden.shape[0] * scaled_weight_golden.shape[1],
                scaled_weight_golden.shape[2]
            )
        scaled_weight_golden = torch.swapaxes(scaled_weight_golden, -1, -2)
    else:
        scaled_weight_golden = torch.swapaxes(scaled_weight_golden, -1, -2)
        if len(scaled_weight_golden.shape) == 3:
            scaled_weight_golden = scaled_weight_golden.reshape(
                scaled_weight_golden.shape[0] * scaled_weight_golden.shape[1],
                scaled_weight_golden.shape[2]
            )

    # Adjust scales for K dimension alignment
    k_dim = x.shape[-1]
    if math.ceil(k_dim / 32) % 2 != 0:
        scaled_x_golden = scaled_x_golden[:, :-1]
        scaled_weight_golden = scaled_weight_golden[:-1, :]

    # Broadcast scale factors to match tensor dimensions
    scaled_x_golden_broadcast = torch.repeat_interleave(scaled_x_golden, repeats=32, dim=-1)
    scaled_weight_golden_broadcast = torch.repeat_interleave(scaled_weight_golden, repeats=32, dim=-2)

    # Calculate padding lengths for alignment
    x1_dims = len(x.shape)
    x2_dims = len(weight.shape)
    x1_pad_len = scaled_x_golden_broadcast.shape[-1] - x.shape[-1]
    x2_pad_len = scaled_weight_golden_broadcast.shape[-2] - weight.shape[-2]

    # Pad input tensor to match broadcasted scale dimensions
    x1_pad = [0, x1_pad_len]
    for _ in range(x1_dims - 1):
        x1_pad += [0, 0]
    x1_golden = torch.nn.functional.pad(x, x1_pad, mode='constant', value=0)

    # Pad weight tensor to match broadcasted scale dimensions
    weight_pad = [0, 0]
    weight_pad += [0, x2_pad_len]
    for _ in range(x2_dims - 2):
        weight_pad += [0, 0]
    weight_golden = torch.nn.functional.pad(weight, weight_pad, mode='constant', value=0)

    # Apply scaling factors in FP32
    x_fp32 = x.to(torch.float32)
    scaled_x_golden_broadcast_fp32 = scaled_x_golden_broadcast.to(torch.float32)
    x1_golden = x_fp32 * scaled_x_golden_broadcast_fp32

    weight_fp32 = weight.to(torch.float32)
    scaled_weight_golden_broadcast_fp32 = scaled_weight_golden_broadcast.to(torch.float32)
    weight_golden = weight_fp32 * scaled_weight_golden_broadcast_fp32

    # Compute matrix multiplication
    golden = torch.matmul(x1_golden, weight_golden)

    return golden


def gen_golden(inputs: GmmGoldenInputs) -> torch.Tensor:
    """
    Generate golden (reference) output for grouped matrix multiplication with inplace add.

    This function splits K-axis uniformly by num_groups, computes matrix multiplication
    for each group, and accumulates results inplace to the initial output tensor.

    Args:
        inputs: Input parameters including tensors, scales, num_groups and transposition flags

    Returns:
        torch.Tensor: Golden output tensor of shape [num_groups, M, N] after inplace accumulation
    """
    a = inputs.a
    b = inputs.b
    scaled_a = inputs.scaled_a
    scaled_b = inputs.scaled_b
    y = inputs.y
    num_groups = inputs.num_groups
    a_trans = inputs.a_trans
    b_trans = inputs.b_trans

    # Calculate K dimension and block size for each group
    k = a.shape[0] if a_trans else a.shape[1]
    k_block = k // num_groups
    golden_result = y.clone()

    # Process each group separately
    for i in range(num_groups):
        begin = i * k_block
        end = (i + 1) * k_block
        scale_offset = begin // 64 + i  # MX quantization format: scale offset = begin/64 + i
        scale_length = k_block // 64

        # Extract input tensor for current group based on transposition
        if a_trans:
            x = a[begin:end, :]
            scaled_x_golden = scaled_a[scale_offset : scale_offset + scale_length, :, :]
        else:
            x = a[:, begin:end]
            scaled_x_golden = scaled_a[:, scale_offset : scale_offset + scale_length, :]

        # Extract weight tensor for current group based on transposition
        if b_trans:
            weight = b[:, begin:end]  # b is [N, K], split K-axis = split second dimension
            scaled_weight_golden = scaled_b[:, scale_offset : scale_offset + scale_length, :]
        else:
            weight = b[begin:end, :]  # b is [K, N], split K-axis = split first dimension
            scaled_weight_golden = scaled_b[scale_offset : scale_offset + scale_length, :, :]

        # Compute golden result for current group
        golden_temp = compute_golden_result(
            GoldenComputeInputs(
                x=x,
                weight=weight,
                scaled_x=scaled_x_golden,
                scaled_weight=scaled_weight_golden,
                a_trans=a_trans,
                b_trans=b_trans,
            )
        )

        # Accumulate result inplace
        golden_result[i] = golden_result[i] + golden_temp

    return golden_result