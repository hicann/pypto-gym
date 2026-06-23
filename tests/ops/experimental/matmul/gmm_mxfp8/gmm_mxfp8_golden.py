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

#
# PyPTO gmm_mxfp8 golden reference (pure torch).
# 类型 `TransposeConfig` / `GroupedMatmulInputs` 见 gmm_mxfp8_impl.py。

import math
from dataclasses import dataclass

import torch

from experimental.matmul.gmm_mxfp8.gmm_mxfp8_impl import (
    TransposeConfig,
)


@dataclass(frozen=True)
class GmmGoldenInputs:
    """Input parameters for generating golden result in grouped matrix multiplication.

    Attributes:
        a: Input tensor of shape [M, K]
        b: Weight tensor of shape [num_groups, K, N] or [num_groups, N, K]
        scaled_a: Scale factors for input tensor
        scaled_b: Scale factors for weight tensor
        group_list: List of group sizes for each weight group
        transpose: Transpose flags for left and right matrices
    """

    a: torch.Tensor
    b: torch.Tensor
    scaled_a: torch.Tensor
    scaled_b: torch.Tensor
    group_list: list[int]
    transpose: TransposeConfig


@dataclass(frozen=True)
class GoldenComputeInputs:
    """Input parameters for computing golden result in matrix multiplication.

    Attributes:
        x: Input tensor of shape [M, K] or [K, M] if transposed
        weight: Weight tensor of shape [K, N] or [N, K] if transposed
        scaled_x: Scale factors for input tensor
        scaled_weight: Scale factors for weight tensor
        transpose: Transpose flags for left and right matrices
    """

    x: torch.Tensor
    weight: torch.Tensor
    scaled_x: torch.Tensor
    scaled_weight: torch.Tensor
    transpose: TransposeConfig


def _reshape_input_scale(scaled_x: torch.Tensor, a_trans: bool) -> torch.Tensor:
    """Reshape and transpose input scale factors based on a_trans flag."""
    if a_trans:
        scaled_x = torch.swapaxes(scaled_x, -1, -2)
        if len(scaled_x.shape) == 3:
            scaled_x = scaled_x.reshape(
                scaled_x.shape[0] * scaled_x.shape[1],
                scaled_x.shape[2],
            )
        scaled_x = torch.swapaxes(scaled_x, -1, -2)
    else:
        if len(scaled_x.shape) == 3:
            scaled_x = scaled_x.reshape(
                scaled_x.shape[0],
                scaled_x.shape[1] * scaled_x.shape[2],
            )
    return scaled_x


def _reshape_weight_scale(scaled_w: torch.Tensor, b_trans: bool) -> torch.Tensor:
    """Reshape and transpose weight scale factors based on b_trans flag."""
    if b_trans:
        if len(scaled_w.shape) == 3:
            scaled_w = scaled_w.reshape(
                scaled_w.shape[0] * scaled_w.shape[1],
                scaled_w.shape[2],
            )
        scaled_w = torch.swapaxes(scaled_w, -1, -2)
    else:
        scaled_w = torch.swapaxes(scaled_w, -1, -2)
        if len(scaled_w.shape) == 3:
            scaled_w = scaled_w.reshape(
                scaled_w.shape[0] * scaled_w.shape[1],
                scaled_w.shape[2],
            )
    return scaled_w


def _apply_dequant_and_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    scaled_x: torch.Tensor,
    scaled_w: torch.Tensor,
) -> torch.Tensor:
    """Broadcast scale factors, dequantize, and compute matrix multiplication."""
    k_dim = x.shape[-1]
    if math.ceil(k_dim / 32) % 2 != 0:
        scaled_x = scaled_x[:, :-1]
        scaled_w = scaled_w[:-1, :]

    # Broadcast scale factors to element-level
    scaled_x_bc = torch.repeat_interleave(scaled_x, repeats=32, dim=-1)
    scaled_w_bc = torch.repeat_interleave(scaled_w, repeats=32, dim=-2)

    # Calculate padding lengths
    x1_pad_len = scaled_x_bc.shape[-1] - x.shape[-1]
    x2_pad_len = scaled_w_bc.shape[-2] - weight.shape[-2]

    # Pad input tensor to match broadcast scale length
    x1_pad = [0, x1_pad_len]
    for _ in range(len(x.shape) - 1):
        x1_pad += [0, 0]
    x_padded = torch.nn.functional.pad(x, x1_pad, mode='constant', value=0)

    # Pad weight tensor to match broadcast scale length
    weight_pad = [0, 0, 0, x2_pad_len]
    for _ in range(len(weight.shape) - 2):
        weight_pad += [0, 0]
    weight_padded = torch.nn.functional.pad(
        weight, weight_pad, mode='constant', value=0
    )

    # Apply scaling factors (dequantize)
    x_dequant = x.to(torch.float32) * scaled_x_bc.to(torch.float32)
    weight_dequant = weight.to(torch.float32) * scaled_w_bc.to(torch.float32)

    # Compute matrix multiplication
    return torch.matmul(x_dequant, weight_dequant)


def compute_golden_result(inputs: GoldenComputeInputs) -> torch.Tensor:
    """Compute golden (reference) result for a single group's matrix multiplication.

    Args:
        inputs: Input parameters including tensors and transposition flags

    Returns:
        torch.Tensor: Golden output tensor
    """
    x = inputs.x
    weight = inputs.weight
    a_trans = inputs.transpose.a_trans
    b_trans = inputs.transpose.b_trans

    # Handle input transposition
    if a_trans:
        x = torch.swapaxes(x, -1, -2)

    # Handle weight transposition
    if b_trans:
        weight = torch.swapaxes(weight, -1, -2)

    # Reshape scale factors
    scaled_x = _reshape_input_scale(inputs.scaled_x, a_trans)
    scaled_w = _reshape_weight_scale(inputs.scaled_weight, b_trans)

    return _apply_dequant_and_matmul(x, weight, scaled_x, scaled_w)


def gen_golden(inputs: GmmGoldenInputs) -> torch.Tensor:
    """Generate golden output for grouped matrix multiplication using PyTorch.

    Args:
        inputs: Input parameters including tensors, scales, and transposition flags

    Returns:
        torch.Tensor: Golden output tensor of shape [M, N]
    """
    a, b = inputs.a, inputs.b
    scaled_a, scaled_b = inputs.scaled_a, inputs.scaled_b
    group_list = inputs.group_list
    a_trans, b_trans = inputs.transpose.a_trans, inputs.transpose.b_trans

    result = []
    begin, end = 0, 0
    for i in range(b.shape[0]):
        if group_list[i] <= 0:
            continue
        begin = end
        end = end + group_list[i]

        x = a[:, begin:end] if a_trans else a[begin:end, :]
        sx = scaled_a[:, begin:end, :] if a_trans else scaled_a[begin:end, :, :]
        golden_temp = compute_golden_result(GoldenComputeInputs(
            x=x, weight=b[i], scaled_x=sx, scaled_weight=scaled_b[i],
            transpose=TransposeConfig(a_trans=a_trans, b_trans=b_trans),
        ))
        result.append(golden_temp)

    return torch.cat(result, dim=0)
