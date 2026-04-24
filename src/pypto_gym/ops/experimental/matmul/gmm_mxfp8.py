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

"""
Grouped Matrix Multiplication with MXFP8 Quantization

This module implements grouped matrix multiplication with MXFP8 quantization using PyPTO.
Supports grouped GEMM operations with different weight groups and MXFP8 quantization format.
"""

import math
from dataclasses import dataclass

import numpy as np
import pypto
import torch
import torch_npu
from numpy.testing import assert_allclose


@dataclass
class GmmGoldenInputs:
    """
    Input parameters for generating golden result in grouped matrix multiplication.

    Attributes:
        a: Input tensor of shape [M, K]
        b: Weight tensor of shape [num_groups, K, N] or [num_groups, N, K]
        scaled_a: Scale factors for input tensor
        scaled_b: Scale factors for weight tensor
        group_list: List of group sizes for each weight group
        a_trans: Whether input tensor is transposed
        b_trans: Whether weight tensor is transposed
    """
    a: torch.Tensor
    b: torch.Tensor
    scaled_a: torch.Tensor
    scaled_b: torch.Tensor
    group_list: list
    a_trans: bool
    b_trans: bool


@dataclass
class GmmMxfp8Inputs:
    """
    Input parameters for generating MXFP8 output.

    Attributes:
        a: Input tensor of shape [M, K]
        b: Weight tensor of shape [num_groups, K, N] or [num_groups, N, K]
        scaled_a: Scale factors for input tensor
        scaled_b: Scale factors for weight tensor
        group_list: List of group sizes for each weight group
        tile_config: Tile configuration for computation
    """
    a: torch.Tensor
    b: torch.Tensor
    scaled_a: torch.Tensor
    scaled_b: torch.Tensor
    group_list: list
    tile_config: 'ShapeConfig'


@dataclass
class GoldenComputeInputs:
    """
    Input parameters for computing golden result in matrix multiplication.

    Attributes:
        x: Input tensor of shape [M, K] or [K, M] if transposed
        weight: Weight tensor of shape [K, N] or [N, K] if transposed
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
        inputs: Input parameters including tensors and transposition flags

    Returns:
        torch.Tensor: Golden output tensor
    """
    x = inputs.x
    weight = inputs.weight
    scaled_x_golden = inputs.scaled_x
    scaled_weight_golden = inputs.scaled_weight
    a_trans = inputs.a_trans
    b_trans = inputs.b_trans

    # Handle input transposition
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

    # Handle weight transposition
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

    # Broadcast scale factors
    scaled_x_golden_broadcast = torch.repeat_interleave(scaled_x_golden, repeats=32, dim=-1)
    scaled_weight_golden_broadcast = torch.repeat_interleave(scaled_weight_golden, repeats=32, dim=-2)

    # Calculate padding lengths
    x1_dims = len(x.shape)
    x2_dims = len(weight.shape)
    x1_pad_len = scaled_x_golden_broadcast.shape[-1] - x.shape[-1]
    x2_pad_len = scaled_weight_golden_broadcast.shape[-2] - weight.shape[-2]

    # Pad input tensor
    x1_pad = [0, x1_pad_len]
    for _ in range(x1_dims - 1):
        x1_pad += [0, 0]
    x1_golden = torch.nn.functional.pad(x, x1_pad, mode='constant', value=0)

    # Pad weight tensor
    weight_pad = [0, 0]
    weight_pad += [0, x2_pad_len]
    for _ in range(x2_dims - 2):
        weight_pad += [0, 0]
    weight_golden = torch.nn.functional.pad(weight, weight_pad, mode='constant', value=0)

    # Apply scaling factors
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
    Generate golden (reference) output for grouped matrix multiplication using PyTorch.

    Args:
        inputs: Input parameters including tensors, scales, and transposition flags

    Returns:
        torch.Tensor: Golden output tensor of shape [M, N]
    """
    a = inputs.a
    b = inputs.b
    scaled_a = inputs.scaled_a
    scaled_b = inputs.scaled_b
    group_list = inputs.group_list
    a_trans = inputs.a_trans
    b_trans = inputs.b_trans

    round_num = b.shape[0]
    result = []
    begin = 0
    end = 0

    for i in range(round_num):
        if group_list[i] <= 0:
            continue
        begin = end
        end = end + group_list[i]

        # Extract input and weight for current group
        if a_trans:
            x = a[:, begin:end]
        else:
            x = a[begin:end, :]
        weight = b[i]

        if a_trans:
            scaled_x_golden = scaled_a[:, begin:end, :]
        else:
            scaled_x_golden = scaled_a[begin:end, :, :]
        scaled_weight_golden = scaled_b[i]

        # Compute golden result for this group
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
        result.append(golden_temp)

    # Concatenate results from all groups
    golden_result = torch.cat(result, dim=0)
    return golden_result


@dataclass
class ShapeConfig:
    """
    Configuration parameters for grouped matrix multiplication with MXFP8 quantization.

    Attributes:
        ori_shape: Original shape [M, K, N]
        group_list: List of group sizes for each weight group
        tile_size: Tile size for computation
        m_tile_shape: Tile shape for M dimension in cube operation
        k_tile_shape: Tile shape for K dimension in cube operation
        n_tile_shape: Tile shape for N dimension in cube operation
        vector_tile_shape: Tile shapes for vector operations
        a_trans: Whether input tensor is transposed (default: False)
        b_trans: Whether weight tensor is transposed (default: False)
        a_format_nz: Whether input uses NZ format (default: False)
        b_format_nz: Whether weight uses NZ format (default: False)
        c_format_nz: Whether output uses NZ format (default: False)
        description: Description of the test case
    """
    ori_shape: list
    group_list: list
    tile_size: int
    m_tile_shape: list
    k_tile_shape: list
    n_tile_shape: list
    vector_tile_shape: list
    a_trans: bool = False
    b_trans: bool = False
    a_format_nz: bool = False
    b_format_nz: bool = False
    c_format_nz: bool = False
    description: str = ""


@pypto.frontend.jit
def scaled_matmul_kernel(
    a: pypto.Tensor,
    b: pypto.Tensor,
    scaled_a: pypto.Tensor,
    scaled_b: pypto.Tensor,
    out: pypto.Tensor,
    group_list,
    tile_config
) -> None:
    """
    Scaled matrix multiplication kernel for grouped GEMM with MXFP8 quantization.

    This kernel performs grouped matrix multiplication where each group uses
    a different weight matrix from the weight tensor, with MXFP8 quantization.

    Args:
        a: Input tensor
        b: Weight tensor containing multiple weight groups
        scaled_a: Scale factors for input tensor
        scaled_b: Scale factors for weight tensor
        out: Output tensor
        group_list: List of group sizes
        tile_config: Tile configuration for computation
    """
    round_num = b.shape[0]
    n_size = b.shape[-1]
    begin = 0
    end = 0

    for i in range(round_num):
        begin = end
        end = end + group_list[i]

        # Extract input and weight for current group
        x = a[begin:end, :]
        weight = b[i]
        scaled_x = scaled_a[begin:end, :, :]

        # Set vector tile shapes for scale processing
        pypto.set_vec_tile_shapes(
            tile_config.vector_tile_shape[0],
            tile_config.vector_tile_shape[1],
            tile_config.vector_tile_shape[2],
            tile_config.vector_tile_shape[3]
        )
        scaled_weight = scaled_b[i]

        # Set cube tile shapes and perform scaled matrix multiplication
        pypto.set_cube_tile_shapes(
            tile_config.m_tile_shape,
            tile_config.k_tile_shape,
            tile_config.n_tile_shape
        )
        out[begin:end, :] = pypto.scaled_mm(x, weight, pypto.DT_FP32, scaled_x, scaled_weight)


def gen_mxfp8(inputs: GmmMxfp8Inputs) -> torch.Tensor:
    """
    Generate MXFP8 output using PyPTO scaled matrix multiplication.

    Args:
        inputs: Input parameters including tensors, scales, group list and tile config

    Returns:
        torch.Tensor: Output tensor of shape [M, N] in FP32
    """
    a = inputs.a
    b = inputs.b
    scaled_a = inputs.scaled_a
    scaled_b = inputs.scaled_b
    group_list = inputs.group_list
    tile_config = inputs.tile_config

    # Move tensors to NPU
    a = a.npu()
    b = b.npu()
    scaled_a = scaled_a.npu()
    scaled_b = scaled_b.npu()

    # Initialize output tensor
    out_shape = (a.shape[0], b.shape[-1])
    out = torch.zeros(out_shape, dtype=torch.float32).npu()

    # Prepare input and output tensors for PyPTO
    input_tensors = {
        a: [],
        b: [],
        scaled_a: [],
        scaled_b: [],
    }
    output_tensors = {
        out: [],
    }

    # Convert torch tensors to PyPTO tensors
    pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in input_tensors.items()]
    pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in output_tensors.items()]

    # Execute scaled matrix multiplication kernel
    scaled_matmul_kernel(*pto_inputs, *pto_outputs, group_list, tile_config)
    out = out.to(torch.float32)
    return out


def test_gmm_mxfp8(tile_config: ShapeConfig):
    """
    Test the grouped matrix multiplication with MXFP8 quantization.

    This function runs a complete test for a given configuration:
    1. Generate test data with MXFP8 format
    2. Compute golden (reference) output using PyTorch
    3. Compute output using PyPTO
    4. Compare results

    Args:
        tile_config: Configuration parameters for the test case
    """
    # Extract configuration parameters
    m = tile_config.ori_shape[0]
    k = tile_config.ori_shape[1]
    n = tile_config.ori_shape[2]
    a_trans = tile_config.a_trans
    b_trans = tile_config.b_trans
    group_list = tile_config.group_list

    # Generate input tensor in MXFP8 format
    a = torch.randn((m, k), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e4m3fn)
    scaled_a = torch.randn((m, k // 64, 2), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu)

    # Generate weight tensor in MXFP8 format
    if b_trans:
        b = torch.randn((len(group_list), n, k), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e4m3fn)
    else:
        b = torch.randn((len(group_list), k, n), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e4m3fn)

    # Generate scale factors for weight tensor
    if b_trans:
        scaled_b = torch.randn(
            (len(group_list), n, k // 64, 2), dtype=torch.float32
        ).uniform_(0, 1).to(torch.float8_e8m0fnu)
    else:
        scaled_b = torch.randn(
            (len(group_list), k // 64, n, 2), dtype=torch.float32
        ).uniform_(0, 1).to(torch.float8_e8m0fnu)

    # Compute golden and PyPTO results
    golden = gen_golden(GmmGoldenInputs(
        a=a,
        b=b,
        scaled_a=scaled_a,
        scaled_b=scaled_b,
        group_list=group_list,
        a_trans=a_trans,
        b_trans=b_trans,
    ))
    result = gen_mxfp8(GmmMxfp8Inputs(
        a=a,
        b=b,
        scaled_a=scaled_a,
        scaled_b=scaled_b,
        group_list=group_list,
        tile_config=tile_config,
    ))

    # Verify results
    assert_allclose(golden.cpu().numpy(), result.cpu().numpy(), rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_gmm_mxfp8(
        ShapeConfig(
            [16, 512, 7168],
            [7, 9],
            256,
            [9, 9],
            [256, 256],
            [256, 256],
            [1, 8, 256, 32],
            False,
            False,
            False,
            False,
            False
        )
    )
