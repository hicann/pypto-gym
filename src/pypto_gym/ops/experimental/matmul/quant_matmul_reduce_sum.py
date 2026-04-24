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
Quantized Matmul Reduce Sum Implementation

This module implements quantized matrix multiplication with reduce sum operation using PyPTO.
Supports INT8 quantized inputs with scale factors for dequantization.
"""

from dataclasses import dataclass

import pypto
import torch
import torch_npu
from numpy.testing import assert_allclose
from torch.nn import functional as F


def trans_nd_to_fractal_nz(data: torch.Tensor, keep_m_dim=False):
    """
    Transform ND tensor to fractal NZ format.

    Args:
        data: Input tensor of shape [..., M, N]
        keep_m_dim: Whether to keep M dimension unchanged (default: False)

    Returns:
        torch.Tensor: Transformed tensor in fractal NZ format
    """
    def _gen_axes_for_transpose(offset, base):
        return [x for x in range(offset)] + [x + offset for x in base]

    def _ceil_div(a, b):
        return (a + b - 1) // b

    ori_shape = data.shape
    m_ori, n_ori = ori_shape[-2:]
    batch_ori = ori_shape[:-2]
    batch_num = len(batch_ori)
    m0 = 16
    n0 = 32 // data.dtype.itemsize
    if data.dtype == torch.int32:
        n0 = 16
    m1, n1 = _ceil_div(m_ori, m0), _ceil_div(n_ori, n0)
    padding_m = m1 * m0 - m_ori
    padding_n = n1 * n0 - n_ori
    if not keep_m_dim:
        pad_list = [0, padding_n, 0, padding_m] + [0, 0] * batch_num
        data = F.pad(data, pad_list, "constant")
        array_trans = _gen_axes_for_transpose(len(data.shape) - 2, [2, 0, 1, 3])
        data = data.reshape(batch_ori + (m1, m0, n1, n0)).permute(*array_trans).contiguous()
    else:
        pad_list = [0, padding_n, 0, 0] + [0, 0] * batch_num
        data = F.pad(data, pad_list, "constant")
        array_trans = _gen_axes_for_transpose(len(data.shape) - 2, [1, 0, 2])
        data = data.reshape(batch_ori + (m_ori, n1, n0)).permute(*array_trans).contiguous()
    return data


@dataclass
class QuantMatmulReduceSumConfig:
    """
    Configuration parameters for quantized matmul reduce sum operation.

    Attributes:
        ori: Original shape [batch, m, k, n]
        m_tile_shape: Tile shape for M dimension in cube operation
        k_tile_shape: Tile shape for K dimension in cube operation
        n_tile_shape: Tile shape for N dimension in cube operation
        in_dtype: Input data type
        out_dtype: Output data type
        x2_format_nz: Whether x2 uses NZ format (default: True)
        vec_tile_shapes: Tile shapes for vector operations (default: [1, 256, 256])
        description: Description of the test case
    """
    ori: list
    m_tile_shape: list
    k_tile_shape: list
    n_tile_shape: list
    in_dtype: pypto.DataType
    out_dtype: pypto.DataType
    x2_format_nz: bool = True
    vec_tile_shapes: list = None
    description: str = ""

    def __post_init__(self):
        if self.vec_tile_shapes is None:
            self.vec_tile_shapes = [1, 256, 256]


def quant_matmul_reduce_sum_pypto(config: QuantMatmulReduceSumConfig):
    """
    Create quantized matmul reduce sum kernel using PyPTO.

    Args:
        config: Configuration parameters for the operation

    Returns:
        function: JIT-compiled kernel function
    """
    batch, m, k, n = config.ori
    x1_shape = [batch, m, k]
    x2_shape = [batch, k, n]
    x1_scale_shape = [batch, m]
    x2_scale_shape = [n]
    out_shape = [m, n]
    x2_format = pypto.TileOpFormat.TILEOP_NZ if config.x2_format_nz else pypto.TileOpFormat.TILEOP_ND

    @pypto.frontend.jit()
    def quant_matmul_reduce_sum_impl(
        x1: pypto.Tensor(x1_shape, pypto.DT_INT8),
        x2: pypto.Tensor(x2_shape, pypto.DT_INT8, format=x2_format),
        x1_scale: pypto.Tensor(x1_scale_shape, pypto.DT_FP32),
        x2_scale: pypto.Tensor(x2_scale_shape, pypto.DT_BF16)
    ) -> pypto.Tensor(out_shape, pypto.DT_BF16):
        pypto.set_cube_tile_shapes(config.m_tile_shape, config.k_tile_shape, config.n_tile_shape)
        pypto.set_vec_tile_shapes(*config.vec_tile_shapes)

        if config.x2_format_nz:
            pypto.set_matrix_size([m, k, n])

        # Matmul computation and convert to FP32
        matmul_result = pypto.matmul(x1, x2, pypto.DT_INT32)
        matmul_result_fp32 = pypto.cast(matmul_result, pypto.DT_FP32)

        # Convert x2_scale to FP32 and broadcast
        x2_scale_fp32 = pypto.cast(x2_scale, pypto.DT_FP32)
        x2_scale_2d = pypto.unsqueeze(x2_scale_fp32, 0)
        x2_scale_broadcast = pypto.expand_clone(x2_scale_2d, [m, n])

        # Broadcast x1_scale
        x1_scale_2d = pypto.unsqueeze(x1_scale, 2)
        x1_scale_broadcast = pypto.expand_clone(x1_scale_2d, [batch, m, n])

        # Compute scale multiplication
        scale_mul = pypto.mul(x1_scale_broadcast, x2_scale_broadcast)

        # Fused multiply and reduce sum
        out = pypto.mul(matmul_result_fp32, scale_mul)
        out_fp32 = pypto.sum(out, 0)
        out_bf16 = pypto.cast(out_fp32, pypto.DT_BF16)

        return out_bf16

    return quant_matmul_reduce_sum_impl


def golden_quant_matmul_reduce_sum(x1, x2, x1_scale, x2_scale):
    """
    Compute golden (reference) output using PyTorch.

    Args:
        x1: First input tensor of shape [batch, m, k]
        x2: Second input tensor of shape [batch, k, n]
        x1_scale: Scale factor for x1 of shape [batch, m]
        x2_scale: Scale factor for x2 of shape [n]

    Returns:
        torch.Tensor: Output tensor of shape [m, n] in bfloat16
    """
    batch, m, k = x1.shape
    _, _, n = x2.shape

    result = torch.zeros((m, n), dtype=torch.float32)

    for i in range(batch):
        matmul_result = torch.matmul(x1[i].float(), x2[i].float())
        scale_broadcast = x1_scale[i].unsqueeze(1).float() * x2_scale.unsqueeze(0).float()
        result += matmul_result * scale_broadcast

    return result.to(torch.bfloat16)


def run_quant_matmul_reduce_sum_case(config: QuantMatmulReduceSumConfig):
    """
    Test the quantized matmul reduce sum implementation.

    This function runs a complete test for a given configuration:
    1. Generate test data
    2. Compute golden (reference) output using PyTorch
    3. Compute output using PyPTO
    4. Compare results

    Args:
        config: Configuration parameters for the test case
    """
    b, m, k, n = config.ori

    torch.manual_seed(42)
    x1 = torch.randint(-10, 10, (b, m, k), dtype=torch.int8).npu()
    x2_nd = torch.randint(-10, 10, (b, k, n), dtype=torch.int8).npu()
    x2_nz = trans_nd_to_fractal_nz(x2_nd).npu()
    x1_scale = torch.randn((b, m), dtype=torch.float32).uniform_(0.5, 1.5).npu()
    x2_scale = torch.randn((n,), dtype=torch.bfloat16).uniform_(0.5, 1.5).npu()

    # Select x2 input based on format configuration
    if config.x2_format_nz:
        x2_input = x2_nz
    else:
        x2_input = x2_nd

    pypto_out = quant_matmul_reduce_sum_pypto(config)(x1, x2_input, x1_scale, x2_scale)
    golden_out = golden_quant_matmul_reduce_sum(x1.cpu(), x2_nd.cpu(), x1_scale.cpu(), x2_scale.cpu())

    pypto_out_cpu = pypto_out.cpu().float()
    golden_out_cpu = golden_out.cpu().float()

    assert_allclose(pypto_out_cpu, golden_out_cpu, rtol=0.001, atol=0.001)


if __name__ == "__main__":
    run_quant_matmul_reduce_sum_case(
        QuantMatmulReduceSumConfig(
            [2, 128, 128, 128],
            [128, 128],
            [128, 128],
            [128, 128],
            pypto.DT_INT8,
            pypto.DT_BF16,
            True,
            [64, 64, 64]
        )
    )
