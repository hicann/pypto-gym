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
Transpose Quantized Batch Matrix Multiplication with MXFP8 Quantization

This module implements batch matrix multiplication with transpose and MXFP8 quantization using PyPTO.
Supports multiple perm combinations (permX1, permX2, permY) and output dtype (FP16/BF16).
M-axis is dynamic — the same compiled kernel supports different M sizes at runtime.
"""

from dataclasses import dataclass
from typing import List
import pypto


@dataclass
class ShapeConfig:
    """
    Configuration parameters for transpose quantized batch matrix multiplication.

    Attributes:
        ori_shape: Original shape [M, K, N] — M is dynamic in kernel, only used for golden/wrapper
        batch_size: B-axis size (compile-time fixed)
        m_tile_shape: Tile shape for M dimension in cube operation
        k_tile_shape: Tile shape for K dimension in cube operation
        n_tile_shape: Tile shape for N dimension in cube operation
        vector_tile_shape: Tile shapes for vector operations
        num_k_groups: Number of groups to split K-axis (default: 1)
        num_n_groups: Number of groups to split N-axis (default: 1)
        in_dtype: Input data type (default: DT_FP8E4M3)
        out_dtype: Output data type (default: DT_BF16)
        permX1: Permutation for x1 input (default: [1, 0, 2] = batch first)
        permX2: Permutation for x2 input (default: [0, 1, 2] = K,N layout)
        permY: Permutation for output (default: [1, 0, 2] = M,B,N layout)
        description: Description of the test case
    """
    ori_shape: list
    batch_size: int
    m_tile_shape: list
    k_tile_shape: list
    n_tile_shape: list
    vector_tile_shape: list
    num_k_groups: int = 1
    num_n_groups: int = 1
    in_dtype: pypto.DataType = pypto.DT_FP8E4M3
    out_dtype: pypto.DataType = pypto.DT_BF16
    permX1: List[int] = None
    permX2: List[int] = None
    permY: List[int] = None
    description: str = ""

    def __post_init__(self):
        if self.permX1 is None:
            self.permX1 = [1, 0, 2]
        if self.permX2 is None:
            self.permX2 = [0, 1, 2]
        if self.permY is None:
            self.permY = [1, 0, 2]


@pypto.frontend.jit(
    pass_options={
        "auto_mix_partition": 1,
        "cube_l1_reuse_setting": {-1: 2},
        "cube_nbuffer_setting": {-1: 2},
        "vec_nbuffer_setting": {-2: 1, -1: 16},
    },
    runtime_options={"stitch_function_max_num": 512, "device_sched_mode": 0}
)
def transpose_quant_batch_mat_mul_kernel(
    x1: pypto.Tensor(),
    x2: pypto.Tensor(),
    x1Scale: pypto.Tensor(),
    x2Scale: pypto.Tensor(),
    out: pypto.Tensor(),
    tile_config: ShapeConfig
):
    """
    Transpose quantized batch matrix multiplication kernel using MXFP8 quantization.

    This kernel performs batch matrix multiplication with MXFP8 quantization and transpose support.
    M-axis is dynamic — runtime obtains actual M size from shape[0].
    B-axis uses parallel loop for batch-wise computation.

    Args:
        x1: Input tensor of shape [M, B, K] in FP8 format — M-axis dynamic
        x2: Input tensor of shape [B, K, N] or [B, N, K] in FP8 format
        x1Scale: Scale factors for x1 in MXFP8 format, shape [M, B, K//64, 2] in E8M0
        x2Scale: Scale factors for x2 in MXFP8 format, shape varies by permX2
        out: Output tensor of shape [M, B, N]
        tile_config: Tile configuration including batch_size, shapes and tile parameters

    Note:
        - M-axis dynamic: same compiled kernel supports different M sizes
        - B-axis parallel loop: each batch computed independently
        - permX2 determines whether x2 uses scaled_mm with b_trans=True
    """
    M = tile_config.ori_shape[0]
    K = tile_config.ori_shape[1]
    N = tile_config.ori_shape[2]
    B = tile_config.batch_size
    out_dtype = tile_config.out_dtype
    permX2 = tile_config.permX2

    x1.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
    x2.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
    x1Scale.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
    x2Scale.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)

    pypto.set_cube_tile_shapes(
        tile_config.m_tile_shape, tile_config.k_tile_shape, tile_config.n_tile_shape
    )
    pypto.set_vec_tile_shapes(
        tile_config.vector_tile_shape[0], tile_config.vector_tile_shape[1],
        tile_config.vector_tile_shape[2], tile_config.vector_tile_shape[3]
    )

    for b_idx in pypto.loop(B, name="LOOP_B", idx_name="b_idx", parallel=True):
        x1_slice = x1[:, b_idx, :]
        x1_scale_slice = x1Scale[:, b_idx, :, :]

        if permX2 == [0, 1, 2]:
            mm_result = pypto.scaled_mm(
                x1_slice, x2[b_idx, :, :], out_dtype,
                x1_scale_slice, x2Scale[b_idx, :, :, :]
            )
        else:
            mm_result = pypto.scaled_mm(
                x1_slice, x2[b_idx, :, :], out_dtype,
                x1_scale_slice, x2Scale[b_idx, :, :, :],
                b_trans=True, scale_b_trans=True
            )

        out[:, b_idx, :] = mm_result