#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PyPTO RMSNorm kernel implementation.

Operator: RMSNorm
Formula: out[b, c, h, w] = x[b, c, h, w] / sqrt(mean(x[b, j, h, w]^2, dim=1) + eps)

Implementation notes:
  - Based on DESIGN.md, full FP32 pipeline, no cast.
  - API sequence: mul -> sum(dim=1) -> div -> add -> sqrt -> div
  - TileShape: [1, 64, 1, 128], fully covers reduction dim C=64, tail 128 (32B aligned)
  - Dynamic axis B(dim=0) uses pypto.loop + pypto.view + pypto.assemble
  - C/H/W are compile-time constants to satisfy PyPTO TileShape and view.shape requirements
  - Ref: examples/02_intermediate/basic_nn/layer_normalization/layer_norm.py (rms_norm_core)

Exported function: RMSNorm_wrapper(x, eps) -> torch.Tensor
"""

import pypto
import torch
from dataclasses import dataclass


@dataclass
class RMSNormConfig:
    """RMSNorm configuration parameters."""
    eps: float = 1e-5
    num_features: int = 64


def rms_norm_core(x: pypto.Tensor, eps: float, num_features: int) -> pypto.Tensor:
    """RMSNorm core computation: pure PyPTO API implementation.

    Formula: out = x / sqrt(sum(x^2, dim=1) / C + eps)

    Args:
        x: input tensor, shape [1, 64, 256, 256], FP32
        eps: small constant to prevent division by zero, float
        num_features: feature dimension size, int (=64)

    Returns:
        normalized tensor, same shape as input
    """
    sq = x * x
    s = pypto.sum(sq, dim=1, keepdim=True)
    mean_sq = s / num_features
    mean_sq_eps = mean_sq + eps
    rms = pypto.sqrt(mean_sq_eps)
    out = x / rms
    return out


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU, "stitch_function_max_num": 128})
def rms_norm_kernel(
    x: pypto.Tensor([pypto.DYNAMIC, 64, 256, 256], pypto.DT_FP32),
    output: pypto.Tensor([pypto.DYNAMIC, 64, 256, 256], pypto.DT_FP32),
    config: RMSNormConfig,
):
    """RMSNorm JIT kernel with dynamic batch dimension support.

    Args:
        x: input tensor, dim=0 is pypto.DYNAMIC
        output: output tensor, dim=0 is pypto.DYNAMIC
        config: RMSNormConfig dataclass with eps and num_features
    """
    C = 64
    H = 256
    W = 256
    eps = config.eps
    B = x.shape[0]

    pypto.set_vec_tile_shapes(1, C, 2, 128)

    for b_idx in pypto.loop(B, unroll_list=[16, 8, 4, 2, 1], name="LOOP_BATCH", idx_name="b_idx"):
        x_slice = pypto.view(x, [1, C, H, W], [b_idx, 0, 0, 0])
        result = rms_norm_core(x_slice, eps, C)
        pypto.assemble(result, [b_idx, 0, 0, 0], output)


def RMSNorm_wrapper(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RMSNorm external interface: torch.Tensor input -> torch.Tensor output.

    Args:
        x: input tensor, shape [B, C, H, W], dtype float32
        eps: small constant to prevent division by zero, default 1e-5

    Returns:
        RMS-normalized tensor, same shape as input
    """
    config = RMSNormConfig(eps=eps, num_features=x.shape[1])
    out = torch.empty_like(x)
    rms_norm_kernel(x, out, config)
    return out
