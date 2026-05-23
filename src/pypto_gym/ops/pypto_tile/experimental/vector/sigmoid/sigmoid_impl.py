#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PyPTO Sigmoid kernel implementation.

Operator: Sigmoid
Formula: sigma(x) = 1 / (1 + exp(-x))
Implementation pattern: @pypto.frontend.jit + pypto.loop + pypto.view + pypto.sigmoid

Exported function: Sigmoid_wrapper() for test_sigmoid.py.
Ref: models/glm_v4_5/glm_gate.py (loop + view + assemble pattern)
"""

import os
import sys
import pypto
import torch


def _peek_run_mode_from_argv(default: str = "npu") -> str:
    """Read run_mode early so module-level decorators can use it."""
    for idx, arg in enumerate(sys.argv):
        if arg == "--run_mode" and idx + 1 < len(sys.argv):
            value = sys.argv[idx + 1]
            if value in ("npu", "sim"):
                return value
        if arg.startswith("--run_mode="):
            value = arg.split("=", 1)[1]
            if value in ("npu", "sim"):
                return value
    return default


global_run_mode = pypto.RunMode.NPU
if _peek_run_mode_from_argv("npu") == "sim":
    global_run_mode = pypto.RunMode.SIM

TILE_B = 16
TILE_D = 512


@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def sigmoid_kernel(
    x: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
):
    """Sigmoid activation kernel:
    out = sigma(x) = 1 / (1 + exp(-x))

    Args:
        x: input [B, D], FP32
        out: output [B, D], FP32
    """
    B = x.shape[0]
    D = x.shape[1]
    bs_loop = (B + TILE_B - 1) // TILE_B
    ds_loop = (D + TILE_D - 1) // TILE_D
    pypto.set_vec_tile_shapes(TILE_B, TILE_D)

    for b_idx in pypto.loop(bs_loop, name="loop_b", idx_name="b_idx"):
        for d_idx in pypto.loop(ds_loop, unroll_list=[64, 16, 4], name="loop_d", idx_name="d_idx"):
            actual_b = (B - b_idx * TILE_B).min(TILE_B)
            actual_d = (D - d_idx * TILE_D).min(TILE_D)
            tile_x = pypto.view(
                x, [TILE_B, TILE_D], [b_idx * TILE_B, d_idx * TILE_D],
                valid_shape=[actual_b, actual_d])
            tile_result = pypto.sigmoid(tile_x)
            out[b_idx * TILE_B:, d_idx * TILE_D:] = tile_result


def Sigmoid_wrapper(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid operator wrapper, encapsulates kernel call.

    Args:
        x: input tensor, shape [B, D], dtype float32.
           B and D are both dynamic axes, supports any 2D shape.

    Returns:
        y: output tensor, same shape as input [B, D], dtype float32.
           Range (0, 1).
    """
    x = x.contiguous()
    out = torch.empty_like(x)
    sigmoid_kernel(x, out)
    return out
