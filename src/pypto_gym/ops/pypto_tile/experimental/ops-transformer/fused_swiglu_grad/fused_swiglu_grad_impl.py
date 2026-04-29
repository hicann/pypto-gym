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
Fused SwiGLU Grad Operator Implementation

This module implements the backward kernels for the Fused SwiGLU operator.
Features:
- Dynamic shape support for batch dimension
- Bias fusion via addition
"""
import os
import pypto


def get_run_mode():
    """Get run mode from environment."""
    if 'TILE_FWK_DEVICE_ID' in os.environ:
        return pypto.RunMode.NPU
    return pypto.RunMode.SIM

global_run_mode = get_run_mode()


@pypto.frontend.jit(
    pass_options={
        "vec_nbuffer_setting": {-1: 2, 0: 4}
    },
    runtime_options={
        "run_mode": global_run_mode,
        "stitch_function_max_num": 128,
        "device_sched_mode": 3
    }
)
def fused_swiglu_bwd_b_kernel(
    dy: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    g: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    fc: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    dg: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    dfc: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    db_g_tmp: pypto.Tensor([], pypto.DT_FP32),
    db_fc_tmp: pypto.Tensor([], pypto.DT_FP32),
    db_g: pypto.Tensor([], pypto.DT_FP32),
    db_fc: pypto.Tensor([], pypto.DT_FP32)
):
    """
    Fused SwiGLU backward kernel - Compute gradients for dg and dfc,
    and accumulate db_g and db_fc.
    """
    pypto.experimental.set_operation_options(combine_axis=True)
    m = dy.shape[0]
    n = dy.shape[1]
    tile_m = 1024
    loop_count = (m + tile_m - 1) // tile_m
    pypto.set_vec_tile_shapes(32)
    index = pypto.zeros((1), dtype=pypto.DT_INT32)

    for idx in pypto.loop(loop_count, name="LOOP_BWD_DG", idx_name="idx"):
        tile_offset = idx * tile_m
        valid_m = (m - tile_offset).min(tile_m)
        dy_tile = pypto.view(dy, [tile_m, n], [tile_offset, 0], valid_shape=[valid_m, n])
        g_tile = pypto.view(g, [tile_m, n], [tile_offset, 0], valid_shape=[valid_m, n])
        fc_tile = pypto.view(fc, [tile_m, n], [tile_offset, 0], valid_shape=[valid_m, n])

        pypto.set_vec_tile_shapes(128, 128)
        exp_g = pypto.exp(g_tile)
        sigmoid_g = pypto.div(exp_g, (1.0 + exp_g), precision_type=pypto.DivAlgorithm.INTRINSIC)
        silu_g = g_tile * sigmoid_g
        dy_mul_fc = dy_tile * fc_tile
        silu_bwd = sigmoid_g * (1.0 + g_tile * (1.0 - sigmoid_g))
        dg_tile = dy_mul_fc * silu_bwd
        dfc_tile = dy_tile * silu_g
        dg[tile_offset:, 0:] = dg_tile
        dfc[tile_offset:, 0:] = dfc_tile

        dg_sum = pypto.sum(dg_tile, dim=0, keepdim=True)
        dfc_sum = pypto.sum(dfc_tile, dim=0, keepdim=True)
        dg_sum_fp32 = pypto.cast(dg_sum, pypto.DT_FP32)
        dfc_sum_fp32 = pypto.cast(dfc_sum, pypto.DT_FP32)
        db_g_view = pypto.view(db_g_tmp, [1, n], [0, 0], valid_shape=[1, n])
        db_fc_view = pypto.view(db_fc_tmp, [1, n], [0, 0], valid_shape=[1, n])
        db_g[:] = pypto.index_add_(db_g_view, 0, index, dg_sum_fp32)
        db_fc[:] = pypto.index_add_(db_fc_view, 0, index, dfc_sum_fp32)


@pypto.frontend.jit(
    pass_options={
        "vec_nbuffer_setting": {-1: 2, 0: 8},
        "cube_l1_reuse_setting": {-1: 2}
    },
    runtime_options={
        "run_mode": global_run_mode,
        "stitch_function_max_num": 128,
        "device_sched_mode": 3
    }
)
def fused_swiglu_bwd_w_kernel(
    x: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    dg: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    dfc: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    dw_g: pypto.Tensor([], pypto.DT_BF16),
    dw_fc: pypto.Tensor([], pypto.DT_BF16)
):
    """
    Fused SwiGLU backward kernel - Compute weight gradients dw_g and dw_fc.
    """
    pypto.experimental.set_operation_options(combine_axis=True)
    m = x.shape[0]
    k = x.shape[1]
    n = dg.shape[1]
    tile_m = 2048
    loop_count = (m + tile_m - 1) // tile_m

    for idx in pypto.loop(loop_count, name="LOOP_BWD_DW", idx_name="idx"):
        tile_offset = idx * tile_m
        valid_m = (m - tile_offset).min(tile_m)
        x_tile = pypto.view(x, [tile_m, k], [tile_offset, 0], valid_shape=[valid_m, k])
        dg_tile = pypto.view(dg, [tile_m, n], [tile_offset, 0], valid_shape=[valid_m, n])
        dfc_tile = pypto.view(dfc, [tile_m, n], [tile_offset, 0], valid_shape=[valid_m, n])
        pypto.set_cube_tile_shapes([128, 128], [128, 256], [128, 256])
        dw_g_tile = pypto.matmul(x_tile, dg_tile, pypto.DT_BF16, a_trans=True, b_trans=False)
        dw_fc_tile = pypto.matmul(x_tile, dfc_tile, pypto.DT_BF16, a_trans=True, b_trans=False)
        pypto.set_vec_tile_shapes(128, 128)
        dw_g[:] = dw_g + dw_g_tile
        dw_fc[:] = dw_fc + dw_fc_tile


@pypto.frontend.jit(
    pass_options={
        "vec_nbuffer_setting": {-1: 2, 0: 4},
        "cube_l1_reuse_setting": {-1: 2}
    },
    runtime_options={
        "run_mode": global_run_mode,
        "stitch_function_max_num": 128,
        "device_sched_mode": 3
    }
)
def fused_swiglu_bwd_x_kernel(
    dg: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    dfc: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    w_g: pypto.Tensor([], pypto.DT_BF16),
    w_fc: pypto.Tensor([], pypto.DT_BF16),
    dx: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16)
):
    """
    Fused SwiGLU backward kernel - Compute input gradient dx.
    """
    pypto.experimental.set_operation_options(combine_axis=True)
    m = dg.shape[0]
    n = dg.shape[1]
    tile_m = 1024
    loop_count = (m + tile_m - 1) // tile_m

    for idx in pypto.loop(loop_count, name="LOOP_BWD_DX", idx_name="idx"):
        tile_offset = idx * tile_m
        valid_m = (m - tile_offset).min(tile_m)
        dg_tile = pypto.view(dg, [tile_m, n], [tile_offset, 0], valid_shape=[valid_m, n])
        dfc_tile = pypto.view(dfc, [tile_m, n], [tile_offset, 0], valid_shape=[valid_m, n])
        pypto.set_cube_tile_shapes([128, 128], [64, 256], [256, 256])
        dx_g_tile = pypto.matmul(dg_tile, w_g, pypto.DT_BF16, a_trans=False, b_trans=True)
        dx_fc_tile = pypto.matmul(dfc_tile, w_fc, pypto.DT_BF16, a_trans=False, b_trans=True)
        if pypto.platform.npuarch == 'DAV_3510':
            pypto.set_vec_tile_shapes(128, 256)
        else:
            pypto.set_vec_tile_shapes(128, 128)
        dx_result = dx_g_tile + dx_fc_tile
        dx[tile_offset:, 0:] = dx_result