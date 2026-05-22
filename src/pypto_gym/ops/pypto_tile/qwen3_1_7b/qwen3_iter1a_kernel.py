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

# Iter 1a: minimal fused kernel for Qwen3-1.7B prefill — RMSNorm + QKV proj.
# Rewritten in GLM-style: outer bs_tile loop converts the dynamic S axis
# to a fixed-shape tile, avoiding the "INT32_MAX" symbolic shape error.

import math
import pypto

H = 2048
Nq = 16
Nkv = 8
D = 128
EPS = 1e-6
BS_TILE = 8


@pypto.frontend.jit(
    runtime_options={
        "stitch_function_max_num": 128,
        "device_sched_mode": 1,
    }
)
def qwen3_pre_qkv_iter1a(
    x:         pypto.Tensor([pypto.DYNAMIC, H], pypto.DT_BF16),                # [S, 2048]
    w_in_norm: pypto.Tensor([H], pypto.DT_BF16),                               # [2048]
    Wq:        pypto.Tensor([Nq * D, H], pypto.DT_BF16),                       # [2048, 2048]
    Wk:        pypto.Tensor([Nkv * D, H], pypto.DT_BF16),                      # [1024, 2048]
    Wv:        pypto.Tensor([Nkv * D, H], pypto.DT_BF16),                      # [1024, 2048]
    q_out:     pypto.Tensor([pypto.DYNAMIC, Nq * D], pypto.DT_BF16),          # [S, 2048]
    k_out:     pypto.Tensor([pypto.DYNAMIC, Nkv * D], pypto.DT_BF16),          # [S, 1024]
    v_out:     pypto.Tensor([pypto.DYNAMIC, Nkv * D], pypto.DT_BF16),          # [S, 1024]
):
    S = x.shape[0]
    bs_loop = (S + BS_TILE - 1) // BS_TILE
    mean_coff = 1.0 / H

    # broadcast w_in_norm to [1, H] FP32 once (compile-time constant work)
    pypto.set_vec_tile_shapes(1, H)
    w_norm_2d = pypto.reshape(w_in_norm, [1, H], inplace=True)
    w_norm_fp32 = pypto.cast(w_norm_2d, pypto.DT_FP32)

    for bs_idx in pypto.loop(bs_loop, name="LOOP_BS", idx_name="bs_idx"):
        cur_bs = (S - bs_idx * BS_TILE).min(BS_TILE)

        # ---- view input tile ----
        x_tile = pypto.view(x, [BS_TILE, H], [bs_idx * BS_TILE, 0],
                            valid_shape=[cur_bs, H])

        # ---- Stage 1: RMSNorm ----
        pypto.set_vec_tile_shapes(1, H)
        x_fp32 = pypto.cast(x_tile, pypto.DT_FP32)
        sq = pypto.mul(x_fp32, x_fp32)
        mean = pypto.sum(sq, -1, keepdim=True)
        mean = pypto.mul(mean, mean_coff)
        mean_eps = pypto.add(mean, EPS)
        rsqrt = pypto.rsqrt(mean_eps)
        n1_fp32 = pypto.mul(x_fp32, rsqrt)
        n1_fp32 = pypto.mul(n1_fp32, w_norm_fp32)        # [BS_TILE, H] FP32 (broadcast w over rows)
        n1 = pypto.cast(n1_fp32, pypto.DT_BF16)          # [BS_TILE, H] BF16

        # ---- Stage 2: QKV matmul ----
        pypto.set_cube_tile_shapes([32, 32], [128, 512], [128, 128])
        q_fp32 = pypto.matmul(n1, Wq, pypto.DT_FP32, b_trans=True)   # [BS_TILE, 2048] FP32
        pypto.set_cube_tile_shapes([32, 32], [128, 512], [128, 128])
        k_fp32 = pypto.matmul(n1, Wk, pypto.DT_FP32, b_trans=True)   # [BS_TILE, 1024] FP32
        pypto.set_cube_tile_shapes([32, 32], [128, 512], [128, 128])
        v_fp32 = pypto.matmul(n1, Wv, pypto.DT_FP32, b_trans=True)   # [BS_TILE, 1024] FP32

        pypto.set_vec_tile_shapes(BS_TILE, Nq * D)
        q_bf = pypto.cast(q_fp32, pypto.DT_BF16)
        pypto.set_vec_tile_shapes(BS_TILE, Nkv * D)
        k_bf = pypto.cast(k_fp32, pypto.DT_BF16)
        v_bf = pypto.cast(v_fp32, pypto.DT_BF16)

        # write back via view
        q_out[bs_idx * BS_TILE: bs_idx * BS_TILE + BS_TILE, :] = q_bf
        k_out[bs_idx * BS_TILE: bs_idx * BS_TILE + BS_TILE, :] = k_bf
        v_out[bs_idx * BS_TILE: bs_idx * BS_TILE + BS_TILE, :] = v_bf
