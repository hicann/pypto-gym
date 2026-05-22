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

# Fused kernel: RMSNorm + QKV proj + Q/K per-head RMSNorm + RoPE
# Combines previous K1 + K2 into one JIT kernel.
#
# Critical insight from pypto-precision-debug skill:
#   inplace=True on reshape causes view+reshape failures in dynamic-shape kernels.
#   We use inplace=False everywhere to avoid memory aliasing issues that triggered
#   the Iter 1b "Shape size mismatch" compile error.

import math
import pypto

H = 2048
Nq = 16
Nkv = 8
D = 128
HALF_D = D // 2
EPS = 1e-6
BS_TILE = 8


def _rms_norm_per_d_fp32(x_3d_fp32, w_fp32_n, mean_coff, eps):
    """RMSNorm along last dim. x: [BS_TILE, N, D] FP32. w: [1, N, D] FP32 broadcast."""
    sq = pypto.mul(x_3d_fp32, x_3d_fp32)
    mean = pypto.sum(sq, -1, keepdim=True)
    mean = pypto.mul(mean, mean_coff)
    rsqrt = pypto.rsqrt(pypto.add(mean, eps))
    return pypto.mul(pypto.mul(x_3d_fp32, rsqrt), w_fp32_n)


@pypto.frontend.jit(
    runtime_options={
        "stitch_function_max_num": 128, 
        "device_sched_mode": 1
    }
)
def qwen3_pre_attn_fused(
    x:         pypto.Tensor([pypto.DYNAMIC, H], pypto.DT_BF16),               # [S, 2048]
    cos:       pypto.Tensor([pypto.DYNAMIC, D], pypto.DT_BF16),               # [S, 128]
    sin:       pypto.Tensor([pypto.DYNAMIC, D], pypto.DT_BF16),               # [S, 128]
    w_in_norm: pypto.Tensor([H], pypto.DT_BF16),
    Wq:        pypto.Tensor([Nq * D, H], pypto.DT_BF16),                      # [2048, 2048]
    Wk:        pypto.Tensor([Nkv * D, H], pypto.DT_BF16),                     # [1024, 2048]
    Wv:        pypto.Tensor([Nkv * D, H], pypto.DT_BF16),
    w_q_norm:  pypto.Tensor([D], pypto.DT_BF16),
    w_k_norm:  pypto.Tensor([D], pypto.DT_BF16),
    q_out:     pypto.Tensor([pypto.DYNAMIC, Nq, D], pypto.DT_BF16),          # [S, 16, 128]
    k_out:     pypto.Tensor([pypto.DYNAMIC, Nkv, D], pypto.DT_BF16),
    v_out:     pypto.Tensor([pypto.DYNAMIC, Nkv, D], pypto.DT_BF16),
):
    S = x.shape[0]
    bs_loop = (S + BS_TILE - 1) // BS_TILE
    h_mean_coff = 1.0 / H
    d_mean_coff = 1.0 / D

    # Pre-prep: broadcast pre-norm weight, build [1, N, D] head-norm weights once.
    pypto.set_vec_tile_shapes(1, H)
    w_in_2d = pypto.reshape(w_in_norm, [1, H])
    w_in_fp32 = pypto.cast(w_in_2d, pypto.DT_FP32)

    pypto.set_vec_tile_shapes(1, 1, D)
    w_qn_3d = pypto.reshape(w_q_norm, [1, 1, D])
    w_kn_3d = pypto.reshape(w_k_norm, [1, 1, D])
    w_qn_fp32 = pypto.cast(w_qn_3d, pypto.DT_FP32)
    w_kn_fp32 = pypto.cast(w_kn_3d, pypto.DT_FP32)
    w_qn_full = pypto.expand_clone(w_qn_fp32, [1, Nq, D])
    w_kn_full = pypto.expand_clone(w_kn_fp32, [1, Nkv, D])

    for bs_idx in pypto.loop(bs_loop, name="LOOP_BS_PRE", idx_name="bs_idx"):
        cur_bs = (S - bs_idx * BS_TILE).min(BS_TILE)

        # ---- Stage 1: pre-norm RMSNorm ----
        x_tile = pypto.view(x, [BS_TILE, H], [bs_idx * BS_TILE, 0],
                            valid_shape=[cur_bs, H])
        pypto.set_vec_tile_shapes(1, H)
        x_fp32 = pypto.cast(x_tile, pypto.DT_FP32)
        sq = pypto.mul(x_fp32, x_fp32)
        mean = pypto.sum(sq, -1, keepdim=True)
        mean = pypto.mul(mean, h_mean_coff)
        rsqrt = pypto.rsqrt(pypto.add(mean, EPS))
        n1_fp32 = pypto.mul(pypto.mul(x_fp32, rsqrt), w_in_fp32)
        # Stage to fixed-shape BF16 buffer for matmul inputs
        n1 = pypto.tensor([BS_TILE, H], pypto.DT_BF16, "n1_buf")
        n1[:] = pypto.cast(n1_fp32, pypto.DT_BF16)

        # ---- Stage 2: QKV matmul (set tile per-matmul) ----
        pypto.set_cube_tile_shapes([8, 8], [128, 512], [128, 128])
        q_flat_fp = pypto.matmul(n1, Wq, pypto.DT_FP32, b_trans=True)
        pypto.set_cube_tile_shapes([8, 8], [128, 512], [128, 128])
        k_flat_fp = pypto.matmul(n1, Wk, pypto.DT_FP32, b_trans=True)
        pypto.set_cube_tile_shapes([8, 8], [128, 512], [128, 128])
        v_flat_fp = pypto.matmul(n1, Wv, pypto.DT_FP32, b_trans=True)

        # Stage to BF16 buffers with concrete shape for downstream reshape.
        # Using preallocated buffers + valid_shape on view ensures invalid rows
        # are not read by per-head RMSNorm.
        q_buf = pypto.tensor([BS_TILE, Nq * D], pypto.DT_BF16, "q_buf")
        k_buf = pypto.tensor([BS_TILE, Nkv * D], pypto.DT_BF16, "k_buf")
        v_buf = pypto.tensor([BS_TILE, Nkv * D], pypto.DT_BF16, "v_buf")
        pypto.set_vec_tile_shapes(BS_TILE, Nq * D)
        q_buf[:] = pypto.cast(q_flat_fp, pypto.DT_BF16)
        pypto.set_vec_tile_shapes(BS_TILE, Nkv * D)
        k_buf[:] = pypto.cast(k_flat_fp, pypto.DT_BF16)
        v_buf[:] = pypto.cast(v_flat_fp, pypto.DT_BF16)

        # ---- Stage 3: reshape to [BS_TILE, N, D] for per-head norm ----
        pypto.set_vec_tile_shapes(BS_TILE, Nq, D)
        q_3d = pypto.reshape(q_buf, [BS_TILE, Nq, D],
                             valid_shape=[cur_bs, Nq, D])
        pypto.set_vec_tile_shapes(BS_TILE, Nkv, D)
        k_3d = pypto.reshape(k_buf, [BS_TILE, Nkv, D],
                             valid_shape=[cur_bs, Nkv, D])
        v_3d = pypto.reshape(v_buf, [BS_TILE, Nkv, D],
                             valid_shape=[cur_bs, Nkv, D])

        # ---- Stage 4: Q/K per-head RMSNorm ----
        pypto.set_vec_tile_shapes(BS_TILE, Nq, D)
        q_3d_fp = pypto.cast(q_3d, pypto.DT_FP32)
        q_normed = _rms_norm_per_d_fp32(q_3d_fp, w_qn_full, d_mean_coff, EPS)
        pypto.set_vec_tile_shapes(BS_TILE, Nkv, D)
        k_3d_fp = pypto.cast(k_3d, pypto.DT_FP32)
        k_normed = _rms_norm_per_d_fp32(k_3d_fp, w_kn_full, d_mean_coff, EPS)

        # ---- Stage 5: RoPE ----
        cos_tile = pypto.view(cos, [BS_TILE, D], [bs_idx * BS_TILE, 0],
                              valid_shape=[cur_bs, D])
        sin_tile = pypto.view(sin, [BS_TILE, D], [bs_idx * BS_TILE, 0],
                              valid_shape=[cur_bs, D])
        pypto.set_vec_tile_shapes(BS_TILE, HALF_D)
        cos_half = pypto.view(cos_tile, [BS_TILE, HALF_D], [0, 0],
                              valid_shape=[cur_bs, HALF_D])
        sin_half = pypto.view(sin_tile, [BS_TILE, HALF_D], [0, 0],
                              valid_shape=[cur_bs, HALF_D])
        cos_half_fp = pypto.cast(cos_half, pypto.DT_FP32)
        sin_half_fp = pypto.cast(sin_half, pypto.DT_FP32)
        cos_b = pypto.reshape(cos_half_fp, [BS_TILE, 1, HALF_D])
        sin_b = pypto.reshape(sin_half_fp, [BS_TILE, 1, HALF_D])

        # Q RoPE
        pypto.set_vec_tile_shapes(BS_TILE, Nq, HALF_D)
        q_left = pypto.view(q_normed, [BS_TILE, Nq, HALF_D], [0, 0, 0],
                             valid_shape=[cur_bs, Nq, HALF_D])
        q_right = pypto.view(q_normed, [BS_TILE, Nq, HALF_D], [0, 0, HALF_D],
                             valid_shape=[cur_bs, Nq, HALF_D])
        q_o1 = pypto.sub(pypto.mul(q_left, cos_b), pypto.mul(q_right, sin_b))
        q_o2 = pypto.add(pypto.mul(q_right, cos_b), pypto.mul(q_left, sin_b))
        q_rope = pypto.concat([q_o1, q_o2], 2)
        q_bf = pypto.cast(q_rope, pypto.DT_BF16)

        # K RoPE
        pypto.set_vec_tile_shapes(BS_TILE, Nkv, HALF_D)
        k_left = pypto.view(k_normed, [BS_TILE, Nkv, HALF_D], [0, 0, 0],
                             valid_shape=[cur_bs, Nkv, HALF_D])
        k_right = pypto.view(k_normed, [BS_TILE, Nkv, HALF_D], [0, 0, HALF_D],
                             valid_shape=[cur_bs, Nkv, HALF_D])
        k_o1 = pypto.sub(pypto.mul(k_left, cos_b), pypto.mul(k_right, sin_b))
        k_o2 = pypto.add(pypto.mul(k_right, cos_b), pypto.mul(k_left, sin_b))
        k_rope = pypto.concat([k_o1, k_o2], 2)
        k_bf = pypto.cast(k_rope, pypto.DT_BF16)

        # ---- Write back ----
        q_out[bs_idx * BS_TILE: bs_idx * BS_TILE + BS_TILE, :, :] = q_bf
        k_out[bs_idx * BS_TILE: bs_idx * BS_TILE + BS_TILE, :, :] = k_bf
        v_out[bs_idx * BS_TILE: bs_idx * BS_TILE + BS_TILE, :, :] = v_3d
