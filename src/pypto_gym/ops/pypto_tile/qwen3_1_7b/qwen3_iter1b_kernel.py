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

# Iter 1b: extend Iter 1a with Q/K per-head RMSNorm + RoPE.
# Output q/k include RoPE; v unchanged.
#
# Layout note: kernel keeps the flat [S, N*D] form (same as Iter 1a) but the
# Nq/Nkv heads are normed/rotated independently along the D axis.

import math
import pypto

H = 2048
Nq = 16
Nkv = 8
D = 128
HALF_D = D // 2
EPS = 1e-6
BS_TILE = 8


def _rms_norm_per_d(x_3d_fp32, w_fp32_1_1_d, mean_coff, eps):
    """RMSNorm along the last dim. x_3d_fp32: [BS_TILE, N, D] FP32. w: [1,1,D] FP32 (broadcast)."""
    sq = pypto.mul(x_3d_fp32, x_3d_fp32)
    mean = pypto.sum(sq, -1, keepdim=True)
    mean = pypto.mul(mean, mean_coff)
    rsqrt = pypto.rsqrt(pypto.add(mean, eps))
    out = pypto.mul(x_3d_fp32, rsqrt)
    out = pypto.mul(out, w_fp32_1_1_d)
    return out


def _rope_half(x_split_fp32_left, x_split_fp32_right, cos_half_fp32, sin_half_fp32):
    """RoPE: returns (o1, o2) FP32 where final = cat([o1, o2], dim=-1).
    cos/sin shape: [BS_TILE, 1, D/2] (broadcast over heads).
    x_left/x_right shape: [BS_TILE, N, D/2]."""
    o1 = pypto.sub(pypto.mul(x_split_fp32_left, cos_half_fp32),
                   pypto.mul(x_split_fp32_right, sin_half_fp32))
    o2 = pypto.add(pypto.mul(x_split_fp32_right, cos_half_fp32),
                   pypto.mul(x_split_fp32_left, sin_half_fp32))
    return o1, o2


@pypto.frontend.jit(
    runtime_options={
        "stitch_function_max_num": 128,
        "device_sched_mode": 1,
    }
)
def qwen3_pre_attn_iter1b(
    x:         pypto.Tensor([pypto.DYNAMIC, H], pypto.DT_BF16),
    cos:       pypto.Tensor([pypto.DYNAMIC, D], pypto.DT_BF16),               # [S, 128]
    sin:       pypto.Tensor([pypto.DYNAMIC, D], pypto.DT_BF16),               # [S, 128]
    w_in_norm: pypto.Tensor([H], pypto.DT_BF16),
    Wq:        pypto.Tensor([H, Nq * D], pypto.DT_BF16),                      # transposed: [2048, 2048]
    Wk:        pypto.Tensor([H, Nkv * D], pypto.DT_BF16),                     # transposed: [2048, 1024]
    Wv:        pypto.Tensor([H, Nkv * D], pypto.DT_BF16),
    w_q_norm:  pypto.Tensor([D], pypto.DT_BF16),
    w_k_norm:  pypto.Tensor([D], pypto.DT_BF16),
    q_out:     pypto.Tensor([pypto.DYNAMIC, Nq * D], pypto.DT_BF16),
    k_out:     pypto.Tensor([pypto.DYNAMIC, Nkv * D], pypto.DT_BF16),
    v_out:     pypto.Tensor([pypto.DYNAMIC, Nkv * D], pypto.DT_BF16),
):
    S = x.shape[0]
    bs_loop = (S + BS_TILE - 1) // BS_TILE
    h_mean_coff = 1.0 / H
    d_mean_coff = 1.0 / D

    # ---- precompute broadcasted norm weights as FP32 ----
    pypto.set_vec_tile_shapes(1, H)
    w_in_norm_2d = pypto.reshape(w_in_norm, [1, H], inplace=True)
    w_in_norm_fp32 = pypto.cast(w_in_norm_2d, pypto.DT_FP32)

    pypto.set_vec_tile_shapes(1, 1, D)
    w_q_norm_3d = pypto.reshape(w_q_norm, [1, 1, D], inplace=True)
    w_k_norm_3d = pypto.reshape(w_k_norm, [1, 1, D], inplace=True)
    w_q_norm_fp32 = pypto.cast(w_q_norm_3d, pypto.DT_FP32)
    w_k_norm_fp32 = pypto.cast(w_k_norm_3d, pypto.DT_FP32)
    w_q_norm_q = pypto.expand_clone(w_q_norm_fp32, [1, Nq, D])
    w_k_norm_k = pypto.expand_clone(w_k_norm_fp32, [1, Nkv, D])

    for bs_idx in pypto.loop(bs_loop, name="LOOP_BS", idx_name="bs_idx"):
        cur_bs = (S - bs_idx * BS_TILE).min(BS_TILE)

        # ---- Pre-norm RMSNorm ----
        x_tile = pypto.view(x, [BS_TILE, H], [bs_idx * BS_TILE, 0],
                            valid_shape=[cur_bs, H])
        pypto.set_vec_tile_shapes(1, H)
        x_fp32 = pypto.cast(x_tile, pypto.DT_FP32)
        sq = pypto.mul(x_fp32, x_fp32)
        mean = pypto.sum(sq, -1, keepdim=True)
        mean = pypto.mul(mean, h_mean_coff)
        rsqrt = pypto.rsqrt(pypto.add(mean, EPS))
        n1_fp32 = pypto.mul(pypto.mul(x_fp32, rsqrt), w_in_norm_fp32)
        # Explicit staging buffer to pin static shape [BS_TILE, H] for matmul input
        n1_buf = pypto.tensor([BS_TILE, H], pypto.DT_BF16, "n1_buf")
        n1_buf[:] = pypto.cast(n1_fp32, pypto.DT_BF16)

        # ---- QKV proj ----
        pypto.set_cube_tile_shapes([32, 32], [128, 512], [128, 128])
        q_fp32 = pypto.matmul(n1_buf, Wq, pypto.DT_FP32)       # [BS_TILE, Nq*D] FP32
        k_fp32 = pypto.matmul(n1_buf, Wk, pypto.DT_FP32)
        v_fp32 = pypto.matmul(n1_buf, Wv, pypto.DT_FP32)

        # cast to BF16 flat first (GLM pattern)
        pypto.set_vec_tile_shapes(BS_TILE, Nq * D)
        q_bf_flat = pypto.cast(q_fp32, pypto.DT_BF16)
        pypto.set_vec_tile_shapes(BS_TILE, Nkv * D)
        k_bf_flat = pypto.cast(k_fp32, pypto.DT_BF16)
        v_bf_flat = pypto.cast(v_fp32, pypto.DT_BF16)

        # ---- Q/K per-head RMSNorm (along D) ----
        pypto.set_vec_tile_shapes(BS_TILE, Nq, D)
        q_3d = pypto.reshape(q_bf_flat, [BS_TILE, Nq, D])
        q_3d_fp32 = pypto.cast(q_3d, pypto.DT_FP32)
        q_normed_fp32 = _rms_norm_per_d(q_3d_fp32, w_q_norm_q, d_mean_coff, EPS)

        pypto.set_vec_tile_shapes(BS_TILE, Nkv, D)
        k_3d = pypto.reshape(k_bf_flat, [BS_TILE, Nkv, D])
        k_3d_fp32 = pypto.cast(k_3d, pypto.DT_FP32)
        k_normed_fp32 = _rms_norm_per_d(k_3d_fp32, w_k_norm_k, d_mean_coff, EPS)

        # ---- RoPE ----
        # Take cos/sin tile [BS_TILE, D] (only first HALF_D values are used; second half is duplicate)
        cos_tile = pypto.view(cos, [BS_TILE, D], [bs_idx * BS_TILE, 0],
                              valid_shape=[cur_bs, D])
        sin_tile = pypto.view(sin, [BS_TILE, D], [bs_idx * BS_TILE, 0],
                              valid_shape=[cur_bs, D])
        # Take only the first HALF_D (cos/sin are mirror-duplicated across halves in Qwen3 RoPE)
        pypto.set_vec_tile_shapes(BS_TILE, HALF_D)
        cos_half_2d = pypto.view(cos_tile, [BS_TILE, HALF_D], [0, 0],
                                 valid_shape=[cur_bs, HALF_D])
        sin_half_2d = pypto.view(sin_tile, [BS_TILE, HALF_D], [0, 0],
                                 valid_shape=[cur_bs, HALF_D])
        cos_half_fp32 = pypto.cast(cos_half_2d, pypto.DT_FP32)
        sin_half_fp32 = pypto.cast(sin_half_2d, pypto.DT_FP32)
        # Reshape to [BS_TILE, 1, HALF_D] for broadcast over heads
        cos_b = pypto.reshape(cos_half_fp32, [BS_TILE, 1, HALF_D], inplace=True)
        sin_b = pypto.reshape(sin_half_fp32, [BS_TILE, 1, HALF_D], inplace=True)

        # Q RoPE
        pypto.set_vec_tile_shapes(BS_TILE, Nq, HALF_D)
        q_left = pypto.view(q_normed_fp32, [BS_TILE, Nq, HALF_D], [0, 0, 0],
                             valid_shape=[cur_bs, Nq, HALF_D])
        q_right = pypto.view(q_normed_fp32, [BS_TILE, Nq, HALF_D], [0, 0, HALF_D],
                             valid_shape=[cur_bs, Nq, HALF_D])
        q_o1 = pypto.sub(pypto.mul(q_left, cos_b), pypto.mul(q_right, sin_b))
        q_o2 = pypto.add(pypto.mul(q_right, cos_b), pypto.mul(q_left, sin_b))
        q_rope = pypto.concat([q_o1, q_o2], 2)                       # [BS_TILE, Nq, D] FP32
        q_rope_bf = pypto.cast(q_rope, pypto.DT_BF16)

        # K RoPE
        pypto.set_vec_tile_shapes(BS_TILE, Nkv, HALF_D)
        k_left = pypto.view(k_normed_fp32, [BS_TILE, Nkv, HALF_D], [0, 0, 0],
                             valid_shape=[cur_bs, Nkv, HALF_D])
        k_right = pypto.view(k_normed_fp32, [BS_TILE, Nkv, HALF_D], [0, 0, HALF_D],
                             valid_shape=[cur_bs, Nkv, HALF_D])
        k_o1 = pypto.sub(pypto.mul(k_left, cos_b), pypto.mul(k_right, sin_b))
        k_o2 = pypto.add(pypto.mul(k_right, cos_b), pypto.mul(k_left, sin_b))
        k_rope = pypto.concat([k_o1, k_o2], 2)
        k_rope_bf = pypto.cast(k_rope, pypto.DT_BF16)

        # ---- Write back as flat [S, N*D] ----
        pypto.set_vec_tile_shapes(BS_TILE, Nq, D)
        q_flat_out = pypto.reshape(q_rope_bf, [BS_TILE, Nq * D],
                                   valid_shape=[cur_bs, Nq * D], inplace=True)
        pypto.set_vec_tile_shapes(BS_TILE, Nkv, D)
        k_flat_out = pypto.reshape(k_rope_bf, [BS_TILE, Nkv * D],
                                   valid_shape=[cur_bs, Nkv * D], inplace=True)

        q_out[bs_idx * BS_TILE: bs_idx * BS_TILE + BS_TILE, :] = q_flat_out
        k_out[bs_idx * BS_TILE: bs_idx * BS_TILE + BS_TILE, :] = k_flat_out
        v_out[bs_idx * BS_TILE: bs_idx * BS_TILE + BS_TILE, :] = v_bf_flat
