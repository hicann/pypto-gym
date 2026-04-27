#!/usr/bin/env python3
# coding: utf-8
# K2: Q/K per-head RMSNorm + RoPE.
# Input/output are 3D [S, N, D] tensors — avoids reshape from matmul output.

import math
import pypto

D = 128
HALF_D = D // 2
EPS = 1e-6
BS_TILE = 8


def _rms_norm_per_d(x_3d_fp32, w_fp32_1_1_d, mean_coff, eps):
    sq = pypto.mul(x_3d_fp32, x_3d_fp32)
    mean = pypto.sum(sq, -1, keepdim=True)
    mean = pypto.mul(mean, mean_coff)
    rsqrt = pypto.rsqrt(pypto.add(mean, eps))
    out = pypto.mul(x_3d_fp32, rsqrt)
    out = pypto.mul(out, w_fp32_1_1_d)
    return out


def _make_qk_rope_kernel(N: int):
    """Factory: returns a JIT kernel specialised for N heads."""

    @pypto.frontend.jit(
        runtime_options={"stitch_function_max_num": 128, "device_sched_mode": 1},
        debug_options={"runtime_debug_mode": 0},
    )
    def kernel(
        x:        pypto.Tensor([pypto.DYNAMIC, N, D], pypto.DT_BF16),    # [S, N, 128]
        cos:      pypto.Tensor([pypto.DYNAMIC, D], pypto.DT_BF16),       # [S, 128]
        sin:      pypto.Tensor([pypto.DYNAMIC, D], pypto.DT_BF16),       # [S, 128]
        w_norm:   pypto.Tensor([D], pypto.DT_BF16),                      # [128]
        out:      pypto.Tensor([pypto.DYNAMIC, N, D], pypto.DT_BF16),    # [S, N, 128]
    ):
        S = x.shape[0]
        bs_loop = (S + BS_TILE - 1) // BS_TILE
        d_mean_coff = 1.0 / D

        pypto.set_vec_tile_shapes(1, 1, D)
        w_3d = pypto.reshape(w_norm, [1, 1, D], inplace=True)
        w_fp32_1 = pypto.cast(w_3d, pypto.DT_FP32)
        w_fp32_n = pypto.expand_clone(w_fp32_1, [1, N, D])

        for bs_idx in pypto.loop(bs_loop, name="LOOP_BS_QKROPE", idx_name="bs_idx"):
            cur_bs = (S - bs_idx * BS_TILE).min(BS_TILE)

            x_tile = pypto.view(x, [BS_TILE, N, D], [bs_idx * BS_TILE, 0, 0],
                                valid_shape=[cur_bs, N, D])
            pypto.set_vec_tile_shapes(BS_TILE, N, D)
            x_fp32 = pypto.cast(x_tile, pypto.DT_FP32)
            normed_fp32 = _rms_norm_per_d(x_fp32, w_fp32_n, d_mean_coff, EPS)

            # cos/sin tile, take first half
            cos_tile = pypto.view(cos, [BS_TILE, D], [bs_idx * BS_TILE, 0],
                                  valid_shape=[cur_bs, D])
            sin_tile = pypto.view(sin, [BS_TILE, D], [bs_idx * BS_TILE, 0],
                                  valid_shape=[cur_bs, D])
            pypto.set_vec_tile_shapes(BS_TILE, HALF_D)
            cos_half = pypto.view(cos_tile, [BS_TILE, HALF_D], [0, 0],
                                  valid_shape=[cur_bs, HALF_D])
            sin_half = pypto.view(sin_tile, [BS_TILE, HALF_D], [0, 0],
                                  valid_shape=[cur_bs, HALF_D])
            cos_half_fp32 = pypto.cast(cos_half, pypto.DT_FP32)
            sin_half_fp32 = pypto.cast(sin_half, pypto.DT_FP32)
            cos_b = pypto.reshape(cos_half_fp32, [BS_TILE, 1, HALF_D], inplace=True)
            sin_b = pypto.reshape(sin_half_fp32, [BS_TILE, 1, HALF_D], inplace=True)

            # RoPE: split into halves, rotate, concat
            pypto.set_vec_tile_shapes(BS_TILE, N, HALF_D)
            x_left  = pypto.view(normed_fp32, [BS_TILE, N, HALF_D], [0, 0, 0],
                                 valid_shape=[cur_bs, N, HALF_D])
            x_right = pypto.view(normed_fp32, [BS_TILE, N, HALF_D], [0, 0, HALF_D],
                                 valid_shape=[cur_bs, N, HALF_D])
            o1 = pypto.sub(pypto.mul(x_left, cos_b), pypto.mul(x_right, sin_b))
            o2 = pypto.add(pypto.mul(x_right, cos_b), pypto.mul(x_left, sin_b))
            roped = pypto.concat([o1, o2], 2)                          # [BS_TILE, N, D] FP32
            roped_bf = pypto.cast(roped, pypto.DT_BF16)

            out[bs_idx * BS_TILE: bs_idx * BS_TILE + BS_TILE, :, :] = roped_bf

    return kernel


qwen3_qk_rope_q = _make_qk_rope_kernel(16)   # for Q (Nq=16)
qwen3_qk_rope_k = _make_qk_rope_kernel(8)    # for K (Nkv=8)
