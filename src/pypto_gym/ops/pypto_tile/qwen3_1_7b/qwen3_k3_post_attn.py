#!/usr/bin/env python3
# coding: utf-8
# K3: post-attention block — O-proj + residual + post-RMSNorm + SwiGLU MLP + residual.
# All inputs/outputs are 2D [S, *] — avoids the matmul→3D-reshape problem.
#
#   attn_in [S, Nq*D=2048] BF16
#   x_res   [S, H=2048]   BF16  (residual from layer input)
#   →  o = matmul(attn_in, Wo)              [S, H]
#      h1 = x_res + o                        [S, H]
#      n2 = RMSNorm(h1, w_post_norm)         [S, H]
#      gate = matmul(n2, Wgate)              [S, INT_SIZE]
#      up   = matmul(n2, Wup)                [S, INT_SIZE]
#      mlp_h = silu(gate) * up               [S, INT_SIZE]
#      down = matmul(mlp_h, Wdown)           [S, H]
#      y    = h1 + down                      [S, H]

import math
import pypto

H = 2048
INT_SIZE = 6144
EPS = 1e-6
BS_TILE = 8


@pypto.frontend.jit(
    runtime_options={
        "stitch_function_max_num": 128, 
        "device_sched_mode": 1
    }
)
def qwen3_post_attn_k3(
    attn_in:     pypto.Tensor([pypto.DYNAMIC, H], pypto.DT_BF16),       # [S, 2048] from attention output
    x_res:       pypto.Tensor([pypto.DYNAMIC, H], pypto.DT_BF16),       # [S, 2048] layer-input residual
    Wo:          pypto.Tensor([H, H], pypto.DT_BF16),                   # [2048, 2048] (already transposed)
    w_post_norm: pypto.Tensor([H], pypto.DT_BF16),                      # [2048]
    Wgate:       pypto.Tensor([H, INT_SIZE], pypto.DT_BF16),            # [2048, 6144]
    Wup:         pypto.Tensor([H, INT_SIZE], pypto.DT_BF16),
    Wdown:       pypto.Tensor([INT_SIZE, H], pypto.DT_BF16),
    y:           pypto.Tensor([pypto.DYNAMIC, H], pypto.DT_BF16),       # [S, 2048] output
):
    S = attn_in.shape[0]
    bs_loop = (S + BS_TILE - 1) // BS_TILE
    h_mean_coff = 1.0 / H

    pypto.set_vec_tile_shapes(1, H)
    w_pn_2d = pypto.reshape(w_post_norm, [1, H], inplace=True)
    w_pn_fp32 = pypto.cast(w_pn_2d, pypto.DT_FP32)

    for bs_idx in pypto.loop(bs_loop, name="LOOP_BS_K3", idx_name="bs_idx"):
        cur_bs = (S - bs_idx * BS_TILE).min(BS_TILE)

        attn_tile = pypto.view(attn_in, [BS_TILE, H], [bs_idx * BS_TILE, 0],
                               valid_shape=[cur_bs, H])
        xres_tile = pypto.view(x_res, [BS_TILE, H], [bs_idx * BS_TILE, 0],
                               valid_shape=[cur_bs, H])

        # --- O proj ---
        attn_buf = pypto.tensor([BS_TILE, H], pypto.DT_BF16, "attn_buf")
        attn_buf[:] = attn_tile
        pypto.set_cube_tile_shapes([32, 32], [128, 512], [128, 128])
        o_fp32 = pypto.matmul(attn_buf, Wo, pypto.DT_FP32)              # [BS_TILE, H] FP32

        # --- residual1 = x_res + o ---
        pypto.set_vec_tile_shapes(1, H)
        xres_fp32 = pypto.cast(xres_tile, pypto.DT_FP32)
        h1_fp32 = pypto.add(xres_fp32, o_fp32)                          # [BS_TILE, H] FP32
        h1_bf = pypto.cast(h1_fp32, pypto.DT_BF16)

        # --- post-RMSNorm on h1 ---
        sq = pypto.mul(h1_fp32, h1_fp32)
        mean = pypto.sum(sq, -1, keepdim=True)
        mean = pypto.mul(mean, h_mean_coff)
        rsqrt = pypto.rsqrt(pypto.add(mean, EPS))
        n2_fp32 = pypto.mul(pypto.mul(h1_fp32, rsqrt), w_pn_fp32)
        n2_buf = pypto.tensor([BS_TILE, H], pypto.DT_BF16, "n2_buf")
        n2_buf[:] = pypto.cast(n2_fp32, pypto.DT_BF16)

        # --- MLP gate / up ---
        pypto.set_cube_tile_shapes([32, 32], [128, 512], [256, 256])
        gate_fp32 = pypto.matmul(n2_buf, Wgate, pypto.DT_FP32)          # [BS_TILE, INT_SIZE] FP32
        up_fp32   = pypto.matmul(n2_buf, Wup,   pypto.DT_FP32)          # [BS_TILE, INT_SIZE] FP32

        # --- SiLU(gate) * up  (in FP32) ---
        pypto.set_vec_tile_shapes(1, INT_SIZE)
        neg_g = pypto.mul(gate_fp32, -1.0)
        exp_ng = pypto.exp(neg_g)
        denom = pypto.add(exp_ng, 1.0)
        silu_g = pypto.div(gate_fp32, denom)                            # silu(g) = g / (1 + e^-g)
        mlp_fp32 = pypto.mul(silu_g, up_fp32)                           # [BS_TILE, INT_SIZE] FP32
        mlp_buf = pypto.tensor([BS_TILE, INT_SIZE], pypto.DT_BF16, "mlp_buf")
        mlp_buf[:] = pypto.cast(mlp_fp32, pypto.DT_BF16)

        # --- down proj + residual2 ---
        pypto.set_cube_tile_shapes([32, 32], [128, 512], [128, 128])
        down_fp32 = pypto.matmul(mlp_buf, Wdown, pypto.DT_FP32)         # [BS_TILE, H] FP32
        pypto.set_vec_tile_shapes(1, H)
        h1_fp32_again = pypto.cast(h1_bf, pypto.DT_FP32)
        y_fp32 = pypto.add(h1_fp32_again, down_fp32)
        y_bf = pypto.cast(y_fp32, pypto.DT_BF16)

        y[bs_idx * BS_TILE: bs_idx * BS_TILE + BS_TILE, :] = y_bf
