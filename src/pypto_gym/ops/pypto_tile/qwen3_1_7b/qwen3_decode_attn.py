#!/usr/bin/env python3
# coding: utf-8
# Decode attention v5: 3D batched matmul over Nkv axis, GQA-native.
# Single batched matmul does ALL Nkv KV groups in parallel.
#
# Inputs:
#   q       [Nq=16, D=128]              == reshaped to [Nkv=8, GROUPS=2, D]
#   k_full  [Nkv=8, Skv_padded, D=128]
#   v_full  [Nkv=8, Skv_padded, D=128]
#   mask    [Skv_padded]                FP32
#   out     [Nq=16, D=128]              == reshaped from [Nkv, GROUPS, D]

import math
import pypto

Nq = 16
Nkv = 8
GROUPS = Nq // Nkv
D = 128
SCALE = 1.0 / math.sqrt(D)
S2_TILE = 64


@pypto.frontend.jit(
    runtime_options={
        "stitch_function_max_num": 128, 
        "device_sched_mode": 1
    },
    pass_options={
        "cube_l1_reuse_setting": {0: 4}
    }
)
def qwen3_decode_attn(
    q:       pypto.Tensor([Nq, D], pypto.DT_BF16),
    k_full:  pypto.Tensor([Nkv, pypto.DYNAMIC, D], pypto.DT_BF16),
    v_full:  pypto.Tensor([Nkv, pypto.DYNAMIC, D], pypto.DT_BF16),
    mask:    pypto.Tensor([Nkv, GROUPS, pypto.DYNAMIC], pypto.DT_FP32),
    out:     pypto.Tensor([Nq, D], pypto.DT_BF16),
):
    Skv = k_full.shape[1]
    s2_loop = (Skv + S2_TILE - 1) // S2_TILE

    # q [Nq, D] -> [Nkv, GROUPS, D] (logical reorder; consecutive Q heads share KV)
    pypto.set_vec_tile_shapes(Nkv, GROUPS, D)
    q_3d_buf = pypto.tensor([Nkv, GROUPS, D], pypto.DT_BF16, "q_3d_buf")
    q_3d_buf[:] = pypto.reshape(q, [Nkv, GROUPS, D])

    # Online accumulators
    oi = pypto.tensor([Nkv, GROUPS, D], pypto.DT_FP32, "oi")
    li = pypto.tensor([Nkv, GROUPS, 1], pypto.DT_FP32, "li")
    mi = pypto.tensor([Nkv, GROUPS, 1], pypto.DT_FP32, "mi")

    for s2_idx in pypto.loop(s2_loop, name="LOOP_S2", idx_name="s2_idx",
                             unroll_list=[4, 2, 1]):
        s2_start = s2_idx * S2_TILE

        # Stage K and V tiles directly: [Nkv, S2_TILE, D]
        pypto.set_vec_tile_shapes(Nkv, S2_TILE, D)
        k_3d_buf = pypto.tensor([Nkv, S2_TILE, D], pypto.DT_BF16, "k_3d_buf")
        k_3d_buf[:] = pypto.view(k_full, [Nkv, S2_TILE, D],
                                 [0, s2_start, 0])
        v_3d_buf = pypto.tensor([Nkv, S2_TILE, D], pypto.DT_BF16, "v_3d_buf")
        v_3d_buf[:] = pypto.view(v_full, [Nkv, S2_TILE, D],
                                 [0, s2_start, 0])

        # Batched 3D matmul: q [Nkv, GROUPS, D] @ k [Nkv, S2_TILE, D]^T → [Nkv, GROUPS, S2_TILE]
        pypto.set_cube_tile_shapes([16, 16], [128, 128], [64, 64])
        sij = pypto.matmul(q_3d_buf, k_3d_buf, pypto.DT_FP32, b_trans=True)

        # Softmax + mask (mask comes in pre-broadcast as [Nkv, GROUPS, Skv_p])
        pypto.set_vec_tile_shapes(Nkv, GROUPS, S2_TILE)
        sij_scaled = pypto.mul(sij, SCALE)
        mask_tile = pypto.view(mask, [Nkv, GROUPS, S2_TILE],
                               [0, 0, s2_start])
        sij_scaled = pypto.add(sij_scaled, mask_tile)
        m_ij = pypto.amax(sij_scaled, -1, keepdim=True)
        p_ij = pypto.exp(pypto.sub(sij_scaled, m_ij))
        l_ij = pypto.sum(p_ij, -1, keepdim=True)

        # Batched 3D matmul: p [Nkv, GROUPS, S2_TILE] @ v [Nkv, S2_TILE, D] → [Nkv, GROUPS, D]
        p_buf = pypto.tensor([Nkv, GROUPS, S2_TILE], pypto.DT_BF16, "p_buf")
        p_buf[:] = pypto.cast(p_ij, pypto.DT_BF16)
        pypto.set_cube_tile_shapes([16, 16], [64, 64], [128, 128])
        o_ij = pypto.matmul(p_buf, v_3d_buf, pypto.DT_FP32)

        pypto.set_vec_tile_shapes(Nkv, GROUPS, D)
        if pypto.is_loop_begin(s2_idx):
            if pypto.is_loop_end(s2_idx):
                o_final = pypto.div(o_ij, l_ij)
                out[:] = pypto.reshape(pypto.cast(o_final, pypto.DT_BF16), [Nq, D])
            else:
                oi[:] = o_ij
            li[:] = l_ij
            mi[:] = m_ij
        else:
            mi_new = pypto.maximum(mi, m_ij)
            alpha = pypto.exp(pypto.sub(mi, mi_new))
            beta = pypto.exp(pypto.sub(m_ij, mi_new))
            li_new = pypto.add(pypto.mul(alpha, li), pypto.mul(beta, l_ij))
            oi_new = pypto.add(pypto.mul(oi, alpha), pypto.mul(o_ij, beta))
            if pypto.is_loop_end(s2_idx):
                o_final = pypto.div(oi_new, li_new)
                out[:] = pypto.reshape(pypto.cast(o_final, pypto.DT_BF16), [Nq, D])
            else:
                oi[:] = oi_new
            li[:] = li_new
            mi[:] = mi_new
