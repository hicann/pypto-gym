#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Flash Attention Backward with Dynamic Variable Length Sequences

语义约定:
  - Q 侧: s1_size (Q seqlen), 张量包括 Q/O/dO/L/M/dQ
  - KV 侧: s2_size (KV seqlen), 张量包括 K/V/dK/dV
  - S2_TILE: KV 序列维度的分块大小 (将 s2_size 切分为多个 tile 迭代)

3 loops: batch + head + kv_tile (KV sequence tiling).
Tiles KV sequence dimension by S2_TILE to reduce intermediate attention matrix
from [s1_size, s2_size] to [s1_size, S2_TILE] per iteration.
dK and dV are accumulated across kv tiles.
"""

import os

import torch

import pypto


NUM_HEADS = 8
HEAD_DIM = 64
HIDDEN_DIM = NUM_HEADS * HEAD_DIM

# KV 序列维度的分块大小 (全局配置常量)
S2_TILE = 320


@pypto.frontend.jit(
    pass_options={
        "cube_nbuffer_setting": {-1: 8},
        "vec_nbuffer_setting": {-1: 16},
        "cube_l1_reuse_setting": {-1: 8},
    },
    runtime_options={
        "stitch_function_max_num": 256,
        "device_sched_mode": 0,
    },
    debug_options={
        "runtime_debug_mode": 1
    }
)
def flash_attention_varlen_backward_kernel(
    # Q 侧输入: shape=[bs, N, D], bs=DYNAMIC, N=num_heads, D=head_dim
    q: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    # KV 侧输入: shape=[bs, N, D]
    k: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    v: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    # Q 侧: 前向输出 O, shape=[bs, N, D]
    o: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    # Q 侧: dO, shape=[bs, N, D]
    do: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    # Q 侧: softmax 中间量 L, shape=[bs, N, 1]
    l_input: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    # Q 侧: softmax 中间量 M, shape=[bs, N, 1]
    m_input: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    # Q 侧输出: dQ, shape=[bs, N*D] (二维)
    dq: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    # KV 侧输出: dK, shape=[bs, N*D] (二维)
    dk: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    # KV 侧输出: dV, shape=[bs, N*D] (二维)
    dv: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    # actual_q: shape=[batch_size], actual_q[i]=第 i 个 batch 的 Q seqlen (s1_size)
    actual_q: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    # actual_kv: shape=[batch_size], actual_kv[i]=第 i 个 batch 的 KV seqlen (s2_size)
    actual_kv: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
):
    """
    Flash Attention Backward - 4 loops (batch + head + q_tile + kv_tile).

    输入张量为三维 [bs, N, D] (bs=DYNAMIC, N=num_heads, D=head_dim),
    进入循环前 reshape inplace 为 [bs, N*D] 以便按 head 做 view 切片。

    张量布局 (Q: s1_size, KV: s2_size):
      Q 侧: Q/O/dO/L/M/dQ — 每个 batch 占用 s1 行 (实际 Q seqlen, 从 actual_q 获取)
      KV 侧: K/V/dK/dV    — 每个 batch 占用 s2_total 行 (实际 KV seqlen, 从 actual_kv 获取)

    actual_q.shape[0]  = batch_size
    actual_q[i]  = 第 i 个 batch 的 Q seqlen
    actual_kv[i] = 第 i 个 batch 的 KV seqlen

    计算流程 (per batch, per head, per q_tile, per kv_tile):
      对 Q seqlen 按 S2_TILE 分块, 对 KV seqlen 按 S2_TILE 分块:
        S_tile = Q_tile @ K_tile^T * scale         [sq, s2]       BF16→FP32
        P_tile = exp(S*scale - M) / L               [sq, s2]       FP32
        dP_tile = dO_tile @ V_tile^T                [sq, s2]       BF16→FP32
        D = sum(O_tile * dO_tile, dim=-1)           [sq, 1]        BF16→FP32
        dS_tile = P * (dP - D)                      [sq, s2]       FP32→BF16
        dK_tile += dS^T @ Q_tile * scale            [s2, D]        BF16→FP32→BF16
        dV_tile += P^T @ dO_tile                    [s2, D]        BF16→BF16
        dQ_partial = dS @ K_tile * scale            [sq, D]        BF16→FP32
      dQ 在 kv_tile 循环中累加 (FP32), 最终 cast 为 BF16 写回。
    """
    pypto.experimental.set_operation_options(combine_axis=True)

    # ---- 从三维输入获取 N(num_heads) 和 D(head_dim), 然后 reshape 为二维 ----
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    hidden_dim = num_heads * head_dim
    bs = q.shape[0]
    scale = 1.0 / (head_dim ** 0.5)

    # reshape inplace: q/k/v/o/do [bs, N, D] → [bs, N*D]
    q_2d = pypto.reshape(q, [bs, hidden_dim], inplace=True)
    k_2d = pypto.reshape(k, [bs, hidden_dim], inplace=True)
    v_2d = pypto.reshape(v, [bs, hidden_dim], inplace=True)
    o_2d = pypto.reshape(o, [bs, hidden_dim], inplace=True)
    do_2d = pypto.reshape(do, [bs, hidden_dim], inplace=True)
    # reshape inplace: l_input/m_input [bs, N, 1] → [bs, N]
    l_input_2d = pypto.reshape(l_input, [bs, num_heads], inplace=True)
    m_input_2d = pypto.reshape(m_input, [bs, num_heads], inplace=True)
    # dq/dk/dv 输入时已经是 [bs, N*D], 无需 reshape

    # bs = actual_q 的 shape (即 batch_size)
    bs = actual_q.shape[0]
    for b_idx in pypto.loop(bs, name="batch_loop", parallel=True):
        # s1: 当前 batch 的 Q seqlen (从 actual_q 获取, 不使用全局变量)
        s1 = actual_q[b_idx]
        s1.as_variable()
        # s2_total: 当前 batch 的 KV seqlen (从 actual_kv 获取, 不使用全局变量)
        s2_total = actual_kv[b_idx]
        s2_total.as_variable()

        # Q 侧偏移: 每个 batch 占 s1 行
        q_start = b_idx * s1
        # KV 侧偏移: 每个 batch 占 s2_total 行
        kv_start = b_idx * s2_total

        # Q seqlen 按 S2_TILE 分块的 tile 数量
        num_q_tiles = (s1 + S2_TILE - 1) // S2_TILE
        num_q_tiles.as_variable()
        # KV seqlen 按 S2_TILE 分块的 tile 数量 (S2_TILE 为全局配置常量)
        num_kv_tiles = (s2_total + S2_TILE - 1) // S2_TILE
        num_kv_tiles.as_variable()

        head_num_loop = num_heads // 2
        for h_idx in pypto.loop(head_num_loop, name="head_loop"):

            for q_tile_idx in pypto.loop(num_q_tiles, name="q_tile_loop"):
                # ---- Q 侧: 当前 Q tile 的偏移和有效长度 ----
                q_ofs = q_tile_idx * S2_TILE
                sq = (s1 - q_ofs).min(S2_TILE)
                sq.as_variable()

                # dQ 累加器 (FP32), shape=[S2_TILE, head_dim], 跨 kv_tile 累加
                dq_update = pypto.tensor([S2_TILE, head_dim], pypto.DT_FP32, "dq_update")

                for kv_tile_idx in pypto.loop(num_kv_tiles, name="kv_tile_loop"):

                    for h_s_idx in range(2):
                        h_ofs = (h_idx * 2 + h_s_idx) * head_dim
                        # L/M reshape 后为 [bs, N], head 索引不乘 head_dim
                        h_idx_lm = h_idx * 2 + h_s_idx

                        # ---- Q 侧: 加载当前 Q tile 的 Q/O/dO/M/L ----
                        # q/o/do reshape 后为 [bs, N*head_dim], 静态 shape=[S2_TILE, head_dim]
                        qi = pypto.view(q_2d, [S2_TILE, head_dim], [q_start + q_ofs, h_ofs],
                                        valid_shape=[sq, head_dim])
                        oi = pypto.view(o_2d, [S2_TILE, head_dim], [q_start + q_ofs, h_ofs],
                                        valid_shape=[sq, head_dim])
                        doi = pypto.view(do_2d, [S2_TILE, head_dim], [q_start + q_ofs, h_ofs],
                                        valid_shape=[sq, head_dim])
                        # l_input/m_input reshape 后为 [bs, N], 每个 head 1 个值
                        m_i = pypto.view(m_input_2d, [S2_TILE, 1], [q_start + q_ofs, h_idx_lm],
                                        valid_shape=[sq, 1])
                        l_i = pypto.view(l_input_2d, [S2_TILE, 1], [q_start + q_ofs, h_idx_lm],
                                        valid_shape=[sq, 1])

                        # ---- KV 侧: 当前 KV tile 的偏移和有效长度 ----
                        s2_ofs = kv_tile_idx * S2_TILE
                        s2 = (s2_total - s2_ofs).min(S2_TILE)
                        s2.as_variable()

                        pypto.set_pass_options(sg_set_scope=1)
                        # KV 侧: 加载 K_tile, V_tile (静态 S2_TILE, 有效 s2)
                        ki_tile = pypto.view(k_2d, [S2_TILE, head_dim], [kv_start + s2_ofs, h_ofs],
                                             valid_shape=[s2, head_dim])
                        vi_tile = pypto.view(v_2d, [S2_TILE, head_dim], [kv_start + s2_ofs, h_ofs],
                                             valid_shape=[s2, head_dim])

                        # 计算公式： head_dim = sum(O * dO, dim=-1, keepdim=True) -> [sq, 1]
                        # dtype: BF16 → cast → FP32, mul, sum
                        pypto.set_vec_tile_shapes(512, 64)
                        oi_fp32 = pypto.cast(oi, pypto.DT_FP32)
                        doi_fp32 = pypto.cast(doi, pypto.DT_FP32)
                        do_o = pypto.mul(oi_fp32, doi_fp32)
                        d_tile = pypto.sum(do_o, dim=-1, keepdim=True)
                        pypto.set_pass_options(sg_set_scope=-1)

                        # 计算公式： dP_tile = dO_tile @ V_tile^T -> [sq, s2]
                        # 数据类型转换： dtype: BF16 matmul → FP32
                        pypto.set_cube_tile_shapes([128, 512], [64, 64], [256, 512])
                        dp = pypto.matmul(doi, vi_tile, out_dtype=pypto.DT_FP32, b_trans=True)
                        pypto.set_vec_tile_shapes(64, 512)
                        dp = pypto.view(dp, [S2_TILE, S2_TILE], [0, 0], valid_shape=[sq, s2])

                        # 计算公式：S_tile = Q_tile @ K_tile^T -> [sq, s2]
                        # 数据类型转换：dtype: BF16 matmul → FP32
                        pypto.set_cube_tile_shapes([128, 512], [64, 64], [256, 512])
                        scores = pypto.matmul(qi, ki_tile, out_dtype=pypto.DT_FP32, b_trans=True)

                        pypto.set_vec_tile_shapes(64, 512)
                        scores = pypto.view(scores, [S2_TILE, S2_TILE], [0, 0], valid_shape=[sq, s2])

                        # 计算公式： P_tile = exp(S * scale - M) / L  (softmax)
                        # 数据类型转换：dtype: FP32 全程
                        pypto.set_pass_options(sg_set_scope=2)
                        p = pypto.div(pypto.exp(pypto.sub(pypto.mul(scores, scale), m_i)), l_i)
                        # 计算公式： dS_tile = P * (dP - head_dim)
                        # 数据类型转换：dtype: FP32
                        ds = pypto.mul(p, pypto.sub(dp, d_tile))
                        # 数据类型转换：dtype: FP32 → cast → BF16 (用于后续 matmul 输入)
                        ds_half = pypto.cast(ds, pypto.DT_BF16)
                        p_half = pypto.cast(p, pypto.DT_BF16)
                        pypto.set_pass_options(sg_set_scope=-1)

                        # 计算公式： dK_tile += dS^T @ Q_tile * scale -> [s2, head_dim]
                        # 数据类型转换：dtype: BF16 matmul → FP32, mul(scale), cast → BF16
                        pypto.set_cube_tile_shapes([128, 512], [256, 512], [64, 64])
                        dk_tile_mm = pypto.matmul(ds_half, qi, out_dtype=pypto.DT_FP32, a_trans=True)
                        pypto.set_vec_tile_shapes(512, 64)
                        dk_tile = pypto.mul(dk_tile_mm, scale)
                        # KV 侧写回: dK[kv_start + s2_ofs]
                        pypto.assemble(pypto.cast(dk_tile, pypto.DT_BF16), [kv_start + s2_ofs, h_ofs], dk)

                        # 计算公式： dV_tile += P^T @ dO_tile -> [s2, head_dim]
                        # 数据类型转换：dtype: BF16 matmul → BF16 (直接输出 BF16)
                        pypto.set_cube_tile_shapes([128, 512], [256, 512], [64, 64])
                        dv_tile = pypto.matmul(p_half, doi, out_dtype=pypto.DT_BF16, a_trans=True)
                        # KV 侧写回: dV[kv_start + s2_ofs]
                        pypto.assemble(dv_tile, [kv_start + s2_ofs, h_ofs], dv)

                        # 计算公式： dQ_partial = dS @ K_tile * scale -> [sq, head_dim]
                        # 数据类型转换：dtype: BF16 matmul → FP32, mul(scale)
                        pypto.set_cube_tile_shapes([128, 512], [256, 512], [64, 64])
                        dq_partial_mm = pypto.matmul(ds_half, ki_tile, out_dtype=pypto.DT_FP32)
                        dq_partial = pypto.mul(dq_partial_mm, scale)

                        # dQ 在 kv_tile 循环中累加 (FP32)
                        if pypto.is_loop_begin(kv_tile_idx):
                            if pypto.is_loop_end(kv_tile_idx):
                                # 单个 KV tile: 直接 cast → BF16 写回 Q 侧对应 tile 位置
                                pypto.assemble(pypto.cast(dq_partial, pypto.DT_BF16), [q_start + q_ofs, h_ofs], dq)
                            else:
                                dq_update[:] = dq_partial
                        else:
                            dq_new = pypto.add(dq_update, dq_partial)
                            if pypto.is_loop_end(kv_tile_idx):
                                # 最后一个 KV tile: cast → BF16 写回 Q 侧对应 tile 位置
                                pypto.assemble(pypto.cast(dq_new, pypto.DT_BF16), [q_start + q_ofs, h_ofs], dq)
                            else:
                                dq_update[:] = dq_new