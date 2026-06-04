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
Flash Attention Forward with Dynamic Variable Length Sequences

语义约定:
  - Q 侧: s1_size (Q seqlen), 张量包括 Q/O/L/M
  - KV 侧: s2_size (KV seqlen), 张量包括 K/V
  - Q_TILE/K_TILE: 序列维度的分块大小 (将 s1/s2 切分为多个 tile 迭代)

4 loops: batch + head + q_tile + kv_tile.
Tiles Q and KV sequence dimensions by Q_TILE/K_TILE to reduce intermediate
attention matrix from [s1_size, s2_size] to [Q_TILE, K_TILE] per iteration.
O, L, M are accumulated across kv tiles (online softmax algorithm).
"""

import pypto


Q_TILE = 320
K_TILE = 320


@pypto.frontend.jit(
    debug_options={
        "runtime_debug_mode": 0,
    },
    runtime_options={
        "device_sched_mode": 0,
        "stitch_function_max_num": 1024,
    },
    pass_options={
        "cube_l1_reuse_setting": {-1: 8},
        "vec_nbuffer_setting": {-1: 8},
        "cube_nbuffer_setting": {-1: 8},
    },
)
def flash_attention_varlen_forward_kernel(
    # Q侧输入: shape=[total_q, N, D], total_q=DYNAMIC, N=num_heads, D=head_dim
    q: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    # KV侧输入: shape=[total_kv, N, D]
    k: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    v: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    # Q侧输出: shape=[total_q, hidden_dim] (二维)
    output: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    # Q侧: softmax中间量L, shape=[total_q, n] (二维)
    l_output: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    # Q侧: softmax中间量M, shape=[total_q, n] (二维)
    m_output: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    # 累积序列长度: shape=[batch_size + 1]
    cu_seqlens_q: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    cu_seqlens_k: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
):
    """
    Flash Attention Forward - 4 loops (batch + head + q_tile + kv_tile).

    输入张量为三维 [total_seq, N, D] (total_seq=DYNAMIC, N=num_heads, D=head_dim),
    输出张量为二维 [total_seq, hidden_dim] (hidden_dim = N*D)。
    
    Kernel入口处从输入tensor shape 获取 num_heads 和 head_dim:
      num_heads = q.shape[1]
      head_dim = q.shape[2]
      hidden_dim = num_heads * head_dim
    
    然后 reshape inplace 输入为二维 [total_seq, hidden_dim] 以便按 head 做 view 切片，
    输出保持二维布局。

    张量布局 (Q: s1_size, KV: s2_size):
      Q 侧: Q/O/L/M — 每个 batch 占用 s1 行 (通过 cu_seqlens_q 定位)
      KV 侧: K/V    — 每个 batch 占用 s2_total 行 (通过 cu_seqlens_k 定位)

    cu_seqlens_q.shape[0] = batch_size + 1
    cu_seqlens_q[i]       = 前 i 个 batch 的累积 Q seqlen
    cu_seqlens_k[i]       = 前 i 个 batch 的累积 KV seqlen

    计算流程 (per batch, per head, per q_tile, per kv_tile):
      对 Q seqlen 按 q_tile 分块, 对 KV seqlen 按 k_tile 分块:
        S_tile = Q_tile @ K_tile^T * scale         [sq, sk]       BF16→FP32
        M_tile = max(S_tile, dim=-1)               [sq, 1]        FP32
        P_tile = exp(S_tile - M_tile)              [sq, sk]       FP32
        L_tile = sum(P_tile, dim=-1)               [sq, 1]        FP32
        P_norm = P_tile / L_tile                   [sq, sk]       FP32→BF16
        O_tile = P_norm @ V_tile                   [sq, D]        BF16
      O/L/M 在 kv_tile 循环中累加 (online softmax), 最终写回。

    Kernel dtype 转换流程:
      1. Q/K/V: BF16 输入
      2. scores = Q(BF16) @ K^T(BF16) → FP32      (matmul out_dtype=FP32)
      3. scores_scaled = scores * scale → FP32    (mul)
      4. M = max(scores_scaled) → FP32            (amax)
      5. P = exp(scores_scaled - M) → FP32        (exp after sub)
      6. L = sum(P) → FP32                        (sum)
      7. P_norm = P / L → FP32                    (div)
      8. P_bf16 = cast(P_norm, BF16)              (pypto.cast → DT_BF16)
      9. O = P_bf16 @ V(BF16) → BF16              (matmul out_dtype=BF16)
     10. L, M 保持 FP32 写回
    """
    # ---- 从三维输入获取 N(num_heads) 和 D(head_dim), 然后 reshape 为二维 ----
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    hidden_dim = num_heads * head_dim
    total_q = q.shape[0]
    total_kv = k.shape[0]
    scale = 1.0 / (head_dim ** 0.5)

    # reshape inplace: q/k/v [total_seq, N, D] → [total_seq, N*D]
    q_2d = pypto.reshape(q, [total_q, hidden_dim], inplace=True)
    k_2d = pypto.reshape(k, [total_kv, hidden_dim], inplace=True)
    v_2d = pypto.reshape(v, [total_kv, hidden_dim], inplace=True)
    # output/l/m 保持二维，无需 reshape

    v1_tile = [64, 512]
    v2_tile = [512, 64]

    q_tile = Q_TILE
    k_tile = K_TILE

    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_cube_tile_shapes([128, 128], [128, 256], [128, 128])
    pypto.set_vec_tile_shapes(64, 256)

    # 累计Q序列长度 batch_size + 1
    batch_size = cu_seqlens_q.shape[0] - 1
    for b_idx in pypto.loop(batch_size, name="batch_loop"):
        q_start = cu_seqlens_q[b_idx]
        q_end = cu_seqlens_q[b_idx + 1]
        seq_len_q = q_end - q_start
        seq_len_q.as_variable()

        k_start = cu_seqlens_k[b_idx]
        k_end = cu_seqlens_k[b_idx + 1]
        seq_len_k = k_end - k_start
        seq_len_k.as_variable()

        q_tile_count = (seq_len_q + q_tile - 1) // q_tile
        k_tile_count = (seq_len_k + k_tile - 1) // k_tile

        h_num = num_heads // 2
        for h_idx in pypto.loop(h_num, name="head_loop"):

            for q_tile_idx in pypto.loop(q_tile_count, name="q_tile_loop"):
                # 创建2个head独立的累加器
                oi_update_0 = pypto.tensor([q_tile, head_dim], pypto.DT_FP32, "oi_update")
                li_update_0 = pypto.tensor([q_tile, 1], pypto.DT_FP32, "li_update")
                mi_update_0 = pypto.tensor([q_tile, 1], pypto.DT_FP32, "mi_update")
                oi_update_1 = pypto.tensor([q_tile, head_dim], pypto.DT_FP32, "oi_update")
                li_update_1 = pypto.tensor([q_tile, 1], pypto.DT_FP32, "li_update")
                mi_update_1 = pypto.tensor([q_tile, 1], pypto.DT_FP32, "mi_update")

                q_tile_start = q_tile_idx * q_tile
                q_tile_end = pypto.min(q_tile_start + q_tile, seq_len_q)
                q_tile_len = q_tile_end - q_tile_start

                for k_tile_idx in pypto.loop(k_tile_count, name="k_tile_loop"):
                    k_tile_start = k_tile_idx * k_tile
                    k_tile_end = pypto.min(k_tile_start + k_tile, seq_len_k)
                    k_tile_len = k_tile_end - k_tile_start

                    for h_s_idx in range(2):
                        h_act_idx = h_idx * 2 + h_s_idx
                        h_offset = h_act_idx * head_dim

                        if h_s_idx == 0:
                            li_update = li_update_0
                            mi_update = mi_update_0
                            oi_update = oi_update_0
                        else:
                            li_update = li_update_1
                            mi_update = mi_update_1
                            oi_update = oi_update_1

                        q_tile_view = pypto.view(q_2d, [q_tile, head_dim],
                                            [q_start + q_tile_start, h_offset],
                                            valid_shape=[q_tile_len, head_dim])

                        k_tile_view = pypto.view(k_2d, [k_tile, head_dim],
                                            [k_start + k_tile_start, h_offset],
                                            valid_shape=[k_tile_len, head_dim])
                        v_tile_view = pypto.view(v_2d, [k_tile, head_dim],
                                            [k_start + k_tile_start, h_offset],
                                            valid_shape=[k_tile_len, head_dim])

                        pypto.set_cube_tile_shapes([64, 512], [64, 64], [512, 512])

                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])

                        if pypto.platform.npuarch == 'DAV_3510':
                            pypto.set_pass_options(sg_set_scope=5001)

                        scores = pypto.matmul(q_tile_view, k_tile_view, out_dtype=pypto.DT_FP32, b_trans=True)

                        scores_scaled = pypto.mul(scores, scale)
                        mij = pypto.amax(scores_scaled, dim=-1, keepdim=True)
                        s_shifted = pypto.sub(scores_scaled, mij)
                        pij = pypto.exp(s_shifted)
                        lij = pypto.sum(pij, dim=-1, keepdim=True)

                        pypto.set_cube_tile_shapes([128, 512], [256, 512], [64, 64])

                        if pypto.is_loop_begin(k_tile_idx):
                            if pypto.is_loop_end(k_tile_idx):
                                pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                                pij_div = pypto.div(pij, lij, precision_type=pypto.PrecisionType.INTRINSIC)
                                pij_bf16 = pypto.cast(pij_div, pypto.DT_BF16)

                                oij = pypto.matmul(pij_bf16, v_tile_view, out_dtype=pypto.DT_BF16)

                                if pypto.platform.npuarch == 'DAV_3510':
                                    pypto.set_pass_options(sg_set_scope=-1)

                                pypto.assemble(lij, [q_start + q_tile_start, h_act_idx], l_output)
                                pypto.assemble(mij, [q_start + q_tile_start, h_act_idx], m_output)
                                pypto.assemble(oij, [q_start + q_tile_start, h_offset], output)

                            else:
                                pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                                pij_bf16 = pypto.cast(pij, pypto.DT_BF16)

                                oij = pypto.matmul(pij_bf16, v_tile_view, out_dtype=pypto.DT_FP32)

                                if pypto.platform.npuarch == 'DAV_3510':
                                    pypto.set_pass_options(sg_set_scope=-1)

                                oi_update[:] = oij
                                li_update[:] = lij
                                mi_update[:] = mij
                        else:
                            pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                            pij_bf16 = pypto.cast(pij, pypto.DT_BF16)

                            oij = pypto.matmul(pij_bf16, v_tile_view, out_dtype=pypto.DT_FP32)

                            if pypto.platform.npuarch == 'DAV_3510':
                                pypto.set_pass_options(sg_set_scope=-1)

                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])

                            li = pypto.view(li_update, [q_tile, 1], [0, 0], valid_shape=[q_tile_len, 1])
                            mi = pypto.view(mi_update, [q_tile, 1], [0, 0], valid_shape=[q_tile_len, 1])
                            oi = pypto.view(oi_update, [q_tile, head_dim], [0, 0], valid_shape=[q_tile_len, head_dim])

                            mi_new = pypto.maximum(mi, mij)
                            t1 = pypto.sub(mi, mi_new)
                            t2 = pypto.exp(t1)
                            t3 = pypto.sub(mij, mi_new)
                            t4 = pypto.exp(t3)

                            li_new = pypto.add(pypto.mul(t2, li), pypto.mul(t4, lij))
                            oi_tmp = pypto.add(pypto.mul(oi, t2), pypto.mul(oij, t4))

                            if pypto.is_loop_end(k_tile_idx):
                                out_fp32 = pypto.div(oi_tmp, li_new, precision_type=pypto.PrecisionType.INTRINSIC)
                                out_bf16 = pypto.cast(out_fp32, pypto.DT_BF16)

                                pypto.assemble(li_new, [q_start + q_tile_start, h_act_idx], l_output)
                                pypto.assemble(mi_new, [q_start + q_tile_start, h_act_idx], m_output)
                                pypto.assemble(out_bf16, [q_start + q_tile_start, h_offset], output)
                            else: 
                                oi_update[:] = oi_tmp
                                li_update[:] = li_new
                                mi_update[:] = mi_new
