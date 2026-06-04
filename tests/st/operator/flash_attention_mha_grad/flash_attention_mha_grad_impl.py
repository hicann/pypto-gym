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
Flash Attention MHA Backward (参考 flash_attention_score_grad_impl.py 实现)

布局 (varlen, 通过 actual_q / actual_kv cumsum 描述每 batch 的 s1 / s2):
  - q/k/v/o/do: [total, num_heads, head_dim] BF16
                Q 侧 total = actual_q[-1], KV 侧 total = actual_kv[-1]
  - l/m:        [total_q, num_heads, 1] FP32
  - dq/dk/dv:   [total, hidden_dim] BF16
  - actual_q/actual_kv: [batch + 1] INT32, 前缀累加
                第 i 个 batch: offset = actual_*[i], seq = actual_*[i+1] - actual_*[i]

两趟 (拆分两个独立 inner loop, 共享外层 batch/head 循环):
  趟1 dQ: 外层 s1_tile, 内层 s2_tile, dQ FP32 累加器在 s2 内层累加
  趟2 dK/dV: 外层 s2_tile, 内层 s1_tile, dK/dV FP32 累加器在 s1 内层累加

关键: s1_loop / s2_loop 基于 per-batch 的 s1 / s2 动态计算 (.as_variable());
is_loop_begin/end 用 Python if 即可。
"""

import pypto


S_TILE_2 = 128
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
        "runtime_debug_mode": 0
    }
)
def flash_attention_varlen_backward_kernel_small_seq(
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
    # actual_q: shape=[batch_size + 1], Q seqlen 的前缀累加 (cumsum)
    actual_q: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    # actual_kv: shape=[batch_size + 1], KV seqlen 的前缀累加 (cumsum)
    actual_kv: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
):
    """
    Flash Attention Backward - 4 loops (batch + head + q_tile + kv_tile).

    输入张量为三维 [bs, N, D] (bs=DYNAMIC, N=num_heads, D=head_dim),
    进入循环前 reshape inplace 为 [bs, N*D] 以便按 head 做 view 切片。

    张量布局 (Q: s1_size, KV: s2_size):
      Q 侧: Q/O/dO/L/M/dQ — 每个 batch 占用 s1 行 (实际 Q seqlen, 从 actual_q 派生)
      KV 侧: K/V/dK/dV    — 每个 batch 占用 s2_total 行 (实际 KV seqlen, 从 actual_kv 派生)

    actual_q.shape[0]  = batch_size + 1, actual_kv.shape[0] = batch_size + 1
    第 i 个 batch:
      Q  起始偏移 = actual_q[i],   s1 = actual_q[i+1]  - actual_q[i]
      KV 起始偏移 = actual_kv[i],  s2 = actual_kv[i+1] - actual_kv[i]
    s1 / s2 tile 循环次数基于 per-batch 的 s1 / s2 动态计算。

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
    # Q 侧 / KV 侧的 flat 总长可能不同 (varlen, 每 batch s1 / s2 不等), 分别取
    total_q = q.shape[0]
    total_kv = k.shape[0]
    scale = 1.0 / (head_dim ** 0.5)

    # reshape inplace: Q 侧 [total_q, N, D] → [total_q, N*D]
    q_2d = pypto.reshape(q, [total_q, hidden_dim], inplace=True)
    o_2d = pypto.reshape(o, [total_q, hidden_dim], inplace=True)
    do_2d = pypto.reshape(do, [total_q, hidden_dim], inplace=True)
    # reshape inplace: KV 侧 [total_kv, N, D] → [total_kv, N*D]
    k_2d = pypto.reshape(k, [total_kv, hidden_dim], inplace=True)
    v_2d = pypto.reshape(v, [total_kv, hidden_dim], inplace=True)
    # reshape inplace: l_input/m_input [total_q, N, 1] → [total_q, N]
    l_input_2d = pypto.reshape(l_input, [total_q, num_heads], inplace=True)
    m_input_2d = pypto.reshape(m_input, [total_q, num_heads], inplace=True)
    # dq/dk/dv 输入时已经是 [total, N*D], 无需 reshape

    # actual_q / actual_kv 为前缀累加, shape=[batch_size + 1], 故 batch_size = shape[0] - 1
    batch_size = actual_q.shape[0] - 1
    for b_idx in pypto.loop(batch_size, name="batch_loop", parallel=True):
        # Q 侧偏移与 seqlen 从 actual_q (cumsum) 派生
        q_start = actual_q[b_idx]
        q_start.as_variable()
        s1 = actual_q[b_idx + 1] - q_start
        s1.as_variable()
        # KV 侧偏移与 seqlen 从 actual_kv (cumsum) 派生
        kv_start = actual_kv[b_idx]
        kv_start.as_variable()
        s2_total = actual_kv[b_idx + 1] - kv_start
        s2_total.as_variable()

        # Q seqlen 按 S2_TILE 分块的 tile 数量 (per-batch 动态)
        num_q_tiles = (s1 + S2_TILE - 1) // S2_TILE
        num_q_tiles.as_variable()
        # KV seqlen 按 S2_TILE 分块的 tile 数量 (per-batch 动态)
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
                        if pypto.platform.npuarch == 'DAV_3510':
                            pypto.set_pass_options(sg_set_scope=10001)
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

                        # KV 侧: 加载 K_tile, V_tile (静态 S2_TILE, 有效 s2)
                        ki_tile = pypto.view(k_2d, [S2_TILE, head_dim], [kv_start + s2_ofs, h_ofs],
                                             valid_shape=[s2, head_dim])
                        vi_tile = pypto.view(v_2d, [S2_TILE, head_dim], [kv_start + s2_ofs, h_ofs],
                                             valid_shape=[s2, head_dim])

                        # 计算公式： head_dim = sum(O * dO, dim=-1, keepdim=True) -> [sq, 1]
                        # dtype: BF16 → cast → FP32, mul, sum
                        if pypto.platform.npuarch == 'DAV_3510':
                            pypto.set_vec_tile_shapes(512, 64)
                        else:
                            pypto.set_vec_tile_shapes(256, 64)
                        oi_fp32 = pypto.cast(oi, pypto.DT_FP32)
                        doi_fp32 = pypto.cast(doi, pypto.DT_FP32)
                        do_o = pypto.mul(oi_fp32, doi_fp32)
                        d_tile = pypto.sum(do_o, dim=-1, keepdim=True)

                        # 计算公式： dP_tile = dO_tile @ V_tile^T -> [sq, s2]
                        # 数据类型转换： dtype: BF16 matmul → FP32
                        pypto.set_cube_tile_shapes([128, 512], [64, 64], [256, 512])
                        dp = pypto.matmul(doi, vi_tile, out_dtype=pypto.DT_FP32, b_trans=True)
                        if pypto.platform.npuarch == 'DAV_3510':
                            pypto.set_vec_tile_shapes(64, 512)
                        else:
                            pypto.set_vec_tile_shapes(64, 256)
                        dp = pypto.view(dp, [S2_TILE, S2_TILE], [0, 0], valid_shape=[sq, s2])

                        # 计算公式：S_tile = Q_tile @ K_tile^T -> [sq, s2]
                        # 数据类型转换：dtype: BF16 matmul → FP32
                        pypto.set_cube_tile_shapes([128, 512], [64, 64], [256, 512])
                        scores = pypto.matmul(qi, ki_tile, out_dtype=pypto.DT_FP32, b_trans=True)

                        if pypto.platform.npuarch == 'DAV_3510':
                            pypto.set_vec_tile_shapes(64, 512)
                        else:
                            pypto.set_vec_tile_shapes(64, 256)
                        scores = pypto.view(scores, [S2_TILE, S2_TILE], [0, 0], valid_shape=[sq, s2])

                        # 计算公式： P_tile = exp(S * scale - M) / L  (softmax)
                        # 数据类型转换：dtype: FP32 全程
                        p = pypto.div(pypto.exp(pypto.sub(pypto.mul(scores, scale), m_i)),
                                      l_i, precision_type=pypto.PrecisionType.INTRINSIC)
                        # 计算公式： dS_tile = P * (dP - head_dim)
                        # 数据类型转换：dtype: FP32
                        ds = pypto.mul(p, pypto.sub(dp, d_tile))
                        # 数据类型转换：dtype: FP32 → cast → BF16 (用于后续 matmul 输入)
                        ds_half = pypto.cast(ds, pypto.DT_BF16)
                        p_half = pypto.cast(p, pypto.DT_BF16)

                        # 计算公式： dK_tile += dS^T @ Q_tile * scale -> [s2, head_dim]
                        # 数据类型转换：dtype: BF16 matmul → FP32, mul(scale), cast → BF16
                        pypto.set_cube_tile_shapes([128, 512], [256, 512], [64, 64])
                        dk_tile_mm = pypto.matmul(ds_half, qi, out_dtype=pypto.DT_FP32, a_trans=True)
                        if pypto.platform.npuarch == 'DAV_3510':
                            pypto.set_vec_tile_shapes(512, 64)
                        else:
                            pypto.set_vec_tile_shapes(256, 64)
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
                        if pypto.platform.npuarch == 'DAV_3510':
                            pypto.set_pass_options(sg_set_scope=-1)


def compute_p_ds(qi, ki, vi, doi, mi, li, d_i, sq, sk, scale, c_tile, v_tile_s):
    """计算单个 (s1_tile, s2_tile) 块的 P_ij 和 dS_ij。"""
    pypto.set_vec_tile_shapes(v_tile_s[0], v_tile_s[1])
    pypto.set_cube_tile_shapes(c_tile[0], c_tile[1], c_tile[2])
    s_ij = pypto.matmul(qi, ki, pypto.DT_FP32, b_trans=True)
    s_ij = pypto.view(s_ij, [S_TILE_2, S_TILE_2], [0, 0], valid_shape=[sq, sk])

    pypto.set_vec_tile_shapes(v_tile_s[0], v_tile_s[1])
    s_ij = pypto.mul(s_ij, scale)
    p_ij = pypto.exp(pypto.sub(s_ij, mi))
    p_ij = pypto.div(p_ij, li, precision_type=pypto.PrecisionType.INTRINSIC)

    pypto.set_vec_tile_shapes(v_tile_s[0], v_tile_s[1])
    pypto.set_cube_tile_shapes(c_tile[0], c_tile[1], c_tile[2])
    dp_ij = pypto.matmul(doi, vi, pypto.DT_FP32, b_trans=True)
    dp_ij = pypto.view(dp_ij, [S_TILE_2, S_TILE_2], [0, 0], valid_shape=[sq, sk])

    pypto.set_vec_tile_shapes(v_tile_s[0], v_tile_s[1])
    ds_ij = pypto.mul(p_ij, pypto.sub(dp_ij, d_i))

    return p_ij, ds_ij


@pypto.frontend.jit(
    runtime_options={
        "stitch_function_max_num": 128,
        "device_sched_mode": 1,
    },
    pass_options={
        "cube_nbuffer_setting": {0: 8},
        "vec_nbuffer_setting": {0: 8},
        "cube_l1_reuse_setting": {0: 8},
    },
    debug_options={
        "runtime_debug_mode": 0,
    },
)
def flash_attention_mha_grad_kernel_long_seq(
    q: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    k: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    v: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    o: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    do: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    l_input: pypto.Tensor([pypto.DYN, ...], pypto.DT_FP32),
    m_input: pypto.Tensor([pypto.DYN, ...], pypto.DT_FP32),
    dq: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    dk: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    dv: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    actual_q: pypto.Tensor([pypto.DYN], pypto.DT_INT32),
    actual_kv: pypto.Tensor([pypto.DYN], pypto.DT_INT32),
):
    """合一 kernel: 两趟 (dQ, dK/dV) 共享外层 batch+head 循环。

    actual_q / actual_kv 为前缀累加, shape=[batch_size + 1], 故 batch_size = shape[0] - 1
    第 i 个 batch: q_start=actual_q[i], s1=actual_q[i+1]-actual_q[i]
                   kv_start=actual_kv[i], s2=actual_kv[i+1]-actual_kv[i]
    s1 / s2 tile 循环次数基于 per-batch 的 s1 / s2 动态计算。
    """
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    hidden_dim = num_heads * head_dim
    total = q.shape[0]
    b = actual_q.shape[0] - 1
    scale = 1.0 / (head_dim ** 0.5)
    pypto.experimental.set_operation_options(combine_axis=True)

    # reshape 三维 → 二维, 便于按 [seq, hidden_dim] 切片
    q_2d = pypto.reshape(q, [total, hidden_dim], inplace=True)
    k_2d = pypto.reshape(k, [total, hidden_dim], inplace=True)
    v_2d = pypto.reshape(v, [total, hidden_dim], inplace=True)
    o_2d = pypto.reshape(o, [total, hidden_dim], inplace=True)
    do_2d = pypto.reshape(do, [total, hidden_dim], inplace=True)
    l_2d = pypto.reshape(l_input, [total, num_heads], inplace=True)
    m_2d = pypto.reshape(m_input, [total, num_heads], inplace=True)

    c_tile = [[S_TILE_2, S_TILE_2], [head_dim, 256], [S_TILE_2, S_TILE_2]]
    v_tile_s = [S_TILE_2, S_TILE_2]
    v_tile_d = [S_TILE_2, head_dim]

    for b_idx in pypto.loop(b, name="LOOP_b", idx_name="b_idx"):
        # per-batch 派生 Q/KV 偏移与 seqlen, 循环次数基于 per-batch 动态值
        q_start = actual_q[b_idx]
        q_start.as_variable()
        s1 = actual_q[b_idx + 1] - q_start
        s1.as_variable()
        kv_start = actual_kv[b_idx]
        kv_start.as_variable()
        s2 = actual_kv[b_idx + 1] - kv_start
        s2.as_variable()

        s1_loop = (s1 + S_TILE_2 - 1) // S_TILE_2
        s1_loop.as_variable()
        s2_loop = (s2 + S_TILE_2 - 1) // S_TILE_2
        s2_loop.as_variable()

        for n_idx in pypto.loop(num_heads, name="LOOP_n", idx_name="n_idx"):
            h_ofs = n_idx * head_dim

            # ===== 趟1: 计算 dQ =====
            for s1_idx in pypto.loop(s1_loop, name="LOOP_s1_dq", idx_name="s1_idx"):
                s1_off = q_start + s1_idx * S_TILE_2
                actual_s1 = (s1 - s1_idx * S_TILE_2).min(S_TILE_2)

                pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                q_i = pypto.view(q_2d, [S_TILE_2, head_dim], [s1_off, h_ofs],
                                 valid_shape=[actual_s1, head_dim])
                do_i = pypto.view(do_2d, [S_TILE_2, head_dim], [s1_off, h_ofs],
                                  valid_shape=[actual_s1, head_dim])
                o_i = pypto.view(o_2d, [S_TILE_2, head_dim], [s1_off, h_ofs],
                                 valid_shape=[actual_s1, head_dim])
                m_i = pypto.view(m_2d, [S_TILE_2, 1], [s1_off, n_idx],
                                 valid_shape=[actual_s1, 1])
                l_i = pypto.view(l_2d, [S_TILE_2, 1], [s1_off, n_idx],
                                 valid_shape=[actual_s1, 1])

                pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                d_i = pypto.sum(pypto.cast(pypto.mul(o_i, do_i), pypto.DT_FP32),
                                -1, keepdim=True)

                dq_acc = pypto.tensor([S_TILE_2, head_dim], pypto.DT_FP32, "dq_acc")

                for s2_idx in pypto.loop(s2_loop, name="LOOP_s2_dq", idx_name="s2_idx",
                                         unroll_list=[8, 4, 2, 1]):
                    s2_off = kv_start + s2_idx * S_TILE_2
                    actual_s2 = (s2 - s2_idx * S_TILE_2).min(S_TILE_2)

                    pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                    k_j = pypto.view(k_2d, [S_TILE_2, head_dim], [s2_off, h_ofs],
                                     valid_shape=[actual_s2, head_dim])
                    v_j = pypto.view(v_2d, [S_TILE_2, head_dim], [s2_off, h_ofs],
                                     valid_shape=[actual_s2, head_dim])

                    _, ds_ij = compute_p_ds(q_i, k_j, v_j, do_i, m_i, l_i, d_i,
                                            actual_s1, actual_s2, scale,
                                            c_tile, v_tile_s)

                    ds_bf16 = pypto.cast(ds_ij, pypto.DT_BF16)
                    pypto.set_cube_tile_shapes(c_tile[0], c_tile[1], c_tile[2])
                    pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                    dq_tile = pypto.matmul(ds_bf16, k_j, pypto.DT_FP32)

                    pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                    if pypto.is_loop_begin(s2_idx):
                        dq_acc[:] = dq_tile
                    else:
                        dq_acc[:] = dq_acc + dq_tile

                    if pypto.is_loop_end(s2_idx):
                        pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                        dq_final = pypto.cast(pypto.mul(dq_acc, scale), pypto.DT_BF16)
                        dq_final_v = pypto.view(dq_final, [S_TILE_2, head_dim], [0, 0],
                                                valid_shape=[actual_s1, head_dim])
                        pypto.assemble(dq_final_v, [s1_off, h_ofs], dq)

            # ===== 趟2: 计算 dK, dV =====
            for s2_idx in pypto.loop(s2_loop, name="LOOP_s2_dkv", idx_name="s2_idx"):
                s2_off = kv_start + s2_idx * S_TILE_2
                actual_s2 = (s2 - s2_idx * S_TILE_2).min(S_TILE_2)

                pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                k_j = pypto.view(k_2d, [S_TILE_2, head_dim], [s2_off, h_ofs],
                                 valid_shape=[actual_s2, head_dim])
                v_j = pypto.view(v_2d, [S_TILE_2, head_dim], [s2_off, h_ofs],
                                 valid_shape=[actual_s2, head_dim])

                dk_acc = pypto.tensor([S_TILE_2, head_dim], pypto.DT_FP32, "dk_acc")
                dv_acc = pypto.tensor([S_TILE_2, head_dim], pypto.DT_FP32, "dv_acc")

                for s1_idx in pypto.loop(s1_loop, name="LOOP_s1_dkv", idx_name="s1_idx",
                                         unroll_list=[8, 4, 2, 1]):
                    s1_off = q_start + s1_idx * S_TILE_2
                    actual_s1 = (s1 - s1_idx * S_TILE_2).min(S_TILE_2)

                    pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                    q_i = pypto.view(q_2d, [S_TILE_2, head_dim], [s1_off, h_ofs],
                                     valid_shape=[actual_s1, head_dim])
                    do_i = pypto.view(do_2d, [S_TILE_2, head_dim], [s1_off, h_ofs],
                                      valid_shape=[actual_s1, head_dim])
                    o_i = pypto.view(o_2d, [S_TILE_2, head_dim], [s1_off, h_ofs],
                                     valid_shape=[actual_s1, head_dim])
                    m_i = pypto.view(m_2d, [S_TILE_2, 1], [s1_off, n_idx],
                                     valid_shape=[actual_s1, 1])
                    l_i = pypto.view(l_2d, [S_TILE_2, 1], [s1_off, n_idx],
                                     valid_shape=[actual_s1, 1])

                    pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                    d_i = pypto.sum(pypto.cast(pypto.mul(o_i, do_i), pypto.DT_FP32),
                                    -1, keepdim=True)

                    p_ij, ds_ij = compute_p_ds(q_i, k_j, v_j, do_i, m_i, l_i, d_i,
                                               actual_s1, actual_s2, scale,
                                               c_tile, v_tile_s)

                    ds_bf16 = pypto.cast(ds_ij, pypto.DT_BF16)
                    p_bf16 = pypto.cast(p_ij, pypto.DT_BF16)
                    pypto.set_cube_tile_shapes(c_tile[0], c_tile[1], c_tile[2])
                    pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                    dk_tile = pypto.matmul(ds_bf16, q_i, pypto.DT_FP32, a_trans=True)
                    dv_tile = pypto.matmul(p_bf16, do_i, pypto.DT_FP32, a_trans=True)

                    pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                    if pypto.is_loop_begin(s1_idx):
                        dk_acc[:] = dk_tile
                        dv_acc[:] = dv_tile
                    else:
                        dk_acc[:] = dk_acc + dk_tile
                        dv_acc[:] = dv_acc + dv_tile

                    if pypto.is_loop_end(s1_idx):
                        pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                        dk_final = pypto.cast(pypto.mul(dk_acc, scale), pypto.DT_BF16)
                        dv_final = pypto.cast(dv_acc, pypto.DT_BF16)
                        dk_final_v = pypto.view(dk_final, [S_TILE_2, head_dim], [0, 0],
                                                valid_shape=[actual_s2, head_dim])
                        dv_final_v = pypto.view(dv_final, [S_TILE_2, head_dim], [0, 0],
                                                valid_shape=[actual_s2, head_dim])
                        pypto.assemble(dk_final_v, [s2_off, h_ofs], dk)
                        pypto.assemble(dv_final_v, [s2_off, h_ofs], dv)
