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

from dataclasses import dataclass
from typing import Any
import pypto


Q_TILE = 320
K_TILE = 320


@dataclass
class _FaSetupDimsOutputs:
    num_heads: Any
    head_dim: Any
    hidden_dim: Any
    total_q: Any
    total_kv: Any
    scale: Any
    q_2d: Any
    k_2d: Any
    v_2d: Any


def _fa_setup_dims(q, k, v):


    """Setup: derive dimensions, reshape inplace, return symbols + 2D views."""
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    hidden_dim = num_heads * head_dim
    total_q = q.shape[0]
    total_kv = k.shape[0]
    scale = 1.0 / (head_dim ** 0.5)
    q_2d = pypto.reshape(q, [total_q, hidden_dim], inplace=True)
    k_2d = pypto.reshape(k, [total_kv, hidden_dim], inplace=True)
    v_2d = pypto.reshape(v, [total_kv, hidden_dim], inplace=True)
    return _FaSetupDimsOutputs(num_heads, head_dim, hidden_dim, total_q, total_kv, scale, q_2d, k_2d, v_2d)


def _fa_compute_head_tile_views(q_2d, k_2d, v_2d, q_tile, k_tile, head_dim,
                                 q_start, q_tile_start, k_start, k_tile_start,
                                 q_tile_len, k_tile_len, h_offset):
    q_tile_view = pypto.view(q_2d, [q_tile, head_dim],
                          [q_start + q_tile_start, h_offset],
                          valid_shape=[q_tile_len, head_dim])
    k_tile_view = pypto.view(k_2d, [k_tile, head_dim],
                          [k_start + k_tile_start, h_offset],
                          valid_shape=[k_tile_len, head_dim])
    v_tile_view = pypto.view(v_2d, [k_tile, head_dim],
                          [k_start + k_tile_start, h_offset],
                          valid_shape=[k_tile_len, head_dim])
    return q_tile_view, k_tile_view, v_tile_view


def _fa_compute_scores_softmax(q_tile_view, k_tile_view, scale):
    pypto.set_vec_tile_shapes(64, 512)
    if pypto.platform.npuarch == 'DAV_3510':
        pypto.set_pass_options(sg_set_scope=5001)
    scores = pypto.matmul(q_tile_view, k_tile_view, out_dtype=pypto.DT_FP32, b_trans=True)
    scores_scaled = pypto.mul(scores, scale)
    mij = pypto.amax(scores_scaled, dim=-1, keepdim=True)
    s_shifted = pypto.sub(scores_scaled, mij)
    pij = pypto.exp(s_shifted)
    lij = pypto.sum(pij, dim=-1, keepdim=True)
    return pij, lij, mij


def _fa_emit_final_tile(k_tile_idx, q_start, q_tile_start, h_act_idx, h_offset,
                         pij, lij, mij, v_tile_view, l_output, m_output, output):
    pypto.set_vec_tile_shapes(64, 512)
    pij_div = pypto.div(pij, lij, precision_type=pypto.PrecisionType.INTRINSIC)
    pij_bf16 = pypto.cast(pij_div, pypto.DT_BF16)
    oij = pypto.matmul(pij_bf16, v_tile_view, out_dtype=pypto.DT_BF16)
    if pypto.platform.npuarch == 'DAV_3510':
        pypto.set_pass_options(sg_set_scope=-1)
    pypto.assemble(lij, [q_start + q_tile_start, h_act_idx], l_output)
    pypto.assemble(mij, [q_start + q_tile_start, h_act_idx], m_output)
    pypto.assemble(oij, [q_start + q_tile_start, h_offset], output)


def _fa_emit_first_tile(v_tile_view, pij, oi_update, li_update, mi_update, lij, mij):
    pypto.set_vec_tile_shapes(64, 512)
    pij_bf16 = pypto.cast(pij, pypto.DT_BF16)
    oij = pypto.matmul(pij_bf16, v_tile_view, out_dtype=pypto.DT_FP32)
    if pypto.platform.npuarch == 'DAV_3510':
        pypto.set_pass_options(sg_set_scope=-1)
    oi_update[:] = oij
    li_update[:] = lij
    mi_update[:] = mij


def _fa_accumulate_tile(k_tile_idx, q_start, q_tile_start, h_act_idx, h_offset,
                         v_tile_view, pij, lij, mij, oi_update, li_update, mi_update,
                         q_tile_len, head_dim, q_tile, l_output, m_output, output):
    pypto.set_vec_tile_shapes(64, 512)
    pij_bf16 = pypto.cast(pij, pypto.DT_BF16)
    oij = pypto.matmul(pij_bf16, v_tile_view, out_dtype=pypto.DT_FP32)
    if pypto.platform.npuarch == 'DAV_3510':
        pypto.set_pass_options(sg_set_scope=-1)
    pypto.set_vec_tile_shapes(512, 64)
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


def _fa_process_ktile(k_tile_idx, k_tile_count, q_start, q_tile_start, k_start, k_tile,
                      seq_len_k, h_idx, head_dim, q_2d, k_2d, v_2d, q_tile, scale,
                      q_tile_len, li_update_0, mi_update_0, oi_update_0,
                      li_update_1, mi_update_1, oi_update_1,
                      l_output, m_output, output):
    k_tile_start = k_tile_idx * k_tile
    k_tile_end = pypto.min(k_tile_start + k_tile, seq_len_k)
    k_tile_len = k_tile_end - k_tile_start
    for h_s_idx in range(2):
        h_act_idx = h_idx * 2 + h_s_idx
        h_offset = h_act_idx * head_dim
        if h_s_idx == 0:
            li_up, mi_up, oi_up = li_update_0, mi_update_0, oi_update_0
        else:
            li_up, mi_up, oi_up = li_update_1, mi_update_1, oi_update_1
        q_tv, k_tv, v_tv = _fa_compute_head_tile_views(
            q_2d, k_2d, v_2d, q_tile, k_tile, head_dim,
            q_start, q_tile_start, k_start, k_tile_start,
            q_tile_len, k_tile_len, h_offset)
        pypto.set_cube_tile_shapes([64, 512], [64, 64], [512, 512])
        pij, lij, mij = _fa_compute_scores_softmax(q_tv, k_tv, scale)
        pypto.set_cube_tile_shapes([128, 512], [256, 512], [64, 64])
        if pypto.is_loop_begin(k_tile_idx):
            if pypto.is_loop_end(k_tile_idx):
                _fa_emit_final_tile(k_tile_idx, q_start, q_tile_start, h_act_idx,
                                     h_offset, pij, lij, mij, v_tv,
                                     l_output, m_output, output)
            else:
                _fa_emit_first_tile(v_tv, pij, oi_up, li_up, mi_up, lij, mij)
        else:
            _fa_accumulate_tile(k_tile_idx, q_start, q_tile_start, h_act_idx,
                                 h_offset, v_tv, pij, lij, mij,
                                 oi_up, li_up, mi_up,
                                 q_tile_len, head_dim, q_tile,
                                 l_output, m_output, output)


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
    q: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    k: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    v: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    output: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    l_output: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    m_output: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    cu_seqlens_q: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    cu_seqlens_k: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
):
    """Flash Attention Forward - 4 loops (batch + head + q_tile + kv_tile)."""
    num_heads, head_dim, hidden_dim, total_q, total_kv, scale, q_2d, k_2d, v_2d = \
        _fa_setup_dims(q, k, v)
    q_tile, k_tile = Q_TILE, K_TILE
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_cube_tile_shapes([128, 128], [128, 256], [128, 128])
    pypto.set_vec_tile_shapes(64, 256)
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
                oi_up0 = pypto.tensor([q_tile, head_dim], pypto.DT_FP32, "oi_update")
                li_up0 = pypto.tensor([q_tile, 1], pypto.DT_FP32, "li_update")
                mi_up0 = pypto.tensor([q_tile, 1], pypto.DT_FP32, "mi_update")
                oi_up1 = pypto.tensor([q_tile, head_dim], pypto.DT_FP32, "oi_update")
                li_up1 = pypto.tensor([q_tile, 1], pypto.DT_FP32, "li_update")
                mi_up1 = pypto.tensor([q_tile, 1], pypto.DT_FP32, "mi_update")
                q_tile_start = q_tile_idx * q_tile
                q_tile_end = pypto.min(q_tile_start + q_tile, seq_len_q)
                q_tile_len = q_tile_end - q_tile_start
                for k_tile_idx in pypto.loop(k_tile_count, name="k_tile_loop"):
                    _fa_process_ktile(
                        k_tile_idx, k_tile_count, q_start, q_tile_start, k_start, k_tile,
                        seq_len_k, h_idx, head_dim, q_2d, k_2d, v_2d, q_tile, scale,
                        q_tile_len, li_up0, mi_up0, oi_up0,
                        li_up1, mi_up1, oi_up1,
                        l_output, m_output, output)
