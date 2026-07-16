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
Engram Backward Operator Implementation

Backward pass of the Engram (memory retrieval attention) operator.
Supports num_heads=1 only via combine_axis=True and axis merging on the host side.
"""
import math

import pypto
import torch


def rms_norm_backward_module(dy, x, gamma, eps=1e-6):
    hidden_dim = x.shape[-1]
    pypto.set_vec_tile_shapes(16, 1024)
    sum_x2 = pypto.sum(x * x, -1, keepdim=True)
    pypto.set_vec_tile_shapes(64, 256)
    mean_x2 = sum_x2 / hidden_dim
    rms = pypto.sqrt(mean_x2 + eps)
    inv_rms = 1 / rms
    pypto.set_vec_tile_shapes(16, 1024)
    sum_dy_x = pypto.sum(dy * x, -1, keepdim=True)
    pypto.set_vec_tile_shapes(64, 256)
    x_hat = x * inv_rms
    pypto.set_vec_tile_shapes(16, 1024)
    d_gamma = pypto.sum(dy * x_hat, 0)
    pypto.set_vec_tile_shapes(64, 256)
    dx = (dy - x * sum_dy_x / (hidden_dim * (rms * rms))) * inv_rms * gamma
    return dx, d_gamma


def linear_backward_module(dy, x, weight):
    """
    dy: [tile, hidden_dim]   upstream gradient
    x:  [tile, hidden_dim]   saved input
    weight: [hidden_dim, hidden_dim]

    Returns: dx [tile, hidden_dim], d_weight [hidden_dim, hidden_dim], db [hidden_dim]
    """
    pypto.set_cube_tile_shapes([128, 128], [64, 256], [128, 128])
    dx = pypto.matmul(dy, weight, pypto.DT_FP32, b_trans=True)
    d_weight = pypto.matmul(x, dy, pypto.DT_FP32, a_trans=True)
    pypto.set_vec_tile_shapes(16, 1024)
    db = pypto.sum(dy, 0)
    return dx, d_weight, db


pass_options = {
    "cube_l1_reuse_setting": {-1: 16},
    "vec_nbuffer_setting": {-2: 1, -1: 4},
}

runtime_options = {
    "stitch_function_max_num": 12,
    "max_workspace_kb": 5469824,
}


@pypto.frontend.jit(
    pass_options=pass_options,
    runtime_options=runtime_options,
)
def engram_backward_kernel(
    # inputs
    grad_out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),       # [batch_size*seq_len, hidden_dim]
    hidden_states: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),  # [batch_size*seq_len, hidden_dim]
    embeddings: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),     # [batch_size*seq_len, hidden_dim]
    key_w: pypto.Tensor([], pypto.DT_FP32),                            # [hidden_dim, hidden_dim]
    value_w: pypto.Tensor([], pypto.DT_FP32),                          # [hidden_dim, hidden_dim]
    key_gamma: pypto.Tensor([], pypto.DT_FP32),                        # [hidden_dim]
    query_gamma: pypto.Tensor([], pypto.DT_FP32),                      # [hidden_dim]
    key_lineared: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),   # [batch_size*seq_len, hidden_dim]
    value_lineared: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32), # [batch_size*seq_len, hidden_dim]
    gate: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),           # [batch_size*seq_len, 1]
    score: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),          # [batch_size*seq_len, 1]
    # outputs
    d_hidden: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),       # [batch_size*seq_len, hidden_dim]
    d_embeddings: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),   # [batch_size*seq_len, hidden_dim]
    d_key_w: pypto.Tensor([], pypto.DT_FP32),                          # [hidden_dim, hidden_dim]
    d_key_b: pypto.Tensor([], pypto.DT_FP32),                          # [hidden_dim]
    d_value_w: pypto.Tensor([], pypto.DT_FP32),                        # [hidden_dim, hidden_dim]
    d_value_b: pypto.Tensor([], pypto.DT_FP32),                        # [hidden_dim]
    d_key_gamma: pypto.Tensor([], pypto.DT_FP32),                      # [hidden_dim]
):
    pypto.experimental.set_operation_options(combine_axis=True)

    total_seq, hidden_dim = grad_out.shape
    sqrt_hidden_dim = math.sqrt(hidden_dim)
    sqrt_eps = 1e-4
    eps = 1e-6
    unroll_list = [256]

    for offset, tile in pypto.loop_unroll(0, total_seq, 1, unroll_list=unroll_list):
        pypto.set_vec_tile_shapes(64, 256)
        value_lineared_view = value_lineared[offset:offset + tile]  # [tile, hidden_dim]
        embeddings_view = embeddings[offset:offset + tile]           # [tile, hidden_dim]
        grad_out_view = grad_out[offset:offset + tile]               # [tile, hidden_dim]
        gate_view = gate[offset:offset + tile]                       # [tile, 1]
        score_view = score[offset:offset + tile]                     # [tile, 1]
        key_lineared_view = key_lineared[offset:offset + tile]       # [tile, hidden_dim]
        query_view = hidden_states[offset:offset + tile]             # [tile, hidden_dim]

        pypto.set_vec_tile_shapes(64, 256)
        d_value = grad_out_view * gate_view                          # [tile, hidden_dim]
        d_gate = grad_out_view * value_lineared_view                 # [tile, hidden_dim]

        pypto.set_vec_tile_shapes(16, 1024)
        d_gate_sum = pypto.sum(d_gate, -1, keepdim=True)            # [tile, 1]
        key_normed = pypto.rms_norm(key_lineared_view, key_gamma, eps)
        query_normed = pypto.rms_norm(query_view, query_gamma, eps)

        pypto.set_vec_tile_shapes(64, 256)
        d_gate_input = d_gate_sum * gate_view * (pypto.neg(gate_view) + 1)   # [tile, 1]
        abs_score = pypto.abs(score_view) + sqrt_eps                          # [tile, 1]
        d_score = d_gate_input / (pypto.sqrt(abs_score) * 2.0)               # [tile, 1]
        d_query_normed = d_score * key_normed / sqrt_hidden_dim                 # [tile, hidden_dim]
        d_key_normed = d_score * query_normed / sqrt_hidden_dim                 # [tile, hidden_dim]
        d_hidden[offset:offset + tile] = d_query_normed

        d_key_lineared_view, d_key_gamma_view = rms_norm_backward_module(
            d_key_normed, key_lineared_view, key_gamma)
        d_key_view, d_key_w_view, d_key_b_view = linear_backward_module(
            d_key_lineared_view, embeddings_view, key_w)
        d_value_view, d_value_w_view, d_value_b_view = linear_backward_module(
            d_value, embeddings_view, value_w)

        pypto.set_vec_tile_shapes(64, 256)
        d_key_w[:] = d_key_w + d_key_w_view
        d_value_b[:] = d_value_b + d_value_b_view
        d_k_v_view = d_key_view + d_value_view                      # [tile, hidden_dim]
        d_value_w[:] = d_value_w + d_value_w_view
        d_embeddings[offset:offset + tile] = d_k_v_view
        d_key_gamma[:] = d_key_gamma + d_key_gamma_view
        d_key_b[:] = d_key_b + d_key_b_view


# Note: engram_backward_pto has 11 input parameters and 7 return values because
# the Engram backward operator naturally decomposes into this set of tensors.
# Grouping via dataclass would obscure the tensor semantics (G.FNM.03/05 waived).
def engram_backward_pto(
    grad_out,        # [batch_size, seq_len, num_heads, hidden_dim]
    hidden_states,   # [batch_size, seq_len, num_heads, hidden_dim]
    embeddings,      # [batch_size, seq_len, hidden_dim]
    key_w,           # [num_heads, hidden_dim, hidden_dim]
    value_w,         # [hidden_dim, hidden_dim]
    key_gamma,       # [num_heads, hidden_dim]
    query_gamma,     # [num_heads, hidden_dim]
    key_lineared,    # [batch_size, seq_len, num_heads, hidden_dim]
    value_lineared,  # [batch_size, seq_len, hidden_dim]
    gate,            # [batch_size, seq_len, num_heads, 1]
    score,           # [batch_size, seq_len, num_heads, 1]
):
    """Host-side wrapper. Requires num_heads==1, flattens batch*seq, calls the kernel, reshapes outputs."""
    batch_size, seq_len, num_heads, hidden_dim = hidden_states.shape
    if num_heads != 1:
        raise ValueError(
            f"engram_backward_kernel only supports num_heads=1 (combine_axis mode), "
            f"got num_heads={num_heads}"
        )
    device = grad_out.device

    d_hidden = torch.zeros_like(hidden_states)
    d_embeddings = torch.zeros([batch_size, seq_len, hidden_dim], dtype=torch.float32, device=device)
    d_key_w = torch.zeros_like(key_w)
    d_key_b = torch.zeros([num_heads, hidden_dim], dtype=torch.float32, device=device)
    d_value_w = torch.zeros_like(value_w)
    d_value_b = torch.zeros([hidden_dim], dtype=torch.float32, device=device)
    d_key_gamma = torch.zeros_like(key_gamma)

    # Flatten num_heads=1 head dimension into batch_size*seq_len axis
    total_seq = batch_size * seq_len
    grad_out_r = grad_out.reshape(total_seq, hidden_dim)
    hidden_states_r = hidden_states.reshape(total_seq, hidden_dim)
    embeddings_r = embeddings.reshape(total_seq, hidden_dim)
    key_w_r = key_w.reshape(hidden_dim, hidden_dim)
    key_gamma_r = key_gamma.reshape(hidden_dim)
    query_gamma_r = query_gamma.reshape(hidden_dim)
    key_lineared_r = key_lineared.reshape(total_seq, hidden_dim)
    value_lineared_r = value_lineared.reshape(total_seq, hidden_dim)
    gate_r = gate.reshape(total_seq, 1)
    score_r = score.reshape(total_seq, 1)

    d_hidden_r = d_hidden.reshape(total_seq, hidden_dim)
    d_embeddings_r = d_embeddings.reshape(total_seq, hidden_dim)
    d_key_w_r = d_key_w.reshape(hidden_dim, hidden_dim)
    d_key_b_r = d_key_b.reshape(hidden_dim)
    d_key_gamma_r = d_key_gamma.reshape(hidden_dim)

    engram_backward_kernel(
        grad_out_r, hidden_states_r, embeddings_r,
        key_w_r, value_w,
        key_gamma_r, query_gamma_r,
        key_lineared_r, value_lineared_r,
        gate_r, score_r,
        d_hidden_r, d_embeddings_r,
        d_key_w_r, d_key_b_r,
        d_value_w, d_value_b,
        d_key_gamma_r,
    )

    return (
        d_hidden_r.reshape(batch_size, seq_len, num_heads, hidden_dim),  # [batch_size, seq_len, num_heads, hidden_dim]
        d_embeddings_r.reshape(batch_size, seq_len, hidden_dim),          # [batch_size, seq_len, hidden_dim]
        d_key_w_r.reshape(num_heads, hidden_dim, hidden_dim),             # [num_heads, hidden_dim, hidden_dim]
        d_key_b_r.reshape(num_heads, hidden_dim),                         # [num_heads, hidden_dim]
        d_value_w,                                                         # [hidden_dim, hidden_dim]
        d_value_b,                                                         # [hidden_dim]
        d_key_gamma_r.reshape(num_heads, hidden_dim),                     # [num_heads, hidden_dim]
    )
