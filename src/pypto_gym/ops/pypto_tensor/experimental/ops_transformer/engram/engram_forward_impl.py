#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Engram Forward PyPTO Kernel Implementation

单头 (HC_MULT=1) Engram 前向算子：Key/Value 投影 + RMSNorm + sign-sqrt gate + 融合输出。

计算路径（与 tests/.../engram_forward_golden.py 一一对应）：
  key       = embeddings @ Wk.T + bk                       # [B*L, hidden_dim]
  value     = embeddings @ Wv.T + bv                       # [B*L, hidden_dim]
  nkey      = rms_norm(key, key_gamma, eps=1e-6)           # [B*L, hidden_dim]
  score     = sum(nkey * query, dim=-1) / sqrt(hidden_dim) # [B*L],  query = hidden_states
  gate      = sigmoid( sign(score) * sqrt(|score|+1e-4) )  # [B*L]
  value_out = gate * value                                  # [B*L, hidden_dim]

输出：value_out, score_back(=score), key_back(=key), value_back(=value), gate_back(=gate)
"""

import math

import torch
import torch_npu
import pypto


def linear(tensor, weight, bias=None):
    """y = tensor @ weight.T + bias  (weight 按 b_trans 传入)

    tensor: [tile, D]
    weight: [D, D]
    """
    pypto.set_cube_tile_shapes([128, 128], [64, 256], [128, 128])
    pypto.set_vec_tile_shapes(64, 256)
    return pypto.matmul(tensor, weight, pypto.DT_FP32, b_trans=True) + bias


def sign_sqrt(tensor, eps):
    """return sign(tensor) * sqrt(|tensor| + eps)"""
    pos = pypto.ge(tensor, 0.0)
    tensor_sqrt = pypto.sqrt(pypto.abs(tensor) + eps)
    neg_sqrt = pypto.neg(tensor_sqrt)
    return pypto.where(pos, tensor_sqrt, neg_sqrt)


_PASS_OPTIONS = {
    "cube_l1_reuse_setting": {"DEFAULT": 1},
    "vec_nbuffer_setting": {"DEFAULT": 1},
    "auto_mix_partition": 1,
}

_RUNTIME_OPTIONS = {
    "stitch_function_max_num": 128,
    "max_workspace_kb": 1048907,
}


# Note: engram_forward_kernel has 13 parameters due to PyPTO framework requirements.
# PyPTO kernels must declare all input/output tensors explicitly in the function signature;
# grouping via dataclass is not applicable here (G.FNM.03 waived).
@pypto.frontend.jit(
    pass_options=_PASS_OPTIONS,
    runtime_options=_RUNTIME_OPTIONS,
)
def engram_forward_kernel(
    hidden_states: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),  # [B*L, hidden_dim]
    embeddings: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),      # [B*L, hidden_dim]
    key_proj_weights: pypto.Tensor([], pypto.DT_FP32),                  # [hidden_dim, hidden_dim]
    key_proj_bias: pypto.Tensor([], pypto.DT_FP32),                     # [hidden_dim]
    value_proj_weights: pypto.Tensor([], pypto.DT_FP32),                # [hidden_dim, hidden_dim]
    value_proj_bias: pypto.Tensor([], pypto.DT_FP32),                   # [hidden_dim]
    key_gamma: pypto.Tensor([], pypto.DT_FP32),                         # [hidden_dim]
    # ! outputs
    value_out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),       # [B*L, hidden_dim]
    score_back: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),      # [B*L]
    key_back: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),        # [B*L, hidden_dim]
    value_back: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),      # [B*L, hidden_dim]
    gate_back: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),       # [B*L, 1]
):
    pypto.experimental.set_operation_options(combine_axis=True)
    total_seq, hidden_dim = hidden_states.shape
    sqrt_hidden_dim = math.sqrt(hidden_dim)
    eps = 1e-6
    sqrt_eps = 1e-4
    unroll_list = [1024]
    for offset, tile in pypto.loop_unroll(0, total_seq, 1, unroll_list=unroll_list):
        pypto.set_vec_tile_shapes(64, 512)
        embedding_view = embeddings[offset:offset + tile]
        key = linear(embedding_view, key_proj_weights, key_proj_bias)              # [tile, hidden_dim]
        value_proj = linear(embedding_view, value_proj_weights, value_proj_bias)   # [tile, hidden_dim]
        pypto.set_vec_tile_shapes(64, 512)
        key_back[offset:offset + tile] = key                                       # ! return key_linear
        normed_query = hidden_states[offset:offset + tile]                         # [tile, hidden_dim]
        pypto.set_vec_tile_shapes(16, 1024)
        normed_key = pypto.rms_norm(key, key_gamma, eps)                           # [tile, hidden_dim]
        score = pypto.sum(normed_key * normed_query, dim=-1) / sqrt_hidden_dim    # [tile]
        pypto.set_vec_tile_shapes(1024)
        sign_qk = sign_sqrt(score, sqrt_eps)                                       # [tile]
        gates = sign_qk.sigmoid()                                                  # [tile]
        pypto.set_vec_tile_shapes(64, 512)
        score_back[offset:offset + tile] = score                                   # ! return score
        pypto.set_vec_tile_shapes(64, 256)
        gates_unsqueezed = gates.unsqueeze(-1)                                     # [tile, 1]
        value = gates_unsqueezed * value_proj                                      # [tile, hidden_dim]
        value_back[offset:offset + tile] = value_proj                              # ! return value_linear
        value_out[offset:offset + tile] = value                                    # ! return value_out
        gate_back[offset:offset + tile] = gates_unsqueezed                         # ! return gate


# Note: pypto_engram_forward has 7 parameters due to the operator's inherent tensor interface.
# All inputs are distinct operator tensors (weights/biases/gamma/activations) and cannot be
# grouped via dataclass; this matches the golden reference signature (G.FNM.03 / G.FNM.05 waived).
def pypto_engram_forward(
    hidden_states,
    embeddings,
    key_proj_weights,
    key_proj_bias,
    value_proj_weights,
    value_proj_bias,
    key_gamma,
) -> tuple:
    """单头 (num_heads=1) 入口包装：输入 reshape、输出张量分配、kernel 调用、输出 reshape。

    Args (与 golden 同名同形):
      hidden_states:        [batch_size, seq_len, num_heads=1, hidden_dim]  FP32
      embeddings:           [batch_size, seq_len, hidden_dim]               FP32
      key_proj_weights:     [num_heads=1, hidden_dim, hidden_dim]           FP32
      key_proj_bias:        [num_heads=1, hidden_dim]                       FP32
      value_proj_weights:   [hidden_dim, hidden_dim]                        FP32
      value_proj_bias:      [hidden_dim]                                    FP32
      key_gamma:            [num_heads=1, hidden_dim]                       FP32

    Returns:
      value_out:  [batch_size, seq_len, num_heads, hidden_dim]
      score_back: [batch_size, seq_len, num_heads, 1]
      key_back:   [batch_size, seq_len, num_heads, hidden_dim]
      value_back: [batch_size, seq_len, hidden_dim]
      gate_back:  [batch_size, seq_len, num_heads, 1]
    """
    batch_size, seq_len, num_heads, hidden_dim = hidden_states.shape
    device = hidden_states.device
    value_output = torch.zeros([batch_size * seq_len, hidden_dim], device=device).float()
    score_back = torch.zeros([batch_size * seq_len], device=device).float()
    key_back = torch.zeros([batch_size * seq_len, hidden_dim], device=device).float()
    value_back = torch.zeros([batch_size * seq_len, hidden_dim], device=device).float()
    gate_back = torch.zeros([batch_size * seq_len, 1], device=device).float()

    hidden_states_reshaped = hidden_states.view(batch_size * seq_len, hidden_dim)
    embeddings_reshaped = embeddings.view(batch_size * seq_len, hidden_dim)
    key_proj_weights_reshaped = key_proj_weights.view(hidden_dim, hidden_dim)
    key_proj_bias_reshaped = key_proj_bias.view(hidden_dim)
    key_gamma_reshaped = key_gamma.view(hidden_dim)

    input_tensors = [
        hidden_states_reshaped,
        embeddings_reshaped,
        key_proj_weights_reshaped,
        key_proj_bias_reshaped,
        value_proj_weights,
        value_proj_bias,
        key_gamma_reshaped,
        value_output,
        score_back,
        key_back,
        value_back,
        gate_back,
    ]
    engram_forward_kernel(*input_tensors)

    value_output = value_output.view(batch_size, seq_len, num_heads, hidden_dim)
    score_back = score_back.view(batch_size, seq_len, num_heads, 1)
    key_back = key_back.view(batch_size, seq_len, num_heads, hidden_dim)
    value_back = value_back.view(batch_size, seq_len, hidden_dim)
    gate_back = gate_back.view(batch_size, seq_len, num_heads, 1)
    return value_output, score_back, key_back, value_back, gate_back
