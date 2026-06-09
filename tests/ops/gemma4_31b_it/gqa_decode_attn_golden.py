# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""GQA decode attention -- pure PyTorch reference with KV head averaging.

Implements GQA attention with KV head averaging for Gemma-4-31B:
    Nq=32, Nkv_orig=16 -> Nkv=4, GROUPS=8, D=256, W=1024

Steps:
    1. Average groups of 4 adjacent KV heads: [B, 16, Skv, D] -> [B, 4, Skv, D]
    2. Expand reduced KV heads: [B, 4, Skv, D] -> [B, 32, Skv, D]
    3. QK^T -> scale -> mask -> softmax
    4. Attention weights @ V -> output
    5. Sliding window: local layers slice KV to last W positions
"""

import math
import torch
import torch.nn.functional as F

Nq = 32
Nkv_orig = 16
KV_GROUP_SIZE = 4
Nkv = Nkv_orig // KV_GROUP_SIZE  # 4
GROUPS = Nq // Nkv  # 8
D = 256
SCALE = 1.0 / math.sqrt(D)
W = 1024


def gqa_decode_attn_golden(query_states, key_states, value_states, attention_mask,
                            scaling=None, layer_kind="global"):
    """FP32 reference GQA attention with KV head averaging.

    Args:
        query_states  : [B, Nq=32, Sq, D=256] bf16
        key_states    : [B, Nkv_orig=16, Skv, D=256] bf16
        value_states  : [B, Nkv_orig=16, Skv, D=256] bf16
        attention_mask : [B, 1, Sq, Skv] fp32 or None
        scaling       : float (default: 1/sqrt(D))
        layer_kind    : "global" or "local"

    Returns:
        [B, Sq, Nq=32, D=256] bf16
    """
    if scaling is None:
        scaling = SCALE

    B, Nq_l, Sq, D_l = query_states.shape
    _, Nkv_in, Skv, _ = key_states.shape

    # Sliding window for local layers
    if layer_kind == "local" and Skv > W:
        key_states = key_states[:, :, Skv - W:, :]
        value_states = value_states[:, :, Skv - W:, :]
        if attention_mask is not None:
            attention_mask = attention_mask[:, :, :, Skv - W:]
        Skv = W

    # Average groups of 4 adjacent KV heads: [B, 16, Skv, D] -> [B, 4, Skv, D]
    k_reduced = (key_states[:, 0::4] + key_states[:, 1::4] + key_states[:, 2::4] + key_states[:, 3::4]).float() / 4.0
    k_reduced = k_reduced.to(key_states.dtype)
    v_reduced = (value_states[:, 0::4] + value_states[:, 1::4] + \
                 value_states[:, 2::4] + value_states[:, 3::4]).float() / 4.0
    v_reduced = v_reduced.to(value_states.dtype)

    G = Nq_l // Nkv  # 8

    # Expand reduced KV heads to match Q heads
    k_exp = k_reduced[:, :, None, :, :].expand(B, Nkv, G, Skv, D_l).reshape(B, Nq_l, Skv, D_l)
    v_exp = v_reduced[:, :, None, :, :].expand(B, Nkv, G, Skv, D_l).reshape(B, Nq_l, Skv, D_l)

    # Attention in FP32
    scores = torch.matmul(query_states.float(), k_exp.float().transpose(-2, -1)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask.float()
    attn_weights = F.softmax(scores, dim=-1)
    out = torch.matmul(attn_weights, v_exp.float())

    return out.transpose(1, 2).to(torch.bfloat16)
