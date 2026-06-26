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
MRoPE Golden 参考实现
来源：GutenOCR-3B multimodal rotary position embedding 的标准实现

数学结构:
    mrope_section_expanded = mrope_section * 2
    cos_new = concat over sections, picking cos[i%3] and sin[i%3]
    q_embed = (q * cos_new) + (rotate_half(q) * sin_new)
    k_embed = (k * cos_new) + (rotate_half(k) * sin_new)

Params:
    mrope_section: [16, 24, 24] — temporal, height, width
"""

import torch


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def mrope_golden(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """
    MRoPE Golden实现

    Args:
        q: query tensor [batch, num_heads, seq_len, head_dim]
        k: key tensor [batch, num_kv_heads, seq_len, head_dim]
        cos: cosine tensor [3, batch, seq_len, head_dim] or [batch, seq_len, head_dim*3]
        sin: sine tensor, same shape as cos
        mrope_section: list of section sizes, e.g. [16, 24, 24]
        unsqueeze_dim: dimension to unsqueeze for broadcasting

    Returns:
        q_embed, k_embed: rotated query and key
    """
    mrope_section_expanded = mrope_section * 2
    cos_new = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section_expanded, dim=-1))],
                        dim=-1).unsqueeze(unsqueeze_dim)
    sin_new = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section_expanded, dim=-1))],
                        dim=-1).unsqueeze(unsqueeze_dim)
    q_embed = (q * cos_new) + (rotate_half(q) * sin_new)
    k_embed = (k * cos_new) + (rotate_half(k) * sin_new)
    return q_embed, k_embed


if __name__ == "__main__":
    torch.manual_seed(42)

    batch, q_heads, k_heads, seq_len, head_dim = 1, 16, 2, 31, 128
    mrope_section = [16, 24, 24]
    total_dim = sum(mrope_section) * 2  # 128

    q = torch.randn(batch, q_heads, seq_len, head_dim, dtype=torch.float16)
    k = torch.randn(batch, k_heads, seq_len, head_dim, dtype=torch.float16)
    cos = torch.randn(3, batch, seq_len, total_dim, dtype=torch.float16)
    sin = torch.randn(3, batch, seq_len, total_dim, dtype=torch.float16)

    q_out, k_out = mrope_golden(q, k, cos, sin, mrope_section)
    print(f"q input: {q.shape}, output: {q_out.shape}")
    print(f"k input: {k.shape}, output: {k_out.shape}")
