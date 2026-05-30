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
Multimodal RoPE Golden 参考实现（场景A：直接复制原始代码）
来源：modeling_qwen2_5_vl.py apply_multimodal_rotary_pos_emb + rotate_half
"""

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def mrope_golden(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list,
    unsqueeze_dim: int = 1
) -> tuple:
    """
    Multimodal Rotary Position Embedding 参考实现
    
    法：
    1. mrope_section扩展为2倍（如[16,24,24] -> [32,48,48]）
    2. 按mrope_section分割cos/sin，交替拼接（i%3实现temporal/height/width交替）
    3. 在unsqueeze_dim维度上unsqueeze
    4. q_embed = (q * cos) + (rotate_half(q) * sin)
    5. k_embed类似
    
    参数：
        q: [batch, num_heads, seq_len, head_dim]
        k: [batch, num_kv_heads, seq_len, head_dim]
        cos: [3, batch, seq_len, head_dim]  # 3个维度：temporal, height, width
        sin: [3, batch, seq_len, head_dim]
        mrope_section: [16, 24, 24]  # 各维度的head_dim分配
        unsqueeze_dim: 默认为1
    
    返回：
        (q_embed, k_embed)：shape与输入相同
    """
    mrope_section = mrope_section * 2
    
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)
    
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    
    return q_embed, k_embed


if __name__ == "__main__":
    import numpy as np
    from numpy.testing import assert_allclose
    
    torch.manual_seed(42)
    
    batch, seq_len, num_heads, head_dim = 1, 11, 16, 128
    num_kv_heads = 2
    mrope_section = [16, 24, 24]
    
    q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    k = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.bfloat16)
    cos = torch.randn(3, batch, seq_len, head_dim, dtype=torch.bfloat16)
    sin = torch.randn(3, batch, seq_len, head_dim, dtype=torch.bfloat16)
    
    q_embed, k_embed = mrope_golden(q, k, cos, sin, mrope_section)
    
    print(f"Input q shape: {q.shape}, dtype: {q.dtype}")
    print(f"Input k shape: {k.shape}, dtype: {k.dtype}")
    print(f"Output q_embed shape: {q_embed.shape}, dtype: {q_embed.dtype}")
    print(f"Output k_embed shape: {k_embed.shape}, dtype: {k_embed.dtype}")
    
    assert q_embed.shape == q.shape
    assert k_embed.shape == k.shape
    assert q_embed.dtype == q.dtype
    assert k_embed.dtype == k.dtype
    print("Golden 自检通过")