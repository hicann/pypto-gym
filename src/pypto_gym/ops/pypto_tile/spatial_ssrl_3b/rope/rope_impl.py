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
PyPTO RoPE 实现（支持 aclgraph）

策略：直接使用 PyTorch 原生算子，用 @allow_in_graph 修饰
理由：
- 原始实现仅使用基础 PyTorch 算子（cat, mul, add, view等）
- 这些算子在 torch.compile 中自动支持
- 无需复杂的 PyPTO kernel 实现
- @allow_in_graph 修饰后自动支持 aclgraph

包含两种 RoPE 实现：
1. apply_rotary_pos_emb_vision: Vision RoPE（2D）
2. apply_multimodal_rotary_pos_emb: Multimodal RoPE（3D）
"""

import torch
from torch._dynamo import allow_in_graph
from typing import Tuple, List


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@allow_in_graph
def apply_rotary_pos_emb_vision_impl(
    q: torch.Tensor, 
    k: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vision RoPE 实现（支持 torch.compile）
    
    直接使用 PyTorch 原生算子，用 @allow_in_graph 修饰
    自动支持 aclgraph（torch.compile + torchair）
    
    Args:
        q: query tensor, shape [seq_len, num_heads, head_dim]
        k: key tensor, shape [seq_len, num_heads, head_dim]
        cos: cosine tensor, shape [seq_len, head_dim // 2]
        sin: sine tensor, shape [seq_len, head_dim // 2]
    
    Returns:
        q_embed, k_embed: 旋转后的 query 和 key
    """
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


@allow_in_graph
def apply_multimodal_rotary_pos_emb_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: List[int],
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Multimodal RoPE 实现（支持 torch.compile）
    
    直接使用 PyTorch 原生算子，用 @allow_in_graph 修饰
    自动支持 aclgraph（torch.compile + torchair）
    
    Args:
        q: query tensor, shape [batch, num_heads, seq_len, head_dim]
        k: key tensor, shape [batch, num_kv_heads, seq_len, head_dim]
        cos: cosine tensor, shape [batch, seq_len, head_dim] 或 [num_sections, batch, seq_len, head_dim]
        sin: sine tensor, shape [batch, seq_len, head_dim] 或 [num_sections, batch, seq_len, head_dim]
        mrope_section: 多模态 RoPE section 列表，如 [16, 24, 24]
        unsqueeze_dim: unsqueeze 维度，默认为 1
    
    Returns:
        q_embed, k_embed: 旋转后的 query 和 key
    """
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


if __name__ == "__main__":
    torch.manual_seed(42)
    
    print("=== RoPE 实现测试 ===")
    print("策略：PyTorch 原生算子 + @allow_in_graph")
    print("自动支持 aclgraph（torch.compile + torchair）")
    
    q = torch.randn(1, 16, 31, 128, dtype=torch.float16)
    k = torch.randn(1, 2, 31, 128, dtype=torch.float16)
    cos = torch.randn(3, 1, 31, 128, dtype=torch.float16)
    sin = torch.randn(3, 1, 31, 128, dtype=torch.float16)
    mrope_section = [16, 24, 24]
    
    q_embed, k_embed = apply_multimodal_rotary_pos_emb_impl(q, k, cos, sin, mrope_section)
    
    print(f"\n输入 shape:")
    print(f"  q: {q.shape}, dtype: {q.dtype}")
    print(f"  k: {k.shape}, dtype: {k.dtype}")
    print(f"  cos: {cos.shape}")
    print(f"  sin: {sin.shape}")
    
    print(f"\n输出 shape:")
    print(f"  q_embed: {q_embed.shape}, dtype: {q_embed.dtype}")
    print(f"  k_embed: {k_embed.shape}, dtype: {k_embed.dtype}")
    
    print(f"\n输出 range:")
    print(f"  q_embed: [{q_embed.min().item():.4f}, {q_embed.max().item():.4f}]")
    print(f"  k_embed: [{k_embed.min().item():.4f}, {k_embed.max().item():.4f}]")