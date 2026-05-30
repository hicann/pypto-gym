#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance of the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
RoPE Golden 参考实现（场景A：直接复制原始代码）
来源：Qwen2.5-VL/core/modeling_qwen2_5_vl.py
包含两种 RoPE：
1. Vision RoPE (2D) - 用于视觉编码器
2. Multimodal RoPE (3D) - 用于多模态语言模型
"""

import torch
from typing import Tuple, List


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision_golden(
    q: torch.Tensor, 
    k: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vision RoPE 参考实现 (2D Rotary Position Embedding)
    
    算法说明：
    1. 将 query/key 转为 float32 精度计算
    2. 对 cos/sin 进行 unsqueeze 扩展维度
    3. 应用旋转：q_embed = q * cos + rotate_half(q) * sin
    4. 转回原始 dtype
    
    参数：
        q: query tensor, shape [seq_len, num_heads, head_dim]
        k: key tensor, shape [seq_len, num_heads, head_dim]
        cos: cosine tensor, shape [seq_len, head_dim // 2]
        sin: sine tensor, shape [seq_len, head_dim // 2]
    
    返回：
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


def apply_multimodal_rotary_pos_emb_golden(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: List[int],
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Multimodal RoPE 参考实现 (3D Rotary Position Embedding)
    
    算法说明：
    1. 扩展 mrope_section 列表（每个 section 复制一次）
    2. 将 cos/sin 按 mrope_section 分割，周期性选择 (m[i % 3])
    3. concat 并 unsqueeze 到指定维度
    4. 应用旋转：q_embed = q * cos + rotate_half(q) * sin
    
    参数：
        q: query tensor, shape [batch, num_heads, seq_len, head_dim]
        k: key tensor, shape [batch, num_kv_heads, seq_len, head_dim]
        cos: cosine tensor, shape [num_sections, batch, seq_len, head_dim]
        sin: sine tensor, shape [num_sections, batch, seq_len, head_dim]
        mrope_section: 多模态 RoPE section 列表，如 [16, 24, 24]
        unsqueeze_dim: unsqueeze 维度，默认为 1
    
    返回：
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
    import numpy as np
    from numpy.testing import assert_allclose
    
    torch.manual_seed(42)
    
    print("=== RoPE Golden 自检 ===")
    
    # Test Multimodal RoPE
    print("\n[1] Multimodal RoPE:")
    q = torch.randn(1, 16, 31, 128, dtype=torch.float16)
    k = torch.randn(1, 2, 31, 128, dtype=torch.float16)
    cos = torch.randn(3, 1, 31, 128, dtype=torch.float16)
    sin = torch.randn(3, 1, 31, 128, dtype=torch.float16)
    mrope_section = [16, 24, 24]
    
    q_embed, k_embed = apply_multimodal_rotary_pos_emb_golden(q, k, cos, sin, mrope_section)
    
    print(f"  Input q shape: {q.shape}, dtype: {q.dtype}")
    print(f"  Input k shape: {k.shape}, dtype: {k.dtype}")
    print(f"  Output q_embed shape: {q_embed.shape}, dtype: {q_embed.dtype}")
    print(f"  Output k_embed shape: {k_embed.shape}, dtype: {k_embed.dtype}")
    
    assert q_embed.shape == q.shape, "q shape mismatch"
    assert k_embed.shape == k.shape, "k shape mismatch"
    assert q_embed.dtype == q.dtype, "q dtype mismatch"
    assert k_embed.dtype == k.dtype, "k dtype mismatch"
    assert not torch.isnan(q_embed).any(), "q_embed contains NaN"
    assert not torch.isnan(k_embed).any(), "k_embed contains NaN"
    print("  ✓ Multimodal RoPE 验证通过")
    
    # Test Vision RoPE
    print("\n[2] Vision RoPE:")
    q_vis = torch.randn(100, 16, 128, dtype=torch.float16)
    k_vis = torch.randn(100, 16, 128, dtype=torch.float16)
    cos_vis = torch.randn(100, 64, dtype=torch.float16)
    sin_vis = torch.randn(100, 64, dtype=torch.float16)
    
    q_vis_embed, k_vis_embed = apply_rotary_pos_emb_vision_golden(q_vis, k_vis, cos_vis, sin_vis)
    
    print(f"  Input q shape: {q_vis.shape}, dtype: {q_vis.dtype}")
    print(f"  Input k shape: {k_vis.shape}, dtype: {k_vis.dtype}")
    print(f"  Output q_embed shape: {q_vis_embed.shape}, dtype: {q_vis_embed.dtype}")
    print(f"  Output k_embed shape: {k_vis_embed.shape}, dtype: {k_vis_embed.dtype}")
    
    assert q_vis_embed.shape == q_vis.shape, "q_vis shape mismatch"
    assert k_vis_embed.shape == k_vis.shape, "k_vis shape mismatch"
    assert q_vis_embed.dtype == q_vis.dtype, "q_vis dtype mismatch"
    assert k_vis_embed.dtype == k_vis.dtype, "k_vis dtype mismatch"
    assert not torch.isnan(q_vis_embed).any(), "q_vis_embed contains NaN"
    assert not torch.isnan(k_vis_embed).any(), "k_vis_embed contains NaN"
    print("  ✓ Vision RoPE 验证通过")
    
    print("\n[TEST_PASS] Golden 自检通过")