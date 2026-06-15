#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
RoPE Golden 参考实现（纯 PyTorch）

包含两种 RoPE 实现：
1. apply_rotary_pos_emb_vision: Vision RoPE（2D，用于视觉部分）
2. apply_multimodal_rotary_pos_emb: Multimodal RoPE（3D，用于多模态）

来源：core/modeling_spatial_ssrl_3b_vl.py line 147-166, 569-611
场景：场景A（仅使用基础 PyTorch 算子，直接复制原始代码）
"""

from typing import Tuple

import logging
import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    y1 = x[..., x.shape[-1] // 2:]
    return torch.cat((-y1, x1), dim=-1)


def apply_rotary_pos_emb_vision_golden(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vision RoPE（2D Rotary Position Embedding）
    
    用于 Spatial_ssrl_3b_VLVisionAttention 的视觉部分
    
    Args:
        q: query tensor, shape [seq_len, num_heads, head_dim]
        k: key tensor, shape [seq_len, num_heads, head_dim]
        cos: cosine tensor, shape [seq_len, head_dim]
        sin: sine tensor, shape [seq_len, head_dim]
    
    Returns:
        q_embed, k_embed: 旋转后的 query 和 key
    """
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()

    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()

    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = (q * cos) + (rotate_half(q) * sin)

    k_embed = k_embed.to(orig_k_dtype)
    q_embed = q_embed.to(orig_q_dtype)
    return q_embed, k_embed


def apply_multimodal_rotary_pos_emb_golden(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Multimodal RoPE（3D Rotary Position Embedding）
    
    用于 Spatial_ssrl_3b_VLAttention 的多模态场景
    
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
    logging.basicConfig(level=logging.INFO)
    logging.info("=== RoPE Golden 参考实现 ===")
    logging.info("包含函数:")
    logging.info("  - rotate_half(x)")
    logging.info("  - apply_rotary_pos_emb_vision_golden(q, k, cos, sin)")
    logging.info("  - apply_multimodal_rotary_pos_emb_golden(q, k, cos, sin, mrope_section)")
    logging.info("\n场景：场景A（纯 PyTorch，直接复制原始代码）")