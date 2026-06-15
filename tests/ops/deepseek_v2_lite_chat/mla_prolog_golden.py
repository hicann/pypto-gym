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
MLA KV Prolog Golden 参考实现
来源：DeepSeek-V2-Lite MLA架构的完整KV路径实现

流程：
1. kv_a_proj (matmul)
2. split (切片)
3. RMSNorm (torch_npu.npu_rms_norm)
4. kv_b_proj (matmul)
5. reshape + split
6. RoPE (2D rotary position embedding)
"""

import torch
import torch_npu


def rotate_half_torch(x):
    """rotate_half torch实现"""
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)


def rope_2d_torch(x, cos, sin):
    """标准2D RoPE torch实现"""
    half_dim = x.shape[-1] // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    x_rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + x_rot * sin


def mla_prolog_golden(
    hidden_states: torch.Tensor,
    kv_a_weight: torch.Tensor,
    kv_b_weight: torch.Tensor,
    ln_weight: torch.Tensor,
    eps: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
    pos_ids: torch.Tensor
) -> tuple:
    """
    MLA KV Prolog Golden实现
    
    Args:
        hidden_states: [bsz, seq_len, hidden_size]
        kv_a_weight: [hidden_size, kv_lora_rank + rope_dim]
        kv_b_weight: [kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim)]
        ln_weight: [kv_lora_rank]
        eps: float (RMSNorm epsilon)
        cos: [seq_len, rope_dim]
        sin: [seq_len, rope_dim]
        pos_ids: [bsz, seq_len]
    
    Returns:
        k_nope: [bsz, num_heads, seq_len, qk_nope_head_dim]
        value: [bsz, num_heads, seq_len, v_head_dim]
        k_pe_embed: [bsz, 1, seq_len, rope_dim]
    """
    bsz, seq_len, hidden_size = hidden_states.shape
    kv_lora_rank = ln_weight.shape[0]
    rope_dim = cos.shape[1]
    num_heads = 16
    qk_nope_head_dim = 128
    v_head_dim = 128
    
    hidden_2d = hidden_states.reshape(bsz * seq_len, hidden_size)
    compressed_kv_total = torch.matmul(hidden_2d, kv_a_weight)
    
    compressed_kv = compressed_kv_total[:, :kv_lora_rank]
    k_pe = compressed_kv_total[:, kv_lora_rank:]
    
    compressed_kv_norm, _ = torch_npu.npu_rms_norm(compressed_kv, ln_weight, epsilon=eps)
    
    kv_total = torch.matmul(compressed_kv_norm, kv_b_weight)
    
    kv_3d = kv_total.reshape(bsz * seq_len, num_heads, qk_nope_head_dim + v_head_dim)
    
    k_nope_2d = kv_3d[:, :, :qk_nope_head_dim]
    value_2d = kv_3d[:, :, qk_nope_head_dim:]
    
    k_nope = k_nope_2d.reshape(bsz, num_heads, seq_len, qk_nope_head_dim)
    value = value_2d.reshape(bsz, num_heads, seq_len, v_head_dim)
    
    cos_expanded = cos.unsqueeze(0).expand(bsz, -1, -1).reshape(bsz * seq_len, rope_dim)
    sin_expanded = sin.unsqueeze(0).expand(bsz, -1, -1).reshape(bsz * seq_len, rope_dim)
    k_pe_embed = rope_2d_torch(k_pe, cos_expanded, sin_expanded)
    k_pe_embed = k_pe_embed.reshape(bsz, 1, seq_len, rope_dim)
    
    return k_nope, value, k_pe_embed


if __name__ == "__main__":
    torch.manual_seed(42)
    
    bsz, seq_len = 1, 2
    hidden_size = 2048
    kv_lora_rank = 512
    rope_dim = 64
    num_heads = 16
    
    hidden = torch.randn(bsz, seq_len, hidden_size, dtype=torch.float16).npu()
    kv_a_weight = torch.randn(hidden_size, kv_lora_rank + rope_dim, dtype=torch.float16).npu()
    kv_b_weight = torch.randn(kv_lora_rank, num_heads * 256, dtype=torch.float16).npu()
    ln_weight = torch.randn(kv_lora_rank, dtype=torch.float16).npu()
    eps = 1e-6
    cos = torch.randn(seq_len, rope_dim, dtype=torch.float16).npu()
    sin = torch.randn(seq_len, rope_dim, dtype=torch.float16).npu()
    pos_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).npu()
    
    k_nope, value, k_pe = mla_prolog_golden(
        hidden, kv_a_weight, kv_b_weight, ln_weight, eps, cos, sin, pos_ids
    )

    assert k_nope.shape == torch.Size([bsz, num_heads, seq_len, 128])
    assert value.shape == torch.Size([bsz, num_heads, seq_len, 128])
    assert k_pe.shape == torch.Size([bsz, 1, seq_len, rope_dim])
