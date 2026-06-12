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
Qwen3-1.7B RMSNorm + RoPE Golden 参考实现
来源：Qwen3-1.7B 架构的 RMSNorm + RoPE 融合算子

流程：
1. RMSNorm (per-head normalization)
2. RoPE (Rotary Position Embedding)
"""

import torch


def rms_norm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    RMSNorm torch实现
    
    Args:
        x: [S, N, D] or [..., D]
        weight: [D]
        eps: epsilon
    
    Returns:
        normed: same shape as x
    """
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return x_normed * weight


def rope_torch(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    RoPE torch实现
    
    Args:
        x: [S, N, D]
        cos: [S, D/2]
        sin: [S, D/2]
    
    Returns:
        rotated: [S, N, D]
    """
    half_dim = x.shape[-1] // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    
    cos_expanded = cos.unsqueeze(1)
    sin_expanded = sin.unsqueeze(1)
    
    o1 = x1 * cos_expanded - x2 * sin_expanded
    o2 = x2 * cos_expanded + x1 * sin_expanded
    
    return torch.cat([o1, o2], dim=-1)


def rms_norm_rope_golden(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    w_norm: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Qwen3-1.7B RMSNorm + RoPE Golden实现
    
    Args:
        x: [S, N, D] - q_proj/k_proj 输出
        cos: [S, D] - cos 值
        sin: [S, D] - sin 值
        w_norm: [D] - q_norm/k_norm 权重
        eps: RMSNorm epsilon
    
    Returns:
        out: [S, N, D] - 经过 RMSNorm + RoPE 的结果
    """
    normed = rms_norm_torch(x, w_norm, eps)
    
    half_dim = cos.shape[-1] // 2
    cos_half = cos[..., :half_dim]
    sin_half = sin[..., :half_dim]
    
    out = rope_torch(normed, cos_half, sin_half)
    
    return out


if __name__ == "__main__":
    torch.manual_seed(42)
    
    S, N_q, N_k, D = 16, 16, 8, 128
    
    x_q = torch.randn(S, N_q, D, dtype=torch.bfloat16)
    x_k = torch.randn(S, N_k, D, dtype=torch.bfloat16)
    cos = torch.randn(S, D, dtype=torch.bfloat16)
    sin = torch.randn(S, D, dtype=torch.bfloat16)
    w_norm = torch.randn(D, dtype=torch.bfloat16)
    eps = 1e-6
    
    out_q = rms_norm_rope_golden(x_q, cos, sin, w_norm, eps)
    out_k = rms_norm_rope_golden(x_k, cos, sin, w_norm, eps)
    
    print(f"Input Q shape: {x_q.shape}, dtype: {x_q.dtype}")
    print(f"Input K shape: {x_k.shape}, dtype: {x_k.dtype}")
    print(f"Output Q shape: {out_q.shape}, dtype: {out_q.dtype}")
    print(f"Output K shape: {out_k.shape}, dtype: {out_k.dtype}")
    
    assert out_q.shape == torch.Size([S, N_q, D])
    assert out_k.shape == torch.Size([S, N_k, D])
    
    print("Golden 自检通过")