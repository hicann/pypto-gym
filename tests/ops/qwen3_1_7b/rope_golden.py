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
RoPE Golden 参考实现 - 部分融合算子
融合范围: Q/K per-head RMSNorm + RoPE

来源：core/modeling_qwen3.py Qwen3Attention.forward
"""

import torch


def rms_norm_per_head(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Per-head RMSNorm
    
    Args:
        x: [S, N, D] or [B, S, N, D]
        weight: [D]
        eps: 1e-6
    
    Returns:
        normed: same shape as x
    """
    input_dtype = x.dtype
    x_f32 = x.float()
    variance = x_f32.pow(2).mean(-1, keepdim=True)
    normed = x_f32 * torch.rsqrt(variance + eps)
    return (weight.float() * normed).to(input_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half for RoPE"""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def rope_golden_3d(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    RoPE Golden (3D version) - 部分融合
    
    Args:
        x: [S, N, D] - q_proj/k_proj output
        cos: [S, D] - cos values
        sin: [S, D] - sin values
        norm_weight: [D] - q_norm/k_norm weight
        eps: 1e-6
    
    Returns:
        output: [S, N, D] - after RMSNorm + RoPE
    """
    # Step 1: RMSNorm
    normed = rms_norm_per_head(x, norm_weight, eps)
    
    # Step 2: RoPE
    # cos/sin: [S, D] -> [S, 1, D] for broadcasting
    cos_expanded = cos.unsqueeze(1)
    sin_expanded = sin.unsqueeze(1)
    
    # Apply RoPE
    output = normed * cos_expanded + rotate_half(normed) * sin_expanded
    
    return output


def rope_golden_batch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    RoPE Golden (4D version with batch)
    
    Args:
        x: [B, S, N, D]
        cos: [S, D]
        sin: [S, D]
        norm_weight: [D]
    
    Returns:
        output: [B, S, N, D]
    """
    B, S, N, D = x.shape
    
    # Process each batch
    outputs = []
    for b in range(B):
        x_3d = x[b]  # [S, N, D]
        out_3d = rope_golden_3d(x_3d, cos, sin, norm_weight, eps)
        outputs.append(out_3d)
    
    return torch.stack(outputs, dim=0)


def generate_cos_sin(S: int, D: int = 128, base: float = 1000000.0, device: str = "cpu") -> tuple:
    """
    Generate cos/sin for RoPE
    
    Args:
        S: sequence length
        D: head_dim (128)
        base: rope_theta (1e6 for Qwen3)
        device: device
    
    Returns:
        cos: [S, D]
        sin: [S, D]
    """
    half = D // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32, device=device) / half))
    pos = torch.arange(S, dtype=torch.float32, device=device).unsqueeze(-1)
    freqs = pos * inv_freq
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().to(torch.bfloat16)
    sin = emb.sin().to(torch.bfloat16)
    return cos, sin


if __name__ == "__main__":
    torch.manual_seed(42)
    
    S = 32
    N_q = 16
    N_kv = 8
    D = 128
    
    # Test Q kernel
    print("=" * 60)
    print("Testing Q RoPE kernel (N_q=16)")
    print("=" * 60)
    
    x_q = torch.randn(S, N_q, D, dtype=torch.bfloat16)
    cos, sin = generate_cos_sin(S, D)
    q_norm_w = torch.randn(D, dtype=torch.bfloat16)
    
    output_q = rope_golden_3d(x_q, cos, sin, q_norm_w, 1e-6)
    
    print(f"Input shape: {x_q.shape}, dtype: {x_q.dtype}")
    print(f"Output shape: {output_q.shape}, dtype: {output_q.dtype}")
    print(f"Output range: [{output_q.min().item():.4f}, {output_q.max().item():.4f}]")
    
    assert output_q.shape == x_q.shape
    assert output_q.dtype == x_q.dtype
    print("✓ Q RoPE golden passed")
    
    # Test K kernel
    print("\n" + "=" * 60)
    print("Testing K RoPE kernel (N_kv=8)")
    print("=" * 60)
    
    x_k = torch.randn(S, N_kv, D, dtype=torch.bfloat16)
    k_norm_w = torch.randn(D, dtype=torch.bfloat16)
    
    output_k = rope_golden_3d(x_k, cos, sin, k_norm_w, 1e-6)
    
    print(f"Input shape: {x_k.shape}, dtype: {x_k.dtype}")
    print(f"Output shape: {output_k.shape}, dtype: {output_k.dtype}")
    print(f"Output range: [{output_k.min().item():.4f}, {output_k.max().item():.4f}]")
    
    assert output_k.shape == x_k.shape
    assert output_k.dtype == x_k.dtype
    print("✓ K RoPE golden passed")
    
    # Test batch version
    print("\n" + "=" * 60)
    print("Testing batch version (B=2)")
    print("=" * 60)
    
    B = 2
    x_q_batch = torch.randn(B, S, N_q, D, dtype=torch.bfloat16)
    output_q_batch = rope_golden_batch(x_q_batch, cos, sin, q_norm_w, 1e-6)
    
    print(f"Input shape: {x_q_batch.shape}, dtype: {x_q_batch.dtype}")
    print(f"Output shape: {output_q_batch.shape}, dtype: {output_q_batch.dtype}")
    
    assert output_q_batch.shape == x_q_batch.shape
    assert output_q_batch.dtype == x_q_batch.dtype
    print("✓ Batch RoPE golden passed")
    
    print("\n" + "=" * 60)
    print("All golden tests passed!")
    print("=" * 60)