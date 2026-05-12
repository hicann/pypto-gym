#!/usr/bin/env python3
# coding: utf-8
"""
RMSNorm Golden 参考实现（场景A：直接复制原始代码）
来源：Qwen3-1.7B/core/modeling_qwen3.py Qwen3RMSNorm.forward
"""

import torch


def rms_norm_golden(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    RMSNorm 参考实现
    
    算法说明：
    1. 计算 hidden_states 的平方均值（沿最后一个维度）
    2. 计算 rsqrt(variance + eps) 作为归一化系数
    3. hidden_states * 归一化系数 * weight
    
    参数：
        hidden_states: 输入 tensor，[..., C]
        weight: gamma 缩放参数，[C]
        eps: 数值稳定性常数
    
    Returns:
        归一化后的tensor，shape与输入相同
    
    数学公式：
        variance = mean(hidden_states^2, axis=-1, keepdim=True)
        hidden_states = hidden_states * rsqrt(variance + eps)
        output = weight * hidden_states
    """
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return weight * hidden_states.to(input_dtype)


if __name__ == "__main__":
    # 测试：input_layernorm 场景
    x = torch.randn(1, 1, 2048, dtype=torch.float16)
    w = torch.ones(2048, dtype=torch.float16)
    out = rms_norm_golden(x, w, eps=1e-6)
    print(f"input_layernorm: shape={out.shape}, dtype={out.dtype}")
    
    # 测试：q_norm 场景
    x = torch.randn(1, 1, 16, 128, dtype=torch.float16)
    w = torch.ones(128, dtype=torch.float16)
    out = rms_norm_golden(x, w, eps=1e-6)
    print(f"q_norm: shape={out.shape}, dtype={out.dtype}")
    
    # 测试：k_norm 场景
    x = torch.randn(1, 1, 8, 128, dtype=torch.float16)
    w = torch.ones(128, dtype=torch.float16)
    out = rms_norm_golden(x, w, eps=1e-6)
    print(f"k_norm: shape={out.shape}, dtype={out.dtype}")