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
    
    返回：
        归一化后的 tensor，shape 与输入相同
    """
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return weight * hidden_states.to(input_dtype)


if __name__ == "__main__":
    import numpy as np
    from numpy.testing import assert_allclose
    
    torch.manual_seed(42)
    
    hidden = torch.randn(1, 11, 2048, dtype=torch.float16)
    weight = torch.ones(2048, dtype=torch.float16)
    
    output = rms_norm_golden(hidden, weight, 1e-6)
    
    print(f"Input shape: {hidden.shape}, dtype: {hidden.dtype}")
    print(f"Output shape: {output.shape}, dtype: {output.dtype}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    assert output.shape == hidden.shape
    assert output.dtype == hidden.dtype
    print("Golden 自检通过")
