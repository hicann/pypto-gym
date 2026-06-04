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
SwiGLU MLP Golden参考实现
直接复制原始PyTorch实现
"""

import torch
import torch.nn.functional as F


def swiglu_mlp_golden(x, gate_weight, gate_bias, up_weight, up_bias, down_weight, down_bias):
    """
    SwiGLU MLP Golden实现
    
    Args:
        x: 输入tensor [batch, hidden_size]
        gate_weight: gate_proj权重 [intermediate_size, hidden_size]
        gate_bias: gate_proj偏置 [intermediate_size] 或 None
        up_weight: up_proj权重 [intermediate_size, hidden_size]
        up_bias: up_proj偏置 [intermediate_size] 或 None
        down_weight: down_proj权重 [hidden_size, intermediate_size]
        down_bias: down_proj偏置 [hidden_size] 或 None
    
    Returns:
        output: [batch, hidden_size]
    """
    gate = F.linear(x, gate_weight, gate_bias)
    gate = F.silu(gate)
    
    up = F.linear(x, up_weight, up_bias)
    
    hidden = gate * up
    
    output = F.linear(hidden, down_weight, down_bias)
    
    return output


def swiglu_mlp_golden_no_bias(x, gate_weight, up_weight, down_weight):
    """
    SwiGLU MLP Golden实现（无bias版本）
    Qwen2模型MLP无bias
    
    Args:
        x: 输入tensor [batch, hidden_size]
        gate_weight: gate_proj权重 [intermediate_size, hidden_size]
        up_weight: up_proj权重 [intermediate_size, hidden_size]
        down_weight: down_proj权重 [hidden_size, intermediate_size]
    
    Returns:
        output: [batch, hidden_size]
    """
    return swiglu_mlp_golden(x, gate_weight, None, up_weight, None, down_weight, None)


if __name__ == "__main__":
    import numpy as np
    from numpy.testing import assert_allclose
    
    torch.manual_seed(42)
    
    batch_size = 4
    hidden_size = 2048
    intermediate_size = 11008
    
    device = "npu:0" if torch.npu.is_available() else "cpu"
    
    x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=device)
    gate_weight = torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16, device=device)
    gate_bias = torch.randn(intermediate_size, dtype=torch.bfloat16, device=device)
    up_weight = torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16, device=device)
    up_bias = torch.randn(intermediate_size, dtype=torch.bfloat16, device=device)
    down_weight = torch.randn(hidden_size, intermediate_size, dtype=torch.bfloat16, device=device)
    down_bias = torch.randn(hidden_size, dtype=torch.bfloat16, device=device)
    
    output = swiglu_mlp_golden(x, gate_weight, gate_bias, up_weight, up_bias, down_weight, down_bias)
    
    print(f"Input shape: {x.shape}, dtype: {x.dtype}")
    print(f"Output shape: {output.shape}, dtype: {output.dtype}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    assert output.shape == torch.Size([batch_size, hidden_size])
    assert output.dtype == torch.bfloat16
    print("Golden 自检通过")