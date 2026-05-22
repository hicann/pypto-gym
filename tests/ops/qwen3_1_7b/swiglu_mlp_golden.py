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
SwiGLU MLP Golden 参考实现（纯 PyTorch）

数学公式：
    gate = matmul(x, Wgate)
    up = matmul(x, Wup)
    silu_gate = silu(gate) = gate / (1 + exp(-gate))
    hidden = silu_gate * up
    output = matmul(hidden, Wdown)
"""

import torch


def swiglu_mlp_golden(
    x: torch.Tensor,
    Wgate: torch.Tensor,
    Wup: torch.Tensor,
    Wdown: torch.Tensor,
) -> torch.Tensor:
    """
    SwiGLU MLP Golden implementation
    
    Args:
        x: input [S, H] or [batch, seq, H]
        Wgate: [H, INT_SIZE] (transposed for matmul)
        Wup: [H, INT_SIZE] (transposed for matmul)
        Wdown: [INT_SIZE, H] (transposed for matmul)
    
    Returns:
        output with same shape as x
    """
    input_shape = x.shape
    if x.dim() == 3:
        batch, seq, hidden = input_shape
        x_2d = x.view(batch * seq, hidden)
    else:
        x_2d = x
    
    # Gate projection: [S, H] @ [H, INT_SIZE] = [S, INT_SIZE]
    gate = torch.matmul(x_2d, Wgate)
    
    # Up projection: [S, H] @ [H, INT_SIZE] = [S, INT_SIZE]
    up = torch.matmul(x_2d, Wup)
    
    # SiLU activation: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    silu_gate = torch.nn.functional.silu(gate)
    
    # Element-wise multiply: [S, INT_SIZE]
    hidden = silu_gate * up
    
    # Down projection: [S, INT_SIZE] @ [INT_SIZE, H] = [S, H]
    output_2d = torch.matmul(hidden, Wdown)
    
    # Restore shape
    if x.dim() == 3:
        output = output_2d.view(input_shape)
    else:
        output = output_2d
    
    return output


if __name__ == "__main__":
    torch.manual_seed(42)
    
    H = 2048
    INT_SIZE = 6144
    
    # Test cases
    test_cases = [
        (1, H),       # Single token
        (4, H),       # Small batch
        (16, H),      # Medium batch
        (32, H),      # Larger batch
    ]
    
    for shape in test_cases:
        x = torch.randn(shape, dtype=torch.float32)
        Wgate = torch.randn(H, INT_SIZE, dtype=torch.float32)
        Wup = torch.randn(H, INT_SIZE, dtype=torch.float32)
        Wdown = torch.randn(INT_SIZE, H, dtype=torch.float32)
        
        output = swiglu_mlp_golden(x, Wgate, Wup, Wdown)
        
        print(f"Input: {shape} → Output: {output.shape}")
        assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
    
    print("\n✓ All golden tests passed")