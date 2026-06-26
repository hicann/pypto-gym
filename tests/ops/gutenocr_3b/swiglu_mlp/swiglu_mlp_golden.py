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
SwiGLU MLP Golden 参考实现
来源：GutenOCR-3B Qwen2MLP 结构的标准实现

数学结构:
    gate = SiLU(Linear_gate(x))   # [B, H] → [B, I]
    up   = Linear_up(x)           # [B, H] → [B, I]
    hidden = gate * up
    output = Linear_down(hidden)  # [B, I] → [B, H]
"""

import torch
import torch.nn.functional as F


def swiglu_mlp_golden(x, gate_weight, gate_bias, up_weight, up_bias,
                      down_weight, down_bias):
    """
    SwiGLU MLP Golden实现

    Args:
        x: input tensor [batch, hidden_size]
        gate_weight: gate projection weight [hidden_size, intermediate_size]
        gate_bias: gate projection bias [intermediate_size]
        up_weight: up projection weight [hidden_size, intermediate_size]
        up_bias: up projection bias [intermediate_size]
        down_weight: down projection weight [intermediate_size, hidden_size]
        down_bias: down projection bias [hidden_size]

    Returns:
        output: [batch, hidden_size]
    """
    gate = F.silu(F.linear(x, gate_weight.T, gate_bias))
    up = F.linear(x, up_weight.T, up_bias)
    hidden = gate * up
    return F.linear(hidden, down_weight.T, down_bias)


if __name__ == "__main__":
    torch.manual_seed(42)

    B, H, I = 2, 2048, 11008

    x = torch.randn(B, H, dtype=torch.bfloat16)
    gate_weight = torch.randn(H, I, dtype=torch.bfloat16)
    gate_bias = torch.randn(I, dtype=torch.bfloat16)
    up_weight = torch.randn(H, I, dtype=torch.bfloat16)
    up_bias = torch.randn(I, dtype=torch.bfloat16)
    down_weight = torch.randn(I, H, dtype=torch.bfloat16)
    down_bias = torch.randn(H, dtype=torch.bfloat16)

    output = swiglu_mlp_golden(x, gate_weight, gate_bias,
                               up_weight, up_bias, down_weight, down_bias)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}, dtype: {output.dtype}")
