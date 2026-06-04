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
RMSNorm Golden 参考实现（场景A：直接复制原始代码）
来源：core/modeling_qwen3.py Qwen3RMSNorm.forward
"""

import torch


def rms_norm_golden(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return weight * hidden_states.to(input_dtype)


if __name__ == "__main__":
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
