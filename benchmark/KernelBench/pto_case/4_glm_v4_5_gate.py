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

import torch
import torch.nn as nn
import math

FORMULA = "out[m, n] = x[m, k] @ W[n, k]^T"
DYNAMIC_AXIS = ["M"]


class Model(nn.Module):
    def __init__(self, num_router_experts: int = 160, hidden_size: int = 5120):
        super().__init__()
        self.gate_weight = nn.Parameter(
            torch.randn(num_router_experts, hidden_size, dtype=torch.float32) / math.sqrt(hidden_size)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.matmul(hidden_states, self.gate_weight.t())


def get_inputs():
    bs = 64
    hidden_size = 5120
    return [torch.randn(bs, hidden_size, dtype=torch.float32) / math.sqrt(hidden_size)]


def get_init_inputs():
    return [160, 5120]
