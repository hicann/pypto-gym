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

"""KernelBench-style level1 case: Fused SwiGLU Forward."""

import math
import torch
import torch.nn as nn


FORMULA = "out[m, n] = SiLU(x[m, k] @ W_g[k, n] + b_g[n]) * (x[m, k] @ W_fc[k, n] + b_fc[n])"
DYNAMIC_AXIS = ["M"]


class Model(nn.Module):
    def __init__(self, k: int = 512, n: int = 1024):
        super().__init__()
        self.k = k
        self.n = n
        self.w_g = nn.Parameter(torch.randn(k, n, dtype=torch.bfloat16) / math.sqrt(k))
        self.w_fc = nn.Parameter(torch.randn(k, n, dtype=torch.bfloat16) / math.sqrt(k))
        self.b_g = nn.Parameter(torch.randn(1, n, dtype=torch.bfloat16) / math.sqrt(n))
        self.b_fc = nn.Parameter(torch.randn(1, n, dtype=torch.bfloat16) / math.sqrt(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = x.float() @ self.w_g.float() + self.b_g
        fc = x.float() @ self.w_fc.float() + self.b_fc
        gate_silu = gate * torch.sigmoid(gate)
        y = (gate_silu * fc).to(torch.bfloat16)
        return y


def get_inputs():
    m = 220000
    k = 512
    x = torch.randn(m, k, dtype=torch.bfloat16) / math.sqrt(m)
    return [x]


def get_init_inputs():
    return [512, 1024]
