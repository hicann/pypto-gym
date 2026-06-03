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

"""KernelBench-style level1 case: Fused SwiGLU Backward."""

import math
import torch
import torch.nn as nn


FORMULA = "dx = dg @ W_g^T + dfc @ W_fc^T, dw_g = x^T @ dg, dw_fc = x^T @ dfc, db_g = sum(dg), db_fc = sum(dfc)"
DYNAMIC_AXIS = ["M"]


class Model(nn.Module):
    def __init__(self, k: int = 512, n: int = 1024):
        super().__init__()
        self.k = k
        self.n = n
        self.w_g = nn.Parameter(torch.randn(k, n, dtype=torch.bfloat16) / math.sqrt(k))
        self.w_fc = nn.Parameter(torch.randn(k, n, dtype=torch.bfloat16) / math.sqrt(k))

    def forward(self, dy: torch.Tensor, g: torch.Tensor, fc: torch.Tensor, x: torch.Tensor):
        sigmoid_g = torch.sigmoid(g)
        silu_g = g * sigmoid_g
        dg = dy * fc * sigmoid_g * (1 + g * (1 - sigmoid_g))
        dfc = dy * silu_g
        db_g = dg.sum(dim=0, keepdim=True)
        db_fc = dfc.sum(dim=0, keepdim=True)
        dw_g = torch.transpose(x, 0, 1) @ dg
        dw_fc = torch.transpose(x, 0, 1) @ dfc
        dx = dg @ torch.transpose(self.w_g, 0, 1) + dfc @ torch.transpose(self.w_fc, 0, 1)
        return dx, dw_g, dw_fc, db_g, db_fc


def get_inputs():
    m = 32768
    k = 512
    n = 1024
    dy = torch.randn(m, n, dtype=torch.bfloat16) / math.sqrt(m)
    g = torch.randn(m, n, dtype=torch.bfloat16) / math.sqrt(m)
    fc = torch.randn(m, n, dtype=torch.bfloat16) / math.sqrt(m)
    x = torch.randn(m, k, dtype=torch.bfloat16) / math.sqrt(m)
    return [dy, g, fc, x]


def get_init_inputs():
    return [512, 1024]
