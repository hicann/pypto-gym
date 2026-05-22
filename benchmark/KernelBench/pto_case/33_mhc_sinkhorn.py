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

FORMULA = "out[b,n,n] = sinkhorn_knopp(softmax(x[b,n,n], dim=-1), num_iters)"
DYNAMIC_AXIS = ["B"]


class Model(nn.Module):
    def __init__(self, eps: float = 1e-6, num_iters: int = 20):
        super().__init__()
        self.eps = eps
        self.num_iters = num_iters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_comb = torch.softmax(x, dim=-1) + self.eps

        col_sum = h_comb.sum(dim=-2, keepdim=True)
        h_comb = h_comb / (col_sum + self.eps)

        for _ in range(max(self.num_iters - 1, 0)):
            row_sum = h_comb.sum(dim=-1, keepdim=True)
            h_comb = h_comb / (row_sum + self.eps)
            col_sum = h_comb.sum(dim=-2, keepdim=True)
            h_comb = h_comb / (col_sum + self.eps)

        return h_comb


def get_inputs():
    bs = 4096
    N = 8
    x = torch.randn(bs, N, N, dtype=torch.float32)
    return [x]


def get_init_inputs():
    return [1e-6, 20]
