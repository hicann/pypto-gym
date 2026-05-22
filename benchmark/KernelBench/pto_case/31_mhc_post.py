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

FORMULA = "out[b, n, d] = h_post[b, n] * h_out[b, d] + sum_i(h_res[b, n, i] * x[b, i, d])"
DYNAMIC_AXIS = ["B"]


class Model(nn.Module):
    def __init__(self, n: int = 4, d: int = 5120):
        super().__init__()
        self.n = n
        self.d = d

    def forward(
        self, x: torch.Tensor, h_res: torch.Tensor, h_out: torch.Tensor, h_post: torch.Tensor
    ) -> torch.Tensor:
        h_out_fp32 = h_out.float()
        x_fp32 = x.float()

        h_post_term = h_post.unsqueeze(-1) * h_out_fp32.unsqueeze(-2)
        h_comb_term = torch.sum(h_res.unsqueeze(-1) * x_fp32.unsqueeze(-2), dim=-3)
        result_fp32 = h_post_term + h_comb_term

        return result_fp32.to(torch.bfloat16)


def get_inputs():
    bs = 1024
    N = 4
    D = 5120
    x = torch.randn(bs, N, D, dtype=torch.bfloat16)
    h_res = torch.randn(bs, N, N, dtype=torch.float32)
    h_out = torch.randn(bs, D, dtype=torch.bfloat16)
    h_post = torch.randn(bs, N, dtype=torch.float32)
    return [x, h_res, h_out, h_post]


def get_init_inputs():
    return [4, 5120]
