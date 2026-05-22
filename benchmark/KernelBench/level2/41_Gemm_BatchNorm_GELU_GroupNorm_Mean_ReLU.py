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


FORMULA = (
    "z[b, n] = x[b, k] @ W^T[k, n] + bias[n]; "
    "bn[b, n] = BatchNorm(z, dim=n); "
    "g[b, n] = GELU(bn[b, n]); "
    "gn[b, n] = GroupNorm(g, dim=n); "
    "m[b, 0] = mean_{n}(gn[b, n]); "
    "out[b, 0] = ReLU(m[b, 0])"
)
DYNAMIC_AXIS = ["B"]


class Model(nn.Module):
    """
    Model that performs a GEMM, BatchNorm, GELU, GroupNorm, Mean, and ReLU operations in sequence.
    """
    def __init__(self, in_features, out_features, num_groups):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.gemm(x)
        x = self.batch_norm(x)
        x = torch.nn.functional.gelu(x)
        x = self.group_norm(x)
        x = torch.mean(x, dim=1, keepdim=True)
        x = torch.relu(x)
        return x

batch_size = 128
in_features = 512
out_features = 1024
num_groups = 8


def get_inputs():
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, num_groups]