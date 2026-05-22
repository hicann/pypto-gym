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
    "s[b, n] = z[b, n] * sigmoid(z[b, n]); "
    "d[b, n] = s[b, n] / 2; "
    "c1[b, n] = clamp(d[b, n], -1, 1); "
    "t[b, n] = tanh(c1[b, n]); "
    "out[b, n] = clamp(t[b, n], -1, 1)"
)
DYNAMIC_AXIS = ["B"]


class Model(nn.Module):
    """
    Simple model that performs a gemm, swish, divide, clamp, tanh, and clamp operations.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.gemm(x)
        x = x * torch.sigmoid(x)  # Swish activation
        x = x / 2.0
        x = torch.clamp(x, min=-1.0, max=1.0)  # Clamp between -1 and 1
        x = torch.tanh(x)  # Tanh activation
        x = torch.clamp(x, min=-1.0, max=1.0)  # Clamp between -1 and 1
        return x

batch_size = 128
in_features = 1024
out_features = 512


def get_inputs():
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features]