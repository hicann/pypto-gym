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
    "z[b, n] = x[b, k] @ W^T[k, n] + bias[n] + add_value[n]; "
    "s[b, n] = sigmoid(z[b, n]) * z[b, n]; "
    "t[b, n] = tanh(s[b, n]); "
    "g[b, n] = GELU(t[b, n]); "
    "out[b, n] = clamp(g[b, n], -1, 1)"
)
DYNAMIC_AXIS = ["B"]


class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, adds a value, applies Swish, Tanh, GELU, and Hardtanh activation functions.
    """
    def __init__(self, in_features, out_features, add_value_shape):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape)) 

    def forward(self, x):
        x = self.matmul(x)
        x = x + self.add_value
        x = torch.sigmoid(x) * x # Swish
        x = torch.tanh(x)
        x = torch.nn.functional.gelu(x) # GELU
        x = torch.nn.functional.hardtanh(x, min_val=-1, max_val=1) # Hardtanh
        return x

batch_size = 128
in_features = 1024
out_features = 512
add_value_shape = (out_features,)


def get_inputs():
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, add_value_shape]