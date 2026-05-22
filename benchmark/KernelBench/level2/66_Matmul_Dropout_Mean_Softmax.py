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
    "d[b, n] = Dropout(z[b, n]); "
    "m[b, 0] = mean_{n}(d[b, n]); "
    "out[b, 0] = Softmax(m, dim=1)"
)
DYNAMIC_AXIS = ["B"]


class Model(nn.Module):
    """
    A model that performs matrix multiplication, applies dropout, calculates the mean, and then applies softmax.
    """
    def __init__(self, in_features, out_features, dropout_p):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.matmul(x)
        x = self.dropout(x)
        x = torch.mean(x, dim=1, keepdim=True)
        x = torch.softmax(x, dim=1)
        return x

batch_size = 128
in_features = 100
out_features = 50
dropout_p = 0.2


def get_inputs():
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, dropout_p]