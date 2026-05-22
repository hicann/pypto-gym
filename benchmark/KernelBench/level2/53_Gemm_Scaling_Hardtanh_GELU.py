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
    "s[b, n] = z[b, n] * scaling_factor; "
    "h[b, n] = HardTanh(s[b, n], hardtanh_min, hardtanh_max); "
    "out[b, n] = GELU(h[b, n])"
)
DYNAMIC_AXIS = ["B"]


class Model(nn.Module):
    """
    Model that performs a GEMM, scaling, hardtanh, and GELU activation.
    """
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.gemm(x)
        x = x * self.scaling_factor
        x = self.hardtanh(x)
        x = self.gelu(x)
        return x

batch_size = 128
in_features = 1024
out_features = 512
scaling_factor = 0.5
hardtanh_min = -2
hardtanh_max = 2


def get_inputs():
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max]