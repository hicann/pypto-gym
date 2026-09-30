# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent host inputs and expected values: no kernel, facade or simulator import."""
import torch

WIDTH = 64
EPS = 1.0 / 1024.0


def make_inputs(case):
    g = torch.Generator().manual_seed(case["seed"])
    rows = 4
    x = (torch.randn(rows, WIDTH, generator=g) * 2).half()
    if case["parameters"].get("zero_row", False):
        # The only case where EPS is observable: without it this row divides by zero.
        x[rows - 1] = 0
    if case["parameters"].get("tiny_row", False):
        x[0] = torch.full((WIDTH,), 2.0 ** -12).half()
    w = torch.randn(1, WIDTH, generator=g).half()
    return {"x": x, "w": w}


def reference(inputs):
    x = inputs["x"].double()
    w = inputs["w"].double()
    scale = (x.pow(2).mean(dim=1, keepdim=True) + EPS).rsqrt()
    return {"o": (x * scale * w).half()}
