# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The same two operations in the same order, on the host. Imports no DSL."""

import torch

LANES = 64


def make_inputs(case):
    generator = torch.Generator().manual_seed(case["seed"])
    spread = case["parameters"]["spread"]
    x = torch.randn(1, LANES, generator=generator) * 3
    y = torch.randn(1, LANES, generator=generator) * 3
    if spread == "mixed":
        # One exponent per lane, alternating sign: the comparison must stay exact across the range.
        scale = torch.pow(2.0, torch.arange(LANES, dtype=torch.float32) - LANES // 2)
        x, y = x * scale, y * scale.flip(0)
    inputs = {"x": x, "y": y}
    validate(inputs)
    return inputs


def validate(inputs):
    for name in ("x", "y"):
        value = inputs[name]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
            raise ValueError(f"{name} must be a float32 tensor")
        if tuple(value.shape) != (1, LANES) or not value.is_contiguous():
            raise ValueError(f"{name} must be one complete contiguous row of {LANES}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")


def reference(inputs):
    validate(inputs)
    # A multiply by two is exact in FP32 and the add is one rounding, the same one the register does.
    return {"o": 2 * inputs["x"] + inputs["y"]}
