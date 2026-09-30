# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The two answers on the host. Imports no DSL.

`tail` is `2 * x + y` over 70 lanes -- not a multiple of the 128-lane capacity, which is the point.
`select` is an element-wise maximum, spelled as the comparison and the choice it drives.
"""

import torch

MODES = {"tail": 70, "select": 64}     # the logical lane count each mode declares
CAPACITY = 128                          # what the staging tile allocates, for either mode
DEVICES = ("a2", "a3")


def make_inputs(case):
    mode = case["parameters"]["mode"]
    generator = torch.Generator().manual_seed(case["seed"])
    lanes = MODES[mode]
    inputs = {"mode": mode,
              "x": torch.randn(1, lanes, generator=generator),
              "y": torch.randn(1, lanes, generator=generator)}
    validate(inputs)
    return inputs


def validate(inputs):
    mode = inputs.get("mode")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {tuple(MODES)}")
    for name in ("x", "y"):
        value = inputs[name]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
            raise ValueError(f"{name} must be float32")
        if tuple(value.shape) != (1, MODES[mode]) or not value.is_contiguous():
            raise ValueError(f"{name} must be one contiguous row of {MODES[mode]}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")


def reference(inputs):
    validate(inputs)
    if inputs["mode"] == "tail":
        return {"o": 2 * inputs["x"] + inputs["y"]}
    return {"o": torch.where(inputs["x"] > inputs["y"], inputs["x"], inputs["y"])}
