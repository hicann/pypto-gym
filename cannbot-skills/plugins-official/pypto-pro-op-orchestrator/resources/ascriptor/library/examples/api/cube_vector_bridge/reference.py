# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Three matrix products, each scaled by two. Imports no DSL.

The operands are integers in [-4, 4] with K = 16, so each product is at most 1024 and the doubling is
exact -- which is what lets the comparison be a byte comparison.
"""

import torch

BEATS, ROWS, WIDTH = 3, 32, 16
BOUND = 4
SCALE = 2.0
DEVICES = ("a2", "a3")


def make_inputs(case):
    generator = torch.Generator().manual_seed(case["seed"])
    inputs = {"x": torch.randint(-BOUND, BOUND + 1, (BEATS, ROWS, WIDTH),
                                 generator=generator).half(),
              "y": torch.randint(-BOUND, BOUND + 1, (WIDTH, WIDTH), generator=generator).half()}
    validate(inputs)
    return inputs


def validate(inputs):
    shapes = {"x": (BEATS, ROWS, WIDTH), "y": (WIDTH, WIDTH)}
    for name, shape in shapes.items():
        value = inputs[name]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float16:
            raise ValueError(f"{name} must be float16")
        if tuple(value.shape) != shape or not value.is_contiguous():
            raise ValueError(f"{name} must be a contiguous {shape} tile")
        if not bool((value.abs() <= BOUND).all()):
            raise ValueError(f"{name} must stay within the exact integer domain")
        if not torch.equal(value.float(), value.float().round()):
            raise ValueError(f"{name} must be integer-valued")


def reference(inputs):
    validate(inputs)
    return {"o": SCALE * (inputs["x"].float() @ inputs["y"].float().T)}
