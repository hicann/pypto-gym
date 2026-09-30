# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A copy is its own reference. Imports no DSL.

The rows are deliberately far apart: each one is offset by ten times its index, so a beat that read
the wrong slot lands about ten away from where it belongs rather than somewhere plausible.
"""

import torch

BEATS = 5
LANES = 64
SEPARATION = 10.0


def make_inputs(case):
    generator = torch.Generator().manual_seed(case["seed"])
    rows = torch.randn(BEATS, LANES, generator=generator)
    inputs = {"x": rows + torch.arange(BEATS)[:, None] * SEPARATION}
    validate(inputs)
    return inputs


def validate(inputs):
    x = inputs["x"]
    if not isinstance(x, torch.Tensor) or x.dtype != torch.float32:
        raise ValueError("the source must be float32")
    if tuple(x.shape) != (BEATS, LANES) or not x.is_contiguous():
        raise ValueError(f"the source must be a contiguous [{BEATS}, {LANES}] tile")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("the source must be finite")


def reference(inputs):
    validate(inputs)
    return {"o": inputs["x"].clone()}
