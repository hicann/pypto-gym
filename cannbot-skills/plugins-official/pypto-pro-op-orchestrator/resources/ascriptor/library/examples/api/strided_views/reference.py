# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The same two slices, spelled with Python's own stride notation. Imports no DSL."""

import torch

ROWS, COLUMNS = 4, 64          # the physical input: 256 FP32 elements
SHAPE = (2, 16)                # the view's logical shape
STRIDES = (128, 2)             # in elements, not bytes
OFFSET = 3                     # in elements


def make_inputs(case):
    generator = torch.Generator().manual_seed(case["seed"])
    inputs = {"x": torch.randn(ROWS, COLUMNS, generator=generator)}
    validate(inputs)
    return inputs


def validate(inputs):
    x = inputs["x"]
    if not isinstance(x, torch.Tensor) or x.dtype != torch.float32:
        raise ValueError("the source must be float32")
    if tuple(x.shape) != (ROWS, COLUMNS) or not x.is_contiguous():
        raise ValueError(f"the source must be a contiguous [{ROWS}, {COLUMNS}] tile")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("the source must be finite")


def reference(inputs):
    validate(inputs)
    flat = inputs["x"].reshape(-1)
    rows = [flat[OFFSET + row * STRIDES[0]:][:SHAPE[1] * STRIDES[1]:STRIDES[1]]
            for row in range(SHAPE[0])]
    return {"o": torch.stack(rows)}


def last_element_read():
    """The highest element index the view touches: 161, of the 256 the source holds."""
    return OFFSET + (SHAPE[0] - 1) * STRIDES[0] + (SHAPE[1] - 1) * STRIDES[1]
