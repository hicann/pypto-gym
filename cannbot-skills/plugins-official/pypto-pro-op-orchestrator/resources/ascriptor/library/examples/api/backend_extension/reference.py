# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The input row and its expected copy. Imports no DSL, no compiler and no backend."""

import torch


def make_inputs(case):
    generator = torch.Generator().manual_seed(case["seed"])
    x = torch.randn(1, 64, generator=generator)
    validate({"x": x})
    return {"x": x}


def validate(inputs):
    x = inputs.get("x")
    if not isinstance(x, torch.Tensor) or x.dtype != torch.float32 or tuple(x.shape) != (1, 64):
        raise ValueError("the copy consumes one complete FP32 row of 64")
    if not x.is_contiguous() or x.device.type != "cpu" or not bool(torch.isfinite(x).all()):
        raise ValueError("the row must be finite, contiguous and on the host")


def reference(inputs):
    validate(inputs)
    # A copy's answer is its input. Cloning it means a kernel that returned the caller's own
    # tensor would still be compared against a separate object.
    return {"copy": inputs["x"].clone()}
