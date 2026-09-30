# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent finite integer-domain ReLU, FP16 boundary and two matrix products."""
import torch


def make_inputs(case):
    if case.get("block_dim", 1) != 1 or case["parameters"] != {"M": 16, "N": 16, "P": 16, "K": 32}:
        raise ValueError("Require the declared complete-tile one-core geometry")
    generator = torch.Generator().manual_seed(case["seed"])
    x = torch.randint(-3, 4, (16, 32), generator=generator).half()
    y = torch.randint(-3, 4, (16, 32), generator=generator).half()
    z = torch.randint(-3, 4, (16, 16), generator=generator).half()
    x[0] = 1
    y[0], y[1] = 1, -1
    z[0].zero_()
    z[0, 1] = 1  # observe the deliberately negative intermediate column directly
    return {"x": x, "y": y, "z": z, "block_dim": 1}


def validate_inputs(inputs, case=None):
    if type(inputs["block_dim"]) is not int or inputs["block_dim"] != 1:
        raise ValueError("One core is required")
    for name, shape in (("x", (16, 32)), ("y", (16, 32)), ("z", (16, 16))):
        value = inputs[name]
        if value.dtype != torch.float16 or value.shape != shape or value.device.type != "cpu" or not value.is_contiguous() or not torch.isfinite(value).all() or not (value.abs() <= 3).all() or not torch.equal(value.float(), value.float().round()):
            raise ValueError("Require finite integer-valued contiguous FP16 tensors within [-3,3]")


def reference(inputs):
    validate_inputs(inputs)
    first = inputs["x"].int() @ inputs["y"].int().T
    intermediate = first.clamp_min(0).half()
    # |first| <= 288: the FP16 boundary is exact for this declared integer domain.
    return {"output": (intermediate.int() @ inputs["z"].int().T).float()}
