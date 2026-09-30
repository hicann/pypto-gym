# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Torch float32 reference for y = (x * mask) * scale."""

import torch


def make_inputs(case):
    """The mask is drawn from {-1, 0, 1}: the zero lanes are the point of this case set,
    because a lane the kernel forgets to write looks identical to a correctly masked one
    unless the destination starts as NaN."""
    p = case["parameters"]
    n = p["n"]
    generator = torch.Generator(device="cpu").manual_seed(case["seed"])
    x = torch.randn(1, n, generator=generator, dtype=torch.float32) * 3.0
    mask = torch.randint(-1, 2, (1, n), generator=generator).to(torch.float32)
    return {"x": x, "mask": mask, "y": torch.full_like(x, float("nan")),
            "n": n, "scale": float(p["scale"]), "tile_len": p["tile_len"]}


def reference(inputs):
    return {"y": (inputs["x"] * inputs["mask"]) * inputs["scale"]}
