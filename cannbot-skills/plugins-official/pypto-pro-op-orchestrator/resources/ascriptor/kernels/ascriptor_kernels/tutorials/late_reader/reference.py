# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent Torch input generation and formula; no DSL or simulator imports.

The generated values are dyadic — multiples of 1/32 and 1/16 inside a bounded range — so every
FP32 product and sum in this formula is exact. The only rounding left is the deliberate FP16
materialisation of the activation, which is the thing the kernel and the reference must agree
about; that is what makes an exact comparison legitimate here rather than a tolerance.
"""
import torch

ROWS = (32, 96, 224)


def make_inputs(case):
    rows, mode = case["parameters"]["rows"], case["parameters"]["mode"]
    if rows not in ROWS:
        raise ValueError(f"This pilot has only full 32-row tiles at M={'/'.join(map(str, ROWS))}")
    gen = torch.Generator().manual_seed(case["seed"])
    x = (torch.randint(-16, 17, (rows, 128), generator=gen).float() / 32).half()
    x[:, 0] = ((torch.arange(rows) // 32).float() / 16).half()  # a per-tile tag in column 0
    w1 = (torch.randint(-16, 17, (128, 128), generator=gen).float() / 32).half()
    w2 = (torch.randint(-4, 5, (128, 128), generator=gen).float() / 16).half()
    return {"x": x, "w1": w1, "w2": w2, "y": torch.full((rows, 128), float("nan")),
            "rows": rows, "mode": mode}


def stages(data):
    """Both P consumers, named. `p` is read early by the activation and again late by the
    residual add, which is the whole point of the unit: its buffer has to stay live across the
    second matmul."""
    p = data["x"].float() @ data["w1"].float()
    activated = torch.relu((p * 0.25).float() + 0.5)
    h = activated.half()
    u = h.float() @ data["w2"].float()
    return {"p": p, "activated": activated, "h": h, "u": u, "y": (u + p).float()}


def reference(inputs):
    return {"y": stages(inputs)["y"]}
