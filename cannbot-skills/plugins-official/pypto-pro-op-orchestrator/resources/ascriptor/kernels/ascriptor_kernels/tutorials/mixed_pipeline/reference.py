# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent Torch input generation and the staged formula every graph must reproduce.

The inputs are dyadic — multiples of 1/8 and 1/32 in a bounded range — so every FP32 matmul
here is exact; the only rounding is the explicit FP16 materialisation between stages, which the
kernels perform at the same points. The tolerance below is therefore small and about that
rounding alone, not a precision budget for the arithmetic.
"""
import torch

PATTERNS = ("CVC", "VCV", "CVCV", "VCVC", "CVCVC")
MODES = ("serial", "pipeline", "resident_serial", "resident")
ROWS = (32, 64, 96, 160, 224)


def make_inputs(case):
    p = case["parameters"]
    rows, pattern, mode = p["rows"], p["pattern"], p["mode"]
    if pattern not in PATTERNS or mode not in MODES or rows not in ROWS:
        raise ValueError(f"unsupported graph/mode/rows: {pattern}/{mode}/{rows}")
    g = torch.Generator().manual_seed(case["seed"])
    x = (torch.randint(-4, 5, (rows, 128), generator=g).float() * 0.125).half()
    # A per-tile mark in column 0. A lost drain, a wrong delayed index or a stale resident slot
    # all move rows between tiles, and this makes that visible instead of merely numeric.
    x[:, 0] = (torch.arange(rows) // 32).float().mul(0.0625).half()
    weights = [(torch.randint(-2, 3, (128, 128), generator=g).float() * 0.03125).half()
               for _ in range(3)]
    return {"x": x, "w0": weights[0], "w1": weights[1], "w2": weights[2],
            "y": torch.full((rows, 128), float("nan")),
            "pattern": pattern, "mode": mode, "rows": rows}


def reference(inputs):
    """Walk the pattern string: 'C' is a matmul against the next weight, 'V' is the vector
    stage (scale, bias, relu). Every stage but the last hands FP16 to its successor."""
    value, cube = inputs["x"].clone(), 0
    for stage, kind in enumerate(inputs["pattern"]):
        if kind == "C":
            value = value.float() @ inputs[f"w{cube}"].float()
            cube += 1
        else:
            dtype = value.dtype
            value = (value * 0.5).to(dtype)
            value = (value + 0.25).to(dtype).relu()
            if stage < len(inputs["pattern"]) - 1:
                value = value.half()
    return {"y": value.float()}
