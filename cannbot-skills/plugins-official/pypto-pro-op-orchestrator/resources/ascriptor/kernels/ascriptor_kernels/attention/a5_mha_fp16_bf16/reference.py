# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Dense FP64 MHA formula, evaluated independently for each batch and head."""

import math
import torch

def attention(inputs):
    p = inputs["parameters"]
    scale = p["scale_value"] if p["scale_value"] > 0 else 1 / math.sqrt(p["D"])
    result = torch.empty_like(inputs["query"])
    if p["is_causal"]:
        q = torch.arange(p["SQ"]).view(-1, 1)
        k = torch.arange(p["SKV"]).view(1, -1)
        valid = k <= q + p["SKV"] - p["SQ"]
    for batch in range(p["B"]):
        for head in range(p["H"]):
            q = inputs["query"][batch, :, head, :].double()
            k = inputs["key"][batch, :, head, :].double()
            v = inputs["value"][batch, :, head, :].double()
            scores = (q @ k.T) * scale
            if p["is_causal"]:
                scores = scores.masked_fill(~valid, float("-inf"))
            result[batch, :, head, :] = (torch.softmax(scores, dim=-1) @ v).to(result.dtype)
    return result


def shapes(p):
    return {"query": (p["B"], p["SQ"], p["H"], p["D"]),
            "key": (p["B"], p["SKV"], p["H"], p["D"]),
            "value": (p["B"], p["SKV"], p["H"], p["D"])}


def make_inputs(case):
    """Deterministic query/key/value for one case. Values stay inside [-1, 1] so the
    FP64 softmax cannot overflow and the comparison tolerance means what it says."""
    p = case["parameters"]
    generator = torch.Generator().manual_seed(case["seed"])
    dtype = getattr(torch, p["dtype"])
    data = {}
    for name, shape in shapes(p).items():
        if p["input_pattern"] == "zero":
            data[name] = torch.zeros(shape, dtype=dtype)
        elif p["input_pattern"] == "equal_kv" and name == "value":
            data[name] = data["key"].clone()  # equal values, independent storage
        else:
            data[name] = (torch.rand(shape, dtype=torch.float64, generator=generator) * 2 - 1).to(dtype)
    if p["input_pattern"] == "head_distinct":
        # every head carries a different signal, so a kernel that shares a head's K or V
        # with its neighbour produces a visibly wrong answer instead of a plausible one
        for index, name in enumerate(("query", "key", "value")):
            shape = shapes(p)[name]
            h = torch.arange(p["H"], dtype=torch.float64).view(1, 1, -1, 1)
            s = torch.arange(shape[1], dtype=torch.float64).view(1, -1, 1, 1)
            d = torch.arange(p["D"], dtype=torch.float64).view(1, 1, 1, -1)
            values = ((h + 1) / (p["H"] + 1) - 0.5) * (1 if index != 2 else -1)
            values = values + ((s % 7) - 3) / 32 + ((d % 5) - 2) / 32 + index / 16
            data[name] = values.expand(shape).to(dtype).clone()
    data["parameters"] = p
    return data


def reference(inputs):
    return {"out": attention(inputs)}
