# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Full mathematical 32-token block-causal attention, independent of any schedule."""

import math
import torch

def attention(q, k, v):
    bh, s1, dim = q.shape
    s2 = k.shape[1]
    scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(dim)
    allowed = (torch.arange(s2).view(1, -1) // 32) <= (torch.arange(s1).view(-1, 1) // 32)
    scores = scores.masked_fill(~allowed, -1.0e30)
    maximum = scores.amax(dim=-1, keepdim=True)
    probability = torch.exp(scores - maximum)
    total = probability.sum(dim=-1, keepdim=True)
    return {"out": ((probability / total) @ v.float()).reshape(bh * s1, dim), "rowmax": maximum.reshape(-1), "rowsum": total.reshape(-1)}


def make_inputs(case):
    """Q/K/V for one case, flattened head-major as the kernel reads them. All six variants
    take the same inputs; `variant` selects a schedule, never a shape or a value."""
    p = case["parameters"]
    generator = torch.Generator().manual_seed(case["seed"])
    return {**p, **{name: torch.randn((p["BH"] * p["S1" if name == "q" else "S2"], 128),
                                      generator=generator).half()
                    for name in ("q", "k", "v")}}


def reference(inputs):
    return attention(*(inputs[name].reshape(inputs["BH"],
                                            inputs["S1" if name == "q" else "S2"], 128)
                       for name in ("q", "k", "v")))
