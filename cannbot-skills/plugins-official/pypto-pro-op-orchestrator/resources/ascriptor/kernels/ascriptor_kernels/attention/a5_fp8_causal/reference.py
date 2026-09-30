# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent Torch causal attention with the kernel's precision order: an E5M2 probability numerator and an FP32 denominator."""

import math
import torch

def causal(q, k, v):
    bh, s1, dim = q.shape
    s2 = k.shape[1]
    maximum = torch.full((bh, s1, 1), -1.0e30)
    total = torch.zeros_like(maximum)
    out = torch.zeros((bh, s1, dim))
    rows = torch.arange(s1).view(1, s1, 1)
    for begin in range(0, s2, 128):
        end = min(begin + 128, s2)
        scores = (q.float() @ k[:, begin:end].float().transpose(-1, -2)) / math.sqrt(dim)
        scores = scores.masked_fill(torch.arange(begin, end).view(1, 1, -1) > rows, -1.0e30)
        new_maximum = torch.maximum(maximum, scores.amax(dim=-1, keepdim=True))
        rescale = torch.exp(maximum - new_maximum)
        probability = torch.exp(scores - new_maximum)
        quantized = probability.to(torch.float8_e5m2).float()
        # The denominator sums the FP32 probabilities, not `quantized`: the cast serves PV only.
        # Summing `quantized` here instead is the plausible variant this formula rejects, and the
        # tight rowsum rule in main.py is what would catch a kernel that did it.
        total = total * rescale + probability.sum(dim=-1, keepdim=True)
        out = out * rescale + quantized @ v[:, begin:end].float()
        maximum = new_maximum
    return {"out": (out / total).reshape(bh * s1, dim), "rowmax": maximum.reshape(bh * s1), "rowsum": total.reshape(bh * s1)}


def make_inputs(case):
    """Deterministic E5M2 q/k/v for one case, flattened as the kernel takes them.

    Q is [BH*S1, 128] and K/V are [BH*S2, 128]: the head axis is folded into the row axis, so
    the causal alignment is top-left per head and the host, not the layout, knows where a head
    begins. The case parameters travel with the tensors because `reference` needs them to
    unflatten."""
    p = case["parameters"]
    generator = torch.Generator().manual_seed(case["seed"])
    return {**p, **{name: torch.randn((p["BH"] * p["S1" if name == "q" else "S2"], 128),
                                      generator=generator).to(torch.float8_e5m2)
                    for name in ("q", "k", "v")}}


def reference(inputs):
    return causal(*(inputs[name].reshape(inputs["BH"],
                                         inputs["S1" if name == "q" else "S2"], 128)
                    for name in ("q", "k", "v")))
