# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent Torch E4M3 attention: the same precision order as the kernel, evaluated tile by tile."""

import math
import torch

def attention(q, k, v, scale_q, scale_k, scale_v, *, probability_scale=16.0):
    maximum = torch.full(q.shape[:2], -99999.0)
    total = torch.zeros_like(maximum)
    output = torch.zeros_like(q, dtype=torch.float32)
    score_scale = torch.tensor(scale_q, dtype=torch.float32) * (1 / math.sqrt(128)) * scale_k
    for begin in range(0, k.shape[1], 256):
        scores = (q.float() @ k[:, begin:begin + 256].float().transpose(-1, -2)) * score_scale
        new_maximum = torch.maximum(maximum, scores.amax(dim=-1))
        rescale = torch.exp(maximum - new_maximum)
        probability = torch.exp(scores - new_maximum.unsqueeze(-1))
        total = total * rescale + probability.sum(dim=-1)
        packed = (probability * probability_scale).to(torch.float8_e4m3fn).float()
        output = output * rescale.unsqueeze(-1) + packed @ v[:, begin:begin + 256].float()
        maximum = new_maximum
    return (output * (scale_v / probability_scale)) / total.unsqueeze(-1)


def make_inputs(case):
    """Deterministic E4M3 q/k/v plus the three independent scales, drawn in that order.

    Each scale is an ordinary positive float in [0.5, 1.5): the kernel takes them as scalar
    arguments, so nothing about them is baked into the schedule and a case that changes one
    changes only the arithmetic."""
    p = case["parameters"]
    generator = torch.Generator().manual_seed(case["seed"])
    values = {name: torch.randn((p["BH"], 1 if name == "q" else p["S"], 128),
                                generator=generator).to(torch.float8_e4m3fn)
              for name in ("q", "k", "v")}
    values.update({name: float(torch.rand((), generator=generator) + 0.5)
                   for name in ("scale_q", "scale_k", "scale_v")})
    return values


def reference(inputs):
    return {"out": attention(*(inputs[name] for name in
                               ("q", "k", "v", "scale_q", "scale_k", "scale_v")))}
