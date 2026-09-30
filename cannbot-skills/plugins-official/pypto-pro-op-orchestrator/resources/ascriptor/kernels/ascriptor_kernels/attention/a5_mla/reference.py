# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent Torch E4M3 MLA formula, streamed over the same 256-key tiles as the kernel."""

import math
import torch

def attention(q_nope, q_rope, k_nope, k_rope, scale_rope):
    maximum = torch.full(q_nope.shape[:2], -99999.0)
    total = torch.zeros_like(maximum)
    out = torch.zeros_like(q_nope, dtype=torch.float32)
    for begin in range(0, k_nope.shape[1], 256):
        nope = q_nope.float() @ k_nope[:, begin:begin + 256].float().transpose(-1, -2)
        rope = q_rope.float() @ k_rope[:, begin:begin + 256].float().transpose(-1, -2)
        scores = (rope * scale_rope + nope) * math.sqrt(1 / 576)
        new_maximum = torch.maximum(maximum, scores.amax(dim=-1))
        rescale = torch.exp(maximum - new_maximum)
        probability = torch.exp(scores - new_maximum.unsqueeze(-1))
        total = total * rescale + probability.sum(dim=-1)
        packed = (probability * 16).to(torch.float8_e4m3fn).float()
        out = out * rescale.unsqueeze(-1) + packed @ k_nope[:, begin:begin + 256].float()
        maximum = new_maximum
    return out / total.unsqueeze(-1)


NAMES = ("q_nope", "q_rope", "k_nope", "k_rope", "scale_rope")


def make_inputs(case):
    """Deterministic E4M3 components for one case, plus the runtime rope scale.

    `scale_rope` is a one-element float32 GM tensor, not a constant: it is read on the device
    per batch, so a case cannot fold it into SCALE without changing the kernel's ABI."""
    p = case["parameters"]
    generator = torch.Generator().manual_seed(case["seed"])
    values = {name: torch.randn((1, p["H"] if name.startswith("q") else p["S"],
                                 512 if name.endswith("nope") else 64),
                                generator=generator).to(torch.float8_e4m3fn)
              for name in NAMES[:-1]}
    values["scale_rope"] = torch.randn((1,), generator=generator)
    return values


def reference(inputs):
    return {"out": attention(*(inputs[name] for name in NAMES))}
