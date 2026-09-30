# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Full FP32 softmax attention in Torch: one query row per batch-head, evaluated in one shot rather than streamed."""

import math
import torch

def attention(q, k, v):
    scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
    return torch.softmax(scores, dim=-1) @ v.float()


def make_inputs(case):
    """Deterministic FP16 q/k/v for one case.

    Every variant draws the same tensors from the same seed, so two cases that differ only in
    `variant` are a controlled comparison: any difference in the result is the schedule, not
    the data."""
    p = case["parameters"]
    generator = torch.Generator().manual_seed(case["seed"])
    return {name: torch.randn((p["BH"], 1 if name == "q" else p["S"], 128),
                              generator=generator).half()
            for name in ("q", "k", "v")}


def reference(inputs):
    return {"out": attention(*(inputs[name] for name in ("q", "k", "v")))}
