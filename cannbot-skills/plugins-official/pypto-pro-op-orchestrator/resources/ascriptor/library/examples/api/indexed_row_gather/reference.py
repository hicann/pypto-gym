# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent host inputs and reference; no kernel, pass or simulator import."""

import torch

SLOTS, S2, DK, DV = 16, 128, 64, 48
GUARD = 1024.0
FILLERS = {"zero": lambda chosen: 0, "last": lambda chosen: S2 - 1, "repeat": lambda chosen: int(chosen[0])}


def make_inputs(case):
    live, padding = case["parameters"]["live"], case["parameters"]["padding"]
    generator = torch.Generator().manual_seed(case["seed"])
    # Multiples of 1/64 below 32 in magnitude are exact in FP16, so every comparison is bytewise.
    k = (torch.randint(-2048, 2049, (S2, DK), generator=generator) / 64).to(torch.float16)
    v = (torch.randint(-2048, 2049, (S2, DV), generator=generator) / 64).to(torch.float16)
    # Column zero labels the row it belongs to, so a wrong index is legible in the gathered output.
    k[:, 0] = torch.arange(S2, dtype=torch.float16)
    v[:, 0] = -torch.arange(S2, dtype=torch.float16)
    chosen = torch.randperm(S2, generator=generator)[:live].int()
    index = torch.full((1, SLOTS), FILLERS[padding](chosen), dtype=torch.int32)
    index[0, :live] = chosen
    return {"index": index, "count": torch.tensor([[live]], dtype=torch.int32), "k_table": k, "v_table": v}


def reference(inputs):
    rows = inputs["index"][0].tolist()
    live = int(inputs["count"][0, 0])
    k, v = inputs["k_table"], inputs["v_table"]
    guarded = torch.full((SLOTS, DV), GUARD, dtype=torch.float16)
    guarded[:live] = v[rows[:live]]
    clamped = torch.full((SLOTS, DV), GUARD, dtype=torch.float16)
    for slot in range(SLOTS):
        clamped[min(slot, live - 1)] = v[rows[slot]]     # the clamp loses row live-1
    return {"k_rows": k[rows], "v_rows": v[rows], "guarded": guarded, "clamped": clamped}
