# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Boolean formulas are independent of register and mask implementations."""

import torch


def make_inputs(case):
    g = torch.Generator().manual_seed(case["seed"])
    x = torch.randperm(64, generator=g, dtype=torch.int32) - 32
    return {
        "x": x.reshape(1, 64),
        "y": (1000 + torch.arange(64, dtype=torch.int32)).reshape(1, 64),
        "count": case["parameters"]["count"],
    }


def reference(inputs):
    x, y = inputs["x"][0], inputs["y"][0]
    count = inputs["count"]
    gate = torch.arange(64) < min(count, 64)
    short = torch.arange(64) < 32
    pred = gate & (x > 0)
    packed = torch.cat((pred[::2], torch.zeros(32, dtype=torch.bool)))
    # b32 observes physical bit 4*i. Pack moves it to 2*i; unpack
    # restores it to 4*i, including bits hidden between packed b32 lanes.
    unpacked = pred.clone()
    interleaved = torch.stack((pred, short), dim=1).flatten()
    return {
        "o": torch.stack(
            (
                pred.int(),
                torch.where(gate, x, y),
                (gate & ~pred).int(),
                (gate & pred & short).int(),
                torch.where(gate, pred | short, False).int(),
                (gate & (pred ^ short)).int(),
                torch.where(gate, pred, False).int(),
                torch.where(gate, pred, short).int(),
                packed.int(),
                unpacked.int(),
                interleaved[:64].int(),
                interleaved[64:].int(),
                pred.int(),
                short.int(),
                torch.full((64,), max(count - 64, 0), dtype=torch.int32),
            )
        )
    }
