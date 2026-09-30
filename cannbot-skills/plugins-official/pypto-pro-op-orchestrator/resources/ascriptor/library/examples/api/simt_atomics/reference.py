# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import torch


def make_inputs(case):
    t = torch.arange(512, dtype=torch.int64)
    x = ((t * 7919 + case["seed"]) % 200 - 100).int().reshape(1, 512)
    initial = torch.zeros(9, 64, dtype=torch.int32)
    initial[0] = case["seed"] % 13
    initial[1] = case["seed"] % 17
    initial[2] = -1000000
    initial[3] = 1000000
    initial[5] = -1
    initial[8] = torch.arange(64)
    return {"x": x, "initial": initial}


def reference(inputs):
    values = inputs["x"].reshape(8, 64)
    out = inputs["initial"].clone()
    out[0] += values.sum(0).int()
    out[1] -= values.sum(0).int()
    out[2] = torch.maximum(out[2], values.max(0).values)
    out[3] = torch.minimum(out[3], values.min(0).values)
    out[4] = torch.arange(64)
    for row in values:
        out[5] &= row
        out[6] |= row
        out[7] ^= row
    out[8] = 1000 + torch.arange(64)
    return {"o": out}
