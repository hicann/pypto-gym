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
    g = torch.Generator().manual_seed(case["seed"])
    mode = case["parameters"]["mode"]
    x = (torch.randperm(128, generator=g).float() + 1000).reshape(1, 128)
    if mode == "elements":
        offsets = torch.randint(0, 128, (1, 64), generator=g) * 4
        offsets[0, :2] = torch.tensor([0, 508])
    else:
        offsets = torch.zeros(1, 64, dtype=torch.int64)
        offsets[0, :8] = torch.randint(0, 16, (8,), generator=g) * 32
        offsets[0, :2] = torch.tensor([0, 480])
    return {"x": x, "offsets": offsets.to(torch.uint32), "mode": mode}


def reference(inputs):
    x = inputs["x"][0]
    offsets = inputs["offsets"][0].tolist()
    out = (
        x[torch.tensor(offsets) // 4]
        if inputs["mode"] == "elements"
        else torch.cat([x[start // 4 : start // 4 + 8] for start in offsets[:8]])
    )
    return {"o": out.reshape(1, 64)}
