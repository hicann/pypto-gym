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
    x = ((torch.randperm(128, generator=g).float() - 64) * 0.375).reshape(4, 32)
    if case["parameters"].get("ties", False):
        x = torch.floor(x / 3) * 3
    indices = (torch.arange(128, dtype=torch.int32) + 1000).to(torch.uint32).reshape(4, 32)
    return {"x": x, "indices": indices}


def records(scores, indices):
    order = torch.argsort(scores, descending=True, stable=True)
    out = torch.empty(2 * scores.numel(), dtype=torch.int32)
    out[::2] = scores[order].contiguous().view(torch.int32)
    out[1::2] = indices.view(torch.int32)[order]
    return out


def reference(inputs):
    x, ids = inputs["x"], inputs["indices"]
    return {
        "sort": torch.stack([records(x[row], ids[row]) for row in range(4)]),
        "merge4": records(x.flatten(), ids.flatten()).reshape(4, 64),
        "merge2": records(x[:2].flatten(), ids[:2].flatten()).reshape(2, 64),
    }
