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
    generator = torch.Generator().manual_seed(case["seed"])
    source = torch.randint(-(1 << 40), 1 << 40, (1, 64), generator=generator, dtype=torch.int64)
    blocks = (
        torch.tensor([15, 0, 15, 7, 7, 8, 1, 14])
        if case["parameters"]["repeated"]
        else torch.randperm(16, generator=generator)[:8]
    )
    indices = torch.zeros(1, 64, dtype=torch.int32)
    indices[0, :8] = (blocks * 32).int()
    return {"src": source, "indices": indices}


def reference(inputs):
    blocks = (inputs["indices"][0, :8] // 32).tolist()
    return {
        "o": torch.cat([inputs["src"][0, block * 4 : (block + 1) * 4] for block in blocks]).reshape(1, 32)
    }
