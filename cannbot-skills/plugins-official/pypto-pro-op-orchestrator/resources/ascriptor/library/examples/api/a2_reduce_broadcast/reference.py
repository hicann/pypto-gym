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
    width = case["parameters"]["width"]
    g = torch.Generator().manual_seed(case["seed"])
    return {"x": torch.randint(-3, 4, (3, width), generator=g).float(), "width": width}


def reference(inputs):
    x = inputs["x"]
    total = x.sum(1, keepdim=True)
    sums = torch.zeros(3, 64)
    sums[:, 0] = total[:, 0]
    return {"o": x * total, "sums": sums}
