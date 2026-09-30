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
    return {
        "x": torch.randint(-3, 4, (32, 32), generator=g).float(),
        "y": torch.randint(-3, 4, (64, 32), generator=g).float(),
        "bias": ((torch.arange(64) - 31) / 4).reshape(1, 64),
        "mode": case["parameters"]["mode"],
    }


def reference(inputs):
    return {"o": inputs["x"] @ inputs["y"].T + inputs["bias"]}
