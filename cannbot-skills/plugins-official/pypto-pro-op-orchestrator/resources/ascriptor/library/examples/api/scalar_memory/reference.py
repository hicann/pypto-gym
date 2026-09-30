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
    return {"x": torch.randn(1, 8, generator=g), "spelling": case["parameters"]["spelling"]}


def reference(inputs):
    out = inputs["x"].clone()
    out[0, 3] = inputs["x"][0, 2] + 1.0
    out[0, 4] = inputs["x"][0, 2]
    return {"o": out}
