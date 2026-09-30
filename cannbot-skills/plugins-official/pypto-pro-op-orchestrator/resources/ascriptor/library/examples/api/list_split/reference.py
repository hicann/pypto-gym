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
    p = case["parameters"]
    x = torch.randn(p["N"], 64, generator=torch.Generator().manual_seed(case["seed"]))
    return {"x": x, "rows": tuple(p[name] for name in ("R0", "R1", "R2"))}


def reference(inputs):
    offset = 0
    result = {}
    for index, rows in enumerate(inputs["rows"]):
        result["y" + str(index)] = inputs["x"][offset : offset + rows].clone()
        offset += rows
    return result
