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
    g = torch.Generator().manual_seed(case["seed"])
    return {
        "x": torch.randint(-10000, 10000, (1, p["cells"]), generator=g, dtype=torch.int32),
        "parameters": dict(p),
    }


def reference(inputs):
    p = inputs["parameters"]
    out = inputs["x"].reshape(p["rows"], p["width"]) + torch.arange(p["rows"], dtype=torch.int32)[:, None]
    return {"o": out.reshape(1, p["cells"])}
