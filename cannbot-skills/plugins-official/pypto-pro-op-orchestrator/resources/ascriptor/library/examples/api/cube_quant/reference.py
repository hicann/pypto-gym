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
    mode = case["parameters"]["mode"]
    g = torch.Generator().manual_seed(case["seed"])
    dtype = torch.int8 if mode.startswith("int32") else torch.float16
    x = torch.randint(-8, 9, (32, 32), generator=g).to(dtype)
    y = torch.randint(-8, 9, (32, 32), generator=g).to(dtype)
    x[:2] = 8
    y[0] = 8
    y[1] = -8
    return {"operands": (x, y), "mode": mode}


def reference(inputs):
    x, y = inputs["operands"]
    mode = inputs["mode"]
    product = x.float() @ y.float().T
    if mode.endswith("f16"):
        scale = 0.25 if mode.startswith("int32") else 0.5
        out = (product * scale).half()
    else:
        offset = 0 if mode.startswith("int32") else 8
        rounded = (product * 0.5).round() + offset
        lo, hi, dtype = (0, 255, torch.uint8) if mode.endswith("u8") else (-128, 127, torch.int8)
        out = rounded.clamp(lo, hi).to(dtype)
    return {"carriers": out.contiguous().view(torch.uint8).flatten()}
