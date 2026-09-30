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
    variant = case["parameters"]["variant"]
    g = torch.Generator().manual_seed(case["seed"])
    if variant == "encode":
        powers = torch.tensor([2.0**exponent for exponent in range(-40, 88)])
        mixed = 1.5 * powers
        edges = torch.randn(128, generator=g) * 100
        edges[:6] = torch.tensor([0.0, float("inf"), float("nan"), -0.0, 2.0**-126, 2.0**-133])
        x = torch.stack((powers, mixed, -mixed, edges)).bfloat16()
    else:
        x = torch.stack(
            (
                torch.arange(128).to(torch.uint8),
                torch.arange(128, 256).to(torch.uint8),
                torch.full((128,), 127, dtype=torch.uint8),
                torch.tensor([0, 1, 126, 127, 128, 253, 254, 255] * 16, dtype=torch.uint8),
            )
        )
    return {"x": x, "variant": variant}


def reference(inputs):
    if inputs["variant"] == "encode":
        result = (inputs["x"].view(torch.uint16).int() >> 7) & 255
    else:
        codes = inputs["x"].int()
        result = torch.where(codes == 255, torch.full_like(codes, 0x7FC0), codes << 7)
    return {"o": result}
