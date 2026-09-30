# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent arithmetic and explicit intermediate-half rounding references."""

import torch


def make_inputs(case):
    variant = case["parameters"]["variant"]
    generator = torch.Generator().manual_seed(case["seed"])
    dtype = torch.float16 if variant == "half_reuse" else torch.float32
    x = torch.randint(-4, 5, (32, 16), generator=generator).to(dtype)
    y = torch.randint(-4, 5, (16, 16), generator=generator).to(dtype)
    if variant == "half_reuse":
        # Exact FP32 products include fractions below an FP16 ULP; skipping conversion is observable.
        y[:, -1] = 2**-12
    return {
        "x": x,
        "y": y,
        "variant": variant,
        "initial_z": torch.randint(1, 8, (32, 16), generator=generator).float(),
        "initial_after": torch.full((32, 16), 11.0),
    }


def reference(inputs):
    product = inputs["x"].float() @ inputs["y"].float().T
    variant = inputs["variant"]
    if variant == "half_reuse":
        rounded = product.half().float()
        if torch.equal(product, rounded):
            raise AssertionError("the generated case must observe the intermediate half conversion")
        result = rounded.unsqueeze(0)
    elif variant == "overwrite":
        result = (2 * product).unsqueeze(0)
    elif variant == "seeded":
        result = torch.stack((inputs["initial_z"] + product, product))
    else:
        raise ValueError("unknown transfer variant")
    return {"o": result}
