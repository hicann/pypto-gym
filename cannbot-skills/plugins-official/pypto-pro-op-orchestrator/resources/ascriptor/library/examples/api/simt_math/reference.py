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
    x = torch.randn(1, 64, generator=g).clamp(-4, 4)
    if case["parameters"]["edges"]:
        x[0, :10] = torch.tensor([-1.25, -1.15, -0.05, -0.0, 0.0, 0.05, 1.15, 1.25, -4.0, 4.0])
    specials = x.clone()
    specials[0, :3] = torch.tensor([float("nan"), float("inf"), float("-inf")])
    bits = torch.randint(-(1 << 31), (1 << 31) - 1, (1, 64), generator=g, dtype=torch.int32)
    bits[0, :4] = torch.tensor([0, -1, -(1 << 31), (1 << 31) - 1])
    return {"x": x, "specials": specials, "bits": bits}


def reference(inputs):
    x = inputs["x"].flatten()
    v10 = x * 10
    shifted = x + 4.5
    specials = inputs["specials"].flatten()
    approximate = torch.stack(
        (
            x.exp(),
            x.exp2(),
            shifted.log(),
            shifted.log2(),
            shifted.log1p(),
            (x * 37).sin(),
            (x * 37).cos(),
            x.tanh(),
            shifted.rsqrt(),
        )
    )
    away = torch.copysign(torch.floor(v10.abs() + 0.5), v10)
    rounding = torch.stack((v10.round(), away, v10.floor(), v10.ceil(), v10.trunc()))
    signed = inputs["bits"].flatten().tolist()
    words = [value & 0xFFFFFFFF for value in signed]
    integer = torch.stack(
        (
            specials.isnan().int(),
            specials.isinf().int(),
            specials.isfinite().int(),
            torch.tensor([word.bit_count() for word in words], dtype=torch.int32),
            torch.tensor([(value * value) >> 32 for value in signed], dtype=torch.int32),
            torch.tensor([(word & -word).bit_length() for word in words], dtype=torch.int32),
        )
    )
    return {
        "approximate": approximate,
        "rounding": rounding,
        "fmod": torch.fmod(v10, 1.7),
        "fma": x * 2 + 1,
        "integer": integer,
    }
