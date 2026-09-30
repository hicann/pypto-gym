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
    mode = case["parameters"]["variant"]
    g = torch.Generator().manual_seed(case["seed"])
    if mode.startswith("b64"):
        big = torch.tensor([-268435472, 268435471, 2**30 - 1, -(2**30) + 1] * 16, dtype=torch.int32)
        x32 = torch.stack(
            (
                torch.arange(64, dtype=torch.int32),
                -torch.arange(64, dtype=torch.int32),
                big,
                torch.randint(-(2**31), 2**31 - 1, (64,), dtype=torch.int32, generator=g),
            )
        )
        x64 = torch.stack(
            (
                torch.arange(32, dtype=torch.int64),
                torch.tensor([-268435472, 268435471] * 16),
                torch.arange(32, dtype=torch.int64) * 2**20,
                torch.randint(-(2**40), 2**40, (32,), dtype=torch.int64, generator=g),
            )
        )
        xf = torch.stack(
            (
                torch.arange(64) + 0.5,
                -(torch.arange(64) + 0.5),
                torch.full((64,), 2.0**33),
                torch.randn(64, generator=g) * 1e6,
            )
        )
        operands = (x32, x64) if mode == "b64_widen" else (x64, xf)
    elif mode == "i4_narrow":
        ramp = torch.arange(-16, 16).float().repeat(4)
        xh = torch.stack((ramp, torch.arange(128) / 8.0 - 8.0, -ramp, torch.randn(128, generator=g) * 6)).half()
        xi = torch.stack((ramp, -ramp, torch.full((128,), 7.0), torch.full((128,), -8.0))).short()
        operands = (xh, xi)
    else:
        operands = (
            torch.stack(
                (
                    torch.arange(64, dtype=torch.uint8),
                    torch.arange(128, 192, dtype=torch.uint8),
                    torch.full((64,), 0x8F, dtype=torch.uint8),
                    torch.full((64,), 0x70, dtype=torch.uint8),
                )
            ),
        )
    return {"variant": mode, "operands": operands}


def rtz_int64_to_float32(value):
    # All inputs fit exactly in float64. Correct a nearest-rounded FP32 only
    # when its magnitude overshoots the integer; nextafter then gives RTZ.
    nearest = value.float()
    overshot = nearest.double().abs() > value.double().abs()
    return torch.where(overshot, torch.nextafter(nearest, torch.zeros_like(nearest)), nearest)


def pack_int4(value):
    nibble = value.to(torch.int16).clamp(-8, 7) & 15
    return (nibble[:, ::2] | (nibble[:, 1::2] << 4)).to(torch.uint8)


def reference(inputs):
    mode = inputs["variant"]
    operands = inputs["operands"]
    if mode == "b64_widen":
        x32, x64 = operands
        values = (x32[:, :32].long(), x64.int())
    elif mode == "b64_float":
        x64, xf = operands
        values = (rtz_int64_to_float32(x64), xf[:, :32].trunc().long())
    elif mode == "i4_narrow":
        xh, xi = operands
        values = (pack_int4(xh.float().round()), pack_int4(xi))
    else:
        codes = operands[0].short()
        nibbles = torch.empty((4, 128), dtype=torch.int16)
        nibbles[:, ::2] = codes & 15
        nibbles[:, 1::2] = codes >> 4
        signed = torch.where(nibbles >= 8, nibbles - 16, nibbles)
        values = (signed.half(), signed.bfloat16(), signed)
    return {"carriers": torch.cat([value.contiguous().view(torch.uint8).flatten() for value in values])}
