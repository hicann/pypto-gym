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
    fmt = case["parameters"]["format"]
    g = torch.Generator().manual_seed(case["seed"])
    if fmt == "fp4_mixed":
        a = torch.randint(0, 256, (16, 64), generator=g, dtype=torch.uint8)
        b = torch.randint(0, 256, (16, 64), generator=g, dtype=torch.uint8)
    else:
        pool = torch.tensor([0x30, 0x38, 0x3C, 0x40] if fmt == "e4m3" else [0x38, 0x3C, 0x3E, 0x40], dtype=torch.int32)
        a = (
            pool[torch.randint(0, 4, (16, 128), generator=g)]
            | torch.randint(0, 2, (16, 128), generator=g) * 128
        ).to(torch.uint8)
        b = (
            pool[torch.randint(0, 4, (16, 128), generator=g)]
            | torch.randint(0, 2, (16, 128), generator=g) * 128
        ).to(torch.uint8)
    sa = (125 + (torch.arange(16)[:, None] + torch.arange(4)[None, :]) % 5).to(torch.uint8)
    sb = (125 + (2 * torch.arange(16)[:, None] + 3 * torch.arange(4)[None, :]) % 5).to(torch.uint8)
    return {"a": a, "b": b, "scale_a": sa, "scale_b": sb, "format": fmt, "path": case["parameters"]["path"]}


def decode_fp8(codes, fmt):
    bits = 3 if fmt == "e4m3" else 2
    bias = 7 if fmt == "e4m3" else 15
    word = codes.int()
    exponent = (word & 127) >> bits
    fraction = word & ((1 << bits) - 1)
    magnitude = (1 + fraction.float() / (1 << bits)) * torch.pow(2.0, exponent - bias)
    magnitude = torch.where(exponent == 0, fraction.float() * 2.0 ** (1 - bias - bits), magnitude)
    return torch.where(word & 128 != 0, -magnitude, magnitude)


def decode_fp4(codes, e2m1):
    table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] if e2m1 else [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75])
    nibble = torch.empty((16, 128), dtype=torch.int64)
    nibble[:, ::2] = (codes & 15).long()
    nibble[:, 1::2] = (codes >> 4).long()
    magnitude = table[nibble & 7]
    return torch.where(nibble & 8 != 0, -magnitude, magnitude)


def reference(inputs):
    fmt = inputs["format"]
    a = decode_fp4(inputs["a"], True) if fmt == "fp4_mixed" else decode_fp8(inputs["a"], fmt)
    b = decode_fp4(inputs["b"], False) if fmt == "fp4_mixed" else decode_fp8(inputs["b"], fmt)
    sa = torch.pow(2.0, inputs["scale_a"].int() - 127).repeat_interleave(32, dim=1)
    sb = torch.pow(2.0, inputs["scale_b"].int() - 127).repeat_interleave(32, dim=1)
    return {"o": (a * sa) @ (b * sb).T}
