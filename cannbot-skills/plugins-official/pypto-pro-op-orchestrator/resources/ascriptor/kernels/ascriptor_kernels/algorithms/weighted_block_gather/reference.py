# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent block addressing in Python ints with a separate half rounding after every
product and every sum."""

import struct
import torch


def half(value):
    """Round one Python float to IEEE binary16 and back. Every arithmetic result the kernel
    stores as a half is rounded here too, so the two agree bit for bit or the kernel is
    fusing (or re-rounding) something it should not."""
    return struct.unpack("<e", struct.pack("<e", value))[0]


def make_inputs(case):
    if type(case.get("block_dim", 1)) is not int or case.get("block_dim", 1) != 1:
        raise ValueError("The fixed whole-output recipe has one vector owner")
    if case["parameters"].get("inner_rep") != 4:
        raise ValueError("The complete recipe requires four gather repeats")
    mode = case["parameters"]["mode"]
    generator = torch.Generator().manual_seed(case["seed"])
    src = torch.arange(1, 513, dtype=torch.float16).reshape(1, 512)
    offsets = (torch.arange(32, dtype=torch.int32) * 32).reshape(1, 32)
    weights = torch.arange(2, 18, dtype=torch.float16).reshape(1, 16)
    if mode == "permuted":
        offsets = (torch.randperm(32, generator=generator).int() * 32).reshape(1, 32)
        weights[0, :4] = torch.tensor([1.25, -2.5, 0.375, 3.0], dtype=torch.float16)
    elif mode == "repeated":
        offsets = (torch.tensor([31, 0, 5, 7, 31, 0, 3, 19] * 4, dtype=torch.int32) * 32).reshape(1, 32)
        src = (torch.randint(-256, 257, (1, 512), generator=generator).float() / 16).half()
        weights[0, :4] = torch.tensor([-0.75, 2.25, 1.5, -1.0], dtype=torch.float16)
    elif mode == "product_rounding":
        src[0, :128] = 1.0009765625
        src[0, 128:256] = 1.001953125
        weights[0, :2] = torch.tensor([1.0009765625, -1.0], dtype=torch.float16)
    elif mode == "half_midpoint":
        src[0, :128] = 1.0029296875
        src[0, 128:256] = 0
        weights[0, :2] = torch.tensor([1.5, 0.0], dtype=torch.float16)
    elif mode == "unused_weights":
        weights[0, 4:] = torch.linspace(-7, 7, 12).half()
    elif mode != "source_literal":
        raise ValueError("Unknown generated weighted-gather case")
    return {"src": src, "offsets": offsets, "weights": weights, "inner_rep": 4, "block_dim": 1}


def reference(inputs):
    """Decode the byte offsets by hand: offset // 2 is the half index of the block's first
    value, and the sixteen halves of that block follow it contiguously. Repeat r reuses
    weights[r] across all eight of its blocks; taps 0+1 and 2+3 then add into two 128-value
    rows. Nothing here reads the kernel, the DSL or a simulator."""
    src = inputs["src"].flatten().tolist()
    offsets = inputs["offsets"].flatten().tolist()
    weights = inputs["weights"].flatten().tolist()
    products = []
    for repeat in range(4):
        products.append([half(src[offsets[repeat * 8 + lane // 16] // 2 + lane % 16] * weights[repeat])
                         for lane in range(128)])
    output = [half(a + b) for tap in (0, 2)
              for a, b in zip(products[tap], products[tap + 1], strict=True)]
    return {"output": torch.tensor([output], dtype=torch.float16)}
