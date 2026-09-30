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
    offset = case["seed"] - 8871
    if mode in ("i64", "reduce"):
        x = (torch.arange(64, dtype=torch.int64) * 1234567 + 2**32 + offset * 65537).reshape(1, 64)
        x[:, ::3] *= -1
        y = (torch.arange(64, dtype=torch.int64) % 13 + 1).reshape(1, 64)
        y[:, ::2] *= -1
        operands = (x, y)
    elif mode == "cast":
        operands = (
            ((torch.arange(64, dtype=torch.int32) - 32) * 8193 + offset).reshape(1, 64),
            (torch.arange(64, dtype=torch.float32) - 32.25 + offset * 0.5).reshape(1, 64),
        )
    elif mode == "memory":
        operands = (
            torch.tensor([2**63 + i * 1234567 + offset for i in range(64)], dtype=torch.uint64).reshape(1, 64),
        )
    else:
        n, dtype = (128, torch.complex32) if mode == "complex" else (64, torch.complex64)
        r = torch.arange(n, dtype=torch.float32).reshape(1, n)
        operands = ((r / 8 + offset / 4 + 1j * ((r % 7) / 4)).to(dtype), (((r % 5) - 2) / 2 + 0.5j).to(dtype))
    return {"variant": mode, "operands": operands}


def reference(inputs):
    mode = inputs["variant"]
    operands = inputs["operands"]
    if mode == "i64":
        a, b = (value.flatten() for value in operands)
        fused = a * b + a
        fused[47:] = 0
        increasing = torch.tensor([18014398509481991 + i for i in range(64)], dtype=torch.int64)
        decreasing = torch.tensor([-18014398509481991 - i for i in range(64)], dtype=torch.int64)
        values = (
            torch.stack((a + b, a - b, a * b, a & b, a % b, fused, torch.maximum(a, b), increasing, decreasing)),
        )
    elif mode == "reduce":
        a, b = (value.flatten() for value in operands)
        out = torch.zeros(7, 64, dtype=torch.int64)
        out[0, 0] = a.sum()
        out[1, 0] = a.max()
        out[2, 0] = a.min()
        out[3:5] = torch.stack((a, b), dim=1).flatten().reshape(2, 64)
        out[5] = a
        out[6] = b
        values = (out,)
    elif mode == "cast":
        x, y = operands
        values = (x.long(), x.clone(), x.float(), y.trunc().long())
    elif mode == "memory":
        source = operands[0].flatten().tolist()
        rows = [[0] * 64 for _ in range(3)]
        for i in range(47):
            rows[0][i] = source[63 - i]
            rows[1][63 - i] = source[i]
        rows[2] = [value >> 17 for value in source]
        values = (torch.tensor(rows, dtype=torch.uint64),)
    else:
        a, b = (value.to(torch.complex64).flatten() for value in operands)
        product = a * b
        if mode == "complex":
            tail = product.clone()
            tail[95:] = 0
            woven = product.clone()
            woven[65::2] = 0
            values = (torch.stack((a + b, tail, a + (1 + 2j), woven)).to(torch.complex32),)
        else:
            product[47:] = 0
            values = (torch.stack((a + b, product)),)
    return {"carriers": torch.cat([value.contiguous().view(torch.uint8).flatten() for value in values])}
