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
    xf = torch.randn(1, 64, generator=g) * 4
    xi = torch.randint(-1000, 1000, (1, 64), generator=g, dtype=torch.int32)
    xl = torch.randint(-(1 << 40), 1 << 40, (1, 32), generator=g, dtype=torch.int64)
    xl[0, 5] = -(1 << 62)
    xl[0, 9] = (1 << 62) + 12345
    return {"xf": xf, "xi": xi, "xl": xl}


def reference(inputs):
    xf, xi, xl = (inputs[name] for name in ("xf", "xi", "xl"))

    def integer_rows(value):
        out = torch.zeros(3, value.numel(), dtype=value.dtype)
        out[:, 0] = torch.stack((value.sum().to(value.dtype), value.max(), value.min()))
        return out

    return {
        "sum": xf.sum().reshape(1),
        "extrema": torch.stack((xf.max(), xf.min())),
        "upper": torch.zeros(3, 63),
        "int32": integer_rows(xi),
        "int64": integer_rows(xl),
    }
