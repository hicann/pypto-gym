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
    return {"n": case["parameters"]["n"]}


def reference(inputs):
    n = inputs["n"]
    aligned = [((n + a - 1) // a) * a for a in (8, 16, 32, 64, 128, 256)]
    values = [
        n + 8,
        n - 8,
        n * 8,
        n // 8,
        n % 8,
        n & 8,
        n | 8,
        n ^ 8,
        n << 2,
        n >> 2,
        ~n,
        min(n, 8),
        max(n, 8),
        (n + 7) // 8,
        *aligned,
        1,
        0,
        0,
        1,
    ]
    return {
        "integers": torch.tensor([values], dtype=torch.int64),
        "floats": torch.tensor([[0.75, 0.75, 3.0, float(n)]]),
    }
