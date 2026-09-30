# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The same control flow in Python. Imports no DSL.

Read it beside the kernel: the unroll contributes 3, the runtime loop stops at 9 however large `n`
is, even indices are skipped, and an index past 4 counts double.
"""

import torch

LIMIT = 9       # where the kernel breaks out, whatever n says
UNROLLED = 3    # the static unroll's contribution


def make_inputs(case):
    n = case["parameters"]["n"]
    inputs = {"n": n}
    validate(inputs)
    return inputs


def validate(inputs):
    n = inputs["n"]
    if type(n) is not int or isinstance(n, bool):
        raise ValueError("the loop bound is a plain int")
    if not 0 <= n <= 64:
        raise ValueError("the declared loop bound is in [0, 64]")


def total(n):
    return UNROLLED + sum(index * (2 if index > 4 else 1)
                          for index in range(min(n, LIMIT)) if index % 2)


def reference(inputs):
    validate(inputs)
    return {"o": torch.tensor([total(inputs["n"])], dtype=torch.int32)}
