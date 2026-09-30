# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Scalar mutation on the device: a static unroll, a runtime loop, branches, break and continue.

One core, one INT32 output, no vector arithmetic and no transfer padding. The loop bound arrives as a
scalar argument, so the trip count is decided at run time while the unroll is decided at compile time.
"""

from ascriptor.a2 import GM, Var, i32, kernel, unroll


@kernel(mode="vec", block_dim=1)
def scalar_control(o: GM[i32, (1,)], n: i32):
    total = Var(0)
    for _ignored in unroll(3):
        total += 1
    for index in range(n):
        if index >= 9:
            break
        if index % 2 == 0:
            continue
        if index > 4:
            total += 2 * index
        else:
            total += index
    total.SetValueTo(o[0:1])
    return o
