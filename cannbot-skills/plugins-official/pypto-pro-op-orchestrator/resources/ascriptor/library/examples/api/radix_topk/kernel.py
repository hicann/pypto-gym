# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One public `radix_topk` call over a complete UB source.

The source tile is always 4096 lanes; `count` says how many of them are live and the rest hold
-infinity. The destinations are 512 lanes and `k` says how many of those are filled.
"""

from ascriptor.a5 import GM, Position, Tensor, auto_sync, f32, i32, kernel, radix_topk


@kernel(mode="vec", block_dim=1)
def select_largest(src: GM[f32, (1, 4096)], values: GM[f32, (1, 512)], indices: GM[i32, (1, 512)],
                   count: i32, k: i32):
    src_ub = Tensor(f32, [1, 4096], Position.UB)
    values_ub = Tensor(f32, [1, 512], Position.UB)
    indices_ub = Tensor(i32, [1, 512], Position.UB)
    with auto_sync():
        src_ub <<= src
        radix_topk(values_ub, indices_ub, src_ub, count, k)
        values <<= values_ub
        indices <<= indices_ub
    return values, indices
