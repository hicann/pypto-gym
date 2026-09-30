# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A reshape and a non-contiguous GM view. Strides and the offset are counted in elements.

The physical input holds 256 FP32 elements; the view's logical shape is [2, 16] with strides
[128, 2] from element 3, so the last element it reads is 161. A non-unit innermost stride is
supported for a read; the matching write is a gap.
"""

from ascriptor.a5 import GM, Position, Tensor, auto_sync, f32, kernel


@kernel(mode="vec", block_dim=1)
def strided_views(x: GM[f32, (4, 64)], o: GM[f32, (2, 16)]):
    view = x.flatten().view([2, 16], [128, 2], offset=3)
    local = Tensor(f32, [2, 16], Position.UB)
    with auto_sync():
        local <<= view
        o <<= local
    return o
