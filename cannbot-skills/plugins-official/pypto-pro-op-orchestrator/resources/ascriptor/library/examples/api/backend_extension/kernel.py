# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The one kernel this example needs: a copy, so the backend is the only variable."""

from ascriptor.a5 import GM, Position, Tensor, auto_sync, f32, kernel


@kernel(mode="vec", block_dim=1)
def copy_tile(x: GM[f32, (1, 64)], o: GM[f32, (1, 64)]):
    local = Tensor(f32, [1, 64], Position.UB)
    with auto_sync():
        local <<= x
        o <<= local
    return o
