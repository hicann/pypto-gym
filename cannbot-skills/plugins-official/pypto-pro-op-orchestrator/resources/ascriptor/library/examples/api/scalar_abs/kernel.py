# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Dynamic A5 scalar absolute value, with observable native-width results."""

import ascriptor.a5 as api


def entry_for(kind):
    dtype = getattr(api, kind)

    @api.kernel(mode="vec", block_dim=1)
    def scalar_abs_dynamic(source: api.GM[dtype, (2, 8)], output: api.GM[dtype, (2, 8)]):
        for row in range(2):
            for col in range(8):
                value = api.Var(0, dtype)
                value.GetValueFrom(source[row:row + 1, col:col + 1])
                magnitude = api.Var(api.scalar_abs(value), dtype)
                magnitude.SetValueTo(output[row:row + 1, col:col + 1])
        return output

    return scalar_abs_dynamic
