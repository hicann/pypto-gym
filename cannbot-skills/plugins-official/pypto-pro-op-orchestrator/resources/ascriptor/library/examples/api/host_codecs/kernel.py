# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The only thing that runs on the device here: a byte-for-byte carrier transport.

The codecs this folder is about are host functions. What a kernel can say about them is that the
carriers survive a trip through UB unchanged, which is what this moves.
"""

from ascriptor import a5 as api


@api.kernel(mode="vec", block_dim=1)
def carrier_transport(x: api.GM[api.u8, (1, 256)], o: api.GM[api.u8, (1, 256)]):
    local = api.Tensor(api.u8, [1, 256], api.Position.UB)
    with api.auto_sync():
        local <<= x
        o <<= local
    return o
