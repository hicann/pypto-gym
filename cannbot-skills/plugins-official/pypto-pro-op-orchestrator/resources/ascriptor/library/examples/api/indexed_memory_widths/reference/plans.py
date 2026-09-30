# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Physical typed shape metadata, with no DSL import."""

# name, source dtype, source elements, index dtype/count, destination dtype/count,
# addressed source/destination width in bytes, number of active indices, operation.
PLANS = [
    ('gather_b8b16_signed', 'int8', 256, 'int16', 128, 'int16', 128, 1, 128, 'gather'),
    ('gather_b8b16_unsigned', 'uint8', 256, 'int16', 128, 'uint16', 128, 1, 128, 'gather'),
    ('gather_b16', 'float16', 128, 'int16', 128, 'float16', 128, 2, 128, 'gather'),
    ('gather_b32', 'float32', 64, 'int32', 64, 'float32', 64, 4, 64, 'gather'),
    ('gather_b64_u32', 'int64', 32, 'int32', 64, 'int64', 32, 8, 32, 'gather'),
    ('gather_b64_u64', 'uint64', 32, 'int64', 32, 'uint64', 32, 8, 32, 'gather'),
    ('scatter_b8', 'uint8', 256, 'int16', 128, 'uint8', 256, 1, 128, 'scatter'),
    ('scatter_b16', 'float16', 128, 'int16', 128, 'float16', 128, 2, 128, 'scatter'),
    ('scatter_b32', 'float32', 64, 'int32', 64, 'float32', 64, 4, 64, 'scatter'),
    ('scatter_b64_u32', 'int64', 32, 'int32', 64, 'int64', 32, 8, 32, 'scatter'),
    ('scatter_b64_u64', 'uint64', 32, 'int64', 32, 'uint64', 32, 8, 32, 'scatter'),
    ('gatherb_b8', 'int8', 512, 'int32', 64, 'int8', 256, 1, 8, 'blocks'),
    ('gatherb_b16', 'float16', 256, 'int32', 64, 'float16', 128, 2, 8, 'blocks'),
    ('gatherb_b32', 'int32', 128, 'int32', 64, 'int32', 64, 4, 8, 'blocks'),
]
