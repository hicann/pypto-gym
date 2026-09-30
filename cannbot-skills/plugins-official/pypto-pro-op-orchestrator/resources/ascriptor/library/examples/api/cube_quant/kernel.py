# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The accumulator/destination dtype pair selects the fixpipe quant mode."""

import ascriptor.a5 as api


def make_quant(mode):
    if mode not in ("fp32_i8", "fp32_u8", "int32_i8", "int32_f16", "fp32_f16"):
        raise ValueError("unknown quant mode")
    integer = mode.startswith("int32")
    input_dtype = api.i8 if integer else api.f16
    accum_dtype = api.i32 if integer else api.f32
    output_dtype = api.f16 if mode.endswith("f16") else (api.u8 if mode.endswith("u8") else api.i8)
    scale = 0.25 if mode == "int32_f16" else 0.5
    offset = 8 if mode in ("fp32_i8", "fp32_u8") else 0

    @api.kernel(mode="cube", block_dim=1)
    def cube_quant(x: api.GM[input_dtype, (32, 32)], y: api.GM[input_dtype, (32, 32)],
        o: api.GM[output_dtype, (32, 32)]):
        a = api.Tensor(input_dtype, [32, 32], api.Position.L1)
        b = api.Tensor(input_dtype, [32, 32], api.Position.L1)
        product = api.Tensor(accum_dtype, [32, 32], api.Position.L0C)
        with api.auto_sync():
            a <<= x
            b <<= y
            api.matmul(product, a, b, m=32, n=32, k=32)
            o <<= product.requant(scale=scale, offset=offset)
        return o

    return cube_quant
