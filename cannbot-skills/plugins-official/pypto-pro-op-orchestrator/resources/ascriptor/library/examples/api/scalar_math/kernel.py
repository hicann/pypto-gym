# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Concrete scalar helpers produce fully observable typed GM outputs."""

import ascriptor.a5 as api


@api.kernel(mode="vec", block_dim=1)
def scalar_math(integers: api.GM[api.i64, (1, 24)], floats: api.GM[api.f32, (1, 4)], n: api.i64):
    a = api.Var(n, api.i64)
    b = api.Var(8, api.i64)
    unsigned = api.Var(2**63 + 15, api.u64)
    integer_values = [
        api.var_add(a, b),
        api.var_sub(a, b),
        api.var_mul(a, b),
        api.var_div(a, b),
        api.var_mod(a, b),
        api.var_and(a, b),
        api.var_or(a, b),
        api.var_xor(a, b),
        api.var_shl(a, 2),
        api.var_shr(a, 2),
        api.var_inv(a),
        api.Min(a, b),
        api.Max(a, b),
        api.CeilDiv(a, b),
        api.Align8(a),
        api.Align16(a),
        api.Align32(a),
        api.Align64(a),
        api.Align128(a),
        api.Align256(a),
        api.GetVecNum(),
        api.GetVecIdx(),
        api.GetSubBlockIdx(),
        api.var_shr(unsigned, 63),
    ]
    for index in api.unroll(24):
        value = api.Var(integer_values[index], api.i64)
        value.SetValueTo(integers[:, index : index + 1])
    numerator = api.Var(1.5, api.f32)
    denominator = api.Var(2.0, api.f32)
    negative = api.Var(-n, api.i64)
    float_values = [
        api.var_div(1.5, 2.0),
        api.var_div(numerator, denominator),
        api.scalar_sqrt(9.0),
        api.scalar_abs(negative),
    ]
    for index in api.unroll(4):
        value = api.Var(float_values[index], api.f32)
        value.SetValueTo(floats[:, index : index + 1])
    return integers, floats
