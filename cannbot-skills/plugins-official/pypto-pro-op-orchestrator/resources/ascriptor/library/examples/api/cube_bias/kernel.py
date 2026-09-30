# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A contiguous bias row is staged once and applied on each init tile."""

import ascriptor.a5 as api


def make_bias(mode="none"):
    if mode not in ("none", "splitn", "splitk"):
        raise ValueError("unknown bias split mode")
    splitn = 32 if mode == "splitn" else None
    splitk = 16 if mode == "splitk" else None

    @api.kernel(mode="cube", block_dim=1)
    def cube_bias(x: api.GM[api.f32, (32, 32)], y: api.GM[api.f32, (64, 32)], bias: api.GM[api.f32, (1, 64)],
        o: api.GM[api.f32, (32, 64)]):
        a = api.Tensor(api.f32, [32, 32], api.Position.L1)
        b = api.Tensor(api.f32, [64, 32], api.Position.L1)
        bias_row = api.Tensor(api.f32, [1, 64], api.Position.L1)
        product = api.Tensor(api.f32, [32, 64], api.Position.L0C)
        with api.auto_sync():
            a <<= x
            b <<= y
            api.gm_to_l1_pad(bias_row, bias)
            api.matmul(product, a, b, m=32, n=64, k=32, splitn=splitn, splitk=splitk, bias=bias_row)
            o <<= product
        return o

    return cube_bias
