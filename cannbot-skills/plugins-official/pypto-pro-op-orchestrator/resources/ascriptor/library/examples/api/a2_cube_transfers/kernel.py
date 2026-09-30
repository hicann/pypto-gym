# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared A2/A3 fixpipe conversion, atomic accumulation and state restoration."""

import importlib


def make_kernels(device="a2"):
    if device not in ("a2", "a3"):
        raise ValueError("this family supports a2 or a3")
    api = importlib.import_module("ascriptor." + device)

    @api.kernel(mode="cube", block_dim=1)
    def half_reuse(x: api.GM[api.f16, (32, 16)], y: api.GM[api.f16, (16, 16)], e: api.GM[api.f16, (16, 16)],
        z: api.GM[api.f32, (32, 16)]):
        lhs = api.Tensor(api.f16, [32, 16], api.Position.L1)
        rhs = api.Tensor(api.f16, [16, 16], api.Position.L1)
        identity = api.Tensor(api.f16, [16, 16], api.Position.L1)
        converted = api.Tensor(api.f16, [32, 16], api.Position.L1)
        product = api.Tensor(api.f32, [32, 16], api.Position.L0C)
        result = api.Tensor(api.f32, [32, 16], api.Position.L0C)
        with api.auto_sync():
            lhs <<= x
            rhs <<= y
            identity <<= e
            api.matmul(product, lhs, rhs, m=32, n=16, k=16, is_init=True)
            converted <<= product
            api.matmul(result, converted, identity, m=32, n=16, k=16, is_init=True)
            z <<= result
        return z

    @api.kernel(mode="cube", block_dim=1)
    def overwrite_then_atomic(x: api.GM[api.f32, (32, 16)], y: api.GM[api.f32, (16, 16)],
        z: api.GM[api.f32, (32, 16)]):
        lhs = api.Tensor(api.f32, [32, 16], api.Position.L1)
        rhs = api.Tensor(api.f32, [16, 16], api.Position.L1)
        product = api.Tensor(api.f32, [32, 16], api.Position.L0C)
        with api.auto_sync():
            lhs <<= x
            rhs <<= y
            api.matmul(product, lhs, rhs, m=32, n=16, k=16, is_init=True)
            z <<= product
            with api.atomic_add():
                z <<= product
        return z

    @api.kernel(mode="cube", block_dim=1)
    def seeded_atomic_restore(x: api.GM[api.f32, (32, 16)], y: api.GM[api.f32, (16, 16)],
        z: api.GM[api.f32, (32, 16)], after: api.GM[api.f32, (32, 16)]):
        lhs = api.Tensor(api.f32, [32, 16], api.Position.L1)
        rhs = api.Tensor(api.f32, [16, 16], api.Position.L1)
        product = api.Tensor(api.f32, [32, 16], api.Position.L0C)
        with api.auto_sync():
            lhs <<= x
            rhs <<= y
            api.matmul(product, lhs, rhs, m=32, n=16, k=16, is_init=True)
            with api.atomic_add():
                z <<= product
            after <<= product
        return z, after

    return {"half_reuse": half_reuse, "overwrite": overwrite_then_atomic, "seeded": seeded_atomic_restore}
