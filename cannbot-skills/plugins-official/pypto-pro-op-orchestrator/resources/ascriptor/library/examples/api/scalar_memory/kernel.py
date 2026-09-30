# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Scalar memory is kernel-level work, ordered around the UB copy."""

import importlib


def make_scalar(device="a5", spelling="methods"):
    if device not in ("a2", "a3", "a5", "a5pr") or spelling not in ("methods", "operators"):
        raise ValueError("unsupported device or scalar spelling")
    api = importlib.import_module("ascriptor." + device)

    @api.kernel(mode="vec", block_dim=1)
    def scalar_memory(x: api.GM[api.f32, (1, 8)], o: api.GM[api.f32, (1, 8)]):
        local = api.Tensor(api.f32, [1, 8], api.Position.UB)
        value = api.Var(-999.0, api.f32)
        with api.auto_sync():
            local <<= x
            if spelling == "methods":
                value.GetValueFrom(local[:, 2:3])
            else:
                value <<= local[:, 2:3]
            copied = api.Var(value)
            value += 1.0
            if spelling == "methods":
                value.SetValueTo(local[:, 3:4])
                copied.SetValueTo(local[:, 4:5])
            else:
                value >>= local[:, 3:4]
                copied >>= local[:, 4:5]
            o <<= local
        return o

    return scalar_memory
