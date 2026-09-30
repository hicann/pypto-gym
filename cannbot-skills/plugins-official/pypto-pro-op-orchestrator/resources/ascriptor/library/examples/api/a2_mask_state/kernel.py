# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A managed count restores the normal mask before the next uncounted op."""

import importlib


def make_masks(device="a2"):
    if device not in ("a2", "a3"):
        raise ValueError("A2/A3 tensor-vector vocabulary required")
    api = importlib.import_module("ascriptor." + device)

    @api.kernel(mode="vec", block_dim=1)
    def a2_masks(x: api.GM[api.f32, (1, 128)], masked: api.GM[api.f32, (1, 128)],
        counted: api.GM[api.f32, (1, 128)], normal: api.GM[api.f32, (1, 128)]):
        source = api.Tensor(api.f32, [1, 128], api.Position.UB)
        first = api.Tensor(api.f32, [1, 128], api.Position.UB)
        second = api.Tensor(api.f32, [1, 128], api.Position.UB)
        third = api.Tensor(api.f32, [1, 128], api.Position.UB)
        with api.auto_sync():
            source <<= x
            api.dup(first, -9.0, repeat=2)
            api.dup(second, -7.0, repeat=2)
            api.set_mask(0, 7)
            api.adds(first, source, 3.0, repeat=1)
            api.muls(second, source, 2.0, count=70)
            api.adds(third, source, 1.0, repeat=2)
            api.reset_mask()
            masked <<= first
            counted <<= second
            normal <<= third
        return masked, counted, normal

    return a2_masks
