# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared A2/A3 vector forms: a counted tail, and a packed compare mask driving a select.

A facade argument selects the profile, so the same source is the A2 and the A3 kernel.
"""

import importlib

from ascriptor.a2 import DT, GM, Position, SelectMode, Tensor, auto_sync, f32


def kernels(device="a2"):
    if device not in ("a2", "a3"):
        raise ValueError("this example supports a2 or a3")
    api = importlib.import_module("ascriptor." + device)

    @api.kernel(mode="vec", block_dim=1)
    def counted_tail(x: GM[f32, (1, "N")], y: GM[f32, (1, "N")], o: GM[f32, (1, "N")]):  # noqa: F821 - DSL shape symbol
        count = x.shape[1]
        capacity = 128
        ub_x = Tensor(f32, [1, capacity], Position.UB)
        ub_y = Tensor(f32, [1, capacity], Position.UB)
        ub_o = Tensor(f32, [1, capacity], Position.UB)
        with auto_sync():
            ub_x <<= x
            ub_y <<= y
            api.muls(ub_o, ub_x, 2.0, count=count)
            api.add(ub_o, ub_o, ub_y, count=count)
            o <<= ub_o
        return o

    @api.kernel(mode="vec", block_dim=1)
    def compare_select(x: GM[f32, (1, 64)], y: GM[f32, (1, 64)], o: GM[f32, (1, 64)]):
        ub_x = Tensor(f32, [1, 64], Position.UB)
        ub_y = Tensor(f32, [1, 64], Position.UB)
        ub_o = Tensor(f32, [1, 64], Position.UB)
        predicate = Tensor(DT.uint8, [1, 32], Position.UB)
        mask_address = Tensor(DT.uint32, [1, 8], Position.UB)
        with auto_sync():
            ub_x <<= x
            ub_y <<= y
            api.compare(predicate, ub_x, ub_y, api.CompareMode.GT)
            api.select(ub_o, predicate, ub_x, ub_y, SelectMode.TENSOR_TENSOR, tmp_addr_buf=mask_address)
            o <<= ub_o
        return o

    return {"tail": counted_tail, "select": compare_select}
