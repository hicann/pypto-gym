# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Explicit local buffer ownership with observable outstanding event depth."""

import ascriptor.a5 as api


def make_ring(depth=3, *, synchronize=True):
    if depth not in (2, 3, 4):
        raise ValueError("depth must be 2, 3 or 4")
    buffer = (api.DBuff, api.TBuff, api.QBuff)[depth - 2]
    event = (api.DEvent, api.TEvent, api.QEvent)[depth - 2]
    iterations = 2 * depth + 1

    @api.kernel(mode="vec", block_dim=1)
    def event_ring(x: api.GM[api.f32, (iterations, 64)], o: api.GM[api.f32, (iterations, 64)]):
        incoming = buffer(api.f32, [1, 64], api.Position.UB)
        ready = event(api.Pipe.MTE2, api.Pipe.MTE3)
        available = event(api.Pipe.MTE3, api.Pipe.MTE2, preset=True)
        beat = api.Var(0)
        for batch in api.unroll(3):
            count = api.Min(depth, iterations - batch * depth)
            for index in range(count):
                beat.set(batch * depth + index)
                source = incoming[beat]
                available.wait()
                source <<= x[beat : beat + 1, :]
                if synchronize:
                    ready.set()
            for index in range(count):
                beat.set(batch * depth + index)
                source = incoming[beat]
                if synchronize:
                    ready.wait()
                o[beat : beat + 1, :] <<= source
                available.set()
        return o

    return event_ring
