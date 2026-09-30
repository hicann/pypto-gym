# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A2/A3 cube-to-vector ownership through a two-slot GM workspace.

A2 and A3 have no direct L0C-to-UB route, so the bridge goes through GM. The cube publishes a 32x16
FP32 product into one slot of a two-slot `GMBuff`; each vector subblock then reads its own 16 rows
back and scales them. `CvMutex` returns the slot only after *both* vector readers have finished their
MTE2 loads. Local `auto_sync()` covers each vector's UB buffers and its final store.

Three beats over two slots, so the third beat reuses the first slot.
"""

import importlib


def make_kernel(device="a2"):
    if device not in ("a2", "a3"):
        raise ValueError("cube_vector_bridge supports the shared a2/a3 family")
    api = importlib.import_module("ascriptor." + device)
    from ascriptor.a2 import CvMutex, GMBuff, GetSubBlockIdx, Pipe, Position, Tensor, auto_sync, f16, f32, matmul

    @api.kernel(mode="mix", block_dim=1)
    def cube_vector_bridge(x: api.GM[api.f16, (3, 32, 16)], y: api.GM[api.f16, (16, 16)], o: api.GM[api.f32, (3, 32, 16)]):
        left = Tensor(f16, [32, 16], Position.L1)
        right = Tensor(f16, [16, 16], Position.L1)
        product = Tensor(f32, [32, 16], Position.L0C)
        incoming = Tensor(f32, [16, 16], Position.UB)
        outgoing = Tensor(f32, [16, 16], Position.UB)
        workspace = GMBuff(f32, [32, 16], slots=2)
        published = CvMutex(0, depth=1, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)
        with auto_sync():
            right <<= y
            for beat in range(3):
                left <<= x[beat, :, :]
                matmul(product, left, right, m=32, n=16, k=16)
                published.lock()
                workspace[beat] <<= product
                published.ready()

                published.wait()
                begin = GetSubBlockIdx() * 16
                incoming <<= workspace[beat][begin:begin + 16, :]
                published.free()
                api.muls(outgoing, incoming, 2.0, count=256)
                o[beat, begin:begin + 16, :] <<= outgoing
        return o

    return cube_vector_bridge
