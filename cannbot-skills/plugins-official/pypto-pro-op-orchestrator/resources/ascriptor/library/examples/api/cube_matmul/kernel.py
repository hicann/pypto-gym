# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One 16x16 FP16 cube tile, through each of the four device facades.

The L1 operands are NZ tiles and the destination is a complete ND FP32 matrix. The facade is a
factory argument, so the same source text is the kernel on A2, A3, A5 and A5PR.
"""

import importlib

from ascriptor.a5 import DT, GM, Position, Tensor, auto_sync, f16, f32, matmul


def make_kernel(device="a5"):
    if device not in ("a2", "a3", "a5", "a5pr"):
        raise ValueError("device must be a2, a3, a5 or a5pr")
    api = importlib.import_module("ascriptor." + device)

    @api.kernel(mode="cube", block_dim=1)
    def cube_matmul(x: GM[f16, (16, 16)], y: GM[f16, (16, 16)], o: GM[f32, (16, 16)]):
        lhs = Tensor(DT.half, [16, 16], Position.L1)
        rhs = Tensor(DT.half, [16, 16], Position.L1)
        accum = Tensor(DT.float, [16, 16], Position.L0C)
        with auto_sync():
            lhs <<= x
            rhs <<= y
            matmul(accum, lhs, rhs)
            o <<= accum
        return o

    return cube_matmul
