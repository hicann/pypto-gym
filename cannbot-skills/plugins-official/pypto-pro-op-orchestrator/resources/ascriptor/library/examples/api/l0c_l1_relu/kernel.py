# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""ReLU and FP16 L0C-to-L1 feedback before a second A5 cube product."""
from ascriptor.a5 import *


@kernel(mode="cube", block_dim=1)
def relu_reuse(x: GM[f16, (16, 32)], y: GM[f16, (16, 32)],
               z: GM[f16, (16, 16)], out: GM[f32, (16, 16)]):
    l1x = Tensor(DT.half, [16, 32], Position.L1)
    l1y = Tensor(DT.half, [16, 32], Position.L1)
    l1z = Tensor(DT.half, [16, 16], Position.L1)
    l1mid = Tensor(DT.half, [16, 16], Position.L1)
    l0c_mid = Tensor(DT.float, [16, 16], Position.L0C)
    l0c_out = Tensor(DT.float, [16, 16], Position.L0C)
    event_fix_mte1 = SEvent(Pipe.FIX, Pipe.MTE1)
    with auto_sync():
        l1x <<= x[:, :]
        l1y <<= y[:, :]
        l1z <<= z[:, :]
        matmul(l0c_mid, l1x, l1y, m=16, n=16, k=32, is_init=True)
        l1mid <<= l0c_mid.relu()
        event_fix_mte1.set()
        event_fix_mte1.wait()
        matmul(l0c_out, l1mid, l1z, m=16, n=16, k=16, is_init=True)
        out[:, :] <<= l0c_out
    return out
