# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Preserved matmul_e5m2_shortcut device body; numerical references are independent and local."""

from ascriptor.a5 import *

@kernel(mode='cube', block_dim=1)
def matmul_e5m2_shortcut_kernel(x: GM[DT.e5m2, ('M', 'K')], y: GM[DT.e5m2, ('N', 'K')], z: GM[f32, ('M', 'N')], M: i32, N: i32, K: i32):
    l1x = Tensor(DT.e5m2, [M, K], Position.L1)
    l1y = Tensor(DT.e5m2, [N, K], Position.L1)
    l0c = Tensor(DT.float, [M, N], Position.L0C)
    with auto_sync():
        l1x <<= x[:, :]
        l1y <<= y[:, :]
        matmul(l0c, l1x, l1y, m=M, n=N, k=K, is_init=True)
        z[:, :] <<= l0c
    return z
