# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F403, F405, F841, F722
"""Reviewed production closures; host inputs/references are local and independent.

Pure-vector teaching units explicitly launch one vector participant. Original
mode=vec bodies, masks, byte footprints and overlapping-write barriers remain.
"""

# SHA256: 943f2f2150cdcd790f892525a951a48d1a8834f8bf05d50fed787e3522bcbdc6

from ascriptor.a5 import *


@kernel(mode="vec", block_dim=1)
def list_split(x: GM[f32, ("N", 64)], ys: GMList[f32, ("?", 64)], N: i32):
    buf = Tensor(DT.float, [8, 64], Position.UB)
    row = Var(0)
    with auto_sync():
        for t in ys:
            rows = t.shape[0]
            for r in range(0, rows, 8):
                buf <<= x[row : row + 8, :]
                t[r : r + 8, :] <<= buf
                row += 8
    return ys
