# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F403, F405, F841
"""Reviewed production closures; host inputs/references are local and independent.

Pure-vector teaching units explicitly launch one vector participant. Original
mode=vec bodies, masks, byte footprints and overlapping-write barriers remain.
"""

from ascriptor.a5 import *  # noqa: F401,F403

ROWS = 4

TILE = 16  # 16 f32 = 64 bytes = two 32-byte blocks

BURST = 9  # 36 bytes: the tail is the 28 bytes up to the next 32-byte boundary

PAD = -1.5  # 0xBFC00000, a value no input carries


@kernel(mode="vec", block_dim=1)
def dma_pad_value(x: GM[f32, (ROWS, TILE)], padded: GM[f32, (ROWS, TILE)]):
    ub = Tensor(DT.float, [1, TILE], Position.UB, name="pv_ub")
    rows_per_vec = CeilDiv(ROWS, GetVecNum())
    row_begin = Var(rows_per_vec * GetVecIdx())
    row_end = Min(row_begin + rows_per_vec, ROWS)
    with auto_sync():
        for row in range(row_begin, row_end):
            gm_to_ub_pad(ub, x[row : row + 1, 0:BURST], n_burst=1, burst_len_element=BURST,
                src_stride_element=0, dst_stride=0, pad=PAD)
            padded[row : row + 1, :] <<= ub
    return padded
