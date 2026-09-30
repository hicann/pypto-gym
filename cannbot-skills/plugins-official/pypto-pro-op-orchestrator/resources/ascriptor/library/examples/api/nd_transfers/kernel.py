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

from ascriptor.a5 import *

ROWS, COLS = 32, 64


@kernel(mode="vec", block_dim=1)
def nd_dma_pad(x: GM[f32, (ROWS, COLS)], xh: GM[bf16, (ROWS, COLS)], y_const: GM[f32, (16, 32)],
    y_near: GM[f32, (16, 32)], y_rows: GM[f32, (8, 32)], y_t: GM[f32, (32, 16)], yh: GM[bf16, (32, 64)]):
    ub_c = Tensor(DT.float, [16, 32], Position.UB)
    ub_n = Tensor(DT.float, [16, 32], Position.UB)
    ub_r = Tensor(DT.float, [8, 32], Position.UB)
    ub_t = Tensor(DT.float, [32, 16], Position.UB)
    ub_h = Tensor(DT.bfloat16, [32, 64], Position.UB)
    with auto_sync():
        # x[2:14, 3:27] into the middle of a 16 x 32 tile: 3 / 5 pad columns, 2 / 2 pad rows, filled with -1.5
        gm_to_ub_nd_dma(ub_c, x[2:14, 3:27], [1, COLS], [1, 32], [24, 12], loop_left_pad=[3, 2],
            loop_right_pad=[5, 2], constant_value=-1.5)
        y_const <<= ub_c
        # the same window; a padded position repeats the nearest source element of its loop
        gm_to_ub_nd_dma(ub_n, x[2:14, 3:27], [1, COLS], [1, 32], [24, 12], loop_left_pad=[3, 2],
            loop_right_pad=[5, 2], nearest_value_mode=True)
        y_near <<= ub_n
        # three loops: 32 columns, 4 rows two apart, 2 blocks eight rows apart -> rows 0, 2, ..., 14 of x[:, :32]
        gm_to_ub_nd_dma(ub_r, x[0:16, 0:32], [1, 2 * COLS, 8 * COLS], [1, 32, 128], [32, 4, 2])
        y_rows <<= ub_r
        # x[0:16, 0:32] transposed to 32 x 16 (the sugar: rows walk the innermost loop)
        gm_to_ub_nd_dma_transpose(ub_t, x[0:16, 0:32])
        y_t <<= ub_t
        # bf16: xh[4:20, 0:48] with 8 pad elements on every side of both loops (config pads), filled with 2.0
        gm_to_ub_nd_dma(ub_h, xh[4:20, 0:48], [1, COLS], [1, 64], [48, 16], config_left_pad=8,
            config_right_pad=8, constant_value=2.0, fence="mte2")
        yh <<= ub_h
    return y_const, y_near, y_rows, y_t, yh
