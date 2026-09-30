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

ROWS = 16


@vf()
def logs_vf(x: Tensor, y_ln: Tensor, y_log2: Tensor, y_log10: Tensor):
    xr = Reg(DT.float)
    r = Reg(DT.float)
    for i in range(ROWS):
        xr <<= x[i * 64]
        r <<= xr.ln()
        y_ln[i * 64] <<= r
        r <<= xr.log2()
        y_log2[i * 64] <<= r
        r <<= xr.log10()
        y_log10[i * 64] <<= r


@kernel(mode="vec", block_dim=1)
def vf_log_family(x: GM[f32, (ROWS, 64)], y_ln: GM[f32, (ROWS, 64)], y_log2: GM[f32, (ROWS, 64)],
    y_log10: GM[f32, (ROWS, 64)]):
    ub_x = Tensor(DT.float, [ROWS, 64], Position.UB)
    ub_ln = Tensor(DT.float, [ROWS, 64], Position.UB)
    ub_log2 = Tensor(DT.float, [ROWS, 64], Position.UB)
    ub_log10 = Tensor(DT.float, [ROWS, 64], Position.UB)
    with auto_sync():
        ub_x <<= x
        logs_vf(ub_x, ub_ln, ub_log2, ub_log10)
        y_ln <<= ub_ln
        y_log2 <<= ub_log2
        y_log10 <<= ub_log10
    return y_ln, y_log2, y_log10
