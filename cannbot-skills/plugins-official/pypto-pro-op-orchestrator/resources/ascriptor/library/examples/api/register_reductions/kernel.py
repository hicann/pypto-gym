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

# SHA256: 0513e9952a31eb6b304f0550051e1ec41ec32bef4b6baa8120abc872dda12a87

from ascriptor.a5 import *


@vf()
def reduce_vf(xf: Tensor, xi: Tensor, xl: Tensor, of: Tensor, oi: Tensor, ol: Tensor):
    rf = Reg(DT.float)
    ri = Reg(DT.int)
    rl = Reg(DT.int64)
    rf <<= xf[0]
    ri <<= xi[0]
    rl <<= xl[0]
    for src, out, dt, cols in ((rf, of, DT.float, 64), (ri, oi, DT.int, 64), (rl, ol, DT.int64, 32)):
        r_add = Reg(dt)
        r_max = Reg(dt)
        r_min = Reg(dt)
        cadd(r_add, src)
        cmax(r_max, src)
        cmin(r_min, src)
        out[0] <<= r_add  # one full register per row (element offsets)
        out[cols] <<= r_max
        out[2 * cols] <<= r_min


@kernel(mode="vec", block_dim=1)
def reduce_family(xf: GM[f32, (1, 64)], xi: GM[i32, (1, 64)], xl: GM[i64, (1, 32)], of: GM[f32, (3, 64)],
    oi: GM[i32, (3, 64)], ol: GM[i64, (3, 32)]):
    ub_xf = Tensor(DT.float, [1, 64], Position.UB)
    ub_xi = Tensor(DT.int, [1, 64], Position.UB)
    ub_xl = Tensor(DT.int64, [1, 32], Position.UB)
    ub_of = Tensor(DT.float, [3, 64], Position.UB)
    ub_oi = Tensor(DT.int, [3, 64], Position.UB)
    ub_ol = Tensor(DT.int64, [3, 32], Position.UB)
    with auto_sync():
        ub_xf <<= xf
        ub_xi <<= xi
        ub_xl <<= xl
        reduce_vf(ub_xf, ub_xi, ub_xl, ub_of, ub_oi, ub_ol)
        of <<= ub_of
        oi <<= ub_oi
        ol <<= ub_ol
    return of, oi, ol
