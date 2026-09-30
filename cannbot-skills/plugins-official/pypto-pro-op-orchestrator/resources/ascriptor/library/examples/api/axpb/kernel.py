# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One register, two operands, three instructions: the smallest complete vector kernel."""

from ascriptor.a5 import DT, GM, Position, Reg, Tensor, add, auto_sync, f32, kernel, muls, vf


@vf
def axpb_vf(x_ub: Tensor, y_ub: Tensor, o_ub: Tensor):
    xr = Reg(DT.float, name="x")
    yr = Reg(DT.float, name="y")
    xr <<= x_ub[0]
    yr <<= y_ub[0]
    scaled = Reg(DT.float, name="scaled")
    muls(scaled, xr, 2.0)
    total = Reg(DT.float, name="total")
    add(total, scaled, yr)
    o_ub[0] <<= total


@kernel(mode="vec", block_dim=1)
def axpb(x: GM[f32, (1, 64)], y: GM[f32, (1, 64)], o: GM[f32, (1, 64)]):
    ub_x = Tensor(DT.float, [1, 64], Position.UB, name="ub_x")
    ub_y = Tensor(DT.float, [1, 64], Position.UB, name="ub_y")
    ub_o = Tensor(DT.float, [1, 64], Position.UB, name="ub_o")
    with auto_sync():
        ub_x <<= x
        ub_y <<= y
        axpb_vf(ub_x, ub_y, ub_o)
        o <<= ub_o
    return o
