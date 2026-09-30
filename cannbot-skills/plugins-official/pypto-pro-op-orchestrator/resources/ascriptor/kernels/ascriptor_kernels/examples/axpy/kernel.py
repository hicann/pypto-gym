# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Preserve the corrected prototype pypto_smoke arithmetic and one-core footprint.

Migration edits rename the kernel, separate host code, and remove recorded data.
See contract.json for immutable source identity and content digest.
"""

from ascriptor.a5 import *


@vf()
def axpb_vf(x_ub: Tensor, y_ub: Tensor, o_ub: Tensor):
    xr = Reg(DT.float, name="ps_x")
    yr = Reg(DT.float, name="ps_y")
    xr <<= x_ub[0]
    yr <<= y_ub[0]
    sx = Reg(DT.float, name="ps_sx")
    muls(sx, xr, 2.0)
    t = Reg(DT.float, name="ps_t")
    add(t, sx, yr)
    o_ub[0] <<= t


@kernel(mode="vec")
def axpy(x: GM[f32, (1, 64)], y: GM[f32, (1, 64)], o: GM[f32, (1, 64)]):
    ub_x = Tensor(DT.float, [1, 64], Position.UB, name="ps_ubx")
    ub_y = Tensor(DT.float, [1, 64], Position.UB, name="ps_uby")
    ub_o = Tensor(DT.float, [1, 64], Position.UB, name="ps_ubo")
    with auto_sync():
        ub_x <<= x
        ub_y <<= y
        axpb_vf(ub_x, ub_y, ub_o)
        o <<= ub_o
    return o
