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


@vf()
def cast_saturation_vf(src: Tensor, srcf: Tensor, dst: Tensor):
    a = Reg(DT.int32)
    f = Reg(DT.float)
    a <<= src[0]
    f <<= srcf[0]
    yes = Reg(DT.int16)
    no = Reg(DT.int16)
    fyes = Reg(DT.int16)
    fno = Reg(DT.int16)
    sat = CastConfig(saturate=True, round_mode=RoundMode.TO_EVEN, reg_layout=RegLayout.ZERO)
    nosat = CastConfig(saturate=False, round_mode=RoundMode.TO_EVEN, reg_layout=RegLayout.ZERO)
    cast(yes, a, sat)
    cast(no, a, nosat)
    cast(fyes, f, sat)
    cast(fno, f, nosat)
    dst[0] <<= yes
    dst[128] <<= no
    dst[256] <<= fyes
    dst[384] <<= fno


@vf()
def restored_flags_vf(dst: Tensor, difference: Var):
    status = Reg(DT.int16)
    status <<= difference
    dst[0] <<= status
    dst[128] <<= status
    dst[256] <<= status
    dst[384] <<= status


@kernel(mode="vec", block_dim=1)
def cast_saturation(x: GM[i32, (1, 64)], xf: GM[f32, (1, 64)], y: GM[i16, (5, 512)], initial_mode: i32):
    ub_x = Tensor(DT.int32, [1, 64], Position.UB)
    ub_f = Tensor(DT.float, [1, 64], Position.UB)
    ub_y = Tensor(DT.int16, [1, 512], Position.UB)
    if GetVecIdx() == 0:
        launch_global = get_saturation_flag("global")
        launch_cast = get_saturation_flag("cast")
        barrier(Pipe.ALL)
        set_saturation_flag("global", (initial_mode & 2) != 0)
        set_saturation_flag("cast", (initial_mode & 1) != 0)
        saved_global = get_saturation_flag("global")
        saved_cast = get_saturation_flag("cast")
        with auto_sync():
            ub_x <<= x
            ub_f <<= xf
            for mode in unroll(4):
                barrier(Pipe.ALL)
                set_saturation_flag("global", bool(mode & 2))
                set_saturation_flag("cast", bool(mode & 1))
                cast_saturation_vf(ub_x, ub_f, ub_y)
                y[mode : mode + 1, :] <<= ub_y
            barrier(Pipe.ALL)
            set_saturation_flag("cast", saved_cast)
            set_saturation_flag("global", saved_global)
            difference = (get_saturation_flag("cast") ^ saved_cast) | (
                get_saturation_flag("global") ^ saved_global
            )
            restored_flags_vf(ub_y, difference)
            y[4:5, :] <<= ub_y
            barrier(Pipe.ALL)
            set_saturation_flag("cast", launch_cast)
            set_saturation_flag("global", launch_global)
    return y
