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

# SHA256: 2ae62a0c17554b7eeefab911484e07ede59ad3c6b40a4f69c51f9dfb2a1f9cac

from ascriptor.a5 import *


@vf()
def flag_status_vf(status: Tensor, difference: Var):
    value = Reg(DT.int32)
    value <<= difference
    status[0] <<= value


@kernel(mode="vec", block_dim=1)
def sat_flags(x: GM[f32, (1, 32)], o: GM[f32, (1, 32)], restored: GM[i32, (1, 8)], initial_mode: i32):
    ub = Tensor(DT.float, [1, 32], Position.UB, name="sat_ub")
    status = Tensor(DT.int32, [1, 64], Position.UB)
    launch_cast = get_saturation_flag("cast")
    launch_float = get_saturation_flag("float")
    barrier(Pipe.ALL)
    set_saturation_flag("cast", (initial_mode & 1) != 0)
    set_saturation_flag("float", (initial_mode & 2) != 0)
    saved_cast = get_saturation_flag("cast")
    saved_float = get_saturation_flag("float")
    with auto_sync():
        set_saturation_flag("cast", True)
        a = get_saturation_flag("cast")  # 1
        set_saturation_flag("float", False)
        b = get_saturation_flag("float")  # 0
        ub[:, :] <<= x
        o[0:1, 0:8] <<= ub[0:1, (a * 8) : (a * 8 + 8)]  # x[8:16]
        o[0:1, 8:16] <<= ub[0:1, (16 + b * 8) : (16 + b * 8 + 8)]  # x[16:24]
        o[0:1, 16:32] <<= ub[0:1, 16:32]  # x[16:32]
        barrier(Pipe.ALL)
        set_saturation_flag("cast", saved_cast)
        set_saturation_flag("float", saved_float)
        difference = (get_saturation_flag("cast") ^ saved_cast) | (get_saturation_flag("float") ^ saved_float)
        flag_status_vf(status, difference)
        restored <<= status[:, 0:8]
        barrier(Pipe.ALL)
        set_saturation_flag("cast", launch_cast)
        set_saturation_flag("float", launch_float)
    return o, restored
