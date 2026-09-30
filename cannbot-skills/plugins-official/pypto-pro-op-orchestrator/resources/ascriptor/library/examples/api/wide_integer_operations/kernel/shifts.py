# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Reviewed shift_reg_onboard device bodies; host cases and references are local and independent."""

from ascriptor.a5 import *

@vf()
def shiftl_i32_vf(data_ub: Tensor, shift_ub: Tensor, dst_ub: Tensor):
    d = Reg(DT.int, name='sl32_d')
    ub_to_reg_normal(d, data_ub)
    s = Reg(DT.int, name='sl32_s')
    ub_to_reg_normal(s, shift_ub)
    y = Reg(DT.int, name='sl32_y')
    shiftl(y, d, s)
    reg_to_ub_normal(dst_ub, y)

@kernel(mode='vec', block_dim=1)
def shiftl_i32_kernel(data: GM[i32, ('rows', 64)], shift: GM[i32, ('rows', 64)], out: GM[i32, ('rows', 64)], rows: i32):
    data_ub = Tensor(DT.int, [1, 64], Position.UB, name='sl32_data')
    shift_ub = Tensor(DT.int, [1, 64], Position.UB, name='sl32_shift')
    dst_ub = Tensor(DT.int, [1, 64], Position.UB, name='sl32_dst')
    with auto_sync():
        data_ub[:, :] <<= data[0:1, :]
        shift_ub[:, :] <<= shift[0:1, :]
        shiftl_i32_vf(data_ub, shift_ub, dst_ub)
        out[0:1, :] <<= dst_ub
    return out

@vf()
def shiftr_i32_vf(data_ub: Tensor, shift_ub: Tensor, dst_ub: Tensor):
    d = Reg(DT.int, name='sr32_d')
    ub_to_reg_normal(d, data_ub)
    s = Reg(DT.int, name='sr32_s')
    ub_to_reg_normal(s, shift_ub)
    y = Reg(DT.int, name='sr32_y')
    shiftr(y, d, s)
    reg_to_ub_normal(dst_ub, y)

@kernel(mode='vec', block_dim=1)
def shiftr_i32_kernel(data: GM[i32, ('rows', 64)], shift: GM[i32, ('rows', 64)], out: GM[i32, ('rows', 64)], rows: i32):
    data_ub = Tensor(DT.int, [1, 64], Position.UB, name='sr32_data')
    shift_ub = Tensor(DT.int, [1, 64], Position.UB, name='sr32_shift')
    dst_ub = Tensor(DT.int, [1, 64], Position.UB, name='sr32_dst')
    with auto_sync():
        data_ub[:, :] <<= data[0:1, :]
        shift_ub[:, :] <<= shift[0:1, :]
        shiftr_i32_vf(data_ub, shift_ub, dst_ub)
        out[0:1, :] <<= dst_ub
    return out

@vf()
def shiftl_i64_vf(data_ub: Tensor, shift_ub: Tensor, dst_ub: Tensor):
    d = Reg(DT.int64, name='sl64_d')
    ub_to_reg_normal(d, data_ub)
    s = Reg(DT.int64, name='sl64_s')
    ub_to_reg_normal(s, shift_ub)
    y = Reg(DT.int64, name='sl64_y')
    shiftl(y, d, s)
    reg_to_ub_normal(dst_ub, y)

@kernel(mode='vec', block_dim=1)
def shiftl_i64_kernel(data: GM[i64, ('rows', 32)], shift: GM[i64, ('rows', 32)], out: GM[i64, ('rows', 32)], rows: i32):
    data_ub = Tensor(DT.int64, [1, 32], Position.UB, name='sl64_data')
    shift_ub = Tensor(DT.int64, [1, 32], Position.UB, name='sl64_shift')
    dst_ub = Tensor(DT.int64, [1, 32], Position.UB, name='sl64_dst')
    with auto_sync():
        data_ub[:, :] <<= data[0:1, :]
        shift_ub[:, :] <<= shift[0:1, :]
        shiftl_i64_vf(data_ub, shift_ub, dst_ub)
        out[0:1, :] <<= dst_ub
    return out

@vf()
def shiftr_i64_vf(data_ub: Tensor, shift_ub: Tensor, dst_ub: Tensor):
    d = Reg(DT.int64, name='sr64_d')
    ub_to_reg_normal(d, data_ub)
    s = Reg(DT.int64, name='sr64_s')
    ub_to_reg_normal(s, shift_ub)
    y = Reg(DT.int64, name='sr64_y')
    shiftr(y, d, s)
    reg_to_ub_normal(dst_ub, y)

@kernel(mode='vec', block_dim=1)
def shiftr_i64_kernel(data: GM[i64, ('rows', 32)], shift: GM[i64, ('rows', 32)], out: GM[i64, ('rows', 32)], rows: i32):
    data_ub = Tensor(DT.int64, [1, 32], Position.UB, name='sr64_data')
    shift_ub = Tensor(DT.int64, [1, 32], Position.UB, name='sr64_shift')
    dst_ub = Tensor(DT.int64, [1, 32], Position.UB, name='sr64_dst')
    with auto_sync():
        data_ub[:, :] <<= data[0:1, :]
        shift_ub[:, :] <<= shift[0:1, :]
        shiftr_i64_vf(data_ub, shift_ub, dst_ub)
        out[0:1, :] <<= dst_ub
    return out
