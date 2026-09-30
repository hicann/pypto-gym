# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Preserved gatherb_onboard indexed movement; old board prose is provenance only."""

from ascriptor.a5 import *

@vf()
def gatherb_b8_vf(src_ub: Tensor, dst_ub: Tensor, idx_u: Tensor):
    ui = Reg(DT.uint32, name='gb8_ui')
    ub_to_reg_normal(ui, idx_u)
    d = Reg(DT.int8, name='gb8_d')
    ub_to_reg_gatherb(d, src_ub, ui)
    reg_to_ub_normal(dst_ub, d)

@kernel(mode='vec', block_dim=1)
def gatherb_b8_kernel(src: GM[i8, ('rows', 512)], idx: GM[i32, ('rows', 64)], out: GM[i8, ('rows', 256)], rows: i32):
    src_ub = Tensor(DT.int8, [1, 512], Position.UB, name='gb8_src')
    dst_ub = Tensor(DT.int8, [1, 256], Position.UB, name='gb8_dst')
    idx_s = Tensor(DT.int, [1, 64], Position.UB, name='gb8_idx_s')
    idx_u = reinterpret(idx_s, DT.uint32, name='gb8_idx_u')
    with auto_sync():
        src_ub[:, :] <<= src[0:1, :]
        idx_s[:, :] <<= idx[0:1, :]
        gatherb_b8_vf(src_ub, dst_ub, idx_u)
        out[0:1, :] <<= dst_ub
    return out

@vf()
def gatherb_b16_vf(src_ub: Tensor, dst_ub: Tensor, idx_u: Tensor):
    ui = Reg(DT.uint32, name='gb16_ui')
    ub_to_reg_normal(ui, idx_u)
    d = Reg(DT.half, name='gb16_d')
    ub_to_reg_gatherb(d, src_ub, ui)
    reg_to_ub_normal(dst_ub, d)

@kernel(mode='vec', block_dim=1)
def gatherb_b16_kernel(src: GM[f16, ('rows', 256)], idx: GM[i32, ('rows', 64)], out: GM[f16, ('rows', 128)], rows: i32):
    src_ub = Tensor(DT.half, [1, 256], Position.UB, name='gb16_src')
    dst_ub = Tensor(DT.half, [1, 128], Position.UB, name='gb16_dst')
    idx_s = Tensor(DT.int, [1, 64], Position.UB, name='gb16_idx_s')
    idx_u = reinterpret(idx_s, DT.uint32, name='gb16_idx_u')
    with auto_sync():
        src_ub[:, :] <<= src[0:1, :]
        idx_s[:, :] <<= idx[0:1, :]
        gatherb_b16_vf(src_ub, dst_ub, idx_u)
        out[0:1, :] <<= dst_ub
    return out

@vf()
def gatherb_b32_vf(src_ub: Tensor, dst_ub: Tensor, idx_u: Tensor):
    ui = Reg(DT.uint32, name='gb32_ui')
    ub_to_reg_normal(ui, idx_u)
    d = Reg(DT.int, name='gb32_d')
    ub_to_reg_gatherb(d, src_ub, ui)
    reg_to_ub_normal(dst_ub, d)

@kernel(mode='vec', block_dim=1)
def gatherb_b32_kernel(src: GM[i32, ('rows', 128)], idx: GM[i32, ('rows', 64)], out: GM[i32, ('rows', 64)], rows: i32):
    src_ub = Tensor(DT.int, [1, 128], Position.UB, name='gb32_src')
    dst_ub = Tensor(DT.int, [1, 64], Position.UB, name='gb32_dst')
    idx_s = Tensor(DT.int, [1, 64], Position.UB, name='gb32_idx_s')
    idx_u = reinterpret(idx_s, DT.uint32, name='gb32_idx_u')
    with auto_sync():
        src_ub[:, :] <<= src[0:1, :]
        idx_s[:, :] <<= idx[0:1, :]
        gatherb_b32_vf(src_ub, dst_ub, idx_u)
        out[0:1, :] <<= dst_ub
    return out
