# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Preserved gather_scatter_onboard indexed movement; old board prose is provenance only."""

from ascriptor.a5 import *

@vf()
def gather_b8b16_signed_vf(src_ub: Tensor, dst_ub: Tensor, idx_u: Tensor):
    ui = Reg(DT.uint16, name='g8s_ui')
    ub_to_reg_normal(ui, idx_u)
    d = Reg(DT.int16, name='g8s_d')
    ub_to_reg_gather(d, src_ub, ui)
    reg_to_ub_normal(dst_ub, d)

@kernel(mode='vec', block_dim=1)
def gather_b8b16_signed_kernel(src: GM[i8, ('rows', 256)], idx: GM[i16, ('rows', 128)], out: GM[i16, ('rows', 128)], rows: i32):
    src_ub = Tensor(DT.int8, [1, 256], Position.UB, name='g8s_src')
    dst_ub = Tensor(DT.int16, [1, 128], Position.UB, name='g8s_dst')
    idx_s = Tensor(DT.int16, [1, 128], Position.UB, name='g8s_idx_s')
    idx_u = reinterpret(idx_s, DT.uint16, name='g8s_idx_u')
    with auto_sync():
        src_ub[:, :] <<= src[0:1, :]
        idx_s[:, :] <<= idx[0:1, :]
        gather_b8b16_signed_vf(src_ub, dst_ub, idx_u)
        out[0:1, :] <<= dst_ub
    return out

@vf()
def gather_b8b16_unsigned_vf(src_ub: Tensor, dst_ub: Tensor, idx_u: Tensor):
    ui = Reg(DT.uint16, name='g8u_ui')
    ub_to_reg_normal(ui, idx_u)
    d = Reg(DT.uint16, name='g8u_d')
    ub_to_reg_gather(d, src_ub, ui)
    reg_to_ub_normal(dst_ub, d)

@kernel(mode='vec', block_dim=1)
def gather_b8b16_unsigned_kernel(src: GM[u8, ('rows', 256)], idx: GM[i16, ('rows', 128)], out: GM[u16, ('rows', 128)], rows: i32):
    src_ub = Tensor(DT.uint8, [1, 256], Position.UB, name='g8u_src')
    dst_ub = Tensor(DT.uint16, [1, 128], Position.UB, name='g8u_dst')
    idx_s = Tensor(DT.int16, [1, 128], Position.UB, name='g8u_idx_s')
    idx_u = reinterpret(idx_s, DT.uint16, name='g8u_idx_u')
    with auto_sync():
        src_ub[:, :] <<= src[0:1, :]
        idx_s[:, :] <<= idx[0:1, :]
        gather_b8b16_unsigned_vf(src_ub, dst_ub, idx_u)
        out[0:1, :] <<= dst_ub
    return out

@vf()
def gather_b16_vf(src_ub: Tensor, dst_ub: Tensor, idx_u: Tensor):
    ui = Reg(DT.uint16, name='g16_ui')
    ub_to_reg_normal(ui, idx_u)
    d = Reg(DT.half, name='g16_d')
    ub_to_reg_gather(d, src_ub, ui)
    reg_to_ub_normal(dst_ub, d)

@kernel(mode='vec', block_dim=1)
def gather_b16_kernel(src: GM[f16, ('rows', 128)], idx: GM[i16, ('rows', 128)], out: GM[f16, ('rows', 128)], rows: i32):
    src_ub = Tensor(DT.half, [1, 128], Position.UB, name='g16_src')
    dst_ub = Tensor(DT.half, [1, 128], Position.UB, name='g16_dst')
    idx_s = Tensor(DT.int16, [1, 128], Position.UB, name='g16_idx_s')
    idx_u = reinterpret(idx_s, DT.uint16, name='g16_idx_u')
    with auto_sync():
        src_ub[:, :] <<= src[0:1, :]
        idx_s[:, :] <<= idx[0:1, :]
        gather_b16_vf(src_ub, dst_ub, idx_u)
        out[0:1, :] <<= dst_ub
    return out

@vf()
def gather_b32_vf(src_ub: Tensor, dst_ub: Tensor, idx_u: Tensor):
    ui = Reg(DT.uint32, name='g32_ui')
    ub_to_reg_normal(ui, idx_u)
    d = Reg(DT.float, name='g32_d')
    ub_to_reg_gather(d, src_ub, ui)
    reg_to_ub_normal(dst_ub, d)

@kernel(mode='vec', block_dim=1)
def gather_b32_kernel(src: GM[f32, ('rows', 64)], idx: GM[i32, ('rows', 64)], out: GM[f32, ('rows', 64)], rows: i32):
    src_ub = Tensor(DT.float, [1, 64], Position.UB, name='g32_src')
    dst_ub = Tensor(DT.float, [1, 64], Position.UB, name='g32_dst')
    idx_s = Tensor(DT.int, [1, 64], Position.UB, name='g32_idx_s')
    idx_u = reinterpret(idx_s, DT.uint32, name='g32_idx_u')
    with auto_sync():
        src_ub[:, :] <<= src[0:1, :]
        idx_s[:, :] <<= idx[0:1, :]
        gather_b32_vf(src_ub, dst_ub, idx_u)
        out[0:1, :] <<= dst_ub
    return out

@vf()
def gather_b64_u32_vf(src_ub: Tensor, dst_ub: Tensor, idx_u: Tensor):
    ui = Reg(DT.uint32, name='g64a_ui')
    ub_to_reg_normal(ui, idx_u)
    d = Reg(DT.int64, name='g64a_d')
    ub_to_reg_gather(d, src_ub, ui)
    reg_to_ub_normal(dst_ub, d)

@kernel(mode='vec', block_dim=1)
def gather_b64_u32_kernel(src: GM[i64, ('rows', 32)], idx: GM[i32, ('rows', 64)], out: GM[i64, ('rows', 32)], rows: i32):
    src_ub = Tensor(DT.int64, [1, 32], Position.UB, name='g64a_src')
    dst_ub = Tensor(DT.int64, [1, 32], Position.UB, name='g64a_dst')
    idx_s = Tensor(DT.int, [1, 64], Position.UB, name='g64a_idx_s')
    idx_u = reinterpret(idx_s, DT.uint32, name='g64a_idx_u')
    with auto_sync():
        src_ub[:, :] <<= src[0:1, :]
        idx_s[:, :] <<= idx[0:1, :]
        gather_b64_u32_vf(src_ub, dst_ub, idx_u)
        out[0:1, :] <<= dst_ub
    return out

@vf()
def gather_b64_u64_vf(src_ub: Tensor, dst_ub: Tensor, idx_u: Tensor):
    ui = Reg(DT.uint64, name='g64b_ui')
    ub_to_reg_normal(ui, idx_u)
    d = Reg(DT.uint64, name='g64b_d')
    ub_to_reg_gather(d, src_ub, ui)
    reg_to_ub_normal(dst_ub, d)

@kernel(mode='vec', block_dim=1)
def gather_b64_u64_kernel(src: GM[u64, ('rows', 32)], idx: GM[i64, ('rows', 32)], out: GM[u64, ('rows', 32)], rows: i32):
    src_ub = Tensor(DT.uint64, [1, 32], Position.UB, name='g64b_src')
    dst_ub = Tensor(DT.uint64, [1, 32], Position.UB, name='g64b_dst')
    idx_s = Tensor(DT.int64, [1, 32], Position.UB, name='g64b_idx_s')
    idx_u = reinterpret(idx_s, DT.uint64, name='g64b_idx_u')
    with auto_sync():
        src_ub[:, :] <<= src[0:1, :]
        idx_s[:, :] <<= idx[0:1, :]
        gather_b64_u64_vf(src_ub, dst_ub, idx_u)
        out[0:1, :] <<= dst_ub
    return out

@vf()
def scatter_b8_vf(src_ub: Tensor, dst_ub: Tensor, idx_u: Tensor):
    z = Reg(DT.uint8, name='s8_z')
    dup(z, 0)
    reg_to_ub_normal(dst_ub, z)
    vf_barrier(VfPipe.STORE, VfPipe.STORE)
    sr = Reg(DT.uint8, name='s8_sr')
    ub_to_reg_normal(sr, src_ub)
    ui = Reg(DT.uint16, name='s8_ui')
    ub_to_reg_normal(ui, idx_u)
    reg_to_ub_scatter(dst_ub, sr, ui)
    vf_barrier(VfPipe.STORE, VfPipe.STORE)

@kernel(mode='vec', block_dim=1)
def scatter_b8_kernel(src: GM[u8, ('rows', 256)], idx: GM[i16, ('rows', 128)], out: GM[u8, ('rows', 256)], rows: i32):
    src_ub = Tensor(DT.uint8, [1, 256], Position.UB, name='s8_src')
    dst_ub = Tensor(DT.uint8, [1, 256], Position.UB, name='s8_dst')
    idx_s = Tensor(DT.int16, [1, 128], Position.UB, name='s8_idx_s')
    idx_u = reinterpret(idx_s, DT.uint16, name='s8_idx_u')
    with auto_sync():
        src_ub[:, :] <<= src[0:1, :]
        idx_s[:, :] <<= idx[0:1, :]
        scatter_b8_vf(src_ub, dst_ub, idx_u)
        out[0:1, :] <<= dst_ub
    return out

@vf()
def scatter_b16_vf(src_ub: Tensor, dst_ub: Tensor, idx_u: Tensor):
    z = Reg(DT.half, name='s16_z')
    dup(z, 0.0)
    reg_to_ub_normal(dst_ub, z)
    vf_barrier(VfPipe.STORE, VfPipe.STORE)
    sr = Reg(DT.half, name='s16_sr')
    ub_to_reg_normal(sr, src_ub)
    ui = Reg(DT.uint16, name='s16_ui')
    ub_to_reg_normal(ui, idx_u)
    reg_to_ub_scatter(dst_ub, sr, ui)
    vf_barrier(VfPipe.STORE, VfPipe.STORE)

@kernel(mode='vec', block_dim=1)
def scatter_b16_kernel(src: GM[f16, ('rows', 128)], idx: GM[i16, ('rows', 128)], out: GM[f16, ('rows', 128)], rows: i32):
    src_ub = Tensor(DT.half, [1, 128], Position.UB, name='s16_src')
    dst_ub = Tensor(DT.half, [1, 128], Position.UB, name='s16_dst')
    idx_s = Tensor(DT.int16, [1, 128], Position.UB, name='s16_idx_s')
    idx_u = reinterpret(idx_s, DT.uint16, name='s16_idx_u')
    with auto_sync():
        src_ub[:, :] <<= src[0:1, :]
        idx_s[:, :] <<= idx[0:1, :]
        scatter_b16_vf(src_ub, dst_ub, idx_u)
        out[0:1, :] <<= dst_ub
    return out

@vf()
def scatter_b32_vf(src_ub: Tensor, dst_ub: Tensor, idx_u: Tensor):
    z = Reg(DT.float, name='s32_z')
    dup(z, 0.0)
    reg_to_ub_normal(dst_ub, z)
    vf_barrier(VfPipe.STORE, VfPipe.STORE)
    sr = Reg(DT.float, name='s32_sr')
    ub_to_reg_normal(sr, src_ub)
    ui = Reg(DT.uint32, name='s32_ui')
    ub_to_reg_normal(ui, idx_u)
    reg_to_ub_scatter(dst_ub, sr, ui)
    vf_barrier(VfPipe.STORE, VfPipe.STORE)

@kernel(mode='vec', block_dim=1)
def scatter_b32_kernel(src: GM[f32, ('rows', 64)], idx: GM[i32, ('rows', 64)], out: GM[f32, ('rows', 64)], rows: i32):
    src_ub = Tensor(DT.float, [1, 64], Position.UB, name='s32_src')
    dst_ub = Tensor(DT.float, [1, 64], Position.UB, name='s32_dst')
    idx_s = Tensor(DT.int, [1, 64], Position.UB, name='s32_idx_s')
    idx_u = reinterpret(idx_s, DT.uint32, name='s32_idx_u')
    with auto_sync():
        src_ub[:, :] <<= src[0:1, :]
        idx_s[:, :] <<= idx[0:1, :]
        scatter_b32_vf(src_ub, dst_ub, idx_u)
        out[0:1, :] <<= dst_ub
    return out

@vf()
def scatter_b64_u32_vf(src_ub: Tensor, dst_ub: Tensor, idx_u: Tensor):
    z = Reg(DT.int64, name='s64a_z')
    dup(z, 0)
    reg_to_ub_normal(dst_ub, z)
    vf_barrier(VfPipe.STORE, VfPipe.STORE)
    sr = Reg(DT.int64, name='s64a_sr')
    ub_to_reg_normal(sr, src_ub)
    ui = Reg(DT.uint32, name='s64a_ui')
    ub_to_reg_normal(ui, idx_u)
    reg_to_ub_scatter(dst_ub, sr, ui)
    vf_barrier(VfPipe.STORE, VfPipe.STORE)

@kernel(mode='vec', block_dim=1)
def scatter_b64_u32_kernel(src: GM[i64, ('rows', 32)], idx: GM[i32, ('rows', 64)], out: GM[i64, ('rows', 32)], rows: i32):
    src_ub = Tensor(DT.int64, [1, 32], Position.UB, name='s64a_src')
    dst_ub = Tensor(DT.int64, [1, 32], Position.UB, name='s64a_dst')
    idx_s = Tensor(DT.int, [1, 64], Position.UB, name='s64a_idx_s')
    idx_u = reinterpret(idx_s, DT.uint32, name='s64a_idx_u')
    with auto_sync():
        src_ub[:, :] <<= src[0:1, :]
        idx_s[:, :] <<= idx[0:1, :]
        scatter_b64_u32_vf(src_ub, dst_ub, idx_u)
        out[0:1, :] <<= dst_ub
    return out

@vf()
def scatter_b64_u64_vf(src_ub: Tensor, dst_ub: Tensor, idx_u: Tensor):
    z = Reg(DT.uint64, name='s64b_z')
    dup(z, 0)
    reg_to_ub_normal(dst_ub, z)
    vf_barrier(VfPipe.STORE, VfPipe.STORE)
    sr = Reg(DT.uint64, name='s64b_sr')
    ub_to_reg_normal(sr, src_ub)
    ui = Reg(DT.uint64, name='s64b_ui')
    ub_to_reg_normal(ui, idx_u)
    reg_to_ub_scatter(dst_ub, sr, ui)
    vf_barrier(VfPipe.STORE, VfPipe.STORE)

@kernel(mode='vec', block_dim=1)
def scatter_b64_u64_kernel(src: GM[u64, ('rows', 32)], idx: GM[i64, ('rows', 32)], out: GM[u64, ('rows', 32)], rows: i32):
    src_ub = Tensor(DT.uint64, [1, 32], Position.UB, name='s64b_src')
    dst_ub = Tensor(DT.uint64, [1, 32], Position.UB, name='s64b_dst')
    idx_s = Tensor(DT.int64, [1, 32], Position.UB, name='s64b_idx_s')
    idx_u = reinterpret(idx_s, DT.uint64, name='s64b_idx_u')
    with auto_sync():
        src_ub[:, :] <<= src[0:1, :]
        idx_s[:, :] <<= idx[0:1, :]
        scatter_b64_u64_vf(src_ub, dst_ub, idx_u)
        out[0:1, :] <<= dst_ub
    return out
