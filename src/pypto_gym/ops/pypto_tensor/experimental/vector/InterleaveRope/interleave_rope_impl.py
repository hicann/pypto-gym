#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
interleave_rope PyPTO implementation 

数学计算 (interleave 模式 RoPE)：
  y_origin[2k]   = x[2k] · cos[2k]   - x[2k+1] · sin[2k]
  y_origin[2k+1] = x[2k] · sin[2k+1] + x[2k+1] · cos[2k+1]

输出 layout (split-half, NOT interleaved):
  out[..., 0:32 ] = y_even = [y_origin[0], y_origin[2], ..., y_origin[62]]
  out[..., 32:64] = y_odd  = [y_origin[1], y_origin[3], ..., y_origin[63]]


Wrapper 做：输入校验 / 输出张量分配 / 按 (N, dtype, S_cs) 派发。
Kernel 全程 4D，无 5D reshape/concat：
  1. ceil-div: s_loops = (S + S_TILE - 1) // S_TILE
  2. valid_s = (S - s_off).min(S_TILE)
  3. view(x|cos|sin, ..., valid_shape=[..., valid_s, ...])
  4. 910 使用 gathermask；950(DAV_3510) 使用 deinterleave → cast fp32 → mul/sub/add → cast 回 → assemble 写左/右半

4 个 kernel 实例：{N=1, N=128} × {bf16, fp16}。
"""

import pypto
import torch

N_TILE_128 = 32
B_TILE_128_SHORT = 4
S_TILE_128 = 16
S_TILE_128_SHORT = 2
S_UNROLL_128 = [4, 2, 1]
S_TILE_1 = 64
S_TILE_128_950 = 12
D = 64
HALF = 32  
ASCEND_950_NPUARCH = "DAV_3510"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
@pypto.frontend.jit(
    runtime_options={
        "run_mode": pypto.RunMode.NPU,
        "stitch_function_max_num": 512,
        "device_sched_mode": 3,
    },
    pass_options={"vec_nbuffer_setting": {-1: 8}},
)
def interleave_rope_kernel_n128_bf16(
    x: pypto.Tensor([pypto.DYNAMIC, 128, pypto.DYNAMIC, 64], pypto.DT_BF16),
    cos: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_BF16),
    sin: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_BF16),
    out: pypto.Tensor([pypto.DYNAMIC, 128, pypto.DYNAMIC, 64], pypto.DT_BF16),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_vec_tile_shapes(1, N_TILE_128, S_TILE_128, D)
    B = x.shape[0]
    S = x.shape[2]
    s_loops = (S + S_TILE_128 - 1) // S_TILE_128
    for b in pypto.loop(B, name="b_loop"):
        for n_blk in pypto.loop(128 // N_TILE_128, name="n_loop"):
            n_off = n_blk * N_TILE_128
            for s_blk in pypto.loop(s_loops, name="s_loop"):
                s_off = s_blk * S_TILE_128
                valid_s = (S - s_off).min(S_TILE_128)
                vshape_x = [1, N_TILE_128, valid_s, D]
                vshape_cs = [1, 1, valid_s, D]
                x_t = pypto.view(x, [1, N_TILE_128, S_TILE_128, D], [b, n_off, s_off, 0],
                                 valid_shape=vshape_x)
                c_t = pypto.view(cos, [1, 1, S_TILE_128, D], [b, 0, s_off, 0],
                                 valid_shape=vshape_cs)
                s_t = pypto.view(sin, [1, 1, S_TILE_128, D], [b, 0, s_off, 0],
                                 valid_shape=vshape_cs)
                pypto.set_pass_options(sg_set_scope=1)
                x_e = pypto.gathermask(x_t, pattern_mode=1)
                x_o = pypto.gathermask(x_t, pattern_mode=2)
                c_e = pypto.gathermask(c_t, pattern_mode=1)
                c_o = pypto.gathermask(c_t, pattern_mode=2)
                s_e = pypto.gathermask(s_t, pattern_mode=1)
                s_o = pypto.gathermask(s_t, pattern_mode=2)
                pypto.set_vec_tile_shapes(1, N_TILE_128, S_TILE_128, HALF)
                ye = pypto.sub(pypto.mul(x_e, c_e), pypto.mul(x_o, s_e))
                yo = pypto.add(pypto.mul(x_e, s_o), pypto.mul(x_o, c_o))
                pypto.assemble(ye, [b, n_off, s_off, 0], out)
                pypto.assemble(yo, [b, n_off, s_off, HALF], out)
                pypto.set_pass_options(sg_set_scope=-1)
                pypto.set_vec_tile_shapes(1, N_TILE_128, S_TILE_128, D)


@pypto.frontend.jit(
    runtime_options={
        "run_mode": pypto.RunMode.NPU,
        "stitch_function_max_num": 512,
        "device_sched_mode": 3,
    },
)
def interleave_rope_kernel_n128_bf16_unroll(
    x: pypto.Tensor([pypto.DYNAMIC, 128, pypto.DYNAMIC, 64], pypto.DT_BF16),
    cos: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_BF16),
    sin: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_BF16),
    out: pypto.Tensor([pypto.DYNAMIC, 128, pypto.DYNAMIC, 64], pypto.DT_BF16),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_vec_tile_shapes(1, N_TILE_128, S_TILE_128, D)
    B = x.shape[0]
    S = x.shape[2]
    s_loops = (S + S_TILE_128 - 1) // S_TILE_128
    for b in pypto.loop(B, name="b_loop"):
        for n_blk in pypto.loop(128 // N_TILE_128, name="n_loop"):
            n_off = n_blk * N_TILE_128
            for s_blk in pypto.loop(s_loops, name="s_loop", unroll_list=S_UNROLL_128):
                s_off = s_blk * S_TILE_128
                valid_s = (S - s_off).min(S_TILE_128)
                vshape_x = [1, N_TILE_128, valid_s, D]
                vshape_cs = [1, 1, valid_s, D]
                x_t = pypto.view(x, [1, N_TILE_128, S_TILE_128, D], [b, n_off, s_off, 0],
                                 valid_shape=vshape_x)
                c_t = pypto.view(cos, [1, 1, S_TILE_128, D], [b, 0, s_off, 0],
                                 valid_shape=vshape_cs)
                s_t = pypto.view(sin, [1, 1, S_TILE_128, D], [b, 0, s_off, 0],
                                 valid_shape=vshape_cs)
                x_e = pypto.gathermask(x_t, pattern_mode=1)
                x_o = pypto.gathermask(x_t, pattern_mode=2)
                c_e = pypto.gathermask(c_t, pattern_mode=1)
                c_o = pypto.gathermask(c_t, pattern_mode=2)
                s_e = pypto.gathermask(s_t, pattern_mode=1)
                s_o = pypto.gathermask(s_t, pattern_mode=2)
                pypto.set_vec_tile_shapes(1, N_TILE_128, S_TILE_128, HALF)
                ye = pypto.sub(pypto.mul(x_e, c_e), pypto.mul(x_o, s_e))
                yo = pypto.add(pypto.mul(x_e, s_o), pypto.mul(x_o, c_o))
                pypto.assemble(ye, [b, n_off, s_off, 0], out)
                pypto.assemble(yo, [b, n_off, s_off, HALF], out)
                pypto.set_vec_tile_shapes(1, N_TILE_128, S_TILE_128, D)


@pypto.frontend.jit(
    runtime_options={
        "run_mode": pypto.RunMode.NPU,
        "stitch_function_max_num": 512,
        "device_sched_mode": 3,
    },
    pass_options={"vec_nbuffer_setting": {-1: 8}},
)
def interleave_rope_kernel_n128_bf16_950(
    x:   pypto.Tensor([pypto.DYNAMIC, 128, pypto.DYNAMIC, 64], pypto.DT_BF16),
    cos: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_BF16),
    sin: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_BF16),
    out: pypto.Tensor([pypto.DYNAMIC, 128, pypto.DYNAMIC, 64], pypto.DT_BF16),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_vec_tile_shapes(1, N_TILE_128, S_TILE_128_950, D)
    B = x.shape[0]
    S = x.shape[2]
    s_loops = (S + S_TILE_128_950 - 1) // S_TILE_128_950
    for b in pypto.loop(B, name="b_loop"):
        for n_blk in pypto.loop(128 // N_TILE_128, name="n_loop"):
            n_off = n_blk * N_TILE_128
            for s_blk in pypto.loop(s_loops, name="s_loop"):
                s_off = s_blk * S_TILE_128_950
                valid_s = (S - s_off).min(S_TILE_128_950)
                vshape_x = [1, N_TILE_128, valid_s, D]
                vshape_cs = [1, 1, valid_s, D]
                x_t = pypto.view(x, [1, N_TILE_128, S_TILE_128_950, D], [b, n_off, s_off, 0],
                                 valid_shape=vshape_x)
                c_t = pypto.view(cos, [1, 1, S_TILE_128_950, D], [b, 0, s_off, 0],
                                 valid_shape=vshape_cs)
                s_t = pypto.view(sin, [1, 1, S_TILE_128_950, D], [b, 0, s_off, 0],
                                 valid_shape=vshape_cs)
                pypto.set_pass_options(sg_set_scope=1)
                x_e, x_o = pypto.deinterleave(x_t)
                c_e, c_o = pypto.deinterleave(c_t)
                s_e, s_o = pypto.deinterleave(s_t)
                pypto.set_vec_tile_shapes(1, N_TILE_128, S_TILE_128_950, HALF)
                ye = pypto.sub(pypto.mul(x_e, c_e), pypto.mul(x_o, s_e))
                yo = pypto.add(pypto.mul(x_e, s_o), pypto.mul(x_o, c_o))
                pypto.assemble(ye, [b, n_off, s_off, 0], out)
                pypto.assemble(yo, [b, n_off, s_off, HALF], out)
                pypto.set_pass_options(sg_set_scope=-1)
                pypto.set_vec_tile_shapes(1, N_TILE_128, S_TILE_128_950, D)


@pypto.frontend.jit(
    runtime_options={
        "run_mode": pypto.RunMode.NPU,
        "stitch_function_max_num": 512,
        "device_sched_mode": 3,
    },
)
def interleave_rope_kernel_n128_bf16_broadcast(
    x: pypto.Tensor([pypto.DYNAMIC, 128, pypto.DYNAMIC, 64], pypto.DT_BF16),
    cos: pypto.Tensor([pypto.DYNAMIC, 1, 1, 64], pypto.DT_BF16),
    sin: pypto.Tensor([pypto.DYNAMIC, 1, 1, 64], pypto.DT_BF16),
    out: pypto.Tensor([pypto.DYNAMIC, 128, pypto.DYNAMIC, 64], pypto.DT_BF16),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_vec_tile_shapes(1, N_TILE_128, S_TILE_128, D)
    B = x.shape[0]
    S = x.shape[2]
    s_loops = (S + S_TILE_128 - 1) // S_TILE_128
    for b in pypto.loop(B, name="b_loop"):
        c_t = pypto.view(cos, [1, 1, 1, D], [b, 0, 0, 0])
        s_t = pypto.view(sin, [1, 1, 1, D], [b, 0, 0, 0])
        for n_blk in pypto.loop(128 // N_TILE_128, name="n_loop"):
            n_off = n_blk * N_TILE_128
            for s_blk in pypto.loop(s_loops, name="s_loop"):
                s_off = s_blk * S_TILE_128
                valid_s = (S - s_off).min(S_TILE_128)
                x_t = pypto.view(
                    x, [1, N_TILE_128, S_TILE_128, D], [b, n_off, s_off, 0],
                    valid_shape=[1, N_TILE_128, valid_s, D])
                pypto.set_pass_options(sg_set_scope=1)
                x_e = pypto.gathermask(x_t, pattern_mode=1)
                x_o = pypto.gathermask(x_t, pattern_mode=2)
                c_e = pypto.gathermask(c_t, pattern_mode=1)
                c_o = pypto.gathermask(c_t, pattern_mode=2)
                s_e = pypto.gathermask(s_t, pattern_mode=1)
                s_o = pypto.gathermask(s_t, pattern_mode=2)
                pypto.set_vec_tile_shapes(1, N_TILE_128, S_TILE_128, HALF)
                ye = pypto.sub(pypto.mul(x_e, c_e), pypto.mul(x_o, s_e))
                yo = pypto.add(pypto.mul(x_e, s_o), pypto.mul(x_o, c_o))
                pypto.assemble(ye, [b, n_off, s_off, 0], out)
                pypto.assemble(yo, [b, n_off, s_off, HALF], out)
                pypto.set_pass_options(sg_set_scope=-1)
                pypto.set_vec_tile_shapes(1, N_TILE_128, S_TILE_128, D)


@pypto.frontend.jit(
    runtime_options={
        "run_mode": pypto.RunMode.NPU,
        "stitch_function_max_num": 512,
        "device_sched_mode": 3,
    },
)
def interleave_rope_kernel_n128_bf16_short_s(
    x: pypto.Tensor([pypto.DYNAMIC, 128, pypto.DYNAMIC, 64], pypto.DT_BF16),
    cos: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_BF16),
    sin: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_BF16),
    out: pypto.Tensor([pypto.DYNAMIC, 128, pypto.DYNAMIC, 64], pypto.DT_BF16),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_vec_tile_shapes(1, N_TILE_128, S_TILE_128_SHORT, D)
    B = x.shape[0]
    S = x.shape[2]
    s_loops = (S + S_TILE_128_SHORT - 1) // S_TILE_128_SHORT
    for b in pypto.loop(B, name="b_loop"):
        for n_blk in pypto.loop(128 // N_TILE_128, name="n_loop"):
            n_off = n_blk * N_TILE_128
            for s_blk in pypto.loop(s_loops, name="s_loop"):
                s_off = s_blk * S_TILE_128_SHORT
                valid_s = (S - s_off).min(S_TILE_128_SHORT)
                vshape_x = [1, N_TILE_128, valid_s, D]
                vshape_cs = [1, 1, valid_s, D]
                x_t = pypto.view(x, [1, N_TILE_128, S_TILE_128_SHORT, D], [b, n_off, s_off, 0],
                                 valid_shape=vshape_x)
                c_t = pypto.view(cos, [1, 1, S_TILE_128_SHORT, D], [b, 0, s_off, 0],
                                 valid_shape=vshape_cs)
                s_t = pypto.view(sin, [1, 1, S_TILE_128_SHORT, D], [b, 0, s_off, 0],
                                 valid_shape=vshape_cs)
                x_e = pypto.gathermask(x_t, pattern_mode=1)
                x_o = pypto.gathermask(x_t, pattern_mode=2)
                c_e = pypto.gathermask(c_t, pattern_mode=1)
                c_o = pypto.gathermask(c_t, pattern_mode=2)
                s_e = pypto.gathermask(s_t, pattern_mode=1)
                s_o = pypto.gathermask(s_t, pattern_mode=2)
                pypto.set_vec_tile_shapes(1, N_TILE_128, S_TILE_128_SHORT, HALF)
                ye = pypto.sub(pypto.mul(x_e, c_e), pypto.mul(x_o, s_e))
                yo = pypto.add(pypto.mul(x_e, s_o), pypto.mul(x_o, c_o))
                pypto.assemble(ye, [b, n_off, s_off, 0], out)
                pypto.assemble(yo, [b, n_off, s_off, HALF], out)
                pypto.set_vec_tile_shapes(1, N_TILE_128, S_TILE_128_SHORT, D)


@pypto.frontend.jit(
    runtime_options={
        "run_mode": pypto.RunMode.NPU,
        "stitch_function_max_num": 512,
        "device_sched_mode": 3,
    },
)
def interleave_rope_kernel_n128_bf16_short_s_btile(
    x: pypto.Tensor([pypto.DYNAMIC, 128, pypto.DYNAMIC, 64], pypto.DT_BF16),
    cos: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_BF16),
    sin: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_BF16),
    out: pypto.Tensor([pypto.DYNAMIC, 128, pypto.DYNAMIC, 64], pypto.DT_BF16),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_vec_tile_shapes(B_TILE_128_SHORT, N_TILE_128, S_TILE_128_SHORT, D)
    B = x.shape[0]
    S = x.shape[2]
    b_loops = (B + B_TILE_128_SHORT - 1) // B_TILE_128_SHORT
    s_loops = (S + S_TILE_128_SHORT - 1) // S_TILE_128_SHORT
    for b_blk in pypto.loop(b_loops, name="b_loop"):
        b_off = b_blk * B_TILE_128_SHORT
        valid_b = (B - b_off).min(B_TILE_128_SHORT)
        for n_blk in pypto.loop(128 // N_TILE_128, name="n_loop"):
            n_off = n_blk * N_TILE_128
            for s_blk in pypto.loop(s_loops, name="s_loop"):
                s_off = s_blk * S_TILE_128_SHORT
                valid_s = (S - s_off).min(S_TILE_128_SHORT)
                vshape_x = [valid_b, N_TILE_128, valid_s, D]
                vshape_cs = [valid_b, 1, valid_s, D]
                x_t = pypto.view(x, [B_TILE_128_SHORT, N_TILE_128, S_TILE_128_SHORT, D],
                                 [b_off, n_off, s_off, 0],
                                 valid_shape=vshape_x)
                c_t = pypto.view(cos, [B_TILE_128_SHORT, 1, S_TILE_128_SHORT, D],
                                 [b_off, 0, s_off, 0],
                                 valid_shape=vshape_cs)
                s_t = pypto.view(sin, [B_TILE_128_SHORT, 1, S_TILE_128_SHORT, D],
                                 [b_off, 0, s_off, 0],
                                 valid_shape=vshape_cs)
                x_e = pypto.gathermask(x_t, pattern_mode=1)
                x_o = pypto.gathermask(x_t, pattern_mode=2)
                c_e = pypto.gathermask(c_t, pattern_mode=1)
                c_o = pypto.gathermask(c_t, pattern_mode=2)
                s_e = pypto.gathermask(s_t, pattern_mode=1)
                s_o = pypto.gathermask(s_t, pattern_mode=2)
                pypto.set_vec_tile_shapes(B_TILE_128_SHORT, N_TILE_128, S_TILE_128_SHORT, HALF)
                ye = pypto.sub(pypto.mul(x_e, c_e), pypto.mul(x_o, s_e))
                yo = pypto.add(pypto.mul(x_e, s_o), pypto.mul(x_o, c_o))
                pypto.assemble(ye, [b_off, n_off, s_off, 0], out)
                pypto.assemble(yo, [b_off, n_off, s_off, HALF], out)
                pypto.set_vec_tile_shapes(B_TILE_128_SHORT, N_TILE_128, S_TILE_128_SHORT, D)


@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
)
def interleave_rope_kernel_n128_fp16(
    x: pypto.Tensor([pypto.DYNAMIC, 128, pypto.DYNAMIC, 64], pypto.DT_FP16),
    cos: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_FP16),
    sin: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_FP16),
    out: pypto.Tensor([pypto.DYNAMIC, 128, pypto.DYNAMIC, 64], pypto.DT_FP16),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_vec_tile_shapes(1, N_TILE_128, S_TILE_128, D)
    B = x.shape[0]
    S = x.shape[2]
    s_loops = (S + S_TILE_128 - 1) // S_TILE_128
    for b in pypto.loop(B, name="b_loop"):
        for n_blk in pypto.loop(128 // N_TILE_128, name="n_loop"):
            n_off = n_blk * N_TILE_128
            for s_blk in pypto.loop(s_loops, name="s_loop"):
                s_off = s_blk * S_TILE_128
                valid_s = (S - s_off).min(S_TILE_128)
                vshape_x = [1, N_TILE_128, valid_s, D]
                vshape_cs = [1, 1, valid_s, D]
                x_t = pypto.view(x, [1, N_TILE_128, S_TILE_128, D], [b, n_off, s_off, 0],
                                 valid_shape=vshape_x)
                c_t = pypto.view(cos, [1, 1, S_TILE_128, D], [b, 0, s_off, 0],
                                 valid_shape=vshape_cs)
                s_t = pypto.view(sin, [1, 1, S_TILE_128, D], [b, 0, s_off, 0],
                                 valid_shape=vshape_cs)
                x_e = pypto.gathermask(x_t, pattern_mode=1)
                x_o = pypto.gathermask(x_t, pattern_mode=2)
                c_e = pypto.gathermask(c_t, pattern_mode=1)
                c_o = pypto.gathermask(c_t, pattern_mode=2)
                s_e = pypto.gathermask(s_t, pattern_mode=1)
                s_o = pypto.gathermask(s_t, pattern_mode=2)
                pypto.set_vec_tile_shapes(1, N_TILE_128, S_TILE_128, HALF)
                xe_f = pypto.cast(x_e, pypto.DT_FP32)
                xo_f = pypto.cast(x_o, pypto.DT_FP32)
                ce_f = pypto.cast(c_e, pypto.DT_FP32)
                co_f = pypto.cast(c_o, pypto.DT_FP32)
                se_f = pypto.cast(s_e, pypto.DT_FP32)
                so_f = pypto.cast(s_o, pypto.DT_FP32)
                ye_f = pypto.sub(pypto.mul(xe_f, ce_f), pypto.mul(xo_f, se_f))
                yo_f = pypto.add(pypto.mul(xe_f, so_f), pypto.mul(xo_f, co_f))
                ye = pypto.cast(ye_f, pypto.DT_FP16)
                yo = pypto.cast(yo_f, pypto.DT_FP16)
                pypto.assemble(ye, [b, n_off, s_off, 0], out)
                pypto.assemble(yo, [b, n_off, s_off, HALF], out)
                pypto.set_vec_tile_shapes(1, N_TILE_128, S_TILE_128, D)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
)
def interleave_rope_kernel_n1_bf16(
    x: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_BF16),
    cos: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_BF16),
    sin: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_BF16),
    out: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_BF16),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_vec_tile_shapes(1, 1, S_TILE_1, D)
    B = x.shape[0]
    S = x.shape[2]
    s_loops = (S + S_TILE_1 - 1) // S_TILE_1
    for b in pypto.loop(B, name="b_loop"):
        for s_blk in pypto.loop(s_loops, name="s_loop"):
            s_off = s_blk * S_TILE_1
            valid_s = (S - s_off).min(S_TILE_1)
            vshape = [1, 1, valid_s, D]
            x_t = pypto.view(x, [1, 1, S_TILE_1, D], [b, 0, s_off, 0], valid_shape=vshape)
            c_t = pypto.view(cos, [1, 1, S_TILE_1, D], [b, 0, s_off, 0], valid_shape=vshape)
            s_t = pypto.view(sin, [1, 1, S_TILE_1, D], [b, 0, s_off, 0], valid_shape=vshape)
            x_e = pypto.gathermask(x_t, pattern_mode=1)
            x_o = pypto.gathermask(x_t, pattern_mode=2)
            c_e = pypto.gathermask(c_t, pattern_mode=1)
            c_o = pypto.gathermask(c_t, pattern_mode=2)
            s_e = pypto.gathermask(s_t, pattern_mode=1)
            s_o = pypto.gathermask(s_t, pattern_mode=2)
            pypto.set_vec_tile_shapes(1, 1, S_TILE_1, HALF)
            ye = pypto.sub(pypto.mul(x_e, c_e), pypto.mul(x_o, s_e))
            yo = pypto.add(pypto.mul(x_e, s_o), pypto.mul(x_o, c_o))
            pypto.assemble(ye, [b, 0, s_off, 0], out)
            pypto.assemble(yo, [b, 0, s_off, HALF], out)
            pypto.set_vec_tile_shapes(1, 1, S_TILE_1, D)


@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
)
def interleave_rope_kernel_n1_bf16_broadcast(
    x: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_BF16),
    cos: pypto.Tensor([pypto.DYNAMIC, 1, 1, 64], pypto.DT_BF16),
    sin: pypto.Tensor([pypto.DYNAMIC, 1, 1, 64], pypto.DT_BF16),
    out: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_BF16),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_vec_tile_shapes(1, 1, S_TILE_1, D)
    B = x.shape[0]
    S = x.shape[2]
    s_loops = (S + S_TILE_1 - 1) // S_TILE_1
    for b in pypto.loop(B, name="b_loop"):
        c_t = pypto.view(cos, [1, 1, 1, D], [b, 0, 0, 0])
        s_t = pypto.view(sin, [1, 1, 1, D], [b, 0, 0, 0])
        for s_blk in pypto.loop(s_loops, name="s_loop"):
            s_off = s_blk * S_TILE_1
            valid_s = (S - s_off).min(S_TILE_1)
            x_t = pypto.view(x, [1, 1, S_TILE_1, D], [b, 0, s_off, 0],
                             valid_shape=[1, 1, valid_s, D])
            x_e = pypto.gathermask(x_t, pattern_mode=1)
            x_o = pypto.gathermask(x_t, pattern_mode=2)
            c_e = pypto.gathermask(c_t, pattern_mode=1)
            c_o = pypto.gathermask(c_t, pattern_mode=2)
            s_e = pypto.gathermask(s_t, pattern_mode=1)
            s_o = pypto.gathermask(s_t, pattern_mode=2)
            pypto.set_vec_tile_shapes(1, 1, S_TILE_1, HALF)
            ye = pypto.sub(pypto.mul(x_e, c_e), pypto.mul(x_o, s_e))
            yo = pypto.add(pypto.mul(x_e, s_o), pypto.mul(x_o, c_o))
            pypto.assemble(ye, [b, 0, s_off, 0], out)
            pypto.assemble(yo, [b, 0, s_off, HALF], out)
            pypto.set_vec_tile_shapes(1, 1, S_TILE_1, D)


@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
)
def interleave_rope_kernel_n1_fp16(
    x: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_FP16),
    cos: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_FP16),
    sin: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_FP16),
    out: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], pypto.DT_FP16),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_vec_tile_shapes(1, 1, S_TILE_1, D)
    B = x.shape[0]
    S = x.shape[2]
    s_loops = (S + S_TILE_1 - 1) // S_TILE_1
    for b in pypto.loop(B, name="b_loop"):
        for s_blk in pypto.loop(s_loops, name="s_loop"):
            s_off = s_blk * S_TILE_1
            valid_s = (S - s_off).min(S_TILE_1)
            vshape = [1, 1, valid_s, D]
            x_t = pypto.view(x, [1, 1, S_TILE_1, D], [b, 0, s_off, 0], valid_shape=vshape)
            c_t = pypto.view(cos, [1, 1, S_TILE_1, D], [b, 0, s_off, 0], valid_shape=vshape)
            s_t = pypto.view(sin, [1, 1, S_TILE_1, D], [b, 0, s_off, 0], valid_shape=vshape)
            x_e = pypto.gathermask(x_t, pattern_mode=1)
            x_o = pypto.gathermask(x_t, pattern_mode=2)
            c_e = pypto.gathermask(c_t, pattern_mode=1)
            c_o = pypto.gathermask(c_t, pattern_mode=2)
            s_e = pypto.gathermask(s_t, pattern_mode=1)
            s_o = pypto.gathermask(s_t, pattern_mode=2)
            pypto.set_vec_tile_shapes(1, 1, S_TILE_1, HALF)
            xe_f = pypto.cast(x_e, pypto.DT_FP32)
            xo_f = pypto.cast(x_o, pypto.DT_FP32)
            ce_f = pypto.cast(c_e, pypto.DT_FP32)
            co_f = pypto.cast(c_o, pypto.DT_FP32)
            se_f = pypto.cast(s_e, pypto.DT_FP32)
            so_f = pypto.cast(s_o, pypto.DT_FP32)
            ye_f = pypto.sub(pypto.mul(xe_f, ce_f), pypto.mul(xo_f, se_f))
            yo_f = pypto.add(pypto.mul(xe_f, so_f), pypto.mul(xo_f, co_f))
            ye = pypto.cast(ye_f, pypto.DT_FP16)
            yo = pypto.cast(yo_f, pypto.DT_FP16)
            pypto.assemble(ye, [b, 0, s_off, 0], out)
            pypto.assemble(yo, [b, 0, s_off, HALF], out)
            pypto.set_vec_tile_shapes(1, 1, S_TILE_1, D)


_KERNELS = {
    (128, torch.bfloat16): interleave_rope_kernel_n128_bf16,
    (128, torch.float16): interleave_rope_kernel_n128_fp16,
    (1, torch.bfloat16): interleave_rope_kernel_n1_bf16,
    (1, torch.float16): interleave_rope_kernel_n1_fp16,
}


def interleave_rope_wrapper(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:

    assert x.is_contiguous(), "x must be contiguous"
    assert cos.is_contiguous() and sin.is_contiguous(), "cos/sin must be contiguous"
    assert x.dim() == 4 and cos.dim() == 4 and sin.dim() == 4
    B, N, S, Dd = x.shape
    assert Dd == 64, f"D must be 64, got {Dd}"
    assert N in (1, 128), f"N must be 1 or 128, got {N}"
    assert cos.shape[0] == B and cos.shape[1] == 1 and cos.shape[3] == 64
    assert sin.shape[0] == B and sin.shape[1] == 1 and sin.shape[3] == 64
    S_cs = cos.shape[2]
    assert S_cs in (1, S), f"S_cs must be 1 or {S}, got {S_cs}"
    assert cos.dtype == x.dtype and sin.dtype == x.dtype, \
        f"cos/sin dtype must match x; got x={x.dtype} cos={cos.dtype} sin={sin.dtype}"
    assert x.dtype in (torch.bfloat16, torch.float16), \
        f"unsupported dtype {x.dtype}"

    out = torch.empty_like(x)
    if (pypto.platform.npuarch == ASCEND_950_NPUARCH and N == 128 and x.dtype == torch.bfloat16
            and S == 1024 and S_cs == S and B in (2, 8)):
        kernel = interleave_rope_kernel_n128_bf16_950
    elif N == 128 and x.dtype == torch.bfloat16 and S == 1:
        kernel = interleave_rope_kernel_n128_bf16_short_s
    elif N == 128 and x.dtype == torch.bfloat16 and S == 2:
        kernel = interleave_rope_kernel_n128_bf16_short_s_btile
    elif S_cs == 1 and x.dtype == torch.bfloat16 and N == 128:
        kernel = interleave_rope_kernel_n128_bf16_broadcast
    elif S_cs == 1 and x.dtype == torch.bfloat16 and N == 1:
        kernel = interleave_rope_kernel_n1_bf16_broadcast
    elif S_cs == 1:
        raise AssertionError("S_cs=1 broadcast is only supported in BF16 kernels")
    elif N == 128 and x.dtype == torch.bfloat16 and S > 2048:
        kernel = interleave_rope_kernel_n128_bf16_unroll
    else:
        kernel = _KERNELS[(N, x.dtype)]
    kernel(x, cos, sin, out)
    return out
