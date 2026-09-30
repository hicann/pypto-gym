# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Two NCHW -> NC1HWC0 transdata bodies that differ only in where the channel tail is zeroed."""

from ascriptor.a2 import *

C0 = 16
DEBUG = False  # True is the historical 40-lane debug valuation; it has no typed signature here.
CHUNK = 1024                            # hw elems per tile; repeat = CHUNK/16 = 64
NLANES = 40 if DEBUG else 2


def transdata_nchw_to_5hd_half(x: GM[f16, (3072, 1)], y: GM[f16, (144, 16)],
                               b: i32, c1n: i32, hw: i32, pad_hw: i32):
    # x: flat NCHW padded on host to the full [B, C1*16, pad_hw] grid (channel
    # tail rows and the hw tail are zero), so every (b, c1) block is 16 full
    # bursts with whole-block gaps and no UB pre-zeroing (a V-pipe dup would
    # race the MTE2 load WAW); y keeps exact hw rows per plane.
    src_buf = Tensor(DT.half, [16, CHUNK], Position.UB)
    dst_buf = Tensor(DT.half, [CHUNK, C0], Position.UB)
    chunks = CeilDiv(pad_hw, CHUNK)
    vec_idx = GetVecIdx()
    with auto_sync():
        for blk in range(b * c1n):
            if blk % NLANES == vec_idx:
                base = blk * C0 * pad_hw                 # first channel elem in flat x
                out0 = blk * hw                          # first 5HD row in y
                for ck in range(chunks):
                    s = Var(ck * CHUNK)
                    span = Min(pad_hw - s, CHUNK)        # 16-aligned by construction
                    gm_to_ub_pad(src_buf, x[base + s:base + s + 1, :],
                                 n_burst=16, burst_len_element=span,
                                 src_stride_element=pad_hw - span,
                                 dst_stride=(CHUNK - span) // C0)
                    transdata5hd(dst_buf, src_buf, repeat=CHUNK // C0,
                                 src_row_stride=CHUNK, dst_row_stride=C0,
                                 src_rep_stride=1, dst_rep_stride=C0)
                    store = Min(hw - s, CHUNK)           # drop the hw pad rows
                    if store > 0:
                        ub_to_gm_pad(y[out0 + s:out0 + s + 1, :], dst_buf,
                                     n_burst=1, burst_len_element=store * C0,
                                     src_stride=0, dst_stride_element=0)
    return y


def kernel_for_host_pad(device):
    from importlib import import_module

    if device not in ("a2", "a3"):
        raise ValueError("This preserved convolution stage supports the A2/A3 CCE path")
    return import_module("ascriptor." + device).kernel(mode="mix", block_dim=1)(transdata_nchw_to_5hd_half)

# ----------------------------------------------------------------------------------------------------
# ub_pad: the same conversion, but the channel tail is zeroed on chip instead of by the host.
# The source's V-to-MTE2 ready/valid pair is what makes that safe: `dup` (V pipe) and the
# `gm_to_ub_pad` load (MTE2 pipe) both write src_buf, so without the two SEvents they race WAW.
# ----------------------------------------------------------------------------------------------------


def transdata_nchw_to_5hd_half_dup(x: GM[f16, (1920, 1)], y: GM[f16, (144, 16)],
                                   b: i32, c: i32, hw: i32, pad_hw: i32):
    src_buf = Tensor(DT.half, [16, CHUNK], Position.UB)
    dst_buf = Tensor(DT.half, [CHUNK, C0], Position.UB)
    dup_ready = SEvent(Pipe.V, Pipe.MTE2, name="t5_dup_ready")
    load_valid = SEvent(Pipe.MTE2, Pipe.V, preset=True, name="t5_load_valid")
    c1n = CeilDiv(c, C0)
    chunks = CeilDiv(pad_hw, CHUNK)
    vec_idx = GetVecIdx()
    with auto_sync():
        for blk in range(b * c1n):
            if blk % NLANES == vec_idx:
                bb = Var(blk // c1n)
                c1 = Var(blk % c1n)
                cvalid = Min(c - c1 * C0, C0)
                base = (bb * c + c1 * C0) * pad_hw
                out0 = (bb * c1n + c1) * hw
                for ck in range(chunks):
                    s = Var(ck * CHUNK)
                    span = Min(pad_hw - s, CHUNK)
                    load_valid.wait()
                    dup(src_buf, 0.0)
                    dup_ready.set()
                    dup_ready.wait()
                    gm_to_ub_pad(src_buf, x[base + s:base + s + 1, :],
                                 n_burst=cvalid, burst_len_element=span,
                                 src_stride_element=pad_hw - span,
                                 dst_stride=(CHUNK - span) // C0)
                    load_valid.set()
                    transdata5hd(dst_buf, src_buf, repeat=CHUNK // C0,
                                 src_row_stride=CHUNK, dst_row_stride=C0,
                                 src_rep_stride=1, dst_rep_stride=C0)
                    store = Min(hw - s, CHUNK)
                    if store > 0:
                        ub_to_gm_pad(y[out0 + s:out0 + s + 1, :], dst_buf,
                                     n_burst=1, burst_len_element=store * C0,
                                     src_stride=0, dst_stride_element=0)
    return y


def kernel_for_ub_pad(device):
    from importlib import import_module

    if device not in ("a2", "a3"):
        raise ValueError("This preserved convolution stage supports the A2/A3 CCE path")
    return import_module("ascriptor." + device).kernel(mode="mix", block_dim=1)(transdata_nchw_to_5hd_half_dup)


def kernel_for(variant, device):
    """Pick the padding protocol the case names. Both bodies write the same NC1HWC0 output;
    they differ only in who zeroes the channel tail, and therefore in what the host must hand
    them -- `host_pad` gets a [B, C1*16, pad_hw] grid already padded to 3072 elements, `ub_pad`
    gets the unpadded [B, C, pad_hw] grid in 1920."""
    if variant == "host_pad":
        return kernel_for_host_pad(device)
    if variant == "ub_pad":
        return kernel_for_ub_pad(device)
    raise ValueError(f"unknown channel-padding variant {variant!r}")
