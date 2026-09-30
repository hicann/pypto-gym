# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The four convolution stages as four separately launchable kernels, over one shared geometry."""

from ascriptor.a2 import *

# ----------------------------------------------------------------------------------------------------
# common.py -- one geometry, shared by all four stages
# Shared geometry for the stepwise NCHW conv2d pipeline (a2, fp16).
#
# Spec (--debug, 910B3): x [1, 64, 256, 256] * w [128, 64, 3, 3],
# stride 2, dilation 2 -> y [1, 128, 128, 128].
# Effective filter is 5x5 (dilated 3x3), so SAME/2 needs pad 2 on every edge:
# Ho = (H + 4 - 5) // 2 + 1 = H / 2.
#
# Simulator runs the identical kernels on a shrunk plane (H = W = 32, all
# channels kept full-size) so the python golden stays fast.
#
# The H-direction pad rows are baked into the 5HD plane as zero rows (HP = H + 4)
# so the compute kernel sees a per-output-row feature map slice with no runtime
# pad-top/bottom switch: the Conv2D descriptor only pads left/right.
# ----------------------------------------------------------------------------------------------------

DEBUG = False

C0 = 16
B = 1
CIN = 64
COUT = 128
KH = KW = 3
STRIDE = 2
DIL = 2
PAD = 2                                  # SAME pad for eff5/s2 (works in all 4 dirs)

H = W = 256 if DEBUG else 32
HO, WO = H // 2, W // 2
M = HO * WO                              # multiple of 16 (W >= 32)
HW = H * W
HP = H + 2 * PAD                         # zero-row padded 5HD height
C1 = CIN // C0                           # 4 (exact, no channel tail)
C1O = COUT // C0                         # 8
K = C1 * KH * KW * C0                    # 576
TILE_K = 96                              # 576 = 6 x 96; 96*128*2B = 24KB <= 32KB L0 slot

N_VEC = 40 if DEBUG else 2
N_CUBE = 20 if DEBUG else 1

# ----------------------------------------------------------------------------------------------------
# transdata_x_to_5hd.py
# ----------------------------------------------------------------------------------------------------


X_CHUNK = 1024
BLOCKS = B * C1
X_CHUNKS = HW // X_CHUNK


def transdata_x_to_5hd(x: GM[f16, (B * CIN, HW)], y: GM[f16, (C1 * HP * W, C0)], nv: i32):
    # contiguous per-lane unit block + DBuff: consecutive units alternate slots
    # so the load of unit u+1 overlaps the transpose/store of unit u.
    src = DBuff(DT.half, [16, X_CHUNK], Position.UB)
    dst = DBuff(DT.half, [X_CHUNK, C0], Position.UB)
    zero = Tensor(DT.half, [PAD * W, C0], Position.UB)
    dup_ready = SEvent(Pipe.V, Pipe.MTE3, name="x5hd_dup_ready")  # autosync misses V-dup->MTE3 RAW
    vec_idx = GetVecIdx()
    units = BLOCKS * X_CHUNKS
    per_lane = (units + N_VEC - 1) // N_VEC
    begin = per_lane * vec_idx
    end = Min(begin + per_lane, units)
    with auto_sync():
        dup(zero, 0.0, repeat=PAD * W * C0 // 128, dst_blk_stride=1, dst_rep_stride=8)
        dup_ready.set()
        dup_ready.wait()
        for blk in range(BLOCKS):
            if blk % nv == vec_idx:                       # H pad rows stay zero
                y[blk * HP * W:blk * HP * W + PAD * W, :] <<= zero
                y[(blk * HP + PAD + H) * W:(blk * HP + PAD + H) * W + PAD * W, :] <<= zero
        for u in range(begin, end):
            blk = var_div(u, X_CHUNKS)
            s = var_mod(u, X_CHUNKS) * X_CHUNK
            s_buf = src[u]
            d_buf = dst[u]
            s_buf <<= x[blk * C0:blk * C0 + 16, s:s + X_CHUNK]
            transdata5hd(d_buf, s_buf, repeat=X_CHUNK // 16,
                         src_row_stride=X_CHUNK, dst_row_stride=C0,
                         src_rep_stride=1, dst_rep_stride=C0)
            out0 = (blk * HP + PAD) * W + s
            y[out0:out0 + X_CHUNK, :] <<= d_buf
    return y


def kernel_for_x_to_5hd(device):
    from importlib import import_module

    if device not in ("a2", "a3"):
        raise ValueError("This preserved convolution stage supports the A2/A3 CCE path")
    return import_module("ascriptor." + device).kernel(mode="vec", block_dim=2)(transdata_x_to_5hd)

# ----------------------------------------------------------------------------------------------------
# transdata_w_to_fractal.py
# ----------------------------------------------------------------------------------------------------


TAPS = KH * KW                  # 9
KC1 = TAPS * C0                 # 144 K columns per c1
NCOB = COUT // 16               # 8 cout blocks


def transdata_w_to_fractal(w: GM[f16, (COUT, K)], wz: GM[f16, (COUT, K)], nv: i32):
    # contiguous per-lane unit block + DBuff (slots alternate per iteration).
    src = DBuff(DT.half, [16, KC1], Position.UB)
    t1 = DBuff(DT.half, [KC1, 16], Position.UB)
    t2 = DBuff(DT.half, [16, KC1], Position.UB)
    vec_idx = GetVecIdx()
    units = NCOB * C1
    per_lane = (units + N_VEC - 1) // N_VEC
    begin = per_lane * vec_idx
    end = Min(begin + per_lane, units)
    with auto_sync():
        for u in range(begin, end):
            cob = var_div(u, C1)
            c1 = var_mod(u, C1)
            sb = src[u]
            tb1 = t1[u]
            tb2 = t2[u]
            sb <<= w[cob * 16:cob * 16 + 16, c1 * KC1:c1 * KC1 + KC1]
            transdata5hd(tb1, sb, repeat=TAPS,
                         src_row_stride=KC1, dst_row_stride=16,
                         src_rep_stride=1, dst_rep_stride=16)
            transdata5hd(tb2, tb1, repeat=TAPS,
                         src_row_stride=TAPS * 16, dst_row_stride=KC1,
                         src_rep_stride=1, dst_rep_stride=1)
            wz[cob * 16:cob * 16 + 16, c1 * KC1:c1 * KC1 + KC1] <<= tb2
    return wz


def kernel_for_w_to_fractal(device):
    from importlib import import_module

    if device not in ("a2", "a3"):
        raise ValueError("This preserved convolution stage supports the A2/A3 CCE path")
    return import_module("ascriptor." + device).kernel(mode="vec", block_dim=2)(transdata_w_to_fractal)

# ----------------------------------------------------------------------------------------------------
# conv_compute.py
# ----------------------------------------------------------------------------------------------------


ROWS = DIL * (KH - 1) + 1               # 5 input rows per output row


def conv_compute(data: GM[f16, (C1 * HP * W, C0)], wz: GM[f16, (COUT, K)], out: GM[f32, (C1O * M, C0)], nc: i32):
    # nc is only the mandatory Var arg; the gate uses static N_CUBE.
    conv = Conv2D(KH, KW, pad=(PAD, PAD, 0, 0), stride=(STRIDE, STRIDE),
                  dilation=(DIL, DIL))
    w_l1 = Tensor(DT.half, [COUT, K], Position.L1)
    # contiguous per-core ho block (not a strided gate): consecutive iterations
    # alternate DBuff slots, which is the assumption behind the per-DBuff
    # depth-2 DEvent credits; the double buffers overlap the next fm load and
    # the previous NZ store with the current mmad.
    fm = DBuff(DT.half, [C1 * ROWS * W, C0], Position.L1)
    l0c = DBuff(DT.float, [WO, COUT], Position.L0C)
    cube_idx = GetCubeIdx()
    per_core = (HO + N_CUBE - 1) // N_CUBE
    begin = per_core * cube_idx
    end = Min(begin + per_core, HO)
    with auto_sync():
        w_l1 <<= wz[:, :]
        for ho in range(begin, end):
            f = fm[ho]
            lc = l0c[ho]
            gm_to_l1(f, data[ho * STRIDE * W:, :],
                     n_burst=C1, burst_len=ROWS * W,
                     src_stride=(HP - ROWS) * W, dst_stride=0)
            conv2d(lc, f, w_l1, conv, h=ROWS, w=W, c=CIN, cout=COUT,
                   m0=0, tile_k=TILE_K)
            l0c_to_gm_nz2nz(out[ho * WO:ho * WO + WO, :], lc, m_pad=M)
    return out


def kernel_for_conv_compute(device):
    from importlib import import_module

    if device not in ("a2", "a3"):
        raise ValueError("This preserved convolution stage supports the A2/A3 CCE path")
    return import_module("ascriptor." + device).kernel(mode="cube", block_dim=1)(conv_compute)

# ----------------------------------------------------------------------------------------------------
# transdata_out_to_nchw.py
# ----------------------------------------------------------------------------------------------------


O_CHUNK = min(512, M)                 # cast repeat = O_CHUNK*16/64 <= 128 <= 255
O_CHUNKS = M // O_CHUNK


def transdata_out_to_nchw(conv_out: GM[f32, (C1O * M, C0)], y: GM[f16, (COUT, M)], nv: i32):
    # contiguous per-lane unit block + DBuff (slots alternate per iteration).
    src = DBuff(DT.float, [O_CHUNK, C0], Position.UB)
    mid = DBuff(DT.half, [O_CHUNK, C0], Position.UB)
    dst = DBuff(DT.half, [16, O_CHUNK], Position.UB)
    vec_idx = GetVecIdx()
    units = C1O * O_CHUNKS
    per_lane = (units + N_VEC - 1) // N_VEC
    begin = per_lane * vec_idx
    end = Min(begin + per_lane, units)
    with auto_sync():
        for u in range(begin, end):
            c1o = var_div(u, O_CHUNKS)
            s = c1o * M + var_mod(u, O_CHUNKS) * O_CHUNK
            sb = src[u]
            mb = mid[u]
            db = dst[u]
            sb <<= conv_out[s:s + O_CHUNK, :]
            cast(mb, sb, round_mode=RoundMode.TO_EVEN)
            transdata5hd(db, mb, repeat=O_CHUNK // 16,
                         src_row_stride=C0, dst_row_stride=O_CHUNK,
                         src_rep_stride=16, dst_rep_stride=1)
            y[c1o * C0:c1o * C0 + 16, var_mod(u, O_CHUNKS) * O_CHUNK:var_mod(u, O_CHUNKS) * O_CHUNK + O_CHUNK] <<= db
    return y


def kernel_for_out_to_nchw(device):
    from importlib import import_module

    if device not in ("a2", "a3"):
        raise ValueError("This preserved convolution stage supports the A2/A3 CCE path")
    return import_module("ascriptor." + device).kernel(mode="vec", block_dim=2)(transdata_out_to_nchw)
