# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One MIX kernel that runs all four convolution stages behind cross-core barriers."""

from ascriptor.a2 import *

DEBUG = False  # True is the historical 256x256 debug valuation; it has no typed signature here.

C0 = 16
B = 1
CIN = 64
COUT = 128
KH = KW = 3
STRIDE = 2
DIL = 2
PAD = 2
H = W = 256 if DEBUG else 32
HO, WO = H // 2, W // 2
M = HO * WO
HW = H * W
HP = H + 2 * PAD
C1 = CIN // C0
C1O = COUT // C0
K = C1 * KH * KW * C0                   # 576
TILE_K = 96
ROWS = DIL * (KH - 1) + 1               # 5 input rows per output row

N_VEC = 40 if DEBUG else 2
N_CUBE = 20 if DEBUG else 1

X_CHUNK = 512                           # halved: DBuff slots of all 3 stages share UB
X_CHUNKS = HW // X_CHUNK
TAPS = KH * KW
KC1 = TAPS * C0                         # 144 K columns per c1
NCOB = COUT // 16
O_CHUNK = min(256, M)                   # stage 4 rows per batch (cast repeat <= 64)
O_CHUNKS = M // O_CHUNK

F_VIN = 0                               # crosscore flags
F_VC = 1
F_CUBE = 2
F_CV = 3


def conv_integrated(x: GM[f16, (64, 1024)], w: GM[f16, (128, 576)], y: GM[f16, (128, 256)], nv: i32):
    data = split_workspace(DT.half, [B * C1 * HP * W, C0], name="data5hd_ws")
    wz = split_workspace(DT.half, [COUT, K], name="wz_ws")
    nzc = split_workspace(DT.float, [C1O * M, C0], name="nz_ws")

    # ---- vec: x -> 5HD (pad rows zeroed) + w -> fractal ----
    with vec_scope():
        vec_idx = GetVecIdx()
        xsrc = DBuff(DT.half, [16, X_CHUNK], Position.UB)
        xdst = DBuff(DT.half, [X_CHUNK, C0], Position.UB)
        zero = Tensor(DT.half, [PAD * W, C0], Position.UB)
        wsrc = DBuff(DT.half, [16, KC1], Position.UB)
        wt1 = DBuff(DT.half, [KC1, 16], Position.UB)
        wt2 = DBuff(DT.half, [16, KC1], Position.UB)
        dup_ready = SEvent(Pipe.V, Pipe.MTE3, name="pad_dup_ready")
        x_units = B * C1 * X_CHUNKS
        x_per_lane = (x_units + N_VEC - 1) // N_VEC
        x_begin = x_per_lane * vec_idx
        x_end = Min(x_begin + x_per_lane, x_units)
        with auto_sync():
            dup(zero, 0.0, repeat=PAD * W * C0 // 128, dst_blk_stride=1, dst_rep_stride=8)
            dup_ready.set()
            dup_ready.wait()
            for blk in range(B * C1):
                if blk % nv == vec_idx:
                    data[blk * HP * W:blk * HP * W + PAD * W, :] <<= zero
                    data[(blk * HP + PAD + H) * W:(blk * HP + PAD + H) * W + PAD * W, :] <<= zero
            for u in range(x_begin, x_end):
                blk = var_div(u, X_CHUNKS)
                s = var_mod(u, X_CHUNKS) * X_CHUNK
                sb = xsrc[u]
                db = xdst[u]
                sb <<= x[blk * C0:blk * C0 + 16, s:s + X_CHUNK]
                transdata5hd(db, sb, repeat=X_CHUNK // 16,
                             src_row_stride=X_CHUNK, dst_row_stride=C0,
                             src_rep_stride=1, dst_rep_stride=C0)
                out0 = (blk * HP + PAD) * W + s
                data[out0:out0 + X_CHUNK, :] <<= db
            w_units = NCOB * C1
            w_per_lane = (w_units + N_VEC - 1) // N_VEC
            w_begin = w_per_lane * vec_idx
            w_end = Min(w_begin + w_per_lane, w_units)
            for u in range(w_begin, w_end):
                cob = var_div(u, C1)
                c1 = var_mod(u, C1)
                wsb = wsrc[u]
                wb1 = wt1[u]
                wb2 = wt2[u]
                wsb <<= w[cob * 16:cob * 16 + 16, c1 * KC1:c1 * KC1 + KC1]
                transdata5hd(wb1, wsb, repeat=TAPS,
                             src_row_stride=KC1, dst_row_stride=16,
                             src_rep_stride=1, dst_rep_stride=16)
                transdata5hd(wb2, wb1, repeat=TAPS,
                             src_row_stride=TAPS * 16, dst_row_stride=KC1,
                             src_rep_stride=1, dst_rep_stride=1)
                wz[cob * 16:cob * 16 + 16, c1 * KC1:c1 * KC1 + KC1] <<= wb2
        allvec_ready(F_VIN)
        allvec_wait(F_VIN)
        vec_ready(F_VC)

    # ---- cube: per-output-row load3d conv ----
    wait_vec(F_VC)
    conv = Conv2D(KH, KW, pad=(PAD, PAD, 0, 0), stride=(STRIDE, STRIDE),
                  dilation=(DIL, DIL))
    w_l1 = Tensor(DT.half, [COUT, K], Position.L1)
    fm = DBuff(DT.half, [C1 * ROWS * W, C0], Position.L1)
    l0c = DBuff(DT.float, [WO, COUT], Position.L0C)
    cube_idx = GetCubeIdx()
    per_core = (HO + N_CUBE - 1) // N_CUBE
    begin = per_core * cube_idx
    end = Min(begin + per_core, HO)
    with auto_sync():
        w_l1 <<= wz[:, :]
        for ho in range(begin, end):                  # contiguous block: slots alternate
            f = fm[ho]
            lc = l0c[ho]
            gm_to_l1(f, data[ho * STRIDE * W:, :],
                     n_burst=C1, burst_len=ROWS * W,
                     src_stride=(HP - ROWS) * W, dst_stride=0)
            conv2d(lc, f, w_l1, conv, h=ROWS, w=W, c=CIN, cout=COUT,
                   m0=0, tile_k=TILE_K)
            l0c_to_gm_nz2nz(nzc[ho * WO:ho * WO + WO, :], lc, m_pad=M)
    allcube_ready(F_CUBE)
    allcube_wait(F_CUBE)
    cube_ready(F_CV)

    # ---- vec: NZ fp32 -> NCHW fp16 ----
    with vec_scope():
        wait_cube(F_CV)
        vec_idx2 = GetVecIdx()
        osrc = DBuff(DT.float, [O_CHUNK, C0], Position.UB)
        omid = DBuff(DT.half, [O_CHUNK, C0], Position.UB)
        odst = DBuff(DT.half, [16, O_CHUNK], Position.UB)
        o_units = C1O * O_CHUNKS
        o_per_lane = (o_units + N_VEC - 1) // N_VEC
        o_begin = o_per_lane * vec_idx2
        o_end = Min(o_begin + o_per_lane, o_units)
        with auto_sync():
            for u in range(o_begin, o_end):
                c1o = var_div(u, O_CHUNKS)
                ck = var_mod(u, O_CHUNKS)
                sb = osrc[u]
                mb = omid[u]
                db = odst[u]
                sb <<= nzc[c1o * M + ck * O_CHUNK:c1o * M + ck * O_CHUNK + O_CHUNK, :]
                cast(mb, sb, round_mode=RoundMode.TO_EVEN)
                transdata5hd(db, mb, repeat=O_CHUNK // 16,
                             src_row_stride=C0, dst_row_stride=O_CHUNK,
                             src_rep_stride=16, dst_rep_stride=1)
                y[c1o * C0:c1o * C0 + 16, ck * O_CHUNK:ck * O_CHUNK + O_CHUNK] <<= db
    return y


def kernel_for(device):
    from importlib import import_module

    if device not in ("a2", "a3"):
        raise ValueError("This preserved convolution stage supports the A2/A3 CCE path")
    return import_module("ascriptor." + device).kernel(mode="mix", block_dim=1)(conv_integrated)
