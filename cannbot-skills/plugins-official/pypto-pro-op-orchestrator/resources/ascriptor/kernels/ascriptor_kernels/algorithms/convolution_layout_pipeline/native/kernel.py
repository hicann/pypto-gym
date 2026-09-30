# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Four native convolution bodies: plain FP16 accumulation, bias on the first K tile, stride-2 with dilation-2, and a 4x4 filter reloaded per batch."""

from ascriptor.a2 import *

# ----------------------------------------------------------------------------------------------------
# basic.py
# ----------------------------------------------------------------------------------------------------

C0 = 16
BASIC_KH = BASIC_KW = 3
BASIC_PAD = (1, 1, 1, 1)
BASIC_TILE_M = 32
# static worst-case capacities (largest shape this binary serves); pairwise
# distinct from every GM dim so OpExec scalar->dim auto-match stays unambiguous
BASIC_C_MAX, BASIC_H_MAX, BASIC_W_MAX, BASIC_COUT_MAX = 32, 8, 8, 48
BASIC_C1_MAX = (BASIC_C_MAX + C0 - 1) // C0
BASIC_K_MAX = BASIC_C1_MAX * BASIC_KH * BASIC_KW * C0
BASIC_COUT_P = (BASIC_COUT_MAX + 15) // 16 * 16
BASIC_TILE_K = 144                                    # divides K for every C1 in 1..BASIC_C1_MAX


def conv_half_basic(data: GM[f16, (128, 16)], weight: GM[f16, (48, 288)], out: GM[f32, (192, 16)],
                    h: i32, w: i32, c: i32, cout: i32, m_pad: i32):
    conv = Conv2D(BASIC_KH, BASIC_KW, pad=BASIC_PAD)
    fm = Tensor(DT.half, [BASIC_C1_MAX * BASIC_H_MAX * BASIC_W_MAX, C0], Position.L1)
    w_l1 = Tensor(DT.half, [BASIC_COUT_P, BASIC_K_MAX], Position.L1)
    l0c = Tensor(DT.float, [BASIC_TILE_M, BASIC_COUT_P], Position.L0C)
    with auto_sync():
        fm <<= data[:, :]                       # GM staged at worst case; Var c/h/w mask the tail
        w_l1 <<= weight[:, :]
        for m0 in range(0, m_pad, BASIC_TILE_M):
            conv2d(l0c, fm, w_l1, conv, h=h, w=w, c=c, cout=cout, m0=m0, tile_k=BASIC_TILE_K)
            l0c_to_gm_nz2nz(out[m0:m0 + BASIC_TILE_M, :], l0c, m_pad=m_pad)
    return out


def kernel_for_basic(device):
    from importlib import import_module

    if device not in ("a2", "a3", "a5"):
        raise ValueError("Native convolution supports the declared A2/A3/A5 CCE paths")
    return import_module("ascriptor." + device).kernel(mode="cube", block_dim=1)(conv_half_basic)

# ----------------------------------------------------------------------------------------------------
# bias.py
# ----------------------------------------------------------------------------------------------------

BIAS_KH = BIAS_KW = 3
BIAS_PAD = (1, 1, 1, 1)
BIAS_TILE_M = 32
# static worst-case capacities; pairwise distinct from every GM dim
BIAS_C_MAX, BIAS_H_MAX, BIAS_W_MAX, BIAS_COUT_MAX = 32, 8, 8, 48
BIAS_C1_MAX = (BIAS_C_MAX + C0 - 1) // C0
BIAS_K_MAX = BIAS_C1_MAX * BIAS_KH * BIAS_KW * C0
BIAS_COUT_P = (BIAS_COUT_MAX + 15) // 16 * 16
BIAS_TILE_K = 144                                    # divides K for every C1 in 1..BIAS_C1_MAX


def conv_half_bias(data: GM[f16, (128, 16)], weight: GM[f16, (48, 288)], bias: GM[f32, (1, 48)], out: GM[f32, (192, 16)],
                   h: i32, w: i32, c: i32, cout: i32, m_pad: i32):
    conv = Conv2D(BIAS_KH, BIAS_KW, pad=BIAS_PAD)
    fm = Tensor(DT.half, [BIAS_C1_MAX * BIAS_H_MAX * BIAS_W_MAX, C0], Position.L1)
    w_l1 = Tensor(DT.half, [BIAS_COUT_P, BIAS_K_MAX], Position.L1)
    b_l1 = Tensor(DT.float, [1, BIAS_COUT_P], Position.L1, layout=Layout.ND)
    l0c = Tensor(DT.float, [BIAS_TILE_M, BIAS_COUT_P], Position.L0C)
    with auto_sync():
        fm <<= data[:, :]
        w_l1 <<= weight[:, :]
        b_l1 <<= bias[:, :]                 # ND layout -> flat gm_to_l1 (BT contract)
        for m0 in range(0, m_pad, BIAS_TILE_M):
            conv2d(l0c, fm, w_l1, conv, h=h, w=w, c=c, cout=cout, m0=m0,
                   tile_k=BIAS_TILE_K, bias=b_l1)
            l0c_to_gm_nz2nz(out[m0:m0 + BIAS_TILE_M, :], l0c, m_pad=m_pad)
    return out


def kernel_for_bias(device):
    from importlib import import_module

    if device not in ("a2", "a3", "a5"):
        raise ValueError("Native convolution supports the declared A2/A3/A5 CCE paths")
    return import_module("ascriptor." + device).kernel(mode="cube", block_dim=1)(conv_half_bias)

# ----------------------------------------------------------------------------------------------------
# dilation.py
# ----------------------------------------------------------------------------------------------------

DILATED_KH = DILATED_KW = 3
DILATED_PAD = (2, 2, 2, 2)
DILATED_STRIDE = (2, 2)
DILATED_DIL = (2, 2)
DILATED_TILE_M = 16
# static worst-case capacities; pairwise distinct from every GM dim
DILATED_C_MAX, DILATED_H_MAX, DILATED_W_MAX, DILATED_COUT_MAX = 16, 12, 12, 32
DILATED_C1_MAX = (DILATED_C_MAX + C0 - 1) // C0
DILATED_K_MAX = DILATED_C1_MAX * DILATED_KH * DILATED_KW * C0
DILATED_COUT_P = (DILATED_COUT_MAX + 15) // 16 * 16
DILATED_TILE_K = 144                                    # == DILATED_K_MAX: runtime K tail loop is empty


def conv_half_dilation(data: GM[f16, (144, 16)], weight: GM[f16, (32, 144)], out: GM[f32, (96, 16)],
                       h: i32, w: i32, c: i32, cout: i32, m_pad: i32):
    conv = Conv2D(DILATED_KH, DILATED_KW, pad=DILATED_PAD, stride=DILATED_STRIDE, dilation=DILATED_DIL)
    fm = Tensor(DT.half, [DILATED_C1_MAX * DILATED_H_MAX * DILATED_W_MAX, C0], Position.L1)
    w_l1 = Tensor(DT.half, [DILATED_COUT_P, DILATED_K_MAX], Position.L1)
    l0c = Tensor(DT.float, [DILATED_TILE_M, DILATED_COUT_P], Position.L0C)
    with auto_sync():
        fm <<= data[:, :]
        w_l1 <<= weight[:, :]
        for m0 in range(0, m_pad, DILATED_TILE_M):
            conv2d(l0c, fm, w_l1, conv, h=h, w=w, c=c, cout=cout, m0=m0, tile_k=DILATED_TILE_K)
            l0c_to_gm_nz2nz(out[m0:m0 + DILATED_TILE_M, :], l0c, m_pad=m_pad)
    return out


def kernel_for_dilation(device):
    from importlib import import_module

    if device not in ("a2", "a3", "a5"):
        raise ValueError("Native convolution supports the declared A2/A3/A5 CCE paths")
    return import_module("ascriptor." + device).kernel(mode="cube", block_dim=1)(conv_half_dilation)

# ----------------------------------------------------------------------------------------------------
# large.py
# ----------------------------------------------------------------------------------------------------


LARGE_DEBUG = False  # True is the historical 64x64 / 20-core valuation; not declared here.
LARGE_KH = LARGE_KW = 4
LARGE_PAD = (1, 2, 1, 2)                      # asymmetric same-pad for even filter
LARGE_TILE_M = 32
LARGE_TILE_K = 256
LARGE_MULTICORE = 20 if LARGE_DEBUG else 1
# static worst-case capacities (full batch/plane this binary serves)
LARGE_C_MAX, LARGE_COUT_MAX = 32, 64
LARGE_H_MAX = LARGE_W_MAX = 64 if LARGE_DEBUG else 16
LARGE_C1_MAX = (LARGE_C_MAX + C0 - 1) // C0
LARGE_K_MAX = LARGE_C1_MAX * LARGE_KH * LARGE_KW * C0          # 512 -> two 256 K-tiles
      # same-pad keeps HoxWo == HxW
LARGE_COUT_P = (LARGE_COUT_MAX + 15) // 16 * 16
LARGE_C1O = LARGE_COUT_P // C0
LARGE_FM_PLANE = LARGE_C1_MAX * LARGE_H_MAX * LARGE_W_MAX      # static per-batch fmap stride (GM rows)


def conv_half_large(data: GM[f16, (1024, 16)], weight: GM[f16, ('cout', 512)], out: GM[f32, (1792, 16)],
                    h: i32, w: i32, c: i32, cout: i32, m_pad: i32, nb: i32):
    conv = Conv2D(LARGE_KH, LARGE_KW, pad=LARGE_PAD)
    w_l1 = Tensor(DT.half, [LARGE_COUT_P, LARGE_K_MAX], Position.L1)
    fm = Tensor(DT.half, [LARGE_FM_PLANE, C0], Position.L1)
    l0c = Tensor(DT.float, [LARGE_TILE_M, LARGE_COUT_P], Position.L0C)
    cube_idx = GetCubeIdx()
    with auto_sync():
        w_l1 <<= weight[:, :]
        for n in range(nb):                             # weight stays resident; fmap reloads
            fm <<= data[n * LARGE_FM_PLANE:n * LARGE_FM_PLANE + LARGE_FM_PLANE, :]
            for m0 in range(0, m_pad, LARGE_TILE_M):
                if cube_idx == (m0 // LARGE_TILE_M) % LARGE_MULTICORE:
                    conv2d(l0c, fm, w_l1, conv, h=h, w=w, c=c, cout=cout,
                           m0=m0, tile_k=LARGE_TILE_K)
                    row0 = n * LARGE_C1O * m_pad + m0
                    l0c_to_gm_nz2nz(out[row0:row0 + LARGE_TILE_M, :], l0c, m_pad=m_pad)
    return out


def kernel_for_large(device):
    from importlib import import_module

    if device not in ("a2", "a3", "a5"):
        raise ValueError("Native convolution supports the declared A2/A3/A5 CCE paths")
    return import_module("ascriptor." + device).kernel(mode="cube", block_dim=1)(conv_half_large)


def kernel_for(mode, device):
    """Pick the body this case names. The four differ only in the convolution descriptor and
    the static worst-case capacities their L1 and L0C tensors are sized against -- every one of
    them is `gm_to_l1`, then `conv2d` over a static M loop, then `l0c_to_gm_nz2nz`."""
    factory = {"basic": kernel_for_basic, "bias": kernel_for_bias,
               "dilation": kernel_for_dilation, "large": kernel_for_large}.get(mode)
    if factory is None:
        raise ValueError(f"unknown native convolution mode {mode!r}")
    return factory(device)
