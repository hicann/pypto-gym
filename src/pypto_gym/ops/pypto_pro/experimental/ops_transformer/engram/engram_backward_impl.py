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
"""PyPTO-Pro engram_backward kernel implementation.

Backward of the Engram Gated Memory operator (forward = engram_v4).

Single-kernel design: a vector section (produces grad_value_ws / grad_key_ws
plus grad_hidden_states and the per-subblock grad_γ FP32 workspace) feeds a
cube section (Nest1 grad_emb + Nest2 grad_W_v + Nest3 grad_W_k). BF16
activation inputs are cast to FP32 once at the computation boundary; all RMS /
gate / scaled-dot math runs in FP32 inside VF helpers. Final output stores
are low precision (BF16).

7-step backward (per head m, reversed Step7->Step1):
  Step7  grad_gates[m]=Σ_h(go·value); grad_value=Σ_m(go·gates)
  Step6  linear_bw(grad_value, E, W_v)  -> grad_emb_v, grad_W_v
  Step5  grad_score   = signed_sqrt_gate_bw(grad_gate, score, gate)
  Step4  grad_nKey    = grad_score·(1/√H)·nQuery ; grad_nQuery symmetric
  Step3  (grad_hidden_m, grad_γ_q) = rms_norm_bw(grad_nQuery, hidden, γ_q)
  Step2  (grad_key_m,  grad_γ_k) = rms_norm_bw(grad_nKey, key, γ_k)
  Step1  linear_bw(grad_key_m, E, W_k[m]) -> grad_emb_k, grad_W_k[m]
  Sum:   grad_emb = grad_emb_v + Σ_m grad_emb_k
"""

import logging

import torch

import pypto_pro.language as pl
from pypto_pro.runtime.platform import get_platform_info
from pypto_pro.runtime.tilingkey import TilingKeyField

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


# TilingKey: two specialization fields (2x4 = 8 compiled variants).
#   HMode (bits=1): vector sub-tile row count on H.
#     0 -> H=1280 (rows: B=8, A=16; H_CHUNK=1280)   1 -> H=2560 (rows: B=4, A=8; H_CHUNK=2560)
#     rows*H is byte-constant -> [rows,H_CHUNK] UB layout is identical for both keys;
#     only [rows,64]/[1,H_CHUNK] tiles need HMode-specialized addresses.
#   BSMode (bits=2): vector worker tile V_TILE (largest with ceil(BS/V_TILE) >= 64):
#     0 -> 128 (BS>=8192)  1 -> 64 (BS>=4096)  2 -> 32 (BS>=2048)  3 -> 16 (small BS)
#     V_TILE only enters scalar arithmetic, never TileType.shape/addrs -> no addr specialization.


class EngramTilingKey:
    HMode = TilingKeyField(bits=1, values=[0, 1])        # 0 -> H=1280, 1 -> H=2560
    BSMode = TilingKeyField(bits=2, values=[0, 1, 2, 3]) # V_TILE = 128/64/32/16


# Compile-time constants (mirror forward engram_v4).
# M_H (head count) is DYNAMIC (1-16), derived in kernel body from grad_output.shape[1].
TILE_M = 128             # CUBE BS-row tile (== cube_mn); vector uses V_TILE
TILE_N = 128             # De/col tile (== cube_mn)
# Vector worker BS-row tile candidates, selected per launch by BSMode so the
# 64 AIV workers (32 cores x 2) stay filled. V_TILE is a multiple of
# TILE_BS_VEC_A (8/16) so a worker's row window never straddles VF sub-tiles.
V_TILE_BS0 = 128           # BSMode 0: BS >= 8192
V_TILE_BS1 = 64            # BSMode 1: BS >= 4096
V_TILE_BS2 = 32            # BSMode 2: BS >= 2048
V_TILE_BS3 = 16            # BSMode 3: small BS
LANES_FP32 = 64          # VF FP32 register width

# Per-HMode vector-tile split sizes. `if pl.constexpr(HMode==1)` binds
# TILE_BS_VEC / TILE_BS_VEC_A / H_CHUNK to one pair so TileType.shape stays
# compile-time constant. H=1280 doubles rows vs H=2560 (byte-identical UB).
TILE_BS_VEC_2560 = 4           # H=2560: vec sub-row tile (B-key/B-query)
TILE_BS_VEC_1280 = 8           # H=1280: 2x rows
TILE_BS_VEC_A_2560 = 8         # H=2560: Pass A dedicated tile
TILE_BS_VEC_A_1280 = 16        # H=1280: 2x rows
H_CHUNK_2560 = 2560           # H=2560: one full-H vector tile
H_CHUNK_1280 = 1280           # H=1280: one full-H vector tile

CLAMP_VALUE = 1.0e-6     # signed_sqrt_gate |s| floor
RMS_EPS = 1.0e-6         # RMSNorm zero-division guard
GATE_EPS = 1.0e-12       # signed_sqrt_gate denominator guard (aligns golden)

# UB addresses (vector section), two groups:
#  (A) [rows,H_CHUNK] tiles: bytes = rows*H is CONSTANT across HMode
#      (4*2560==8*1280), so the same addresses serve both keys.
#  (B) [rows,64]/[1,H_CHUNK] tiles: column count does NOT scale with H, so
#      doubling rows doubles bytes -> addresses MUST be specialized per HMode
#      (a shared 0x400 stride would overlap at H=1280 -> RMS-bw reads garbage).
#  HMode-dependent addrs use a parser-foldable select `HMode*A_2560 + (1-HMode)*A_1280`
#  (HMode is a ConstInt) — parser can't see names bound in `if pl.constexpr` blocks.

# (A) H-invariant addresses (byte-constant [rows,H_CHUNK] tiles).
VA_GOUT16 = 0x00000  # [rows,H_CHUNK] BF16 grad_output load (Pass A & B)
VA_GOUT32 = 0x05000  # [rows,H_CHUNK] FP32 grad_output FP32 (Pass A & B)
VA_PTNR16 = 0x0F000  # [rows,H_CHUNK] BF16 partner BF16 (value/nKey/nQuery/hidden/key)
VA_PTNR32 = 0x14000  # [rows,H_CHUNK] FP32 partner FP32
VA_GVAL32 = 0x1E000  # [rows,H_CHUNK] FP32 Pass-A grad_value accumulator
VA_GN32 = 0x28000  # [rows,H_CHUNK] FP32 Pass-B grad_nQuery/grad_nKey
VA_GBUF16 = 0x1E000  # [rows,H_CHUNK] BF16 final vector output (aliases VA_GVAL32)
# RMS-x cache ([rows,H_CHUNK] FP32): holds keys (Phase B-key) / hidden (Phase B-query)
# so the 3-pass RMS backward loads each tile from GM ONCE instead of 3x.  Aliases
# grad_value (VA_GVAL32), which is dead after GV_DONE -- Phase-B only, same pattern
# as VA_GBUF16 aliasing VA_GVAL32.  Reuses existing UB, no extra footprint.
VA_XCACHE32 = VA_GVAL32

# (B) HMode-specialized addresses (module-level int pairs).
#     [rows,64] FP32 stride: 0x400 (H=2560, rows=4) / 0x800 (H=1280, rows=8).
#     Listed grouped by HMode (all _2560 first, then all _1280) for readability.

# ---- HMode = 1 (H = 2560) ----
VA_SCORE32_2560 = 0x32000  # [rows,64] FP32 score
VA_GATE32_2560 = 0x32400
VA_GGATE32_2560 = 0x32800
VA_GSCORE32_2560 = 0x32C00
VA_RSQ32_2560 = 0x33000
VA_RINV32_2560 = 0x33400
VA_RMEAN32_2560 = 0x33800
VA_GAMMA16_2560 = 0x33C00  # [1,H_CHUNK] BF16 gamma
VA_GAMMA32_2560 = 0x35000  # [1,H_CHUNK] FP32 gamma
VA_GGAMQ32_2560 = 0x37800  # [1,H_CHUNK] FP32 grad_gamma -- query
VA_GGAMK32_2560 = 0x3A000  # [1,H_CHUNK] FP32 grad_gamma -- key
VA_GGAMQ_ACC32_2560 = 0x3C800  # [1,H_CHUNK] FP32 RMW acc -- query
# GGAMK_ACC aliases GGAMQ32 (query partial dead before key RMW):
VA_GGAMK_ACC32_2560 = 0x37800

# ---- HMode = 0 (H = 1280) ----
VA_SCORE32_1280 = 0x32000  # [rows,64] FP32 score
VA_GATE32_1280 = 0x32800
VA_GGATE32_1280 = 0x33000
VA_GSCORE32_1280 = 0x33800
VA_RSQ32_1280 = 0x34000
VA_RINV32_1280 = 0x34800
VA_RMEAN32_1280 = 0x35000
VA_GAMMA16_1280 = 0x35800  # [1,H_CHUNK] BF16 gamma
VA_GAMMA32_1280 = 0x36200  # [1,H_CHUNK] FP32 gamma
VA_GGAMQ32_1280 = 0x37600  # [1,H_CHUNK] FP32 grad_gamma -- query
VA_GGAMK32_1280 = 0x38A00  # [1,H_CHUNK] FP32 grad_gamma -- key
VA_GGAMQ_ACC32_1280 = 0x39E00  # [1,H_CHUNK] FP32 RMW acc -- query
# GGAMK_ACC aliases GGAMQ32 (query partial dead before key RMW):
VA_GGAMK_ACC32_1280 = 0x37600

# L1 addresses (cube section) — two generic L1 buffers, time-shared
LA_LEFT = 0x00000
LA_RIGHT = 0x10000

L0A_BASE = 0x0000
L0B_BASE = 0x0000
L0C_BASE = 0x0000

# Cross-core handoff event IDs. Two producer phases let Cube overlap with Vector:
#   Phase A: grad_value_ws -> GV_DONE (Cube Nest1-val + Nest2 may start)
#   Phase B: grad_key_ws   -> GK_DONE (Cube Nest1-key + Nest3 may start)
EVENT_GV_DONE = 0   # grad_value_ws ready (Phase A / Pass A)
EVENT_GK_DONE = 2   # grad_key_ws ready   (Phase B / Pass B)


# ═══════════════════════════════════════════════════════════════════
# Layer C: VF helpers — ALL FP32 in/out (NO vf.astype)
# ═══════════════════════════════════════════════════════════════════

@pl.vector_function
def vf_zero_2d(tile, n_rows, n_cols, row_stride):
    """Zero a [n_rows, n_cols] FP32 tile with an explicit row stride."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    zero = vf.full(0.0, preg, dtype=pl.DT_FP32)
    n_regs = (n_cols + LANES_FP32 - 1) // LANES_FP32
    for m in pl.range(0, n_rows):
        for r in pl.range(0, n_regs):
            vf.store_align(tile + m * row_stride + r * LANES_FP32, zero, preg)


@pl.vector_function
def vf_zero_1d(tile, n_cols):
    """Zero a [1, n_cols] FP32 tile."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    zero = vf.full(0.0, preg, dtype=pl.DT_FP32)
    n_regs = (n_cols + LANES_FP32 - 1) // LANES_FP32
    for r in pl.range(0, n_regs):
        vf.store_align(tile + r * LANES_FP32, zero, preg)


@pl.vector_function
def vf_grad_value_accum(
    go_f32,            # [TILE_BS_VEC, H_CHUNK] FP32  grad_output for one head
    gate_f32,          # [TILE_BS_VEC, 64]      FP32  gate (scalar per row, lane 0)
    gv_acc,            # [TILE_BS_VEC, H_CHUNK] FP32 in/out  grad_value accumulator
    n_rows, n_cols, row_stride,
):
    """Step7a^{-1}: gv_acc[r,h] += go[r,h] * gate[r]   (gate broadcast over h)."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    n_regs = (n_cols + LANES_FP32 - 1) // LANES_FP32
    for m in pl.range(0, n_rows):
        gate_reg = vf.load_align(gate_f32, m * 64)
        gate_b = vf.full(gate_reg, preg)
        for r in pl.range(0, n_regs):
            off = m * row_stride + r * LANES_FP32
            g = vf.load_align(go_f32, off)
            a = vf.load_align(gv_acc, off)
            prod = vf.mul(g, gate_b, preg)
            res = vf.add(a, prod, preg)
            vf.store_align(gv_acc + off, res, preg)


@pl.vector_function
def vf_grad_gate_accum(
    go_f32,            # [TILE_BS_VEC, H_CHUNK] FP32  grad_output for this head
    val_f32,           # [TILE_BS_VEC, H_CHUNK] FP32  value (shared)
    gg_acc,            # [TILE_BS_VEC, 64] FP32 in/out  grad_gate accumulator
    n_rows, n_cols, row_stride,
):
    """Step7b^{-1}: gg_acc[r] += Σ_h(go[r,h] * val[r,h]) for this H-chunk."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    n_regs = (n_cols + LANES_FP32 - 1) // LANES_FP32
    for m in pl.range(0, n_rows):
        gg = vf.load_align(gg_acc, m * 64)
        for r in pl.range(0, n_regs):
            off = m * row_stride + r * LANES_FP32
            g = vf.load_align(go_f32, off)
            v = vf.load_align(val_f32, off)
            prod = vf.mul(g, v, preg)
            part = vf.reduce_sum(prod, preg, merge_mode=pl.MergeMode.ZEROING)
            gg = vf.add(gg, part, preg)
        vf.store_align(gg_acc + m * 64, gg, preg)


@pl.vector_function
def vf_gate_bw(
    gg_f32,            # [TILE_BS_VEC, 64] FP32  grad_gate (scalar per row)
    score_f32,         # [TILE_BS_VEC, 64] FP32  score (scalar per row)
    gate_f32,          # [TILE_BS_VEC, 64] FP32  gate g (scalar per row)
    gs_out,            # [TILE_BS_VEC, 64] FP32 write  grad_score
    n_rows,
):
    """Step5^{-1}: signed_sqrt_gate backward.
       grad_score = grad_gate · g(1−g) · mask / (2·√max(|s|,c) + 1e-12)
       mask = (|s| > c)
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    clamp_reg = vf.full(CLAMP_VALUE, preg, dtype=pl.DT_FP32)
    eps_reg = vf.full(GATE_EPS, preg, dtype=pl.DT_FP32)
    two_reg = vf.full(2.0, preg, dtype=pl.DT_FP32)
    zero_reg = vf.full(0.0, preg, dtype=pl.DT_FP32)
    one_reg = vf.full(1.0, preg, dtype=pl.DT_FP32)

    for m in pl.range(0, n_rows):
        gg = vf.load_align(gg_f32, m * 64)
        gg_b = vf.full(gg, preg)
        sc = vf.load_align(score_f32, m * 64)
        sc_b = vf.full(sc, preg)
        gt = vf.load_align(gate_f32, m * 64)
        gt_b = vf.full(gt, preg)

        # sigmoid_grad = g - g*g = g(1-g)
        g_sq = vf.mul(gt_b, gt_b, preg)
        sig_grad = vf.sub(gt_b, g_sq, preg)
        # mask = (|s| > clamp) ? 1 : 0
        abs_s = vf.abs(sc_b, preg)
        mask_gt = vf.gt(abs_s, clamp_reg, preg)
        mask = vf.select(one_reg, zero_reg, mask_gt)
        # sqrt_abs = sqrt(max(|s|, clamp))
        clamped = vf.max(abs_s, clamp_reg, preg)
        sqrt_abs = vf.sqrt(clamped, preg)
        # logits_grad = mask / (2*sqrt_abs + 1e-12)
        denom = vf.mul(two_reg, sqrt_abs, preg)
        denom = vf.add(denom, eps_reg, preg)
        logits_grad = vf.div(mask, denom, preg)
        # grad_score = grad_gate · sigmoid_grad · logits_grad
        result = vf.mul(gg_b, sig_grad, preg)
        result = vf.mul(result, logits_grad, preg)
        vf.store_align(gs_out + m * 64, result, preg)


@pl.vector_function
def vf_scaled_dot(
    gs_f32,            # [TILE_BS_VEC, 64]      FP32  grad_score (scalar per row)
    partner_f32,       # [TILE_BS_VEC, H_CHUNK] FP32  the "other" normed vector
    out_f32,           # [TILE_BS_VEC, H_CHUNK] FP32 write  result
    n_rows, n_cols, row_stride, h_value,
):
    """Step4^{-1}: out[r,h] = grad_score[r] · (1/√H) · partner[r,h].
       Used as: partner=normed_key -> grad_nQuery; partner=normed_query -> grad_nKey.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    n_regs = (n_cols + LANES_FP32 - 1) // LANES_FP32
    # Do not pass 1/sqrt(H) as a VF scalar: scalar constants are lowered
    # through the BF16 scalar path on this target.  H itself is exact, so
    # construct the FP32 scale in vector registers.
    h_reg = vf.full(h_value, preg, dtype=pl.DT_FP32)
    one_reg = vf.full(1.0, preg, dtype=pl.DT_FP32)
    sqrt_h = vf.sqrt(h_reg, preg)
    scale_reg = vf.div(one_reg, sqrt_h, preg)
    for m in pl.range(0, n_rows):
        gs = vf.load_align(gs_f32, m * 64)
        gs_b = vf.full(gs, preg)
        gs_scaled = vf.mul(gs_b, scale_reg, preg)
        for r in pl.range(0, n_regs):
            off = m * row_stride + r * LANES_FP32
            p = vf.load_align(partner_f32, off)
            res = vf.mul(gs_scaled, p, preg)
            vf.store_align(out_f32 + off, res, preg)


# ── rms_norm_backward (3-pass; H-split cross-chunk accumulation) ──

@pl.vector_function
def vf_rmsbw_sq(x_f32, sq_acc, n_rows, n_cols, row_stride):
    """rms_bw Pass 1: sq_acc[r] += Σ_h(x[r,h]^2) for this H-chunk."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    n_regs = (n_cols + LANES_FP32 - 1) // LANES_FP32
    for m in pl.range(0, n_rows):
        sq = vf.load_align(sq_acc, m * 64)
        for r in pl.range(0, n_regs):
            off = m * row_stride + r * LANES_FP32
            x = vf.load_align(x_f32, off)
            xq = vf.mul(x, x, preg)
            part = vf.reduce_sum(xq, preg, merge_mode=pl.MergeMode.ZEROING)
            sq = vf.add(sq, part, preg)
        vf.store_align(sq_acc + m * 64, sq, preg)


@pl.vector_function
def vf_rmsbw_invrms(sq_acc, rms_out, n_rows, h_value):
    """rms_bw: compute inv_rms in FP32 using an exact integer H divisor."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    eps_reg = vf.full(RMS_EPS, preg, dtype=pl.DT_FP32)
    h_reg = vf.full(h_value, preg, dtype=pl.DT_FP32)
    one_reg = vf.full(1.0, preg, dtype=pl.DT_FP32)
    for m in pl.range(0, n_rows):
        sq = vf.load_align(sq_acc, m * 64)
        mean = vf.div(sq, h_reg, preg)
        mean_eps = vf.add(mean, eps_reg, preg)
        rms_val = vf.sqrt(mean_eps, preg)
        inv_rms = vf.div(one_reg, rms_val, preg)
        vf.store_align(rms_out + m * 64, inv_rms, preg)


@pl.vector_function
def vf_rmsnorm_fwd(x_f32, gam_f32, inv_rms_f32, out_f32, n_rows, n_cols, row_stride):
    """Recompute RMSNorm(x, gamma) in FP32 without materializing a cache."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    n_regs = (n_cols + LANES_FP32 - 1) // LANES_FP32
    for m in pl.range(0, n_rows):
        inv_rms = vf.load_align(inv_rms_f32, m * 64)
        inv_rms_b = vf.full(inv_rms, preg)
        for r in pl.range(0, n_regs):
            off = m * row_stride + r * LANES_FP32
            x = vf.load_align(x_f32, off)
            gam = vf.load_align(gam_f32, r * LANES_FP32)
            normed = vf.mul(x, inv_rms_b, preg)
            out = vf.mul(normed, gam, preg)
            vf.store_align(out_f32 + off, out, preg)


@pl.vector_function
def vf_rmsbw_rmean(
    gx_f32,            # [TILE_BS_VEC, H_CHUNK] FP32  grad_xhat (= grad_n)
    x_f32,             # [TILE_BS_VEC, H_CHUNK] FP32  x
    gam_f32,           # [1, H_CHUNK] FP32  gamma
    rms_f32,           # [TILE_BS_VEC, 64] FP32  inv_rms per row
    rmean_acc,         # [TILE_BS_VEC, 64] FP32 in/out
    gg_acc,            # [1, H_CHUNK] FP32 in/out  grad_gamma partial
    n_rows, n_cols, row_stride,
):
    """rms_bw Pass 2: rmean[r] += Σ(grad_n·n); gg_acc[h] += Σ grad_xhat·n.
       n = x · inv_rms ; grad_n = grad_xhat · gamma.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    n_regs = (n_cols + LANES_FP32 - 1) // LANES_FP32
    for m in pl.range(0, n_rows):
        inv_rms = vf.load_align(rms_f32, m * 64)
        inv_rms_b = vf.full(inv_rms, preg)
        rmean = vf.load_align(rmean_acc, m * 64)
        for r in pl.range(0, n_regs):
            off = m * row_stride + r * LANES_FP32
            x = vf.load_align(x_f32, off)
            n_reg = vf.mul(x, inv_rms_b, preg)
            gx = vf.load_align(gx_f32, off)
            gam = vf.load_align(gam_f32, r * LANES_FP32)
            grad_n = vf.mul(gx, gam, preg)
            dn = vf.mul(grad_n, n_reg, preg)
            part = vf.reduce_sum(dn, preg, merge_mode=pl.MergeMode.ZEROING)
            rmean = vf.add(rmean, part, preg)
            dg = vf.mul(gx, n_reg, preg)
            prev_gg = vf.load_align(gg_acc, r * LANES_FP32)
            new_gg = vf.add(prev_gg, dg, preg)
            vf.store_align(gg_acc + r * LANES_FP32, new_gg, preg)
        vf.store_align(rmean_acc + m * 64, rmean, preg)


@pl.vector_function
def vf_rmsbw_rmean_finalize(rmean_acc, n_rows, h_value):
    """Finalize rmean by dividing the FP32 sum by the exact integer H."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    h_reg = vf.full(h_value, preg, dtype=pl.DT_FP32)
    for m in pl.range(0, n_rows):
        val = vf.load_align(rmean_acc, m * 64)
        val = vf.div(val, h_reg, preg)
        vf.store_align(rmean_acc + m * 64, val, preg)


@pl.vector_function
def vf_rmsbw_gradx(
    gx_f32,            # [TILE_BS_VEC, H_CHUNK] FP32  grad_xhat
    x_f32,             # [TILE_BS_VEC, H_CHUNK] FP32  x
    gam_f32,           # [1, H_CHUNK] FP32  gamma
    rms_f32,           # [TILE_BS_VEC, 64] FP32  inv_rms
    rmean_f32,         # [TILE_BS_VEC, 64] FP32  rmean (mean)
    gx_out_f32,        # [TILE_BS_VEC, H_CHUNK] FP32 write  grad_x
    n_rows, n_cols, row_stride,
):
    """rms_bw Pass 3: grad_x = (grad_n − n·rmean) · inv_rms.
       Recomputes n and grad_n from x and gamma.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    n_regs = (n_cols + LANES_FP32 - 1) // LANES_FP32
    for m in pl.range(0, n_rows):
        inv_rms = vf.load_align(rms_f32, m * 64)
        inv_rms_b = vf.full(inv_rms, preg)
        rmean = vf.load_align(rmean_f32, m * 64)
        rmean_b = vf.full(rmean, preg)
        for r in pl.range(0, n_regs):
            off = m * row_stride + r * LANES_FP32
            x = vf.load_align(x_f32, off)
            n_reg = vf.mul(x, inv_rms_b, preg)
            gx = vf.load_align(gx_f32, off)
            gam = vf.load_align(gam_f32, r * LANES_FP32)
            grad_n = vf.mul(gx, gam, preg)
            n_rmean = vf.mul(n_reg, rmean_b, preg)
            diff = vf.sub(grad_n, n_rmean, preg)
            grad_x = vf.mul(diff, inv_rms_b, preg)
            vf.store_align(gx_out_f32 + off, grad_x, preg)


# ═══════════════════════════════════════════════════════════════════
# Layer D: engram_backward_kernel
# ═══════════════════════════════════════════════════════════════════

@pl.jit(auto_mutex=True, tiling_key=EngramTilingKey)
def engram_backward_kernel(
    # Inputs (scores/gates are FP32; other activation inputs are BF16)
    grad_output: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    hidden_states: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    embeddings: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    key_gamma: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    query_gamma: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    scores: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, 64], pl.DT_FP32],
    gates: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, 64], pl.DT_FP32],
    keys: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    value: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    # host-pre-transposed inputs (plain NZ loads, NO is_transpose / layout=ZN)
    emb_t: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],  # [De, BS] BF16
    wv_t: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],  # [H, De] BF16
    wk_t: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],  # [M_H, H, De] BF16
    # BF16 cube operand workspaces (vector produces FP32, casts to BF16 on store)
    grad_value_ws: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],  # [M, H]
    grad_key_ws: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],  # [M, M_H, H]
    gscore_ws: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, 64], pl.DT_FP32],  # [M, M_H, 64] grad_score cache
    # FP32 per-subblock gamma workspace [num_cores*2, M_H, H]
    grad_qgamma_acc: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    grad_kgamma_acc: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    # FP32 grad_emb accumulator [M, De]: Nest1 value + per-head key all atomic-add
    # here in FP32; host casts to BF16 grad_embeddings. MUST be pre-zeroed.
    grad_emb_acc: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],  # [M, De]
    # BF16 outputs (grad_embeddings produced by host from grad_emb_acc)
    grad_hidden_states: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    grad_key_proj_weights: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    grad_value_proj_weights: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
):
    """Single kernel: vector section (produces grad_value_ws/grad_key_ws +
    grad_hidden + grad_γ) then cube section (consumes ws, produces grad_emb +
    grad_W). BF16 activation inputs are cast once at the computation boundary;
    grad_value_ws and grad_key_ws feed legal FP32 Cube paths directly.
    Final output stores are low precision.
    """
    bs = grad_output.shape[0]
    m_h = grad_output.shape[1]  # head count, dynamic (1-16)
    h = grad_output.shape[2]
    de = embeddings.shape[1]

    # Bind vector tile sizes to compile-time constants via TilingKey. HMode/BSMode
    # are ConstInt -> `if pl.constexpr` parses only the taken branch, so these names
    # are usable in TileType.shape and pl.range steps.
    if pl.constexpr(HMode == 1):  # H = 2560
        tile_bs_vec = TILE_BS_VEC_2560  # 4
        tile_bs_vec_a = TILE_BS_VEC_A_2560  # 8
        h_chunk_size = H_CHUNK_2560  # 2560
    else:  # H = 1280
        tile_bs_vec = TILE_BS_VEC_1280  # 8
        tile_bs_vec_a = TILE_BS_VEC_A_1280  # 16
        h_chunk_size = H_CHUNK_1280  # 1280

    if pl.constexpr(BSMode == 0):  # BS >= 8192
        v_tile = V_TILE_BS0  # 128
    elif pl.constexpr(BSMode == 1):  # BS >= 4096
        v_tile = V_TILE_BS1  # 64
    elif pl.constexpr(BSMode == 2):  # BS >= 2048
        v_tile = V_TILE_BS2  # 32
    else:  # BSMode == 3: small BS
        v_tile = V_TILE_BS3  # 16
    # NOTE: UB addresses for [rows,64]/[1,H_CHUNK] tiles are NOT bound here (parser
    # can't see names from this `if` block); they use the HMode arithmetic-select in
    # make_tile_group below. TileType.shape DOES see these names (parser const_env).
    n_h_chunks = (h + h_chunk_size - 1) // h_chunk_size  # exactly one full-H tile
    cube_k = 128  # cube K tile: BF16 [128,128]=32KB fits L0A/L0B double-buffer
    cube_mn = 128  # cube M/N output tile
    n_kh = (h + cube_k - 1) // cube_k  # K=H slice count (Nest1 reduces over H)
    n_kbs = (bs + cube_k - 1) // cube_k  # K=BS slice count (Nest2/3 reduce over BS)
    n_de = (de + cube_mn - 1) // cube_mn  # De output tile count
    n_h = (h + cube_mn - 1) // cube_mn  # H output tile count (Nest2/3 N=H)
    n_bs = (bs + cube_mn - 1) // cube_mn  # cube BS output tile count (Nest1 M=BS)
    n_bs_v = (bs + v_tile - 1) // v_tile  # vector worker tile count (>=64 fills workers)
    num_cores = pl.get_block_num()
    # In section_vector, get_block_idx() is the AI Core index (0..num_cores-1),
    # shared by that core's 2 AIV subblocks. Combine into a global worker id so
    # the 64 workers split BS-tiles and write disjoint gamma-workspace slots.
    # NOTE: get_subblock_num() was removed from pypto; an AIV block has a fixed
    # 2 subblocks, so num_subcores is a compile-time constant (no API call needed).
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    num_subcores = 2                       # AIV block has 2 subblocks (sub_idx 0/1)
    sub_id = pl.get_subblock_idx()         # 0 or 1 within this AI Core
    total_subblocks = num_cores * num_subcores
    worker_id = core_id * num_subcores + sub_id   # global vector-subblock index
    iters_per_core = (n_bs + num_cores - 1) // num_cores      # per-core BS-tile iterations (cube)
    iters_per_subblock = (n_bs_v + total_subblocks - 1) // total_subblocks  # per-worker

    # ═══════════════════════════════════════════════════════════════
    # VECTOR SECTION (runs first; produces grad_value_ws / grad_key_ws)
    # ═══════════════════════════════════════════════════════════════
    tt_mv16 = pl.TileType(shape=[tile_bs_vec, h_chunk_size], dtype=pl.DT_BF16,
                           target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    tt_mv32 = pl.TileType(shape=[tile_bs_vec, h_chunk_size], dtype=pl.DT_FP32,
                           target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    tt_score32 = pl.TileType(shape=[tile_bs_vec, 64], dtype=pl.DT_FP32,
                           target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    tt_gamma16 = pl.TileType(shape=[1, h_chunk_size], dtype=pl.DT_BF16,
                           target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    tt_gamma32 = pl.TileType(shape=[1, h_chunk_size], dtype=pl.DT_FP32,
                           target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])

    # Tile capacity is H_CHUNK (1280/2560), but every vector op uses the actual H
    # as its valid width (set_validshape). H <= H_CHUNK, so H is never split.

    gout16_grp = pl.make_tile_group(type=tt_mv16, addrs=VA_GOUT16, mutex_ids=[0])
    gout32_grp = pl.make_tile_group(type=tt_mv32, addrs=VA_GOUT32, mutex_ids=[1])
    ptnr16_grp = pl.make_tile_group(type=tt_mv16, addrs=VA_PTNR16, mutex_ids=[2])
    ptnr32_grp = pl.make_tile_group(type=tt_mv32, addrs=VA_PTNR32, mutex_ids=[3])
    gval32_grp = pl.make_tile_group(type=tt_mv32, addrs=VA_GVAL32, mutex_ids=[4])
    gn32_grp = pl.make_tile_group(type=tt_mv32, addrs=VA_GN32, mutex_ids=[17])
    gbuf16_grp = pl.make_tile_group(type=tt_mv16, addrs=VA_GBUF16, mutex_ids=[4])
    # HMode-dependent addresses via arithmetic select (folds to _2560/_1280 per key).
    score32_grp = pl.make_tile_group(type=tt_score32, addrs=HMode * VA_SCORE32_2560 + (1 - HMode) * VA_SCORE32_1280, mutex_ids=[7])
    gate32_grp = pl.make_tile_group(type=tt_score32, addrs=HMode * VA_GATE32_2560 + (1 - HMode) * VA_GATE32_1280, mutex_ids=[8])
    ggate_grp = pl.make_tile_group(type=tt_score32, addrs=HMode * VA_GGATE32_2560 + (1 - HMode) * VA_GGATE32_1280, mutex_ids=[9])
    gscore_grp = pl.make_tile_group(type=tt_score32, addrs=HMode * VA_GSCORE32_2560 + (1 - HMode) * VA_GSCORE32_1280, mutex_ids=[10])
    rsq_grp = pl.make_tile_group(type=tt_score32, addrs=HMode * VA_RSQ32_2560 + (1 - HMode) * VA_RSQ32_1280, mutex_ids=[11])
    rinv_grp = pl.make_tile_group(type=tt_score32, addrs=HMode * VA_RINV32_2560 + (1 - HMode) * VA_RINV32_1280, mutex_ids=[12])
    rmean_grp = pl.make_tile_group(type=tt_score32, addrs=HMode * VA_RMEAN32_2560 + (1 - HMode) * VA_RMEAN32_1280, mutex_ids=[13])
    gamma16_grp = pl.make_tile_group(type=tt_gamma16, addrs=HMode * VA_GAMMA16_2560 + (1 - HMode) * VA_GAMMA16_1280, mutex_ids=[14])
    gamma32_grp = pl.make_tile_group(type=tt_gamma32, addrs=HMode * VA_GAMMA32_2560 + (1 - HMode) * VA_GAMMA32_1280, mutex_ids=[15])
    ggamq_grp = pl.make_tile_group(type=tt_gamma32, addrs=HMode * VA_GGAMQ32_2560 + (1 - HMode) * VA_GGAMQ32_1280, mutex_ids=[16])
    ggamk_grp = pl.make_tile_group(type=tt_gamma32, addrs=HMode * VA_GGAMK32_2560 + (1 - HMode) * VA_GGAMK32_1280, mutex_ids=[18])
    ggamq_acc_grp = pl.make_tile_group(type=tt_gamma32, addrs=HMode * VA_GGAMQ_ACC32_2560 + (1 - HMode) * VA_GGAMQ_ACC32_1280, mutex_ids=[19])
    ggamk_acc_grp = pl.make_tile_group(type=tt_gamma32, addrs=HMode * VA_GGAMK_ACC32_2560 + (1 - HMode) * VA_GGAMK_ACC32_1280, mutex_ids=[20])
    # RMS-x cache: keys (B-key) / hidden (B-query), loaded once in RMS Pass1,
    # reused in Pass2/Pass3. Aliases grad_value (dead after GV_DONE).
    xcache_grp = pl.make_tile_group(type=tt_mv32, addrs=VA_XCACHE32, mutex_ids=[21])

    # Pass A dedicated large tiles ([tile_bs_vec_a, h_chunk_size], 2x B-phase rows).
    # Pass A reuses B-phase UB space (all B tiles dead), isolated by GV_DONE.
    tt_pa16 = pl.TileType(shape=[tile_bs_vec_a, h_chunk_size], dtype=pl.DT_BF16,
                          target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    tt_pa32 = pl.TileType(shape=[tile_bs_vec_a, h_chunk_size], dtype=pl.DT_FP32,
                          target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    tt_pa_sc = pl.TileType(shape=[tile_bs_vec_a, 64], dtype=pl.DT_FP32,
                           target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    pa_go16_grp = pl.make_tile_group(type=tt_pa16, addrs=0x00000, mutex_ids=[22])
    pa_go32_grp = pl.make_tile_group(type=tt_pa32, addrs=0x0A000, mutex_ids=[23])
    pa_gv32_grp = pl.make_tile_group(type=tt_pa32, addrs=0x1E000, mutex_ids=[24])
    pa_gv16_grp = pl.make_tile_group(type=tt_pa16, addrs=0x32000, mutex_ids=[25])
    pa_gate32_grp = pl.make_tile_group(type=tt_pa_sc, addrs=0x3C000, mutex_ids=[26])

    with pl.section_vector():
        for tile_idx in pl.range(0, iters_per_subblock):
            bsi = worker_id + tile_idx * total_subblocks
            if bsi < n_bs_v:
                row_off = bsi * v_tile
                bs_tile_rows = pl.min(v_tile, bs - row_off)
                for bs_start in pl.range(0, bs_tile_rows, tile_bs_vec_a):
                    valid_bsv = pl.min(tile_bs_vec_a, bs_tile_rows - bs_start)
                    bs_off = row_off + bs_start
                    # Pass A: Step7a -> grad_value_ws = Σ_m(go·gate)
                    for h_chunk in pl.range(0, n_h_chunks):
                        h_off = h_chunk * h
                        gv = pa_gv32_grp.current()
                        pl.set_validshape(gv, [valid_bsv, h])
                        vf_zero_2d(gv, valid_bsv, h, h_chunk_size)
                        for m_head in pl.range(0, m_h):
                            pa_go16 = pa_go16_grp.current()
                            pl.set_validshape(pa_go16, [valid_bsv, h])
                            pl.load(pa_go16, grad_output,
                                    [bs_off, m_head, h_off], order=[0, 2])
                            pa_go32 = pa_go32_grp.current()
                            pl.set_validshape(pa_go32, [valid_bsv, h])
                            pl.cast(pa_go32, pa_go16, mode=pl.RoundMode.CAST_NONE)
                            pa_gate32 = pa_gate32_grp.current()
                            pl.set_validshape(pa_gate32, [valid_bsv, 64])
                            pl.load(pa_gate32, gates, [bs_off, m_head, 0], order=[0, 2])
                            vf_grad_value_accum(pa_go32, pa_gate32, gv, valid_bsv, h, h_chunk_size)
                        pa_gv16 = pa_gv16_grp.current()
                        pl.set_validshape(pa_gv16, [valid_bsv, h])
                        pl.cast(pa_gv16, gv, mode=pl.RoundMode.CAST_ROUND)  # FP32 -> BF16
                        pl.store(grad_value_ws, pa_gv16, [bs_off, h_off])
                    # Drain Pass-A stores before Pass-B reuses the GV32 address.

        # ===== GV_DONE: grad_value_ws ready (Phase A). Cube Nest1-val + Nest2 may
        # start and overlap with Vector Phase B below. =====
        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=EVENT_GV_DONE,
                                 sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

        # ===== Phase B-key: grad_key_ws / grad_γ_k (key path first so GK_DONE
        # fires early and Nest1-key/Nest3 overlap with Phase B-query). =====
        for tile_idx in pl.range(0, iters_per_subblock):
            bsi = worker_id + tile_idx * total_subblocks
            if bsi < n_bs_v:
                row_off = bsi * v_tile
                bs_tile_rows = pl.min(v_tile, bs - row_off)
                for bs_start in pl.range(0, bs_tile_rows, tile_bs_vec):
                    valid_bsv = pl.min(tile_bs_vec, bs_tile_rows - bs_start)
                    bs_off = row_off + bs_start
                    pl.system.bar_all()
                    for m_head in pl.range(0, m_h):
                        # ── Step7b: grad_gate = Σ_h(go·val) over the full-H tile ──
                        gg = ggate_grp.current()
                        pl.set_validshape(gg, [valid_bsv, 64])
                        vf_zero_2d(gg, valid_bsv, 64, 64)
                        for h_chunk in pl.range(0, n_h_chunks):
                            h_off = h_chunk * h
                            gout16 = gout16_grp.current()
                            pl.set_validshape(gout16, [valid_bsv, h])
                            pl.load(gout16, grad_output,
                                    [bs_off, m_head, h_off], order=[0, 2])
                            gout32 = gout32_grp.current()
                            pl.set_validshape(gout32, [valid_bsv, h])
                            pl.cast(gout32, gout16, mode=pl.RoundMode.CAST_NONE)
                            ptnr16 = ptnr16_grp.current()
                            pl.set_validshape(ptnr16, [valid_bsv, h])
                            pl.load(ptnr16, value, [bs_off, h_off])
                            ptnr32 = ptnr32_grp.current()
                            pl.set_validshape(ptnr32, [valid_bsv, h])
                            pl.cast(ptnr32, ptnr16, mode=pl.RoundMode.CAST_NONE)
                            vf_grad_gate_accum(gout32, ptnr32, gg, valid_bsv, h, h_chunk_size)

                        # ── Step5: gate_bw(gg, score, gate) -> gs ──
                        score32 = score32_grp.current()                 # score FP32
                        pl.set_validshape(score32, [valid_bsv, 64])
                        pl.load(score32, scores, [bs_off, m_head, 0], order=[0, 2])
                        gate32 = gate32_grp.current()                 # gate FP32
                        pl.set_validshape(gate32, [valid_bsv, 64])
                        pl.load(gate32, gates, [bs_off, m_head, 0], order=[0, 2])
                        gs = gscore_grp.current()                     # grad_score output
                        pl.set_validshape(gs, [valid_bsv, 64])
                        vf_gate_bw(gg, score32, gate32, gs, valid_bsv)
                        pl.store(gscore_ws, gs, [bs_off, m_head, 0], order=[0, 2])  # cache gs for Phase B-query
                        # Step4 + Step2 (key): grad_nKey -> rms_bw(keys, γ_k).
                        # Cache keys (RMS-x) ONCE; reused in Pass2/Pass3 (3-pass pattern).
                        sq = rsq_grp.current()
                        pl.set_validshape(sq, [valid_bsv, 64])
                        vf_zero_2d(sq, valid_bsv, 64, 64)
                        xcache = xcache_grp.current()
                        pl.set_validshape(xcache, [valid_bsv, h])
                        for h_chunk in pl.range(0, n_h_chunks):
                            h_off = h_chunk * h
                            ptnr16 = ptnr16_grp.current()
                            pl.set_validshape(ptnr16, [valid_bsv, h])
                            pl.load(ptnr16, keys,
                                    [bs_off, m_head, h_off], order=[0, 2])
                            pl.cast(xcache, ptnr16, mode=pl.RoundMode.CAST_NONE)
                            vf_rmsbw_sq(xcache, sq, valid_bsv, h, h_chunk_size)
                        rms = rinv_grp.current()
                        pl.set_validshape(rms, [valid_bsv, 64])
                        vf_rmsbw_invrms(sq, rms, valid_bsv, h)
                        # Recompute hidden inv_rms (normed_query) for the key backward.
                        # norm32 (hidden FP32) stays in UB; Pass2 reuses it (n_h_chunks==1).
                        partner_sq = rsq_grp.current()
                        pl.set_validshape(partner_sq, [valid_bsv, 64])
                        vf_zero_2d(partner_sq, valid_bsv, 64, 64)
                        for norm_h_chunk in pl.range(0, n_h_chunks):
                            norm_h_off = norm_h_chunk * h
                            norm16 = ptnr16_grp.current()
                            pl.set_validshape(norm16, [valid_bsv, h])
                            pl.load(norm16, hidden_states,
                                    [bs_off, m_head, norm_h_off], order=[0, 2])
                            norm32 = ptnr32_grp.current()
                            pl.set_validshape(norm32, [valid_bsv, h])
                            pl.cast(norm32, norm16, mode=pl.RoundMode.CAST_NONE)
                            vf_rmsbw_sq(norm32, partner_sq, valid_bsv, h, h_chunk_size)
                        vf_rmsbw_invrms(partner_sq, partner_sq, valid_bsv, h)

                        rmean = rmean_grp.current()
                        pl.set_validshape(rmean, [valid_bsv, 64])
                        vf_zero_2d(rmean, valid_bsv, 64, 64)
                        for h_chunk in pl.range(0, n_h_chunks):
                            h_off = h_chunk * h
                            # grad_nKey = gs·(1/√H)·RMSNorm(hidden); reuse hidden FP32 tile from Pass1b.
                            ptnr32 = ptnr32_grp.current()
                            gamma16 = gamma16_grp.current()
                            pl.set_validshape(gamma16, [1, h])
                            pl.load(gamma16, query_gamma, [m_head, h_off], order=[0])
                            gamma32 = gamma32_grp.current()
                            pl.set_validshape(gamma32, [1, h])
                            pl.cast(gamma32, gamma16, mode=pl.RoundMode.CAST_NONE)
                            vf_rmsnorm_fwd(ptnr32, gamma32, partner_sq, ptnr32,
                                           valid_bsv, h, h_chunk_size)
                            gx32 = gn32_grp.current()
                            pl.set_validshape(gx32, [valid_bsv, h])
                            vf_scaled_dot(gs, ptnr32, gx32, valid_bsv, h,
                                           h_chunk_size, h)
                            x32 = xcache_grp.current()           # reuse cached keys
                            pl.set_validshape(x32, [valid_bsv, h])
                            pl.load(gamma16, key_gamma, [m_head, h_off], order=[0])
                            pl.cast(gamma32, gamma16, mode=pl.RoundMode.CAST_NONE)
                            ggam = ggamk_grp.current()
                            pl.set_validshape(ggam, [1, h])
                            vf_zero_1d(ggam, h)
                            vf_rmsbw_rmean(gx32, x32, gamma32, rms, rmean, ggam,
                                           valid_bsv, h, h_chunk_size)
                            # RMW this chunk's grad_γ_k in FP32 workspace (per-subblock).
                            ggamk_acc = ggamk_acc_grp.current()
                            pl.set_validshape(ggamk_acc, [1, h])
                            pl.load(ggamk_acc, grad_kgamma_acc,
                                    [worker_id, m_head, h_off], order=[1, 2])
                            pl.add(ggamk_acc, ggamk_acc, ggam)
                            pl.store(grad_kgamma_acc, ggamk_acc,
                                     [worker_id, m_head, h_off], order=[1, 2])
                        vf_rmsbw_rmean_finalize(rmean, valid_bsv, h)
                        # Pass 3: grad_key_m -> store GM.  Reuse gx32 (grad_nKey
                        # = gs·RMSNorm(hidden)) still live in gn32 from Pass 2.
                        for h_chunk in pl.range(0, n_h_chunks):
                            h_off = h_chunk * h
                            gx32 = gn32_grp.current()
                            pl.set_validshape(gx32, [valid_bsv, h])
                            x32 = xcache_grp.current()           # reuse cached keys
                            pl.set_validshape(x32, [valid_bsv, h])
                            gamma32 = gamma32_grp.current()      # reuse key_gamma FP32 from Pass2
                            gk32 = gout32_grp.current()
                            pl.set_validshape(gk32, [valid_bsv, h])
                            vf_rmsbw_gradx(gx32, x32, gamma32, rms, rmean, gk32,
                                           valid_bsv, h, h_chunk_size)
                            gk16 = gbuf16_grp.current()         # reuse OUT16(=GV32), free in Pass B-key
                            pl.set_validshape(gk16, [valid_bsv, h])
                            pl.cast(gk16, gk32, mode=pl.RoundMode.CAST_ROUND)  # FP32 -> BF16
                            pl.store(grad_key_ws, gk16,
                                     [bs_off, m_head, h_off], order=[0, 2])

        # ===== GK_DONE: Phase B-key (grad_key_ws) complete. sync_all(MIX) is
        # REQUIRED: both AIV subblocks split the work, so cube must wait for ALL
        # subblocks (a per-block signal would miss the other subblock's tiles). =====
        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=EVENT_GK_DONE,
                                 sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

        # ===== Phase B-query: grad_hidden / grad_γ_q (overlaps Cube Nest1-key/Nest3) =====
        # gs (Step7b/Step5) is recomputed here; it does not persist across loops.
        for tile_idx in pl.range(0, iters_per_subblock):
            bsi = worker_id + tile_idx * total_subblocks
            if bsi < n_bs_v:
                row_off = bsi * v_tile
                bs_tile_rows = pl.min(v_tile, bs - row_off)
                for bs_start in pl.range(0, bs_tile_rows, tile_bs_vec):
                    valid_bsv = pl.min(tile_bs_vec, bs_tile_rows - bs_start)
                    bs_off = row_off + bs_start
                    # OUT16 aliases VA_GVAL32 (Phase A); Phase A is globally done.
                    pl.system.bar_all()

                    # Pass B-query: per-head Step4 -> Step3 (rms_q). gs loaded from
                    # gscore_ws (computed once in Phase B-key, skips Step7b+Step5 here).
                    for m_head in pl.range(0, m_h):
                        gs = gscore_grp.current()
                        pl.set_validshape(gs, [valid_bsv, 64])
                        pl.load(gs, gscore_ws, [bs_off, m_head, 0], order=[0, 2])
                        # Step4 + Step3 (query): grad_nQuery -> rms_bw(hidden, γ_q).
                        # Pass 1: sq over hidden_states
                        sq = rsq_grp.current()
                        pl.set_validshape(sq, [valid_bsv, 64])
                        vf_zero_2d(sq, valid_bsv, 64, 64)
                        # Cache hidden_states (the RMS-x of the query backward) ONCE;
                        # reuse in Pass2/Pass3 instead of reloading 3x.
                        xcache = xcache_grp.current()
                        pl.set_validshape(xcache, [valid_bsv, h])
                        for h_chunk in pl.range(0, n_h_chunks):
                            h_off = h_chunk * h
                            ptnr16 = ptnr16_grp.current()
                            pl.set_validshape(ptnr16, [valid_bsv, h])
                            pl.load(ptnr16, hidden_states,
                                    [bs_off, m_head, h_off], order=[0, 2])
                            pl.cast(xcache, ptnr16, mode=pl.RoundMode.CAST_NONE)
                            vf_rmsbw_sq(xcache, sq, valid_bsv, h, h_chunk_size)
                        rms = rinv_grp.current()
                        pl.set_validshape(rms, [valid_bsv, 64])
                        vf_rmsbw_invrms(sq, rms, valid_bsv, h)
                        # Recompute keys inv_rms (normed_key) for the query backward.
                        # norm32 (keys FP32) stays in UB; Pass2 reuses it (n_h_chunks==1).
                        partner_sq = rsq_grp.current()
                        pl.set_validshape(partner_sq, [valid_bsv, 64])
                        vf_zero_2d(partner_sq, valid_bsv, 64, 64)
                        for norm_h_chunk in pl.range(0, n_h_chunks):
                            norm_h_off = norm_h_chunk * h
                            norm16 = ptnr16_grp.current()
                            pl.set_validshape(norm16, [valid_bsv, h])
                            pl.load(norm16, keys,
                                    [bs_off, m_head, norm_h_off], order=[0, 2])
                            norm32 = ptnr32_grp.current()
                            pl.set_validshape(norm32, [valid_bsv, h])
                            pl.cast(norm32, norm16, mode=pl.RoundMode.CAST_NONE)
                            vf_rmsbw_sq(norm32, partner_sq, valid_bsv, h, h_chunk_size)
                        vf_rmsbw_invrms(partner_sq, partner_sq, valid_bsv, h)

                        # Pass 2: rmean over H + grad_γ_q
                        rmean = rmean_grp.current()
                        pl.set_validshape(rmean, [valid_bsv, 64])
                        vf_zero_2d(rmean, valid_bsv, 64, 64)
                        for h_chunk in pl.range(0, n_h_chunks):
                            h_off = h_chunk * h
                            # grad_nQuery = gs·(1/√H)·RMSNorm(key); reuse keys FP32 tile from Pass1b.
                            ptnr32 = ptnr32_grp.current()
                            gamma16 = gamma16_grp.current()
                            pl.set_validshape(gamma16, [1, h])
                            pl.load(gamma16, key_gamma, [m_head, h_off], order=[0])
                            gamma32 = gamma32_grp.current()
                            pl.set_validshape(gamma32, [1, h])
                            pl.cast(gamma32, gamma16, mode=pl.RoundMode.CAST_NONE)
                            vf_rmsnorm_fwd(ptnr32, gamma32, partner_sq, ptnr32,
                                           valid_bsv, h, h_chunk_size)
                            gx32 = gn32_grp.current()
                            pl.set_validshape(gx32, [valid_bsv, h])
                            vf_scaled_dot(gs, ptnr32, gx32, valid_bsv, h,
                                           h_chunk_size, h)
                            # reload hidden chunk
                            x32 = xcache_grp.current()           # reuse cached hidden
                            pl.set_validshape(x32, [valid_bsv, h])
                            pl.load(gamma16, query_gamma, [m_head, h_off], order=[0])
                            pl.cast(gamma32, gamma16, mode=pl.RoundMode.CAST_NONE)
                            ggam = ggamq_grp.current()
                            pl.set_validshape(ggam, [1, h])
                            vf_zero_1d(ggam, h)
                            vf_rmsbw_rmean(gx32, x32, gamma32, rms, rmean, ggam,
                                           valid_bsv, h, h_chunk_size)
                            # RMW this chunk's grad_γ_q in FP32 workspace (per-subblock).
                            ggamq_acc = ggamq_acc_grp.current()
                            pl.set_validshape(ggamq_acc, [1, h])
                            pl.load(ggamq_acc, grad_qgamma_acc,
                                    [worker_id, m_head, h_off], order=[1, 2])
                            pl.add(ggamq_acc, ggamq_acc, ggam)
                            pl.store(grad_qgamma_acc, ggamq_acc,
                                     [worker_id, m_head, h_off], order=[1, 2])
                        vf_rmsbw_rmean_finalize(rmean, valid_bsv, h)
                        # Pass 3: grad_hidden_m -> store GM. gx32 (grad_nQuery) and
                        # partner_sq (keys inv_rms) are still live from Pass2/Pass1b,
                        # so Pass3 reuses them (n_h_chunks==1 -> H <= H_CHUNK).
                        for h_chunk in pl.range(0, n_h_chunks):
                            h_off = h_chunk * h
                            gx32 = gn32_grp.current()
                            pl.set_validshape(gx32, [valid_bsv, h])
                            x32 = xcache_grp.current()           # reuse cached hidden
                            pl.set_validshape(x32, [valid_bsv, h])
                            gamma32 = gamma32_grp.current()      # reuse query_gamma FP32 from Pass2
                            gh32 = gout32_grp.current()
                            pl.set_validshape(gh32, [valid_bsv, h])
                            vf_rmsbw_gradx(gx32, x32, gamma32, rms, rmean, gh32,
                                           valid_bsv, h, h_chunk_size)
                            gh16 = gbuf16_grp.current()
                            pl.set_validshape(gh16, [valid_bsv, h])
                            pl.cast(gh16, gh32, mode=pl.RoundMode.CAST_ROUND)
                            pl.store(grad_hidden_states, gh16,
                                     [bs_off, m_head, h_off], order=[0, 2])

                        pl.system.bar_all()

    # ═══════════════════════════════════════════════════════════════
    # CUBE SECTION (consumes grad_value_ws / grad_key_ws and transposed inputs)
    # Cube tile: BF16 [128,128,128] (A/B/L0A/L0B BF16, L0C accumulator FP32).
    tt_a_l1 = pl.TileType(shape=[cube_mn, cube_k], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                            valid_shape=[-1, -1], compact=1)
    tt_b_l1 = pl.TileType(shape=[cube_k, cube_mn], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                            valid_shape=[-1, -1], compact=1)
    tt_a_l0 = pl.TileType(shape=[cube_mn, cube_k], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Left, layout=pl.NZ,
                            valid_shape=[-1, -1], compact=1)
    tt_b_l0 = pl.TileType(shape=[cube_k, cube_mn], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Right, layout=pl.ZN,
                            valid_shape=[-1, -1], compact=1)
    tt_acc = pl.TileType(shape=[cube_mn, cube_mn], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Acc, fractal=1024,
                            layout=pl.NZ, valid_shape=[-1, -1], compact=1)

    # L1/L0 double-buffered (overlap GM->L1 load + L1->L0 move with matmul);
    # acc single-buffered (matmul_acc K-chain is a true dependency).
    a_l1_grp = pl.make_tile_group(type=tt_a_l1, addrs=[0x20000, 0x28000], mutex_ids=[10, 11])
    b_l1_grp = pl.make_tile_group(type=tt_b_l1, addrs=[0x30000, 0x38000], mutex_ids=[12, 13])
    a_l0_grp = pl.make_tile_group(type=tt_a_l0, addrs=[0x0000, 0x8000], mutex_ids=[14, 15])
    b_l0_grp = pl.make_tile_group(type=tt_b_l0, addrs=[0x0000, 0x8000], mutex_ids=[16, 17])
    acc_grp = pl.make_tile_group(type=tt_acc, addrs=0x0000, mutex_ids=[18])

    with pl.section_cube():
        # wait GV_DONE: grad_value_ws ready. Nest1-val + Nest2 may now run,
        # overlapping Vector Phase B.
        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=EVENT_GV_DONE,
                                 sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.system.set_mm_layout_transform(enabled=True)

        # Nest 1 (value): grad_value_ws @ wv_t -> atomic-add into grad_emb_acc.
        # M-strided partitioning -> each output address has one cube owner ->
        # atomicAdd is a local FP32 accumulate (no cross-core race).
        for tile_idx in pl.range(0, iters_per_core):
            bsi = core_id + tile_idx * num_cores
            for ni in pl.range(0, n_de):
                if bsi < n_bs:
                    row_off = bsi * cube_mn
                    valid_bs = pl.min(cube_mn, bs - row_off)
                    col_off = ni * cube_mn
                    valid_n = pl.min(cube_mn, de - col_off)
                    ac = acc_grp.current()
                    pl.set_validshape(ac, [valid_bs, valid_n])
                    for k_idx in pl.range(0, n_kh):
                        k_off = k_idx * cube_k
                        valid_k = pl.min(cube_k, h - k_off)
                        left = a_l1_grp.next()
                        right = b_l1_grp.next()
                        pl.set_validshape(left, [valid_bs, valid_k])
                        pl.set_validshape(right, [valid_k, valid_n])
                        pl.load(left, grad_value_ws, [row_off, k_off])
                        pl.load(right, wv_t, [k_off, col_off])
                        al = a_l0_grp.next()
                        br = b_l0_grp.next()
                        pl.set_validshape(al, [valid_bs, valid_k])
                        pl.set_validshape(br, [valid_k, valid_n])
                        pl.move(al, left)
                        pl.move(br, right)
                        if k_idx == 0:
                            pl.matmul(ac, al, br)
                        else:
                            pl.matmul_acc(ac, ac, al, br)
                    # Value contribution: atomic-add into grad_emb_acc (FP32 sum with
                    # per-head key stores below).
                    pl.store(grad_emb_acc, ac, [row_off, col_off],
                             atomic=pl.AtomicType.AtomicAdd)

        # Nest 2: grad_W_v[De, H] = emb_t @ grad_value_ws. Only depends on
        # grad_value_ws -> runs before wait(GK_DONE), overlapping Vector Phase B.
        total_wv = n_de * n_h
        iters_wv = (total_wv + num_cores - 1) // num_cores
        for wv_idx in pl.range(0, iters_wv):
            flat = core_id + wv_idx * num_cores
            if flat < total_wv:
                de_idx = flat // n_h
                h_idx = flat % n_h
                de_off = de_idx * cube_mn
                h_off = h_idx * cube_mn
                valid_de = pl.min(cube_mn, de - de_off)
                valid_h = pl.min(cube_mn, h - h_off)
                ac = acc_grp.current()
                pl.set_validshape(ac, [valid_de, valid_h])
                for k_idx in pl.range(0, n_kbs):
                    k_off = k_idx * cube_k
                    valid_k = pl.min(cube_k, bs - k_off)
                    left = a_l1_grp.next()
                    right = b_l1_grp.next()
                    pl.set_validshape(left, [valid_de, valid_k])
                    pl.set_validshape(right, [valid_k, valid_h])
                    pl.load(left, emb_t, [de_off, k_off])
                    pl.load(right, grad_value_ws, [k_off, h_off])
                    al = a_l0_grp.next()
                    br = b_l0_grp.next()
                    pl.set_validshape(al, [valid_de, valid_k])
                    pl.set_validshape(br, [valid_k, valid_h])
                    pl.move(al, left)
                    pl.move(br, right)
                    if k_idx == 0:
                        pl.matmul(ac, al, br)
                    else:
                        pl.matmul_acc(ac, ac, al, br)
                pl.store(grad_value_proj_weights, ac, [de_off, h_off])

        # wait GK_DONE: grad_key_ws ready (both AIV subblocks split its production).
        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=EVENT_GK_DONE,
                                 sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

        # Nest 1 (per-head key): grad_key_ws[m] @ wk_t[m] -> atomic-add into the
        # same grad_emb_acc slot as the value store above.
        for m_head in pl.range(0, m_h):
            for tile_idx in pl.range(0, iters_per_core):
                bsi = core_id + tile_idx * num_cores
                for n_idx in pl.range(0, n_de):
                    if bsi < n_bs:
                        row_off = bsi * cube_mn
                        valid_bs = pl.min(cube_mn, bs - row_off)
                        col_off = n_idx * cube_mn
                        valid_n = pl.min(cube_mn, de - col_off)
                        ac = acc_grp.current()
                        pl.set_validshape(ac, [valid_bs, valid_n])
                        for k_idx in pl.range(0, n_kh):
                            k_off = k_idx * cube_k
                            valid_k = pl.min(cube_k, h - k_off)
                            left = a_l1_grp.next()
                            right = b_l1_grp.next()
                            pl.set_validshape(left, [valid_bs, valid_k])
                            pl.set_validshape(right, [valid_k, valid_n])
                            pl.load(left, grad_key_ws,
                                    [row_off, m_head, k_off], order=[0, 2])
                            pl.load(right, wk_t,
                                    [m_head, k_off, col_off], order=[1, 2])
                            al = a_l0_grp.next()
                            br = b_l0_grp.next()
                            pl.set_validshape(al, [valid_bs, valid_k])
                            pl.set_validshape(br, [valid_k, valid_n])
                            pl.move(al, left)
                            pl.move(br, right)
                            if k_idx == 0:
                                pl.matmul(ac, al, br)
                            else:
                                pl.matmul_acc(ac, ac, al, br)
                        # Per-head key contribution: atomic-add into the same grad_emb_acc slot.
                        pl.store(grad_emb_acc, ac, [row_off, col_off],
                                 atomic=pl.AtomicType.AtomicAdd)

        # Nest 3: grad_W_k[m][De, H] = emb_t @ grad_key_ws[m].
        total_wk = m_h * n_de * n_h
        iters_wk = (total_wk + num_cores - 1) // num_cores
        for wk_idx in pl.range(0, iters_wk):
            flat = core_id + wk_idx * num_cores
            if flat < total_wk:
                m_head = flat // (n_de * n_h)
                rem = flat % (n_de * n_h)
                de_idx = rem // n_h
                h_idx = rem % n_h
                de_off = de_idx * cube_mn
                h_off = h_idx * cube_mn
                valid_de = pl.min(cube_mn, de - de_off)
                valid_h = pl.min(cube_mn, h - h_off)
                ac = acc_grp.current()
                pl.set_validshape(ac, [valid_de, valid_h])
                for k_idx in pl.range(0, n_kbs):
                    k_off = k_idx * cube_k
                    valid_k = pl.min(cube_k, bs - k_off)
                    left = a_l1_grp.next()
                    right = b_l1_grp.next()
                    pl.set_validshape(left, [valid_de, valid_k])
                    pl.set_validshape(right, [valid_k, valid_h])
                    pl.load(left, emb_t, [de_off, k_off])
                    pl.load(right, grad_key_ws,
                            [k_off, m_head, h_off], order=[0, 2])
                    al = a_l0_grp.next()
                    br = b_l0_grp.next()
                    pl.set_validshape(al, [valid_de, valid_k])
                    pl.set_validshape(br, [valid_k, valid_h])
                    pl.move(al, left)
                    pl.move(br, right)
                    if k_idx == 0:
                        pl.matmul(ac, al, br)
                    else:
                        pl.matmul_acc(ac, ac, al, br)
                pl.store(grad_key_proj_weights, ac,
                         [m_head, de_off, h_off], order=[1, 2])

        pl.system.set_mm_layout_transform(enabled=False)
        # grad_emb_acc is final (Nest1 value + per-head key atomic-add). Host casts it -> BF16.


# ═══════════════════════════════════════════════════════════════════
# Layer E: Host wrapper
# ═══════════════════════════════════════════════════════════════════

def engram_backward_wrapper(
    grad_output,        # [B, S, M_H, H] BF16
    hidden_states,      # [B, S, M_H, H] BF16
    embeddings,         # [B, S, De]     BF16
    key_proj_weights,   # [M_H, De, H]   BF16  (host-pre-transposed below)
    value_proj_weights, # [De, H]        BF16  (host-pre-transposed below)
    key_gamma,          # [M_H, H]       BF16
    query_gamma,        # [M_H, H]       BF16
    scores,             # [B, S, M_H, 1] FP32
    gates,              # [B, S, M_H, 1] FP32
    keys,               # [B, S, M_H, H] BF16 cache; cast to FP32 in kernel
    value,              # [B, S, H]      BF16 cache; cast to FP32 in kernel
    clamp_value=1e-6,
    eps=1e-6,
    use_kernel_transpose=True,   # API-compat flag; only host-pre-transpose path exists
):
    """Host wrapper. Mirrors forward's reshape + launch + reshape-back, and
    additionally:
      * host pre-transposes W_v, W_k, E (rule 3: no is_transpose / layout=ZN)
      * pads scores/gates scalar dim 1->64 (PyPTO tile load does not broadcast 1)
      * pre-zeros FP32 grad_γ GM (FP32 tile RMW targets)
      * keeps only scores/gates in FP32 cache form; other cache tensors stay BF16.
      * returns low-precision final gradients; gamma RMW and reduction use an
        FP32 per-core GM workspace, with one vector subblock as the writer.
    """
    del use_kernel_transpose  # only the host-pre-transpose path is implemented
    if scores.dtype != torch.float32 or gates.dtype != torch.float32:
        raise TypeError(
            "engram_backward_wrapper expects scores and gates to be torch.float32"
        )
    b, s, m_dim, h = grad_output.shape
    bs = b * s
    de = embeddings.shape[-1]
    device = grad_output.device
    dtype = grad_output.dtype

    # ── TilingKey selection: only H=1280/2560 have a specialized kernel. ──
    if h not in (1280, 2560):
        raise ValueError(
            f"engram_backward_wrapper only supports H in {{1280, 2560}} (TilingKey "
            f"specialization), got H={h}"
        )
    # BSMode: largest V_TILE that fills all 64 vector workers
    # (need n_bs_v = ceil(BS/V_TILE) >= 64; small BS floors at V_TILE=16).
    if bs >= 128 * 64:    # 8192
        v_tile, bsmode = V_TILE_BS0, 0
    elif bs >= 64 * 64:   # 4096
        v_tile, bsmode = V_TILE_BS1, 1
    elif bs >= 32 * 64:   # 2048
        v_tile, bsmode = V_TILE_BS2, 2
    else:
        v_tile, bsmode = V_TILE_BS3, 3
    tiling_key = {"HMode": 1 if h == 2560 else 0, "BSMode": bsmode}

    # ── Host reshape: merge B·S -> BS ──
    go_bs = grad_output.reshape(bs, m_dim, h).contiguous()
    hid_bs = hidden_states.reshape(bs, m_dim, h).contiguous()
    emb_bs = embeddings.reshape(bs, de).contiguous()
    keys_bs = keys.reshape(bs, m_dim, h).contiguous()
    val_bs = value.reshape(bs, h).contiguous()
    # Pad scores/gates 1->64 (forward stores score_back/gate_back as [.,.,64]).
    sc_bs = scores.reshape(bs, m_dim, 1).expand(-1, -1, 64).contiguous()
    gt_bs = gates.reshape(bs, m_dim, 1).expand(-1, -1, 64).contiguous()

    # ── Host pre-transpose (rule 3): kernel uses plain NZ loads.  Weights stay
    #    BF16 (cube is BF16) -- no .float() upcast. ──
    emb_t = emb_bs.t().contiguous()                                      # [De, BS] BF16
    wv_t = value_proj_weights.t().contiguous()                          # [H, De] BF16
    wk_t = key_proj_weights.transpose(-1, -2).contiguous()              # [M_H, H, De] BF16

    # ── BF16 cube-operand workspaces (vector produces FP32, casts to BF16 on store) ──
    grad_value_ws = torch.empty((bs, h), dtype=torch.bfloat16, device=device)
    grad_key_ws = torch.empty((bs, m_dim, h), dtype=torch.bfloat16, device=device)

    # Launch on all AI Cores (num_cores AIC x 2 AIV = num_cores*2 vector workers).
    # NOTE: the TilingKey V_TILE / BSMode split and the gamma workspace sizing are
    # currently tuned for 32 cores (a5 / DAV_3510, 64 workers). Other SoCs run but
    # may need TilingKey retuning for best occupancy.
    num_cores = get_platform_info().core_num

    # FP32 per-core gamma workspace (one slot per AIV subblock); host reduces + casts.
    grad_qgamma_acc = torch.zeros((num_cores * 2, m_dim, h), dtype=torch.float32, device=device)  # B-query split
    grad_kgamma_acc = torch.zeros((num_cores * 2, m_dim, h), dtype=torch.float32, device=device)  # B-key split

    # FP32 grad_emb accumulator [BS, De]: Nest1 value + per-head key all atomic-add
    # here. MUST be pre-zeroed (atomicAdd accumulates onto the existing value).
    grad_emb_acc = torch.zeros((bs, de), dtype=torch.float32, device=device)

    # ── grad_score cache (B-key writes, B-query reads; avoids recomputing Step7b+Step5) ──
    gscore_ws = torch.empty((bs, m_dim, 64), dtype=torch.float32, device=device)

    # ── Low-precision final outputs; all source calculations stay FP32 ──
    # (grad_embeddings is now produced by host from grad_emb_acc, see below.)
    grad_hidden_states = torch.empty((bs, m_dim, h), dtype=dtype, device=device)
    grad_key_proj_weights = torch.empty((m_dim, de, h), dtype=dtype, device=device)
    grad_value_proj_weights = torch.empty((de, h), dtype=dtype, device=device)

    engram_backward_kernel[None, num_cores, tiling_key](
        go_bs, hid_bs, emb_bs,
        key_gamma, query_gamma,
        sc_bs, gt_bs, keys_bs, val_bs,
        emb_t, wv_t, wk_t,
        grad_value_ws, grad_key_ws, gscore_ws,
        grad_qgamma_acc, grad_kgamma_acc, grad_emb_acc,
        grad_hidden_states, grad_key_proj_weights, grad_value_proj_weights,
    )
    torch.npu.synchronize()

    # Cast FP32 grad_emb_acc -> BF16 grad_embeddings on host.
    grad_embeddings = grad_emb_acc.to(dtype)

    # ── Reduce FP32 per-core gamma workspace, then cast once at the output ──
    grad_kgamma_out = grad_kgamma_acc.sum(dim=0).to(dtype)
    grad_qgamma_out = grad_qgamma_acc.sum(dim=0).to(dtype)

    grad_hidden_states = grad_hidden_states.reshape(b, s, m_dim, h)
    grad_embeddings = grad_embeddings.reshape(b, s, de)
    grad_key_proj_weights = grad_key_proj_weights.reshape(m_dim, de, h)
    return (
        grad_hidden_states,
        grad_embeddings,
        grad_key_proj_weights,
        grad_value_proj_weights,
        grad_kgamma_out,
        grad_qgamma_out,
    )
