#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

"""PyPTO-Pro engram_backward kernel implementation (Stage 4) — MINIMAL rewrite.

Backward of the Engram Gated Memory operator (forward = engram_v4).

Design goal: COMPILE + RUN FIRST. No optimization. Mirror the forward
(engram_v4_impl.py) structure as closely as possible.

Hard rules respected (see task brief):
  (2) NO `vf.astype` inside @pl.vector_function. BF16->FP32 is done in the main
      flow via `pl.cast(..., target_type=pl.DT_FP32, ...)`; every VF helper eats
      FP32 tiles only (exactly like forward).
  (3) NO `is_transpose`, NO `layout=pl.ZN`. W_v / W_k / E are host-pre-transposed
      in the wrapper; the kernel loads them as plain NZ.
  (4) The bs_start loop is truncated via a stored variable:
        bs_tile_rows = pl.min(TILE_M, bs - row_off)
        for bs_start in pl.range(0, m_tile_rows, TILE_BS_VEC): ...
  (5) Tile-group count kept lean (cube=5, vec=17 single-id groups; comparable
      to forward which uses ~22 vec groups).
  (6) Reduction outputs use the STABLE options only:
        - grad_W_v / grad_W_k / grad_emb : cube output-tile-distributed,
          FULL K-reduction per output tile, cover-write (same as forward value
          projection). NO atomicAdd from L0C anywhere.
        - grad_γ_q / grad_γ_k            : FP32 tile RMW into a host
          pre-zeroed FP32 GM workspace; only vector subblock 0 performs
          the RMW.

Math truth:  custom/engram_backward/engram_backward_golden.py
Struct tmpl: engram/engram_v4_impl.py (forward)

7-step backward (per head m, reversed Step7->Step1):
  Step7  grad_gates[m]=Σ_h(go·value); grad_value=Σ_m(go·gates)
  Step6  linear_bw(grad_value, E, W_v)  -> grad_emb_v, grad_W_v
  Step5  grad_score   = signed_sqrt_gate_bw(grad_gate, score, gate)
  Step4  grad_nKey    = grad_score·(1/√H)·nQuery ; grad_nQuery symmetric
  Step3  (grad_hidden_m, grad_γ_q) = rms_norm_bw(grad_nQuery, hidden, γ_q)
  Step2  (grad_key_m,  grad_γ_k) = rms_norm_bw(grad_nKey, key, γ_k)
  Step1  linear_bw(grad_key_m, E, W_k[m]) -> grad_emb_k, grad_W_k[m]
  Sum:   grad_emb = grad_emb_v + Σ_m grad_emb_k

Kernel layout (single kernel, vector section FIRST then cube):
  VECTOR section:
  Pass A  Step7a : grad_value_ws  (cover-write per row and full h; Σ over m_h)
    Pass b  per-head Step7b -> Step5 -> Step4 -> Step3(rms_q) -> Step2(rms_k):
            grad_hidden_states (cover-write), grad_key_ws (cover-write),
            grad_γ_q / grad_γ_k (FP32 tile RMW into low-precision GM).
  handoff: vector set_cross_core(MTE3, event=0) -> cube wait_cross_core(MTE1, event=0)
           (V->Cube proven pattern, see custom/lhz_design/test_lhz_design.py:644/656
            and 接口手册-05 §set/wait_cross_core: "V->Cube: set(MTE3) + wait(MTE1)")
  CUBE section:
    Nest 1 grad_emb : per (M_tile, De_tile) long L0C Partial chain
                      (1 value + m_h keys) -> single Final cover-write.
  Nest 2 grad_W_v : output-tile distributed, full K=bs, cover-write.
  Nest 3 grad_W_k : loop m_h, output-tile distributed, full K=bs, cover-write.

PRECISION STATUS: [PRECISION_UNKNOWN] (no Python/NPU on this host).
NPU repro:  python custom/engram_backward/test_engram_backward.py
"""

import logging
import os

import torch
import torch_npu  # noqa: F401  (registers NPU backend)

import pypto_pro.language as pl
from pypto_pro.runtime.platform import get_platform_info

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# ═══════════════════════════════════════════════════════════════════
# Layer b: Compile-time constants  (mirror forward engram_v4 exactly)
# ═══════════════════════════════════════════════════════════════════

# m_h (head count) is DYNAMIC (1-16), derived in kernel body from grad_output.shape[1].
TILE_M = 128  # cube M tile
TILE_K = 128  # cube K tile
TILE_N = 128  # cube N tile
TILE_BS_VEC = 4  # vec sub-row tile (conservative, < 248KB UB)
# m_h is now DYNAMIC (1-16), derived from grad_output.shape[1] in kernel body.
H_CHUNK = 2560  # one full-h vector tile (h=1280 uses valid_shape tail)
LANES_FP32 = 64  # VF FP32 register width
VPW_K_CHUNK = 1024  # split the long BS reduction before the final vector sum

CLAMP_VALUE = 1.0e-6  # signed_sqrt_gate |s| floor
RMS_EPS = 1.0e-6  # RMSNorm zero-division guard
GATE_EPS = 1.0e-12  # signed_sqrt_gate denominator guard (aligns golden)

# ═══════════════════════════════════════════════════════════════════
# UB addresses (vector section). All non-overlapping; each [4,64]/[1,2560]
# scratch tile has its OWN address so simultaneous-live tiles never collide.
# BF16 [4,2560]=0x5000  FP32 [4,2560]=0xA000; h is processed as one tile.
# Peak allocated footprint ≈ 288 KB; the A5 Vector UB supports this layout.
# VA_OUT16 aliases VA_GV32: Pass-A grad_value is fully stored before Pass-b.
# VA_GKACC32 reuses the query partial buffer: query partials are dead before
# key gamma RMW begins.
# ═══════════════════════════════════════════════════════════════════

VA_GO16 = 0x00000  # [4,2560] BF16 grad_output load (Pass A & b)
VA_GO32 = 0x05000  # [4,2560] FP32 grad_output FP32 (Pass A & b)
VA_PT16 = 0x0F000  # [4,2560] BF16 partner BF16 (value/nKey/nQuery/hidden/key)
VA_PT32 = 0x14000  # [4,2560] FP32 partner FP32
VA_GV32 = 0x1E000  # [4,2560] FP32 Pass-A grad_value accumulator
VA_OUT32 = 0x28000  # [4,2560] FP32 Pass-b grad_nQuery/grad_nKey
VA_OUT16 = 0x1E000  # [4,2560] BF16 final vector output (aliases VA_GV32)
VA_SC32 = 0x32000  # [4,64] FP32 score
VA_GT32 = 0x32400  # [4,64] FP32 gate
VA_GG32 = 0x32800  # [4,64] FP32 grad_gate (Step7b accumulator)
VA_GS32 = 0x32C00  # [4,64] FP32 grad_score (Step5 output)
VA_SQ32 = 0x33000  # [4,64] FP32 rms sq_sum accumulator
VA_RMS32 = 0x33400  # [4,64] FP32 rms inv_rms
VA_INNER32 = 0x33800  # [4,64] FP32 rms inner (mean)
VA_GAM16 = 0x33C00  # [1,2560] BF16 gamma
VA_GAM32 = 0x35000  # [1,2560] FP32 gamma FP32
VA_GGP32 = 0x37800  # [1,2560] FP32 grad_gamma partial -- query path
VA_GGP32_K = 0x3A000  # [1,2560] FP32 grad_gamma partial -- key path
VA_GQACC32 = 0x3C800  # [1,2560] FP32 RMW accumulator load buf -- query gamma
VA_GKACC32 = 0x37800  # [1,2560] FP32 RMW accumulator load buf -- key gamma (aliases query partial)

# L1 addresses (cube section) — two generic L1 buffers, time-shared
LA_LEFT = 0x00000
LA_RIGHT = 0x10000

L0A_BASE = 0x0000
L0B_BASE = 0x0000
L0C_BASE = 0x0000


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
    go_f32,  # [TILE_BS_VEC, H_CHUNK] FP32 grad_output for one head
    gate_f32,  # [TILE_BS_VEC, 64] FP32 gate (scalar per row, lane 0)
    gv_acc,  # [TILE_BS_VEC, H_CHUNK] FP32 in/out grad_value accumulator
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
    go_f32,  # [TILE_BS_VEC, H_CHUNK] FP32 grad_output for this head
    val_f32,  # [TILE_BS_VEC, H_CHUNK] FP32 value (shared)
    gg_acc,  # [TILE_BS_VEC, 64] FP32 in/out grad_gate accumulator
    n_rows, n_cols, row_stride,
):
    """Step7b^{-1}: gg_acc[r] += Σ_h(go[r,h] * val[r,h]) for this h-chunk."""
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
    gg_f32,  # [TILE_BS_VEC, 64] FP32 grad_gate (scalar per row)
    score_f32,  # [TILE_BS_VEC, 64] FP32 score (scalar per row)
    gate_f32,  # [TILE_BS_VEC, 64] FP32 gate g (scalar per row)
    gs_out,  # [TILE_BS_VEC, 64] FP32 write grad_score
    n_rows,
):
    """Step5^{-1}: signed_sqrt_gate backward.
       公式：grad_score = grad_gate · g(1−g) · mask / (2·√max(|s|,c) + 1e-12)
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

        # 公式：sigmoid_grad = g - g*g = g(1-g)
        g_sq = vf.mul(gt_b, gt_b, preg)
        sig_grad = vf.sub(gt_b, g_sq, preg)
        # 公式：mask = (|s| > clamp) ? 1 : 0
        abs_s = vf.abs(sc_b, preg)
        mask_gt = vf.gt(abs_s, clamp_reg, preg)
        mask = vf.select(one_reg, zero_reg, mask_gt)
        # 公式：sqrt_abs = sqrt(max(|s|, clamp))
        clamped = vf.max(abs_s, clamp_reg, preg)
        sqrt_abs = vf.sqrt(clamped, preg)
        # 公式：logits_grad = mask / (2*sqrt_abs + 1e-12)
        denom = vf.mul(two_reg, sqrt_abs, preg)
        denom = vf.add(denom, eps_reg, preg)
        logits_grad = vf.div(mask, denom, preg)
        # 公式：grad_score = grad_gate · sigmoid_grad · logits_grad
        result = vf.mul(gg_b, sig_grad, preg)
        result = vf.mul(result, logits_grad, preg)
        vf.store_align(gs_out + m * 64, result, preg)


@pl.vector_function
def vf_scaled_dot(
    gs_f32,  # [TILE_BS_VEC, 64] FP32 grad_score (scalar per row)
    partner_f32,  # [TILE_BS_VEC, H_CHUNK] FP32 the "other" normed vector
    out_f32,  # [TILE_BS_VEC, H_CHUNK] FP32 write result
    n_rows, n_cols, row_stride, h_value,
):
    """Step4^{-1}: 公式：out[r,h] = grad_score[r] · (1/√H) · partner[r,h].
       Used as: partner=normed_key -> grad_nQuery; partner=normed_query -> grad_nKey.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    n_regs = (n_cols + LANES_FP32 - 1) // LANES_FP32
    # Do not pass 1/sqrt(h) as a VF scalar: scalar constants are lowered
    # through the BF16 scalar path on this target.  h itself is exact, so
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


# ── rms_norm_backward (3-pass; h-split cross-chunk accumulation) ──

@pl.vector_function
def vf_rmsbw_sq(x_f32, sq_acc, n_rows, n_cols, row_stride):
    """rms_bw Pass 1: sq_acc[r] += Σ_h(x[r,h]^2) for this h-chunk."""
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
    """rms_bw: compute inv_rms in FP32 using an exact integer h divisor."""
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
def vf_rmsbw_inner(
    gx_f32,  # [TILE_BS_VEC, H_CHUNK] FP32 grad_xhat (= grad_n)
    x_f32,  # [TILE_BS_VEC, H_CHUNK] FP32 x
    gam_f32,  # [1, H_CHUNK] FP32 gamma
    rms_f32,  # [TILE_BS_VEC, 64] FP32 inv_rms per row
    inner_acc,  # [TILE_BS_VEC, 64] FP32 in/out
    gg_acc,  # [1, H_CHUNK] FP32 in/out grad_gamma partial
    n_rows, n_cols, row_stride,
):
    """rms_bw Pass 2: 公式：inner[r] += Σ(grad_n·n); gg_acc[h] += Σ grad_xhat·n.
       n = x · inv_rms ; grad_n = grad_xhat · gamma.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    n_regs = (n_cols + LANES_FP32 - 1) // LANES_FP32
    for m in pl.range(0, n_rows):
        inv_rms = vf.load_align(rms_f32, m * 64)
        inv_rms_b = vf.full(inv_rms, preg)
        inner = vf.load_align(inner_acc, m * 64)
        for r in pl.range(0, n_regs):
            off = m * row_stride + r * LANES_FP32
            x = vf.load_align(x_f32, off)
            n_reg = vf.mul(x, inv_rms_b, preg)
            gx = vf.load_align(gx_f32, off)
            gam = vf.load_align(gam_f32, r * LANES_FP32)
            grad_n = vf.mul(gx, gam, preg)
            dn = vf.mul(grad_n, n_reg, preg)
            part = vf.reduce_sum(dn, preg, merge_mode=pl.MergeMode.ZEROING)
            inner = vf.add(inner, part, preg)
            dg = vf.mul(gx, n_reg, preg)
            prev_gg = vf.load_align(gg_acc, r * LANES_FP32)
            new_gg = vf.add(prev_gg, dg, preg)
            vf.store_align(gg_acc + r * LANES_FP32, new_gg, preg)
        vf.store_align(inner_acc + m * 64, inner, preg)


@pl.vector_function
def vf_rmsbw_inner_finalize(inner_acc, n_rows, h_value):
    """Finalize inner by dividing the FP32 sum by the exact integer h."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    h_reg = vf.full(h_value, preg, dtype=pl.DT_FP32)
    for m in pl.range(0, n_rows):
        val = vf.load_align(inner_acc, m * 64)
        val = vf.div(val, h_reg, preg)
        vf.store_align(inner_acc + m * 64, val, preg)


@pl.vector_function
def vf_rmsbw_gradx(
    gx_f32,  # [TILE_BS_VEC, H_CHUNK] FP32 grad_xhat
    x_f32,  # [TILE_BS_VEC, H_CHUNK] FP32 x
    gam_f32,  # [1, H_CHUNK] FP32 gamma
    rms_f32,  # [TILE_BS_VEC, 64] FP32 inv_rms
    inner_f32,  # [TILE_BS_VEC, 64] FP32 inner (mean)
    gx_out_f32,  # [TILE_BS_VEC, H_CHUNK] FP32 write grad_x
    n_rows, n_cols, row_stride,
):
    """rms_bw Pass 3: grad_x = (grad_n − n·inner) · inv_rms.
       Recomputes n and grad_n from x and gamma.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    n_regs = (n_cols + LANES_FP32 - 1) // LANES_FP32
    for m in pl.range(0, n_rows):
        inv_rms = vf.load_align(rms_f32, m * 64)
        inv_rms_b = vf.full(inv_rms, preg)
        inner = vf.load_align(inner_f32, m * 64)
        inner_b = vf.full(inner, preg)
        for r in pl.range(0, n_regs):
            off = m * row_stride + r * LANES_FP32
            x = vf.load_align(x_f32, off)
            n_reg = vf.mul(x, inv_rms_b, preg)
            gx = vf.load_align(gx_f32, off)
            gam = vf.load_align(gam_f32, r * LANES_FP32)
            grad_n = vf.mul(gx, gam, preg)
            n_inner = vf.mul(n_reg, inner_b, preg)
            diff = vf.sub(grad_n, n_inner, preg)
            grad_x = vf.mul(diff, inv_rms_b, preg)
            vf.store_align(gx_out_f32 + off, grad_x, preg)


# ═══════════════════════════════════════════════════════════════════
# Layer D: engram_backward_kernel
# ═══════════════════════════════════════════════════════════════════

@pl.jit(auto_mutex=True)
def engram_backward_kernel(
    # ── Inputs (scores/gates are FP32; other activation inputs are BF16) ──
    grad_output: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    hidden_states: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    embeddings: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    key_gamma: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    query_gamma: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    scores: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, 64], pl.DT_FP32],
    gates: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, 64], pl.DT_FP32],
    keys: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    value: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    # ── host-pre-transposed inputs (plain NZ loads, NO is_transpose / layout=ZN) ──
    emb_t_f32: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],  # [de, bs]
    wv_t_f32: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],  # [h, de]
    wk_t_f32: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],  # [m_h, h, de]
    # ── Mixed-precision intermediate workspaces ──
    grad_value_ws: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],  # [M, h]
    grad_key_ws: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],  # [M, m_h, h] (m_h dynamic)
    grad_vpw_ws: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],  # [K-split, De, h]
    # ── FP32 per-core gamma workspace [num_cores, m_h, h] ──
    grad_qgamma_acc: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    grad_kgamma_acc: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    # ── FP32 workspace (cube cover-write slots, vector sums to grad_embeddings) ──
    grad_emb_ws: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],  # [m_h+1, M, de] (m_h dynamic)
    # ── BF16 outputs ──
    grad_hidden_states: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    grad_embeddings: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    grad_key_proj_weights: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    grad_value_proj_weights: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
):
    """Single kernel: vector section (produces grad_value_ws/grad_key_ws +
    grad_hidden + grad_γ) then cube section (consumes ws, produces grad_emb +
    grad_W). BF16 activation inputs are cast once at the computation boundary;
    grad_value_ws and grad_key_ws feed legal FP32 64x64 Cube paths directly.
    Final output stores are low precision.
    No VF performs vf.astype.
    """
    bs = grad_output.shape[0]
    m_h = grad_output.shape[1]  # head count, dynamic (1-16)
    h = grad_output.shape[2]
    de = embeddings.shape[1]
    # Padded bs for cube Nest1 row-tiling: the wrapper zero-pads grad_value_ws /
    # grad_key_ws / grad_emb_ws to a multiple of f32_tile so Nest1 stores FULL
    # FP32 tiles (avoids the undersized-tail FIX-store scramble on grad_emb_ws).
    # The real bs is still used by the vector section, V2, and Nest2/3 K-reduce.
    bs_f32 = grad_value_ws.shape[0]
    n_h_chunks = (h + H_CHUNK - 1) // H_CHUNK  # exactly one full-h tile
    n_k_h = (h + TILE_K - 1) // TILE_K  # K=h tile count
    n_n_de = (de + TILE_N - 1) // TILE_N  # N=de tile count
    n_de_tiles = (de + TILE_M - 1) // TILE_M  # de tile count (TILE_M stride)
    n_h_tiles = (h + TILE_N - 1) // TILE_N  # h tile count (TILE_N stride)
    n_k_bs = (bs + TILE_K - 1) // TILE_K  # K=bs tile count (grad_W reduction)
    f32_tile = 64
    n_k_h_f32 = (h + f32_tile - 1) // f32_tile
    n_n_de_f32 = (de + f32_tile - 1) // f32_tile
    n_k_bs_f32 = (bs + f32_tile - 1) // f32_tile
    n_vpw_parts = (bs + VPW_K_CHUNK - 1) // VPW_K_CHUNK
    n_de_tiles_f32 = n_n_de_f32
    n_h_tiles_f32 = (h + f32_tile - 1) // f32_tile
    num_cores = pl.get_block_num()
    # Baseline c3c9e0e4: get_block_idx() already returns the physical AI Core
    # index directly. The //get_subblock_num() split was introduced later on
    # master (commit 272b1fa3c) and is absent from this baseline.
    core_id = pl.get_block_idx()
    # A5 executes section_vector on both vector subblocks. The gamma
    # workspace is private per AI core, not per subblock.
    sub_id = pl.get_subblock_idx()
    n_bs_tiles = (bs + TILE_M - 1) // TILE_M
    iters_per_core = (n_bs_tiles + num_cores - 1) // num_cores
    n_bs_tiles_f32 = bs_f32 // f32_tile  # Nest1 tiles over padded bs (all full tiles)
    iters_f32_per_core = (n_bs_tiles_f32 + num_cores - 1) // num_cores

    # ═══════════════════════════════════════════════════════════════
    # VECTOR SECTION (runs first; produces grad_value_ws / grad_key_ws)
    # ═══════════════════════════════════════════════════════════════
    tt_mv16 = pl.TileType(shape=[TILE_BS_VEC, H_CHUNK], dtype=pl.DT_BF16,
                          target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    tt_mv32 = pl.TileType(shape=[TILE_BS_VEC, H_CHUNK], dtype=pl.DT_FP32,
                          target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    tt_sc32 = pl.TileType(shape=[TILE_BS_VEC, 64], dtype=pl.DT_FP32,
                          target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    tt_gam16 = pl.TileType(shape=[1, H_CHUNK], dtype=pl.DT_BF16,
                           target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    tt_gam32 = pl.TileType(shape=[1, H_CHUNK], dtype=pl.DT_FP32,
                           target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])

    # The tile capacity is fixed at 2560, but every vector operation uses the
    # actual h as its valid width.  Thus h=1280 is a tail of the same tile and
    # h is never split into multiple vector chunks.
    h_dim = h

    go16_grp = pl.make_tile_group(type=tt_mv16, addrs=VA_GO16, mutex_ids=[0])
    go32_grp = pl.make_tile_group(type=tt_mv32, addrs=VA_GO32, mutex_ids=[1])
    pt16_grp = pl.make_tile_group(type=tt_mv16, addrs=VA_PT16, mutex_ids=[2])
    pt32_grp = pl.make_tile_group(type=tt_mv32, addrs=VA_PT32, mutex_ids=[3])
    gv32_grp = pl.make_tile_group(type=tt_mv32, addrs=VA_GV32, mutex_ids=[4])
    out32_grp = pl.make_tile_group(type=tt_mv32, addrs=VA_OUT32, mutex_ids=[17])
    out16_grp = pl.make_tile_group(type=tt_mv16, addrs=VA_OUT16, mutex_ids=[4])
    sc32_grp = pl.make_tile_group(type=tt_sc32, addrs=VA_SC32, mutex_ids=[7])
    gt32_grp = pl.make_tile_group(type=tt_sc32, addrs=VA_GT32, mutex_ids=[8])
    gg_grp = pl.make_tile_group(type=tt_sc32, addrs=VA_GG32, mutex_ids=[9])
    gs_grp = pl.make_tile_group(type=tt_sc32, addrs=VA_GS32, mutex_ids=[10])
    sq_grp = pl.make_tile_group(type=tt_sc32, addrs=VA_SQ32, mutex_ids=[11])
    rms_grp = pl.make_tile_group(type=tt_sc32, addrs=VA_RMS32, mutex_ids=[12])
    inner_grp = pl.make_tile_group(type=tt_sc32, addrs=VA_INNER32, mutex_ids=[13])
    gam16_grp = pl.make_tile_group(type=tt_gam16, addrs=VA_GAM16, mutex_ids=[14])
    gam32_grp = pl.make_tile_group(type=tt_gam32, addrs=VA_GAM32, mutex_ids=[15])
    ggp_grp = pl.make_tile_group(type=tt_gam32, addrs=VA_GGP32, mutex_ids=[16])
    ggp_k_grp = pl.make_tile_group(type=tt_gam32, addrs=VA_GGP32_K, mutex_ids=[18])
    gqacc_grp = pl.make_tile_group(type=tt_gam32, addrs=VA_GQACC32, mutex_ids=[19])
    gkacc_grp = pl.make_tile_group(type=tt_gam32, addrs=VA_GKACC32, mutex_ids=[20])

    with pl.section_vector():
        for tile_idx in pl.range(0, iters_per_core):
            bsi = core_id + tile_idx * num_cores
            if bsi < n_bs_tiles:
                row_off = bsi * TILE_M
                bs_tile_rows = pl.min(TILE_M, bs - row_off)  # (rule 4) stored variable
                for bs_start in pl.range(0, bs_tile_rows, TILE_BS_VEC):
                    valid_bsv = pl.min(TILE_BS_VEC, bs_tile_rows - bs_start)
                    bs_off = row_off + bs_start

                    # 公式：══ Pass A: Step7a -> grad_value_ws = Σ_m(go·gate) ══
                    for h_chunk in pl.range(0, n_h_chunks):
                        h_off = h_chunk * h_dim
                        gv = gv32_grp.current()
                        pl.set_validshape(gv, [valid_bsv, h_dim])
                        vf_zero_2d(gv, valid_bsv, h_dim, H_CHUNK)
                        for m_head in pl.range(0, m_h):
                            go16 = go16_grp.current()
                            pl.set_validshape(go16, [valid_bsv, h_dim])
                            pl.load(go16, grad_output,
                                    [bs_off, m_head, h_off], order=[0, 2])
                            go32 = go32_grp.current()
                            pl.set_validshape(go32, [valid_bsv, h_dim])
                            pl.cast(go32, go16, mode=pl.RoundMode.CAST_NONE)
                            sc32 = sc32_grp.current()
                            pl.set_validshape(sc32, [valid_bsv, 64])
                            pl.load(sc32, gates, [bs_off, m_head, 0], order=[0, 2])
                            vf_grad_value_accum(go32, sc32, gv, valid_bsv, h_dim, H_CHUNK)
                        pl.store(grad_value_ws, gv, [bs_off, h_off])

                    # OUT16 reuses the Pass-A GV32 address.  Complete the
                    # Pass-A store before Pass-b starts using that address.
                    pl.system.bar_all()

                    # ══ Pass b: per-head Step7b -> Step5 -> Step4 -> Step3 -> Step2 ══
                    for m_head in pl.range(0, m_h):
                        # ── Step7b: grad_gate = Σ_h(go·val) over the full-h tile ──
                        gg = gg_grp.current()
                        pl.set_validshape(gg, [valid_bsv, 64])
                        vf_zero_2d(gg, valid_bsv, 64, 64)
                        for h_chunk in pl.range(0, n_h_chunks):
                            h_off = h_chunk * h_dim
                            go16 = go16_grp.current()
                            pl.set_validshape(go16, [valid_bsv, h_dim])
                            pl.load(go16, grad_output,
                                    [bs_off, m_head, h_off], order=[0, 2])
                            go32 = go32_grp.current()
                            pl.set_validshape(go32, [valid_bsv, h_dim])
                            pl.cast(go32, go16, mode=pl.RoundMode.CAST_NONE)
                            pt16 = pt16_grp.current()
                            pl.set_validshape(pt16, [valid_bsv, h_dim])
                            pl.load(pt16, value, [bs_off, h_off])
                            pt32 = pt32_grp.current()
                            pl.set_validshape(pt32, [valid_bsv, h_dim])
                            pl.cast(pt32, pt16, mode=pl.RoundMode.CAST_NONE)
                            vf_grad_gate_accum(go32, pt32, gg, valid_bsv, h_dim, H_CHUNK)

                        # ── Step5: gate_bw(gg, score, gate) -> gs ──
                        sc32 = sc32_grp.current()  # score FP32
                        pl.set_validshape(sc32, [valid_bsv, 64])
                        pl.load(sc32, scores, [bs_off, m_head, 0], order=[0, 2])
                        gt32 = gt32_grp.current()  # gate FP32
                        pl.set_validshape(gt32, [valid_bsv, 64])
                        pl.load(gt32, gates, [bs_off, m_head, 0], order=[0, 2])
                        gs = gs_grp.current()  # grad_score output
                        pl.set_validshape(gs, [valid_bsv, 64])
                        vf_gate_bw(gg, sc32, gt32, gs, valid_bsv)
                        # ══ Step4 + Step3 (query): grad_nQuery -> rms_bw(hidden, γ_q) ══
                        # Pass 1: sq over hidden_states
                        sq = sq_grp.current()
                        pl.set_validshape(sq, [valid_bsv, 64])
                        vf_zero_2d(sq, valid_bsv, 64, 64)
                        for h_chunk in pl.range(0, n_h_chunks):
                            h_off = h_chunk * h_dim
                            pt16 = pt16_grp.current()
                            pl.set_validshape(pt16, [valid_bsv, h_dim])
                            pl.load(pt16, hidden_states,
                                    [bs_off, m_head, h_off], order=[0, 2])
                            pt32 = pt32_grp.current()
                            pl.set_validshape(pt32, [valid_bsv, h_dim])
                            pl.cast(pt32, pt16, mode=pl.RoundMode.CAST_NONE)
                            vf_rmsbw_sq(pt32, sq, valid_bsv, h_dim, H_CHUNK)
                        rms = rms_grp.current()
                        pl.set_validshape(rms, [valid_bsv, 64])
                        vf_rmsbw_invrms(sq, rms, valid_bsv, h)
                        # Recompute normed keys in FP32 for the query backward.
                        partner_sq = sq_grp.current()
                        pl.set_validshape(partner_sq, [valid_bsv, 64])
                        vf_zero_2d(partner_sq, valid_bsv, 64, 64)
                        for norm_h_chunk in pl.range(0, n_h_chunks):
                            norm_h_off = norm_h_chunk * h_dim
                            norm16 = pt16_grp.current()
                            pl.set_validshape(norm16, [valid_bsv, h_dim])
                            pl.load(norm16, keys,
                                    [bs_off, m_head, norm_h_off], order=[0, 2])
                            norm32 = pt32_grp.current()
                            pl.set_validshape(norm32, [valid_bsv, h_dim])
                            pl.cast(norm32, norm16, mode=pl.RoundMode.CAST_NONE)
                            vf_rmsbw_sq(norm32, partner_sq, valid_bsv, h_dim, H_CHUNK)
                        vf_rmsbw_invrms(partner_sq, partner_sq, valid_bsv, h)

                        # Pass 2: inner over h + grad_γ_q
                        inner = inner_grp.current()
                        pl.set_validshape(inner, [valid_bsv, 64])
                        vf_zero_2d(inner, valid_bsv, 64, 64)
                        for h_chunk in pl.range(0, n_h_chunks):
                            h_off = h_chunk * h_dim
                            # 公式：grad_nQuery chunk = gs·(1/√H)·RMSNorm(key)
                            pt16 = pt16_grp.current()
                            pl.set_validshape(pt16, [valid_bsv, h_dim])
                            pl.load(pt16, keys,
                                    [bs_off, m_head, h_off], order=[0, 2])
                            pt32 = pt32_grp.current()
                            pl.set_validshape(pt32, [valid_bsv, h_dim])
                            pl.cast(pt32, pt16, mode=pl.RoundMode.CAST_NONE)
                            gam16 = gam16_grp.current()
                            pl.set_validshape(gam16, [1, h_dim])
                            pl.load(gam16, key_gamma, [m_head, h_off], order=[0])
                            gam32 = gam32_grp.current()
                            pl.set_validshape(gam32, [1, h_dim])
                            pl.cast(gam32, gam16, mode=pl.RoundMode.CAST_NONE)
                            vf_rmsnorm_fwd(pt32, gam32, partner_sq, pt32,
                                           valid_bsv, h_dim, H_CHUNK)
                            gx32 = out32_grp.current()
                            pl.set_validshape(gx32, [valid_bsv, h_dim])
                            vf_scaled_dot(gs, pt32, gx32, valid_bsv, h_dim,
                                           H_CHUNK, h)
                            # reload hidden chunk
                            x16 = pt16_grp.current()
                            pl.set_validshape(x16, [valid_bsv, h_dim])
                            pl.load(x16, hidden_states,
                                    [bs_off, m_head, h_off], order=[0, 2])
                            x32 = pt32_grp.current()
                            pl.set_validshape(x32, [valid_bsv, h_dim])
                            pl.cast(x32, x16, mode=pl.RoundMode.CAST_NONE)
                            pl.load(gam16, query_gamma, [m_head, h_off], order=[0])
                            pl.cast(gam32, gam16, mode=pl.RoundMode.CAST_NONE)
                            ggp = ggp_grp.current()
                            pl.set_validshape(ggp, [1, h_dim])
                            vf_zero_1d(ggp, h_dim)
                            vf_rmsbw_inner(gx32, x32, gam32, rms, inner, ggp,
                                           valid_bsv, h_dim, H_CHUNK)
                            # RMW this chunk's grad_γ_q in FP32 workspace.
                            if sub_id == 0:
                                gqacc = gqacc_grp.current()
                                pl.set_validshape(gqacc, [1, h_dim])
                                pl.load(gqacc, grad_qgamma_acc,
                                        [core_id, m_head, h_off], order=[1, 2])
                                pl.add(gqacc, gqacc, ggp)
                                pl.store(grad_qgamma_acc, gqacc,
                                         [core_id, m_head, h_off], tile_dims=[1, 2])
                        vf_rmsbw_inner_finalize(inner, valid_bsv, h)
                        # Pass 3: grad_hidden_m -> store GM
                        partner_sq = sq_grp.current()
                        pl.set_validshape(partner_sq, [valid_bsv, 64])
                        vf_zero_2d(partner_sq, valid_bsv, 64, 64)
                        for norm_h_chunk in pl.range(0, n_h_chunks):
                            norm_h_off = norm_h_chunk * h_dim
                            norm16 = pt16_grp.current()
                            pl.set_validshape(norm16, [valid_bsv, h_dim])
                            pl.load(norm16, keys,
                                    [bs_off, m_head, norm_h_off], order=[0, 2])
                            norm32 = pt32_grp.current()
                            pl.set_validshape(norm32, [valid_bsv, h_dim])
                            pl.cast(norm32, norm16, mode=pl.RoundMode.CAST_NONE)
                            vf_rmsbw_sq(norm32, partner_sq, valid_bsv, h_dim, H_CHUNK)
                        vf_rmsbw_invrms(partner_sq, partner_sq, valid_bsv, h)
                        for h_chunk in pl.range(0, n_h_chunks):
                            h_off = h_chunk * h_dim
                            pt16 = pt16_grp.current()
                            pl.set_validshape(pt16, [valid_bsv, h_dim])
                            pl.load(pt16, keys,
                                    [bs_off, m_head, h_off], order=[0, 2])
                            pt32 = pt32_grp.current()
                            pl.set_validshape(pt32, [valid_bsv, h_dim])
                            pl.cast(pt32, pt16, mode=pl.RoundMode.CAST_NONE)
                            gam16 = gam16_grp.current()
                            pl.set_validshape(gam16, [1, h_dim])
                            pl.load(gam16, key_gamma, [m_head, h_off], order=[0])
                            gam32 = gam32_grp.current()
                            pl.set_validshape(gam32, [1, h_dim])
                            pl.cast(gam32, gam16, mode=pl.RoundMode.CAST_NONE)
                            vf_rmsnorm_fwd(pt32, gam32, partner_sq, pt32,
                                           valid_bsv, h_dim, H_CHUNK)
                            gx32 = out32_grp.current()
                            pl.set_validshape(gx32, [valid_bsv, h_dim])
                            vf_scaled_dot(gs, pt32, gx32, valid_bsv, h_dim,
                                           H_CHUNK, h)
                            x16 = pt16_grp.current()
                            pl.set_validshape(x16, [valid_bsv, h_dim])
                            pl.load(x16, hidden_states,
                                    [bs_off, m_head, h_off], order=[0, 2])
                            x32 = pt32_grp.current()
                            pl.set_validshape(x32, [valid_bsv, h_dim])
                            pl.cast(x32, x16, mode=pl.RoundMode.CAST_NONE)
                            pl.load(gam16, query_gamma, [m_head, h_off], order=[0])
                            pl.cast(gam32, gam16, mode=pl.RoundMode.CAST_NONE)
                            gh32 = go32_grp.current()
                            pl.set_validshape(gh32, [valid_bsv, h_dim])
                            vf_rmsbw_gradx(gx32, x32, gam32, rms, inner, gh32,
                                           valid_bsv, h_dim, H_CHUNK)
                            gh16 = out16_grp.current()
                            pl.set_validshape(gh16, [valid_bsv, h_dim])
                            pl.cast(gh16, gh32, mode=pl.RoundMode.CAST_ROUND)
                            pl.store(grad_hidden_states, gh16,
                                     [bs_off, m_head, h_off], tile_dims=[0, 2])

                        # Query gamma RMW and its dependent vector work use
                        # the same local tile as key gamma RMW below.
                        pl.system.bar_all()

                        # ══ Step4 + Step2 (key): grad_nKey -> rms_bw(keys, γ_k) ══
                        sq = sq_grp.current()
                        pl.set_validshape(sq, [valid_bsv, 64])
                        vf_zero_2d(sq, valid_bsv, 64, 64)
                        for h_chunk in pl.range(0, n_h_chunks):
                            h_off = h_chunk * h_dim
                            pt16 = pt16_grp.current()
                            pl.set_validshape(pt16, [valid_bsv, h_dim])
                            pl.load(pt16, keys,
                                    [bs_off, m_head, h_off], order=[0, 2])
                            pt32 = pt32_grp.current()
                            pl.set_validshape(pt32, [valid_bsv, h_dim])
                            pl.cast(pt32, pt16, mode=pl.RoundMode.CAST_NONE)
                            vf_rmsbw_sq(pt32, sq, valid_bsv, h_dim, H_CHUNK)
                        rms = rms_grp.current()
                        pl.set_validshape(rms, [valid_bsv, 64])
                        vf_rmsbw_invrms(sq, rms, valid_bsv, h)
                        # Recompute normed queries in FP32 for the key backward.
                        partner_sq = sq_grp.current()
                        pl.set_validshape(partner_sq, [valid_bsv, 64])
                        vf_zero_2d(partner_sq, valid_bsv, 64, 64)
                        for norm_h_chunk in pl.range(0, n_h_chunks):
                            norm_h_off = norm_h_chunk * h_dim
                            norm16 = pt16_grp.current()
                            pl.set_validshape(norm16, [valid_bsv, h_dim])
                            pl.load(norm16, hidden_states,
                                    [bs_off, m_head, norm_h_off], order=[0, 2])
                            norm32 = pt32_grp.current()
                            pl.set_validshape(norm32, [valid_bsv, h_dim])
                            pl.cast(norm32, norm16, mode=pl.RoundMode.CAST_NONE)
                            vf_rmsbw_sq(norm32, partner_sq, valid_bsv, h_dim, H_CHUNK)
                        vf_rmsbw_invrms(partner_sq, partner_sq, valid_bsv, h)

                        inner = inner_grp.current()
                        pl.set_validshape(inner, [valid_bsv, 64])
                        vf_zero_2d(inner, valid_bsv, 64, 64)
                        for h_chunk in pl.range(0, n_h_chunks):
                            h_off = h_chunk * h_dim
                            # 公式：grad_nKey chunk = gs·(1/√H)·RMSNorm(hidden)
                            pt16 = pt16_grp.current()
                            pl.set_validshape(pt16, [valid_bsv, h_dim])
                            pl.load(pt16, hidden_states,
                                    [bs_off, m_head, h_off], order=[0, 2])
                            pt32 = pt32_grp.current()
                            pl.set_validshape(pt32, [valid_bsv, h_dim])
                            pl.cast(pt32, pt16, mode=pl.RoundMode.CAST_NONE)
                            gam16 = gam16_grp.current()
                            pl.set_validshape(gam16, [1, h_dim])
                            pl.load(gam16, query_gamma, [m_head, h_off], order=[0])
                            gam32 = gam32_grp.current()
                            pl.set_validshape(gam32, [1, h_dim])
                            pl.cast(gam32, gam16, mode=pl.RoundMode.CAST_NONE)
                            vf_rmsnorm_fwd(pt32, gam32, partner_sq, pt32,
                                           valid_bsv, h_dim, H_CHUNK)
                            gx32 = out32_grp.current()
                            pl.set_validshape(gx32, [valid_bsv, h_dim])
                            vf_scaled_dot(gs, pt32, gx32, valid_bsv, h_dim,
                                           H_CHUNK, h)
                            x16 = pt16_grp.current()
                            pl.set_validshape(x16, [valid_bsv, h_dim])
                            pl.load(x16, keys,
                                    [bs_off, m_head, h_off], order=[0, 2])
                            x32 = pt32_grp.current()
                            pl.set_validshape(x32, [valid_bsv, h_dim])
                            pl.cast(x32, x16, mode=pl.RoundMode.CAST_NONE)
                            pl.load(gam16, key_gamma, [m_head, h_off], order=[0])
                            pl.cast(gam32, gam16, mode=pl.RoundMode.CAST_NONE)
                            ggp = ggp_k_grp.current()
                            pl.set_validshape(ggp, [1, h_dim])
                            vf_zero_1d(ggp, h_dim)
                            vf_rmsbw_inner(gx32, x32, gam32, rms, inner, ggp,
                                           valid_bsv, h_dim, H_CHUNK)
                            # RMW this chunk's grad_γ_k in FP32 workspace.
                            if sub_id == 0:
                                gkacc = gkacc_grp.current()
                                pl.set_validshape(gkacc, [1, h_dim])
                                pl.load(gkacc, grad_kgamma_acc,
                                        [core_id, m_head, h_off], order=[1, 2])
                                pl.add(gkacc, gkacc, ggp)
                                pl.store(grad_kgamma_acc, gkacc,
                                         [core_id, m_head, h_off], tile_dims=[1, 2])
                        vf_rmsbw_inner_finalize(inner, valid_bsv, h)
                        for h_chunk in pl.range(0, n_h_chunks):
                            h_off = h_chunk * h_dim
                            pt16 = pt16_grp.current()
                            pl.set_validshape(pt16, [valid_bsv, h_dim])
                            pl.load(pt16, hidden_states,
                                    [bs_off, m_head, h_off], order=[0, 2])
                            pt32 = pt32_grp.current()
                            pl.set_validshape(pt32, [valid_bsv, h_dim])
                            pl.cast(pt32, pt16, mode=pl.RoundMode.CAST_NONE)
                            gam16 = gam16_grp.current()
                            pl.set_validshape(gam16, [1, h_dim])
                            pl.load(gam16, query_gamma, [m_head, h_off], order=[0])
                            gam32 = gam32_grp.current()
                            pl.set_validshape(gam32, [1, h_dim])
                            pl.cast(gam32, gam16, mode=pl.RoundMode.CAST_NONE)
                            vf_rmsnorm_fwd(pt32, gam32, partner_sq, pt32,
                                           valid_bsv, h_dim, H_CHUNK)
                            gx32 = out32_grp.current()
                            pl.set_validshape(gx32, [valid_bsv, h_dim])
                            vf_scaled_dot(gs, pt32, gx32, valid_bsv, h_dim,
                                           H_CHUNK, h)
                            x16 = pt16_grp.current()
                            pl.set_validshape(x16, [valid_bsv, h_dim])
                            pl.load(x16, keys,
                                    [bs_off, m_head, h_off], order=[0, 2])
                            x32 = pt32_grp.current()
                            pl.set_validshape(x32, [valid_bsv, h_dim])
                            pl.cast(x32, x16, mode=pl.RoundMode.CAST_NONE)
                            gam16 = gam16_grp.current()
                            pl.set_validshape(gam16, [1, h_dim])
                            pl.load(gam16, key_gamma, [m_head, h_off], order=[0])
                            gam32 = gam32_grp.current()
                            pl.set_validshape(gam32, [1, h_dim])
                            pl.cast(gam32, gam16, mode=pl.RoundMode.CAST_NONE)
                            gk32 = go32_grp.current()
                            pl.set_validshape(gk32, [valid_bsv, h_dim])
                            vf_rmsbw_gradx(gx32, x32, gam32, rms, inner, gk32,
                                           valid_bsv, h_dim, H_CHUNK)
                            pl.store(grad_key_ws, gk32,
                                     [bs_off, m_head, h_off], tile_dims=[0, 2])

        # ── vector -> cube handoff (V->Cube: V->MTE3 fence, then
        #    set MTE3 / wait MTE1) ──
        # set_cross_core alone only publishes the cross-core event; it does
        # not replace the intra-core ordering between Vector arithmetic and
        # the MTE3 GM stores.  Without this fence, a following dynamic-shape
        # launch can make Cube observe a partially materialized workspace.
        # A hard MIX rendezvous prevents a previous dynamic-shape launch's
        # per-core event state from being mistaken for this launch's handoff.
        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3,
                           event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3,
                           event_id=0)
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=0,
                                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

    # ═══════════════════════════════════════════════════════════════
    # CUBE SECTION (consumes grad_value_ws / grad_key_ws and transposed inputs)
    # ═══════════════════════════════════════════════════════════════
    tt_f32_mat_a = pl.TileType(shape=[f32_tile, f32_tile], dtype=pl.DT_FP32,
                               target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                               valid_shape=[-1, -1], compact=1)
    tt_f32_mat_b = pl.TileType(shape=[f32_tile, f32_tile], dtype=pl.DT_FP32,
                               target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                               valid_shape=[-1, -1], compact=1)
    tt_f32_left = pl.TileType(shape=[f32_tile, f32_tile], dtype=pl.DT_FP32,
                              target_memory=pl.MemorySpace.Left, layout=pl.NZ,
                              valid_shape=[-1, -1], compact=1)
    tt_f32_right = pl.TileType(shape=[f32_tile, f32_tile], dtype=pl.DT_FP32,
                               target_memory=pl.MemorySpace.Right, layout=pl.ZN,
                               valid_shape=[-1, -1], compact=1)
    tt_f32_acc = pl.TileType(shape=[f32_tile, f32_tile], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Acc, fractal=1024,
                             layout=pl.NZ, valid_shape=[-1, -1], compact=1)

    # Keep the FP32 Cube path single-buffered.  The dynamic-shape kernel is
    # launched repeatedly with different bs/de/h; a double-buffered ``next``
    # sequence can retain an unsafe in-flight slot across such launches.
    # Explicit bar_all() calls below serialize reuse without changing math.
    f32_l1_left_grp = pl.make_tile_group(type=tt_f32_mat_a, addrs=0x20000, mutex_ids=[10])
    f32_l1_right_grp = pl.make_tile_group(type=tt_f32_mat_b, addrs=0x28000, mutex_ids=[12])
    f32_left_grp = pl.make_tile_group(type=tt_f32_left, addrs=0x0000, mutex_ids=[14])
    f32_right_grp = pl.make_tile_group(type=tt_f32_right, addrs=0x0000, mutex_ids=[16])
    f32_acc_grp = pl.make_tile_group(type=tt_f32_acc, addrs=0x0000, mutex_ids=[18])

    with pl.section_cube():
        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=0,
                                 sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.system.set_mm_layout_transform(enabled=True)

        # 公式：══ Nest 1: grad_emb[bs, de] = grad_value@W_v_t + Σ_m grad_key_m@W_k_t[m] ══
        # 17 INDEPENDENT matmuls (each n_k_h=10 deep K-split), each Final + FIX store
        # to grad_emb_ws[i] (cover-write). Vector section sums all 17 slots.
        # NO atomicAdd, NO 170-deep Partial chain.

        # Value contribution also uses the legal 64x64 FP32 Cube tiles:
        # grad_value_ws @ wv_t_f32 -> grad_emb_ws[0].
        for tile_idx in pl.range(0, iters_f32_per_core):
            bsi = core_id + tile_idx * num_cores
            for ni in pl.range(0, n_n_de_f32):
                if bsi < n_bs_tiles_f32:
                    row_off = bsi * f32_tile
                    valid_bs = pl.min(f32_tile, bs_f32 - row_off)
                    col_off = ni * f32_tile
                    valid_n = pl.min(f32_tile, de - col_off)
                    pl.system.bar_all()
                    ac = f32_acc_grp.current()
                    pl.set_validshape(ac, [valid_bs, valid_n])
                    for k_idx in pl.range(0, n_k_h_f32):
                        k_off = k_idx * f32_tile
                        valid_k = pl.min(f32_tile, h - k_off)
                        left = f32_l1_left_grp.current()
                        right = f32_l1_right_grp.current()
                        pl.set_validshape(left, [valid_bs, valid_k])
                        pl.set_validshape(right, [valid_k, valid_n])
                        pl.load(left, grad_value_ws, [row_off, k_off])
                        pl.load(right, wv_t_f32, [k_off, col_off])
                        al = f32_left_grp.current()
                        br = f32_right_grp.current()
                        pl.set_validshape(al, [valid_bs, valid_k])
                        pl.set_validshape(br, [valid_k, valid_n])
                        pl.move(al, left)
                        pl.move(br, right)
                        if k_idx == 0 and k_idx == n_k_h_f32 - 1:
                            pl.matmul(ac, al, br)
                        elif k_idx == 0:
                            pl.matmul(ac, al, br)
                        elif k_idx == n_k_h_f32 - 1:
                            pl.matmul_acc(ac, ac, al, br)
                        else:
                            pl.matmul_acc(ac, ac, al, br)
                    pl.store(grad_emb_ws, ac, [0, row_off, col_off],
                             tile_dims=[1, 2])
                # Every core reaches the barrier, including tail owners that
                # have no valid bs tile.  This makes the tile-group reuse
                # deterministic for arbitrary bs, not only aligned cases.
                pl.system.bar_all()

        pl.system.bar_all()

        # Per-head key contributions use legal 64x64 FP32 Cube tiles:
        # grad_key_ws[m] @ wk_t_f32[m] -> grad_emb_ws[m + 1].
        for m_head in pl.range(0, m_h):
            for tile_idx in pl.range(0, iters_f32_per_core):
                bsi = core_id + tile_idx * num_cores
                for n_idx in pl.range(0, n_n_de_f32):
                    if bsi < n_bs_tiles_f32:
                        row_off = bsi * f32_tile
                        valid_bs = pl.min(f32_tile, bs_f32 - row_off)
                        col_off = n_idx * f32_tile
                        valid_n = pl.min(f32_tile, de - col_off)
                        pl.system.bar_all()
                        ac = f32_acc_grp.current()
                        pl.set_validshape(ac, [valid_bs, valid_n])
                        for k_idx in pl.range(0, n_k_h_f32):
                            k_off = k_idx * f32_tile
                            valid_k = pl.min(f32_tile, h - k_off)
                            left = f32_l1_left_grp.current()
                            right = f32_l1_right_grp.current()
                            pl.set_validshape(left, [valid_bs, valid_k])
                            pl.set_validshape(right, [valid_k, valid_n])
                            pl.load(left, grad_key_ws,
                                    [row_off, m_head, k_off], order=[0, 2])
                            pl.load(right, wk_t_f32,
                                    [m_head, k_off, col_off], order=[1, 2])
                            al = f32_left_grp.current()
                            br = f32_right_grp.current()
                            pl.set_validshape(al, [valid_bs, valid_k])
                            pl.set_validshape(br, [valid_k, valid_n])
                            pl.move(al, left)
                            pl.move(br, right)
                            if k_idx == 0 and k_idx == n_k_h_f32 - 1:
                                pl.matmul(ac, al, br)
                            elif k_idx == 0:
                                pl.matmul(ac, al, br)
                            elif k_idx == n_k_h_f32 - 1:
                                pl.matmul_acc(ac, ac, al, br)
                            else:
                                pl.matmul_acc(ac, ac, al, br)
                        pl.store(grad_emb_ws, ac, [m_head + 1, row_off, col_off],
                                 tile_dims=[1, 2])
                    # Keep the barrier outside the ownership predicate so
                    # tail tiles cannot deadlock the other cores.
                    pl.system.bar_all()
#
        pl.system.bar_all()

        # ══ Nest 2: split grad_W_v reduction into shorter FP32 Cube chains.
        # Each partial is stored in FP32 GM; Vector section reduces the
        # partials before the single low-precision output cast.  This avoids
        # a long BS-deep L0C accumulation chain for BS=8192/16384.
        total_wv_f32 = n_de_tiles_f32 * n_h_tiles_f32
        iters_wv_f32 = (total_wv_f32 + num_cores - 1) // num_cores
        for part_idx in pl.range(0, n_vpw_parts):
            part_start = part_idx * VPW_K_CHUNK
            part_len = pl.min(VPW_K_CHUNK, bs - part_start)
            n_k_part = (part_len + f32_tile - 1) // f32_tile
            for wv_idx in pl.range(0, iters_wv_f32):
                flat = core_id + wv_idx * num_cores
                if flat < total_wv_f32:
                    de_idx = flat // n_h_tiles_f32
                    h_idx = flat % n_h_tiles_f32
                    de_off = de_idx * f32_tile
                    h_off = h_idx * f32_tile
                    valid_de = pl.min(f32_tile, de - de_off)
                    valid_h = pl.min(f32_tile, h - h_off)
                    pl.system.bar_all()
                    ac = f32_acc_grp.current()
                    pl.set_validshape(ac, [valid_de, valid_h])
                    for k_idx in pl.range(0, n_k_part):
                        k_off = part_start + k_idx * f32_tile
                        valid_k = pl.min(f32_tile, bs - k_off)
                        valid_k = pl.min(valid_k, part_len - k_idx * f32_tile)
                        left = f32_l1_left_grp.current()
                        right = f32_l1_right_grp.current()
                        pl.set_validshape(left, [valid_de, valid_k])
                        pl.set_validshape(right, [valid_k, valid_h])
                        pl.load(left, emb_t_f32, [de_off, k_off])
                        pl.load(right, grad_value_ws, [k_off, h_off])
                        al = f32_left_grp.current()
                        br = f32_right_grp.current()
                        pl.set_validshape(al, [valid_de, valid_k])
                        pl.set_validshape(br, [valid_k, valid_h])
                        pl.move(al, left)
                        pl.move(br, right)
                        if k_idx == 0 and k_idx == n_k_part - 1:
                            pl.matmul(ac, al, br, phase=pl.AccPhase.Final)
                        elif k_idx == 0:
                            pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
                        elif k_idx == n_k_part - 1:
                            pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
                        else:
                            pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)
                    pl.store(grad_vpw_ws, ac, [part_idx, de_off, h_off],
                             tile_dims=[1, 2], phase=pl.STPhase.Final)

        pl.system.bar_all()

        # 公式：══ Nest 3: grad_W_k[m][de, h] = emb_t_f32 @ grad_key_ws[m] ══
        # This key path also uses 64x64 FP32 Cube tiles.  The final store
        # narrows the FP32 accumulator into the low-precision output tensor.
        total_wk_f32 = m_h * n_de_tiles_f32 * n_h_tiles_f32
        iters_wk_f32 = (total_wk_f32 + num_cores - 1) // num_cores
        for wk_idx in pl.range(0, iters_wk_f32):
            flat = core_id + wk_idx * num_cores
            if flat < total_wk_f32:
                m_head = flat // (n_de_tiles_f32 * n_h_tiles_f32)
                rem = flat % (n_de_tiles_f32 * n_h_tiles_f32)
                de_idx = rem // n_h_tiles_f32
                h_idx = rem % n_h_tiles_f32
                de_off = de_idx * f32_tile
                h_off = h_idx * f32_tile
                valid_de = pl.min(f32_tile, de - de_off)
                valid_h = pl.min(f32_tile, h - h_off)
                pl.system.bar_all()
                ac = f32_acc_grp.current()
                pl.set_validshape(ac, [valid_de, valid_h])
                for k_idx in pl.range(0, n_k_bs_f32):
                    k_off = k_idx * f32_tile
                    valid_k = pl.min(f32_tile, bs - k_off)
                    left = f32_l1_left_grp.current()
                    right = f32_l1_right_grp.current()
                    pl.set_validshape(left, [valid_de, valid_k])
                    pl.set_validshape(right, [valid_k, valid_h])
                    pl.load(left, emb_t_f32, [de_off, k_off])
                    pl.load(right, grad_key_ws,
                            [k_off, m_head, h_off], order=[0, 2])
                    al = f32_left_grp.current()
                    br = f32_right_grp.current()
                    pl.set_validshape(al, [valid_de, valid_k])
                    pl.set_validshape(br, [valid_k, valid_h])
                    pl.move(al, left)
                    pl.move(br, right)
                    if k_idx == 0 and k_idx == n_k_bs_f32 - 1:
                        pl.matmul(ac, al, br, phase=pl.AccPhase.Final)
                    elif k_idx == 0:
                        pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
                    elif k_idx == n_k_bs_f32 - 1:
                        pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
                    else:
                        pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)
                pl.store(grad_key_proj_weights, ac,
                         [m_head, de_off, h_off], tile_dims=[1, 2],
                         phase=pl.STPhase.Final)
#
        pl.system.set_mm_layout_transform(enabled=False)
        # ── cube -> vector handoff (Cube->V: M->FIX fence, then
        #    set FIX / wait MTE2) ──
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX,
                           event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX,
                           event_id=1)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=1,
                                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

    # ═══════════════════════════════════════════════════════════════
    #    # VECTOR SECTION 2: sum grad_emb_ws[0..m_h] -> grad_embeddings
#    # ═══════════════════════════════════════════════════════════════
    with pl.section_vector():
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=1,
                                     sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        # Cube has completed the GM stores at this point.  The two AIV
        # subblocks share the Vector MTE/UB pipeline, so serialize the
        # consumer section before selecting the single GM writer.
        pl.system.bar_all()
        # tile types for sum
        tt_sum_f32 = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32,
                                    target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
        tt_sum_f16 = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_BF16,
                                    target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
        sum_grp = pl.make_tile_group(type=tt_sum_f32, addrs=0x00000, mutex_ids=[0])
        slot_grp = pl.make_tile_group(type=tt_sum_f32, addrs=0x10000, mutex_ids=[1])
        out_grp = pl.make_tile_group(type=tt_sum_f16, addrs=0x20000, mutex_ids=[2])

        # Reduce the shorter FP32 grad_W_v Cube partials in Vector before
        # casting the final result to the output dtype.
        total_vpw = n_de_tiles_f32 * n_h_tiles_f32
        iters_vpw = (total_vpw + num_cores - 1) // num_cores
        if sub_id == 0:
            for vpw_idx in pl.range(0, iters_vpw):
                flat = core_id + vpw_idx * num_cores
                if flat < total_vpw:
                    de_idx = flat // n_h_tiles_f32
                    h_idx = flat % n_h_tiles_f32
                    de_off = de_idx * f32_tile
                    h_off = h_idx * f32_tile
                    valid_de = pl.min(f32_tile, de - de_off)
                    valid_h = pl.min(f32_tile, h - h_off)
                    acc = sum_grp.current()
                    pl.set_validshape(acc, [valid_de, valid_h])
                    for part_idx in pl.range(0, n_vpw_parts):
                        if part_idx == 0:
                            pl.load(acc, grad_vpw_ws,
                                    [part_idx, de_off, h_off], order=[1, 2])
                        else:
                            part = slot_grp.current()
                            pl.set_validshape(part, [valid_de, valid_h])
                            pl.load(part, grad_vpw_ws,
                                    [part_idx, de_off, h_off], order=[1, 2])
                            pl.add(acc, acc, part)
                    out = out_grp.current()
                    pl.set_validshape(out, [valid_de, valid_h])
                    pl.cast(out, acc, mode=pl.RoundMode.CAST_ROUND)
                    pl.store(grad_value_proj_weights, out, [de_off, h_off])

        pl.system.bar_all()

        total_sum = n_bs_tiles * n_n_de
        iters_sum = (total_sum + num_cores - 1) // num_cores
        if sub_id == 0:
            for sum_idx in pl.range(0, iters_sum):
                flat = core_id + sum_idx * num_cores
                if flat < total_sum:
                    bsi = flat // n_n_de
                    ni = flat % n_n_de
                    row_off = bsi * TILE_M
                    col_off = ni * TILE_N
                    valid_bs = pl.min(TILE_M, bs - row_off)
                    valid_n = pl.min(TILE_N, de - col_off)
                    acc = sum_grp.current()
                    pl.set_validshape(acc, [valid_bs, valid_n])
                    # Sum all m_h+1 slots
                    for slot_i in pl.range(0, m_h + 1):
                        if slot_i == 0:
                            pl.load(acc, grad_emb_ws, [slot_i, row_off, col_off],
                                    order=[1, 2])
                        else:
                            slot = slot_grp.current()
                            pl.set_validshape(slot, [valid_bs, valid_n])
                            pl.load(slot, grad_emb_ws, [slot_i, row_off, col_off],
                                    order=[1, 2])
                            pl.add(acc, acc, slot)
                    out = out_grp.current()
                    pl.set_validshape(out, [valid_bs, valid_n])
                    pl.cast(out, acc, mode=pl.RoundMode.CAST_ROUND)
                    pl.store(grad_embeddings, out, [row_off, col_off])

        # Do not let a following dynamic-shape launch observe unfinished
        # Vector/MTE3 work from this output section.
        pl.system.bar_all()


# ═══════════════════════════════════════════════════════════════════
# Layer E: Host wrapper
# ═══════════════════════════════════════════════════════════════════

def engram_backward_wrapper(
    grad_output,  # [b, s, m_h, h] BF16
    hidden_states,  # [b, s, m_h, h] BF16
    embeddings,  # [b, s, de] BF16
    key_proj_weights,  # [m_h, de, h] BF16 (host-pre-transposed below)
    value_proj_weights,  # [de, h] BF16 (host-pre-transposed below)
    key_gamma,  # [m_h, h] BF16
    query_gamma,  # [m_h, h] BF16
    scores,  # [b, s, m_h, 1] FP32
    gates,  # [b, s, m_h, 1] FP32
    keys,  # [b, s, m_h, h] BF16 cache; cast to FP32 in kernel
    value,  # [b, s, h] BF16 cache; cast to FP32 in kernel
    clamp_value=1e-6,
    eps=1e-6,
    use_kernel_transpose=True,  # API-compat flag; only host-pre-transpose path exists
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
    # Pad bs -> multiple of f32_tile(64): cube Nest1 must store grad_emb_ws as
    # FULL 64-row FP32 tiles.  An undersized tail tile (bs not a multiple of 64)
    # corrupts the FIX NZ->linear de-pad and scrambles the valid rows
    # (this was the grad_embeddings failure at small M).  Padded workspace rows
    # are zero; V2 and the final output still index by the real bs.
    bs_f32 = ((bs + 63) // 64) * 64
    de = embeddings.shape[-1]
    device = grad_output.device
    dtype = grad_output.dtype

    # ── Host reshape: merge B·S -> bs ──
    go_bs = grad_output.reshape(bs, m_dim, h).contiguous()
    hid_bs = hidden_states.reshape(bs, m_dim, h).contiguous()
    emb_bs = embeddings.reshape(bs, de).contiguous()
    keys_bs = keys.reshape(bs, m_dim, h).contiguous()
    val_bs = value.reshape(bs, h).contiguous()
    # Pad scores/gates 1->64 (forward stores score_back/gate_back as [.,.,64]).
    sc_bs = scores.reshape(bs, m_dim, 1).expand(-1, -1, 64).contiguous()
    gt_bs = gates.reshape(bs, m_dim, 1).expand(-1, -1, 64).contiguous()

    # ── Host pre-transpose (rule 3): kernel uses plain NZ loads ──
    emb_t_f32 = emb_bs.float().t().contiguous()  # [de, bs] FP32
    wv_t_f32 = value_proj_weights.float().t().contiguous()  # [h, de] FP32
    wk_t_f32 = key_proj_weights.float().transpose(-1, -2).contiguous()  # [m_h, h, de] FP32

    # ── FP32 intermediate workspace (cover-write; vector produces, cube consumes) ──
    # ZEROS (not empty): padded rows [bs, bs_f32) must read as 0 inside the cube
    # so the matmul yields full zero-padded tiles and grad_emb_ws stores cleanly.
    grad_value_ws = torch.zeros((bs_f32, h), dtype=torch.float32, device=device)
    grad_key_ws = torch.zeros((bs_f32, m_dim, h), dtype=torch.float32, device=device)
    n_vpw_parts = (bs + VPW_K_CHUNK - 1) // VPW_K_CHUNK
    grad_vpw_ws = torch.zeros((n_vpw_parts, de, h),
                              dtype=torch.float32, device=device)

    # ── num_cores (needed for per-core gamma workspace sizing) ──
    natural_cores = max(1, min(32, (bs + TILE_M - 1) // TILE_M))
    debug_cores = int(os.environ.get("ENGRAM_GAMMA_CORES", "0"))
    num_cores = (
        max(1, min(natural_cores, debug_cores))
        if debug_cores > 0 else natural_cores
    )

    # ── FP32 per-core gamma workspace; reduce in FP32, cast output once ──
    grad_qgamma_acc = torch.zeros((num_cores, m_dim, h), dtype=torch.float32, device=device)
    grad_kgamma_acc = torch.zeros((num_cores, m_dim, h), dtype=torch.float32, device=device)

    # ── FP32 grad_emb workspace (cube cover-write slots, vector sums to grad_embeddings) ──
    grad_emb_ws = torch.zeros((m_dim + 1, bs_f32, de), dtype=torch.float32, device=device)

    # ── Low-precision final outputs; all source calculations stay FP32 ──
    grad_hidden_states = torch.empty((bs, m_dim, h), dtype=dtype, device=device)
    grad_embeddings = torch.empty((bs, de), dtype=dtype, device=device)
    grad_kpw = torch.empty((m_dim, de, h), dtype=dtype, device=device)
    grad_vpw = torch.empty((de, h), dtype=dtype, device=device)

    engram_backward_kernel[None, num_cores](
        go_bs, hid_bs, emb_bs,
        key_gamma, query_gamma,
        sc_bs, gt_bs, keys_bs, val_bs,
        emb_t_f32, wv_t_f32, wk_t_f32,
        grad_value_ws, grad_key_ws,
        grad_vpw_ws,
        grad_qgamma_acc, grad_kgamma_acc, grad_emb_ws,
        grad_hidden_states, grad_embeddings, grad_kpw, grad_vpw,
    )
    torch.npu.synchronize()

    # ── Reduce FP32 per-core gamma workspace, then cast once at the output ──
    grad_kgamma_out = grad_kgamma_acc.sum(dim=0).to(dtype)
    grad_qgamma_out = grad_qgamma_acc.sum(dim=0).to(dtype)

    grad_hidden_states = grad_hidden_states.reshape(b, s, m_dim, h)
    grad_embeddings = grad_embeddings.reshape(b, s, de)
    grad_kpw = grad_kpw.reshape(m_dim, de, h)
    return (
        grad_hidden_states,
        grad_embeddings,
        grad_kpw,
        grad_vpw,
        grad_kgamma_out,
        grad_qgamma_out,
    )


# ═══════════════════════════════════════════════════════════════════
# Layer F: Self-validation
# ═══════════════════════════════════════════════════════════════════

def _validate(b=1, s=64, m_dim=4, h=1280, de=512):
    """Small-shape self-check against golden.
    On NPU machine:  python custom/engram_backward/engram_backward_impl.py
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from engram_backward_golden import (
        engram_backward_golden, engram_forward_with_cache,
        _get_device,
    )

    device = _get_device()
    torch.manual_seed(42)

    hidden_states = torch.randn(b, s, m_dim, h, dtype=torch.bfloat16, device=device)
    embeddings = torch.randn(b, s, de, dtype=torch.bfloat16, device=device)
    key_proj_weights = torch.randn(m_dim, de, h, dtype=torch.bfloat16, device=device) * 0.5
    value_proj_weights = torch.randn(de, h, dtype=torch.bfloat16, device=device) * 0.5
    key_gamma = torch.ones(m_dim, h, dtype=torch.bfloat16, device=device)
    query_gamma = torch.ones(m_dim, h, dtype=torch.bfloat16, device=device)

    clamp_value, eps = 1e-6, 1e-6
    with torch.no_grad():
        _, cache = engram_forward_with_cache(
            hidden_states, embeddings, key_proj_weights, value_proj_weights,
            key_gamma, query_gamma, clamp_value, eps,
        )

    grad_output = torch.randn(b, s, m_dim, h, dtype=torch.bfloat16, device=device)

    (ghs, ge, gkpw, gvpw, gkg, gqg) = engram_backward_wrapper(
        grad_output,
        hidden_states, embeddings, key_proj_weights, value_proj_weights,
        key_gamma, query_gamma,
        cache["scores"].float(), cache["gates"].float(),
        cache["keys"], cache["value"],
        clamp_value, eps,
    )

    (g_ghs, g_ge, g_gkpw, g_gvpw, g_gkg, g_gqg) = engram_backward_golden(
        grad_output,
        hidden_states, embeddings, key_proj_weights, value_proj_weights,
        key_gamma, query_gamma,
        cache["scores"].float(), cache["gates"].float(),
        cache["keys"], cache["value"],
        clamp_value, eps,
    )

    names = ["grad_hidden_states", "grad_embeddings", "grad_key_proj_weights",
             "grad_value_proj_weights", "grad_key_gamma", "grad_query_gamma"]
    npu_t = [ghs, ge, gkpw, gvpw, gkg, gqg]
    gold_t = [g_ghs, g_ge, g_gkpw, g_gvpw, g_gkg, g_gqg]

    log.info("=== engram_backward self-check (b=%s,s=%s,h=%s,de=%s) ===", b, s, h, de)
    all_ok = True
    atol, rtol = 0.001, 0.02
    for name, nt, gt in zip(names, npu_t, gold_t):
        nt = nt.cpu().float()
        gt = gt.cpu().float()
        max_diff = (nt - gt).abs().max().item()
        max_val = gt.abs().max().item() + 1e-8
        rel = max_diff / max_val
        ok = (max_diff <= atol) or (rel <= rtol)
        all_ok = all_ok and ok
        log.info("  %s: max_diff=%.4e  rel=%.4e  shape=%s  %s",
                 name, max_diff, rel, tuple(nt.shape), "OK" if ok else "FAIL")
    tag = "[PRECISION_PASS]" if all_ok else "[PRECISION_FAIL]"
    log.info("  => %s %s", "ALL OK" if all_ok else "FAIL", tag)
    return all_ok


if __name__ == "__main__":
    ok = _validate(b=1, s=128, h=1280, de=512)
    if not ok:
        raise SystemExit(1)
    log.info("=== Small shape PASS ===")
    raise SystemExit(0)
