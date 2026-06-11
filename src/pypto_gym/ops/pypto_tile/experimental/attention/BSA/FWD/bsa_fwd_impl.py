#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You can not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
BSA Forward PyPTO Kernel Implementation (Single-Phase, Auto-Configured)

Dynamic axes: B, Hq, Hkv, Sq, Skv (the 5 primitive shape dimensions that vary
across test cases). Derived values (numQB, maxSel, BH, Sq_pad, etc.) are
carried via hint tensors or derived from data tensor shapes inside the kernel.

Single jit kernel with nested loops:
  - Outer loop (LOOP_fwd_outer): TOTAL_OUTER = BH * numQB iterations, parallel
  - Inner loop (LOOP_fwd_kblk): maxSel iterations, sequential (online softmax)

Each outer iteration processes one Q block across all its valid KV blocks,
computing the full online softmax accumulation (m, l, o) and writing the
final O and LSE outputs.

Auto-Configuration (Sq-based):
  When Sq >= _FWD_PERF_THRESHOLD_SQ (1024), the kernel automatically uses
  optimized l1/sched settings (l1=64, sched=1) that improve S1024 performance
  by ~13.5% (kernel task time). For Sq < 1024, the default settings
  (l1=16, sched=3) are used, which are optimal for S256/S512.
  The kernel cache stores separate compiled binaries for each config combination.

Performance History:
  Baseline (l1=16, sched=3) — the original single-phase approach:
    S256 ≈ 149us/37.5%, S512 ≈ 146us/34.2%, S1024 ≈ 1.04ms/34.1%, S2048 ≈ 1.95ms/22.5%

  Optimized (l1=64, sched=1) for S1024+:
    S1024 ≈ 0.90ms/37.4% (+13.5% kernel improvement, +3.3% util)
    S2048 ≈ 1.96ms/22.2% (no significant change)

  Attempted alternatives that were rejected:
    - Concurrent (per-BH streams): +110% slower (S1024), stream dispatch overhead
    - Two-kernel flash: +343% slower, kernel dispatch + sync overhead
    - Single-kernel flash: NaN/inf for BH>4, PyPTO tile alignment hard constraint
    - BH=16 tuning: only +1.9% util improvement
"""

import math
import os

import torch
import pypto
from torch._dynamo import allow_in_graph

from bsa_common import (
    DEFAULT_CONFIG, SparseKvBuildConfig,
    _pad_to_block_aligned, _build_sparse_kv_cached,
    _make_jit_opts,
    _VEC_TILE_LOAD, _CUBE_TILE, _CUBE_TILE_LIST,
)

# ===========================================================================
# Dynamic Axis Symbols (B, Hq, Hkv, Sq, Skv + derived)
# ===========================================================================
DYNAMIC_B = pypto.frontend.dynamic('DYNAMIC_B')
DYNAMIC_Hq = pypto.frontend.dynamic('DYNAMIC_Hq')
DYNAMIC_Hkv = pypto.frontend.dynamic('DYNAMIC_Hkv')
DYNAMIC_Sq = pypto.frontend.dynamic('DYNAMIC_Sq')
DYNAMIC_Skv = pypto.frontend.dynamic('DYNAMIC_Skv')
DYNAMIC_numQB = pypto.frontend.dynamic('DYNAMIC_numQB')
DYNAMIC_maxSel = pypto.frontend.dynamic('DYNAMIC_maxSel')
DYNAMIC_BH = pypto.frontend.dynamic('DYNAMIC_BH')
DYNAMIC_Sq_pad = pypto.frontend.dynamic('DYNAMIC_Sq_pad')
DYNAMIC_TotalQ = pypto.frontend.dynamic('DYNAMIC_TotalQ')
DYNAMIC_TotalKV = pypto.frontend.dynamic('DYNAMIC_TotalKV')
DYNAMIC_TotalMask = pypto.frontend.dynamic('DYNAMIC_TotalMask')

_fwd_cache = {}
_PERF_OUTPUT_BASE = os.path.abspath(os.path.join(os.getcwd(), "output"))

# Sq threshold for auto-switching to optimized l1/sched config.
# When Sq >= this value, l1=64/sched=1 is used (13.5% kernel improvement for S1024).
# When Sq < this value, default l1=16/sched=3 is used (optimal for S256/S512).
_FWD_PERF_THRESHOLD_SQ = 1024


def _snapshot_output_dirs():
    if not os.path.isdir(_PERF_OUTPUT_BASE):
        return set()
    return {d for d in os.listdir(_PERF_OUTPUT_BASE)
            if os.path.isdir(os.path.join(_PERF_OUTPUT_BASE, d))}


def _find_newest_created_dir(before):
    after = _snapshot_output_dirs()
    created = after - before
    best_dir, best_mt = None, 0.0
    for name in created:
        d = os.path.join(_PERF_OUTPUT_BASE, name)
        swim = os.path.join(d, "merged_swimlane.json")
        if os.path.isfile(swim):
            mt = os.path.getmtime(swim)
            if mt > best_mt:
                best_dir, best_mt = d, mt
    return best_dir


last_forward_perf_dir = None


def _get_fwd_kernel(cfg, *, extra_pass_options=None, extra_runtime_options=None):
    """Return the single-phase FWD kernel (nested loops, online softmax).

    The kernel cache key includes the effective l1 and sched settings so that
    different configurations (e.g. l1=16/sched=3 vs l1=64/sched=1) get their
    own compiled binaries.
    """
    # Build hashable cache key from effective configuration
    from bsa_common import _CUBE_L1_REUSE_SETTING, _DEVICE_SCHED_MODE
    _eff_l1 = (extra_pass_options or {}).get('cube_l1_reuse_setting', _CUBE_L1_REUSE_SETTING)
    _eff_sched = (extra_runtime_options or {}).get('device_sched_mode', _DEVICE_SCHED_MODE)
    _l1_val = _eff_l1.get(-1, 16) if isinstance(_eff_l1, dict) else _eff_l1
    key = ("fwd", _l1_val, _eff_sched)
    if key in _fwd_cache:
        return _fwd_cache[key]

    before = _snapshot_output_dirs()

    bx = cfg.block_shape_x
    by = cfg.block_shape_y
    D = cfg.head_dim
    BLOCK = bx
    SUB_SPLIT = 1
    SUB_BLOCK = bx // SUB_SPLIT  # = bx when SUB_SPLIT=1
    KV_BLOCK = by
    ct = _CUBE_TILE_LIST
    vtl = _VEC_TILE_LOAD

    jit_opts = _make_jit_opts(cfg, extra_pass_options=extra_pass_options,
                              extra_runtime_options=extra_runtime_options)

    @pypto.frontend.jit(**jit_opts)
    def fwd_kernel(
        b_hint: pypto.Tensor([DYNAMIC_B, 1], pypto.DT_FP32),
        hq_hint: pypto.Tensor([DYNAMIC_Hq, 1], pypto.DT_FP32),
        hkv_hint: pypto.Tensor([DYNAMIC_Hkv, 1], pypto.DT_FP32),
        sq_hint: pypto.Tensor([DYNAMIC_Sq, 1], pypto.DT_FP32),
        skv_hint: pypto.Tensor([DYNAMIC_Skv, 1], pypto.DT_FP32),
        numqb_hint: pypto.Tensor([DYNAMIC_numQB, 1], pypto.DT_FP32),
        maxsel_hint: pypto.Tensor([DYNAMIC_maxSel, 1], pypto.DT_FP32),
        q_2d: pypto.Tensor([DYNAMIC_TotalQ, D], pypto.DT_FP16),
        k_compact: pypto.Tensor([DYNAMIC_TotalKV, D], pypto.DT_FP16),
        v_compact: pypto.Tensor([DYNAMIC_TotalKV, D], pypto.DT_FP16),
        scaled_mask: pypto.Tensor([DYNAMIC_TotalMask, KV_BLOCK], pypto.DT_FP16),
        neg_inf_mask: pypto.Tensor([DYNAMIC_TotalMask, KV_BLOCK], pypto.DT_FP16),
        output_3d: pypto.Tensor([DYNAMIC_BH, DYNAMIC_Sq_pad, D], pypto.DT_FP16),
        lse_2d: pypto.Tensor([DYNAMIC_BH, DYNAMIC_Sq_pad], pypto.DT_FP32),
    ):
        dtype = q_2d.dtype

        BH = output_3d.shape[0]
        numQB = numqb_hint.shape[0]
        maxSel = maxsel_hint.shape[0]
        TOTAL_OUTER = BH * numQB

        for outer_local in pypto.loop(TOTAL_OUTER, name="LOOP_fwd_outer",
                                        idx_name="outer_local_idx", parallel=True):
            u = outer_local % numQB
            bh_ofs = outer_local // numQB

            q_row_ofs = outer_local * BLOCK

            mi_acc = pypto.tensor([SUB_BLOCK, 1], pypto.DT_FP32, "mi_acc")
            li_acc = pypto.tensor([SUB_BLOCK, 1], pypto.DT_FP32, "li_acc")
            oi_acc = pypto.tensor([SUB_BLOCK, D], pypto.DT_FP32, "oi_acc")

            for v_idx in pypto.loop(maxSel, name="LOOP_fwd_kblk",
                                    idx_name="kblk_idx"):
                kv_row_ofs = outer_local * maxSel * KV_BLOCK + v_idx * KV_BLOCK

                pypto.set_vec_tile_shapes(*vtl)
                q_sub = pypto.view(q_2d, [SUB_BLOCK, D], [q_row_ofs, 0])
                k_block = pypto.view(k_compact, [KV_BLOCK, D], [kv_row_ofs, 0])
                v_block = pypto.view(v_compact, [KV_BLOCK, D], [kv_row_ofs, 0])

                pypto.set_cube_tile_shapes(ct, ct, ct)
                S = pypto.matmul(q_sub, k_block, pypto.DT_FP32,
                                a_trans=False, b_trans=True)

                mask_row_ofs = outer_local * maxSel * BLOCK + v_idx * BLOCK
                scaled_mask_block = pypto.view(scaled_mask, [SUB_BLOCK, KV_BLOCK],
                                                [mask_row_ofs, 0])
                neg_inf_block = pypto.view(neg_inf_mask, [SUB_BLOCK, KV_BLOCK],
                                            [mask_row_ofs, 0])
                scaled_mask_fp32 = pypto.cast(scaled_mask_block, pypto.DT_FP32)
                neg_inf_fp32 = pypto.cast(neg_inf_block, pypto.DT_FP32)
                S_masked = pypto.add(pypto.mul(S, scaled_mask_fp32), neg_inf_fp32)

                m_ij = pypto.amax(S_masked, dim=-1, keepdim=True)
                P_ij = pypto.exp(pypto.sub(S_masked, m_ij))
                l_ij = pypto.sum(P_ij, dim=-1, keepdim=True)
                P_ij_fp16 = pypto.cast(P_ij, dtype)
                pypto.set_cube_tile_shapes(ct, ct, ct)
                o_ij = pypto.matmul(P_ij_fp16, v_block, pypto.DT_FP32)

                if pypto.is_loop_begin(v_idx):
                    if pypto.is_loop_end(v_idx):
                        O_final = pypto.div(o_ij, l_ij)
                        O_reshaped = pypto.reshape(O_final, [1, SUB_BLOCK, D])
                        pypto.set_vec_tile_shapes(1, 128, 128)
                        O_cast = pypto.cast(O_reshaped, dtype)
                        pypto.assemble(O_cast, [bh_ofs, u * BLOCK, 0], output_3d)
                        lse_val = pypto.add(m_ij, pypto.log(l_ij))
                        lse_cast = pypto.reshape(lse_val, [1, SUB_BLOCK])
                        pypto.set_vec_tile_shapes(1, 128)
                        pypto.assemble(lse_cast, [bh_ofs, u * BLOCK], lse_2d)
                    else:
                        oi_acc[:] = o_ij
                    li_acc[:] = l_ij
                    mi_acc[:] = m_ij
                else:
                    mi_new = pypto.maximum(mi_acc, m_ij)
                    alpha = pypto.exp(pypto.sub(mi_acc, mi_new))
                    beta = pypto.exp(pypto.sub(m_ij, mi_new))
                    li_new = pypto.add(pypto.mul(alpha, li_acc), pypto.mul(beta, l_ij))
                    oi_scaled = pypto.mul(oi_acc, alpha)
                    o_ij_scaled = pypto.mul(o_ij, beta)
                    oi_new = pypto.add(oi_scaled, o_ij_scaled)
                    if pypto.is_loop_end(v_idx):
                        O_final = pypto.div(oi_new, li_new)
                        O_reshaped = pypto.reshape(O_final, [1, SUB_BLOCK, D])
                        pypto.set_vec_tile_shapes(1, 128, 128)
                        O_cast = pypto.cast(O_reshaped, dtype)
                        pypto.assemble(O_cast, [bh_ofs, u * BLOCK, 0], output_3d)
                        lse_val = pypto.add(mi_new, pypto.log(li_new))
                        lse_cast = pypto.reshape(lse_val, [1, SUB_BLOCK])
                        pypto.set_vec_tile_shapes(1, 128)
                        pypto.assemble(lse_cast, [bh_ofs, u * BLOCK], lse_2d)
                    else:
                        oi_acc[:] = oi_new
                    li_acc[:] = li_new
                    mi_acc[:] = mi_new

    output_dir = _find_newest_created_dir(before)
    _fwd_cache[key] = (fwd_kernel, output_dir)
    return _fwd_cache[key]





@allow_in_graph
def block_sparse_attention_forward(
    query, key, value, block_sparse_mask,
    actual_seq_lengths=None, actual_seq_lengths_kv=None,
    block_shape=None, cfg=DEFAULT_CONFIG,
    extra_pass_options=None, extra_runtime_options=None,
):
    """BSA forward with auto-configuration.

    When Sq >= _FWD_PERF_THRESHOLD_SQ (1024) and no explicit options are
    provided, automatically uses optimized l1=64/sched=1 settings for ~13.5%
    kernel improvement. Explicit extra_pass_options/extra_runtime_options
    override the auto-config.
    """
    global last_forward_perf_dir
    last_forward_perf_dir = None

    bx = block_shape[0] if block_shape else cfg.block_shape_x
    by = block_shape[1] if block_shape else cfg.block_shape_y

    B, Hq, Sq, D = query.shape
    _, Hkv, Skv, _ = key.shape
    assert D == cfg.head_dim

    # Auto-configuration: select optimized l1/sched for long sequences
    # Only auto-configure when caller hasn't explicitly provided options
    if extra_pass_options is None and Sq >= _FWD_PERF_THRESHOLD_SQ:
        extra_pass_options = {'cube_l1_reuse_setting': {-1: 64}}
    if extra_runtime_options is None and Sq >= _FWD_PERF_THRESHOLD_SQ:
        extra_runtime_options = {'device_sched_mode': 1}

    numQB = math.ceil(Sq / bx)
    numKB = math.ceil(Skv / by)
    Sq_pad = numQB * bx
    Skv_pad = numKB * by

    Q_pad, _ = _pad_to_block_aligned(query, bx)
    K_pad, _ = _pad_to_block_aligned(key, by)
    V_pad, _ = _pad_to_block_aligned(value, by)

    q_2d = Q_pad.reshape(B * Hq * Sq_pad, D)
    k_2d = K_pad.reshape(B * Hkv * Skv_pad, D)
    v_2d = V_pad.reshape(B * Hkv * Skv_pad, D)

    output_3d = torch.zeros(B * Hq, Sq_pad, D, dtype=cfg.torch_dtype, device=query.device)
    lse_2d = torch.full([B * Hq, Sq_pad], cfg.lse_init, dtype=cfg.accum_torch_dtype, device=query.device)

    k_compact, v_compact, valid_mask, maxSel = _build_sparse_kv_cached(SparseKvBuildConfig(
        block_sparse_mask=block_sparse_mask, k_2d=k_2d, v_2d=v_2d,
        B=B, Hq=Hq, Hkv=Hkv, Sq=Sq, Skv=Skv, Sq_pad=Sq_pad, Skv_pad=Skv_pad,
        numQB=numQB, numKB=numKB, bx=bx, by=by, D=D, device=query.device))

    softmax_scale = D ** -0.5
    large_neg = cfg.large_neg
    scaled_mask = (valid_mask * softmax_scale).to(torch.float16)
    neg_inf_mask = ((1.0 - valid_mask) * large_neg).to(torch.float16)

    b_hint = torch.zeros(B, 1, dtype=torch.float32, device=query.device)
    hq_hint = torch.zeros(Hq, 1, dtype=torch.float32, device=query.device)
    hkv_hint = torch.zeros(Hkv, 1, dtype=torch.float32, device=query.device)
    sq_hint = torch.zeros(Sq, 1, dtype=torch.float32, device=query.device)
    skv_hint = torch.zeros(Skv, 1, dtype=torch.float32, device=query.device)
    numqb_hint = torch.zeros(numQB, 1, dtype=torch.float32, device=query.device)
    maxsel_hint = torch.zeros(maxSel, 1, dtype=torch.float32, device=query.device)

    kernel_fn, kernel_dir = _get_fwd_kernel(cfg,
        extra_pass_options=extra_pass_options,
        extra_runtime_options=extra_runtime_options)
    before = _snapshot_output_dirs()
    kernel_fn(
        b_hint, hq_hint, hkv_hint, sq_hint, skv_hint,
        numqb_hint, maxsel_hint,
        q_2d, k_compact, v_compact, scaled_mask, neg_inf_mask,
        output_3d, lse_2d)
    new_dir = _find_newest_created_dir(before)
    if new_dir:
        # Update cache with config-aware key (same logic as _get_fwd_kernel)
        from bsa_common import _CUBE_L1_REUSE_SETTING, _DEVICE_SCHED_MODE
        _eff_l1 = (extra_pass_options or {}).get('cube_l1_reuse_setting', _CUBE_L1_REUSE_SETTING)
        _eff_sched = (extra_runtime_options or {}).get('device_sched_mode', _DEVICE_SCHED_MODE)
        _l1_val = _eff_l1.get(-1, 16) if isinstance(_eff_l1, dict) else _eff_l1
        cache_key = ("fwd", _l1_val, _eff_sched)
        _fwd_cache[cache_key] = (kernel_fn, new_dir)
        kernel_dir = new_dir

    last_forward_perf_dir = kernel_dir

    attention_out = output_3d[:, :Sq, :].reshape(B, Hq, Sq, D)
    softmax_lse = lse_2d[:, :Sq].reshape(B, Hq, Sq)
    return attention_out, softmax_lse


# ===========================================================================
# A1: Auto-Dispatch (delegates to auto-configured baseline)
# ===========================================================================
# NOTE: Concurrent mode was evaluated and proven 110% slower for S1024
# (stream dispatch overhead far exceeds AICore utilization gains).
# block_sparse_attention_forward now auto-configures l1/sched based on Sq,
# so this auto-dispatch simply delegates to it.


@allow_in_graph
def block_sparse_attention_forward_auto(
    query, key, value, block_sparse_mask,
    actual_seq_lengths=None, actual_seq_lengths_kv=None,
    block_shape=None, cfg=DEFAULT_CONFIG,
    extra_pass_options=None, extra_runtime_options=None,
):
    """Auto-dispatch: delegates to auto-configured baseline.

    block_sparse_attention_forward now auto-selects l1/sched based on Sq,
    so this function simply calls it directly. The concurrent mode was
    evaluated and proven significantly slower (stream dispatch overhead).
    """
    return block_sparse_attention_forward(
        query, key, value, block_sparse_mask,
        actual_seq_lengths, actual_seq_lengths_kv,
        block_shape, cfg, extra_pass_options, extra_runtime_options)


@allow_in_graph
def block_sparse_attention_forward_concurrent(
    query, key, value, block_sparse_mask,
    actual_seq_lengths=None, actual_seq_lengths_kv=None,
    block_shape=None, cfg=DEFAULT_CONFIG,
    extra_pass_options=None, extra_runtime_options=None,
):
    """Per-BH concurrent FWD: launch B*Hq kernels on separate NPU streams,
    each using the single fwd_kernel with BH=1 slices.
    """
    global last_forward_perf_dir
    last_forward_perf_dir = None

    bx = block_shape[0] if block_shape else cfg.block_shape_x
    by = block_shape[1] if block_shape else cfg.block_shape_y

    B, Hq, Sq, D = query.shape
    _, Hkv, Skv, _ = key.shape
    assert D == cfg.head_dim

    numQB = math.ceil(Sq / bx)
    numKB = math.ceil(Skv / by)
    Sq_pad = numQB * bx
    Skv_pad = numKB * by
    BH = B * Hq

    Q_pad, _ = _pad_to_block_aligned(query, bx)
    K_pad, _ = _pad_to_block_aligned(key, by)
    V_pad, _ = _pad_to_block_aligned(value, by)

    q_2d = Q_pad.reshape(B * Hq * Sq_pad, D)
    k_2d = K_pad.reshape(B * Hkv * Skv_pad, D)
    v_2d = V_pad.reshape(B * Hkv * Skv_pad, D)

    output_3d = torch.zeros(B * Hq, Sq_pad, D, dtype=cfg.torch_dtype, device=query.device)
    lse_2d = torch.full([B * Hq, Sq_pad], cfg.lse_init, dtype=cfg.accum_torch_dtype, device=query.device)

    k_compact, v_compact, valid_mask, maxSel = _build_sparse_kv_cached(SparseKvBuildConfig(
        block_sparse_mask=block_sparse_mask, k_2d=k_2d, v_2d=v_2d,
        B=B, Hq=Hq, Hkv=Hkv, Sq=Sq, Skv=Skv, Sq_pad=Sq_pad, Skv_pad=Skv_pad,
        numQB=numQB, numKB=numKB, bx=bx, by=by, D=D, device=query.device))

    softmax_scale = D ** -0.5
    large_neg = cfg.large_neg
    scaled_mask = (valid_mask * softmax_scale).to(torch.float16)
    neg_inf_mask = ((1.0 - valid_mask) * large_neg).to(torch.float16)

    # Hint tensors shared across all BH slices (dimensions don't change per BH)
    b_hint_1 = torch.zeros(1, 1, dtype=torch.float32, device=query.device)
    hq_hint_1 = torch.zeros(1, 1, dtype=torch.float32, device=query.device)
    hkv_hint = torch.zeros(Hkv, 1, dtype=torch.float32, device=query.device)
    sq_hint = torch.zeros(Sq, 1, dtype=torch.float32, device=query.device)
    skv_hint = torch.zeros(Skv, 1, dtype=torch.float32, device=query.device)
    numqb_hint = torch.zeros(numQB, 1, dtype=torch.float32, device=query.device)
    maxsel_hint = torch.zeros(maxSel, 1, dtype=torch.float32, device=query.device)

    kernel_fn, kernel_dir = _get_fwd_kernel(cfg,
        extra_pass_options=extra_pass_options,
        extra_runtime_options=extra_runtime_options)
    before = _snapshot_output_dirs()

    streams = [torch.npu.Stream() for _ in range(BH)]
    bh_stride_q = Sq_pad
    bh_stride_kv = numQB * maxSel * by
    bh_stride_mask = numQB * maxSel * bx

    for bh_idx in range(BH):
        q_bh = q_2d[bh_idx * bh_stride_q: bh_idx * bh_stride_q + Sq_pad]
        k_bh = k_compact[bh_idx * bh_stride_kv: bh_idx * bh_stride_kv + numQB * maxSel * by]
        v_bh = v_compact[bh_idx * bh_stride_kv: bh_idx * bh_stride_kv + numQB * maxSel * by]
        sm_bh = scaled_mask[bh_idx * bh_stride_mask: bh_idx * bh_stride_mask + numQB * maxSel * bx]
        nm_bh = neg_inf_mask[bh_idx * bh_stride_mask: bh_idx * bh_stride_mask + numQB * maxSel * bx]
        out_bh = output_3d[bh_idx: bh_idx + 1]
        lse_bh = lse_2d[bh_idx: bh_idx + 1]

        with torch.npu.Stream(streams[bh_idx]):
            kernel_fn(
                b_hint_1, hq_hint_1, hkv_hint, sq_hint, skv_hint,
                numqb_hint, maxsel_hint,
                q_bh, k_bh, v_bh, sm_bh, nm_bh,
                out_bh, lse_bh)

    torch.npu.synchronize()

    new_dir = _find_newest_created_dir(before)
    if new_dir:
        from bsa_common import _CUBE_L1_REUSE_SETTING, _DEVICE_SCHED_MODE
        _eff_l1 = (extra_pass_options or {}).get('cube_l1_reuse_setting', _CUBE_L1_REUSE_SETTING)
        _eff_sched = (extra_runtime_options or {}).get('device_sched_mode', _DEVICE_SCHED_MODE)
        _l1_val = _eff_l1.get(-1, 16) if isinstance(_eff_l1, dict) else _eff_l1
        cache_key = ("fwd", _l1_val, _eff_sched)
        _fwd_cache[cache_key] = (kernel_fn, new_dir)
        kernel_dir = new_dir

    last_forward_perf_dir = kernel_dir

    attention_out = output_3d[:, :Sq, :].reshape(B, Hq, Sq, D)
    softmax_lse = lse_2d[:, :Sq].reshape(B, Hq, Sq)
    return attention_out, softmax_lse


