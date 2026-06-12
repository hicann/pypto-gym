#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
BSA Backward PyPTO Kernel Implementation (Fully Dynamic Axis Pattern)

Dynamic axes: B, Hq, Hkv, Sq, Skv (the 5 primitive shape dimensions that vary
across test cases). Derived values (numQB, numKB, max_sel, maxInner, BH, BHKV,
Sq_pad, Skv_pad) are computed inside the kernel from hint tensor shapes.

Single compiled kernel handles ALL (B, Hq, Hkv, Sq, Skv) combinations — no
per-shape re-compilation (cache key = ("bwd") only).

Performance optimizations:
  B0-1: Mask preprocessing — scaled_mask + neg_inf_mask precomputed in wrapper
  B0-2: D_row precomputed outside kernel (eliminates mul+cast+sum)
  B0-3: Redundant inv_mask eliminated — P already masked via S_masked
  B0-4: Simplified accumulation — consistent iteration graph for stitch merge
  B0-5: parallel=True on outer loops for multi-core scheduling
"""

import math
import os

import torch
import pypto
from torch._dynamo import allow_in_graph

from bsa_common import (
    DEFAULT_CONFIG, SparseKvBuildConfig,
    _pad_to_block_aligned, _build_sparse_kv_cached,
    _build_sparse_q_dkdv_cached,
    _make_jit_opts,
    _VEC_TILE_LOAD, _CUBE_TILE, _CUBE_TILE_LIST,
)

# ===========================================================================
# Dynamic Axis Symbols (B, Hq, Hkv, Sq, Skv + derived)
# ===========================================================================
# 5 primitive dynamic axes (per user requirement)
DYNAMIC_B = pypto.frontend.dynamic('DYNAMIC_B')
DYNAMIC_Hq = pypto.frontend.dynamic('DYNAMIC_Hq')
DYNAMIC_Hkv = pypto.frontend.dynamic('DYNAMIC_Hkv')
DYNAMIC_Sq = pypto.frontend.dynamic('DYNAMIC_Sq')
DYNAMIC_Skv = pypto.frontend.dynamic('DYNAMIC_Skv')

# Derived dynamic axes for loop bounds (carried via hint tensors)
DYNAMIC_numQB = pypto.frontend.dynamic('DYNAMIC_numQB')
DYNAMIC_numKB = pypto.frontend.dynamic('DYNAMIC_numKB')
DYNAMIC_maxSel = pypto.frontend.dynamic('DYNAMIC_maxSel')
DYNAMIC_maxInner = pypto.frontend.dynamic('DYNAMIC_maxInner')

# Derived dynamic axes for tensor dimension symbols
DYNAMIC_TotalQ = pypto.frontend.dynamic('DYNAMIC_TotalQ')
DYNAMIC_TotalKV_compact = pypto.frontend.dynamic('DYNAMIC_TotalKV_compact')
DYNAMIC_TotalMask = pypto.frontend.dynamic('DYNAMIC_TotalMask')
DYNAMIC_TotalQ_dkdv = pypto.frontend.dynamic('DYNAMIC_TotalQ_dkdv')
DYNAMIC_TotalMask_dkdv = pypto.frontend.dynamic('DYNAMIC_TotalMask_dkdv')
DYNAMIC_TotalKV = pypto.frontend.dynamic('DYNAMIC_TotalKV')

_BWD_PASS_OPTS = {
    "cube_l1_reuse_setting": {-1: 64},
}
_BWD_RT_OPTS = {}

_bwd_cache = {}

_PERF_OUTPUT_BASE = os.path.abspath(os.path.join(os.getcwd(), "output"))


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


last_backward_perf_dirs = {}


def _get_bwd_kernel(cfg):
    key = ("bwd",)
    if key in _bwd_cache:
        return _bwd_cache[key]

    before = _snapshot_output_dirs()

    softmax_scale = cfg.softmax_scale
    bx = cfg.block_shape_x
    by = cfg.block_shape_y
    D = cfg.head_dim
    large_neg = cfg.large_neg
    BLOCK = bx
    KV_BLOCK = by
    SUB_SPLIT = bx // 64
    SUB_BLOCK = bx // SUB_SPLIT
    ct = _CUBE_TILE_LIST
    vtl = _VEC_TILE_LOAD

    @pypto.frontend.jit(**_make_jit_opts(cfg, extra_pass_options=_BWD_PASS_OPTS,
                                          extra_runtime_options=_BWD_RT_OPTS))
    def kernel(
        # --- Hint tensors for 5 primitive dynamic axes ---
        b_hint: pypto.Tensor([DYNAMIC_B, 1], pypto.DT_FP32),
        hq_hint: pypto.Tensor([DYNAMIC_Hq, 1], pypto.DT_FP32),
        hkv_hint: pypto.Tensor([DYNAMIC_Hkv, 1], pypto.DT_FP32),
        sq_hint: pypto.Tensor([DYNAMIC_Sq, 1], pypto.DT_FP32),
        skv_hint: pypto.Tensor([DYNAMIC_Skv, 1], pypto.DT_FP32),
        # --- Hint tensors for derived loop-bound values ---
        numqb_hint: pypto.Tensor([DYNAMIC_numQB, 1], pypto.DT_FP32),
        numkb_hint: pypto.Tensor([DYNAMIC_numKB, 1], pypto.DT_FP32),
        maxsel_hint: pypto.Tensor([DYNAMIC_maxSel, 1], pypto.DT_FP32),
        maxinner_hint: pypto.Tensor([DYNAMIC_maxInner, 1], pypto.DT_FP32),
        # --- Data tensors ---
        q_2d: pypto.Tensor([DYNAMIC_TotalQ, D], pypto.DT_FP16),
        k_compact: pypto.Tensor([DYNAMIC_TotalKV_compact, D], pypto.DT_FP16),
        v_compact: pypto.Tensor([DYNAMIC_TotalKV_compact, D], pypto.DT_FP16),
        do_2d: pypto.Tensor([DYNAMIC_TotalQ, D], pypto.DT_FP16),
        lse_2d: pypto.Tensor([DYNAMIC_TotalQ, 1], pypto.DT_FP32),
        scaled_mask_dq: pypto.Tensor([DYNAMIC_TotalMask, KV_BLOCK], pypto.DT_FP16),
        neg_inf_mask_dq: pypto.Tensor([DYNAMIC_TotalMask, KV_BLOCK], pypto.DT_FP16),
        d_row_dq: pypto.Tensor([DYNAMIC_TotalQ, 1], pypto.DT_FP32),
        dq_2d: pypto.Tensor([DYNAMIC_TotalQ, D], pypto.DT_FP16),
        q_compact: pypto.Tensor([DYNAMIC_TotalQ_dkdv, D], pypto.DT_FP16),
        do_compact: pypto.Tensor([DYNAMIC_TotalQ_dkdv, D], pypto.DT_FP16),
        lse_compact: pypto.Tensor([DYNAMIC_TotalQ_dkdv, 1], pypto.DT_FP32),
        scaled_mask_dkdv: pypto.Tensor([DYNAMIC_TotalMask_dkdv, KV_BLOCK], pypto.DT_FP16),
        neg_inf_mask_dkdv: pypto.Tensor([DYNAMIC_TotalMask_dkdv, KV_BLOCK], pypto.DT_FP16),
        d_row_dkdv: pypto.Tensor([DYNAMIC_TotalQ_dkdv, 1], pypto.DT_FP32),
        k_2d: pypto.Tensor([DYNAMIC_TotalKV, D], pypto.DT_FP16),
        v_2d: pypto.Tensor([DYNAMIC_TotalKV, D], pypto.DT_FP16),
        dk_2d: pypto.Tensor([DYNAMIC_TotalKV, D], pypto.DT_FP16),
        dv_2d: pypto.Tensor([DYNAMIC_TotalKV, D], pypto.DT_FP16),
    ):
        dtype = q_2d.dtype

        # Derive runtime dimensions from hint tensors
        B = b_hint.shape[0]
        Hq = hq_hint.shape[0]
        Hkv = hkv_hint.shape[0]
        BH = B * Hq
        BHKV = B * Hkv
        numQB_rt = numqb_hint.shape[0]
        numKB_rt = numkb_hint.shape[0]
        maxSel_rt = maxsel_hint.shape[0]
        maxInner_rt = maxinner_hint.shape[0]

        Sq_pad = numQB_rt * BLOCK
        Skv_pad = numKB_rt * KV_BLOCK

        TOTAL_DQ_OUTER = BH * numQB_rt * SUB_SPLIT
        TOTAL_DKDV_OUTER = BHKV * numKB_rt

        # ===== Phase 1: dQ computation =====
        for outer in pypto.loop(TOTAL_DQ_OUTER, name="LOOP_dqd_sub",
                                idx_name="dq_outer_idx", parallel=True):
            sub = outer % SUB_SPLIT
            rest1 = outer // SUB_SPLIT

            q_row_ofs = rest1 * BLOCK + sub * SUB_BLOCK

            pypto.set_vec_tile_shapes(*vtl)
            q_sub = pypto.view(q_2d, [SUB_BLOCK, D], [q_row_ofs, 0])
            do_sub = pypto.view(do_2d, [SUB_BLOCK, D], [q_row_ofs, 0])
            lse_sub = pypto.view(lse_2d, [SUB_BLOCK, 1], [q_row_ofs, 0])

            D_row = pypto.view(d_row_dq, [SUB_BLOCK, 1], [q_row_ofs, 0])

            dq_acc = pypto.tensor([SUB_BLOCK, D], pypto.DT_FP32, "dq_acc")

            for v_idx in pypto.loop(maxSel_rt, name="LOOP_dqd_kblk", idx_name="v_idx"):
                kv_row_ofs = rest1 * maxSel_rt * KV_BLOCK + v_idx * KV_BLOCK

                pypto.set_vec_tile_shapes(*vtl)
                k_block = pypto.view(k_compact, [KV_BLOCK, D], [kv_row_ofs, 0])
                v_block = pypto.view(v_compact, [KV_BLOCK, D], [kv_row_ofs, 0])

                mask_row_ofs = rest1 * maxSel_rt * BLOCK + v_idx * BLOCK + sub * SUB_BLOCK
                scaled_mask_block = pypto.view(scaled_mask_dq, [SUB_BLOCK, KV_BLOCK],
                                               [mask_row_ofs, 0])
                neg_inf_block = pypto.view(neg_inf_mask_dq, [SUB_BLOCK, KV_BLOCK],
                                           [mask_row_ofs, 0])

                pypto.set_cube_tile_shapes(ct, ct, ct)
                S = pypto.matmul(q_sub, k_block, pypto.DT_FP32,
                                a_trans=False, b_trans=True)

                scaled_mask_fp32 = pypto.cast(scaled_mask_block, pypto.DT_FP32)
                neg_inf_fp32 = pypto.cast(neg_inf_block, pypto.DT_FP32)
                S_masked = pypto.add(pypto.mul(S, scaled_mask_fp32), neg_inf_fp32)

                P = pypto.exp(pypto.sub(S_masked, lse_sub))

                pypto.set_cube_tile_shapes(ct, ct, ct)
                do_v = pypto.matmul(do_sub, v_block, pypto.DT_FP32,
                                    a_trans=False, b_trans=True)

                dP = pypto.mul(P, pypto.sub(do_v, D_row))

                dP_fp16 = pypto.cast(dP, dtype)
                pypto.set_cube_tile_shapes(ct, ct, ct)
                dq_contrib = pypto.matmul(dP_fp16, k_block, pypto.DT_FP32)
                dq_contrib = pypto.mul(dq_contrib, softmax_scale)

                if pypto.is_loop_begin(v_idx):
                    dq_acc[:] = dq_contrib
                else:
                    dq_new = pypto.add(dq_acc, dq_contrib)
                    dq_acc[:] = dq_new

                if pypto.is_loop_end(v_idx):
                    dq_fp16 = pypto.cast(dq_acc, dtype)
                    pypto.assemble(dq_fp16, [q_row_ofs, 0], dq_2d)

        # ===== Phase 2: dK/dV computation =====
        for outer in pypto.loop(TOTAL_DKDV_OUTER, name="LOOP_dkdvd_kblk",
                                idx_name="dkdv_outer_idx", parallel=True):
            kv_row_ofs = outer * KV_BLOCK

            pypto.set_vec_tile_shapes(*vtl)
            k_block = pypto.view(k_2d, [KV_BLOCK, D], [kv_row_ofs, 0])
            v_block = pypto.view(v_2d, [KV_BLOCK, D], [kv_row_ofs, 0])

            dk_acc = pypto.tensor([KV_BLOCK, D], pypto.DT_FP32, "dk_acc")
            dv_acc = pypto.tensor([KV_BLOCK, D], pypto.DT_FP32, "dv_acc")

            for inner in pypto.loop(maxInner_rt, name="LOOP_dkdvd_inner", idx_name="inner_idx"):
                q_row_ofs = outer * maxInner_rt * BLOCK + inner * BLOCK

                D_row = pypto.view(d_row_dkdv, [BLOCK, 1], [q_row_ofs, 0])

                pypto.set_vec_tile_shapes(*vtl)
                q_block = pypto.view(q_compact, [BLOCK, D], [q_row_ofs, 0])
                do_block = pypto.view(do_compact, [BLOCK, D], [q_row_ofs, 0])
                lse_block = pypto.view(lse_compact, [BLOCK, 1], [q_row_ofs, 0])

                mask_row_ofs = outer * maxInner_rt * BLOCK + inner * BLOCK
                scaled_mask_block = pypto.view(scaled_mask_dkdv, [BLOCK, KV_BLOCK],
                                               [mask_row_ofs, 0])
                neg_inf_block = pypto.view(neg_inf_mask_dkdv, [BLOCK, KV_BLOCK],
                                           [mask_row_ofs, 0])

                pypto.set_cube_tile_shapes(ct, ct, ct)
                S = pypto.matmul(q_block, k_block, pypto.DT_FP32,
                                a_trans=False, b_trans=True)

                scaled_mask_fp32 = pypto.cast(scaled_mask_block, pypto.DT_FP32)
                neg_inf_fp32 = pypto.cast(neg_inf_block, pypto.DT_FP32)
                S_masked = pypto.add(pypto.mul(S, scaled_mask_fp32), neg_inf_fp32)

                P = pypto.exp(pypto.sub(S_masked, lse_block))

                pypto.set_cube_tile_shapes(ct, ct, ct)
                do_v = pypto.matmul(do_block, v_block, pypto.DT_FP32,
                                    a_trans=False, b_trans=True)

                dP = pypto.mul(P, pypto.sub(do_v, D_row))

                dP_fp16 = pypto.cast(dP, dtype)
                pypto.set_cube_tile_shapes(ct, ct, ct)
                dk_contrib = pypto.matmul(dP_fp16, q_block, pypto.DT_FP32,
                                          a_trans=True, b_trans=False)
                dk_contrib = pypto.mul(dk_contrib, softmax_scale)

                P_fp16 = pypto.cast(P, dtype)
                pypto.set_cube_tile_shapes(ct, ct, ct)
                dv_contrib = pypto.matmul(P_fp16, do_block, pypto.DT_FP32,
                                          a_trans=True, b_trans=False)

                if pypto.is_loop_begin(inner):
                    dk_acc[:] = dk_contrib
                    dv_acc[:] = dv_contrib
                else:
                    dk_new = pypto.add(dk_acc, dk_contrib)
                    dv_new = pypto.add(dv_acc, dv_contrib)
                    dk_acc[:] = dk_new
                    dv_acc[:] = dv_new

                if pypto.is_loop_end(inner):
                    dk_fp16 = pypto.cast(dk_acc, dtype)
                    dv_fp16 = pypto.cast(dv_acc, dtype)
                    pypto.assemble(dk_fp16, [kv_row_ofs, 0], dk_2d)
                    pypto.assemble(dv_fp16, [kv_row_ofs, 0], dv_2d)

    output_dir = _find_newest_created_dir(before)
    _bwd_cache[key] = (kernel, output_dir)
    return _bwd_cache[key]


@allow_in_graph
def block_sparse_attention_backward(
    dout, query, key, value, attention_out, softmax_lse, block_sparse_mask,
    actual_seq_lengths=None, actual_seq_lengths_kv=None,
    block_shape=None, cfg=DEFAULT_CONFIG,
):
    global last_backward_perf_dirs
    last_backward_perf_dirs = {}

    bx = block_shape[0] if block_shape else cfg.block_shape_x
    by = block_shape[1] if block_shape else cfg.block_shape_y

    B, Hq, Sq, D = query.shape
    _, Hkv, Skv, _ = key.shape

    numQB = math.ceil(Sq / bx)
    numKB = math.ceil(Skv / by)
    Sq_pad = numQB * bx
    Skv_pad = numKB * by

    Q_pad, _ = _pad_to_block_aligned(query, bx)
    K_pad, _ = _pad_to_block_aligned(key, by)
    V_pad, _ = _pad_to_block_aligned(value, by)
    dO_pad, _ = _pad_to_block_aligned(dout, bx)
    O_pad, _ = _pad_to_block_aligned(attention_out, bx)

    q_2d = Q_pad.reshape(B * Hq * Sq_pad, D)
    k_2d = K_pad.reshape(B * Hkv * Skv_pad, D)
    v_2d = V_pad.reshape(B * Hkv * Skv_pad, D)
    do_2d = dO_pad.reshape(B * Hq * Sq_pad, D)
    o_2d = O_pad.reshape(B * Hq * Sq_pad, D)

    lse_pad = torch.full([B, Hq, Sq_pad], cfg.lse_pad_value,
                         dtype=cfg.accum_torch_dtype, device=query.device)
    lse_pad[:, :, :Sq] = softmax_lse
    lse_2d = lse_pad.reshape(B * Hq * Sq_pad, 1)

    dq_2d = torch.zeros(B * Hq * Sq_pad, D, dtype=cfg.torch_dtype, device=query.device)
    dk_2d = torch.zeros(B * Hkv * Skv_pad, D, dtype=cfg.torch_dtype, device=query.device)
    dv_2d = torch.zeros(B * Hkv * Skv_pad, D, dtype=cfg.torch_dtype, device=query.device)

    # Create hint tensors for 5 primitive dynamic axes
    b_hint = torch.zeros(B, 1, dtype=torch.float32, device=query.device)
    hq_hint = torch.zeros(Hq, 1, dtype=torch.float32, device=query.device)
    hkv_hint = torch.zeros(Hkv, 1, dtype=torch.float32, device=query.device)
    sq_hint = torch.zeros(Sq, 1, dtype=torch.float32, device=query.device)
    skv_hint = torch.zeros(Skv, 1, dtype=torch.float32, device=query.device)
    # Derived loop-bound hints
    numqb_hint = torch.zeros(numQB, 1, dtype=torch.float32, device=query.device)
    numkb_hint = torch.zeros(numKB, 1, dtype=torch.float32, device=query.device)

    k_compact, v_compact, valid_mask, max_sel = _build_sparse_kv_cached(SparseKvBuildConfig(
        block_sparse_mask=block_sparse_mask, k_2d=k_2d, v_2d=v_2d,
        B=B, Hq=Hq, Hkv=Hkv, Sq=Sq, Skv=Skv, Sq_pad=Sq_pad, Skv_pad=Skv_pad,
        numQB=numQB, numKB=numKB, bx=bx, by=by, D=D, device=query.device))

    maxsel_hint = torch.zeros(max_sel, 1, dtype=torch.float32, device=query.device)

    q_result = _build_sparse_q_dkdv_cached(
        block_sparse_mask, q_2d, do_2d, o_2d, lse_2d,
        B, Hq, Hkv, Sq, Sq_pad, numQB, numKB,
        bx, by, D, query.device)
    q_c = q_result.q_compact
    do_c = q_result.do_compact
    o_c = q_result.o_compact
    lse_c = q_result.lse_compact
    inner_mask = q_result.inner_mask
    maxInner = q_result.maxInner

    torch.npu.synchronize()

    maxinner_hint = torch.zeros(maxInner, 1, dtype=torch.float32, device=query.device)

    softmax_scale = cfg.softmax_scale
    large_neg = cfg.large_neg
    scaled_mask_dq = (valid_mask * softmax_scale).to(torch.float16)
    neg_inf_mask_dq = ((1.0 - valid_mask) * large_neg).to(torch.float16)

    d_row_dq = (do_2d.to(torch.float32) * o_2d.to(torch.float32)).sum(dim=-1, keepdim=True)

    scaled_mask_dkdv = (inner_mask * softmax_scale).to(torch.float16)
    neg_inf_mask_dkdv = ((1.0 - inner_mask) * large_neg).to(torch.float16)

    d_row_dkdv = (do_c.to(torch.float32) * o_c.to(torch.float32)).sum(dim=-1, keepdim=True)

    torch.npu.synchronize()

    bwd_kernel_fn, bwd_dir = _get_bwd_kernel(cfg)
    before_bwd = _snapshot_output_dirs()
    bwd_kernel_fn(
        b_hint, hq_hint, hkv_hint, sq_hint, skv_hint,
        numqb_hint, numkb_hint, maxsel_hint, maxinner_hint,
        q_2d, k_compact, v_compact, do_2d, lse_2d,
        scaled_mask_dq, neg_inf_mask_dq, d_row_dq, dq_2d,
        q_c, do_c, lse_c,
        scaled_mask_dkdv, neg_inf_mask_dkdv, d_row_dkdv,
        k_2d, v_2d, dk_2d, dv_2d)
    new_bwd_dir = _find_newest_created_dir(before_bwd)
    if new_bwd_dir:
        key = ("bwd",)
        _bwd_cache[key] = (bwd_kernel_fn, new_bwd_dir)
        bwd_dir = new_bwd_dir
    last_backward_perf_dirs["dQ"] = bwd_dir
    last_backward_perf_dirs["dK/dV"] = bwd_dir

    dq = dq_2d.reshape(B, Hq, Sq_pad, D)[:, :, :Sq, :].contiguous()
    dk = dk_2d.reshape(B, Hkv, Skv_pad, D)[:, :, :Skv, :].contiguous()
    dv = dv_2d.reshape(B, Hkv, Skv_pad, D)[:, :, :Skv, :].contiguous()
    return dq, dk, dv


@allow_in_graph
def block_sparse_attention_backward_concurrent(
    dout, query, key, value, attention_out, softmax_lse, block_sparse_mask,
    actual_seq_lengths=None, actual_seq_lengths_kv=None,
    block_shape=None, cfg=DEFAULT_CONFIG,
    extra_pass_options=None, extra_runtime_options=None,
):
    """Per-BH concurrent BWD: now uses the single combined kernel (merged dQ+dK/dV).

    Since dK/dV requires contributions from all BH for accumulation,
    per-BH/BHKV streaming is not feasible with a single kernel.
    This function delegates to the baseline combined kernel.
    """
    return block_sparse_attention_backward(
        dout, query, key, value, attention_out, softmax_lse, block_sparse_mask,
        actual_seq_lengths, actual_seq_lengths_kv,
        block_shape, cfg)


# ===========================================================================
# BWD Auto-Dispatch (now delegates to baseline only)
# ===========================================================================

@allow_in_graph
def block_sparse_attention_backward_auto(
    dout, query, key, value, attention_out, softmax_lse, block_sparse_mask,
    actual_seq_lengths=None, actual_seq_lengths_kv=None,
    block_shape=None, cfg=DEFAULT_CONFIG,
    extra_pass_options=None, extra_runtime_options=None,
):
    """Auto-dispatch: now uses baseline only (single merged kernel)."""
    return block_sparse_attention_backward(
        dout, query, key, value, attention_out, softmax_lse, block_sparse_mask,
        actual_seq_lengths, actual_seq_lengths_kv,
        block_shape, cfg)