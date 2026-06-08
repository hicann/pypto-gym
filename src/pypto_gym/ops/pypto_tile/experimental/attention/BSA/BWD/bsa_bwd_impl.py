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
BSA Backward PyPTO Kernel Implementation

Implements block sparse attention backward pass using PyPTO,
targeting Huawei Ascend NPU.
"""

import math
import os

import torch
import pypto
from torch._dynamo import allow_in_graph

from bsa_common import (
    DEFAULT_CONFIG,
    _pad_to_block_aligned, _is_dense_mask, _build_sparse_kv,
    _make_jit_opts,
    _VEC_TILE_LOAD, _CUBE_TILE, _VEC_TILE_OUTPUT, _CUBE_TILE_LIST,
)

# Backward-specific pass options (swimlane-tuned)
# Reducing cube_l1_reuse from the global {-1:64} to {-1:8} for backward:
#   - dQ inner loop t/iter=2, dK/dV inner loop t/iter=8
#   - {-1:64} over-merges for dQ (wastes L1, increases compile time)
# Backward-specific pass options (swimlane-tuned, per-kernel)
# Dense dQ benefits from high L1 reuse (64) for large shapes' inner loop.
# Other kernels benefit from moderate L1 reuse (16) for small shapes.
_BWD_DENSE_DQ_PASS_OPTS = {
    "cube_l1_reuse_setting": {-1: 64},
}
_BWD_PASS_OPTS = {
    "cube_l1_reuse_setting": {-1: 16},
}
_BWD_RT_OPTS = {}


# ---------------------------------------------------------------------------
# Output-dir tracking helpers
# ---------------------------------------------------------------------------
_PERF_OUTPUT_BASE = os.path.abspath(os.path.join(os.getcwd(), "output"))


def _snapshot_output_dirs():
    """Return the set of output-dir basenames that exist right now."""
    if not os.path.isdir(_PERF_OUTPUT_BASE):
        return set()
    return {d for d in os.listdir(_PERF_OUTPUT_BASE)
            if os.path.isdir(os.path.join(_PERF_OUTPUT_BASE, d))}


def _find_newest_created_dir(before):
    """Among dirs created after *before*, return the one with the newest
    ``merged_swimlane.json``, or *None*."""
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


# Module-level dict exposing the output dirs of the **last** backward call.
# Keys: "dQ", "dK/dV".  Values: absolute path to the output dir (or None).
last_backward_perf_dirs = {}


# ---------------------------------------------------------------------------
# Compacted Q builder (backward dK/dV only)
# ---------------------------------------------------------------------------
def _collect_valid_q_per_kvblock(block_sparse_mask, B, Hq, Hkv, numQB, numKB):
    """Phase 1 of _build_sparse_q_dkdv: collect valid Q blocks per KV block."""
    group = Hq // Hkv
    nkv_cols = block_sparse_mask.shape[3]
    kvblock_info = []
    maxInner = 0

    for flat_idx in range(B * Hkv):
        b = flat_idx // Hkv
        h_kv = flat_idx % Hkv
        for v_blk in range(numKB):
            valid_q = [  # pylint: disable=complicate-comprehension
                (h_kv * group + g_idx, u)
                for g_idx in range(group)
                for u in range(numQB)
                if v_blk < nkv_cols and block_sparse_mask[b, h_kv * group + g_idx, u, v_blk].item()
            ]
            kvblock_info.append((b, valid_q))
            maxInner = max(maxInner, len(valid_q))

    return kvblock_info, max(maxInner, 1)


def _fill_compacted_q(kvblock_info, maxInner, q_2d, do_2d, o_2d, lse_2d,  # pylint: disable=huawei-too-many-arguments
                       Hq, Sq_pad, bx, by, D, device, total_kv):
    """Phase 2 of _build_sparse_q_dkdv: build compacted Q/dO/O/LSE + inner_mask."""
    q_compact = torch.zeros(total_kv * maxInner * bx, D, dtype=torch.float16, device=device)
    do_compact = torch.zeros(total_kv * maxInner * bx, D, dtype=torch.float16, device=device)
    o_compact = torch.zeros(total_kv * maxInner * bx, D, dtype=torch.float16, device=device)
    lse_compact = torch.full([total_kv * maxInner * bx, 1], 1e30, dtype=torch.float32, device=device)
    inner_mask = torch.zeros(total_kv * maxInner * bx, by, dtype=torch.float32, device=device)

    for i, (b, valid_q) in enumerate(kvblock_info):
        for j, (h_q, u) in enumerate(valid_q):
            src = (b * Hq + h_q) * Sq_pad + u * bx
            dst = i * maxInner * bx + j * bx
            q_compact[dst:dst + bx] = q_2d[src:src + bx]
            do_compact[dst:dst + bx] = do_2d[src:src + bx]
            o_compact[dst:dst + bx] = o_2d[src:src + bx]
            lse_compact[dst:dst + bx] = lse_2d[src:src + bx]
            inner_mask[dst:dst + bx, :] = 1.0

        if valid_q:
            h_q0, u0 = valid_q[0]
            src0 = (b * Hq + h_q0) * Sq_pad + u0 * bx
            for j in range(len(valid_q), maxInner):
                dst = i * maxInner * bx + j * bx
                q_compact[dst:dst + bx] = q_2d[src0:src0 + bx]
                do_compact[dst:dst + bx] = do_2d[src0:src0 + bx]
                o_compact[dst:dst + bx] = o_2d[src0:src0 + bx]
                lse_compact[dst:dst + bx] = lse_2d[src0:src0 + bx]

    return q_compact, do_compact, o_compact, lse_compact, inner_mask


def _apply_q_boundary_inner_mask(inner_mask, kvblock_info, maxInner, bx, numQB, remaining_q):
    """Zero out rows in inner_mask for the last Q block's padded rows."""
    for i, (b, valid_q) in enumerate(kvblock_info):
        for j, (h_q, u) in enumerate(valid_q):
            if u == numQB - 1:
                m_dst = i * maxInner * bx + j * bx
                inner_mask[m_dst + remaining_q:m_dst + bx, :] = 0.0


def _build_sparse_q_dkdv(block_sparse_mask, q_2d, do_2d, o_2d, lse_2d,  # pylint: disable=huawei-too-many-arguments
                           B, Hq, Hkv, Sq, Sq_pad, numQB, numKB,
                           bx, by, D, device):
    """Build compacted Q/dO/O/LSE + inner_mask for sparse dK/dV kernel."""
    total_kv = B * Hkv * numKB

    kvblock_info, maxInner = _collect_valid_q_per_kvblock(
        block_sparse_mask, B, Hq, Hkv, numQB, numKB)

    q_compact, do_compact, o_compact, lse_compact, inner_mask = _fill_compacted_q(
        kvblock_info, maxInner, q_2d, do_2d, o_2d, lse_2d,
        Hq, Sq_pad, bx, by, D, device, total_kv)

    if Sq_pad > Sq:
        _apply_q_boundary_inner_mask(
            inner_mask, kvblock_info, maxInner, bx, numQB,
            Sq - (numQB - 1) * bx)

    return q_compact, do_compact, o_compact, lse_compact, inner_mask, maxInner


# ===========================================================================
# Backward dQ Kernel (compacted KV — skip invalid blocks)
# ===========================================================================
_dq_cache = {}


def _get_dq_kernel(B, Hq, Hkv, Sq, D, numQB, maxSel, cfg):  # pylint: disable=too-many-return-values
    key = ("dq", B, Hq, Hkv, Sq, D, numQB, maxSel)
    if key in _dq_cache:
        return _dq_cache[key]

    before = _snapshot_output_dirs()

    softmax_scale = cfg.softmax_scale
    bx = cfg.block_shape_x
    by = cfg.block_shape_y
    ct = _CUBE_TILE_LIST
    vtl = _VEC_TILE_LOAD
    TOTAL_OUTER = B * Hq * numQB

    @pypto.frontend.jit(**_make_jit_opts(cfg, total_outer=TOTAL_OUTER,
                                          extra_pass_options=_BWD_PASS_OPTS,
                                          extra_runtime_options=_BWD_RT_OPTS))
    def kernel(
        q_2d: pypto.Tensor([B * Hq * Sq, D], pypto.DT_FP16),
        k_compact: pypto.Tensor([B * Hq * numQB * maxSel * by, D], pypto.DT_FP16),
        v_compact: pypto.Tensor([B * Hq * numQB * maxSel * by, D], pypto.DT_FP16),
        do_2d: pypto.Tensor([B * Hq * Sq, D], pypto.DT_FP16),
        o_2d: pypto.Tensor([B * Hq * Sq, D], pypto.DT_FP16),
        lse_2d: pypto.Tensor([B * Hq * Sq, 1], pypto.DT_FP32),
        valid_mask: pypto.Tensor([B * Hq * numQB * maxSel * bx, by], pypto.DT_FP32),
        dq_2d: pypto.Tensor([B * Hq * Sq, D], pypto.DT_FP16),
    ):
        dtype = q_2d.dtype
        BLOCK = bx
        KV_BLOCK = by

        for outer in pypto.loop(TOTAL_OUTER, name="LOOP_dq_qblk", idx_name="outer_idx"):
            u = outer % numQB
            rest = outer // numQB
            h_q_idx = rest % Hq
            b_idx = rest // Hq
            bh_ofs = b_idx * Hq + h_q_idx
            q_row_ofs = bh_ofs * Sq + u * BLOCK

            pypto.set_vec_tile_shapes(*vtl)
            q_block = pypto.view(q_2d, [BLOCK, D], [q_row_ofs, 0])
            do_block = pypto.view(do_2d, [BLOCK, D], [q_row_ofs, 0])
            o_block = pypto.view(o_2d, [BLOCK, D], [q_row_ofs, 0])
            lse_block = pypto.view(lse_2d, [BLOCK, 1], [q_row_ofs, 0])

            do_o = pypto.mul(do_block, o_block)
            pypto.set_vec_tile_shapes(*vtl)
            do_o_fp32 = pypto.cast(do_o, pypto.DT_FP32)
            D_row = pypto.sum(do_o_fp32, dim=-1, keepdim=True)

            dq_acc = pypto.tensor([BLOCK, D], pypto.DT_FP32, "dq_acc")

            for v_idx in pypto.loop(maxSel, name="LOOP_dq_kblk", idx_name="v_idx"):
                kv_row_ofs = outer * maxSel * KV_BLOCK + v_idx * KV_BLOCK

                pypto.set_vec_tile_shapes(*vtl)
                k_block = pypto.view(k_compact, [KV_BLOCK, D], [kv_row_ofs, 0])
                v_block = pypto.view(v_compact, [KV_BLOCK, D], [kv_row_ofs, 0])

                pypto.set_cube_tile_shapes(ct, ct, ct)
                S = pypto.matmul(q_block, k_block, pypto.DT_FP32,
                                a_trans=False, b_trans=True)
                S_scaled = pypto.mul(S, softmax_scale)
                P = pypto.exp(pypto.sub(S_scaled, lse_block))

                mask_row_ofs = outer * maxSel * BLOCK + v_idx * BLOCK
                mask_float = pypto.view(valid_mask, [BLOCK, KV_BLOCK], [mask_row_ofs, 0])
                inv_mask = pypto.add(pypto.mul(mask_float, -1.0), 1.0)
                P_masked = pypto.add(pypto.mul(P, mask_float), pypto.mul(inv_mask, 0.0))

                pypto.set_cube_tile_shapes(ct, ct, ct)
                do_v = pypto.matmul(do_block, v_block, pypto.DT_FP32,
                                    a_trans=False, b_trans=True)
                dP = pypto.mul(P_masked, pypto.sub(do_v, D_row))

                dP_fp16 = pypto.cast(dP, dtype)
                pypto.set_cube_tile_shapes(ct, ct, ct)
                dq_contrib = pypto.matmul(dP_fp16, k_block, pypto.DT_FP32)
                dq_contrib = pypto.mul(dq_contrib, softmax_scale)

                if pypto.is_loop_begin(v_idx):
                    if pypto.is_loop_end(v_idx):
                        dq_fp16 = pypto.cast(dq_contrib, dtype)
                        pypto.assemble(dq_fp16, [q_row_ofs, 0], dq_2d)
                    else:
                        dq_acc[:] = dq_contrib
                else:
                    dq_new = pypto.add(dq_acc, dq_contrib)
                    if pypto.is_loop_end(v_idx):
                        dq_fp16 = pypto.cast(dq_new, dtype)
                        pypto.assemble(dq_fp16, [q_row_ofs, 0], dq_2d)
                    else:
                        dq_acc[:] = dq_new

    output_dir = _find_newest_created_dir(before)
    _dq_cache[key] = (kernel, output_dir)
    return _dq_cache[key]


# ===========================================================================
# Backward dK/dV Kernel (compacted Q — skip invalid blocks)
# ===========================================================================
_dkdv_cache = {}


def _get_dk_dv_kernel(B, Hq, Hkv, Sq, Skv, D, numQB, numKB, maxInner, cfg):
    key = ("dk_dv", B, Hq, Hkv, Sq, Skv, D, numQB, numKB, maxInner)
    if key in _dkdv_cache:
        return _dkdv_cache[key]

    before = _snapshot_output_dirs()

    softmax_scale = cfg.softmax_scale
    bx = cfg.block_shape_x
    by = cfg.block_shape_y
    ct = _CUBE_TILE_LIST
    vtl = _VEC_TILE_LOAD
    TOTAL_OUTER = B * Hkv * numKB

    @pypto.frontend.jit(**_make_jit_opts(cfg, total_outer=TOTAL_OUTER,
                                          extra_pass_options=_BWD_PASS_OPTS,
                                          extra_runtime_options=_BWD_RT_OPTS))
    def kernel(
        q_compact: pypto.Tensor([B * Hkv * numKB * maxInner * bx, D], pypto.DT_FP16),
        do_compact: pypto.Tensor([B * Hkv * numKB * maxInner * bx, D], pypto.DT_FP16),
        o_compact: pypto.Tensor([B * Hkv * numKB * maxInner * bx, D], pypto.DT_FP16),
        lse_compact: pypto.Tensor([B * Hkv * numKB * maxInner * bx, 1], pypto.DT_FP32),
        inner_mask: pypto.Tensor([B * Hkv * numKB * maxInner * bx, by], pypto.DT_FP32),
        k_2d: pypto.Tensor([B * Hkv * Skv, D], pypto.DT_FP16),
        v_2d: pypto.Tensor([B * Hkv * Skv, D], pypto.DT_FP16),
        dk_2d: pypto.Tensor([B * Hkv * Skv, D], pypto.DT_FP16),
        dv_2d: pypto.Tensor([B * Hkv * Skv, D], pypto.DT_FP16),
    ):
        dtype = q_compact.dtype
        BLOCK = bx
        KV_BLOCK = by

        for outer in pypto.loop(TOTAL_OUTER, name="LOOP_dkdv_kblk", idx_name="outer_idx"):
            v_blk = outer % numKB
            rest = outer // numKB
            h_kv_idx = rest % Hkv
            b_idx = rest // Hkv

            kv_row_ofs = (b_idx * Hkv + h_kv_idx) * Skv + v_blk * KV_BLOCK

            pypto.set_vec_tile_shapes(*vtl)
            k_block = pypto.view(k_2d, [KV_BLOCK, D], [kv_row_ofs, 0])
            v_block = pypto.view(v_2d, [KV_BLOCK, D], [kv_row_ofs, 0])

            dk_acc = pypto.tensor([KV_BLOCK, D], pypto.DT_FP32, "dk_acc")
            dv_acc = pypto.tensor([KV_BLOCK, D], pypto.DT_FP32, "dv_acc")

            for inner in pypto.loop(maxInner, name="LOOP_dkdv_inner", idx_name="inner_idx"):
                q_row_ofs = outer * maxInner * BLOCK + inner * BLOCK

                pypto.set_vec_tile_shapes(*vtl)
                q_block = pypto.view(q_compact, [BLOCK, D], [q_row_ofs, 0])
                do_block = pypto.view(do_compact, [BLOCK, D], [q_row_ofs, 0])
                o_block = pypto.view(o_compact, [BLOCK, D], [q_row_ofs, 0])
                lse_block = pypto.view(lse_compact, [BLOCK, 1], [q_row_ofs, 0])

                do_o = pypto.mul(do_block, o_block)
                pypto.set_vec_tile_shapes(*vtl)
                do_o_fp32 = pypto.cast(do_o, pypto.DT_FP32)
                D_row = pypto.sum(do_o_fp32, dim=-1, keepdim=True)

                pypto.set_cube_tile_shapes(ct, ct, ct)
                S = pypto.matmul(q_block, k_block, pypto.DT_FP32,
                                a_trans=False, b_trans=True)
                S_scaled = pypto.mul(S, softmax_scale)
                P = pypto.exp(pypto.sub(S_scaled, lse_block))

                mask_row_ofs = outer * maxInner * BLOCK + inner * BLOCK
                mask_float = pypto.view(inner_mask, [BLOCK, KV_BLOCK], [mask_row_ofs, 0])
                inv_mask = pypto.add(pypto.mul(mask_float, -1.0), 1.0)
                P_masked = pypto.add(pypto.mul(P, mask_float), pypto.mul(inv_mask, 0.0))

                pypto.set_cube_tile_shapes(ct, ct, ct)
                do_v = pypto.matmul(do_block, v_block, pypto.DT_FP32,
                                    a_trans=False, b_trans=True)
                dP = pypto.mul(P_masked, pypto.sub(do_v, D_row))

                dP_fp16 = pypto.cast(dP, dtype)
                pypto.set_cube_tile_shapes(ct, ct, ct)
                dk_contrib = pypto.matmul(dP_fp16, q_block, pypto.DT_FP32,
                                          a_trans=True, b_trans=False)
                dk_contrib = pypto.mul(dk_contrib, softmax_scale)

                P_fp16 = pypto.cast(P_masked, dtype)
                pypto.set_cube_tile_shapes(ct, ct, ct)
                dv_contrib = pypto.matmul(P_fp16, do_block, pypto.DT_FP32,
                                          a_trans=True, b_trans=False)

                if pypto.is_loop_begin(inner):
                    if pypto.is_loop_end(inner):
                        dk_fp16 = pypto.cast(dk_contrib, dtype)
                        dv_fp16 = pypto.cast(dv_contrib, dtype)
                        pypto.assemble(dk_fp16, [kv_row_ofs, 0], dk_2d)
                        pypto.assemble(dv_fp16, [kv_row_ofs, 0], dv_2d)
                    else:
                        dk_acc[:] = dk_contrib
                        dv_acc[:] = dv_contrib
                else:
                    dk_new = pypto.add(dk_acc, dk_contrib)
                    dv_new = pypto.add(dv_acc, dv_contrib)
                    if pypto.is_loop_end(inner):
                        dk_fp16 = pypto.cast(dk_new, dtype)
                        dv_fp16 = pypto.cast(dv_new, dtype)
                        pypto.assemble(dk_fp16, [kv_row_ofs, 0], dk_2d)
                        pypto.assemble(dv_fp16, [kv_row_ofs, 0], dv_2d)
                    else:
                        dk_acc[:] = dk_new
                        dv_acc[:] = dv_new

    output_dir = _find_newest_created_dir(before)
    _dkdv_cache[key] = (kernel, output_dir)
    return _dkdv_cache[key]


# ===========================================================================
# Dense Backward dQ Kernel (no mask, direct 2D access)
# ===========================================================================
_dq_dense_cache = {}


def _get_dense_dq_kernel(B, Hq, Hkv, Sq, Skv, D, numQB, numKB, cfg):
    key = ("dq_dense", B, Hq, Hkv, Sq, Skv, D, numQB, numKB)
    if key in _dq_dense_cache:
        return _dq_dense_cache[key]

    before = _snapshot_output_dirs()

    softmax_scale = cfg.softmax_scale
    bx = cfg.block_shape_x
    by = cfg.block_shape_y
    group = Hq // Hkv
    ct = _CUBE_TILE_LIST
    vtl = _VEC_TILE_LOAD

    # Sub-block splitting: split each Q-block (bx=256) into SUB_SPLIT=2
    # sub-blocks of 128 rows each, doubling the outer loop count.
    # This increases parallelism: more tasks fill the 22 NPU cores better.
    # Applied unconditionally — even 1024×1024 benefits from the extra
    # task parallelism (TOTAL_OUTER 16→32) despite smaller matmul M-dim.
    SUB_SPLIT = bx // 128   # = 2
    SUB_BLOCK = bx // SUB_SPLIT  # = 128
    TOTAL_OUTER = B * Hq * numQB * SUB_SPLIT

    @pypto.frontend.jit(**_make_jit_opts(cfg, total_outer=TOTAL_OUTER,
                                          extra_pass_options=_BWD_DENSE_DQ_PASS_OPTS,
                                          extra_runtime_options=_BWD_RT_OPTS))
    def kernel(
        q_2d: pypto.Tensor([B * Hq * Sq, D], pypto.DT_FP16),
        k_2d: pypto.Tensor([B * Hkv * Skv, D], pypto.DT_FP16),
        v_2d: pypto.Tensor([B * Hkv * Skv, D], pypto.DT_FP16),
        do_2d: pypto.Tensor([B * Hq * Sq, D], pypto.DT_FP16),
        o_2d: pypto.Tensor([B * Hq * Sq, D], pypto.DT_FP16),
        lse_2d: pypto.Tensor([B * Hq * Sq, 1], pypto.DT_FP32),
        dq_2d: pypto.Tensor([B * Hq * Sq, D], pypto.DT_FP16),
    ):
        dtype = q_2d.dtype
        KV_BLOCK = by

        for outer in pypto.loop(TOTAL_OUTER, name="LOOP_dqd_sub", idx_name="outer_idx"):
            sub = outer % SUB_SPLIT
            rest1 = outer // SUB_SPLIT
            u = rest1 % numQB
            rest2 = rest1 // numQB
            h_q_idx = rest2 % Hq
            b_idx = rest2 // Hq
            bh_ofs = b_idx * Hq + h_q_idx
            q_row_ofs = bh_ofs * Sq + u * bx + sub * SUB_BLOCK

            pypto.set_vec_tile_shapes(*vtl)
            q_sub = pypto.view(q_2d, [SUB_BLOCK, D], [q_row_ofs, 0])
            do_sub = pypto.view(do_2d, [SUB_BLOCK, D], [q_row_ofs, 0])
            o_sub = pypto.view(o_2d, [SUB_BLOCK, D], [q_row_ofs, 0])
            lse_sub = pypto.view(lse_2d, [SUB_BLOCK, 1], [q_row_ofs, 0])

            do_o = pypto.mul(do_sub, o_sub)
            pypto.set_vec_tile_shapes(*vtl)
            do_o_fp32 = pypto.cast(do_o, pypto.DT_FP32)
            D_row = pypto.sum(do_o_fp32, dim=-1, keepdim=True)

            dq_acc = pypto.tensor([SUB_BLOCK, D], pypto.DT_FP32, "dq_acc")

            h_kv_idx = h_q_idx // group

            for v_blk in pypto.loop(numKB, name="LOOP_dqd_kblk", idx_name="v_blk"):
                kv_row_ofs = (b_idx * Hkv + h_kv_idx) * Skv + v_blk * KV_BLOCK

                pypto.set_vec_tile_shapes(*vtl)
                k_block = pypto.view(k_2d, [KV_BLOCK, D], [kv_row_ofs, 0])
                v_block = pypto.view(v_2d, [KV_BLOCK, D], [kv_row_ofs, 0])

                pypto.set_cube_tile_shapes(ct, ct, ct)
                S = pypto.matmul(q_sub, k_block, pypto.DT_FP32,
                                a_trans=False, b_trans=True)
                S_scaled = pypto.mul(S, softmax_scale)
                P = pypto.exp(pypto.sub(S_scaled, lse_sub))

                pypto.set_cube_tile_shapes(ct, ct, ct)
                do_v = pypto.matmul(do_sub, v_block, pypto.DT_FP32,
                                    a_trans=False, b_trans=True)
                dP = pypto.mul(P, pypto.sub(do_v, D_row))

                dP_fp16 = pypto.cast(dP, dtype)
                pypto.set_cube_tile_shapes(ct, ct, ct)
                dq_contrib = pypto.matmul(dP_fp16, k_block, pypto.DT_FP32)
                dq_contrib = pypto.mul(dq_contrib, softmax_scale)

                if pypto.is_loop_begin(v_blk):
                    if pypto.is_loop_end(v_blk):
                        dq_fp16 = pypto.cast(dq_contrib, dtype)
                        pypto.assemble(dq_fp16, [q_row_ofs, 0], dq_2d)
                    else:
                        dq_acc[:] = dq_contrib
                else:
                    dq_new = pypto.add(dq_acc, dq_contrib)
                    if pypto.is_loop_end(v_blk):
                        dq_fp16 = pypto.cast(dq_new, dtype)
                        pypto.assemble(dq_fp16, [q_row_ofs, 0], dq_2d)
                    else:
                        dq_acc[:] = dq_new

    output_dir = _find_newest_created_dir(before)
    _dq_dense_cache[key] = (kernel, output_dir)
    return _dq_dense_cache[key]


# ===========================================================================
# Dense Backward dK/dV Kernel (no mask, direct 2D access)
# ===========================================================================
_dkdv_dense_cache = {}


def _get_dense_dk_dv_kernel(B, Hq, Hkv, Sq, Skv, D, numQB, numKB, cfg):
    key = ("dk_dv_dense", B, Hq, Hkv, Sq, Skv, D, numQB, numKB)
    if key in _dkdv_dense_cache:
        return _dkdv_dense_cache[key]

    before = _snapshot_output_dirs()

    softmax_scale = cfg.softmax_scale
    bx = cfg.block_shape_x
    by = cfg.block_shape_y
    group = Hq // Hkv
    ct = _CUBE_TILE_LIST
    vtl = _VEC_TILE_LOAD
    TOTAL_OUTER = B * Hkv * numKB
    TOTAL_INNER = group * numQB  # all Q blocks for this KV head

    @pypto.frontend.jit(**_make_jit_opts(cfg, total_outer=TOTAL_OUTER,
                                          extra_pass_options=_BWD_PASS_OPTS,
                                          extra_runtime_options=_BWD_RT_OPTS))
    def kernel(
        q_2d: pypto.Tensor([B * Hq * Sq, D], pypto.DT_FP16),
        do_2d: pypto.Tensor([B * Hq * Sq, D], pypto.DT_FP16),
        o_2d: pypto.Tensor([B * Hq * Sq, D], pypto.DT_FP16),
        lse_2d: pypto.Tensor([B * Hq * Sq, 1], pypto.DT_FP32),
        k_2d: pypto.Tensor([B * Hkv * Skv, D], pypto.DT_FP16),
        v_2d: pypto.Tensor([B * Hkv * Skv, D], pypto.DT_FP16),
        dk_2d: pypto.Tensor([B * Hkv * Skv, D], pypto.DT_FP16),
        dv_2d: pypto.Tensor([B * Hkv * Skv, D], pypto.DT_FP16),
    ):
        dtype = q_2d.dtype
        BLOCK = bx
        KV_BLOCK = by

        for outer in pypto.loop(TOTAL_OUTER, name="LOOP_dkdvd_kblk", idx_name="outer_idx"):
            v_blk = outer % numKB
            rest = outer // numKB
            h_kv_idx = rest % Hkv
            b_idx = rest // Hkv

            kv_row_ofs = (b_idx * Hkv + h_kv_idx) * Skv + v_blk * KV_BLOCK

            pypto.set_vec_tile_shapes(*vtl)
            k_block = pypto.view(k_2d, [KV_BLOCK, D], [kv_row_ofs, 0])
            v_block = pypto.view(v_2d, [KV_BLOCK, D], [kv_row_ofs, 0])

            dk_acc = pypto.tensor([KV_BLOCK, D], pypto.DT_FP32, "dk_acc")
            dv_acc = pypto.tensor([KV_BLOCK, D], pypto.DT_FP32, "dv_acc")

            for inner in pypto.loop(TOTAL_INNER, name="LOOP_dkdvd_inner", idx_name="inner_idx"):
                g_idx = inner // numQB
                u = inner % numQB
                h_q_idx = h_kv_idx * group + g_idx
                q_row_ofs = (b_idx * Hq + h_q_idx) * Sq + u * BLOCK

                pypto.set_vec_tile_shapes(*vtl)
                q_block = pypto.view(q_2d, [BLOCK, D], [q_row_ofs, 0])
                do_block = pypto.view(do_2d, [BLOCK, D], [q_row_ofs, 0])
                o_block = pypto.view(o_2d, [BLOCK, D], [q_row_ofs, 0])
                lse_block = pypto.view(lse_2d, [BLOCK, 1], [q_row_ofs, 0])

                do_o = pypto.mul(do_block, o_block)
                pypto.set_vec_tile_shapes(*vtl)
                do_o_fp32 = pypto.cast(do_o, pypto.DT_FP32)
                D_row = pypto.sum(do_o_fp32, dim=-1, keepdim=True)

                pypto.set_cube_tile_shapes(ct, ct, ct)
                S = pypto.matmul(q_block, k_block, pypto.DT_FP32,
                                a_trans=False, b_trans=True)
                S_scaled = pypto.mul(S, softmax_scale)
                P = pypto.exp(pypto.sub(S_scaled, lse_block))

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
                    if pypto.is_loop_end(inner):
                        dk_fp16 = pypto.cast(dk_contrib, dtype)
                        dv_fp16 = pypto.cast(dv_contrib, dtype)
                        pypto.assemble(dk_fp16, [kv_row_ofs, 0], dk_2d)
                        pypto.assemble(dv_fp16, [kv_row_ofs, 0], dv_2d)
                    else:
                        dk_acc[:] = dk_contrib
                        dv_acc[:] = dv_contrib
                else:
                    dk_new = pypto.add(dk_acc, dk_contrib)
                    dv_new = pypto.add(dv_acc, dv_contrib)
                    if pypto.is_loop_end(inner):
                        dk_fp16 = pypto.cast(dk_new, dtype)
                        dv_fp16 = pypto.cast(dv_new, dtype)
                        pypto.assemble(dk_fp16, [kv_row_ofs, 0], dk_2d)
                        pypto.assemble(dv_fp16, [kv_row_ofs, 0], dv_2d)
                    else:
                        dk_acc[:] = dk_new
                        dv_acc[:] = dv_new

    output_dir = _find_newest_created_dir(before)
    _dkdv_dense_cache[key] = (kernel, output_dir)
    return _dkdv_dense_cache[key]


# ===========================================================================
# Public wrapper
# ===========================================================================
@allow_in_graph
def block_sparse_attention_backward(  # pylint: disable=huawei-too-many-arguments
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

    is_dense = _is_dense_mask(block_sparse_mask)
    is_aligned = (Sq == Sq_pad) and (Skv == Skv_pad)

    if is_dense and is_aligned:
        # --- Dense path: direct 2D access, no compaction, no masks ---
        dq_kernel_fn, dq_dir = _get_dense_dq_kernel(B, Hq, Hkv, Sq_pad, Skv_pad, D, numQB, numKB, cfg)
        before_dq = _snapshot_output_dirs()
        dq_kernel_fn(q_2d, k_2d, v_2d, do_2d, o_2d, lse_2d, dq_2d)
        new_dq_dir = _find_newest_created_dir(before_dq)
        if new_dq_dir:
            # Update cache with the actual output dir (has swimlane now)
            key = ("dq_dense", B, Hq, Hkv, Sq_pad, Skv_pad, D, numQB, numKB)
            _dq_dense_cache[key] = (dq_kernel_fn, new_dq_dir)
            dq_dir = new_dq_dir
        last_backward_perf_dirs["dQ"] = dq_dir

        dk_dv_kernel_fn, dkdv_dir = _get_dense_dk_dv_kernel(B, Hq, Hkv, Sq_pad, Skv_pad, D, numQB, numKB, cfg)
        before_dkdv = _snapshot_output_dirs()
        dk_dv_kernel_fn(q_2d, do_2d, o_2d, lse_2d, k_2d, v_2d, dk_2d, dv_2d)
        new_dkdv_dir = _find_newest_created_dir(before_dkdv)
        if new_dkdv_dir:
            key = ("dk_dv_dense", B, Hq, Hkv, Sq_pad, Skv_pad, D, numQB, numKB)
            _dkdv_dense_cache[key] = (dk_dv_kernel_fn, new_dkdv_dir)
            dkdv_dir = new_dkdv_dir
        last_backward_perf_dirs["dK/dV"] = dkdv_dir
    else:
        # --- Sparse path: compacted tensors + masks ---
        # --- dQ kernel: compacted K/V (same schedule as forward) ---
        k_compact, v_compact, valid_mask, maxSel = _build_sparse_kv(
            block_sparse_mask, k_2d, v_2d,
            B, Hq, Hkv, Sq, Skv, Sq_pad, Skv_pad, numQB, numKB,
            bx, by, D, query.device)
        torch.npu.synchronize()

        dq_kernel_fn, dq_dir = _get_dq_kernel(B, Hq, Hkv, Sq_pad, D, numQB, maxSel, cfg)
        before_dq = _snapshot_output_dirs()
        dq_kernel_fn(q_2d, k_compact, v_compact, do_2d, o_2d, lse_2d, valid_mask, dq_2d)
        new_dq_dir = _find_newest_created_dir(before_dq)
        if new_dq_dir:
            key = ("dq", B, Hq, Hkv, Sq_pad, D, numQB, maxSel)
            _dq_cache[key] = (dq_kernel_fn, new_dq_dir)
            dq_dir = new_dq_dir
        last_backward_perf_dirs["dQ"] = dq_dir

        # --- dK/dV kernel: compacted Q/dO/O/LSE ---
        q_c, do_c, o_c, lse_c, inner_mask, maxInner = _build_sparse_q_dkdv(
            block_sparse_mask, q_2d, do_2d, o_2d, lse_2d,
            B, Hq, Hkv, Sq, Sq_pad, numQB, numKB,
            bx, by, D, query.device)
        torch.npu.synchronize()

        dk_dv_kernel_fn, dkdv_dir = _get_dk_dv_kernel(B, Hq, Hkv, Sq_pad, Skv_pad, D, numQB, numKB, maxInner, cfg)
        before_dkdv = _snapshot_output_dirs()
        dk_dv_kernel_fn(q_c, do_c, o_c, lse_c, inner_mask,
                     k_2d, v_2d, dk_2d, dv_2d)
        new_dkdv_dir = _find_newest_created_dir(before_dkdv)
        if new_dkdv_dir:
            key = ("dk_dv", B, Hq, Hkv, Sq_pad, Skv_pad, D, numQB, numKB, maxInner)
            _dkdv_cache[key] = (dk_dv_kernel_fn, new_dkdv_dir)
            dkdv_dir = new_dkdv_dir
        last_backward_perf_dirs["dK/dV"] = dkdv_dir

    dq = dq_2d.reshape(B, Hq, Sq_pad, D)[:, :, :Sq, :].contiguous()
    dk = dk_2d.reshape(B, Hkv, Skv_pad, D)[:, :, :Skv, :].contiguous()
    dv = dv_2d.reshape(B, Hkv, Skv_pad, D)[:, :, :Skv, :].contiguous()
    return dq, dk, dv
