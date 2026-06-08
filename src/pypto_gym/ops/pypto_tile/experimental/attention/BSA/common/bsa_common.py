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
BSA (Block Sparse Attention) Shared Configuration and Utilities

Contains all shared code used by both forward and backward:
  - BSAConfig / DEFAULT_CONFIG
  - Golden helpers (_resolve_defaults, _block_ranges, _is_valid_mask)
  - Impl helpers (_pad_to_block_aligned, _is_dense_mask, _make_jit_opts)
  - Compacted KV builders (shared by forward and backward dQ)
  - Mask generation (generate_block_sparse_mask)
"""

import math
from dataclasses import dataclass

import torch


# ===========================================================================
# Configuration (shared by golden + impl)
# ===========================================================================
@dataclass
class BSAConfig:
    """Centralized configuration for BSA (Block Sparse Attention)."""

    # --- Shape Constraints ---
    head_dim: int = 128
    block_shape_x: int = 256
    block_shape_y: int = 512

    # --- Data Types ---
    torch_dtype: torch.dtype = torch.float16
    accum_torch_dtype: torch.dtype = torch.float32

    # --- Numerical Constants ---
    large_neg: float = -65504.0
    lse_init: float = float('-inf')
    lse_pad_value: float = 1e30

    # --- Precision Tolerance (per BSA.md) ---
    fwd_atol: float = 0.0001
    fwd_rtol: float = 0.0078125
    bwd_atol: float = 0.0001
    bwd_rtol: float = 0.0078125

    # --- Derived helpers ---
    @property
    def softmax_scale(self) -> float:
        return self.head_dim ** -0.5

    # --- Validation ---
    def __post_init__(self):
        assert self.head_dim == 128, f"head_dim must be 128, got {self.head_dim}"
        assert self.block_shape_x % 64 == 0
        assert self.block_shape_y % 64 == 0
        assert self.block_shape_y >= 128
        assert self.torch_dtype == torch.float16


DEFAULT_CONFIG = BSAConfig()


# ===========================================================================
# PyPTO-specific configuration (shared by fwd_impl and bwd_impl)
# ===========================================================================
_VEC_TILE_LOAD = (128, 128)
_CUBE_TILE = (128, 128)
_VEC_TILE_OUTPUT = (16, 128, 128)
_CUBE_TILE_LIST = list(_CUBE_TILE)

# Swimlane tuning: testing sched_mode=1 (L2 affinity only) for perf
# Baseline sched_mode=3: S1024 1.05ms, S2048 2.00ms, Util 22-38%
_DEVICE_SCHED_MODE = 3
_CUBE_L1_REUSE_SETTING = {-1: 16}
_RUNTIME_DEBUG_MODE = 1
_STITCH_FUNCTION_NUM_INITIAL = 128
_STITCH_FUNCTION_NUM_STEP = 64


def _make_jit_opts(cfg, total_outer=None, *, extra_pass_options=None,
                    extra_runtime_options=None):
    """Build JIT options for pypto.frontend.jit.

    Args:
        extra_pass_options: optional dict merged into pass_options
            (e.g. {"cube_l1_reuse_setting": {-1: 4}} for backward kernels).
        extra_runtime_options: optional dict merged into runtime_options
            (e.g. {"device_sched_mode": 1} for L2 affinity).
    """
    pass_opts = {"cube_l1_reuse_setting": _CUBE_L1_REUSE_SETTING}
    if extra_pass_options:
        pass_opts.update(extra_pass_options)
    rt_opts = {
        "device_sched_mode": _DEVICE_SCHED_MODE,
    }
    if extra_runtime_options:
        rt_opts.update(extra_runtime_options)
    return dict(
        runtime_options=rt_opts,
        pass_options=pass_opts,
        debug_options={"runtime_debug_mode": _RUNTIME_DEBUG_MODE},
    )


# ===========================================================================
# Golden helpers (shared by fwd_golden and bwd_golden)
# ===========================================================================
def _resolve_defaults(query, key, block_shape_x, block_shape_y,
                      actual_seq_lengths, actual_seq_lengths_kv,
                      scale_value, cfg):
    """Resolve default parameters and validate shapes."""
    B, Hq, Sq, D = query.shape
    _, Hkv, Skv, _ = key.shape

    bx = block_shape_x or cfg.block_shape_x
    by = block_shape_y or cfg.block_shape_y
    scale = scale_value if scale_value is not None else cfg.softmax_scale
    asq = actual_seq_lengths if actual_seq_lengths is not None else torch.full([B], Sq, dtype=torch.int64)
    askv = actual_seq_lengths_kv if actual_seq_lengths_kv is not None else torch.full([B], Skv, dtype=torch.int64)

    assert D == cfg.head_dim, f"head_dim must be {cfg.head_dim}, got {D}"
    assert Hq >= Hkv and Hq % Hkv == 0, \
        f"GQA constraint: Hq({Hq}) >= Hkv({Hkv}) and Hq % Hkv == 0"

    return B, Hq, Hkv, Sq, Skv, D, bx, by, scale, asq, askv


def _block_ranges(seq_len, block_size, actual_len):
    """Yield (block_index, start, end) for each block up to actual_len."""
    num_blocks = math.ceil(actual_len / block_size)
    for blk in range(num_blocks):
        start = blk * block_size
        end = min(start + block_size, actual_len)
        yield blk, start, end


def _is_valid_mask(block_sparse_mask, b, h_q, u, v):
    """Check whether mask[b, h_q, u, v] is True, with bounds guard."""
    if u >= block_sparse_mask.shape[2] or v >= block_sparse_mask.shape[3]:
        return False
    return block_sparse_mask[b, h_q, u, v].item()


# ===========================================================================
# Impl helpers (shared by fwd_impl and bwd_impl)
# ===========================================================================
def _pad_to_block_aligned(tensor, block_size):
    B, H, S, D = tensor.shape
    S_padded = math.ceil(S / block_size) * block_size
    if S == S_padded:
        return tensor, S_padded
    return torch.nn.functional.pad(tensor, (0, 0, 0, S_padded - S), mode='constant', value=0.0), S_padded


def _is_dense_mask(block_sparse_mask):
    return block_sparse_mask.all().item()


# ---------------------------------------------------------------------------
# Boundary mask helpers (used by compacted KV/Q builders)
# ---------------------------------------------------------------------------
def _apply_kv_boundary_mask(valid_mask, qblock_info, maxSel, bx, numKB, remaining_kv):
    """Zero out columns in valid_mask for the last KV block's padded rows."""
    for i, (b, h_kv, valid_v) in enumerate(qblock_info):
        for j, v_blk in enumerate(valid_v):
            if v_blk == numKB - 1:
                m_start = i * maxSel * bx + j * bx
                valid_mask[m_start:m_start + bx, remaining_kv:] = 0.0


def _apply_q_boundary_mask(valid_mask, total_qblocks, maxSel, bx, numQB, remaining_q):
    """Zero out rows in valid_mask for the last Q block's padded rows."""
    for i in range(total_qblocks):
        if i % numQB == numQB - 1:
            for j in range(maxSel):
                m_dst = i * maxSel * bx + j * bx
                valid_mask[m_dst + remaining_q:m_dst + bx, :] = 0.0


# ---------------------------------------------------------------------------
# Compacted KV builders (shared by forward sparse and backward dQ)
# ---------------------------------------------------------------------------
def _collect_valid_kv_per_qblock(block_sparse_mask, B, Hq, Hkv, numQB, numKB):
    """Phase 1 of _build_sparse_kv: collect valid KV indices per Q block."""
    group = Hq // Hkv
    nkv_cols = block_sparse_mask.shape[3]
    qblock_info = []
    maxSel = 0

    for flat_idx in range(B * Hq):
        b = flat_idx // Hq
        h_q = flat_idx % Hq
        h_kv = h_q // group
        for u in range(numQB):
            valid_v = [v for v in range(min(numKB, nkv_cols))
                       if block_sparse_mask[b, h_q, u, v].item()]
            if not valid_v:
                valid_v = [0]
            qblock_info.append((b, h_kv, valid_v))
            maxSel = max(maxSel, len(valid_v))

    return qblock_info, max(maxSel, 1)


def _fill_compacted_kv(qblock_info, maxSel, k_2d, v_2d, valid_mask,
                        Hkv, Skv_pad, bx, by, D, device, total_qblocks):
    """Phase 2 of _build_sparse_kv: build compacted K/V tensors."""
    k_compact = torch.zeros(total_qblocks * maxSel * by, D,
                            dtype=torch.float16, device=device)
    v_compact = torch.zeros(total_qblocks * maxSel * by, D,
                            dtype=torch.float16, device=device)
    valid_mask_new = torch.zeros(total_qblocks * maxSel * bx, by,
                                 dtype=torch.float32, device=device)

    for i, (b, h_kv, valid_v) in enumerate(qblock_info):
        for j, v_blk in enumerate(valid_v):
            src = (b * Hkv + h_kv) * Skv_pad + v_blk * by
            dst = i * maxSel * by + j * by
            k_compact[dst:dst + by] = k_2d[src:src + by]
            v_compact[dst:dst + by] = v_2d[src:src + by]
            valid_mask_new[i * maxSel * bx + j * bx:i * maxSel * bx + j * bx + bx, :] = 1.0

        first_src = (b * Hkv + h_kv) * Skv_pad + valid_v[0] * by
        for j in range(len(valid_v), maxSel):
            dst = i * maxSel * by + j * by
            k_compact[dst:dst + by] = k_2d[first_src:first_src + by]
            v_compact[dst:dst + by] = v_2d[first_src:first_src + by]

    return k_compact, v_compact, valid_mask_new


def _build_sparse_kv(block_sparse_mask, k_2d, v_2d,
                      B, Hq, Hkv, Sq, Skv, Sq_pad, Skv_pad, numQB, numKB,
                      bx, by, D, device):
    """Build compacted K/V + valid_mask for sparse forward and dQ kernels."""
    total_qblocks = B * Hq * numQB

    qblock_info, maxSel = _collect_valid_kv_per_qblock(
        block_sparse_mask, B, Hq, Hkv, numQB, numKB)

    k_compact, v_compact, valid_mask = _fill_compacted_kv(
        qblock_info, maxSel, k_2d, v_2d, None,
        Hkv, Skv_pad, bx, by, D, device, total_qblocks)

    if Skv_pad > Skv:
        _apply_kv_boundary_mask(
            valid_mask, qblock_info, maxSel, bx, numKB,
            Skv - (numKB - 1) * by)

    if Sq_pad > Sq:
        _apply_q_boundary_mask(valid_mask, total_qblocks, maxSel, bx, numQB,
                               Sq - (numQB - 1) * bx)

    return k_compact, v_compact, valid_mask, maxSel


# ===========================================================================
# C3: Mask structure cache (reuse valid_mask/maxSel across calls)
# ===========================================================================
_mask_cache = {}


def _build_sparse_kv_cached(block_sparse_mask, k_2d, v_2d,
                             B, Hq, Hkv, Sq, Skv, Sq_pad, Skv_pad, numQB, numKB,
                             bx, by, D, device):
    """Cached version: reuse mask structure if mask is unchanged."""
    global _mask_cache
    total_qblocks = B * Hq * numQB

    # Cache key = mask hash + shape info (not k_2d/v_2d which change per call)
    cache_key = (block_sparse_mask.data_ptr(), B, Hq, Hkv, numQB, numKB, bx, by)

    cached = _mask_cache.get(cache_key)
    if cached is not None:
        qblock_info, maxSel = cached
    else:
        qblock_info, maxSel = _collect_valid_kv_per_qblock(
            block_sparse_mask, B, Hq, Hkv, numQB, numKB)
        _mask_cache[cache_key] = (qblock_info, maxSel)

    k_compact, v_compact, valid_mask = _fill_compacted_kv(
        qblock_info, maxSel, k_2d, v_2d, None,
        Hkv, Skv_pad, bx, by, D, device, total_qblocks)

    if Skv_pad > Skv:
        _apply_kv_boundary_mask(
            valid_mask, qblock_info, maxSel, bx, numKB,
            Skv - (numKB - 1) * by)

    if Sq_pad > Sq:
        _apply_q_boundary_mask(valid_mask, total_qblocks, maxSel, bx, numQB,
                               Sq - (numQB - 1) * bx)

    return k_compact, v_compact, valid_mask, maxSel


# ---------------------------------------------------------------------------
# Compacted Q builders (shared by backward dK/dV — baseline and concurrent)
# ---------------------------------------------------------------------------
def _collect_valid_q_per_kvblock(block_sparse_mask, B, Hq, Hkv, numQB, numKB):
    group = Hq // Hkv
    nkv_cols = block_sparse_mask.shape[3]
    kvblock_info = []
    maxInner = 0

    for flat_idx in range(B * Hkv):
        b = flat_idx // Hkv
        h_kv = flat_idx % Hkv
        for v_blk in range(numKB):
            valid_q = [
                (h_kv * group + g_idx, u)
                for g_idx in range(group)
                for u in range(numQB)
                if v_blk < nkv_cols and block_sparse_mask[b, h_kv * group + g_idx, u, v_blk].item()
            ]
            kvblock_info.append((b, valid_q))
            maxInner = max(maxInner, len(valid_q))

    return kvblock_info, max(maxInner, 1)


def _fill_compacted_q(kvblock_info, maxInner, q_2d, do_2d, o_2d, lse_2d,
                       Hq, Sq_pad, bx, by, D, device, total_kv):
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
    for i, (b, valid_q) in enumerate(kvblock_info):
        for j, (h_q, u) in enumerate(valid_q):
            if u == numQB - 1:
                m_dst = i * maxInner * bx + j * bx
                inner_mask[m_dst + remaining_q:m_dst + bx, :] = 0.0


def _build_sparse_q_dkdv(block_sparse_mask, q_2d, do_2d, o_2d, lse_2d,
                           B, Hq, Hkv, Sq, Sq_pad, numQB, numKB,
                           bx, by, D, device):
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
# C3: Mask structure cache for dK/dV (reuse kvblock_info/maxInner across calls)
# ===========================================================================
_q_dkdv_cache = {}


def _build_sparse_q_dkdv_cached(block_sparse_mask, q_2d, do_2d, o_2d, lse_2d,
                                  B, Hq, Hkv, Sq, Sq_pad, numQB, numKB,
                                  bx, by, D, device):
    """Cached version: reuse mask structure if mask is unchanged."""
    global _q_dkdv_cache
    total_kv = B * Hkv * numKB

    cache_key = (block_sparse_mask.data_ptr(), B, Hq, Hkv, numQB, numKB, bx, by)

    cached = _q_dkdv_cache.get(cache_key)
    if cached is not None:
        kvblock_info, maxInner = cached
    else:
        kvblock_info, maxInner = _collect_valid_q_per_kvblock(
            block_sparse_mask, B, Hq, Hkv, numQB, numKB)
        _q_dkdv_cache[cache_key] = (kvblock_info, maxInner)

    q_compact, do_compact, o_compact, lse_compact, inner_mask = _fill_compacted_q(
        kvblock_info, maxInner, q_2d, do_2d, o_2d, lse_2d,
        Hq, Sq_pad, bx, by, D, device, total_kv)

    if Sq_pad > Sq:
        _apply_q_boundary_inner_mask(
            inner_mask, kvblock_info, maxInner, bx, numQB,
            Sq - (numQB - 1) * bx)

    return q_compact, do_compact, o_compact, lse_compact, inner_mask, maxInner


# ===========================================================================
# Utility: mask generation
# ===========================================================================
def generate_block_sparse_mask(
    batch, head_num_q, num_q_blocks, num_kv_blocks,
    sparsity=0.5, device="cpu", seed=None,
):
    """Generate a random block-sparse mask with given sparsity ratio."""
    if seed is not None:
        torch.manual_seed(seed)

    mask = torch.rand(batch, head_num_q, num_q_blocks, num_kv_blocks, device=device) < sparsity

    any_valid = mask.any(dim=-1)
    empty_qblocks = ~any_valid

    if empty_qblocks.any():
        forced_kv = torch.randint(0, num_kv_blocks,
                                  empty_qblocks.shape, device=device)
        b_idx, h_idx, u_idx = torch.where(empty_qblocks)
        mask[b_idx, h_idx, u_idx, forced_kv[b_idx, h_idx, u_idx]] = True

    return mask
