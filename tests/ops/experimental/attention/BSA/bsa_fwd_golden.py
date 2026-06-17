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
BSA Forward Golden Reference Implementation

Pure PyTorch forward reference, mathematically equivalent to AscendC BSA.
"""

from collections import namedtuple

import torch

from bsa_common import (
    DEFAULT_CONFIG, BSAForwardResult,
    _resolve_defaults, _block_ranges, _is_valid_mask,
)


BSAForwardInputs = namedtuple(
    'BSAForwardInputs',
    ['query', 'key', 'value', 'block_sparse_mask',
     'block_shape_x', 'block_shape_y',
     'actual_seq_lengths', 'actual_seq_lengths_kv',
     'scale_value', 'cfg'])

_ProcessQBlockConfig = namedtuple('_ProcessQBlockConfig',
    ['k_f', 'v_f', 'block_sparse_mask', 'b_idx', 'h_q', 'h_kv',
     'skv_val', 'by', 'scale', 'ftype'])


# --- Online softmax update helpers ---
def _softmax_first_block(s_scores, v_block, ftype):
    """Online softmax for the first valid KV block."""
    block_max = s_scores.max(dim=-1).values
    p_probs = torch.exp(s_scores - block_max.unsqueeze(-1))
    block_sum = p_probs.sum(dim=-1)
    block_out = torch.matmul(p_probs, v_block)
    return block_max, block_sum, block_out


def _softmax_accumulate(s_scores, v_block, block_max, block_sum, block_out):
    """Online softmax for subsequent valid KV blocks."""
    cur_max = s_scores.max(dim=-1).values
    new_max = torch.maximum(block_max, cur_max)
    correction = torch.exp(block_max - new_max)
    p_probs = torch.exp(s_scores - new_max.unsqueeze(-1))
    block_sum = block_sum * correction + p_probs.sum(dim=-1)
    block_out = block_out * correction.unsqueeze(-1) + torch.matmul(p_probs, v_block)
    return new_max, block_sum, block_out


def _process_q_block(q_block, u, config):
    """Process one Q block's forward pass across all valid KV blocks.

    Returns (block_out, block_max, block_sum, has_valid) or None if no valid KV.
    """
    k_f = config.k_f
    v_f = config.v_f
    block_sparse_mask = config.block_sparse_mask
    b_idx = config.b_idx
    h_q = config.h_q
    h_kv = config.h_kv
    skv_val = config.skv_val
    by = config.by
    scale = config.scale
    ftype = config.ftype

    block_max = None
    block_sum = None
    block_out = None
    has_valid = False

    for v, k_start, k_end in _block_ranges(skv_val, by, skv_val):
        if not _is_valid_mask(block_sparse_mask, b_idx, h_q, u, v):
            continue

        k_block = k_f[b_idx, h_kv, k_start:k_end, :]
        v_block = v_f[b_idx, h_kv, k_start:k_end, :]
        s_scores = torch.matmul(q_block, k_block.t()) * scale

        if not has_valid:
            block_max, block_sum, block_out = _softmax_first_block(s_scores, v_block, ftype)
            has_valid = True
        else:
            block_max, block_sum, block_out = _softmax_accumulate(
                s_scores, v_block, block_max, block_sum, block_out)

    if has_valid:
        return block_out, block_max, block_sum, True
    return None


# --- Forward ---
def bsa_forward_golden(inputs):
    """BSA Forward Golden Reference (Pure PyTorch, BNSD layout, FP16).

    Implements block sparse attention with online softmax:
        S_uv = Q_u @ K_v^T / sqrt(d),  M[u][v] = 1
        p_uv = exp(S_uv - LSE)
        O_u  = sum_v( p_uv @ V_v ),  for valid v

    Args:
        inputs: BSAForwardInputs namedtuple containing:
            query, key, value, block_sparse_mask,
            block_shape_x, block_shape_y,
            actual_seq_lengths, actual_seq_lengths_kv,
            scale_value, cfg
    """
    query = inputs.query
    key = inputs.key
    value = inputs.value
    block_sparse_mask = inputs.block_sparse_mask
    block_shape_x = inputs.block_shape_x
    block_shape_y = inputs.block_shape_y
    actual_seq_lengths = inputs.actual_seq_lengths
    actual_seq_lengths_kv = inputs.actual_seq_lengths_kv
    scale_value = inputs.scale_value
    cfg = inputs.cfg

    resolved = _resolve_defaults(
        query, key, block_shape_x, block_shape_y,
        actual_seq_lengths, actual_seq_lengths_kv, scale_value, cfg)
    b = resolved.b
    hq = resolved.hq
    hkv = resolved.hkv
    sq = resolved.sq
    skv = resolved.skv
    d = resolved.d
    bx = resolved.bx
    by = resolved.by
    scale = resolved.scale
    asq = resolved.asq
    askv = resolved.askv

    dtype = query.dtype
    ftype = cfg.accum_torch_dtype
    group = hq // hkv

    o_out = torch.zeros(b, hq, sq, d, dtype=ftype, device=query.device)
    softmax_lse = torch.full([b, hq, sq], cfg.lse_init, dtype=ftype, device=query.device)

    q_f, k_f, v_f = query.to(ftype), key.to(ftype), value.to(ftype)

    for flat_idx in range(b * hq):
        b_idx = flat_idx // hq
        h_q = flat_idx % hq
        h_kv = h_q // group

        sq_val = asq[b_idx].item()
        skv_val = askv[b_idx].item()

        for u, q_start, q_end in _block_ranges(sq_val, bx, sq_val):
            q_block = q_f[b_idx, h_q, q_start:q_end, :]
            result = _process_q_block(
                q_block, u,
                _ProcessQBlockConfig(
                    k_f=k_f, v_f=v_f, block_sparse_mask=block_sparse_mask,
                    b_idx=b_idx, h_q=h_q, h_kv=h_kv, skv_val=skv_val,
                    by=by, scale=scale, ftype=ftype))

            if result is not None:
                block_out, block_max, block_sum, _ = result
                o_out[b_idx, h_q, q_start:q_end, :] = block_out / block_sum.unsqueeze(-1)
                softmax_lse[b_idx, h_q, q_start:q_end] = block_max + torch.log(block_sum)

    return BSAForwardResult(o=o_out.to(dtype), lse=softmax_lse)
