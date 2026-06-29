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
BSA Backward Golden Reference Implementation

Pure PyTorch backward reference, mathematically equivalent to AscendC BSAGrad.
"""

import math
from collections import namedtuple

import torch

from bsa_common import (
    DEFAULT_CONFIG, BSABackwardResult,
    _resolve_defaults, _block_ranges, _is_valid_mask,
)


BSABackwardBlockInputs = namedtuple(
    'BSABackwardBlockInputs',
    ['b', 'h_q', 'h_kv', 'u', 'sq', 'skv', 'bx', 'by', 'scale',
     'q_f', 'k_f', 'v_f', 'dout_f', 'out_f', 'lse',
     'block_sparse_mask', 'd_q', 'd_k', 'd_v', 'ftype'])

BSABackwardInputs = namedtuple(
    'BSABackwardInputs',
    ['dout', 'query', 'key', 'value', 'attention_out', 'softmax_lse',
     'block_sparse_mask',
     'block_shape_x', 'block_shape_y',
     'actual_seq_lengths', 'actual_seq_lengths_kv',
     'scale_value', 'cfg'])


# --- Backward ---
def _process_backward_block(block_inputs):
    """Process one Q block's backward pass across all valid KV blocks."""
    b = block_inputs.b
    h_q = block_inputs.h_q
    h_kv = block_inputs.h_kv
    u = block_inputs.u
    sq = block_inputs.sq
    skv = block_inputs.skv
    bx = block_inputs.bx
    by = block_inputs.by
    scale = block_inputs.scale
    q_f = block_inputs.q_f
    k_f = block_inputs.k_f
    v_f = block_inputs.v_f
    dout_f = block_inputs.dout_f
    out_f = block_inputs.out_f
    lse = block_inputs.lse
    block_sparse_mask = block_inputs.block_sparse_mask
    dq = block_inputs.d_q
    dk = block_inputs.d_k
    dv = block_inputs.d_v
    ftype = block_inputs.ftype

    q_start = u * bx
    q_end = min(q_start + bx, sq)
    q_block = q_f[b, h_q, q_start:q_end, :]
    do_block = dout_f[b, h_q, q_start:q_end, :]
    lse_block = lse[b, h_q, q_start:q_end]

    sg_block = (do_block * out_f[b, h_q, q_start:q_end, :]).sum(dim=-1)

    for v, k_start, k_end in _block_ranges(skv, by, skv):
        if not _is_valid_mask(block_sparse_mask, b, h_q, u, v):
            continue

        k_block = k_f[b, h_kv, k_start:k_end, :]
        v_block = v_f[b, h_kv, k_start:k_end, :]

        s_scores = torch.matmul(q_block, k_block.t()).to(ftype) * scale
        p_probs = torch.exp(s_scores - lse_block.unsqueeze(-1))
        ds = p_probs * (torch.matmul(do_block, v_block.t()).to(ftype) - sg_block.unsqueeze(-1))

        dq[b, h_q, q_start:q_end, :] += torch.matmul(ds.to(k_block.dtype), k_block).to(ftype) * scale
        dk[b, h_kv, k_start:k_end, :] += torch.matmul(ds.t().to(q_block.dtype), q_block).to(ftype) * scale
        dv[b, h_kv, k_start:k_end, :] += torch.matmul(p_probs.t().to(do_block.dtype), do_block).to(ftype)


def bsa_backward_golden(inputs):
    """BSA Backward Golden Reference (Pure PyTorch, BNSD layout, FP16).

    Recompute p from saved S and LSE, then compute gradients:
        softmaxGrad_i = sum(dO_i * O_i)
        dp = dO @ V^T,  p = exp(scale * S - LSE)
        ds = p * (dp - softmaxGrad)
        dq += ds @ K * scale,  dk += ds^T @ Q * scale,  dv += P^T @ dO

    Args:
        inputs: BSABackwardInputs namedtuple containing:
            dout, query, key, value, attention_out, softmax_lse,
            block_sparse_mask, block_shape_x, block_shape_y,
            actual_seq_lengths, actual_seq_lengths_kv, scale_value, cfg
    """
    dout, query, key, value, attention_out, softmax_lse = (
        inputs.dout, inputs.query, inputs.key, inputs.value,
        inputs.attention_out, inputs.softmax_lse)
    block_sparse_mask, block_shape_x, block_shape_y = (
        inputs.block_sparse_mask, inputs.block_shape_x, inputs.block_shape_y)
    actual_seq_lengths, actual_seq_lengths_kv, scale_value, cfg = (
        inputs.actual_seq_lengths, inputs.actual_seq_lengths_kv, inputs.scale_value, inputs.cfg)

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

    dq = torch.zeros_like(query, dtype=ftype)
    dk = torch.zeros_like(key, dtype=ftype)
    dv = torch.zeros_like(value, dtype=ftype)

    q_f, k_f, v_f = query, key, value
    out_f = attention_out.to(ftype)
    dout_f = dout

    for flat_idx in range(b * hq):
        b_idx = flat_idx // hq
        h_q = flat_idx % hq
        h_kv = h_q // group

        sq_val = asq[b_idx].item()
        skv_val = askv[b_idx].item()
        nqb = math.ceil(sq_val / bx)

        for u in range(nqb):
            block_inputs = BSABackwardBlockInputs(
                b=b_idx, h_q=h_q, h_kv=h_kv, u=u, sq=sq_val, skv=skv_val,
                bx=bx, by=by, scale=scale,
                q_f=q_f, k_f=k_f, v_f=v_f, dout_f=dout_f, out_f=out_f,
                lse=softmax_lse, block_sparse_mask=block_sparse_mask,
                d_q=dq, d_k=dk, d_v=dv, ftype=ftype)
            _process_backward_block(block_inputs)

    return BSABackwardResult(d_q=dq, d_k=dk, d_v=dv)
