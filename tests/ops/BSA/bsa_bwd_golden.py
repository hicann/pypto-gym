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

import torch

from bsa_common import DEFAULT_CONFIG, _resolve_defaults, _block_ranges, _is_valid_mask


# ===========================================================================
# Backward
# ===========================================================================
def _process_backward_block(b, h_q, h_kv, u, sq, skv, bx, by, scale,
                            Q_f, K_f, V_f, dO_f, O_f, lse,
                            block_sparse_mask, dQ, dK, dV):
    """Process one Q block's backward pass across all valid KV blocks."""
    q_start = u * bx
    q_end = min(q_start + bx, sq)
    q_block = Q_f[b, h_q, q_start:q_end, :]
    do_block = dO_f[b, h_q, q_start:q_end, :]
    lse_block = lse[b, h_q, q_start:q_end]

    sg_block = (do_block * O_f[b, h_q, q_start:q_end, :]).sum(dim=-1)

    for v, k_start, k_end in _block_ranges(skv, by, skv):
        if not _is_valid_mask(block_sparse_mask, b, h_q, u, v):
            continue

        k_block = K_f[b, h_kv, k_start:k_end, :]
        v_block = V_f[b, h_kv, k_start:k_end, :]

        S = torch.matmul(q_block, k_block.t()) * scale
        P = torch.exp(S - lse_block.unsqueeze(-1))
        dS = P * (torch.matmul(do_block, v_block.t()) - sg_block.unsqueeze(-1))

        dQ[b, h_q, q_start:q_end, :] += torch.matmul(dS, k_block) * scale
        dK[b, h_kv, k_start:k_end, :] += torch.matmul(dS.t(), q_block) * scale
        dV[b, h_kv, k_start:k_end, :] += torch.matmul(P.t(), do_block)


def bsa_backward_golden(
    dout, query, key, value, attention_out, softmax_lse,
    block_sparse_mask,
    block_shape_x=None, block_shape_y=None,
    actual_seq_lengths=None, actual_seq_lengths_kv=None,
    scale_value=None, cfg=DEFAULT_CONFIG,
):
    """BSA Backward Golden Reference (Pure PyTorch, BNSD layout, FP16).

    Recompute P from saved S and LSE, then compute gradients:
        softmaxGrad_i = sum(dO_i * O_i)
        dP = dO @ V^T,  P = exp(scale * S - LSE)
        dS = P * (dP - softmaxGrad)
        dQ += dS @ K * scale,  dK += dS^T @ Q * scale,  dV += P^T @ dO
    """
    B, Hq, Hkv, Sq, Skv, D, bx, by, scale, asq, askv = _resolve_defaults(
        query, key, block_shape_x, block_shape_y,
        actual_seq_lengths, actual_seq_lengths_kv, scale_value, cfg)

    dtype = query.dtype
    ftype = cfg.accum_torch_dtype
    group = Hq // Hkv

    dQ = torch.zeros_like(query, dtype=ftype)
    dK = torch.zeros_like(key, dtype=ftype)
    dV = torch.zeros_like(value, dtype=ftype)

    Q_f, K_f, V_f = query.to(ftype), key.to(ftype), value.to(ftype)
    O_f = attention_out.to(ftype)
    dO_f = dout.to(ftype)

    for flat_idx in range(B * Hq):
        b = flat_idx // Hq
        h_q = flat_idx % Hq
        h_kv = h_q // group

        sq = asq[b].item()
        skv = askv[b].item()
        nqb = math.ceil(sq / bx)

        for u in range(nqb):
            _process_backward_block(
                b, h_q, h_kv, u, sq, skv, bx, by, scale,
                Q_f, K_f, V_f, dO_f, O_f, softmax_lse,
                block_sparse_mask, dQ, dK, dV)

    return dQ.to(dtype), dK.to(dtype), dV.to(dtype)
