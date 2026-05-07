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

import torch

from bsa_common import DEFAULT_CONFIG, _resolve_defaults, _block_ranges, _is_valid_mask


# ===========================================================================
# Online softmax update helpers
# ===========================================================================
def _softmax_first_block(S, v_block, ftype):
    """Online softmax for the first valid KV block."""
    block_max = S.max(dim=-1).values
    P = torch.exp(S - block_max.unsqueeze(-1))
    block_sum = P.sum(dim=-1)
    block_out = torch.matmul(P, v_block)
    return block_max, block_sum, block_out


def _softmax_accumulate(S, v_block, block_max, block_sum, block_out):
    """Online softmax for subsequent valid KV blocks."""
    cur_max = S.max(dim=-1).values
    new_max = torch.maximum(block_max, cur_max)
    correction = torch.exp(block_max - new_max)
    P = torch.exp(S - new_max.unsqueeze(-1))
    block_sum = block_sum * correction + P.sum(dim=-1)
    block_out = block_out * correction.unsqueeze(-1) + torch.matmul(P, v_block)
    return new_max, block_sum, block_out


# ===========================================================================
# Forward
# ===========================================================================
def bsa_forward_golden(
    query, key, value, block_sparse_mask,
    block_shape_x=None, block_shape_y=None,
    actual_seq_lengths=None, actual_seq_lengths_kv=None,
    scale_value=None, cfg=DEFAULT_CONFIG,
):
    """BSA Forward Golden Reference (Pure PyTorch, BNSD layout, FP16).

    Implements block sparse attention with online softmax:
        S_uv = Q_u @ K_v^T / sqrt(d),  M[u][v] = 1
        P_uv = exp(S_uv - LSE)
        O_u  = sum_v( P_uv @ V_v ),  for valid v
    """
    B, Hq, Hkv, Sq, Skv, D, bx, by, scale, asq, askv = _resolve_defaults(
        query, key, block_shape_x, block_shape_y,
        actual_seq_lengths, actual_seq_lengths_kv, scale_value, cfg)

    dtype = query.dtype
    ftype = cfg.accum_torch_dtype
    group = Hq // Hkv

    O = torch.zeros(B, Hq, Sq, D, dtype=ftype, device=query.device)
    softmax_lse = torch.full([B, Hq, Sq], cfg.lse_init, dtype=ftype, device=query.device)

    Q_f, K_f, V_f = query.to(ftype), key.to(ftype), value.to(ftype)

    for flat_idx in range(B * Hq):
        b = flat_idx // Hq
        h_q = flat_idx % Hq
        h_kv = h_q // group

        sq = asq[b].item()
        skv = askv[b].item()

        for u, q_start, q_end in _block_ranges(sq, bx, sq):
            q_block = Q_f[b, h_q, q_start:q_end, :]

            block_max = torch.full([q_end - q_start], float('-inf'), dtype=ftype, device=query.device)
            block_sum = torch.zeros([q_end - q_start], dtype=ftype, device=query.device)
            block_out = torch.zeros([q_end - q_start, D], dtype=ftype, device=query.device)
            has_valid = False

            for v, k_start, k_end in _block_ranges(skv, by, skv):
                if not _is_valid_mask(block_sparse_mask, b, h_q, u, v):
                    continue

                k_block = K_f[b, h_kv, k_start:k_end, :]
                v_block = V_f[b, h_kv, k_start:k_end, :]
                S = torch.matmul(q_block, k_block.t()) * scale

                if not has_valid:
                    block_max, block_sum, block_out = _softmax_first_block(S, v_block, ftype)
                    has_valid = True
                else:
                    block_max, block_sum, block_out = _softmax_accumulate(
                        S, v_block, block_max, block_sum, block_out)

            if has_valid:
                O[b, h_q, q_start:q_end, :] = block_out / block_sum.unsqueeze(-1)
                softmax_lse[b, h_q, q_start:q_end] = block_max + torch.log(block_sum)

    return O.to(dtype), softmax_lse
