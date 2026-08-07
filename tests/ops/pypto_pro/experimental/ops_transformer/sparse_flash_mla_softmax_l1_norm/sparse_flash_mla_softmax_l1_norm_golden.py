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
"""CPU FP32 golden reference for sparse_flash_mla_softmax_l1_norm.

精确复刻 kernel 逐行 ``used_t1_index`` 的定位与 ``s2_real_size`` 语义，用于与 NPU PyPTO-Pro
kernel 输出做精度比对。
"""

import torch


def _find_batch_seqused(seqused, r):
    off = 0
    for bi in range(len(seqused)):
        if r < off + int(seqused[bi]):
            return bi, r - off
        off += int(seqused[bi])
    return len(seqused) - 1, r - off


def _find_batch_cuseq(cu_lens, r):
    c = cu_lens
    lo, hi = 0, len(c) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if c[mid] <= r:
            lo = mid
        else:
            hi = mid - 1
    return lo


def softmax_l1_norm_golden(
    q, k, lse, sparse_indices, cu_q, cu_k, seq_q, seq_k, cmp_res, topk,
    tiling, is_tnd, is_sparse, sparse_mode,
):
    """通用 CPU FP32 参考实现，返回与 kernel 相同布局的输出。"""
    b, sq, sk, g = tiling.b, tiling.sq, tiling.sk, tiling.g
    k_length = tiling.k_length
    max_seqlen_k = tiling.max_seqlen_k
    cmp_ratio = tiling.cmp_ratio
    scale = tiling.softmax_scale
    has_seq_q = tiling.has_seqused_q
    has_seq_k = tiling.has_seqused_k
    has_topk = tiling.has_topk_length

    if is_tnd:
        out_len = k_length if is_sparse else max_seqlen_k
        out = torch.zeros((tiling.t1, 1, out_len), dtype=torch.float32)
        sq_tiles = int(sum(seq_q)) if has_seq_q else int(cu_q[-1])
    else:
        out_len = k_length if is_sparse else sk
        out = torch.zeros((b, sq, 1, out_len), dtype=torch.float32)
        sq_tiles = int(sum(seq_q)) if has_seq_q else b * sq

    qf = q.float()
    for r in range(sq_tiles):
        # --- 定位 batch_id / s1_idx / offsets（复刻 get_seq_length）---
        if has_seq_q:
            batch_id, s1_idx = _find_batch_seqused(seq_q, r)
            if is_tnd:
                t1_off = int(cu_q[batch_id])
                t2_off = int(cu_k[batch_id])
                s1_size = int(cu_q[batch_id + 1]) - t1_off
                s2_size = int(cu_k[batch_id + 1]) - t2_off
            else:
                t1_off = batch_id * sq
                t2_off = batch_id * sk
                s1_size = sq
                s2_size = sk
        else:
            if is_tnd:
                batch_id = _find_batch_cuseq(cu_q, r)
                s1_idx = r - int(cu_q[batch_id])
                t1_off = int(cu_q[batch_id])
                t2_off = int(cu_k[batch_id])
                s1_size = int(cu_q[batch_id + 1]) - t1_off
                s2_size = int(cu_k[batch_id + 1]) - t2_off
            else:
                batch_id = r // sq
                s1_idx = r % sq
                t1_off = batch_id * sq
                t2_off = batch_id * sk
                s1_size = sq
                s2_size = sk

        cur_s1 = int(seq_q[batch_id]) if has_seq_q else s1_size
        cur_s2 = int(seq_k[batch_id]) if has_seq_k else s2_size

        # --- s2_real_size ---
        ori_s2 = cur_s2 * cmp_ratio + (int(cmp_res[batch_id]) if cmp_ratio > 1 else 0)
        if not is_sparse:
            if sparse_mode == 0:
                s2_real = cur_s2
            else:
                s2_real = max((ori_s2 - cur_s1 + s1_idx + 1) // cmp_ratio, 0)
        else:
            cur_k_length = k_length
            if has_topk:
                if is_tnd:
                    cur_k_length = min(cur_k_length, int(topk[t1_off + s1_idx]))
                else:
                    cur_k_length = min(cur_k_length, int(topk[batch_id, s1_idx]))
            if sparse_mode == 0:
                s2_real = min(cur_k_length, cur_s2)
            else:
                s2_valid = max((ori_s2 - cur_s1 + s1_idx + 1) // cmp_ratio, 0)
                s2_real = min(cur_k_length, s2_valid)

        if s2_real <= 0:
            continue

        # --- q 行 ---
        q_row = qf[t1_off + s1_idx] if is_tnd else qf[batch_id, s1_idx]

        # --- 选中的 kv [s2_real, d] ---
        if is_sparse:
            if is_tnd:
                idx = sparse_indices[t1_off + s1_idx, 0, :s2_real].long()
                kv = k[t2_off + idx, 0, :].float()
            else:
                idx = sparse_indices[batch_id, s1_idx, 0, :s2_real].long()
                kv = k[batch_id, idx, 0, :].float()
        else:
            if is_tnd:
                kv = k[t2_off:t2_off + s2_real, 0, :].float()
            else:
                kv = k[batch_id, :s2_real, 0, :].float()

        # --- lse 行 ---
        lse_row = lse[0, t1_off + s1_idx] if is_tnd else lse[batch_id, s1_idx, 0]

        qk = kv @ q_row.t()
        p = torch.exp(qk * scale - lse_row.float().unsqueeze(0))
        row_out = p.sum(dim=-1) / g

        if is_tnd:
            out[t1_off + s1_idx, 0, :s2_real] = row_out
        else:
            out[batch_id, s1_idx, 0, :s2_real] = row_out
    return out