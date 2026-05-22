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

import math
import torch
import torch.nn as nn

FORMULA = "out[t,n,d] = sliding_window_attention(q[t,n,d], kv_cache[b,s,n,d], sink[n], block_table, seqused_kv)"
DYNAMIC_AXIS = ["T", "S"]


class Model(nn.Module):
    def __init__(self, n_q: int = 64, d: int = 512, win_size: int = 128, block_size: int = 128):
        super().__init__()
        self.n_q = n_q
        self.d = d
        self.win_size = win_size
        self.block_size = block_size

    def forward(
        self, q: torch.Tensor, kv_cache: torch.Tensor, block_table: torch.Tensor,
        seqused_kv: torch.Tensor, sinks: torch.Tensor, cu_seqlens_q: torch.Tensor
    ) -> torch.Tensor:
        t = q.shape[0]
        scalar = self.d ** -0.5
        b = len(seqused_kv)
        atten_out = torch.zeros(t, self.n_q, self.d, dtype=torch.bfloat16, device=q.device)

        for b_idx in range(b):
            cur_s_q = cu_seqlens_q[b_idx + 1] - cu_seqlens_q[b_idx]
            for s1_idx in range(cur_s_q):
                t_idx = cu_seqlens_q[b_idx] + s1_idx
                actual_seq = seqused_kv[b_idx].item()
                qi = q[t_idx, :, :]

                cur_loc = actual_seq - cur_s_q + s1_idx + 1
                valid_len = min(cur_loc, self.win_size)
                cur_start = cur_loc - valid_len
                end_pos = cur_loc

                start_block = cur_start // self.block_size
                start_offset = cur_start % self.block_size
                end_block = (end_pos - 1) // self.block_size

                kv_list = []
                for blk_idx in range(start_block, end_block + 1):
                    physical_block_id = block_table[b_idx, blk_idx].item()
                    if physical_block_id >= 0:
                        kv_block = kv_cache[physical_block_id * self.block_size:(physical_block_id + 1) * self.block_size, :self.d]
                        kv_list.append(kv_block)

                if not kv_list:
                    continue

                kv_cur = torch.cat(kv_list, dim=0)
                kv_cur = kv_cur[start_offset:start_offset + valid_len, :]

                acc_s = torch.matmul(qi.to(torch.float32), kv_cur.to(torch.float32).T) * scalar
                scores_max = acc_s.max(dim=-1, keepdim=True)[0]
                acc_s_exp = torch.exp(acc_s - scores_max)
                sum_exp = acc_s_exp.sum(dim=-1, keepdim=True)
                sum_exp += torch.exp(sinks.reshape(self.n_q, 1) - scores_max)
                attn_w = (acc_s_exp / sum_exp).to(torch.bfloat16)
                atten_out[t_idx, :, :] = torch.matmul(attn_w, kv_cur)

        return atten_out


def get_inputs():
    b = 1
    n_q, d, win, block_size = 64, 512, 128, 128
    t = 2
    s = 1024
    kv_blocks = (s + block_size - 1) // block_size

    q = torch.randn(t, n_q, d, dtype=torch.bfloat16)
    kv_cache = torch.randn(kv_blocks * block_size, d, dtype=torch.bfloat16)
    block_table = torch.arange(kv_blocks, dtype=torch.int32).reshape(b, kv_blocks)
    seqused_kv = torch.tensor([s], dtype=torch.int32)
    sinks = torch.randn(n_q, dtype=torch.float32)
    cu_seqlens_q = torch.tensor([0, t], dtype=torch.int32)

    return [q, kv_cache, block_table, seqused_kv, sinks, cu_seqlens_q]


def get_init_inputs():
    return [64, 512, 128, 128]
