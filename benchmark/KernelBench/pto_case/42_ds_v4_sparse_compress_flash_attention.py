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

FORMULA = ("out[t,n,d] = scfa(q, compress_kv, origin_kv, topk_indices, block_table, actual_seq_q,\n"
           "  cmp_actual_seq, origin_block_table, origin_actual_seq, atten_sink)\n"
           "  = flash_attn( Q @ [win_KV; compress_sparse_KV]^T / sqrt(d) + sinks ) @ [win_KV; compress_sparse_KV]")
DYNAMIC_AXIS = ["T", "S"]


def _gen_uniform_data(data_shape, min_value, max_value, dtype):
    if min_value == 0 and max_value == 0:
        return torch.zeros(data_shape, dtype=dtype)
    if dtype == torch.bool:
        return torch.randint(0, 2, data_shape, dtype=dtype)
    if torch.is_floating_point(torch.tensor(0, dtype=dtype)):
        return min_value + (max_value - min_value) * torch.rand(data_shape, dtype=dtype)
    else:
        return torch.randint(low=min_value, high=max_value, size=data_shape, dtype=dtype)


def _gen_block_table(act_seq, block_size):
    block_num = 0
    block_num_each = []
    b = act_seq.shape[0]
    max_kv = max(act_seq).item() if isinstance(act_seq, torch.Tensor) else max(act_seq)
    for cur_s in act_seq:
        s_val = cur_s.item() if isinstance(cur_s, torch.Tensor) else cur_s
        cur_block_num = math.ceil(s_val / block_size)
        block_num_each.append(cur_block_num)
        block_num += cur_block_num
    max_blocks = max(math.ceil(max_kv / block_size), 1)
    block_table_shape = (b, max_blocks)
    block_idx_list = torch.arange(0, max(block_num, 1), dtype=torch.int32)
    if block_num > 0:
        block_idx_list = block_idx_list[torch.randperm(block_idx_list.size(0))]
    block_table = -torch.ones(block_table_shape, dtype=torch.int32)
    bi = 0
    btb = 0
    for cur_block in block_num_each:
        for j in range(cur_block):
            block_table[btb, j] = block_idx_list[bi]
            bi += 1
        btb += 1
    return block_num, block_table


def _gen_topk_indices(actual_seq, actual_seq_q, topk, n_kv):
    t_q = actual_seq_q[-1]
    b = len(actual_seq)
    topk_indices = torch.zeros(t_q, n_kv * topk, dtype=torch.int32)
    slc_actual_seq = [min(actual_seq[i], topk) for i in range(b)]
    for b_i in range(b):
        s_q = actual_seq_q[b_i + 1] - actual_seq_q[b_i]
        for s_q_i in range(s_q):
            t_idx = actual_seq_q[b_i] + s_q_i
            if slc_actual_seq[b_i] < topk:
                topk_indices[t_idx, :slc_actual_seq[b_i]] = torch.arange(0, slc_actual_seq[b_i])
            else:
                perm = torch.randperm(slc_actual_seq[b_i])
                topk_indices[t_idx, :] = perm[:topk]
    return topk_indices


class Model(nn.Module):
    def __init__(self, n_q=64, d=512, n_kv=1, block_size=128, cmp_ratio=4, win_size=128, topk=512):
        super().__init__()
        self.n_q = n_q
        self.d = d
        self.n_kv = n_kv
        self.block_size = block_size
        self.cmp_ratio = cmp_ratio
        self.win_size = win_size
        self.topk = topk

    def forward(self, q, compress_kv, origin_kv, topk_indices, block_table, actual_seq_q,
                cmp_actual_seq, origin_block_table, origin_actual_seq, atten_sink):
        scalar = self.d ** -0.5
        s2_tile = 512

        t, n1, d = q.shape
        t = int(actual_seq_q[-1].item())
        b = len(cmp_actual_seq)

        if topk_indices.ndim > 2:
            topk_indices = topk_indices.reshape(t, self.topk)

        input_dtype = q.dtype
        kv_dtype = compress_kv.dtype
        attention_output = torch.zeros(t, n1, d, dtype=input_dtype, device=q.device)
        atten_sink_2d = atten_sink.unsqueeze(-1)

        for b_idx in range(b):
            cur_k_seq = int(cmp_actual_seq[b_idx].item())
            origin_cur_k_seq = int(origin_actual_seq[b_idx].item())
            s1 = int(actual_seq_q[b_idx + 1].item()) - int(actual_seq_q[b_idx].item())

            for s1_idx in range(s1):
                t_idx = int(actual_seq_q[b_idx].item()) + s1_idx

                cur_len = max(origin_cur_k_seq - s1 + 1 + s1_idx, 0)
                origin_cur_win_size = min(cur_len, self.win_size)
                valid_start_pos = cur_len - origin_cur_win_size
                valid_end_pos = cur_len - 1
                start_block = valid_start_pos // self.block_size
                start_offset = valid_start_pos % self.block_size
                end_block = valid_end_pos // self.block_size

                cur_seq = min(max(cur_k_seq - s1 + 1 + s1_idx, 0), self.topk)

                bn_per_batch = math.ceil(cur_seq / s2_tile)
                for s2_idx in range(bn_per_batch):
                    s2_tile_cur = min(s2_tile, cur_seq - s2_idx * s2_tile)
                    s2_start = s2_tile * s2_idx
                    s2_end = s2_start + s2_tile_cur

                    topk_indices_tmp = topk_indices[t_idx, s2_start:s2_end]
                    slc_compress_kv = torch.zeros(s2_tile_cur, self.d, dtype=kv_dtype, device=q.device)
                    offset = torch.zeros(s2_tile_cur, dtype=torch.int32, device=q.device)
                    for cur_s2_idx in range(s2_tile_cur):
                        topk_index = int(topk_indices_tmp[cur_s2_idx].item())
                        block_idx_in_batch = topk_index // self.block_size
                        slc_block_idx = int(block_table[b_idx, block_idx_in_batch].item())
                        tail = topk_index % self.block_size
                        offset[cur_s2_idx] = slc_block_idx * self.block_size + tail
                    for cur_s2_idx in range(s2_tile_cur):
                        slc_idx = int(offset[cur_s2_idx].item())
                        slc_compress_kv[cur_s2_idx, :] = compress_kv[slc_idx, :]

                    kv_list = []
                    for block_idx in range(start_block, end_block + 1):
                        physical_block_id = int(origin_block_table[b_idx, block_idx].item())
                        kv_block = origin_kv[physical_block_id * self.block_size:
                                             (physical_block_id + 1) * self.block_size, :]
                        kv_list.append(kv_block)
                    kv_cur = torch.cat(kv_list, dim=0)
                    win_kv_cache = kv_cur[start_offset:start_offset + origin_cur_win_size, :]

                    kj = torch.zeros(origin_cur_win_size + s2_tile_cur, self.d, dtype=kv_dtype, device=q.device)
                    kj[0:origin_cur_win_size, :] = win_kv_cache
                    kj[origin_cur_win_size:origin_cur_win_size + s2_tile_cur, :] = slc_compress_kv

                    qi = q[t_idx, :, :].reshape(n1, self.d)
                    sij = torch.matmul(qi.to(torch.float32), kj.transpose(1, 0).to(torch.float32))
                    sij_scale = sij * scalar
                    tilda_mij = sij_scale.amax(dim=-1, keepdim=True)
                    t_sub = sij_scale - tilda_mij
                    tilda_pij = torch.exp(t_sub)
                    tilda_lij = tilda_pij.sum(dim=-1, keepdim=True)

                    sink_t_sub = atten_sink_2d - tilda_mij
                    sink_tilda_pij = torch.exp(sink_t_sub)
                    tilda_lij = tilda_lij + sink_tilda_pij

                    tmp_softmax = (tilda_pij / tilda_lij).to(input_dtype)
                    atten_out_part = torch.matmul(tmp_softmax.to(torch.float32),
                                                  kj.to(torch.float32)).to(torch.float32)

                attention_output[t_idx, :, :] = atten_out_part.to(input_dtype)

        return attention_output


def get_inputs():
    b, n_q, n_kv, s_per_batch = 2, 4, 1, 4
    kv_lora_rank = 64
    topk = 32
    win_size = 64
    block_size = 32
    cmp_ratio = 32

    torch.manual_seed(42)
    t = b * s_per_batch
    origin_actual_seq = [128, 256]
    cmp_actual_seq = [i // cmp_ratio for i in origin_actual_seq]
    actual_seq_q = [i * s_per_batch for i in range(b + 1)]

    _, block_table = _gen_block_table(torch.tensor(cmp_actual_seq), block_size)
    _, origin_block_table = _gen_block_table(torch.tensor(origin_actual_seq), block_size)

    topk_indices = _gen_topk_indices(cmp_actual_seq, actual_seq_q, topk, n_kv)

    q = _gen_uniform_data((t, n_q, kv_lora_rank), -1, 1, torch.bfloat16)
    compress_kv_shape = block_table.max().item() + 1
    origin_kv_shape = origin_block_table.max().item() + 1
    compress_kv = _gen_uniform_data((compress_kv_shape * block_size, kv_lora_rank), -1, 1, torch.bfloat16)
    origin_kv = _gen_uniform_data((origin_kv_shape * block_size, kv_lora_rank), -1, 1, torch.bfloat16)
    atten_sink = _gen_uniform_data((n_q,), -1, 1, torch.float32)

    return [q, compress_kv, origin_kv, topk_indices, block_table,
            torch.tensor(actual_seq_q, dtype=torch.int32),
            torch.tensor(cmp_actual_seq, dtype=torch.int32),
            origin_block_table,
            torch.tensor(origin_actual_seq, dtype=torch.int32),
            atten_sink]


def get_init_inputs():
    return [4, 64, 1, 32, 32, 64, 32]
