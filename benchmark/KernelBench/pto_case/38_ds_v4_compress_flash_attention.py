#!/usr/bin/env python3
# coding: utf-8

import math
import torch
import torch.nn as nn

FORMULA = ("out[t,n,d] = ifa_golden(q, cmp_kv, sinks, cmp_block_table, seqused_kv, ori_kv, ori_block_table)\n"
           "  = softmax( Q @ [win_KV; compress_KV]^T / sqrt(d) + sinks ) @ [win_KV; compress_KV]")
DYNAMIC_AXIS = ["T", "S"]


def _matmul_proxy(left, right):
    return torch.matmul(left.to(torch.float32), right.to(torch.float32)).to(left.dtype)


def _softmax_with_sinks(x, sinks, is_new_sink=True):
    x_max = x.max(dim=-1, keepdim=True).values
    x_sub = x - x_max
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdim=True)
    if is_new_sink:
        x_sum += torch.exp(sinks.unsqueeze(-1) - x_max)
    else:
        x_sum += sinks.unsqueeze(-1)
    return y / x_sum


def _kv_cache_concat_bsnd(kv_cache_out, block_table, actual_seqs):
    b = actual_seqs.shape[0]
    n2 = kv_cache_out.shape[2]
    d = kv_cache_out.shape[3]
    block_size = kv_cache_out.shape[1]
    dtype = kv_cache_out.dtype
    kv_max = max(int(torch.max(actual_seqs).item() + block_size - 1) // block_size * block_size, 1)
    cmp_kv = torch.zeros([b, kv_max, n2, d], dtype=dtype, device=kv_cache_out.device)
    for b_idx in range(b):
        block_list = block_table[b_idx]
        kv_tmp = torch.zeros([1, kv_max, n2, d], dtype=dtype, device=kv_cache_out.device)
        s_idx = 0
        for block_idx in block_list:
            if block_idx == -1:
                break
            start_idx = s_idx * block_size
            end_idx = (s_idx + 1) * block_size
            kv_tmp[0, start_idx:end_idx, :, :] = kv_cache_out[int(block_idx):int(block_idx) + 1, :, :, :]
            s_idx += 1
        cmp_kv[b_idx:b_idx + 1, :, :, :] = kv_tmp
    return cmp_kv


class Model(nn.Module):
    def __init__(self, n_q=64, d=512, n_kv=1, block_size=128, cmp_ratio=128):
        super().__init__()
        self.n_q = n_q
        self.d = d
        self.n_kv = n_kv
        self.block_size = block_size
        self.cmp_ratio = cmp_ratio

    def forward(self, q, cmp_kv, sinks, cmp_block_table, seqused_kv, ori_kv, ori_block_table):
        b = int(seqused_kv.shape[0])
        bs = q.shape[0]
        s1 = bs // b
        nkv = cmp_kv.shape[2]
        d = cmp_kv.shape[3]
        softmax_scale = d ** -0.5

        compress_actual_seqs = seqused_kv // self.cmp_ratio
        k_cfa_bsnd = _kv_cache_concat_bsnd(cmp_kv, cmp_block_table, compress_actual_seqs)
        k_win_bsnd = _kv_cache_concat_bsnd(ori_kv, ori_block_table, seqused_kv)
        kv_bsnd = torch.cat([k_cfa_bsnd], dim=1)

        output_flash = torch.zeros_like(q)

        for i in range(b):
            for j in range(s1):
                seq_end = int(seqused_kv[i].item()) - (s1 - 1 - j)
                seq_len = max(seq_end // self.cmp_ratio, 0)
                if seq_len == 0:
                    continue
                for n2_idx in range(nkv):
                    q_bs = q[i * s1 + j]
                    win_start = max(seq_end - 128, 0)
                    kv_win_view = k_win_bsnd[i, win_start:seq_end, :, :].reshape(-1, d)
                    kv_bs = kv_bsnd[i, :seq_len, n2_idx:n2_idx + 1].reshape(seq_len, d)
                    kv_bs_cat = torch.cat([kv_win_view, kv_bs], dim=0)

                    qk = _matmul_proxy(q_bs.float(), kv_bs_cat.float().T) * softmax_scale
                    sm = _softmax_with_sinks(qk.float(), sinks, is_new_sink=True)
                    bmm2 = _matmul_proxy(sm.to(q.dtype), kv_bs_cat.to(q.dtype))
                    output_flash[i * s1 + j] = bmm2.to(q.dtype)

        return output_flash


def _gen_block_table(actual_seq_len, block_size, block_table_shape, cmp_ratio=128):
    block_num_per_batch = []
    block_num = 0
    for s in actual_seq_len:
        s_val = s.item() if isinstance(s, torch.Tensor) else s
        block_num_per_batch.append(math.ceil(s_val // cmp_ratio / block_size))
        block_num += math.ceil(s_val / block_size)
    block_idx_list = torch.arange(0, block_num, dtype=torch.int32)
    block_idx_list = block_idx_list[torch.randperm(block_idx_list.size(0))]
    cmp_block_table = torch.full(block_table_shape, -1, dtype=torch.int32)
    bi = 0
    btb = 0
    for idx in block_num_per_batch:
        for j in range(idx):
            cmp_block_table[btb, j] = block_idx_list[bi]
            bi += 1
        btb += 1
    return cmp_block_table


def get_inputs():
    b, s2, n_q, d, block_size, cmp_ratio, s1 = 1, 1024, 64, 512, 128, 128, 2
    t = b * s1
    seqused_kv = torch.tensor([s2] * b, dtype=torch.int32)

    max_cmp_blocks = (s2 // cmp_ratio + block_size - 1) // block_size
    max_ori_blocks = (s2 + s1 - 1 + block_size - 1) // block_size

    cmp_blk_tbl_shape = (b, max(max_cmp_blocks, 1))
    ori_blk_tbl_shape = (b, max(max_ori_blocks, 1))

    cmp_block_table = _gen_block_table(seqused_kv, block_size, cmp_blk_tbl_shape, cmp_ratio=cmp_ratio)
    ori_block_table = _gen_block_table(seqused_kv, block_size, ori_blk_tbl_shape, cmp_ratio=1)

    cmp_kv_blocks = max(cmp_block_table.max().item() + 1, 1)
    ori_kv_blocks = max(ori_block_table.max().item() + 1, 1)

    q = torch.randn(t, n_q, d, dtype=torch.bfloat16)
    cmp_kv = torch.randn(cmp_kv_blocks, block_size, 1, d, dtype=torch.bfloat16)
    sinks = torch.randn(n_q, dtype=torch.float32)
    ori_kv = torch.randn(ori_kv_blocks, block_size, 1, d, dtype=torch.bfloat16)

    return [q, cmp_kv, sinks, cmp_block_table, seqused_kv, ori_kv, ori_block_table]


def get_init_inputs():
    return [64, 512, 1, 128, 128]
