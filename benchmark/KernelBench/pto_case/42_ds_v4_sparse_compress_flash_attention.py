#!/usr/bin/env python3
# coding: utf-8

import math
import torch
import torch.nn as nn

FORMULA = "out[t,n,d] = sparse_compress_fa(q[t,n,d], kv_cache, compress_kv, block_table, sparse_indices, sinks)"
DYNAMIC_AXIS = ["T", "S"]


class Model(nn.Module):
    def __init__(self, n_q: int = 64, d: int = 512, block_size: int = 128, cmp_ratio: int = 128):
        super().__init__()
        self.n_q = n_q
        self.d = d
        self.block_size = block_size
        self.cmp_ratio = cmp_ratio

    def forward(
        self, q: torch.Tensor, kv_cache: torch.Tensor, compress_kv: torch.Tensor,
        block_table: torch.Tensor, cmp_block_table: torch.Tensor,
        seqused_kv: torch.Tensor, sparse_indices: torch.Tensor, sinks: torch.Tensor
    ) -> torch.Tensor:
        t = q.shape[0]
        b = block_table.shape[0]
        scalar = self.d ** -0.5
        atten_out = torch.zeros(t, self.n_q, self.d, dtype=torch.bfloat16, device=q.device)
        n1 = self.n_q

        for b_idx in range(b):
            actual_seq = seqused_kv[b_idx].item()
            cur_s_q = t // b if b > 0 else t

            for s1_idx in range(min(cur_s_q, t - b_idx * cur_s_q)):
                t_idx = b_idx * cur_s_q + s1_idx
                if t_idx >= t:
                    break

                qi = q[t_idx, :, :]

                win_size = min(actual_seq, self.block_size)
                win_start = max(0, actual_seq - cur_s_q + s1_idx + 1 - win_size)

                kv_list = []
                for pos in range(win_start, actual_seq - cur_s_q + s1_idx + 1):
                    blk = pos // self.block_size
                    off = pos % self.block_size
                    pid = block_table[b_idx, blk].item()
                    if pid >= 0:
                        kv_list.append(kv_cache[pid * self.block_size + off, :self.d])

                if not kv_list:
                    continue

                kj = torch.stack(kv_list, dim=0)
                acc_s = torch.matmul(qi.to(torch.float32), kj.T.to(torch.float32)) * scalar
                scores_max = acc_s.max(dim=-1, keepdim=True)[0]
                acc_s_exp = torch.exp(acc_s - scores_max)
                sum_exp = acc_s_exp.sum(-1, keepdim=True)
                sum_exp += torch.exp(sinks.reshape(n1, 1) - scores_max)
                atten_out[t_idx, :, :] = torch.matmul(acc_s_exp / sum_exp, kj.to(torch.float32)).to(torch.bfloat16)

        return atten_out


def get_inputs():
    b, s, n_q, d, block_size = 1, 1024, 64, 512, 128
    t = 2
    kv_blocks = (s + block_size - 1) // block_size
    cmp_blocks = (s // 128 + block_size - 1) // block_size

    q = torch.randn(t, n_q, d, dtype=torch.bfloat16)
    kv_cache = torch.randn(kv_blocks * block_size, d, dtype=torch.bfloat16)
    compress_kv = torch.randn(cmp_blocks * block_size, d, dtype=torch.bfloat16)
    block_table = torch.arange(kv_blocks, dtype=torch.int32).reshape(b, kv_blocks)
    cmp_block_table = torch.arange(cmp_blocks, dtype=torch.int32).reshape(b, cmp_blocks)
    seqused_kv = torch.tensor([s], dtype=torch.int32)
    sparse_indices = torch.zeros(t, 128, dtype=torch.int32)
    sinks = torch.randn(n_q, dtype=torch.float32)

    return [q, kv_cache, compress_kv, block_table, cmp_block_table, seqused_kv, sparse_indices, sinks]


def get_init_inputs():
    return [64, 512, 128, 128]
