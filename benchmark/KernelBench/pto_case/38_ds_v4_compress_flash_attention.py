#!/usr/bin/env python3
# coding: utf-8

import math
import torch
import torch.nn as nn

FORMULA = "out[t,n,d] = softmax(q[t,n,d] @ kv_cache[t,s,d]^T / sqrt(d) + sinks) @ kv_cache[t,s,d]"
DYNAMIC_AXIS = ["T", "S"]


class Model(nn.Module):
    def __init__(self, n_q: int = 64, d: int = 512, block_size: int = 128):
        super().__init__()
        self.n_q = n_q
        self.d = d
        self.block_size = block_size

    def forward(
        self, q: torch.Tensor, kv_cache: torch.Tensor, block_table: torch.Tensor,
        seqused_kv: torch.Tensor, sinks: torch.Tensor
    ) -> torch.Tensor:
        t, n1, d = q.shape
        b = block_table.shape[0]
        scalar = d ** -0.5
        atten_out = torch.zeros(t, n1, d, dtype=torch.bfloat16, device=q.device)

        for b_idx in range(b):
            actual_seq = seqused_kv[b_idx].item()
            cur_s_q = t // b
            for s1_idx in range(min(cur_s_q, t - b_idx * cur_s_q)):
                t_idx = b_idx * cur_s_q + s1_idx
                if t_idx >= t:
                    break

                qi = q[t_idx, :, :]
                cur_seq = actual_seq - cur_s_q + s1_idx + 1
                valid_len = min(cur_seq, actual_seq)
                start_block = 0
                end_block = (valid_len - 1) // self.block_size
                start_offset = 0

                kv_list = []
                for blk in range(start_block, end_block + 1):
                    pid = block_table[b_idx, blk].item()
                    if pid >= 0:
                        nv = min(self.block_size, valid_len - blk * self.block_size)
                        kv_list.append(kv_cache[pid * self.block_size: pid * self.block_size + nv, :d])

                if not kv_list:
                    continue

                kj = torch.cat(kv_list, dim=0)
                sij = torch.matmul(qi.to(torch.float32), kj.T.to(torch.float32)) * scalar
                sij_exp = torch.exp(sij - sij.max(dim=-1, keepdim=True)[0])
                sij_sum = sij_exp.sum(-1, keepdim=True)
                attn_w = sij_exp / (sij_sum + torch.exp(sinks.unsqueeze(1) - sij.max(dim=-1, keepdim=True)[0]))
                atten_out[t_idx:t_idx+1, :, :] = torch.matmul(attn_w, kj.to(torch.float32)).to(torch.bfloat16)

        return atten_out


def get_inputs():
    b, s, n_q, d, block_size = 1, 1024, 64, 512, 128
    t = b * 1
    kv_blocks = (s + block_size - 1) // block_size
    kv_total = kv_blocks * block_size

    q = torch.randn(t, n_q, d, dtype=torch.bfloat16)
    kv_cache = torch.randn(kv_blocks * block_size, d, dtype=torch.bfloat16)
    block_table = torch.arange(kv_blocks, dtype=torch.int32).reshape(b, kv_blocks)
    seqused_kv = torch.tensor([s], dtype=torch.int32)
    sinks = torch.randn(n_q, dtype=torch.float32)
    return [q, kv_cache, block_table, seqused_kv, sinks]


def get_init_inputs():
    return [64, 512, 128]
