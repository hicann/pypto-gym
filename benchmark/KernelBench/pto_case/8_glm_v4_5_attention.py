#!/usr/bin/env python3
# coding: utf-8
"""KernelBench case: GLM V4.5 Paged Attention — flash attention with paged KV cache."""

import math
import torch
import torch.nn as nn

FORMULA = "attn_out[t, h, d] = Softmax( Zoom( Q[t, h, d] @ K_cache[t, seq, h_kv, d]^T ) ) @ V_cache[t, seq, h_kv, d]"
DYNAMIC_AXIS = ["B", "S"]


class Model(nn.Module):
    def __init__(self, num_heads: int = 12, head_size: int = 128,
                 kv_num_heads: int = 1, block_size: int = 128,
                 num_blocks: int = 256, max_seq_len: int = 16384):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.kv_num_heads = kv_num_heads
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.max_seq_len = max_seq_len
        self.softmax_scale = head_size ** -0.5

    def forward(self, query: torch.Tensor, key_cache: torch.Tensor,
                value_cache: torch.Tensor, block_tables: torch.Tensor,
                actual_seqs: torch.Tensor
                ) -> torch.Tensor:
        b = actual_seqs.shape[0]
        num_tokens = query.shape[0]
        nq = self.num_heads
        nkv = self.kv_num_heads
        d = self.head_size
        block_size = self.block_size
        group = nq // nkv
        s1 = num_tokens // b

        k_nope = key_cache.view(-1, nkv * d)
        v_nope = value_cache.view(-1, nkv * d)

        output = torch.zeros(num_tokens, nq, d, dtype=query.dtype, device=query.device)

        for b_idx in range(b):
            for s1_idx in range(s1):
                cur_seq = actual_seqs[b_idx].item() - (s1 - 1 - s1_idx)
                token_idx = b_idx * s1 + s1_idx

                for n2_idx in range(nkv):
                    q_group = query[token_idx, n2_idx * group:(n2_idx + 1) * group]

                    k_list = []
                    v_list = []
                    for blk_i in range(block_tables.shape[1]):
                        blk = block_tables[b_idx, blk_i].item()
                        if blk < 0:
                            break
                        k_list.append(k_nope[blk * block_size:(blk + 1) * block_size, n2_idx * d:(n2_idx + 1) * d])
                        v_list.append(v_nope[blk * block_size:(blk + 1) * block_size, n2_idx * d:(n2_idx + 1) * d])
                    if not k_list:
                        continue
                    k_seq = torch.cat(k_list, dim=0)[:cur_seq]
                    v_seq = torch.cat(v_list, dim=0)[:cur_seq]

                    scores = torch.matmul(q_group.float(), k_seq.float().t()) * self.softmax_scale
                    attn_weights = torch.softmax(scores, dim=-1).to(query.dtype)
                    attn_out = torch.matmul(attn_weights.float(), v_seq.float()).to(query.dtype)
                    output[token_idx, n2_idx * group:(n2_idx + 1) * group] = attn_out

        return output


def get_inputs():
    b = 4
    s1 = 1
    nq = 12
    nkv = 1
    d = 128
    block_size = 128
    num_blocks = 128
    max_seq_len = 16384
    s2 = 2048

    q = torch.randn(b * s1, nq, d, dtype=torch.bfloat16) / math.sqrt(d)
    k = torch.randn(num_blocks, block_size, nkv, d, dtype=torch.bfloat16) / math.sqrt(d)
    v = torch.randn(num_blocks, block_size, nkv, d, dtype=torch.bfloat16) / math.sqrt(d)

    block_tables = torch.zeros(b, (s2 + block_size - 1) // block_size, dtype=torch.int32)
    blk_idx = 0
    for bi in range(b):
        for j in range((s2 + block_size - 1) // block_size):
            if blk_idx < num_blocks:
                block_tables[bi, j] = blk_idx
                blk_idx += 1

    actual_seqs = torch.full((b,), s2, dtype=torch.int32)
    return [q, k, v, block_tables, actual_seqs]


def get_init_inputs():
    return [12, 128, 1, 128, 256, 16384]
