#!/usr/bin/env python3
# coding: utf-8
"""KernelBench-style level1 case: Flash Attention Score."""

import math
import torch
import torch.nn as nn


FORMULA = "out[b, h, sq, d] = Softmax((Q[b, h, sq, d] @ K[b, h, skv, d]^T / sqrt(d)) * mask[sq, skv]) @ V[b, h, skv, d]"
DYNAMIC_AXIS = ["B", "SQ", "SKV"]


class Model(nn.Module):
    def __init__(self, num_heads: int = 4, head_dim: int = 64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, atten_mask: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(atten_mask.unsqueeze(0).unsqueeze(0) == 1, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, value.float())
        return output.to(torch.bfloat16)


def get_inputs():
    batch_size = 2
    num_heads = 4
    seq_len_q = 64
    seq_len_kv = 64
    head_dim = 64

    query = torch.randn(batch_size, num_heads, seq_len_q, head_dim, dtype=torch.bfloat16)
    key = torch.randn(batch_size, num_heads, seq_len_kv, head_dim, dtype=torch.bfloat16)
    value = torch.randn(batch_size, num_heads, seq_len_kv, head_dim, dtype=torch.bfloat16)

    atten_mask = torch.zeros(seq_len_q, seq_len_kv, dtype=torch.float32)
    atten_mask[:, seq_len_kv // 2:] = 1.0

    return [query, key, value, atten_mask]


def get_init_inputs():
    return [4, 64]
