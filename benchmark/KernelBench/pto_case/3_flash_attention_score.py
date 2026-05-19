#!/usr/bin/env python3
# coding: utf-8
"""KernelBench-style level1 case: Flash Attention Score."""

import math
import torch
import torch.nn as nn


FORMULA = "out[b, h, sq, d] = OnlineSoftmax(Q[b, h, sq, d] @ K[b, h, skv, d]^T / sqrt(d), mask[sq, skv]) @ V[b, h, skv, d]"
DYNAMIC_AXIS = ["B", "SQ", "SKV"]


class Model(nn.Module):
    def __init__(self, num_heads: int = 4, head_dim: int = 64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, atten_mask: torch.Tensor) -> torch.Tensor:
        b, n, sq, d = query.shape
        _, _, skv, _ = key.shape

        query_fp32 = query.float()
        key_fp32 = key.float()
        value_fp32 = value.float()

        output = torch.zeros(b, n, sq, d, dtype=torch.float32, device=query.device)

        for b_idx in range(b):
            for n_idx in range(n):
                for q_idx in range(sq):
                    q_vec = query_fp32[b_idx, n_idx, q_idx, :]

                    max_score = float('-inf')
                    sum_exp = 0.0
                    output_vec = torch.zeros(d, dtype=torch.float32, device=query.device)

                    for kv_idx in range(skv):
                        if atten_mask[q_idx, kv_idx] == 1:
                            continue

                        k_vec = key_fp32[b_idx, n_idx, kv_idx, :]
                        score = torch.dot(q_vec, k_vec) * self.scale

                        new_max = max(max_score, score.item())

                        if new_max > max_score:
                            correction = math.exp(max_score - new_max)
                            sum_exp = sum_exp * correction
                            output_vec = output_vec * correction
                            max_score = new_max

                        exp_score = math.exp(score - max_score)
                        sum_exp += exp_score

                        v_vec = value_fp32[b_idx, n_idx, kv_idx, :]
                        output_vec += exp_score * v_vec

                    if sum_exp > 0:
                        output[b_idx, n_idx, q_idx, :] = output_vec / sum_exp

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
