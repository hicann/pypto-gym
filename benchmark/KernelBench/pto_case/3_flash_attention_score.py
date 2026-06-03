#!/usr/bin/env python3
# coding: utf-8

"""KernelBench case: Flash Attention Score — online softmax with PSE, dropout, GQA."""

import math
import torch
import torch.nn as nn


FORMULA = "out[b, h, sq, d] = OnlineSoftmax((Q[b, h, sq, d] @ K[b, h_kv, skv, d]^T + PSE) * scale, mask, dropout) @ V[b, h_kv, skv, d]"
DYNAMIC_AXIS = ["B", "SQ", "SKV"]


class Model(nn.Module):
    def __init__(self, num_heads: int = 4, head_dim: int = 64, num_kv_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.scale = 1.0 / math.sqrt(head_dim)
        self.group = num_heads // num_kv_heads

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        atten_mask: torch.Tensor,
        pse: torch.Tensor,
        drop_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Flash attention score with PSE, dropout, GQA.

        Args:
            query: [B, N, Sq, D] BF16
            key: [B, N_kv, Skv, D] BF16
            value: [B, N_kv, Skv, D] BF16
            atten_mask: [Sq, Skv] float32, 0=valid, 1=invalid
            pse: [B, N, Sq, Skv] BF16 — positional score encoding
            drop_mask: [Sq, Skv] BF16 — dropout mask (binary 0/1, 1=keep)
        Returns:
            output: [B, N, Sq, D] BF16
            softmax_max: [B, N, Sq, 1] FP32
            softmax_sum: [B, N, Sq, 1] FP32
        """
        b, n, sq, d = query.shape
        _, n_kv, skv, _ = key.shape
        group = self.group
        scale = self.scale

        query_fp32 = query.float()
        key_fp32 = key.float()
        value_fp32 = value.float()
        mask_fp32 = atten_mask.float()
        pse_fp32 = pse.float()
        drop_fp32 = drop_mask.float()

        output = torch.zeros(b, n, sq, d, dtype=torch.float32, device=query.device)
        softmax_max = torch.zeros(b, n, sq, 1, dtype=torch.float32, device=query.device)
        softmax_sum = torch.zeros(b, n, sq, 1, dtype=torch.float32, device=query.device)

        for b_idx in range(b):
            for h_idx in range(n):
                kv_h = h_idx // group
                q_h = query_fp32[b_idx, h_idx, :, :]  # [Sq, D]
                k_h = key_fp32[b_idx, kv_h, :, :]     # [Skv, D]
                v_h = value_fp32[b_idx, kv_h, :, :]   # [Skv, D]

                scores = torch.matmul(q_h, torch.transpose(k_h, 0, 1))  # [Sq, Skv]
                scores = (scores + pse_fp32[b_idx, h_idx, :, :]) * scale

                scores = scores - mask_fp32 * 1e4

                scores = scores * drop_fp32

                row_max = scores.max(dim=-1, keepdim=True)[0]
                exp_scores = torch.exp(scores - row_max)
                row_sum = exp_scores.sum(dim=-1, keepdim=True)
                p = exp_scores / row_sum

                out_h = torch.matmul(p, v_h)

                output[b_idx, h_idx, :, :] = out_h
                softmax_max[b_idx, h_idx, :, :] = row_max
                softmax_sum[b_idx, h_idx, :, :] = row_sum

        return output.to(torch.bfloat16), softmax_max, softmax_sum


def get_inputs():
    batch_size = 2
    num_heads = 4
    num_kv_heads = 4
    seq_len_q = 64
    seq_len_kv = 64
    head_dim = 64

    torch.manual_seed(42)
    query = torch.randn(batch_size, num_heads, seq_len_q, head_dim, dtype=torch.bfloat16) * 0.1
    key = torch.randn(batch_size, num_kv_heads, seq_len_kv, head_dim, dtype=torch.bfloat16) * 0.1
    value = torch.randn(batch_size, num_kv_heads, seq_len_kv, head_dim, dtype=torch.bfloat16) * 0.1

    atten_mask = torch.zeros(seq_len_q, seq_len_kv, dtype=torch.float32)
    atten_mask[:, seq_len_kv // 2:] = 1.0

    pse = torch.randn(batch_size, num_heads, seq_len_q, seq_len_kv, dtype=torch.bfloat16) * 0.01
    drop_mask = torch.ones(seq_len_q, seq_len_kv, dtype=torch.bfloat16)

    return [query, key, value, atten_mask, pse, drop_mask]


def get_init_inputs():
    return [4, 64, 4]
