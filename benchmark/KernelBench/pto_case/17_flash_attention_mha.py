#!/usr/bin/env python3
# coding: utf-8
"""KernelBench case: Flash Attention MHA Forward — varlen multi-head attention with online softmax."""

import math
import torch
import torch.nn as nn

FORMULA = "scores=Q@K^T*scale; O,L,M = online_softmax(scores)@V per head per batch"
DYNAMIC_AXIS = ["M"]


class Model(nn.Module):
    def __init__(self, num_heads: int = 8, head_dim: int = 64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Varlen flash attention forward.

        Args:
            q: [total_q, num_heads*head_dim] BF16
            k: [total_kv, num_heads*head_dim] BF16
            v: [total_kv, num_heads*head_dim] BF16
            cu_seqlens_q: [batch+1] INT32 — cumulative Q seq lens
            cu_seqlens_k: [batch+1] INT32 — cumulative KV seq lens
        Returns:
            o:   [total_q, num_heads*head_dim] BF16 — attention output
            l:   [total_q, 1] FP32 — softmax sum
            m:   [total_q, 1] FP32 — softmax max
        """
        hidden_dim = self.num_heads * self.head_dim
        batch_size = cu_seqlens_q.shape[0] - 1

        o = torch.zeros(q.shape[0], hidden_dim, dtype=torch.float32, device=q.device)
        l_out = torch.zeros(q.shape[0], 1, dtype=torch.float32, device=q.device)
        m_out = torch.zeros(q.shape[0], 1, dtype=torch.float32, device=q.device)

        for b in range(batch_size):
            q_start = cu_seqlens_q[b].item()
            q_end = cu_seqlens_q[b + 1].item()
            sq = q_end - q_start
            k_start = cu_seqlens_k[b].item()
            k_end = cu_seqlens_k[b + 1].item()
            sk = k_end - k_start

            for h in range(self.num_heads):
                h_off = h * self.head_dim
                q_h = q[q_start:q_end, h_off:h_off + self.head_dim].float()
                k_h = k[k_start:k_end, h_off:h_off + self.head_dim].float()
                v_h = v[k_start:k_end, h_off:h_off + self.head_dim].float()

                scores = torch.matmul(q_h, k_h.T) * self.scale
                m_h = scores.max(dim=-1, keepdim=True)[0]
                p = torch.exp(scores - m_h)
                l_h = p.sum(dim=-1, keepdim=True)
                o_h = torch.matmul(p / l_h, v_h)

                o[q_start:q_end, h_off:h_off + self.head_dim] = o_h
                l_out[q_start:q_end, 0:1] = l_h
                m_out[q_start:q_end, 0:1] = m_h

        return o.to(torch.bfloat16), l_out, m_out


def get_inputs():
    batch_size = 8
    num_heads = 8
    head_dim = 64
    s1 = 320
    s2 = 320
    hidden_dim = num_heads * head_dim
    total_q = batch_size * s1
    total_kv = batch_size * s2

    torch.manual_seed(42)
    q = torch.randn(total_q, hidden_dim, dtype=torch.bfloat16) * 0.1
    k = torch.randn(total_kv, hidden_dim, dtype=torch.bfloat16) * 0.1
    v = torch.randn(total_kv, hidden_dim, dtype=torch.bfloat16) * 0.1
    cu_seqlens_q = torch.tensor([i * s1 for i in range(batch_size + 1)], dtype=torch.int32)
    cu_seqlens_k = torch.tensor([i * s2 for i in range(batch_size + 1)], dtype=torch.int32)
    return [q, k, v, cu_seqlens_q, cu_seqlens_k]


def get_init_inputs():
    return [8, 64]
