#!/usr/bin/env python3
# coding: utf-8
"""KernelBench case: Flash Attention MHA Backward — varlen gradient computation for flash attention."""

import math
import torch
import torch.nn as nn

FORMULA = "dQ=dS@K*scale; dK=dS^T@Q*scale; dV=P^T@dO; dS=P*(dP-D); dP=dO@V^T; D=sum(O*dO)"
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
        o: torch.Tensor,
        do: torch.Tensor,
        l_input: torch.Tensor,
        m_input: torch.Tensor,
        actual_q: torch.Tensor,
        actual_kv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Flash attention backward.

        Args:
            q: [total_q, num_heads, head_dim] BF16
            k: [total_kv, num_heads, head_dim] BF16
            v: [total_kv, num_heads, head_dim] BF16
            o: [total_q, num_heads, head_dim] BF16 — forward output
            do: [total_q, num_heads, head_dim] BF16 — output gradient
            l_input: [total_q, num_heads, 1] FP32 — softmax sum
            m_input: [total_q, num_heads, 1] FP32 — softmax max
            actual_q: [batch] INT32 — per-batch Q seqlens
            actual_kv: [batch] INT32 — per-batch KV seqlens
        Returns:
            dq: [total_q, hidden_dim] BF16
            dk: [total_kv, hidden_dim] BF16
            dv: [total_kv, hidden_dim] BF16
        """
        batch_size = actual_q.shape[0]
        hidden_dim = self.num_heads * self.head_dim
        s1_size = q.shape[0] // batch_size
        s2_size = k.shape[0] // batch_size

        dq_out = torch.zeros(q.shape[0], hidden_dim, dtype=torch.float32, device=q.device)
        dk_out = torch.zeros(k.shape[0], hidden_dim, dtype=torch.float32, device=k.device)
        dv_out = torch.zeros(k.shape[0], hidden_dim, dtype=torch.float32, device=k.device)

        for b in range(batch_size):
            sq = actual_q[b].item()
            skv = actual_kv[b].item()
            q_off = b * s1_size
            kv_off = b * s2_size

            for h in range(self.num_heads):
                h_off = h * self.head_dim
                q_h = q[q_off:q_off + sq, h, :].float()
                k_h = k[kv_off:kv_off + skv, h, :].float()
                v_h = v[kv_off:kv_off + skv, h, :].float()
                o_h = o[q_off:q_off + sq, h, :].float()
                do_h = do[q_off:q_off + sq, h, :].float()
                m_h = m_input[q_off:q_off + sq, h, 0:1].float()
                l_h = l_input[q_off:q_off + sq, h, 0:1].float()

                scores = torch.matmul(q_h, k_h.T) * self.scale
                p = torch.exp(scores - m_h) / l_h

                d_val = (o_h * do_h).sum(dim=-1, keepdim=True)
                dp = torch.matmul(do_h, v_h.T)
                ds = p * (dp - d_val)

                ds_bf16 = ds.to(torch.bfloat16).float()
                p_bf16 = p.to(torch.bfloat16).float()

                dq = torch.matmul(ds_bf16, k_h) * self.scale
                dk = torch.matmul(ds_bf16.T, q_h) * self.scale
                dv = torch.matmul(p_bf16.T, do_h)

                dq_out[q_off:q_off + sq, h_off:h_off + self.head_dim] = dq
                dk_out[kv_off:kv_off + skv, h_off:h_off + self.head_dim] = dk
                dv_out[kv_off:kv_off + skv, h_off:h_off + self.head_dim] = dv

        return dq_out.to(torch.bfloat16), dk_out.to(torch.bfloat16), dv_out.to(torch.bfloat16)


def get_inputs():
    batch_size = 8
    num_heads = 8
    head_dim = 64
    s1 = 320
    s2 = 320
    total_q = batch_size * s1
    total_kv = batch_size * s2
    hidden_dim = num_heads * head_dim
    scale = 1.0 / math.sqrt(head_dim)

    torch.manual_seed(42)
    q = torch.randn(total_q, num_heads, head_dim, dtype=torch.bfloat16) * 0.01
    k = torch.randn(total_kv, num_heads, head_dim, dtype=torch.bfloat16) * 0.01
    v = torch.randn(total_kv, num_heads, head_dim, dtype=torch.bfloat16) * 0.01

    # Precompute forward outputs L, M, O
    l_in = torch.empty(total_q, num_heads, 1, dtype=torch.float32)
    m_in = torch.empty(total_q, num_heads, 1, dtype=torch.float32)
    o_in = torch.empty(total_q, num_heads, head_dim, dtype=torch.bfloat16)

    for b in range(batch_size):
        q_off = b * s1
        kv_off = b * s2
        for h in range(num_heads):
            q_h = q[q_off:q_off + s1, h, :].float()
            k_h = k[kv_off:kv_off + s2, h, :].float()
            v_h = v[kv_off:kv_off + s2, h, :].float()
            scores = torch.matmul(q_h, k_h.T) * scale
            m = scores.max(dim=-1, keepdim=True)[0]
            p = torch.exp(scores - m)
            l = p.sum(dim=-1, keepdim=True)
            o = torch.matmul(p / l, v_h)
            l_in[q_off:q_off + s1, h, :] = l
            m_in[q_off:q_off + s1, h, :] = m
            o_in[q_off:q_off + s1, h, :] = o.to(torch.bfloat16)

    torch.manual_seed(2026)
    do = torch.randn(total_q, num_heads, head_dim, dtype=torch.bfloat16) * 0.01

    actual_q = torch.tensor([s1] * batch_size, dtype=torch.int32)
    actual_kv = torch.tensor([s2] * batch_size, dtype=torch.int32)

    return [q, k, v, o_in, do, l_in, m_in, actual_q, actual_kv]


def get_init_inputs():
    return [8, 64]
