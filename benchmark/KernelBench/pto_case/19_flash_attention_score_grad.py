#!/usr/bin/env python3
# coding: utf-8
"""KernelBench case: Flash Attention Score Grad — backward for flash attention (online softmax recompute P)."""

import math
import torch
import torch.nn as nn

FORMULA = "P=exp(Q@K^T*scale-max)/sum; D=sum(dO*O); dP=dO@V^T; dS=P*(dP-D); dV=P^T@dO; dQ=dS@K*scale; dK=dS^T@Q*scale"
DYNAMIC_AXIS = ["M"]


class Model(nn.Module):
    def __init__(self, num_heads: int = 8, head_dim: int = 64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dy: torch.Tensor,
        softmax_max: torch.Tensor,
        softmax_sum: torch.Tensor,
        attention_out: torch.Tensor,
        scale_value: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Flash attention score gradient.

        Args:
            query: [B, N, S, D] BF16
            key: [B, N, S, D] BF16
            value: [B, N, S, D] BF16
            dy: [B, N, S, D] BF16 — output gradient
            softmax_max: [B, N, S, 1] FP32 — from forward
            softmax_sum: [B, N, S, 1] FP32 — from forward
            attention_out: [B, N, S, D] BF16 — from forward
            scale_value: float
        Returns:
            dq: [B, N, S, D] BF16
            dk: [B, N, S, D] BF16
            dv: [B, N, S, D] BF16
        """
        orig_dtype = query.dtype

        q = query.float()
        k = key.float()
        v = value.float()
        dy_f = dy.float()
        attn_out_f = attention_out.float()
        s_max = softmax_max[:, :, :, 0:1]
        s_sum = softmax_sum[:, :, :, 0:1]

        scores = torch.matmul(q, k.transpose(-2, -1)) * scale_value
        p_mat = torch.exp(scores - s_max) / s_sum

        d_var = (dy_f * attn_out_f).sum(dim=-1, keepdim=True)
        dp_var = torch.matmul(dy_f, v.transpose(-2, -1))
        ds_var = p_mat * (dp_var - d_var)

        dq_out = torch.matmul(ds_var, k) * scale_value
        dk_out = torch.matmul(ds_var.transpose(-2, -1), q) * scale_value
        dv_out = torch.matmul(p_mat.transpose(-2, -1), dy_f)

        return dq_out.to(orig_dtype), dk_out.to(orig_dtype), dv_out.to(orig_dtype)


def get_inputs():
    B = 2
    N = 8
    S = 64
    D = 64
    scale_value = 1.0 / math.sqrt(D)

    torch.manual_seed(42)
    q = torch.randn(B, N, S, D, dtype=torch.bfloat16) * 0.01
    k = torch.randn(B, N, S, D, dtype=torch.bfloat16) * 0.01
    v = torch.randn(B, N, S, D, dtype=torch.bfloat16) * 0.01
    dy = torch.randn(B, N, S, D, dtype=torch.bfloat16) * 0.01

    # Precompute forward outputs
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale_value
    row_max = scores.amax(dim=-1, keepdim=True)
    exp_scores = torch.exp(scores - row_max)
    row_sum = exp_scores.sum(dim=-1, keepdim=True)
    p = exp_scores / row_sum
    attn_out = torch.matmul(p, v_f).to(torch.bfloat16)

    sm = torch.zeros(B, N, S, 8, dtype=torch.float32)
    sm[:, :, :, 0:1] = row_max
    ss = torch.zeros(B, N, S, 8, dtype=torch.float32)
    ss[:, :, :, 0:1] = row_sum

    # Slice to [B,N,S,1]
    sm = sm[:, :, :, 0:1]
    ss = ss[:, :, :, 0:1]

    return [q, k, v, dy, sm, ss, attn_out, scale_value]


def get_init_inputs():
    return [8, 64]
