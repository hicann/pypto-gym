#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn

FORMULA = "q[t,idx_nq*hd], weights[t,idx_nq], q_scale[t,idx_nq] = lightning_indexer_quant(qr, idx_wq_b, x, weights_proj, cos, sin, hadamard)"
DYNAMIC_AXIS = ["T"]


class Model(nn.Module):
    def __init__(
        self, h: int = 2048, idx_nq: int = 32, head_dim: int = 192,
        q_lora_rank: int = 256, rope_dim: int = 64
    ):
        super().__init__()
        self.h = h
        self.idx_nq = idx_nq
        self.head_dim = head_dim
        self.q_lora_rank = q_lora_rank
        self.rope_dim = rope_dim

        self.idx_wq_b = nn.Parameter(torch.randn(q_lora_rank, idx_nq * head_dim, dtype=torch.float32))
        self.weights_proj = nn.Parameter(torch.randn(h, idx_nq, dtype=torch.bfloat16))
        self.hadamard = nn.Parameter(torch.randn(head_dim, head_dim, dtype=torch.bfloat16))

    def forward(
        self, qr: torch.Tensor, x: torch.Tensor,
        cos: torch.Tensor, sin: torch.Tensor,
        qr_scale: torch.Tensor, idx_wq_b_scale: torch.Tensor,
    ):
        t = qr.shape[0]
        calc_dtype = torch.bfloat16

        q_fp32 = torch.matmul(qr.to(torch.float32), self.idx_wq_b.to(torch.float32)).to(torch.float32)
        q_fp32 = q_fp32 * qr_scale * idx_wq_b_scale.reshape(1, self.idx_nq * self.head_dim)
        q_re = q_fp32.to(calc_dtype).reshape(t, self.idx_nq, self.head_dim)
        q_nope, q_rope = torch.split(q_re, [self.head_dim - self.rope_dim, self.rope_dim], dim=-1)

        k_rope = q_rope[:, :1, :]
        q_rope_r = apply_rotary_pos_emb(q_rope, cos, sin)
        k_rope_r = apply_rotary_pos_emb(k_rope, cos, sin)
        q_full = torch.cat([q_nope, q_rope_r], dim=-1)

        q_hadamard = torch.matmul(
            q_full.to(torch.float32), self.hadamard.reshape(1, self.head_dim, self.head_dim).to(torch.float32)
        ).to(calc_dtype)

        q_int8, q_scale_out = per_token_quant(q_hadamard)
        q_scale_out = q_scale_out.to(torch.float16).reshape(t, self.idx_nq)

        weights = torch.matmul(
            x.to(torch.float32), self.weights_proj.to(torch.float32).unsqueeze(0)
        ).to(torch.float16).squeeze(1)

        return q_int8.reshape(t, self.idx_nq * self.head_dim), weights, q_scale_out


def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, cos, sin):
    orig = q.dtype
    q_f32 = q.to(torch.float32)
    cos_f32 = cos.to(torch.float32).unsqueeze(1)
    sin_f32 = sin.to(torch.float32).unsqueeze(1)
    t, n, d = q_f32.shape
    q_re = q_f32.reshape(t, n, d//2, 2)
    q_rot = rotate_half(q_re).reshape(t, n, d)
    q_embed = (q_f32 * cos_f32) + (q_rot * sin_f32)
    return q_embed.to(orig)


def per_token_quant(x):
    x_f32 = x.to(torch.float32)
    abs_res = torch.abs(x_f32)
    max_val = torch.max(abs_res, dim=-1, keepdims=True)[0]
    scale = 127.0 / max_val
    out = torch.round(x_f32 * scale).clamp(-127, 127).to(torch.int8)
    deq_scale = 1.0 / scale
    return out, deq_scale


def get_inputs():
    t, h, rope_dim = 128, 2048, 64
    idx_nq, head_dim, q_lora_rank = 32, 192, 256

    qr = torch.randint(-10, 10, (t, q_lora_rank), dtype=torch.int8)
    x = torch.empty(t, h, dtype=torch.bfloat16).uniform_(-1, 1)
    cos = torch.empty(t, rope_dim, dtype=torch.bfloat16).uniform_(-1, 1)
    sin = torch.empty(t, rope_dim, dtype=torch.bfloat16).uniform_(-1, 1)
    qr_scale = torch.ones(t, 1, dtype=torch.float32)
    idx_wq_b_scale = torch.ones(idx_nq * head_dim, 1, dtype=torch.float32)

    return [qr, x, cos, sin, qr_scale, idx_wq_b_scale]


def get_init_inputs():
    return [2048, 32, 192, 256, 64]
