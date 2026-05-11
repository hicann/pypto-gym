#!/usr/bin/env python3
# coding: utf-8
"""KernelBench case: DeepSeek V3.2 Lightning Indexer Prolog Quant — quantized query+key+weights for indexer."""

import math
import torch
import torch.nn as nn

FORMULA = (
    "q = DeQuant(Hadamard(RoPE(DeQuant(q_norm @ w_qb)))); "
    "k = DeQuant(Hadamard(RoPE(LayerNorm(x @ wk)))); "
    "weights = (x @ w_proj) * (n*d)^-0.5"
)
DYNAMIC_AXIS = ["M"]


class Model(nn.Module):
    def __init__(
        self,
        t: int = 4,
        h: int = 7168,
        q_lora_rank: int = 1536,
        idx_n_heads: int = 64,
        idx_head_dim: int = 128,
        rope_head_dim: int = 64,
    ):
        super().__init__()
        self.t = t
        self.h = h
        self.q_lora_rank = q_lora_rank
        self.idx_n_heads = idx_n_heads
        self.idx_head_dim = idx_head_dim
        self.rope_head_dim = rope_head_dim

        nph = idx_n_heads * idx_head_dim
        self.w_qb = nn.Parameter(
            torch.randn(q_lora_rank, nph, dtype=torch.float32) * 0.01 / math.sqrt(q_lora_rank),
        )
        self.wk = nn.Parameter(
            torch.randn(h, idx_head_dim, dtype=torch.float32) * 0.01 / math.sqrt(h),
        )
        self.w_proj = nn.Parameter(
            torch.randn(h, idx_n_heads, dtype=torch.float32) * 0.01 / math.sqrt(h),
        )
        self.ln_gamma = nn.Parameter(torch.ones(idx_head_dim, dtype=torch.float32))
        self.ln_beta = nn.Parameter(torch.zeros(idx_head_dim, dtype=torch.float32))
        self.hadamard_q = nn.Parameter(
            torch.randn(idx_head_dim, idx_head_dim, dtype=torch.float32) * 0.01,
        )
        self.hadamard_k = nn.Parameter(
            torch.randn(idx_head_dim, idx_head_dim, dtype=torch.float32) * 0.01,
        )

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def _rope_3d(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        cos_f32 = cos.float().unsqueeze(1)
        sin_f32 = sin.float().unsqueeze(1)
        x_embed = x_f32 * cos_f32 + Model._rotate_half(x_f32) * sin_f32
        return x_embed.to(x.dtype)

    @staticmethod
    def _layer_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x_f32 = x.float()
        mean = x_f32.mean(dim=-1, keepdim=True)
        var = ((x_f32 - mean) ** 2).mean(dim=-1, keepdim=True)
        x_norm = (x_f32 - mean) / torch.sqrt(var + 1e-6)
        return (x_norm * gamma.float() + beta.float()).to(x_dtype)

    @staticmethod
    def _per_token_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        max_val = x.float().abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-8)
        scale = 127.0 / max_val
        q = (x.float() * scale).round().clamp(-128, 127).to(torch.int8)
        deq_scale = 1.0 / scale
        return q, deq_scale

    def forward(
        self, hidden_states: torch.Tensor, q_norm_in: torch.Tensor, q_norm_scale_in: torch.Tensor,
        cos_in: torch.Tensor, sin_in: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        x_dtype = hidden_states.dtype
        t, h = hidden_states.shape

        q_norm_i32 = q_norm_in.float()
        w_qb = self.w_qb.float()
        q_i32 = torch.matmul(q_norm_i32, w_qb)
        q_fp32 = q_i32.float() * q_norm_scale_in.float()
        q_bf16 = q_fp32.reshape(t, self.idx_n_heads, self.idx_head_dim).to(x_dtype)

        q_rope, q_nope = torch.split(q_bf16, [self.rope_head_dim, self.idx_head_dim - self.rope_head_dim], dim=-1)
        q_rope = self._rope_3d(q_rope, cos_in, sin_in)
        q_cat = torch.cat([q_rope, q_nope], dim=-1)
        q_hd = (q_cat.float() @ self.hadamard_q.float()).to(x_dtype)
        q_int8, q_scale = self._per_token_quantize(q_hd)
        q_scale = q_scale.to(torch.float16)

        k_bf16 = (hidden_states.float() @ self.wk.float()).to(x_dtype)
        k_norm = self._layer_norm(k_bf16, self.ln_gamma, self.ln_beta)
        k_rope, k_nope = torch.split(k_norm, [self.rope_head_dim, self.idx_head_dim - self.rope_head_dim], dim=-1)
        k_rope = self._rope_3d(k_rope.unsqueeze(1), cos_in, sin_in).squeeze(1)
        k_cat = torch.cat([k_rope, k_nope], dim=-1)
        k_hd = (k_cat.float() @ self.hadamard_k.float()).to(x_dtype)
        k_int8, k_scale = self._per_token_quantize(k_hd)

        weights_f32 = (hidden_states.float() @ self.w_proj.float()).to(x_dtype).float()
        weights = weights_f32 * (self.idx_n_heads ** -0.5) * (self.idx_head_dim ** -0.5)
        weights = weights.to(torch.float16)

        return (q_int8, q_scale, k_int8, weights)


def get_inputs():
    t = 4
    h = 7168
    q_lora_rank = 1536
    idx_n_heads = 64
    idx_head_dim = 128
    rope_head_dim = 64

    x = torch.randn(t, h, dtype=torch.bfloat16) * 0.01 / math.sqrt(h)
    q_norm = torch.randint(-128, 128, (t, q_lora_rank), dtype=torch.int8)
    q_norm_scale = torch.randn(t, 1, dtype=torch.float32).abs().clamp(min=0.01)
    cos = torch.randn(t, rope_head_dim, dtype=torch.bfloat16) * 0.01
    sin = torch.randn(t, rope_head_dim, dtype=torch.bfloat16) * 0.01
    return [x, q_norm, q_norm_scale, cos, sin]


def get_init_inputs():
    return [4, 7168, 1536, 64, 128, 64]
