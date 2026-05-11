#!/usr/bin/env python3
# coding: utf-8
"""KernelBench case: DeepSeek V3.2 MLA Indexer Prolog Quant — fused MLA prolog + indexer prolog in single forward pass."""

import math
import torch
import torch.nn as nn

FORMULA = (
    "[MLA]: q_norm,q_nope,q_rope = MLA_Proj(x, w_dq, w_uq_qr, w_uk, w_dkv_kr); "
    "[IP]:  q_idx = Hadamard(RoPE(DeQuant(q_norm @ w_qb))); "
    "k_idx = Hadamard(RoPE(LayerNorm(x @ wk))); "
    "weights = (x @ w_proj) * (n*d)^-0.5"
)
DYNAMIC_AXIS = ["M"]


class Model(nn.Module):
    def __init__(
        self,
        t: int = 4,
        n: int = 128,
        h: int = 7168,
        q_lora_rank: int = 1536,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        kv_lora_rank: int = 512,
        idx_n_heads: int = 64,
        idx_head_dim: int = 128,
        rope_head_dim: int = 64,
    ):
        super().__init__()
        self.t = t
        self.n = n
        self.h = h
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.idx_n_heads = idx_n_heads
        self.idx_head_dim = idx_head_dim
        self.rope_head_dim = rope_head_dim
        self.q_head_dim = qk_nope_head_dim + qk_rope_head_dim

        scale = 0.01
        self.w_dq = nn.Parameter(torch.randn(h, q_lora_rank, dtype=torch.float32) * scale / math.sqrt(h))
        self.w_uq_qr = nn.Parameter(
            torch.randn(q_lora_rank, n * self.q_head_dim, dtype=torch.float32) * scale / math.sqrt(q_lora_rank),
        )
        self.w_uk = nn.Parameter(
            torch.randn(n, qk_nope_head_dim, kv_lora_rank, dtype=torch.float32) * scale,
        )
        self.w_dkv_kr = nn.Parameter(
            torch.randn(h, kv_lora_rank + qk_rope_head_dim, dtype=torch.float32) * scale / math.sqrt(h),
        )
        self.gamma_cq = nn.Parameter(torch.randn(q_lora_rank, dtype=torch.float32))
        self.gamma_ckv = nn.Parameter(torch.randn(kv_lora_rank, dtype=torch.float32))

        nph = idx_n_heads * idx_head_dim
        self.w_qb = nn.Parameter(torch.randn(q_lora_rank, nph, dtype=torch.float32) * scale / math.sqrt(q_lora_rank))
        self.wk = nn.Parameter(torch.randn(h, idx_head_dim, dtype=torch.float32) * scale / math.sqrt(h))
        self.w_proj = nn.Parameter(torch.randn(h, idx_n_heads, dtype=torch.float32) * scale / math.sqrt(h))
        self.ln_gamma = nn.Parameter(torch.ones(idx_head_dim, dtype=torch.float32))
        self.ln_beta = nn.Parameter(torch.zeros(idx_head_dim, dtype=torch.float32))
        self.hadamard_q = nn.Parameter(torch.randn(idx_head_dim, idx_head_dim, dtype=torch.float32) * scale)
        self.hadamard_k = nn.Parameter(torch.randn(idx_head_dim, idx_head_dim, dtype=torch.float32) * scale)

    @staticmethod
    def _rms_norm(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        rms = torch.sqrt((x_f32 * x_f32).mean(dim=-1, keepdim=True) + 1e-5)
        return ((x_f32 / rms) * gamma.float()).to(x.dtype)

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
        x_f32 = x.float()
        mean = x_f32.mean(dim=-1, keepdim=True)
        var = ((x_f32 - mean) ** 2).mean(dim=-1, keepdim=True)
        x_norm = (x_f32 - mean) / torch.sqrt(var + 1e-6)
        return (x_norm * gamma.float() + beta.float()).to(x.dtype)

    @staticmethod
    def _per_token_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        max_val = x.float().abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)
        scale = 127.0 / max_val
        q = (x.float() * scale).round().clamp(-128, 127).to(torch.int8)
        deq_scale = 1.0 / scale
        return q, deq_scale

    def forward(self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x_dtype = hidden_states.dtype
        t, h = hidden_states.shape

        q_a = (hidden_states.float() @ self.w_dq.float()).to(x_dtype)
        q_a_norm = self._rms_norm(q_a, self.gamma_cq)

        q_b = (q_a_norm.float() @ self.w_uq_qr.float()).to(x_dtype)
        q_reshape = q_b.reshape(t, self.n, self.q_head_dim)

        q_nope_raw = q_reshape[:, :, :self.qk_nope_head_dim]
        q_nope_t = q_nope_raw.permute(1, 0, 2)
        q_nope_proj = (q_nope_t.float() @ self.w_uk.float()).to(x_dtype)
        q_nope = q_nope_proj.permute(1, 0, 2)

        q_rope_raw = q_reshape[:, :, self.qk_nope_head_dim:]
        q_rope_out = self._rope_3d(q_rope_raw, cos, sin)

        kv_a = (hidden_states.float() @ self.w_dkv_kr.float()).to(x_dtype)
        compressed_kv = kv_a[:, :self.kv_lora_rank]
        k_nope_norm = self._rms_norm(compressed_kv, self.gamma_ckv)

        q_norm_q, q_norm_scale = self._per_token_quantize(q_a_norm)
        q_idx_i32 = q_norm_q.float() @ self.w_qb.float()
        q_idx = q_idx_i32.float() * q_norm_scale.float()
        q_idx_3d = q_idx.reshape(t, self.idx_n_heads, self.idx_head_dim).to(x_dtype)

        q_r, q_np = torch.split(q_idx_3d, [self.rope_head_dim, self.idx_head_dim - self.rope_head_dim], dim=-1)
        q_r = self._rope_3d(q_r, cos, sin)
        q_cat = torch.cat([q_r, q_np], dim=-1)
        q_hd = (q_cat.float() @ self.hadamard_q.float()).to(x_dtype)
        q_idx_i8, q_idx_scale = self._per_token_quantize(q_hd)
        q_idx_scale = q_idx_scale.to(torch.float16)

        k_raw = (hidden_states.float() @ self.wk.float()).to(x_dtype)
        k_ln = self._layer_norm(k_raw, self.ln_gamma, self.ln_beta)
        k_r, k_np = torch.split(k_ln, [self.rope_head_dim, self.idx_head_dim - self.rope_head_dim], dim=-1)
        k_r = self._rope_3d(k_r.unsqueeze(1), cos, sin).squeeze(1)
        k_cat = torch.cat([k_r, k_np], dim=-1)
        k_hd = (k_cat.float() @ self.hadamard_k.float()).to(x_dtype)
        k_idx_i8, k_idx_scale = self._per_token_quantize(k_hd)

        w_f32 = hidden_states.float() @ self.w_proj.float()
        weights = (w_f32 * (self.idx_n_heads ** -0.5) * (self.idx_head_dim ** -0.5)).to(torch.float16)

        return (q_nope, q_rope_out, k_nope_norm, q_idx_i8, q_idx_scale, weights)


def get_inputs():
    t = 4
    h = 7168
    rope_dim = 64
    x = torch.randn(t, h, dtype=torch.bfloat16) * 0.01 / math.sqrt(h)
    cos = torch.randn(t, rope_dim, dtype=torch.bfloat16) * 0.01
    sin = torch.randn(t, rope_dim, dtype=torch.bfloat16) * 0.01
    return [x, cos, sin]


def get_init_inputs():
    return [4, 128, 7168, 1536, 128, 64, 512, 64, 128, 64]
