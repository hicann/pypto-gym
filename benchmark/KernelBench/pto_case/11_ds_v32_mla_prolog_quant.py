#!/usr/bin/env python3
# coding: utf-8
"""KernelBench case: DeepSeek V3.2 MLA Prolog Quant — hidden→query(nope+rope)+kv cache via quantized projections."""

import math
import torch
import torch.nn as nn

FORMULA = (
    "q_norm = RMSNorm(x @ w_dq, gamma_cq); "
    "q_norm_q = PerTokenQuant(q_norm); "
    "q_proj = Dequant(q_norm_q @ w_uq_qr, q_norm_scale, w_uq_scale); "
    "q_nope = (q_proj[:,:knh]) · w_uk → KV-lora; "
    "q_rope = RoPE(q_proj[:,knh:], cos, sin); "
    "kv = x @ w_dkv_kr; "
    "k_nope = PerChannelQuant(RMSNorm(kv[:,:kvl], gamma_ckv)); "
    "k_rope = RoPE(kv[:,kvl:], cos, sin)"
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
    ):
        super().__init__()
        self.t = t
        self.n = n
        self.h = h
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.q_head_dim = qk_nope_head_dim + qk_rope_head_dim

        # Weights — quantize w_dq/w_uq_qr to int8 per-channel, keep w_uk/w_dkv_kr in BF16
        _w_dq_raw = torch.randn(h, q_lora_rank, dtype=torch.float32) * 0.01
        _w_dq, _w_dq_deq_scale = self._per_channel_quantize(_w_dq_raw)
        self.register_buffer('w_dq', _w_dq)
        self.w_dq_deq_scale = nn.Parameter(_w_dq_deq_scale.reshape(-1).to(torch.float32))

        _w_uq_qr_raw = torch.randn(q_lora_rank, n * self.q_head_dim, dtype=torch.float32) * 0.01
        _w_uq_qr, _w_uq_deq_scale = self._per_channel_quantize(_w_uq_qr_raw)
        self.register_buffer('w_uq_qr', _w_uq_qr)
        self.w_uq_deq_scale = nn.Parameter(_w_uq_deq_scale.reshape(-1).to(torch.float32))

        self.w_uk = nn.Parameter(
            torch.randn(n, qk_nope_head_dim, kv_lora_rank, dtype=torch.float32) * 0.01
        )
        self.w_dkv_kr = nn.Parameter(
            torch.randn(h, kv_lora_rank + qk_rope_head_dim, dtype=torch.float32) * 0.01
        )

        self.gamma_cq = nn.Parameter(torch.randn(q_lora_rank, dtype=torch.float32))
        self.gamma_ckv = nn.Parameter(torch.randn(kv_lora_rank, dtype=torch.float32))

    @staticmethod
    def _per_channel_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        max_val = x.abs().max(dim=0, keepdim=True)[0].clamp(min=1e-8)
        scale = 127.0 / max_val
        q = (x * scale).round().clamp(-128, 127).to(torch.int8)
        deq_scale = 1.0 / scale
        return q, deq_scale

    @staticmethod
    def _per_token_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        max_val = x.float().abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)
        scale = 127.0 / max_val
        q = (x.float() * scale).round().clamp(-128, 127).to(torch.int8)
        deq_scale = 1.0 / scale
        return q, deq_scale

    @staticmethod
    def _rms_norm(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x_f32 = x.float()
        rms = torch.sqrt((x_f32 * x_f32).mean(dim=-1, keepdim=True) + 1e-5)
        return ((x_f32 / rms) * gamma.float()).to(x_dtype)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def _rope_3d(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: (t, heads, rope_dim), cos/sin: (t, rope_dim)
        x_dtype = x.dtype
        x_f32 = x.float()
        cos_f32 = cos.float().unsqueeze(1)  # (t, 1, rope_dim)
        sin_f32 = sin.float().unsqueeze(1)
        x_embed = x_f32 * cos_f32 + Model._rotate_half(x_f32) * sin_f32
        return x_embed.to(x_dtype)

    @staticmethod
    def _rope_2d(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x_f32 = x.float()
        cos_f32 = cos.float()
        sin_f32 = sin.float()
        x_embed = x_f32 * cos_f32 + Model._rotate_half(x_f32) * sin_f32
        return x_embed.to(x_dtype)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Args:
            hidden_states: (t, h) BF16
        Returns:
            q_nope: (t, n, kv_lora_rank) BF16
            q_rope: (t, n, qk_rope_head_dim) BF16
            q_norm: (t, q_lora_rank) INT8
            q_norm_scale: (t, 1) FP32
        """
        x_dtype = hidden_states.dtype
        t, h = hidden_states.shape
        x_f32 = hidden_states.float()

        # === Query path ===
        # q_a: x @ w_dq → dequant → RMSNorm
        x_quant, x_scale = self._per_token_quantize(hidden_states)
        q_a_i32 = x_quant.float() @ self.w_dq.float()
        q_a = (q_a_i32.float() * x_scale.float() *
               self.w_dq_deq_scale.float().unsqueeze(0)).to(x_dtype)
        q_a_norm = self._rms_norm(q_a, self.gamma_cq)  # (t, q_lora_rank)

        # q_b: quantize norm → matmul with w_uq_qr → dequant
        q_norm_int8, q_norm_scale = self._per_token_quantize(q_a_norm)
        q_b_i32 = q_norm_int8.float() @ self.w_uq_qr.float()  # (t, n*q_head_dim)
        q_b = (q_b_i32.float() * q_norm_scale.float() *
               self.w_uq_deq_scale.float().unsqueeze(0)).to(x_dtype)

        # Split nope / rope
        q_reshape = q_b.reshape(t, self.n, self.q_head_dim)
        q_nope_raw = q_reshape[:, :, :self.qk_nope_head_dim]  # (t, n, qk_nope_head_dim)

        # q_nope → transpose → matmul w_uk → transpose back
        q_nope_t = q_nope_raw.permute(1, 0, 2)  # (n, t, qk_nope_head_dim)
        q_nope_proj = (q_nope_t.float() @ self.w_uk.float()).to(x_dtype)  # (n, t, kv_lora_rank)
        q_nope = q_nope_proj.permute(1, 0, 2)  # (t, n, kv_lora_rank)

        # q_rope: RoPE
        q_rope_raw = q_reshape[:, :, self.qk_nope_head_dim:]  # (t, n, qk_rope_head_dim)

        return (q_nope, q_rope_raw, q_norm_int8, q_norm_scale)


def get_inputs():
    t = 4
    h = 7168
    x = torch.randn(t, h, dtype=torch.bfloat16) * 0.01 / math.sqrt(h)
    return [x]


def get_init_inputs():
    return [4, 128, 7168, 1536, 128, 64, 512]
