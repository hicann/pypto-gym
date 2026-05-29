#!/usr/bin/env python3
# coding: utf-8
"""KernelBench case: DeepSeek V3.2 MLA Prolog (decode, non-quantized) — hidden→query(nope+rope)+kv projections."""

import math
import torch
import torch.nn as nn

FORMULA = (
    "q_norm = RMSNorm(x @ w_dq, gamma_cq); "
    "q_proj = q_norm @ w_uqqr; "
    "q_nope = (q_proj[:,:knh]) @ w_uk → KV-lora; "
    "q_rope = RoPE(q_proj[:,knh:], cos, sin); "
    "kv = x @ w_dkv_kr; "
    "k_nope = RMSNorm(kv[:,:kvl], gamma_ckv); "
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
        self.n = n
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.q_head_dim = qk_nope_head_dim + qk_rope_head_dim

        self.w_dq = nn.Parameter(torch.randn(h, q_lora_rank, dtype=torch.float32) * 0.01)
        self.w_uqqr = nn.Parameter(
            torch.randn(q_lora_rank, n * self.q_head_dim, dtype=torch.float32) * 0.01
        )
        self.w_uk = nn.Parameter(
            torch.randn(n, qk_nope_head_dim, kv_lora_rank, dtype=torch.float32) * 0.01
        )
        self.w_dkvkr = nn.Parameter(
            torch.randn(h, kv_lora_rank + qk_rope_head_dim, dtype=torch.float32) * 0.01
        )

        self.gamma_cq = nn.Parameter(torch.randn(q_lora_rank, dtype=torch.float32))
        self.gamma_ckv = nn.Parameter(torch.randn(kv_lora_rank, dtype=torch.float32))

    @staticmethod
    def _rms_norm(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        mean_coff = 1.0 / x.shape[-1]
        x_f32 = x.float()
        square = x_f32 * x_f32
        mean_res = square * mean_coff
        reduce_sum = torch.sum(mean_res, dim=-1, keepdim=True)
        reduce_sqrt = torch.sqrt(reduce_sum)
        res_div = x_f32 / reduce_sqrt
        res = res_div * gamma.float()
        if x_dtype != torch.float32:
            res = res.to(x_dtype)
        return res

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def _rope_3d(
        x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Interleaved RoPE for (t, num_heads, head_dim)."""
        x_dtype = x.dtype
        x_f32 = x.float()
        t, hx, d = x_f32.shape
        x_interleaved = x_f32.reshape(t, hx, d // 2, 2).permute(0, 1, 3, 2).reshape(t, hx, d)
        cos_f32 = cos.float().unsqueeze(1)
        sin_f32 = sin.float().unsqueeze(1)
        x_embed = x_interleaved * cos_f32 + Model._rotate_half(x_interleaved) * sin_f32
        return x_embed.to(x_dtype)

    @staticmethod
    def _rope_2d(
        x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Interleaved RoPE for (t, head_dim)."""
        x_dtype = x.dtype
        x_f32 = x.float()
        t, d = x_f32.shape
        x_interleaved = x_f32.reshape(t, d // 2, 2).permute(0, 2, 1).reshape(t, d)
        cos_f32 = cos.float()
        sin_f32 = sin.float()
        x_embed = x_interleaved * cos_f32 + Model._rotate_half(x_interleaved) * sin_f32
        return x_embed.to(x_dtype)

    def forward(
        self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """
        Args:
            hidden_states: (t, h) BF16
            cos: (t, qk_rope_head_dim) BF16
            sin: (t, qk_rope_head_dim) BF16
        Returns:
            q_nope:  (t, n, kv_lora_rank) BF16
            q_rope:  (t, n, qk_rope_head_dim) BF16
            k_nope:  (t, kv_lora_rank) BF16
            k_rope:  (t, qk_rope_head_dim) BF16
            q_norm:  (t, q_lora_rank) BF16
        """
        x_dtype = hidden_states.dtype
        t = hidden_states.shape[0]

        # === Q path: x → w_dq → RMSNorm → w_uqqr → split nope/rope ===
        q_a = (hidden_states.float() @ self.w_dq.float()).to(x_dtype)
        q_a_norm = self._rms_norm(q_a, self.gamma_cq)

        q_b = (q_a_norm.float() @ self.w_uqqr.float()).to(x_dtype)
        q_reshape = q_b.reshape(t, self.n, self.q_head_dim)

        q_nope_raw = q_reshape[:, :, : self.qk_nope_head_dim]
        q_nope_t = q_nope_raw.permute(1, 0, 2)
        q_nope_proj = (q_nope_t.float() @ self.w_uk.float()).to(x_dtype)
        q_nope = q_nope_proj.permute(1, 0, 2)

        q_rope_raw = q_reshape[:, :, self.qk_nope_head_dim:]
        q_rope = self._rope_3d(q_rope_raw, cos, sin)

        # === KV path: x → w_dkvkr → split ckv/k_rope ===
        kv = (hidden_states.float() @ self.w_dkvkr.float()).to(x_dtype)
        compressed_kv = kv[:, : self.kv_lora_rank]
        k_nope = self._rms_norm(compressed_kv, self.gamma_ckv)

        k_rope_raw = kv[:, self.kv_lora_rank:]
        k_rope = self._rope_2d(k_rope_raw, cos, sin)

        return q_nope, q_rope, k_nope, k_rope, q_a_norm


def get_inputs():
    t = 4
    h = 7168
    qk_rope_head_dim = 64
    x = torch.randn(t, h, dtype=torch.bfloat16) * 0.01 / math.sqrt(h)
    cos = torch.randn(t, qk_rope_head_dim, dtype=torch.bfloat16) * 0.01
    sin = torch.randn(t, qk_rope_head_dim, dtype=torch.bfloat16) * 0.01
    return [x, cos, sin]


def get_init_inputs():
    return [4, 128, 7168, 1536, 128, 64, 512]
