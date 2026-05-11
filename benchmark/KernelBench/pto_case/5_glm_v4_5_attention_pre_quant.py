#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn
import math

FORMULA = (
    "q[m, q_size] = RoPE(RMSNorm(DeQuantMatMul(RMSNorm(x[m, h] + residual[m, h]), W_qkv[h, t])))), "
    "k[m, kv_size] = RoPE(RMSNorm(DeQuantMatMul(...)[q_size:q_size+kv_size]))), "
    "v[m, kv_size] = DeQuantMatMul(...)[q_size+kv_size:]), "
    "residual_out[m, h] = x[m, h] + residual[m, h]"
)
DYNAMIC_AXIS = ["M"]


class Model(nn.Module):
    def __init__(
        self,
        hidden_size: int = 5120,
        total_head_size: int = 1792,
        head_size: int = 128,
        q_size: int = 1536,
        kv_size: int = 128,
        rotary_dim: int = 64,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.total_head_size = total_head_size
        self.head_size = head_size
        self.q_size = q_size
        self.kv_size = kv_size
        self.rotary_dim = rotary_dim
        self.half_rotary_dim = rotary_dim // 2
        self.eps = 1e-05

        # Input LayerNorm
        self.input_layernorm_weight = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.bfloat16) / math.sqrt(hidden_size)
        )
        self.input_layernorm_bias = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.bfloat16) / math.sqrt(hidden_size)
        )

        # Quantization params
        self.atten_qkv_input_scale_reciprocal = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.bfloat16) / math.sqrt(hidden_size)
        )
        self.atten_qkv_input_offset = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.bfloat16) / math.sqrt(hidden_size)
        )

        # QKV weight (int8)
        self.atten_qkv_weight = nn.Parameter(
            torch.randint(-128, 127, (hidden_size, total_head_size), dtype=torch.int8),
            requires_grad=False,
        )
        self.atten_qkv_quant_bias = nn.Parameter(
            torch.randint(-128, 127, (total_head_size,), dtype=torch.int32),
            requires_grad=False,
        )
        self.atten_qkv_deq_scale = nn.Parameter(
            torch.randn(total_head_size, dtype=torch.float32) / math.sqrt(total_head_size)
        )

        # Q/K LayerNorm
        self.atten_q_norm_weight = nn.Parameter(
            torch.randn(head_size, dtype=torch.bfloat16) / math.sqrt(head_size)
        )
        self.atten_q_norm_bias = nn.Parameter(
            torch.randn(head_size, dtype=torch.bfloat16) / math.sqrt(head_size)
        )
        self.atten_k_norm_weight = nn.Parameter(
            torch.randn(head_size, dtype=torch.bfloat16) / math.sqrt(head_size)
        )
        self.atten_k_norm_bias = nn.Parameter(
            torch.randn(head_size, dtype=torch.bfloat16) / math.sqrt(head_size)
        )

    def _rms_norm(self, x: torch.Tensor, gamma: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.to(torch.float32)
        x_mean_coff = 1.0 / x.shape[-1]
        x_square = x_fp32 * x_fp32
        x_mean = x_square * x_mean_coff
        x_reduce_sum = torch.sum(x_mean, dim=-1, keepdim=True) + self.eps
        x_reduce_sqrt = torch.sqrt(x_reduce_sum)
        x_res_div = x_fp32 / x_reduce_sqrt
        x_mul_res = x_res_div * gamma.to(torch.float32)
        x_add_bias = x_mul_res + bias.to(torch.float32)
        return x_add_bias.to(torch.bfloat16)

    def _apply_rotary_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return torch.cat((o1, o2), dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bs = hidden_states.shape[0]

        # 1. Add residual + RMSNorm
        x_fp32 = hidden_states.to(torch.float32) + residual.to(torch.float32)
        x_g = self._rms_norm(
            x_fp32.to(torch.bfloat16),
            self.input_layernorm_weight,
            self.input_layernorm_bias,
        )
        residual_out = x_fp32.to(torch.bfloat16)

        # 2. Quantize (approximate pure PyTorch)
        x_scale = self.atten_qkv_input_scale_reciprocal.float().unsqueeze(0)
        x_offset = self.atten_qkv_input_offset.float().unsqueeze(0)
        x_quant = (x_g.float() * x_scale + x_offset).round().clamp(-128, 127).to(torch.int8)

        # 3. Quantized MatMul + Dequantize (approximate pure PyTorch)
        mm = x_quant.float() @ self.atten_qkv_weight.float()
        mm = mm.float() * self.atten_qkv_deq_scale.float().unsqueeze(0)
        mm = mm + self.atten_qkv_quant_bias.float().unsqueeze(0)
        mm_golden = mm.to(torch.bfloat16)

        # 4. Split QKV
        q_g, k_g, v_g = mm_golden.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # 5. RMSNorm Q/K
        q_by_head = q_g.view(bs, self.q_size // self.head_size, self.head_size)
        q_norm = self._rms_norm(q_by_head, self.atten_q_norm_weight, self.atten_q_norm_bias)

        k_by_head = k_g.view(bs, self.kv_size // self.head_size, self.head_size)
        k_norm = self._rms_norm(k_by_head, self.atten_k_norm_weight, self.atten_k_norm_bias)

        # 6. Apply RoPE
        q_rot = q_norm[..., : self.rotary_dim]
        q_pass = q_norm[..., self.rotary_dim :]
        k_rot = k_norm[..., : self.rotary_dim]
        k_pass = k_norm[..., self.rotary_dim :]

        q_r = self._apply_rotary_emb(q_rot.to(torch.float32), cos.to(torch.float32), sin.to(torch.float32))
        k_r = self._apply_rotary_emb(k_rot.to(torch.float32), cos.to(torch.float32), sin.to(torch.float32))

        q_cat = torch.cat((q_r.to(torch.bfloat16), q_pass), dim=-1)
        k_cat = torch.cat((k_r.to(torch.bfloat16), k_pass), dim=-1)

        q_out = q_cat.view(bs, self.q_size)
        k_out = k_cat.view(bs, self.kv_size)
        v_out = v_g

        return q_out, k_out, v_out, residual_out


def get_inputs():
    bs = 8
    hidden_size = 5120
    half_rotary_dim = 32
    return [
        torch.randn(bs, hidden_size, dtype=torch.bfloat16) / math.sqrt(hidden_size),
        torch.randn(bs, hidden_size, dtype=torch.bfloat16) / math.sqrt(hidden_size),
        torch.randn(bs, 1, half_rotary_dim, dtype=torch.bfloat16) / math.sqrt(half_rotary_dim),
        torch.randn(bs, 1, half_rotary_dim, dtype=torch.bfloat16) / math.sqrt(half_rotary_dim),
    ]


def get_init_inputs():
    return [5120, 1792, 128, 1536, 128, 64]
