#!/usr/bin/env python3
# coding: utf-8
"""KernelBench case: GLM V4.5 Attention Fusion — attention_pre_quant + paged_attention end-to-end."""

import math
import torch
import torch.nn as nn

FORMULA = "attn_out = PagedAttention( RMSNorm_QK( RoPE( RMSNorm( QuantMatMul( RMSNorm( x + residual ) ) ) ) ) )"
DYNAMIC_AXIS = ["M", "S"]


class Model(nn.Module):
    def __init__(self, hidden_size: int = 5120, total_head_size: int = 1792,
                 head_size: int = 128, q_size: int = 1536, kv_size: int = 128,
                 half_rotary_dim: int = 32, num_heads: int = 12,
                 kv_num_heads: int = 1, block_size: int = 128,
                 num_blocks: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.total_head_size = total_head_size
        self.head_size = head_size
        self.q_size = q_size
        self.kv_size = kv_size
        self.half_rotary_dim = half_rotary_dim
        self.rotary_dim = half_rotary_dim * 2
        self.num_heads = num_heads
        self.kv_num_heads = kv_num_heads
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.eps = 1e-5
        self.softmax_scale = head_size ** -0.5

        self.input_layernorm_weight = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.bfloat16) / math.sqrt(hidden_size))
        self.input_layernorm_bias = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.bfloat16) / math.sqrt(hidden_size))
        self.input_scale = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.bfloat16) / math.sqrt(hidden_size))
        self.input_offset = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.bfloat16) / math.sqrt(hidden_size))
        self.register_buffer('qkv_weight',
            torch.randint(-128, 128, (hidden_size, total_head_size), dtype=torch.int8))
        self.register_buffer('qkv_quant_bias',
            torch.randint(-128, 128, (total_head_size,), dtype=torch.int32))
        self.qkv_deq_scale = nn.Parameter(
            torch.randn(total_head_size, dtype=torch.float32) / math.sqrt(total_head_size))
        self.q_norm_weight = nn.Parameter(
            torch.randn(head_size, dtype=torch.bfloat16) / math.sqrt(head_size))
        self.q_norm_bias = nn.Parameter(
            torch.randn(head_size, dtype=torch.bfloat16) / math.sqrt(head_size))
        self.k_norm_weight = nn.Parameter(
            torch.randn(head_size, dtype=torch.bfloat16) / math.sqrt(head_size))
        self.k_norm_bias = nn.Parameter(
            torch.randn(head_size, dtype=torch.bfloat16) / math.sqrt(head_size))

    def _rms_norm(self, x: torch.Tensor, gamma: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        mean_coff = 1.0 / x.shape[-1]
        rms = torch.sqrt((x_fp32 * x_fp32 * mean_coff).sum(dim=-1, keepdim=True) + self.eps)
        return ((x_fp32 / rms) * gamma.float() + bias.float()).to(x.dtype)

    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        rot = x[..., :self.rotary_dim]
        x1, x2 = rot.chunk(2, dim=-1)
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return torch.cat([torch.cat([o1, o2], dim=-1), x[..., self.rotary_dim:]], dim=-1)

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor,
                key_cache: torch.Tensor, value_cache: torch.Tensor,
                block_tables: torch.Tensor, actual_seqs: torch.Tensor,
                num_decode_tokens: int = 1
                ) -> tuple[torch.Tensor, torch.Tensor]:
        bs = hidden_states.shape[0]
        input_dtype = hidden_states.dtype
        d = self.head_size
        nq = self.num_heads
        nkv = self.kv_num_heads
        group = nq // nkv

        residual_out = hidden_states + residual
        x_norm = self._rms_norm(residual_out, self.input_layernorm_weight, self.input_layernorm_bias)

        isr = self.input_scale.float()
        iof = self.input_offset.float()
        x_quant = (x_norm.float() * isr + iof).round().clamp(-128, 127).to(torch.int8)

        qkv = (x_quant.float() @ self.qkv_weight.float()).float()
        qkv = qkv * self.qkv_deq_scale.float().unsqueeze(0) + self.qkv_quant_bias.float().unsqueeze(0)
        qkv = qkv.to(input_dtype)

        q_raw = qkv[:, :self.q_size]
        k = qkv[:, self.q_size:self.q_size + self.kv_size]
        v = qkv[:, self.q_size + self.kv_size:self.q_size + self.kv_size * 2]

        q_by_head = q_raw.view(bs, nq, d)
        k_by_head = k.view(bs, nkv, d)

        q_norm = self._rms_norm(q_by_head, self.q_norm_weight, self.q_norm_bias)
        k_norm = self._rms_norm(k_by_head, self.k_norm_weight, self.k_norm_bias)

        cos_s = cos.view(bs, 1, self.half_rotary_dim)
        sin_s = sin.view(bs, 1, self.half_rotary_dim)

        q_rope = self._apply_rope(q_norm, cos_s, sin_s)
        k_rope = self._apply_rope(k_norm, cos_s, sin_s)

        k_nope_tbl = key_cache.view(-1, nkv * d)
        v_nope_tbl = value_cache.view(-1, nkv * d)

        attn_out = torch.zeros(bs, nq, d, dtype=input_dtype)

        for b_idx in range(bs):
            cur_seq = actual_seqs[b_idx].item()
            for n2_idx in range(nkv):
                q_group = q_rope[b_idx, n2_idx * group:(n2_idx + 1) * group]
                k_list, v_list = [], []
                for blk_i in range(block_tables.shape[1]):
                    blk = block_tables[b_idx, blk_i].item()
                    if blk < 0:
                        break
                    k_list.append(k_nope_tbl[blk * self.block_size:(blk + 1) * self.block_size,
                                            n2_idx * d:(n2_idx + 1) * d])
                    v_list.append(v_nope_tbl[blk * self.block_size:(blk + 1) * self.block_size,
                                            n2_idx * d:(n2_idx + 1) * d])
                if not k_list:
                    continue
                k_seq = torch.cat(k_list, dim=0)[:cur_seq]
                v_seq = torch.cat(v_list, dim=0)[:cur_seq]
                scores = torch.matmul(q_group.float(), k_seq.float().t()) * self.softmax_scale
                attn_w = torch.softmax(scores, dim=-1).to(input_dtype)
                attn_out[b_idx, n2_idx * group:(n2_idx + 1) * group] = (
                    torch.matmul(attn_w.float(), v_seq.float()).to(input_dtype))

        return attn_out.reshape(bs, -1), residual_out.to(input_dtype)


def get_inputs():
    bs = 4
    hidden_size = 5120
    half_rotary_dim = 32
    d = 128
    block_size = 128
    num_blocks = 64
    s2 = 1024

    x = torch.randn(bs, hidden_size, dtype=torch.bfloat16) / math.sqrt(hidden_size)
    residual = torch.randn(bs, hidden_size, dtype=torch.bfloat16) / math.sqrt(hidden_size)
    cos = torch.randn(bs, 1, half_rotary_dim, dtype=torch.bfloat16) / math.sqrt(half_rotary_dim)
    sin = torch.randn(bs, 1, half_rotary_dim, dtype=torch.bfloat16) / math.sqrt(half_rotary_dim)
    k = torch.randn(num_blocks, block_size, 1, d, dtype=torch.bfloat16) / math.sqrt(d)
    v = torch.randn(num_blocks, block_size, 1, d, dtype=torch.bfloat16) / math.sqrt(d)

    block_tables = torch.zeros(bs, (s2 + block_size - 1) // block_size, dtype=torch.int32)
    blk_idx = 0
    for bi in range(bs):
        for j in range((s2 + block_size - 1) // block_size):
            if blk_idx < num_blocks:
                block_tables[bi, j] = blk_idx
                blk_idx += 1
    actual_seqs = torch.full((bs,), s2, dtype=torch.int32)

    return [x, residual, cos, sin, k, v, block_tables, actual_seqs, 1]


def get_init_inputs():
    return [5120, 1792, 128, 1536, 128, 32, 12, 1, 128, 256]
