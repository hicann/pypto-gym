#!/usr/bin/env python3
# coding: utf-8

import math
import torch
import torch.nn as nn


FORMULA = (
    "q_fp8[t, h, d] = fp8e4m3_quant(Hq @ RoPE(split((Q_norm @ W_qb) * q_norm_scale * w_qb_scale))); "
    "k_fp8 = scatter_update(cache, fp8e4m3_quant(Hk @ RoPE(split(LayerNorm(x @ W_k, gamma, beta))))); "
    "weights[t, h] = (x @ W_proj) * (n^-0.5) * (d^-0.5)"
)
DYNAMIC_AXIS = ["B", "S"]


def _gen_block_table(act_seq, block_size, s1):
    b = act_seq.shape[0]
    max_kv = int(act_seq.max().item())
    block_num_each = [math.ceil(int(s) / block_size) for s in act_seq.tolist()]
    block_num = sum(block_num_each)
    block_table_shape = [b, math.ceil(max_kv / block_size)]
    perm = torch.randperm(block_num, dtype=torch.int32)
    block_table = -torch.ones(block_table_shape, dtype=torch.int32)
    block_idx = 0
    for bidx, cur_block in enumerate(block_num_each):
        for j in range(cur_block):
            block_table[bidx, j] = perm[block_idx].item()
            block_idx += 1
    cache_index = -torch.ones((b, s1), dtype=torch.int64)
    for i in range(b):
        cur_act = int(act_seq[i].item())
        for j in range(s1):
            pos = cur_act - s1 + j
            block_idx_in_seq = pos // block_size
            global_block_id = int(block_table[i, block_idx_in_seq].item())
            if global_block_id >= 0:
                offset_in_block = pos % block_size
                global_index = global_block_id * block_size + offset_in_block
                cache_index[i, j] = global_index
    return block_num, block_table, cache_index


class Model(nn.Module):
    def __init__(
        self,
        hidden_size: int = 2560,
        q_lora_rank: int = 1024,
        n_heads: int = 24,
        head_dim: int = 128,
        rope_head_dim: int = 64,
    ):
        super().__init__()
        n, d = n_heads, head_dim
        h, r = hidden_size, q_lora_rank

        self.hidden_size = h
        self.q_lora_rank = r
        self.n_heads = n
        self.head_dim = d
        self.rope_head_dim = rope_head_dim

        self.w_qb = nn.Parameter(torch.randn(r, n * d, dtype=torch.float32) / math.sqrt(r))
        self.w_qb_scale = nn.Parameter(torch.randn(1, n * d, dtype=torch.float32) * 0.1 + 1.0)
        self.w_k = nn.Parameter(torch.randn(h, d, dtype=torch.float32) / math.sqrt(h))
        self.w_proj = nn.Parameter(torch.randn(h, n, dtype=torch.float32) / math.sqrt(h))
        self.gamma = nn.Parameter(torch.ones(d, dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros(d, dtype=torch.float32))
        self.hadamard_q = nn.Parameter(torch.randn(d, d, dtype=torch.float32) / math.sqrt(d))
        self.hadamard_k = nn.Parameter(torch.randn(d, d, dtype=torch.float32) / math.sqrt(d))

    @staticmethod
    def _rotate_half(x):
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def _apply_rope(x, cos, sin):
        return x * cos + Model._rotate_half(x) * sin

    @staticmethod
    def _layer_norm(x, gamma, beta, eps=1e-6):
        x_fp32 = x.float()
        mean = x_fp32.mean(dim=-1, keepdim=True)
        var = ((x_fp32 - mean) ** 2).mean(dim=-1, keepdim=True)
        x_norm = (x_fp32 - mean) / torch.sqrt(var + eps)
        return (x_norm * gamma.float() + beta.float())

    @staticmethod
    def _fp8e4m3_quantize(x_fp32):
        max_val = torch.amax(torch.abs(x_fp32), dim=-1, keepdim=True).clamp(min=1e-10)
        scale = 448.0 / max_val
        y_scaled = x_fp32 * scale
        y_rounded = torch.round(y_scaled)
        y_clamped = y_rounded.clamp(-448.0, 448.0)
        dequant_scale = 1.0 / scale
        return y_clamped, dequant_scale

    @staticmethod
    def _scatter_update(cache, values, cache_index):
        block_num, block_size, n_kv = cache.shape[:3]
        flat = cache.reshape(block_num * block_size * n_kv, -1)
        b, s = cache_index.shape[:2]
        values_2d = values.reshape(b * s, -1)
        for b_i in range(b):
            for s_i in range(s):
                idx = int(cache_index[b_i, s_i].item())
                if idx >= 0:
                    flat[idx] = values_2d[b_i * s + s_i]
        return flat.reshape_as(cache)

    def forward(
        self,
        x: torch.Tensor,
        q_norm: torch.Tensor,
        q_norm_scale: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        k_cache: torch.Tensor,
        k_scale_cache: torch.Tensor,
        cache_index: torch.Tensor,
    ):
        t = x.shape[0]
        n = self.n_heads
        d = self.head_dim
        rdim = self.rope_head_dim
        x_dtype = x.dtype

        # ── Q path ──
        q = torch.matmul(q_norm.float(), self.w_qb.float())
        q_f32 = q.to(torch.float32)
        q_f32 = q_f32 * q_norm_scale.float()
        q_f32 = q_f32 * self.w_qb_scale.float()
        q_bf16 = q_f32.view(-1, n, d).to(x_dtype)

        q_rope, q_nope = torch.split(q_bf16, [rdim, d - rdim], dim=-1)
        cos_r = cos.view(-1, 1, 1, rdim).float()
        sin_r = sin.view(-1, 1, 1, rdim).float()
        q_rope = self._apply_rope(q_rope.float().view(-1, n, 1, rdim), cos_r, sin_r)
        q_rope = q_rope.to(x_dtype).view(-1, n, rdim)
        q_nope = q_nope.float().to(x_dtype)
        q_cat = torch.cat((q_rope, q_nope), dim=-1)
        q_hadamard = (q_cat.float() @ self.hadamard_q).to(x_dtype)
        q_fp8, q_scale = self._fp8e4m3_quantize(q_hadamard.float())
        q_scale = q_scale.to(torch.float16)

        # ── K path ──
        k_proj = x.float() @ self.w_k
        k_ln = self._layer_norm(k_proj, self.gamma, self.beta).to(x_dtype)

        k_rope, k_nope = torch.split(k_ln, [rdim, d - rdim], dim=-1)
        k_rope = self._apply_rope(k_rope.float().view(-1, 1, 1, rdim), cos_r, sin_r)
        k_rope = k_rope.to(x_dtype).view(-1, rdim)
        k_nope = k_nope.float().to(x_dtype)
        k_cat = torch.cat((k_rope, k_nope), dim=-1)
        k_hadamard = (k_cat.float() @ self.hadamard_k).to(x_dtype)
        k_fp8, k_scale = self._fp8e4m3_quantize(k_hadamard.float())
        k_scale = k_scale.to(torch.float16)

        b = cache_index.shape[0]
        s_val = k_hadamard.shape[0] // b
        k_cache_out = self._scatter_update(k_cache.clone(), k_fp8.view(b, s_val, 1, d), cache_index)
        k_scale_out = self._scatter_update(k_scale_cache.clone(), k_scale.view(b, s_val, 1, 1), cache_index)

        # ── W path ──
        weights = (x.float() @ self.w_proj).to(x_dtype).to(torch.float32)
        weights = weights * (n ** -0.5) * (d ** -0.5)
        weights = weights.to(torch.float16)

        return q_fp8, q_scale, k_cache_out, k_scale_out, weights


def get_inputs():
    b = 1
    s = 4
    s2 = 256
    t = b * s
    h = 2560
    qr = 1024
    n = 24
    d = 128
    rdim = 64
    block_size = 128
    n_kv = 1
    block_num = b * s2 // block_size

    torch.manual_seed(42)

    x = torch.randn(t, h, dtype=torch.bfloat16) / math.sqrt(h)

    q_norm = torch.randint(-128, 128, (t, qr), dtype=torch.int8)

    q_norm_scale = torch.randn(t, 1, dtype=torch.float32) * 0.5 + 1.0

    random_angles = torch.rand(t, rdim, dtype=torch.float32) * 2 * math.pi
    cos = torch.cos(random_angles).to(torch.bfloat16) / math.sqrt(rdim)
    sin = torch.sin(random_angles).to(torch.bfloat16) / math.sqrt(rdim)

    k_cache = torch.randn(block_num, block_size, n_kv, d, dtype=torch.float32) / math.sqrt(d)

    k_scale_cache = torch.randn(block_num, block_size, n_kv, 1, dtype=torch.float32) * 0.5 + 1.0

    act_seq = torch.tensor([s2] * b)
    _, _, cache_index = _gen_block_table(act_seq, block_size, s)

    return [x, q_norm, q_norm_scale, cos, sin, k_cache, k_scale_cache, cache_index]


def get_init_inputs():
    return [2560, 1024, 24, 128, 64]
