#!/usr/bin/env python3
# coding: utf-8

import math
import torch
import torch.nn as nn


FORMULA = (
    "q_hif8[t, h, d] = hif8_quant(Hq @ RoPE(split(dequant(Q_norm) @ dequant(W_qb)))); "
    "k_hif8 = scatter_update(cache, hif8_quant(Hk @ RoPE(split(RMSNorm(x @ W_k))))); "
    "weights[t, h] = (x @ W_proj) / sqrt(h * d)"
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


def _gen_cache_tensor(k_cache_bsnd, block_table, block_num, block_size, k_cache):
    dtype = k_cache_bsnd.dtype
    b, s2, n_kv, d = k_cache_bsnd.shape
    k_cache = k_cache.view(block_num, block_size, n_kv, d)
    s2_pad = ((s2 + block_size - 1) // block_size) * block_size
    k_cache_raw = torch.zeros((b, s2_pad, n_kv, d), dtype=dtype)
    k_cache_raw[:, :s2, :, :] = k_cache_bsnd
    for b_idx in range(b):
        for block_idx in range(int(math.ceil(s2 / block_size))):
            bid = block_idx
            if bid < block_table.shape[1]:
                cache_block_idx = int(block_table[b_idx, bid].item())
                if cache_block_idx >= 0:
                    block_offset = bid * block_size
                    k_cache[cache_block_idx, :, :, :] = k_cache_raw[
                        b_idx, block_offset : block_offset + block_size, :, :
                    ]
    return k_cache


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
    def _rms_norm(x, gamma, eps=1e-6):
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps)
        return (x.float() / rms * gamma.float())

    @staticmethod
    def _sim_hif8_quantize(x_fp32):
        max_val = torch.amax(torch.abs(x_fp32), dim=-1, keepdim=True).clamp(min=1e-10)
        scale = 32768.0 / max_val
        y_int = torch.round(x_fp32 * scale).clamp(-32768, 32767)
        y_uint8 = (y_int.to(torch.int32) & 0xFF).to(torch.uint8)
        dequant_scale = 1.0 / scale
        return y_uint8, dequant_scale

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
        qr = self.q_lora_rank
        rdim = self.rope_head_dim
        x_dtype = x.dtype

        # ── Q path ──
        q_norm_fp = q_norm.float() * q_norm_scale.float()
        w_qb_fp = self.w_qb * self.w_qb_scale
        q_proj = (q_norm_fp @ w_qb_fp).to(x_dtype)
        q_proj = q_proj.view(-1, n, d)

        q_rope, q_nope = torch.split(q_proj, [rdim, d - rdim], dim=-1)
        cos_r = cos.view(-1, 1, 1, rdim).float()
        sin_r = sin.view(-1, 1, 1, rdim).float()
        q_rope = self._apply_rope(q_rope.float().view(-1, n, 1, rdim), cos_r, sin_r)
        q_rope = q_rope.to(x_dtype).view(-1, n, rdim)
        q_cat = torch.cat((q_rope, q_nope), dim=-1)
        q_hadamard = (q_cat.float() @ self.hadamard_q).to(x_dtype)
        q_hif8, q_scale = self._sim_hif8_quantize(q_hadamard.float())

        # ── K path ──
        k_proj = x.float() @ self.w_k
        k_rms_norm = self._rms_norm(k_proj, self.gamma).to(x_dtype)

        k_rope, k_nope = torch.split(k_rms_norm, [rdim, d - rdim], dim=-1)
        k_rope = self._apply_rope(k_rope.float().view(-1, 1, 1, rdim), cos_r, sin_r)
        k_rope = k_rope.to(x_dtype).view(-1, rdim)
        k_cat = torch.cat((k_rope, k_nope), dim=-1)
        k_hadamard = (k_cat.float() @ self.hadamard_k).to(x_dtype)
        k_hif8, k_scale = self._sim_hif8_quantize(k_hadamard.float())

        b = cache_index.shape[0]
        s_val = k_hadamard.shape[0] // b
        k_cache_out = self._scatter_update(k_cache.clone(), k_hif8.view(b, s_val, 1, d), cache_index)
        k_scale_out = self._scatter_update(k_scale_cache.clone(), k_scale.view(b, s_val, 1, 1), cache_index)

        # ── W path ──
        weights = (x.float() @ self.w_proj).to(x_dtype)
        weights = weights * (n ** -0.5) * (d ** -0.5)

        return q_hif8, q_scale, k_cache_out, k_scale_out, weights


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

    q_norm_fp32 = torch.randn(t, qr, dtype=torch.float32) * 5.0
    q_norm_scale = torch.randn(t, 1, dtype=torch.float32) * 0.5 + 1.0
    max_val_qn = torch.amax(torch.abs(q_norm_fp32), dim=-1, keepdim=True).clamp(min=1e-6)
    q_scale_qn = 32768.0 / max_val_qn
    q_norm_int = torch.round(q_norm_fp32 * q_scale_qn).clamp(-32768, 32767)
    q_norm = (q_norm_int.to(torch.int32) & 0xFF).to(torch.uint8)

    random_angles = torch.rand(t, rdim, dtype=torch.float32) * 2 * math.pi
    cos = torch.cos(random_angles).to(torch.bfloat16) / math.sqrt(rdim)
    sin = torch.sin(random_angles).to(torch.bfloat16) / math.sqrt(rdim)

    k_cache_fp32 = torch.randn(block_num, block_size, n_kv, d, dtype=torch.float32) * 5.0
    max_val_kc = torch.amax(torch.abs(k_cache_fp32), dim=-1, keepdim=True).clamp(min=1e-6)
    kc_scale = 32768.0 / max_val_kc
    k_cache_int = torch.round(k_cache_fp32 * kc_scale).clamp(-32768, 32767)
    k_cache = (k_cache_int.to(torch.int32) & 0xFF).to(torch.uint8)

    k_scale_cache = torch.randn(block_num, block_size, n_kv, 1, dtype=torch.float32) * 0.5 + 1.0

    act_seq = torch.tensor([s2] * b)
    _, _, cache_index = _gen_block_table(act_seq, block_size, s)

    return [x, q_norm, q_norm_scale, cos, sin, k_cache, k_scale_cache, cache_index]


def get_init_inputs():
    return [2560, 1024, 24, 128, 64]
