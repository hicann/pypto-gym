#!/usr/bin/env python3
# coding: utf-8

"""KernelBench case: DeepSeek V3.2 MLA Prolog Quant — hidden→query(nope+rope)+kv via quantized projections.

Config: is_quant_a=False, is_quant_b=True
  - First-stage matmuls (x@w_dq, x@w_dkv_kr): FP32 (no quantization)
  - Second-stage Q matmul (q_norm@w_uq_qr): INT8 quantized
  - compressed_kv_norm: per-group(4) INT8 quantized
  - RoPE: interleaved (reshape(d//2,2).permute)
"""

import math
import torch
import torch.nn as nn

FORMULA = (
    "q_a = RMSNorm(x @ w_dq, gamma_cq); "
    "q_a_q, q_a_scale = PerTokenQuant(q_a); "
    "q_b = Dequant(q_a_q @ w_uq_qr, q_a_scale, w_qb_scale); "
    "q_nope = (q_b[:,:knh]) @ w_uk; "
    "q_rope = InterleavedRoPE(q_b[:,knh:], cos, sin); "
    "kv = x @ w_dkv_kr; "
    "k_nope_q, k_scale = PerTokenQuant(RMSNorm(kv[:,:kvl], gamma_ckv).split(g=4)); "
    "k_rope = InterleavedRoPE(kv[:,kvl:], cos, sin); "
    "kv_cache_out = scatter(kv_cache, k_nope_q); "
    "kr_cache_out = scatter(kr_cache, k_rope); "
    "k_scale_cache_out = scatter(k_scale_cache, k_scale)"
)
DYNAMIC_AXIS = ["M"]


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

        # is_quant_a=False: w_dq, w_dkv_kr are BF16 (not quantized)
        self.w_dq = nn.Parameter(
            torch.randn(h, q_lora_rank, dtype=torch.float32) * 0.01
        )
        self.w_dkv_kr = nn.Parameter(
            torch.randn(h, kv_lora_rank + qk_rope_head_dim, dtype=torch.float32) * 0.01
        )

        # is_quant_b=True: w_uq_qr is INT8 with dequant scale
        nph = n * self.q_head_dim
        _w_uq_qr_raw = torch.randint(-128, 128, (q_lora_rank, nph), dtype=torch.int32).to(torch.int8)
        self.register_buffer('w_uq_qr', _w_uq_qr_raw)
        self.w_qb_scale = nn.Parameter(
            torch.randn(1, nph, dtype=torch.float32) * 0.1 + 1.0
        )

        # w_uk: BF16 (not quantized)
        self.w_uk = nn.Parameter(
            torch.randn(n, qk_nope_head_dim, kv_lora_rank, dtype=torch.float32) * 0.01
        )

        # RMSNorm gammas
        self.gamma_cq = nn.Parameter(torch.randn(q_lora_rank, dtype=torch.float32))
        self.gamma_ckv = nn.Parameter(torch.randn(kv_lora_rank, dtype=torch.float32))

    @staticmethod
    def _rms_norm(x, gamma):
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
    def _per_token_quantize(x):
        x_f32 = x.float()
        abs_res = torch.abs(x_f32)
        max_value = torch.max(abs_res, dim=-1, keepdim=True)[0]
        scale_quant = 127.0 / max_value
        out_fp32 = x_f32 * scale_quant
        out_int32 = torch.round(out_fp32).to(torch.int32)
        out_fp16 = out_int32.to(torch.float16)
        out_int8 = torch.trunc(out_fp16).to(torch.int8)
        scale_dequant = 1.0 / scale_quant
        return out_int8, scale_dequant

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def _interleaved_rope_3d(x, cos, sin):
        """Interleaved RoPE for 3D tensor [t, n, d]."""
        x_dtype = x.dtype
        x_f32 = x.float()
        t, h, d = x_f32.shape
        x_interleaved = x_f32.reshape(t, h, d // 2, 2).permute(0, 1, 3, 2).reshape(t, h, d)
        cos_f32 = cos.float().unsqueeze(1)
        sin_f32 = sin.float().unsqueeze(1)
        x_embed = x_interleaved * cos_f32 + Model._rotate_half(x_interleaved) * sin_f32
        return x_embed.to(x_dtype)

    @staticmethod
    def _interleaved_rope_2d(x, cos, sin):
        """Interleaved RoPE for 2D tensor [t, d]."""
        x_dtype = x.dtype
        x_f32 = x.float()
        t, d = x_f32.shape
        x_interleaved = x_f32.reshape(t, d // 2, 2).permute(0, 2, 1).reshape(t, d)
        cos_f32 = cos.float()
        sin_f32 = sin.float()
        x_embed = x_interleaved * cos_f32 + Model._rotate_half(x_interleaved) * sin_f32
        return x_embed.to(x_dtype)

    @staticmethod
    def _scatter_update_4d(cache, key_states, indices):
        """Scatter update for 4D cache [block_num, block_size, n2, d]."""
        block_number, block_size, n2, d = cache.shape
        res = cache.reshape(block_number * block_size * n2, d)
        b, s1 = indices.shape
        for b_i in range(b):
            for s1_i in range(s1):
                index_value = indices[b_i][s1_i]
                res[index_value][:] = key_states[b_i * s1 + s1_i][:]
        return res.reshape(block_number, block_size, n2, d)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: torch.Tensor,
        kr_cache: torch.Tensor,
        kv_quant_scale_cache: torch.Tensor,
        cache_index: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        x_dtype = hidden_states.dtype
        t, h = hidden_states.shape
        b = cache_index.shape[0]
        s = t // b

        # === Q path ===
        # is_quant_a=False: simple FP32 matmul
        q_a = (hidden_states.float() @ self.w_dq.float()).to(x_dtype)
        q_a_norm = self._rms_norm(q_a, self.gamma_cq)

        # is_quant_b=True: quantize q_a_norm → INT8 matmul → dequant
        q_norm_int8, q_norm_scale = self._per_token_quantize(q_a_norm)
        q_b_i32 = q_norm_int8.float() @ self.w_uq_qr.float()
        q_b = (q_b_i32.float() * q_norm_scale.float() *
               self.w_qb_scale.float()).to(x_dtype)

        # Split nope / rope
        q_reshape = q_b.reshape(t, self.n, self.q_head_dim)
        q_nope_raw = q_reshape[:, :, :self.qk_nope_head_dim]
        q_rope_raw = q_reshape[:, :, self.qk_nope_head_dim:]

        # q_nope → transpose → matmul w_uk → transpose back
        q_nope_t = q_nope_raw.permute(1, 0, 2)
        q_nope = (q_nope_t.float() @ self.w_uk.float()).to(x_dtype)
        q_nope = q_nope.permute(1, 0, 2)

        # q_rope: interleaved RoPE
        q_rope = self._interleaved_rope_3d(q_rope_raw, cos, sin)

        # === KV path ===
        # is_quant_a=False: simple FP32 matmul
        kv = (hidden_states.float() @ self.w_dkv_kr.float()).to(x_dtype)

        compressed_kv = kv[:, :self.kv_lora_rank]
        k_rope_raw = kv[:, self.kv_lora_rank:]

        # RMSNorm → split into 4 groups → PerTokenQuant (is_quant_b=True)
        kv_norm = self._rms_norm(compressed_kv, self.gamma_ckv)
        kv_norm_split = kv_norm.reshape(t, 4, self.kv_lora_rank // 4)
        k_nope_quant, k_nope_deq_scale = self._per_token_quantize(kv_norm_split)
        k_nope_quant_flat = k_nope_quant.reshape(t, self.kv_lora_rank)
        k_scale_4group = k_nope_deq_scale.reshape(t, 4)

        # k_rope: interleaved RoPE
        k_rope = self._interleaved_rope_2d(k_rope_raw, cos, sin)

        # === Scatter update ===
        kv_cache_out = self._scatter_update_4d(
            kv_cache.clone(), k_nope_quant_flat, cache_index)
        kr_cache_out = self._scatter_update_4d(
            kr_cache.clone(), k_rope, cache_index)
        kv_quant_scale_cache_out = self._scatter_update_4d(
            kv_quant_scale_cache.clone(), k_scale_4group, cache_index)

        return (q_nope, q_rope, q_norm_int8, q_norm_scale,
                kv_cache_out, kr_cache_out, kv_quant_scale_cache_out)


def get_inputs():
    t = 4
    h = 7168
    qk_rope_head_dim = 64
    kv_lora_rank = 512
    block_size = 128
    n_kv = 1
    b = 1
    s = 4
    s2 = 256
    block_num = b * s2 // block_size

    torch.manual_seed(42)

    x = torch.randn(t, h, dtype=torch.bfloat16) * 0.01 / math.sqrt(h)
    cos = torch.randn(t, qk_rope_head_dim, dtype=torch.bfloat16) * 0.01
    sin = torch.randn(t, qk_rope_head_dim, dtype=torch.bfloat16) * 0.01

    kv_cache = torch.randn(block_num, block_size, n_kv, kv_lora_rank, dtype=torch.bfloat16) * 0.01
    kr_cache = torch.randn(block_num, block_size, n_kv, qk_rope_head_dim, dtype=torch.bfloat16) * 0.01
    kv_quant_scale_cache = torch.randn(block_num, block_size, n_kv, 4, dtype=torch.float32) * 0.5 + 1.0

    act_seq = torch.tensor([s2] * b)
    _, _, cache_index = _gen_block_table(act_seq, block_size, s)

    return [x, cos, sin, kv_cache, kr_cache, kv_quant_scale_cache, cache_index]


def get_init_inputs():
    return [4, 128, 7168, 1536, 128, 64, 512]
