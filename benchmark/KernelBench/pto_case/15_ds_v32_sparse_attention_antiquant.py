#!/usr/bin/env python3
# coding: utf-8
"""KernelBench case: DeepSeek V3.2 Sparse Attention Antiquant — sparse attention from packed anti-quantized nope cache."""

import math
import torch
import torch.nn as nn

FORMULA = (
    "K_nope_fp = DeQuant(K_nope_i8, K_scale_packed) from packed nope_cache; "
    "attn = softmax(Q_i @ [K_nope_fp|K_rope_from_cache]^T * scale) @ K_nope_fp"
)
DYNAMIC_AXIS = ["M"]


class Model(nn.Module):
    def __init__(
        self,
        nq: int = 128,
        n_kv: int = 1,
        kv_lora_rank: int = 512,
        qk_rope_dim: int = 64,
        softmax_scale: float = 0.041666,
        topk: int = 2048,
        block_size: int = 128,
    ):
        super().__init__()
        self.nq = nq
        self.n_kv = n_kv
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_dim = qk_rope_dim
        self.softmax_scale = softmax_scale
        self.topk = topk
        self.block_size = block_size
        self.packed_dim = kv_lora_rank + qk_rope_dim * 2 + 4 * 4

    def forward(
        self, q_nope: torch.Tensor, q_rope: torch.Tensor, nope_cache: torch.Tensor,
        topk_idx: torch.Tensor, block_table: torch.Tensor, act_seq: torch.Tensor,
    ) -> torch.Tensor:
        b_s1_nq = q_nope.shape[0]
        b = act_seq.shape[0]
        s1 = b_s1_nq // (self.nq * b)

        q = torch.cat([q_nope, q_rope], dim=-1).reshape(b, s1, self.nq, -1)
        cache_2d = nope_cache.reshape(-1, self.packed_dim)

        attn_out = torch.zeros(b, s1, self.nq, self.kv_lora_rank, dtype=torch.bfloat16, device=q_nope.device)
        s2_tile = 2048

        for b_idx in range(b):
            cur_k_seq = int(act_seq[b_idx].item())
            for s1_idx in range(s1):
                cur_seq = min(max(cur_k_seq - s1 + 1 + s1_idx, 0), self.topk)
                bn_per_batch = (cur_seq + s2_tile - 1) // s2_tile
                qi = q[b_idx, s1_idx, :, :]

                for s2_idx in range(bn_per_batch):
                    s2_end = min(s2_tile, cur_seq - s2_idx * s2_tile)
                    s2_start = s2_tile * s2_idx

                    kv_up = torch.zeros(s2_end, self.kv_lora_rank + self.qk_rope_dim, dtype=torch.bfloat16, device=q_nope.device)
                    for l_idx in range(s2_end):
                        topk_val = int(topk_idx[b_idx * s1 + s1_idx, s2_start + l_idx].item())
                        blk_b = topk_val // self.block_size
                        blk_id = int(block_table[b_idx, blk_b].item())
                        offset = blk_id * self.block_size + (topk_val % self.block_size)

                        kn_i8 = cache_2d[offset, :self.kv_lora_rank]
                        kr_vint8 = cache_2d[offset, self.kv_lora_rank:self.kv_lora_rank + self.qk_rope_dim * 2]
                        sc_vint8 = cache_2d[offset, self.kv_lora_rank + self.qk_rope_dim * 2:]

                        kn_f32 = kn_i8.reshape(-1, 128).float()
                        sc_f32 = sc_vint8.view(torch.float32).reshape(-1, 1)
                        kn_deq = (kn_f32 * sc_f32).reshape(self.kv_lora_rank).to(torch.bfloat16)
                        kr_fp = kr_vint8.view(torch.bfloat16)

                        kv_up[l_idx, :self.kv_lora_rank] = kn_deq
                        kv_up[l_idx, self.kv_lora_rank:] = kr_fp

                    vj = kv_up[:, :self.kv_lora_rank]
                    sij = qi.float() @ kv_up.float().t()
                    sij_s = sij * self.softmax_scale
                    m = sij_s.amax(dim=-1, keepdim=True)
                    p = (sij_s - m).exp()
                    l_sum = p.sum(dim=-1, keepdim=True)
                    attn_w = (p / l_sum).to(torch.bfloat16)
                    o_part = attn_w.float() @ vj.float()
                    attn_out[b_idx, s1_idx, :, :] = o_part.to(torch.bfloat16)

        return attn_out.reshape(b * s1 * self.nq, self.kv_lora_rank)


def get_inputs():
    b, s1, nq, n_kv = 1, 2, 128, 1
    kv_lora_rank, qk_rope_dim = 512, 64
    topk_val = 2048
    block_size = 128
    actual_seq = [1024]
    packed_dim = kv_lora_rank + qk_rope_dim * 2 + 4 * 4

    block_num = sum((s + block_size - 1) // block_size for s in actual_seq)
    max_kv = max(actual_seq)
    block_table = torch.zeros(b, (max_kv + block_size - 1) // block_size, dtype=torch.int32)
    for i in range(block_num):
        block_table[0, i] = i

    q = torch.randn(b, s1, nq, kv_lora_rank + qk_rope_dim, dtype=torch.bfloat16) * 0.01 / math.sqrt(kv_lora_rank + qk_rope_dim)
    q_nope = q[:, :, :, :kv_lora_rank].reshape(b * s1 * nq, kv_lora_rank)
    q_rope = q[:, :, :, kv_lora_rank:].reshape(b * s1 * nq, qk_rope_dim)

    kn_raw = torch.randn(block_num, block_size, 1, kv_lora_rank, dtype=torch.bfloat16) * 0.01 / math.sqrt(kv_lora_rank)
    kn_reshape = kn_raw.reshape(block_num * block_size, 4, 128).float()
    kn_scale = kn_reshape.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    kn_i8 = (kn_reshape / kn_scale).round().clamp(-128, 127).to(torch.int8)

    kr = torch.randn(block_num, block_size, 1, qk_rope_dim, dtype=torch.bfloat16) * 0.01 / math.sqrt(qk_rope_dim)

    nope_cache = torch.zeros(block_num * block_size, packed_dim, dtype=torch.int8)
    nope_cache[:, :kv_lora_rank] = kn_i8.reshape(block_num * block_size, kv_lora_rank)
    nope_cache[:, kv_lora_rank:kv_lora_rank + qk_rope_dim * 2] = kr.reshape(block_num * block_size, qk_rope_dim).view(torch.int8)
    nope_cache[:, kv_lora_rank + qk_rope_dim * 2:] = kn_scale.reshape(block_num * block_size, 4).contiguous().view(torch.int8)

    topk_indices = torch.zeros(b * s1, n_kv * topk_val, dtype=torch.int32)
    for b_i in range(b):
        for s_i in range(s1):
            slc = min(actual_seq[b_i], topk_val)
            topk_indices[b_i * s1 + s_i, :slc] = torch.arange(slc)

    act_seq_t = torch.tensor(actual_seq, dtype=torch.int32)
    return [q_nope, q_rope, nope_cache, topk_indices, block_table, act_seq_t]


def get_init_inputs():
    return [128, 1, 512, 64, 1.0 / math.sqrt(576), 2048, 128]
