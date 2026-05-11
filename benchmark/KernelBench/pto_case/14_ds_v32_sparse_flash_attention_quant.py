#!/usr/bin/env python3
# coding: utf-8
"""KernelBench case: DeepSeek V3.2 Sparse Flash Attention Quant — sparse attention over top-K keys with INT8 dequant."""

import math
import torch
import torch.nn as nn

FORMULA = (
    "Kj = [DeQuant(K_nope_i8, K_scale) | K_rope]; "
    "attn = softmax(Q_i @ Kj^T * scale) @ K_nope_fp"
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
        self.dq = kv_lora_rank + qk_rope_dim

    def forward(
        self, q_nope: torch.Tensor, q_rope: torch.Tensor, k_nope: torch.Tensor,
        k_rope: torch.Tensor, k_scale: torch.Tensor, topk_idx: torch.Tensor,
        block_table: torch.Tensor, act_seq: torch.Tensor,
    ) -> torch.Tensor:
        b_s1_nq = q_nope.shape[0]
        b = act_seq.shape[0]
        s1 = b_s1_nq // (self.nq * b)

        q = torch.cat([q_nope, q_rope], dim=-1).reshape(b, s1, self.nq, self.dq)
        k_nope_2d = k_nope.reshape(-1, self.kv_lora_rank)
        k_rope_2d = k_rope.reshape(-1, self.qk_rope_dim)
        k_scale_2d = k_scale.reshape(-1, 4)

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
                    slc_kn = torch.zeros(s2_end, self.kv_lora_rank, dtype=k_nope.dtype, device=q_nope.device)
                    slc_kr = torch.zeros(s2_end, self.qk_rope_dim, dtype=torch.bfloat16, device=q_nope.device)
                    slc_sc = torch.zeros(s2_end, 4, dtype=torch.float32, device=q_nope.device)

                    for l_idx in range(s2_end):
                        topk_val = int(topk_idx[b_idx * s1 + s1_idx, s2_start + l_idx].item())
                        blk_b = topk_val // self.block_size
                        blk_id = int(block_table[b_idx, blk_b].item())
                        offset = blk_id * self.block_size + (topk_val % self.block_size)
                        slc_kn[l_idx, :] = k_nope_2d[offset, :]
                        slc_kr[l_idx, :] = k_rope_2d[offset, :]
                        slc_sc[l_idx, :] = k_scale_2d[offset, :]

                    kn_int8 = slc_kn.reshape(-1, 128).float()
                    sc = slc_sc.reshape(-1, 1)
                    kn_fp = (kn_int8 * sc).reshape(-1, self.kv_lora_rank).to(torch.bfloat16)

                    kj = torch.cat([kn_fp, slc_kr], dim=-1)
                    sij = qi.float() @ kj.float().t()
                    sij_s = sij * self.softmax_scale
                    m = sij_s.amax(dim=-1, keepdim=True)
                    p = (sij_s - m).exp()
                    l = p.sum(dim=-1, keepdim=True)
                    attn_w = (p / l).to(torch.bfloat16)
                    o_part = attn_w.float() @ kn_fp.float()
                    attn_out[b_idx, s1_idx, :, :] = o_part.to(torch.bfloat16)

        attn_out_2d = attn_out.reshape(b * s1 * self.nq, self.kv_lora_rank)
        return attn_out_2d


def get_inputs():
    b, s1, nq, n_kv = 1, 2, 128, 1
    kv_lora_rank, qk_rope_dim = 512, 64
    topk_val = 2048
    block_size = 128
    actual_seq = [512]

    block_num = sum((s + block_size - 1) // block_size for s in actual_seq)
    max_kv = max(actual_seq)
    block_table = torch.zeros(b, (max_kv + block_size - 1) // block_size, dtype=torch.int32)
    for i in range(block_num):
        block_table[0, i] = i

    q = torch.randn(b, s1, nq, kv_lora_rank + qk_rope_dim, dtype=torch.bfloat16) * 0.01 / math.sqrt(576)
    q_nope = q[:, :, :, :kv_lora_rank].reshape(b * s1 * nq, kv_lora_rank)
    q_rope = q[:, :, :, kv_lora_rank:].reshape(b * s1 * nq, qk_rope_dim)

    kn_raw = torch.randn(block_num, block_size, kv_lora_rank, dtype=torch.bfloat16) * 0.01 / math.sqrt(kv_lora_rank)
    kn_reshape = kn_raw.reshape(block_num * block_size, 4, 128).float()
    kn_scale = kn_reshape.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    kn_i8 = (kn_reshape / kn_scale).round().clamp(-128, 127).to(torch.int8)
    k_nope = kn_i8.reshape(block_num * block_size, kv_lora_rank)
    k_rope = torch.randn(block_num, block_size, qk_rope_dim, dtype=torch.bfloat16) * 0.01 / math.sqrt(qk_rope_dim)
    k_rope = k_rope.reshape(block_num * block_size, qk_rope_dim)
    k_scale = kn_scale.reshape(block_num * block_size, 4)

    topk_indices = torch.zeros(b * s1, n_kv * topk_val, dtype=torch.int32)
    for b_i in range(b):
        for s_i in range(s1):
            slc = min(actual_seq[b_i], topk_val)
            topk_indices[b_i * s1 + s_i, :slc] = torch.arange(slc)

    act_seq_t = torch.tensor(actual_seq, dtype=torch.int32)
    return [q_nope, q_rope, k_nope, k_rope, k_scale, topk_indices, block_table, act_seq_t]


def get_init_inputs():
    return [128, 1, 512, 64, 1.0 / math.sqrt(576), 2048, 128]
