#!/usr/bin/env python3
# coding: utf-8
"""KernelBench case: DeepSeek V3.2 Lightning Indexer Quant — compute top-K index positions via quantized dot-product scoring."""

import math
import torch
import torch.nn as nn

FORMULA = (
    "scores = (q_scale * weight) @ ReLU(q_int8 @ k_int8^T / 2048); "
    "scores = scores * k_scale; topk_res = TopK(scores, k=selected_count)"
)
DYNAMIC_AXIS = ["M"]


class Model(nn.Module):
    def __init__(
        self,
        b: int = 2,
        s1: int = 2,
        n1: int = 64,
        d: int = 128,
        s2: int = 1024,
        block_size: int = 128,
        selected_count: int = 2048,
    ):
        super().__init__()
        self.b = b
        self.s1 = s1
        self.n1 = n1
        self.d = d
        self.s2 = s2
        self.block_size = block_size
        self.selected_count = selected_count
        self.avoid_overflow_scale = 1.0 / 2048

    def forward(
        self, query: torch.Tensor, q_scale: torch.Tensor, key_cache: torch.Tensor,
        k_scale: torch.Tensor, weight: torch.Tensor, act_seq: torch.Tensor,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        t = query.shape[0]
        q_2d = query.reshape(-1, self.d)
        q_scale_3d = q_scale.reshape(t, 1, self.n1)
        k_2d = key_cache.reshape(-1, self.d)
        k_scale_2d = k_scale.reshape(-1, self.block_size)
        weight_3d = weight.reshape(t, 1, self.n1)

        max_block_num = block_table.shape[1]
        mm_out = torch.zeros(t, max_block_num * self.block_size, dtype=torch.float32, device=query.device)

        for b_idx in range(self.b):
            cur_seq = int(act_seq[b_idx].item())
            cur_block = (cur_seq + self.block_size - 1) // self.block_size
            cur_qs = q_scale_3d[b_idx * self.s1:(b_idx + 1) * self.s1, :, :]
            cur_w = weight_3d[b_idx * self.s1:(b_idx + 1) * self.s1, :, :]
            w_s = cur_qs * cur_w

            for block_idx in range(cur_block):
                cur_block_idx = int(block_table[b_idx, block_idx].item())
                if cur_block_idx < 0:
                    continue
                tail_seq = min(self.block_size, cur_seq - self.block_size * block_idx)
                cur_k = k_2d[cur_block_idx * self.block_size:cur_block_idx * self.block_size + tail_seq, :]

                qk_dot = torch.matmul(
                    q_2d[b_idx * self.s1 * self.n1:(b_idx + 1) * self.s1 * self.n1, :].float(),
                    cur_k.float().t(),
                ).relu()
                qk_dot = qk_dot * self.avoid_overflow_scale

                qk_dot_3d = qk_dot.reshape(self.s1, self.n1, tail_seq)
                w_qk = torch.bmm(w_s.to(torch.float32), qk_dot_3d.to(torch.float32))
                w_qk_2d = w_qk.reshape(self.s1, tail_seq)

                cur_ks = k_scale_2d[cur_block_idx:cur_block_idx + 1, :tail_seq].to(torch.float32)
                k_res = w_qk_2d * cur_ks
                mm_out[b_idx * self.s1:(b_idx + 1) * self.s1,
                       block_idx * self.block_size:block_idx * self.block_size + tail_seq] = k_res

        topk_res = torch.empty(t, self.selected_count, dtype=torch.int32, device=query.device)
        for b_idx in range(self.b):
            cur_seq = int(act_seq[b_idx].item())
            for s_idx in range(self.s1):
                eff_seq = cur_seq - (self.s1 - s_idx - 1)
                row_idx = b_idx * self.s1 + s_idx
                topk_in = mm_out[row_idx:row_idx + 1, :eff_seq]
                if eff_seq < self.selected_count:
                    _, cur_idx = torch.topk(topk_in, k=eff_seq, dim=-1)
                    pad_idx = torch.full((1, self.selected_count - eff_seq), -1, dtype=torch.int32, device=query.device)
                    cur_idx = torch.cat([cur_idx, pad_idx], dim=1)
                else:
                    _, cur_idx = torch.topk(topk_in, k=self.selected_count, dim=-1)
                topk_res[row_idx:row_idx + 1, :] = cur_idx

        return topk_res


def get_inputs():
    b, s1, n1, d = 2, 2, 64, 128
    s2 = 1024
    block_size = 128
    selected_count = 2048
    block_num = b * ((s2 + block_size - 1) // block_size)
    max_block_num = (s2 + block_size - 1) // block_size

    query = torch.randint(-128, 128, (b * s1, n1, d), dtype=torch.int8)
    q_scale = torch.randn(b * s1, n1, dtype=torch.float16).abs().clamp(min=0.01) / math.sqrt(d)
    key_cache = torch.randint(-128, 128, (block_num, block_size, 1, d), dtype=torch.int8)
    k_scale = torch.randn(block_num, block_size, 1, dtype=torch.float16).abs().clamp(min=0.01)
    weight = torch.randn(b * s1, n1, dtype=torch.float16) * 0.01 / math.sqrt(d)
    act_seq = torch.tensor([s2] * b, dtype=torch.int32)
    block_table = torch.tensor([list(range(max_block_num))] * b, dtype=torch.int32)
    return [query, q_scale, key_cache, k_scale, weight, act_seq, block_table]


def get_init_inputs():
    return [2, 2, 64, 128, 1024, 128, 2048]
