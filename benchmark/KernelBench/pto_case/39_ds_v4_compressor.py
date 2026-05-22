#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn

FORMULA = ("out[b,d] = golden_compress(x, sin, cos, wkv, wgate, ape, weight, kv_state, score_state,\n"
           "  kv_block_table, score_block_table, hadamard, ratio, start_pos_dy, rope_head_dim, rotate)\n"
           "  = sum(softmax(scores) * KV_states) -> rms_norm -> RoPE -> [Hadamard]")
DYNAMIC_AXIS = ["T", "S"]


def _rms_norm(x, eps, weight):
    dtype = x.dtype
    xf = x.float()
    var = xf.square().mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    return (weight * xf).to(dtype)


def _apply_rope_interleave(x, sin, cos):
    input_dtype = x.dtype
    if input_dtype != torch.float32:
        x = x.to(torch.float32)
    if cos.dtype != torch.float32:
        cos = cos.to(torch.float32)
        sin = sin.to(torch.float32)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    p = torch.stack((-x2, x1), dim=-1).flatten(-2)
    x_embed = (x * cos) + (p * sin)
    return x_embed.to(input_dtype)


class Model(nn.Module):
    def __init__(self, d=128, rope_head_dim=64, ratio=128, rotate=False, eps=1e-6):
        super().__init__()
        self.d = d
        self.rope_head_dim = rope_head_dim
        self.ratio = ratio
        self.rotate = rotate
        self.eps = eps

    def forward(self, x, sin, cos, wkv, wgate, ape, weight, kv_state, score_state,
                kv_block_table, score_block_table, hadamard, start_pos_dy):
        bsz, s1, _ = x.size()
        overlap = self.ratio == 4
        dtype = x.dtype

        xf = x.float()
        wkv_t = wkv.transpose(-2, -1).to(torch.float32)
        wgate_t = wgate.transpose(-2, -1).to(torch.float32)
        d = wkv_t.size(1) // (1 + overlap)

        kv_total = torch.matmul(xf, wkv_t)
        score_total = torch.matmul(xf, wgate_t)

        block_size = kv_state.shape[1]
        kv_output = torch.zeros(
            (min(bsz * s1, bsz * s1 // self.ratio + bsz), d),
            dtype=torch.bfloat16, device=x.device
        )

        for b_idx in range(bsz):
            start_pos = int(start_pos_dy[b_idx].item())
            for i in range(s1):
                should_compress = (start_pos + i + 1) % self.ratio == 0
                pos = (start_pos + i) % self.ratio
                kv = kv_total[b_idx, i:i + 1, :].clone()
                score = score_total[b_idx, i:i + 1, :].clone()
                score += ape[pos]

                if overlap:
                    kv_block_idx = int(kv_block_table[b_idx, (start_pos + i) // block_size].item())
                    score_block_idx = int(score_block_table[b_idx, (start_pos + i) // block_size].item())
                    cur_pos = (start_pos + i) % block_size
                    kv_state[kv_block_idx, cur_pos, :] = kv.squeeze(0)
                    score_state[score_block_idx, cur_pos, :] = score.squeeze(0)

                    if should_compress:
                        pre_kv_block_idx = int(kv_block_table[b_idx, (start_pos + i - 2 * self.ratio + 1) // block_size].item())
                        pre_score_block_idx = int(score_block_table[b_idx, (start_pos + i - 2 * self.ratio + 1) // block_size].item())
                        pre_start = (start_pos + i - 2 * self.ratio + 1) % block_size
                        pre_end = pre_start + self.ratio
                        cur_start = (start_pos + i - self.ratio + 1) % block_size
                        cur_end = cur_start + self.ratio

                        if start_pos < self.ratio:
                            kv_state_tmp = torch.cat([
                                kv_state[pre_kv_block_idx, pre_start:pre_end, :d] * 0,
                                kv_state[kv_block_idx, cur_start:cur_end, d:],
                            ], dim=0)
                            score_state_tmp = torch.cat([
                                score_state[pre_score_block_idx, pre_start:pre_end, :d] - float("inf"),
                                score_state[score_block_idx, cur_start:cur_end, d:],
                            ], dim=0)
                        else:
                            kv_state_tmp = torch.cat([
                                kv_state[pre_kv_block_idx, pre_start:pre_end, :d],
                                kv_state[kv_block_idx, cur_start:cur_end, d:],
                            ], dim=0)
                            score_state_tmp = torch.cat([
                                score_state[pre_score_block_idx, pre_start:pre_end, :d],
                                score_state[score_block_idx, cur_start:cur_end, d:],
                            ], dim=0)

                        kv_c = (kv_state_tmp * score_state_tmp.softmax(dim=0)).sum(dim=0, keepdim=False)
                else:
                    kv_block_idx = int(kv_block_table[b_idx, (start_pos + i) // block_size].item())
                    score_block_idx = int(score_block_table[b_idx, (start_pos + i) // block_size].item())
                    cur_pos = (start_pos + i) % block_size
                    kv_state[kv_block_idx, cur_pos, :] = kv.squeeze(0)
                    score_state[score_block_idx, cur_pos, :] = score.squeeze(0)

                    if should_compress:
                        kv_tmp = torch.cat((kv_state[kv_block_idx, :-1, :], kv), dim=0)
                        score_tmp = torch.cat((score_state[score_block_idx, :-1, :], score), dim=0)
                        kv_c = (kv_tmp * score_tmp.softmax(dim=0)).sum(dim=0, keepdim=False)

                if should_compress:
                    kv_c = _rms_norm(kv_c.to(dtype), self.eps, weight)
                    kv_rope = kv_c[..., -self.rope_head_dim:].clone()
                    kv_new = kv_c.clone()
                    kv_new[..., -self.rope_head_dim:] = _apply_rope_interleave(
                        kv_rope, sin[b_idx, ...], cos[b_idx, ...]
                    )
                    if self.rotate:
                        kv_output[b_idx, :] = torch.matmul(kv_new, hadamard)
                    else:
                        kv_output[b_idx, :] = kv_new

        return kv_output


def _gen_block_table(bsz, overlap, device):
    if overlap:
        return torch.ones(bsz, 100, dtype=torch.int32, device=device) + \
               torch.arange(bsz, dtype=torch.int32, device=device).view(-1, 1) * 2
    else:
        return (torch.arange(100, dtype=torch.int32, device=device) % 2 + 1) + \
               torch.arange(bsz, dtype=torch.int32, device=device).view(-1, 1) * 2


def get_inputs():
    bsz, seq, h, d, rope_head_dim, ratio = 2, 2, 64, 32, 16, 8
    rotate = False
    overlap = (ratio == 4)
    coff = 1 + overlap

    torch.manual_seed(42)
    x = torch.rand((bsz, seq, h), dtype=torch.bfloat16)
    rope_axis0 = min(bsz * seq, bsz * seq // ratio + bsz)
    sin = torch.rand((rope_axis0, rope_head_dim), dtype=torch.bfloat16)
    cos = torch.rand((rope_axis0, rope_head_dim), dtype=torch.bfloat16)
    wkv = torch.rand((coff * d, h), dtype=torch.bfloat16)
    wgate = torch.rand((coff * d, h), dtype=torch.bfloat16)
    ape = torch.rand((ratio, coff * d), dtype=torch.float32)
    weight = torch.ones(d, dtype=torch.float32)
    block_table = _gen_block_table(bsz, overlap, "cpu")
    max_block_id = block_table.max().item()
    kv_state = torch.zeros((max_block_id + 1, 128, coff * d), dtype=torch.float32)
    score_state = torch.zeros((max_block_id + 1, 128, coff * d), dtype=torch.float32)
    hadamard = torch.rand((d, d), dtype=torch.bfloat16) * (d ** -0.5)
    start_pos_dy = torch.tensor([ratio - 2] * bsz, dtype=torch.int32)

    return [x, sin, cos, wkv, wgate, ape, weight, kv_state, score_state,
            block_table, block_table, hadamard, start_pos_dy]


def get_init_inputs():
    return [32, 16, 8, False, 1e-6]
