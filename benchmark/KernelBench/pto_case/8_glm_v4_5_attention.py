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

"""KernelBench case: GLM V4.5 Full Attention — RMSNorm + QuantQKV + RoPE + Scatter + IFA online softmax."""

import math
import torch
import torch.nn as nn

FORMULA = "attn_out = IFA_online_softmax( QKV_split( RMSNorm_QK( RoPE( QK_RMSNORM( QuantMatMul( AddRMSNorm( x + residual ) ) ) ) ) ), Scatter(k, v, slot_mapping) )"
DYNAMIC_AXIS = ["M", "S"]


class Model(nn.Module):
    def __init__(self, hidden_size: int = 5120, q_size: int = 1536,
                 head_size: int = 128, num_kv_heads: int = 1,
                 block_size: int = 128, num_blocks: int = 32,
                 eps: float = 1e-5, enable_residual: bool = True,
                 num_decode_tokens: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.q_size = q_size
        self.kv_size = head_size
        self.num_kv_heads = num_kv_heads
        self.total_head_size = q_size + 2 * head_size
        self.half_rotary_dim = head_size // 4
        self.rotary_dim = head_size // 2
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.eps = eps
        self.enable_residual = enable_residual
        self.num_decode_tokens = num_decode_tokens
        self.softmax_scale = head_size ** -0.5

        self.input_layernorm_weight = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.bfloat16) / math.sqrt(hidden_size))
        self.input_layernorm_bias = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.bfloat16) / math.sqrt(hidden_size))
        self.qkv_proj_scale = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.float32) / math.sqrt(hidden_size))
        self.qkv_proj_offset = nn.Parameter(
            torch.zeros(hidden_size, dtype=torch.float32))
        self.register_buffer('qkv_proj_weight',
            torch.randint(-128, 128, (hidden_size, self.total_head_size), dtype=torch.int8))
        self.register_buffer('qkv_proj_quant_bias',
            torch.randint(-128, 128, (self.total_head_size,), dtype=torch.int32))
        self.qkv_proj_deq_scale = nn.Parameter(
            torch.randn(self.total_head_size, dtype=torch.float32) / math.sqrt(self.total_head_size))
        self.q_norm_weight = nn.Parameter(
            torch.randn(head_size, dtype=torch.bfloat16) / math.sqrt(head_size))
        self.q_norm_bias = nn.Parameter(
            torch.randn(head_size, dtype=torch.bfloat16) / math.sqrt(head_size))
        self.k_norm_weight = nn.Parameter(
            torch.randn(head_size, dtype=torch.bfloat16) / math.sqrt(head_size))
        self.k_norm_bias = nn.Parameter(
            torch.randn(head_size, dtype=torch.bfloat16) / math.sqrt(head_size))

    def _add_rms_norm(self, hidden_states: torch.Tensor, residual: torch.Tensor,
                      gamma: torch.Tensor, bias: torch.Tensor
                      ) -> tuple[torch.Tensor, torch.Tensor]:
        x_fp32 = residual.float()
        residual_fp32 = hidden_states.float()
        x_fp32 = x_fp32 + residual_fp32
        mean_coff = 1.0 / x_fp32.shape[-1]
        x_square = x_fp32 * x_fp32
        x_mean = x_square * mean_coff
        x_reduce_sum = torch.sum(x_mean, dim=-1, keepdim=True) + self.eps
        x_reduce_sqrt = torch.sqrt(x_reduce_sum)
        x_res_div = x_fp32 / x_reduce_sqrt
        x_mul_res = x_res_div * gamma.float()
        x_add_bias = x_mul_res + bias.float()
        return x_add_bias.to(hidden_states.dtype), x_fp32.to(hidden_states.dtype)

    def _rms_norm(self, x: torch.Tensor, gamma: torch.Tensor, bias: torch.Tensor
                  ) -> torch.Tensor:
        x_fp32 = x.float()
        mean_coff = 1.0 / x_fp32.shape[-1]
        x_square = x_fp32 * x_fp32
        x_mean = x_square * mean_coff
        x_reduce_sum = torch.sum(x_mean, dim=-1, keepdim=True) + self.eps
        x_reduce_sqrt = torch.sqrt(x_reduce_sum)
        x_res_div = x_fp32 / x_reduce_sqrt
        x_mul_res = x_res_div * gamma.float()
        x_add_bias = x_mul_res + bias.float()
        return x_add_bias.to(x.dtype)

    def _apply_rotary_emb_neuron(self, x: torch.Tensor, cos: torch.Tensor,
                                  sin: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return torch.cat((o1, o2), dim=-1)

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor,
                       cos: torch.Tensor, sin: torch.Tensor
                       ) -> tuple[torch.Tensor, torch.Tensor]:
        x_dtype = q.dtype
        q_fp32 = q.float()
        k_fp32 = k.float()
        cos_fp32 = cos.float()
        sin_fp32 = sin.float()

        q_embed = self._apply_rotary_emb_neuron(q_fp32, cos_fp32, sin_fp32)
        k_embed = self._apply_rotary_emb_neuron(k_fp32, cos_fp32, sin_fp32)

        if x_dtype != torch.float32:
            q_embed = q_embed.to(x_dtype)
            k_embed = k_embed.to(x_dtype)
        return q_embed, k_embed

    @staticmethod
    def _matmul_proxy(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.matmul(left.float(), right.float())

    def _scatter_update(self, value: torch.Tensor, cache: torch.Tensor,
                         slots: torch.Tensor):
        bs = value.shape[0]
        for bs_idx in range(bs):
            index = slots[bs_idx].item()
            dim_0 = index // self.block_size
            dim_1 = index % self.block_size
            cache[dim_0, dim_1, :] = value[bs_idx]

    def _ifa_flash_torch(self, q: torch.Tensor, k: torch.Tensor,
                          v: torch.Tensor, block_table: torch.Tensor,
                          kv_act_seqs: torch.Tensor, out: torch.Tensor):
        q_shape = q.shape
        bs1, n1, d = q_shape[0], q_shape[1], q_shape[2]
        b = kv_act_seqs.shape[0]
        s1 = bs1 // b
        k_shape = k.shape
        block_num, _, n2, _ = k_shape
        g = n1 // n2
        g_tile = g
        k_2d = k.reshape(-1, d)
        v_2d = v.reshape(-1, d)
        q_2d = q.reshape(-1, d)

        for b_idx in range(b):
            for s1_idx in range(s1):
                cur_seq_tensor = kv_act_seqs[b_idx] - (s1 - 1 - s1_idx)
                cur_seq = max(cur_seq_tensor.item(), 0)
                s2_loop = math.ceil(cur_seq / self.block_size)

                for n2_idx in range(n2):
                    for g_idx in range(g // g_tile):
                        device = q.device
                        oi_upd = torch.zeros((g_tile, d), device=device, dtype=torch.float32)
                        li_upd = torch.zeros(g_tile, device=device, dtype=torch.float32)
                        mi_upd = torch.zeros(g_tile, device=device, dtype=torch.float32)

                        for s2_idx in range(s2_loop):
                            block_idx = block_table[b_idx][s2_idx].item()
                            bs_ofs = b_idx * s1 + s1_idx
                            n2g_ofs = n2_idx * g + g_idx * g_tile
                            actual_s2_tile = min(self.block_size, cur_seq - s2_idx * self.block_size)

                            qi_start = bs_ofs * n1 + n2g_ofs
                            qi_end = qi_start + g_tile
                            qi = q_2d[qi_start:qi_end, :]

                            kj_start = block_idx * self.block_size
                            kj_end = kj_start + actual_s2_tile
                            kj = k_2d[kj_start:kj_end, :]
                            vj = v_2d[kj_start:kj_end, :]

                            mm1 = self._matmul_proxy(qi, kj.t())
                            muls_res = mm1 * self.softmax_scale
                            tilda_mij, _ = torch.max(muls_res, dim=-1, keepdim=True)

                            if s2_idx == 0:
                                tsub = muls_res - tilda_mij
                                tilda_pij = torch.exp(tsub)
                                tilda_lij = torch.sum(tilda_pij, dim=-1, keepdim=True)
                                oi_tmp = self._matmul_proxy(tilda_pij.to(q.dtype), vj)
                                oi_upd = oi_tmp
                                li_upd = tilda_lij.squeeze(-1)
                                mi_upd = tilda_mij.squeeze(-1)
                            else:
                                mi = mi_upd.unsqueeze(-1)
                                max_new, _ = torch.max(torch.cat([mi, tilda_mij], dim=-1),
                                                       dim=-1, keepdim=True)
                                tsub = muls_res - max_new
                                tilda_pij = torch.exp(tsub)
                                tilda_lij = torch.sum(tilda_pij, dim=-1, keepdim=True)
                                tsub2 = torch.sub(mi, max_new)
                                mi_upd = max_new.squeeze(-1)
                                update_mul = torch.exp(tsub2)
                                li = li_upd.unsqueeze(-1)
                                sum_new = li * update_mul + tilda_lij
                                li_upd = sum_new.squeeze(-1)
                                q1 = self._matmul_proxy(tilda_pij.to(q.dtype), vj)
                                oi_upd = oi_upd * update_mul + q1

                            if s2_idx == s2_loop - 1:
                                li = li_upd.unsqueeze(-1)
                                oi_final = oi_upd / li
                                oi_upd_3d = oi_final.unsqueeze(0)
                                attn_out_start_col = n2g_ofs
                                attn_out_end_col = n2g_ofs + g_tile
                                if attn_out_end_col > out.shape[1]:
                                    attn_out_end_col = out.shape[1]
                                    attn_out_start_col = attn_out_end_col - g_tile
                                out[bs_ofs:bs_ofs + 1, attn_out_start_col:attn_out_end_col, :] = \
                                    oi_upd_3d.to(q.dtype)

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor,
                key_cache: torch.Tensor, value_cache: torch.Tensor,
                block_tables: torch.Tensor, actual_seqs: torch.Tensor,
                slot_mapping: torch.Tensor,
                ) -> tuple[torch.Tensor, torch.Tensor]:
        bs = hidden_states.shape[0]
        d = self.head_size

        x_g, residual_out = self._add_rms_norm(
            hidden_states, residual,
            self.input_layernorm_weight, self.input_layernorm_bias)

        x_quant = (x_g.float() * self.qkv_proj_scale.float() + self.qkv_proj_offset.float()
                   ).round().clamp(-128, 127).to(torch.int8)

        mm_fp32 = x_quant.float() @ self.qkv_proj_weight.float()
        mm_with_bias = mm_fp32 + self.qkv_proj_quant_bias.float().unsqueeze(0)
        mm_golden = (mm_with_bias * self.qkv_proj_deq_scale.float().unsqueeze(0)).to(x_g.dtype)

        q_g, k_g, v_g = mm_golden.split([self.q_size, d, d], dim=-1)

        q_by_head = q_g.view(*q_g.shape[:-1], q_g.shape[-1] // d, d)
        q_by_head = self._rms_norm(q_by_head, self.q_norm_weight, self.q_norm_bias)

        k_by_head = k_g.view(*k_g.shape[:-1], k_g.shape[-1] // d, d)
        k_by_head = self._rms_norm(k_by_head, self.k_norm_weight, self.k_norm_bias)

        cos_s = cos.view(cos.shape[0], 1, self.half_rotary_dim)
        sin_s = sin.view(sin.shape[0], 1, self.half_rotary_dim)

        q_rot = q_by_head[..., :self.rotary_dim]
        q_pass = q_by_head[..., self.rotary_dim:]
        k_rot = k_by_head[..., :self.rotary_dim]
        k_pass = k_by_head[..., self.rotary_dim:]

        q_r, k_r = self._apply_rotary(q_rot, k_rot, cos_s, sin_s)
        q_cat = torch.cat((q_r, q_pass), dim=-1)
        k_cat = torch.cat((k_r, k_pass), dim=-1)

        q_out = q_cat.view(bs, self.q_size)
        k_out = k_cat.view(bs, d)

        n2 = key_cache.shape[2]
        self._scatter_update(k_out.view(bs, n2, d), key_cache, slot_mapping)
        self._scatter_update(v_g.view(bs, n2, d), value_cache, slot_mapping)

        q_3d = q_out.view(bs, -1, d)
        attn_out = torch.zeros(q_3d.shape, dtype=q_3d.dtype, device=q_3d.device)
        self._ifa_flash_torch(q_3d, key_cache, value_cache, block_tables,
                              actual_seqs, attn_out)

        return attn_out, residual_out


def get_inputs():
    bs = 2
    hidden_size = 5120
    half_rotary_dim = 32
    d = 128
    block_size = 128
    num_blocks = 32
    s2 = 512

    x = torch.randn(bs, hidden_size, dtype=torch.bfloat16) / math.sqrt(hidden_size)
    residual = torch.randn(bs, hidden_size, dtype=torch.bfloat16) / math.sqrt(hidden_size)
    cos = torch.randn(bs, 1, half_rotary_dim, dtype=torch.bfloat16) / math.sqrt(half_rotary_dim)
    sin = torch.randn(bs, 1, half_rotary_dim, dtype=torch.bfloat16) / math.sqrt(half_rotary_dim)
    k_cache = torch.randn(num_blocks, block_size, 1, d, dtype=torch.bfloat16) / math.sqrt(d)
    v_cache = torch.randn(num_blocks, block_size, 1, d, dtype=torch.bfloat16) / math.sqrt(d)

    max_blocks = (s2 + block_size - 1) // block_size
    block_tables = torch.full((bs, max_blocks), -1, dtype=torch.int32)
    blk_idx = 0
    for bi in range(bs):
        for j in range(max_blocks):
            if blk_idx < num_blocks:
                block_tables[bi, j] = blk_idx
                blk_idx += 1

    actual_seqs = torch.full((bs,), s2, dtype=torch.int32)
    slot_mapping = torch.arange(bs, dtype=torch.int32)

    return [x, residual, cos, sin, k_cache, v_cache, block_tables, actual_seqs, slot_mapping]


def get_init_inputs():
    return [5120, 1536, 128, 1, 128, 32, 1e-5, True, 1]
