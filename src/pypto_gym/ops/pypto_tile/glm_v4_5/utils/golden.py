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
import math
import torch
import torch_npu  # noqa: F401


class AttnGolden:

    @staticmethod
    def gen_block_table(actual_seq_len, block_size, block_table_shape):
        block_num_per_batch = []
        block_num = 0

        if isinstance(actual_seq_len, torch.Tensor):
            if actual_seq_len.device.type != 'cpu':
                actual_seq_len_cpu = actual_seq_len.cpu()
            else:
                actual_seq_len_cpu = actual_seq_len
            for actual_seq in actual_seq_len_cpu:
                block_num_per_batch.append(math.ceil(actual_seq.item() / block_size))
                block_num += math.ceil(actual_seq.item() / block_size)
        else:
            for actual_seq in actual_seq_len:
                block_num_per_batch.append(math.ceil(actual_seq / block_size))
                block_num += math.ceil(actual_seq / block_size)

        block_idx_list = torch.arange(0, block_num, dtype=torch.int32)
        block_idx_list = block_idx_list[torch.randperm(block_idx_list.size(0))]

        block_table = torch.full(block_table_shape, -1, dtype=torch.int32)
        block_idx = 0
        block_table_batch_idx = 0
        for idx in block_num_per_batch:
            for j in range(idx):
                block_table[block_table_batch_idx][j] = block_idx_list[block_idx]
                block_idx += 1
            block_table_batch_idx += 1
        return block_table

    @staticmethod
    def _rms_norm_bias(x, gamma, bias, eps):
        in_dtype = x.dtype
        xf = x.to(torch.float32)
        mean_coff = 1.0 / x.shape[-1]
        square = xf * xf
        mean_res = square * mean_coff
        reduce_sum = torch.sum(mean_res, dim=-1, keepdim=True) + eps
        reduce_sqrt = torch.sqrt(reduce_sum)
        res_div = xf / reduce_sqrt
        res = res_div * gamma.to(torch.float32) + bias.to(torch.float32)
        return res.to(in_dtype)

    @staticmethod
    def _apply_rotary_emb(x, cos, sin):
        x1, x2 = torch.chunk(x, 2, dim=-1)
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return torch.cat((o1, o2), dim=-1)

    @staticmethod
    def attention_golden(
        hidden_states,
        residual,
        input_layernorm_weight,
        input_layernorm_bias,
        qkv_proj_scale,
        qkv_proj_offset,
        qkv_proj_weight,
        qkv_proj_quant_bias,
        qkv_proj_deq_scale,
        q_norm_weight,
        q_norm_bias,
        k_norm_weight,
        k_norm_bias,
        cos,
        sin,
        key_cache,
        value_cache,
        block_tables,
        actual_seq_lens,
        slot_mapping,
        eps,
        enable_residual=True,
        num_decode_tokens=0,
    ):
        bs, hidden_size = hidden_states.shape
        total_head_size = qkv_proj_weight.shape[1]
        head_size = q_norm_weight.shape[0]
        n1 = total_head_size // head_size - 2
        n2 = 1
        q_size = n1 * head_size
        rotary_dim = head_size // 2
        half_rotary_dim = rotary_dim // 2

        x_fp32 = hidden_states.to(torch.float32)
        residual_fp32 = residual.to(torch.float32) if enable_residual else torch.zeros_like(x_fp32)
        x_fp32 = x_fp32 + residual_fp32

        x_mean_coff = 1.0 / hidden_size
        square = x_fp32 * x_fp32
        mean_res = square * x_mean_coff
        reduce_sum = torch.sum(mean_res, dim=-1, keepdim=True) + eps
        reduce_sqrt = torch.sqrt(reduce_sum)
        x_norm_before_gamma = x_fp32 / reduce_sqrt

        residual_bf16 = x_fp32.to(torch.bfloat16)

        x_g_bf16 = (x_norm_before_gamma * input_layernorm_weight + input_layernorm_bias).to(torch.bfloat16)
        x_quant = torch_npu.npu_quantize(x_g_bf16, qkv_proj_scale, qkv_proj_offset, torch.qint8, -1, False)
        mm_bf16 = torch_npu.npu_quant_matmul(x_quant, qkv_proj_weight, qkv_proj_deq_scale,
                                             bias=qkv_proj_quant_bias, output_dtype=torch.bfloat16)

        q_raw, k_raw, v_raw = torch.split(mm_bf16, [q_size, head_size, head_size], dim=-1)
        q_3d = q_raw.view(bs, n1, head_size)
        k_3d = k_raw.view(bs, n2, head_size)
        v_3d = v_raw.view(bs, n2, head_size)

        q_normed = AttnGolden._rms_norm_bias(q_3d, q_norm_weight, q_norm_bias, eps)
        k_normed = AttnGolden._rms_norm_bias(k_3d, k_norm_weight, k_norm_bias, eps)

        q_rot = q_normed[..., :rotary_dim]
        q_pass = q_normed[..., rotary_dim:]
        k_rot = k_normed[..., :rotary_dim]
        k_pass = k_normed[..., rotary_dim:]

        cos_3d = cos.view(bs, 1, half_rotary_dim)
        sin_3d = sin.view(bs, 1, half_rotary_dim)
        q_rope = AttnGolden._apply_rotary_emb(q_rot, cos_3d, sin_3d)
        k_rope = AttnGolden._apply_rotary_emb(k_rot, cos_3d, sin_3d)

        q_final_3d = torch.cat([q_rope, q_pass], dim=-1)
        q_final_2d = q_final_3d.view(bs, q_size)
        k_final = torch.cat([k_rope, k_pass], dim=-1).view(bs, head_size)
        v_final = v_3d.view(bs, head_size)

        key_cache_flat = key_cache.view(-1, n2 * head_size)
        key_cache_flat[slot_mapping.long()] = k_final
        value_cache_flat = value_cache.view(-1, n2 * head_size)
        value_cache_flat[slot_mapping.long()] = v_final

        kv_cache_n_blocks, block_size, _, _ = key_cache.shape
        attention_output = torch.zeros(bs, n1, head_size, dtype=hidden_states.dtype, device=hidden_states.device)

        actual_on_cpu = actual_seq_lens if actual_seq_lens.device.type == 'cpu' else actual_seq_lens.cpu()
        block_tables_cpu = block_tables if block_tables.device.type == 'cpu' else block_tables.cpu()
        softmax_scale = head_size ** -0.5

        for b_idx in range(bs):
            seq_len = actual_on_cpu[b_idx].item()
            for q_idx in range(n1):
                q_vec = q_final_3d[b_idx, q_idx].view(1, head_size)
                kv_blocks = AttnGolden._gather_kv(key_cache, value_cache, block_tables_cpu, b_idx, seq_len,
                                                  block_size, n2, head_size)
                k_seq, v_seq = kv_blocks
                scores = (q_vec @ k_seq.T) * softmax_scale
                scores_fp32 = scores.float()
                scores_max = scores_fp32.max(dim=-1, keepdim=True).values
                scores_exp = torch.exp(scores_fp32 - scores_max)
                attn_weights = scores_exp / scores_exp.sum(dim=-1, keepdim=True)
                attn_out = (attn_weights.to(v_seq.dtype) @ v_seq)
                attention_output[b_idx, q_idx] = attn_out

        return attention_output, residual_bf16

    @staticmethod
    def _gather_kv(key_cache, value_cache, block_tables, b_idx, seq_len, block_size, n2, head_size):
        device = key_cache.device
        dtype = key_cache.dtype
        kv_max = (seq_len + block_size - 1) // block_size * block_size

        k_out = torch.zeros(kv_max, head_size, dtype=dtype, device=device)
        v_out = torch.zeros(kv_max, head_size, dtype=dtype, device=device)

        block_list = block_tables[b_idx]
        s_idx = 0
        for _, block_idx in enumerate(block_list):
            if block_idx == -1:
                break
            start_idx = s_idx * block_size
            end_idx = min((s_idx + 1) * block_size, kv_max)
            valid = min(block_size, kv_max - start_idx)

            k_out[start_idx:end_idx, :] = key_cache[block_idx, :valid, :, :].view(valid, head_size)
            v_out[start_idx:end_idx, :] = value_cache[block_idx, :valid, :, :].view(valid, head_size)
            s_idx += 1
            if end_idx >= kv_max:
                break

        k_out = k_out[:seq_len, :]
        v_out = v_out[:seq_len, :]
        return k_out, v_out


attn_golden = AttnGolden()
