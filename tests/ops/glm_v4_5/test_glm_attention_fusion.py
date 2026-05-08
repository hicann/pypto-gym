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
"""
"""
import os
import numpy as np
import torch
import torch_npu

import sys, os; _p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')): _p = os.path.dirname(_p)
sys.path.insert(0, os.path.join(_p, 'src')); sys.path.insert(0, os.path.join(_p, 'src', 'pypto_gym', 'ops', 'pypto_tile'))

import pypto
import pytest
from glm_v4_5.glm_attention_fusion_impl import attention, get_qwen_common_config
from glm_v4_5.utils.np_compare import detailed_allclose_manual as compare
from glm_v4_5.utils.golden import attn_golden


np.random.seed(0)
torch.manual_seed(0)
np.set_printoptions(formatter={'float': '{:.6f}'.format})


@pytest.mark.soc("950", "910")
def test_attention():
    torch_npu.npu.config.allow_internal_format = True
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    device = f'npu:{device_id}'
    npu = 'npu'
    torch.npu.set_device(int(device_id))
    attn_cfg, _ = get_qwen_common_config(device=device)

    torch_dtype = torch.bfloat16
    b = attn_cfg.b
    s1 = attn_cfg.s1
    d = attn_cfg.q_d
    n1 = attn_cfg.n1
    n2 = attn_cfg.n2
    bs = b * s1
    hidden_size = attn_cfg.hidden_size
    q_size = n1 * d
    total_head_size = q_size + 2 * d
    rotary_dim = d // 2
    half_rotary_dim = rotary_dim // 2

    block_num = attn_cfg.kv_num_blocks
    block_size = attn_cfg.block_size
    max_num_blocks_per_query = attn_cfg.max_num_blocks_per_query

    actual_seq_lens = attn_cfg.actual_seq.to(dtype=torch.int32, device=device)

    kv_cache_shape = [attn_cfg.kv_num_blocks, block_size, n2, d]
    block_table_shape = [attn_cfg.block_table_batch, max_num_blocks_per_query]

    slot_mapping = torch.randperm(block_num * block_size, dtype=torch.int32)[:b].to(npu)

    key_cache = torch.empty(kv_cache_shape, dtype=torch_dtype).uniform_(-1, 1).to(npu) * 0
    value_cache = torch.empty(kv_cache_shape, dtype=torch_dtype).uniform_(-1, 1).to(npu) * 0
    block_tables = attn_golden.gen_block_table(actual_seq_lens, block_size, block_table_shape)
    block_tables = block_tables.to(dtype=torch.int32, device=f'npu:{device_id}')
    key_cache_clone = key_cache.clone()
    value_cache_clone = value_cache.clone()

    hidden_states = torch.rand(bs, hidden_size, dtype=torch.bfloat16).to(npu)
    residual = torch.rand(bs, hidden_size, dtype=torch.bfloat16).to(npu)
    input_layernorm_weight = torch.rand(hidden_size, dtype=torch.bfloat16).to(npu)
    input_layernorm_bias = torch.rand(hidden_size, dtype=torch.bfloat16).to(npu)
    qkv_proj_scale = torch.rand(hidden_size, dtype=torch.bfloat16).to(npu)
    qkv_proj_offset = torch.rand(hidden_size, dtype=torch.bfloat16).to(npu)

    qkv_proj_weight = torch.randint(0, 128, size=(hidden_size, total_head_size), dtype=torch.int8,
                                    device=f'npu:{device_id}')
    qkv_proj_weight = torch_npu.npu_format_cast(qkv_proj_weight, 29)
    qkv_proj_quant_bias = torch.randint(0, 128, size=(total_head_size,), dtype=torch.int32, device=f'npu:{device_id}')
    qkv_proj_deq_scale = torch.rand(total_head_size, dtype=torch.float32).to(npu)
    q_norm_weight = torch.rand(d, dtype=torch.bfloat16).to(npu)
    q_norm_bias = torch.rand(d, dtype=torch.bfloat16).to(npu)
    k_norm_weight = torch.rand(d, dtype=torch.bfloat16).to(npu)
    k_norm_bias = torch.rand(d, dtype=torch.bfloat16).to(npu)
    cos = torch.rand(bs, 1, half_rotary_dim, dtype=torch.bfloat16).to(npu)
    sin = torch.rand(bs, 1, half_rotary_dim, dtype=torch.bfloat16).to(npu)

    loop_times = 1
    for _ in range(loop_times):
        output, residual_tmp = attention(
            hidden_states=hidden_states,
            residual=residual,
            input_layernorm_weight=input_layernorm_weight,
            input_layernorm_bias=input_layernorm_bias,
            qkv_proj_scale=qkv_proj_scale,
            qkv_proj_offset=qkv_proj_offset,
            qkv_proj_weight=qkv_proj_weight,
            qkv_proj_quant_bias=qkv_proj_quant_bias,
            qkv_proj_deq_scale=qkv_proj_deq_scale,
            q_norm_weight=q_norm_weight,
            q_norm_bias=q_norm_bias,
            k_norm_weight=k_norm_weight,
            k_norm_bias=k_norm_bias,
            cos=cos,
            sin=sin,
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=block_tables,
            actual_seq_lens=actual_seq_lens,
            slot_mapping=slot_mapping,
            eps=attn_cfg.eps,
            enable_residual=True,
            num_decode_tokens=0
        )

    attention_output, residual_g = attn_golden.attention_golden(
        hidden_states=hidden_states,
        residual=residual,
        input_layernorm_weight=input_layernorm_weight,
        input_layernorm_bias=input_layernorm_bias,
        qkv_proj_scale=qkv_proj_scale,
        qkv_proj_offset=qkv_proj_offset,
        qkv_proj_weight=qkv_proj_weight,
        qkv_proj_quant_bias=qkv_proj_quant_bias,
        qkv_proj_deq_scale=qkv_proj_deq_scale,
        q_norm_weight=q_norm_weight,
        q_norm_bias=q_norm_bias,
        k_norm_weight=k_norm_weight,
        k_norm_bias=k_norm_bias,
        cos=cos,
        sin=sin,
        key_cache=key_cache_clone,
        value_cache=value_cache_clone,
        block_tables=block_tables,
        actual_seq_lens=actual_seq_lens,
        slot_mapping=slot_mapping,
        eps=attn_cfg.eps,
        enable_residual=True,
        num_decode_tokens=0
    )

    compare(np.array(residual_g.cpu().flatten().tolist()), np.array(residual_tmp.flatten().tolist()),
            "residual_g", rtol=0.001, atol=0.001)
    compare(np.array(attention_output.flatten().tolist()), np.array(output.flatten().tolist()),
            "golden vs pypto", rtol=0.003, atol=0.003)


if __name__ == "__main__":
    test_attention()
