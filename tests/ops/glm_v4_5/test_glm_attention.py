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
import os
import math
import torch
import torch_npu
import pytest
import numpy as np
from numpy.testing import assert_allclose
import pypto
from pypto_gym.ops.pypto_tile.glm_v4_5.glm_attention_impl import (
    attention, attention_for_950, set_qwen_common_config,
    get_common_config,
)


np.random.seed(0)
torch.manual_seed(0)
np.set_printoptions(formatter={'float': '{:.6f}'.format})


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


def kv_cache_concat_bsnd(kr_cache_out, kv_cache_out, block_table, atten_config):
    b = atten_config.b
    n2 = atten_config.n2
    kv_lora_rank = atten_config.q_d
    rope_dim = atten_config.kv_d
    block_size = atten_config.block_size
    kv_cache_actual_seq = atten_config.actual_seq
    dtype = kv_cache_out.dtype

    if isinstance(kv_cache_actual_seq, torch.Tensor):
        if kv_cache_actual_seq.device.type != 'cpu':
            kv_cache_actual_seq_cpu = kv_cache_actual_seq.cpu()
        else:
            kv_cache_actual_seq_cpu = kv_cache_actual_seq
        kv_max = (torch.max(kv_cache_actual_seq_cpu).item() + block_size - 1) // block_size * block_size
    else:
        kv_max = (max(kv_cache_actual_seq) + block_size - 1) // block_size * block_size

    device = kr_cache_out.device
    k_cache = torch.zeros([b, kv_max, n2, kv_lora_rank], dtype=dtype, device=device)
    v_cache = torch.zeros([b, kv_max, n2, rope_dim], dtype=dtype, device=device)

    for b_idx in range(b):
        block_list = block_table[b_idx]
        kv_nope_temp_tensor = torch.zeros([1, kv_max, n2, kv_lora_rank], dtype=dtype, device=device)
        kv_rope_temp_tensor = torch.zeros([1, kv_max, n2, rope_dim], dtype=dtype, device=device)
        s_idx = 0

        for _, block_idx in enumerate(block_list):
            if block_idx == -1:
                break
            start_idx = s_idx * block_size
            end_idx = (s_idx + 1) * block_size

            kv_nope_temp_tensor[:, start_idx:end_idx, :, :] = kv_cache_out[block_idx:block_idx + 1, :, :, :]
            kv_rope_temp_tensor[:, start_idx:end_idx, :, :] = kr_cache_out[block_idx:block_idx + 1, :, :, :]
            s_idx += 1

        v_cache[b_idx:b_idx + 1, :, :, :] = kv_nope_temp_tensor
        k_cache[b_idx:b_idx + 1, :, :, :] = kv_rope_temp_tensor

    return k_cache, v_cache


def get_special_array(m, n):
    q_shape = [m, n]
    base = np.arange(1, m + 1)
    q = base[:, np.newaxis]
    q = np.broadcast_to(q, q_shape)
    q = q.astype(np.float16)
    return q


def softmax(x, is_fp16=False):
    if is_fp16:
        original_dtype = x.dtype
        x = x.float()
    x_max = x.max(dim=-1, keepdim=True).values
    x_sub = x - x_max
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdim=True)
    ans = y / x_sum
    if is_fp16:
        ans = ans.to(original_dtype)
        x_max = x_max.to(original_dtype)
        x_sum = x_sum.to(original_dtype)

    return ans, x_max, x_sum


def ifa(atten_cfg, case_950=0):
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch_dtype = torch.bfloat16
    torch.npu.set_device(int(device_id))

    b = atten_cfg.b
    s1 = atten_cfg.s1
    d = atten_cfg.q_d
    nq = atten_cfg.n1
    nkv = atten_cfg.n2

    block_size = atten_cfg.block_size
    max_num_blocks_per_query = atten_cfg.max_num_blocks_per_query

    kv_cache_actual_seq = atten_cfg.actual_seq

    q_shape = [b * s1, nq, d]
    kv_shape = [atten_cfg.kv_num_blocks, block_size, nkv, d]
    block_table_shape = [atten_cfg.block_table_batch, max_num_blocks_per_query]

    device = f'npu:{device_id}'
    q = torch.empty(q_shape, dtype=torch_dtype).uniform_(-1, 1).to(device=device)
    k = torch.empty(kv_shape, dtype=torch_dtype).uniform_(-1, 1).to(device=device)
    v = torch.empty(kv_shape, dtype=torch_dtype).uniform_(-1, 1).to(device=device)
    attention_output = torch.zeros(q_shape, dtype=torch_dtype).to(device=device)

    block_table = gen_block_table(kv_cache_actual_seq, block_size, block_table_shape)

    k_cache_bsnd, v_cache_bsnd = kv_cache_concat_bsnd(k, v, block_table, atten_cfg)

    for i in range(b):
        for j in range(s1):
            for n2_idx in range(nkv):
                kv_seq_len = kv_cache_actual_seq[i].item()
                seq_len = kv_seq_len - s1 + 1 + j
                q_bs = q[i * s1 + j]
                k_bs = k_cache_bsnd[i, :seq_len, n2_idx:n2_idx + 1].reshape(seq_len, d)
                v_bs = v_cache_bsnd[i, :seq_len, n2_idx:n2_idx + 1].reshape(seq_len, d)
                qk_bmm_res = torch.matmul(q_bs, k_bs.transpose(1, 0))
                qk_ele_res = qk_bmm_res * atten_cfg.softmax_scale
                softmax_res, _, _ = softmax(qk_ele_res, True)

                bmm2_res = torch.matmul(softmax_res, v_bs)

                attention_output[i * s1 + j] = bmm2_res

    block_table_torch = block_table.to(dtype=torch.int32, device=device)
    act_seq_torch = kv_cache_actual_seq.to(dtype=torch.int32, device=device)

    out_torch = torch.zeros(q_shape, dtype=torch_dtype).to(device=device)

    inputs = [
        q,
        k,
        v,
        block_table_torch,
        act_seq_torch,
        out_torch
    ]
    if case_950 == 1:
        attention_for_950(*inputs)
    else:
        attention(*inputs)

    assert_allclose(np.array(attention_output.cpu().flatten().tolist()),
                    np.array(out_torch.cpu().flatten().tolist()),
                    rtol=0.0078125, atol=0.0001)


@pytest.mark.soc("950")
def test_ifa_for_950():
    for case_i in range(4):
        set_qwen_common_config(case_950=1, b=16, s1=1, s2=8192)
        if case_i == 1:
            set_qwen_common_config(case_950=1, b=64, s1=1, s2=8192)
        if case_i == 2:
            set_qwen_common_config(case_950=1, b=64, s1=2, s2=8192)
        if case_i == 3:
            set_qwen_common_config(case_950=1, b=16, s1=1, s2=16384)
        atten_cfg, _ = get_common_config()

        assert atten_cfg.b == len(
            atten_cfg.actual_seq), f'{atten_cfg.b} {atten_cfg.actual_seq} B的大小必须和actual_seq长度相等'

        if atten_cfg.actual_seq.device.type != 'cpu':
            actual_seq_cpu = atten_cfg.actual_seq.cpu()
        else:
            actual_seq_cpu = atten_cfg.actual_seq

        assert all(x <= atten_cfg.s2 for x in actual_seq_cpu), "所有值都必须小于s2"
        ifa(atten_cfg, case_950=1)


@pytest.mark.soc("950", "910")
def test_ifa():
    for case_i in range(2):
        set_qwen_common_config(case_950=0, b=8, s1=1, s2=16384)
        if case_i == 1:
            set_qwen_common_config(case_950=0, b=16, s1=1, s2=16384)
        atten_cfg, _ = get_common_config()
        assert atten_cfg.b == len(
            atten_cfg.actual_seq), f'{atten_cfg.b} {atten_cfg.actual_seq} B的大小必须和actual_seq长度相等'

        if atten_cfg.actual_seq.device.type != 'cpu':
            actual_seq_cpu = atten_cfg.actual_seq.cpu()
        else:
            actual_seq_cpu = atten_cfg.actual_seq

        assert all(x <= atten_cfg.s2 for x in actual_seq_cpu), "所有值都必须小于s2"
        ifa(atten_cfg, case_950=0)

if __name__ == "__main__":
    test_ifa()
    if pypto.platform.npuarch == 'DAV_3510':
        test_ifa_for_950()
