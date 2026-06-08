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
import logging
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
import numpy as np
from numpy.testing import assert_allclose

import sys
import os
_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')):
    _p = os.path.dirname(_p)
sys.path.insert(0, os.path.join(_p, 'src'))
sys.path.insert(0, os.path.join(_p, 'src', 'pypto_gym', 'ops', 'pypto_tile'))

from experimental.attention.incre_flash_attention_mla.utils import create_logger, get_device, compare
from experimental.attention.incre_flash_attention_mla.incre_flash_attention_mla_impl import (
    MlaConfig, AttentionTileConfig, incre_flash_attention_mla)


logger = create_logger(__name__)


def gen_inputs(mla_config: MlaConfig, device: str):
    dtype = torch.bfloat16
    query_shape = [mla_config.b, mla_config.s1, mla_config.n1, mla_config.q_d]  # BSND
    query = torch.empty(query_shape, dtype=dtype).uniform_(-1, 1).to(device=device)
    query_rope_shape = [mla_config.b, mla_config.s1, mla_config.n1, mla_config.q_rope_d]  # BSND
    query_rope = torch.empty(query_rope_shape, dtype=dtype).uniform_(-1, 1).to(device=device)

    max_num_blocks_per_query = math.ceil(mla_config.s2 / mla_config.block_size)
    kv_num_blocks = mla_config.b * max_num_blocks_per_query
    key_cache_shape = [kv_num_blocks, mla_config.n2, mla_config.block_size, mla_config.kv_d]  # PA_BnNBsD format
    key_cache = torch.empty(key_cache_shape, dtype=dtype).uniform_(-1, 1).to(device=device)
    value_cache = key_cache

    key_rope_cache_shape = [
    kv_num_blocks,
    mla_config.n2,
    mla_config.block_size,
     mla_config.k_rope_d]  # PA_BnNBsD format
    key_rope_cache = torch.empty(key_rope_cache_shape, dtype=dtype).uniform_(-1, 1).to(device=device)

    kv_actual_seqs = torch.tensor([mla_config.s2] * mla_config.b, dtype=torch.int32, device=device)

    block_table = gen_block_table(mla_config, kv_actual_seqs, device)

    key = kv_cache_concat(key_cache, kv_actual_seqs, block_table, mla_config, device)
    value = key

    key_rope = kv_cache_concat(key_rope_cache, kv_actual_seqs, block_table, mla_config, device)

    logger.info(f"query_shape: {query.shape}")
    logger.info(f"key_shape: {key.shape}")
    logger.info(f"value_shape: {value.shape}")
    logger.info(f"key_cache_shape: {key_cache.shape}")
    logger.info(f"value_cache_shape: {value_cache.shape}")
    logger.info(f"query_rope_shape: {query_rope.shape}")
    logger.info(f"key_rope_shape: {key_rope.shape}")
    logger.info(f"key_rope_cache_shape: {key_rope_cache.shape}")
    logger.info(f"block_table_shape: {block_table.shape}")
    logger.info(f"layout: {mla_config.layout}")
    logger.info(f"kv_actual_seqs_shape: {kv_actual_seqs.shape}")
    logger.info(f"kv_actual_seqs: {kv_actual_seqs}")
    logger.info(f"num_heads: {mla_config.n1}")
    logger.info(f"key_num_heads: {mla_config.n2}")
    logger.info(f"block_size: {mla_config.block_size}")

    mla_inputs = dict(
        query=query, key=key, value=value, key_cache=key_cache, 
        value_cache=value_cache, query_rope=query_rope, key_rope=key_rope,
        key_rope_cache=key_rope_cache, block_table=block_table, 
        kv_actual_seqs=kv_actual_seqs,
    )

    return mla_inputs


def gen_block_table(mla_config, kv_actual_seqs, device: str):
    """
    Generate a block table for paged KV cache.

    The block table maps logical block indices to physical block indices,
    enabling non-contiguous memory access patterns. This is essential for
    efficient memory management in autoregressive generation.

    Args:
        mla_config: Mla configuration
        kv_actual_seqs: Tensor containing kv actual sequence lengths for each batch
        device: Device to create tensors on (e.g., 'npu:0', 'cpu')

    Returns:
        torch: Block table tensor of shape [batch_size, max_blocks_per_query]
               Contains physical block indices, or -1 for invalid blocks
    """
    block_num_per_batch = []
    block_num = 0  # Total number of blocks needed

    block_size = mla_config.block_size
    block_table_batch = mla_config.b
    max_num_blocks_per_query = math.ceil(mla_config.s2 / block_size)
    block_table_shape = [block_table_batch, max_num_blocks_per_query]

    # Calculate number of blocks needed for each batch element
    for actual_seq in kv_actual_seqs:
        block_num_per_batch.append(math.ceil(actual_seq.item() / block_size))
        block_num += math.ceil(actual_seq.item() / block_size)

    # Create all block indices and randomly permute them
    # This simulates non-contiguous physical memory allocation
    block_idx_list = torch.arange(0, block_num, dtype=torch.int32)
    block_idx_list = block_idx_list[torch.randperm(block_idx_list.size(0))]

    # Create block table
    block_table = torch.full(block_table_shape, -1, dtype=torch.int32, device=device)
    block_idx = 0
    block_table_batch_idx = 0
    for idx in block_num_per_batch:
        for j in range(idx):
            block_table[block_table_batch_idx][j] = block_idx_list[block_idx]
            block_idx += 1
        block_table_batch_idx += 1
    return block_table


def kv_cache_concat(cache_tensor, kv_actual_seqs, block_table, mla_config, device: str):
    batch_size = mla_config.b
    kv_num_heads = mla_config.n2 
    kv_seqs_len = mla_config.s2
    d = cache_tensor.shape[3]  # BNSD

    block_size = mla_config.block_size
    dtype = cache_tensor.dtype

    # Initializes output tensors
    result = torch.zeros([batch_size, kv_num_heads, kv_seqs_len, d], dtype=dtype, device=device)

    # Reconstruct tensors by following block table
    for b_idx in range(batch_size):
        block_list = block_table[b_idx]
        temp_tensor = torch.zeros([1, kv_num_heads, kv_seqs_len, d], dtype=dtype, device=device)
        s_idx = 0

        # Copy blocks according to block table
        for _, block_idx in enumerate(block_list):
            if block_idx == -1:
                break
            start_idx = s_idx * block_size
            end_idx = (s_idx + 1) * block_size

            # Copy block from cache to temp tensor
            temp_tensor[:, :, start_idx:end_idx, :] = cache_tensor[block_idx:block_idx + 1, :, :, :]
            s_idx += 1

        result[b_idx:b_idx + 1, :, :, :] = temp_tensor
    return result


def ifa_mla_golden(query, key, value, query_rope, key_rope):
    b = query.shape[0]
    s1 = query.shape[1]
    n1 = query.shape[2]
    n2 = key.shape[1]
    s2 = key.shape[2]
    kv_d = key.shape[3]
    group_size = n1 // n2

    d_full = query.shape[3] + query_rope.shape[3]
    softmax_scale = d_full ** -0.5
    logger.info(f"softmax_scale: {softmax_scale}")
    logger.info(f"group_size: {group_size}, n1: {n1}, n2: {n2}")

    attention_out = torch.zeros([b, s1, n1, kv_d], dtype=query.dtype, device=query.device)

    for n2_idx in range(n2):
        q_group = query[:, :, n2_idx * group_size:(n2_idx + 1) * group_size, :]
        qr_group = query_rope[:, :, n2_idx * group_size:(n2_idx + 1) * group_size, :]
        q_full = torch.cat([q_group, qr_group], dim=-1)

        k_nope = key[:, n2_idx:n2_idx + 1, :, :]
        k_rope = key_rope[:, n2_idx:n2_idx + 1, :, :]
        k_full = torch.cat([k_nope, k_rope], dim=-1)

        v_head = value[:, n2_idx:n2_idx + 1, :, :]

        qk_mm_res = torch.matmul(q_full, k_full.transpose(-2, -1))
        logger.info(f"n2_idx={n2_idx}, qk_mm_res.shape: {qk_mm_res.shape}")
        qk_ele_res = qk_mm_res * softmax_scale
        softmax_res = F.softmax(qk_ele_res, dim=-1)
        logger.info(f"n2_idx={n2_idx}, softmax_res.shape: {softmax_res.shape}")
        attn_group = torch.matmul(softmax_res, v_head)
        logger.info(f"n2_idx={n2_idx}, attn_group.shape: {attn_group.shape}")

        attention_out[:, :, n2_idx * group_size:(n2_idx + 1) * group_size, :] = attn_group

    logger.info(f"attention_out.shape: {attention_out.shape}")
    return attention_out


def get_case_config(case_name):
    base_params = {"layout": "BSND", "block_size": 128, "d": 512, "dr": 64, "softmax_scale": 576 ** -0.5}
    if case_name.startswith("1b4k"):
        params = {"b": 1, "n1": 128, "s1": 1, "s2": 4 * 1024, "n2": 1}
    elif case_name.startswith("8b4k"):
        params = {"b": 8, "n1": 128, "s1": 1, "s2": 4 * 1024, "n2": 1}
    elif case_name.startswith("16b8k"):
        params = {"b": 16, "n1": 128, "s1": 1, "s2": 8 * 1024, "n2": 1}
    elif case_name.startswith("32b2k"):
        params = {"b": 32, "n1": 128, "s1": 1, "s2": 2 * 1024, "n2": 1}
    elif case_name.startswith("32b4k"):
        params = {"b": 32, "n1": 128, "s1": 1, "s2": 4 * 1024, "n2": 1}
    elif case_name.startswith("4b8k"):
        params = {"b": 4, "n1": 128, "s1": 2, "s2": 8 * 1024, "n2": 1}
    elif case_name.startswith("64b8k"):
        params = {"b": 64, "n1": 128, "s1": 2, "s2": 8 * 1024, "n2": 1}
    elif case_name.startswith("qs3_1b4k"):
        params = {"b": 1, "n1": 128, "s1": 3, "s2": 4 * 1024, "n2": 1}
    elif case_name.startswith("nkv2_qs3_1b4k"):
        params = {"b": 1, "n1": 128, "s1": 3, "s2": 4 * 1024, "n2": 2}
    elif case_name.startswith("dn128_qs3_1b4k"):
        params = {"b": 1, "n1": 128, "s1": 3, "s2": 4 * 1024, "n2": 2, "d": 128, "dr": 64, "softmax_scale": 192 ** -0.5}

    base_params.update(params)
    group = base_params["n1"] // base_params["n2"]

    case_config = MlaConfig(layout=base_params["layout"], b=base_params["b"], n1=base_params["n1"], 
                            s1=base_params["s1"], q_d=base_params["d"], q_rope_d=base_params["dr"], 
                            n2=base_params["n2"], s2=base_params["s2"], kv_d=base_params["d"], 
                            k_rope_d=base_params["dr"], block_size=base_params["block_size"], 
                            softmax_scale=base_params["softmax_scale"], group=group)
    return case_config


def get_tile_config(case_config):

    tile_config = AttentionTileConfig(
        g_tile=case_config.group,
        s2_tile=2048,
        v0_tile=[128, 576],
        c1_tile=[[128, 128], [128, 128], [128, 128]],
        v1_tile=[8, 2048],
        c2_tile=[[128, 128], [128, 128], [128, 128]],
        v2_tile=[64, 512],
        v2_update_tile=[32, 512]
    )
    return tile_config


def do_test_incre_flash_attention_mla(case_name):
    logger.info("*" * 60)
    logger.info(f"Run incre_flash_attention_mla {case_name} case")
    logger.info("*" * 60 + "\n")

    device = get_device()

    case_config = get_case_config(case_name)
    mla_inputs = gen_inputs(case_config, device)

    query = mla_inputs['query']
    key = mla_inputs['key']
    value = mla_inputs['value']
    query_rope = mla_inputs['query_rope']
    key_rope = mla_inputs['key_rope']

    mla_golden = ifa_mla_golden(query, key, value, query_rope, key_rope)

    tile_config = get_tile_config(case_config)
    pypto_kernel_inputs = dict(
        query=query,
        key=mla_inputs['key_cache'],
        value=mla_inputs['value_cache'],
        query_rope=query_rope,
        key_rope=mla_inputs['key_rope_cache'],
        kv_actual_seqs=mla_inputs['kv_actual_seqs'],
        block_table=mla_inputs['block_table'],
        kernel_config=case_config,
        tile_config=tile_config
    )
    pypto_atten_out = incre_flash_attention_mla(**pypto_kernel_inputs)
    compare(
    pypto_atten_out.cpu(),
    mla_golden.cpu(),
    "pypto_atten_out",
    atol=0.0001,
    rtol=0.0078125,
     max_error_ratio=0.005)
    print("[PRECISION_PASS]")


def test_incre_flash_attention_mla_1b4k():
    do_test_incre_flash_attention_mla("1b4k")


def test_incre_flash_attention_mla_8b4k():
    do_test_incre_flash_attention_mla("8b4k")


def test_incre_flash_attention_mla_16b8k():
    do_test_incre_flash_attention_mla("16b8k")


def test_incre_flash_attention_mla_32b2k():
    do_test_incre_flash_attention_mla("32b2k")


def test_incre_flash_attention_mla_32b4k():
    do_test_incre_flash_attention_mla("32b4k")


def test_incre_flash_attention_mla_4b8k():
    do_test_incre_flash_attention_mla("4b8k")


def test_incre_flash_attention_mla_64b8k():
    do_test_incre_flash_attention_mla("64b8k")


def test_incre_flash_attention_mla_qs3_1b4k():
    do_test_incre_flash_attention_mla("qs3_1b4k")


def test_incre_flash_attention_mla_nkv2_qs3_1b4k():
    do_test_incre_flash_attention_mla("nkv2_qs3_1b4k")


def test_incre_flash_attention_mla_dn128_qs3_1b4k():
    do_test_incre_flash_attention_mla("dn128_qs3_1b4k")


def main():
    logger.info("\n")
    logger.info("=" * 60)
    logger.info("PyPTO incre_flash_attention_mla example")
    logger.info("=" * 60 + "\n")

    test_incre_flash_attention_mla_32b4k()
    test_incre_flash_attention_mla_1b4k()
    test_incre_flash_attention_mla_8b4k()
    test_incre_flash_attention_mla_16b8k()
    test_incre_flash_attention_mla_32b2k()
    test_incre_flash_attention_mla_qs3_1b4k()
    test_incre_flash_attention_mla_nkv2_qs3_1b4k()
    test_incre_flash_attention_mla_dn128_qs3_1b4k()

    # prof case
    test_incre_flash_attention_mla_4b8k()
    test_incre_flash_attention_mla_64b8k()


if __name__ == "__main__":
    main()