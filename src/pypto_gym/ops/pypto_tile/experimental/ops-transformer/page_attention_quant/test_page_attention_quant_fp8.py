#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
GLM-4.5 Attention Module

This module implements the Attention mechanism for GLM-4.5 model, which uses
a paged memory management approach similar to operating systems to efficiently
handle variable-length sequences and dynamic batch sizes in attention computation.

Main Functions:
    - attention: Main attention function with Attention support
    - ifa_func: JIT compiled kernel implementing Flash Attention with paged KV cache
    - gen_block_table: Generate block mapping table for Attention
    - kv_cache_concat_bsnd: Convert paged KV cache to BSND format
"""
import os
import math
import enum
import torch
import torch_npu
import pytest
import numpy as np
from numpy.testing import assert_allclose
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph
from page_attention_quant_fp8_impl import ifa_func_kernel, set_qwen_common_config, get_common_config
import pypto

np.random.seed(0)
torch.manual_seed(0)
np.set_printoptions(formatter={'float': '{:.6f}'.format})


def check_cond(cond, msg):
    if not cond:
        raise ValueError(msg)


class TileOpFormat(enum.Enum):
    ND = "ND"
    NZ = "NZ"


def get_format(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("input type error")
    if not tensor.is_contiguous():
        raise TypeError("input type error")

    tile_op_format = TileOpFormat.ND.value
    if tensor.device.type == "npu":
        if torch_npu.get_npu_format(tensor) == 29:
            tile_op_format = TileOpFormat.NZ.value
    return tile_op_format


def check_args(
    query,
    key_cache,
    value_cache,
    block_tables,
    actual_seqs,
    attn_res
):
    check_cond(query.dim() == 3, "invalid query dim.")
    check_cond(get_format(query) == 'ND', "invalid query format.")
    check_cond(query.dtype == torch.float8_e4m3fn, "invalid query dtype.")
    check_cond(key_cache.dim() == 4, "invalid key_cache dim.")
    check_cond(get_format(key_cache) == 'ND', "invalid key_cache format.")
    check_cond(key_cache.dtype == torch.float8_e4m3fn, "invalid key_cache dtype.")
    check_cond(value_cache.dim() == 4, "invalid value_cache dim.")
    check_cond(get_format(value_cache) == 'ND', "invalid value_cache format.")
    check_cond(value_cache.dtype == torch.float8_e4m3fn, "invalid value_cache dtype.")
    check_cond(block_tables.dim() == 2, "invalid block_tables dim.")
    check_cond(get_format(block_tables) == 'ND', "invalid block_tables format.")
    check_cond(block_tables.dtype == torch.int32, "invalid block_tables dtype.")
    check_cond(actual_seqs.dim() == 1, "invalid actual_seqs dim.")
    check_cond(get_format(actual_seqs) == 'ND', "invalid actual_seqs format.")
    check_cond(actual_seqs.dtype == torch.int32, "invalid actual_seqs dtype.")
    check_cond(attn_res.dim() == 3, "invalid attn_res dim.")
    check_cond(get_format(attn_res) == 'ND', "invalid attn_res format.")
    check_cond(attn_res.dtype == torch.bfloat16, "invalid attn_res dtype.")


def gen_block_table(actual_seq_len, block_size, block_table_shape):
    block_num_per_batch = []
    block_num = 0

    # 处理 torch tensor 类型的 actual_seq_len
    if isinstance(actual_seq_len, torch.Tensor):
        # 如果 tensor 在 GPU/NPU 上，先移动到 CPU
        if actual_seq_len.device.type != 'cpu':
            actual_seq_len_cpu = actual_seq_len.cpu()
        else:
            actual_seq_len_cpu = actual_seq_len

        # 转换为 numpy 数组进行处理，或者直接使用 torch 操作
        for actual_seq in actual_seq_len_cpu:
            block_num_per_batch.append(math.ceil(actual_seq.item() / block_size))
            block_num += math.ceil(actual_seq.item() / block_size)
    else:
        # 保持对 list 的兼容
        for actual_seq in actual_seq_len:
            block_num_per_batch.append(math.ceil(actual_seq / block_size))
            block_num += math.ceil(actual_seq / block_size)

    # 使用 torch 替换 numpy
    block_idx_list = torch.arange(0, block_num, dtype=torch.int32)
    block_idx_list = block_idx_list[torch.randperm(block_idx_list.size(0))]

    # 创建 block_table 张量
    block_table = torch.full(block_table_shape, -1, dtype=torch.int32)
    block_idx = 0
    block_table_batch_idx = 0
    for idx in block_num_per_batch:
        for j in range(idx):
            block_table[block_table_batch_idx][j] = block_idx_list[block_idx]
            block_idx += 1
        block_table_batch_idx += 1
    return block_table


def kv_cache_concat_bsnd(kr_cache_out, kv_cache_out, k_scale, block_table, atten_config):
    b = atten_config.b
    n2 = atten_config.n2
    kv_lora_rank = atten_config.q_d
    rope_dim = atten_config.kv_d
    block_size = atten_config.block_size
    kv_cache_actual_seq = atten_config.actual_seq
    dtype = kr_cache_out.dtype
    scale_dtype = k_scale.dtype

    # 处理 torch tensor 类型的 kv_cache_actual_seq
    if isinstance(kv_cache_actual_seq, torch.Tensor):
        if kv_cache_actual_seq.device.type != 'cpu':
            kv_cache_actual_seq_cpu = kv_cache_actual_seq.cpu()
        else:
            kv_cache_actual_seq_cpu = kv_cache_actual_seq
        kv_max = (torch.max(kv_cache_actual_seq_cpu).item() + block_size - 1) // block_size * block_size
    else:
        kv_max = (max(kv_cache_actual_seq) + block_size - 1) // block_size * block_size

    # 使用 torch 创建张量，保持在同一设备上
    device = kr_cache_out.device
    k_cache = torch.zeros([b, kv_max, n2, kv_lora_rank], dtype=dtype, device=device)
    k_sclae_cache = torch.zeros([b, kv_max, n2, 1], dtype=scale_dtype, device=device)
    v_cache = torch.zeros([b, kv_max, n2, rope_dim], dtype=kv_cache_out.dtype, device=device)

    for b_idx in range(b):
        block_list = block_table[b_idx]
        kv_nope_temp_tensor = torch.zeros([1, kv_max, n2, kv_lora_rank], dtype=kv_cache_out.dtype, device=device)
        kv_rope_temp_tensor = torch.zeros([1, kv_max, n2, rope_dim], dtype=dtype, device=device)
        k_scale_temp_tensor = torch.zeros([1, kv_max, n2, 1], dtype=scale_dtype, device=device)
        s_idx = 0

        for _, block_idx in enumerate(block_list):
            if block_idx == -1:
                break
            # 使用 torch 的切片操作
            start_idx = s_idx * block_size
            end_idx = (s_idx + 1) * block_size

            kv_nope_temp_tensor[:, start_idx:end_idx, :, :] = kv_cache_out[block_idx:block_idx + 1, :, :, :]
            kv_rope_temp_tensor[:, start_idx:end_idx, :, :] = kr_cache_out[block_idx:block_idx + 1, :, :, :]
            k_scale_temp_tensor[:, start_idx:end_idx, :, :] = k_scale[block_idx:block_idx + 1, :, :, :]
            s_idx += 1

        v_cache[b_idx:b_idx + 1, :, :, :] = kv_nope_temp_tensor
        k_cache[b_idx:b_idx + 1, :, :, :] = kv_rope_temp_tensor
        k_sclae_cache[b_idx:b_idx + 1, :, :, :] = k_scale_temp_tensor

    return k_cache, v_cache, k_sclae_cache


def get_special_array(m, n):
    q_shape = [m, n]

    # 生成递增的行值
    base = np.arange(1, m + 1)  # 生成 [1, 2, ..., m]

    # 将 base 扩展到二维形状 [m, n]
    q = base[:, np.newaxis]  # 增加一个新维度，形状变为 [m, 1]
    q = np.broadcast_to(q, q_shape)  # 广播到目标形状 [m, n]

    # 转换为 float16 类型
    q = q.astype(np.float16)
    return q


def softmax(x, is_fp16=False):
    # 使用 torch 的 softmax 实现
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


def quant_fp8e4m3_per_token(x: torch.Tensor):
    # perblock
    x_fp32 = x.to(torch.float32)
    max_value = torch.amax(torch.abs(x_fp32), dim=-1, keepdim=True)
    scale_quant = 448.0 / max_value
    y_fp32 = x_fp32 * scale_quant
    y_fp32 = y_fp32.view(x.shape)
    y_fp8e4m3 = y_fp32.to(torch.float8_e4m3fn)
    scale_dequant = 1.0 / scale_quant
    # shape是(b, s, n, d) fp8e4m3, (b, s, n, 1) fp32
    return y_fp8e4m3, scale_dequant


def quant_fp8e4m3_per_token_key(x: torch.Tensor):
    # perblock
    x_fp32 = x.to(torch.float32)
    max_value = torch.amax(torch.abs(x_fp32), dim=-1, keepdim=True)
    scale_quant = 448.0 / max_value
    y_fp32 = x_fp32 * scale_quant
    y_fp32 = y_fp32.view(x.shape)
    y_fp8e4m3 = y_fp32.to(torch.float8_e4m3fn)
    scale_dequant = 1.0 / scale_quant
    # shape是(b, s, n, d) fp8e4m3, (b, s, n, 1) fp32
    return y_fp8e4m3, scale_dequant


def quant_fp8e4m3_per_channel_value(x: torch.Tensor):
    # perblock
    x_fp32 = x.to(torch.float32)
    max_value = torch.amax(torch.abs(x_fp32), dim=1, keepdim=True)
    scale_quant = 448.0 / max_value
    y_fp32 = x_fp32 * scale_quant
    y_fp32 = y_fp32.view(x.shape)
    y_fp8e4m3 = y_fp32.to(torch.float8_e4m3fn)
    scale_dequant = 1.0 / scale_quant
    # shape是(b, s, n, d) fp8e4m3, (b, 1, n, d) fp32
    return y_fp8e4m3, scale_dequant


def fp8_bsnd_to_pa_format(tensor_bsnd, block_table, actual_seq, block_size):
    """
    转换为最终 PA 格式：shape [total_blocks, 128, n, d]
    每个全局 block 存储 128 个 token 位置，有效 token 填充，无效位置补 0
    """
    b, s, n, d = tensor_bsnd.shape
    num_blocks_per_batch = s // block_size
    total_blocks = num_blocks_per_batch * b

    # 2. 初始化 PA 张量（核心：shape [total_blocks, 128, n, d]，全 0 填充）
    pa_tensor = torch.zeros((total_blocks, block_size, n, d), dtype=tensor_bsnd.dtype, device="cpu")

    # 3. 逐 Batch + 逐逻辑 Block 填充
    for batch_idx in range(b):
        curr_actual_seq = actual_seq[batch_idx].item()
        if curr_actual_seq <= 0:
            continue
        # 截断有效序列长度到总长度以内
        curr_actual_seq = min(curr_actual_seq, s)

        # 当前 batch 的原始数据和 Block Table
        curr_tokens = tensor_bsnd[batch_idx]  # [s, n, d]
        curr_global_block_ids = block_table[batch_idx]  # [num_blocks_per_batch]：每个逻辑 block 对应的全局 ID

        # 按 128 切分逻辑 block，逐个处理
        for logical_block_idx in range(num_blocks_per_batch):
            # 步骤 1：获取当前逻辑 block 对应的全局 PA block ID（唯一）
            global_pa_block_id = curr_global_block_ids[logical_block_idx].item()
            # 校验全局 ID 合法性
            if global_pa_block_id < 0 or global_pa_block_id >= total_blocks:
                raise ValueError(f"全局 Block ID {global_pa_block_id} 超出范围 [0, {total_blocks-1}]")

            # 步骤 2：计算当前逻辑 block 的 token 范围
            token_start = logical_block_idx * block_size  # 逻辑 block 起始 token
            token_end = min((logical_block_idx + 1) * block_size, curr_actual_seq)  # 结束 token（不超过有效长度）
            # 无有效 token，跳过
            if token_start >= token_end:
                continue

            # 步骤 3：填充当前逻辑 block 的有效 token 到全局 PA block 中
            # token_offset：token 在 block 内的偏移（0~127）
            for token_in_block_offset in range(token_end - token_start):
                src_token_idx = token_start + token_in_block_offset  # 原始张量的 token 索引
                # 填充：PA[全局block_id, 块内偏移, :, :] = 原始token数据
                pa_tensor[global_pa_block_id, token_in_block_offset] = curr_tokens[src_token_idx]
    return pa_tensor


@allow_in_graph
def attention(
    query: torch.Tensor,
    query_scale: torch.Tensor,
    key_cache: torch.Tensor,
    key_cache_scale: torch.Tensor,
    value_cache: torch.Tensor,
    value_cache_sclae: torch.Tensor,
    block_tables: torch.Tensor,
    actual_seqs: torch.Tensor,
    attn_res: torch.Tensor
) -> None:
    """
    Main attention function with Attention support.

    This function implements scaled dot-product attention using Attention
    mechanism, which efficiently handles variable-length sequences and dynamic
    batch sizes by managing KV cache in non-contiguous blocks.

    Args:
        query: Query tensor with shape [num_tokens, num_head, head_size]
        key_cache: Key cache tensor with shape [num_blocks, block_size, kv_head_num, head_size]
        value_cache: Value cache tensor with shape [num_blocks, block_size, kv_head_num, head_size]
        block_tables: Block mapping table with shape [batch_size, max_num_blocks_per_query]
        actual_seqs: Actual sequence lengths with shape [batch_size]
        attn_res: Output attention tensor with shape [num_tokens, num_head, head_size]

    Note:
        This function is decorated with @allow_in_graph to enable integration
        with PyTorch's compilation graph.
    """
    if isinstance(query, FakeTensor):
        return
    check_args(
        query,
        key_cache,
        value_cache,
        block_tables,
        actual_seqs,
        attn_res
    )

    inputs = [query, query_scale, key_cache, key_cache_scale, value_cache, value_cache_sclae, block_tables,
              actual_seqs, attn_res]
    ifa_func_kernel(*inputs)


def ifa(atten_cfg):
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

    # 获取 torch tensor 类型的 actual_seq
    kv_cache_actual_seq = atten_cfg.actual_seq

    q_shape = [b * s1, nq, d]
    kv_shape = [atten_cfg.kv_num_blocks, block_size, nkv, d]
    block_table_shape = [atten_cfg.block_table_batch, max_num_blocks_per_query]

    # 使用 torch 生成数据
    device = f'npu:{device_id}'
    q = torch.empty(q_shape, dtype=torch_dtype).uniform_(-1, 1).to(device=device)
    q_fp8_e4m3, q_scale = quant_fp8e4m3_per_token(q)
    k = torch.empty(kv_shape, dtype=torch_dtype).uniform_(-1, 1).to(device=device)
    k_fp8_e4m3, k_scale = quant_fp8e4m3_per_token_key(k)
    v = torch.empty(kv_shape, dtype=torch_dtype).uniform_(-1, 1).to(device=device)
    attention_output = torch.zeros(q_shape, dtype=torch_dtype).to(device=device)

    # 2. 生成block table - 传入 torch tensor
    block_table = gen_block_table(kv_cache_actual_seq, block_size, block_table_shape)

    # 3. 根据block table 将pa格式的数据转换成
    k_cache_bsnd, v_cache_bsnd, k_sclae_bsnd = kv_cache_concat_bsnd(k_fp8_e4m3, v, k_scale, block_table, atten_cfg)
    v_fp8_e4m3_bsnd, v_scale = quant_fp8e4m3_per_channel_value(v_cache_bsnd)
    v_fp8_e4m3 = fp8_bsnd_to_pa_format(v_fp8_e4m3_bsnd.cpu(), block_table.cpu(), kv_cache_actual_seq.cpu(),
                                       atten_cfg.block_size)
    v_scale = v_scale.reshape(b * 1, nkv, d)
    k_cache_bsnd_cpu = k_cache_bsnd.cpu()
    v_cache_bsnd_cpu = v_fp8_e4m3_bsnd.cpu()
    q_fp8_e4m3 = q_fp8_e4m3.cpu()

    for i in range(b):
        for j in range(s1):
            for n2_idx in range(nkv):
                # 从 torch tensor 获取值
                kv_seq_len = kv_cache_actual_seq[i].item()  # 使用 .item() 获取标量值
                seq_len = kv_seq_len - s1 + 1 + j
                q_bs = q_fp8_e4m3[i * s1 + j]
                q_scale_b = q_scale[i]
                k_bs = k_cache_bsnd_cpu[i, :seq_len, n2_idx:n2_idx + 1].reshape(seq_len, d)
                k_value_bs = k_sclae_bsnd[i, :seq_len, n2_idx:n2_idx + 1].reshape(seq_len, 1)
                v_bs = v_cache_bsnd_cpu[i, :seq_len, n2_idx:n2_idx + 1].reshape(seq_len, d)
                v_bs_scale = v_scale[i, n2_idx:n2_idx + 1].reshape(1, d)
                # MM1: 矩阵乘法
                # 1,nq, d  -> n_q,d @ d, s2_actual_len
                qk_bmm_res = torch.matmul(q_bs.to(torch.float32), k_bs.to(torch.float32).transpose(1, 0))
                qk_bmm_res_npu = qk_bmm_res.npu()
                q_scale_b_npu = q_scale_b.npu()
                k_value_bs_npu = k_value_bs.npu()
                qk_bmm_res_npu = qk_bmm_res_npu * q_scale_b_npu
                qk_bmm_res_npu = qk_bmm_res_npu * k_value_bs_npu.transpose(1, 0)
                qk_ele_res = qk_bmm_res_npu * atten_cfg.softmax_scale
                # Softmax计算
                softmax_res, _, _ = softmax(qk_ele_res, False)
                softmax_res_fp8_e4m3, softmax_res_scale = quant_fp8e4m3_per_token(softmax_res)
                softmax_res_cpu = softmax_res_fp8_e4m3.cpu()
                # MM2: 矩阵乘法
                bmm2_res = torch.matmul(softmax_res_cpu.to(torch.float32), v_bs.to(torch.float32))
                bmm2_res_npu = bmm2_res.npu()
                bmm2_res_npu = bmm2_res_npu * softmax_res_scale
                bmm2_res_npu = bmm2_res_npu * v_bs_scale
                bmm2_res_npu = bmm2_res_npu.to(torch_dtype)
                bmm2_res_cpu = bmm2_res_npu.cpu()
                # 存储结果
                attention_output[i * s1 + j] = bmm2_res_cpu

    # 4. 准备测试数据 - 直接使用 torch 张量
    block_table_torch = block_table.to(dtype=torch.int32, device=device)
    act_seq_torch = kv_cache_actual_seq.to(dtype=torch.int32, device=device)
    q_fp8_e4m3 = q_fp8_e4m3.to(device=device)
    q_scale = q_scale.to(device=device)
    k_fp8_e4m3 = k_fp8_e4m3.to(device=device)
    k_scale = k_scale.to(device=device)
    v_fp8_e4m3 = v_fp8_e4m3.to(device=device)
    v_scale = v_scale.to(device=device)
    out_torch = torch.zeros(q_shape, dtype=torch_dtype).to(device=device)

    inputs = [
        q_fp8_e4m3,
        q_scale,
        k_fp8_e4m3,
        k_scale,
        v_fp8_e4m3,
        v_scale,
        block_table_torch,
        act_seq_torch,
        out_torch
    ]
    # 5. 执行kernel并获取结果
    attention(*inputs)

    # 6. 与PyTorch参考实现对比
    assert_allclose(np.array(attention_output.cpu().flatten().tolist()),
                    np.array(out_torch.cpu().flatten().tolist()),
                    rtol=0.0078125, atol=0.001)


def ifa_test_impl(b=16, s1=1, s2=8192):
    # 1. 设置参数
    set_qwen_common_config(b=b, s1=s1, s2=s2)
    atten_cfg, _ = get_common_config()

    # 检查 B 的大小和 actual_seq 长度是否相等
    check_cond(atten_cfg.b == len(atten_cfg.actual_seq), \
               f'{atten_cfg.b} {atten_cfg.actual_seq} B的大小必须和actual_seq长度相等')

    # 检查所有值是否都小于 s2
    if atten_cfg.actual_seq.device.type != 'cpu':
        actual_seq_cpu = atten_cfg.actual_seq.cpu()
    else:
        actual_seq_cpu = atten_cfg.actual_seq

    check_cond(all(x <= atten_cfg.s2 for x in actual_seq_cpu), "所有值都必须小于s2")
    ifa(atten_cfg)


@pytest.mark.soc("950")
def test_ifa_01():
    ifa_test_impl(b=16, s1=1, s2=8192)


@pytest.mark.soc("950")
@pytest.mark.skip(reason="large test case")
def test_ifa_02():
    ifa_test_impl(b=8, s1=1, s2=8192)


if __name__ == "__main__":
    if pypto.platform.npuarch == 'DAV_3510':
        # 950上板
        test_ifa_01()
        test_ifa_02()