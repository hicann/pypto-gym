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
from dataclasses import dataclass
import os
import logging
import math
import pytest
import torch
import torch_npu
import numpy as np
import pypto
from utils.compare import compare


@dataclass
class LightningIndexerConfigs:
    # graph optimization params
    # used for copy in merge graph
    mg_copy_in_upper_bound = 2 * 1024 * 1024
    # used for graph partition
    pg_upper_bound = 16 * 8192
    # l1 reuse merge params
    cube_l1_reuse_setting = {
        0: 16
    }
    # vector graph fuse optimization
    vec_merge_mode = 2
    vec_nbuffer_setting = {
        -1: 16
    }
    # tile params
    s1_tile = 2
    topk_tile = 8192
    # set the tileshape size in cube computation
    c1_tile = [64, 64, 128, 128, 128, 128] # (m, M), (k, K), (n, N)
    c2_tile = [128, 128, 64, 64, 128, 128] # (m, M), (k, K), (n, N)
    # matmul relu fuse params
    extend_param = {'scale': 1 / 2048.0, 'relu_type': pypto.ReLuType.RELU}


def gen_cache_tensor(k_tensor, block_table, block_num, block_size, b):
    logging.info("Entering into gen_cache_tensor!")
    # 获取输入张量的数据类型
    dtype = k_tensor.dtype
    b, s, n, d = k_tensor.shape

    # 初始化KV缓存张量
    k_cache = torch.zeros([block_num, block_size, n * d], dtype=dtype)
    k_tensor_bsh_raw = k_tensor.reshape(b, s, n * d)

    # 创建填充后的张量，长度扩展到块对齐
    k_tensor_bsh = torch.zeros(
        (b, block_table.shape[1] * block_size, n * d), dtype=dtype)

    # 将原始数据填充到新张量的前部
    k_tensor_bsh[:, : k_tensor_bsh_raw.shape[1], :] = k_tensor_bsh_raw[:, :, :]

    # 遍历每个样本和块进行缓存填充
    for b_idx in range(b):  # 遍历batch维度
        for block_idx, cache_block_idx in enumerate(block_table[b_idx]):  # 遍历块映射表
            block_offset = block_idx * block_size  # 计算当前块在序列中的起始位置
            # 如果cache_block_idx有效（非-1），则执行数据拷贝
            if cache_block_idx != -1:
                # 将数据从k_tensor_bsh复制到k_cache的指定块位置
                # 注意：block_offset到block_offset+block_size的切片对应当前块的数据
                k_cache[cache_block_idx, :, :] = k_tensor_bsh[b_idx,
                                                              block_offset: (block_offset + block_size), :]

    k_cache = k_cache.reshape(block_num, block_size, n, d)
    return k_cache


def gen_block_table(b, block_size, max_kv, act_kv):
    logging.info("Entering into gen_block_table!")

    # 初始化总块数和每个样本的块数列表
    block_num = 0
    block_num_each = []

    # 计算每个样本需要的块数（向上取整）并累加总块数
    for cur_s in act_kv:
        cur_block_num = math.ceil(cur_s / block_size)  # 当前样本需要的块数
        block_num_each.append(cur_block_num)
        block_num += cur_block_num

    shape_bt = [b, math.ceil(max_kv / block_size)]

    # 生成物理块索引列表，并随机排列以优化缓存效率
    block_idx_list = np.arange(0, block_num, 1)
    block_idx_list = np.random.permutation(block_idx_list).astype(np.int32)

    # 初始化块表，无效块标记为-1
    block_table = [-1] * shape_bt[1]
    block_table = np.tile(block_table, (shape_bt[0], 1)).astype(np.int32)

    block_table_bidx = 0
    block_idx = 0

    # 填充块表：将物理块索引分配给每个样本的有效块位置
    for cur_block in block_num_each:
        for j in range(cur_block):
            block_table[block_table_bidx][j] = block_idx_list[block_idx]
            block_idx += 1
        block_table_bidx += 1

    return block_num, block_table


def gen_data_for_compute(params, is_quant: bool):
    b = params.get("b")
    s1 = params.get("s1")
    n1 = params.get("n1")
    n2 = params.get("n2")
    d = params.get("d")
    dtype = params.get("dtype")
    s2 = params.get("s2")
    act_seq_len = params.get("act_seq")
    block_size = params.get("block_size")
    block_num = params.get("block_num")
    selected_count = params.get("selected_count")

    # 生成query张量 [b, s1, n1, d]
    query = torch.randn([b, s1, n1, d]).to(torch.int8)
    # 生成权重张量 [b, s1, n1]
    weights = torch.randn([b, s1, n1], dtype=dtype).to(torch.float16)

    # 生成key张量 [b, s2, n2, d]
    k_bsnd = torch.randn([b, s2, n2, d]).to(torch.int8)

    # 生成块表
    _, block_table_list = gen_block_table(b, block_size, s2, act_seq_len)
    block_table = torch.tensor(block_table_list, dtype=torch.int32)
    act_seq = torch.tensor(act_seq_len, dtype=torch.int32)

    # 初始化输出张量
    topk_res = torch.ones([b, s1, n2, selected_count], dtype=torch.int32)

    # 生成KV缓存张量
    key = gen_cache_tensor(k_bsnd, block_table_list, block_num, block_size, b)

    input_data_map = {}

    # 量化处理逻辑
    if is_quant:
        # 计算query的缩放因子（最大值/127，最小1e-3）
        q_scale = (query.abs().max(dim=-1, keepdim=True).values / 127).\
                    to(dtype=torch.float16).maximum(torch.tensor(1e-3))
        # 计算key的缩放因子
        k_scale = (key.abs().max(dim=-1, keepdim=True).values / 127).to(dtype=torch.float16).maximum(torch.tensor(1e-3))

        # 量化：归一化后四舍五入并裁剪到int8范围
        query = torch.round(query / q_scale).clip(-127, 127).to(dtype=torch.int8)
        key = torch.round(key / k_scale).clip(-127, 127).to(dtype=torch.int8)

        # 存储量化数据和缩放因子
        input_data_map["query"] = query
        input_data_map["key"] = key
        input_data_map["q_scale"] = q_scale
        input_data_map["k_scale"] = k_scale
    else:
        # 非量化模式直接存储原始数据
        input_data_map["query"] = query
        input_data_map["key"] = key

    input_data_map["weights"] = weights
    input_data_map["act_seq"] = act_seq
    input_data_map["block_table"] = block_table
    input_data_map["selected_count"] = selected_count
    input_data_map["topk_res"] = topk_res

    return input_data_map


def lightning_indexer_compute(input_data_map, params):
    # 提取参数
    block_size = params.get("block_size")
    selected_count = params.get("selected_count")
    b = params.get("b")
    s1 = params.get("s1")
    n1 = params.get("n1")
    d = params.get("d")
    block_num = params.get("block_num")
    max_block_num = params.get("max_block_num")

    query = input_data_map.get("query")
    key = input_data_map.get("key")
    q_scale = input_data_map.get("q_scale")
    k_scale = input_data_map.get("k_scale")
    weights = input_data_map.get("weights")
    act_seq = input_data_map.get("act_seq")
    block_table = input_data_map.get("block_table")

    topk_res = torch.zeros([b * s1, 1, selected_count], dtype=torch.int32)
    first_mm = torch.zeros(b * s1 * n1, max_block_num * block_size, dtype=torch.float16)
    mm_out = torch.zeros([b * s1 * 1, max_block_num * block_size], dtype=torch.float32)
    avoid_fp32_to_fp16_overflow_scale = 1.0 / 2048

    # 重塑张量形状
    query = query.reshape(b * s1 * n1, d)
    q_scale = q_scale.reshape(b * s1, 1, n1)
    key = key.reshape(block_num * block_size, d)
    k_scale = k_scale.reshape(block_num, block_size)
    weights = weights.reshape(b * s1, 1, n1)

    for b_idx in range(b):
        cur_seq = act_seq[b_idx]
        cur_block = (cur_seq + block_size - 1) // block_size
        # cur_qs的形状为(s1, 1, n1)
        cur_qs = q_scale[b_idx * s1:(b_idx + 1) * s1, :, :]
        # cur_w的形状为(s1, 1, n1)
        cur_w = weights[b_idx * s1:(b_idx + 1) * s1, :, :]
        w_scale = cur_qs * cur_w # (s1, 1, n1), fp16

        for block_idx in range(cur_block):
            # cur_q的形状为(s1 * n1, d)
            cur_q = query[b_idx * s1 * n1: (b_idx + 1) * s1 * n1, :]
            cur_block_idx = block_table[b_idx][block_idx]
            tail_seq = min(block_size, cur_seq - block_size * block_idx)
            # cur_k形状为(tail_seq, d)
            cur_k = key[cur_block_idx * block_size: (cur_block_idx * block_size + tail_seq), :]
            # 使用随路量化计算，qk_dot形状为(s1 * n1, tail_seq)
            qk_dot = torch.matmul(cur_q.to(torch.int32),
                                  cur_k.transpose(1, 0).to(torch.int32)).to(torch.float32).relu()
            qk_dot = qk_dot * avoid_fp32_to_fp16_overflow_scale
            qk_dot = qk_dot.to(torch.float16)
            first_mm[b_idx * s1 * n1:(b_idx + 1) * s1 * n1, block_idx * block_size:(block_idx * \
                                                                block_size + tail_seq)] = qk_dot
            qk_dot = qk_dot.reshape(s1, n1, tail_seq)

            # cur_ks形状为(1, tail_seq)
            cur_ks = k_scale[cur_block_idx:(cur_block_idx + 1), :tail_seq]
            cur_ks = cur_ks.to(torch.float32)
            # w_qk形状为(s1, 1, tail_seq)
            w_qk = torch.bmm(w_scale.to(torch.float32), qk_dot.to(torch.float32))
            w_qk = w_qk.reshape(s1, tail_seq)
            # k_res形状为(s1, tail_seq), fp32
            k_res = w_qk * cur_ks
            mm_out[b_idx * s1:(b_idx + 1) * s1, block_idx * block_size:(block_idx * block_size + tail_seq)] = k_res
    # Top-k选择，对每个批次 b_idx 和每个序列 s_idx 进行处理
    for b_idx in range(b):
        cur_seq = act_seq[b_idx]
        for s_idx in range(s1):
            # 计算当前序列的有效长度 eff_seq
            eff_seq = cur_seq - (s1 - s_idx - 1)
            # 从mm_out中提取当前序列的点积结果topk_in
            topk_in = mm_out[(b_idx * s1 + s_idx):(b_idx * s1 + s_idx + 1), :eff_seq] # (1, act_seq)
            # 如果有效长度小于 selected_count，则进行Top-k选择，并填充结果
            if (eff_seq < selected_count):
                cur_res, cur_idx = torch.topk(topk_in, k=eff_seq, dim=-1) # (1, eff_seq)
                pad_res = torch.full((1, selected_count - eff_seq), float("-inf"), dtype=torch.float32)
                pad_idx = torch.full((1, selected_count - eff_seq), -1, dtype=torch.int32)
                cur_res = torch.cat([cur_res, pad_res], dim=1)
                cur_idx = torch.cat([cur_idx, pad_idx], dim=1)
                topk_res[(b_idx * s1 + s_idx):(b_idx * s1 + s_idx + 1), :, :] = cur_idx.reshape(1, 1, selected_count)
            else:
                cur_res, cur_idx = torch.topk(topk_in, k=selected_count, dim=-1) # (1, selected_count)
                topk_res[(b_idx * s1 + s_idx):(b_idx * s1 + s_idx + 1), :, :] = cur_idx.reshape(1, 1, selected_count)

    return topk_res


def topk_idx_compare(t: torch.Tensor, t_ref: torch.Tensor, name, atol, error_count_threshold):
    part_result_dict = {}
    err_msg = None

    # 按元素遍历比较
    for idx, (act, exp) in enumerate(zip(t.flatten().tolist(), t_ref.flatten().tolist())):
        # 按误差阈值分组（每组包含error_count_threshold个元素）
        part_index = idx // error_count_threshold
        # 记录不匹配的索引
        if exp != act:
            if part_index not in part_result_dict:
                part_result_dict[part_index] = {
                    "exp": [],  # 预期索引列表
                    "act": []   # 实际索引列表
                }
            part_result_dict[part_index]["exp"].append(exp)
            part_result_dict[part_index]["act"].append(act)

    # 初始化精度状态
    precision = "PASS"

    # 遍历所有分组进行比较
    for idx_index in part_result_dict.keys():
        exp_list = part_result_dict[idx_index]["exp"]
        act_list = part_result_dict[idx_index]["act"]

        # 排序后比较
        exp_list.sort()
        act_list.sort()

        # 统计错误数量
        error_count = 0
        for topk_id in exp_list:
            if topk_id not in act_list:
                error_count += 1

        if error_count > int(error_count_threshold * atol):
            precision = "FAIL"
            err_msg = f"compare fail: {name}, error_count: {error_count}, \
                        error_count_threshold: {int(error_count_threshold * atol)}"
            break
    assert precision == "PASS", err_msg


def lightning_indexer(case_name: str) -> bool:
    from lightning_indexer_quant_impl import lightning_indexer_decode
    # 设置设备ID
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    # 基础参数配置
    n1, d = 64, 128  # query头数和维度
    n2 = 1  # key头数
    block_size = 128  # 块大小
    dtype = torch.float16  # 数据类型

    # 根据测试用例名称配置参数
    if case_name == "LightningIndexerSTest.lightning_indexer_quant_4_b_2_s1_64k_s2":
        b, s1 = 4, 2  # batch size和query序列长度
        act_seq = [64 * 1024, 971, 32 * 1024 + 101, 16 * 1024 - 1] # 每个样本的实际序列长度
    elif case_name == "LightningIndexerSTest.lightning_indexer_quant_8_b_2_s1_64k_s2":
        b, s1 = 8, 2
        act_seq = [32767, 32656, 384, 2000, 64 * 1024, 971, 32 * 1024 + 101, 129090]
    elif case_name == "LightningIndexerSTest.lightning_indexer_quant_4_b_2_s1_64k_s2_perf":
        b, s1 = 4, 2  # batch size和query序列长度
        act_seq = [64 * 1024] * b # 每个样本的实际序列长度
    else:
        logging.error("Fail to gen golden for Case(%s)", case_name)
        return False
    # 计算关键参数
    s2 = max(act_seq)  # 最大序列长度
    block_num = sum([(s + block_size - 1) // block_size for s in act_seq])  # 总块数
    max_block_num = (s2 + block_size - 1) // block_size  # 最大块数
    selected_count = 2048  # TopK选择数量

    # 构建参数字典
    params = {
        "b": b,
        "s1": s1,
        "n1": n1,
        "n2": n2,
        "d": d,
        "dtype": dtype,
        "s2": s2,
        "act_seq": act_seq,
        "block_size": block_size,
        "block_num": block_num,
        "max_block_num": max_block_num,
        "selected_count": selected_count
    }

    input_data_map = gen_data_for_compute(params, is_quant=True)

    idx_query_npu = input_data_map["query"].reshape(b * s1, n1, d).npu()
    idx_query_scale_npu = input_data_map["q_scale"].reshape(b * s1, n1).npu()
    idx_key_cache_npu = input_data_map["key"].npu()
    idx_key_scale_npu = input_data_map["k_scale"].reshape(block_num, block_size, 1).npu()
    idx_weight_npu = input_data_map["weights"].reshape(b * s1, n1).npu()
    act_seq_key_npu = input_data_map["act_seq"].npu()
    block_table_npu = input_data_map["block_table"].npu()

    topk_res_out = torch.zeros([b * s1, 1, selected_count], dtype=torch.int32)
    topk_res_npu = topk_res_out.npu()

    unroll_list = [128, 64, 32, 16, 8, 4, 1]
    configs = LightningIndexerConfigs()

    lightning_indexer_decode(idx_query_npu, idx_query_scale_npu, idx_key_cache_npu, idx_key_scale_npu,
                         idx_weight_npu, act_seq_key_npu, block_table_npu, topk_res_npu,
                         unroll_list, configs, selected_count)

    torch_npu.npu.synchronize()

    topk_res_golden = lightning_indexer_compute(input_data_map, params)
    topk_idx_compare(topk_res_npu.cpu(), topk_res_golden.cpu(), "topk_res", 5e-4, selected_count)

    return True


def test_lightning_indexer_topk_quant_4_b_2_s1_64k_s2():
    lightning_indexer("LightningIndexerSTest.lightning_indexer_quant_4_b_2_s1_64k_s2")


def test_lightning_indexer_topk_quant_8_b_2_s1_64k_s2():
    lightning_indexer("LightningIndexerSTest.lightning_indexer_quant_8_b_2_s1_64k_s2")


@pytest.mark.skip(reason="large shape")
def test_lightning_indexer_topk_quant_4_b_2_s1_64k_s2_perf():
    lightning_indexer("LightningIndexerSTest.lightning_indexer_quant_4_b_2_s1_64k_s2_perf")


if __name__ == "__main__":
    test_lightning_indexer_topk_quant_4_b_2_s1_64k_s2()
    test_lightning_indexer_topk_quant_8_b_2_s1_64k_s2()
    test_lightning_indexer_topk_quant_4_b_2_s1_64k_s2_perf()
