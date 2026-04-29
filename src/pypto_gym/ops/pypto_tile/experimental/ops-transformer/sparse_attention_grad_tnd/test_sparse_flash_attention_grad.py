#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Sparse Flash Attention Grad - Test & Golden (TND format, nope/rope split)

测试SFA反向算子：Q/K拆分为nope和rope，V仅nope。
TND格式输入，actual_seq_qlen/actual_seq_kvlen前缀和格式。
t_1, batch 动态轴。
"""
import os
import sys
import math
import random
import logging
import torch
import torch_npu
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'deepseek_v32_exp'))
from utils.compare import compare
import pytest

# ============================================================
# Golden: PyTorch 参考实现 (per-batch, n_2=1)
# ============================================================


def generate_random_sequence(length, total_sum):
    # random.seed(42)  # 你可以选择任意整数作为种子
    # 确保每个元素至少为1
    sequence = [1] * length
    remaining = total_sum - length  # 剩余部分需要随机分配

    # 随机分配剩余部分到各个元素
    for i in range(length):
        # 确保 remaining 至少为1，避免 randint 出现空范围
        if remaining <= 0:
            break
        # 随机分配一个值，最大不超过 remaining
        if remaining <= total_sum // 8:
            max_ = remaining
        else:
            max_ = remaining // 4

        add = random.randint(1, max_)
        sequence[i] += add
        remaining -= add

    # 计算前缀和序列
    prefix_sums = []
    current_sum = 0
    for num in sequence:
        current_sum += num
        prefix_sums.append(current_sum)

    # 打印原序列和前缀和序列
    logging.info(f"原序列：{sequence}")
    logging.info(f"前缀和序列：{prefix_sums}")
    assert sum(sequence) == total_sum, "总和不正确"

    return sequence


def gen_uniform_data(data_shape, min_value, max_value, dtype):
    """
    PyTorch版本的均匀分布数据生成，与NumPy版本行为完全一致
    严格保持 [min_value, max_value) 左闭右开区间特性
    """
    # 特殊情况：全零张量
    if min_value == 0 and max_value == 0:
        return torch.zeros(data_shape, dtype=dtype)
    # 布尔类型处理：等概率生成True/False
    if dtype == torch.bool:
        # 生成[0,2)的整数，转换为bool即等概率True/False
        return torch.randint(0, 2, data_shape, dtype=dtype)
    # 浮点类型：[min_value, max_value)
    if torch.is_floating_point(torch.tensor(0, dtype=dtype)):
        # torch.rand生成[0,1)，缩放后得到[min_value, max_value)
        return min_value + (max_value - min_value) * torch.rand(data_shape, dtype=dtype)
    # 整数类型：[min_value, max_value)
    else:
        # torch.randint的high参数为开区间，直接对应[min_value, max_value)
        return torch.randint(low=min_value, high=max_value, size=data_shape, dtype=dtype)


def sfa_grad_golden_tnd(q_nope_tnd, q_pe_tnd, k_nope_tnd, k_pe_tnd, value_tnd,
                        sparse_indices_tnd, d_out_tnd, out_tnd,
                        sm_max_tnd, sm_sum_tnd,
                        actual_q_lens, actual_kv_lens, scale_value):
    """
    TND格式的golden实现。

    Args:
        q_nope_tnd:       (t_1, n_1, d) BF16
        q_pe_tnd:         (t_1, n_1, dr) BF16
        k_nope_tnd:       (t_2, n_2, d) BF16
        k_pe_tnd:         (t_2, n_2, dr) BF16
        value_tnd:        (t_2, n_2, d) BF16
        sparse_indices_tnd: (t_1, n_2, k) INT64, t_2-level global indices
        d_out_tnd:        (t_1, n_1, d) BF16
        out_tnd:          (t_1, n_1, d) BF16
        sm_max_tnd:       (n_2, t_1, group) FP32
        sm_sum_tnd:       (n_2, t_1, group) FP32
        actual_q_lens:    list of int, per-batch q lengths
        actual_kv_lens:   list of int, per-batch kv lengths
        scale_value:      float

    Returns:
        dq_nope: (t_1, n_1, d) BF16
        dq_pe:   (t_1, n_1, dr) BF16
        dk_nope: (t_2, n_2, d) BF16
        dk_pe:   (t_2, n_2, dr) BF16
        dv:      (t_2, n_2, d) BF16
    """
    t_1, n_1, d = q_nope_tnd.shape
    dr = q_pe_tnd.shape[-1]
    t_2, n_2, _ = k_nope_tnd.shape
    k = sparse_indices_tnd.shape[-1]
    group = n_1 // n_2
    batch = len(actual_q_lens)

    dq_nope = torch.zeros(t_1, n_1, d, dtype=torch.float32)
    dq_pe = torch.zeros(t_1, n_1, dr, dtype=torch.float32)
    dk_nope = torch.zeros(t_2, n_2, d, dtype=torch.float32)
    dk_pe = torch.zeros(t_2, n_2, dr, dtype=torch.float32)
    dv = torch.zeros(t_2, n_2, d, dtype=torch.float32)

    q_offset = 0
    kv_offset = 0
    for b in range(batch):
        s_count = actual_q_lens[b]
        kv_len = actual_kv_lens[b]

        for s in range(s_count):
            t_idx = q_offset + s
            for n2 in range(n_2):
                slc_kv_len = min(max(kv_len - s_count + 1 + s, 0), k)
                indices = sparse_indices_tnd[t_idx, n2, :slc_kv_len].long()  # (k,) t_2-level

                # Gather from k/v
                sel_k_nope = k_nope_tnd[indices, n2, :].float()  # (k, d)
                sel_k_pe = k_pe_tnd[indices, n2, :].float()       # (k, dr)
                sel_k = torch.cat([sel_k_nope, sel_k_pe], dim=-1)  # (k, d+dr)
                sel_v = value_tnd[indices, n2, :].float()          # (k, d)

                for g in range(group):
                    n1 = n2 * group + g
                    q_nope_vec = q_nope_tnd[t_idx, n1, :].float()
                    q_pe_vec = q_pe_tnd[t_idx, n1, :].float()
                    q_vec = torch.cat([q_nope_vec, q_pe_vec]).to(torch.bfloat16).to(torch.float32)

                    do_vec = d_out_tnd[t_idx, n1, :].float()
                    o_vec = out_tnd[t_idx, n1, :].float()

                    mi = sm_max_tnd[n2, t_idx, g].float()
                    li = sm_sum_tnd[n2, t_idx, g].float()

                    s_scores = (q_vec.unsqueeze(0) @ sel_k.T).squeeze(0) * scale_value
                    s_shifted = s_scores - mi
                    p_vec = (torch.exp(s_shifted) / li)

                    dp_vec = (do_vec.unsqueeze(0) @ sel_v.T).squeeze(0)
                    dv_local = p_vec.unsqueeze(1).to(torch.bfloat16).to(torch.float32) * do_vec.unsqueeze(0)

                    d_val = (do_vec * o_vec).sum()
                    ds_vec = (p_vec * (dp_vec - d_val)).to(torch.bfloat16).to(torch.float32)

                    dq_full = (ds_vec.unsqueeze(0) @ sel_k).squeeze(0) * scale_value
                    dq_nope[t_idx, n1, :] += dq_full[:d]
                    dq_pe[t_idx, n1, :] += dq_full[d:]

                    dk_full = (ds_vec.unsqueeze(1) * q_vec.unsqueeze(0)) * scale_value
                    for ki in range(slc_kv_len):
                        idx = indices[ki]
                        dk_nope[idx, n2, :] += dk_full[ki, :d]
                        dk_pe[idx, n2, :] += dk_full[ki, d:]
                        dv[idx, n2, :] += dv_local[ki]

        q_offset += s_count
        kv_offset += kv_len

    dtype = q_nope_tnd.dtype
    return (dq_nope.to(dtype), dq_pe.to(dtype),
            (dk_nope + dv).to(dtype), dk_pe.to(dtype), dv.to(dtype))


def sfa_forward_golden_tnd(q_nope_tnd, q_pe_tnd, k_nope_tnd, k_pe_tnd, value_tnd,
                           sparse_indices_tnd, actual_q_lens, actual_kv_lens, scale_value):
    """
    TND格式的前向golden。

    Returns:
        out:         (t_1, n_1, d) BF16
        softmax_max: (n_2, t_1, group) FP32
        softmax_sum: (n_2, t_1, group) FP32
    """
    t_1, n_1, d = q_nope_tnd.shape
    dr = q_pe_tnd.shape[-1]
    t_2, n_2, _ = k_nope_tnd.shape
    k = sparse_indices_tnd.shape[-1]
    group = n_1 // n_2
    batch = len(actual_q_lens)

    out = torch.zeros(t_1, n_1, d, dtype=q_nope_tnd.dtype)
    softmax_max = torch.zeros(n_2, t_1, group, dtype=torch.float32)
    softmax_sum = torch.zeros(n_2, t_1, group, dtype=torch.float32)

    q_offset = 0
    for b in range(batch):
        s_len = actual_q_lens[b]
        kv_len = actual_kv_lens[b]
        for s in range(s_len):
            t_idx = q_offset + s
            for n2 in range(n_2):
                slc_kv_len = min(max(kv_len - s_len + 1 + s, 0), k)
                indices = sparse_indices_tnd[t_idx, n2, :slc_kv_len].long()
                sel_k_nope = k_nope_tnd[indices, n2, :].float()
                sel_k_pe = k_pe_tnd[indices, n2, :].float()
                sel_k = torch.cat([sel_k_nope, sel_k_pe], dim=-1)
                sel_v = value_tnd[indices, n2, :].float()
                for g in range(group):
                    n1 = n2 * group + g
                    q_nope_vec = q_nope_tnd[t_idx, n1, :].float()
                    q_pe_vec = q_pe_tnd[t_idx, n1, :].float()
                    q_vec = torch.cat([q_nope_vec, q_pe_vec])

                    s_scores = (q_vec.unsqueeze(0) @ sel_k.T).squeeze(0) * scale_value
                    mi = s_scores.max()
                    s_shifted = s_scores - mi
                    exp_s = torch.exp(s_shifted)
                    li = exp_s.sum()
                    p = exp_s / li

                    o = (p.unsqueeze(0) @ sel_v).squeeze(0)
                    out[t_idx, n1, :] = o.to(q_nope_tnd.dtype)
                    softmax_max[n2, t_idx, g] = mi
                    softmax_sum[n2, t_idx, g] = li

        q_offset += s_len

    return out, softmax_max, softmax_sum


# ============================================================
# 数据生成 (TND format)
# ============================================================


def gen_test_data_tnd(actual_q_lens, actual_kv_lens, n_1, n_2, d, dr, k,
                      dtype=torch.bfloat16, seed=42):
    """
    生成TND格式的SFA反向测试数据。

    Args:
        actual_q_lens:  list[int], per-batch q token counts
        actual_kv_lens: list[int], per-batch kv token counts
        n_1, n_2, d, dr, k: static dims
    """
    torch.manual_seed(seed)
    batch = len(actual_q_lens)
    group = n_1 // n_2
    d_full = d + dr
    scale_value = 1.0 / math.sqrt(d_full)

    t_1 = sum(actual_q_lens)
    t_2 = sum(actual_kv_lens)

    # TND tensors
    q_nope = torch.randn(t_1, n_1, d, dtype=dtype) * 0.5 + 0.5
    q_pe = torch.randn(t_1, n_1, dr, dtype=dtype) * 0.5 + 0.5
    k_nope = torch.randn(t_2, n_2, d, dtype=dtype) * 0.5 + 0.5
    k_pe = torch.randn(t_2, n_2, dr, dtype=dtype) * 0.5 + 0.5
    value = k_nope.clone()
    d_out = torch.randn(t_1, n_1, d, dtype=dtype) * 0.5 + 0.5

    # sparse_indices: (t_1, n_2, k), t_2-level global indices
    sparse_indices = torch.zeros(t_1, n_2, k, dtype=torch.int64) - 1
    q_offset = 0
    for b in range(batch):
        s_count = actual_q_lens[b]
        kv_len = actual_kv_lens[b]
        for s in range(s_count):
            t_idx = q_offset + s
            for n2 in range(n_2):
                slc_kv_len = min(max(kv_len - s_count + 1 + s, 0), k)
                perm = torch.randperm(slc_kv_len)
                sparse_indices[t_idx, n2, :slc_kv_len] = perm
        q_offset += s_count

    actual_seq_qlen = torch.tensor(actual_q_lens, dtype=torch.int32).cumsum(0).to(torch.int32)
    actual_seq_kvlen = torch.tensor(actual_kv_lens, dtype=torch.int32).cumsum(0).to(torch.int32)

    # Forward golden (for out, sm_max, sm_sum)
    out, sm_max, sm_sum = sfa_forward_golden_tnd(
        q_nope, q_pe, k_nope, k_pe, value,
        sparse_indices, actual_q_lens, actual_kv_lens, scale_value
    )

    logging.info("sfa_forward_golden_tnd success!!")

    # Grad golden
    dq_nope_g, dq_pe_g, dk_nope_g, dk_pe_g, dv_g = sfa_grad_golden_tnd(
        q_nope, q_pe, k_nope, k_pe, value,
        sparse_indices, d_out, out,
        sm_max, sm_sum,
        actual_q_lens, actual_kv_lens, scale_value
    )

    return {
        'q_nope': q_nope, 'q_pe': q_pe,
        'k_nope': k_nope, 'k_pe': k_pe,
        'value': value,
        'sparse_indices': sparse_indices,
        'd_out': d_out, 'out': out,
        'sm_max': sm_max, 'sm_sum': sm_sum,
        'actual_seq_qlen': actual_seq_qlen,
        'actual_seq_kvlen': actual_seq_kvlen,
        'scale_value': scale_value,
        't_1': t_1, 't_2': t_2, 'batch': batch,
        'dq_nope_golden': dq_nope_g, 'dq_pe_golden': dq_pe_g,
        'dk_nope_golden': dk_nope_g, 'dk_pe_golden': dk_pe_g,
        'dv_golden': dv_g,
    }


# ============================================================
# 测试入口
# ============================================================


def do_test_sfa_grad_npu_eager(case_name, actual_q_lens, actual_kv_lens,
                      n_1, n_2, d, dr, k, seed=42):
    """Run SFA grad test on NPU (TND format)."""
    logging.info("=" * 50)
    logging.info(f"Test: SFA grad NPU ({case_name})")
    logging.info("=" * 50)

    torch.npu.set_device(int(os.environ.get('TILE_FWK_DEVICE_ID', 0)))

    data = gen_test_data_tnd(actual_q_lens, actual_kv_lens, n_1, n_2, d, dr, k, seed=seed)
    scale_value = data['scale_value']
    batch = data['batch']
    t_1 = data['t_1']
    t_2 = data['t_2']
    group = n_1 // n_2
    logging.info(f"  batch={batch}, t_1={t_1}, t_2={t_2}, n_1={n_1}, n_2={n_2}, d={d}, dr={dr}, k={k}, group={group}")
    logging.info(f"  actual_q_lens={actual_q_lens}, actual_kv_lens={actual_kv_lens}")
    logging.info(f"  scale_value={scale_value:.6f}")

    from sparse_flash_attention_grad_impl import npu_sfa_sparse_attention_grad

    # Move to NPU - all in TND 3D format
    q_nope_npu = data['q_nope'].npu()
    q_pe_npu = data['q_pe'].npu()
    k_nope_npu = data['k_nope'].npu()
    k_pe_npu = data['k_pe'].npu()
    value_npu = data['value'].npu()
    sparse_idx_npu = data['sparse_indices'].to(torch.int32).npu()
    d_out_npu = data['d_out'].npu()
    out_npu = data['out'].npu()
    sm_max_npu = data['sm_max'].npu()
    sm_sum_npu = data['sm_sum'].npu()
    actual_seq_qlen_npu = data['actual_seq_qlen'].npu()
    actual_seq_kvlen_npu = data['actual_seq_kvlen'].npu()

    logging.info("Running eager eager eager PyPTO SFA grad (TND, nope/rope split, dynamic) on NPU...")

    dq_nope_out_npu, dq_pe_out_npu, dk_nope_out_npu, dk_pe_out_npu, dv_out_npu = npu_sfa_sparse_attention_grad(
        q_nope_npu, q_pe_npu, k_nope_npu, k_pe_npu, value_npu, sparse_idx_npu, d_out_npu, out_npu,
        sm_max_npu, sm_sum_npu, actual_seq_qlen_npu, actual_seq_kvlen_npu, scale_value)

    torch_npu.npu.synchronize()
    logging.info("NPU computation done.")

    # Compare (outputs are TND 3D)
    compare(dq_nope_out_npu.cpu(), data['dq_nope_golden'], "dQ_nope",
            atol=0.0001, rtol=0.0078125, max_error_count=100)
    compare(dq_pe_out_npu.cpu(), data['dq_pe_golden'], "dQ_pe",
            atol=0.0001, rtol=0.0078125, max_error_count=100)
    compare(dk_nope_out_npu.cpu(), data['dk_nope_golden'], "dK_nope",
            atol=0.0001, rtol=0.0078125, max_error_count=100)
    compare(dk_pe_out_npu.cpu(), data['dk_pe_golden'], "dK_pe",
            atol=0.0001, rtol=0.0078125, max_error_count=100)

    logging.info(f"Test {case_name} PASSED!")


def do_test_sfa_grad_npu(case_name, actual_q_lens, actual_kv_lens,
                      n_1, n_2, d, dr, k, seed=42):
    """Run SFA grad test on NPU (TND format)."""
    logging.info("=" * 60)
    logging.info(f"Test: SFA grad NPU ({case_name})")
    logging.info("=" * 60)

    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    data = gen_test_data_tnd(actual_q_lens, actual_kv_lens,
                             n_1, n_2, d, dr, k, seed=seed)
    scale_value = data['scale_value']
    batch = data['batch']
    t_1 = data['t_1']
    t_2 = data['t_2']
    group = n_1 // n_2

    ### save data
    path = f"./golden/B_{batch}_T1_{t_1}/"
    # 确保路径存在
    os.makedirs(path, exist_ok=True)

    for name, tensor in data.items():
        torch.save(tensor, path + f"{name}.pt")
    logging.info("save success !!!")

    logging.info(f"  batch={batch}, t_1={t_1}, t_2={t_2}, n_1={n_1}, n_2={n_2}, d={d}, dr={dr}, k={k}, group={group}")
    logging.info(f"  actual_q_lens={actual_q_lens}, actual_kv_lens={actual_kv_lens}")
    logging.info(f"  scale_value={scale_value:.6f}")

    from sparse_flash_attention_grad_impl import sparse_flash_attention_grad

    # Move to NPU - all in TND 3D format
    q_pe_npu = data['q_pe'].npu()
    k_nope_npu = data['k_nope'].npu()
    q_nope_npu = data['q_nope'].npu()
    k_pe_npu = data['k_pe'].npu()
    value_npu = data['value'].npu()
    actual_seq_qlen_npu = data['actual_seq_qlen'].npu()
    actual_seq_kvlen_npu = data['actual_seq_kvlen'].npu()
    d_out_npu = data['d_out'].npu()
    sparse_idx_npu = data['sparse_indices'].to(torch.int32).npu()
    sm_max_npu = data['sm_max'].npu()
    sm_sum_npu = data['sm_sum'].npu()
    out_npu = data['out'].npu()

    dq_nope_out_shape = (t_1 * n_1, d)
    dq_pe_out_shape = (t_1 * n_1, dr)
    dk_nope_out_shape = (t_2 * n_2, d)
    dk_pe_out_shape = (t_2 * n_2, dr)
    dv_out_shape = (t_2 * n_2, d)
    dtype = data['q_nope'].dtype

    dq_nope_out = torch.empty(dq_nope_out_shape, dtype=dtype)
    dq_nope_out_npu = dq_nope_out.npu()
    dq_pe_out = torch.empty(dq_pe_out_shape, dtype=dtype)
    dq_pe_out_npu = dq_pe_out.npu()
    dk_nope_out = torch.zeros(dk_nope_out_shape, dtype=torch.float32) + 0
    dk_nope_out_npu = dk_nope_out.npu()
    dk_pe_out = torch.zeros(dk_pe_out_shape, dtype=torch.float32) + 0
    dk_pe_out_npu = dk_pe_out.npu()
    dv_out = torch.zeros(dv_out_shape, dtype=torch.float32) + 0
    dv_out_npu = dv_out.npu()
    dk_out = torch.zeros((t_2 * n_2, d + dr), dtype=torch.float32) + 0
    dk_out_npu = dk_out.npu()

    logging.info("Running PyPTO SFA grad (TND, nope/rope split, dynamic) on NPU...")
    sparse_flash_attention_grad(
        q_nope_npu, q_pe_npu, k_nope_npu, k_pe_npu, value_npu,
        sparse_idx_npu,
        d_out_npu, out_npu, sm_max_npu, sm_sum_npu,
        actual_seq_qlen_npu, actual_seq_kvlen_npu,
        dq_nope_out_npu, dq_pe_out_npu, dk_nope_out_npu, dk_pe_out_npu, dv_out_npu,
        dk_nope_out_npu, dk_pe_out_npu, dk_out_npu, dk_out_npu,
        n_1, n_2, d, dr, k, group, scale_value
    )
    torch_npu.npu.synchronize()
    logging.info("NPU computation done.")

    dk_nope_out_slice = dk_out_npu.cpu()[:, :d]
    dk_pe_out_slice = dk_out_npu.cpu()[:, d:]

    # Compare (outputs are TND 3D)
    compare(dq_nope_out_npu.cpu(), data['dq_nope_golden'].reshape(t_1 * n_1, d), "dQ_nope",
            atol=0.0001, rtol=0.0078125, max_error_count=100)
    compare(dq_pe_out_npu.cpu(), data['dq_pe_golden'].reshape(t_1 * n_1, dr), "dQ_pe",
            atol=0.0001, rtol=0.0078125, max_error_count=100)
    compare(dk_nope_out_slice.to(dtype), data['dk_nope_golden'].reshape(t_2 * n_2, d), "dK_nope",
            atol=0.0001, rtol=0.0078125, max_error_count=100)
    compare(dk_pe_out_slice.to(dtype), data['dk_pe_golden'].reshape(t_2 * n_2, dr), "dK_pe",
            atol=0.0001, rtol=0.0078125, max_error_count=100)

    logging.info(f"Test {case_name} PASSED!")


def test_level0_tiny():
    """Level 0: batch=1, S1=1, S2=16, n_1=2, d=8, dr=8, k=8"""
    do_test_sfa_grad_npu("level0_tiny",
                      actual_q_lens=[1], actual_kv_lens=[16],
                      n_1=2, n_2=1, d=8, dr=8, k=8)


@pytest.mark.skip(reason="large test case")
def test_level1_small():
    """Level 1: batch=1, S1=1, S2=2048, n_1=16, d=512, dr=64, k=1024"""
    do_test_sfa_grad_npu("level1_b1_s1",
                      actual_q_lens=[1], actual_kv_lens=[2048],
                      n_1=16, n_2=1, d=512, dr=64, k=1024)


@pytest.mark.skip(reason="large test case")
def test_level2_medium():
    """Level 2: batch=1, S1=4, S2=2048, n_1=16, d=512, dr=64, k=1024"""
    do_test_sfa_grad_npu("level2_b1_s4",
                      actual_q_lens=[128], actual_kv_lens=[32768],
                      n_1=2, n_2=1, d=512, dr=64, k=2048,
                      seed=123)
    
    
@pytest.mark.skip(reason="large test case")
def test_level2_t1k():
    do_test_sfa_grad_npu("level2_b1_s4",
                      actual_q_lens=[1024], actual_kv_lens=[32768],
                      n_1=2, n_2=1, d=512, dr=64, k=2048,
                      seed=123)
    
    
@pytest.mark.skip(reason="large test case")
def test_level2_t256():
    do_test_sfa_grad_npu("level2_b1_s4",
                      actual_q_lens=[256], actual_kv_lens=[32768],
                      n_1=64, n_2=1, d=512, dr=64, k=2048,
                      seed=123)
    
    
@pytest.mark.skip(reason="large test case")
def test_level2_t128():
    do_test_sfa_grad_npu("level2_b1_s4",
                      actual_q_lens=[128], actual_kv_lens=[32768],
                      n_1=32, n_2=1, d=512, dr=64, k=2048,
                      seed=123)


@pytest.mark.skip(reason="large test case")
def test_eager():
    do_test_sfa_grad_npu_eager("level_eager",
                      actual_q_lens=[128, ], actual_kv_lens=[32768, ],
                      n_1=2, n_2=1, d=512, dr=64, k=2048,
                      seed=123)


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
        level=logging.INFO
    )

    logging.info("Starting SFA Grad (TND, nope/rope split) tests...")
    test_level0_tiny()
    logging.info("All tests completed!")
