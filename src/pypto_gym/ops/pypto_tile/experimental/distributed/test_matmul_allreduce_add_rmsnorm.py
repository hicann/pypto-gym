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
GLM-4.5 MatMul AllReduce Add RmsNorm Module

This module implements a fused matmul, all-reduce, add, and RMSNorm operation for large-scale distributed models.
It efficiently combines computation and communication, reducing memory overhead and accelerating training and inference.

Main Functions:
    - matmul_allreduce_add_rmsnorm: Main function for fused matmul, all-reduce, add, and RMSNorm computation
"""

import multiprocessing as mp
import traceback

import numpy as np
import pytest
import torch
from torch._dynamo import allow_in_graph
from torch._subclasses import fake_tensor

import pypto

from distributed_config import DistributedConfig, collect_process_errors


@pypto.frontend.jit()
def matmul_allreduce_add_rmsnorm_kernel(
    in_tensor: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    matmul_weight: pypto.Tensor(),
    residual: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    gamma: pypto.Tensor(),
    bias: pypto.Tensor(),
    out_tensor: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    residual_out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    eps,
    group_name,
    world_size,
):
    batch_size = in_tensor.shape[0]
    hidden_size = matmul_weight.shape[0]

    in_tensor_mean_coff = 1.0 / hidden_size
    view_row_shape = 8
    bs_loop = (batch_size + view_row_shape - 1) // view_row_shape

    pypto.set_vec_tile_shapes(hidden_size)
    gamma_2d = pypto.reshape(gamma, [1, hidden_size], inplace=True)
    bias_2d = pypto.reshape(bias, [1, hidden_size], inplace=True)

    for bs_idx in pypto.loop(bs_loop, name="LOOP_MM_ALLREDUCE_ADD_RMSNORM", idx_name="bs_idx"):
        # 1. create shmem tesnor
        shmem_shape = [view_row_shape, hidden_size]
        shmem_tensor = pypto.distributed.create_shmem_tensor(
            group_name, world_size, pypto.DT_BF16, shmem_shape)
        shmem_barrier_signal = pypto.distributed.create_shmem_signal(group_name, world_size)
        my_pe = pypto.distributed.my_symbolic_pe(group_name)
        in_tensor_tile = pypto.view(
            in_tensor, (view_row_shape, in_tensor.shape[1]), [bs_idx * view_row_shape, 0],
            valid_shape=[(batch_size - bs_idx * view_row_shape).min(view_row_shape), in_tensor.shape[1]])

        # 2. clear data
        pypto.set_vec_tile_shapes(view_row_shape, hidden_size)
        data_clear_out = pypto.distributed.shmem_clear_data(
            shmem_tensor, shmem_shape, [0, 0], pred=[in_tensor_tile])
        barrier_out = pypto.distributed.shmem_barrier_all(
            shmem_barrier_signal, [data_clear_out])

        # 3. matmul
        pypto.set_cube_tile_shapes([8, 8], [128, 256], [256, 512])
        matmul_result = pypto.matmul(in_tensor_tile, matmul_weight, pypto.DT_BF16, b_trans=True)

        # 4. allreduce
        pypto.set_vec_tile_shapes(view_row_shape, hidden_size)
        for dyn_idx in range(world_size):
            put_out = pypto.distributed.shmem_put(matmul_result, [0, 0], shmem_tensor, dyn_idx,
                put_op=pypto.AtomicType.ADD, pred=[barrier_out])
            pypto.distributed.shmem_signal(shmem_tensor, dyn_idx, 1, shmem_shape,
                [0, 0], target_pe=dyn_idx, sig_op=pypto.AtomicType.ADD, pred=[put_out])
        wait_until_out = pypto.distributed.shmem_wait_until(shmem_tensor, my_pe, world_size,
            shmem_shape, [0, 0], cmp=pypto.OpType.EQ, clear_signal=True, pred=[in_tensor_tile])
        pypto.set_vec_tile_shapes(1, hidden_size)
        all_reduce_out = pypto.experimental.shmem_load(
            shmem_tensor, my_pe, shmem_shape, [0, 0], pred=[wait_until_out],
            valid_shape=[(batch_size - bs_idx * view_row_shape).min(view_row_shape), hidden_size]
        )

        # 5. Add RmsNorm
        residual_tile = pypto.view(
            residual, (view_row_shape, hidden_size), [bs_idx * view_row_shape, 0],
            valid_shape=[(batch_size - bs_idx * view_row_shape).min(view_row_shape), hidden_size])

        # add
        all_reduce_out_fp32 = pypto.cast(all_reduce_out, pypto.DT_FP32)
        residual_tile_fp32 = pypto.cast(residual_tile, pypto.DT_FP32)
        add_out = pypto.add(all_reduce_out_fp32, residual_tile_fp32)

        # rms norm
        square = pypto.mul(add_out, add_out)
        mean_res = pypto.mul(square, in_tensor_mean_coff)
        reduce_asum = pypto.sum(mean_res, -1, True)
        reduce_sum = pypto.add(reduce_asum, eps)
        reduce_sqrt = pypto.sqrt(reduce_sum)
        res_div = pypto.div(add_out, reduce_sqrt)

        hidden_bf16 = pypto.tensor([view_row_shape, hidden_size], pypto.DT_BF16, "hidden_bf16")
        residual_bf16_tmp = pypto.cast(add_out, in_tensor.dtype)
        for tmp_idx in range(view_row_shape):
            gamma_2d_fp32 = pypto.cast(gamma_2d, pypto.DT_FP32)
            bias_2d_fp32 = pypto.cast(bias_2d, pypto.DT_FP32)
            res_div_single = pypto.view(res_div, [1, hidden_size], [tmp_idx, 0])
            res = pypto.mul(res_div_single, gamma_2d_fp32)
            res_add = pypto.add(res, bias_2d_fp32)
            in_tensor_norm = pypto.cast(res_add, in_tensor.dtype)
            hidden_bf16[tmp_idx:tmp_idx + 1] = in_tensor_norm

        residual_out[bs_idx * pypto.symbolic_scalar(view_row_shape):] = residual_bf16_tmp
        out_tensor[bs_idx * pypto.symbolic_scalar(view_row_shape):] = hidden_bf16


def generate_golden_data(config: DistributedConfig):
    # 设置参数
    batch_size = 8
    attn_dim_per_tp = 1536
    hidden_size = 5120
    world_size = config.world_size
    torch.manual_seed(42)

    #构造每张卡上需要的数据
    input_datas = []
    for rank in range(world_size):
        physical_device_id = config.get_physical_device_id(rank)
        device = f'npu:{physical_device_id}'
        in_tensor = torch.randn((batch_size, attn_dim_per_tp), dtype=torch.bfloat16, device=device)
        matmul_weight = torch.randn((hidden_size, attn_dim_per_tp), dtype=torch.bfloat16, device=device)
        residual = torch.randn((batch_size, hidden_size), dtype=torch.bfloat16, device=device)
        gamma = torch.randn((hidden_size), dtype=torch.bfloat16, device=device)
        bias = torch.randn((hidden_size), dtype=torch.bfloat16, device=device)
        eps = 1e-5
        input_data = [in_tensor, matmul_weight, residual, gamma, bias, eps]
        input_datas.append(input_data)
    output_datas = matmul_allreduce_add_rmsnorm_result_golden(batch_size, hidden_size, input_datas)
    return input_datas, output_datas


def matmul_allreduce_add_rmsnorm_result_golden(batch_size, num, input_datas):
    output_datas = []
    # 计算 matmul & allreduce 结果， 该结果所有卡上一致
    matmul_allreduce_result_bf16 = torch.zeros((batch_size, num), dtype=torch.bfloat16)
    for input_data in input_datas:
        in_tensor, matmul_weight = input_data[:2]
        matmul_result = torch.matmul(in_tensor, matmul_weight.T)
        matmul_allreduce_result_bf16 += matmul_result.cpu()
    matmul_allreduce_result_fp32 = matmul_allreduce_result_bf16.to(torch.float32)
    # 计算各卡上add_rmsnorm之后的结果
    for input_data in input_datas:
        residual, gamma, bias, eps = input_data[-4:]
        res_add = residual.to(torch.float32) + matmul_allreduce_result_fp32.to(residual.device)
        mean_coff = 1.0 / res_add.shape[-1]
        in_tensor_f32 = res_add
        square = in_tensor_f32 * in_tensor_f32
        square = square.sum(dim=-1, keepdim=True)
        mean_res = square * mean_coff
        reduce_sum = mean_res + eps
        reduce_sqrt = torch.sqrt(reduce_sum)
        res_div = in_tensor_f32 / reduce_sqrt
        res = res_div * gamma.to(torch.float32)
        res = res + bias.to(res.dtype)
        output_data = [res.to(torch.bfloat16), in_tensor_f32.to(torch.bfloat16)]
        output_datas.append(output_data)
    return output_datas


def matmul_allreduce_add_rmsnorm_worker(
    config: DistributedConfig,
    input_data: list,
    output_data: list,
    logical_rank_id: int,
    error_queue: mp.Queue,
):
    try:
        groups = config.init_hccl_comm(logical_rank_id)
        physical_device_id = config.get_physical_device_id(logical_rank_id)
        device = f'npu:{physical_device_id}'

        in_tensor, matmul_weight, residual, gamma, bias, eps = input_data
        golden_out_tensor, golden_residual = output_data

        out_tensor = torch.empty(residual.shape, dtype=torch.bfloat16, device=device)
        residual_out = torch.empty(residual.shape, dtype=torch.bfloat16, device=device)

        inputs = [in_tensor, matmul_weight, residual, gamma, bias, out_tensor, residual_out]

        matmul_allreduce_add_rmsnorm_kernel(*inputs, eps, groups[0], config.world_size)

        np.testing.assert_allclose(
            np.array(out_tensor.cpu().flatten().tolist()),
            np.array(golden_out_tensor.cpu().flatten().tolist()),
            rtol=8e-3,
            atol=8e-3,
        )

        np.testing.assert_allclose(
            np.array(residual_out.cpu().flatten().tolist()),
            np.array(golden_residual.cpu().flatten().tolist()),
            rtol=8e-3,
            atol=8e-3,
        )
    except Exception as e:
        if error_queue is not None:
            error_queue.put((logical_rank_id, str(e), traceback.format_exc()))
        raise


@allow_in_graph
def matmul_allreduce_add_rmsnorm(
    in_tensor: torch.Tensor,
    matmul_weight: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    group_name: str,
    world_size: int,
):
    if isinstance(in_tensor, fake_tensor.FakeTensor):
        return None, None

    out_tensor = torch.empty(residual.shape, dtype=torch.bfloat16, device=residual.device)
    residual_out = torch.empty(residual.shape, dtype=torch.bfloat16, device=residual.device)

    inputs = [in_tensor, matmul_weight, residual, gamma, bias, out_tensor, residual_out]

    matmul_allreduce_add_rmsnorm_kernel(*inputs, eps, group_name, world_size)

    return out_tensor, residual_out


@pytest.mark.skip(reason="temporarily disabled")
@pytest.mark.world_size(4)
def test_matmul_allreduce_add_rmsnorm():
    mp.set_start_method('spawn', force=True)
    config = DistributedConfig(world_size=4)
    input_datas, output_datas = generate_golden_data(config)

    error_queue = mp.Queue()

    processes = []
    for i in range(config.world_size):
        p = mp.Process(
            target=matmul_allreduce_add_rmsnorm_worker,
            args=(config, input_datas[i], output_datas[i], i, error_queue)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    collect_process_errors(processes, error_queue)


def main():
    test_matmul_allreduce_add_rmsnorm()


if __name__ == '__main__':
    main()
