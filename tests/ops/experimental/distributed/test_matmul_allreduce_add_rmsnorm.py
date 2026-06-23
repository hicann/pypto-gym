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
GLM-4.5 MatMul AllReduce Add RmsNorm Module

This module implements a fused matmul, all-reduce, add, and RMSNorm operation for large-scale distributed models.
It efficiently combines computation and communication, reducing memory overhead and accelerating training and inference.

Main Functions:
    - matmul_allreduce_add_rmsnorm: Main function for fused matmul, all-reduce, add, and RMSNorm computation
"""


import sys
import os
_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')):
    _p = os.path.dirname(_p)
sys.path.insert(0, os.path.join(_p, 'src'))
sys.path.insert(0, os.path.join(_p, 'src', 'pypto_gym', 'ops', 'pypto_tile'))

import dataclasses
import multiprocessing as mp
import traceback

import numpy as np
import pytest
import torch
import torch_npu  # noqa: F401
from torch._dynamo import allow_in_graph
from torch._subclasses import fake_tensor

import pypto

from experimental.distributed.distributed_config import DistributedConfig, collect_process_errors


@dataclasses.dataclass
class DoAllreduceMatmulInputs:
    matmul_weight: object
    in_tensor_tile: object
    shmem_tensor: object
    shmem_barrier_signal: object
    my_pe: object
    world_size: int
    shmem_shape: list
    view_row_shape: int
    hidden_size: int
    batch_size: int
    bs_idx: int


@dataclasses.dataclass
class AddRmsnormAndStoreOutputInputs:
    all_reduce_out: object
    residual: object
    batch_size: int
    bs_idx: int
    view_row_shape: int
    hidden_size: int
    in_tensor_mean_coff: float
    eps: float
    gamma_2d: object
    bias_2d: object
    in_tensor: object
    residual_out: object
    out_tensor: object


def _do_allreduce_matmul(inputs: DoAllreduceMatmulInputs):
    pypto.set_vec_tile_shapes(inputs.view_row_shape, inputs.hidden_size)
    data_clear_out = pypto.distributed.shmem_clear_data(
        inputs.shmem_tensor, inputs.shmem_shape, [0, 0], pred=[inputs.in_tensor_tile])
    barrier_out = pypto.distributed.shmem_barrier_all(
        inputs.shmem_barrier_signal, [data_clear_out])
    pypto.set_cube_tile_shapes([8, 8], [128, 256], [256, 512])
    matmul_result = pypto.matmul(inputs.in_tensor_tile, inputs.matmul_weight, pypto.DT_BF16, b_trans=True)
    pypto.set_vec_tile_shapes(inputs.view_row_shape, inputs.hidden_size)
    for dyn_idx in range(inputs.world_size):
        put_out = pypto.distributed.shmem_put(matmul_result, [0, 0], inputs.shmem_tensor, dyn_idx,
            put_op=pypto.AtomicType.ADD, pred=[barrier_out])
        pypto.distributed.shmem_signal(inputs.shmem_tensor, dyn_idx, 1, inputs.shmem_shape,
            [0, 0], target_pe=dyn_idx, sig_op=pypto.AtomicType.ADD, pred=[put_out])
    wait_until_out = pypto.distributed.shmem_wait_until(inputs.shmem_tensor, inputs.my_pe, inputs.world_size,
        inputs.shmem_shape, [0, 0], cmp=pypto.OpType.EQ, clear_signal=True, pred=[inputs.in_tensor_tile])
    pypto.set_vec_tile_shapes(1, inputs.hidden_size)
    return pypto.experimental.shmem_load(
        inputs.shmem_tensor, inputs.my_pe, inputs.shmem_shape, [0, 0], pred=[wait_until_out],
        valid_shape=[(inputs.batch_size - inputs.bs_idx * inputs.view_row_shape).min(
            inputs.view_row_shape), inputs.hidden_size]
    )


def _add_rmsnorm_and_store_output(inputs: AddRmsnormAndStoreOutputInputs):
    residual_tile = pypto.view(
        inputs.residual, (inputs.view_row_shape, inputs.hidden_size), [inputs.bs_idx * inputs.view_row_shape, 0],
        valid_shape=[(inputs.batch_size - inputs.bs_idx * inputs.view_row_shape).min(
            inputs.view_row_shape), inputs.hidden_size])
    all_reduce_out_fp32 = pypto.cast(inputs.all_reduce_out, pypto.DT_FP32)
    residual_tile_fp32 = pypto.cast(residual_tile, pypto.DT_FP32)
    add_out = pypto.add(all_reduce_out_fp32, residual_tile_fp32)
    square = pypto.mul(add_out, add_out)
    mean_res = pypto.mul(square, inputs.in_tensor_mean_coff)
    reduce_asum = pypto.sum(mean_res, -1, True)
    reduce_sum = pypto.add(reduce_asum, inputs.eps)
    reduce_sqrt = pypto.sqrt(reduce_sum)
    res_div = pypto.div(add_out, reduce_sqrt)
    hidden_bf16 = pypto.tensor([inputs.view_row_shape, inputs.hidden_size], pypto.DT_BF16, "hidden_bf16")
    residual_bf16_tmp = pypto.cast(add_out, inputs.in_tensor.dtype)
    for tmp_idx in range(inputs.view_row_shape):
        gamma_2d_fp32 = pypto.cast(inputs.gamma_2d, pypto.DT_FP32)
        bias_2d_fp32 = pypto.cast(inputs.bias_2d, pypto.DT_FP32)
        res_div_single = pypto.view(res_div, [1, inputs.hidden_size], [tmp_idx, 0])
        res = pypto.mul(res_div_single, gamma_2d_fp32)
        res_add = pypto.add(res, bias_2d_fp32)
        in_tensor_norm = pypto.cast(res_add, inputs.in_tensor.dtype)
        hidden_bf16[tmp_idx:tmp_idx + 1] = in_tensor_norm
    inputs.residual_out[inputs.bs_idx * pypto.symbolic_scalar(inputs.view_row_shape):] = residual_bf16_tmp
    inputs.out_tensor[inputs.bs_idx * pypto.symbolic_scalar(inputs.view_row_shape):] = hidden_bf16


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
        shmem_shape = [view_row_shape, hidden_size]
        shmem_tensor = pypto.distributed.create_shmem_tensor(
            group_name, world_size, pypto.DT_BF16, shmem_shape)
        shmem_barrier_signal = pypto.distributed.create_shmem_signal(group_name, world_size)
        my_pe = pypto.distributed.my_symbolic_pe(group_name)
        in_tensor_tile = pypto.view(
            in_tensor, (view_row_shape, in_tensor.shape[1]), [bs_idx * view_row_shape, 0],
            valid_shape=[(batch_size - bs_idx * view_row_shape).min(view_row_shape), in_tensor.shape[1]])

        all_reduce_out = _do_allreduce_matmul(DoAllreduceMatmulInputs(
            matmul_weight=matmul_weight, in_tensor_tile=in_tensor_tile,
            shmem_tensor=shmem_tensor, shmem_barrier_signal=shmem_barrier_signal,
            my_pe=my_pe, world_size=world_size, shmem_shape=shmem_shape,
            view_row_shape=view_row_shape, hidden_size=hidden_size,
            batch_size=batch_size, bs_idx=bs_idx))

        _add_rmsnorm_and_store_output(AddRmsnormAndStoreOutputInputs(
            all_reduce_out=all_reduce_out, residual=residual, batch_size=batch_size,
            bs_idx=bs_idx, view_row_shape=view_row_shape, hidden_size=hidden_size,
            in_tensor_mean_coff=in_tensor_mean_coff, eps=eps, gamma_2d=gamma_2d,
            bias_2d=bias_2d, in_tensor=in_tensor, residual_out=residual_out,
            out_tensor=out_tensor))


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
