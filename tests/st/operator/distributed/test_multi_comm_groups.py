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
The scripts provides an example for multiple comm groups with allreduce cascading case

Main Functions:
    - allreduce_cascading_worker: Main function for allreduce_cascading
    - test_allreduce_cascading: Performance test function
"""

import multiprocessing as mp
import traceback
import numpy as np
import pytest
import torch
import pypto
from distributed_config import DistributedConfig, collect_process_errors


def _create_shmem_tensors(group_names, world_sizes, shmem_shape):
    shmem_tensor = pypto.distributed.create_shmem_tensor(
        group_names[0], world_sizes[0], pypto.DT_FP32, shmem_shape)
    my_pe = pypto.distributed.my_symbolic_pe(group_names[0])
    
    shmem_tensor1 = pypto.distributed.create_shmem_tensor(
        group_names[1], world_sizes[1], pypto.DT_FP32, shmem_shape)
    my_pe1 = pypto.distributed.my_symbolic_pe(group_names[1])
    
    return shmem_tensor, my_pe, shmem_tensor1, my_pe1


def _get_input_tile(in_tensor, batch_size, bs_idx, view_row_shape, hidden_size):
    in_tensor_tile = pypto.view(
        in_tensor, (view_row_shape, in_tensor.shape[1]), [bs_idx * view_row_shape, 0],
        valid_shape=[(batch_size - bs_idx * view_row_shape).min(view_row_shape), in_tensor.shape[1]])
    
    pypto.set_vec_tile_shapes(view_row_shape, hidden_size)
    in_tensor_tile_fp32 = pypto.cast(in_tensor_tile, pypto.DT_FP32)
    
    return in_tensor_tile_fp32


def _perform_allreduce(input_tensor, stage_params):
    """执行 AllReduce 阶段
    Args:
        input_tensor: 输入tensor
        stage_params: 列表参数 [shmem_tensor, world_size, shmem_shape, my_pe,
                                 view_row_shape, hidden_size, batch_size, bs_idx]
    Returns:
        AllReduce 结果 (BF16)
    """
    (shmem_tensor, world_size, shmem_shape, my_pe,
     view_row_shape, hidden_size, batch_size, bs_idx) = stage_params
    pypto.set_vec_tile_shapes(view_row_shape, hidden_size)
    
    for dyn_idx in range(world_size):
        put_out = pypto.distributed.shmem_put(
            input_tensor, [0, 0], shmem_tensor, dyn_idx,
            put_op=pypto.AtomicType.ADD, pred=[input_tensor]
        )
        pypto.distributed.shmem_signal(
            shmem_tensor, dyn_idx, 1, shmem_shape,
            [0, 0], target_pe=dyn_idx, sig_op=pypto.AtomicType.ADD, pred=[put_out]
        )
    
    wait_until_out = pypto.distributed.shmem_wait_until(
        shmem_tensor, my_pe, world_size,
        shmem_shape, [0, 0], cmp=pypto.OpType.EQ, clear_signal=True, pred=[input_tensor]
    )
    all_reduce_out_fp32 = pypto.distributed.shmem_get(
        shmem_tensor, my_pe, shmem_shape, [0, 0], pred=[wait_until_out],
        valid_shape=[(batch_size - bs_idx * view_row_shape).min(view_row_shape), hidden_size]
    )
    
    pypto.set_vec_tile_shapes(1, hidden_size)
    all_reduce_out_bf16 = pypto.cast(all_reduce_out_fp32, pypto.DT_BF16)
    
    return all_reduce_out_bf16


@pypto.frontend.jit()
def allreduce_cascading_kernel(
    in_tensor: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    out_tensor: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    group_names,
    world_sizes,
):
    batch_size = in_tensor.shape[0]
    hidden_size = in_tensor.shape[1]

    view_row_shape = 8
    bs_loop = (batch_size + view_row_shape - 1) // view_row_shape
    shmem_shape = [view_row_shape, hidden_size]

    for bs_idx in pypto.loop(bs_loop, name="LOOP_ALLREDUCE_CASCADING", idx_name="bs_idx"):
        shmem_tensor, my_pe, shmem_tensor1, my_pe1 = _create_shmem_tensors(
            group_names, world_sizes, shmem_shape)
        
        in_tensor_tile_fp32 = _get_input_tile(
            in_tensor, batch_size, bs_idx, view_row_shape, hidden_size)
        
        # 第一阶段参数列表
        stage1_params = [shmem_tensor, world_sizes[0], shmem_shape, my_pe,
                         view_row_shape, hidden_size, batch_size, bs_idx]
        all_reduce_out_bf16 = _perform_allreduce(in_tensor_tile_fp32, stage1_params)
        
        # 第二阶段参数列表
        stage2_params = [shmem_tensor1, world_sizes[1], shmem_shape, my_pe1,
                         view_row_shape, hidden_size, batch_size, bs_idx]
        all_reduce_out_bf161 = _perform_allreduce(all_reduce_out_bf16, stage2_params)
        
        out_tensor[bs_idx * pypto.symbolic_scalar(view_row_shape):] = all_reduce_out_bf161


def generate_group_splits(world_size, group_info=None):
    if group_info is not None:
        return group_info
    group_info = {
        "even_odd_0": list(range(0, world_size, 2)),  # 偶数组
        "even_odd_1": list(range(1, world_size, 2)),  # 奇数组
        "half_0": list(range(0, world_size // 2)),    # 前半部分
        "half_1": list(range(world_size // 2, world_size)),  # 后半部分
    }
    return group_info


def generate_allreduce_cascading_golden_data(config: DistributedConfig, group_info=None):
    batch_size = 13
    hidden_size = 256
    world_size = config.world_size
    input_datas = []
    for _ in range(world_size):
        in_tensor = torch.randn((batch_size, hidden_size), dtype=torch.bfloat16).share_memory_()
        input_datas.append([in_tensor])

    group_info = generate_group_splits(world_size, group_info)

    intermediate_datas = [None] * world_size
    # allreduce
    for group_name in ["even_odd_0", "even_odd_1"]:
        group_ranks = group_info.get(group_name, [])
        if not group_ranks:
            continue
        group_sum_fp32 = torch.zeros((batch_size, hidden_size), dtype=torch.float32)
        for rank in group_ranks:
            group_sum_fp32 += input_datas[rank][0].to(torch.float32).cpu()
        group_sum_bf16 = group_sum_fp32.to(torch.bfloat16)

        for rank in group_ranks:
            intermediate_datas[rank] = group_sum_bf16

    output_datas = [None] * world_size
    # allreduce
    for group_name in ["half_0", "half_1"]:
        group_ranks = group_info.get(group_name, [])
        if not group_ranks:
            continue
        group_sum_fp32 = torch.zeros((batch_size, hidden_size), dtype=torch.float32)
        for rank in group_ranks:
            group_sum_fp32 += intermediate_datas[rank].to(torch.float32).cpu()
        group_sum_bf16 = group_sum_fp32.to(torch.bfloat16)
        
        for rank in group_ranks:
            output_datas[rank] = group_sum_bf16

    return input_datas, output_datas


def allreduce_cascading_worker(worker_params, error_queue: mp.Queue):
    """
    Args:
        worker_params: 列表参数 [config, input_data, output_data, logical_rank_id, group_info]
        error_queue: 错误队列
    """
    try:
        config, input_data, output_data, logical_rank_id, group_info = worker_params
        
        group_info = generate_group_splits(config.world_size, group_info)
        groups = config.init_hccl_comm(logical_rank_id, group_info)
        physical_device_id = config.get_physical_device_id(logical_rank_id)
        device = f'npu:{physical_device_id}'

        in_tensor = input_data[0]
        golden_out_tensor = output_data

        out_tensor = torch.empty(in_tensor.shape, dtype=torch.bfloat16, device=device)
        inputs = [in_tensor.to(device)]

        group_key0 = "even_odd_0" if logical_rank_id % 2 == 0 else "even_odd_1"
        group_name0 = groups[0]
        world_size0 = len(group_info.get(group_key0, []))
        
        mid = config.world_size // 2
        group_key1 = "half_0" if logical_rank_id < mid else "half_1"
        group_name1 = groups[1]
        world_size1 = len(group_info.get(group_key1, []))

        group_names = [group_name0, group_name1]
        world_sizes = [world_size0, world_size1]
        allreduce_cascading_kernel(*inputs, out_tensor, group_names, world_sizes)

        np.testing.assert_allclose(
            np.array(out_tensor.cpu().flatten().tolist()),
            np.array(golden_out_tensor.cpu().flatten().tolist()),
            rtol=8e-3,
            atol=8e-3,
        )
    except Exception as e:
        if error_queue is not None:
            error_queue.put((logical_rank_id, str(e), traceback.format_exc()))
        raise


@pytest.mark.world_size(4)
def test_allreduce_cascading():
    mp.set_start_method('spawn', force=True)
    config = DistributedConfig(world_size=4)
    default_group_info = {
        "even_odd_0": [0, 2],
        "even_odd_1": [1, 3],
        "half_0": [0, 1],
        "half_1": [2, 3],
    }
    input_datas, output_datas = generate_allreduce_cascading_golden_data(config)

    error_queue = mp.Queue()
    processes = []
    
    for i in range(config.world_size):
        worker_params = [config, input_datas[i], output_datas[i], i, None]
        p = mp.Process(
            target=allreduce_cascading_worker,
            args=(worker_params, error_queue)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    collect_process_errors(processes, error_queue)


def main():
    test_allreduce_cascading()


if __name__ == '__main__':
    main()
