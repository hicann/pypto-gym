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
GLM-4.5 Distributed MoE Dispatch And Combine Module

This module implements the distributed token dispatch and combine stage for
MoE in an expert parallel setting, based on open shared memory.

Main Functions:
    - moe_distributed_dispatch_kernel: JIT compiled dispatch kernel
    - moe_distributed_combine_kernel: JIT compiled combine kernel
"""

import dataclasses
import random
from typing import Callable, Optional, Union
import traceback

import multiprocessing as mp
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch._dynamo import allow_in_graph
from torch._subclasses import fake_tensor

import pypto

from distributed_config import DistributedConfig, collect_process_errors

TensorList = list[torch.Tensor]

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def check_cond(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def align_up(value: int, alignment: int) -> int:
    check_cond(
        alignment > 0 and (alignment & (alignment - 1)) == 0,
        f'alignment must be a power of two, but got {alignment}',
    )
    return (value + alignment - 1) & ~(alignment - 1)


def assert_allclose_with_eps(expected: torch.Tensor, actual: torch.Tensor, eps: float = 0.001) -> None:
    if expected.shape != actual.shape:
        raise ValueError(f'Shape mismatch: {expected.shape=}, {actual.shape=}')
    if expected.dtype != actual.dtype:
        raise ValueError(f'Dtype mismatch: {expected.dtype=}, {actual.dtype=}')

    numel = expected.numel()

    abs_err = torch.abs(expected - actual)
    rel_err = abs_err / torch.clamp(torch.abs(expected), min=1e-12)
    err_mask = (abs_err > eps) | (rel_err > eps)
    err_count = err_mask.sum().item()

    zero_mask = (torch.abs(expected) > 1e-6) & (torch.abs(actual) <= 1e-6)
    zero_count = zero_mask.sum().item()

    check_cond(err_count <= numel * eps and zero_count <= 1000, 'Allclose failed')


def assert_allcolse_whit_rtol_and_atol(out, act):
    np.testing.assert_allclose(
        np.array(out.cpu().flatten().tolist()),
        np.array(act.cpu().flatten().tolist()),
        rtol=0,
        atol=0,
    )


@dataclasses.dataclass
class MoeCase:
    batch_size: int
    hidden_size: int
    moe_expert_num: int
    topk: int
    data_type: pypto.DataType
    ep_world_size: int

    def __post_init__(self):
        check_cond(
            self.topk <= self.moe_expert_num,
            f'topk ({self.topk}) must be <= moe_expert_num ({self.moe_expert_num})',
        )


@dataclasses.dataclass
class DispatchInOperands:
    x: torch.Tensor
    expert_ids: torch.Tensor
    x_active_mask: Optional[torch.Tensor] = None


@dataclasses.dataclass
class DispatchOutOperands:
    expand_x: torch.Tensor
    assist_info_for_combine: torch.Tensor
    expert_token_nums: torch.Tensor
    recv_counts: torch.Tensor


@dataclasses.dataclass
class CombineInOperands:
    expand_x: torch.Tensor
    assist_info_for_combine: torch.Tensor
    recv_counts: torch.Tensor
    expert_scales: torch.Tensor
    x_active_mask: torch.Tensor


@dataclasses.dataclass
class CombineOutOperands:
    out: torch.Tensor


def to_device(
    obj: Union[DispatchInOperands, DispatchOutOperands, CombineInOperands, CombineOutOperands],
    device: torch.device,
) -> None:
    fields = dataclasses.fields(obj)
    for field in fields:
        value = getattr(obj, field.name)
        if isinstance(value, torch.Tensor):
            setattr(obj, field.name, value.to(device))


def empty_like(obj: Union[DispatchOutOperands, CombineOutOperands]):
    fields = dataclasses.fields(obj)
    kwargs = {}
    for field in fields:
        value = getattr(obj, field.name)
        if isinstance(value, torch.Tensor):
            kwargs[field.name] = torch.empty_like(value.cpu()).to(value.device)
            # 如果直接用 torch.empty_like(value)，会有警告 UserWarning: Cannot create tensor with interal format while
            # allow_internel_format=False, tensor will be created with base format.
    return type(obj)(**kwargs)


def generate_random_tensor(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    float_dtypes = (torch.float16, torch.float32, torch.float64, torch.bfloat16)
    int_dtypes = (torch.int8, torch.int16, torch.int32, torch.int64)
    if dtype in float_dtypes:
        return torch.randn(shape, dtype=dtype)
    elif dtype in int_dtypes:
        return torch.randint(-10, 10, size=shape, dtype=dtype)
    else:
        raise ValueError(f'Unsupported dtype: {dtype}. Supported: {float_dtypes + int_dtypes}')


def generate_zero_tensor_list(shape: tuple[int, ...], dtype: torch.dtype, world_size: int) -> TensorList:
    return [torch.zeros(shape, dtype=dtype) for _ in range(world_size)]


def generate_inputs(
    moe_case: MoeCase,
    torch_data_type: torch.dtype,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    operands_list = []
    for _ in range(moe_case.ep_world_size):
        x = generate_random_tensor((moe_case.batch_size, moe_case.hidden_size), torch_data_type)

        expert_scores = generate_random_tensor((moe_case.batch_size, moe_case.moe_expert_num), torch.float32)
        topk_expert_scores, moe_expert_ids = expert_scores.topk(k=moe_case.topk)
        expert_scales = topk_expert_scores.softmax(dim=-1)

        moe_expert_ids = moe_expert_ids.to(dtype=torch.int32)

        x_active_mask = torch.zeros([moe_case.batch_size], dtype=torch.int32)
        active_count = random.randint(0, moe_case.batch_size)
        x_active_mask[:active_count] = 1

        operands_list.append((x, expert_scales, moe_expert_ids, x_active_mask))

    return operands_list


def get_moe_expert_num_per_rank(moe_case: MoeCase) -> int:
    return (moe_case.moe_expert_num + moe_case.ep_world_size - 1) // moe_case.ep_world_size


def get_moe_expert_rank_id_and_expert_offset(expert_id: int, moe_expert_num_per_rank: int) -> tuple[int, int]:
    return divmod(expert_id, moe_expert_num_per_rank)


def get_dispatch_output_row(moe_case: MoeCase) -> int:
    max_send_token_num = moe_case.topk * moe_case.batch_size * moe_case.ep_world_size
    max_receive_token_num = moe_case.batch_size * moe_case.moe_expert_num
    return min(max_send_token_num, max_receive_token_num)


def dispatch_tokens(
    moe_case: MoeCase,
    torch_data_type: torch.dtype,
    input_operands_list: list[DispatchInOperands],
) -> list[DispatchOutOperands]:
    expert_num_per_rank, total_send_tasks = get_moe_expert_num_per_rank(moe_case), moe_case.batch_size * moe_case.topk

    send_rank_cumsum_tables_list, send_rank_token_counts_list = [], []
    for op in input_operands_list:
        expert_ids_flat = op.expert_ids.flatten().to(torch.long)
        x_active_mask_flat = op.x_active_mask.unsqueeze(1).expand(-1, moe_case.topk).flatten()
        active_indices = x_active_mask_flat.nonzero().squeeze(-1)
        active_expert_ids = expert_ids_flat[active_indices]

        one_hot_table = F.one_hot(active_expert_ids, num_classes=moe_case.moe_expert_num)
        cumsum_table = torch.cumsum(one_hot_table, dim=0)
        send_rank_cumsum_tables_list.append((cumsum_table, active_indices))
        send_rank_token_counts_list.append(
            cumsum_table[-1, :] if cumsum_table.size(0) > 0 else torch.zeros(moe_case.moe_expert_num, dtype=torch.int64)
        )

    receive_rank_token_counts_list = generate_zero_tensor_list(
        [moe_case.moe_expert_num + 1], torch.int32, moe_case.ep_world_size,)

    for send_rank_id, token_count_per_expert in enumerate(send_rank_token_counts_list):
        for expert_id in range(moe_case.moe_expert_num):
            receive_rank_id, expert_offset = divmod(expert_id, expert_num_per_rank)
            offset = expert_offset * moe_case.ep_world_size + send_rank_id + 1
            receive_rank_token_counts_list[receive_rank_id][offset] = token_count_per_expert[expert_id]

    token_counts_tensor = torch.stack(receive_rank_token_counts_list)
    cumsum_result = torch.cumsum(token_counts_tensor, dim=1).to(torch.int32)
    recv_counts_tensor = cumsum_result[:, -1].unsqueeze(1)
    recv_counts_list = [recv_counts_tensor[i] for i in range(moe_case.ep_world_size)]

    indices = torch.arange(0, moe_case.moe_expert_num + 1, moe_case.ep_world_size)
    start_indices, end_indices = indices[:-1], indices[1:]
    expert_token_nums_tensor = cumsum_result[:, end_indices] - cumsum_result[:, start_indices]
    expert_token_nums_list = [expert_token_nums_tensor[i] for i in range(moe_case.ep_world_size)]

    row = get_dispatch_output_row(moe_case)
    expand_x_list = generate_zero_tensor_list([row, moe_case.hidden_size], torch_data_type, moe_case.ep_world_size)
    assist_info_for_combine_list = generate_zero_tensor_list([row, 3], torch.int32, moe_case.ep_world_size)

    for send_rank_id, (op, cumsum_table_data) in enumerate(zip(input_operands_list, send_rank_cumsum_tables_list)):
        cumsum_table, active_indices = cumsum_table_data
        for local_idx, index in enumerate(active_indices):
            token_id, k_offset = divmod(index.item(), moe_case.topk)
            expert_id = op.expert_ids[token_id, k_offset].item()
            receive_rank_id, expert_offset = divmod(expert_id, expert_num_per_rank)
            cumsum_offset = cumsum_table[local_idx, expert_id] - 1
            offset = expert_offset * moe_case.ep_world_size + send_rank_id
            token_offset = cumsum_result[receive_rank_id, offset] + cumsum_offset
            expand_x_list[receive_rank_id][token_offset] = op.x[token_id]
            info = [send_rank_id, token_id, k_offset]
            assist_info_for_combine_list[receive_rank_id][token_offset] = torch.tensor(info, dtype=torch.int32)

    return [
        DispatchOutOperands(
            expand_x_list[i], assist_info_for_combine_list[i], expert_token_nums_list[i], recv_counts_list[i])
        for i in range(moe_case.ep_world_size)]


def combine_tokens(
    moe_case: MoeCase,
    torch_data_type: torch.dtype,
    input_operands_list: list[CombineInOperands],
) -> list[CombineOutOperands]:
    # 初始化变量
    output_operands_golden_list = []
    moe_expert_tokens_list = generate_zero_tensor_list(
        [moe_case.batch_size, moe_case.topk, moe_case.hidden_size], torch_data_type, moe_case.ep_world_size,
    )

    # 发送 token
    for op in input_operands_list:
        for row_index in range(op.recv_counts.item()):
            token = op.expand_x[row_index]
            dispatch_send_rank_id, token_id, k_offset = op.assist_info_for_combine[row_index]
            moe_expert_tokens_list[dispatch_send_rank_id][token_id, k_offset] = token

    # 接收 token
    for moe_expert_tokens, op in zip(moe_expert_tokens_list, input_operands_list):
        out = torch.zeros([moe_case.batch_size, moe_case.hidden_size], dtype=torch_data_type)
        for token_id in range(moe_case.batch_size):
            if op.x_active_mask[token_id]:
                out[token_id] = (
                    op.expert_scales[token_id:(token_id + 1)].
                    matmul(moe_expert_tokens[token_id:(token_id + 1)].to(torch.float32))
                    .squeeze(0)
                    .to(torch_data_type)
                )
        output_operands_golden_list.append(CombineOutOperands(out))

    return output_operands_golden_list


def generate_moe_golden(
    moe_case: MoeCase,
    torch_data_type: torch.dtype,
    mode: str = 'dispatch_combine',
) -> Union[
    tuple[list[DispatchInOperands], list[DispatchOutOperands]],
    tuple[list[CombineInOperands], list[CombineOutOperands]],
    list[tuple[DispatchInOperands, DispatchOutOperands, CombineInOperands, CombineOutOperands]],
]:
    check_cond(mode in ('dispatch', 'combine', 'dispatch_combine'),
        f'mode must be dispatch/combine/dispatch_combine, but got {mode}')
    
    operands_list = generate_inputs(moe_case, torch_data_type)
    dispatch_input_operands_list = [
        DispatchInOperands(x, moe_expert_ids, x_active_mask)
        for x, _, moe_expert_ids, x_active_mask in operands_list
    ]
    dispatch_golden_output_operands_list = dispatch_tokens(moe_case, torch_data_type, dispatch_input_operands_list)
    
    if mode == 'dispatch':
        return dispatch_input_operands_list, dispatch_golden_output_operands_list
    
    combine_input_operands_list = [
        CombineInOperands(op.expand_x, op.assist_info_for_combine, op.recv_counts, expert_scales, x_active_mask)
        for op, (_, expert_scales, _, x_active_mask) in zip(dispatch_golden_output_operands_list, operands_list)
    ]
    combine_golden_output_operands_list = combine_tokens(moe_case, torch_data_type, combine_input_operands_list)
    
    if mode == 'combine':
        return combine_input_operands_list, combine_golden_output_operands_list
    
    return list(zip(
        dispatch_input_operands_list,
        dispatch_golden_output_operands_list,
        combine_input_operands_list,
        combine_golden_output_operands_list,
    ))


def generate_dispatch_golden(
    moe_case: MoeCase,
    torch_data_type: torch.dtype,
) -> tuple[list[DispatchInOperands], list[DispatchOutOperands]]:
    return generate_moe_golden(moe_case, torch_data_type, mode='dispatch')


def generate_combine_golden(
    moe_case: MoeCase,
    torch_data_type: torch.dtype,
) -> tuple[list[CombineInOperands], list[CombineOutOperands]]:
    return generate_moe_golden(moe_case, torch_data_type, mode='combine')


def generate_dispatch_combine_golden(
    moe_case: MoeCase,
    torch_data_type: torch.dtype,
) -> list[tuple[DispatchInOperands, DispatchOutOperands, CombineInOperands, CombineOutOperands]]:
    return generate_moe_golden(moe_case, torch_data_type, mode='dispatch_combine')


def moe_distributed_dispatch_kernel(
    moe_case: MoeCase,
    group_name: str,
) -> Callable[[pypto.Tensor, pypto.Tensor, pypto.Tensor, pypto.Tensor, pypto.Tensor, pypto.Tensor], None]:
    batch_size = moe_case.batch_size
    hidden_size = moe_case.hidden_size
    moe_expert_num = moe_case.moe_expert_num
    topk = moe_case.topk
    data_type = moe_case.data_type
    ep_world_size = moe_case.ep_world_size

    valid_ep_batch_pairs = [(2, 8), (4, 8), (8, 8), (2, 1), (4, 1), (8, 1), (16, 1)]
    allowed_str = ', '.join([f'(ep_world_size={ep}, batch_size={bs})' for ep, bs in valid_ep_batch_pairs])
    check_cond(
        (ep_world_size, batch_size) in valid_ep_batch_pairs,
        f'Invalid ep_world_size and batch_size: ep_world_size={ep_world_size}, batch_size={batch_size}. '
        f'Allowed ep_world_size and batch_size: {allowed_str}'
    )
    check_cond(hidden_size == 5120, f'hidden_size must be 5120, but got {hidden_size}')
    check_cond(moe_expert_num == 160, f'moe_expert_num must be 160, but got {moe_expert_num}')
    check_cond(topk == 8, f'topk must be 8, but got {topk}')
    check_cond(data_type == pypto.DT_BF16, f'data_type must be pypto.DT_BF16, but got {data_type}')
    check_cond(isinstance(group_name, str), f'type of group_name must be str, but got {type(group_name)}')
    check_cond(group_name.strip(), f"group_name can't be empty string")
    check_cond(
        1 <= len(group_name) < 128,
        f'the length of group_name only supports [1, 128), but got {len(group_name)}',
    )

    expand_x_row = min(topk * batch_size * ep_world_size, batch_size * moe_expert_num)
    expert_num_per_rank = moe_expert_num // ep_world_size
    info_out_size = 3
    info_size = 4
    cum_sum_row_size = align_up(moe_expert_num, 256)
    count_size = 8
    total_send_tasks = batch_size * topk

    @pypto.frontend.jit()
    def kernel(
        x: pypto.Tensor([batch_size, hidden_size], data_type, format=pypto.TileOpFormat.TILEOP_ND),
        expert_ids: pypto.Tensor([batch_size, topk], pypto.DT_INT32, format=pypto.TileOpFormat.TILEOP_ND),
        x_active_mask: pypto.Tensor([batch_size], pypto.DT_INT32, format=pypto.TileOpFormat.TILEOP_ND),
        expand_x: pypto.Tensor([expand_x_row, hidden_size], data_type, format=pypto.TileOpFormat.TILEOP_ND),
        assist_info_for_combine: pypto.Tensor(
            [expand_x_row, info_out_size],
            pypto.DT_INT32,
            format=pypto.TileOpFormat.TILEOP_ND,
        ),
        expert_token_nums: pypto.Tensor([expert_num_per_rank], pypto.DT_INT32, format=pypto.TileOpFormat.TILEOP_ND),
        recv_counts: pypto.Tensor([1], pypto.DT_INT32, format=pypto.TileOpFormat.TILEOP_ND),
    ):
        this_rank = pypto.distributed.my_symbolic_pe(group_name)

        # 创建通信共享区域
        shmem_data = pypto.distributed.create_shmem_tensor(
            group_name, ep_world_size, x.dtype, [moe_expert_num * batch_size, hidden_size])
        shmem_info = pypto.distributed.create_shmem_tensor(
            group_name, ep_world_size, pypto.DT_INT32, [moe_expert_num * batch_size, info_size])
        shmem_count = pypto.distributed.create_shmem_tensor(
            group_name, ep_world_size, pypto.DT_INT32, [cum_sum_row_size, count_size])

        # 根据专家表计算发送偏移
        pypto.set_vec_tile_shapes(total_send_tasks)
        expert_ids_flat = pypto.reshape(expert_ids, [total_send_tasks], inplace=True)
        pypto.set_vec_tile_shapes(total_send_tasks, moe_expert_num)
        one_hot_table = pypto.one_hot(expert_ids_flat, moe_expert_num)
        one_hot_table_int32 = pypto.cast(one_hot_table, pypto.DT_INT32, pypto.CastMode.CAST_TRUNC)
        cumsum_table = pypto.cumsum(one_hot_table_int32, 0)
        cumsum_table_int32 = pypto.cast(cumsum_table, pypto.DT_INT32, pypto.CastMode.CAST_TRUNC)

        for index in pypto.loop(batch_size * topk, name='MOE_DISTRIBUTED_DISPATCH_SEND', idx_name='index'):
            # 发送 token 与 info 信息
            shmem_data_out_put = pypto.tensor([1, 1], pypto.DT_INT32, 'shmem_data_out_put')
            shmem_info_out_put = pypto.tensor([1, 1], pypto.DT_INT32, 'shmem_info_out_put')
            shmem_count_out_put = pypto.tensor([1, 1], pypto.DT_INT32, 'shmem_count_out_put')
            if x_active_mask[index // topk] == 1:
                token_id = index // topk
                k_offset = index % topk
                moe_info = pypto.Tensor([1, info_size], pypto.DT_INT32)
                tensor_tile = x[token_id:token_id + 1, :]
                pypto.set_vec_tile_shapes(1, info_size)
                moe_info[0, 0] = this_rank
                moe_info[0, 1] = token_id
                moe_info[0, 2] = k_offset
                remote_expert_id = expert_ids[token_id, k_offset]
                remote_rank_id = remote_expert_id // pypto.SymbolicScalar(expert_num_per_rank)
                remote_expert_offset = remote_expert_id % expert_num_per_rank
                if index == 0:
                    token_offset = pypto.SymbolicScalar(0)
                else:
                    token_offset = cumsum_table_int32[index - 1, remote_expert_id]
                pypto.set_vec_tile_shapes(1, hidden_size)
                shmem_data_out_put[:] = pypto.distributed.shmem_put(
                    tensor_tile,
                    [(remote_expert_offset * ep_world_size + this_rank) * batch_size + token_offset, 0],
                    shmem_data,
                    remote_rank_id,
                )
                pypto.set_vec_tile_shapes(1, info_size)
                shmem_info_out_put[:] = pypto.distributed.shmem_put(
                    moe_info,
                    [(remote_expert_offset * ep_world_size + this_rank) * batch_size + token_offset, 0],
                    shmem_info,
                    remote_rank_id,
                )
                count = pypto.full([1, 1], 1, pypto.DT_INT32)
                pypto.set_vec_tile_shapes(1, 1)
                shmem_count_out_put[:] = pypto.distributed.shmem_put(
                    count,
                    [remote_expert_offset * ep_world_size + this_rank + 1, 0],
                    shmem_count,
                    remote_rank_id,
                    put_op=pypto.AtomicType.ADD,
                )
            pypto.set_vec_tile_shapes(1, hidden_size)
            pypto.distributed.shmem_signal(
                shmem_data,
                0,
                1,
                [1, hidden_size],
                [0, 0],
                target_pe=-1,
                sig_op=pypto.AtomicType.ADD,
                pred=[shmem_data_out_put, shmem_info_out_put, shmem_count_out_put],
            )

        # 接收 count 值，计算专家接收数据在输出上的偏移
        cum_sum_result = pypto.tensor([cum_sum_row_size, count_size], pypto.DT_INT32, 'cumSumResult')
        local_expert_recv_count = pypto.tensor([cum_sum_row_size, count_size], pypto.DT_INT32, 'localExpertRecvCount')
        for _ in pypto.loop(1, name='MOE_DISTRIBUTED_DISPATCH_CUM_SUM', idx_name='_'):
            pypto.set_vec_tile_shapes(1, hidden_size)
            shmem_data_wait_out = pypto.distributed.shmem_wait_until(
                shmem_data,
                0,
                batch_size * topk * ep_world_size,
                [1, hidden_size],
                [0, 0],
                cmp=pypto.OpType.EQ,
                clear_signal=True,
                pred=[x],
            )
            pypto.set_vec_tile_shapes(cum_sum_row_size, count_size)
            local_expert_recv_count = pypto.distributed.shmem_get(
                shmem_count,
                this_rank,
                [cum_sum_row_size, count_size],
                [0, 0],
                pred=[shmem_data_wait_out],
            )
            pypto.set_vec_tile_shapes(cum_sum_row_size, count_size)
            cum_sum_input = pypto.distributed.shmem_get(
                shmem_count,
                this_rank,
                [cum_sum_row_size, count_size],
                [0, 0],
                pred=[shmem_data_wait_out],
            )
            cum_sum_current = pypto.cumsum(cum_sum_input, 0)
            cum_sum_result[:] = pypto.cast(cum_sum_current, pypto.DT_INT32, pypto.CastMode.CAST_TRUNC)

            recv_count_result = cum_sum_result[expert_num_per_rank * ep_world_size, 0]
            recv_counts[0] = recv_count_result
            for expert_id in range(expert_num_per_rank):
                cum_sum_start_row = expert_id * ep_world_size + 1
                cum_sum_end_row = cum_sum_start_row + ep_world_size
                expert_valid_cnt = cum_sum_input[cum_sum_start_row:cum_sum_end_row, :]
                expert_valid_cum_sum = pypto.cumsum(expert_valid_cnt, 0)
                expert_valid_cum_sum_int32 = pypto.cast(expert_valid_cum_sum, pypto.DT_INT32, pypto.CastMode.CAST_TRUNC)
                recv_valid_result = expert_valid_cum_sum_int32[ep_world_size - 1, 0]
                expert_token_nums[expert_id] = recv_valid_result

        # 根据偏移值，做 token 与 info 的数据接收
        for index in pypto.loop(moe_expert_num, name='MOE_DISTRIBUTED_DISPATCH_RECEIVE', idx_name='index'):
            cur_count = local_expert_recv_count[index + 1, 0]
            offset = cum_sum_result[index, 0]
            pypto.set_vec_tile_shapes(batch_size, hidden_size)
            local_data_recv_count = pypto.distributed.shmem_get(
                shmem_data,
                this_rank,
                [batch_size, hidden_size],
                [index * batch_size, 0],
                valid_shape=[cur_count, hidden_size],
            )
            expand_x[offset:offset + cur_count, ...] = local_data_recv_count
            pypto.set_vec_tile_shapes(batch_size, info_out_size)
            local_info_recv_count = pypto.distributed.shmem_get(
                shmem_info,
                this_rank,
                [batch_size, info_out_size],
                [index * batch_size, 0],
                valid_shape=[cur_count, info_out_size],
            )
            assist_info_for_combine[offset:offset + cur_count, ...] = local_info_recv_count

    return kernel


def moe_distributed_dispatch(
    config: DistributedConfig,
    moe_case: MoeCase,
    input_operands: DispatchInOperands,
    golden_output_operands: DispatchOutOperands,
    logical_rank_id: int,
) -> None:
    groups = config.init_hccl_comm(logical_rank_id)
    physical_device_id = config.get_physical_device_id(logical_rank_id)
    device = torch.device(f'npu:{physical_device_id}')
    to_device(input_operands, device)
    to_device(golden_output_operands, device)
    actual_output_operands = empty_like(golden_output_operands)

    kernel = moe_distributed_dispatch_kernel(moe_case=moe_case, group_name=groups[0])
    kernel(
        input_operands.x,
        input_operands.expert_ids,
        input_operands.x_active_mask,
        actual_output_operands.expand_x,
        actual_output_operands.assist_info_for_combine,
        actual_output_operands.expert_token_nums,
        actual_output_operands.recv_counts,
    )

    recv_count = actual_output_operands.recv_counts.item()
    assert_allcolse_whit_rtol_and_atol(
        actual_output_operands.expand_x[:recv_count],
        golden_output_operands.expand_x[:recv_count]
    )
    assert_allcolse_whit_rtol_and_atol(
        actual_output_operands.assist_info_for_combine[:recv_count],
        golden_output_operands.assist_info_for_combine[:recv_count],
    )
    assert_allcolse_whit_rtol_and_atol(
        actual_output_operands.expert_token_nums,
        golden_output_operands.expert_token_nums,
    )
    assert_allcolse_whit_rtol_and_atol(actual_output_operands.recv_counts, golden_output_operands.recv_counts)


@pytest.mark.skip(reason="CI 上仅看护 test_moe_distributed_dispatch_combine")
@pytest.mark.world_size(4)
def test_moe_distributed_dispatch() -> None:
    config = DistributedConfig(world_size=4)
    mp.set_start_method('spawn', force=True)
    processes = []
    moe_case = MoeCase(8, 5120, 160, 8, pypto.DT_BF16, config.world_size)

    input_operands_list, golden_output_operands_list = generate_dispatch_golden(moe_case, torch.bfloat16)
    for input_operands, golden_output_operands, logical_rank_id in zip(
        input_operands_list, golden_output_operands_list, config.logical_ranks,
    ):
        p = mp.Process(
            target=moe_distributed_dispatch,
            args=(config, moe_case, input_operands, golden_output_operands, logical_rank_id),
        )
        p.start()
        processes.append(p)
    for i, p in enumerate(processes):
        p.join()
        if p.exitcode != 0:
            raise AssertionError(f"process {i} failed, return: {p.exitcode}")


def moe_distributed_combine_kernel(
    moe_case: MoeCase,
    group_name: str,
) -> Callable[[pypto.Tensor, pypto.Tensor, pypto.Tensor, pypto.Tensor, pypto.Tensor], None]:
    batch_size = moe_case.batch_size
    hidden_size = moe_case.hidden_size
    moe_expert_num = moe_case.moe_expert_num
    topk = moe_case.topk
    data_type = moe_case.data_type
    ep_world_size = moe_case.ep_world_size
    row = min(topk * batch_size * ep_world_size, batch_size * moe_expert_num)

    allowed_combinations = [(2, 8), (4, 8), (4, 256), (8, 8), (8, 256), (16, 1)]
    allowed_str = ', '.join([f'(ep_world_size={ep}, batch_size={bs})' for ep, bs in allowed_combinations])
    check_cond(
        (ep_world_size, batch_size) in allowed_combinations,
        f'Invalid combination: ep_world_size={ep_world_size}, batch_size={batch_size}. '
        f'Allowed combinations: {allowed_str}'
    )
    check_cond(hidden_size == 5120, f'hidden_size must be 5120, but got {hidden_size}')
    check_cond(moe_expert_num == 160, f'moe_expert_num must be 160, but got {moe_expert_num}')
    check_cond(topk == 8, f'topk must be 8, but got {topk}')
    check_cond(data_type == pypto.DT_BF16, f'data_type must be pypto.DT_BF16, but got {data_type}')
    check_cond(isinstance(group_name, str), f'type of group_name must be str, but got {type(group_name)}')
    check_cond(group_name.strip(), f"group_name can't be empty string")
    check_cond(
        1 <= len(group_name) < 128,
        f'The length of group_name only supports [1, 128), but got {len(group_name)}',
    )

    stitch_function_max_num = 128 if batch_size in (1, 8) else 10

    @pypto.frontend.jit(runtime_options={"stitch_function_max_num": stitch_function_max_num})
    def kernel(
        expand_x: pypto.Tensor([row, hidden_size], data_type, format=pypto.TileOpFormat.TILEOP_ND),
        assist_info_for_combine: pypto.Tensor([row, 3], pypto.DT_INT32, format=pypto.TileOpFormat.TILEOP_ND),
        recv_counts: pypto.Tensor([1], pypto.DT_INT32, format=pypto.TileOpFormat.TILEOP_ND),
        expert_scales: pypto.Tensor([batch_size, topk], pypto.DT_FP32, format=pypto.TileOpFormat.TILEOP_ND),
        x_active_mask: pypto.Tensor([batch_size], pypto.DT_INT32, format=pypto.TileOpFormat.TILEOP_ND),
        out: pypto.Tensor([batch_size, hidden_size], data_type, format=pypto.TileOpFormat.TILEOP_ND),
    ):
        # 创建 shmem_data
        shmem_data = pypto.distributed.create_shmem_tensor(
            group_name,
            ep_world_size,
            expand_x.dtype,
            [topk * batch_size, hidden_size],
        )

        # 发送 token
        recv_counts_scalar = recv_counts[0]
        for row_index in pypto.loop(recv_counts_scalar, name='MOE_DISTRIBUTED_SEND', idx_name='row_index'):
            rank_id = assist_info_for_combine[row_index, 0]
            token_id = assist_info_for_combine[row_index, 1]
            k_offset = assist_info_for_combine[row_index, 2]

            pypto.set_vec_tile_shapes(1, hidden_size)
            expand_x_tile = expand_x[row_index:row_index + 1, ...]
            shmem_put_out = pypto.distributed.shmem_put(
                expand_x_tile,
                [topk * token_id + k_offset, 0],
                shmem_data,
                rank_id,
            )

            pypto.distributed.shmem_signal(
                shmem_data,
                0,
                1,
                [1, hidden_size],
                [token_id, 0],
                target_pe=rank_id,
                sig_op=pypto.AtomicType.ADD,
                pred=[shmem_put_out],
            )

        # 接收 token
        my_pe = pypto.distributed.my_symbolic_pe(group_name)
        for token_id in pypto.loop(batch_size, name='MOE_DISTRIBUTED_RECEIVE', idx_name='token_id'):
            if x_active_mask[token_id] == 1:
                pypto.set_vec_tile_shapes(1, hidden_size)
                wait_until_out = pypto.distributed.shmem_wait_until(
                    shmem_data,
                    0,
                    topk,
                    [1, hidden_size],
                    [token_id, 0],
                    cmp=pypto.OpType.EQ,
                    clear_signal=True,
                    pred=[expand_x],
                )

                pypto.set_vec_tile_shapes(topk, hidden_size)
                shmem_get_out = pypto.distributed.shmem_get(
                    shmem_data,
                    my_pe,
                    [topk, hidden_size],
                    [topk * token_id, 0],
                    pred=[wait_until_out],
                )
                shmem_get_out = shmem_get_out.view([topk, hidden_size], [0, 0], valid_shape=[topk, hidden_size])

                pypto.set_vec_tile_shapes(topk // 2, hidden_size)
                shmem_get_out_fp32 = pypto.cast(shmem_get_out, pypto.DT_FP32)

                k_tile_shape = align_up(topk, 16)
                l0b_size = 65536
                n_tile_shape = l0b_size // pypto.bytes_of(pypto.DT_FP32) // k_tile_shape
                pypto.set_cube_tile_shapes([1, 1], [k_tile_shape, k_tile_shape], [n_tile_shape, n_tile_shape])
                expert_scales_tile = expert_scales[token_id:token_id + 1, :topk]
                matmul_out_fp32 = expert_scales_tile.matmul(shmem_get_out_fp32, pypto.DT_FP32)

                matmul_out_fp16 = pypto.cast(matmul_out_fp32, expand_x.dtype)

                out[token_id:(token_id + 1), :] = matmul_out_fp16

    return kernel


@allow_in_graph
def moe_distributed_combine_graph(
    expand_x: torch.Tensor,
    assist_info_for_combine: torch.Tensor,
    recv_counts: torch.Tensor,
    expert_scales: torch.Tensor,
    x_active_mask: torch.Tensor,
    moe_expert_num: int,
    group_name: str,
    world_size: int,
) -> Optional[torch.Tensor]:
    if isinstance(expand_x, fake_tensor.FakeTensor):
        return None

    batch_size = expert_scales.shape[0]
    hidden_size = expand_x.shape[1]
    topk = expert_scales.shape[1]
    dtype_from_method = getattr(pypto.converter, '_dtype_from')
    data_type = dtype_from_method(expand_x.dtype)
    moe_case = MoeCase(batch_size, hidden_size, moe_expert_num, topk, data_type, world_size)

    out = torch.empty([batch_size, hidden_size], dtype=expand_x.dtype, device=expand_x.device)
    kernel = moe_distributed_combine_kernel(moe_case, group_name)
    kernel(expand_x, assist_info_for_combine, recv_counts, expert_scales, x_active_mask, out)
    return out


def moe_distributed_combine(
    config: DistributedConfig,
    moe_case: MoeCase,
    input_operands: CombineInOperands,
    golden_output_operands: CombineOutOperands,
    logical_rank_id: int,
) -> None:
    groups = config.init_hccl_comm(logical_rank_id)
    physical_device_id = config.get_physical_device_id(logical_rank_id)
    device = torch.device(f'npu:{physical_device_id}')
    to_device(input_operands, device)
    to_device(golden_output_operands, device)
    actual_output_operands = empty_like(golden_output_operands)

    kernel = moe_distributed_combine_kernel(moe_case=moe_case, group_name=groups[0])
    kernel(
        input_operands.expand_x,
        input_operands.assist_info_for_combine,
        input_operands.recv_counts,
        input_operands.expert_scales,
        input_operands.x_active_mask,
        actual_output_operands.out,
    )

    active_indices = torch.nonzero(input_operands.x_active_mask.cpu()).squeeze(-1)
    golden_out_filtered = golden_output_operands.out.cpu()[active_indices]
    actual_out_filtered = actual_output_operands.out.cpu()[active_indices]
    assert_allclose_with_eps(golden_out_filtered, actual_out_filtered)


@pytest.mark.world_size(8)
def test_moe_distributed_combine() -> None:
    config = DistributedConfig(world_size=8)
    mp.set_start_method('spawn', force=True)
    processes = []
    moe_case = MoeCase(256, 5120, 160, 8, pypto.DT_BF16, config.world_size)

    input_operands_list, golden_output_operands_list = generate_combine_golden(moe_case, torch.bfloat16)
    for input_operands, golden_output_operands, logical_rank_id in zip(
        input_operands_list, golden_output_operands_list, config.logical_ranks,
    ):
        p = mp.Process(
            target=moe_distributed_combine,
            args=(config, moe_case, input_operands, golden_output_operands, logical_rank_id),
        )
        p.start()
        processes.append(p)
    for i, p in enumerate(processes):
        p.join()
        if p.exitcode != 0:
            raise AssertionError(f"process {i} failed, return: {p.exitcode}")


def moe_distributed_dispatch_combine(
    config: DistributedConfig,
    moe_case: MoeCase,
    operands: tuple[DispatchInOperands, DispatchOutOperands, CombineInOperands, CombineOutOperands],
    logical_rank_id: int,
    error_queue: mp.Queue,
) -> None:
    try:
        (
            dispatch_input_operands,
            dispatch_golden_output_operands,
            combine_input_operands,
            combine_golden_output_operands,
        ) = operands
        groups = config.init_hccl_comm(logical_rank_id)
        physical_device_id = config.get_physical_device_id(logical_rank_id)
        device = torch.device(f'npu:{physical_device_id}')
        to_device(dispatch_input_operands, device)
        to_device(dispatch_golden_output_operands, device)
        to_device(combine_input_operands, device)
        to_device(combine_golden_output_operands, device)
        dispatch_actual_output_operands = empty_like(dispatch_golden_output_operands)
        combine_actual_output_operands = empty_like(combine_golden_output_operands)

        dispatch_kernel = moe_distributed_dispatch_kernel(moe_case=moe_case, group_name=groups[0])
        dispatch_kernel(
            dispatch_input_operands.x,
            dispatch_input_operands.expert_ids,
            dispatch_input_operands.x_active_mask,
            dispatch_actual_output_operands.expand_x,
            dispatch_actual_output_operands.assist_info_for_combine,
            dispatch_actual_output_operands.expert_token_nums,
            dispatch_actual_output_operands.recv_counts,
        )

        combine_kernel = moe_distributed_combine_kernel(moe_case=moe_case, group_name=groups[0])
        combine_kernel(
            dispatch_actual_output_operands.expand_x,
            dispatch_actual_output_operands.assist_info_for_combine,
            dispatch_actual_output_operands.recv_counts,
            combine_input_operands.expert_scales,
            combine_input_operands.x_active_mask,
            combine_actual_output_operands.out,
        )

        active_indices = torch.nonzero(combine_input_operands.x_active_mask.cpu()).squeeze(-1)
        golden_out_filtered = combine_golden_output_operands.out.cpu()[active_indices]
        actual_out_filtered = combine_actual_output_operands.out.cpu()[active_indices]
        assert_allclose_with_eps(golden_out_filtered, actual_out_filtered)
    except Exception as e:
        if error_queue is not None:
            error_queue.put((logical_rank_id, str(e), traceback.format_exc()))
        raise


@pytest.mark.world_size(4)
def test_moe_distributed_dispatch_combine() -> None:
    config = DistributedConfig(world_size=4)
    mp.set_start_method('spawn', force=True)
    processes = []
    moe_case = MoeCase(8, 5120, 160, 8, pypto.DT_BF16, config.world_size)

    operands_list = generate_dispatch_combine_golden(moe_case, torch.bfloat16)
    error_queue = mp.Queue()
    for operands, logical_rank_id in zip(operands_list, config.logical_ranks):
        p = mp.Process(
            target=moe_distributed_dispatch_combine,
            args=(config, moe_case, operands, logical_rank_id, error_queue),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    collect_process_errors(processes, error_queue)


if __name__ == '__main__':
    test_moe_distributed_dispatch_combine()