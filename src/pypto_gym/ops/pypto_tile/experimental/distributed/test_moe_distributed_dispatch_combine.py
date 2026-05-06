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
from typing import Callable
import traceback

import multiprocessing as mp
import numpy as np
import pytest
import torch
import torch.nn.functional as F

import pypto

from distributed_config import DistributedConfig, collect_process_errors

TensorList = list[torch.Tensor]

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


@dataclasses.dataclass(frozen=True)
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


@dataclasses.dataclass(frozen=True)
class MoeDispatchOperands:
    x: torch.Tensor
    expert_ids: torch.Tensor
    expand_x_golden: torch.Tensor
    assist_info_for_combine_golden: torch.Tensor
    expert_token_nums_golden: torch.Tensor
    recv_counts_golden: torch.Tensor


@dataclasses.dataclass(frozen=True)
class MoeDispatchOperandLists:
    x_list: TensorList
    expert_ids_list: TensorList
    expand_x_golden_list: TensorList
    assist_info_for_combine_golden_list: TensorList
    expert_token_nums_golden_list: TensorList
    recv_counts_golden_list: TensorList


@dataclasses.dataclass(frozen=True)
class MoeCombineOperands:
    expand_x: torch.Tensor
    assist_info_for_combine: torch.Tensor
    recv_counts: torch.Tensor
    expert_scales: torch.Tensor
    out_golden: torch.Tensor


@dataclasses.dataclass(frozen=True)
class MoeCombineOperandLists:
    expand_x_list: TensorList
    assist_info_for_combine_list: TensorList
    recv_counts_list: TensorList
    expert_scales_list: TensorList
    out_golden_list: TensorList = None


@dataclasses.dataclass(frozen=True)
class MoeDispatchCombineOperands:
    x: torch.Tensor
    expert_ids: torch.Tensor
    expert_scales: torch.Tensor
    expand_x_golden: torch.Tensor
    assist_info_for_combine_golden: torch.Tensor
    expert_token_nums_golden: torch.Tensor
    recv_counts_golden: torch.Tensor
    out_golden: torch.Tensor


@dataclasses.dataclass(frozen=True)
class MoeDispatchCombineOperandLists:
    x_list: TensorList
    expert_ids_list: TensorList
    expert_scales_list: TensorList
    expand_x_golden_list: TensorList
    assist_info_for_combine_golden_list: TensorList
    expert_token_nums_golden_list: TensorList
    recv_counts_golden_list: TensorList
    out_golden_list: TensorList


def generate_random_tensor(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    float_dtypes = (torch.float16, torch.float32, torch.float64, torch.bfloat16)
    int_dtypes = (torch.int8, torch.int16, torch.int32, torch.int64)
    if dtype in float_dtypes:
        return torch.randn(shape, dtype=dtype)
    elif dtype in int_dtypes:
        return torch.randint(-10, 10, size=shape, dtype=dtype)
    else:
        raise ValueError(f'Unsupported dtype: {dtype}. Supported: {float_dtypes + int_dtypes}')


def generate_inputs(moe_case: MoeCase, torch_data_type: torch.dtype) -> tuple[TensorList, TensorList, TensorList]:
    x_list = [
        generate_random_tensor((moe_case.batch_size, moe_case.hidden_size), torch_data_type)
        for _ in range(moe_case.ep_world_size)
    ]
    moe_expert_ids_list = []
    topk_expert_scales_list = []

    for _ in range(moe_case.ep_world_size):
        expert_scores = generate_random_tensor((moe_case.batch_size, moe_case.moe_expert_num), torch.float32)
        topk_expert_scores, moe_expert_ids = expert_scores.topk(k=moe_case.topk)
        topk_expert_scales = topk_expert_scores.softmax(dim=-1)
        topk_expert_scales_list.append(topk_expert_scales)
        moe_expert_ids = moe_expert_ids.to(dtype=torch.int32)
        moe_expert_ids_list.append(moe_expert_ids)

    return x_list, moe_expert_ids_list, topk_expert_scales_list


def create_tensor_on_npu(golden_tensor, device_id):
    return torch.empty(golden_tensor.shape,
                    dtype=golden_tensor.dtype,
                    device=f'npu:{device_id}')


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
    x_list: TensorList,
    moe_expert_ids_list: TensorList,
) -> tuple[TensorList, TensorList, TensorList, TensorList]:
    # 初始化变量
    moe_expert_num_per_rank = get_moe_expert_num_per_rank(moe_case)
    expand_x_per_expert = [[[] for _ in range(moe_expert_num_per_rank)] for _ in range(moe_case.ep_world_size)]
    assist_info_for_combine_per_expert = [
        [[] for _ in range(moe_expert_num_per_rank)]
        for _ in range(moe_case.ep_world_size)
    ]

    # 发送 token
    for sending_rank_id, (x, moe_expert_ids) in enumerate(zip(x_list, moe_expert_ids_list)):
        for token_id, (token, topk_moe_expert_ids) in enumerate(zip(x, moe_expert_ids)):
            for k_offset, moe_expert_id in enumerate(topk_moe_expert_ids):
                receiving_expert_id = moe_expert_id.item()
                receiving_rank_id, expert_offset = get_moe_expert_rank_id_and_expert_offset(
                    receiving_expert_id, moe_expert_num_per_rank)
                expand_x_per_expert[receiving_rank_id][expert_offset].append(token)
                assist_info_for_combine_per_expert[receiving_rank_id][expert_offset].append(
                    (sending_rank_id, token_id, k_offset))

    # 接收 token
    row = get_dispatch_output_row(moe_case)
    combine_info_col = 3
    expand_x_per_rank = []
    assist_info_for_combine_per_rank = []
    expert_token_nums_per_rank = []
    recv_counts_per_rank = []
    for logical_rank_id in range(moe_case.ep_world_size):
        fixed_shape_expand_x = torch.zeros((row, moe_case.hidden_size), dtype=torch_data_type)
        fixed_shape_assist_info_for_combine = torch.zeros((row, combine_info_col), dtype=torch.int32)
        expert_token_nums = torch.zeros([moe_expert_num_per_rank], dtype=torch.int32)
        offset = 0

        for expert_offset in range(moe_expert_num_per_rank):
            tokens = expand_x_per_expert[logical_rank_id][expert_offset]
            if tokens:
                actual_expand_x = torch.stack(tokens, dim=0)
                end = offset + actual_expand_x.size(0)
                fixed_shape_expand_x[offset:end] = actual_expand_x
                actual_assist_info_for_combine = torch.tensor(
                    assist_info_for_combine_per_expert[logical_rank_id][expert_offset],
                    dtype=torch.int32,
                )
                fixed_shape_assist_info_for_combine[offset:end, -3:] = actual_assist_info_for_combine
                expert_token_nums[expert_offset] = actual_expand_x.size(0)
                offset = end

        expand_x_per_rank.append(fixed_shape_expand_x)
        assist_info_for_combine_per_rank.append(fixed_shape_assist_info_for_combine)
        expert_token_nums_per_rank.append(expert_token_nums)
        recv_counts_per_rank.append(expert_token_nums.sum(dtype=torch.int32).unsqueeze(0))

    return expand_x_per_rank, assist_info_for_combine_per_rank, expert_token_nums_per_rank, recv_counts_per_rank


def combine_tokens(
    moe_case: MoeCase,
    torch_data_type: torch.dtype,
    operand_lists: MoeCombineOperandLists
) -> TensorList:
    # 初始化变量
    out_golden_list = []
    moe_expert_tokens_list = [
        torch.zeros([moe_case.batch_size, moe_case.topk, moe_case.hidden_size], dtype=torch_data_type)
        for _ in range(moe_case.ep_world_size)
    ]

    # 发送 token
    for expand_x, assist_info_for_combine, recv_counts in zip(
        operand_lists.expand_x_list, operand_lists.assist_info_for_combine_list, operand_lists.recv_counts_list
    ):
        for row_index in range(recv_counts.item()):
            token = expand_x[row_index]
            dispatch_sending_rank_id, token_id, k_offset = assist_info_for_combine[row_index]
            moe_expert_tokens_list[dispatch_sending_rank_id][token_id, k_offset] = token

    # 接收 token
    for moe_expert_tokens, expert_scales in zip(moe_expert_tokens_list, operand_lists.expert_scales_list):
        out_golden = (
            expert_scales.unsqueeze(1).
            matmul(moe_expert_tokens.to(torch.float32))
            .squeeze(1)
            .to(torch.bfloat16)
        )
        out_golden_list.append(out_golden)

    return out_golden_list


def generate_dispatch_golden(moe_case: MoeCase, torch_data_type: torch.dtype) -> MoeDispatchOperandLists:
    x_list, moe_expert_ids_list, _ = generate_inputs(moe_case, torch_data_type)
    (
        expand_x_golden_list,
        assist_info_for_combine_golden_list,
        expert_token_nums_golden_list,
        recv_counts_golden_list,
    ) = dispatch_tokens(moe_case, torch_data_type, x_list, moe_expert_ids_list)
    return MoeDispatchOperandLists(
        x_list,
        moe_expert_ids_list,
        expand_x_golden_list,
        assist_info_for_combine_golden_list,
        expert_token_nums_golden_list,
        recv_counts_golden_list,
    )


def generate_combine_golden(
    moe_case: MoeCase,
    torch_data_type: torch.dtype,
) -> MoeCombineOperandLists:
    x_list, moe_expert_ids_list, expert_scales_list = generate_inputs(moe_case, torch_data_type)
    (
        expand_x_list,
        assist_info_for_combine_list,
        _,
        recv_counts_list,
    ) = dispatch_tokens(moe_case, torch_data_type, x_list, moe_expert_ids_list)
    operand_lists = MoeCombineOperandLists(
        expand_x_list,
        assist_info_for_combine_list,
        recv_counts_list,
        expert_scales_list,
    )
    out_golden_list = combine_tokens(moe_case, torch_data_type, operand_lists)
    return MoeCombineOperandLists(
        expand_x_list,
        assist_info_for_combine_list,
        recv_counts_list,
        expert_scales_list,
        out_golden_list,
    )


def generate_dispatch_combine_golden(
    moe_case: MoeCase,
    torch_data_type: torch.dtype,
) -> MoeDispatchCombineOperandLists:
    x_list, moe_expert_ids_list, expert_scales_list = generate_inputs(moe_case, torch_data_type)
    (
        expand_x_list,
        assist_info_for_combine_list,
        expert_token_nums_golden_list,
        recv_counts_list,
    ) = dispatch_tokens(moe_case, torch_data_type, x_list, moe_expert_ids_list)

    operand_lists = MoeCombineOperandLists(
        expand_x_list,
        assist_info_for_combine_list,
        recv_counts_list,
        expert_scales_list,
    )
    out_golden_list = combine_tokens(moe_case, torch_data_type, operand_lists)

    return MoeDispatchCombineOperandLists(
        x_list,
        moe_expert_ids_list,
        expert_scales_list,
        expand_x_list,
        assist_info_for_combine_list,
        expert_token_nums_golden_list,
        recv_counts_list,
        out_golden_list,
    )


def dispatch_calc_occurrences(
    expert_ids: pypto.Tensor, expert_id: pypto.SymbolicScalar, calc_index: int,
) -> pypto.Tensor:
    pypto.set_vec_tile_shapes(1, expert_ids.shape[1])
    expert_ids_dup = pypto.full(expert_ids.shape, expert_id, pypto.DT_INT32)
    sub_result = pypto.sub(expert_ids_dup, expert_ids)
    sub_result_fp32 = pypto.cast(sub_result, pypto.DT_FP32, pypto.CastMode.CAST_TRUNC)
    abs_sub_result = pypto.abs(sub_result_fp32)
    abs_sub_result_int32 = pypto.cast(abs_sub_result, pypto.DT_INT32, pypto.CastMode.CAST_TRUNC)
    count_of_equals = pypto.clip(abs_sub_result_int32, 0, 1)
    cum_sum_offset = pypto.cumsum(count_of_equals, 1)
    cum_sum_offset_int32 = pypto.cast(cum_sum_offset, pypto.DT_INT32, pypto.CastMode.CAST_TRUNC)
    expert_offset_result = pypto.sub(cum_sum_offset_int32, calc_index)
    expert_offset_result_fp32 = pypto.cast(expert_offset_result, pypto.DT_FP32, pypto.CastMode.CAST_TRUNC)
    expert_offset_abs_fp32 = pypto.abs(expert_offset_result_fp32)
    expert_offset = pypto.cast(expert_offset_abs_fp32, pypto.DT_INT32, pypto.CastMode.CAST_TRUNC)
    return expert_offset


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

    check_cond(batch_size == 8, f'batch_size must be 8, but got {batch_size}')
    check_cond(hidden_size == 5120, f'hidden_size must be 5120, but got {hidden_size}')
    check_cond(moe_expert_num == 160, f'moe_expert_num must be 160, but got {moe_expert_num}')
    check_cond(topk == 8, f'topk must be 8, but got {topk}')
    check_cond(data_type == pypto.DT_BF16, f'data_type must be pypto.DT_BF16, but got {data_type}')
    check_cond(ep_world_size in (4, 8), f'ep_world_size must be 4 or 8, but got {ep_world_size}')
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

    @pypto.frontend.jit()
    def kernel(
        x: pypto.Tensor([batch_size, hidden_size], data_type, format=pypto.TileOpFormat.TILEOP_ND),
        expert_ids: pypto.Tensor([batch_size, topk], pypto.DT_INT32, format=pypto.TileOpFormat.TILEOP_ND),
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
        pypto.set_vec_tile_shapes(1, batch_size * topk)
        expert_ids_vec = pypto.reshape(expert_ids, [1, batch_size * topk], inplace=True)

        for _ in pypto.loop(1, name='MOE_DISTRIBUTED_DISPATCH_SEND', idx_name='_'):
            # 发送 token 与 info 信息
            for index in range(batch_size * topk):
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
                token_offset_result = dispatch_calc_occurrences(
                    expert_ids_vec,
                    remote_expert_id,
                    index,
                )
                token_index = max(index - 1, 0)
                token_offset = token_offset_result[0, token_index]
                pypto.set_vec_tile_shapes(1, hidden_size)
                shmem_data_out_put = pypto.distributed.shmem_put(
                    tensor_tile,
                    [(remote_expert_offset * ep_world_size + this_rank) * batch_size + token_offset, 0],
                    shmem_data,
                    remote_rank_id,
                )
                pypto.set_vec_tile_shapes(1, info_size)
                shmem_info_out_put = pypto.distributed.shmem_put(
                    moe_info,
                    [(remote_expert_offset * ep_world_size + this_rank) * batch_size + token_offset, 0],
                    shmem_info,
                    remote_rank_id,
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
                    pred=[shmem_data_out_put, shmem_info_out_put],
                )
            # 发送每个专家的 token 有效发送数目
            for expert_id in range(moe_expert_num):
                expert_offset = dispatch_calc_occurrences(
                    expert_ids_vec,
                    expert_id,
                    batch_size * topk,
                )
                remote_rank_id = expert_id // expert_num_per_rank
                remote_expert_offset = expert_id % expert_num_per_rank
                total_offset_tile = expert_offset[:, -1:]
                pypto.set_vec_tile_shapes(1, 1)
                shmem_put_out = pypto.distributed.shmem_put(
                    total_offset_tile,
                    [remote_expert_offset * ep_world_size + this_rank + 1, 0],
                    shmem_count,
                    remote_rank_id,
                    pred=[total_offset_tile],
                )
                pypto.set_vec_tile_shapes(1, count_size)
                pypto.distributed.shmem_signal(
                    shmem_count,
                    0,
                    1,
                    [1, count_size],
                    [0, 0],
                    target_pe=remote_rank_id,
                    sig_op=pypto.AtomicType.ADD, pred=[shmem_put_out],
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
            pypto.set_vec_tile_shapes(1, count_size)
            shmem_count_wait_out = pypto.distributed.shmem_wait_until(
                shmem_count,
                0,
                moe_expert_num,
                [1, count_size],
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
                pred=[shmem_data_wait_out, shmem_count_wait_out],
            )
            pypto.set_vec_tile_shapes(cum_sum_row_size, count_size)
            cum_sum_input = pypto.distributed.shmem_get(
                shmem_count,
                this_rank,
                [cum_sum_row_size, count_size],
                [0, 0],
                pred=[shmem_data_wait_out, shmem_count_wait_out],
            )
            cum_sum_current = pypto.cumsum(cum_sum_input, 0)
            cum_sum_result = pypto.cast(cum_sum_current, pypto.DT_INT32, pypto.CastMode.CAST_TRUNC)

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
        for index in pypto.loop(moe_expert_num, name='MOE_DISTRIBUTED_DISPATCH_RECEIVE', idx_name='_'):
            cur_count = local_expert_recv_count[index + 1, 0]
            offset = cum_sum_result[index, 0]
            pypto.set_vec_tile_shapes(batch_size, hidden_size)
            local_data_recv_count = pypto.distributed.shmem_get(
                shmem_data,
                this_rank,
                [batch_size, hidden_size],
                [index * batch_size, 0],
                pred=[cum_sum_result],
                valid_shape=[cur_count, hidden_size],
            )
            expand_x[offset:offset + cur_count, ...] = local_data_recv_count
            pypto.set_vec_tile_shapes(batch_size, info_out_size)
            local_info_recv_count = pypto.distributed.shmem_get(
                shmem_info,
                this_rank,
                [batch_size, info_out_size],
                [index * batch_size, 0],
                pred=[cum_sum_result],
                valid_shape=[cur_count, info_out_size],
            )
            assist_info_for_combine[offset:offset + cur_count, ...] = local_info_recv_count

    return kernel


def moe_distributed_dispatch(
    config: DistributedConfig,
    moe_case: MoeCase,
    operands: MoeDispatchOperands,
    logical_rank_id: int,
) -> None:
    groups = config.init_hccl_comm(logical_rank_id)

    x = operands.x
    expert_ids = operands.expert_ids
    expand_x_golden = operands.expand_x_golden
    assist_info_for_combine_golden = operands.assist_info_for_combine_golden
    expert_token_nums_golden = operands.expert_token_nums_golden
    recv_counts_golden = operands.recv_counts_golden

    x.share_memory_()
    expert_ids.share_memory_()
    expand_x_golden.share_memory_()
    assist_info_for_combine_golden.share_memory_()
    expert_token_nums_golden.share_memory_()
    recv_counts_golden.share_memory_()

    physical_device_id = config.get_physical_device_id(logical_rank_id)
    x = x.to(f'npu:{physical_device_id}')
    expert_ids = expert_ids.to(f'npu:{physical_device_id}')
    expand_x_golden = expand_x_golden.to(f'npu:{physical_device_id}')
    assist_info_for_combine_golden = assist_info_for_combine_golden.to(f'npu:{physical_device_id}')
    expert_token_nums_golden = expert_token_nums_golden.to(f'npu:{physical_device_id}')
    recv_counts_golden = recv_counts_golden.to(f'npu:{physical_device_id}')

    expand_x_actual = create_tensor_on_npu(expand_x_golden, physical_device_id)
    assist_info_for_combine_actual = create_tensor_on_npu(assist_info_for_combine_golden, physical_device_id)
    expert_token_nums_actual = create_tensor_on_npu(expert_token_nums_golden, physical_device_id)
    recv_counts_actual = create_tensor_on_npu(recv_counts_golden, physical_device_id)

    kernel = moe_distributed_dispatch_kernel(moe_case=moe_case, group_name=groups[0])
    kernel(x, expert_ids, expand_x_actual, assist_info_for_combine_actual, expert_token_nums_actual, recv_counts_actual)

    for out, act in [
        (expand_x_actual, expand_x_golden),
        (assist_info_for_combine_actual, assist_info_for_combine_golden),
        (expert_token_nums_actual, expert_token_nums_golden),
        (recv_counts_actual, recv_counts_golden),
    ]:
        assert_allcolse_whit_rtol_and_atol(out, act)


@pytest.mark.skip(reason="功能未实现，暂不执行")
@pytest.mark.world_size(4)
def test_moe_distributed_dispatch() -> None:
    config = DistributedConfig(world_size=4)
    mp.set_start_method('spawn', force=True)
    processes = []
    moe_case = MoeCase(8, 5120, 160, 8, pypto.DT_BF16, config.world_size)

    operand_lists = generate_dispatch_golden(moe_case, torch.bfloat16)
    for (
        x,
        moe_expert_ids,
        expand_x_golden,
        assist_info_for_combine_golden,
        expert_token_nums_golden,
        recv_counts_golden,
        logical_rank_id,
    ) in zip(
        operand_lists.x_list,
        operand_lists.expert_ids_list,
        operand_lists.expand_x_golden_list,
        operand_lists.assist_info_for_combine_golden_list,
        operand_lists.expert_token_nums_golden_list,
        operand_lists.recv_counts_golden_list,
        config.logical_ranks,
    ):
        operands = MoeDispatchOperands(
            x,
            moe_expert_ids,
            expand_x_golden,
            assist_info_for_combine_golden,
            expert_token_nums_golden,
            recv_counts_golden,
        )
        p = mp.Process(target=moe_distributed_dispatch, args=(config, moe_case, operands, logical_rank_id))
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

    check_cond(batch_size == 8 or batch_size == 256, f'batch_size must be 8 or 256, but got {batch_size}')
    check_cond(hidden_size == 5120, f'hidden_size must be 5120, but got {hidden_size}')
    check_cond(moe_expert_num == 160, f'moe_expert_num must be 160, but got {moe_expert_num}')
    check_cond(topk == 8, f'topk must be 8, but got {topk}')
    check_cond(data_type == pypto.DT_BF16, f'data_type must be pypto.DT_BF16, but got {data_type}')
    check_cond(ep_world_size in (4, 8), f'ep_world_size must be 4 or 8, but got {ep_world_size}')
    check_cond(isinstance(group_name, str), f'type of group_name must be str, but got {type(group_name)}')
    check_cond(group_name.strip(), f"group_name can't be empty string")
    check_cond(
        1 <= len(group_name) < 128,
        f'The length of group_name only supports [1, 128), but got {len(group_name)}',
    )

    @pypto.frontend.jit()
    def kernel(
        expand_x: pypto.Tensor([row, hidden_size], data_type, format=pypto.TileOpFormat.TILEOP_ND),
        assist_info_for_combine: pypto.Tensor([row, 3], pypto.DT_INT32, format=pypto.TileOpFormat.TILEOP_ND),
        recv_counts: pypto.Tensor([1], pypto.DT_INT32, format=pypto.TileOpFormat.TILEOP_ND),
        expert_scales: pypto.Tensor([batch_size, topk], pypto.DT_FP32, format=pypto.TileOpFormat.TILEOP_ND),
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

            out[token_id:, :] = matmul_out_fp16

    return kernel


def moe_distributed_combine(
    config: DistributedConfig,
    moe_case: MoeCase,
    operands: MoeCombineOperands,
    logical_rank_id: int,
) -> None:
    groups = config.init_hccl_comm(logical_rank_id)

    expand_x = operands.expand_x
    assist_info_for_combine = operands.assist_info_for_combine
    recv_counts = operands.recv_counts
    expert_scales = operands.expert_scales
    out_golden = operands.out_golden

    expand_x.share_memory_()
    assist_info_for_combine.share_memory_()
    recv_counts.share_memory_()
    expert_scales.share_memory_()
    out_golden.share_memory_()

    physical_device_id = config.get_physical_device_id(logical_rank_id)
    expand_x = expand_x.to(f'npu:{physical_device_id}')
    assist_info_for_combine = assist_info_for_combine.to(f'npu:{physical_device_id}')
    recv_counts = recv_counts.to(f'npu:{physical_device_id}')
    expert_scales = expert_scales.to(f'npu:{physical_device_id}')
    out_golden = out_golden.to(f'npu:{physical_device_id}')
    out = create_tensor_on_npu(out_golden, physical_device_id)

    kernel = moe_distributed_combine_kernel(moe_case=moe_case, group_name=groups[0])
    kernel(expand_x, assist_info_for_combine, recv_counts, expert_scales, out)

    assert_allclose_with_eps(out_golden.cpu(), out.cpu())


@pytest.mark.skip(reason="功能未实现，暂不执行")
@pytest.mark.world_size(4)
def test_moe_distributed_combine() -> None:
    config = DistributedConfig(world_size=4)
    mp.set_start_method('spawn', force=True)
    processes = []
    moe_case = MoeCase(8, 5120, 160, 8, pypto.DT_BF16, config.world_size)

    operand_lists = generate_combine_golden(moe_case, torch.bfloat16)
    for expand_x, assist_info_for_combine, recv_counts, expert_scales, out_golden, logical_rank_id in zip(
        operand_lists.expand_x_list,
        operand_lists.assist_info_for_combine_list,
        operand_lists.recv_counts_list,
        operand_lists.expert_scales_list,
        operand_lists.out_golden_list,
        config.logical_ranks,
    ):
        operands = MoeCombineOperands(expand_x, assist_info_for_combine, recv_counts, expert_scales, out_golden)
        p = mp.Process(target=moe_distributed_combine, args=(config, moe_case, operands, logical_rank_id))
        p.start()
        processes.append(p)
    for i, p in enumerate(processes):
        p.join()
        if p.exitcode != 0:
            raise AssertionError(f"process {i} failed, return: {p.exitcode}")


def moe_distributed_dispatch_combine(
    config: DistributedConfig,
    moe_case: MoeCase,
    operands: MoeDispatchCombineOperands,
    logical_rank_id: int,
    error_queue: mp.Queue,
) -> None:
    try:
        groups = config.init_hccl_comm(logical_rank_id)

        x = operands.x
        expert_ids = operands.expert_ids
        expand_x_golden = operands.expand_x_golden
        assist_info_for_combine_golden = operands.assist_info_for_combine_golden
        expert_token_nums_golden = operands.expert_token_nums_golden
        recv_counts_golden = operands.recv_counts_golden

        physical_device_id = config.get_physical_device_id(logical_rank_id)
        x = x.to(f'npu:{physical_device_id}')
        expert_ids = expert_ids.to(f'npu:{physical_device_id}')
        expand_x_golden = expand_x_golden.to(f'npu:{physical_device_id}')
        assist_info_for_combine_golden = assist_info_for_combine_golden.to(f'npu:{physical_device_id}')
        expert_token_nums_golden = expert_token_nums_golden.to(f'npu:{physical_device_id}')
        recv_counts_golden = recv_counts_golden.to(f'npu:{physical_device_id}')

        expand_x_actual = create_tensor_on_npu(expand_x_golden, physical_device_id)
        assist_info_for_combine_actual = create_tensor_on_npu(assist_info_for_combine_golden, physical_device_id)
        expert_token_nums_actual = create_tensor_on_npu(expert_token_nums_golden, physical_device_id)
        recv_counts_actual = create_tensor_on_npu(recv_counts_golden, physical_device_id)

        kernel = moe_distributed_dispatch_kernel(moe_case=moe_case, group_name=groups[0])
        kernel(
            x, expert_ids, expand_x_actual, assist_info_for_combine_actual,
            expert_token_nums_actual, recv_counts_actual
        )

        assist_info_for_combine = assist_info_for_combine_actual[:, -3:].cpu()
        assist_info_for_combine = assist_info_for_combine.to(f'npu:{physical_device_id}')
        expert_scales = operands.expert_scales
        expert_scales = expert_scales.to(f'npu:{physical_device_id}')
        out_golden = operands.out_golden
        out_golden = out_golden.to(f'npu:{physical_device_id}')
        out = create_tensor_on_npu(out_golden, physical_device_id)

        kernel = moe_distributed_combine_kernel(moe_case=moe_case, group_name=groups[0])
        kernel(expand_x_actual, assist_info_for_combine, recv_counts_actual, expert_scales, out)

        assert_allclose_with_eps(out_golden.cpu(), out.cpu())
    except Exception as e:
        if error_queue is not None:
            error_queue.put((logical_rank_id, str(e), traceback.format_exc()))
        raise


@pytest.mark.skip(reason="精度问题，暂不执行")
@pytest.mark.world_size(4)
def test_moe_distributed_dispatch_combine() -> None:
    config = DistributedConfig(world_size=4)
    mp.set_start_method('spawn', force=True)
    processes = []
    moe_case = MoeCase(8, 5120, 160, 8, pypto.DT_BF16, config.world_size)

    operand_lists = generate_dispatch_combine_golden(moe_case, torch.bfloat16)
    error_queue = mp.Queue()
    for (
        x,
        moe_expert_ids,
        expert_scale,
        expand_x_golden,
        assist_info_for_combine_golden,
        expert_token_nums_golden,
        recv_counts_golden,
        out_golden,
        logical_rank_id,
    ) in zip(
        operand_lists.x_list,
        operand_lists.expert_ids_list,
        operand_lists.expert_scales_list,
        operand_lists.expand_x_golden_list,
        operand_lists.assist_info_for_combine_golden_list,
        operand_lists.expert_token_nums_golden_list,
        operand_lists.recv_counts_golden_list,
        operand_lists.out_golden_list,
        config.logical_ranks,
    ):
        operands = MoeDispatchCombineOperands(
            x,
            moe_expert_ids,
            expert_scale,
            expand_x_golden,
            assist_info_for_combine_golden,
            expert_token_nums_golden,
            recv_counts_golden,
            out_golden,
        )
        p = mp.Process(target=moe_distributed_dispatch_combine,
            args=(config, moe_case, operands, logical_rank_id, error_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    collect_process_errors(processes, error_queue)


if __name__ == '__main__':
    test_moe_distributed_dispatch_combine()
