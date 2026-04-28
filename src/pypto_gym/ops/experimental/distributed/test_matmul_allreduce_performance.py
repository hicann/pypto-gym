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
GLM-4.5 MatMul AllReduce Module for performance

This module implements a fused matmul and all-reduce operation for large-scale distributed models.
It efficiently combines computation and communication, reducing memory overhead and accelerating training.

Main Functions:
    - matmul_allreduce: Main function for fused matmul and all-reduce computation
"""

import logging
import multiprocessing as mp
import os
import csv
from pathlib import Path
import time
import statistics

import numpy as np
import pytest
import torch

import pypto

from distributed_config import DistributedConfig
from swimlane_analyzer import SwimlaneAnalyzer

logger = logging.getLogger(__name__)


def _get_soc_version():
    """获取 soc version"""
    try:
        import torch_npu
        return torch_npu.npu.get_soc_version()
    except Exception:
        return None


@pypto.frontend.jit(
    debug_options={"runtime_debug_mode": 1},
    runtime_options={"stitch_function_max_num": 128},
)
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
            group_name, world_size, pypto.DT_FP32, shmem_shape)
        shmem_barrier_signal = pypto.distributed.create_shmem_signal(group_name, world_size)
        my_pe = pypto.distributed.my_symbolic_pe(group_name)
        for _ in pypto.loop(1, name="LOOP_MM_AR_ARMS_L0", idx_name="_"):
            in_tensor_tile = pypto.view(
                in_tensor, (view_row_shape, in_tensor.shape[1]), [bs_idx * view_row_shape, 0],
                valid_shape=[(batch_size - bs_idx * view_row_shape).min(view_row_shape), in_tensor.shape[1]])

            # 2. clear data
            pypto.set_vec_tile_shapes(view_row_shape, hidden_size)
            data_clear_out = pypto.distributed.shmem_clear_data(
                shmem_tensor, shmem_shape, [0, 0], pred=[in_tensor_tile])
            signal_clear_out = pypto.distributed.shmem_clear_signal(
                shmem_tensor, pred=[in_tensor_tile])
            barrier_out = pypto.distributed.shmem_barrier_all(
                shmem_barrier_signal, [data_clear_out, signal_clear_out])

            # 3. matmul
            pypto.set_cube_tile_shapes([8, 8], [128, 256], [256, 512])
            matmul_result = pypto.matmul(in_tensor_tile, matmul_weight, pypto.DT_FP32, b_trans=True)

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
            residual_tile_fp32 = pypto.cast(residual_tile, pypto.DT_FP32)
            add_out = pypto.add(all_reduce_out, residual_tile_fp32)

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


def generate_golden_data(world_size: int):
    # 设置参数
    batch_size = 8
    attn_dim_per_tp = 1536
    hidden_size = 5120
    torch.manual_seed(42)

    #构造每张卡上需要的数据
    input_datas = []
    for _ in range(world_size):
        in_tensor = torch.randn((batch_size, attn_dim_per_tp), dtype=torch.bfloat16).share_memory_()
        matmul_weight = torch.randn((hidden_size, attn_dim_per_tp), dtype=torch.bfloat16).share_memory_()
        residual = torch.randn((batch_size, hidden_size), dtype=torch.bfloat16).share_memory_()
        gamma = torch.randn((hidden_size), dtype=torch.bfloat16).share_memory_()
        bias = torch.randn((hidden_size), dtype=torch.bfloat16).share_memory_()
        eps = 1e-5
        input_data = [in_tensor, matmul_weight, residual, gamma, bias, eps]
        input_datas.append(input_data)
    output_datas = matmul_allreduce_add_rmsnorm_result_golden(batch_size, hidden_size, input_datas)
    return input_datas, output_datas


def matmul_allreduce_add_rmsnorm_result_golden(batch_size, num, input_datas):
    output_datas = []
    # 计算 matmul & allreduce 结果， 该结果所有卡上一致
    matmul_allreduce_result_fp32 = torch.zeros((batch_size, num), dtype=torch.float32)
    for input_data in input_datas:
        in_tensor, matmul_weight = input_data[:2]
        matmul_result = torch.matmul(in_tensor.to(torch.float32), matmul_weight.to(torch.float32).T)
        matmul_allreduce_result_fp32 += matmul_result

    # 计算各卡上add_rmsnorm之后的结果
    for input_data in input_datas:
        residual, gamma, bias, eps = input_data[-4:]
        res_add = residual.to(torch.float32) + matmul_allreduce_result_fp32
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
):
    groups = config.init_hccl_comm(logical_rank_id)
    physical_device_id = config.get_physical_device_id(logical_rank_id)
    device = f'npu:{physical_device_id}'

    in_tensor, matmul_weight, residual, gamma, bias, eps = input_data
    golden_out_tensor, golden_residual = output_data

    out_tensor = torch.empty(residual.shape, dtype=torch.bfloat16, device=device)
    residual_out = torch.empty(residual.shape, dtype=torch.bfloat16, device=device)

    inputs = [in_tensor.to(device), matmul_weight.to(device), residual.to(device), gamma.to(device),
        bias.to(device), out_tensor, residual_out]

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


@pytest.mark.skip(reason="performance test case")
@pytest.mark.world_size(4)
def test_matmul_allreduce_add_rmsnorm_performance():
    logger.info("=" * 60)
    logger.info("开始运行matmul_allreduce_add_rmsnorm性能测试")
    logger.info("=" * 60)

    mp.set_start_method('spawn', force=True)
    soc_version = _get_soc_version()
    world_size = 2 if soc_version == 260 else 8
    config = DistributedConfig(world_size=world_size)
    logger.info(f"检测到 soc_version={soc_version}, 使用 world_size={world_size}")

    expected_total_time = 80 if soc_version == 260 else 60
    all_min_times = []

    for run_num in range(1, 11):
        min_time = _run_single_iteration(run_num, config, expected_total_time)
        if min_time is not None:
            all_min_times.append(min_time)
        if run_num < 10:
            time.sleep(1)

    if all_min_times:
        std_dev = statistics.stdev(all_min_times) if len(all_min_times) > 1 else 0.0
        _print_statistics(all_min_times, expected_total_time, std_dev)
        _save_statistics_to_csv(config, all_min_times, expected_total_time, std_dev)


def _run_single_iteration(run_num, config, expected_total_time):
    logger.info(f"第 {run_num}/10 次运行开始")

    timestamp_dir = time.strftime('%Y-%m-%d_%H-%M-%S')
    output_name = f"output_run_{run_num}"
    output_dir = f"{Path.cwd()}/output/{timestamp_dir}/{output_name}"
    os.environ["TILE_FWK_OUTPUT_DIR"] = output_dir
    os.makedirs(output_dir, exist_ok=True)

    processes = []
    input_datas, output_datas = generate_golden_data(config.world_size)

    for i in range(config.world_size):
        p = mp.Process(target=matmul_allreduce_add_rmsnorm_worker,
                      args=(config, input_datas[i], output_datas[i], i))
        p.start()
        processes.append(p)

    for i, p in enumerate(processes):
        p.join()
        if p.exitcode != 0:
            raise AssertionError(f"process {i} failed, return: {p.exitcode}")

    try:
        analyzer = SwimlaneAnalyzer(output_dir, expected_total_time=expected_total_time)
        stats = analyzer.calculate_stats(config.world_size)
        min_time = stats['min_time']
        is_within = analyzer.check_within_expected(config.world_size)

        if not is_within:
            logger.warning(f"第{run_num}次运行执行时间超出预期: "
                f"实际{min_time:.3f}us > 预期{expected_total_time:.3f}us")
        else:
            logger.info(f"第{run_num}次运行执行时间在预期范围内: "
                f"实际{min_time:.3f}us <= 预期{expected_total_time:.3f}us")

        return min_time

    except Exception as e:
        logger.error(f"第{run_num}次运行分析失败: {e}")
        return None

    finally:
        logger.info(f"第{run_num}次运行性能数据保存在: {output_dir}")


def _print_statistics(all_min_times, expected_total_time, std_dev):
    logger.info("=" * 60)
    logger.info("10次运行统计结果")
    logger.info("=" * 60)

    avg_min_time = statistics.mean(all_min_times) if len(all_min_times) > 1 else all_min_times[0]
    min_min_time = min(all_min_times)
    max_min_time = max(all_min_times)

    logger.info(f"运行次数: {len(all_min_times)}")
    logger.info(f"平均值: {avg_min_time:.3f} us")
    logger.info(f"最小值: {min_min_time:.3f} us")
    logger.info(f"最大值: {max_min_time:.3f} us")

    if len(all_min_times) > 1:
        logger.info(f"标准差: {std_dev:.3f} us")
        logger.info(f"波动范围: {max_min_time - min_min_time:.3f} us")

    if expected_total_time is not None:
        logger.info(f"预期总时间: {expected_total_time:.3f} us")
        all_within = all(min_time <= expected_total_time for min_time in all_min_times)
        if all_within:
            logger.info("[PASS] 所有10次运行的最小值都在预期时间内")
        else:
            failed_runs = [i + 1 for i, min_time in enumerate(all_min_times)
                          if min_time > expected_total_time]
            logger.warning(f"[FAIL] 有{len(failed_runs)}次运行的最小值超出预期: "
                f"第{', '.join(map(str, failed_runs))}次")

    logger.info("=" * 60)


def _save_statistics_to_csv(config, all_min_times, expected_total_time, std_dev):
    stats_csv_file = "performance_statistics_matmul_allreduce_add_rmsnorm.csv"
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    avg_min_time = statistics.mean(all_min_times) if len(all_min_times) > 1 else all_min_times[0]
    min_min_time = min(all_min_times)
    max_min_time = max(all_min_times)

    stats_record = {
        'timestamp': timestamp,
        'world_size': config.world_size,
        'expected_time_us': expected_total_time,
        'min_of_mins_us': round(min_min_time, 3),
        'avg_of_mins_us': round(avg_min_time, 3),
        'max_of_mins_us': round(max_min_time, 3),
        'std_dev_us': round(std_dev, 3) if len(all_min_times) > 1 else 0.0,
        'num_runs': len(all_min_times)
    }

    with open(stats_csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'timestamp', 'world_size', 'expected_time_us',
            'min_of_mins_us', 'avg_of_mins_us', 'max_of_mins_us',
            'std_dev_us', 'num_runs'
        ])
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(stats_record)

    logger.info(f"汇总统计已追加保存到: {stats_csv_file}")


def summarize_performance_results(
    stats_csv_file: str = "performance_statistics_matmul_allreduce_add_rmsnorm.csv"
):
    """汇总并显示统计结果"""
    if not os.path.exists(stats_csv_file):
        logger.error(f"统计CSV文件不存在: {stats_csv_file}")
        return

    with open(stats_csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        logger.error("统计CSV文件为空")
        return

    recent_rows = rows[-10:] if len(rows) > 10 else rows
    _print_summary_table(recent_rows)
    _print_overall_statistics(recent_rows)


def _print_summary_table(recent_rows):
    logger.info("=" * 120)
    logger.info(f"性能统计结果汇总 (最近{len(recent_rows)}次)")
    logger.info("=" * 120)

    header_format = "{:<5} {:<20} {:<10} {:<12} {:<12} {:<12} {:<12} {:<12} {:<8}"
    logger.info(header_format.format('序号', '时间', 'world_size', '预期时间(us)',
                               '最小值(us)', '平均值(us)', '最大值(us)',
                               '标准差(us)', '运行次数'))
    logger.info("-" * 120)

    for i, row in enumerate(recent_rows, 1):
        timestamp = row.get('timestamp', 'N/A')
        world_size = row.get('world_size', 'N/A')
        expected_time = row.get('expected_time_us', 'N/A')
        min_of_mins = row.get('min_of_mins_us', 'N/A')
        avg_of_mins = row.get('avg_of_mins_us', 'N/A')
        max_of_mins = row.get('max_of_mins_us', 'N/A')
        std_dev = row.get('std_dev_us', 'N/A')
        num_runs = row.get('num_runs', 'N/A')

        logger.info(header_format.format(
            i, timestamp, world_size, expected_time,
            min_of_mins, avg_of_mins, max_of_mins, std_dev, num_runs
        ))

    logger.info("-" * 120)


def _print_overall_statistics(recent_rows):
    if len(recent_rows) <= 1:
        logger.info("=" * 120)
        return

    all_min_of_mins = []
    all_avg_of_mins = []
    all_max_of_mins = []

    for row in recent_rows:
        min_of_mins_str = row.get('min_of_mins_us', '')
        avg_of_mins_str = row.get('avg_of_mins_us', '')
        max_of_mins_str = row.get('max_of_mins_us', '')

        if min_of_mins_str:
            all_min_of_mins.append(float(min_of_mins_str))
        if avg_of_mins_str:
            all_avg_of_mins.append(float(avg_of_mins_str))
        if max_of_mins_str:
            all_max_of_mins.append(float(max_of_mins_str))

    if all_min_of_mins and all_avg_of_mins and all_max_of_mins:
        logger.info(f"\n总体统计 (基于{len(recent_rows)}次统计记录):")
        _print_stat_category("最小值统计", all_min_of_mins)
        _print_stat_category("平均值统计", all_avg_of_mins)
        _print_stat_category("最大值统计", all_max_of_mins)

    logger.info("=" * 120)


def _print_stat_category(category_name, values):
    min_val = min(values)
    avg_val = statistics.mean(values) if len(values) > 1 else values[0]
    max_val = max(values)
    logger.info(f"  {category_name}:")
    logger.info(f"    平均: {avg_val:.3f} us, 最小: {min_val:.3f} us, 最大: {max_val:.3f} us")


def main():
    test_matmul_allreduce_add_rmsnorm_performance()

    logger.info("\n")
    summarize_performance_results()


if __name__ == '__main__':
    main()