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

import numpy as np
import pytest
import torch

import pypto

from distributed_config import DistributedConfig
from swimlane_analyzer import SwimlaneAnalyzer

logging.basicConfig(level=logging.INFO, format='%(message)s')


@pypto.frontend.jit(
    debug_options={"runtime_debug_mode": 1},
    runtime_options={"stitch_function_max_num": 128,
                     "stitch_cfgcache_size": 100000000},
)
def matmul_allreduce_kernel(
    in_tensor: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    matmul_weight: pypto.Tensor(),
    out_tensor: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    group_name,
    world_size,
):
    batch_size = in_tensor.shape[0]
    hidden_size = matmul_weight.shape[0]

    view_row_shape = 8
    bs_loop = (batch_size + view_row_shape - 1) // view_row_shape

    for bs_idx in pypto.loop(bs_loop, name="LOOP_MM_ALLREDUCE", idx_name="bs_idx"):
        shmem_shape = [view_row_shape, hidden_size]
        shmem_tensor = pypto.distributed.create_shmem_tensor(
            group_name, world_size, pypto.DT_FP32, shmem_shape)
        shmem_barrier_signal = pypto.distributed.create_shmem_signal(group_name, world_size)
        my_pe = pypto.distributed.my_symbolic_pe(group_name)
        for _ in pypto.loop(1, name="LOOP_MM_AR_L0", idx_name="_"):
            in_tensor_tile = pypto.view(
                in_tensor, (view_row_shape, in_tensor.shape[1]), [bs_idx * view_row_shape, 0],
                valid_shape=[(batch_size - bs_idx * view_row_shape).min(view_row_shape), in_tensor.shape[1]])

            pypto.set_vec_tile_shapes(view_row_shape, hidden_size)
            data_clear_out = pypto.distributed.shmem_clear_data(
                shmem_tensor, shmem_shape, [0, 0], pred=[in_tensor_tile])
            signal_clear_out = pypto.distributed.shmem_clear_signal(
                shmem_tensor, pred=[in_tensor_tile])
            barrier_out = pypto.distributed.shmem_barrier_all(
                shmem_barrier_signal, [data_clear_out, signal_clear_out])

            pypto.set_cube_tile_shapes([8, 8], [128, 256], [256, 512])
            matmul_result = pypto.matmul(in_tensor_tile, matmul_weight, pypto.DT_FP32, b_trans=True)

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

            all_reduce_bf16 = pypto.cast(all_reduce_out, pypto.DT_BF16)
            out_tensor[bs_idx * pypto.symbolic_scalar(view_row_shape):] = all_reduce_bf16


def generate_golden_data(world_size: int):
    batch_size = 8
    attn_dim_per_tp = 1536
    hidden_size = 5120
    torch.manual_seed(42)

    input_datas = []
    for _ in range(world_size):
        in_tensor = torch.randn((batch_size, attn_dim_per_tp), dtype=torch.bfloat16).share_memory_()
        matmul_weight = torch.randn((hidden_size, attn_dim_per_tp), dtype=torch.bfloat16).share_memory_()
        input_data = [in_tensor, matmul_weight]
        input_datas.append(input_data)
    output_datas = matmul_allreduce_result_golden(batch_size, hidden_size, input_datas)
    return input_datas, output_datas


def matmul_allreduce_result_golden(batch_size, num, input_datas):
    output_datas = []
    matmul_allreduce_result_fp32 = torch.zeros((batch_size, num), dtype=torch.float32)
    for input_data in input_datas:
        in_tensor, matmul_weight = input_data[:2]
        matmul_result = torch.matmul(in_tensor.to(torch.float32), matmul_weight.to(torch.float32).T)
        matmul_allreduce_result_fp32 += matmul_result

    for _ in input_datas:
        output_data = matmul_allreduce_result_fp32.to(torch.bfloat16)
        output_datas.append(output_data)
    return output_datas


def matmul_allreduce_worker(
    config: DistributedConfig,
    input_data: list,
    output_data: torch.Tensor,
    logical_rank_id: int,
    output_dir: str,
):
    groups = config.init_hccl_comm(logical_rank_id)
    physical_device_id = config.get_physical_device_id(logical_rank_id)
    device = f'npu:{physical_device_id}'

    in_tensor, matmul_weight = input_data

    out_tensor = torch.empty((8, 5120), dtype=torch.bfloat16, device=device)

    inputs = [in_tensor.to(device), matmul_weight.to(device), out_tensor]

    matmul_allreduce_kernel(*inputs, groups[0], config.world_size)

    np.testing.assert_allclose(
        np.array(out_tensor.cpu().flatten().tolist()),
        np.array(output_data.cpu().flatten().tolist()),
        rtol=8e-3,
        atol=8e-3,
    )


@pytest.mark.skip(reason="performance test case")
def test_matmul_allreduce_performance():
    mp.set_start_method('spawn', force=True)
    config = DistributedConfig(world_size=2)

    output_name = f"output"
    output_dir = f"{Path.cwd()}/{output_name}"
    os.environ["TILE_FWK_OUTPUT_DIR"] = output_dir

    expected_total_time = 70.0

    csv_file = "performance_results.csv"

    processes = []
    input_datas, output_datas = generate_golden_data(config.world_size)

    for i in range(config.world_size):
        p = mp.Process(target=matmul_allreduce_worker,
                      args=(config, input_datas[i], output_datas[i], i, output_dir))
        p.start()
        processes.append(p)

    for i, p in enumerate(processes):
        p.join()
        if p.exitcode != 0:
            raise AssertionError(f"process {i} failed, return: {p.exitcode}")

    try:
        analyzer = SwimlaneAnalyzer(output_dir, expected_total_time=expected_total_time)

        analyzer.print_swimlane_files_info(config.world_size)

        report = analyzer.generate_comparison_report(config.world_size, debug=False)
        logging.info(report)

        csv_path = analyzer.save_to_csv(config.world_size, csv_file)
        logging.info(f"结果已保存到: {csv_path}")

        is_within = analyzer.check_within_expected(config.world_size)

        if not is_within:
            actual_min_time = analyzer.calculate_min_total_time(config.world_size)
            pytest.fail(f"总体执行时间超出预期: 实际{actual_min_time:.3f}us > 预期{expected_total_time:.3f}us")
        else:
            logging.info("✓ 总体执行时间在预期范围内")

    finally:
        logging.info(f"\n性能数据保存在: {output_dir}")


def _print_results_table(rows):
    """打印结果表格"""
    logging.info("=" * 80)
    logging.info("性能测试结果汇总")
    logging.info("=" * 80)
    header = f"{'序号':<4} {'时间':<20} {'world_size':<10} {'rank':<8} " \
             f"{'预期时间(us)':<12} {'实际总体时间(us)':<16} {'状态':<8}"
    logging.info(header)
    logging.info("-" * 80)

    for i, row in enumerate(rows, 1):
        row_str = f"{i:<4} {row.get('timestamp', 'N/A'):<20} " \
                  f"{row.get('world_size', 'N/A'):<10} {row.get('rank', 'N/A'):<8} " \
                  f"{row.get('expected_time_us', 'N/A'):<12} " \
                  f"{row.get('actual_total_time_us', 'N/A'):<16} " \
                  f"{row.get('status', 'N/A'):<8}"
        logging.info(row_str)


def _print_summary_stats(rows):
    """打印统计摘要"""
    total_tests = len(rows)
    passed_tests = sum(1 for row in rows if row.get('status') == 'PASS')
    failed_tests = total_tests - passed_tests

    logging.info("-" * 80)
    logging.info(f"总计测试: {total_tests}")
    logging.info(f"通过测试: {passed_tests}")
    logging.info(f"失败测试: {failed_tests}")


def _build_ws_stats(rows):
    """构建 world_size 统计数据"""
    ws_stats = {}
    for row in rows:
        ws = row.get('world_size', 'unknown')
        if ws not in ws_stats:
            ws_stats[ws] = {'count': 0, 'times': [], 'passed': 0}

        ws_stats[ws]['count'] += 1
        if row.get('actual_total_time_us', 'N/A') != 'N/A':
            try:
                ws_stats[ws]['times'].append(float(row['actual_total_time_us']))
            except (ValueError, TypeError):
                pass
        if row.get('status') == 'PASS':
            ws_stats[ws]['passed'] += 1
    return ws_stats


def _print_ws_stats(ws_stats):
    """打印 world_size 统计"""
    if not ws_stats:
        return

    logging.info(f"\n按world_size统计:")
    for ws, stats in sorted(ws_stats.items()):
        if stats['times']:
            avg_time = sum(stats['times']) / len(stats['times'])
            min_time = min(stats['times'])
            max_time = max(stats['times'])
            pass_rate = (stats['passed'] / stats['count'] * 100) if stats['count'] > 0 else 0

            logging.info(f"  world_size={ws}:")
            logging.info(f"    测试次数: {stats['count']}, 通过率: {pass_rate:.1f}%")
            logging.info(f"    平均时间: {avg_time:.3f} us, 最小: {min_time:.3f} us, 最大: {max_time:.3f} us")


def summarize_performance_results(csv_file: str = "performance_results.csv"):
    """汇总并显示所有性能测试结果"""
    if not os.path.exists(csv_file):
        logging.warning(f"CSV文件不存在: {csv_file}")
        return

    with open(csv_file, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    if not rows:
        logging.warning("CSV文件为空")
        return

    _print_results_table(rows)
    _print_summary_stats(rows)

    if rows:
        ws_stats = _build_ws_stats(rows)
        _print_ws_stats(ws_stats)

    logging.info("=" * 80)


def main():
    test_matmul_allreduce_performance()

    logging.info("\n")
    summarize_performance_results()


if __name__ == '__main__':
    main()