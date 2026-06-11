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
import numpy as np
from numpy.testing import assert_allclose


class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'


def _get_multi_index(flat_index, shape):
    indices = []
    remaining = flat_index
    for dim in reversed(shape):
        indices.append(remaining % dim)
        remaining = remaining // dim
    return tuple(reversed(indices))


def _print_forced_elements(cpu_flat, npu_flat, force_print_first_n, total_elements, shape):
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    if force_print_first_n <= 0:
        return
    print(f"{YELLOW}强制打印前 {force_print_first_n} 个元素:{RESET}")
    for flat_idx in range(min(force_print_first_n, total_elements)):
        cpu_val = cpu_flat[flat_idx]
        npu_val = npu_flat[flat_idx]
        multi_idx = _get_multi_index(flat_idx, shape)
        if np.isnan(cpu_val) or np.isnan(npu_val):
            diff_str = "NaN"
        else:
            diff_val = np.abs(cpu_val - npu_val)
            diff_str = f"{diff_val:.6e}"
        cpu_str = "NaN" if np.isnan(cpu_val) else f"{cpu_val:.6e}"
        npu_str = "NaN" if np.isnan(npu_val) else f"{npu_val:.6e}"
        print(f"{YELLOW}索引 {multi_idx}: cpu={cpu_str}, npu={npu_str}, 差值={diff_str}{RESET}")
    print("-" * 80)


def _scan_anomalies(cpu_flat, npu_flat, total_elements, shape, atol, rtol, max_prints):
    abnormal_count = 0
    nan_count = 0
    exceed_tolerance_count = 0
    for flat_idx in range(total_elements):
        cpu_val = cpu_flat[flat_idx]
        npu_val = npu_flat[flat_idx]
        multi_idx = _get_multi_index(flat_idx, shape)
        if np.isnan(npu_val):
            abnormal_count += 1
            nan_count += 1
            if abnormal_count <= max_prints:
                cpu_str = "NaN" if np.isnan(cpu_val) else f"{cpu_val:.6e}"
                print(f"索引 {multi_idx}: cpu={cpu_str}, npu=NaN, 差值=NaN (NPU包含NaN)")
        elif np.isnan(cpu_val):
            abnormal_count += 1
            exceed_tolerance_count += 1
            if abnormal_count <= max_prints:
                print(f"索引 {multi_idx}: cpu=NaN, npu={npu_val:.6e}, 差值=NaN (CPU包含NaN)")
        else:
            abs_diff = np.abs(cpu_val - npu_val)
            allowed_diff = atol + rtol * np.abs(npu_val)
            if abs_diff > allowed_diff:
                abnormal_count += 1
                exceed_tolerance_count += 1
                if abnormal_count <= max_prints:
                    print(f"索引 {multi_idx}: cpu={cpu_val:.6e}, npu={npu_val:.6e}, "
                          f"差值={abs_diff:.6e} (超过容差 {allowed_diff:.6e})")
    return abnormal_count, nan_count, exceed_tolerance_count


def _print_comparison_stats(abnormal_count, nan_count, exceed_tolerance_count,
                             total_elements, name, max_prints):
    print("=" * 80)
    print(f"{Colors.BOLD}{Colors.PURPLE}{name} 比较结果统计:{Colors.RESET}")
    print(f"总元素数量: {total_elements}")
    print(f"异常元素数量: {abnormal_count}")
    print(f"  - NaN 数量: {nan_count}")
    print(f"  - 超出容差数量: {exceed_tolerance_count}")
    print(f"异常比例: {abnormal_count / total_elements * 100:.4f}%")
    is_allclose = (abnormal_count == 0)
    print(f"\nnp.allclose 等价结果: {is_allclose}")
    if abnormal_count > max_prints:
        print(f"\n注意: 只显示了前 {max_prints} 个异常，共有 {abnormal_count} 个异常元素")


def detailed_allclose_manual(cpu, npu, name, rtol=1e-3, atol=1e-3, max_prints=50, force_print_first_n=5):
    if cpu.shape != npu.shape:
        print(f"错误: 形状不一致 - cpu {cpu.shape} vs npu {npu.shape}")
        return False

    total_elements = cpu.size
    print(f"开始比较数组，形状: {cpu.shape}, 总元素数: {total_elements}")
    print(f"容差条件: rtol={rtol}, atol={atol}")
    print("=" * 80)

    cpu_flat = cpu.reshape(-1)
    npu_flat = npu.reshape(-1)

    _print_forced_elements(cpu_flat, npu_flat, force_print_first_n, total_elements, cpu.shape)

    abnormal_count, nan_count, exceed_tolerance_count = _scan_anomalies(
        cpu_flat, npu_flat, total_elements, cpu.shape, atol, rtol, max_prints)

    _print_comparison_stats(abnormal_count, nan_count, exceed_tolerance_count,
                             total_elements, name, max_prints)
    assert_allclose(cpu, npu, rtol, atol)
    return (abnormal_count == 0)
