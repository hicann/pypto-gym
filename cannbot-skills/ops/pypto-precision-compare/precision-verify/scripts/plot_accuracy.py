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
PyPTO 精度变化折线图绘制工具
从验证结果日志文件中提取精度数据并绘制变化趋势图
"""
import logging
import os
import re
import sys

import matplotlib.pyplot as plt

CHECKPOINT_PATTERN = re.compile(r'^(\d+_[^:\s]+):')
NUMBER_PATTERN = r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
TOLERANCE_PATTERN = re.compile(f'Tolerance: rtol={NUMBER_PATTERN}, atol={NUMBER_PATTERN}')
ACTUAL_PATTERN = re.compile(f'Actual: rtol={NUMBER_PATTERN}, atol={NUMBER_PATTERN}')


logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def extract_operator_name(log_file_path):
    """从日志文件路径中提取算子名称"""
    log_file_name = os.path.basename(log_file_path)
    if log_file_name.endswith('_verify_result.log'):
        return log_file_name[:-len('_verify_result.log')]
    return 'operator'


def _checkpoint_starts(lines):
    starts = []
    for index, line in enumerate(lines):
        match = CHECKPOINT_PATTERN.match(line)
        if match:
            starts.append((match.group(1), index))
    return starts


def _parse_checkpoint_metrics(lines, start_line):
    tolerance = None
    actual = None
    for line in lines[start_line:start_line + 20]:
        tolerance_match = TOLERANCE_PATTERN.search(line)
        if tolerance_match:
            tolerance = tuple(float(tolerance_match.group(i)) for i in (1, 2))
            continue
        actual_match = ACTUAL_PATTERN.search(line)
        if actual_match:
            actual = tuple(float(actual_match.group(i)) for i in (1, 2))
            break
    if tolerance is None or actual is None:
        return None
    return tolerance + actual


def parse_checkpoints(lines):
    """从日志行中提取检查点名称和对应的精度数据"""
    results = []
    for checkpoint, start_line in _checkpoint_starts(lines):
        metrics = _parse_checkpoint_metrics(lines, start_line)
        if metrics is not None:
            results.append((checkpoint, *metrics))
    return results


def plot_accuracy(results, output_path):
    """绘制精度变化折线图"""
    checkpoints_list = [r[0] for r in results]
    tol_rtol_values = [r[1] for r in results]
    tol_atol_values = [r[2] for r in results]
    act_rtol_values = [r[3] for r in results]
    act_atol_values = [r[4] for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    ax1.plot(range(len(checkpoints_list)), act_rtol_values, marker='o', linewidth=2, markersize=8,
             color='blue', label='Actual rtol')
    ax1.plot(range(len(checkpoints_list)), tol_rtol_values, marker='s', linewidth=2, markersize=6,
             color='green', linestyle='--', label='Tolerance rtol')
    ax1.set_xlabel('Checkpoint', fontsize=14)
    ax1.set_ylabel('Relative Tolerance (rtol)', fontsize=14)
    ax1.set_title('Relative Tolerance (rtol) Change Across Checkpoints', fontsize=16, fontweight='bold', y=1.02)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(len(checkpoints_list)))
    ax1.set_xticklabels(checkpoints_list, rotation=45, ha='right', fontsize=11)
    ax1.legend(loc='upper right', fontsize=11)

    for i, (ckpt, rtol) in enumerate(zip(checkpoints_list, act_rtol_values)):
        if rtol > 0:
            ax1.annotate(f'{rtol:.4f}', (i, rtol), textcoords="offset points",
                       xytext=(0, 10), ha='center', fontsize=8, color='blue', fontweight='bold')

    ax2.plot(range(len(checkpoints_list)), act_atol_values, marker='o', linewidth=2, markersize=8,
             color='red', label='Actual atol')
    ax2.plot(range(len(checkpoints_list)), tol_atol_values, marker='s', linewidth=2, markersize=6,
             color='green', linestyle='--', label='Tolerance atol')
    ax2.set_xlabel('Checkpoint', fontsize=14)
    ax2.set_ylabel('Absolute Tolerance (atol)', fontsize=14)
    ax2.set_title('Absolute Tolerance (atol) Change Across Checkpoints', fontsize=16, fontweight='bold', y=1.02)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(len(checkpoints_list)))
    ax2.set_xticklabels(checkpoints_list, rotation=45, ha='right', fontsize=11)
    ax2.legend(loc='upper right', fontsize=11)

    for i, (ckpt, atol) in enumerate(zip(checkpoints_list, act_atol_values)):
        if atol > 0:
            ax2.annotate(f'{atol:.2f}', (i, atol), textcoords="offset points",
                       xytext=(0, 10), ha='center', fontsize=8, color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    return output_path


def print_summary(results):
    """Print accuracy data summary"""
    logger.info("\nAccuracy data summary:")
    logger.info("-" * 100)
    logger.info(f"{'Checkpoint':<30} {'Tolerance':<25} {'Actual':<25} {'Status':<10}")
    logger.info("-" * 100)
    for ckpt, tol_rtol, tol_atol, act_rtol, act_atol in results:
        status = "FAIL" if (act_rtol > tol_rtol or act_atol > tol_atol) else "PASS"
        tol_str = f"rt={tol_rtol:.6f}, at={tol_atol:.6f}"
        act_str = f"rt={act_rtol:.6f}, at={act_atol:.6f}"
        logger.info(f"{ckpt:<30} {tol_str:<25} {act_str:<25} {status:<10}")
    logger.info("-" * 100)


def main():
    """主函数"""
    if len(sys.argv) < 2:
        logger.error("Error: a log file path argument is required")
        logger.error("Usage: python3 plot_accuracy.py <verify_result.log>")
        sys.exit(1)

    log_file = sys.argv[1]
    log_file_path = os.path.abspath(log_file)
    log_file_dir = os.path.dirname(log_file_path)

    operator_name = extract_operator_name(log_file_path)

    with open(log_file, 'r') as f:
        lines = f.readlines()

    results = parse_checkpoints(lines)

    logger.info(f'Extracted accuracy data of {len(results)} checkpoints')
    for i, (ckpt, tol_rtol, tol_atol, act_rtol, act_atol) in enumerate(results):
        tol_str = f'Tolerance(rt={tol_rtol:.6f}, at={tol_atol:.6f})'
        act_str = f'Actual(rt={act_rtol:.6f}, at={act_atol:.6f})'
        logger.info(f'{i + 1}. {ckpt}: {tol_str} | {act_str}')

    output_image = os.path.join(log_file_dir, f'{operator_name}_accuracy_change.png')
    plot_accuracy(results, output_image)
    logger.info(f"\nAccuracy change line chart saved to: {output_image}")

    print_summary(results)


if __name__ == "__main__":
    main()
