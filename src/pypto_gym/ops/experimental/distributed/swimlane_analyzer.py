#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---
"""
Performance use
"""

import os
import json
import csv
import logging
from typing import List, Optional, Tuple
from datetime import datetime, timezone


class SwimlaneAnalyzer:
    def __init__(self, output_dir: str, expected_total_time: Optional[float] = None):
        """
        初始化Swimlane分析器

        参数:
        output_dir: 输出目录路径
        expected_total_time: 预期总时间（微秒），在测试函数中设置
        """
        self.output_dir = output_dir
        self.performance_data = None
        self.expected_total_time = expected_total_time
        self._cached_min_file_info = None
    
    @staticmethod
    def find_swimlane_files(rank_dir: str) -> List[str]:
        """在目录中查找 swimlane 文件"""
        swimlane_files = []
        for root, _, files in os.walk(rank_dir):
            for file in files:
                if file == "merged_swimlane.json":
                    swimlane_files.append(os.path.join(root, file))
        return swimlane_files

    @staticmethod
    def _calculate_total_time_from_data(performance_data: dict) -> float:
        """从性能数据计算总体执行时间（第一个任务开始到最后一个任务结束的时间跨度）"""
        if 'traceEvents' not in performance_data:
            raise ValueError("Performance data does not contain traceEvents")

        start_times = []
        end_times = []

        for event in performance_data['traceEvents']:
            if event.get('ph') == 'X':
                if 'fake' in event.get('name', '').lower():
                    continue

                start_time = event.get('ts', 0)
                duration = event.get('dur', 0)

                if duration <= 0:
                    continue

                end_time = start_time + duration
                start_times.append(start_time)
                end_times.append(end_time)

        if not start_times or not end_times:
            return 0.0

        overall_start = min(start_times)
        overall_end = max(end_times)
        total_time = overall_end - overall_start

        return total_time

    def find_all_rank_dirs(self) -> List[str]:
        """查找所有rank目录"""
        rank_dirs = []

        for item in os.listdir(self.output_dir):
            item_path = os.path.join(self.output_dir, item)
            if os.path.isdir(item_path):
                if 'rank' in item.lower():
                    rank_dirs.append(item_path)

        if not rank_dirs:
            for item in os.listdir(self.output_dir):
                item_path = os.path.join(self.output_dir, item)
                if os.path.isdir(item_path):
                    rank_dirs.append(item_path)

        return rank_dirs

    def find_recent_rank_dirs(self, world_size: int) -> List[str]:
        """查找最近的world_size个rank目录"""
        rank_dirs = self.find_all_rank_dirs()

        if not rank_dirs:
            return []

        rank_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)

        return rank_dirs[:world_size]

    def find_min_time_file_in_ranks(self, world_size: int) -> Tuple[str, float]:
        """
        在最近的world_size个rank目录中找到耗时最少的文件

        返回:
        (文件路径, 最小耗时)
        """
        if self._cached_min_file_info is not None:
            return self._cached_min_file_info

        rank_dirs = self.find_recent_rank_dirs(world_size)

        if not rank_dirs:
            raise FileNotFoundError(f"No rank directories found in {self.output_dir}")

        all_times = self._collect_swimlane_times(rank_dirs)

        if not all_times:
            raise ValueError("Could not find any valid swimlane file")

        min_file, min_time = min(all_times, key=lambda x: x[1])
        self._cached_min_file_info = (min_file, min_time)

        return min_file, min_time

    def calculate_min_total_time(self, world_size: int) -> float:
        """计算所有rank文件中总体执行时间的最小值"""
        _, min_time = self.find_min_time_file_in_ranks(world_size)
        return min_time

    def generate_comparison_report(self, world_size: int, debug: bool = True) -> str:
        """生成实际最小总体执行时间与预期总时间的对比报告"""
        actual_min_time = self.calculate_min_total_time(world_size)
        expected_total_time = self.expected_total_time
        min_file_path, _ = self.find_min_time_file_in_ranks(world_size)

        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("Swimlane总体执行时间对比报告")
        report_lines.append("=" * 60)
        report_lines.append("")
        report_lines.append(f"分析目录: {self.output_dir}")
        report_lines.append(f"World Size: {world_size}")
        report_lines.append(f"最小耗时文件: {os.path.basename(os.path.dirname(min_file_path))}")
        report_lines.append(f"文件路径: {min_file_path}")
        report_lines.append(f"分析时间: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        if expected_total_time is not None:
            is_within = actual_min_time <= expected_total_time
            diff = actual_min_time - expected_total_time

            if is_within:
                status = "✓"
                status_text = f"实际总体执行时间在预期范围内"
            else:
                status = "✗"
                status_text = f"实际总体执行时间超出预期"

            report_lines.append(f"实际总体执行时间: {actual_min_time:.3f} us")
            report_lines.append(f"预期总时间: {expected_total_time:.3f} us")
            report_lines.append(f"差值: {diff:+.3f} us ({diff/expected_total_time*100:.1f}%)")
            report_lines.append(f"状态: {status} {status_text}")
        else:
            report_lines.append(f"实际总体执行时间: {actual_min_time:.3f} us")
            report_lines.append(f"预期总时间: 未设置")

        report_lines.append("")
        report_lines.append("=" * 60)

        return "\n".join(report_lines)

    def check_within_expected(self, world_size: int) -> bool:
        """检查实际总体执行时间是否在预期总时间内"""
        if self.expected_total_time is None:
            return True

        actual_min_time = self.calculate_min_total_time(world_size)
        return actual_min_time <= self.expected_total_time

    def save_to_csv(self, world_size: int, csv_file: str = "performance_results.csv"):
        """将结果保存到CSV文件"""
        actual_min_time = self.calculate_min_total_time(world_size)
        min_file_path, _ = self.find_min_time_file_in_ranks(world_size)
        expected_total_time = self.expected_total_time
        is_within = self.check_within_expected(world_size)
        timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

        rank = "unknown"
        rank_dir = os.path.dirname(min_file_path)
        for part in rank_dir.split(os.sep):
            if 'rank' in part.lower():
                rank = part
                break

        fieldnames = [
            'timestamp',
            'world_size',
            'rank',
            'expected_time_us',
            'actual_total_time_us',
            'is_within_expected',
            'min_file_path',
            'status'
        ]

        file_exists = os.path.isfile(csv_file)

        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            expected_value = expected_total_time if expected_total_time is not None else 'N/A'

            writer.writerow({
                'timestamp': timestamp,
                'world_size': world_size,
                'rank': rank,
                'expected_time_us': expected_value,
                'actual_total_time_us': round(actual_min_time, 3),
                'is_within_expected': is_within,
                'min_file_path': min_file_path,
                'status': 'PASS' if is_within else 'FAIL'
            })

        return csv_file

    def print_swimlane_files_info(self, world_size: int):
        """打印找到的swimlane文件信息"""
        if self._cached_min_file_info is None:
            self.find_min_time_file_in_ranks(world_size)

        rank_dirs = self.find_recent_rank_dirs(world_size)

        if not rank_dirs:
            logging.warning(f"No rank directories found in {self.output_dir}")
            return

        all_times = self._collect_swimlane_times(rank_dirs)

        if all_times:
            logging.info(f"Found {len(all_times)} swimlane files with times:")
            for file_path, time in sorted(all_times, key=lambda x: x[1]):
                rank_name = os.path.basename(os.path.dirname(file_path))
                logging.info(f"  {rank_name}: {time:.3f} us")

    def _collect_swimlane_times(self, rank_dirs: List[str]) -> List[Tuple[str, float]]:
        """收集所有 swimlane 文件的时间信息"""
        all_times = []
        for rank_dir in rank_dirs:
            swimlane_files = SwimlaneAnalyzer.find_swimlane_files(rank_dir)
            for file_path in swimlane_files:
                time_info = self._parse_swimlane_file(file_path)
                if time_info:
                    all_times.append(time_info)
        return all_times

    def _parse_swimlane_file(self, file_path: str) -> Optional[Tuple[str, float]]:
        """解析单个 swimlane 文件并返回时间信息"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            total_time = self._calculate_total_time_from_data(data)
            return (file_path, total_time)

        except (json.JSONDecodeError, OSError) as e:
            logging.warning(f"Failed to parse swimlane file {file_path}: {e}")
            return None
