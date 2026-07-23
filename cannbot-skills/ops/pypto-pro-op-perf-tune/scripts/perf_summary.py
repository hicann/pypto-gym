# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Generate a statistical summary from msprof op CSV output and archive to operator directory.

Usage:
    python3 perf_summary.py <OPPROF_dir> <ops_dir>

Example:
    python3 perf_summary.py /path/to/OPPROF_xxx ops/Softmax

The script:
1. Creates docs/perf/round_NNN/ under <ops_dir> (auto-incrementing)
2. Copies all CSV files from OPPROF directory to the archive
3. Generates summary.txt with min/avg/max statistics for ALL non-zero metrics
4. Does NOT make any judgments, thresholds, or optimization recommendations
"""

import argparse
import csv
import glob
import logging
import os
import re
import shutil
import statistics
import sys
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
_stdout_handler = logging.StreamHandler(sys.stdout)
_stdout_handler.setLevel(logging.INFO)
_stdout_handler.addFilter(lambda r: r.levelno < logging.ERROR)
_stderr_handler = logging.StreamHandler(sys.stderr)
_stderr_handler.setLevel(logging.ERROR)
LOGGER.addHandler(_stdout_handler)
LOGGER.addHandler(_stderr_handler)
LOGGER.propagate = False


def safe_float(val: Any, default: float = 0.0) -> float:
    if val is None or str(val).strip() in ("", "N/A", "NA"):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def stat_line(name: str, values: List[float], fmt: str = ".1f", unit: str = "", multiply: float = 1.0) -> Optional[str]:
    """Generate a min/avg/max stat line. Returns None if all values are zero."""
    scaled = [v * multiply for v in values]
    if all(abs(v) < 0.001 for v in scaled):
        return None
    mn, avg, mx = min(scaled), statistics.mean(scaled), max(scaled)
    u = f"    ({unit})" if unit else ""
    return f"{'  ' + name:<24s} {mn:>10{fmt}} {avg:>10{fmt}} {mx:>10{fmt}}{u}"


def detect_core_prefix(rows: List[Dict[str, str]]) -> str:
    """Detect whether this is a vector (aiv) or cube (aic) workload."""
    if not rows:
        return "aiv"
    row = rows[0]
    aiv_time = safe_float(row.get("aiv_time(us)"))
    aic_time = safe_float(row.get("aic_time(us)"))
    return "aic" if aic_time > aiv_time else "aiv"


def find_next_round(perf_dir: str) -> str:
    """Find the next round_NNN directory number."""
    if not os.path.exists(perf_dir):
        return os.path.join(perf_dir, "round_001")
    existing = [d for d in os.listdir(perf_dir) if re.match(r"round_\d+", d)]
    if not existing:
        return os.path.join(perf_dir, "round_001")
    nums = [int(re.search(r"\d+", d).group()) for d in existing]
    return os.path.join(perf_dir, f"round_{max(nums) + 1:03d}")


def archive_csvs(opprof_dir: str, round_dir: str) -> List[str]:
    """Copy all CSV files from OPPROF directory to archive. Returns list of copied files."""
    os.makedirs(round_dir, exist_ok=True)
    copied = []
    for csv_file in sorted(glob.glob(os.path.join(opprof_dir, "*.csv"))):
        dest = os.path.join(round_dir, os.path.basename(csv_file))
        shutil.copy2(csv_file, dest)
        copied.append(os.path.basename(csv_file))
    return copied


def _append_stat_lines(lines, rows, fields, *, multiply=1.0, unit=""):
    for display_name, field in fields:
        values = [safe_float(row.get(field)) for row in rows]
        line = stat_line(display_name, values, ".2f", unit=unit, multiply=multiply)
        if line:
            lines.append(line)


def _append_bandwidth_lines(lines, rows, fields):
    for display_name, field in fields:
        values = [safe_float(row.get(field)) for row in rows]
        line = stat_line(display_name, values, ".1f", unit="GB/s")
        if line:
            lines.append(line)


def _basic_summary(opprof_dir):
    rows = read_csv(os.path.join(opprof_dir, "OpBasicInfo.csv"))
    lines = ["=== 上板性能统计摘要 ==="]
    if not rows:
        lines.append("(OpBasicInfo.csv 未找到)")
        return lines, 0.0
    row = rows[0]
    duration = safe_float(row.get("Task Duration(us)"))
    lines.append(
        f"Op: {row.get('Op Name', 'unknown')} | Type: {row.get('Op Type', 'unknown')} | "
        f"Duration: {duration}us | BlockDim: {int(safe_float(row.get('Block Dim', '1')))} | "
        f"Freq: {safe_float(row.get('Current Freq')):.0f}/{safe_float(row.get('Rated Freq')):.0f}"
    )
    return lines, duration


def _pipe_ratio_fields(prefix):
    fields = [
        ("vec_ratio%", f"{prefix}_vec_ratio"),
        ("scalar_ratio%", f"{prefix}_scalar_ratio"),
        ("mte2_ratio%", f"{prefix}_mte2_ratio"),
        ("mte3_ratio%", f"{prefix}_mte3_ratio"),
        ("icache_miss%", f"{prefix}_icache_miss_rate"),
    ]
    if prefix == "aic":
        fields.extend([
            ("cube_ratio%", "aic_cube_ratio"),
            ("fixpipe_ratio%", "aic_fixpipe_ratio"),
            ("mte1_ratio%", "aic_mte1_ratio"),
        ])
    return fields


def _pipe_bandwidth_fields(prefix):
    fields = [
        ("mte2_active_bw", f"{prefix}_mte2_active_bw(GB/s)"),
        ("mte3_active_bw", f"{prefix}_mte3_active_bw(GB/s)"),
    ]
    if prefix == "aic":
        fields.extend([
            ("mte1_active_bw", "aic_mte1_active_bw(GB/s)"),
            ("fixpipe_active_bw", "aic_fixpipe_active_bw(GB/s)"),
        ])
    return fields


def _scalar_summary(rows, prefix):
    fields = [
        ("single", f"{prefix}_scalar_single_time(us)"),
        ("dual", f"{prefix}_scalar_dual_time(us)"),
        ("wait", f"{prefix}_scalar_wait_time(us)"),
        ("mte2_stall", f"{prefix}_scalar_mte2_stall_time(us)"),
        ("mte3_stall", f"{prefix}_scalar_mte3_stall_time(us)"),
    ]
    if prefix == "aiv":
        fields.extend([
            ("vec_stall", "aiv_scalar_vector_stall_time(us)"),
            ("ub_stall", "aiv_scalar_stall_by_ub_time(us)"),
        ])
    else:
        fields.extend([
            ("cube_stall", "aic_scalar_cube_stall_time(us)"),
            ("mte1_stall", "aic_scalar_mte1_stall_time(us)"),
        ])
    fields.append(("wait_ib", f"{prefix}_scalar_wait_ib_time(us)"))
    parts = []
    for display_name, field in fields:
        average = statistics.mean(safe_float(row.get(field)) for row in rows)
        if average > 0.001:
            parts.append(f"{display_name}: {average:.2f}us")
    return parts


def _pipe_summary(rows, duration):
    if not rows:
        return []
    prefix = detect_core_prefix(rows)
    lines = ["", f"--- PipeUtilization ({len(rows)} cores, prefix={prefix}) ---"]
    lines.append(f"  {'':24s} {'min':>10s} {'avg':>10s} {'max':>10s}")
    core_times = [safe_float(row.get(f"{prefix}_time(us)")) for row in rows]
    core_line = stat_line(f"{prefix}_time(us)", core_times, ".2f")
    if core_line:
        lines.append(core_line)
    _append_stat_lines(lines, rows, _pipe_ratio_fields(prefix), multiply=100)
    _append_bandwidth_lines(lines, rows, _pipe_bandwidth_fields(prefix))
    scalar_parts = _scalar_summary(rows, prefix)
    if scalar_parts:
        lines.extend(["", "--- SCALAR 子类耗时 (avg) ---", "  " + " | ".join(scalar_parts)])
    if duration > 0 and core_times:
        max_core = max(core_times)
        overhead = max(0, duration - max_core)
        lines.extend([
            "",
            "--- 头开销 ---",
            f"  Task Duration: {duration}us | 最长核: {max_core:.2f}us | "
            f"头开销: {overhead:.2f}us ({overhead / duration * 100:.1f}%)",
        ])
    return lines


def _append_transfer(lines, rows, data_field, rate_field, label):
    values = [safe_float(row.get(data_field)) for row in rows]
    total = sum(values)
    if total <= 0:
        return
    rates = [safe_float(row.get(rate_field)) for row in rows]
    lines.append(
        f"  {label}: {total:.1f}KB total ({statistics.mean(values):.1f}KB/core), "
        f"BW usage: {statistics.mean(rates):.2f}%"
    )


def _memory_summary(rows):
    if not rows:
        return []
    lines = ["", "--- Memory ---"]
    _append_transfer(lines, rows, "GM_to_UB_datas(KB)", "GM_to_UB_bw_usage_rate(%)", "GM→UB")
    _append_transfer(lines, rows, "UB_to_GM_datas(KB)", "UB_to_GM_bw_usage_rate(%)", "UB→GM")
    _append_transfer(lines, rows, "GM_to_L1_datas(KB)", "GM_to_L1_bw_usage_rate(%)", "GM→L1")
    read_total = sum(safe_float(row.get("read_main_memory_datas(KB)")) for row in rows)
    write_total = sum(safe_float(row.get("write_main_memory_datas(KB)")) for row in rows)
    if read_total > 0:
        lines.append(f"  主存读取: {read_total:.1f}KB total")
    if write_total > 0:
        lines.append(f"  主存写入: {write_total:.1f}KB total")
    gm_to_ub = sum(safe_float(row.get("GM_to_UB_datas(KB)")) for row in rows)
    instruction_count = sum(
        safe_float(row.get("aiv_mte2_instructions", 0))
        + safe_float(row.get("aic_mte2_instructions", 0))
        for row in rows
    )
    if instruction_count > 0 and gm_to_ub > 0:
        average_kb = gm_to_ub / instruction_count
        lines.append(f"  Avg MTE2 transfer: {average_kb:.2f}KB ({int(instruction_count)} instructions total)")
    for label, field in (("GM→UB", "aiv_gm_to_ub_bw(GB/s)"), ("UB→GM", "aiv_ub_to_gm_bw(GB/s)")):
        values = [safe_float(row.get(field)) for row in rows]
        if sum(values) > 0:
            lines.append(f"  {label} avg BW: {statistics.mean(values):.2f} GB/s")
    return lines


def _average_field_parts(rows, fields, suffix=""):
    parts = []
    for display, field in fields:
        average = statistics.mean(safe_float(row.get(field)) for row in rows)
        if average > 0.001:
            parts.append(f"{display}: {average:.1f}{suffix}")
    return parts


def _percentage_parts(rows, fields, precision=1, threshold=0.01):
    parts = []
    for display, field in fields:
        average = statistics.mean(safe_float(row.get(field)) * 100 for row in rows)
        if average > threshold:
            parts.append(f"{display}: {average:.{precision}f}%")
    return parts


def _memory_ub_summary(rows):
    if not rows:
        return []
    fields = [
        ("UB read BW (vector)", "aiv_ub_read_bw_vector(GB/s)"),
        ("UB write BW (vector)", "aiv_ub_write_bw_vector(GB/s)"),
        ("UB read BW (scalar)", "aiv_ub_read_bw_scalar(GB/s)"),
        ("UB write BW (scalar)", "aiv_ub_write_bw_scalar(GB/s)"),
    ]
    lines = ["", "--- MemoryUB ---"]
    for display, field in fields:
        average = statistics.mean(safe_float(row.get(field)) for row in rows)
        if average > 0.001:
            lines.append(f"  {display}: avg={average:.1f} GB/s")
    return lines


def _memory_l0_summary(rows):
    if not rows:
        return []
    fields = [
        ("L0A read BW", "aic_l0a_read_bw(GB/s)"),
        ("L0A write BW", "aic_l0a_write_bw(GB/s)"),
        ("L0B read BW", "aic_l0b_read_bw(GB/s)"),
        ("L0B write BW", "aic_l0b_write_bw(GB/s)"),
        ("L0C read BW (cube)", "aic_l0c_read_bw_cube(GB/s)"),
        ("L0C write BW (cube)", "aic_l0c_write_bw_cube(GB/s)"),
    ]
    parts = _average_field_parts(rows, fields, " GB/s")
    return ["", "--- MemoryL0 ---", *(f"  {part}" for part in parts)] if parts else []


def _l2_summary(rows, prefix):
    if not rows:
        return []
    lines = ["", "--- L2Cache ---"]
    rate_fields = [
        ("total_hit", f"{prefix}_total_hit_rate(%)"),
        ("read_hit", f"{prefix}_read_hit_rate(%)"),
        ("write_hit", f"{prefix}_write_hit_rate(%)"),
    ]
    for rate_name, field in rate_fields:
        values = [safe_float(row.get(field)) for row in rows]
        if any(value > 0 for value in values):
            lines.append(
                f"  {rate_name}: avg={statistics.mean(values):.1f}% "
                f"(min={min(values):.1f}%, max={max(values):.1f}%)"
            )
    total_hit = sum(safe_float(row.get(f"{prefix}_write_cache_hit")) for row in rows)
    total_miss = sum(safe_float(row.get(f"{prefix}_write_cache_miss_allocate")) for row in rows)
    if total_hit + total_miss > 0:
        lines.append(f"  write cache: hit={int(total_hit)} miss={int(total_miss)}")
    return lines


def _resource_conflict_summary(rows, prefix):
    if not rows:
        return []
    conflict_fields = [
        ("vec_total_cflt", f"{prefix}_vec_total_cflt_ratio"),
        ("bankgroup_cflt", f"{prefix}_vec_bankgroup_cflt_ratio"),
        ("bank_cflt", f"{prefix}_vec_bank_cflt_ratio"),
        ("resc_cflt", f"{prefix}_vec_resc_cflt_ratio"),
        ("mte_cflt", f"{prefix}_vec_mte_cflt_ratio"),
    ]
    wait_fields = [
        ("vec_wait", f"{prefix}_vec_wait_ratio"),
        ("mte2_wait", f"{prefix}_mte2_wait_ratio"),
        ("mte3_wait", f"{prefix}_mte3_wait_ratio"),
    ]
    if prefix == "aic":
        wait_fields.insert(0, ("cube_wait", "aic_cube_wait_ratio"))
    groups = []
    for fields in (conflict_fields, wait_fields):
        parts = _percentage_parts(rows, fields, precision=2, threshold=-1)
        groups.append("  " + " | ".join(parts))
    return ["", "--- ResourceConflict ---", *groups]


def _arithmetic_summary(rows):
    if not rows:
        return []
    lines = ["", "--- ArithmeticUtilization ---"]
    vector_fields = [
        ("vec_fp32", "aiv_vec_fp32_ratio"),
        ("vec_fp16", "aiv_vec_fp16_ratio"),
        ("vec_int32", "aiv_vec_int32_ratio"),
        ("vec_int16", "aiv_vec_int16_ratio"),
        ("vec_misc", "aiv_vec_misc_ratio"),
    ]
    vector_parts = _percentage_parts(rows, vector_fields)
    if vector_parts:
        lines.append("  " + " | ".join(vector_parts))
    vector_fops = statistics.mean(safe_float(row.get("aiv_vec_fops")) for row in rows)
    if vector_fops > 0:
        lines.append(f"  vec_fops: {vector_fops:.0f}/core")
    cube_fields = [("cube_fp16", "aic_cube_fp16_ratio"), ("cube_int8", "aic_cube_int8_ratio")]
    cube_parts = _percentage_parts(rows, cube_fields)
    cube_fops = statistics.mean(safe_float(row.get("aic_cube_fops")) for row in rows)
    if cube_fops > 0:
        cube_parts.append(f"cube_fops: {cube_fops:.0f}/core")
    if cube_parts:
        lines.append("  " + " | ".join(cube_parts))
    return lines


def generate_summary(opprof_dir: str, round_dir: str, ops_dir: str) -> str:
    """Generate summary.txt content from CSV data."""
    lines, duration = _basic_summary(opprof_dir)

    # === PipeUtilization ===
    pipe_rows = read_csv(os.path.join(opprof_dir, "PipeUtilization.csv"))
    lines.extend(_pipe_summary(pipe_rows, duration))

    # === Memory ===
    mem_rows = read_csv(os.path.join(opprof_dir, "Memory.csv"))
    lines.extend(_memory_summary(mem_rows))

    # === MemoryUB ===
    ub_rows = read_csv(os.path.join(opprof_dir, "MemoryUB.csv"))
    lines.extend(_memory_ub_summary(ub_rows))

    # === MemoryL0 ===
    l0_rows = read_csv(os.path.join(opprof_dir, "MemoryL0.csv"))
    lines.extend(_memory_l0_summary(l0_rows))

    # === L2Cache ===
    l2_rows = read_csv(os.path.join(opprof_dir, "L2Cache.csv"))
    prefix = detect_core_prefix(pipe_rows) if pipe_rows else "aiv"
    lines.extend(_l2_summary(l2_rows, prefix))

    # === ResourceConflict ===
    rc_rows = read_csv(os.path.join(opprof_dir, "ResourceConflictRatio.csv"))
    lines.extend(_resource_conflict_summary(rc_rows, prefix))

    # === ArithmeticUtilization ===
    arith_rows = read_csv(os.path.join(opprof_dir, "ArithmeticUtilization.csv"))
    lines.extend(_arithmetic_summary(arith_rows))

    # === Footer ===
    lines.append("")
    lines.append("--- 原始数据位置 ---")
    lines.append(f"  CSV 文件: {round_dir}/")
    lines.append("  如需逐核详情，请 Read 对应 CSV 文件。")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate msprof statistical summary and archive CSV data.")
    parser.add_argument("opprof_dir", help="Path to OPPROF_{timestamp}_XXX directory")
    parser.add_argument("ops_dir", help="Path to operator directory (e.g., ops/Softmax)")
    parser.add_argument("--round-name", help="Override round directory name (default: auto-increment)")
    args = parser.parse_args()

    opprof_dir = os.path.abspath(args.opprof_dir)
    ops_dir = os.path.abspath(args.ops_dir)

    if not os.path.isdir(opprof_dir):
        LOGGER.error("Error: '%s' is not a directory.", opprof_dir)
        sys.exit(1)
    if not os.path.isdir(ops_dir):
        LOGGER.error("Error: '%s' is not a directory.", ops_dir)
        sys.exit(1)

    # Check PipeUtilization.csv exists
    if not os.path.exists(os.path.join(opprof_dir, "PipeUtilization.csv")):
        LOGGER.error("Error: PipeUtilization.csv not found in %s", opprof_dir)
        sys.exit(1)

    # Determine round directory
    perf_dir = os.path.join(ops_dir, "docs", "perf")
    if args.round_name:
        round_dir = os.path.join(perf_dir, args.round_name)
    else:
        round_dir = find_next_round(perf_dir)

    # Archive CSVs
    copied = archive_csvs(opprof_dir, round_dir)
    LOGGER.info("Archived %d CSV files to: %s", len(copied), round_dir)

    # Generate summary
    summary = generate_summary(opprof_dir, round_dir, ops_dir)
    summary_path = os.path.join(round_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    LOGGER.info("Summary written to: %s", summary_path)
    LOGGER.info("")
    LOGGER.info(summary)


if __name__ == "__main__":
    main()
