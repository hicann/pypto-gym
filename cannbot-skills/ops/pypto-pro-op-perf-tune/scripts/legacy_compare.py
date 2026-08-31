#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.

"""既有 compare/quick/batch 流程的保留实现。

这些函数是 Stage 5 证据协议引入前既有代码的逐字保留（仅改名 *_legacy），由
msprof_perf_summary.py 在"未传 --case-manifest"时延迟导入，行为与既有流程一致，
不参与 Stage 5 证据协议。
"""

from __future__ import annotations

import inspect
import json
import re
import statistics
import time
from importlib import import_module
from pathlib import Path
from typing import Any, Dict

# 延迟导入触发时主模块已完整加载，不会形成循环依赖。
from msprof_perf_summary import (  # noqa: E402
    LOGGER,
    CompareSummaryInput,
    _cleanup_prof_dirs,
    _compute_speedup_stats,
    _extract_trace_table_rows,
    _load_performance_json,
    _measure_pypto_runs,
)




def _find_cls(module, preferred: str):
    nn = import_module("torch.nn")
    c = getattr(module, preferred, None)
    if inspect.isclass(c) and issubclass(c, nn.Module):
        return c
    for _, v in vars(module).items():
        if inspect.isclass(v) and issubclass(v, nn.Module) and v is not nn.Module:
            return v
    raise AttributeError(f"no nn.Module subclass found in {module.__file__}")



def _move(v, d):
    torch = import_module("torch")
    if isinstance(v, torch.Tensor):
        return v.to(d)
    if isinstance(v, (list, tuple)):
        return type(v)(_move(x, d) for x in v)
    return v



def _clone(v):
    """Deep clone tensor / list of tensors."""
    torch = import_module("torch")
    if isinstance(v, torch.Tensor):
        return v.clone()
    if isinstance(v, (list, tuple)):
        return type(v)(_clone(x) for x in v)
    return v



def _parse_golden_table_row(line):
    if not line.startswith("|") or "---" in line:
        return None
    parts = [part.strip() for part in line.split("|")]
    if len(parts) < 5:
        return None
    try:
        return parts[1], parts[2], parts[3], float(parts[4].replace("us", "").strip())
    except ValueError:
        return None



def _parse_golden_summary_table(content):
    cases = []
    in_summary_table = False
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("|") and "case" in line.lower() and "E2E" in line:
            in_summary_table = True
            continue
        if not in_summary_table:
            continue
        if not line.startswith("|"):
            in_summary_table = False
            continue
        parsed = _parse_golden_table_row(line)
        if parsed is not None:
            cases.append(parsed)
    return cases



def _parse_legacy_golden(content):
    shape, dtype, duration = "?", "?", None
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("- **Input Shape**:"):
            shape = line.split(":", 1)[1].strip()
        elif line.startswith("- **dtype**:"):
            dtype = line.split(":", 1)[1].strip()
        elif "Total kernel duration" in line:
            match = re.search(r'([\d.]+)\s*us', line)
            if match:
                duration = float(match.group(1))
    return [("P0", shape, dtype, duration)] if duration is not None else []



def _parse_golden_report(out_dir: Path):
    """从 GOLDEN_PERF_REPORT.md 解析 golden 性能数据。"""
    golden_path = out_dir / "GOLDEN_PERF_REPORT.md"
    if not golden_path.exists():
        return None, "GOLDEN_PERF_REPORT.md not found"
    content = golden_path.read_text(encoding="utf-8")
    cases = _parse_golden_summary_table(content) or _parse_legacy_golden(content)

    if not cases:
        return None, "no golden E2E data found in GOLDEN_PERF_REPORT.md"
    return cases, None



def _profile_case_contract_error_legacy(cases):
    if len(cases) <= 1:
        return None
    return (
        f"GOLDEN_PERF_REPORT.md contains {len(cases)} cases, but test_<op>.py "
        "has no standard per-case selector. Refusing to reuse one aggregate "
        "msprof duration for every case; profile one case per report instead."
    )



def _validated_golden_cases(out_dir):
    golden_cases, golden_error = _parse_golden_report(out_dir)
    if not golden_cases:
        LOGGER.error("[ERROR] Failed to load golden data: %s", golden_error)
        return None
    LOGGER.info(
        "[INFO] Loaded %s golden cases from GOLDEN_PERF_REPORT.md",
        len(golden_cases),
    )
    contract_error = _profile_case_contract_error_legacy(golden_cases)
    if contract_error:
        LOGGER.error("[ERROR] %s", contract_error)
        return None
    return golden_cases



def _measure_pypto_legacy(out_dir, args, device_id):
    """既有测量包装：返回 (duration, err, prof_dir) 三元组。"""
    duration, err, evidence_dir, _meta = _measure_pypto_runs(
        out_dir, args, device_id, quick=False, case_name=None
    )
    return duration, err, evidence_dir



def _measure_pypto_quick_legacy(out_dir, args, device_id):
    """既有 quick 测量包装：返回 (duration, err, prof_dir) 三元组。"""
    duration, err, evidence_dir, _meta = _measure_pypto_runs(
        out_dir, args, device_id, quick=True, case_name=None
    )
    return duration, err, evidence_dir




def _run_measurement_loop_legacy(out_dir, golden_cases, args, device_id, measure):
    rows, speedups, ref_times, asc_times = [], [], [], []
    pypto_us, pypto_err, pypto_prof_dir = measure(out_dir, args, device_id)
    for case_name, golden_shape, golden_dtype, golden_us in golden_cases:
        if pypto_us is not None and pypto_us > 0:
            speedup = golden_us / pypto_us
            speedups.append(speedup)
            ref_times.append(golden_us)
            asc_times.append(pypto_us)
            LOGGER.info(f"{case_name:<20} {golden_shape:<35} {golden_dtype:<10} "
                  f"{golden_us:>12.2f} {pypto_us:>12.2f} {speedup:>9.3f}x")
        else:
            LOGGER.warning(f"{case_name:<20} {golden_shape:<35} {golden_dtype:<10} "
                  f"{'N/A' if golden_us is None else f'{golden_us:.2f}':>12} "
                  f"{'N/A' if pypto_us is None else f'{pypto_us:.2f}':>12} "
                  f"{'N/A':>10}  (pypto_err={pypto_err})")

        rows.append({
            "case": case_name, "shape": golden_shape, "dtype": golden_dtype,
            "ref_us": golden_us, "asc_us": pypto_us,
            "speedup": (golden_us / pypto_us) if (golden_us and pypto_us and pypto_us > 0) else None,
            "ref_error": None,
            "asc_error": pypto_err,
            "ref_prof_dir": None,
            "asc_prof_dir": pypto_prof_dir,
        })
    if not args.keep_prof:
        _cleanup_prof_dirs(pypto_prof_dir)
    return rows, speedups, ref_times, asc_times



def _run_compare_loop_legacy(out_dir, golden_cases, args, device_id):
    """Run standard msprof measurement for all golden cases."""
    return _run_measurement_loop_legacy(
        out_dir, golden_cases, args, device_id, _measure_pypto_legacy
    )



def _run_quick_loop_legacy(out_dir, golden_cases, args, device_id):
    """Run lightweight msprof measurement for the single golden case."""
    return _run_measurement_loop_legacy(
        out_dir, golden_cases, args, device_id, _measure_pypto_quick_legacy
    )



def _compute_compare_summary_legacy(csi):
    """Compute the summary statistics dict for compare mode."""
    speedup_stats = _compute_speedup_stats(csi.speedups)
    timing_stats = _compute_timing_stats_legacy(csi.ref_times, csi.asc_times)
    return {
        "task": csi.out_dir.name,
        "task_dir": str(csi.out_dir),
        "n_cases_total": csi.n_cases,
        **speedup_stats,
        **timing_stats,
        "warmup": csi.args.warmup,
        "repeats": csi.args.repeats,
        "seed": csi.args.seed,
        "device_id": csi.device_id,
        "device_select_source": csi.device_src,
        "timing_method": "msprof.op_summary.Task_Duration",
        "per_case": csi.rows,
    }



def _compute_timing_stats_legacy(ref_times: list, asc_times: list) -> dict:
    """Compute timing statistics for reference and AscendC implementations.

    Returns a dict with mean/median/total for ref and asc, plus total_speedup.
    """
    if ref_times:
        ref_stats = {
            "mean_ref_us": statistics.mean(ref_times),
            "median_ref_us": statistics.median(ref_times),
            "total_ref_us": sum(ref_times),
        }
    else:
        ref_stats = {"mean_ref_us": None, "median_ref_us": None, "total_ref_us": None}

    if asc_times:
        asc_stats = {
            "mean_asc_us": statistics.mean(asc_times),
            "median_asc_us": statistics.median(asc_times),
            "total_asc_us": sum(asc_times),
        }
    else:
        asc_stats = {"mean_asc_us": None, "median_asc_us": None, "total_asc_us": None}

    total_speedup = None
    if ref_times and asc_times:
        asc_sum = sum(asc_times)
        if asc_sum > 0:
            total_speedup = sum(ref_times) / asc_sum

    return {**ref_stats, **asc_stats, "total_speedup": total_speedup}



def _log_and_save_compare_reports_legacy(summary, out_dir, speedups, n_cases):
    """Log summary results and save JSON/log/Markdown reports."""
    LOGGER.info("-" * 100)
    if speedups:
        LOGGER.info("--- Speedup ---")
        LOGGER.info(f"  Geomean : {summary['geomean_speedup']:.2f}x  ← primary metric")
        LOGGER.info(f"  Mean    : {summary['mean_speedup']:.2f}x")
        LOGGER.info(f"  Median  : {summary['median_speedup']:.2f}x")
        LOGGER.info(f"  Min/Max : {summary['min_speedup']:.2f}x / {summary['max_speedup']:.2f}x")
        LOGGER.info(f"  Valid   : {len(speedups)}/{n_cases}")
    if summary.get("mean_ref_us") is not None:
        LOGGER.info("--- Task Duration (us) ---")
        LOGGER.info(
            f"  Ref  mean/median/total : {summary['mean_ref_us']:.2f} / "
            f"{summary['median_ref_us']:.2f} / {summary['total_ref_us']:.2f}"
        )
        LOGGER.info(
            f"  Asc  mean/median/total : {summary['mean_asc_us']:.2f} / "
            f"{summary['median_asc_us']:.2f} / {summary['total_asc_us']:.2f}"
        )
        LOGGER.info(f"  Total speedup (Σref/Σasc) : {summary['total_speedup']:.2f}x")
    LOGGER.info("=" * 100)

    # 保存 JSON 报告
    json_path = out_dir / "performance.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    LOGGER.info(f"\n[INFO] JSON report saved to: {json_path}")

    # 保存打屏日志
    log_path = out_dir / "performance.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(_report_compare_to_text_legacy(summary))
    LOGGER.info(f"[INFO] Console report saved to: {log_path}")

    # 保存 Markdown 报告
    md_path = out_dir / "perf_report.md"
    md = _report_compare_to_markdown_legacy(summary)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    LOGGER.info(f"[INFO] Markdown report saved to: {md_path}")



def _log_compare_header_legacy(out_dir, args):
    """Log the compare mode header."""
    LOGGER.info("=" * 100)
    LOGGER.info(f"Kernel-level Performance (msprof): {out_dir.name}  "
          f"(warmup={args.warmup}, repeats={args.repeats}, seed={args.seed})")
    LOGGER.info("=" * 100)
    LOGGER.info(f"{'Case':<5} {'Shape':<35} {'dtype':<10} {'Golden(us)':>12} {'PyPTO(us)':>12} {'Speedup':>10}")
    LOGGER.info("-" * 100)



def _add_compare_header_legacy(lines, report):
    """Add the header section to the compare markdown report."""
    lines.append("# 性能评估结果")
    lines.append("")
    lines.append(f"- **Operator**: {report['task']}")
    lines.append(f"- **Device**: npu:{report['device_id']} (source={report['device_select_source']})")
    lines.append(f"- **Warmup**: {report['warmup']}")
    lines.append(f"- **Repeats**: {report['repeats']}")
    lines.append(f"- **Seed**: {report['seed']}")
    lines.append(f"- **Timing method**: {report['timing_method']}")
    lines.append("")



def _add_per_case_table_legacy(lines, report):
    """Add the per-case comparison table."""
    if not report.get("per_case"):
        return
    lines.append("## 性能对比")
    lines.append("")
    lines.append("| Case | Shape | DType | PyPTO算子(us) | Golden(us) | 加速比 |")
    lines.append("| ---- | ----- | ----- | ------------- | -------- | -------------- |")
    for case in report["per_case"]:
        shape = case.get("shape", "?")
        dtype = case.get("dtype", "?")
        ref = case.get("ref_us")
        asc = case.get("asc_us")
        sp = case.get("speedup")
        ref_str = f"{ref:.2f}" if ref is not None else "N/A"
        asc_str = f"{asc:.2f}" if asc is not None else "N/A"
        sp_str = f"{sp:.3f}" if sp is not None else "N/A"
        lines.append(f"| {case['case']} | {shape} | {dtype} | {asc_str} | {ref_str} | {sp_str} |")
    lines.append("")



def _add_summary_section_legacy(lines, report):
    """Add the summary and dtype tables."""
    if report.get("geomean_speedup") is None:
        return
    lines.append("## 全量汇总")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("| ---- | -- |")
    lines.append(f"| 用例数 | {report['n_cases_total']} |")
    lines.append(f"| 平均加速比（>1 表示PyPTO算子更快） | {report['mean_speedup']:.3f} |")
    better = sum(1 for c in report.get('per_case', []) if c.get('speedup') and c['speedup'] > 1)
    worse = sum(1 for c in report.get('per_case', []) if c.get('speedup') and c['speedup'] < 1)
    lines.append(f"| PyPTO算子更优（比值>1） | {better} |")
    lines.append(f"| Golden更优（比值<1） | {worse} |")
    lines.append("")

    dtype_groups = {}
    for case in report.get("per_case", []):
        dtype = case.get("dtype", "?")
        sp = case.get("speedup")
        if sp is not None:
            dtype_groups.setdefault(dtype, []).append(sp)
    if dtype_groups:
        lines.append("### 按数据类型汇总")
        lines.append("")
        lines.append("| DType | 用例数 | 平均加速比 | PyPTO算子更优 | Golden更优 |")
        lines.append("| ----- | ------ | ------------------- | ------------- | -------- |")
        for dtype, sps in sorted(dtype_groups.items()):
            mean_sp = statistics.mean(sps)
            better = sum(1 for sp in sps if sp > 1)
            worse = sum(1 for sp in sps if sp < 1)
            lines.append(f"| {dtype} | {len(sps)} | {mean_sp:.3f} | {better} | {worse} |")
        lines.append("")



def _add_analysis_sections_legacy(lines, report):
    """Add the short analysis and deep bottleneck analysis sections."""
    lines.append("## 简短分析")
    lines.append("")
    if report.get("mean_speedup") is not None:
        if report["mean_speedup"] > 1:
            lines.append(f"- 平均加速比 {report['mean_speedup']:.3f} 大于 1，PyPTO算子整体有优势。")
        else:
            lines.append(f"- 平均加速比 {report['mean_speedup']:.3f} 小于 1，Golden路径整体更优。")
    lines.append("- 详细瓶颈分析见 msprof 归档目录（op_summary_*.csv + summary.txt）。")
    lines.append("")

    lines.append("## 深度瓶颈分析")
    lines.append("")
    lines.append(
        "如需进一步分析性能瓶颈（各流水线利用率、核间负载均衡、主 Bound 判定），"
        "可运行："
    )
    lines.append("```bash")
    lines.append(
        f"python3 ${{SKILL_PATH}}/scripts/msprof_perf_summary.py "
        f"{report.get('prof_group_dir', './PROF_GROUP_*')} {report['task']}"
    )
    lines.append("```")
    lines.append("")
    lines.append("")
    lines.append("")



def _report_compare_to_markdown_legacy(report: Dict[str, Any]) -> str:
    lines = []
    _add_compare_header_legacy(lines, report)
    _add_per_case_table_legacy(lines, report)
    _add_summary_section_legacy(lines, report)
    _add_analysis_sections_legacy(lines, report)
    return "\n".join(lines)



def _report_compare_to_text_legacy(report: Dict[str, Any]) -> str:
    lines = []
    lines.append("=" * 100)
    lines.append(f"Kernel-level Performance (msprof): {report['task']}  "
                 f"(warmup={report['warmup']}, repeats={report['repeats']}, seed={report['seed']})")
    lines.append("=" * 100)
    lines.append(f"{'Case':<5} {'Shape':<35} {'dtype':<10} {'Golden(us)':>12} {'PyPTO(us)':>12} {'Speedup':>10}")
    lines.append("-" * 100)

    for case in report.get("per_case", []):
        shape = case.get("shape", "?")
        dtype = case.get("dtype", "?")
        ref_us = case.get("ref_us")
        asc_us = case.get("asc_us")
        sp = case.get("speedup")
        if ref_us is not None and asc_us is not None and sp is not None:
            lines.append(f"{case['case']:<5} {shape:<35} {dtype:<10} {ref_us:>12.2f} {asc_us:>12.2f} {sp:>9.3f}x")
        else:
            ref_str = f"{ref_us:.2f}" if ref_us is not None else "N/A"
            asc_str = f"{asc_us:.2f}" if asc_us is not None else "N/A"
            ref_err = case.get("ref_error", "")
            asc_err = case.get("asc_error", "")
            lines.append(f"{case['case']:<5} {shape:<35} {dtype:<10} "
                         f"{ref_str:>12} {asc_str:>12} "
                         f"{'N/A':>10}  (ref_err={ref_err}, asc_err={asc_err})")

    lines.append("-" * 100)
    if report.get("geomean_speedup") is not None:
        lines.append("--- Speedup ---")
        lines.append(f"  Geomean : {report['geomean_speedup']:.2f}x  ← 主指标")
        lines.append(f"  Mean    : {report['mean_speedup']:.2f}x")
        lines.append(f"  Median  : {report['median_speedup']:.2f}x")
        lines.append(f"  Min/Max : {report['min_speedup']:.2f}x / {report['max_speedup']:.2f}x")
        lines.append(f"  Valid   : {report['n_cases_valid']}/{report['n_cases_total']}")
    if report.get("mean_ref_us") is not None:
        lines.append("--- Task Duration (us) ---")
        lines.append(
            f"  Ref  mean/median/total : {report['mean_ref_us']:.2f} / "
            f"{report['median_ref_us']:.2f} / {report['total_ref_us']:.2f}"
        )
        lines.append(
            f"  Asc  mean/median/total : {report['mean_asc_us']:.2f} / "
            f"{report['median_asc_us']:.2f} / {report['total_asc_us']:.2f}"
        )
        lines.append(f"  Total speedup (Σref/Σasc) : {report['total_speedup']:.2f}x")
    lines.append("=" * 100)

    return "\n".join(lines)



def _run_compare_mode_legacy(args, out_dir, device_id, device_src):
    """既有对比模式：GOLDEN_PERF_REPORT.md (golden) vs test_{op}.py (PyPTO 算子)。

    保持既有行为；未传 --case-manifest 时由 run_compare_mode 分发到此。
    """
    LOGGER.info(f"[INFO] Using NPU device {device_id} (source={device_src})")

    golden_cases = _validated_golden_cases(out_dir)
    if golden_cases is None:
        return 1

    _log_compare_header_legacy(out_dir, args)
    rows, speedups, ref_times, asc_times = _run_compare_loop_legacy(
        out_dir, golden_cases, args, device_id)

    csi = CompareSummaryInput(
        out_dir, rows, speedups, ref_times, asc_times, len(golden_cases), args, device_id, device_src)
    summary = _compute_compare_summary_legacy(csi)
    _log_and_save_compare_reports_legacy(summary, out_dir, speedups, len(golden_cases))
    return 0



def _run_quick_mode_legacy(args, out_dir, device_id, device_src):
    """Legacy quick mode: only fetch kernel time for each repeat.

    Keeps existing behavior; dispatched here by run_quick_mode when --case-manifest is not passed.
    """
    LOGGER.info(f"[INFO] Using NPU device {device_id} (source={device_src})")
    LOGGER.info("[INFO] Quick mode: kernel timing only (no aic-metrics)")

    golden_cases = _validated_golden_cases(out_dir)
    if golden_cases is None:
        return 1

    _log_compare_header_legacy(out_dir, args)
    rows, speedups, ref_times, asc_times = _run_quick_loop_legacy(
        out_dir, golden_cases, args, device_id)

    csi = CompareSummaryInput(
        out_dir, rows, speedups, ref_times, asc_times, len(golden_cases), args, device_id, device_src)
    summary = _compute_compare_summary_legacy(csi)
    summary["timing_method"] = "msprof.quick.Task_Duration"
    _log_and_save_compare_reports_legacy(summary, out_dir, speedups, len(golden_cases))
    return 0



def _build_batch_md_summary_table_legacy(op_results):
    """Build the batch summary table markdown lines."""
    md_lines = []
    if op_results:
        md_lines.append("## 性能汇总")
        md_lines.append("")
        md_lines.append(
            "| 算子名称 | 用例数 | 有效用例 | 几何平均加速比 | 平均加速比 | 状态 |"
        )
        md_lines.append("| -------- | ------ | -------- | -------------- | ---------- | ---- |")
        for op in op_results:
            data = op["data"]
            name = op["name"]
            n_total = data.get("n_cases_total", 0)
            n_valid = data.get("n_cases_valid", 0)
            geo = data.get("geomean_speedup")
            mean = data.get("mean_speedup")
            geo_str = f"{geo:.3f}" if geo is not None else "N/A"
            mean_str = f"{mean:.3f}" if mean is not None else "N/A"
            status = "✅" if geo is not None and geo > 1 else "⚠️" if geo is not None else "❌"
            md_lines.append(f"| {name} | {n_total} | {n_valid} | {geo_str} | {mean_str} | {status} |")
        md_lines.append("")
    return md_lines



def _build_batch_md_per_op_details_legacy(op_results):
    """Build per-operator detail tables in markdown."""
    md_lines = []
    for op in op_results:
        data = op["data"]
        name = op["name"]
        md_lines.append(f"## {name}")
        md_lines.append("")
        if data.get("per_case"):
            md_lines.append("| Case | Shape | DType | PyPTO算子(us) | Golden(us) | 加速比 |")
            md_lines.append("| ---- | ----- | ----- | ------------- | -------- | -------------- |")
            for case in data["per_case"]:
                shape = case.get("shape", "?")
                dtype = case.get("dtype", "?")
                ref = case.get("ref_us")
                asc = case.get("asc_us")
                sp = case.get("speedup")
                ref_str = f"{ref:.2f}" if ref is not None else "N/A"
                asc_str = f"{asc:.2f}" if asc is not None else "N/A"
                sp_str = f"{sp:.3f}" if sp is not None else "N/A"
                md_lines.append(f"| {case['case']} | {shape} | {dtype} | {asc_str} | {ref_str} | {sp_str} |")
            md_lines.append("")
    return md_lines



def _build_batch_md_trace_section_legacy(trace_rows):
    """Build trace table section in markdown if rows exist."""
    if not trace_rows:
        return []
    md_lines = [
        "## Trace 汇总表",
        "",
        ("| Level | Problem ID | 算子名称 | 算子类型 | 编译通过 | 精度正确 | "
         "PyTorch 参考延迟 | 生成AscendC代码延迟 | 加速比 | 最终状态 | "
         "精度正确 | 性能0.6x pytorch | 性能0.8x pytorch |"),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    md_lines.extend(trace_rows)
    md_lines.append("")
    return md_lines



def _generate_batch_md_report_legacy(args, op_results, trace_rows, base_dir):
    """Generate and save the batch Markdown report."""
    md_lines = []
    md_lines.append("# 📊 算子批量性能汇总报告")
    md_lines.append("")
    md_lines.append(f"- **扫描目录**: {base_dir}")
    md_lines.append(f"- **算子总数**: {len(op_results)}")
    md_lines.append(f"- **生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md_lines.append("")

    md_lines.extend(_build_batch_md_summary_table_legacy(op_results))
    md_lines.extend(_build_batch_md_per_op_details_legacy(op_results))
    md_lines.extend(_build_batch_md_trace_section_legacy(trace_rows))

    md_path = Path(args.output_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    LOGGER.info("Batch markdown report saved to: %s", md_path)



def _generate_batch_json_report_legacy(args, op_results, base_dir):
    """Generate and save the batch JSON summary."""
    batch_summary = {
        "base_dir": str(base_dir),
        "n_operators": len(op_results),
        "operators": [
            {
                "name": op["name"],
                "n_cases_total": op["data"].get("n_cases_total", 0),
                "n_cases_valid": op["data"].get("n_cases_valid", 0),
                "geomean_speedup": op["data"].get("geomean_speedup"),
                "mean_speedup": op["data"].get("mean_speedup"),
                "mean_ref_us": op["data"].get("mean_ref_us"),
                "mean_asc_us": op["data"].get("mean_asc_us"),
            }
            for op in op_results
        ],
        "generated_at": time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(batch_summary, f, indent=2, ensure_ascii=False)
    LOGGER.info("Batch JSON summary saved to: %s", json_path)



def run_batch_mode_legacy(args):
    """执行批量模式：扫描 base_dir 下所有子目录，汇总性能报告。"""
    base_dir = Path(args.batch).resolve()

    if not base_dir.is_dir():
        raise ValueError("'%s' is not a directory." % base_dir)

    # 收集所有子目录的 performance.json
    op_results = []
    for subdir in sorted(base_dir.iterdir()):
        if not subdir.is_dir():
            continue
        perf_data = _load_performance_json(subdir)
        if perf_data:
            op_results.append({
                "name": subdir.name,
                "data": perf_data,
                "dir": subdir,
            })

    # 同时收集 trace.md 中的表格行（兼容旧 batch_report.py 功能）
    trace_rows = []
    for subdir in sorted(base_dir.iterdir()):
        if not subdir.is_dir():
            continue
        trace_file = subdir / "trace.md"
        if trace_file.exists():
            rows = _extract_trace_table_rows(str(trace_file))
            trace_rows.extend(rows)

    LOGGER.info("Found %d operators with performance.json in %s", len(op_results), base_dir)

    if args.output_md:
        _generate_batch_md_report_legacy(args, op_results, trace_rows, base_dir)

    if args.output_json:
        _generate_batch_json_report_legacy(args, op_results, base_dir)

    return 0


# ============================================================================
# 主入口
# ============================================================================


# ============================================================================
# compare/quick：case manifest 驱动 PyPTO target-kernel 采集；Golden 提供默认目标
# ============================================================================
