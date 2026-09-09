#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.

"""性能采集批量汇总的证据复核与渲染。

batch 汇总前从每个算子的七组原始 CSV 重建计时与 bound 诊断，与
performance.json/measurement.json/evidence_status.json 逐项对照，防止派生数字
被篡改或手滑。由 msprof_perf_summary.py 的 batch 证据路径延迟导入。
"""

from __future__ import annotations

import csv
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Tuple

from golden_contract import (
    parse_golden_contract,
    performance_case_source_error,
)
# 延迟导入触发时主模块已完整加载，不会形成循环依赖。
from msprof_perf_summary import (  # noqa: E402
    DEFAULT_GOLDEN_TARGET_THRESHOLD,
    LOGGER,
    METRICS,
    _aggregate_durations,
    _atomic_write_text,
    _compute_speedup_stats,
    _is_positive_finite,
    _merge_row_values,
    _resolved_evidence_round,
    _selected_executable_sha256,
    safe_case_dir_name,
)
from evidence_cli import diagnose_bound_route


def _read_metric_rows(repeat_dir: Path, metric: str):
    """读取并校验单个指标的归档 CSV；返回 (rows, error)。"""
    metric_path = repeat_dir / f"op_summary_{metric}.csv"
    if not metric_path.is_file() or metric_path.is_symlink():
        return None, f"missing archived metric: {metric_path.name}"
    try:
        with metric_path.open(
                "r", encoding="utf-8", errors="strict", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            fieldnames = reader.fieldnames
            if (
                not fieldnames
                or "Op Name" not in fieldnames
                or len(fieldnames) != len(set(fieldnames))
            ):
                return None, (
                    f"invalid archived metric schema: {metric_path.name}"
                )
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        return None, (
            f"cannot parse archived metric {metric_path.name}: {error}"
        )
    return rows, None


def _read_archived_repeat_evidence(
        repeat_dir: Path, target_op_name: str
) -> Tuple[Optional[Dict[str, Any]], Optional[float], Optional[str]]:
    """Rebuild one repeat's timing and bound route from its seven raw CSVs."""
    merged: Dict[str, Any] = {"_metric_sources": {}, "_missing_metrics": []}
    pipe_duration = None
    for metric in METRICS:
        metric_path = repeat_dir / f"op_summary_{metric}.csv"
        rows, metric_error = _read_metric_rows(repeat_dir, metric)
        if metric_error:
            return None, None, metric_error
        matches = [
            row for row in rows
            if str(row.get("Op Name", "")).strip() == target_op_name
        ]
        if len(matches) != 1:
            return None, None, (
                f"archived metric {metric_path.name} expected exactly one row for "
                f"target_op_name={target_op_name!r}; found {len(matches)}"
            )
        row = matches[0]
        merged["_metric_sources"][metric] = str(metric_path)
        _merge_row_values(merged, row)
        if metric == "PipeUtilization":
            try:
                pipe_duration = float(row.get("Task Duration(us)", ""))
            except (TypeError, ValueError):
                pipe_duration = None
            if not _is_positive_finite(pipe_duration):
                return None, None, (
                    "archived PipeUtilization target row has invalid Task Duration(us)"
                )
    try:
        diagnosis = diagnose_bound_route(merged)
    except ValueError as error:
        return None, None, f"archived bound diagnosis cannot be recomputed: {error}"
    return diagnosis, float(pipe_duration), None


def _check_collection_protocol(
    op_dir: Path, manifest: Dict[str, Any], data: Dict[str, Any]
) -> Optional[str]:
    """校验 collection.json 与 performance.json 的协议字段一致性。"""
    target_op_name = data.get("target_op_name")
    if not isinstance(target_op_name, str) or not target_op_name.strip():
        return "target op name is missing"
    for field, minimum in (
        ("device_id", 0), ("seed", 0), ("warmup", 0), ("repeats", 1),
    ):
        value = data.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            return f"performance.json {field} is invalid"
        if manifest.get(field) != value:
            return f"collection protocol mismatch: {field}"
    if data["seed"] != 42:
        return "collection seed must be 42"
    if manifest.get("performance_cases") != data.get("performance_cases"):
        return "performance case source mismatch between manifest and performance.json"
    source_error = performance_case_source_error(manifest.get("performance_cases"))
    if source_error:
        return source_error
    expected_sha256 = manifest.get("executable_sha256")
    if not isinstance(expected_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_sha256):
        return "collection.json executable_sha256 is invalid"
    performance_sha256 = data.get("executable_sha256")
    if not isinstance(performance_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", performance_sha256):
        return "performance.json executable_sha256 is invalid"
    if expected_sha256 != performance_sha256:
        return "executable sha256 mismatch between collection.json and performance.json"
    _, current_sha256, executable_error = _selected_executable_sha256(op_dir)
    if executable_error:
        return executable_error
    if current_sha256 != expected_sha256:
        return "selected executable sha256 differs from the final compare"
    return None


class CollectionManifest(NamedTuple):
    """校验通过的 collection 证据上下文；校验失败时除 error 外均为 None。"""
    round_path: Optional[Path]
    manifest: Optional[Dict[str, Any]]
    data_cases: Optional[Dict[str, Any]]
    expected_cases: Optional[list]
    repeats: Optional[int]
    error: Optional[str]


def _check_collection_manifest(
    op_dir: Path, data: Dict[str, Any]
) -> CollectionManifest:
    """校验 collection.json 与 performance.json 的协议一致性。"""
    if data.get("profiling_mode") != "compare":
        return CollectionManifest(None, None, None, None, None, "profiling_mode is not compare")
    try:
        round_path = _resolved_evidence_round(op_dir, data.get("deep_profile_round"))
    except (OSError, ValueError) as error:
        return CollectionManifest(None, None, None, None, None, str(error))
    manifest_path = round_path / "collection.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return CollectionManifest(None, None, None, None, None, f"cannot load collection manifest: {error}")
    if manifest.get("status") != "complete" or manifest.get("mode") != "compare":
        return CollectionManifest(None, None, None, None, None, "collection manifest is not a complete compare")
    if manifest.get("collection_id") != data.get("collection_id"):
        return CollectionManifest(None, None, None, None, None, "collection id mismatch")
    if manifest.get("target_op_name") != data.get("target_op_name"):
        return CollectionManifest(None, None, None, None, None, "target op mismatch")
    protocol_error = _check_collection_protocol(op_dir, manifest, data)
    if protocol_error:
        return CollectionManifest(None, None, None, None, None, protocol_error)
    expected_cases = manifest.get("expected_cases")
    if not isinstance(expected_cases, list) or not expected_cases:
        return CollectionManifest(None, None, None, None, None, "collection manifest has no expected cases")
    if manifest.get("completed_cases") != len(expected_cases):
        return CollectionManifest(None, None, None, None, None, (
            "collection manifest did not complete every expected case"
        ))
    per_case = data.get("per_case")
    if not isinstance(per_case, list) or len(per_case) != len(expected_cases):
        return CollectionManifest(None, None, None, None, None, (
            "performance.json per_case does not cover every expected case"
        ))
    data_cases = {
        str(record.get("case")): record
        for record in per_case
        if isinstance(record, dict) and record.get("case") is not None
    }
    if set(data_cases) != {str(case_id) for case_id in expected_cases}:
        return CollectionManifest(None, None, None, None, None, (
            "performance.json per_case ids do not match collection manifest"
        ))
    repeats = data.get("repeats")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats <= 0:
        return CollectionManifest(None, None, None, None, None, "performance.json repeats is invalid")
    return CollectionManifest(round_path, manifest, data_cases, expected_cases, repeats, None)


class RepeatEvidence(NamedTuple):
    """逐 repeat 证据集（目录对、诊断与样本）。"""
    repeat_dirs: list
    expected_repeat_dirs: list
    diagnoses: list
    samples: list


def _check_repeat(
    case_dir: Path, case_id: str, data: Dict[str, Any], evidence: RepeatEvidence,
) -> Optional[str]:
    """校验单个 case 的逐 repeat 原始证据（evidence_status.json + 七组 CSV）。"""
    for repeat_index, (raw_repeat_dir, expected_repeat_dir) in enumerate(
            zip(evidence.repeat_dirs, evidence.expected_repeat_dirs), 1):
        try:
            repeat_dir = Path(str(raw_repeat_dir)).resolve(strict=True)
            repeat_dir.relative_to(case_dir.resolve(strict=True))
        except (OSError, ValueError) as error:
            return f"unsafe repeat evidence directory for {case_id} repeat {repeat_index}: {error}"
        if repeat_dir != expected_repeat_dir.resolve() or repeat_dir.is_symlink():
            return f"repeat evidence directory mismatch: {case_id} repeat {repeat_index}"
        status_path = repeat_dir / "evidence_status.json"
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            return f"cannot read repeat evidence status {case_id}/{repeat_index}: {error}"
        status_ok = (
            status.get("collection_id") == data.get("collection_id")
            and str(status.get("case")) == str(case_id)
            and status.get("target_op_name") == data.get("target_op_name")
            and status.get("seven_metric_status") == "complete"
            and status.get("bound_diagnosis") == evidence.diagnoses[repeat_index - 1]
        )
        if not status_ok:
            return f"repeat evidence status mismatch: {case_id} repeat {repeat_index}"
        raw_diagnosis, raw_duration, raw_error = _read_archived_repeat_evidence(
            repeat_dir, data.get("target_op_name")
        )
        if raw_error:
            return f"invalid raw evidence for {case_id} repeat {repeat_index}: {raw_error}"
        if not math.isclose(
            raw_duration, float(evidence.samples[repeat_index - 1]),
            rel_tol=1e-9, abs_tol=1e-6,
        ):
            return f"raw PipeUtilization duration mismatch: {case_id} repeat {repeat_index}"
        if raw_diagnosis != evidence.diagnoses[repeat_index - 1]:
            return f"raw bound diagnosis mismatch: {case_id} repeat {repeat_index}"
    return None


def _check_diagnoses(
    diagnoses: list, repeats: int, data_cases: Dict[str, Any], case_id: str
) -> Optional[str]:
    """校验诊断列表数量与合同字段。"""
    if not isinstance(diagnoses, list) or len(diagnoses) != repeats:
        return f"per-case bound diagnoses are invalid: {case_id}"
    output_diagnoses = data_cases[str(case_id)].get("repeat_bound_diagnoses")
    if output_diagnoses != diagnoses:
        return f"performance.json bound diagnoses mismatch: {case_id}"
    for diagnosis in diagnoses:
        kind_ok = (
            isinstance(diagnosis, dict)
            and diagnosis.get("evidence_kind") == "op_summary_ratio_route"
            and diagnosis.get("roofline_terminal_status") == "insufficient_evidence"
            and diagnosis.get("pipeline_overlap_status") == "unverified"
            and diagnosis.get("completion_eligible") is False
        )
        if not kind_ok:
            return f"per-case bound diagnosis contract is invalid: {case_id}"
    return None


def _check_case_samples(
    record: Dict[str, Any], output_case: Dict[str, Any], case_id: str,
    repeats: int, case_dir: Path,
) -> Tuple[Optional[dict], Optional[str]]:
    """校验样本、selected Op Names 与计时来源；返回 (context, error)。"""
    samples = record.get("duration_samples_us")
    samples_ok = (
        isinstance(samples, list)
        and len(samples) == repeats
        and all(_is_positive_finite(value) for value in samples)
    )
    if not samples_ok:
        return None, f"per-case duration samples are invalid: {case_id}"
    if output_case.get("duration_samples_us") != samples:
        return None, f"performance.json duration samples mismatch: {case_id}"
    selected_names = record.get("selected_op_names")
    if (
        not isinstance(selected_names, list)
        or len(selected_names) != repeats
        or any(name != record.get("target_op_name") for name in selected_names)
    ):
        return None, f"per-case selected Op Names are invalid: {case_id}"
    if output_case.get("selected_op_names") != selected_names:
        return None, f"performance.json selected Op Names mismatch: {case_id}"
    repeat_dirs = record.get("repeat_evidence_dirs")
    if not isinstance(repeat_dirs, list) or len(repeat_dirs) != repeats:
        return None, f"per-case repeat evidence directories are invalid: {case_id}"
    expected_repeat_dirs = [case_dir / f"repeat_{index:03d}" for index in range(1, repeats + 1)]
    expected_timing_sources = [
        str(path / "op_summary_PipeUtilization.csv")
        for path in expected_repeat_dirs
    ]
    timing_metric_ok = (
        record.get("timing_source_metric") == "PipeUtilization"
        and record.get("timing_source_files") == expected_timing_sources
        and output_case.get("timing_source_metric") == "PipeUtilization"
        and output_case.get("timing_source_files") == expected_timing_sources
    )
    if not timing_metric_ok:
        return None, f"per-case timing source mismatch: {case_id}"
    context = {
        "samples": samples,
        "repeat_dirs": repeat_dirs,
        "expected_repeat_dirs": expected_repeat_dirs,
    }
    return context, None


def _check_case_record(
    case_dir: Path, case_id: str, data_cases: Dict[str, Any], data: Dict[str, Any],
    repeats: int,
):
    """校验 measurement.json 与逐 repeat 元数据；返回校验上下文或 error。"""
    measurement = case_dir / "measurement.json"
    if not measurement.is_file():
        return None, f"missing per-case measurement: {case_id}"
    try:
        record = json.loads(measurement.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, f"cannot read per-case measurement {case_id}: {error}"
    if (
        record.get("collection_id") != data.get("collection_id")
        or record.get("target_op_name") != data.get("target_op_name")
        or str(record.get("case")) != str(case_id)
    ):
        return None, f"per-case measurement metadata mismatch: {case_id}"
    diagnoses = record.get("repeat_bound_diagnoses")
    diagnosis_error = _check_diagnoses(diagnoses, repeats, data_cases, case_id)
    if diagnosis_error:
        return None, diagnosis_error
    output_case = data_cases[str(case_id)]
    samples_context, samples_error = _check_case_samples(
        record, output_case, case_id, repeats, case_dir
    )
    if samples_error:
        return None, samples_error
    context = {
        "record": record,
        "output_case": output_case,
        "diagnoses": diagnoses,
        **samples_context,
    }
    return context, None


def _check_per_case(
    round_path: Path, case_id: str, data_cases: Dict[str, Any], data: Dict[str, Any],
    repeats: int,
) -> Optional[str]:
    """校验单个 case 的 measurement.json 与逐 repeat 原始证据。"""
    case_dir = round_path / f"case_{safe_case_dir_name(case_id)}"
    context, record_error = _check_case_record(
        case_dir, case_id, data_cases, data, repeats
    )
    if record_error:
        return record_error
    record = context["record"]
    output_case = context["output_case"]
    diagnoses = context["diagnoses"]
    samples = context["samples"]
    repeat_error = _check_repeat(
        case_dir, case_id, data,
        RepeatEvidence(
            context["repeat_dirs"], context["expected_repeat_dirs"],
            diagnoses, samples,
        ),
    )
    if repeat_error:
        return repeat_error
    expected_aggregation = "trimmed_mean" if repeats >= 3 else "median"
    if record.get("aggregation_method") != expected_aggregation:
        return f"per-case aggregation method mismatch: {case_id}"
    aggregate = _aggregate_durations([float(value) for value in samples])
    if not _is_positive_finite(record.get("aggregate_us")) or not math.isclose(
        float(record["aggregate_us"]), aggregate, rel_tol=1e-9, abs_tol=1e-6
    ):
        return f"per-case aggregate is not reproducible: {case_id}"
    if not _is_positive_finite(output_case.get("asc_us")) or not math.isclose(
        float(output_case["asc_us"]), aggregate, rel_tol=1e-9, abs_tol=1e-6
    ):
        return f"performance.json PyPTO duration mismatch: {case_id}"
    return None


def _output_ratio_fields_error(
    output_case: Dict[str, Any], ratio: float, case_id: str,
) -> Optional[str]:
    """对照重算比值校验 performance.json 的比值字段；返回 error。"""
    for field in ("golden_reference_ratio", "default_target_ratio"):
        if not _is_positive_finite(output_case.get(field)) or not math.isclose(
            float(output_case[field]), ratio, rel_tol=1e-9, abs_tol=1e-6
        ):
            return f"performance.json {field} mismatch: {case_id}"
    return None


def _check_golden_contract(
    data: Dict[str, Any], data_cases: Dict[str, Any], expected_cases: list,
) -> Tuple[list, Optional[str]]:
    """从冻结合同重算 Golden 目标比值，与 performance.json 逐项对照。"""
    source = data.get("performance_cases") or {}
    golden_contract = source.get("golden_contract")
    canonical_ratios = []
    if isinstance(golden_contract, dict):
        try:
            golden_cases, golden_metadata, golden_error = parse_golden_contract(
                Path(golden_contract["path"]).parent
            )
        except (KeyError, OSError, ValueError) as error:
            return [], f"cannot reload Golden target contract: {error}"
        if golden_error:
            return [], golden_error
        if golden_metadata.get("device_id") != data.get("device_id"):
            return [], "Golden/PyPTO physical device mismatch"
        golden_map = {str(case[0]): float(case[3]) for case in golden_cases}
        if set(golden_map) != {str(case_id) for case_id in expected_cases}:
            return [], "Golden target case ids do not exactly match collection manifest"
        for case_id in expected_cases:
            output_case = data_cases[str(case_id)]
            ratio = golden_map[str(case_id)] / float(output_case["asc_us"])
            if not _is_positive_finite(ratio):
                return [], f"Golden target ratio is invalid: {case_id}"
            canonical_ratios.append(ratio)
            field_error = _output_ratio_fields_error(output_case, ratio, case_id)
            if field_error:
                return [], field_error
            if output_case.get("default_target_met") is not (
                ratio >= DEFAULT_GOLDEN_TARGET_THRESHOLD
            ):
                return [], f"performance.json case target decision mismatch: {case_id}"
    return canonical_ratios, None


def _check_target_decisions(
    data: Dict[str, Any], canonical_ratios: list, expected_cases: list,
) -> Optional[str]:
    """对照重算的比值校验 performance.json 中的目标判定与统计。"""
    expected_valid = bool(canonical_ratios) and len(canonical_ratios) == len(expected_cases)
    expected_met = expected_valid and all(
        ratio >= DEFAULT_GOLDEN_TARGET_THRESHOLD for ratio in canonical_ratios
    )
    if bool(data.get("valid_for_target_met")) != expected_valid:
        return "performance.json target-evidence validity mismatch"
    if bool(data.get("default_target_met")) != expected_met:
        return "performance.json target decision mismatch"
    recomputed_stats = _compute_speedup_stats(canonical_ratios)
    output_stats = data.get("golden_reference_ratio_stats") or {}
    for output_key, source_key in (
        ("geomean", "geomean_speedup"), ("mean", "mean_speedup"),
        ("median", "median_speedup"), ("min", "min_speedup"),
        ("max", "max_speedup"),
    ):
        expected = recomputed_stats[source_key]
        actual = output_stats.get(output_key)
        if expected is None:
            if actual is not None:
                return f"performance.json Golden ratio stats mismatch: {output_key}"
        elif not _is_positive_finite(actual) or not math.isclose(
            float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-6
        ):
            return f"performance.json Golden ratio stats mismatch: {output_key}"
    return None


def batch_evidence_error(op_dir: Path, data: Dict[str, Any]) -> Optional[str]:
    """Check the lightweight compare evidence needed by batch summaries."""
    checked = _check_collection_manifest(op_dir, data)
    if checked.error:
        return checked.error
    for case_id in checked.expected_cases:
        per_case_error = _check_per_case(
            checked.round_path, case_id, checked.data_cases, data, checked.repeats
        )
        if per_case_error:
            return per_case_error
    # Recompute the Golden target fields from the frozen contract and the
    # verified per-case measurement aggregates.  Batch must never trust
    # mutable derived booleans or ratios from performance.json.
    canonical_ratios, golden_error = _check_golden_contract(
        data, checked.data_cases, checked.expected_cases
    )
    if golden_error:
        return golden_error
    return _check_target_decisions(data, canonical_ratios, checked.expected_cases)


def _batch_result_status(data):
    n_total = data.get("n_cases_total", 0)
    n_valid = data.get("n_cases_valid", 0)
    if not n_total or n_valid != n_total:
        return "incomplete"
    return "complete"


def _batch_target_status(data):
    if not data.get("valid_for_target_met"):
        return "unavailable"
    return "met" if data.get("default_target_met") else "not_met"


def _build_batch_md_summary_table(op_results):
    """Build the batch summary table markdown lines."""
    md_lines = []
    if op_results:
        md_lines.append("## Golden 默认目标汇总")
        md_lines.append("")
        md_lines.append(
            "| 算子名称 | 用例数 | 有效用例 | 几何平均目标比值 | 平均目标比值 | 采集状态 | 目标状态 |"
        )
        md_lines.append("| -------- | ------ | -------- | -------------- | ---------- | -------- | -------- |")
        for op in op_results:
            data = op["data"]
            name = op["name"]
            n_total = data.get("n_cases_total", 0)
            n_valid = data.get("n_cases_valid", 0)
            ratio_stats = data.get("golden_reference_ratio_stats") or {}
            geo = ratio_stats.get("geomean")
            mean = ratio_stats.get("mean")
            geo_str = f"{geo:.3f}" if geo is not None else "N/A"
            mean_str = f"{mean:.3f}" if mean is not None else "N/A"
            status = _batch_result_status(data)
            target_status = _batch_target_status(data)
            md_lines.append(
                f"| {name} | {n_total} | {n_valid} | {geo_str} | {mean_str} | "
                f"{status} | {target_status} |"
            )
        md_lines.append("")
    return md_lines


def _build_batch_md_per_op_details(op_results):
    """Build per-operator detail tables in markdown."""
    md_lines = []
    for op in op_results:
        data = op["data"]
        name = op["name"]
        md_lines.append(f"## {name}")
        md_lines.append("")
        if data.get("per_case"):
            md_lines.append("| Case | Shape | DType | PyPTO目标kernel(us) | Golden每迭代E2E(us) | 默认目标比值 |")
            md_lines.append("| ---- | ----- | ----- | ------------- | -------- | -------------- |")
            for case in data["per_case"]:
                shape = case.get("shape", "?")
                dtype = case.get("dtype", "?")
                ref = case.get("ref_us")
                asc = case.get("asc_us")
                ratio = case.get("golden_reference_ratio")
                ref_str = f"{ref:.2f}" if ref is not None else "N/A"
                asc_str = f"{asc:.2f}" if asc is not None else "N/A"
                ratio_str = f"{ratio:.3f}" if ratio is not None else "N/A"
                md_lines.append(
                    f"| {case['case']} | {shape} | {dtype} | {asc_str} | "
                    f"{ref_str} | {ratio_str} |"
                )
            md_lines.append("")
    return md_lines


def _build_batch_md_trace_section(trace_rows):
    """Build trace table section in markdown if rows exist."""
    if not trace_rows:
        return []
    md_lines = [
        "## Trace 汇总表",
        "",
        ("| Level | Problem ID | 算子名称 | 算子类型 | 编译通过 | 精度正确 | "
         "PyTorch 参考延迟 | PyPTO-Pro Kernel 延迟 | 异口径历史比值 | 历史状态 | "
         "精度正确 | 历史0.6x字段 | 历史0.8x字段 |"),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    md_lines.extend(trace_rows)
    md_lines.append("")
    return md_lines


def _generate_batch_md_report(args, op_results, trace_rows, base_dir):
    """Generate and save the batch Markdown report."""
    md_lines = []
    md_lines.append("# 📊 算子批量 Golden 默认目标报告")
    md_lines.append("")
    md_lines.append(f"- **扫描目录**: {base_dir}")
    md_lines.append(f"- **Collection ID**: {args.collection_id}")
    md_lines.append(f"- **算子总数**: {len(op_results)}")
    md_lines.append(f"- **生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md_lines.append("")

    md_lines.extend(_build_batch_md_summary_table(op_results))
    md_lines.extend(_build_batch_md_per_op_details(op_results))
    md_lines.extend(_build_batch_md_trace_section(trace_rows))

    md_path = Path(args.output_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(md_path, "\n".join(md_lines))
    LOGGER.info("Batch markdown report saved to: %s", md_path)


def _operator_summary(op: Dict[str, Any]) -> Dict[str, Any]:
    """构造单个算子在批量 JSON 汇总中的条目。"""
    return {
        "name": op["name"],
        "n_cases_total": op["data"].get("n_cases_total", 0),
        "n_cases_valid": op["data"].get("n_cases_valid", 0),
        "n_cross_scope_cases": op["data"].get("n_cross_scope_cases", 0),
        "n_default_target_cases": op["data"].get("n_default_target_cases", 0),
        "golden_reference_ratio_stats": op["data"].get(
            "golden_reference_ratio_stats"
        ),
        "aggregate_golden_reference_ratio": op["data"].get(
            "aggregate_golden_reference_ratio"
        ),
        # Compatibility aliases retained in the machine-readable output.
        "geomean_speedup": op["data"].get("geomean_speedup"),
        "mean_speedup": op["data"].get("mean_speedup"),
        "mean_ref_us": op["data"].get("mean_ref_us"),
        "mean_asc_us": op["data"].get("mean_asc_us"),
        "comparison_scope": op["data"].get("comparison_scope"),
        "performance_cases": op["data"].get("performance_cases"),
        "default_target_metric": op["data"].get("default_target_metric"),
        "default_target_threshold": op["data"].get("default_target_threshold"),
        "valid_for_target_met": bool(op["data"].get("valid_for_target_met")),
        "default_target_met": bool(op["data"].get("default_target_met")),
        "default_target_status": _batch_target_status(op["data"]),
        "valid_for_optimization_speedup": False,
    }


def generate_batch_json_report(args, op_results, base_dir):
    """Generate and save the batch JSON summary."""
    all_targets_valid = bool(op_results) and all(
        bool(op["data"].get("valid_for_target_met")) for op in op_results
    )
    all_targets_met = all_targets_valid and all(
        bool(op["data"].get("default_target_met")) for op in op_results
    )
    batch_summary = {
        "base_dir": str(base_dir),
        "collection_id": args.collection_id,
        "status": "complete",
        "measurement_scope": "pypto_target_kernel",
        "has_cross_scope_diagnostics": any(
            bool(op["data"].get("n_cross_scope_cases")) for op in op_results
        ),
        "default_target_metric": "golden_per_iteration_e2e_us / pypto_target_kernel_us",
        "default_target_threshold": DEFAULT_GOLDEN_TARGET_THRESHOLD,
        "valid_for_target_met": all_targets_valid,
        "default_target_met": bool(all_targets_met),
        "default_target_status": (
            "met" if all_targets_met else (
                "not_met" if all_targets_valid else "unavailable"
            )
        ),
        "valid_for_optimization_speedup": False,
        "n_operators": len(op_results),
        "operators": [_operator_summary(op) for op in op_results],
        "generated_at": time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        json_path, json.dumps(batch_summary, indent=2, ensure_ascii=False) + "\n"
    )
    LOGGER.info("Batch JSON summary saved to: %s", json_path)
