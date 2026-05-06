#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""单 case 与汇总报告写入工具.

输入:
    - ``CaseRunRecord``: 单个用例的两阶段结果 (pypto 生成 + verifier 验证).

输出:
    - ``<report-dir>/<op>/result.json``: 单用例结构化结果.
    - ``<report-dir>/summary.json``: 全量 JSON.
    - ``<report-dir>/summary.md``: 人读 Markdown.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any


logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# 单 case 记录
# ────────────────────────────────────────────────────────────

@dataclass
class CaseRunRecord:
    """单 case 端到端结果."""
    op_name: str
    case_id: str
    source_file: str
    level: str = ""
    report_subdir: str = ""

    # pypto 生成阶段
    pypto_status: str = ""           # PyptoRunStatus.value
    pypto_message: str = ""
    pypto_duration_sec: float = 0.0
    pypto_log_file: Optional[str] = None
    pypto_attempt_log_files: List[str] = field(default_factory=list)
    pypto_retry_count: int = 0
    pypto_session_id: Optional[str] = None
    pypto_session_md_file: Optional[str] = None
    pypto_session_export_message: str = ""
    pypto_artifacts: Dict[str, str] = field(default_factory=dict)

    # verifier 验证阶段
    verifier_status: str = ""        # VerifierStatus.value
    verifier_message: str = ""
    verifier_duration_sec: float = 0.0
    verifier_log_file: Optional[str] = None
    verifier_session_id: Optional[str] = None
    verifier_session_md_file: Optional[str] = None
    verifier_session_export_message: str = ""
    correctness: Optional[bool] = None

    # 性能字段 (仅 mode=performance/full 有数值; mode=correctness 全为 None).
    perf_gen_time_us: Optional[float] = None       # agent 生成实现执行时间 (us)
    perf_base_time_us: Optional[float] = None      # KernelBench Model 基线时间 (us)
    perf_speedup: Optional[float] = None           # base_time / gen_time
    perf_roofline_time_us: Optional[float] = None
    perf_roofline_speedup: Optional[float] = None
    perf_message: str = ""                         # perf 阶段独立消息 (如 "skipped: correctness failed")

    # 合并视角
    overall_status: str = ""         # success / pypto_failed / verify_failed / baseline_failed / verify_error
    failure_category: str = ""       # verifier / skill_report; PASS 或未分类为空
    started_at: str = ""
    finished_at: str = ""

    @property
    def succeeded(self) -> bool:
        return self.overall_status == "success"

    def total_duration_sec(self) -> float:
        """两阶段耗时累加 — pypto + verifier."""
        return self.pypto_duration_sec + self.verifier_duration_sec

    def wall_duration_sec(self) -> Optional[float]:
        """完整流程墙钟耗时 = ``finished_at - started_at``.

        含 case_loader / 调度 / 报告写盘等 ``pypto + verifier`` 之外的开销;
        多 case 并发场景下也是单 case 真实端到端时长 (而非跟其他 case 重叠的累加).
        ``started_at`` / ``finished_at`` 任一缺失返回 ``None``.
        """
        if not self.started_at or not self.finished_at:
            return None
        try:
            t0 = dt.datetime.fromisoformat(self.started_at)
            t1 = dt.datetime.fromisoformat(self.finished_at)
        except ValueError:
            return None
        return max(0.0, (t1 - t0).total_seconds())


def derive_overall_status(pypto_ok: bool, verifier_status: Optional[str],
                          correctness: Optional[bool]) -> str:
    """三个状态合一."""
    if not pypto_ok:
        return "pypto_failed"
    if verifier_status == "passed" and correctness is True:
        return "success"
    if verifier_status == "baseline_failed":
        return "baseline_failed"
    if verifier_status == "failed":
        return "verify_failed"
    return "verify_error"


# ────────────────────────────────────────────────────────────
# 落盘
# ────────────────────────────────────────────────────────────

def write_case_result(record: CaseRunRecord, report_dir: Path) -> Path:
    """把单 case 结果写到 ``<report-dir>/<op>/result.json``."""
    op_dir = report_dir / (record.report_subdir or record.op_name)
    op_dir.mkdir(parents=True, exist_ok=True)
    out = op_dir / "result.json"
    out.write_text(json.dumps(asdict(record), indent=2, ensure_ascii=False),
                   encoding="utf-8")
    return out


def write_summary(records: Iterable[CaseRunRecord], report_dir: Path,
                  meta: Optional[Dict[str, Any]] = None) -> Dict[str, Path]:
    """写 summary.json + summary.md, 返回两个路径."""
    records = list(records)
    records.sort(key=_case_sort_key)
    report_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "meta": _build_meta(meta),
        "totals": _compute_totals(records),
        "cases": [asdict(r) for r in records],
    }

    json_path = report_dir / "summary.json"
    json_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False),
                         encoding="utf-8")

    md_path = report_dir / "summary.md"
    md_path.write_text(_render_markdown(summary_payload), encoding="utf-8")

    return {"json": json_path, "md": md_path}


# ────────────────────────────────────────────────────────────
# 内部: 统计 / 渲染
# ────────────────────────────────────────────────────────────

def _build_meta(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    if meta:
        base.update(meta)
    return base


def _compute_totals(records: List[CaseRunRecord], *, include_by_level: bool = True) -> Dict[str, Any]:
    total = len(records)
    if total == 0:
        return {"total": 0}

    overall_buckets: Dict[str, int] = {}
    correctness_pass = 0
    correctness_fail = 0
    correctness_unknown = 0
    by_failure_category: Dict[str, int] = {}
    for r in records:
        overall_buckets[r.overall_status] = overall_buckets.get(r.overall_status, 0) + 1
        fc_key = (r.failure_category or "").strip()
        by_failure_category[fc_key] = by_failure_category.get(fc_key, 0) + 1
        if r.correctness is True:
            correctness_pass += 1
        elif r.correctness is False:
            correctness_fail += 1
        else:
            correctness_unknown += 1

    pypto_durations = [r.pypto_duration_sec for r in records if r.pypto_duration_sec > 0]
    verify_durations = [r.verifier_duration_sec for r in records if r.verifier_duration_sec > 0]
    stage_sum_durations = [r.total_duration_sec() for r in records]
    wall_durations = [d for r in records if (d := r.wall_duration_sec()) is not None]

    speedups = [r.perf_speedup for r in records
                if isinstance(r.perf_speedup, (int, float)) and r.perf_speedup is not None]

    success = overall_buckets.get("success", 0)
    out: Dict[str, Any] = {
        "total": total,
        "success": success,
        "success_rate": round(success / total, 4),
        "by_status": overall_buckets,
        "correctness": {
            "pass": correctness_pass,
            "fail": correctness_fail,
            "unknown": correctness_unknown,
        },
        "by_failure_category": by_failure_category,
        "duration_sec": {
            "pypto_total": round(sum(pypto_durations), 2),
            "pypto_mean": round(statistics.mean(pypto_durations), 2) if pypto_durations else 0.0,
            "verify_total": round(sum(verify_durations), 2),
            "verify_mean": round(statistics.mean(verify_durations), 2) if verify_durations else 0.0,
            "stage_sum_total": round(sum(stage_sum_durations), 2),
            "wall_total": round(sum(wall_durations), 2) if wall_durations else 0.0,
            "wall_mean": round(statistics.mean(wall_durations), 2) if wall_durations else 0.0,
            "wall_min": round(min(wall_durations), 2) if wall_durations else 0.0,
            "wall_max": round(max(wall_durations), 2) if wall_durations else 0.0,
        },
    }
    if speedups:
        out["perf"] = {
            "cases_with_perf": len(speedups),
            "speedup_min": round(min(speedups), 4),
            "speedup_max": round(max(speedups), 4),
            "speedup_mean": round(statistics.mean(speedups), 4),
            "speedup_geomean": round(_geo_mean(speedups), 4),
        }
    if include_by_level:
        grouped: Dict[str, List[CaseRunRecord]] = {}
        for r in records:
            level = r.level or "unknown"
            grouped.setdefault(level, []).append(r)
        out["by_level"] = {
            level: _compute_totals(group_records, include_by_level=False)
            for level, group_records in sorted(grouped.items())
        }
    return out


def _geo_mean(values: List[float]) -> float:
    """几何平均, 适合 speedup 这类 ratio 聚合 (避免少数极端值拉偏算数平均)."""
    pos = [v for v in values if v > 0]
    if not pos:
        return 0.0
    import math
    return math.exp(sum(math.log(v) for v in pos) / len(pos))


_STATUS_BADGE = {
    "success": "PASS",
    "pypto_failed": "FAIL (pypto)",
    "verify_failed": "FAIL (verify)",
    "baseline_failed": "FAIL (baseline)",
    "verify_error": "ERROR",
}


# skill_report.failure_category / verifier 推导值 → summary.md 「错误类别」列中文标签
_FAILURE_CATEGORY_LABEL: Dict[str, str] = {
    # pypto-kernel-validate 表 (大写)
    "CHEAT_MULTI_KERNEL": "多内核作弊",
    "CHEAT_SEMANTIC": "语义作弊",
    "BASELINE_FAILED": "基线失败",
    "CORRECTNESS_RUNTIME": "精度：运行/AICore",
    "CORRECTNESS_SHAPE_OR_IO": "精度：形状/IO",
    "CORRECTNESS_NUMERICAL": "精度：数值",
    "PERFORMANCE_FAILED": "性能失败",
    "ERROR_INPUT_OR_ARTIFACT": "输入/产物错误",
    "ERROR_VERIFIER_OR_CLI": "验证器/CLI 错误",
    "ERROR_OTHER": "其他错误",
    # final_verdict 回退 / direct 模式等 (小写 snake)
    "cheat": "作弊",
    "semantic_cheat": "语义作弊",
    "correctness": "精度失败",
    "performance": "性能失败",
    "baseline_failed": "基线失败",
    "error": "错误",
    "unknown_verdict": "未知裁决",
    "missing_input": "缺少输入",
    "config_error": "配置错误",
    "worker_error": "Worker 错误",
    "runtime_error": "运行时错误",
    "opencode_unavailable": "OpenCode 不可用",
    "invalid_skill_report": "无效验证报告",
}


def _failure_category_display(raw: Any) -> str:
    """用于 Markdown: PASS / 空 / 空白 → —, 其余尽量转为中文标签."""
    s = str(raw or "").strip()
    if not s or s.upper() == "PASS":
        return "—"
    if s in _FAILURE_CATEGORY_LABEL:
        return _FAILURE_CATEGORY_LABEL[s]
    sup = s.upper()
    if sup in _FAILURE_CATEGORY_LABEL:
        return _FAILURE_CATEGORY_LABEL[sup]
    low = s.lower()
    if low in _FAILURE_CATEGORY_LABEL:
        return _FAILURE_CATEGORY_LABEL[low]
    return s


def _fmt_perf_num(value: Any, suffix: str = "", spec: str = ".2f") -> str:
    if isinstance(value, (int, float)) and value is not None:
        return f"{value:{spec}}{suffix}"
    return "—"


def _render_markdown(payload: Dict[str, Any]) -> str:
    meta = payload["meta"]
    totals = payload["totals"]
    cases = payload["cases"]

    lines: List[str] = []
    lines.append("# pypto KernelBench 批处理报告")
    lines.append("")
    lines.append("## 元数据")
    for k, v in meta.items():
        lines.append(f"- **{k}**: `{v}`")
    lines.append("")

    lines.append("## 总览")
    if totals.get("total", 0) == 0:
        lines.append("> 无 case.")
    else:
        lines.append(f"- **总用例数**: {totals['total']}")
        lines.append(f"- **总通过 (overall=success)**: {totals['success']} "
                     f"({totals['success_rate']*100:.1f}%)")
        cor = totals.get("correctness", {})
        lines.append(
            f"- **精度**: pass={cor.get('pass', 0)}, "
            f"fail={cor.get('fail', 0)}, unknown={cor.get('unknown', 0)}"
        )
        bucket_str = ", ".join(f"{k}={v}" for k, v in totals.get("by_status", {}).items())
        lines.append(f"- **总状态分布**: {bucket_str}")

        fc_totals = totals.get("by_failure_category") or {}
        fc_fail = {k: v for k, v in fc_totals.items() if k}
        if fc_fail:
            fc_bits = [
                f"{_failure_category_display(k)}={v}"
                for k, v in sorted(fc_fail.items(), key=lambda kv: (-kv[1], kv[0]))
            ]
            lines.append(f"- **失败类别分布**: {', '.join(fc_bits)}")
        else:
            lines.append("- **失败类别分布**: 无")

        perf_agg = totals.get("perf")
        if perf_agg:
            lines.append(
                f"- **性能聚合 (speedup, n={perf_agg['cases_with_perf']})**: "
                f"min={perf_agg['speedup_min']:.3f}x, "
                f"max={perf_agg['speedup_max']:.3f}x, "
                f"mean={perf_agg['speedup_mean']:.3f}x, "
                f"geomean={perf_agg['speedup_geomean']:.3f}x"
            )

        dur = totals.get("duration_sec", {})
        lines.append(
            "- **阶段累加耗时 (秒)**: "
            f"pypto_total={dur.get('pypto_total', 0)}, "
            f"pypto_mean={dur.get('pypto_mean', 0)}, "
            f"verify_total={dur.get('verify_total', 0)}, "
            f"verify_mean={dur.get('verify_mean', 0)}, "
            f"stage_sum_total={dur.get('stage_sum_total', 0)}"
        )
        lines.append(
            "- **完整流程墙钟 (秒)**: "
            f"wall_total={dur.get('wall_total', 0)}, "
            f"wall_mean={dur.get('wall_mean', 0)}, "
            f"wall_min={dur.get('wall_min', 0)}, "
            f"wall_max={dur.get('wall_max', 0)}"
        )
    lines.append("")

    by_level = totals.get("by_level") or {}
    if by_level:
        lines.append("## 按 Level 汇总")
        lines.append("")
        lines.append("| level | total | success | success_rate | correctness | status |")
        lines.append("|-------|------:|--------:|-------------:|-------------|--------|")
        for level, level_totals in by_level.items():
            cor = level_totals.get("correctness", {})
            cor_cell = (
                f"pass={cor.get('pass', 0)}, "
                f"fail={cor.get('fail', 0)}, unknown={cor.get('unknown', 0)}"
            )
            status_cell = ", ".join(
                f"{k}={v}" for k, v in level_totals.get("by_status", {}).items()
            )
            lines.append(
                f"| `{level}` | {level_totals.get('total', 0)} "
                f"| {level_totals.get('success', 0)} "
                f"| {level_totals.get('success_rate', 0) * 100:.1f}% "
                f"| {cor_cell} | {status_cell} |"
            )
        lines.append("")

    if cases:
        lines.append("## 用例明细")
        lines.append("")
        lines.append(
            "| level | op | case | 总状态 | 错误类别 | 精度 | pypto阶段 | verify阶段 | "
            "pypto retry | pypto(s) | verify(s) | 完整流程(s) | 备注 |"
        )
        lines.append(
            "|-------|----|------|--------|----------|:----:|-----------|------------|"
            "------------:|---------:|----------:|------------:|------|"
        )
        for c in cases:
            badge = _STATUS_BADGE.get(c.get("overall_status", ""), c.get("overall_status", ""))
            fc_cell = _failure_category_display(c.get("failure_category"))
            cor = c.get("correctness")
            cor_cell = "✓" if cor is True else ("✗" if cor is False else "—")
            note_src = c.get("verifier_message") or c.get("pypto_message") or ""
            note = _table_text(note_src)
            pypto_sec = round(c.get("pypto_duration_sec") or 0, 1)
            verify_sec = round(c.get("verifier_duration_sec") or 0, 1)
            wall_sec = _wall_from_record_dict(c)
            wall_cell = f"{wall_sec:.1f}" if wall_sec is not None else "—"
            op_cell = f"`{c['op_name']}`"
            case_cell = f"`{c['case_id']}`"
            level_cell = f"`{c.get('level') or '—'}`"
            pypto_st = c.get("pypto_status", "")
            ver_st = c.get("verifier_status", "")
            pypto_retry = int(c.get("pypto_retry_count") or 0)
            lines.append(
                f"| {level_cell} | {op_cell} | {case_cell} | {badge} | {fc_cell} | {cor_cell} "
                f"| {pypto_st} | {ver_st} "
                f"| {pypto_retry} | {pypto_sec} | {verify_sec} | {wall_cell} | {note} |"
            )
        lines.append("")

        has_session_exports = any(
            c.get("pypto_session_md_file")
            or c.get("verifier_session_md_file")
            or c.get("pypto_attempt_log_files")
            or c.get("verifier_attempt_log_files")
            or c.get("pypto_session_export_message")
            or c.get("verifier_session_export_message")
            for c in cases
        )
        if has_session_exports:
            lines.append("## OpenCode 会话导出")
            lines.append("")
            lines.append("| level | op | pypto session | verifier session |")
            lines.append("|-------|----|---------------|------------------|")
            for c in cases:
                lines.append(
                    f"| `{c.get('level') or '—'}` "
                    f"| `{c['op_name']}` "
                    f"| {_export_cell(c, 'pypto')} "
                    f"| {_export_cell(c, 'verifier')} |"
                )
            lines.append("")

        # 性能明细 — 仅当至少一个 case 有 gen_time 数值时输出整张表.
        has_perf = any(c.get("perf_gen_time_us") is not None for c in cases)
        if has_perf:
            lines.append("## 性能明细 (gen vs base, time 单位 us)")
            lines.append("")
            lines.append("| level | op | gen_time | base_time | speedup | roofline_time | roofline_speedup | 完整流程(s) | 备注 |")
            lines.append("|-------|----|---------:|----------:|--------:|--------------:|-----------------:|------------:|------|")
            for c in cases:
                gen = c.get("perf_gen_time_us")
                base = c.get("perf_base_time_us")
                sp = c.get("perf_speedup")
                rl_t = c.get("perf_roofline_time_us")
                rl_sp = c.get("perf_roofline_speedup")
                pm = _table_text(c.get("perf_message"))
                wall_sec = _wall_from_record_dict(c)
                wall_cell = f"{wall_sec:.1f}" if wall_sec is not None else "—"
                op_perf = f"`{c['op_name']}`"
                lines.append(
                    f"| `{c.get('level') or '—'}` "
                    f"| {op_perf} "
                    f"| {_fmt_perf_num(gen)} "
                    f"| {_fmt_perf_num(base)} "
                    f"| {_fmt_perf_num(sp, suffix='x', spec='.3f')} "
                    f"| {_fmt_perf_num(rl_t)} "
                    f"| {_fmt_perf_num(rl_sp, suffix='x', spec='.3f')} "
                    f"| {wall_cell} | {pm} |"
                )
            lines.append("")

    return "\n".join(lines) + "\n"


def _md_file_link(value: Any) -> str:
    if not value:
        return "—"
    path = Path(str(value))
    label = path.name or str(value)
    target = str(value)
    if any(ch.isspace() for ch in target):
        return f"[{label}](<{target}>)"
    return f"[{label}]({target})"


def _export_cell(case: Dict[str, Any], label: str) -> str:
    parts: List[str] = []
    md_file = case.get(f"{label}_session_md_file")
    if md_file:
        parts.append(_md_file_link(md_file))

    attempt_logs = case.get(f"{label}_attempt_log_files")
    if isinstance(attempt_logs, list) and attempt_logs:
        links = ", ".join(_md_file_link(p) for p in attempt_logs)
        parts.append(f"attempt logs: {links}")

    log_file = case.get(f"{label}_log_file")
    if log_file:
        status_file = Path(str(log_file)).parent / f"{label}_session_export.json"
        parts.append(_md_file_link(status_file))

    message = case.get(f"{label}_session_export_message")
    if message:
        parts.append(_table_text(message))

    return "<br>".join(parts) if parts else "—"


def _table_text(value: Any) -> str:
    text = str(value or "").replace("\n", " ").replace("|", "\\|").strip()
    if not text:
        return "—"
    return text


def _wall_from_record_dict(case: Dict[str, Any]) -> Optional[float]:
    """从已 dict 化的 record 算 wall_duration_sec, 复用 CaseRunRecord.wall_duration_sec 逻辑."""
    started = case.get("started_at") or ""
    finished = case.get("finished_at") or ""
    if not started or not finished:
        return None
    try:
        t0 = dt.datetime.fromisoformat(started)
        t1 = dt.datetime.fromisoformat(finished)
    except ValueError:
        return None
    return max(0.0, (t1 - t0).total_seconds())


def _case_sort_key(record: CaseRunRecord) -> tuple:
    """按 KernelBench case_id 前缀数字排序, 无数字时退化到字符串排序."""
    head = (record.case_id or "").split("_", 1)[0]
    try:
        return (record.level or "", int(head), record.case_id)
    except ValueError:
        return (record.level or "", sys.maxsize, record.case_id)


# ────────────────────────────────────────────────────────────
# CLI (调试用)
# ────────────────────────────────────────────────────────────

def _main_cli() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Aggregate per-case result.json files into summary")
    parser.add_argument("report_dir", type=Path)
    args = parser.parse_args()

    records: List[CaseRunRecord] = []
    for result_file in sorted(args.report_dir.rglob("result.json")):
        if not result_file.is_file():
            continue
        try:
            data = json.loads(result_file.read_text(encoding="utf-8"))
            records.append(CaseRunRecord(**data))
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("skip %s: %s", result_file, e)

    records.sort(key=_case_sort_key)

    meta = None
    summary_json = args.report_dir / "summary.json"
    if summary_json.is_file():
        try:
            existing = json.loads(summary_json.read_text(encoding="utf-8"))
            existing_meta = existing.get("meta")
            if isinstance(existing_meta, dict):
                meta = existing_meta
        except json.JSONDecodeError:
            logger.warning("ignore invalid existing summary metadata: %s", summary_json)

    paths = write_summary(records, args.report_dir, meta=meta)
    for k, p in paths.items():
        logger.info("%s: %s", k, p)
    return 0


if __name__ == "__main__":
    sys.exit(_main_cli())
