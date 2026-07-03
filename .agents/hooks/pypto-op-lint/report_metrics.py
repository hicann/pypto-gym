#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
"""
Lint 指标报告生成器

从 lint_events.jsonl 读取 lint 检查事件，生成统计摘要并输出为 JSON 和 Markdown。

使用方法：
    # 默认读取 logs/lint_events.jsonl，输出到 logs/summary_latest.json 和 logs/summary_latest.md
    python3 report_metrics.py

    # 指定输入文件
    python3 report_metrics.py --input /path/to/lint_events.jsonl

    # 指定输出路径
    python3 report_metrics.py --output-json /path/to/summary.json --output-md /path/to/summary.md

    # 按算子目录过滤
    python3 report_metrics.py --op-dir /path/to/custom/my_op

    # 只统计某个时间点之后的事件（Unix 时间戳）
    python3 report_metrics.py --since 1700000000

输出内容：
    - 概览：规则检查总数、通过/失败/警告/跳过数、门禁阻断次数、总耗时
    - S0/S1 失败规则列表
    - 修复成本分布：按 E1/E2/E3 分组统计失败和警告规则
    - 每次调用明细：invocation_id、阶段、模式、通过/失败/警告、S0 失败、耗时
    - 按算子统计：每个算子目录的调用次数和通过/失败/警告数

作为模块导入：
    from report_metrics import load_events, build_summary, write_outputs
    events = load_events(path=Path("lint_events.jsonl"))
    summary = build_summary(events)
    write_outputs(summary, out_json=Path("out.json"), out_md=Path("out.md"))
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

BASE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVENTS = BASE / "logs" / "lint_events.jsonl"
OUT_JSON = BASE / "logs" / "summary_latest.json"
OUT_MD = BASE / "logs" / "summary_latest.md"

_log = logging.getLogger(__name__)


def load_events(path: Path | None = None,
                since_ts: int | None = None,
                op_dir_filter: str | None = None) -> list[dict[str, Any]]:
    target = path or EVENTS
    events: list[dict[str, Any]] = []
    if not target.exists():
        return events
    malformed = 0
    with open(target, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if since_ts is not None and event.get("ts", 0) < since_ts:
                continue
            if op_dir_filter and event.get("op_dir", "") != op_dir_filter:
                continue
            events.append(event)
    if malformed:
        _log.warning("skipped %d malformed JSONL line(s)", malformed)
    return events


def _build_from_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(e.get("total", 0) for e in summaries)
    pass_c = sum(e.get("pass", 0) for e in summaries)
    fail_c = sum(e.get("fail", 0) for e in summaries)
    warn_c = sum(e.get("warn", 0) for e in summaries)
    skip_c = sum(e.get("skip", 0) for e in summaries)
    duration = sum(float(e.get("total_duration_ms", 0)) for e in summaries)

    all_failed: set[str] = set()
    s0_fails: set[str] = set()
    s1_fails: set[str] = set()
    effort_agg: dict[str, dict[str, set[str]]] = defaultdict(lambda: {"fails": set(), "warns": set()})
    for s in summaries:
        all_failed.update(s.get("s0_fails", []))
        all_failed.update(s.get("s1_fails", []))
        s0_fails.update(s.get("s0_fails", []))
        s1_fails.update(s.get("s1_fails", []))
        for eff, data in s.get("effort_stats", {}).items():
            effort_agg[eff]["fails"].update(data.get("fails", []))
            effort_agg[eff]["warns"].update(data.get("warns", []))

    effort_stats = {
        eff: {"fails": sorted(d["fails"]), "warns": sorted(d["warns"])}
        for eff, d in sorted(effort_agg.items())
    }
    return {
        "total_rule_checks": total,
        "pass": pass_c,
        "fail": fail_c,
        "warn": warn_c,
        "skip": skip_c,
        "failed_rules": sorted(all_failed),
        "s0_fails": sorted(s0_fails),
        "s1_fails": sorted(s1_fails),
        "effort_stats": effort_stats,
        "total_duration_ms": round(duration, 3),
    }


def _build_from_rule_checks(rule_checks: list[dict[str, Any]]) -> dict[str, Any]:
    pass_c = sum(1 for e in rule_checks if e.get("status") == "PASS")
    fail_c = sum(1 for e in rule_checks if e.get("status") == "FAIL")
    warn_c = sum(1 for e in rule_checks if e.get("status") == "WARN")
    skip_c = sum(1 for e in rule_checks if e.get("status") == "SKIP")
    duration = sum(float(e.get("duration_ms", 0)) for e in rule_checks)
    failed_rules = sorted({e.get("rule_id") for e in rule_checks if e.get("status") == "FAIL"})

    effort_agg: dict[str, dict[str, set[str]]] = defaultdict(lambda: {"fails": set(), "warns": set()})
    for e in rule_checks:
        eff = e.get("fix_effort") or "unknown"
        if e.get("status") == "FAIL":
            effort_agg[eff]["fails"].add(e.get("rule_id", ""))
        elif e.get("status") == "WARN":
            effort_agg[eff]["warns"].add(e.get("rule_id", ""))
    effort_stats = {
        eff: {"fails": sorted(d["fails"]), "warns": sorted(d["warns"])}
        for eff, d in sorted(effort_agg.items())
    }
    return {
        "total_rule_checks": len(rule_checks),
        "pass": pass_c,
        "fail": fail_c,
        "warn": warn_c,
        "skip": skip_c,
        "failed_rules": failed_rules,
        "s0_fails": [],
        "s1_fails": [],
        "effort_stats": effort_stats,
        "total_duration_ms": round(duration, 3),
    }


def _build_invocation_breakdown(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for s in summaries:
        rows.append({
            "invocation_id": s.get("invocation_id", ""),
            "op_dir": s.get("op_dir", ""),
            "stage": s.get("stage", ""),
            "mode": s.get("mode", ""),
            "total": s.get("total", 0),
            "pass": s.get("pass", 0),
            "fail": s.get("fail", 0),
            "warn": s.get("warn", 0),
            "s0_fails": s.get("s0_fails", []),
            "s1_fails": s.get("s1_fails", []),
            "duration_ms": s.get("total_duration_ms", 0),
        })
    return rows


def _build_op_breakdown(summaries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    ops: dict[str, dict[str, Any]] = {}
    for s in summaries:
        op = s.get("op_dir", "unknown")
        if op not in ops:
            ops[op] = {"invocations": 0, "total": 0, "pass": 0, "fail": 0, "warn": 0, "blocks": 0}
        ops[op]["invocations"] += 1
        ops[op]["total"] += s.get("total", 0)
        ops[op]["pass"] += s.get("pass", 0)
        ops[op]["fail"] += s.get("fail", 0)
        ops[op]["warn"] += s.get("warn", 0)
    return ops


def build_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    rule_checks = [e for e in events if e.get("event_type") == "rule_check"]
    gate_decisions = [e for e in events if e.get("event_type") == "gate_decision"]
    summaries = [e for e in events if e.get("event_type") == "run_check_summary"]

    if summaries:
        core = _build_from_summaries(summaries)
        core["data_source"] = "run_check_summary"
    else:
        core = _build_from_rule_checks(rule_checks)
        core["data_source"] = "rule_check"

    core["total_gate_blocks"] = sum(1 for e in gate_decisions if e.get("blocked"))
    core["total_summaries"] = len(summaries)
    core["invocations"] = _build_invocation_breakdown(summaries)
    core["per_op"] = _build_op_breakdown(summaries)
    return core


def _md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def write_outputs(summary: dict[str, Any],
                  out_json: Path | None = None,
                  out_md: Path | None = None) -> None:
    json_path = out_json or OUT_JSON
    md_path = out_md or OUT_MD

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = [
        "# Lint 指标汇总",
        "",
        f"数据来源：`{summary.get('data_source', 'unknown')}`",
        "",
        "## 概览",
        "",
    ]
    lines.extend(_md_table(
        ["指标", "值"],
        [
            ["规则检查总数", str(summary["total_rule_checks"])],
            ["通过", str(summary["pass"])],
            ["失败", str(summary["fail"])],
            ["警告", str(summary["warn"])],
            ["跳过", str(summary["skip"])],
            ["门禁阻断", str(summary["total_gate_blocks"])],
            ["总耗时", f"{summary['total_duration_ms']} ms"],
        ],
    ))

    if summary.get("s0_fails"):
        lines.append("")
        lines.append(f"**S0 致命失败**：{', '.join(summary['s0_fails'])}")
    if summary.get("s1_fails"):
        lines.append(f"**S1 失败**：{', '.join(summary['s1_fails'])}")
    elif summary.get("failed_rules"):
        lines.append("")
        lines.append(f"**失败规则**：{', '.join(summary['failed_rules'])}")

    effort_stats = summary.get("effort_stats", {})
    if effort_stats:
        lines.append("")
        lines.append("## 修复成本分布")
        lines.append("")
        effort_rows = []
        for eff, data in effort_stats.items():
            fails_str = ", ".join(data["fails"]) if data["fails"] else "-"
            warns_str = ", ".join(data["warns"]) if data["warns"] else "-"
            effort_rows.append([eff, fails_str, warns_str])
        lines.extend(_md_table(["修复成本", "失败", "警告"], effort_rows))

    invocations = summary.get("invocations", [])
    if invocations:
        lines.append("")
        lines.append("## 每次调用明细")
        lines.append("")
        inv_rows = []
        for inv in invocations:
            s0 = ", ".join(inv.get("s0_fails", [])) or "-"
            inv_rows.append([
                inv.get("invocation_id", ""),
                str(inv.get("stage", "")),
                inv.get("mode", ""),
                f"{inv['pass']}/{inv['fail']}/{inv['warn']}",
                s0,
                f"{inv.get('duration_ms', 0)} ms",
            ])
        lines.extend(_md_table(
            ["调用ID", "阶段", "模式", "通过/失败/警告", "S0 失败", "耗时"],
            inv_rows,
        ))

    per_op = summary.get("per_op", {})
    if per_op:
        lines.append("")
        lines.append("## 按算子统计")
        lines.append("")
        op_rows = []
        for op, data in sorted(per_op.items()):
            op_rows.append([
                op,
                str(data["invocations"]),
                f"{data['pass']}/{data['fail']}/{data['warn']}",
            ])
        lines.extend(_md_table(["算子目录", "调用次数", "通过/失败/警告"], op_rows))

    lines.append("")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Lint 指标报告生成器")
    parser.add_argument("--input", type=Path, default=None,
                        help="lint_events.jsonl 路径（默认：logs/lint_events.jsonl）")
    parser.add_argument("--output-json", type=Path, default=None,
                        help="输出 JSON 路径（默认：logs/summary_latest.json）")
    parser.add_argument("--output-md", type=Path, default=None,
                        help="输出 Markdown 路径（默认：logs/summary_latest.md）")
    parser.add_argument("--op-dir", type=str, default=None,
                        help="按算子目录过滤事件")
    parser.add_argument("--since", type=int, default=None,
                        help="仅包含 ts >= 此 Unix 时间戳的事件")
    args = parser.parse_args()

    events = load_events(path=args.input, since_ts=args.since, op_dir_filter=args.op_dir)
    summary = build_summary(events)
    write_outputs(summary, out_json=args.output_json, out_md=args.output_md)
    _log.info(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
