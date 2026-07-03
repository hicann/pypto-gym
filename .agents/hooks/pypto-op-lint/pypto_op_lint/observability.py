#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional

from .core import SCRIPT_DIR, STRICT_ENV, CheckContext, Finding, _has_error_fail

LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")
LOGS_EVENTS_FILE = os.path.join(LOGS_DIR, "lint_events.jsonl")

_LOG_SKIP = os.environ.get("PYPTO_OP_LINT_LOG_SKIP", "0") == "1"


@dataclass
class _RunMeta:
    mode: str
    strict: bool
    invocation_id: str = ""


_metrics_batch: list[dict[str, Any]] = []


def _output_hook_json(event: str, **kwargs):
    """输出 hookSpecificOutput JSON 到 stdout"""
    output = {"hookSpecificOutput": {"hookEventName": event, **kwargs}}
    _write_stdout_json(output)


def _print_findings(findings: list[Finding]):
    has_error_fail = _has_error_fail(findings)
    result = {
        "passed": not any(f.status == "FAIL" for f in findings),
        "findings": [asdict(f) for f in findings],
        "summary": {
            "pass": sum(1 for f in findings if f.status == "PASS"),
            "warn": sum(1 for f in findings if f.status == "WARN"),
            "info": sum(1 for f in findings if f.status == "INFO"),
            "fail": sum(1 for f in findings if f.status == "FAIL"),
            "skip": sum(1 for f in findings if f.status == "SKIP"),
            "has_error_fail": has_error_fail,
        },
    }
    _write_stdout_json(result, indent=2)


def _write_stdout_json(payload: dict[str, Any], indent: Optional[int] = None) -> None:
    content = json.dumps(payload, ensure_ascii=False, indent=indent)
    os.write(1, f"{content}\n".encode("utf-8"))


def _ensure_logs_dir() -> None:
    os.makedirs(LOGS_DIR, exist_ok=True)


def _append_jsonl(path: str, record: dict[str, Any]) -> None:
    _ensure_logs_dir()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _emit_metric_event_buffered(ctx: CheckContext, finding: Finding,
                                run_meta: _RunMeta, duration_ms: float) -> None:
    if finding.status == "SKIP" and not _LOG_SKIP:
        return
    message_hash = hashlib.sha1(finding.message.encode("utf-8")).hexdigest()[:12]
    _metrics_batch.append({
        "ts": int(time.time()),
        "invocation_id": run_meta.invocation_id,
        "op_dir": ctx.op_dir,
        "stage": ctx.stage,
        "rule_id": finding.rule_id,
        "severity": finding.severity,
        "fix_effort": finding.fix_effort,
        "status": finding.status,
        "mode": run_meta.mode,
        "strict": run_meta.strict,
        "duration_ms": round(float(duration_ms), 3),
        "file": finding.file,
        "message_hash": message_hash,
        "event_type": "rule_check",
    })


def _flush_metrics_batch() -> None:
    if not _metrics_batch:
        return
    try:
        _ensure_logs_dir()
        with open(LOGS_EVENTS_FILE, "a", encoding="utf-8") as f:
            for record in _metrics_batch:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass
    finally:
        _metrics_batch.clear()


def _emit_summary_event(ctx: CheckContext, findings: list[Finding],
                        run_meta: _RunMeta, total_duration_ms: float) -> None:
    try:
        fails = [f for f in findings if f.status == "FAIL"]
        warns = [f for f in findings if f.status == "WARN"]
        skip_rules = sorted(f.rule_id for f in findings if f.status == "SKIP")
        skip_count = len(skip_rules)
        non_skip_count = len(findings) - skip_count
        effort_stats: dict[str, dict[str, list[str]]] = {}
        for f in findings:
            eff = f.fix_effort or "unknown"
            if eff not in effort_stats:
                effort_stats[eff] = {"fails": [], "warns": []}
            if f.status == "FAIL":
                effort_stats[eff]["fails"].append(f.rule_id)
            elif f.status == "WARN":
                effort_stats[eff]["warns"].append(f.rule_id)
        for eff in effort_stats:
            effort_stats[eff]["fails"].sort()
            effort_stats[eff]["warns"].sort()
        event = {
            "ts": int(time.time()),
            "invocation_id": run_meta.invocation_id,
            "event_type": "run_check_summary",
            "op_dir": ctx.op_dir,
            "stage": ctx.stage,
            "mode": run_meta.mode,
            "strict": run_meta.strict,
            "total": len(findings),
            "pass": sum(1 for f in findings if f.status == "PASS"),
            "fail": len(fails),
            "warn": len(warns),
            "info": sum(1 for f in findings if f.status == "INFO"),
            "skip": skip_count,
            "skip_rules": skip_rules,
            "events_written": non_skip_count if not _LOG_SKIP else len(findings),
            "s0_fails": sorted(f.rule_id for f in fails if f.severity == "S0"),
            "s1_fails": sorted(f.rule_id for f in fails if f.severity == "S1"),
            "effort_stats": dict(sorted(effort_stats.items())),
            "total_duration_ms": round(total_duration_ms, 3),
        }
        _append_jsonl(LOGS_EVENTS_FILE, event)
    except OSError:
        pass


def _emit_gate_event(ctx: CheckContext, blocked: bool, blocking_rules: list[str],
                     invocation_id: str = "") -> None:
    try:
        event = {
            "ts": int(time.time()),
            "invocation_id": invocation_id,
            "op_dir": ctx.op_dir,
            "stage": ctx.stage,
            "mode": "stop",
            "strict": os.environ.get(STRICT_ENV, "1") == "1",
            "event_type": "gate_decision",
            "blocked": blocked,
            "blocking_rules": blocking_rules,
        }
        _append_jsonl(LOGS_EVENTS_FILE, event)
    except OSError:
        pass
