#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
"""report.CaseRunRecord.failure_category 与 summary.md 渲染."""

from __future__ import annotations

from benchmark.report import (
    CaseRunRecord,
    _compute_totals,
    _failure_category_display,
    _render_markdown,
)


def test_case_run_record_accepts_missing_failure_category_from_json() -> None:
    data = {
        "op_name": "x",
        "case_id": "1",
        "source_file": "p.py",
        "overall_status": "success",
    }
    r = CaseRunRecord(**data)
    assert r.failure_category == ""


def test_compute_totals_by_failure_category() -> None:
    records = [
        CaseRunRecord(op_name="a", case_id="1", source_file="a.py", failure_category=""),
        CaseRunRecord(
            op_name="b",
            case_id="2",
            source_file="b.py",
            failure_category="error",
            overall_status="verify_error",
        ),
        CaseRunRecord(
            op_name="c",
            case_id="3",
            source_file="c.py",
            failure_category="error",
            overall_status="verify_error",
        ),
    ]
    t = _compute_totals(records, include_by_level=False)
    assert t["by_failure_category"] == {"": 1, "error": 2}


def test_failure_category_display_pass_empty() -> None:
    assert _failure_category_display("") == "—"
    assert _failure_category_display("PASS") == "—"
    assert _failure_category_display("  pass  ") == "—"


def test_render_markdown_detail_and_overview_fc() -> None:
    md = _render_markdown(
        {
            "meta": {"m": "v"},
            "totals": {
                "total": 2,
                "success": 1,
                "success_rate": 0.5,
                "correctness": {"pass": 1, "fail": 1, "unknown": 0},
                "by_status": {"success": 1, "verify_error": 1},
                "by_failure_category": {"": 1, "missing_input": 1},
                "duration_sec": {
                    "pypto_total": 0,
                    "pypto_mean": 0,
                    "verify_total": 0,
                    "verify_mean": 0,
                    "stage_sum_total": 0,
                    "wall_total": 0,
                    "wall_mean": 0,
                    "wall_min": 0,
                    "wall_max": 0,
                },
            },
            "cases": [
                {
                    "op_name": "op",
                    "case_id": "1",
                    "source_file": "s",
                    "level": "L1",
                    "overall_status": "success",
                    "failure_category": "",
                    "correctness": True,
                    "pypto_status": "done",
                    "verifier_status": "passed",
                    "pypto_retry_count": 0,
                    "pypto_duration_sec": 0.1,
                    "verifier_duration_sec": 0.2,
                    "started_at": "2026-01-01T00:00:00",
                    "finished_at": "2026-01-01T00:01:01",
                    "pypto_message": "",
                    "verifier_message": "",
                },
                {
                    "op_name": "op",
                    "case_id": "2",
                    "source_file": "s",
                    "level": "L1",
                    "overall_status": "verify_error",
                    "failure_category": "missing_input",
                    "correctness": False,
                    "pypto_status": "done",
                    "verifier_status": "error",
                    "pypto_retry_count": 0,
                    "pypto_duration_sec": 0,
                    "verifier_duration_sec": 0,
                    "started_at": "",
                    "finished_at": "",
                    "pypto_message": "",
                    "verifier_message": "oops",
                },
            ],
        }
    )
    assert "| level | op | case | 总状态 | 错误类别 | 精度 |" in md
    assert "失败类别分布" in md
    assert "缺少输入=1" in md
    assert "PASS | — | ✓ |" in md
    assert "| ERROR | 缺少输入 | ✗ |" in md
