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

"""report.CaseRunRecord.failure_category 与 summary.md 渲染."""

from __future__ import annotations

from dataclasses import asdict

from benchmark.report import (
    CaseRunRecord,
    _compute_totals,
    _failure_category_display,
    _render_markdown,
    counts_as_aggregate_success,
    is_profile_only_failure,
)


def test_profile_only_failure_counts_as_success_in_totals_not_in_failure_category() -> None:
    prof = CaseRunRecord(
        op_name="p",
        case_id="9",
        source_file="p.py",
        overall_status="verify_failed",
        failure_category="performance",
        correctness=True,
        pypto_status="success",
        verifier_status="failed",
        pypto_duration_sec=0.0,
        verifier_duration_sec=0.0,
        pypto_retry_count=0,
    )
    bad = CaseRunRecord(
        op_name="b",
        case_id="10",
        source_file="b.py",
        overall_status="verify_error",
        failure_category="error",
        correctness=False,
        pypto_status="success",
        verifier_status="error",
        pypto_duration_sec=0.0,
        verifier_duration_sec=0.0,
        pypto_retry_count=0,
    )
    assert is_profile_only_failure(prof) is True
    assert is_profile_only_failure(bad) is False
    t = _compute_totals([prof, bad], include_by_level=False)
    assert t["total"] == 2
    assert t["success"] == 1
    assert t["by_status"] == {"success": 1, "verify_error": 1}
    assert t["profile_only_failures"] == 1
    assert t["by_failure_category"] == {"error": 1}


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
    assert t["profile_only_failures"] == 0


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
                "profile_only_failures": 0,
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
    assert "总通过 (汇总口径)" in md
    assert "失败类别分布" in md
    assert "缺少输入=1" in md
    assert "PASS | — | ✓ |" in md
    assert "| ERROR | 缺少输入 | ✗ |" in md


def test_render_markdown_profile_only_shows_badge_and_count() -> None:
    prof = CaseRunRecord(
        op_name="p",
        case_id="9_p",
        source_file="p.py",
        level="L1",
        overall_status="verify_failed",
        failure_category="performance",
        correctness=True,
        pypto_status="success",
        verifier_status="failed",
        pypto_duration_sec=0.1,
        verifier_duration_sec=0.2,
        pypto_retry_count=0,
        pypto_message="",
        verifier_message="perf bad",
        started_at="2026-01-01T00:00:00",
        finished_at="2026-01-01T00:01:00",
    )
    md = _render_markdown(
        {
            "meta": {},
            "totals": _compute_totals([prof], include_by_level=False),
            "cases": [asdict(prof)],
        }
    )
    assert "PASS (prof 失败)" in md
    assert "仅 profile/性能失败" in md
    assert "| `L1` | `p` | `9_p` | PASS (prof 失败) | 性能失败 | ✓ |" in md


def test_verifier_skipped_counts_as_aggregate_success_but_keeps_status() -> None:
    skipped = CaseRunRecord(
        op_name="p",
        case_id="9_p",
        source_file="p.py",
        level="L1",
        overall_status="verifier_skipped",
        correctness=None,
        pypto_status="success",
        verifier_status="skipped",
        pypto_duration_sec=0.1,
        verifier_duration_sec=0.0,
        pypto_retry_count=0,
        pypto_message="",
        verifier_message="verifier.skip=true",
        started_at="2026-01-01T00:00:00",
        finished_at="2026-01-01T00:01:00",
    )

    assert counts_as_aggregate_success(skipped) is True
    totals = _compute_totals([skipped], include_by_level=False)
    assert totals["success"] == 1
    assert totals["by_status"] == {"verifier_skipped": 1}
    assert totals["correctness"]["unknown"] == 1

    md = _render_markdown(
        {
            "meta": {},
            "totals": totals,
            "cases": [asdict(skipped)],
        }
    )
    assert "| `L1` | `p` | `9_p` | SKIP (verify) | — | — |" in md


def test_compute_totals_aggregates_opencode_token_usage() -> None:
    record = CaseRunRecord(
        op_name="p",
        case_id="9_p",
        source_file="p.py",
        level="L1",
        overall_status="success",
        correctness=True,
        pypto_status="success",
        verifier_status="passed",
        pypto_token_usage={
            "supported": True,
            "message_count": 1,
            "session_count": 1,
            "total": 100,
            "input": 10,
            "output": 20,
            "reasoning": 30,
            "cache": {"read": 40, "write": 0},
            "cost": 0.01,
            "by_model": {},
        },
        verifier_token_usage={
            "supported": True,
            "message_count": 2,
            "session_count": 1,
            "total": 200,
            "input": 50,
            "output": 60,
            "reasoning": 70,
            "cache": {"read": 20, "write": 0},
            "cost": 0.02,
            "by_model": {},
        },
    )

    totals = _compute_totals([record], include_by_level=False)
    assert totals["tokens"]["total"] == 300
    assert totals["tokens"]["input"] == 60
    assert totals["tokens"]["output"] == 80
    assert totals["tokens"]["reasoning"] == 100
    assert totals["tokens"]["cache"]["read"] == 60
    assert totals["tokens"]["cost"] == 0.03
    assert totals["tokens_by_phase"]["pypto"]["total"] == 100
    assert totals["tokens_by_phase"]["verifier"]["total"] == 200

    md = _render_markdown({"meta": {}, "totals": totals, "cases": [asdict(record)]})
    assert "OpenCode token 用量" in md
    assert "total=300" in md
    assert "pypto=100" in md
    assert "verifier=200" in md
