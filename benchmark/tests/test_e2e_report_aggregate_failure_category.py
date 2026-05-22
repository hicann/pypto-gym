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

"""汇总 CLI 与旧版 result.json（缺 failure_category）兼容性."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from benchmark.report import CaseRunRecord, write_case_result

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_legacy_record_without_failure_category(report_dir: Path) -> None:
    """模拟旧落盘：完全省略 failure_category 字段."""
    op_dir = report_dir / "level1" / "BadOp"
    op_dir.mkdir(parents=True, exist_ok=True)
    legacy = {
        "op_name": "BadOp",
        "case_id": "99_BadOp",
        "source_file": "x.py",
        "level": "level1",
        "report_subdir": "level1/BadOp",
        "overall_status": "verify_error",
        "pypto_status": "success",
        "verifier_status": "error",
        "correctness": False,
        "pypto_duration_sec": 0.0,
        "verifier_duration_sec": 0.0,
        "pypto_retry_count": 0,
        "pypto_message": "",
        "verifier_message": "boom",
        "started_at": "",
        "finished_at": "",
    }
    (op_dir / "result.json").write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")


def test_benchmark_report_cli_pass_shows_em_dash_legacy_aggregates(tmp_path: Path) -> None:
    report_dir = tmp_path / "report"
    report_dir.mkdir(parents=True)

    ok = CaseRunRecord(
        op_name="GoodOp",
        case_id="1_GoodOp",
        source_file="g.py",
        level="level1",
        report_subdir="level1/GoodOp",
        overall_status="success",
        pypto_status="success",
        verifier_status="passed",
        correctness=True,
        failure_category="",
        pypto_duration_sec=0.1,
        verifier_duration_sec=0.2,
        pypto_retry_count=0,
        started_at="2026-01-01T00:00:00",
        finished_at="2026-01-01T00:00:10",
    )
    write_case_result(ok, report_dir)
    _write_legacy_record_without_failure_category(report_dir)

    env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT)}
    proc = subprocess.run(
        [sys.executable, "-m", "benchmark.report", str(report_dir)],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    md = (report_dir / "summary.md").read_text(encoding="utf-8")
    assert "| 总状态 | 错误类别 | 精度 |" in md
    assert "- **失败类别分布**: " in md
    assert "PASS | — | ✓ |" in md

    payload = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    totals = payload["totals"]
    assert "by_failure_category" in totals
    assert totals["by_failure_category"] == {"": 2}
    assert totals["total"] == 2


def test_benchmark_summary_subcommand_idempotent_with_preserved_meta(tmp_path: Path) -> None:
    """``python -m benchmark summary`` 与 ``benchmark.report`` 同源逻辑；保留 meta 时再导出应逐字节一致."""
    report_dir = tmp_path / "report"
    report_dir.mkdir(parents=True)

    ok = CaseRunRecord(
        op_name="GoodOp",
        case_id="1_GoodOp",
        source_file="g.py",
        level="level1",
        report_subdir="level1/GoodOp",
        overall_status="success",
        pypto_status="success",
        verifier_status="passed",
        correctness=True,
        failure_category="",
        pypto_duration_sec=0.1,
        verifier_duration_sec=0.2,
        pypto_retry_count=0,
        started_at="2026-01-01T00:00:00",
        finished_at="2026-01-01T00:00:10",
    )
    write_case_result(ok, report_dir)
    _write_legacy_record_without_failure_category(report_dir)

    env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT)}
    proc_report = subprocess.run(
        [sys.executable, "-m", "benchmark.report", str(report_dir)],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_report.returncode == 0, proc_report.stdout + proc_report.stderr
    md_after_report = (report_dir / "summary.md").read_text(encoding="utf-8")

    proc_summary = subprocess.run(
        [sys.executable, "-m", "benchmark", "summary", str(report_dir)],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_summary.returncode == 0, proc_summary.stdout + proc_summary.stderr
    md_after_summary = (report_dir / "summary.md").read_text(encoding="utf-8")
    assert md_after_summary == md_after_report
