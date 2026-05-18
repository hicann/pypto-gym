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
"""VerifierResult.failure_category 与 skill_report / direct 推导."""

from __future__ import annotations

from benchmark.verifier_runner import (
    VerifierResult,
    VerifierStatus,
    _failure_category_for_direct,
    _skill_report_to_result,
)


def test_to_dict_includes_failure_category() -> None:
    r = VerifierResult(
        op_name="x",
        status=VerifierStatus.PASSED,
        failure_category="",
        opencode_token_usage={"supported": True, "total": 10},
        opencode_token_usage_attempts=[
            {"attempt": 1, "token_usage": {"supported": True, "total": 10}},
        ],
    )
    data = r.to_dict()
    assert "failure_category" in data
    assert data["failure_category"] == ""
    assert data["opencode_token_usage"]["total"] == 10
    assert data["opencode_token_usage_attempts"][0]["attempt"] == 1


def test_skill_report_explicit_failure_category() -> None:
    report = {
        "final_verdict": "FAIL_CHEAT",
        "failure_category": "semantic_cheat",
        "correctness": {"status": "failed"},
        "performance": {},
    }
    r = _skill_report_to_result(
        op_name="op",
        report=report,
        log_text="",
        log_file=None,
        duration=0.0,
    )
    assert r.failure_category == "semantic_cheat"


def test_skill_report_fallback_covers_all_final_verdicts() -> None:
    cases = [
        ("PASS", ""),
        ("FAIL_CHEAT", "cheat"),
        ("FAIL_CORRECTNESS", "correctness"),
        ("FAIL_PERFORMANCE", "performance"),
        ("BASELINE_FAILED", "baseline_failed"),
        ("ERROR", "error"),
        ("NOT_A_REAL_VERDICT", "unknown_verdict"),
    ]
    for final, expected in cases:
        report = {"final_verdict": final, "correctness": {}, "performance": {}}
        r = _skill_report_to_result(
            op_name="op",
            report=report,
            log_text="",
            log_file=None,
            duration=0.0,
        )
        assert r.failure_category == expected, final


def test_skill_report_whitespace_only_failure_category_falls_back() -> None:
    report = {
        "final_verdict": "FAIL_CORRECTNESS",
        "failure_category": "   ",
        "correctness": {},
        "performance": {},
    }
    r = _skill_report_to_result(
        op_name="op",
        report=report,
        log_text="",
        log_file=None,
        duration=0.0,
    )
    assert r.failure_category == "correctness"


def test_direct_failure_category_for_status_and_message() -> None:
    assert _failure_category_for_direct(VerifierStatus.PASSED, "") == ""
    assert _failure_category_for_direct(VerifierStatus.MISSING_INPUT, "x") == "missing_input"
    assert _failure_category_for_direct(
        VerifierStatus.FAILED, "kernelverifier.run 返回 false"
    ) == "correctness"
    assert _failure_category_for_direct(
        VerifierStatus.FAILED, "x", perf_message="run_profile 异常: boom"
    ) == "performance"
    assert _failure_category_for_direct(
        VerifierStatus.ERROR, "load_config('pypto') 失败: x"
    ) == "config_error"
    assert _failure_category_for_direct(
        VerifierStatus.ERROR, "WorkerManager 中没有匹配"
    ) == "worker_error"
    assert _failure_category_for_direct(
        VerifierStatus.ERROR, "KernelVerifier.run/run_profile 异常: x"
    ) == "runtime_error"
    assert _failure_category_for_direct(VerifierStatus.ERROR, "something else") == "error"
