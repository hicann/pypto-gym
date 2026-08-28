#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .observability import (
    _emit_metric_event_buffered,
    _emit_summary_event,
    _flush_metrics_batch,
    _RunMeta,
)

logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SPEC_FILE = "SPEC.md"
DESIGN_FILE = "DESIGN.md"
MODULE_INTERFACES_FILE = "module_interfaces.yaml"
STATE_FILE = ".orchestrator_state.json"

MODE_ENV = "PYPTO_PRO_OP_LINT_MODE"
HOOK_INPUT_ENV = "PYPTO_PRO_OP_LINT_HOOK_INPUT"
POST_EDIT_BLOCK_ENV = "PYPTO_PRO_OP_LINT_POST_EDIT_BLOCK"

TEST_RULE_IDS = ["PL01", "PL02", "PL15", "PL16"]
GOLDEN_RULE_IDS = ["PL11", "PL13"]
GATE_RULES_BY_STAGE: dict[int, list[str]] = {
    1: ["PL03", "PL10"],
    2: ["PL04", "PL05", "PL10", "PL11"],
    3: ["PL07", "PL08", "PL10"],
    4: [
        "PL01", "PL02", "PL04", "PL05", "PL09", "PL10", "PL11", "PL13",
        "PL15", "PL16",
    ],
}

MODULE_GATE_RULE_IDS = ["PL01", "PL02"]

POST_EDIT_TEST_RULES = ["PL01", "PL02"]
POST_EDIT_GOLDEN_RULES = ["PL11", "PL13"]


@dataclass
class Finding:
    rule_id: str = ""
    severity: str = ""
    dimension: str = ""
    fix_effort: str = ""
    status: str = "SKIP"
    message: str = ""
    file: str = ""
    line: int = 0


@dataclass
class CheckContext:
    op_dir: str
    op_name: str
    stage: int
    rules: list[dict[str, Any]]
    file_scope: Optional[str] = None
    module_scope: Optional[str] = None
    _file_cache: dict[str, str] = field(default_factory=dict, repr=False)

    def file_path(self, filename: str) -> str:
        return os.path.join(self.op_dir, filename)

    def file_exists(self, filename: str) -> bool:
        return os.path.isfile(self.file_path(filename))

    def read_file(self, filename: str) -> str:
        if filename in self._file_cache:
            return self._file_cache[filename]
        path = self.file_path(filename)
        if not os.path.isfile(path):
            return ""
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        self._file_cache[filename] = content
        return content

    def get_rule(self, rule_id: str) -> dict[str, Any]:
        for r in self.rules:
            if r["id"] == rule_id:
                return r
        return {}

    def make_finding(self, rule_id: str, status: str, message: str,
                     file: str = "", line: int = 0) -> Finding:
        rule = self.get_rule(rule_id)
        return Finding(
            rule_id=rule_id,
            severity=rule.get("severity", "S2"),
            dimension=rule.get("dimension", ""),
            fix_effort=rule.get("fix_effort", ""),
            status=status,
            message=message,
            file=file,
            line=line,
        )


CHECKERS: dict[str, Callable[[CheckContext], Finding]] = {}


class LintConfigurationError(RuntimeError):
    """Raised when the lint configuration cannot safely drive a hard gate."""


def register(rule_id: str):
    def decorator(fn: Callable[[CheckContext], Finding]):
        CHECKERS[rule_id] = fn
        return fn
    return decorator


def _read_rules_file() -> list[dict[str, Any]]:
    rules_path = os.path.join(SCRIPT_DIR, "rules.json")
    try:
        with open(rules_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        message = f"[pypto-pro-op-lint FATAL] failed to load rules.json: {e}"
        logger.error(message)
        raise LintConfigurationError(message) from e

    if not isinstance(data, dict) or not isinstance(data.get("rules"), list):
        raise LintConfigurationError(
            "[pypto-pro-op-lint FATAL] rules.json 缺少 rules 数组"
        )
    return data["rules"]


def _validate_rule(rule: Any, index: int) -> str:
    if not isinstance(rule, dict) or not isinstance(rule.get("id"), str):
        raise LintConfigurationError(
            f"[pypto-pro-op-lint FATAL] rules[{index}] 缺少合法 id"
        )
    rule_id = rule["id"]
    if not isinstance(rule.get("stages"), list):
        raise LintConfigurationError(
            f"[pypto-pro-op-lint FATAL] {rule_id} 缺少 stages 数组"
        )
    invalid_stage = any(
        not isinstance(stage, int) or stage not in GATE_RULES_BY_STAGE
        for stage in rule["stages"]
    )
    if not rule["stages"] or invalid_stage:
        raise LintConfigurationError(
            f"[pypto-pro-op-lint FATAL] {rule_id} 含非法 stages"
        )
    if rule.get("severity") not in {"S0", "S1", "S2", "S3"}:
        raise LintConfigurationError(
            f"[pypto-pro-op-lint FATAL] {rule_id} 缺少合法 severity"
        )
    triggers = rule.get("triggers")
    if (
        not isinstance(triggers, list)
        or not triggers
        or any(not isinstance(trigger, str) or not trigger for trigger in triggers)
    ):
        raise LintConfigurationError(
            f"[pypto-pro-op-lint FATAL] {rule_id} 缺少合法 triggers"
        )
    for field_name in ("dimension", "fix_effort", "target", "rule", "fix_hint"):
        if not isinstance(rule.get(field_name), str) or not rule[field_name].strip():
            raise LintConfigurationError(
                f"[pypto-pro-op-lint FATAL] {rule_id} 缺少合法 {field_name}"
            )
    return rule_id


def _validate_rule_schema(rules: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not rules:
        raise LintConfigurationError(
            "[pypto-pro-op-lint FATAL] rules.json 中没有任何规则"
        )
    rule_ids = [_validate_rule(rule, index) for index, rule in enumerate(rules)]
    seen: set[str] = set()
    duplicate_ids: set[str] = set()
    for rule_id in rule_ids:
        if rule_id in seen:
            duplicate_ids.add(rule_id)
        else:
            seen.add(rule_id)
    if duplicate_ids:
        raise LintConfigurationError(
            "[pypto-pro-op-lint FATAL] rules.json 存在重复规则: "
            + ", ".join(sorted(duplicate_ids))
        )
    return {rule["id"]: rule for rule in rules}


def _referenced_rule_ids() -> set[str]:
    referenced_ids = set(TEST_RULE_IDS + GOLDEN_RULE_IDS)
    referenced_ids.update(MODULE_GATE_RULE_IDS)
    referenced_ids.update(POST_EDIT_TEST_RULES + POST_EDIT_GOLDEN_RULES)
    for gate_rule_ids in GATE_RULES_BY_STAGE.values():
        referenced_ids.update(gate_rule_ids)
    return referenced_ids


def _validate_referenced_rule_ids(rule_by_id: dict[str, dict[str, Any]]) -> None:
    referenced_ids = _referenced_rule_ids()
    missing_ids = sorted(referenced_ids - set(rule_by_id))
    if missing_ids:
        raise LintConfigurationError(
            "[pypto-pro-op-lint FATAL] 门禁引用了未定义规则: "
            + ", ".join(missing_ids)
        )


def _validate_checker_coverage(rule_by_id: dict[str, dict[str, Any]]) -> None:
    configured_ids = set(rule_by_id)
    checker_ids = set(CHECKERS)
    missing_checkers = sorted(configured_ids - checker_ids)
    undocumented_checkers = sorted(checker_ids - configured_ids)
    if missing_checkers or undocumented_checkers:
        details = []
        if missing_checkers:
            details.append("Missing checkers for rules: " + ", ".join(missing_checkers))
        if undocumented_checkers:
            details.append("Checkers missing rule definitions: " + ", ".join(undocumented_checkers))
        raise LintConfigurationError(
            "[pypto-pro-op-lint FATAL] rules/checkers 不一致: " + "; ".join(details)
        )


def _validate_stage_gate_mapping(
    stage: int,
    gate_rule_ids: list[str],
    rule_by_id: dict[str, dict[str, Any]],
) -> None:
    mismatched = [
        rule_id
        for rule_id in gate_rule_ids
        if stage not in rule_by_id[rule_id]["stages"]
    ]
    if mismatched:
        raise LintConfigurationError(
            f"[pypto-pro-op-lint FATAL] Stage {stage} 门禁规则阶段不匹配: "
            + ", ".join(mismatched)
        )

    expected_trigger = f"gate:S{stage}"
    missing_triggers = [
        rule_id
        for rule_id in gate_rule_ids
        if expected_trigger not in rule_by_id[rule_id].get("triggers", [])
    ]
    if missing_triggers:
        raise LintConfigurationError(
            f"[pypto-pro-op-lint FATAL] Stage {stage} 门禁规则缺少 trigger: "
            + ", ".join(missing_triggers)
        )


def _validate_stage_gate_mappings(
    rule_by_id: dict[str, dict[str, Any]],
) -> None:
    for stage, gate_rule_ids in GATE_RULES_BY_STAGE.items():
        _validate_stage_gate_mapping(stage, gate_rule_ids, rule_by_id)


def _validate_required_trigger_group(
    rule_ids: list[str],
    expected_trigger: str,
    rule_by_id: dict[str, dict[str, Any]],
) -> None:
    missing_triggers = [
        rule_id
        for rule_id in rule_ids
        if expected_trigger not in rule_by_id[rule_id].get("triggers", [])
    ]
    if missing_triggers:
        raise LintConfigurationError(
            f"[pypto-pro-op-lint FATAL] 规则缺少 {expected_trigger} trigger: "
            + ", ".join(missing_triggers)
        )


def _parse_stage_trigger(rule_id: str, trigger: str) -> int:
    try:
        stage = int(trigger.removeprefix("gate:S"))
    except ValueError as error:
        raise LintConfigurationError(
            f"[pypto-pro-op-lint FATAL] {rule_id} 含非法 trigger: {trigger}"
        ) from error
    if stage not in GATE_RULES_BY_STAGE:
        raise LintConfigurationError(
            f"[pypto-pro-op-lint FATAL] {rule_id} 含非法 trigger: {trigger}"
        )
    return stage


def _validate_declared_trigger(
    rule_id: str,
    trigger: str,
    post_edit_ids: set[str],
) -> None:
    if trigger.startswith("gate:S"):
        stage = _parse_stage_trigger(rule_id, trigger)
        if rule_id not in GATE_RULES_BY_STAGE[stage]:
            raise LintConfigurationError(
                f"[pypto-pro-op-lint FATAL] {rule_id} 声明了 {trigger} "
                "但未进入对应 Stage 门禁"
            )
        return

    if trigger == "gate:M":
        if rule_id not in MODULE_GATE_RULE_IDS:
            raise LintConfigurationError(
                f"[pypto-pro-op-lint FATAL] {rule_id} 声明了 gate:M "
                "但未进入 Module 门禁"
            )
        return

    if trigger == "post-edit":
        if rule_id not in post_edit_ids:
            raise LintConfigurationError(
                f"[pypto-pro-op-lint FATAL] {rule_id} 声明了 post-edit "
                "但未进入 post-edit 门禁"
            )
        return

    raise LintConfigurationError(
        f"[pypto-pro-op-lint FATAL] {rule_id} 含未知 trigger: {trigger}"
    )


def _validate_declared_triggers(
    rule_by_id: dict[str, dict[str, Any]],
) -> None:
    post_edit_ids = set(POST_EDIT_TEST_RULES + POST_EDIT_GOLDEN_RULES)
    for rule_id, rule in rule_by_id.items():
        for trigger in rule["triggers"]:
            _validate_declared_trigger(rule_id, trigger, post_edit_ids)


def _validate_rule_references(rule_by_id: dict[str, dict[str, Any]]) -> None:
    _validate_referenced_rule_ids(rule_by_id)
    _validate_checker_coverage(rule_by_id)
    _validate_stage_gate_mappings(rule_by_id)
    _validate_required_trigger_group(
        MODULE_GATE_RULE_IDS,
        "gate:M",
        rule_by_id,
    )
    _validate_required_trigger_group(
        POST_EDIT_TEST_RULES + POST_EDIT_GOLDEN_RULES,
        "post-edit",
        rule_by_id,
    )
    _validate_declared_triggers(rule_by_id)


def _load_rules() -> list[dict[str, Any]]:
    rules = _read_rules_file()
    rule_by_id = _validate_rule_schema(rules)
    _validate_rule_references(rule_by_id)
    return rules


def _system_failure(rule_id: str, message: str) -> Finding:
    """Build an S0 failure for lint-engine/configuration faults."""
    return Finding(
        rule_id=rule_id,
        severity="S0",
        dimension="SYSTEM",
        fix_effort="E1",
        status="FAIL",
        message=message,
    )


def _run_single_check(ctx: CheckContext, rule_id: str) -> Finding:
    rule = ctx.get_rule(rule_id)
    if not rule:
        return _system_failure(rule_id, f"Rule {rule_id} is not defined in rules.json")
    if ctx.stage not in rule.get("stages", []):
        return ctx.make_finding(rule_id, "SKIP", "Not applicable in the current stage")
    checker = CHECKERS.get(rule_id)
    if not checker:
        return _system_failure(rule_id, f"Checker function for rule {rule_id} is not registered")
    try:
        return checker(ctx)
    except Exception as error:  # hard gates must fail closed
        return _system_failure(
            rule_id,
            f"Checker for rule {rule_id} raised an exception: {type(error).__name__}: {error}",
        )


def _run_checks(ctx: CheckContext, rule_ids: list[str]) -> tuple[list[Finding], str]:
    invocation_id = uuid.uuid4().hex[:8]
    findings: list[Finding] = []
    run_meta = _RunMeta(
        mode=os.environ.get(MODE_ENV, "cli"),
        invocation_id=invocation_id,
    )
    total_duration_ms = 0.0
    try:
        if not rule_ids:
            finding = _system_failure("LINT_CONFIG", "No check rules selected, gate refused to pass")
            findings.append(finding)
            _emit_metric_event_buffered(ctx, finding, run_meta, 0.0)
        for rule_id in rule_ids:
            start = time.perf_counter()
            finding = _run_single_check(ctx, rule_id)
            findings.append(finding)
            dur = (time.perf_counter() - start) * 1000
            total_duration_ms += dur
            _emit_metric_event_buffered(ctx, finding, run_meta, dur)
    finally:
        _flush_metrics_batch()
    _emit_summary_event(ctx, findings, run_meta, total_duration_ms)
    return findings, invocation_id


def _has_error_fail(findings: list[Finding]) -> bool:
    return any(
        finding.status == "FAIL" and finding.severity in ("S0", "S1")
        for finding in findings
    )
