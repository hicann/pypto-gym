# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Optimization budget and explicit early-exit semantics."""
from __future__ import annotations

from .common import ContractError, finite


def resolve_policy(overrides=None, project=None, *, from_pro=False):
    defaults = {"enabled": from_pro, "max_iterations": 10 if from_pro else 5,
                "stop_on_criteria": True, "time_budget_s": None,
                "max_no_improvement_rounds": None}
    for source in (project or {}, overrides or {}):
        if not isinstance(source, dict) or set(source) - set(defaults):
            raise ContractError("unknown optimization policy field")
        defaults.update(source)
    for key in ("enabled", "stop_on_criteria"):
        if not isinstance(defaults[key], bool):
            raise ContractError(f"optimization.{key} must be boolean")
    for key in ("max_iterations", "max_no_improvement_rounds"):
        value = defaults[key]
        if key == "max_iterations" and value in ("infinite", "unlimited", "无限", "无限轮"):
            value = defaults[key] = None
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ContractError(f"{key} must be a positive integer or null")
    if defaults["time_budget_s"] is not None:
        finite(defaults["time_budget_s"], "time_budget_s", positive=True)
    return defaults


def stop_reason(policy, rounds, *, criteria_met=False, elapsed=0, no_improvement=0):
    if criteria_met and policy.get("stop_on_criteria", True):
        return "criteria_met"
    if policy["max_iterations"] is not None and rounds >= policy["max_iterations"]:
        return "iteration_budget_exhausted"
    if policy["time_budget_s"] is not None and elapsed >= policy["time_budget_s"]:
        return "time_budget_exhausted"
    if (policy["max_no_improvement_rounds"] is not None and
            no_improvement >= policy["max_no_improvement_rounds"]):
        return "no_improvement"
    return None
