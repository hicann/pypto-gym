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
"""D2 工件完整性: PL03-PL10."""

from __future__ import annotations

import json
import os

from ..core import (
    CheckContext,
    DESIGN_FILE,
    Finding,
    MODULE_INTERFACES_FILE,
    SPEC_FILE,
    STATE_FILE,
    register,
)


def _check_file_exists(ctx: CheckContext, rule_id: str, filename: str) -> Finding:
    if ctx.file_exists(filename):
        return ctx.make_finding(
            rule_id, "PASS", f"{filename} 存在", file=filename
        )
    return ctx.make_finding(
        rule_id, "FAIL", f"{filename} 不存在", file=filename
    )


@register("PL03")
def check_pl03(ctx: CheckContext) -> Finding:
    """SPEC.md 存在且非空."""
    if not ctx.file_exists(SPEC_FILE):
        return ctx.make_finding("PL03", "FAIL", f"{SPEC_FILE} 不存在", file=SPEC_FILE)
    content = ctx.read_file(SPEC_FILE)
    if not content.strip():
        return ctx.make_finding("PL03", "FAIL", f"{SPEC_FILE} 为空", file=SPEC_FILE)
    return ctx.make_finding("PL03", "PASS", f"{SPEC_FILE} 存在且非空", file=SPEC_FILE)


@register("PL04")
def check_pl04(ctx: CheckContext) -> Finding:
    """{op}_golden.py 存在."""
    filename = f"{ctx.op_name}_golden.py"
    return _check_file_exists(ctx, "PL04", filename)


@register("PL05")
def check_pl05(ctx: CheckContext) -> Finding:
    """{op}_golden_cpu.py 存在."""
    filename = f"{ctx.op_name}_golden_cpu.py"
    return _check_file_exists(ctx, "PL05", filename)


@register("PL07")
def check_pl07(ctx: CheckContext) -> Finding:
    """DESIGN.md 存在."""
    return _check_file_exists(ctx, "PL07", DESIGN_FILE)


@register("PL08")
def check_pl08(ctx: CheckContext) -> Finding:
    """module_interfaces.yaml 存在."""
    return _check_file_exists(ctx, "PL08", MODULE_INTERFACES_FILE)


@register("PL09")
def check_pl09(ctx: CheckContext) -> Finding:
    """test_{op}.py 存在."""
    filename = f"test_{ctx.op_name}.py"
    return _check_file_exists(ctx, "PL09", filename)


@register("PL10")
def check_pl10(ctx: CheckContext) -> Finding:
    """orchestrator_state.json 是合法 JSON 且 max_stage == 4."""
    state_path = ctx.file_path(STATE_FILE)
    if not os.path.isfile(state_path):
        return ctx.make_finding(
            "PL10", "FAIL", f"{STATE_FILE} 不存在", file=STATE_FILE
        )
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return ctx.make_finding(
            "PL10", "FAIL", f"{STATE_FILE} 不是合法 JSON: {e}", file=STATE_FILE
        )
    max_stage = data.get("max_stage")
    if max_stage != 4:
        return ctx.make_finding(
            "PL10", "FAIL",
            f"{STATE_FILE} max_stage={max_stage}（期望 4）",
            file=STATE_FILE
        )
    return ctx.make_finding(
        "PL10", "PASS", f"{STATE_FILE} 合法，max_stage=4", file=STATE_FILE
    )
