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
import os
import select
import sys
from typing import Any, Optional

from .core import (
    CheckContext,
    HOOK_INPUT_ENV,
    STATE_FILE,
    _load_rules,
)


def _infer_op_dir(file_path: str) -> Optional[str]:
    """Resolve the operator directory that owns ``file_path``.

    Covers three layouts:
    * Standard: ``<op_dir>/test_{op}.py`` → ``<op_dir>``
    * Module dev (L1): ``<op_dir>/modules/test_{op}_module*.py`` → ``<op_dir>``
    * Stateless: no ``.orchestrator_state.json``, heuristic match.
    """
    if not file_path:
        return None
    op_dir = os.path.dirname(os.path.abspath(file_path))
    if os.path.isfile(os.path.join(op_dir, STATE_FILE)):
        return op_dir
    # modules/ subdirectory → walk up to operator main directory
    if os.path.basename(op_dir) == "modules":
        parent = os.path.dirname(op_dir)
        if parent and os.path.isfile(os.path.join(parent, STATE_FILE)):
            return parent
        parent_basename = os.path.basename(parent)
        if parent_basename and _looks_like_stateless_op_dir(parent, parent_basename):
            return parent
    basename = os.path.basename(file_path)
    inferred_op_name = _infer_op_name_from_filename(basename)
    if inferred_op_name and _looks_like_stateless_op_dir(op_dir, inferred_op_name):
        return op_dir
    return None


def _infer_op_name_from_filename(filename: str) -> str:
    if filename.startswith("test_") and filename.endswith(".py"):
        return filename[len("test_"):-len(".py")]
    if filename.endswith("_golden.py"):
        return filename[:-len("_golden.py")]
    if filename.endswith("_golden_cpu.py"):
        return filename[:-len("_golden_cpu.py")]
    return ""


def _looks_like_stateless_op_dir(op_dir: str, op_name: str) -> bool:
    try:
        files = set(os.listdir(op_dir))
    except OSError:
        return False
    expected = {
        f"{op_name}_golden.py",
        f"{op_name}_golden_cpu.py",
        f"test_{op_name}.py",
        "SPEC.md",
        "DESIGN.md",
        "module_interfaces.yaml",
        "GOLDEN_PERF_REPORT.md",
    }
    return len(files & expected) >= 2


def _load_state_json(state_path: str) -> dict:
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (ValueError, OSError):
        return {}


def _get_current_stage(op_dir: str) -> int:
    state_path = os.path.join(op_dir, STATE_FILE)
    if not os.path.isfile(state_path):
        return 0
    data = _load_state_json(state_path)
    return int(data.get("current_stage", 0))


def _get_op_name(op_dir: str) -> str:
    state_path = os.path.join(op_dir, STATE_FILE)
    if not os.path.isfile(state_path):
        return os.path.basename(op_dir)
    data = _load_state_json(state_path)
    name = data.get("operator_name", "")
    return name if name else os.path.basename(op_dir)


def _build_context(op_dir: str, stage: Optional[int] = None) -> CheckContext:
    rules = _load_rules()
    op_dir = os.path.abspath(op_dir)
    if stage is None:
        stage = _get_current_stage(op_dir)
    op_name = _get_op_name(op_dir)
    return CheckContext(op_dir=op_dir, op_name=op_name, stage=stage, rules=rules)


def _load_hook_input() -> dict[str, Any]:
    raw = ""
    if not sys.stdin.closed:
        try:
            if select.select([sys.stdin], [], [], 0)[0]:
                raw = sys.stdin.read()
        except (ValueError, OSError):
            pass
    if not raw.strip():
        raw = os.environ.get(HOOK_INPUT_ENV, "")
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _is_test_file(filename: str) -> bool:
    """Match test_{op}.py and modules/test_{op}_module*.py."""
    basename = os.path.basename(filename)
    if basename.startswith("test_") and basename.endswith(".py"):
        return True
    return False


def _is_golden_file(filename: str) -> bool:
    """Match {op}_golden.py, {op}_golden_cpu.py, {op}_golden_stage*.py."""
    basename = os.path.basename(filename)
    if basename.endswith("_golden.py"):
        return True
    if basename.endswith("_golden_cpu.py"):
        return True
    if "_golden_stage" in basename and basename.endswith(".py"):
        return True
    return False


def _rule_ids_for_filename(filename: str) -> list[str]:
    """Return rule IDs for post-edit checks on a single file.

    Only includes rules that validate the edited file itself. Cross-file
    rules (gate class) are deferred to the gate check.
    """
    from .core import POST_EDIT_GOLDEN_RULES, POST_EDIT_TEST_RULES

    if _is_test_file(filename):
        return POST_EDIT_TEST_RULES
    if _is_golden_file(filename):
        return POST_EDIT_GOLDEN_RULES
    return []


def _module_staged_filename(op_name: str, module_suffix: str) -> str:
    """Return the relative path of a module's staged impl file.

    module_suffix is the cumulative suffix, e.g. "1", "12", "123".
    """
    return os.path.join("modules", f"test_{op_name}_module{module_suffix}.py")


_MAX_OP_DIR_SEARCH_DEPTH = 8


def _find_nearest_op_dir(cwd: str) -> Optional[str]:
    """Walk up from cwd to find the owning operator directory."""
    abs_cwd = os.path.abspath(cwd)
    current = abs_cwd
    for _ in range(_MAX_OP_DIR_SEARCH_DEPTH):
        if os.path.isfile(os.path.join(current, STATE_FILE)):
            return current
        if _looks_like_stateless_op_dir(current, os.path.basename(current)):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return None


def _resolve_module_suffix(op_dir: str, module: str) -> Optional[str]:
    """Compute the cumulative suffix for a given module number.

    Module 1 → "1", Module 2 → "12", Module 3 → "123", etc.
    Reads module_count from state to validate the module number.
    """
    state_path = os.path.join(op_dir, STATE_FILE)
    if not os.path.isfile(state_path):
        return None
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, ValueError):
        return None
    stage4_modules = state.get("stage4_modules")
    if not isinstance(stage4_modules, dict):
        return None
    module_count = stage4_modules.get("module_count", 0)
    if not isinstance(module_count, int):
        return None
    try:
        module_num = int(module)
    except (TypeError, ValueError):
        return None
    if module_num < 1 or module_num > module_count:
        return None
    suffix = ""
    for i in range(1, module_num + 1):
        suffix += str(i)
    return suffix
