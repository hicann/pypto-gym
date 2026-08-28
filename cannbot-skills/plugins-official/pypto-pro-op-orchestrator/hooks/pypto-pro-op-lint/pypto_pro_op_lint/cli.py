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

import argparse
import os

from . import checks  # noqa: F401
from .core import (
    GATE_RULES_BY_STAGE,
    MODULE_GATE_RULE_IDS,
    Finding,
    _has_error_fail,
    _run_checks,
    _system_failure,
)
from .hooks import hook_post_edit, hook_stop
from .infer import _build_context, _module_staged_filename, _resolve_module_suffix
from .observability import _print_findings


def _cmd_run(findings: list[Finding]) -> int:
    _print_findings(findings)
    return 2 if _has_error_fail(findings) else 0


def cmd_lint_test(op_dir: str, stage: int) -> int:
    from .core import TEST_RULE_IDS
    ctx = _build_context(op_dir, stage)
    findings, _ = _run_checks(ctx, TEST_RULE_IDS)
    return _cmd_run(findings)


def cmd_lint_golden(op_dir: str, stage: int) -> int:
    from .core import GOLDEN_RULE_IDS
    ctx = _build_context(op_dir, stage)
    findings, _ = _run_checks(ctx, GOLDEN_RULE_IDS)
    return _cmd_run(findings)


def cmd_check_gate(op_dir: str, stage: int) -> int:
    """Gate check for complete_stage(N).

    Runs all rules applicable to the given stage (see GATE_RULES_BY_STAGE).
    """
    ctx = _build_context(op_dir, stage)
    gate_rules = GATE_RULES_BY_STAGE.get(stage, [])
    if not gate_rules:
        return _cmd_run([
            _system_failure("LINT_CONFIG", f"Stage {stage} is not supported, gate refused to pass")
        ])
    findings, _ = _run_checks(ctx, gate_rules)
    return _cmd_run(findings)


def cmd_check_module_gate(op_dir: str, module: str, stage: int = 4) -> int:
    """Module gate check for submit_for_verify / complete_module.

    Only scans the current module's staged file, running PL01 + PL02.
    """
    ctx = _build_context(op_dir, stage)
    if stage != 4:
        return _cmd_run([
            _system_failure(
                "LINT_CONFIG",
                f"Module gate only applies to Stage 4, received Stage {stage}",
            )
        ])
    suffix = _resolve_module_suffix(op_dir, module)
    if suffix is None:
        return _cmd_run([
            _system_failure(
                "LINT_CONFIG",
                f"cannot resolve the suffix of module {module} (no valid stage4_modules in state.json)",
            )
        ])
    staged_file = _module_staged_filename(ctx.op_name, suffix)
    if not os.path.isfile(ctx.file_path(staged_file)):
        return _cmd_run([
            _system_failure(
                "LINT_CONFIG",
                f"staged file for Module {module} does not exist: {staged_file}",
            )
        ])
    ctx.module_scope = suffix
    findings, _ = _run_checks(ctx, MODULE_GATE_RULE_IDS)
    return _cmd_run(findings)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PyPTO-Pro 算子开发流程确定性检查工具"
    )
    parser.add_argument(
        "--hook",
        choices=["post-edit", "stop"],
        help="Hook 模式（从环境变量/stdin 读 JSON）",
    )
    parser.add_argument("--lint-test", action="store_true", help="检查 test 文件")
    parser.add_argument("--lint-golden", action="store_true", help="检查 golden 文件")
    parser.add_argument("--check-gate", action="store_true", help="检查阶段门禁")
    parser.add_argument(
        "--check-module-gate", action="store_true",
        help="检查 Stage 4 module 门禁（仅扫当前 module 的 staged 文件）",
    )
    parser.add_argument("--module", help="Module 序号（如 1, 2, 3），与 --check-module-gate 配合")
    parser.add_argument("--op-dir", help="算子工作目录")
    parser.add_argument("--stage", type=int, default=4, help="当前阶段 (1-4)")
    return parser


def _require_op_dir(parser: argparse.ArgumentParser, args: argparse.Namespace, option: str) -> str:
    if not args.op_dir:
        parser.error(f"{option} requires --op-dir")
    return args.op_dir


def _dispatch_command(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if args.hook:
        if args.hook == "post-edit":
            return hook_post_edit()
        elif args.hook == "stop":
            return hook_stop()
        parser.error(f"unsupported hook: {args.hook}")

    if args.lint_test:
        op_dir = _require_op_dir(parser, args, "--lint-test")
        return cmd_lint_test(op_dir, args.stage)

    if args.lint_golden:
        op_dir = _require_op_dir(parser, args, "--lint-golden")
        return cmd_lint_golden(op_dir, args.stage)

    if args.check_gate:
        op_dir = _require_op_dir(parser, args, "--check-gate")
        return cmd_check_gate(op_dir, args.stage)

    if args.check_module_gate:
        op_dir = _require_op_dir(parser, args, "--check-module-gate")
        if not args.module:
            parser.error("--check-module-gate requires --module (e.g. 1, 2, 3)")
        return cmd_check_module_gate(op_dir, args.module, args.stage)

    parser.print_help()
    return 0


def main() -> int:
    parser = _build_parser()
    return _dispatch_command(parser, parser.parse_args())
