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
"""D4 测试规范: PL15 (_assert_precision / 禁止 assert_close) + PL16 (import golden_cpu)."""

from __future__ import annotations

import ast

from ..ast_helpers import imports_module, parse_python_source
from ..core import CheckContext, Finding, register


def _get_test_file_content(ctx: CheckContext) -> tuple[str, str]:
    """Return (relative_filename, content) of test_{op}.py.

    Returns ("", "") if the file does not exist.
    """
    filename = f"test_{ctx.op_name}.py"
    if not ctx.file_exists(filename):
        return ("", "")
    return (filename, ctx.read_file(filename))


@register("PL15")
def check_pl15(ctx: CheckContext) -> Finding:
    """test_{op}.py 须含 _assert_precision; 禁止 assert_close."""
    filename, content = _get_test_file_content(ctx)
    if not filename:
        return ctx.make_finding("PL15", "SKIP", f"test_{ctx.op_name}.py does not exist")

    tree, parse_error = parse_python_source(content, filename)
    if tree is None:
        return ctx.make_finding(
            "PL15", "FAIL", f"{filename}: {parse_error}", file=filename
        )

    failures: list[str] = []

    has_assert_precision = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_assert_precision"
        for node in ast.walk(tree)
    )
    if not has_assert_precision:
        failures.append("`_assert_precision` not found (Plan A hybrid tolerance criteria)")

    has_assert_close = any(
        (isinstance(node, ast.Name) and node.id == "assert_close")
        or (isinstance(node, ast.Attribute) and node.attr == "assert_close")
        or (
            isinstance(node, (ast.Import, ast.ImportFrom))
            and any(alias.name.split(".")[-1] == "assert_close" for alias in node.names)
        )
        for node in ast.walk(tree)
    )
    if has_assert_close:
        failures.append("found `assert_close` (forbidden, use _assert_precision instead)")

    if failures:
        return ctx.make_finding(
            "PL15", "FAIL", "\n".join(failures), file=filename
        )
    return ctx.make_finding("PL15", "PASS", "Precision check specification passed", file=filename)


@register("PL16")
def check_pl16(ctx: CheckContext) -> Finding:
    """test_{op}.py 须 import {op}_golden_cpu."""
    filename, content = _get_test_file_content(ctx)
    if not filename:
        return ctx.make_finding("PL16", "SKIP", f"test_{ctx.op_name}.py does not exist")

    tree, parse_error = parse_python_source(content, filename)
    if tree is None:
        return ctx.make_finding(
            "PL16", "FAIL", f"{filename}: {parse_error}", file=filename
        )

    golden_cpu_module = f"{ctx.op_name}_golden_cpu"
    has_import = imports_module(tree, golden_cpu_module)

    if not has_import:
        return ctx.make_finding(
            "PL16", "FAIL",
            f"import `{golden_cpu_module}` not found (precision comparison must use CPU FP32 golden)",
            file=filename
        )
    return ctx.make_finding(
        "PL16", "PASS", f"imported {golden_cpu_module}", file=filename
    )
