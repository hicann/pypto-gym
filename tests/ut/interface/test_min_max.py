#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Regression tests: verify that Python's builtin min is invoked as-is when all
arguments are non-dynamic (i.e. no SymbolicScalar is involved), and that the
framework intercepts the call only when at least one SymbolicScalar is present.
"""

import ast
import pypto

from pypto.frontend.parser.diagnostics import Diagnostics, Source
from pypto.frontend.parser.evaluator import ExprEvaluator


def _diag_source():
    return None


def _eval_expr(expr: str, **var_table):
    node = ast.parse(expr, mode="eval").body
    return ExprEvaluator.eval(node, var_table, Diagnostics(Source(_diag_source)))


def _assert_min_result(
    result,
    *,
    expect_symbolic: bool,
    expected_value=None,
) -> None:
    if expect_symbolic:
        assert isinstance(result, pypto.SymbolicScalar), (
            f"Expected SymbolicScalar, got {type(result).__name__!r}. "
            "min() over a dynamic operand must be intercepted by the framework."
        )
    else:
        assert not isinstance(result, pypto.SymbolicScalar), (
            "Expected a plain Python value, got SymbolicScalar. "
            "min() over non-dynamic operands must use the Python builtin."
        )

    if expected_value is not None:
        assert result == expected_value, (
            f"Value mismatch: expected {expected_value!r}, got {result!r}"
        )


def test_min_dispatch_expr_evaluator_builtin_cases():
    _assert_min_result(
        _eval_expr("min(['bbb', 'a', 'cc'], key=len)"),
        expect_symbolic=False,
        expected_value="a",
    )
    _assert_min_result(
        _eval_expr("min([10, 3, 7])"),
        expect_symbolic=False,
        expected_value=3,
    )
    _assert_min_result(
        _eval_expr("min(8, 16)"),
        expect_symbolic=False,
        expected_value=8,
    )
    _assert_min_result(
        _eval_expr("min([], key=len, default='fallback')"),
        expect_symbolic=False,
        expected_value="fallback",
    )
    _assert_min_result(
        _eval_expr("min(static_dim, 16)", static_dim=8),
        expect_symbolic=False,
        expected_value=8,
    )
    _assert_min_result(
        _eval_expr("min([static_dim, 64, 1])", static_dim=8),
        expect_symbolic=False,
        expected_value=1,
    )
    concrete_m = pypto.symbolic_scalar(10)
    concrete_n = pypto.symbolic_scalar(20)
    _assert_min_result(
        _eval_expr(
            "min(concrete_m, concrete_n)",
            concrete_m=concrete_m,
            concrete_n=concrete_n,
        ),
        expect_symbolic=False,
        expected_value=10,
    )


def test_min_dispatch_expr_evaluator_symbolic_cases():
    dynamic_dim = pypto.SymbolicScalar("n")

    _assert_min_result(
        _eval_expr("min(dynamic_dim, 16)", dynamic_dim=dynamic_dim),
        expect_symbolic=True,
    )
    _assert_min_result(
        _eval_expr("min(min(dynamic_dim, 16), 32)", dynamic_dim=dynamic_dim),
        expect_symbolic=True,
    )
    _assert_min_result(
        _eval_expr("min(symbolic_value, 16)", symbolic_value=pypto.SymbolicScalar("m")),
        expect_symbolic=True,
    )


if __name__ == "__main__":
    test_min_dispatch_expr_evaluator_builtin_cases()
    test_min_dispatch_expr_evaluator_symbolic_cases()
