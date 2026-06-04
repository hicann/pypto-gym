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
"""
Unit tests for SymbolicScalar.simplify() algebraic simplification.
"""
import pypto


def _sym(name):
    return pypto.symbolic_scalar(name)


def _val(v):
    return pypto.symbolic_scalar(v)


# ============================================================================
# Unary rules
# ============================================================================

def test_neg_neg():
    x = _sym("x")
    result = (-(-x)).simplify()
    assert str(result) == "x"


def test_neg_sub():
    x = _sym("x")
    y = _sym("y")
    result = (-(x - y)).simplify()
    assert str(result) == "(y-x)"


def test_pos_idempotent():
    x = _sym("x")
    result = (+x).simplify()
    assert str(result) == "x"


def test_not_not():
    x = _sym("x")
    result = (~(~x)).simplify()
    assert str(result) == "x"


def test_not_lt():
    x = _sym("x")
    y = _sym("y")
    result = (~(x < y)).simplify()
    assert str(result) == "(y<=x)"


def test_not_le():
    x = _sym("x")
    y = _sym("y")
    result = (~(x <= y)).simplify()
    assert str(result) == "(y<x)"


def test_not_eq():
    x = _sym("x")
    y = _sym("y")
    result = (~(x == y)).simplify()
    assert str(result) == "RUNTIME_Ne(x, y)"


def test_not_ne():
    x = _sym("x")
    y = _sym("y")
    result = (~(x != y)).simplify()
    assert str(result) == "RUNTIME_Eq(x, y)"


# ============================================================================
# Add rules
# ============================================================================

def test_add_const_reassociate():
    x = _sym("x")
    result = ((x + 3) + 5).simplify()
    assert str(result) == "(x+8)"


def test_add_cancel_sub():
    x = _sym("x")
    y = _sym("y")
    result = ((x - y) + y).simplify()
    assert str(result) == "x"


def test_add_cancel_sub_reverse():
    x = _sym("x")
    y = _sym("y")
    result = (x + (y - x)).simplify()
    assert str(result) == "y"


def test_add_self_to_mul():
    x = _sym("x")
    result = (x + x).simplify()
    assert str(result) == "(x*2)"


def test_add_const_canonicalize():
    x = _sym("x")
    result = (5 + x).simplify()
    assert str(result) == "(x+5)"


# ============================================================================
# Sub rules
# ============================================================================

def test_sub_self():
    x = _sym("x")
    result = (x - x).simplify()
    assert str(result) == "0"


def test_sub_cancel_add():
    x = _sym("x")
    y = _sym("y")
    result = ((x + y) - y).simplify()
    assert str(result) == "x"


def test_sub_cancel_add_reverse():
    x = _sym("x")
    y = _sym("y")
    result = ((x + y) - x).simplify()
    assert str(result) == "y"


def test_sub_cross_cancellation():
    x = _sym("x")
    y = _sym("y")
    z = _sym("z")
    result = ((x - y) - (x - z)).simplify()
    assert str(result) == "(z-y)"


def test_sub_const_reassociate():
    x = _sym("x")
    result = ((x + 5) - 3).simplify()
    assert str(result) == "(x+2)"


def test_sub_add_canonicalize():
    x = _sym("x")
    y = _sym("y")
    result = ((x + 3) - y).simplify()
    assert str(result) == "((x-y)+3)"


# ============================================================================
# Mul rules
# ============================================================================

def test_mul_const_associativity():
    x = _sym("x")
    result = ((x * 3) * 5).simplify()
    assert str(result) == "(x*15)"


def test_mul_const_canonicalize():
    x = _sym("x")
    result = (3 * x).simplify()
    assert str(result) == "(x*3)"


def test_mul_min_max():
    x = _sym("x")
    y = _sym("y")
    expr = x.min(y) * x.max(y)
    result = expr.simplify()
    assert str(result) == "(x*y)"


# ============================================================================
# Div rules
# ============================================================================

def test_div_mul_const():
    x = _sym("x")
    result = ((x * 6) / 3).simplify()
    assert str(result) == "(x*2)"


def test_div_nested():
    x = _sym("x")
    result = (((x / 4) / 2)).simplify()
    assert str(result) == "(x/8)"


# ============================================================================
# Mod rules
# ============================================================================

def test_mod_mul_const():
    x = _sym("x")
    result = ((x * 6) % 3).simplify()
    assert str(result) == "0"


def test_mod_add_const():
    x = _sym("x")
    result = ((x + 6) % 3).simplify()
    assert str(result) == "(x%3)"


# ============================================================================
# Min rules
# ============================================================================

def test_min_self():
    x = _sym("x")
    result = x.min(x).simplify()
    assert str(result) == "x"


def test_min_common_add():
    x = _sym("x")
    y = _sym("y")
    z = _sym("z")
    result = (x + y).min(x + z).simplify()
    assert str(result) == "(x+RUNTIME_Min(y, z))"


def test_min_absorption():
    x = _sym("x")
    y = _sym("y")
    result = x.max(y).min(y).simplify()
    assert str(result) == "y"


def test_min_const():
    x = _sym("x")
    result = x.min(5).simplify()
    # constants should be on the right
    assert str(result) == "RUNTIME_Min(x, 5)"


# ============================================================================
# Max rules
# ============================================================================

def test_max_self():
    x = _sym("x")
    result = x.max(x).simplify()
    assert str(result) == "x"


def test_max_common_add():
    x = _sym("x")
    y = _sym("y")
    z = _sym("z")
    result = (x + y).max(x + z).simplify()
    assert str(result) == "(x+RUNTIME_Max(y, z))"


def test_max_absorption():
    x = _sym("x")
    y = _sym("y")
    result = x.min(y).max(y).simplify()
    assert str(result) == "y"


# ============================================================================
# Comparison rules
# ============================================================================

def test_eq_self():
    x = _sym("x")
    result = (x == x).simplify()
    assert str(result) == "1"


def test_ne_self():
    x = _sym("x")
    result = (x != x).simplify()
    assert str(result) == "0"


def test_lt_self():
    x = _sym("x")
    result = (x < x).simplify()
    assert str(result) == "0"


def test_le_self():
    x = _sym("x")
    result = (x <= x).simplify()
    assert str(result) == "1"


def test_gt_delegates_lt():
    x = _sym("x")
    result = (x > x).simplify()
    assert str(result) == "0"


def test_ge_delegates_le():
    x = _sym("x")
    result = (x >= x).simplify()
    assert str(result) == "1"


def test_eq_cancel_add():
    x = _sym("x")
    y = _sym("y")
    z = _sym("z")
    result = (x + y == x + z).simplify()
    assert str(result) == "RUNTIME_Eq(y, z)"


def test_ne_cancel_add():
    x = _sym("x")
    y = _sym("y")
    z = _sym("z")
    result = (x + y != x + z).simplify()
    assert str(result) == "RUNTIME_Ne(y, z)"


def test_lt_cancel_add():
    x = _sym("x")
    y = _sym("y")
    z = _sym("z")
    result = (x + y < x + z).simplify()
    assert str(result) == "(y<z)"


def test_le_cancel_add():
    x = _sym("x")
    y = _sym("y")
    z = _sym("z")
    result = (x + y <= x + z).simplify()
    assert str(result) == "(y<=z)"


# ============================================================================
# Mixed/complex simplification
# ============================================================================

def test_simplify_preserves_immediate():
    x = _val(10)
    result = x.simplify()
    assert result.is_immediate()
    assert result.concrete() == 10


def test_simplify_preserves_symbol():
    x = _sym("x")
    result = x.simplify()
    assert result.is_symbol()
    assert str(result) == "x"


def test_nested_simplify():
    x = _sym("x")
    y = _sym("y")
    # (x - y) + y => x
    result = ((x - y) + y).simplify()
    assert str(result) == "x"


def test_complex_cancellation():
    x = _sym("x")
    y = _sym("y")
    z = _sym("z")
    # x*y + x*z => (y+z) * x
    result = (x * y + x * z).simplify()
    assert str(result) == "((y+z)*x)"
