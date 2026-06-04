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
"""
import pypto


def test_init_symbolic_scalar_value_arg():
    expected_value = 123
    scalar = pypto.symbolic_scalar(expected_value)

    assert scalar.is_concrete() == True
    assert scalar.concrete() == expected_value

    scalar = pypto.symbolic_scalar(scalar)
    assert scalar.is_concrete() == True
    assert scalar.concrete() == expected_value


def test_init_symbolic_scalar_name_value_args():
    expected_value = 123
    scalar = pypto.symbolic_scalar("scalar", expected_value)

    assert scalar.is_concrete() == True
    assert scalar.concrete() == expected_value


def test_symbolic_scalar_dump():
    scalar = pypto.symbolic_scalar(10)
    assert str(scalar) == "10"
    assert int(scalar) == 10


def test_symbolic_scalar_prop():
    scalar = pypto.symbolic_scalar(10)
    assert scalar.is_symbol() == False
    assert scalar.is_expression() == False
    assert scalar.is_immediate() == True
    assert scalar.is_concrete() == True
    assert scalar.concrete() == 10

    scalar2 = pypto.symbolic_scalar("s")
    assert scalar2.is_symbol() == True
    assert scalar2.is_expression() == False
    assert scalar.is_immediate() == True
    assert scalar2.is_concrete() == False

    scalar3 = scalar < 2
    assert isinstance(scalar3, pypto.symbolic_scalar)
    assert scalar3.is_symbol() == False
    assert scalar3.is_expression() == False
    assert scalar3.is_immediate() == True
    assert scalar3.is_concrete() == True
    assert scalar3.concrete() == 0

    scalar4 = scalar2 < 2
    assert isinstance(scalar4, pypto.symbolic_scalar)
    assert scalar4.is_symbol() == False
    assert scalar4.is_expression() == True
    assert scalar4.is_immediate() == False
    assert scalar4.is_concrete() == False


def test_symbolic_scalar_uniop():
    scalar = pypto.symbolic_scalar(10)
    pos_s = +scalar
    neg_s = -scalar
    not_s = ~scalar
    assert isinstance(pos_s, pypto.symbolic_scalar)
    assert isinstance(neg_s, pypto.symbolic_scalar)
    assert isinstance(not_s, pypto.symbolic_scalar)
    assert scalar.concrete() == 10
    assert pos_s.concrete() == 10
    assert neg_s.concrete() == -10
    assert not_s.concrete() == 0


def test_binary_ops():
    c = 10
    x = pypto.symbolic_scalar(10)
    y = pypto.symbolic_scalar('y')
    z = pypto.symbolic_scalar(20)

    tests = [
        (x + y, c + y, x + z, c + z, 30),
        (y + x, y + c, z + x, z + c, 30),
        (y - x, y - c, z - x, z - c, 10),
        (x - y, c - y, x - z, c - z, -10),
        (y * x, y * c, z * x, c * z, 200),
        (x * y, c * y, x * z, z * c, 200),
        (y / x, y / c, z / x, z / c, 2.0),
        (x / y, c / y, x / z, c / z, 0),  # always floordiv
        (y // x, y // c, z // x, z // c, 2.0),
        (x // y, c // y, x // z, c // z, 0),
        (y % x, y % c, z % x, z % c, 0),
        (x % y, c % y, x % z, c % z, 10),
        (y > x, y > c, z > x, z > c, True),
        (x > y, c > y, x > z, c > z, False),
        (y >= x, y >= c, z >= x, z >= c, True),
        (x >= y, c >= y, x >= z, c >= z, False),
        (y < x, y < c, z < x, z < c, False),
        (x < y, c < y, x < z, c < z, True),
        (y <= x, y <= c, z <= x, z <= c, False),
        (x <= y, c <= y, x <= z, c <= z, True),
        (y == x, y == c, z == x, z == c, False),
        (x == y, c == y, x == z, c == z, False),
        (x != y, c != y, z != x, z != c, True),
        (y != x, y != c, x != z, c != z, True),
    ]

    for (expr, expr1, expr2, expr3, val) in tests:
        assert isinstance(expr, pypto.symbolic_scalar)
        assert expr.is_symbol() == False
        assert expr.is_expression() == True
        assert expr.is_immediate() == False
        assert expr.is_concrete() == False

        assert isinstance(expr1, pypto.symbolic_scalar)
        assert expr1.is_symbol() == False
        assert expr1.is_expression() == True
        assert expr1.is_immediate() == False
        assert expr1.is_concrete() == False

        assert isinstance(expr2, pypto.symbolic_scalar)
        assert expr2.concrete() == val
        assert expr2.is_symbol() == False
        assert expr2.is_expression() == False
        assert expr2.is_immediate() == True
        assert expr2.is_concrete() == True

        assert isinstance(expr3, pypto.symbolic_scalar)
        assert expr3.concrete() == val
        assert expr3.is_symbol() == False
        assert expr3.is_expression() == False
        assert expr3.is_immediate() == True
        assert expr3.is_concrete() == True


def test_symbolic_scalar_add():
    ten = pypto.symbolic_scalar("10", 10)
    twenty = pypto.symbolic_scalar("20", 20)
    thirty = pypto.symbolic_scalar("30", 30)

    assert ten + twenty == thirty


def test_symbolic_scalar_sub():
    ten = pypto.symbolic_scalar("10", 10)
    twenty = pypto.symbolic_scalar("20", 20)
    thirty = pypto.symbolic_scalar("30", 30)

    assert twenty == thirty - ten


def test_symbolic_scalar_mul():
    ten = pypto.symbolic_scalar("10", 10)
    two = pypto.symbolic_scalar("2", 3)
    assert ten * two == pypto.symbolic_scalar("20", 30)


def test_symbolic_scalar_div():
    ten = pypto.symbolic_scalar("10", 10)
    twenty = pypto.symbolic_scalar("20", 20)
    assert (twenty / ten) == pypto.symbolic_scalar("2", 2)


def test_symbolic_scalar_mod():
    one = pypto.symbolic_scalar("one", 1)
    scalar = pypto.symbolic_scalar("31", 31)
    two = pypto.symbolic_scalar("2", 3)
    assert scalar % two == one


def test_symbolic_scalar_binop():
    a = pypto.symbolic_scalar(6)
    b = pypto.symbolic_scalar(4)
    c = a + b
    d = a - b
    e = a * b
    f = a / b
    f_floor = a // b
    g = a % b
    h = a.max(b)
    i = a.min(b)

    for op in [c, d, e, f, f_floor, g, h, i]:
        assert isinstance(op, pypto.symbolic_scalar)
    assert c.concrete() == 10
    assert d.concrete() == 2
    assert e.concrete() == 24
    assert f.concrete() == 1
    assert f_floor.concrete() == 1
    assert g.concrete() == 2
    assert h.concrete() == 6
    assert i.concrete() == 4


def test_symbolic_scalar_binop_with_int():
    a = pypto.symbolic_scalar(6)
    b = 4
    c = a + b
    d = a - b
    e = a * b
    f = a / b
    f_floor = a // b
    g = a % b

    h = b + a
    i = b - a
    j = b * a
    k = b / a
    k_floor = b // a
    l = b % a

    for op in [c, d, e, f, f_floor, g, h, i, j, k, k_floor, l]:
        assert isinstance(op, pypto.symbolic_scalar)
    assert c.concrete() == 10
    assert d.concrete() == 2
    assert e.concrete() == 24
    assert f.concrete() == 1
    assert f_floor.concrete() == 1
    assert g.concrete() == 2
    assert h.concrete() == 10
    assert i.concrete() == -2
    assert j.concrete() == 24
    assert k.concrete() == 0
    assert k_floor.concrete() == 0
    assert l.concrete() == 4


def test_symbolic_scalar_le():
    ten = pypto.symbolic_scalar("scalar", 10)
    twenty = pypto.symbolic_scalar("scalar", 20)

    assert ten <= twenty


def test_symbolic_scalar_lt():
    ten = pypto.symbolic_scalar("scalar", 10)
    twenty = pypto.symbolic_scalar("scalar", 20)

    assert ten < twenty


def test_symbolic_scalar_gt():
    ten = pypto.symbolic_scalar("scalar", 10)
    twenty = pypto.symbolic_scalar("scalar", 20)

    assert twenty > ten


def test_symbolic_scalar_ge():
    ten = pypto.symbolic_scalar("scalar", 10)
    twenty = pypto.symbolic_scalar("scalar", 20)

    assert twenty >= ten


def test_symbolic_scalar_ne():
    ten = pypto.symbolic_scalar("scalar", 10)
    twenty = pypto.symbolic_scalar("scalar", 20)

    assert twenty != ten


def test_symbolic_scalar_eq():
    ten = pypto.symbolic_scalar("scalar", 10)
    another_ten = pypto.symbolic_scalar("scalar", 10)

    assert ten == another_ten


def test_symbolic_scalar_comp_op():
    a = pypto.symbolic_scalar(6)
    b = pypto.symbolic_scalar(4)
    assert (a == b).concrete() == 0
    assert (a != b).concrete() == 1
    assert (a < b).concrete() == 0
    assert (a <= b).concrete() == 0
    assert (a > b).concrete() == 1
    assert (a >= b).concrete() == 1

    assert (a == 6).concrete() == 1
    assert (a != 6).concrete() == 0
    assert (a < 6).concrete() == 0
    assert (a <= 6).concrete() == 1
    assert (a > 6).concrete() == 0
    assert (a >= 6).concrete() == 1

    assert (6 == a).concrete() == 1
    assert (6 != a).concrete() == 0
    assert (6 < a).concrete() == 0
    assert (6 <= a).concrete() == 1
    assert (6 > a).concrete() == 0
    assert (6 >= a).concrete() == 1


def test_symbolic_scalar_issue36():
    b = pypto.symbolic_scalar('b')
    a = (b >= 2) * (b < 8)
    assert str(a) == '((b>=2)*(b<8))'


def test_as_expr():
    b = pypto.symbolic_scalar('b').as_expr()
    assert str(b) == 'b'


test_as_expr()
