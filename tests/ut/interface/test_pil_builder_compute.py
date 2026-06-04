#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 CANN community contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from test_pil_builder_utils import TestParser, Expr


def test_pil_builder_boolop():

    with TestParser():

        @TestParser.test
        def true_and_true_and_true():
            if Expr.true(0) and Expr.true(1) and Expr.true(2):
                Expr.str(3)
            else:
                Expr.str(4)

        @TestParser.test
        def true_and_false_and_true():
            if Expr.true(0) and Expr.false(1) and Expr.true(2):
                Expr.str(3)
            else:
                Expr.str(4)

        @TestParser.test
        def false_and_true_and_true():
            if Expr.false(0) and Expr.true(1) and Expr.true(2):
                Expr.str(3)
            else:
                Expr.str(4)

        @TestParser.test
        def false_or_false_or_false():
            if Expr.true(0) and Expr.true(1) and Expr.true(2):
                Expr.str(3)
            else:
                Expr.str(4)

        @TestParser.test
        def false_or_true_or_false():
            if Expr.true(0) and Expr.false(1) and Expr.true(2):
                Expr.str(3)
            else:
                Expr.str(4)

        @TestParser.test
        def true_or_false_or_false():
            if Expr.false(0) and Expr.true(1) and Expr.true(2):
                Expr.str(3)
            else:
                Expr.str(4)


def test_pil_builder_ifexp():

    with TestParser():

        @TestParser.test
        def true_test_body_selected():
            var_x = Expr.str(0) if Expr.true(1) else Expr.str(2)

        @TestParser.test
        def false_test_orelse_selected():
            var_x = Expr.str(0) if Expr.false(1) else Expr.str(2)

        @TestParser.test
        def nested_ifexp_in_body():
            var_x = (Expr.str(0) if Expr.true(1) else Expr.str(2)) if Expr.true(3) else Expr.str(4)

        @TestParser.test
        def nested_ifexp_in_orelse():
            var_x = Expr.str(0) if Expr.false(1) else (Expr.str(2) if Expr.true(3) else Expr.str(4))

        @TestParser.test
        def nested_ifexp_in_orelse_false():
            var_x = Expr.str(0) if Expr.false(1) else (Expr.str(2) if Expr.false(3) else Expr.str(4))

        @TestParser.test
        def ifexp_as_if_test():
            if Expr.str(0) if Expr.true(1) else Expr.str(2):
                Expr.str(3)
            else:
                Expr.str(4)


def test_pil_builder_bin_op():

    with TestParser():

        @TestParser.test
        def add():
            var_x = Expr.int(0) + Expr.int(1) * Expr.int(2)
            var_y = Expr.int(0) - Expr.int(1) // Expr.int(2)


def test_pil_builder_unary_op():

    with TestParser():

        @TestParser.test
        def add():
            var_x = -Expr.int(0) + Expr.int(1) * Expr.int(2)
            var_y = -Expr.int(0) - Expr.int(1) // Expr.int(2)


def test_pil_builder_compare():

    with TestParser():

        # ================================================================
        # Part 1: single-op comparisons with each operator
        # ================================================================

        @TestParser.test
        def cmp_lt_true():
            var_x = Expr.int(1) < Expr.int(2)
            Expr.str(var_x)

        @TestParser.test
        def cmp_lt_false():
            var_x = Expr.int(2) < Expr.int(1)
            Expr.str(var_x)

        @TestParser.test
        def cmp_lte():
            var_x = Expr.int(1) <= Expr.int(1)
            Expr.str(var_x)

        @TestParser.test
        def cmp_gt():
            var_x = Expr.int(2) > Expr.int(1)
            Expr.str(var_x)

        @TestParser.test
        def cmp_gte():
            var_x = Expr.int(2) >= Expr.int(2)
            Expr.str(var_x)

        @TestParser.test
        def cmp_eq_true():
            var_x = Expr.int(1) == Expr.int(1)
            Expr.str(var_x)

        @TestParser.test
        def cmp_eq_false():
            var_x = Expr.int(1) == Expr.int(2)
            Expr.str(var_x)

        @TestParser.test
        def cmp_neq():
            var_x = Expr.int(1) != Expr.int(2)
            Expr.str(var_x)

        @TestParser.test
        def cmp_is():
            var_a = Expr.int(0)
            var_x = var_a is var_a
            Expr.str(var_x)

        @TestParser.test
        def cmp_is_not():
            var_a = Expr.int(0)
            var_b = Expr.int(1)
            var_x = var_a is not var_b
            Expr.str(var_x)

        @TestParser.test
        def cmp_in():
            var_l = [Expr.int(0), Expr.int(1)]
            var_x = Expr.int(0) in var_l
            Expr.str(var_x)

        @TestParser.test
        def cmp_not_in():
            var_l = [Expr.int(0), Expr.int(1)]
            var_x = Expr.int(2) not in var_l
            Expr.str(var_x)

        # ================================================================
        # Part 2: chained comparisons (PIL short-circuits via if)
        # ================================================================

        @TestParser.test
        def cmp_chain_lt_lt_all_true():
            # e.g. 1 < 2 < 3 — both sub-comparisons true, b evaluated once
            var_x = Expr.int(1) < Expr.int(2) < Expr.int(3)
            Expr.str(var_x)

        @TestParser.test
        def cmp_chain_lt_lt_first_false():
            # e.g. 3 < 2 < 4 — first false, third operand not evaluated
            var_x = Expr.int(3) < Expr.int(2) < Expr.int(4)
            Expr.str(var_x)

        @TestParser.test
        def cmp_chain_lt_eq():
            # e.g. 1 < 2 == 2
            var_x = Expr.int(1) < Expr.int(2) == Expr.int(2)
            Expr.str(var_x)

        @TestParser.test
        def cmp_chain_three_ops():
            # e.g. 1 < 2 <= 3 < 4
            var_x = Expr.int(1) < Expr.int(2) <= Expr.int(3) < Expr.int(4)
            Expr.str(var_x)

        # ================================================================
        # Part 3: compare result used in various expression contexts
        # ================================================================

        # --- compare result in binop ---

        @TestParser.test
        def cmp_in_binop():
            var_x = (Expr.int(1) < Expr.int(2)) + 0
            Expr.str(var_x)

        # --- compare result in unary op ---

        @TestParser.test
        def cmp_in_unary():
            var_x = not (Expr.int(1) == Expr.int(2))
            Expr.str(var_x)

        # --- compare result as if test ---

        @TestParser.test
        def cmp_if_true():
            if Expr.int(1) < Expr.int(2):
                Expr.str(0)
            else:
                Expr.str(1)

        @TestParser.test
        def cmp_if_false():
            if Expr.int(2) < Expr.int(1):
                Expr.str(0)
            else:
                Expr.str(1)

        # --- compare result as for iter ---

        @TestParser.test
        def cmp_for_iter():
            for var_x in [Expr.int(0) < Expr.int(1), Expr.int(2) < Expr.int(1)]:
                Expr.str(var_x)

        # --- compare result as while test ---

        @TestParser.test
        def cmp_while_test():
            var_n = [0]
            while var_n[0] < 3:
                Expr.str(var_n[0])
                var_n[0] = var_n[0] + 1

        # --- compare result as call positional arg ---

        @TestParser.test
        def cmp_call_pos_arg():
            Expr.str(Expr.int(1) < Expr.int(2))

        # --- compare result as call keyword arg ---

        @TestParser.test
        def cmp_call_kw_arg():

            def func(x):
                Expr.str(x)
            func(x=Expr.int(1) == Expr.int(1))

        # --- compare result in tuple literal ---

        @TestParser.test
        def cmp_in_tuple():
            var_t = (Expr.int(1) < Expr.int(2), Expr.int(3) > Expr.int(4))
            Expr.str(var_t[0])
            Expr.str(var_t[1])

        # --- compare result in list literal ---

        @TestParser.test
        def cmp_in_list():
            var_l = [Expr.int(1) < Expr.int(2), Expr.int(3) > Expr.int(4)]
            Expr.str(var_l[0])
            Expr.str(var_l[1])

        # --- compare result as dict key and value ---

        @TestParser.test
        def cmp_dict_value():
            var_d = {0: Expr.int(1) < Expr.int(2)}
            Expr.str(var_d[0])

        @TestParser.test
        def cmp_dict_key():
            var_d = {Expr.int(1) == Expr.int(1): Expr.int(0)}
            Expr.str(var_d[True])

        # --- compare result in set literal ---

        @TestParser.test
        def cmp_in_set():
            var_s = {Expr.int(1) < Expr.int(2), Expr.int(3) > Expr.int(4)}
            Expr.str(True in var_s)

        # --- compare result as subscript index ---

        @TestParser.test
        def cmp_as_subscript_index():
            var_arr = Expr(0)
            var_arr[True] = Expr.int(99)
            var_x = var_arr[Expr.int(1) == Expr.int(1)]
            Expr.str(var_x)

        # --- compare result as slice bound ---

        @TestParser.test
        def cmp_as_slice_bound():
            var_l = [Expr.int(0), Expr.int(1), Expr.int(2)]
            # True == 1, so slice [True:3] == [1:3]
            var_s = var_l[Expr.int(1) == Expr.int(1):]
            Expr.str(var_s[0])

        # --- compare result as type annotation ---

        @TestParser.test
        def cmp_annotation_with_value():
            var_x: bool = Expr.int(1) < Expr.int(2)
            Expr.str(var_x)
