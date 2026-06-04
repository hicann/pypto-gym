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


def test_pil_builder_for():

    with TestParser():

        # --- simple name target, no orelse ---

        @TestParser.test
        def for_simple_name():
            for var_x in [Expr.int(0), Expr.int(1), Expr.int(2)]:
                Expr.str(var_x)

        # --- with orelse (loop exhausts normally → else fires) ---

        @TestParser.test
        def for_with_orelse():
            for var_x in [Expr.int(0), Expr.int(1)]:
                Expr.str(var_x)
            else:
                Expr.str(99)

        # --- tuple unpack target ---

        @TestParser.test
        def for_tuple_unpack_target():
            for var_x, var_y in [(Expr.int(0), Expr.int(1)), (Expr.int(2), Expr.int(3))]:
                Expr.str(var_x)
                Expr.str(var_y)

        @TestParser.test
        def for_tuple_unpack_target_with_orelse():
            for var_x, var_y in [(Expr.int(0), Expr.int(1))]:
                Expr.str(var_x)
                Expr.str(var_y)
            else:
                Expr.str(99)

        # --- attribute target ---

        @TestParser.test
        def for_attribute_target():
            var_obj = Expr(0)
            var_obj.val = Expr.int(-1)
            for var_obj.val in [Expr.int(0), Expr.int(1)]:
                Expr.str(var_obj.val)

        # --- subscript target ---

        @TestParser.test
        def for_subscript_target():
            var_obj = Expr(0)
            var_obj[0] = Expr.int(-1)
            for var_obj[0] in [Expr.int(0), Expr.int(1)]:
                Expr.str(var_obj[0])

        # --- call iter: iter expression is a function call ---

        @TestParser.test
        def for_call_iter():
            for var_x in range(Expr.int(3)):
                Expr.str(var_x)

        @TestParser.test
        def for_call_iter_with_orelse():
            for var_x in range(Expr.int(2)):
                Expr.str(var_x)
            else:
                Expr.str(99)

        # --- nested for ---

        @TestParser.test
        def for_nested():
            for var_i in [Expr.int(0), Expr.int(1)]:
                for var_j in [Expr.int(0), Expr.int(1)]:
                    Expr.str(var_i)
                    Expr.str(var_j)

        @TestParser.test
        def for_nested_outer_orelse():
            for var_i in [Expr.int(0), Expr.int(1)]:
                for var_j in [Expr.int(0)]:
                    Expr.str(var_i)
                    Expr.str(var_j)
            else:
                Expr.str(99)

        @TestParser.test
        def for_nested_both_orelse():
            for var_i in [Expr.int(0), Expr.int(1)]:
                for var_j in [Expr.int(0)]:
                    Expr.str(var_i)
                    Expr.str(var_j)
                else:
                    Expr.str(88)
            else:
                Expr.str(99)

        # --- 3-level nesting: tuple unpack inner, call iter outer ---

        @TestParser.test
        def for_three_level_nesting():
            for var_i in range(Expr.int(2)):
                for var_j in [Expr.int(0), Expr.int(1)]:
                    for var_x, var_y in [(Expr.int(0), Expr.int(1))]:
                        Expr.str(var_i)
                        Expr.str(var_j)
                        Expr.str(var_x)
                        Expr.str(var_y)


def test_pil_builder_while():

    with TestParser():

        # --- function condition, loop never enters ---

        @TestParser.test
        def while_false_no_body():
            while Expr.false(0):
                Expr.str(1)

        @TestParser.test
        def while_false_with_orelse():
            # orelse fires on natural exit (condition was False from the start)
            while Expr.false(0):
                Expr.str(1)
            else:
                Expr.str(2)

        # --- function condition, body + break (condition always True) ---

        @TestParser.test
        def while_true_break():
            while Expr.true(0):
                Expr.str(1)
                break

        @TestParser.test
        def while_true_break_orelse_not_fired():
            # break suppresses orelse
            while Expr.true(0):
                Expr.str(1)
                break
            else:
                Expr.str(99)

        @TestParser.test
        def while_true_body_then_break():
            while Expr.true(0):
                Expr.str(1)
                Expr.str(2)
                break

        # --- constant condition (variable holding True), exits via break ---
        # Note: bare `while True:` is not used because PIL requires the test
        # expression to produce a string identifier (not a raw Constant node).

        @TestParser.test
        def while_const_var_break():
            while 1:
                Expr.str(0)
                break

        @TestParser.test
        def while_const_var_break_orelse_not_fired():
            while 1:
                Expr.str(0)
                break
            else:
                Expr.str(99)

        @TestParser.test
        def while_const_var_multi_body_break():
            while 1:
                Expr.str(0)
                Expr.str(1)
                Expr.str(2)
                break

        # --- nested: function-cond outer, function-cond inner ---

        @TestParser.test
        def while_nested_func_func():
            while Expr.true(0):
                while Expr.true(1):
                    Expr.str(2)
                    break
                Expr.str(3)
                break

        @TestParser.test
        def while_nested_func_func_inner_orelse():
            while Expr.true(0):
                while Expr.false(1):
                    Expr.str(2)
                else:
                    Expr.str(3)
                break

        # --- nested: constant-cond outer, function-cond inner ---

        @TestParser.test
        def while_nested_const_outer_func_inner():
            while 1:
                while Expr.true(0):
                    Expr.str(1)
                    break
                break

        @TestParser.test
        def while_nested_const_outer_false_inner_orelse():
            while True:
                while Expr.false(0):
                    Expr.str(1)
                else:
                    Expr.str(2)
                break

        # --- 3-level nesting ---

        @TestParser.test
        def while_three_level_nesting():
            while 1:
                while Expr.true(0):
                    while Expr.true(1):
                        Expr.str(2)
                        break
                    Expr.str(3)
                    break
                break

        @TestParser.test
        def while_three_level_mixed_orelse():
            var_outer = True
            while var_outer:
                while Expr.true(0):
                    while Expr.false(1):
                        Expr.str(2)
                    else:
                        Expr.str(3)
                    break
                break


def test_pil_builder_if():

    with TestParser():

        @TestParser.test
        def simple_if_true():
            if Expr.true(0):
                Expr.str(1)

        @TestParser.test
        def simple_if_false():
            if Expr.false(0):
                Expr.str(1)

        @TestParser.test
        def if_else_true():
            if Expr.true(0):
                Expr.str(1)
            else:
                Expr.str(2)

        @TestParser.test
        def if_else_false():
            if Expr.false(0):
                Expr.str(1)
            else:
                Expr.str(2)

        @TestParser.test
        def if_elif_else_first():
            if Expr.true(0):
                Expr.str(1)
            elif Expr.true(2):
                Expr.str(3)
            else:
                Expr.str(4)

        @TestParser.test
        def if_elif_else_second():
            if Expr.false(0):
                Expr.str(1)
            elif Expr.true(2):
                Expr.str(3)
            else:
                Expr.str(4)

        @TestParser.test
        def if_elif_else_third():
            if Expr.false(0):
                Expr.str(1)
            elif Expr.false(2):
                Expr.str(3)
            else:
                Expr.str(4)

        @TestParser.test
        def nested_if_true_true():
            if Expr.true(0):
                if Expr.true(1):
                    Expr.str(2)
                else:
                    Expr.str(3)
            else:
                Expr.str(4)

        @TestParser.test
        def nested_if_true_false():
            if Expr.true(0):
                if Expr.false(1):
                    Expr.str(2)
                else:
                    Expr.str(3)
            else:
                Expr.str(4)

        @TestParser.test
        def nested_if_false():
            if Expr.false(0):
                if Expr.true(1):
                    Expr.str(2)
                else:
                    Expr.str(3)
            else:
                Expr.str(4)

        @TestParser.test
        def if_and_condition_both_true():
            if Expr.true(0) and Expr.true(1):
                Expr.str(2)
            else:
                Expr.str(3)

        @TestParser.test
        def if_and_condition_first_false():
            if Expr.false(0) and True:
                Expr.str(2)
            else:
                Expr.str(3)

        @TestParser.test
        def if_and_condition_second_false():
            if Expr.true(0) and False:
                Expr.str(2)
            else:
                Expr.str(3)

        @TestParser.test
        def if_or_condition_both_false():
            if False or Expr.false(1):
                Expr.str(2)
            else:
                Expr.str(3)

        @TestParser.test
        def if_or_condition_first_true():
            if Expr.true(0) or Expr.false(1):
                Expr.str(2)
            else:
                Expr.str(3)

        @TestParser.test
        def if_or_condition_second_true():
            if Expr.false(0) or Expr.true(1):
                Expr.str(2)
            else:
                Expr.str(3)

        @TestParser.test
        def if_and_or_combined():
            if Expr.true(0) and Expr.false(1) or Expr.true(2):
                Expr.str(3)
            else:
                Expr.str(4)

        @TestParser.test
        def sequential_ifs():
            if Expr.true(0):
                Expr.str(1)
            if Expr.false(2):
                Expr.str(3)
            if Expr.true(4):
                Expr.str(5)


def test_pil_builder_break():

    with TestParser():

        # --- break in for, first iteration ---

        @TestParser.test
        def break_for_first():
            for var_x in [Expr.int(0), Expr.int(1), Expr.int(2)]:
                Expr.str(var_x)
                break

        # --- break in for, conditional ---

        @TestParser.test
        def break_for_conditional():
            for var_x in [Expr.int(0), Expr.int(1), Expr.int(2)]:
                Expr.str(var_x)
                if Expr.true(var_x):
                    break

        # --- break in for suppresses orelse ---

        @TestParser.test
        def break_for_suppresses_orelse():
            for var_x in [Expr.int(0), Expr.int(1)]:
                Expr.str(var_x)
                break
            else:
                Expr.str(99)

        # --- break in nested for, inner only ---

        @TestParser.test
        def break_for_nested_inner():
            for var_i in [Expr.int(0), Expr.int(1)]:
                for var_j in [Expr.int(0), Expr.int(1), Expr.int(2)]:
                    Expr.str(var_j)
                    break
                Expr.str(var_i)

        # --- break in nested for, outer ---

        @TestParser.test
        def break_for_nested_outer():
            for var_i in [Expr.int(0), Expr.int(1)]:
                for var_j in [Expr.int(0), Expr.int(1)]:
                    Expr.str(var_j)
                Expr.str(var_i)
                break

        # --- break in while, constant condition ---

        @TestParser.test
        def break_while_const():
            while True:
                Expr.str(0)
                break

        # --- break in while, function condition ---

        @TestParser.test
        def break_while_func_cond():
            while Expr.true(0):
                Expr.str(1)
                break

        # --- break in while suppresses orelse ---

        @TestParser.test
        def break_while_suppresses_orelse():
            while Expr.true(0):
                Expr.str(1)
                break
            else:
                Expr.str(99)

        # --- break in while, condition is named expr ---

        @TestParser.test
        def break_while_named_expr_cond():
            var_items = [Expr.int(0), Expr.int(1), Expr.int(2)]
            var_i = [0]
            while var_n := var_i[0] < len(var_items):
                Expr.str(var_items[var_i[0]])
                var_i[0] = var_i[0] + 1
                if var_i[0] == 2:
                    break
            Expr.str(var_n)

        # --- break in while, named expr cond, suppresses orelse ---

        @TestParser.test
        def break_while_named_expr_cond_suppresses_orelse():
            var_items = [Expr.int(0), Expr.int(1)]
            var_i = [0]
            while var_n := var_i[0] < len(var_items):
                Expr.str(var_items[var_i[0]])
                break
            else:
                Expr.str(99)
            Expr.str(var_n)

        # --- break in nested while ---

        @TestParser.test
        def break_while_nested():
            while Expr.true(0):
                while Expr.true(1):
                    Expr.str(2)
                    break
                Expr.str(3)
                break


def test_pil_builder_continue():

    with TestParser():

        # --- continue in for, skip remaining body ---

        @TestParser.test
        def continue_for_basic():
            for var_x in [Expr.int(0), Expr.int(1), Expr.int(2)]:
                continue
                Expr.str(var_x)

        # --- continue in for, conditional ---

        @TestParser.test
        def continue_for_conditional():
            for var_x in [Expr.int(0), Expr.int(1), Expr.int(2)]:
                if Expr.true(var_x):
                    continue
                Expr.str(var_x)

        # --- continue in for does not suppress orelse ---

        @TestParser.test
        def continue_for_orelse_fires():
            for var_x in [Expr.int(0), Expr.int(1)]:
                Expr.str(var_x)
                continue
            else:
                Expr.str(99)

        # --- continue in nested for, inner only ---

        @TestParser.test
        def continue_for_nested_inner():
            for var_i in [Expr.int(0), Expr.int(1)]:
                Expr.str(var_i)
                for var_j in [Expr.int(0), Expr.int(1), Expr.int(2)]:
                    if Expr.true(var_j):
                        continue
                    Expr.str(var_j)

        # --- continue in nested for, outer ---

        @TestParser.test
        def continue_for_nested_outer():
            for var_i in [Expr.int(0), Expr.int(1), Expr.int(2)]:
                if Expr.true(var_i):
                    continue
                for var_j in [Expr.int(0), Expr.int(1)]:
                    Expr.str(var_j)
                Expr.str(var_i)

        # --- continue in while, constant condition ---

        @TestParser.test
        def continue_while_const():
            var_n = [0]
            while var_n[0] < 3:
                var_n[0] = var_n[0] + 1
                if var_n[0] == 2:
                    continue
                Expr.str(var_n[0])

        # --- continue in while, function condition (PIL re-evaluates test) ---

        @TestParser.test
        def continue_while_func_cond():
            var_i = [0]
            while Expr.true(var_i[0]):
                var_i[0] = var_i[0] + 1
                if var_i[0] < 2:
                    continue
                Expr.str(var_i[0])
                break

        # --- continue in while does not suppress orelse ---

        @TestParser.test
        def continue_while_orelse_fires():
            var_n = [0]
            while var_n[0] < 2:
                var_n[0] = var_n[0] + 1
                continue
            else:
                Expr.str(99)

        # --- continue in while, condition is named expr (PIL re-evaluates) ---

        @TestParser.test
        def continue_while_named_expr_cond():
            var_items = [Expr.int(0), Expr.int(1), Expr.int(2)]
            var_i = [0]
            while var_n := var_i[0] < len(var_items):
                var_cur = var_i[0]
                var_i[0] = var_i[0] + 1
                if var_cur == 1:
                    continue
                Expr.str(var_items[var_cur])
            Expr.str(var_n)

        # --- continue in nested while ---

        @TestParser.test
        def continue_while_nested():
            var_i = [0]
            while Expr.true(var_i[0]):
                var_j = [0]
                while Expr.true(var_j[0]):
                    var_j[0] = var_j[0] + 1
                    if var_j[0] < 2:
                        continue
                    Expr.str(var_j[0])
                    break
                var_i[0] = var_i[0] + 1
                if var_i[0] < 2:
                    continue
                Expr.str(var_i[0])
                break


def test_pil_builder_with():

    with TestParser():

        # --- single item, no as-binding ---

        @TestParser.test
        def with_single_no_as():
            with Expr.ContextManager(enter_n=0, exit_n=1):
                Expr.str(2)

        # --- single item, with as-binding (name target) ---

        @TestParser.test
        def with_single_as_name():
            with Expr.ContextManager(enter_n=0, exit_n=1) as var_ctx:
                Expr.str(2)
                Expr.str(var_ctx._enter_n)

        # --- context_expr is a complex call expression ---

        @TestParser.test
        def with_ctx_from_call():

            def make_cm():
                return Expr.ContextManager(init_n=Expr.int(0))
            with make_cm():
                Expr.str(1)

        # --- multiple items, no as-bindings ---

        @TestParser.test
        def with_multiple_no_as():
            with Expr.ContextManager(enter_n=0, exit_n=10), Expr.ContextManager(enter_n=1, exit_n=11):
                Expr.str(2)

        # --- multiple items, both with as-bindings ---

        @TestParser.test
        def with_multiple_as_names():
            with (
                Expr.ContextManager(enter_n=0, exit_n=10) as var_a,
                Expr.ContextManager(enter_n=1, exit_n=11) as var_b,
            ):
                Expr.str(2)
                Expr.str(var_a._enter_n)
                Expr.str(var_b._enter_n)

        # --- multiple items, mixed as and no-as ---

        @TestParser.test
        def with_multiple_mixed_as():
            with (
                Expr.ContextManager(enter_n=0, exit_n=10) as var_a,
                Expr.ContextManager(enter_n=1, exit_n=11),
            ):
                Expr.str(2)
                Expr.str(var_a._enter_n)

        # --- three items ---

        @TestParser.test
        def with_three_items():
            with (
                Expr.ContextManager(enter_n=0, exit_n=10),
                Expr.ContextManager(enter_n=1, exit_n=11),
                Expr.ContextManager(enter_n=2, exit_n=12),
            ):
                Expr.str(3)

        # --- as-binding to attribute target ---

        @TestParser.test
        def with_as_attr_target():
            var_obj = Expr(2)
            with Expr.ContextManager(enter_n=0, exit_n=1) as var_obj.val:
                Expr.str(3)

        # --- as-binding to subscript target ---

        @TestParser.test
        def with_as_subscript_target():
            var_obj = Expr(2)
            var_obj[0] = None
            with Expr.ContextManager(enter_n=0, exit_n=1) as var_obj[0]:
                Expr.str(3)
