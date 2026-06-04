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


def test_pil_builder_constant():

    with TestParser():

        # --- constant in binop ---

        @TestParser.test
        def const_binop_left():
            var_x = 2 + Expr.int(0)
            Expr.str(var_x)

        @TestParser.test
        def const_binop_right():
            var_x = Expr.int(0) + 3
            Expr.str(var_x)

        @TestParser.test
        def const_binop_both():
            var_x = 2 + 3
            Expr.str(var_x)

        # --- constant in unary op ---

        @TestParser.test
        def const_unary_neg():
            var_x = -1
            Expr.str(var_x)

        @TestParser.test
        def const_unary_not():
            var_x = not False
            Expr.str(var_x)

        # --- constant as if test ---

        @TestParser.test
        def const_if_true():
            if 1:
                Expr.str(0)
            else:
                Expr.str(1)

        @TestParser.test
        def const_if_false():
            if 0:
                Expr.str(0)
            else:
                Expr.str(1)

        # --- constant as for iter ---

        @TestParser.test
        def const_for_iter():
            for var_x in (0, 1, 2):
                Expr.str(var_x)

        # --- constant as while test ---

        @TestParser.test
        def const_while_test():
            while 1:
                Expr.str(0)
                break

        # --- constant as call positional arg ---

        @TestParser.test
        def const_call_pos_arg():

            def func(x):
                Expr.str(x)
            func(42)

        # --- constant as call keyword arg ---

        @TestParser.test
        def const_call_kw_arg():

            def func(x):
                Expr.str(x)
            func(x=42)

        # --- constant in tuple literal ---

        @TestParser.test
        def const_in_tuple():
            var_t = (0, Expr.int(1), 2)
            Expr.str(var_t[0])
            Expr.str(var_t[1])
            Expr.str(var_t[2])

        # --- constant in list literal ---

        @TestParser.test
        def const_in_list():
            var_l = [0, Expr.int(1), 2]
            Expr.str(var_l[0])
            Expr.str(var_l[1])
            Expr.str(var_l[2])

        # --- constant as dict key and value ---

        @TestParser.test
        def const_dict_key():
            var_d = {0: Expr.int(1)}
            Expr.str(var_d[0])

        @TestParser.test
        def const_dict_value():
            var_d = {Expr.int(0): 99}
            Expr.str(var_d[0])

        @TestParser.test
        def const_dict_both():
            var_d = {0: 99}
            Expr.str(var_d[0])

        # --- constant in set literal ---

        @TestParser.test
        def const_in_set():
            var_s = {0, Expr.int(1)}
            Expr.str(0 in var_s)

        # --- constant as subscript index ---

        @TestParser.test
        def const_subscript_index():
            var_obj = Expr(0)
            var_obj[0] = Expr.int(1)
            var_x = var_obj[0]
            Expr.str(var_x)

        # --- constant as slice bounds ---

        @TestParser.test
        def const_slice_bounds():
            var_l = [Expr.int(0), Expr.int(1), Expr.int(2), Expr.int(3)]
            var_s = var_l[1:3]
            Expr.str(var_s[0])

        # --- constant as type annotation (no side-effect) ---

        @TestParser.test
        def const_annotation_no_value():
            var_x: int

        @TestParser.test
        def const_annotation_with_value():
            var_x: int = Expr.int(0)
            Expr.str(var_x)


def test_pil_builder_joined_str():

    with TestParser():

        # --- 单个插值, 无 conversion, 无 format_spec ---
        # PIL: single-part path, returns the expr directly (no join)
        # Expr.str returns a str, so format(s, '') == s — values match

        @TestParser.test
        def fstr_single_expr():
            var_x = f"{Expr.str(0)}"

        # --- 多个部分: 字面量前缀 + 插值 ---

        @TestParser.test
        def fstr_prefix_and_expr():
            var_x = f"prefix_{Expr.str(0)}"

        # --- 多个部分: 插值 + 字面量后缀 ---

        @TestParser.test
        def fstr_expr_and_suffix():
            var_x = f"{Expr.str(0)}_suffix"

        # --- 多个部分: 前缀 + 插值 + 后缀 ---

        @TestParser.test
        def fstr_prefix_expr_suffix():
            var_x = f"prefix_{Expr.str(0)}_suffix"

        # --- 多个插值, 无字面量 ---

        @TestParser.test
        def fstr_two_exprs():
            var_x = f"{Expr.str(0)}{Expr.str(1)}"

        # --- 多个插值, 中间有字面量 ---

        @TestParser.test
        def fstr_expr_sep_expr():
            var_x = f"{Expr.str(0)}_{Expr.str(1)}"

        # --- conversion !s ---

        @TestParser.test
        def fstr_conversion_s():
            var_x = f"{Expr.int(0)!s}"

        # --- conversion !r ---

        @TestParser.test
        def fstr_conversion_r():
            var_x = f"{Expr.int(0)!r}"

        # --- conversion !a ---

        @TestParser.test
        def fstr_conversion_a():
            var_x = f"{Expr.int(0)!a}"

        # --- format_spec 是字面量 ---

        @TestParser.test
        def fstr_format_spec_const():
            var_x = f"{Expr.str(0):>10}"

        # --- format_spec 是变量表达式 ---

        @TestParser.test
        def fstr_format_spec_expr():
            var_fmt = '>10'
            var_x = f"{Expr.str(0):{var_fmt}}"

        # --- conversion + format_spec ---

        @TestParser.test
        def fstr_conversion_and_format_spec():
            var_x = f"{Expr.int(0)!r:>10}"

        # --- 插值表达式本身是嵌套调用 ---

        @TestParser.test
        def fstr_nested_call_expr():
            var_x = f"{Expr.str(Expr.int(0))}"


def test_pil_builder_dict():

    with TestParser():

        # --- empty dict ---

        @TestParser.test
        def dict_empty():
            var_d = {}

        # --- single pair: constant key, call value ---

        @TestParser.test
        def dict_const_key_call_value():
            var_d = {0: Expr.int(1)}

        # --- single pair: call key, call value ---

        @TestParser.test
        def dict_call_key_call_value():
            var_d = {Expr.int(0): Expr.int(1)}

        # --- multiple pairs: all call keys and values (eval order: k0,v0,k1,v1,...) ---

        @TestParser.test
        def dict_multiple_pairs():
            var_d = {Expr.int(0): Expr.int(1), Expr.int(2): Expr.int(3)}

        @TestParser.test
        def dict_three_pairs():
            var_d = {Expr.int(0): Expr.int(1), Expr.int(2): Expr.int(3), Expr.int(4): Expr.int(5)}

        # --- spread: **other (key is None in the AST) ---

        @TestParser.test
        def dict_spread_only():
            var_other = {0: Expr.int(0)}
            var_d = {**var_other}

        # --- mixed: normal pair then spread ---

        @TestParser.test
        def dict_normal_then_spread():
            var_other = {2: Expr.int(2)}
            var_d = {Expr.int(0): Expr.int(1), **var_other}

        # --- spread in the middle ---

        @TestParser.test
        def dict_spread_in_middle():
            var_other = {1: Expr.int(2)}
            var_d = {Expr.int(0): Expr.int(0), **var_other, Expr.int(3): Expr.int(4)}

        # --- multiple spreads ---

        @TestParser.test
        def dict_multiple_spreads():
            var_a = {0: Expr.int(0)}
            var_b = {1: Expr.int(1)}
            var_d = {**var_a, **var_b}

        # --- spread where the source is a function call result ---

        @TestParser.test
        def dict_spread_from_func_call():

            def make():
                return {Expr.int(0): Expr.int(1)}
            var_d = {**make()}

        @TestParser.test
        def dict_normal_then_spread_from_call():

            def make():
                return {Expr.int(2): Expr.int(3)}
            var_d = {Expr.int(0): Expr.int(1), **make()}

        @TestParser.test
        def dict_spread_from_call_then_normal():

            def make():
                return {Expr.int(0): Expr.int(1)}
            var_d = {**make(), Expr.int(2): Expr.int(3)}

        @TestParser.test
        def dict_multiple_spreads_from_calls():

            def make_a():
                return {Expr.int(0): Expr.int(1)}

            def make_b():
                return {Expr.int(2): Expr.int(3)}
            var_d = {**make_a(), **make_b()}

        # --- nested: value is itself a dict literal ---

        @TestParser.test
        def dict_nested_value():
            var_d = {Expr.int(0): {Expr.int(1): Expr.int(2)}}


def test_pil_builder_set():

    with TestParser():

        # --- single element: constant ---

        @TestParser.test
        def set_single_const():
            var_s = {0}

        # --- single element: call ---

        @TestParser.test
        def set_single_call():
            var_s = {Expr.int(0)}

        # --- multiple elements: all calls (eval order left-to-right) ---

        @TestParser.test
        def set_multiple_calls():
            var_s = {Expr.int(0), Expr.int(1), Expr.int(2)}

        # --- spread: *other where other is a variable ---

        @TestParser.test
        def set_spread_only():
            var_other = [Expr.int(0), Expr.int(1)]
            var_s = {*var_other}

        # --- spread where source is a function call result ---

        @TestParser.test
        def set_spread_from_func_call():

            def make():
                return [Expr.int(0), Expr.int(1)]
            var_s = {*make()}

        # --- normal element then spread ---

        @TestParser.test
        def set_normal_then_spread():

            def make():
                return [Expr.int(2), Expr.int(3)]
            var_s = {Expr.int(0), Expr.int(1), *make()}

        # --- spread then normal element ---

        @TestParser.test
        def set_spread_then_normal():

            def make():
                return [Expr.int(0), Expr.int(1)]
            var_s = {*make(), Expr.int(2), Expr.int(3)}

        # --- spread in the middle ---

        @TestParser.test
        def set_spread_in_middle():

            def make():
                return [Expr.int(1), Expr.int(2)]
            var_s = {Expr.int(0), *make(), Expr.int(3)}

        # --- multiple spreads from calls ---

        @TestParser.test
        def set_multiple_spreads_from_calls():

            def make_a():
                return [Expr.int(0), Expr.int(1)]

            def make_b():
                return [Expr.int(2), Expr.int(3)]
            var_s = {*make_a(), *make_b()}


def test_pil_builder_list():

    with TestParser():

        # --- empty ---

        @TestParser.test
        def list_empty():
            var_l = []

        # --- single element: constant ---

        @TestParser.test
        def list_single_const():
            var_l = [0]

        # --- single element: call ---

        @TestParser.test
        def list_single_call():
            var_l = [Expr.int(0)]

        # --- multiple elements: all calls (eval order left-to-right) ---

        @TestParser.test
        def list_multiple_calls():
            var_l = [Expr.int(0), Expr.int(1), Expr.int(2)]

        # --- starred: *other where other is a variable ---

        @TestParser.test
        def list_starred_only():
            var_other = [Expr.int(0), Expr.int(1)]
            var_l = [*var_other]

        # --- starred: source is a function call result ---

        @TestParser.test
        def list_starred_from_call():

            def make():
                return [Expr.int(0), Expr.int(1)]
            var_l = [*make()]

        # --- normal then starred ---

        @TestParser.test
        def list_normal_then_starred():

            def make():
                return [Expr.int(2), Expr.int(3)]
            var_l = [Expr.int(0), Expr.int(1), *make()]

        # --- starred then normal ---

        @TestParser.test
        def list_starred_then_normal():

            def make():
                return [Expr.int(0), Expr.int(1)]
            var_l = [*make(), Expr.int(2), Expr.int(3)]

        # --- starred in the middle ---

        @TestParser.test
        def list_starred_in_middle():

            def make():
                return [Expr.int(1), Expr.int(2)]
            var_l = [Expr.int(0), *make(), Expr.int(3)]

        # --- multiple starred from calls ---

        @TestParser.test
        def list_multiple_starred_from_calls():

            def make_a():
                return [Expr.int(0), Expr.int(1)]

            def make_b():
                return [Expr.int(2), Expr.int(3)]
            var_l = [*make_a(), *make_b()]

        # --- nested list literal as element ---

        @TestParser.test
        def list_nested():
            var_l = [Expr.int(0), [Expr.int(1), Expr.int(2)]]


def test_pil_builder_tuple():

    with TestParser():

        # --- single element (trailing comma) ---

        @TestParser.test
        def tuple_single_const():
            var_t = (0,)

        # --- single element: call ---

        @TestParser.test
        def tuple_single_call():
            var_t = (Expr.int(0),)

        # --- multiple elements: all calls (eval order left-to-right) ---

        @TestParser.test
        def tuple_multiple_calls():
            var_t = (Expr.int(0), Expr.int(1), Expr.int(2))

        # --- starred: *other where other is a variable ---

        @TestParser.test
        def tuple_starred_only():
            var_other = [Expr.int(0), Expr.int(1)]
            var_t = (*var_other,)

        # --- starred: source is a function call result ---

        @TestParser.test
        def tuple_starred_from_call():

            def make():
                return [Expr.int(0), Expr.int(1)]
            var_t = (*make(),)

        # --- normal then starred ---

        @TestParser.test
        def tuple_normal_then_starred():

            def make():
                return [Expr.int(2), Expr.int(3)]
            var_t = (Expr.int(0), Expr.int(1), *make())

        # --- starred then normal ---

        @TestParser.test
        def tuple_starred_then_normal():

            def make():
                return [Expr.int(0), Expr.int(1)]
            var_t = (*make(), Expr.int(2), Expr.int(3))

        # --- starred in the middle ---

        @TestParser.test
        def tuple_starred_in_middle():

            def make():
                return [Expr.int(1), Expr.int(2)]
            var_t = (Expr.int(0), *make(), Expr.int(3))

        # --- multiple starred from calls ---

        @TestParser.test
        def tuple_multiple_starred_from_calls():

            def make_a():
                return [Expr.int(0), Expr.int(1)]

            def make_b():
                return [Expr.int(2), Expr.int(3)]
            var_t = (*make_a(), *make_b())

        # --- nested tuple literal as element ---

        @TestParser.test
        def tuple_nested():
            var_t = (Expr.int(0), (Expr.int(1), Expr.int(2)))


def test_pil_builder_attribute():

    with TestParser():

        # --- attribute in binop ---

        @TestParser.test
        def attr_binop_left():
            var_obj = Expr(0)
            var_obj.val = Expr.int(2)
            var_x = var_obj.val + Expr.int(1)
            Expr.str(var_x)

        @TestParser.test
        def attr_binop_right():
            var_obj = Expr(0)
            var_obj.val = Expr.int(3)
            var_x = Expr.int(1) + var_obj.val
            Expr.str(var_x)

        @TestParser.test
        def attr_binop_both():
            var_a = Expr(0)
            var_a.val = Expr.int(2)
            var_b = Expr(1)
            var_b.val = Expr.int(3)
            var_x = var_a.val + var_b.val
            Expr.str(var_x)

        # --- attribute in unary op ---

        @TestParser.test
        def attr_unary_neg():
            var_obj = Expr(0)
            var_obj.val = Expr.int(5)
            var_x = -var_obj.val
            Expr.str(var_x)

        @TestParser.test
        def attr_unary_not():
            var_obj = Expr(0)
            var_obj.val = Expr.true(0)
            var_x = not var_obj.val
            Expr.str(var_x)

        # --- attribute as if test ---

        @TestParser.test
        def attr_if_true():
            var_obj = Expr(0)
            var_obj.val = Expr.true(0)
            if var_obj.val:
                Expr.str(1)
            else:
                Expr.str(2)

        @TestParser.test
        def attr_if_false():
            var_obj = Expr(0)
            var_obj.val = Expr.false(0)
            if var_obj.val:
                Expr.str(1)
            else:
                Expr.str(2)

        # --- attribute as for iter ---

        @TestParser.test
        def attr_for_iter():
            var_obj = Expr(0)
            var_obj.val = [Expr.int(0), Expr.int(1), Expr.int(2)]
            for var_x in var_obj.val:
                Expr.str(var_x)

        # --- attribute as while test ---

        @TestParser.test
        def attr_while_test():
            var_obj = Expr(0)
            var_obj.val = Expr.true(0)
            while var_obj.val:
                Expr.str(1)
                var_obj.val = False

        # --- attribute as call positional arg ---

        @TestParser.test
        def attr_call_pos_arg():
            var_obj = Expr(0)
            var_obj.val = Expr.int(1)
            Expr.str(var_obj.val)

        # --- attribute as call keyword arg ---

        @TestParser.test
        def attr_call_kw_arg():

            def func(x):
                Expr.str(x)
            var_obj = Expr(0)
            var_obj.val = Expr.int(1)
            func(x=var_obj.val)

        # --- attribute in tuple literal ---

        @TestParser.test
        def attr_in_tuple():
            var_obj = Expr(0)
            var_obj.val = Expr.int(1)
            var_t = (var_obj.val, Expr.int(2))
            Expr.str(var_t[0])
            Expr.str(var_t[1])

        # --- attribute in list literal ---

        @TestParser.test
        def attr_in_list():
            var_obj = Expr(0)
            var_obj.val = Expr.int(1)
            var_l = [var_obj.val, Expr.int(2)]
            Expr.str(var_l[0])
            Expr.str(var_l[1])

        # --- attribute as dict key and value ---

        @TestParser.test
        def attr_dict_key():
            var_obj = Expr(0)
            var_obj.val = Expr.int(0)
            var_d = {var_obj.val: Expr.int(1)}
            Expr.str(var_d[0])

        @TestParser.test
        def attr_dict_value():
            var_obj = Expr(0)
            var_obj.val = Expr.int(99)
            var_d = {Expr.int(0): var_obj.val}
            Expr.str(var_d[0])

        # --- attribute in set literal ---

        @TestParser.test
        def attr_in_set():
            var_obj = Expr(0)
            var_obj.val = Expr.int(1)
            var_s = {var_obj.val, Expr.int(2)}
            Expr.str(1 in var_s)

        # --- attribute as subscript index ---

        @TestParser.test
        def attr_subscript_index():
            var_obj = Expr(0)
            var_obj.val = Expr.int(0)
            var_arr = Expr(1)
            var_arr[0] = Expr.int(99)
            var_x = var_arr[var_obj.val]
            Expr.str(var_x)

        # --- attribute as slice bounds ---

        @TestParser.test
        def attr_slice_lower():
            var_obj = Expr(0)
            var_obj.val = Expr.int(1)
            var_l = [Expr.int(0), Expr.int(1), Expr.int(2)]
            var_s = var_l[var_obj.val:3]
            Expr.str(var_s[0])

        @TestParser.test
        def attr_slice_upper():
            var_obj = Expr(0)
            var_obj.val = Expr.int(2)
            var_l = [Expr.int(0), Expr.int(1), Expr.int(2)]
            var_s = var_l[0:var_obj.val]
            Expr.str(var_s[0])

        @TestParser.test
        def attr_slice_both():
            var_lo = Expr(0)
            var_lo.val = Expr.int(1)
            var_hi = Expr(1)
            var_hi.val = Expr.int(3)
            var_l = [Expr.int(0), Expr.int(1), Expr.int(2), Expr.int(3)]
            var_s = var_l[var_lo.val:var_hi.val]
            Expr.str(var_s[0])

        # --- attribute as type annotation ---

        @TestParser.test
        def attr_annotation_no_value():
            var_x: Expr.ContextManager

        @TestParser.test
        def attr_annotation_with_value():
            var_x: Expr.ContextManager = Expr.int(0)
            Expr.str(var_x)


def test_pil_builder_subscript():

    with TestParser():

        # ================================================================
        # Part 1: subscript result used in various expression contexts
        # ================================================================

        # --- subscript result in binop ---

        @TestParser.test
        def subscript_binop_left():
            var_l = [Expr.int(2), Expr.int(3)]
            var_x = var_l[0] + Expr.int(1)
            Expr.str(var_x)

        @TestParser.test
        def subscript_binop_right():
            var_l = [Expr.int(3), Expr.int(4)]
            var_x = Expr.int(1) + var_l[0]
            Expr.str(var_x)

        @TestParser.test
        def subscript_binop_both():
            var_l = [Expr.int(2), Expr.int(3)]
            var_x = var_l[0] + var_l[1]
            Expr.str(var_x)

        # --- subscript result in unary op ---

        @TestParser.test
        def subscript_unary_neg():
            var_l = [Expr.int(5)]
            var_x = -var_l[0]
            Expr.str(var_x)

        @TestParser.test
        def subscript_unary_not():
            var_l = [Expr.true(0)]
            var_x = not var_l[0]
            Expr.str(var_x)

        # --- subscript result as if test ---

        @TestParser.test
        def subscript_if_true():
            var_l = [Expr.true(0)]
            if var_l[0]:
                Expr.str(1)
            else:
                Expr.str(2)

        @TestParser.test
        def subscript_if_false():
            var_l = [Expr.false(0)]
            if var_l[0]:
                Expr.str(1)
            else:
                Expr.str(2)

        # --- subscript result as for iter ---

        @TestParser.test
        def subscript_for_iter():
            var_l = [[Expr.int(0), Expr.int(1), Expr.int(2)]]
            for var_x in var_l[0]:
                Expr.str(var_x)

        # --- subscript result as while test ---

        @TestParser.test
        def subscript_while_test():
            var_l = [True]
            while var_l[0]:
                Expr.str(0)
                var_l[0] = False

        # --- subscript result as call positional arg ---

        @TestParser.test
        def subscript_call_pos_arg():
            var_l = [Expr.int(0)]
            Expr.str(var_l[0])

        # --- subscript result as call keyword arg ---

        @TestParser.test
        def subscript_call_kw_arg():

            def func(x):
                Expr.str(x)
            var_l = [Expr.int(0)]
            func(x=var_l[0])

        # --- subscript result in tuple literal ---

        @TestParser.test
        def subscript_in_tuple():
            var_l = [Expr.int(1), Expr.int(2)]
            var_t = (var_l[0], var_l[1])
            Expr.str(var_t[0])
            Expr.str(var_t[1])

        # --- subscript result in list literal ---

        @TestParser.test
        def subscript_in_list():
            var_l = [Expr.int(1), Expr.int(2)]
            var_r = [var_l[0], var_l[1]]
            Expr.str(var_r[0])
            Expr.str(var_r[1])

        # --- subscript result as dict key and value ---

        @TestParser.test
        def subscript_dict_key():
            var_keys = [Expr.int(0)]
            var_d = {var_keys[0]: Expr.int(99)}
            Expr.str(var_d[0])

        @TestParser.test
        def subscript_dict_value():
            var_vals = [Expr.int(99)]
            var_d = {Expr.int(0): var_vals[0]}
            Expr.str(var_d[0])

        # --- subscript result in set literal ---

        @TestParser.test
        def subscript_in_set():
            var_l = [Expr.int(1), Expr.int(2)]
            var_s = {var_l[0], var_l[1]}
            Expr.str(1 in var_s)

        # --- subscript result as subscript index ---

        @TestParser.test
        def subscript_as_index():
            var_idx = [Expr.int(0)]
            var_arr = Expr(0)
            var_arr[0] = Expr.int(99)
            var_x = var_arr[var_idx[0]]
            Expr.str(var_x)

        # --- subscript result as type annotation ---

        @TestParser.test
        def subscript_annotation_no_value():
            var_x: list[int]

        @TestParser.test
        def subscript_annotation_with_value():
            var_x: list[int] = [Expr.int(0)]
            Expr.str(var_x[0])

        # ================================================================
        # Part 2: slice expressions of various kinds
        # ================================================================

        # --- slice index: attr ---

        @TestParser.test
        def slice_index_attr():
            var_obj = Expr(0)
            var_obj.val = Expr.int(1)
            var_arr = Expr(1)
            var_arr[1] = Expr.int(99)
            var_x = var_arr[var_obj.val]
            Expr.str(var_x)

        # --- slice index: call ---

        @TestParser.test
        def slice_index_call():

            def idx():
                return Expr.int(0)
            var_arr = Expr(0)
            var_arr[0] = Expr.int(99)
            var_x = var_arr[idx()]
            Expr.str(var_x)

        # --- slice index: binop ---

        @TestParser.test
        def slice_index_binop():
            var_arr = Expr(0)
            var_arr[2] = Expr.int(99)
            var_x = var_arr[Expr.int(1) + Expr.int(1)]
            Expr.str(var_x)

        # --- slice index: unary op ---

        @TestParser.test
        def slice_index_unary():
            var_arr = Expr(0)
            var_arr[-1] = Expr.int(99)
            var_x = var_arr[-Expr.int(1)]
            Expr.str(var_x)

        # --- slice index: subscript ---

        @TestParser.test
        def slice_index_subscript():
            var_idxs = [Expr.int(0)]
            var_arr = Expr(0)
            var_arr[0] = Expr.int(99)
            var_x = var_arr[var_idxs[0]]
            Expr.str(var_x)

        # --- slice index: dict value ---

        @TestParser.test
        def slice_index_dict():
            var_d = {0: Expr.int(1)}
            var_arr = Expr(0)
            var_arr[1] = Expr.int(99)
            var_x = var_arr[var_d[0]]
            Expr.str(var_x)

        # --- slice index: set membership (bool) ---

        @TestParser.test
        def slice_index_set():
            var_s = {0}
            var_arr = Expr(0)
            var_arr[True] = Expr.int(99)
            var_x = var_arr[0 in var_s]
            Expr.str(var_x)

        # --- slice index: named expr ---

        @TestParser.test
        def slice_index_named_expr():
            var_arr = Expr(0)
            var_arr[0] = Expr.int(99)
            var_x = var_arr[(var_k := Expr.int(0))]
            Expr.str(var_x)
            Expr.str(var_k)

        # --- slice range: attr bounds ---

        @TestParser.test
        def slice_range_attr_bounds():
            var_lo = Expr(0)
            var_lo.val = Expr.int(1)
            var_hi = Expr(1)
            var_hi.val = Expr.int(3)
            var_l = [Expr.int(0), Expr.int(1), Expr.int(2), Expr.int(3)]
            var_s = var_l[var_lo.val:var_hi.val]
            Expr.str(var_s[0])

        # --- slice range: call bounds ---

        @TestParser.test
        def slice_range_call_bounds():

            def lo():
                return Expr.int(1)

            def hi():
                return Expr.int(3)
            var_l = [Expr.int(0), Expr.int(1), Expr.int(2), Expr.int(3)]
            var_s = var_l[lo():hi()]
            Expr.str(var_s[0])

        # --- slice range: binop bounds ---

        @TestParser.test
        def slice_range_binop_bounds():
            var_l = [Expr.int(0), Expr.int(1), Expr.int(2), Expr.int(3)]
            var_s = var_l[Expr.int(0) + 1: Expr.int(1) + 2]
            Expr.str(var_s[0])

        # --- slice range: subscript bounds ---

        @TestParser.test
        def slice_range_subscript_bounds():
            var_bounds = [Expr.int(1), Expr.int(3)]
            var_l = [Expr.int(0), Expr.int(1), Expr.int(2), Expr.int(3)]
            var_s = var_l[var_bounds[0]:var_bounds[1]]
            Expr.str(var_s[0])

        # --- slice range: dict bounds ---

        @TestParser.test
        def slice_range_dict_bounds():
            var_d = {'lo': Expr.int(1), 'hi': Expr.int(3)}
            var_l = [Expr.int(0), Expr.int(1), Expr.int(2), Expr.int(3)]
            var_s = var_l[var_d['lo']:var_d['hi']]
            Expr.str(var_s[0])
