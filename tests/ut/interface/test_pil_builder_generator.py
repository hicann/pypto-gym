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


def test_pil_builder_list_comp():

    with TestParser():

        # --- 1 for, no if ---

        @TestParser.test
        def listcomp_simple():
            var_l = [Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1), Expr.int(2)]]

        # --- 1 for, 1 if ---

        @TestParser.test
        def listcomp_one_if():
            var_l = [Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1), Expr.int(2)] if Expr.true(var_x)]

        # --- 1 for, 2 if ---

        @TestParser.test
        def listcomp_two_ifs():
            var_l = [Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1), Expr.int(2)]
                     if Expr.true(var_x) if var_x != 2]

        # --- 1 for, if with boolop ---

        @TestParser.test
        def listcomp_if_boolop():
            var_l = [Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1), Expr.int(2)]
                     if Expr.true(0) and Expr.true(var_x)]

        # --- 1 for, if with ifexp ---

        @TestParser.test
        def listcomp_if_ifexp():
            var_l = [Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1)]
                     if (Expr.true(var_x) if Expr.true(0) else Expr.false(var_x))]

        # --- for target: tuple unpack ---

        @TestParser.test
        def listcomp_target_tuple():
            var_l = [Expr.str(var_a) for var_a, var_b in [(Expr.int(0), Expr.int(1)), (Expr.int(2), Expr.int(3))]]

        # --- for target: list unpack ---

        @TestParser.test
        def listcomp_target_list():
            var_l = [Expr.str(var_a) for [var_a, var_b] in [[Expr.int(0), Expr.int(1)], [Expr.int(2), Expr.int(3)]]]

        # --- for target: attribute ---

        @TestParser.test
        def listcomp_target_attr():
            var_obj = Expr(0)
            var_obj.val = None
            var_l = [Expr.str(var_obj.val) for var_obj.val in [Expr.int(0), Expr.int(1)]]

        # --- for target: subscript ---

        @TestParser.test
        def listcomp_target_subscript():
            var_arr = Expr(0)
            var_arr[0] = None
            var_l = [Expr.str(var_arr[0]) for var_arr[0] in [Expr.int(0), Expr.int(1)]]

        # --- 2 for (nested) ---

        @TestParser.test
        def listcomp_two_fors():
            var_l = [Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1)]
                     for var_y in [Expr.int(2), Expr.int(3)]]

        # --- 2 for with if ---

        @TestParser.test
        def listcomp_two_fors_with_if():
            var_l = [Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1)] if Expr.true(var_x)
                     for var_y in [Expr.int(2), Expr.int(3)] if Expr.true(var_y)]

        # --- 3 for (deep nesting) ---

        @TestParser.test
        def listcomp_three_fors():
            var_l = [Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1)]
                     for var_y in [Expr.int(0), Expr.int(1)]
                     for var_z in [Expr.int(0), Expr.int(1)]]


def test_pil_builder_set_comp():

    with TestParser():

        # --- 1 for, no if ---

        @TestParser.test
        def setcomp_simple():
            var_s = {Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1), Expr.int(2)]}

        # --- 1 for, 1 if ---

        @TestParser.test
        def setcomp_one_if():
            var_s = {Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1), Expr.int(2)] if Expr.true(var_x)}

        # --- 1 for, if with boolop ---

        @TestParser.test
        def setcomp_if_boolop():
            var_s = {Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1)]
                     if Expr.true(0) and Expr.true(var_x)}

        # --- 1 for, if with ifexp ---

        @TestParser.test
        def setcomp_if_ifexp():
            var_s = {Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1)]
                     if (Expr.true(var_x) if Expr.true(0) else Expr.false(var_x))}

        # --- for target: tuple unpack ---

        @TestParser.test
        def setcomp_target_tuple():
            var_s = {Expr.str(var_a) for var_a, var_b in [(Expr.int(0), Expr.int(1)), (Expr.int(2), Expr.int(3))]}

        # --- for target: list unpack ---

        @TestParser.test
        def setcomp_target_list():
            var_s = {Expr.str(var_a) for [var_a, var_b] in [[Expr.int(0), Expr.int(1)], [Expr.int(2), Expr.int(3)]]}

        # --- for target: attribute ---

        @TestParser.test
        def setcomp_target_attr():
            var_obj = Expr(0)
            var_obj.val = None
            var_s = {Expr.str(var_obj.val) for var_obj.val in [Expr.int(0), Expr.int(1)]}

        # --- for target: subscript ---

        @TestParser.test
        def setcomp_target_subscript():
            var_arr = Expr(0)
            var_arr[0] = None
            var_s = {Expr.str(var_arr[0]) for var_arr[0] in [Expr.int(0), Expr.int(1)]}

        # --- 2 for (nested) ---

        @TestParser.test
        def setcomp_two_fors():
            var_s = {Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1)]
                     for var_y in [Expr.int(2), Expr.int(3)]}

        # --- 3 for (deep nesting) ---

        @TestParser.test
        def setcomp_three_fors():
            var_s = {Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1)]
                     for var_y in [Expr.int(0), Expr.int(1)]
                     for var_z in [Expr.int(0), Expr.int(1)]}


def test_pil_builder_dict_comp():

    with TestParser():

        # --- 1 for, no if ---

        @TestParser.test
        def dictcomp_simple():
            var_d = {Expr.int(var_x): Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1)]}

        # --- 1 for, 1 if ---

        @TestParser.test
        def dictcomp_one_if():
            var_d = {Expr.int(var_x): Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1)] if Expr.true(var_x)}

        # --- 1 for, if with boolop ---

        @TestParser.test
        def dictcomp_if_boolop():
            var_d = {Expr.int(var_x): Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1)]
                     if Expr.true(0) and Expr.true(var_x)}

        # --- 1 for, if with ifexp ---

        @TestParser.test
        def dictcomp_if_ifexp():
            var_d = {Expr.int(var_x): Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1)]
                     if (Expr.true(var_x) if Expr.true(0) else Expr.false(var_x))}

        # --- for target: tuple unpack ---

        @TestParser.test
        def dictcomp_target_tuple():
            var_d = {Expr.int(var_a): Expr.str(var_b)
                     for var_a, var_b in [(Expr.int(0), Expr.int(1)), (Expr.int(2), Expr.int(3))]}

        # --- for target: list unpack ---

        @TestParser.test
        def dictcomp_target_list():
            var_d = {Expr.int(var_a): Expr.str(var_b)
                     for [var_a, var_b] in [[Expr.int(0), Expr.int(1)], [Expr.int(2), Expr.int(3)]]}

        # --- for target: attribute ---

        @TestParser.test
        def dictcomp_target_attr():
            var_obj = Expr(0)
            var_obj.val = None
            var_d = {Expr.int(var_obj.val): Expr.str(var_obj.val) for var_obj.val in [Expr.int(0), Expr.int(1)]}

        # --- for target: subscript ---

        @TestParser.test
        def dictcomp_target_subscript():
            var_arr = Expr(0)
            var_arr[0] = None
            var_d = {Expr.int(var_arr[0]): Expr.str(var_arr[0]) for var_arr[0] in [Expr.int(0), Expr.int(1)]}

        # --- 2 for (nested) ---

        @TestParser.test
        def dictcomp_two_fors():
            var_d = {Expr.int(var_x): Expr.str(var_y)
                     for var_x in [Expr.int(0), Expr.int(1)]
                     for var_y in [Expr.int(2), Expr.int(3)]}

        # --- 3 for (deep nesting) ---

        @TestParser.test
        def dictcomp_three_fors():
            var_d = {Expr.int(var_x): Expr.str(var_z)
                     for var_x in [Expr.int(0), Expr.int(1)]
                     for var_y in [Expr.int(0), Expr.int(1)]
                     for var_z in [Expr.int(0), Expr.int(1)]}


def test_pil_builder_generator_exp():

    with TestParser():

        # --- 1 for, no if: consume via list() ---

        @TestParser.test
        def genexp_simple():
            g = (Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1), Expr.int(2)])
            var_l = list(g)

        # --- 1 for, 1 if ---

        @TestParser.test
        def genexp_one_if():
            g = (Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1), Expr.int(2)] if Expr.true(var_x))
            var_l = list(g)

        # --- 1 for, if with boolop ---

        @TestParser.test
        def genexp_if_boolop():
            g = (Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1)]
                     if Expr.true(0) and Expr.true(var_x))
            var_l = list(g)

        # --- 1 for, if with ifexp ---

        @TestParser.test
        def genexp_if_ifexp():
            g = (Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1)]
                     if (Expr.true(var_x) if Expr.true(0) else Expr.false(var_x)))
            var_l = list(g)

        # --- for target: tuple unpack ---

        @TestParser.test
        def genexp_target_tuple():
            g = (Expr.str(var_a) for var_a, var_b in [(Expr.int(0), Expr.int(1)), (Expr.int(2), Expr.int(3))])
            var_l = list(g)

        # --- for target: list unpack ---

        @TestParser.test
        def genexp_target_list():
            g = (Expr.str(var_a) for [var_a, var_b] in [[Expr.int(0), Expr.int(1)], [Expr.int(2), Expr.int(3)]])
            var_l = list(g)

        # --- for target: attribute ---

        @TestParser.test
        def genexp_target_attr():
            var_obj = Expr(0)
            var_obj.val = None
            g = (Expr.str(var_obj.val) for var_obj.val in [Expr.int(0), Expr.int(1)])
            var_l = list(g)

        # --- for target: subscript ---

        @TestParser.test
        def genexp_target_subscript():
            var_arr = Expr(0)
            var_arr[0] = None
            g = (Expr.str(var_arr[0]) for var_arr[0] in [Expr.int(0), Expr.int(1)])
            var_l = list(g)

        # --- 2 for (nested) ---

        @TestParser.test
        def genexp_two_fors():
            g = (Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1)]
                     for var_y in [Expr.int(2), Expr.int(3)])
            var_l = list(g)

        # --- 3 for (deep nesting) ---

        @TestParser.test
        def genexp_three_fors():
            g = (Expr.str(var_x) for var_x in [Expr.int(0), Expr.int(1)]
                     for var_y in [Expr.int(0), Expr.int(1)]
                     for var_z in [Expr.int(0), Expr.int(1)])
            var_l = list(g)


def test_pil_builder_yield():

    with TestParser():

        # --- bare yield (value is None) ---

        @TestParser.test
        def yield_bare():

            def gen():
                yield
            var_l = list(gen())

        # --- yield name ---

        @TestParser.test
        def yield_name():

            def gen():
                var_x = Expr.int(0)
                yield var_x
            var_l = list(gen())
            Expr.str(var_l[0])

        # --- yield call expr (PIL inserts a temp) ---

        @TestParser.test
        def yield_call_expr():

            def gen():
                yield Expr.int(0)
            var_l = list(gen())
            Expr.str(var_l[0])

        # --- yield constant ---

        @TestParser.test
        def yield_constant():

            def gen():
                yield 42
            var_l = list(gen())
            Expr.str(var_l[0])

        # --- yield binop expression ---

        @TestParser.test
        def yield_binop():

            def gen():
                yield Expr.int(0) + Expr.int(1)
            var_l = list(gen())
            Expr.str(var_l[0])

        # --- yield attribute expression ---

        @TestParser.test
        def yield_attr():

            def gen():
                var_obj = Expr(0)
                var_obj.val = Expr.int(99)
                yield var_obj.val
            var_l = list(gen())
            Expr.str(var_l[0])

        # --- yield subscript expression ---

        @TestParser.test
        def yield_subscript():

            def gen():
                var_arr = [Expr.int(0), Expr.int(1)]
                yield var_arr[0]
            var_l = list(gen())
            Expr.str(var_l[0])

        # --- multiple yields in sequence ---

        @TestParser.test
        def yield_multiple():

            def gen():
                yield Expr.int(0)
                yield Expr.int(1)
                yield Expr.int(2)
            var_l = list(gen())
            Expr.str(var_l[0])
            Expr.str(var_l[1])
            Expr.str(var_l[2])

        # --- yield inside if branch ---

        @TestParser.test
        def yield_in_if():

            def gen(flag):
                if flag:
                    yield Expr.int(0)
                else:
                    yield Expr.int(1)
            var_l0 = list(gen(Expr.true(0)))
            Expr.str(var_l0[0])
            var_l1 = list(gen(Expr.false(1)))
            Expr.str(var_l1[0])

        # --- yield inside for loop ---

        @TestParser.test
        def yield_in_for():

            def gen():
                for var_x in [Expr.int(0), Expr.int(1), Expr.int(2)]:
                    yield var_x
            var_l = list(gen())
            Expr.str(var_l[0])
            Expr.str(var_l[1])
            Expr.str(var_l[2])

        # --- yield from name ---

        @TestParser.test
        def yield_from_name():

            def inner():
                yield Expr.int(0)
                yield Expr.int(1)

            def gen():
                var_g = inner()
                yield from var_g
            var_l = list(gen())
            Expr.str(var_l[0])
            Expr.str(var_l[1])

        # --- yield from call expr ---

        @TestParser.test
        def yield_from_call():

            def inner():
                yield Expr.int(0)
                yield Expr.int(1)

            def gen():
                yield from inner()
            var_l = list(gen())
            Expr.str(var_l[0])
            Expr.str(var_l[1])

        # --- send value captured via yield assignment ---

        @TestParser.test
        def yield_send_value():

            def gen():
                var_sent = yield Expr.int(0)
                Expr.str(var_sent)
            g = gen()
            next(g)
            try:
                g.send(Expr.int(1))
            except StopIteration:
                pass
