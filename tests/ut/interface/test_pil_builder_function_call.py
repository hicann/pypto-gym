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


def test_pil_builder_function_def():

    with TestParser():

        # --- 3-level nesting ---

        @TestParser.test
        def three_level_nesting():

            def outer():

                def middle():

                    def inner():
                        Expr.str(0)
                    inner()
                middle()
            outer()

        @TestParser.test
        def three_level_nesting_with_return_values():

            def outer():

                def middle():

                    def inner():
                        Expr.str(0)
                        Expr.str(1)
                    inner()
                    Expr.str(2)
                middle()
                Expr.str(3)
            outer()

        # --- decorator ---

        @TestParser.test
        def function_with_decorator():
            var_e = Expr(0)

            @var_e.decorate(1)
            def func():
                Expr.str(2)
            func()

        @TestParser.test
        def function_with_multiple_decorators():
            var_e = Expr(0)

            @var_e.decorate(1)
            @var_e.decorate(2)
            def func():
                Expr.str(3)
            func()

        # --- arg with default values ---

        @TestParser.test
        def function_with_default_arg():

            def func(x=Expr.int(0)):
                Expr.str(x)
            func()
            func(Expr.int(1))

        @TestParser.test
        def function_with_multiple_defaults():

            def func(x=Expr.int(0), y=Expr.int(1)):
                Expr.str(x)
                Expr.str(y)
            func()

        # --- combined: 3-level nesting + decorator + default args ---

        @TestParser.test
        def three_level_nesting_with_decorator_and_default():
            var_e = Expr(0)

            @var_e.decorate(1)
            def outer(x=Expr.int(2)):

                @var_e.decorate(3)
                def middle(y=Expr.int(4)):

                    def inner():
                        Expr.str(x)
                        Expr.str(y)
                    inner()
                middle()
            outer()


def test_pil_builder_lambda():

    with TestParser():

        # --- 无参 lambda, body 是调用 ---

        @TestParser.test
        def lambda_no_args():
            f = lambda: Expr.str(0)
            var_r = f()

        # --- 单参数, body 是调用 ---

        @TestParser.test
        def lambda_single_arg():
            f = lambda x: Expr.str(x)
            var_r = f(Expr.int(0))

        # --- 多参数 ---

        @TestParser.test
        def lambda_multiple_args():
            f = lambda x, y: Expr.str(x)
            var_r = f(Expr.int(0), Expr.int(1))

        # --- 带默认值, default 未被覆盖 ---

        @TestParser.test
        def lambda_default_not_overridden():
            f = lambda x=Expr.int(0): Expr.str(x)
            var_r = f()

        # --- 带默认值, default 被覆盖 ---

        @TestParser.test
        def lambda_default_overridden():
            f = lambda x=Expr.int(0): Expr.str(x)
            var_r = f(Expr.int(1))

        # --- *args 可变参数 ---

        @TestParser.test
        def lambda_vararg():
            f = lambda *args: Expr.str(args[0])
            var_r = f(Expr.int(0), Expr.int(1))

        # --- keyword-only 参数 ---

        @TestParser.test
        def lambda_kwonly():
            f = lambda *, key: Expr.str(key)
            var_r = f(key=Expr.int(0))

        @TestParser.test
        def lambda_kwonly_default_not_overridden():
            f = lambda *, key=0: Expr.str(key)
            var_r = f()

        # --- **kwargs ---

        @TestParser.test
        def lambda_kwargs():
            f = lambda **kw: Expr.str(kw['x'])
            var_r = f(x=Expr.int(0))

        # --- body 是常数 ---

        @TestParser.test
        def lambda_body_const():
            f = lambda: 42
            var_x = f()
            Expr.str(var_x)

        # --- body 是 binop ---

        @TestParser.test
        def lambda_body_binop():
            f = lambda x: x + Expr.int(1)
            var_x = f(Expr.int(0))
            Expr.str(var_x)

        # --- body 是 ifexp ---

        @TestParser.test
        def lambda_body_ifexp():
            f = lambda x: Expr.str(0) if Expr.true(x) else Expr.str(1)
            var_r = f(Expr.int(0))

        # --- body 是嵌套调用 ---

        @TestParser.test
        def lambda_body_nested_call():
            f = lambda x: Expr.str(Expr.int(x))
            var_r = f(0)

        # --- 嵌套 lambda: outer 返回 lambda, inner 不加 var_ ---

        @TestParser.test
        def lambda_nested():
            outer = lambda x: lambda y: Expr.str(x)
            inner = outer(Expr.int(0))
            var_r = inner(Expr.int(1))

        # --- lambda 作为高阶函数参数 ---

        @TestParser.test
        def lambda_as_argument():

            def apply(fn, val):
                return fn(val)
            var_r = apply(lambda x: Expr.int(x), Expr.int(0))
            Expr.str(var_r)

        # --- lambda 在列表推导中作为元素 ---

        @TestParser.test
        def lambda_in_listcomp():
            fs = [lambda x=Expr.int(i): Expr.str(x) for i in range(Expr.int(3))]
            for f in fs:
                var_r = f()


def test_pil_builder_return():

    with TestParser():

        # --- bare return (value is None) ---

        @TestParser.test
        def return_bare():

            def func():
                Expr.str(0)
                return
            func()

        # --- return identifier ---

        @TestParser.test
        def return_name():

            def func():
                var_x = Expr.int(0)
                return var_x
            var_r = func()
            Expr.str(var_r)

        # --- return call expression (PIL inserts a temp) ---

        @TestParser.test
        def return_call_expr():

            def func():
                return Expr.int(0)
            var_r = func()
            Expr.str(var_r)

        # --- return constant ---

        @TestParser.test
        def return_constant():

            def func():
                return 42
            var_r = func()
            Expr.str(var_r)

        # --- return binop expression ---

        @TestParser.test
        def return_binop():

            def func():
                return Expr.int(0) + Expr.int(1)
            var_r = func()
            Expr.str(var_r)

        # --- return tuple literal ---

        @TestParser.test
        def return_tuple():

            def func():
                return (Expr.int(0), Expr.int(1))
            var_r = func()
            Expr.str(var_r[0])
            Expr.str(var_r[1])

        # --- return tuple with starred ---

        @TestParser.test
        def return_tuple_starred():

            def make():
                return [Expr.int(1), Expr.int(2)]

            def func():
                return (Expr.int(0), *make())
            var_r = func()
            Expr.str(var_r[0])
            Expr.str(var_r[1])

        # --- return constant tuple (all elements are constants) ---

        @TestParser.test
        def return_const_tuple():

            def func():
                return (0, 1, 2)
            var_r = func()
            Expr.str(var_r[0])

        # --- early return: only one branch executes ---

        @TestParser.test
        def return_early():

            def func(flag):
                if flag:
                    return Expr.int(0)
                return Expr.int(1)
            var_a = func(Expr.true(0))
            Expr.str(var_a)
            var_b = func(Expr.false(1))
            Expr.str(var_b)

        # --- nested function: each level has its own return ---

        @TestParser.test
        def return_nested():

            def outer():

                def inner():
                    return Expr.int(0)
                var_x = inner()
                Expr.str(var_x)
                return Expr.int(1)
            var_r = outer()
            Expr.str(var_r)


def test_pil_builder_call():

    with TestParser():

        # --- 常数: constant literals as positional call arguments ---

        @TestParser.test
        def call_pos_int_const():
            Expr.str(0)

        @TestParser.test
        def call_pos_str_const():
            Expr.str('hello')

        @TestParser.test
        def call_pos_none_const():

            def func(x):
                Expr.str(x)
            func(None)

        @TestParser.test
        def call_pos_bool_const():

            def func(x):
                Expr.str(x)
            func(True)

        @TestParser.test
        def call_pos_multiple_consts():

            def func(x, y):
                Expr.str(x)
                Expr.str(y)
            func(0, 1)

        @TestParser.test
        def call_pos_mixed_const_and_expr():

            def func(x, y):
                Expr.str(x)
                Expr.str(y)
            func(0, Expr.int(1))

        # --- named arg 传常数: named argument with constant value ---

        @TestParser.test
        def call_named_arg_int_const():

            def func(x):
                Expr.str(x)
            func(x=0)

        @TestParser.test
        def call_named_arg_str_const():

            def func(x):
                Expr.str(x)
            func(x='hello')

        @TestParser.test
        def call_named_arg_none_const():

            def func(x):
                Expr.str(x)
            func(x=None)

        @TestParser.test
        def call_named_arg_multiple_consts():

            def func(x, y):
                Expr.str(x)
                Expr.str(y)
            func(x=0, y=1)

        @TestParser.test
        def call_named_arg_mixed_const_and_expr():

            def func(x, y):
                Expr.str(x)
                Expr.str(y)
            func(x=0, y=Expr.int(1))

        # --- 默认参数: keyword arguments overriding defaults ---

        @TestParser.test
        def call_keyword_override_one_default():

            def func(x=0, y=1):
                Expr.str(x)
                Expr.str(y)
            func(x=Expr.int(0))

        @TestParser.test
        def call_keyword_override_all_defaults():

            def func(x=0, y=0):
                Expr.str(x)
                Expr.str(y)
            func(x=Expr.int(0), y=Expr.int(1))

        @TestParser.test
        def call_keyword_arg_expr_value():

            def func(x):
                Expr.str(x)
            func(x=Expr.int(0))

        # --- 可变参数: *args parameter ---

        @TestParser.test
        def call_vararg_empty():

            def func(*args):
                for var_a in args:
                    Expr.str(var_a)
            func()

        @TestParser.test
        def call_vararg_one():

            def func(*args):
                for var_a in args:
                    Expr.str(var_a)
            func(Expr.int(0))

        @TestParser.test
        def call_vararg_many():

            def func(*args):
                for var_a in args:
                    Expr.str(var_a)
            func(Expr.int(0), Expr.int(1), Expr.int(2))

        @TestParser.test
        def call_pos_and_vararg():

            def func(x, *args):
                Expr.str(x)
                for var_a in args:
                    Expr.str(var_a)
            func(Expr.int(0), Expr.int(1), Expr.int(2))

        # --- name only 参数: keyword-only parameters ---

        @TestParser.test
        def call_kwonly_required():

            def func(*, key):
                Expr.str(key)
            func(key=Expr.int(0))

        @TestParser.test
        def call_kwonly_default_not_overridden():

            def func(*, key=0):
                Expr.str(key)
            func()

        @TestParser.test
        def call_kwonly_default_overridden():

            def func(*, key=0):
                Expr.str(key)
            func(key=Expr.int(1))

        @TestParser.test
        def call_pos_and_kwonly():

            def func(x, *, y):
                Expr.str(x)
                Expr.str(y)
            func(Expr.int(0), y=Expr.int(1))

        @TestParser.test
        def call_vararg_and_kwonly():

            def func(*args, key):
                for var_a in args:
                    Expr.str(var_a)
                Expr.str(key)
            func(Expr.int(0), Expr.int(1), key=Expr.int(2))

        # --- keyword参数: **dict expansion ---

        @TestParser.test
        def call_double_star_expand():

            def func(x, y):
                Expr.str(x)
                Expr.str(y)
            var_d = {'x': Expr.int(0), 'y': Expr.int(1)}
            func(**var_d)

        @TestParser.test
        def call_double_star_from_func():

            def make():
                return {'x': Expr.int(0), 'y': Expr.int(1)}

            def func(x, y):
                Expr.str(x)
                Expr.str(y)
            func(**make())

        @TestParser.test
        def call_pos_and_double_star():

            def func(x, y, z):
                Expr.str(x)
                Expr.str(y)
                Expr.str(z)
            var_d = {'y': Expr.int(1), 'z': Expr.int(2)}
            func(Expr.int(0), **var_d)

        @TestParser.test
        def call_keyword_and_double_star():

            def func(x, y, z):
                Expr.str(x)
                Expr.str(y)
                Expr.str(z)
            var_d = {'z': Expr.int(2)}
            func(Expr.int(0), y=Expr.int(1), **var_d)

        # --- 函数调用嵌套场景: nested function calls as arguments ---

        @TestParser.test
        def call_nested_pos_arg():
            Expr.str(Expr.int(0))

        @TestParser.test
        def call_nested_multiple_pos_args():

            def func(x, y):
                Expr.str(x)
                Expr.str(y)
            func(Expr.int(0), Expr.int(1))

        @TestParser.test
        def call_nested_keyword_arg():

            def func(key):
                Expr.str(key)
            func(key=Expr.int(0))

        @TestParser.test
        def call_nested_two_deep():

            def inner():
                return Expr.int(0)
            Expr.str(inner())

        @TestParser.test
        def call_nested_three_deep():

            def inner():
                return Expr.int(0)

            def middle(x):
                return x
            Expr.str(middle(inner()))

        @TestParser.test
        def call_star():

            def inner():
                return [Expr.int(0), Expr.int(1)]

            def middle(a, b):
                return a, b, Expr.int(3)

            middle(*inner())
