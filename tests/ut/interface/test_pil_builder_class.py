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


def test_pil_builder_class():

    with TestParser():

        # ================================================================
        # Part 1: single-op comparisons with each operator
        # ================================================================

        @TestParser.test
        def cls_basic():

            class Cls:
                a = Expr.int(20) + Expr.int(30)
                b = a + Expr.int(20)
                c = b + a
            var_a = Cls.a
            var_b = Cls.b
            var_c = Cls.c

        @TestParser.test
        def cls_decorate():

            var_e = Expr(0)

            @var_e.decorate(1)
            @var_e.decorate(2)
            class Cls:
                a = Expr.int(20) + Expr.int(30)
                b = a + Expr.int(20)
                c = b + a
            var_a = Cls.a
            var_b = Cls.b
            var_c = Cls.c


        @TestParser.test
        def cls_inherit():

            var_e = Expr(0)

            class Base:
                d = Expr.int(20) + Expr.int(30)

            e = [Base]

            @var_e.decorate(1)
            @var_e.decorate(2)
            class Cls(e[0]):
                a = Expr.int(20) + Expr.int(30)
                b = a + Expr.int(20)
                c = b + a
            var_a = Cls.a
            var_b = Cls.b
            var_c = Cls.c
            var_d = Cls.d

        @TestParser.test
        def cls_inherit_list():

            var_e = Expr(0)

            class Base:
                d = Expr.int(20) + Expr.int(30)

            class Base2:
                d = Expr.int(100) + Expr.int(110)

            e = [Base, Base2]

            @var_e.decorate(1)
            @var_e.decorate(2)
            class Cls(*e):
                a = Expr.int(320) + Expr.int(330)
                b = a + Expr.int(420)
                c = b + a
            var_a = Cls.a
            var_b = Cls.b
            var_c = Cls.c
            var_d = Cls.d


def test_pil_builder_with():

    with TestParser():

        # ================================================================
        # Part 1: single-op comparisons with each operator
        # ================================================================

        @TestParser.test
        def with_basic():

            class Cls:
                @classmethod
                def __enter__(cls):
                    return Expr.str(20) + Expr.str(30)

                @classmethod
                def __exit__(cls, exc_type, exc_value, traceback):
                    Expr.int(40)

            with Cls():
                var_g = Expr.int(20)


        @TestParser.test
        def with_target():

            class Cls:
                @classmethod
                def __enter__(cls):
                    return Expr.str(20) + Expr.str(30)

                @classmethod
                def __exit__(cls, exc_type, exc_value, traceback):
                    Expr.int(40)

            with Cls() as var_f:
                var_g = var_f[0]


        @TestParser.test
        def with_multiple_target():

            class Cls:
                @classmethod
                def __enter__(cls):
                    return Expr.str(20) + Expr.str(30)

                @classmethod
                def __exit__(cls, exc_type, exc_value, traceback):
                    Expr.int(40)

            with Cls() as var_f0, Cls() as var_f1:
                var_g0 = var_f0[0]
                var_g1 = var_f1[1]
