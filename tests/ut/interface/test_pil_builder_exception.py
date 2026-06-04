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


def test_pil_builder_raise():

    with TestParser():

        # --- bare raise (re-raise inside except) ---

        @TestParser.test
        def raise_bare():
            try:
                raise Expr.TypeA(0)
            except Expr.TypeA:
                try:
                    raise
                except Expr.TypeA:
                    Expr.str(1)

        @TestParser.test
        def raise_bare_simple():
            try:
                try:
                    raise Expr.TypeA(0)
                except Expr.TypeA:
                    raise
            except Expr.TypeA as e:
                Expr.int(e.value)

        # --- raise a name ---

        @TestParser.test
        def raise_name():
            try:
                exc = Expr.TypeA(0)
                raise exc
            except Expr.TypeA as e:
                Expr.int(e.value)

        # --- raise an attribute expression ---

        @TestParser.test
        def raise_attribute():
            var_obj = Expr(0)
            var_obj.val = Expr.TypeB(1)
            try:
                raise var_obj.val
            except Expr.TypeB as e:
                Expr.int(e.value)

        # --- raise a subscript expression ---

        @TestParser.test
        def raise_subscript():
            var_obj = Expr(0)
            var_obj[Expr.str(1)] = Expr.TypeC(2)
            try:
                raise var_obj[Expr.str(1)]
            except Expr.TypeC as e:
                Expr.int(e.value)

        # --- raise an ifexp expression ---

        @TestParser.test
        def raise_ifexp_true():
            try:
                raise Expr.TypeA(0) if Expr.true(1) else Expr.TypeB(2)
            except Expr.TypeA as e:
                Expr.int(e.value)

        @TestParser.test
        def raise_ifexp_false():
            try:
                raise Expr.TypeA(0) if Expr.false(1) else Expr.TypeB(2)
            except Expr.TypeB as e:
                Expr.int(e.value)

        # --- raise from (chain cause with a name) ---

        @TestParser.test
        def raise_from_name():
            try:
                cause = Expr.TypeA(0)
                raise Expr.TypeB(1) from cause
            except Expr.TypeB as e:
                Expr.int(e.value)

        # --- raise from with attribute as cause ---

        @TestParser.test
        def raise_from_attribute():
            var_obj = Expr(0)
            var_obj.val = Expr.TypeA(1)
            try:
                raise Expr.TypeB(2) from var_obj.val
            except Expr.TypeB as e:
                Expr.int(e.value)

        # --- raise with attribute exc, subscript cause ---

        @TestParser.test
        def raise_attribute_from_subscript():
            var_obj = Expr(0)
            var_obj.val = Expr.TypeB(1)
            var_obj[Expr.str(2)] = Expr.TypeA(3)
            try:
                raise var_obj.val from var_obj[Expr.str(2)]
            except Expr.TypeB as e:
                Expr.int(e.value)


def test_pil_builder_try():

    with TestParser():

        # --- single typed handler, no binding ---

        @TestParser.test
        def try_single_typed_no_binding():
            try:
                raise Expr.TypeA(0)
            except Expr.TypeA:
                Expr.str(1)

        # --- single typed handler, with binding ---

        @TestParser.test
        def try_single_typed_with_binding():
            try:
                raise Expr.TypeA(0)
            except Expr.TypeA as e:
                Expr.str(e.value)

        # --- multiple typed handlers, dispatch to correct branch ---

        @TestParser.test
        def try_multi_handler_typea():
            try:
                raise Expr.TypeA(0)
            except Expr.TypeA:
                Expr.str(10)
            except Expr.TypeB:
                Expr.str(20)
            except Expr.TypeC:
                Expr.str(30)

        @TestParser.test
        def try_multi_handler_typeb():
            try:
                raise Expr.TypeB(0)
            except Expr.TypeA:
                Expr.str(10)
            except Expr.TypeB:
                Expr.str(20)
            except Expr.TypeC:
                Expr.str(30)

        @TestParser.test
        def try_multi_handler_typec():
            try:
                raise Expr.TypeC(0)
            except Expr.TypeA:
                Expr.str(10)
            except Expr.TypeB:
                Expr.str(20)
            except Expr.TypeC:
                Expr.str(30)

        # --- bare except catches everything ---

        @TestParser.test
        def try_bare_except():
            try:
                raise Expr.TypeA(0)
            except Expr.TypeB:
                Expr.str(10)
            except:
                Expr.str(20)

        # --- else branch fires when no exception ---

        @TestParser.test
        def try_else_no_exception():
            try:
                Expr.str(0)
            except Expr.TypeA:
                Expr.str(10)
            else:
                Expr.str(20)

        @TestParser.test
        def try_else_with_exception():
            try:
                if Expr.true(0):
                    raise Expr.TypeA(1)
            except Expr.TypeA:
                Expr.str(10)
            else:
                Expr.str(20)

        # --- finally always fires ---

        @TestParser.test
        def try_finally_no_exception():
            try:
                Expr.str(0)
            finally:
                Expr.str(1)

        @TestParser.test
        def try_finally_with_exception():
            try:
                raise Expr.TypeA(0)
            except Expr.TypeA:
                Expr.str(10)
            finally:
                Expr.str(20)

        # --- combined: multiple handlers + else + finally ---

        @TestParser.test
        def try_combined_no_exception():
            try:
                Expr.str(0)
            except Expr.TypeA as e:
                Expr.int(e.value)
            except Expr.TypeB as e:
                Expr.str(e.value)
            else:
                Expr.str(30)
            finally:
                Expr.str(40)

        @TestParser.test
        def try_combined_typea():
            try:
                if Expr.true(0):
                    raise Expr.TypeA(1)
            except Expr.TypeA as e:
                Expr.int(e.value)
            except Expr.TypeB as e:
                Expr.str(e.value)
            else:
                Expr.str(30)
            finally:
                Expr.str(40)

        @TestParser.test
        def try_combined_typeb():
            try:
                if Expr.true(0):
                    raise Expr.TypeB(1)
            except Expr.TypeA as e:
                Expr.int(e.value)
            except Expr.TypeB as e:
                Expr.str(e.value)
            else:
                Expr.str(30)
            finally:
                Expr.str(40)

        # --- nested try ---

        @TestParser.test
        def try_nested():
            try:
                try:
                    raise Expr.TypeA(0)
                except Expr.TypeB:
                    Expr.str(10)
            except Expr.TypeA as e:
                Expr.int(e.value)


def test_pil_builder_assert():

    with TestParser():

        # --- assert passes: subscript condition, no message ---

        @TestParser.test
        def assert_pass_subscript():
            var_obj = Expr(0)
            var_obj[Expr.str(1)] = Expr.true(2)
            assert var_obj[Expr.str(1)]

        # --- assert passes: subscript condition, message is function call (not evaluated) ---

        @TestParser.test
        def assert_pass_subscript_msg_call():
            var_obj = Expr(0)
            var_obj[Expr.str(1)] = Expr.true(2)
            assert var_obj[Expr.str(1)], Expr.str(3)

        # --- assert passes: subscript condition, message is non-function-call (not evaluated) ---

        @TestParser.test
        def assert_pass_subscript_msg_name():
            var_msg = 'assertion failed'
            var_obj = Expr(0)
            var_obj[Expr.str(1)] = Expr.true(2)
            assert var_obj[Expr.str(1)], var_msg

        # --- assert fails: subscript condition, no message, raises AssertionError ---

        @TestParser.test
        def assert_fail_subscript_no_msg():
            var_obj = Expr(0)
            var_obj[Expr.str(1)] = Expr.false(2)
            try:
                assert var_obj[Expr.str(1)]
            except AssertionError:
                Expr.str(3)

        # --- assert fails: subscript condition, message is function call, raises AssertionError ---

        @TestParser.test
        def assert_fail_subscript_msg_call():
            var_obj = Expr(0)
            var_obj[Expr.str(1)] = Expr.false(2)
            try:
                assert var_obj[Expr.str(1)], Expr.str(3)
            except AssertionError:
                Expr.str(4)

        # --- assert fails: subscript condition, message is non-function-call, raises AssertionError ---

        @TestParser.test
        def assert_fail_subscript_msg_name():
            var_msg = 'assertion failed'
            var_obj = Expr(0)
            var_obj[Expr.str(1)] = Expr.false(2)
            try:
                assert var_obj[Expr.str(1)], var_msg
            except AssertionError:
                Expr.str(3)
