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

import ast
import inspect
import logging
import textwrap

import pypto.frontend.parser.pil as pil


LOGGER = logging.getLogger(__name__)


class Expr:

    trace = []

    def __init__(self, value):
        self._item_dict = {}
        self._attr_dict = {}
        self._value = value
        Expr.trace.append(('init', self._value))

    def __getitem__(self, item):
        Expr.trace.append(('getitem', self._value, item))
        return self._item_dict[Expr._normalize_item_key(item)]

    def __setitem__(self, item, value):
        Expr.trace.append(('setitem', self._value, item, value))
        self._item_dict[Expr._normalize_item_key(item)] = value

    def __delitem__(self, item):
        Expr.trace.append(('delitem', self._value, item))
        del self._item_dict[Expr._normalize_item_key(item)]

    def __eq__(self, other):
        return (
            self._attr_dict == other._attr_dict
            and self._item_dict == other._item_dict
            and self._value == other._value
        )

    @property
    def value(self):
        return self._value

    @property
    def attr_dict(self):
        return self._attr_dict

    @staticmethod
    def clear():
        Expr.trace.clear()

    @staticmethod
    def true(n):
        Expr.trace.append(('true', n))
        return True

    @staticmethod
    def false(n):
        Expr.trace.append(('false', n))
        return False

    @staticmethod
    def str(n):
        Expr.trace.append(('str', n))
        return f'str({n})'

    @staticmethod
    def int(n):
        Expr.trace.append(('int', n))
        return n


    @staticmethod
    def decorate(n):
        Expr.trace.append(('decorate', n))

        def wrapper(func):
            Expr.trace.append(('decorate.wrapper', n))
            return func
        return wrapper

    @staticmethod
    def attr(method_dict, name):

        @property
        def field(self):
            Expr.trace.append(('getattr', self.value, name))
            return self.attr_dict[name]

        @field.setter
        def field(self, value):
            Expr.trace.append(('setattr', self.value, name, value))
            self.attr_dict[name] = value

        @field.deleter
        def field(self):
            Expr.trace.append(('delattr', self.value, name))
            del self.attr_dict[name]

        method_dict[name] = field

    @staticmethod
    def _normalize_item_key(item):
        if isinstance(item, slice):
            return ('slice',
                    Expr._normalize_item_key(item.start),
                    Expr._normalize_item_key(item.stop),
                    Expr._normalize_item_key(item.step))
        if isinstance(item, tuple):
            return tuple(Expr._normalize_item_key(sub_item) for sub_item in item)
        return item

    attr.__func__(locals(), 'val')

    class ContextManager:

        def __init__(self, enter_n=None, exit_n=None, init_n=None):
            self._enter_n = enter_n
            self._exit_n = exit_n
            if init_n is not None:
                Expr.str(init_n)

        def __enter__(self):
            if self._enter_n is not None:
                Expr.str(self._enter_n)
            return self

        def __exit__(self, *a):
            if self._exit_n is not None:
                Expr.str(self._exit_n)

        def __eq__(self, other):
            return self._enter_n == other._enter_n and self._exit_n == other._exit_n

    class ValueError(Exception):

        def __init__(self, value):
            super().__init__(value)
            Expr.trace.append(('error', value))
            self._value = value

        def __eq__(self, other):
            return self._value == other._value

        @property
        def value(self):
            return self._value


    class TypeA(ValueError):
        pass

    class TypeB(ValueError):
        pass

    class TypeC(ValueError):
        pass


class TestParser:
    __test__ = False

    target_list = []

    def __init__(self):
        Expr.trace.clear()

        TestParser.target_list = []

    @staticmethod
    def __enter__():
        TestParser.target_list.clear()

    def __exit__(self, exc_type, exc, tb):
        self.run()

    @staticmethod
    def test(target):
        TestParser.target_list.append(target)

    @staticmethod
    def run_test(global_dict):
        for test_name in global_dict:
            if test_name.startswith('test_'):
                global_dict[test_name]()

    @staticmethod
    def run_ast(stmt_list):
        Expr.clear()
        src = ast.unparse(stmt_list)
        exec_global = {'Expr': Expr}
        try:
            LOGGER.debug('%s', '-' * 100)
            LOGGER.debug('%s', src)
            exec(src, exec_global)
        except Exception:
            LOGGER.exception(
                "Failed to exec generated source:\n%s",
                '\n'.join([f'{lineno + 1:3d} | {line}' for lineno, line in enumerate(src.strip().split('\n'))]),
            )
            raise
        run_trace = Expr.trace[:]
        run_vardict = {name: value for name, value in exec_global.items() if name.startswith('var_')}
        return run_trace, run_vardict

    def run(self):
        for target in TestParser.target_list:
            source_lines, _ = inspect.getsourcelines(target)
            source = textwrap.dedent(''.join(source_lines))
            stmt_list = ast.parse(source).body[0].body

            python_trace, python_vardict = self.run_ast(stmt_list)
            pil_trace, pil_vardict = self.run_ast(pil.parse_stmts(stmt_list))

            assert python_trace == pil_trace, f'{target.__name__}: {python_trace=} {pil_trace=}'
            assert python_vardict == pil_vardict, f'{target.__name__}: {python_vardict=} {pil_vardict=}'
