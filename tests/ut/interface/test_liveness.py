#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import ast
import sys
from pypto.frontend.parser.liveness import ScopeLivenessAnalyzer, LivenessResult, Scope, VarInfo


def test_rule2_same_scope():
    """Rule 2: Variable defined and used in same scope."""
    code = """
def kernel(a, result):
    b = pypto.matmul(a, a)
    c = pypto.cast(b)
    result[:] = c
"""
    tree = ast.parse(code)
    analyzer = ScopeLivenessAnalyzer()
    result = analyzer.analyze(tree, exempt_vars={'a', 'result'})

    assert 'b' in result.var_info
    assert 'c' in result.var_info
    assert result.scope_lift_suggestions == []


def test_nested_loop_usage():
    """Test bias_2d nested loop scenario."""
    code = """
def kernel(input_tensor, bias_input, output):
    bias_2d = pypto.reshape(bias_input, [1, 16])

    for bs_idx, tile_batch in pypto.loop_unroll(0, 32, 1):
        tile_bias = pypto.tensor([tile_batch, 16], bias_2d.dtype)

        for tmp_idx in pypto.loop(tile_batch):
            pypto.assemble(bias_2d, [tmp_idx, 0], tile_bias)

        tile_result = pypto.add(input_tensor, tile_bias)
        output[:] = tile_result
"""
    tree = ast.parse(code)
    analyzer = ScopeLivenessAnalyzer()
    result = analyzer.analyze(tree, exempt_vars={'input_tensor',
        'bias_input', 'output', 'bs_idx', 'tile_batch', 'tmp_idx'})

    bias_2d_info = result.var_info.get('bias_2d')
    assert bias_2d_info is not None

    delete_scope = result.scope_map.get(bias_2d_info.delete_scope_id)
    if delete_scope:
        assert delete_scope.scope_type == 'root', f"Expected 'root', got {delete_scope.scope_type}"


def test_mixed_usage():
    """Test tile_bias mixed usage scenario."""
    code = """
def kernel(input_tensor, bias_input, output):
    for bs_idx, tile_batch in pypto.loop_unroll(0, 32, 1):
        tile_bias = pypto.tensor([tile_batch, 16])

        for tmp_idx in pypto.loop(tile_batch):
            pypto.assemble(bias_2d, [tmp_idx, 0], tile_bias)

        tile_result = pypto.add(input_tensor, tile_bias)
        output[:] = tile_result
"""
    tree = ast.parse(code)
    analyzer = ScopeLivenessAnalyzer()
    result = analyzer.analyze(tree, exempt_vars={'input_tensor',
        'bias_input', 'output', 'bs_idx', 'tile_batch', 'tmp_idx'})

    tile_bias_info = result.var_info.get('tile_bias')
    assert tile_bias_info is not None
    assert tile_bias_info.delete_after_scope_exit == False


def test_scope_lift():
    """Test Rule B: Scope lifting for non-nested usage."""
    code = """
def kernel(a, result):
    if True:
        b = pypto.add(a, a)
    else:
        b = pypto.sub(a, a)
    c = pypto.cast(b)
    result[:] = c
"""
    tree = ast.parse(code)
    analyzer = ScopeLivenessAnalyzer()
    result = analyzer.analyze(tree, exempt_vars={'a', 'result'})

    assert 'b' in result.var_info
    b_info = result.var_info['b']
    delete_scope = result.scope_map.get(b_info.delete_scope_id)
    assert delete_scope.scope_type == 'root'


def test_all_vars_recorded():
    """Test that ALL variables are recorded."""
    code = """
def kernel(a, result):
    T = 4
    D = 256
    for i in range(T):
        temp = pypto.view(a, [D], [i])
        result[:] = temp
"""
    tree = ast.parse(code)
    analyzer = ScopeLivenessAnalyzer()
    result = analyzer.analyze(tree, exempt_vars={'a', 'result', 'i'})

    assert 'T' in result.var_info
    assert 'D' in result.var_info
    assert 'temp' in result.var_info


def run_all_tests():
    """Run all tests."""
    test_rule2_same_scope()
    test_nested_loop_usage()
    test_mixed_usage()
    test_scope_lift()
    test_all_vars_recorded()


if __name__ == "__main__":
    run_all_tests()