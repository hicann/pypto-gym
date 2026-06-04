#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 CANN community contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Unit tests for pypto._error module.

This test suite verifies that error handling works correctly in PyPTO by
testing error scenarios that trigger different error types through the frontend.
"""

import pytest
import pypto
import torch

from pypto.error import ParserError, PyptoError, PyptoGeneralError, _catch_and_wrap_error
import pypto.config


def test_varargs_error():
    """Test that variable-length arguments trigger proper error handling."""
    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
    def varargs_kernel(
        x: pypto.Tensor([], pypto.DT_FP32),
        out: pypto.Tensor([], pypto.DT_FP32),
        *args):
        out[:] = x
    
    x = torch.randn(4, 4, dtype=torch.float32)
    out = torch.zeros(4, 4, dtype=torch.float32)
    
    with pytest.raises(ParserError, match="Variable-length arguments"):
        varargs_kernel(x, out)


def test_kwargs_error():
    """Test that keyword arguments trigger proper error handling."""
    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
    def kwargs_kernel(
        x: pypto.Tensor([], pypto.DT_FP32),
        out: pypto.Tensor([], pypto.DT_FP32),
        **kwargs):
        out[:] = x
    
    x = torch.randn(4, 4, dtype=torch.float32)
    out = torch.zeros(4, 4, dtype=torch.float32)
    
    with pytest.raises(ParserError, match="Keyword argument packing"):
        kwargs_kernel(x, out)


def test_error_message_contains_error_code():
    """Test that error messages contain error codes."""
    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
    def varargs_kernel(
        x: pypto.Tensor([], pypto.DT_FP32),
        out: pypto.Tensor([], pypto.DT_FP32),
        *args):
        out[:] = x
    
    x = torch.randn(4, 4, dtype=torch.float32)
    out = torch.zeros(4, 4, dtype=torch.float32)
    
    try:
        varargs_kernel(x, out)
        assert False, "Should have raised an error"
    except Exception as e:
        error_str = str(e)
        assert "ErrCode: F00005" in error_str
        assert len(error_str) > 0


def test_pypto_error_init():
    err = PyptoError(0xF00001, "test error")
    assert "ErrCode: F00001" in str(err)
    assert "test error" in str(err)


def test_pypto_error_no_duplicate_errcode():
    err = PyptoError(0xF00001, "ErrCode: F00003, original error")
    assert "ErrCode: F00003" in str(err)
    assert str(err).count("ErrCode:") == 1


def test_catch_and_wrap_error_normal():
    @_catch_and_wrap_error("test operation")
    def normal_func(x):
        return x * 2
    
    assert normal_func(5) == 10


def test_catch_and_wrap_error_wraps_exception():
    @_catch_and_wrap_error("test operation")
    def failing_func():
        raise ValueError("test error")
    
    with pytest.raises(PyptoGeneralError) as exc_info:
        failing_func()
    
    assert "Failed to test operation" in str(exc_info.value)
    assert "test error" in str(exc_info.value)


def test_catch_and_wrap_error_preserves_errcode():
    @_catch_and_wrap_error("test operation")
    def failing_func():
        raise RuntimeError("ErrCode: F00003, some error")
    
    with pytest.raises(PyptoGeneralError) as exc_info:
        failing_func()
    
    assert "ErrCode: F00003" in str(exc_info.value)


def test_error_on_input_tensor_reassign():
    """Test that reassigning input tensor triggers ParserError."""
    @pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.SIM},
        host_options={"compile_stage": pypto.CompStage.TENSOR_GRAPH}
    )
    def error_assign_input(
        a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
        b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
        c: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    ):
        pypto.set_vec_tile_shapes(32, 32)
        c = a + b
    
    a = torch.rand((32, 32), dtype=torch.float16)
    b = torch.rand((32, 32), dtype=torch.float16)
    c = torch.zeros((32, 32), dtype=torch.float16)
    
    with pytest.raises(ParserError, match="Input tensor 'c' cannot be reassigned"):
        error_assign_input(a, b, c)


def test_error_on_first_input_tensor_reassign():
    """Test that reassigning the first input tensor triggers ParserError."""
    @pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.SIM},
        host_options={"compile_stage": pypto.CompStage.TENSOR_GRAPH}
    )
    def error_assign_first_input(
        a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
        b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
        c: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    ):
        pypto.set_vec_tile_shapes(32, 32)
        a = b + c
    
    a = torch.rand((32, 32), dtype=torch.float16)
    b = torch.rand((32, 32), dtype=torch.float16)
    c = torch.zeros((32, 32), dtype=torch.float16)
    
    with pytest.raises(ParserError, match="Input tensor 'a' cannot be reassigned"):
        error_assign_first_input(a, b, c)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
