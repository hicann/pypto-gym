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
"""
"""
import pytest
import pypto
import torch

from pypto.error import ParserError


def test_return_statement_error():
    """Test that return statements in kernel functions raise ParserError."""
    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
    def kernel_with_return(
        x: pypto.Tensor([4, 4], pypto.DT_FP32),
        out: pypto.Tensor([4, 4], pypto.DT_FP32)):
        out[:] = x
        return out
    
    x = torch.randn(4, 4, dtype=torch.float32)
    out = torch.zeros(4, 4, dtype=torch.float32)
    
    with pytest.raises(ParserError, match="Return statements are not allowed"):
        kernel_with_return(x, out)


def test_return_nested_in_if_error():
    """Test that return statements nested in if structures raise ParserError."""
    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
    def kernel_with_return_in_if(
        x: pypto.Tensor([4, 4], pypto.DT_FP32),
        flag: pypto.Tensor([], pypto.DT_INT32),
        out: pypto.Tensor([4, 4], pypto.DT_FP32)):
        out[:] = x
        if flag > 0:
            return out
        return out
    
    x = torch.randn(4, 4, dtype=torch.float32)
    flag = torch.tensor(1, dtype=torch.int32)
    out = torch.zeros(4, 4, dtype=torch.float32)
    
    with pytest.raises(ParserError, match="Return statements are not allowed"):
        kernel_with_return_in_if(x, flag, out)


def test_return_nested_in_for_error():
    """Test that return statements nested in for loops raise ParserError."""
    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
    def kernel_with_return_in_for(
        x: pypto.Tensor([4, 4], pypto.DT_FP32),
        out: pypto.Tensor([4, 4], pypto.DT_FP32)):
        for i in range(4):
            out[i, :] = x[i, :]
            if i == 3:
                return out
        return out
    
    x = torch.randn(4, 4, dtype=torch.float32)
    out = torch.zeros(4, 4, dtype=torch.float32)
    
    with pytest.raises(ParserError, match="Return statements are not allowed"):
        kernel_with_return_in_for(x, out)


def test_break_statement_error():
    """Test that break statements in loops raise ParserError."""
    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
    def kernel_with_break(
        x: pypto.Tensor([4, 4], pypto.DT_FP32),
        out: pypto.Tensor([4, 4], pypto.DT_FP32)):
        for i in range(4):
            if i == 2:
                break
    
    x = torch.randn(4, 4, dtype=torch.float32)
    out = torch.zeros(4, 4, dtype=torch.float32)
    
    with pytest.raises(ParserError, match="Break is not supported by the PTO parser"):
        kernel_with_break(x, out)


def test_continue_statement_error():
    """Test that continue statements in loops raise ParserError."""
    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
    def kernel_with_continue(
        x: pypto.Tensor([4, 4], pypto.DT_FP32),
        out: pypto.Tensor([4, 4], pypto.DT_FP32)):
        for i in range(4):
            if i % 2 == 0:
                continue
            out[i, :] = x[i, :]
    
    x = torch.randn(4, 4, dtype=torch.float32)
    out = torch.zeros(4, 4, dtype=torch.float32)
    
    with pytest.raises(ParserError, match="Continue is not supported by the PTO parser"):
        kernel_with_continue(x, out)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])