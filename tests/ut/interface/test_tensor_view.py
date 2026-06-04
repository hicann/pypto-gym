#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
"""
import pypto
import pytest


def init_tensors():
    dtype = pypto.DT_FP32
    shape = (128, 128)
    a = pypto.tensor(shape, dtype, "a")
    b = pypto.tensor(shape, dtype, "b")
    c = pypto.tensor(shape, dtype, "c")
    return a, b, c


def test_tensor_view():
    a, b, c = init_tensors()
    with pypto.function("MAIN", a, b, c):
        pypto.set_vec_tile_shapes(16, 16)

        for k in pypto.loop(10):
            a_view = a[k * 16:(k + 1) * 16, k * 16:(k + 1) * 16]
            b_view = b[:16, :16]

            assert isinstance(a_view, pypto.tensor)
            assert isinstance(b_view, pypto.tensor)
            assert a_view.shape == [16, 16]
            assert b_view.shape == [16, 16]


def test_tensor_get_tensor_data():
    a = pypto.tensor((128, 128), pypto.DT_INT32, "a")
    with pypto.function("MAIN", a):
        pypto.set_vec_tile_shapes(16, 16)
        t = a[0, 0]


def test_slice_neg_index():
    """Test negative index"""
    x_shape = [4, 8]
    dtype = pypto.DT_FP32
    x = pypto.tensor(x_shape, dtype)

    with pypto.function("SLICE_NEG_INDEX", [x]):
        pypto.set_vec_tile_shapes(4, 8)
        res = x[-3:-1, -2:-1]
        assert res.shape == [2, 1]


def test_slice_int_index():
    """Test mix use of slice and int"""
    x_shape = [4, 8, 8, 8, 8]
    dtype = pypto.DT_FP32
    x = pypto.tensor(x_shape, dtype)

    with pypto.function("SLICE_INT_INDEX", [x]):
        pypto.set_vec_tile_shapes(4, 4, 4, 4, 8)
        res = x[-2, -3:8, :, 1:4, 2]
        assert res.shape == [3, 8, 3]


def test_slice_ellipsis_index():
    """Test mix use of ellipsis, slice and int"""
    x_shape = [4, 8, 8, 8]
    dtype = pypto.DT_FP32
    x = pypto.tensor(x_shape, dtype)

    with pypto.function("SLICE_INT_ELLIPSIS_INDEX", [x]):
        pypto.set_vec_tile_shapes(4, 4, 4, 8)
        res1 = x[..., 2]
        res2 = x[1:2, :, ..., 3:5]
        res3 = x[2, 3, ...]
        res4 = x[...] + 0.0
        assert res1.shape == [4, 8, 8]
        assert res2.shape == [1, 8, 8, 2]
        assert res3.shape == [8, 8]
        assert res4.shape == [4, 8, 8, 8]


def test_tensor_batch_assemble():
    a = pypto.tensor((32, 32), pypto.DT_FP32, "a")
    b = pypto.tensor((32, 32), pypto.DT_FP32, "b")
    c = pypto.tensor((64, 64), pypto.DT_FP32, "c")
    with pypto.function("MAIN", a, b, c):
        pypto.set_vec_tile_shapes(32, 32)
        pypto.assemble([
            (a, (0, 0)),
            (b, (0, 32)),
        ], c, parallel=True)


def test_view_dimension_mismatch_shapes():
    """Test that shapes dimension mismatch with operand raises error (LogicalTensor constraint)"""
    x_shape = [4, 8, 16]
    dtype = pypto.DT_FP32
    x = pypto.tensor(x_shape, dtype)
    
    view_shape = [2, 4]
    offset = [0, 0]
    
    with pytest.raises(pypto.error.FeError):
        with pypto.function("VIEW_DIM_MISMATCH_SHAPES", x):
            pypto.set_vec_tile_shapes(4, 8, 16)
            pypto.view(x, view_shape, offset)


def test_view_validshape_exceeds_input_shape():
    """Test that valid_shape exceeding input tensor shape raises error"""
    x_shape = [4, 8]
    dtype = pypto.DT_FP32
    x = pypto.tensor(x_shape, dtype)
    
    view_shape = [4, 8]
    offset = [0, 0]
    valid_shape = [5, 8]
    
    with pytest.raises(pypto.error.FeError):
        with pypto.function("VIEW_VALIDSHAPE_EXCEEDS_INPUT", x):
            pypto.set_vec_tile_shapes(4, 8)
            pypto.view(x, view_shape, offset, valid_shape=valid_shape)


def test_view_validshape_dimension_mismatch():
    """Test that valid_shape dimension mismatch with input raises error"""
    x_shape = [4, 8]
    dtype = pypto.DT_FP32
    x = pypto.tensor(x_shape, dtype)
    
    view_shape = [4, 8]
    offset = [0, 0]
    valid_shape = [4]
    
    with pytest.raises(pypto.error.FeError):
        with pypto.function("VIEW_VALIDSHAPE_DIM_MISMATCH", x):
            pypto.set_vec_tile_shapes(4, 8)
            pypto.view(x, view_shape, offset, valid_shape=valid_shape)