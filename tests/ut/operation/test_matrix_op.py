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


def test_matrix_matmul():
    dtype = pypto.DT_FP32
    a = pypto.tensor((32, 64), dtype, "A")
    b = pypto.tensor((64, 32), dtype, "B")
    c = None

    with pypto.function("MATMUL", a, b):
        pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
        c = pypto.matmul(a, b, dtype)
        d = pypto.matmul(a, b, dtype, a_trans=True, b_trans=True)

    assert isinstance(c, pypto.tensor)
    assert c.shape == [32, 32]

    assert isinstance(d, pypto.tensor)
    assert d.shape == [64, 64]


def test_matrix_batch_matmul():
    dtype = pypto.DT_FP32
    a = pypto.tensor((2, 64, 32), dtype, "A")
    b = pypto.tensor((2, 32, 64), dtype, "B")
    c = None

    with pypto.function("BATCH_MATMUL", a, b):
        pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
        c = pypto.matmul(a, b, dtype)
        d = pypto.matmul(a, b, dtype, a_trans=True, b_trans=True)

    assert isinstance(c, pypto.tensor)
    assert c.shape == [2, 64, 64]

    assert isinstance(d, pypto.tensor)
    assert d.shape == [2, 32, 32]


def test_matrix_matmul_with_syntactic_sugar():
    dtype = pypto.DT_FP16
    a = pypto.tensor((64, 32), dtype, "A")
    b = pypto.tensor((32, 64), dtype, "B")
    c = None

    with pypto.function("MATMUL", a, b):
        pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
        c = a @ b

    assert isinstance(c, pypto.tensor)
    assert c.dtype == pypto.DT_FP16
    assert c.shape == [64, 64]


def test_matrix_matmul_with_tensor_interface():
    input_dtype = pypto.DT_INT8
    out_dtype = pypto.DT_INT32
    a = pypto.tensor((3, 64, 32), input_dtype, "A")
    b = pypto.tensor((3, 32, 64), input_dtype, "B")
    c = None

    with pypto.function("BATCH_MATMUL", a, b):
        pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
        c = a.matmul(b, out_dtype, a_trans=True, b_trans=True)

    assert isinstance(c, pypto.tensor)
    assert c.dtype == pypto.DT_INT32
    assert c.shape == [3, 32, 32]


def test_matrix_matmul_with_trans_mode_cast_rint():
    input_dtype = pypto.DT_FP32
    out_dtype = pypto.DT_FP32
    a = pypto.tensor((64, 32), input_dtype, "A")
    b = pypto.tensor((32, 64), input_dtype, "B")
    c = None

    with pypto.function("MATMUL", a, b):
        pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
        c = pypto.matmul(a, b, out_dtype, extend_params={"trans_mode": pypto.TransMode.CAST_RINT})

    assert isinstance(c, pypto.tensor)
    assert c.shape == [64, 64]


def test_matrix_matmul_with_trans_mode_cast_rint():
    input_dtype = pypto.DT_FP32
    out_dtype = pypto.DT_FP32
    a = pypto.tensor((64, 32), input_dtype, "A")
    b = pypto.tensor((32, 64), input_dtype, "B")
    c = None

    with pypto.function("MATMUL", a, b):
        pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
        c = pypto.matmul(a, b, out_dtype, extend_params={"trans_mode": pypto.TransMode.CAST_ROUND})

    assert isinstance(c, pypto.tensor)
    assert c.shape == [64, 64]
