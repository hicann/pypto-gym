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
import torch
import pypto
import pytest


def test_init_tensor():
    dtype = pypto.DT_FP16
    shape = [32, 1]

    a = pypto.tensor(shape, dtype, "a")
    assert a.name == "a"
    assert a.dtype == dtype
    assert a.shape == shape
    assert a.dim == len(shape)
    assert a.format == pypto.TileOpFormat.TILEOP_ND

    b = pypto.tensor([-1, 32], dtype, "b", pypto.TileOpFormat.TILEOP_NZ)
    assert b.dtype == dtype
    assert b.name == "b"
    assert b.format == pypto.TileOpFormat.TILEOP_NZ
    with pytest.raises(ValueError):
        # dynamic shape could not be compared
        assert b.shape == [-1, 32]


def test_init_tensor_no_name():
    expected_dtype = pypto.DT_FP16
    shape = [32, 1]
    tensor = pypto.tensor(shape, expected_dtype)

    assert tensor.dtype == expected_dtype
    assert tensor.shape == shape


def test_tensor_add_plus_op():
    dtype = pypto.DT_FP16
    shape = [8, 8]
    a = pypto.tensor(shape, dtype, "tensor_a")
    b = pypto.tensor(shape, dtype, "tensor_b")

    with pypto.function("ADD", a, b):
        pypto.set_vec_tile_shapes(8, 8)
        c = a + b

    assert c.shape == shape
    assert c.dtype == dtype


def test_tensor_add_tensor_element():
    dtype = pypto.DT_FP16
    shape = [8, 8]
    a = pypto.tensor(shape, dtype, "tensor_a")

    with pypto.function("ADD", a):
        pypto.set_vec_tile_shapes(8, 8)
        c = a + 3.14

    assert c.shape == shape
    assert c.dtype == dtype


def test_tensor_add_element_tensor():
    dtype = pypto.DT_FP16
    shape = [8, 8]
    a = pypto.tensor(shape, dtype, "tensor_a")

    with pypto.function("ADD", a):
        pypto.set_vec_tile_shapes(8, 8)
        c = 3.14 + a

    assert c.shape == shape
    assert c.dtype == dtype


def test_tensor_sub_op():
    dtype = pypto.DT_FP16
    shape = [8, 8]
    a = pypto.tensor(shape, dtype, "tensor_a")
    b = pypto.tensor(shape, dtype, "tensor_b")

    with pypto.function("SUB", a, b):
        pypto.set_vec_tile_shapes(8, 8)
        c = a - b

    assert c.shape == shape
    assert c.dtype == dtype


def test_tensor_subs_tensor_element():
    dtype = pypto.DT_FP16
    shape = [8, 8]
    a = pypto.tensor(shape, dtype, "tensor_a")

    with pypto.function("SUBS", a):
        pypto.set_vec_tile_shapes(8, 8)
        c = a - 3.14
        assert [x.concrete() for x in c.valid_shape] == shape

    assert c.shape == shape
    assert c.dtype == dtype


def test_tensor_nz_format():
    a = pypto.from_torch(torch.randn((64, 64)), "a", tensor_format=pypto.TileOpFormat.TILEOP_NZ)
    assert a.format == pypto.TileOpFormat.TILEOP_NZ

    b = pypto.from_torch(torch.randn((64, 64)), "b")
    assert b.format == pypto.TileOpFormat.TILEOP_ND

    c = pypto.from_torch(torch.randn((64, 64)), "c", tensor_format=pypto.TileOpFormat.TILEOP_ND)
    assert c.format == pypto.TileOpFormat.TILEOP_ND


def test_tensor_starred():
    dtype = pypto.DT_INT32
    shape = [64, 64]
    a = pypto.tensor(shape, dtype, "a")
    c1 = pypto.tensor(shape, dtype, "c1")
    c2 = pypto.tensor(shape, dtype, "c2")

    with pytest.raises(pypto.error.FeError):
        with pypto.function("SUBS", a, c1, c2):
            pypto.set_vec_tile_shapes(32, 32)
            c = a - 3
            c1, *c2 = c
