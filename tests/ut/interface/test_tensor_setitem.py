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


def test_tensor_setitem_inside_loop():
    a, b, c = init_tensors()
    with pypto.function("MAIN", a, b, c):
        pypto.set_vec_tile_shapes(16, 16)
        for k in pypto.loop(10, name="LOOP", idx_name="k"):
            b[:] = pypto.add(a, a)

            if pypto.cond(k < 2):
                b[:] = pypto.add(b, a)
            else:
                b[:] = pypto.sub(b, a)

            if pypto.cond(k < 5):
                b[:] = pypto.mul(b, a)
            else:
                b[:] = pypto.div(b, a)
            c[:] = pypto.sub(b, a)

    assert isinstance(b, pypto.tensor)


def test_tensor_assmble_slice():
    a, b, c = init_tensors()
    with pypto.function("MAIN", a, b, c):
        pypto.set_vec_tile_shapes(16, 16)

        for k in pypto.loop(10, name="LOOP", idx_name="k"):
            b[k * 16:, 0:] = pypto.add(a, a)

            if pypto.cond(k < 2):
                b[k * 16:, 0:] = pypto.add(a, a)
            else:
                b[k * 16:, 0:] = pypto.sub(a, a)

            if pypto.cond(k < 5):
                b[0:, :k * 16] = pypto.mul(a, a)
            else:
                b[0:, :k * 16] = pypto.div(a, a)
            c[:] = pypto.sub(b, a)

    assert isinstance(c, pypto.tensor)


@pytest.mark.skip(reason="Case is no longer maintained")
def test_set_tensor_data():
    a = pypto.tensor([32, 32], pypto.DT_INT32, "a")
    b = pypto.tensor([32, 32], pypto.DT_INT32, "b")
    c = pypto.tensor([32, 32], pypto.DT_INT32, "c")
    with pypto.function("MAIN", a, b, c):
        pypto.set_vec_tile_shapes(16, 16)
        sym_a = a[0, 0]
        sym_b = b[0, 0]
        assert isinstance(sym_a, pypto.SymbolicScalar)
        assert isinstance(sym_b, pypto.SymbolicScalar)
        c[0, 0] = sym_a + sym_b
        c[1, 1] = 1


@pytest.mark.parametrize("dtype", [pypto.DT_INT32])
def test_get_input_shape(dtype):
    shape = (128, 128)
    a = pypto.tensor(shape, dtype, "a")
    b = pypto.tensor(shape, dtype, "b")
    with pypto.function("MAIN", a, b):
        pypto.set_vec_tile_shapes(16, 16)
        val = a[0, 0]
        b[:] = a + a[val: val + 128, :]
