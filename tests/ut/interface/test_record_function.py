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
import pypto._controller as controller

def test_record_function():
    dtype = pypto.DT_FP16
    shape = (8, 8)
    a = pypto.tensor(shape, dtype, "tensor_a")
    b = pypto.tensor(shape, dtype, "tensor_b")
    c = None

    with pypto.function("ADD", a, b):
        pypto.set_vec_tile_shapes(8, 8)
        c = pypto.add(a, b)

    print(controller.dump())
    # Replace True with False to see graph
    assert isinstance(c, pypto.tensor)


def test_begin_inplaceadd_end_function():
    dtype = pypto.DT_FP16
    shape = (8, 8)
    a = pypto.tensor(shape, dtype, "tensor_a")
    b = pypto.tensor(shape, dtype, "tensor_b")
    c = None

    with pypto.function("ADD_INPLACE", a, b):
        pypto.set_vec_tile_shapes(8, 8)
        c = a + b

    print(controller.dump())
    assert isinstance(c, pypto.tensor)


@pytest.mark.skip(reason="Case is no longer maintained")
def test_empty_begin_end_function():
    dtype = pypto.DT_FP16
    a = pypto.tensor((8, 8), dtype, "tensor_a")

    with pypto.function("MAIN", a):
        pypto.set_vec_tile_shapes(8, 8)

    assert True