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


def test_pybind_context_manager():
    dtype = pypto.DT_FP16
    shape = (8, 8)
    a = pypto.tensor(shape, dtype, "tensor_a")
    b = pypto.tensor(shape, dtype, "tensor_a")
    c = pypto.tensor(shape, dtype, "tensor_c")

    with pypto.function("fnc_name", a, b):
        pypto.set_vec_tile_shapes(8, 8)
        c.move(pypto.add(a, b))

    assert isinstance(c, pypto.tensor)
