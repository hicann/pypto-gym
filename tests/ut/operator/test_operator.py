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
""" """
import pypto
from test_base import BaseTest

dtype = pypto.DT_FP16
shape = (64, 64)
tiles = (32, 32)


class TestOperator(BaseTest):

    def test_sin(self):
        a = pypto.tensor(shape, dtype, "a")
        c = pypto.tensor(shape, dtype, "c")
        with pypto.function("sign", a, c):
            for _ in pypto.loop(1, name="signLoop"):
                pypto.set_vec_tile_shapes(*tiles)
                c[:] = pypto.sin(a)

    def test_cos(self):
        a = pypto.tensor(shape, dtype, "a")
        c = pypto.tensor(shape, dtype, "c")
        with pypto.function("cos", a, c):
            for _ in pypto.loop(1, name="cosLoop"):
                pypto.set_vec_tile_shapes(*tiles)
                c[:] = pypto.cos(a)

    def test_sigmoid(self):
        a = pypto.tensor(shape, dtype, "a")
        c = pypto.tensor(shape, dtype, "c")
        with pypto.function("sigmoid", a, c):
            for _ in pypto.loop(1, name="sigmoidLoop"):
                pypto.set_vec_tile_shapes(*tiles)
                c[:] = pypto.sigmoid(a)

    def test_softmax(self):
        a = pypto.tensor(shape, dtype, "a")
        c = pypto.tensor(shape, dtype, "c")
        with pypto.function("softmax", a, c):
            for _ in pypto.loop(1, name="softmaxLoop"):
                pypto.set_vec_tile_shapes(*tiles)
                c[:] = pypto.softmax(a, dim=-1)
