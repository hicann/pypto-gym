#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
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


def test_unsqueeze_validshape():
    """Test whether unsqueeze correctly propagates validShape"""
    dtype = pypto.DT_FP32
    shape = [32, 32]
    x = pypto.tensor(shape, dtype, "x")

    with pypto.function("UNSQUEEZE_VALIDSHAPE", x):
        pypto.set_vec_tile_shapes(32, 32)

        # Create a view with validShape different from shape to test validShape propagation
        # View shape is [32, 32], but validShape is [16, 16]
        x_view = pypto.view(x, [32, 32], [0, 0], valid_shape=[16, 16])

        # Test unsqueeze at dimension 0
        res = pypto.unsqueeze(x_view, 0)

        # Verify shape: [32, 32] -> [1, 32, 32]
        assert res.shape == [1, 32, 32]

        # Verify validShape: [16, 16] -> [1, 16, 16]
        assert len(res.valid_shape) == 3
        assert res.valid_shape[0].concrete() == 1
        assert res.valid_shape[1].concrete() == 16
        assert res.valid_shape[2].concrete() == 16

        assert pypto.reshape(res, [-1]).shape == [1024]
