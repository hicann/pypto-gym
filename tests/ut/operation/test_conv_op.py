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
Conv operation unit tests.
Tests cover:
- Different dtype: FP16, BF16, FP32 (交叉测试在动态轴用例中)
- Dynamic axis tiling: batch, cout, hout, wout, dout
"""
import pypto
import pytest


# ============================================================================
# Original Conv Tests (保持原有 conv_tile_shapes 配置不变)
# ============================================================================

def test_conv1d_op():
    # conv1d op test
    dtype = pypto.DT_FP32
    a = pypto.tensor((1, 16, 64), dtype, "fmap")
    b = pypto.tensor((64, 16, 3), dtype, "weight")
    c = None

    with pypto.function("CONV", a, b):
        pypto.set_conv_tile_shapes(
            pypto.pypto_impl.TileL1Info(
                tileHin=1,
                tileHout=1,
                tileWin=64,
                tileWout=64,
                tileCinFmap=8,
                tileCinWeight=8,
                tileN=32,
                tileBatch=1
            ),
            pypto.pypto_impl.TileL0Info(
                tileH=1,
                tileW=64,
                tileK=24,
                tileN=32
            )
        )
        c = pypto.conv(a, b, dtype, [1], [1, 1], [1], extend_params={}, groups=1)

    assert isinstance(c, pypto.tensor)
    assert c.shape == [1, 64, 64]


def test_conv2d_op():
    dtype = pypto.DT_FP16
    a = pypto.tensor((2, 16, 16, 16), dtype, "fmap")
    b = pypto.tensor((64, 16, 3, 3), dtype, "weight")
    c = None

    with pypto.function("CONV", a, b):
        pypto.set_conv_tile_shapes(
            pypto.pypto_impl.TileL1Info(
                tileHin=3,
                tileHout=3,
                tileWin=16,
                tileWout=16,
                tileCinFmap=16,
                tileCinWeight=16,
                tileN=64,
                tileBatch=1
            ),
            pypto.pypto_impl.TileL0Info(
                tileH=3,
                tileW=16,
                tileK=48,
                tileN=64
            )
        )
        c = pypto.conv(a, b, dtype, [1, 1], [1, 1, 1, 1], [1, 1], extend_params={}, groups=1)

    assert isinstance(c, pypto.tensor)
    assert c.shape == [2, 64, 16, 16]


def test_conv3d_op():
    dtype = pypto.DT_FP16
    a = pypto.tensor((1, 16, 2, 16, 32), dtype, "fmap")
    b = pypto.tensor((64, 16, 2, 3, 3), dtype, "weight")

    with pypto.function("CONV", a, b):
        pypto.set_conv_tile_shapes(
            pypto.pypto_impl.TileL1Info(
                tileHin=1,
                tileHout=1,
                tileWin=32,
                tileWout=32,
                tileCinFmap=16,
                tileCinWeight=16,
                tileN=64,
                tileBatch=1
            ),
            pypto.pypto_impl.TileL0Info(
                tileH=1,
                tileW=32,
                tileK=48,
                tileN=64
            )
        )
        c = pypto.conv(a, b, dtype, [1, 1, 1], [0, 0, 1, 1, 1, 1], [1, 1, 1], extend_params={}, groups=1)

    assert isinstance(c, pypto.tensor)
    assert c.shape == [1, 64, 1, 16, 32]


def test_conv2d_bias_op():
    dtype = pypto.DT_FP16
    a = pypto.tensor((2, 16, 16, 64), dtype, "fmap")
    b = pypto.tensor((64, 16, 3, 3), dtype, "weight")
    c = pypto.tensor((64,), dtype, "bias")

    with pypto.function("CONV", a, b):
        pypto.set_conv_tile_shapes(
            pypto.pypto_impl.TileL1Info(
                tileHin=2,
                tileHout=2,
                tileWin=64,
                tileWout=64,
                tileCinFmap=16,
                tileCinWeight=16,
                tileN=32,
                tileBatch=1
            ),
            pypto.pypto_impl.TileL0Info(
                tileH=2,
                tileW=64,
                tileK=48,
                tileN=32
            )
        )
        c = pypto.conv(a, b, dtype, [1, 1], [1, 1, 1, 1], [1, 1], extend_params={"bias_tensor": c}, groups=1)

    assert isinstance(c, pypto.tensor)
    assert c.shape == [2, 64, 16, 64]


def test_conv1d_group_op():
    dtype = pypto.DT_FP32
    a = pypto.tensor((1, 16, 128), dtype, "fmap")
    b = pypto.tensor((64, 1, 3), dtype, "weight")

    with pypto.function("CONV", a, b):
        pypto.set_conv_tile_shapes(
            pypto.pypto_impl.TileL1Info(
                tileHin=1,
                tileHout=1,
                tileWin=128,
                tileWout=128,
                tileCinFmap=8,
                tileCinWeight=8,
                tileN=16,
                tileBatch=1
            ),
            pypto.pypto_impl.TileL0Info(
                tileH=1,
                tileW=64,
                tileK=8,
                tileN=16
            )
        )
        c = pypto.conv(a, b, dtype, [1], [1, 1], [1], extend_params={}, groups=16)

    assert isinstance(c, pypto.tensor)
    assert c.shape == [1, 64, 128]