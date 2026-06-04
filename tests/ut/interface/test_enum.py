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


def test_dtype():
    # Make sure all data types are defined
    assert pypto.bytes_of(pypto.DT_INT4) == 1
    assert pypto.bytes_of(pypto.DT_INT8) == 1
    assert pypto.bytes_of(pypto.DT_INT16) == 2
    assert pypto.bytes_of(pypto.DT_INT32) == 4
    assert pypto.bytes_of(pypto.DT_INT64) == 8
    assert pypto.bytes_of(pypto.DT_FP8) == 1
    assert pypto.bytes_of(pypto.DT_FP16) == 2
    assert pypto.bytes_of(pypto.DT_FP32) == 4
    assert pypto.bytes_of(pypto.DT_BF16) == 2
    assert pypto.bytes_of(pypto.DT_HF4) == 1
    assert pypto.bytes_of(pypto.DT_HF8) == 1
    assert pypto.bytes_of(pypto.DT_FP8E4M3) == 1
    assert pypto.bytes_of(pypto.DT_FP8E5M2) == 1
    assert pypto.bytes_of(pypto.DT_FP8E8M0) == 1
    assert pypto.bytes_of(pypto.DT_UINT8) == 1
    assert pypto.bytes_of(pypto.DT_UINT16) == 2
    assert pypto.bytes_of(pypto.DT_UINT32) == 4
    assert pypto.bytes_of(pypto.DT_UINT64) == 8
    assert pypto.bytes_of(pypto.DT_BOOL) == 1
    assert pypto.bytes_of(pypto.DT_DOUBLE) == 8

    assert repr(pypto.DT_INT4) == "DataType.DT_INT4"


def test_tile_op_format():
    assert repr(pypto.TileOpFormat.TILEOP_ND) == "TileOpFormat.TILEOP_ND"
    assert repr(pypto.TileOpFormat.TILEOP_NZ) == "TileOpFormat.TILEOP_NZ"


def test_cache_policy():
    assert repr(pypto.CachePolicy.NONE_CACHEABLE) == "CachePolicy.NONE_CACHEABLE"


def test_reduce_mode():
    assert repr(pypto.ReduceMode.ATOMIC_ADD) == "ReduceMode.ATOMIC_ADD"


def test_cast_mode():
    assert repr(pypto.CastMode.CAST_RINT) == "CastMode.CAST_RINT"
    assert repr(pypto.CastMode.CAST_ROUND) == "CastMode.CAST_ROUND"
    assert repr(pypto.CastMode.CAST_FLOOR) == "CastMode.CAST_FLOOR"
    assert repr(pypto.CastMode.CAST_CEIL) == "CastMode.CAST_CEIL"
    assert repr(pypto.CastMode.CAST_TRUNC) == "CastMode.CAST_TRUNC"
    assert repr(pypto.CastMode.CAST_ODD) == "CastMode.CAST_ODD"


def test_op_type():
    assert repr(pypto.OpType.EQ) == "OpType.EQ"
    assert repr(pypto.OpType.NE) == "OpType.NE"
    assert repr(pypto.OpType.LT) == "OpType.LT"
    assert repr(pypto.OpType.LE) == "OpType.LE"
    assert repr(pypto.OpType.GT) == "OpType.GT"
    assert repr(pypto.OpType.GE) == "OpType.GE"


def test_out_type():
    assert repr(pypto.OutType.BOOL) == "OutType.BOOL"
    assert repr(pypto.OutType.BIT) == "OutType.BIT"


def test_topk_algo():
    assert repr(pypto.TopKAlgo.MERGE_SORT) == "TopKAlgo.MERGE_SORT"
    assert repr(pypto.TopKAlgo.RADIX_SELECT) == "TopKAlgo.RADIX_SELECT"
