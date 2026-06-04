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
""" """
import torch
import pypto


def test_from_torch():
    a_rawdata = torch.ones((32, 32)) * 2
    a_data = a_rawdata.to(dtype=torch.int32)
    a_pto_dt16 = pypto.from_torch(a_data, dtype=pypto.DT_INT16)
    assert (a_pto_dt16.dtype == pypto.DT_INT16)

    a_pto_hf8 = pypto.from_torch(a_data, dtype=pypto.DT_HF8)
    assert (a_pto_hf8.dtype == pypto.DT_HF8)

    a_pto_default = pypto.from_torch(a_data)
    assert (a_pto_default.dtype == pypto.DT_INT32)
