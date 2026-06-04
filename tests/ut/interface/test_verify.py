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
import torch


def _to_device_tensor_data(tensor: torch.Tensor, shape=None, dtype=pypto.DT_FP32):
    target_shape = list(tensor.shape) if shape is None else shape
    return pypto.pypto_impl.DeviceTensorData(dtype, tensor.data_ptr(), target_shape)


@pytest.mark.skip(reason="Verify not supported")
def test_verify_dynamic_ops_assemble():
    s = 32
    n = 2
    m = 1
    t0 = pypto.tensor((n * s, m * s), pypto.DT_FP32)
    t1 = pypto.tensor((n * s, m * s), pypto.DT_FP32)
    out = pypto.tensor((n * s, m * s), pypto.DT_FP32)
    t0_data = torch.ones((n * s, m * s))
    t1_data = torch.ones((n * s, m * s)) * 2
    out_data = torch.zeros((n * s, m * s))
    golden = torch.ones((n * s, m * s)) * 3
    pypto.set_verify_options(
        enable_pass_verify=True
    )
    pypto.set_verify_golden_data([t0_data, t1_data, out_data], [t0_data, t1_data, golden])
    with pypto.function("main", t0, t1, out):
        pypto.set_vec_tile_shapes(8, 8)
        for idx in pypto.loop(10):
            pypto.pass_verify_print(t0)
            t0a = pypto.view(t0, [s, s], [0, 0])
            t0b = pypto.view(t0, [s, s], [s, 0])
            t1a = pypto.view(t1, [s, s], [0, 0])
            t1b = pypto.view(t1, [s, s], [s, 0])
            t2a = t0a + t1a
            t2b = t0b + t1b
            pypto.assemble(t2a, [0, 0], out)
            pypto.assemble(t2b, [s, 0], out)
            pypto.pass_verify_save(out, "tensor_out_idx$idx", idx=idx)


def test_set_verify_data_dtype_mismatch_intercept():
    input_tensor = torch.zeros((2, 3), dtype=torch.float32)
    output_tensor = torch.zeros((2, 3), dtype=torch.float32)
    input_golden = torch.zeros((2, 3), dtype=torch.float32)
    output_golden = torch.zeros((2, 3), dtype=torch.float32)

    input_data = _to_device_tensor_data(input_tensor, dtype=pypto.DT_FP32)
    output_data = _to_device_tensor_data(output_tensor, dtype=pypto.DT_FP32)
    input_golden_data = _to_device_tensor_data(input_golden, dtype=pypto.DT_FP32)
    output_golden_data = _to_device_tensor_data(output_golden, dtype=pypto.DT_FP16)

    with pytest.raises(Exception, match="ErrCode:\\s*FB4003"):
        pypto.pypto_impl.SetVerifyData([input_data], [output_data], [input_golden_data, output_golden_data])


def test_set_verify_data_shape_mismatch_intercept():
    input_tensor = torch.zeros((2, 3), dtype=torch.float32)
    output_tensor = torch.zeros((2, 3), dtype=torch.float32)
    input_golden = torch.zeros((2, 3), dtype=torch.float32)
    output_golden = torch.zeros((2, 3), dtype=torch.float32)

    input_data = _to_device_tensor_data(input_tensor)
    input_golden_data = _to_device_tensor_data(input_golden)
    output_golden_exact = _to_device_tensor_data(output_golden, [2, 3])
    output_bad_shape = _to_device_tensor_data(output_tensor, [2, 4])
    with pytest.raises(Exception, match="ErrCode:\\s*FB4002"):
        pypto.pypto_impl.SetVerifyData([input_data], [output_bad_shape], [input_golden_data, output_golden_exact])
