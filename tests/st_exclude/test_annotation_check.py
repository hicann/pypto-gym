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
"""Test input tensor annotation check of @jit decorator."""

import os
import logging
import pypto
import torch

logging.basicConfig(level=logging.INFO, format='', force=True)


def create_compute_func(shape, dtype):
    """Factory function that creates a new compute function with annotations."""
    tiling = 16

    @pypto.frontend.jit()
    def compute_add(
        a: pypto.Tensor(shape, dtype),
        b: pypto.Tensor(shape, dtype),
        c: pypto.Tensor(shape, dtype),
    ):
        pypto.set_cube_tile_shapes([tiling, tiling], [tiling, tiling], [tiling, tiling])
        c[:] = pypto.matmul(a, b, a.dtype)
    return compute_add


def test_correct_annotation():
    """Test that correct annotation causes no error."""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    shape = (16, 16)
    dtype = pypto.DT_FP32
    kernel = create_compute_func(shape, dtype)

    kernel(torch.rand(shape, dtype=torch.float32, device=f'npu:{device_id}'),
           torch.rand(shape, dtype=torch.float32, device=f'npu:{device_id}'),
           torch.rand(shape, dtype=torch.float32, device=f'npu:{device_id}'))
    logging.info(f"✓ Verified: Correct annotation causes no error.")


def test_number_of_input_not_match():
    """Test that number of input not match causes an error."""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    shape = (16, 16)
    dtype = pypto.DT_FP32
    kernel = create_compute_func(shape, dtype)
    try:
        kernel(torch.rand(shape, dtype=torch.float32, device=f'npu:{device_id}'),
               torch.rand(shape, dtype=torch.float32, device=f'npu:{device_id}'))
    except pypto.error.FeError as e:
        logging.info(f"✓ Verified: Number of input not match causes an error: {e}")
    else:
        raise RuntimeError("Number of input not match causes no error.")


def test_incorrect_number_of_dimensions_annotation():
    """Test that incorrect number of dimensions annotation causes an error."""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    shape = (16, 16)
    dtype = pypto.DT_FP32
    kernel = create_compute_func(shape, dtype)

    try:
        kernel(torch.rand((16, 16, 16), dtype=torch.float32, device=f'npu:{device_id}'),
               torch.rand(shape, dtype=torch.float32, device=f'npu:{device_id}'),
               torch.rand(shape, dtype=torch.float32, device=f'npu:{device_id}'))
    except pypto.error.FeError as e:
        logging.info(f"✓ Verified: Incorrect number of dimensions annotation causes an error: {e}")
    else:
        raise ValueError("Incorrect number of dimensions annotation causes no error.")


def test_incorrect_shape_annotation():
    """Test that incorrect shape annotation causes an error."""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    shape = (16, 16)
    dtype = pypto.DT_FP32
    kernel = create_compute_func(shape, dtype)
    try:
        kernel(torch.rand((16, 18), dtype=torch.float32, device=f'npu:{device_id}'),
               torch.rand((16, 16), dtype=torch.float32, device=f'npu:{device_id}'),
               torch.rand((16, 16), dtype=torch.float32, device=f'npu:{device_id}'))
    except pypto.error.FeError as e:
        logging.info(f"✓ Verified: Incorrect shape annotation causes an error: {e}")
    else:
        raise ValueError("Incorrect shape annotation causes no error.")


def test_incorrect_dtype_annotation():
    """Test that incorrect annotation causes an error."""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    shape = (16, 16)
    dtype = pypto.DT_FP16
    kernel = create_compute_func(shape, dtype)
    try:
        kernel(torch.rand(shape, dtype=torch.float16, device=f'npu:{device_id}'),
               torch.rand(shape, dtype=torch.float32, device=f'npu:{device_id}'),
               torch.rand(shape, dtype=torch.float16, device=f'npu:{device_id}'))
    except pypto.error.FeError as e:
        logging.info(f"✓ Verified: Incorrect dtype annotation causes an error: {e}")
    else:
        raise ValueError("Incorrect dtype annotation causes no error.")


if __name__ == "__main__":
    logging.info("=" * 70)
    logging.info("Testing @pypto.frontend.jit decorator annotation check")
    logging.info("=" * 70)

    test_correct_annotation()
    test_number_of_input_not_match()
    test_incorrect_number_of_dimensions_annotation()
    test_incorrect_shape_annotation()
    test_incorrect_dtype_annotation()

    logging.info("\n" + "=" * 70)
    logging.info("All tests passed!")
    logging.info("=" * 70)
