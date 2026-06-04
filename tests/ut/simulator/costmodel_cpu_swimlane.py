#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
This test case verifies that the swimlane diagram generation for the costmodel works correctly
regardless of whether the CANN is installed in the environment or not.
"""

import os
import sys
import argparse
import json
import pypto
import numpy as np
from numpy.testing import assert_allclose
import torch


def safe_json_load(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data, None
    except FileNotFoundError:
        return None, "File not found"
    except json.JSONDecodeError as e:
        return None, f"Invalid json format: {e}"
    except PermissionError:
        return None, "Permission Error"
    except Exception as e:
        return None, f"Load json fail, unknow error: {e}"


def get_out_put_path():
    out_path = "./output"
    if os.path.exists(out_path):
        subdirs = [os.path.join(out_path, d) for d in os.listdir(out_path)
                if os.path.isdir(os.path.join(out_path, d))]
        if subdirs:
            latest_dir = max(subdirs, key=os.path.getctime)
            return latest_dir
    return None


def softmax_core(input_tensor: pypto.tensor) -> pypto.tensor:
    """
    Core softmax computation: exp(x - max(x)) / sum(exp(x - max(x))).

    Parameters
    ----------
    input_tensor : pypto.tensor
        Input tensor to apply softmax to

    Returns
    -------
    pypto.tensor
        Softmax normalized tensor
    """
    # Find maximum for numerical stability
    row_max = pypto.amax(input_tensor, dim=-1, keepdim=True)

    # Subtract maximum
    sub = pypto.sub(input_tensor, row_max)

    # Compute exponentials
    exp = pypto.exp(sub)

    # Sum exponentials
    esum = pypto.sum(exp, dim=-1, keepdim=True)

    return pypto.div(exp, esum)


@pypto.frontend.jit(
    runtime_options={
    "run_mode": 1}
)
def softmax(input_tensor: pypto.Tensor(), output_tensor: pypto.Tensor()):
    """
    Softmax implementation with dynamic batch size support.

    This function processes input tensors in batches, applying softmax
    to each batch independently. The batch dimension is marked as dynamic,
    allowing variable batch sizes at runtime.

    Parameters
    ----------
    inputs : list
        List containing input tensor [batch, n1, n2, dim]
    outputs : list
        List containing output tensor [batch, n1, n2, dim]
    """

    # After the dynamic axis of tensor is marked, get the tensor shape accordingly
    tensor_shape = input_tensor.shape
    b = tensor_shape[0]  # Dynamic batch size
    n1, n2, dim = tensor_shape[1:]  # Static dimensions
    tile_b = 1  # Process one batch at a time
    b_loop = b // tile_b

    # Tiling shape setting for efficient execution
    pypto.set_vec_tile_shapes(1, 4, 1, 64)

    # for idx in pypto.loop(0, b_loop, 1, name="LOOP_L0_bIdx", idx_name="idx"):
    for idx in range(b_loop):
        b_offset = idx * tile_b
        b_offset_end = (idx + 1) * tile_b

        # Extract batch slice
        input_view = input_tensor[b_offset:b_offset_end, :n1, :n2, :dim]

        # Apply softmax to batch slice
        softmax_out = softmax_core(input_view)

        # Assemble result back to output tensor
        pypto.assemble(softmax_out, [b_offset, 0, 0, 0], output_tensor)


def test_softmax():
    """
    Test softmax implementation against PyTorch reference.

    Tests with shape [batch, n1, n2, dim] where batch is dynamic.
    """
    cann_is_configed: bool = bool(os.environ.get("ASCEND_HOME_PATH"))

    # Shape for verification: NCHW format, N can be any integer number as it is defined as dynamic axis
    shape = (32, 32, 1, 256)

    # Prepare data
    input_data = torch.rand(shape, dtype=torch.float32)
    output_data = torch.zeros(shape, dtype=torch.float32)

    softmax(input_data, output_data)

    # Verify against PyTorch reference
    torch_softmax = torch.softmax(input_data, dim=3)
    npu_data = output_data.cpu()
    torch_data = torch_softmax.cpu()

    max_diff = np.abs(npu_data.numpy() - torch_data.numpy()).max()

    output_path = get_out_put_path()
    assert output_path

    merged_swimlane_path = os.path.join(output_path, "CostModelSimulationOutput", "merged_swimlane.json")
    merged_swimlane, error = safe_json_load(merged_swimlane_path)
    assert not error, f"safe_json_load({merged_swimlane_path}): {error}"


if __name__ == "__main__":
    test_softmax()
