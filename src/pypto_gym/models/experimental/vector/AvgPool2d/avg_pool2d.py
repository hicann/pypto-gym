#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE; IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
AvgPool2d Example for PyPTO

This example demonstrates average pooling 2D operation.
"""
import logging
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from numpy.testing import assert_allclose

import pypto


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PoolParams:
    """Pooling parameters for avg_pool_2d operation"""
    batch_size: int
    channels: int
    in_h: int
    in_w: int
    k_h: int
    k_w: int
    s_h: int
    s_w: int
    t_pad: int
    b_pad: int
    l_pad: int
    r_pad: int
    out_h: int
    out_w: int


@dataclass
class AvgPool2DConfig:
    """Configuration for avg_pool_2d operation"""
    shape: Tuple[int, int, int, int]
    kernel_size: Tuple[int, int]
    stride: Optional[Tuple[int, int]] = None
    padding_mode: str = 'SAME'
    run_mode: str = 'npu'
    dynamic: bool = True


@dataclass
class TestConfig(AvgPool2DConfig):
    """Configuration for avg_pool_2d test case"""
    device_id: Optional[int] = None


def get_device_id():
    """
    Get and validate TILE_FWK_DEVICE_ID from environment variable.

    Returns:
        int: The device ID if valid, None otherwise.
    """
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        logger.warning("If no NPU environment is available, set --run_mode sim to run in simulation mode;")
        logger.warning("otherwise, set the environment variable TILE_FWK_DEVICE_ID.")
        logger.warning("Please set it before running this example:")
        logger.warning("  export TILE_FWK_DEVICE_ID=0")
        return None

    try:
        device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
        return device_id
    except ValueError:
        logger.error(f"ERROR: TILE_FWK_DEVICE_ID must be an integer, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None


def calculate_pool2d_params(config: AvgPool2DConfig) -> PoolParams:
    """
    Calculate pooling parameters including output dimensions and padding values.

    Args:
        config: AvgPool2DConfig containing shape, kernel_size, stride, and padding_mode

    Returns:
        PoolParams: Dataclass containing all pooling parameters

    Raises:
        ValueError: If padding_mode is invalid
    """
    batch_size, channels, in_h, in_w = config.shape
    k_h, k_w = config.kernel_size

    stride = config.stride if config.stride is not None else config.kernel_size
    s_h, s_w = stride

    if config.padding_mode.upper() == 'VALID':
        t_pad = b_pad = l_pad = r_pad = 0
    elif config.padding_mode.upper() == 'SAME':
        out_h = (in_h + s_h - 1) // s_h
        out_w = (in_w + s_w - 1) // s_w
        pad_h = max(0, (out_h - 1) * s_h + k_h - in_h)
        pad_w = max(0, (out_w - 1) * s_w + k_w - in_w)
        t_pad = pad_h // 2
        b_pad = pad_h - t_pad
        l_pad = pad_w // 2
        r_pad = pad_w - l_pad
    else:
        raise ValueError(f"Invalid padding_mode: {config.padding_mode}. Must be 'VALID' or 'SAME'")

    out_h = (in_h + t_pad + b_pad - k_h) // s_h + 1
    out_w = (in_w + l_pad + r_pad - k_w) // s_w + 1

    return PoolParams(
        batch_size=batch_size,
        channels=channels,
        in_h=in_h,
        in_w=in_w,
        k_h=k_h,
        k_w=k_w,
        s_h=s_h,
        s_w=s_w,
        t_pad=t_pad,
        b_pad=b_pad,
        l_pad=l_pad,
        r_pad=r_pad,
        out_h=out_h,
        out_w=out_w
    )


def avg_pool_2d(config: AvgPool2DConfig):
    """
    Create avg_pool_2d kernel based on configuration.

    Args:
        config: AvgPool2DConfig containing all parameters for the pooling operation

    Returns:
        Compiled kernel function
    """
    params = calculate_pool2d_params(config)

    batch_size = params.batch_size
    channels = params.channels
    in_h = params.in_h
    in_w = params.in_w
    k_h = params.k_h
    k_w = params.k_w
    s_h = params.s_h
    s_w = params.s_w
    t_pad = params.t_pad
    b_pad = params.b_pad
    l_pad = params.l_pad
    r_pad = params.r_pad
    out_h = params.out_h
    out_w = params.out_w

    if config.dynamic:
        batch_size = pypto.frontend.dynamic("batch_size")
        channels = pypto.frontend.dynamic("channels")

    if config.run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif config.run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {config.run_mode}. Must be 'npu' or 'sim'")

    @pypto.frontend.jit(pass_options={"vec_nbuffer_setting": {-1: 2, 0: 8}},
                        runtime_options={"run_mode": mode, "stitch_function_max_num": 128},
                        debug_options=dict(runtime_debug_mode=1, compile_debug_mode=1))
    def avg_pool_2d_kernel(
        input_tensor: pypto.Tensor((batch_size, channels, in_h, in_w), pypto.DT_FP32),
        output_result: pypto.Tensor((batch_size, channels, out_h, out_w), pypto.DT_FP32),
    ):
        bc_total = batch_size * channels
        pypto.set_vec_tile_shapes(16, 16, 4, 128)
        input_reshaped = pypto.reshape(input_tensor, [batch_size * channels, in_h, in_w], inplace=True)
        output_tmp = pypto.tensor((bc_total, out_h, out_w), pypto.DT_FP32)

        for bc_idx, unroll_length in pypto.loop_unroll(0, bc_total, 1, name="LOOP_BC",
                                                       idx_name="bc_idx", unroll_list=[8, 4, 2, 1]):
            input_cur = input_reshaped[bc_idx: bc_idx + unroll_length, :, :]
            for oh in range(out_h):
                h_start = oh * s_h - t_pad
                h_end = h_start + k_h
                h_start_clamped = max(h_start, 0)
                h_end_clamped = min(h_end, in_h)

                cur_k_h = h_end_clamped - h_start_clamped

                pypto.set_vec_tile_shapes(16, 16, 128)
                if cur_k_h > 0:
                    input_single_row = input_cur[:, h_start_clamped:h_end_clamped, :]
                else:
                    input_single_row = None

                input_single_row_1 = pypto.sum(input_single_row, 1, keepdim=True)

                for ow in range(out_w):
                    w_start = ow * s_w - l_pad
                    w_end = w_start + k_w
                    w_start_clamped = max(w_start, 0)
                    w_end_clamped = min(w_end, in_w)

                    cur_k_w = w_end_clamped - w_start_clamped

                    if cur_k_h > 0 and cur_k_w > 0:
                        window = input_single_row_1[:, :, w_start_clamped:w_end_clamped]
                        sum_val = pypto.sum(window, dim=2, keepdim=True)
                        avg_val = sum_val / (k_h * k_w)
                        pypto.set_vec_tile_shapes(unroll_length, 4, 128)
                        pypto.assemble(avg_val, [bc_idx, oh, ow], output_tmp)
                    else:
                        zero_val = pypto.zeros([unroll_length, 1, 1], dtype=pypto.DT_FP32)
                        pypto.set_vec_tile_shapes(unroll_length, 4, 128)
                        pypto.assemble(zero_val, [bc_idx, oh, ow], output_tmp)
            pypto.set_vec_tile_shapes(unroll_length, 4, 128)
            output_result.move(pypto.reshape(output_tmp, [batch_size, channels, out_h, out_w], inplace=True))
    return avg_pool_2d_kernel


def avg_pool_2d_golden(x, kernel_size, stride, padding_mode):
    """
    Compute golden output using numpy (equivalent to tf.compat.v1.nn.avg_pool).

    Args:
        x: Input tensor (input_n, input_c, input_h, input_w)
        kernel_size: (k_h, k_w)
        stride: (s_h, s_w)
        padding_mode: 'VALID' or 'SAME'

    Returns:
        Output tensor (input_n, input_c, out_h, out_w)
    """
    input_n, input_c, input_h, input_w = x.shape
    k_h, k_w = kernel_size
    s_h, s_w = stride

    if padding_mode == 'VALID':
        t_pad = b_pad = l_pad = r_pad = 0
    elif padding_mode == 'SAME':
        out_h = (input_h + s_h - 1) // s_h
        out_w = (input_w + s_w - 1) // s_w
        pad_h = max(0, (out_h - 1) * s_h + k_h - input_h)
        pad_w = max(0, (out_w - 1) * s_w + k_w - input_w)
        t_pad = pad_h // 2
        b_pad = pad_h - t_pad
        l_pad = pad_w // 2
        r_pad = pad_w - l_pad
    else:
        raise ValueError(f"Invalid padding_mode: {padding_mode}")

    out_h = (input_h + t_pad + b_pad - k_h) // s_h + 1
    out_w = (input_w + l_pad + r_pad - k_w) // s_w + 1

    x_padded = np.pad(x, ((0, 0), (0, 0), (t_pad, b_pad), (l_pad, r_pad)), mode='constant', constant_values=0)

    output = np.zeros((input_n, input_c, out_h, out_w), dtype=x.dtype)

    for oh in range(out_h):
        h_start = oh * s_h
        h_end = h_start + k_h

        for ow in range(out_w):
            w_start = ow * s_w
            w_end = w_start + k_w

            window = x_padded[:, :, h_start:h_end, w_start:w_end]
            output[:, :, oh, ow] = np.mean(window, axis=(2, 3))

    return output


def test_avg_pool_2d_single(config: TestConfig):
    """
    Single test case for avg_pool_2d

    Args:
        config: TestConfig containing all test parameters
    """
    device = f'npu:{config.device_id}' if (config.run_mode == "npu" and config.device_id is not None) else 'cpu'

    np.random.seed(42)
    x_np = np.random.randn(*config.shape).astype(np.float32)
    x = torch.from_numpy(x_np).to(device)

    params = calculate_pool2d_params(config)

    y = torch.empty((params.batch_size, params.channels, params.out_h, params.out_w),
                    dtype=torch.float32, device=device)
    avg_pool_2d(config)(x, y)
    y = y.cpu().numpy()

    stride = config.stride if config.stride is not None else config.kernel_size
    golden_output = avg_pool_2d_golden(x_np, config.kernel_size, stride, config.padding_mode)

    if config.run_mode == "npu":
        assert_allclose(y, golden_output, rtol=1e-3, atol=1e-3)
    else:
        assert_allclose(y, golden_output, rtol=1e-3, atol=1e-3)

    logger.info(
        f"✓ test_avg_pool_2d ({config.padding_mode}) passed: "
        f"shape={config.shape}, kernel={config.kernel_size}, stride={stride}"
    )


def test_avg_pool_2d_all():
    """Run all avg_pool_2d test cases"""
    logger.info("PyPTO avg_pool_2d Example")
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    import torch_npu
    torch_npu.npu.set_device(device_id)

    test_configs = [
        TestConfig(
            shape=(2, 3, 6, 6),
            kernel_size=(2, 2),
            stride=(2, 2),
            padding_mode='SAME',
            device_id=device_id,
            run_mode="npu",
            dynamic=True
        ),
        TestConfig(
            shape=(4, 8, 12, 12),
            kernel_size=(3, 3),
            stride=(2, 2),
            padding_mode='VALID',
            device_id=device_id,
            run_mode="npu",
            dynamic=True
        )
    ]

    for test_config in test_configs:
        test_avg_pool_2d_single(test_config)

    logger.info("All avg_pool_2d tests passed!")


def main():
    test_avg_pool_2d_all()


if __name__ == "__main__":
    main()
