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
Test file for Arctic Sum LSTM operator.

This file contains:
- Torch golden reference implementations
- Test data preparation
- Precision and performance tests
"""

import os
import sys
import time
import argparse
import logging
from typing import Optional, Tuple, List, Dict, Any

import torch
from numpy.testing import assert_allclose
import pytest

from sum_lstm import sum_lstm, LstmConfig

BATCH_SIZE = 32
D_GATE = 4096
D_GATE_4 = 16384


def rms_norm_golden(x: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.to(torch.float32)
    mean_square = x.pow(2).mean(-1, keepdim=True)
    inv_rms = torch.rsqrt(mean_square + eps)
    return x * inv_rms


def gelu_approx_sigmoid_golden(x: torch.Tensor) -> torch.Tensor:
    """
    GELU approximation using Sigmoid: x * sigmoid(1.702 * x).
    Matches the NPU implementation for alignment.
    """
    return x * torch.sigmoid(1.702 * x)


def sum_lstm_golden(
    states_4d: torch.Tensor,
    z4_4d: torch.Tensor,
    prev_cell: torch.Tensor,
    alpha: float,
    eps_cell: float,
    eps_state: float,
    w_cell: Optional[torch.Tensor] = None,
    b_cell: Optional[torch.Tensor] = None,
    w_state: Optional[torch.Tensor] = None,
    b_state: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Golden reference for Arctic LSTM kernel."""

    # 1. Input Fusion
    fused = states_4d + alpha * z4_4d

    # 2. Chunking: [Forget, Input, Output, Cell_Candidate]
    chunk_size = fused.shape[-1] // 4
    pre_f, pre_i, pre_o, pre_c = torch.split(fused, chunk_size, dim=-1)

    # 3. Gates
    f_gate = torch.sigmoid(pre_f)
    i_gate = torch.sigmoid(pre_i)

    # 4. Pre-Cell Path
    c_cand_norm = rms_norm_golden(pre_c, eps_cell)

    if w_cell is not None:
        c_cand_norm = c_cand_norm * w_cell
    if b_cell is not None:
        c_cand_norm = c_cand_norm + b_cell

    c_act = gelu_approx_sigmoid_golden(c_cand_norm)

    # 5. Cell Update
    c_new = prev_cell * f_gate + c_act * i_gate

    # 6. Post-Cell Path
    h_temp = rms_norm_golden(c_new, eps_state)

    if w_state is not None:
        h_temp = h_temp * w_state
    if b_state is not None:
        h_temp = h_temp + b_state

    h_act = gelu_approx_sigmoid_golden(h_temp)

    # 7. Output Gate & Final Output
    o_gate = torch.sigmoid(pre_o)
    h_new = h_act * o_gate

    return h_new, c_new


def prepare_test_data(device) -> Dict[str, Any]:
    """Prepare common data for both precision and performance tests."""
    # Data
    states_4d = torch.randn(BATCH_SIZE, D_GATE_4, dtype=torch.float16, device=device)
    z4_4d = torch.randn(BATCH_SIZE, D_GATE_4, dtype=torch.float16, device=device)
    prev_cell = torch.randn(BATCH_SIZE, D_GATE, dtype=torch.float16, device=device)

    # Weights (None for this test case, can be tensors)
    w_c = torch.randn(D_GATE, dtype=torch.float16, device=device)
    b_c = torch.randn(D_GATE, dtype=torch.float16, device=device)
    w_s = torch.randn(D_GATE, dtype=torch.float16, device=device)
    b_s = torch.randn(D_GATE, dtype=torch.float16, device=device)

    # Outputs
    h_out = torch.zeros(BATCH_SIZE, D_GATE, dtype=torch.float16, device=device)
    c_out = torch.zeros(BATCH_SIZE, D_GATE, dtype=torch.float16, device=device)

    config = LstmConfig(alpha=0.1, eps_cell=1e-6, eps_state=1e-6)

    # PyPTO Wrappers
    inputs_torch = [states_4d, z4_4d, prev_cell, w_c, b_c, w_s, b_s]
    outputs_torch = [h_out, c_out]

    return {
        "torch_inputs": inputs_torch,
        "torch_outputs": outputs_torch,
        "pto_inputs": inputs_torch,
        "pto_outputs": outputs_torch,
        "config": config,
    }


def run_precision_test(kernel_func, data: Dict[str, Any]):
    """Run correctness verification."""
    logging.info("\n" + "=" * 40)
    logging.info("Running [Precision Test]")
    logging.info("=" * 40)

    # Unpack data
    t_in = data["torch_inputs"]
    h_out, c_out = data["torch_outputs"]
    pto_inputs = data["pto_inputs"]
    pto_outputs = data["pto_outputs"]
    cfg = data["config"]

    # 1. Run NPU Kernel
    # Reset outputs to 0 to ensure we are reading fresh results
    kernel_func(*pto_inputs, *pto_outputs, cfg)

    # 2. Run Golden
    golden_h, golden_c = sum_lstm_golden(
        t_in[0], t_in[1], t_in[2],
        alpha=cfg.alpha, eps_cell=cfg.eps_cell, eps_state=cfg.eps_state,
        w_cell=t_in[3], b_cell=t_in[4], w_state=t_in[5], b_state=t_in[6]
    )

    # 3. Compare
    diff_h = (h_out - golden_h).abs().max().item()
    diff_c = (c_out - golden_c).abs().max().item()
    logging.info(f"Max Diff Hidden: {diff_h:.6f}")
    logging.info(f"Max Diff Cell:   {diff_c:.6f}")

    try:
        assert_allclose(h_out.cpu().numpy(), golden_h.cpu().numpy(), rtol=0.001, atol=5e-3)
        assert_allclose(c_out.cpu().numpy(), golden_c.cpu().numpy(), rtol=5e-3, atol=5e-3)
        logging.info(">> Precision Test PASSED!")
    except AssertionError as e:
        logging.error(">> Precision Test FAILED!")
        raise e


def benchmark_func(func, name: str, n_warmup=1, n_repeat=2) -> float:
    """Helper for measuring execution time."""
    logging.info(f"Benchmarking {name} ...")
    # Warmup
    for _ in range(n_warmup):
        func()
    torch.npu.synchronize()

    # Timing
    t0 = time.time()
    for _ in range(n_repeat):
        func()
    torch.npu.synchronize()
    t1 = time.time()

    avg_ms = (t1 - t0) * 1000 / n_repeat
    logging.info(f" -> {name}: {avg_ms:.4f} ms")
    return avg_ms


def run_performance_test(kernel_func, data: Dict[str, Any]):
    """Run performance benchmarking."""
    logging.info("\n" + "=" * 40)
    logging.info("Running [Performance Test]")
    logging.info("=" * 40)

    # Unpack data
    t_in = data["torch_inputs"]
    pto_inputs = data["pto_inputs"]
    pto_outputs = data["pto_outputs"]
    cfg = data["config"]

    # Wrap calls
    def run_npu():
        kernel_func(*pto_inputs, *pto_outputs, cfg)

    def run_golden():
        sum_lstm_golden(
            t_in[0], t_in[1], t_in[2],
            alpha=cfg.alpha, eps_cell=cfg.eps_cell, eps_state=cfg.eps_state,
            w_cell=t_in[3], b_cell=t_in[4], w_state=t_in[5], b_state=t_in[6]
        )

    # Benchmark
    time_npu = benchmark_func(run_npu, "PyPTO NPU Kernel")
    time_gold = benchmark_func(run_golden, "PyTorch Golden")

    if time_npu > 0:
        logging.info(f"\n>> Speedup: {time_gold / time_npu:.2f}x")


def get_device_id():
    """
    Get and validate TILE_FWK_DEVICE_ID from environment variable.

    Returns:
        int: The device ID if valid, None otherwise.
    """
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        logging.info("If no NPU environment is available, set --run_mode sim to run in simulation mode;")
        logging.info("otherwise, set the environment variable TILE_FWK_DEVICE_ID.")
        logging.info("Please set it before running this example:")
        logging.info("  export TILE_FWK_DEVICE_ID=0")
        return None

    try:
        device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
        return device_id
    except ValueError:
        logging.error(f"ERROR: TILE_FWK_DEVICE_ID must be an integer, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None


@pytest.mark.skip("precision test")
def main():
    parser = argparse.ArgumentParser(description="Run Arctic LSTM PyPTO Example")
    parser.add_argument('--run_mode', type=str, default="npu", choices=["npu", "sim"])
    parser.add_argument('--test_type', type=str, default="precision",
                    choices=["precision", "performance", "all"],
                    help="Choose test type: check correctness or measure performance.")
    args = parser.parse_args()

    # # Enable debug options for development
    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        import torch_npu
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    # 1. Compile Kernel (JIT)
    kernel_func = sum_lstm(args.run_mode)

    # 2. Prepare Data
    data = prepare_test_data(device_id)

    # 3. Dispatch Tests
    if args.test_type in ["precision", "all"]:
        run_precision_test(kernel_func, data)

    if args.test_type in ["performance", "all"]:
        # Only meaningful on NPU hardware
        if args.run_mode == "npu":
            run_performance_test(kernel_func, data)
        else:
            logging.info("\n[INFO] Skipping performance test in simulation mode.")


if __name__ == "__main__":
    main()
