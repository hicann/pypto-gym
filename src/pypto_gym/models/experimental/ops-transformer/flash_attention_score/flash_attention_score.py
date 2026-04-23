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
Flash Attention Score with Online Softmax

This module provides:
1. Multiple PyPTO kernel implementations (imported from impl)
2. Golden reference implementations for validation
3. Test suite for all kernels and datatypes

Kernels (Stage 3: Multi-datatype support):
- with_mask: Basic attention with mask support (BF16/FP16/FP32)
- with_pse_and_dropout: Full features (PSE + Dropout + Mask) (BF16/FP16/FP32)
- Configurable scale_value parameter

Datatype Strategy:
- BF16/FP16: Input -> FP32 compute -> Output (with cast)
- FP32: Input -> FP32 compute -> Output (no cast)
"""

import os
import sys
import math
import argparse
import logging
from typing import Optional
from dataclasses import dataclass
import torch
import numpy as np
from numpy.testing import assert_allclose

from flash_attention_score_impl import (
    flash_attention_score_kernel_with_mask_origin,
    flash_attention_score_kernel_with_mask,
    flash_attention_score_kernel_with_pse_and_dropout,
    flash_attention_score_kernel_with_mask_fp32,
    flash_attention_score_kernel_with_pse_and_dropout_fp32,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")


BATCH_SIZE = 4
NUM_HEADS = 8
SEQ_LEN_Q = 128
SEQ_LEN_KV = 128
HEAD_DIM = 64


def get_device_id():
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        logging.info("Please set the environment variable TILE_FWK_DEVICE_ID before running:")
        logging.info("  export TILE_FWK_DEVICE_ID=0")
        return None
    try:
        device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
        return device_id
    except ValueError:
        logging.info(f"ERROR: TILE_FWK_DEVICE_ID must be an integer, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None


def check_nan(tensor: torch.Tensor, name: str) -> bool:
    if torch.isnan(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        total_count = tensor.numel()
        logging.error(f"  {name} contains {nan_count}/{total_count} NaN values!")
        return True
    return False


def flash_attention_score_golden_origin(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    atten_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Golden reference for flash_attention_score_kernel_with_mask_origin.
    
    This version only returns output, without softmax_max and softmax_sum.
    Uses fixed scale = 1/sqrt(HEAD_DIM).
    """
    b, n, sq, d = query.shape
    _, _, skv, _ = key.shape

    scale = 1.0 / math.sqrt(d)

    query_fp32 = query.float()
    key_fp32 = key.float()
    value_fp32 = value.float()

    output = torch.zeros(b, n, sq, d, dtype=torch.float32, device=query.device)

    for b_idx in range(b):
        for n_idx in range(n):
            for q_idx in range(sq):
                q_vec = query_fp32[b_idx, n_idx, q_idx, :]

                max_score = float('-inf')
                sum_exp = 0.0
                output_vec = torch.zeros(d, dtype=torch.float32, device=query.device)

                for kv_idx in range(skv):
                    if atten_mask is not None and atten_mask[q_idx, kv_idx] == 1:
                        continue

                    k_vec = key_fp32[b_idx, n_idx, kv_idx, :]
                    score = torch.dot(q_vec, k_vec) * scale

                    new_max = max(max_score, score.item())

                    if new_max > max_score:
                        correction = math.exp(max_score - new_max)
                        sum_exp = sum_exp * correction
                        output_vec = output_vec * correction
                        max_score = new_max

                    exp_score = math.exp(score - max_score)
                    sum_exp += exp_score

                    v_vec = value_fp32[b_idx, n_idx, kv_idx, :]
                    output_vec += exp_score * v_vec

                if sum_exp > 0:
                    output[b_idx, n_idx, q_idx, :] = output_vec / sum_exp

    return output.to(torch.bfloat16)


def flash_attention_score_golden(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    atten_mask: Optional[torch.Tensor] = None,
    scale_value: Optional[float] = None,
) -> tuple:
    """Golden reference for flash_attention_score_kernel_with_mask.
    
    Args:
        scale_value: Scaling factor for attention scores (default: 1/sqrt(HEAD_DIM))
    """
    b, n, sq, d = query.shape
    _, _, skv, _ = key.shape

    scale = scale_value if scale_value is not None else 1.0 / math.sqrt(d)

    query_fp32 = query.float()
    key_fp32 = key.float()
    value_fp32 = value.float()

    output = torch.zeros(b, n, sq, d, dtype=torch.float32, device=query.device)
    softmax_max = torch.zeros(b, n, sq, 1, dtype=torch.float32, device=query.device)
    softmax_sum = torch.zeros(b, n, sq, 1, dtype=torch.float32, device=query.device)

    for b_idx in range(b):
        for n_idx in range(n):
            for q_idx in range(sq):
                q_vec = query_fp32[b_idx, n_idx, q_idx, :]

                max_score = float('-inf')
                sum_exp = 0.0
                output_vec = torch.zeros(d, dtype=torch.float32, device=query.device)

                for kv_idx in range(skv):
                    if atten_mask is not None and atten_mask[q_idx, kv_idx] == 1:
                        continue

                    k_vec = key_fp32[b_idx, n_idx, kv_idx, :]
                    score = torch.dot(q_vec, k_vec) * scale

                    new_max = max(max_score, score.item())

                    if new_max > max_score:
                        correction = math.exp(max_score - new_max)
                        sum_exp = sum_exp * correction
                        output_vec = output_vec * correction
                        max_score = new_max

                    exp_score = math.exp(score - max_score)
                    sum_exp += exp_score

                    v_vec = value_fp32[b_idx, n_idx, kv_idx, :]
                    output_vec += exp_score * v_vec

                if sum_exp > 0:
                    output[b_idx, n_idx, q_idx, :] = output_vec / sum_exp
                    
                softmax_max[b_idx, n_idx, q_idx, 0] = max_score
                softmax_sum[b_idx, n_idx, q_idx, 0] = sum_exp

    return output.to(torch.bfloat16), softmax_max, softmax_sum


@dataclass
class FlashAttentionInputs:
    """Flash Attention inputs container."""
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    atten_mask: Optional[torch.Tensor]
    pse: torch.Tensor
    drop_mask: torch.Tensor
    pse_type: int = 0
    keep_prob: float = 1.0
    scale_value: Optional[float] = None


def flash_attention_score_golden_with_pse_and_dropout(inputs: FlashAttentionInputs) -> tuple:
    """Golden reference for flash_attention_score_kernel_with_pse_and_dropout.
    
    Args:
        inputs: FlashAttentionInputs containing all input tensors and parameters
    """
    query = inputs.query
    key = inputs.key
    value = inputs.value
    atten_mask = inputs.atten_mask
    pse = inputs.pse
    drop_mask = inputs.drop_mask
    pse_type = inputs.pse_type
    keep_prob = inputs.keep_prob
    scale_value = inputs.scale_value
    b, n, sq, d = query.shape
    _, _, skv, _ = key.shape

    scale = scale_value if scale_value is not None else 1.0 / math.sqrt(d)

    query_fp32 = query.float()
    key_fp32 = key.float()
    value_fp32 = value.float()
    pse_fp32 = pse.float()

    output = torch.zeros(b, n, sq, d, dtype=torch.float32, device=query.device)
    softmax_max = torch.zeros(b, n, sq, 1, dtype=torch.float32, device=query.device)
    softmax_sum = torch.zeros(b, n, sq, 1, dtype=torch.float32, device=query.device)

    for b_idx in range(b):
        for n_idx in range(n):
            for q_idx in range(sq):
                q_vec = query_fp32[b_idx, n_idx, q_idx, :]

                max_score = float('-inf')
                sum_exp = 0.0
                output_vec = torch.zeros(d, dtype=torch.float32, device=query.device)

                for kv_idx in range(skv):
                    if atten_mask is not None and atten_mask[q_idx, kv_idx] == 1:
                        continue

                    k_vec = key_fp32[b_idx, n_idx, kv_idx, :]
                    
                    if pse_type == 1:
                        score = (torch.dot(q_vec, k_vec) + pse_fp32[b_idx, n_idx, q_idx, kv_idx]) * scale
                    else:
                        score = torch.dot(q_vec, k_vec) * scale + pse_fp32[b_idx, n_idx, q_idx, kv_idx]

                    new_max = max(max_score, score.item())

                    if new_max > max_score:
                        correction = math.exp(max_score - new_max)
                        sum_exp = sum_exp * correction
                        output_vec = output_vec * correction
                        max_score = new_max

                    exp_score = math.exp(score - max_score)
                    exp_score = exp_score * drop_mask[q_idx, kv_idx]
                    
                    if keep_prob < 1.0:
                        exp_score = exp_score / keep_prob
                    
                    sum_exp += exp_score

                    v_vec = value_fp32[b_idx, n_idx, kv_idx, :]
                    output_vec += exp_score * v_vec

                if sum_exp > 0:
                    output[b_idx, n_idx, q_idx, :] = output_vec / sum_exp
                    
                softmax_max[b_idx, n_idx, q_idx, 0] = max_score
                softmax_sum[b_idx, n_idx, q_idx, 0] = sum_exp

    return output.to(torch.bfloat16), softmax_max, softmax_sum


def test_kernel_with_mask_origin(device_id=None, run_mode: str = "npu", skip_golden: bool = False):
    """Test flash_attention_score_kernel_with_mask_origin.
    
    This kernel only outputs attention_out, without softmax_max and softmax_sum.
    Uses fixed scale = 1/sqrt(HEAD_DIM), BF16 only.
    
    Args:
        skip_golden: Skip golden comparison (faster for large shapes)
    """
    logging.info("=" * 70)
    logging.info("Test: flash_attention_score_kernel_with_mask_origin (BF16)")
    logging.info("=" * 70)

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'

    query = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM,
                        dtype=torch.bfloat16, device=device)
    key = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM,
                      dtype=torch.bfloat16, device=device)
    value = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM,
                        dtype=torch.bfloat16, device=device)

    atten_mask = torch.zeros(SEQ_LEN_Q, SEQ_LEN_KV, dtype=torch.uint8, device=device)
    atten_mask[:, SEQ_LEN_KV // 2:] = 1

    output = torch.empty(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM,
                         dtype=torch.bfloat16, device=device)

    atten_mask_fp32 = atten_mask.float()

    flash_attention_score_kernel_with_mask_origin(query, key, value, atten_mask_fp32, output)

    logging.info(f"Input shape: query={query.shape}, key={key.shape}, value={value.shape}")
    logging.info(f"Output shape: {output.shape}")

    has_nan_output = check_nan(output, "output")
    
    if has_nan_output:
        raise RuntimeError("Kernel with_mask_origin test failed due to NaN values")
    
    logging.info("  No NaN values detected in output")

    if skip_golden:
        logging.info("  Golden comparison skipped")
        logging.info("  Kernel with_mask_origin test passed!")
        return

    if run_mode == "npu":
        golden = flash_attention_score_golden_origin(query, key, value, atten_mask)

        output_fp32 = output.float()
        golden_fp32 = golden.float()
        max_diff = (output_fp32 - golden_fp32).abs().max().item()
        mean_diff = (output_fp32 - golden_fp32).abs().mean().item()

        logging.info(f"Output max difference: {max_diff:.6f}")
        logging.info(f"Output mean difference: {mean_diff:.6f}")

        assert_allclose(
            output_fp32.cpu().numpy().flatten(),
            golden_fp32.cpu().numpy().flatten(),
            rtol=0.0078125,
            atol=0.0001
        )
        
        logging.info("  Kernel with_mask_origin test passed!")


def test_kernel_with_mask(
    device_id=None,
    run_mode: str = "npu",
    dtype: str = "bf16",
    scale_value: Optional[float] = None,
    skip_golden: bool = False
):
    """Test flash_attention_score_kernel_with_mask.
    
    Args:
        dtype: Data type to test (bf16, fp16, fp32)
        scale_value: Custom scale value (default: 1/sqrt(HEAD_DIM))
        skip_golden: Skip golden comparison (faster for large shapes)
    """
    logging.info("=" * 70)
    logging.info(f"Test: flash_attention_score_kernel_with_mask ({dtype.upper()})")
    logging.info("=" * 70)

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float32
    
    query = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM,
                        dtype=torch_dtype, device=device)
    key = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM,
                      dtype=torch_dtype, device=device)
    value = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM,
                        dtype=torch_dtype, device=device)

    atten_mask = torch.zeros(SEQ_LEN_Q, SEQ_LEN_KV, dtype=torch.uint8, device=device)
    atten_mask[:, SEQ_LEN_KV // 2:] = 1

    output = torch.empty(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM,
                         dtype=torch_dtype, device=device)
    softmax_max = torch.empty(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, 1,
                              dtype=torch.float32, device=device)
    softmax_sum = torch.empty(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, 1,
                              dtype=torch.float32, device=device)

    atten_mask_fp32 = atten_mask.float()
    
    default_scale = 1.0 / math.sqrt(HEAD_DIM)
    test_scale = scale_value if scale_value is not None else default_scale
    
    logging.info(f"Scale value: {test_scale}")

    if dtype == "bf16":
        kernel_func = flash_attention_score_kernel_with_mask
    else:
        kernel_func = flash_attention_score_kernel_with_mask_fp32
    
    kernel_func(query, key, value, atten_mask_fp32, output, softmax_max, softmax_sum, test_scale)

    logging.info(f"Input shape: query={query.shape}, key={key.shape}, value={value.shape}")
    logging.info(f"Output shape: {output.shape}")

    has_nan_output = check_nan(output, "output")
    has_nan_softmax_max = check_nan(softmax_max, "softmax_max")
    has_nan_softmax_sum = check_nan(softmax_sum, "softmax_sum")
    
    if has_nan_output or has_nan_softmax_max or has_nan_softmax_sum:
        raise RuntimeError("Kernel with_mask test failed due to NaN values")
    
    logging.info("  No NaN values detected in outputs")

    if skip_golden:
        logging.info("  Golden comparison skipped")
        logging.info(f"  Kernel with_mask test passed for {dtype.upper()}!")
        return

    if run_mode == "npu":
        golden, golden_max, golden_sum = flash_attention_score_golden(query, key, value, atten_mask, test_scale)

        output_fp32 = output.float()
        golden_fp32 = golden.float()
        max_diff = (output_fp32 - golden_fp32).abs().max().item()
        mean_diff = (output_fp32 - golden_fp32).abs().mean().item()

        logging.info(f"Output max difference: {max_diff:.6f}")
        logging.info(f"Output mean difference: {mean_diff:.6f}")
        
        rtol = 0.0078125 if dtype == "bf16" else 0.01
        atol = 0.0001 if dtype == "bf16" else 0.003

        assert_allclose(
            output_fp32.cpu().numpy().flatten(),
            golden_fp32.cpu().numpy().flatten(),
            rtol=rtol,
            atol=atol
        )
        
        logging.info(f"  Kernel with_mask test passed for {dtype.upper()}!")


def test_kernel_with_pse_and_dropout(
    device_id=None,
    run_mode: str = "npu",
    dtype: str = "bf16",
    scale_value: Optional[float] = None,
    skip_golden: bool = False
):
    """Test flash_attention_score_kernel_with_pse_and_dropout.
    
    Args:
        dtype: Data type to test (bf16, fp16, fp32)
        scale_value: Custom scale value (default: 1/sqrt(HEAD_DIM))
        skip_golden: Skip golden comparison (faster for large shapes)
    """
    logging.info("\n" + "=" * 70)
    logging.info(f"Test: flash_attention_score_kernel_with_pse_and_dropout ({dtype.upper()})")
    logging.info("=" * 70)

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float32

    query = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM,
                        dtype=torch_dtype, device=device)
    key = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM,
                      dtype=torch_dtype, device=device)
    value = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM,
                        dtype=torch_dtype, device=device)

    atten_mask = torch.zeros(SEQ_LEN_Q, SEQ_LEN_KV, dtype=torch.uint8, device=device)
    atten_mask[:, SEQ_LEN_KV // 4:] = 1

    pse = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, SEQ_LEN_KV,
                     dtype=torch_dtype, device=device)

    drop_mask = torch.ones(SEQ_LEN_Q, SEQ_LEN_KV, dtype=torch.float32, device=device)
    keep_prob = 0.8

    output = torch.empty(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM,
                         dtype=torch_dtype, device=device)
    softmax_max = torch.empty(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, 1,
                              dtype=torch.float32, device=device)
    softmax_sum = torch.empty(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, 1,
                              dtype=torch.float32, device=device)

    atten_mask_fp32 = atten_mask.float()

    logging.info(f"Input shape: query={query.shape}, key={key.shape}, value={value.shape}")
    logging.info(f"PSE shape: {pse.shape}, drop_mask shape: {drop_mask.shape}")
    
    default_scale = 1.0 / math.sqrt(HEAD_DIM)
    test_scale = scale_value if scale_value is not None else default_scale
    
    logging.info(f"Scale value: {test_scale}")

    if dtype == "bf16":
        kernel_func = flash_attention_score_kernel_with_pse_and_dropout
    else:
        kernel_func = flash_attention_score_kernel_with_pse_and_dropout_fp32

    for pse_type in [0, 1]:
        logging.info(f"\nTesting pse_type={pse_type}")
        
        kernel_func(
            query, key, value, atten_mask_fp32, pse, drop_mask,
            output, softmax_max, softmax_sum, pse_type, keep_prob, test_scale
        )
        
        has_nan_output = check_nan(output, "output")
        has_nan_softmax_max = check_nan(softmax_max, "softmax_max")
        has_nan_softmax_sum = check_nan(softmax_sum, "softmax_sum")
        
        if has_nan_output or has_nan_softmax_max or has_nan_softmax_sum:
            raise RuntimeError(f"Kernel with_pse_and_dropout test failed due to NaN values (pse_type={pse_type})")
        
        logging.info(f"  No NaN values detected in outputs")
        
        if skip_golden:
            logging.info("  Golden comparison skipped")
            logging.info(f"  Kernel with_pse_and_dropout test passed for pse_type={pse_type} ({dtype.upper()}!")
            continue
        
        if run_mode == "npu":
            inputs = FlashAttentionInputs(
                query=query,
                key=key,
                value=value,
                atten_mask=atten_mask,
                pse=pse,
                drop_mask=drop_mask,
                pse_type=pse_type,
                keep_prob=keep_prob,
                scale_value=test_scale
            )
            golden, golden_max, golden_sum = flash_attention_score_golden_with_pse_and_dropout(inputs)
            
            output_fp32 = output.float()
            golden_fp32 = golden.float()
            
            max_diff = (output_fp32 - golden_fp32).abs().max().item()
            mean_diff = (output_fp32 - golden_fp32).abs().mean().item()
            
            logging.info(f"Output max difference: {max_diff:.6f}")
            logging.info(f"Output mean difference: {mean_diff:.6f}")
            
            rtol = 0.0078125 if dtype == "bf16" else 0.01
            atol = 0.0001 if dtype == "bf16" else 0.003
            
            assert_allclose(
                output_fp32.cpu().numpy().flatten(),
                golden_fp32.cpu().numpy().flatten(),
                rtol=rtol,
                atol=atol
            )
            
            logging.info(f"  Kernel with_pse_and_dropout test passed for pse_type={pse_type} ({dtype.upper()}!")


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO Flash Attention Score Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on NPU (default)
  export TILE_FWK_DEVICE_ID=0
  python flash_attention_score.py
  
  # Run in sim mode
  python flash_attention_score.py --run_mode sim
"""
    )
    parser.add_argument(
        '--run_mode',
        type=str,
        default='npu',
        choices=["npu", "sim"],
        help='Run mode: npu or sim (default: npu)'
    )
    parser.add_argument(
        '--kernel',
        type=str,
        default='all',
        choices=["all", "mask_origin", "mask", "pse_dropout"],
        help='Which kernel to test: all, mask_origin, mask, or pse_dropout (default: all)'
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='all',
        choices=["all", "bf16", "fp32"],
        help='Data type to test: all, bf16, or fp32 (default: all)'
    )
    parser.add_argument(
        '--scale_value',
        type=float,
        default=None,
        help='Custom scale value (default: 1/sqrt(HEAD_DIM))'
    )
    parser.add_argument(
        '--skip_golden',
        action='store_true',
        help='Skip golden comparison (faster for large shapes)'
    )
    args = parser.parse_args()

    logging.info("\n" + "=" * 70)
    logging.info("PyPTO Flash Attention Score Test Suite")
    logging.info("=" * 70 + "\n")

    device_id = None
    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        import torch_npu
        torch.npu.set_device(device_id)
        logging.info(f"Running on NPU device {device_id}\n")

    try:
        dtypes_to_test = ["bf16", "fp32"] if args.dtype == "all" else [args.dtype]
        
        for dtype in dtypes_to_test:
            logging.info(f"\n{'=' * 70}")
            logging.info(f"Testing dtype: {dtype.upper()}")
            logging.info(f"{'=' * 70}")
            
            if args.kernel == "all":
                if dtype == "bf16":
                    test_kernel_with_mask_origin(device_id, args.run_mode, args.skip_golden)
                test_kernel_with_mask(device_id, args.run_mode, dtype, args.scale_value, args.skip_golden)
                test_kernel_with_pse_and_dropout(device_id, args.run_mode, dtype, args.scale_value, args.skip_golden)
            elif args.kernel == "mask_origin":
                test_kernel_with_mask_origin(device_id, args.run_mode, args.skip_golden)
            elif args.kernel == "mask":
                test_kernel_with_mask(device_id, args.run_mode, dtype, args.scale_value, args.skip_golden)
            elif args.kernel == "pse_dropout":
                test_kernel_with_pse_and_dropout(device_id, args.run_mode, dtype, args.scale_value, args.skip_golden)

        logging.info("\n" + "=" * 70)
        logging.info("All tests passed!")
        logging.info("=" * 70)
        logging.info("Available kernels (Stage 3: Multi-datatype support):")
        logging.info("")
        logging.info("  1. flash_attention_score_kernel_with_mask (BF16/FP32)")
        logging.info("     - Basic attention with mask support")
        logging.info("     - Configurable scale_value parameter")
        logging.info("     - Outputs: attention_out + softmax_max + softmax_sum")
        logging.info("")
        logging.info("  2. flash_attention_score_kernel_with_pse_and_dropout (BF16/FP32)")
        logging.info("     - Full features: Mask + PSE + Dropout")
        logging.info("     - PSE modes: pse_type 0,1,2,3 (add/mul order control)")
        logging.info("     - Dropout: drop_mask + keep_prob")
        logging.info("     - Configurable scale_value parameter")
        logging.info("     - Outputs: attention_out + softmax_max + softmax_sum")
        logging.info("")
        logging.info("  Precision Strategy:")
        logging.info("  - BF16: Input -> FP32 compute -> Output (with cast)")
        logging.info("  - FP32: Input -> FP32 compute -> Output (no cast)")
        logging.info("=" * 70)
    except Exception as e:
        logging.info(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()