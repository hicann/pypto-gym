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
3. Test suite for all kernels

Kernels:
- with_mask_origin: Basic attention with mask (BF16, no softmax_max/sum output)
- with_mask: Attention with mask support (BF16)
- with_pse_and_dropout: Full features (PSE + Dropout + Mask) (BF16)
- Configurable scale_value parameter

Datatype Strategy:
- BF16: Input -> FP32 compute -> BF16 Output (with cast)
"""

import os
import math
import argparse
import logging
from typing import Optional
from dataclasses import dataclass
import torch
import torch_npu

import sys, os; _p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')): _p = os.path.dirname(_p)
sys.path.insert(0, os.path.join(_p, 'src')); sys.path.insert(0, os.path.join(_p, 'src', 'pypto_gym', 'ops', 'pypto_tile'))

import numpy as np
from numpy.testing import assert_allclose

from experimental.ops_transformer.flash_attention_score.flash_attention_score_impl import (
    flash_attention_score_kernel_with_mask_origin,
    flash_attention_score_kernel_with_mask,
    flash_attention_score_kernel_with_pse_and_dropout,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")


BATCH_SIZE = 2
NUM_HEADS = 4
SEQ_LEN_Q = 64
SEQ_LEN_KV = 64
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


BLOCK_SIZE_Q = 32
BLOCK_SIZE_KV = 64


def flash_attention_score_golden_origin(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    atten_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Golden reference for flash_attention_score_kernel_with_mask_origin.
    
    This version only returns output, without softmax_max and softmax_sum.
    Uses fixed scale = 1/sqrt(HEAD_DIM).
    
    Computation flow matches the kernel exactly:
    - Tiling: BLOCK_SIZE_Q=32, BLOCK_SIZE_KV=64
    - Q*K^T: BF16 matmul -> FP32 output
    - Mask: 0=valid, 1=masked -> valid_mask = 1-mask, applied after exp
    - Online softmax: m_ij computed over all scores (incl masked), then p_ij zeroed
    """
    b, n, sq, d = query.shape
    _, _, skv, _ = key.shape

    scale = 1.0 / math.sqrt(d)

    if atten_mask is not None:
        atten_mask_fp32 = atten_mask.float()
    else:
        atten_mask_fp32 = None

    output = torch.zeros(b, n, sq, d, dtype=torch.bfloat16, device=query.device)

    num_blocks_kv = (skv + BLOCK_SIZE_KV - 1) // BLOCK_SIZE_KV
    num_blocks_q = (sq + BLOCK_SIZE_Q - 1) // BLOCK_SIZE_Q

    for b_idx in range(b):
        for n_idx in range(n):
            for q_block_idx in range(num_blocks_q):
                q_start = q_block_idx * BLOCK_SIZE_Q
                cur_q_size = min(BLOCK_SIZE_Q, sq - q_start)

                q_block_2d = query[b_idx, n_idx, q_start:q_start + cur_q_size, :].reshape(cur_q_size, d)

                mi_update = torch.full((cur_q_size, 1), float('-inf'), dtype=torch.float32, device=query.device)
                li_update = torch.zeros(cur_q_size, 1, dtype=torch.float32, device=query.device)
                oi_update = torch.zeros(cur_q_size, d, dtype=torch.float32, device=query.device)

                for kv_block_idx in range(num_blocks_kv):
                    kv_start = kv_block_idx * BLOCK_SIZE_KV
                    cur_block_size = min(BLOCK_SIZE_KV, skv - kv_start)

                    k_block_2d = key[b_idx, n_idx, kv_start:kv_start + cur_block_size, :].reshape(cur_block_size, d)

                    scores = torch.matmul(q_block_2d.float(), k_block_2d.float().T)
                    scores_scaled = scores * scale

                    if atten_mask_fp32 is not None:
                        mask_block = atten_mask_fp32[q_start:q_start + cur_q_size, kv_start:kv_start + cur_block_size]
                        valid_mask = 1.0 - mask_block
                    else:
                        valid_mask = torch.ones(cur_q_size, cur_block_size, dtype=torch.float32, device=query.device)

                    m_ij = torch.amax(scores_scaled, dim=-1, keepdim=True)
                    s_ij_sub_m = scores_scaled - m_ij
                    p_ij = torch.exp(s_ij_sub_m)
                    p_ij = p_ij * valid_mask
                    l_ij = torch.sum(p_ij, dim=-1, keepdim=True)

                    v_block_2d = value[b_idx, n_idx, kv_start:kv_start + cur_block_size, :].reshape(cur_block_size, d).float()
                    o_ij = torch.matmul(p_ij, v_block_2d)

                    if kv_block_idx == 0:
                        mi_update = m_ij
                        li_update = l_ij
                        oi_update = o_ij
                    else:
                        mi_new = torch.maximum(mi_update, m_ij)
                        alpha = torch.exp(mi_update - mi_new)
                        beta = torch.exp(m_ij - mi_new)
                        li_update = alpha * li_update + beta * l_ij
                        oi_update = alpha * oi_update + beta * o_ij
                        mi_update = mi_new

                o_final = oi_update / li_update
                output[b_idx, n_idx, q_start:q_start + cur_q_size, :] = o_final.to(torch.bfloat16)

    return output


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
    
    Computation flow matches the kernel exactly:
    - Tiling: BLOCK_SIZE_Q=32, BLOCK_SIZE_KV=64
    - Q*K^T: BF16 matmul -> FP32 output
    - Mask: 0=valid, 1=masked -> valid_mask = 1-mask, applied after exp
    - Online softmax: m_ij computed over all scores (incl masked), then p_ij zeroed
    """
    b, n, sq, d = query.shape
    _, _, skv, _ = key.shape

    scale = scale_value if scale_value is not None else 1.0 / math.sqrt(d)

    if atten_mask is not None:
        atten_mask_fp32 = atten_mask.float()
    else:
        atten_mask_fp32 = None

    output = torch.zeros(b, n, sq, d, dtype=torch.bfloat16, device=query.device)
    softmax_max = torch.zeros(b, n, sq, 1, dtype=torch.float32, device=query.device)
    softmax_sum = torch.zeros(b, n, sq, 1, dtype=torch.float32, device=query.device)

    num_blocks_kv = (skv + BLOCK_SIZE_KV - 1) // BLOCK_SIZE_KV
    num_blocks_q = (sq + BLOCK_SIZE_Q - 1) // BLOCK_SIZE_Q

    for b_idx in range(b):
        for n_idx in range(n):
            for q_block_idx in range(num_blocks_q):
                q_start = q_block_idx * BLOCK_SIZE_Q
                cur_q_size = min(BLOCK_SIZE_Q, sq - q_start)

                q_block_2d = query[b_idx, n_idx, q_start:q_start + cur_q_size, :].reshape(cur_q_size, d)

                mi_update = torch.full((cur_q_size, 1), float('-inf'), dtype=torch.float32, device=query.device)
                li_update = torch.zeros(cur_q_size, 1, dtype=torch.float32, device=query.device)
                oi_update = torch.zeros(cur_q_size, d, dtype=torch.float32, device=query.device)

                for kv_block_idx in range(num_blocks_kv):
                    kv_start = kv_block_idx * BLOCK_SIZE_KV
                    cur_block_size = min(BLOCK_SIZE_KV, skv - kv_start)

                    k_block_2d = key[b_idx, n_idx, kv_start:kv_start + cur_block_size, :].reshape(cur_block_size, d)

                    scores = torch.matmul(q_block_2d.float(), k_block_2d.float().T)
                    scores_scaled = scores * scale

                    if atten_mask_fp32 is not None:
                        mask_block = atten_mask_fp32[q_start:q_start + cur_q_size, kv_start:kv_start + cur_block_size]
                        valid_mask = 1.0 - mask_block
                    else:
                        valid_mask = torch.ones(cur_q_size, cur_block_size, dtype=torch.float32, device=query.device)

                    m_ij = torch.amax(scores_scaled, dim=-1, keepdim=True)
                    s_ij_sub_m = scores_scaled - m_ij
                    p_ij = torch.exp(s_ij_sub_m)
                    p_ij = p_ij * valid_mask
                    l_ij = torch.sum(p_ij, dim=-1, keepdim=True)

                    v_block_2d = value[b_idx, n_idx, kv_start:kv_start + cur_block_size, :].reshape(cur_block_size, d).float()
                    o_ij = torch.matmul(p_ij, v_block_2d)

                    if kv_block_idx == 0:
                        mi_update = m_ij
                        li_update = l_ij
                        oi_update = o_ij
                    else:
                        mi_new = torch.maximum(mi_update, m_ij)
                        alpha = torch.exp(mi_update - mi_new)
                        beta = torch.exp(m_ij - mi_new)
                        li_update = alpha * li_update + beta * l_ij
                        oi_update = alpha * oi_update + beta * o_ij
                        mi_update = mi_new

                o_final = oi_update / li_update
                output[b_idx, n_idx, q_start:q_start + cur_q_size, :] = o_final.to(torch.bfloat16)
                softmax_max[b_idx, n_idx, q_start:q_start + cur_q_size, :] = mi_update.reshape(cur_q_size, 1)
                softmax_sum[b_idx, n_idx, q_start:q_start + cur_q_size, :] = li_update.reshape(cur_q_size, 1)

    return output, softmax_max, softmax_sum


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
    
    Computation flow matches the kernel exactly:
    - Tiling: BLOCK_SIZE_Q=32, BLOCK_SIZE_KV=64
    - Q*K^T: BF16 matmul -> FP32 output
    - PSE: BF16 -> FP32 cast; pse_type==1: (PSE+QKT)*scale, else: QKT*scale+PSE
    - Mask: 0=valid, 1=masked -> valid_mask = 1-mask, applied after exp
    - Dropout: p_ij *= drop_mask, if keep_prob<1: p_ij /= keep_prob
    - Online softmax: m_ij computed over all scores (incl masked), then p_ij zeroed
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

    if atten_mask is not None:
        atten_mask_fp32 = atten_mask.float()
    else:
        atten_mask_fp32 = None

    output = torch.zeros(b, n, sq, d, dtype=torch.bfloat16, device=query.device)
    softmax_max = torch.zeros(b, n, sq, 1, dtype=torch.float32, device=query.device)
    softmax_sum = torch.zeros(b, n, sq, 1, dtype=torch.float32, device=query.device)

    num_blocks_kv = (skv + BLOCK_SIZE_KV - 1) // BLOCK_SIZE_KV
    num_blocks_q = (sq + BLOCK_SIZE_Q - 1) // BLOCK_SIZE_Q

    for b_idx in range(b):
        for n_idx in range(n):
            for q_block_idx in range(num_blocks_q):
                q_start = q_block_idx * BLOCK_SIZE_Q
                cur_q_size = min(BLOCK_SIZE_Q, sq - q_start)

                q_block_2d = query[b_idx, n_idx, q_start:q_start + cur_q_size, :].reshape(cur_q_size, d)

                mi_update = torch.full((cur_q_size, 1), float('-inf'), dtype=torch.float32, device=query.device)
                li_update = torch.zeros(cur_q_size, 1, dtype=torch.float32, device=query.device)
                oi_update = torch.zeros(cur_q_size, d, dtype=torch.float32, device=query.device)

                for kv_block_idx in range(num_blocks_kv):
                    kv_start = kv_block_idx * BLOCK_SIZE_KV
                    cur_block_size = min(BLOCK_SIZE_KV, skv - kv_start)

                    k_block_2d = key[b_idx, n_idx, kv_start:kv_start + cur_block_size, :].reshape(cur_block_size, d)

                    scores = torch.matmul(q_block_2d.float(), k_block_2d.float().T)

                    pse_block_2d = pse[b_idx, n_idx, q_start:q_start + cur_q_size, kv_start:kv_start + cur_block_size].reshape(cur_q_size, cur_block_size)
                    pse_fp32 = pse_block_2d.float()

                    if pse_type == 1:
                        scores = scores + pse_fp32
                        scores_scaled = scores * scale
                    else:
                        scores_scaled = scores * scale
                        scores_scaled = scores_scaled + pse_fp32

                    if atten_mask_fp32 is not None:
                        mask_block = atten_mask_fp32[q_start:q_start + cur_q_size, kv_start:kv_start + cur_block_size]
                        valid_mask = 1.0 - mask_block
                    else:
                        valid_mask = torch.ones(cur_q_size, cur_block_size, dtype=torch.float32, device=query.device)

                    m_ij = torch.amax(scores_scaled, dim=-1, keepdim=True)
                    s_ij_sub_m = scores_scaled - m_ij
                    p_ij = torch.exp(s_ij_sub_m)
                    p_ij = p_ij * valid_mask

                    drop_mask_block = drop_mask[q_start:q_start + cur_q_size, kv_start:kv_start + cur_block_size]
                    p_ij = p_ij * drop_mask_block

                    if keep_prob < 1.0:
                        p_ij = p_ij / keep_prob

                    l_ij = torch.sum(p_ij, dim=-1, keepdim=True)

                    v_block_2d = value[b_idx, n_idx, kv_start:kv_start + cur_block_size, :].reshape(cur_block_size, d).float()
                    o_ij = torch.matmul(p_ij, v_block_2d)

                    if kv_block_idx == 0:
                        mi_update = m_ij
                        li_update = l_ij
                        oi_update = o_ij
                    else:
                        mi_new = torch.maximum(mi_update, m_ij)
                        alpha = torch.exp(mi_update - mi_new)
                        beta = torch.exp(m_ij - mi_new)
                        li_update = alpha * li_update + beta * l_ij
                        oi_update = alpha * oi_update + beta * o_ij
                        mi_update = mi_new

                o_final = oi_update / li_update
                output[b_idx, n_idx, q_start:q_start + cur_q_size, :] = o_final.to(torch.bfloat16)
                softmax_max[b_idx, n_idx, q_start:q_start + cur_q_size, :] = mi_update.reshape(cur_q_size, 1)
                softmax_sum[b_idx, n_idx, q_start:q_start + cur_q_size, :] = li_update.reshape(cur_q_size, 1)

    return output, softmax_max, softmax_sum


def test_kernel_with_mask_origin(device_id=None, run_mode: str = "npu", skip_golden: bool = False,
                                  batch_size=None, num_heads=None, seq_len_q=None,
                                  seq_len_kv=None, head_dim=None):
    """Test flash_attention_score_kernel_with_mask_origin."""
    bs = batch_size if batch_size is not None else BATCH_SIZE
    nh = num_heads if num_heads is not None else NUM_HEADS
    sq = seq_len_q if seq_len_q is not None else SEQ_LEN_Q
    skv = seq_len_kv if seq_len_kv is not None else SEQ_LEN_KV
    hd = head_dim if head_dim is not None else HEAD_DIM

    logging.info("=" * 70)
    logging.info("Test: flash_attention_score_kernel_with_mask_origin (BF16)")
    logging.info("=" * 70)

    if device_id is not None:
        torch.npu.set_device(device_id)

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'

    query = torch.randn(bs, nh, sq, hd, dtype=torch.bfloat16, device=device)
    key = torch.randn(bs, nh, skv, hd, dtype=torch.bfloat16, device=device)
    value = torch.randn(bs, nh, skv, hd, dtype=torch.bfloat16, device=device)

    atten_mask = torch.zeros(sq, skv, dtype=torch.uint8, device=device)
    atten_mask[:, skv // 2:] = 1

    output = torch.empty(bs, nh, sq, hd, dtype=torch.bfloat16, device=device)

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
    scale_value: Optional[float] = None,
    skip_golden: bool = False,
    batch_size=None, num_heads=None, seq_len_q=None,
    seq_len_kv=None, head_dim=None,
):
    """Test flash_attention_score_kernel_with_mask."""
    bs = batch_size if batch_size is not None else BATCH_SIZE
    nh = num_heads if num_heads is not None else NUM_HEADS
    sq = seq_len_q if seq_len_q is not None else SEQ_LEN_Q
    skv = seq_len_kv if seq_len_kv is not None else SEQ_LEN_KV
    hd = head_dim if head_dim is not None else HEAD_DIM

    logging.info("=" * 70)
    logging.info("Test: flash_attention_score_kernel_with_mask (BF16)")
    logging.info("=" * 70)

    if device_id is not None:
        torch.npu.set_device(device_id)

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    torch_dtype = torch.bfloat16
    
    query = torch.randn(bs, nh, sq, hd, dtype=torch_dtype, device=device)
    key = torch.randn(bs, nh, skv, hd, dtype=torch_dtype, device=device)
    value = torch.randn(bs, nh, skv, hd, dtype=torch_dtype, device=device)

    atten_mask = torch.zeros(sq, skv, dtype=torch.uint8, device=device)
    atten_mask[:, skv // 2:] = 1

    output = torch.empty(bs, nh, sq, hd, dtype=torch_dtype, device=device)
    softmax_max = torch.empty(bs, nh, sq, 1, dtype=torch.float32, device=device)
    softmax_sum = torch.empty(bs, nh, sq, 1, dtype=torch.float32, device=device)

    atten_mask_fp32 = atten_mask.float()
    
    default_scale = 1.0 / math.sqrt(hd)
    test_scale = scale_value if scale_value is not None else default_scale
    
    logging.info(f"Scale value: {test_scale}")

    flash_attention_score_kernel_with_mask(
        query, key, value, atten_mask_fp32, output, softmax_max, softmax_sum, test_scale
    )

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
        logging.info("  Kernel with_mask test passed for BF16!")
        return

    if run_mode == "npu":
        golden, golden_max, golden_sum = flash_attention_score_golden(query, key, value, atten_mask, test_scale)

        output_fp32 = output.float()
        golden_fp32 = golden.float()

        rtol = 0.0078125
        atol = 0.0001

        assert_allclose(
            output_fp32.cpu().numpy().flatten(),
            golden_fp32.cpu().numpy().flatten(),
            rtol=rtol,
            atol=atol
        )
        
        logging.info("  Kernel with_mask test passed for BF16!")


def test_kernel_with_pse_and_dropout(
    device_id=None,
    run_mode: str = "npu",
    scale_value: Optional[float] = None,
    skip_golden: bool = False,
    batch_size=None, num_heads=None, seq_len_q=None,
    seq_len_kv=None, head_dim=None,
):
    """Test flash_attention_score_kernel_with_pse_and_dropout."""
    bs = batch_size if batch_size is not None else BATCH_SIZE
    nh = num_heads if num_heads is not None else NUM_HEADS
    sq = seq_len_q if seq_len_q is not None else SEQ_LEN_Q
    skv = seq_len_kv if seq_len_kv is not None else SEQ_LEN_KV
    hd = head_dim if head_dim is not None else HEAD_DIM

    logging.info("\n" + "=" * 70)
    logging.info("Test: flash_attention_score_kernel_with_pse_and_dropout (BF16)")
    logging.info("=" * 70)

    if device_id is not None:
        torch.npu.set_device(device_id)

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    torch_dtype = torch.bfloat16

    query = torch.randn(bs, nh, sq, hd, dtype=torch_dtype, device=device)
    key = torch.randn(bs, nh, skv, hd, dtype=torch_dtype, device=device)
    value = torch.randn(bs, nh, skv, hd, dtype=torch_dtype, device=device)

    atten_mask = torch.zeros(sq, skv, dtype=torch.uint8, device=device)
    atten_mask[:, skv // 4:] = 1

    pse = torch.randn(bs, nh, sq, skv, dtype=torch_dtype, device=device)

    drop_mask = torch.ones(sq, skv, dtype=torch.float32, device=device)
    keep_prob = 0.8

    output = torch.empty(bs, nh, sq, hd, dtype=torch_dtype, device=device)
    softmax_max = torch.empty(bs, nh, sq, 1, dtype=torch.float32, device=device)
    softmax_sum = torch.empty(bs, nh, sq, 1, dtype=torch.float32, device=device)

    atten_mask_fp32 = atten_mask.float()

    logging.info(f"Input shape: query={query.shape}, key={key.shape}, value={value.shape}")
    logging.info(f"PSE shape: {pse.shape}, drop_mask shape: {drop_mask.shape}")
    
    default_scale = 1.0 / math.sqrt(hd)
    test_scale = scale_value if scale_value is not None else default_scale
    
    logging.info(f"Scale value: {test_scale}")

    for pse_type in [0, 1]:
        logging.info(f"\nTesting pse_type={pse_type}")
        
        flash_attention_score_kernel_with_pse_and_dropout(
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
            logging.info(f"  Kernel with_pse_and_dropout test passed for pse_type={pse_type} (BF16)!")
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
            
            rtol = 0.0078125
            atol = 0.0001
            
            assert_allclose(
                output_fp32.cpu().numpy().flatten(),
                golden_fp32.cpu().numpy().flatten(),
                rtol=rtol,
                atol=atol
            )
            
            logging.info(f"  Kernel with_pse_and_dropout test passed for pse_type={pse_type} (BF16)!")


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
        torch.npu.set_device(device_id)
        logging.info(f"Running on NPU device {device_id}\n")

    try:
        if args.kernel == "all":
            test_kernel_with_mask_origin(device_id, args.run_mode, args.skip_golden)
            test_kernel_with_mask(device_id, args.run_mode, args.scale_value, args.skip_golden)
            test_kernel_with_pse_and_dropout(device_id, args.run_mode, args.scale_value, args.skip_golden)
        elif args.kernel == "mask_origin":
            test_kernel_with_mask_origin(device_id, args.run_mode, args.skip_golden)
        elif args.kernel == "mask":
            test_kernel_with_mask(device_id, args.run_mode, args.scale_value, args.skip_golden)
        elif args.kernel == "pse_dropout":
            test_kernel_with_pse_and_dropout(device_id, args.run_mode, args.scale_value, args.skip_golden)

        logging.info("\n" + "=" * 70)
        logging.info("All tests passed!")
        logging.info("=" * 70)
        logging.info("Available kernels:")
        logging.info("")
        logging.info("  1. flash_attention_score_kernel_with_mask_origin (BF16)")
        logging.info("     - Basic attention with mask support")
        logging.info("     - Fixed scale = 1/sqrt(HEAD_DIM)")
        logging.info("     - Outputs: attention_out only")
        logging.info("")
        logging.info("  2. flash_attention_score_kernel_with_mask (BF16)")
        logging.info("     - Basic attention with mask support")
        logging.info("     - Configurable scale_value parameter")
        logging.info("     - Outputs: attention_out + softmax_max + softmax_sum")
        logging.info("")
        logging.info("  3. flash_attention_score_kernel_with_pse_and_dropout (BF16)")
        logging.info("     - Full features: Mask + PSE + Dropout")
        logging.info("     - PSE modes: pse_type 0,1,2,3 (add/mul order control)")
        logging.info("     - Dropout: drop_mask + keep_prob")
        logging.info("     - Configurable scale_value parameter")
        logging.info("     - Outputs: attention_out + softmax_max + softmax_sum")
        logging.info("")
        logging.info("  Precision Strategy:")
        logging.info("  - BF16: Input -> FP32 compute -> BF16 Output (with cast)")
        logging.info("=" * 70)
    except Exception as e:
        logging.info(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()