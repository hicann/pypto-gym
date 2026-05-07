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
LightningIndexer Operator for PyPTO

This operator computes the Top-k indices based on:
    Indices = Top-k(W ⊙ ReLU(Q @ K^T))

For a given Index Query Q_index of shape [B, Sq, N, D] and Index Key K_index of shape [B, Skv, N, D],
with weights W of shape [B, Sq, N], the operator returns the top-k indices for each token.
"""

import os
import sys
import argparse
import logging
import torch
import pypto

logging.basicConfig(level=logging.INFO, format='%(message)s')

BATCH_SIZE = 1
NUM_HEADS = 8
SEQ_LEN_Q = 64
SEQ_LEN_KV = 64
HEAD_DIM = 128
TOPK = 8


def get_device_id():
    """
    Get and validate TILE_FWK_DEVICE_ID from environment variable.

    Returns:
        int: The device ID if valid, None otherwise.
    """
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        logging.info("Please set the environment variable TILE_FWK_DEVICE_ID before running:")
        logging.info("  export TILE_FWK_DEVICE_ID=0")
        return None

    try:
        device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
        return device_id
    except ValueError:
        logging.error(f"ERROR: TILE_FWK_DEVICE_ID must be an integer, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None


def lightning_indexer_golden(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    topk: int
) -> torch.Tensor:
    """
    PyTorch reference implementation of LightningIndexer.

    Args:
        query: [B, Sq, N, D] - Query tensor
        key: [B, Skv, N, D] - Key tensor
        weights: [B, Sq, N] - Weights tensor
        topk: Number of top indices to return

    Returns:
        indices: [B, Sq, N, topk] - Top-k indices
    """

    query_t = query.transpose(1, 2)
    key_t = key.transpose(1, 2)

    scores = torch.matmul(query_t, key_t.transpose(-2, -1))

    scores_relu = torch.relu(scores)

    weights_expanded = weights.transpose(1, 2).unsqueeze(-1)
    weighted_scores = weights_expanded * scores_relu

    _, indices = torch.topk(weighted_scores, topk, dim=-1, largest=True)

    indices = indices.transpose(1, 2)

    return indices


@pypto.frontend.jit(
    runtime_options={"stitch_function_max_num": 128, "device_sched_mode": 1},
    pass_options={"cube_l1_reuse_setting": {0: 8}}
)
def lightning_indexer_kernel(
    query: pypto.Tensor([pypto.DYNAMIC, SEQ_LEN_Q, NUM_HEADS, HEAD_DIM], pypto.DT_BF16),
    key: pypto.Tensor([pypto.DYNAMIC, SEQ_LEN_KV, NUM_HEADS, HEAD_DIM], pypto.DT_BF16),
    weights: pypto.Tensor([pypto.DYNAMIC, NUM_HEADS, SEQ_LEN_Q, 1], pypto.DT_BF16),
    indices: pypto.Tensor([pypto.DYNAMIC, SEQ_LEN_Q, NUM_HEADS, TOPK], pypto.DT_INT32),
):
    """
    LightningIndexer kernel implementation.

    Computes: Indices = Top-k(W ⊙ ReLU(Q @ K^T))

    Input shapes:
        query: [B, Sq, N, D]
        key: [B, Skv, N, D]
        weights: [B, N, Sq, 1] (pre-transposed and expanded)

    Output shape:
        indices: [B, Sq, N, topk]
    """
    batch_size = query.shape[0]
    batch_tile = 1

    pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
    pypto.set_vec_tile_shapes(1, 1, SEQ_LEN_Q, HEAD_DIM)

    batch_loop = (batch_size + batch_tile - 1) // batch_tile

    for batch_idx in pypto.loop(batch_loop, name="LOOP_BATCH", idx_name="batch_idx"):
        act_batch_tile = (batch_size - batch_idx * batch_tile).min(batch_tile)

        query_tile = pypto.view(query, [batch_tile, SEQ_LEN_Q, NUM_HEADS, HEAD_DIM],
                                [batch_idx * batch_tile, 0, 0, 0],
                                valid_shape=[act_batch_tile, SEQ_LEN_Q, NUM_HEADS, HEAD_DIM])
        key_tile = pypto.view(key, [batch_tile, SEQ_LEN_KV, NUM_HEADS, HEAD_DIM],
                             [batch_idx * batch_tile, 0, 0, 0],
                             valid_shape=[act_batch_tile, SEQ_LEN_KV, NUM_HEADS, HEAD_DIM])
        weights_tile = pypto.view(weights, [batch_tile, NUM_HEADS, SEQ_LEN_Q, 1],
                                  [batch_idx * batch_tile, 0, 0, 0],
                                  valid_shape=[act_batch_tile, NUM_HEADS, SEQ_LEN_Q, 1])

        query_t = pypto.transpose(query_tile, 1, 2)
        key_t = pypto.transpose(key_tile, 1, 2)

        scores = pypto.matmul(query_t, key_t, out_dtype=pypto.DT_BF16, b_trans=True)

        pypto.set_vec_tile_shapes(1, 1, SEQ_LEN_Q, SEQ_LEN_KV)
        scores_relu = pypto.relu(scores)

        weighted_scores = pypto.mul(scores_relu, weights_tile)

        weighted_scores_fp32 = pypto.cast(weighted_scores, pypto.DT_FP32)

        _, topk_indices = pypto.topk(weighted_scores_fp32, TOPK, dim=-1, largest=True)

        indices_t = pypto.cast(topk_indices, pypto.DT_INT32)
        indices_tile = pypto.transpose(indices_t, 1, 2)

        indices[batch_idx * batch_tile:, 0:, 0:, 0:] = indices_tile


def test_lightning_indexer(device_id=None, run_mode: str = "npu", batch_size: int = BATCH_SIZE) -> None:
    """Test LightningIndexer function."""
    logging.info("=" * 60)
    logging.info("Test: LightningIndexer")
    logging.info("=" * 60)

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'

    query = torch.randn(batch_size, SEQ_LEN_Q, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    key = torch.randn(batch_size, SEQ_LEN_KV, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    weights_raw = torch.randn(batch_size, SEQ_LEN_Q, NUM_HEADS, dtype=torch.bfloat16, device=device)

    weights = weights_raw.transpose(1, 2).unsqueeze(-1).contiguous()

    indices = torch.empty(batch_size, SEQ_LEN_Q, NUM_HEADS, TOPK, dtype=torch.int32, device=device)

    lightning_indexer_kernel(query, key, weights, indices)

    golden_indices = lightning_indexer_golden(query, key, weights_raw, TOPK)

    logging.info(f"Input query shape: {query.shape}")
    logging.info(f"Input key shape: {key.shape}")
    logging.info(f"Input weights shape: {weights.shape}")
    logging.info(f"Output indices shape: {indices.shape}")

    if run_mode == "npu":
        match = torch.allclose(indices, golden_indices.to(torch.int32), rtol=0, atol=0)
        match_count = (indices == golden_indices.to(torch.int32)).sum().item()
        total_count = indices.numel()
        match_ratio = match_count / total_count * 100

        logging.info(f"Exact match: {match}")
        logging.info(f"Match ratio: {match_ratio:.2f}% ({match_count}/{total_count})")

        if not match:
            logging.info("Sample indices (first batch, first head):")
            logging.info(f"  PyPTO:   {indices[0, 0, 0, :].tolist()}")
            logging.info(f"  PyTorch: {golden_indices[0, 0, 0, :].tolist()}")

    logging.info("✓ LightningIndexer test completed")
    logging.info("")


def main():
    """Run LightningIndexer examples."""
    parser = argparse.ArgumentParser(
        description="PyPTO LightningIndexer Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--run_mode',
        type=str,
        nargs='?',
        default='npu',
        choices=["npu"],
        help='Run mode, currently only support npu.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        nargs='?',
        default=1,
        help='Batch size for testing (supports dynamic batch size).'
    )
    args = parser.parse_args()

    logging.info("\n" + "=" * 60)
    logging.info("PyPTO LightningIndexer Example")
    logging.info("=" * 60 + "\n")

    device_id = None
    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        import torch_npu
        torch.npu.set_device(device_id)
        logging.info("Running on NPU...")
        logging.info("Make sure CANN environment is configured and NPU is available\n")

    try:
        test_lightning_indexer(device_id, args.run_mode, args.batch_size)
    except Exception as e:
        logging.info(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
