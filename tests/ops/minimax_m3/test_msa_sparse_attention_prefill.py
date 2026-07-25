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
"""
MiniMax M3 MSA Main Branch Test Module

Data flow:
  - Q: [n, hq, dh]
  - K/V blocks: [topk, bk, hkv, dh] (gathered from raw KV by topk_indices)
  - topk_indices: [topk] int32 (selected KV block original indices, must include last block)
  - causal_mask: [bk, bk] (fixed lower-triangular)
  - block_mask: [num_blocks, topk, bk, bk] (precomputed from topk_indices + causal_mask)
  - Kernel: loop hkv(parallel) -> g -> qb -> i(topk), online softmax
  - For each selected KV block topk_indices[i], it contributes to all query blocks
    whose index >= topk_indices[i]; when query block index == topk_indices[i],
    causal mask is applied (within-block lower-triangular).
  - All causal logic is precomputed into block_mask on host; kernel has zero
    dynamic conditions (required by PyPTO AssignMemoryType pass).

Parameters: Hq=64, Hkv=4 (GQA group=16), topk=16, N=2048, head_dim=128, Bk=128.
"""
import sys
import os
import random
import logging
_DIR = os.path.dirname(os.path.abspath(__file__))
_p = _DIR
while _p != "/" and not os.path.isdir(os.path.join(_p, "src")):
    _p = os.path.dirname(_p)
sys.path.insert(0, os.path.join(_p, "src"))

import torch
import torch_npu
import pypto
from pypto_gym.ops.pypto_tensor.minimax.msa_main_branch_impl import msa_main_branch_prefill


HQ = 64
HKV = 4
TOPK = 16
N = 2048
HEAD_DIM = 128
BK = 128
NUM_BLOCKS = N // BK
GROUP = HQ // HKV
MASK_NEG = -1e9


def gen_inputs(dtype=torch.float32):
    query = torch.randn(N, HQ, HEAD_DIM, dtype=dtype)
    key = torch.randn(N, HKV, HEAD_DIM, dtype=dtype)
    value = torch.randn(N, HKV, HEAD_DIM, dtype=dtype)
    return query, key, value


def gen_topk_indices(seed=42):
    rng = random.Random(seed)
    all_blocks = list(range(NUM_BLOCKS))
    rng.shuffle(all_blocks)
    selected = all_blocks[:TOPK]
    if (NUM_BLOCKS - 1) not in selected:
        replace_idx = rng.randrange(len(selected))
        selected[replace_idx] = NUM_BLOCKS - 1
    return torch.tensor(selected, dtype=torch.int32)


def gen_causal_mask():
    row_idx = torch.arange(BK).view(-1, 1)
    col_idx = torch.arange(BK).view(1, -1)
    return torch.where(col_idx <= row_idx, 0.0, float(MASK_NEG)).to(torch.float32)


def gen_block_mask(topk_indices, causal_mask):
    """Precompute per (qb, i) mask. Returns [num_blocks, topk, bk, bk].

    For each (qb, i) with kv_seq = topk_indices[i]:
      - kv_seq > qb: MASK_NEG (block not yet reachable, softmax skips it)
      - kv_seq == qb: causal_mask (within-block lower-triangular)
      - kv_seq < qb: 0.0 (full attention, no mask)
    """
    block_mask = torch.full(
        (NUM_BLOCKS, TOPK, BK, BK), float(MASK_NEG), dtype=torch.float32
    )
    for qb in range(NUM_BLOCKS):
        for i in range(TOPK):
            kv_seq = int(topk_indices[i])
            if kv_seq > qb:
                pass
            elif kv_seq == qb:
                block_mask[qb, i] = causal_mask
            else:
                block_mask[qb, i] = 0.0
    return block_mask


def golden_msa_main_branch(query, key_blocks, value_blocks, topk_indices, causal_mask):
    """
    Golden: sparse softmax attention on selected KV blocks.

    query : [n, hq, dh]
    key_blocks : [topk, bk, hkv, dh]
    value_blocks : [topk, bk, hkv, dh]
    topk_indices : [topk] int32 (original block indices)
    causal_mask : [bk, bk]
    """
    scale = 1.0 / (HEAD_DIM ** 0.5)
    q_4d = query.reshape(NUM_BLOCKS, BK, HQ, HEAD_DIM)
    output = torch.zeros(N, HQ, HEAD_DIM, dtype=torch.float32)

    for h in range(HQ):
        h_kv = h // GROUP
        for qb in range(NUM_BLOCKS):
            q = q_4d[qb, :, h, :]
            scores_list = []
            v_list = []
            for i in range(TOPK):
                kv_seq = int(topk_indices[i])
                if kv_seq > qb:
                    continue
                k = key_blocks[i, :, h_kv, :]
                v = value_blocks[i, :, h_kv, :]
                s = torch.matmul(q, k.t()) * scale
                if kv_seq == qb:
                    s = s + causal_mask
                scores_list.append(s)
                v_list.append(v)
            if not scores_list:
                continue
            scores_all = torch.cat(scores_list, dim=-1)
            v_all = torch.cat(v_list, dim=0)
            p = torch.softmax(scores_all, dim=-1)
            out = torch.matmul(p, v_all)
            qi = qb * BK
            output[qi:qi + BK, h, :] = out
    return output


def compare(actual, expected, name="tensor", atol_abs=1e-3, atol_rel=1e-3):
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{name}: shape mismatch actual={tuple(actual.shape)} expected={tuple(expected.shape)}"
        )
    if not torch.isfinite(actual).all():
        raise AssertionError(f"{name}: actual contains non-finite values")
    if not torch.isfinite(expected).all():
        raise AssertionError(f"{name}: expected contains non-finite values")

    diff = torch.abs(actual - expected)
    tol = atol_abs + atol_rel * torch.abs(expected)
    out_count = (diff > tol).sum().item()
    total = actual.numel()
    max_diff = diff.max().item()

    if out_count > 0:
        bad_idx = torch.nonzero(diff > tol, as_tuple=False)[0].tolist()
        idx = tuple(bad_idx)
        raise AssertionError(
            f"{name}: {out_count}/{total} out of tolerance, "
            f"max_diff={max_diff:.6e}, atol_abs={atol_abs}, atol_rel={atol_rel}, "
            f"first_bad_idx={bad_idx}, "
            f"actual={actual[idx].item():.9e}, expected={expected[idx].item():.9e}"
        )
    logging.info(
        f"[PASS] {name}: shape={tuple(actual.shape)}, max_diff={max_diff:.6e}"
    )


def do_test():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    collect_swimlane = bool(os.environ.get("COLLECT_SWIMLANE"))
    if collect_swimlane:
        pypto.set_debug_options(runtime_debug_mode=1)
        logging.info("[swimlane] runtime_debug_mode=1 enabled")

    torch.manual_seed(0)
    query, key, value = gen_inputs(torch.float32)
    topk_indices = gen_topk_indices(seed=42)
    causal_mask = gen_causal_mask()

    key_blocks_4d = key.reshape(NUM_BLOCKS, BK, HKV, HEAD_DIM)[topk_indices.long()]
    value_blocks_4d = value.reshape(NUM_BLOCKS, BK, HKV, HEAD_DIM)[topk_indices.long()]

    block_mask = gen_block_mask(topk_indices, causal_mask)

    golden = golden_msa_main_branch(query, key_blocks_4d, value_blocks_4d, topk_indices, causal_mask)

    query_npu = query.npu()
    key_blocks_npu = key_blocks_4d.reshape(TOPK * BK, HKV, HEAD_DIM).npu()
    value_blocks_npu = value_blocks_4d.reshape(TOPK * BK, HKV, HEAD_DIM).npu()
    block_mask_npu = block_mask.npu()
    output_npu = torch.zeros_like(golden).npu()

    kernel = msa_main_branch_prefill(HQ, HKV, HEAD_DIM, BK, TOPK)
    kernel(query_npu, key_blocks_npu, value_blocks_npu, block_mask_npu, output_npu)
    torch.npu.synchronize()

    compare(output_npu.cpu(), golden, name="msa_main_branch_output")

    if collect_swimlane:
        perf_path = pypto.pypto_impl.LogTopFolder()
        logging.info("[swimlane] Perf log dir: %s", perf_path)
        if perf_path and os.path.isdir(perf_path):
            merged = os.path.join(perf_path, "merged_swimlane.json")
            if os.path.exists(merged):
                logging.info(
                    "[swimlane] merged_swimlane.json: %.2f MB",
                    os.path.getsize(merged) / 1024 / 1024,
                )
                logging.info("[swimlane] View at https://ui.perfetto.dev/")
            else:
                logging.info("[swimlane] merged_swimlane.json NOT generated")
                logging.info("[swimlane] Contents: %s", sorted(os.listdir(perf_path)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    do_test()
