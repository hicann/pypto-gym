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
from dataclasses import dataclass
import logging
import pytest
import pypto
from conftest import duration_estimate


@dataclass
class NSASimpleParamsObj:
    def __init__(self, block_size=128, topk=2048, **kwargs):
        self.block_size = int(block_size)
        self.topk = int(topk)
        defaults = {
            "n1": 128,
            "n2": 1,
            "idx_n_heads": 64,
            "idx_head_dim": 128,
            "q_lora_rank": 1536,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "kv_lora_rank": 512,
        }
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, int(v))


@dataclass
class GatherInputs:
    top_k_indices: pypto.tensor
    k_nope_cache: pypto.tensor
    k_rope_cache: pypto.tensor
    block_table: pypto.tensor
    act_seqs: pypto.tensor
    gather_res: pypto.tensor
    nsa_params: NSASimpleParamsObj
    b: pypto.symbolic_scalar
    s1: pypto.symbolic_scalar


def gather_after_prolog_compute(args: GatherInputs):
    top_k_indices = args.top_k_indices
    k_nope_cache = args.k_nope_cache
    k_rope_cache = args.k_rope_cache
    block_table = args.block_table
    act_seqs = args.act_seqs
    gather_res = args.gather_res
    nsa_params = args.nsa_params
    b = args.b
    s1 = args.s1
    d_n = k_nope_cache.shape[-1]
    d_r = k_rope_cache.shape[-1]
    n2 = top_k_indices.shape[2]
    block_size = nsa_params.block_size
    topk = nsa_params.topk
    unroll_list = {64, 32, 16, 8, 4, 2, 1}
    input_tensors = [top_k_indices, k_nope_cache,
                     k_rope_cache, block_table, act_seqs]
    output_tensors = [gather_res]
    with pypto.function("main", *input_tensors, *output_tensors):
        for b_idx in pypto.loop(b, submit_before_loop=True):
            for s1_idx in pypto.loop(s1):
                for n2_idx in pypto.loop(n2):
                    pypto.set_semantic_label("gather0")
                    cur_kv_seq = act_seqs[b_idx]
                    top_k_loop = ((cur_kv_seq - s1 + 1 + s1_idx).max(0).min(topk))
                    for topk_idx in pypto.loop(top_k_loop, unroll_list=unroll_list):
                        pypto.set_vec_tile_shapes(1, 1, 1, 16)
                        topk_index = top_k_indices[
                            b_idx, s1_idx, n2_idx, topk_idx
                        ]
                        block_idx_in_batch = (
                            topk_index // block_size
                        )
                        tail = topk_index % block_size
                        slc_block_idx = block_table[
                            b_idx, block_idx_in_batch
                        ]
                        pypto.set_vec_tile_shapes(1, d_n)
                        kv_slc_block = pypto.view(
                            k_nope_cache,
                            [1, d_n],
                            [slc_block_idx * block_size + tail, 0],
                        )
                        kr_slc_block = pypto.view(
                            k_rope_cache,
                            [1, d_r],
                            [slc_block_idx * block_size + tail, 0],
                        )
                        pypto.set_semantic_label("gather1")
                        kv_slc_block_fp32 = pypto.cast(
                            kv_slc_block, pypto.DT_FP32
                        )
                        kr_slc_block_fp32 = pypto.cast(
                            kr_slc_block, pypto.DT_FP32
                        )
                        pypto.set_semantic_label("gather2")
                        kv_slc_block_fp16 = pypto.cast(
                            kv_slc_block_fp32, gather_res.dtype
                        )
                        kr_slc_block_fp16 = pypto.cast(
                            kr_slc_block_fp32, gather_res.dtype
                        )
                        ofs = (
                            b_idx * s1 * n2 * topk
                            + s1_idx * n2 * topk
                            + n2_idx * topk
                            + topk_idx
                        )
                        pypto.assemble(
                            kv_slc_block_fp16, [ofs, 0], gather_res
                        )
                        pypto.assemble(
                            kr_slc_block_fp16,
                            [ofs, d_n],
                            gather_res,
                        )


@dataclass
class BuildConfig:
    b: int = 32
    s1: int = 4
    n2: int = 128
    d_n: int = 512
    d_r: int = 64
    block_size: int = 128
    topk: int = 2048
    num_blocks: int = 32
    s2: int = 4096


def build_gather_args(cfg: BuildConfig = BuildConfig()):
    cache_dtype = pypto.DT_FP16
    index_dtype = pypto.DT_INT32
    cache_rows = cfg.num_blocks * cfg.block_size
    max_block_per_batch = cfg.s2 // cfg.block_size
    top_k_indices = pypto.tensor(
        [cfg.b, cfg.s1, cfg.n2, cfg.topk], index_dtype, "topKIndices"
    )
    k_nope_cache = pypto.tensor(
        [cache_rows, cfg.d_n], cache_dtype, "kNopeCache")
    k_rope_cache = pypto.tensor(
        [cache_rows, cfg.d_r], cache_dtype, "kRopeCache")
    block_table = pypto.tensor(
        [cfg.b, max_block_per_batch], index_dtype, "blockTable")
    act_seqs = pypto.tensor([cfg.b], index_dtype, "actSeqs")
    gather_res = pypto.tensor(
        [cfg.b * cfg.s1 * cfg.topk, cfg.d_n + cfg.d_r], cache_dtype, "gatherRes"
    )
    nsa_params = NSASimpleParamsObj(block_size=cfg.block_size, topk=cfg.topk)
    args = GatherInputs(
        top_k_indices=top_k_indices,
        k_nope_cache=k_nope_cache,
        k_rope_cache=k_rope_cache,
        block_table=block_table,
        act_seqs=act_seqs,
        gather_res=gather_res,
        nsa_params=nsa_params,
        b=pypto.symbolic_scalar(cfg.b),
        s1=pypto.symbolic_scalar(cfg.s1),
    )
    meta = {
        "b": cfg.b,
        "s1": cfg.s1,
        "n2": cfg.n2,
        "dims": {
            "topKIndices": [cfg.b, cfg.s1, cfg.n2, cfg.topk],
            "kNopeCache": [cache_rows, cfg.d_n],
            "kRopeCache": [cache_rows, cfg.d_r],
            "blockTable": [cfg.b, max_block_per_batch],
            "actSeqs": [cfg.b],
            "gatherRes": [cfg.b * cfg.s1 * cfg.topk, cfg.d_n + cfg.d_r],
        },
        "params": {
            "blockSize": cfg.block_size,
            "topk": cfg.topk,
            "numBlocks": cfg.num_blocks,
            "s2": cfg.s2,
            "dN": cfg.d_n,
            "dR": cfg.d_r,
        },
        "dtypes": {
            "cache": str(cache_dtype),
            "index": str(index_dtype),
        },
    }
    return args, meta


@duration_estimate(32)
def test_gather_with_builder():
    logging.basicConfig(level=logging.INFO)
    args, meta = build_gather_args()
    logging.info({"Sanity": meta})
    gather_after_prolog_compute(args)
    assert True
