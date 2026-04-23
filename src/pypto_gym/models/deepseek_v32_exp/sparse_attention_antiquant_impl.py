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
Sparse Flash Attention Quantization Module

This module implements sparse flash attention with quantization support for DeepSeek V32.
It performs attention computation on top-k selected key-value pairs from cache,
supporting both standard and flash attention algorithms.

Main Functions:
    - sparse_flash_attention_quant_compute: Standard sparse attention computation
    - sparse_flash_attention_quant_compute_flash: Flash attention variant with online softmax
    - sparse_flash_attention_quant_d: JIT-compiled decode version
    - sparse_flash_attention_quant_p: JIT-compiled prefill version

Example:
    See deepseekv32_sparse_attention_antiquant.py for usage examples.
"""
import os
import math
from dataclasses import dataclass
import numpy as np
import pypto
from pypto.experimental import gather_in_ub


@dataclass
class SaTileShapeConfig:
    g_tile: int
    s_kv_tile: int
    c1_tile_shape: list
    v1_tile_shape: list
    c2_tile_shape: list
    v2_tile_shape: list


def sparse_attention_antiquant_compute(query_nope, query_rope, nope_cache, topk_indices,
                                            block_table, kv_act_seqs, attention_out,
                                            nq, n_kv, softmax_scale, topk, block_size,
                                            max_blocknum_perbatch, tile_config):
    """Compute sparse flash attention with quantization support.

    Performs attention computation on top-k selected key-value pairs from cache.
    The function processes queries and keys in batches, computing attention scores
    and aggregating values. Supports both quantized (INT8) and non-quantized keys.

    Args:
        query_nope: Query tensor without RoPE, shape (t * n_q, kv_lora_rank), dtype BF16
        query_rope: Query tensor with RoPE, shape (t * n_q, rope_dim), dtype BF16
        nope_cache: Key tensor without RoPE, Key tensor with RoPE, Dequantization scales for quantized keys,
                    shape (block_num * block_size, kv_lora_rank + rope_dim*2 + 4*4),
                    dtype INT8
        topk_indices: Top-k indices for each query token, shape (t, n_kv * topk), dtype INT32
        block_table: Block mapping table for PagedAttention, shape (b, max_blocknum_perbatch),
                     dtype INT32
        kv_act_seqs: Actual sequence lengths for each batch, shape (b,), dtype INT32
        attention_out: Output attention tensor, shape (b, s, n_q, kv_lora_rank), dtype BF16
        nq: Number of query heads
        n_kv: Number of key-value heads
        softmax_scale: Scaling factor for attention scores, typically 1/sqrt(head_dim)
        topk: Number of top-k keys to attend to
        block_size: Size of each block in PagedAttention
        max_blocknum_perbatch: Maximum number of blocks per batch
        tile_config: SaTileShapeConfig object containing tiling parameters:
            - g_tile: Group tile size
            - s_kv_tile: Key-value sequence tile size
            - c1_tile_shape: Cube tile shape for first matmul
            - v1_tile_shape: Vector tile shape for softmax
            - c2_tile_shape: Cube tile shape for second matmul

    Note:
        The function uses nested loops to process batches, sequences, heads, and groups.
        For quantized keys, it performs dequantization before attention computation.
        The attention computation uses standard softmax normalization.
    """
    dtype = query_nope.dtype
    dn = query_nope.shape[1]
    dr = query_rope.shape[1]
    group = nq // n_kv
    group_tile = tile_config.g_tile
    s2_tile = tile_config.s_kv_tile
    c1_tile = tile_config.c1_tile_shape
    v1_tile = tile_config.v1_tile_shape
    c2_tile = tile_config.c2_tile_shape
    n_kv_sym = n_kv

    batch_size_sym = kv_act_seqs.shape[0]

    s1_n2_gsym = query_nope.shape[0] // batch_size_sym
    s1_sym = s1_n2_gsym // nq

    g_loop_sym = group // group_tile

    for batch_idx in pypto.loop(0, batch_size_sym, 1, name="LOOP_L0_idx", idx_name="bIdx"):
        cur_act_seq = kv_act_seqs[batch_idx]
        for slc_idx in pypto.loop(0, s1_sym, 1, name="LOOP_L1_s1_SA", idx_name="s1Idx"):
            cur_seq = (cur_act_seq - s1_sym + 1 + slc_idx).max(0).min(topk)
            cur_seq.as_variable()
            bn_per_batch = (cur_seq + s2_tile - 1) // s2_tile

            for n_kv_idx in pypto.loop(0, n_kv_sym, 1, name="LOOP_L2_n_kv_SA", idx_name="n_kvIdx"):
                for group_idx in pypto.loop(0, g_loop_sym, 1, name="LOOP_L3_g_SA", idx_name="gIdx"):
                    cur_group_tile = group_tile
                    cur_offset = batch_idx * s1_n2_gsym + slc_idx * nq + n_kv_idx * group + group_idx * cur_group_tile
                    for s2_idx, _ in pypto.loop_unroll(0, bn_per_batch, 1,
                        name="LOOP_L4_s2_SA", idx_name="s2_idx", unroll_list={1}):
                        cur_s2_tile = s2_tile

                        cur_topk_indices = pypto.view(topk_indices, [1, cur_s2_tile],
                                                  [batch_idx * s1_sym + slc_idx, s2_idx * cur_s2_tile],
                                                  valid_shape=[1, (cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile)])
                        cur_block_table = pypto.view(block_table, [1, max_blocknum_perbatch], [batch_idx, 0])

                        # V0
                        # nope_cache索引
                        pypto.set_semantic_label("Sa_V0")

                        # kv尾轴512 int8， kr尾轴64 bf16/fp16，kv scale尾轴4 fp32，共656; 然后最后一维要32对齐，变成672
                        pypto.set_vec_tile_shapes(16, 672)

                        # [512:640:656] kv_quant 512*int8, kr 64*bf16, kv_scale 4*fp32
                        cur_topk_indices = pypto.view(topk_indices, [1, cur_s2_tile],
                                                  [batch_idx * s1_sym + slc_idx, s2_idx * cur_s2_tile],
                                                  valid_shape=[1, (cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile)])
                        cur_block_table = pypto.view(block_table, [1, max_blocknum_perbatch], [batch_idx, 0])
                        nope_cache_view = pypto.view(
                            nope_cache,
                            [nope_cache.shape[0], 672],
                            [0, 0],
                            valid_shape=[nope_cache.shape[0], 656]
                        )

                        # ---- gather: GM --> UB  ----  UB非连续：shape [16, 672]， vaildshape：[16, 656]
                        slc_nope_cache = gather_in_ub(nope_cache_view, cur_topk_indices, cur_block_table,
                                                      block_size, -2)

                        pypto.set_vec_tile_shapes(16, 512)

                        # get kn
                        kn_quant = pypto.view(
                            input=slc_nope_cache,
                            shape=[cur_s2_tile, 512],
                            offsets=[0, 0],
                            valid_shape=[(cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile), 512]
                        )

                        # ---- cast: UB --> UB  ---- [16, 672]  --view->  [16, 0:512]  -cast-> [16, 512]
                        kn_quant_fp16 = pypto.cast(kn_quant, pypto.DT_FP16)

                        # ------------------ cast: UB --> UB  ---- [32, 512]  -cast-> [32, 512]
                        kn_quant_fp32 = pypto.cast(kn_quant_fp16, pypto.DT_FP32)

                        pypto.set_vec_tile_shapes(16, 1024)
                        kn_quant_fp32 = pypto.concat([kn_quant_fp32, kn_quant_fp32], -1)
                        kn_quant_fp32_reshape = pypto.reshape(kn_quant_fp32, [s2_tile * 4 * 2, 128])

                        kn_scale_vint8 = pypto.view(
                            input=slc_nope_cache,
                            shape=[cur_s2_tile, 16 * 2],
                            offsets=[0, dn + dr * 2],
                            valid_shape=[(cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile), 16]
                        )
                        kn_scale = pypto.view(input=kn_scale_vint8, dtype=pypto.DT_FP32)

                        kn_scale_t = pypto.add(kn_scale, 0)
                        kn_scale_reshape = pypto.reshape(kn_scale_t, [s2_tile * 4 * 2, 1])  # [32*4, 1]

                        pypto.set_vec_tile_shapes(16 * 4 * 2, 128)

                        # mul 附带 scale [32*4*2, 1] expand [32*4*2, 128]
                        kn_fp32 = pypto.mul(kn_quant_fp32_reshape, kn_scale_reshape)
                        kn_fp32_reshape = pypto.reshape(kn_fp32, [s2_tile, dn * 2])
                        pypto.set_vec_tile_shapes(16, 512)
                        cur_kn_fp32 = pypto.view(
                            input=kn_fp32_reshape,
                            shape=[cur_s2_tile, dn],
                            offsets=[0, 0],
                            valid_shape=[(cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile), dn]
                        )
                        kn = pypto.cast(cur_kn_fp32, dtype)

                        # get kr， UB --> GM
                        kr_vint8 = pypto.view(  # slc_nope_cache view
                            input=slc_nope_cache,
                            shape=[cur_s2_tile, dr * 2],
                            offsets=[0, dn],
                            valid_shape=[(cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile), dr * 2]
                        )
                        kr = pypto.view(input=kr_vint8, dtype=dtype)

                        # （1）kr和kn分开搬出，（2）kr和kb UB内assemble，再连续内存搬出
                        kj = pypto.Tensor([cur_s2_tile, dn + dr], dtype, "kj")
                        pypto.assemble(kn, [0, 0], kj)
                        pypto.assemble(pypto.clone(kr), [0, dn], kj)
                        kj_view = pypto.view(kj, [cur_s2_tile, dn + dr], [0, 0],
                                             valid_shape=[(cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile), dn + dr])

                        # C1
                        pypto.set_semantic_label("Sa_C1")
                        pypto.set_vec_tile_shapes(32, 512)
                        pypto.set_cube_tile_shapes([c1_tile[0],
                            c1_tile[1]], [c1_tile[2], c1_tile[3]], [c1_tile[4], c1_tile[5]])

                        qn = pypto.view(query_nope, [cur_group_tile, dn], [cur_offset, 0],
                                        valid_shape=[cur_group_tile, dn])
                        qr = pypto.view(query_rope, [cur_group_tile, dr], [cur_offset, 0],
                                        valid_shape=[cur_group_tile, dr])
                        qi = pypto.Tensor([cur_group_tile, dn + dr], dtype, "qi")
                        pypto.assemble(qn, [0, 0], qi)
                        pypto.assemble(qr, [0, dn], qi)

                        sij = pypto.matmul(qi, kj_view, pypto.DT_FP32, a_trans=False, b_trans=True)

                        # V1 softmax
                        pypto.set_semantic_label("Sa_V1")
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                        sij_scale = pypto.mul(sij, softmax_scale)
                        tilda_mij_reduce = pypto.amax(sij_scale, dim=-1, keepdim=True)
                        t_sub = pypto.sub(sij_scale, tilda_mij_reduce)
                        tilda_pij = pypto.exp(t_sub)
                        tilda_lij_reduce = pypto.sum(tilda_pij, dim=-1, keepdim=True)
                        t_softmax = pypto.div(tilda_pij, tilda_lij_reduce)
                        tilda_pij_f16 = pypto.cast(t_softmax, dtype)

                        # C2
                        pypto.set_semantic_label("Sa_C2")
                        pypto.set_cube_tile_shapes([c2_tile[0],
                            c2_tile[1]], [c2_tile[2], c2_tile[3]], [c2_tile[4], c2_tile[5]])
                        pypto.set_matrix_size([tilda_pij_f16.shape[0],
                            tilda_pij_f16.shape[1], kn.shape[1]])
                        vj = pypto.view(kn, [cur_s2_tile, dn], [0, 0],
                                        valid_shape=[(cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile), dn])
                        q1 = pypto.matmul(tilda_pij_f16, vj, dtype)

                        pypto.assemble(q1, [cur_offset, 0], attention_out)


@pypto.frontend.jit(
    host_options={
        "compile_monitor_enable": True,
        "compile_timeout": 10,
        "compile_timeout_stage": 5,
        "compile_monitor_print_interval": 2},
    pass_options={
        "pg_upper_bound": 5000000,
        "vec_nbuffer_setting": {-1: 2, 0: 4},
        "cube_l1_reuse_setting": {-1: 2},
    },
    runtime_options={
        "stitch_function_max_num": 128,
        "device_sched_mode": 3
    }
)
def sparse_attention_antiquant_d(
    query_nope: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    query_rope: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    nope_cache: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    topk_indices: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_INT32),
    block_table: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_INT32),
    kv_act_seqs: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    attention_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),

    nq, n_kv, softmax_scale, topk, block_size, max_blocknum_perbatch, tile_config
):
    """JIT-compiled sparse flash attention for decode phase.

    Optimized version for decode phase with specific pass configurations.
    Uses flash attention algorithm with online softmax for numerical stability.

    Args:
        query_nope: Query tensor without RoPE, shape (t * n_q, kv_lora_rank), dtype BF16
        query_rope: Query tensor with RoPE, shape (t * n_q, rope_dim), dtype BF16
        nope_cache: Key tensor without RoPE, Key tensor with RoPE, Dequantization scales for quantized keys,
                    shape (block_num * block_size, kv_lora_rank + rope_dim*2 + 4*4),
                    dtype INT8
        topk_indices: Top-k indices for each query token, shape (t, n_kv * topk), dtype INT32
        block_table: Block mapping table for PagedAttention, shape (b, max_blocknum_perbatch),
                    dtype INT32
        kv_act_seqs: Actual sequence lengths for each batch, shape (b,), dtype INT32
        attention_out: Output attention tensor, shape (b, s, n_q, kv_lora_rank), dtype BF16
        nq: Number of query heads
        n_kv: Number of key-value heads
        softmax_scale: Scaling factor for attention scores
        topk: Number of top-k keys to attend to
        block_size: Size of each block in PagedAttention
        max_blocknum_perbatch: Maximum number of blocks per batch
        tile_config: SaTileShapeConfig object containing tiling parameters

    Note:
        Configured for decode phase with optimized memory and parallelism settings.
        Uses flash attention algorithm for better numerical stability.
    """
    pypto.experimental.set_operation_options(combine_axis=True)

    sparse_attention_antiquant_compute(query_nope, query_rope, nope_cache, topk_indices,
                                            block_table, kv_act_seqs, attention_out,
                                            nq, n_kv, softmax_scale, topk, block_size,
                                            max_blocknum_perbatch, tile_config)


@pypto.frontend.jit(
    host_options={
        "compile_monitor_enable": True,
        "compile_timeout": 10,
        "compile_timeout_stage": 5,
        "compile_monitor_print_interval": 2},
    pass_options={
        "pg_upper_bound": 5000000,
        "vec_nbuffer_setting": {-1: 4, 0: 4},
        "cube_l1_reuse_setting": {-1: 4},
    },
    runtime_options={
        "stitch_function_max_num": 128
    }
)
def sparse_attention_antiquant_p(
    query_nope: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    query_rope: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    nope_cache: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    topk_indices: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_INT32),
    block_table: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_INT32),
    kv_act_seqs: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    attention_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),

    nq, n_kv, softmax_scale, topk, block_size, max_blocknum_perbatch, tile_config
):
    """JIT-compiled sparse flash attention for prefill phase.

    Optimized version for prefill phase with specific pass configurations.
    Uses flash attention algorithm with online softmax for numerical stability.

    Args:
        query_nope: Query tensor without RoPE, shape (t * n_q, kv_lora_rank), dtype BF16
        query_rope: Query tensor with RoPE, shape (t * n_q, rope_dim), dtype BF16
        nope_cache: Key tensor without RoPE, Key tensor with RoPE, Dequantization scales for quantized keys,
                    shape (block_num * block_size, kv_lora_rank + rope_dim*2 + 4*4),
                    dtype INT8
        topk_indices: Top-k indices for each query token, shape (t, n_kv * topk), dtype INT32
        block_table: Block mapping table for PagedAttention, shape (b, max_blocknum_perbatch),
                    dtype INT32
        kv_act_seqs: Actual sequence lengths for each batch, shape (b,), dtype INT32
        attention_out: Output attention tensor, shape (b, s, n_q, kv_lora_rank), dtype BF16
        nq: Number of query heads
        n_kv: Number of key-value heads
        softmax_scale: Scaling factor for attention scores
        topk: Number of top-k keys to attend to
        block_size: Size of each block in PagedAttention
        max_blocknum_perbatch: Maximum number of blocks per batch
        tile_config: SaTileShapeConfig object containing tiling parameters

    Note:
        Configured for prefill phase with optimized memory and parallelism settings.
        Uses flash attention algorithm for better numerical stability.
    """
    pypto.experimental.set_operation_options(combine_axis=True)

    sparse_attention_antiquant_compute(query_nope, query_rope, nope_cache, topk_indices,
                                            block_table, kv_act_seqs, attention_out,
                                            nq, n_kv, softmax_scale, topk, block_size,
                                            max_blocknum_perbatch, tile_config)
