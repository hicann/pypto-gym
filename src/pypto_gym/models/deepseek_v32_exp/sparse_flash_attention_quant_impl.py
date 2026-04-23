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
    See deepseekv32_sparse_flash_attention_quant.py for usage examples.
"""
import os
import math
from dataclasses import dataclass
import numpy as np
import pypto
from pypto.experimental import gather_in_l1, gather_in_ub


@dataclass
class SaTileShapeConfig:
    g_tile: int
    s_kv_tile: int
    c1_tile_shape: list
    v1_tile_shape: list
    c2_tile_shape: list
    v2_tile_shape: list


def sparse_flash_attention_quant_compute(query_nope, query_rope, key_nope_2d, key_rope_2d,
                                         k_nope_scales, topk_indices, block_table, kv_act_seqs,
                                         attention_out, nq, n_kv, softmax_scale, topk,
                                         block_size, max_blocknum_perbatch, tile_config):
    """Compute sparse flash attention with quantization support.

    Performs attention computation on top-k selected key-value pairs from cache.
    The function processes queries and keys in batches, computing attention scores
    and aggregating values. Supports both quantized (INT8) and non-quantized keys.

    Args:
        query_nope: Query tensor without RoPE, shape (t * n_q, kv_lora_rank), dtype BF16
        query_rope: Query tensor with RoPE, shape (t * n_q, rope_dim), dtype BF16
        key_nope_2d: Key tensor without RoPE, shape (block_num * block_size, kv_lora_rank),
                     dtype BF16 or INT8
        key_rope_2d: Key tensor with RoPE, shape (block_num * block_size, rope_dim), dtype BF16
        k_nope_scales: Dequantization scales for quantized keys, shape (block_num * block_size, 4),
                       dtype FP32. Only used when key_nope_2d is INT8.
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
    kn_dtype = key_nope_2d.dtype
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

    atten_out_2dim = pypto.tensor([batch_size_sym * s1_n2_gsym, dn], dtype, "attenOut2Dim")
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

                        kn = pypto.tensor([s2_tile, dn], dtype, "kn")
                        if kn_dtype == pypto.DT_INT8:
                            pypto.set_semantic_label("Sa_V0")
                            pypto.set_vec_tile_shapes(16, 1024)
                            k_nope_scale_view = pypto.view(k_nope_scales, [k_nope_scales.shape[0], 8],
                                [0, 0], valid_shape=[k_nope_scales.shape[0], 4])
                            kn_scale = gather_in_ub(k_nope_scale_view, cur_topk_indices, cur_block_table,
                                                    block_size, -2)
                            k_nope_2d_view = pypto.view(key_nope_2d, [key_nope_2d.shape[0], dn],
                                [0, 0], valid_shape=[key_nope_2d.shape[0], dn])
                            kn_quant = gather_in_ub(k_nope_2d_view, cur_topk_indices, cur_block_table, block_size, -2)
                            kn_quant_fp16 = pypto.cast(kn_quant, pypto.DT_FP16)
                            kn_quant_fp32 = pypto.cast(kn_quant_fp16, pypto.DT_FP32)
                            kn_quant_fp32 = pypto.concat([kn_quant_fp32, kn_quant_fp32], -1)
                            kn_quant_fp32_tmp = pypto.reshape(kn_quant_fp32, [s2_tile * 8, 128])
                            kn_scale_tmp = pypto.reshape(kn_scale, [s2_tile * 8, 1])
                            pypto.set_vec_tile_shapes(128, 128)
                            kn_fp32 = pypto.mul(kn_quant_fp32_tmp, kn_scale_tmp)
                            kn_fp32_reshape = pypto.reshape(kn_fp32, [s2_tile, dn * 2])
                            pypto.set_vec_tile_shapes(16, 512)
                            cur_kn_fp32 = pypto.view(kn_fp32_reshape, [cur_s2_tile, dn], [0, 0],
                                valid_shape=[(cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile), dn])
                            kn = pypto.cast(cur_kn_fp32, dtype)
                        else:
                            pypto.set_semantic_label("Sa_V0")
                            pypto.set_vec_tile_shapes(32, 512)
                            k_nope_2d_view = pypto.view(key_nope_2d, [key_nope_2d.shape[0], dn],
                                [0, 0], valid_shape=[key_nope_2d.shape[0], dn])
                            kn = gather_in_ub(k_nope_2d_view, cur_topk_indices, cur_block_table, block_size, -2)

                        # C1
                        pypto.set_semantic_label("Sa_C1")
                        pypto.set_vec_tile_shapes(32, 512)
                        pypto.set_cube_tile_shapes([c1_tile[0],
                            c1_tile[1]], [c1_tile[2], c1_tile[3]], [c1_tile[4], c1_tile[5]])

                        if kn_dtype == pypto.DT_INT8:
                            kr = gather_in_l1(key_rope_2d, cur_topk_indices, cur_block_table, block_size, dr,
                                            is_b_matrix=True, is_trans=True)
                        else:
                            key_rope_2d_view = pypto.view(key_rope_2d, [key_rope_2d.shape[0], dr],
                                                            [0, 0], valid_shape=[key_rope_2d.shape[0], dr])
                            kr = gather_in_ub(key_rope_2d_view, cur_topk_indices, cur_block_table, block_size, -2)

                        kj = pypto.tensor([cur_s2_tile, dn + dr], dtype, "kj")
                        pypto.assemble(kn, [0, 0], kj)
                        pypto.assemble(kr, [0, dn], kj)
                        kj_view = pypto.view(kj, [cur_s2_tile, dn + dr], [0, 0],
                                             valid_shape=[(cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile), dn + dr])

                        qn = pypto.view(query_nope, [cur_group_tile, dn], [cur_offset, 0],
                                        valid_shape=[cur_group_tile, dn])
                        qr = pypto.view(query_rope, [cur_group_tile, dr], [cur_offset, 0],
                                        valid_shape=[cur_group_tile, dr])
                        qi = pypto.tensor([cur_group_tile, dn + dr], dtype, "qi")
                        pypto.assemble(qn, [0, 0], qi)
                        pypto.assemble(qr, [0, dn], qi)

                        sij = pypto.matmul(qi, kj_view, pypto.DT_FP32, a_trans=False, b_trans=True)

                        pypto.set_semantic_label("Sa_V1")
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                        sij_scale = pypto.mul(sij, softmax_scale)
                        tilda_mij_reduce = pypto.amax(sij_scale, dim=-1, keepdim=True)
                        t_sub = pypto.sub(sij_scale, tilda_mij_reduce)
                        tilda_pij = pypto.exp(t_sub)
                        tilda_lij_reduce = pypto.sum(tilda_pij, dim=-1, keepdim=True)
                        t_softmax = pypto.div(tilda_pij, tilda_lij_reduce)
                        tilda_pij_f16 = pypto.cast(t_softmax, dtype)

                        pypto.set_semantic_label("Sa_C2")
                        pypto.set_cube_tile_shapes([c2_tile[0],
                            c2_tile[1]], [c2_tile[2], c2_tile[3]], [c2_tile[4], c2_tile[5]])
                        pypto.set_matrix_size([tilda_pij_f16.shape[0],
                            tilda_pij_f16.shape[1], kn.shape[1]])

                        q1 = pypto.tensor([cur_group_tile, dn], dtype)
                        vj = pypto.view(kn, [cur_s2_tile, dn], [0, 0],
                                        valid_shape=[(cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile), dn])
                        q1 = pypto.matmul(tilda_pij_f16, vj, dtype)

                        pypto.assemble(q1, [cur_offset, 0], atten_out_2dim)
                        attention_out[:] = pypto.reshape(atten_out_2dim,
                                                    [attention_out.shape[0], attention_out.shape[1],
                                                     attention_out.shape[2], attention_out.shape[3]], inplace=True)


def sparse_flash_attention_quant_compute_flash(query_nope, query_rope, key_nope_2d, key_rope_2d,
                                               k_nope_scales, topk_indices, block_table, kv_act_seqs,
                                               attention_out, nq, n_kv, softmax_scale, topk,
                                               block_size, max_blocknum_perbatch, tile_config):
    """Compute sparse flash attention with online softmax (flash attention variant).

    Implements flash attention algorithm with online softmax computation for better
    numerical stability and memory efficiency. Uses incremental updates of attention
    output, normalization factor, and maximum values across key-value blocks.

    Args:
        query_nope: Query tensor without RoPE, shape (t * n_q, kv_lora_rank), dtype BF16
        query_rope: Query tensor with RoPE, shape (t * n_q, rope_dim), dtype BF16
        key_nope_2d: Key tensor without RoPE, shape (block_num * block_size, kv_lora_rank),
                     dtype BF16 or INT8
        key_rope_2d: Key tensor with RoPE, shape (block_num * block_size, rope_dim), dtype BF16
        k_nope_scales: Dequantization scales for quantized keys, shape (block_num * block_size, 4),
                       dtype FP32. Only used when key_nope_2d is INT8.
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
        tile_config: SaTileShapeConfig object containing tiling parameters, including
                     v2_tile_shape for flash attention updates

    Note:
        Flash attention algorithm maintains running statistics:
        - oi_update: Running attention output
        - li_update: Running normalization factor (sum of exp values)
        - mi_update: Running maximum value

        These are incrementally updated across key-value blocks using the online softmax
        formula to maintain numerical stability.
    """
    dtype = query_nope.dtype
    kn_dtype = key_nope_2d.dtype
    dn = query_nope.shape[1]
    dr = query_rope.shape[1]
    group = nq // n_kv
    group_tile = tile_config.g_tile
    s2_tile = tile_config.s_kv_tile
    c1_tile = tile_config.c1_tile_shape
    v1_tile = tile_config.v1_tile_shape
    c2_tile = tile_config.c2_tile_shape
    v2_tile = tile_config.v2_tile_shape
    n_kv_sym = n_kv

    batch_size_sym = kv_act_seqs.shape[0]

    s1_n2_gsym = query_nope.shape[0] // batch_size_sym
    s1_sym = s1_n2_gsym // nq

    g_loop_sym = group // group_tile

    for batch_idx in pypto.loop(0, batch_size_sym, 1, name="FLASH_LOOP_L0_idx", idx_name="bIdx"):
        cur_act_seq = kv_act_seqs[batch_idx]
        for slc_idx in pypto.loop(0, s1_sym, 1, name="FLASH_LOOP_L1_s1_SA", idx_name="s1Idx"):
            cur_seq = (cur_act_seq - s1_sym + 1 + slc_idx).max(0).min(topk)
            cur_seq.as_variable()
            bn_per_batch = (cur_seq + s2_tile - 1) // s2_tile

            for n_kv_idx in pypto.loop(0, n_kv_sym, 1, name="FLASH_LOOP_L2_n_kv_SA", idx_name="n_kvIdx"):
                for group_idx in pypto.loop(0, g_loop_sym, 1, name="FLASH_LOOP_L3_g_SA", idx_name="gIdx"):
                    cur_group_tile = group_tile
                    oi_update = pypto.tensor([cur_group_tile, dn], pypto.DT_FP32, "oi_update")
                    li_update = pypto.tensor([1, cur_group_tile], pypto.DT_FP32, "li_update")
                    mi_update = pypto.tensor([1, cur_group_tile], pypto.DT_FP32, "mi_update")

                    cur_offset = batch_idx * s1_n2_gsym + slc_idx * nq + n_kv_idx * group + group_idx * cur_group_tile
                    oi_offset = [batch_idx, slc_idx, n_kv_idx * group + group_idx * cur_group_tile, 0]
                    for s2_idx, _ in pypto.loop_unroll(0, bn_per_batch, 1,
                        name="FLASH_LOOP_L4_s2_SA", idx_name="s2_idx", unroll_list={1}):
                        cur_s2_tile = s2_tile

                        pypto.set_semantic_label("Sa_V0")
                        cur_topk_indices = pypto.view(topk_indices, [1, cur_s2_tile],
                                                  [batch_idx * s1_sym + slc_idx, s2_idx * cur_s2_tile],
                                                  valid_shape=[1, (cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile)])
                        cur_block_table = pypto.view(block_table, [1, max_blocknum_perbatch], [batch_idx, 0])
                        k_nope_2d_view = pypto.view(key_nope_2d, [key_nope_2d.shape[0], dn],
                            [0, 0], valid_shape=[key_nope_2d.shape[0], dn])
                        k_nope_scale_view = pypto.view(k_nope_scales, [k_nope_scales.shape[0], 4],
                            [0, 0], valid_shape=[k_nope_scales.shape[0], 4])

                        kn = pypto.tensor([s2_tile, dn], dtype, "kn")

                        if kn_dtype == pypto.DT_INT8:
                            pypto.set_vec_tile_shapes(32, 512)
                            kn_scale = gather_in_ub(k_nope_scale_view, cur_topk_indices,
                                                    cur_block_table, block_size, -2)
                            kn_quant = gather_in_ub(k_nope_2d_view, cur_topk_indices, cur_block_table, block_size, -2)
                            kn_quant_fp16 = pypto.cast(kn_quant, pypto.DT_FP16)
                            kn_quant_fp32 = pypto.cast(kn_quant_fp16, pypto.DT_FP32)
                            kn_quant_fp32_tmp = pypto.reshape(kn_quant_fp32, [s2_tile * 4, 128])
                            kn_scale_tmp = pypto.reshape(kn_scale, [s2_tile * 4, 1])
                            pypto.set_vec_tile_shapes(128, 128)
                            kn_fp32 = pypto.mul(kn_quant_fp32_tmp, kn_scale_tmp)
                            kn_fp32_reshape = pypto.reshape(kn_fp32, [s2_tile, dn])
                            pypto.set_vec_tile_shapes(32, 512)
                            cur_kn_fp32 = pypto.view(kn_fp32_reshape, [cur_s2_tile, dn], [0, 0],
                                valid_shape=[(cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile), dn])
                            kn = pypto.cast(cur_kn_fp32, dtype)
                        else:
                            pypto.set_cube_tile_shapes([c1_tile[0], c1_tile[1]],
                                [c1_tile[2], c1_tile[3]], [c1_tile[4], c1_tile[5]])
                            kn = gather_in_l1(key_nope_2d,
                                cur_topk_indices, cur_block_table, block_size, dn, is_b_matrix=True, is_trans=True)
                        # C1
                        pypto.set_semantic_label("Sa_C1")
                        pypto.set_cube_tile_shapes([c1_tile[0],
                            c1_tile[1]], [c1_tile[2], c1_tile[3]], [c1_tile[4], c1_tile[5]])

                        kr = gather_in_l1(key_rope_2d, cur_topk_indices, cur_block_table, block_size, dr,
                                          is_b_matrix=True, is_trans=True)
                        kj = pypto.tensor([cur_s2_tile, dn + dr], dtype, "kj")
                        pypto.assemble(kn, [0, 0], kj)
                        pypto.assemble(kr, [0, dn], kj)
                        kj_view = pypto.view(kj, [cur_s2_tile, dn + dr], [0, 0],
                                             valid_shape=[(cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile), dn + dr])

                        qn = pypto.view(query_nope, [cur_group_tile, dn], [cur_offset, 0],
                                        valid_shape=[cur_group_tile, dn])
                        qr = pypto.view(query_rope, [cur_group_tile, dr], [cur_offset, 0],
                                        valid_shape=[cur_group_tile, dr])
                        qi = pypto.tensor([cur_group_tile, dn + dr], dtype, "qi")
                        pypto.assemble(qn, [0, 0], qi)
                        pypto.assemble(qr, [0, dn], qi)

                        sij = pypto.matmul(qi, kj_view, pypto.DT_FP32, a_trans=False, b_trans=True)

                        pypto.set_semantic_label("Sa_V1")
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                        sij_scale = pypto.mul(sij, softmax_scale)
                        tilda_mij_reduce = pypto.amax(sij_scale, dim=-1, keepdim=True)
                        tilda_mij = pypto.reshape(tilda_mij_reduce, [1, cur_group_tile])
                        t_sub = pypto.sub(sij_scale, tilda_mij_reduce)
                        tilda_pij = pypto.exp(t_sub)
                        tilda_pij_f16 = pypto.cast(tilda_pij, dtype)
                        tilda_lij_reduce = pypto.sum(tilda_pij, dim=-1, keepdim=True)
                        tilda_lij = pypto.reshape(tilda_lij_reduce, [1, cur_group_tile])

                        pypto.set_semantic_label("Sa_C2")
                        pypto.set_cube_tile_shapes([c2_tile[0],
                            c2_tile[1]], [c2_tile[2], c2_tile[3]], [c2_tile[4], c2_tile[5]])
                        pypto.set_matrix_size([tilda_pij_f16.shape[0],
                            tilda_pij_f16.shape[1], kn.shape[1]])

                        q1 = pypto.tensor([cur_group_tile, dn], dtype)
                        if kn_dtype == pypto.DT_INT8:
                            vj = pypto.view(kn, [cur_s2_tile, dn], [0, 0],
                                            valid_shape=[(cur_seq - s2_idx * cur_s2_tile).min(cur_s2_tile), dn])
                            q1 = pypto.matmul(tilda_pij_f16, vj, pypto.DT_FP32)
                        else:
                            vj = gather_in_l1(key_nope_2d, cur_topk_indices, cur_block_table, block_size,
                                dn, is_b_matrix=True, is_trans=False)
                            q1 = pypto.matmul(tilda_pij_f16, vj, pypto.DT_FP32)

                        if pypto.cond(pypto.is_loop_begin(s2_idx)):
                            oi_tmp = q1
                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            if pypto.cond(pypto.is_loop_end(s2_idx)):
                                pypto.set_semantic_label("Sa_V2")
                                oi_update[:] = oi_tmp / tilda_lij_reduce
                                pypto.set_vec_tile_shapes(1, 1, v2_tile[0], v2_tile[1])
                                oi_update_4_dim = pypto.cast(pypto.reshape(oi_update,
                                    [1, 1, cur_group_tile, dn]), dtype)
                                pypto.assemble(oi_update_4_dim, oi_offset, attention_out)
                            else:
                                oi_update[:] = oi_tmp
                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            li_update[:] = tilda_lij
                            mi_update[:] = tilda_mij
                        else:
                            pypto.set_semantic_label("Sa_UpdateVec2")
                            oi = oi_update
                            li = li_update
                            mi = mi_update
                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            mi_new = pypto.maximum(mi, tilda_mij)
                            t1 = pypto.sub(mi, mi_new)
                            t2 = pypto.exp(t1)
                            t3 = pypto.sub(tilda_mij, mi_new)
                            t4 = pypto.exp(t3)
                            t5 = pypto.mul(t4, tilda_lij)
                            t6 = pypto.mul(t2, li)
                            li_new = pypto.add(t6, t5)
                            q3 = pypto.mul(oi, pypto.reshape(t2, [cur_group_tile, 1]))
                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            q2 = pypto.mul(q1, pypto.reshape(t4, [cur_group_tile, 1]))
                            oi_tmp = pypto.add(q3, q2)
                            if pypto.cond(pypto.is_loop_end(s2_idx)):
                                oi_update[:] = pypto.div(oi_tmp,
                                    pypto.reshape(li_new, [cur_group_tile, 1]))
                                pypto.set_vec_tile_shapes(1, 1, v2_tile[0], v2_tile[1])
                                oi_update_4_dim = pypto.cast(pypto.reshape(oi_update,
                                    [1, 1, cur_group_tile, dn]), dtype)
                                pypto.assemble(oi_update_4_dim, oi_offset, attention_out)
                            else:
                                oi_update[:] = oi_tmp
                            li_update[:] = li_new
                            mi_update[:] = mi_new


@pypto.frontend.jit(
    pass_options={
        "pg_upper_bound": 5000000,
        "vec_nbuffer_setting": {-1: 2, 0: 8},
        "cube_l1_reuse_setting": {-1: 2},
    },
    runtime_options={
        "stitch_function_max_num": 128,
        "device_sched_mode": 3
    }
)
def sparse_flash_attention_quant_d(
    query_nope: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    query_rope: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    key_nope_2d: pypto.Tensor([pypto.STATIC, pypto.STATIC], ), # int8 or bf16
    key_rope_2d: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    k_nope_scales: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    topk_indices: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_INT32),
    block_table: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_INT32),
    kv_act_seqs: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    attention_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16),

    nq, n_kv, softmax_scale, topk, block_size, max_blocknum_perbatch, tile_config
):
    """JIT-compiled sparse flash attention for decode phase.

    Optimized version for decode phase with specific pass configurations.
    Uses flash attention algorithm with online softmax for numerical stability.

    Args:
        query_nope: Query tensor without RoPE, shape (t * n_q, kv_lora_rank), dtype BF16
        query_rope: Query tensor with RoPE, shape (t * n_q, rope_dim), dtype BF16
        key_nope_2d: Key tensor without RoPE, shape (block_num * block_size, kv_lora_rank),
                    dtype BF16 or INT8
        key_rope_2d: Key tensor with RoPE, shape (block_num * block_size, rope_dim), dtype BF16
        k_nope_scales: Dequantization scales for quantized keys, shape (block_num * block_size, 4),
                    dtype FP32
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

    sparse_flash_attention_quant_compute(query_nope, query_rope, key_nope_2d, key_rope_2d,
                                        k_nope_scales, topk_indices, block_table, kv_act_seqs,
                                        attention_out, nq, n_kv, softmax_scale, topk,
                                        block_size, max_blocknum_perbatch, tile_config)


@pypto.frontend.jit(
    pass_options={
        "pg_upper_bound": 5000000,
        "vec_nbuffer_setting": {-1: 4, 0: 16},
        "cube_l1_reuse_setting": {-1: 4},
    },
    runtime_options={
        "stitch_function_max_num": 128
    }
)
def sparse_flash_attention_quant_p(
    query_nope: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    query_rope: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    key_nope_2d: pypto.Tensor([pypto.STATIC, pypto.STATIC],), # int8 or bf16
    key_rope_2d: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    k_nope_scales: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    topk_indices: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_INT32),
    block_table: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_INT32),
    kv_act_seqs: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    attention_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16),

    nq, n_kv, softmax_scale, topk, block_size, max_blocknum_perbatch, tile_config
):
    """JIT-compiled sparse flash attention for prefill phase.

    Optimized version for prefill phase with specific pass configurations.
    Uses flash attention algorithm with online softmax for numerical stability.

    Args:
        query_nope: Query tensor without RoPE, shape (t * n_q, kv_lora_rank), dtype BF16
        query_rope: Query tensor with RoPE, shape (t * n_q, rope_dim), dtype BF16
        key_nope_2d: Key tensor without RoPE, shape (block_num * block_size, kv_lora_rank),
                    dtype BF16 or INT8
        key_rope_2d: Key tensor with RoPE, shape (block_num * block_size, rope_dim), dtype BF16
        k_nope_scales: Dequantization scales for quantized keys, shape (block_num * block_size, 4),
                    dtype FP32
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

    sparse_flash_attention_quant_compute(query_nope, query_rope, key_nope_2d, key_rope_2d,
                                          k_nope_scales, topk_indices, block_table, kv_act_seqs,
                                          attention_out, nq, n_kv, softmax_scale, topk,
                                          block_size, max_blocknum_perbatch, tile_config)
