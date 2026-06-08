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

from dataclasses import dataclass, replace

import torch
from torch._dynamo import allow_in_graph

import pypto


@dataclass
class IFAKernelCfg:
    """
    Parameters for the IFA kernel computation.

    Attributes:
        n1: Number of query heads
        d: Head dimension
        block_num: Total number of KV blocks
        n2: Number of key/value heads
        block_size: Size of each block
        b: Batch size
        s1: Query sequence length
        group: Number of head groups (n1 // n2)
        softmax_scale: Softmax scaling factor
    """
    n1: int
    d: int
    block_num: int
    n2: int
    block_size: int
    b: int
    s1: int
    group: int
    softmax_scale: float


@dataclass
class AttentionTileConfig:
    """
    Configuration for tile sizes used in attention computation.

    Tiling is used to break large computations into smaller, cache-friendly chunks.

    Attributes:
        g_tile: Tile size for group dimension
        s2_tile: Tile size for kv sequence dimension
        c1_tile: Tile configuration for first matrix multiplication (Q x K^T)
        v1_tile: Tile configuration for vector operations in first MM
        c2_tile: Tile configuration for second matrix multiplication (Softmax x V)
        v2_tile: Tile configuration for vector operations in second MM
    """
    g_tile: int
    s2_tile: int
    c1_tile: list
    v1_tile: list
    c2_tile: list
    v2_tile: list


@dataclass
class LoopTensor:
    """
    Tensors used during loop iterations.

    These are the main data structures accessed during attention computation.

    Attributes:
        q_2d: Query tensor reshaped to 2D (bs*n1, d)
        k_2d: Key tensor reshaped to 2D (block_num*block_size*n2, d)
        v_2d: Value tensor reshaped to 2D (block_num*block_size*n2, d)
        block_table: Mapping from logical block indices to physical block indices
        kv_act_seqs: Key/Value actual sequence lengths for each batch element
        atten_out: Output tensor for attention results
    """
    q_2d: pypto.Tensor = None
    k_2d: pypto.Tensor = None
    v_2d: pypto.Tensor = None
    block_table: pypto.Tensor = None
    kv_act_seqs: pypto.Tensor = None
    atten_out: pypto.Tensor = None
    key_antiquant_scale: pypto.Tensor = None
    value_antiquant_scale: pypto.Tensor = None


@dataclass
class LoopIndex:
    """
    Current indices for nested loop iterations.

    Attributes:
        b_idx: Batch index
        s1_idx: Query sequence index
        n2_idx: Key/Value head index
        group_idx: Group index within heads
        s2_idx: Key/Value sequence tile index
    """
    b_idx: int = 0
    s1_idx: int = 0
    n2_idx: int = 0
    group_idx: int = 0
    s2_idx: int = 0


@dataclass
class TempUpdateTensor:
    """
    Temporary tensors for online softmax computation.

    These tensors accumulate results across sequence tiles using the
    online softmax algorithm (Welford's algorithm variant).

    Attributes:
        out_update: Accumulated output (weighted sum of values)
        sum_update: Accumulated softmax denominator
        max_update: Accumulated softmax maximum value
    """
    out_update: pypto.Tensor = None
    sum_update: pypto.Tensor = None
    max_update: pypto.Tensor = None


@dataclass
class LoopSize:
    """
    Loop iteration counts.

    Attributes:
        group_loop: Number of groups to iterate (group_num // g_tile ——> (n1 // n2) // g_tile)
        s2_loop: Number of sequence tiles to iterate
    """
    group_loop: int = 0
    s2_loop: int = 0


@dataclass
class LoopOfs:
    """
    Offset parameters for loop iterations.

    Used to track current positions in the output tensor during nested loops.

    Attributes:
        bs_ofs: Batch-sequence offset (b_idx * s1 + s1_idx)
        n1g_ofs: Head-group offset (n2_idx * group + group_idx * g_tile)
        out_ofs: Output tensor offset [bs_ofs, n1g_ofs, 0]
    """
    bs_ofs: int = 0
    n1g_ofs: int = 0
    out_ofs: int = 0


@dataclass
class ContextParams:
    """
    Container for all context parameters passed between functions.

    This dataclass groups all the parameters needed for attention computation
    to avoid passing many individual arguments.

    Attributes:
        kernel_cfg: Kernel computation parameters
        tile_cfg: Tile configuration
        loop_tensors: Tensors used in loops
        loop_index: Current loop indices
        loop_size: Loop iteration counts
        loop_ofs: Loop offsets
        temp_update_tensors: Temporary update tensors
    """
    kernel_cfg: IFAKernelCfg = None
    tile_cfg: AttentionTileConfig = None
    loop_tensors: LoopTensor = None
    loop_index: LoopIndex = None
    loop_size: LoopSize = None
    loop_ofs: LoopOfs = None
    temp_update_tensors: TempUpdateTensor = None


def init_kernel_cfg(query, key, block_table):
    """
    Initialize kernel parameters from input tensors.

    This function extracts and computes all the parameters needed
    for the IFA kernel computation.

    Args:
        query: Query tensor
        key: Key cache tensor
        block_table: Block table

    Returns:
        IFAKernelCfg: Initialized kernel config
    """
    b, n1, s1, d = query.shape
    block_num, n2, block_size, _ = key.shape
    block_table_shape = block_table.shape
    b = block_table_shape[0]
    group = n1 // n2
    softmax_scale = d ** -0.5
    kernel_cfg = IFAKernelCfg(
        n1=n1, d=d, block_num=block_num, n2=n2, block_size=block_size,
        b=b, s1=s1, group=group, softmax_scale=softmax_scale
    )
    return kernel_cfg


def reshape_qkv_to_2d(query, key, value, kernel_cfg):
    b = kernel_cfg.b
    s1 = kernel_cfg.s1
    n1 = kernel_cfg.n1
    d = kernel_cfg.d
    block_num = kernel_cfg.block_num
    block_size = kernel_cfg.block_size
    n2 = kernel_cfg.n2

    q_2d_shape = (b * n1 * s1, d)
    kv_2d_shape = (block_num * block_size * n2, d)

    q_2d = pypto.reshape(query, q_2d_shape, inplace=True)
    k_2d = pypto.reshape(key, kv_2d_shape, inplace=True)
    v_2d = pypto.reshape(value, kv_2d_shape, inplace=True)
    return q_2d, k_2d, v_2d


def get_ifa_tile_cfg(group):
    """
    Get tile configuration for IFA computation.

    Args:
        group: Number of query heads per KV head (n1 // n2)

    Returns:
        AttentionTileConfig: Tile configuration with optimal sizes
    """
    m_tile = 128
    k_tile = 128
    n_tile = 128
    s2_tile = 2048

    tile_cfg = AttentionTileConfig(
        g_tile=group,
        s2_tile=s2_tile,
        c1_tile=[[m_tile, m_tile], [k_tile, k_tile], [n_tile, n_tile]],
        v1_tile=[m_tile, s2_tile],
        c2_tile=[[m_tile, m_tile], [k_tile, k_tile], [n_tile, n_tile]],
        v2_tile=[m_tile, m_tile]
    )
    return tile_cfg


@pypto.frontend.jit(
    runtime_options={
        "stitch_function_max_num": 256,
        "device_sched_mode": 1
    },
    pass_options={
        "cube_l1_reuse_setting": {0: 16},
        "vec_nbuffer_setting": {0: 16},
    }
)
def incre_flash_attention_gqa_antiquant_kernel(
    query: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    key: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP8E4M3),
    value: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP8E4M3),
    key_antiquant_scale: pypto.Tensor([...], pypto.DT_BF16),
    value_antiquant_scale: pypto.Tensor([...], pypto.DT_BF16),
    kv_actual_seqs: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    block_table: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_INT32),
    atten_out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16)
):
    pypto.experimental.set_operation_options(combine_axis=True)

    # Step 1: Initialize kernel config
    dtype = query.dtype
    kernel_cfg = init_kernel_cfg(query, key, block_table)

    # Step 2: Get tile configuration
    tile_cfg = get_ifa_tile_cfg(kernel_cfg.group)

    # Step 3: Reshape Q, K, V to 2D
    q_2d, k_2d, v_2d = reshape_qkv_to_2d(query, key, value, kernel_cfg)

    loop_tensors = LoopTensor(
    q_2d,
    k_2d,
    v_2d,
    block_table,
    kv_actual_seqs,
    atten_out,
    key_antiquant_scale,
     value_antiquant_scale)

    # Calculate number of groups to iterate
    group_loop = kernel_cfg.group // tile_cfg.g_tile
    loop_size = LoopSize(group_loop=group_loop)

    # Create context parameters
    ctx_params = ContextParams(
        kernel_cfg=kernel_cfg, tile_cfg=tile_cfg, loop_tensors=loop_tensors,
        loop_size=loop_size
    )

    # Step 4: Implement kernel logic with nested loops
    # Loop over batch dimension
    for b_idx in pypto.loop(kernel_cfg.b, name="LOOP_b", idx_name="b_idx"):
        loop_index = LoopIndex(b_idx=b_idx)
        ctx_params = replace(ctx_params, loop_index=loop_index)
        compute_loop_b(dtype, ctx_params)


def compute_loop_b(dtype, ctx_params):
    """
    Compute attention loop over batch dimension.

    Args:
        dtype: Data type for computation
        ctx_params: Context parameters
    """
    # Get needed kernel params
    s1 = ctx_params.kernel_cfg.s1
    d = ctx_params.kernel_cfg.d
    n1 = ctx_params.kernel_cfg.n1

    # Get needed tile cfg
    s2_tile = ctx_params.tile_cfg.s2_tile

    # Get needed loop tensors
    kv_act_seqs = ctx_params.loop_tensors.kv_act_seqs

    # Get needed loop index
    b_idx = ctx_params.loop_index.b_idx

    loop_size = ctx_params.loop_size
    loop_index = ctx_params.loop_index

    # Loop over query sequence positions
    for s1_idx in pypto.loop(s1, name="LOOP_s1", idx_name="s1_idx"):
        # Calculate effective sequence length
        cur_seq_len = kv_act_seqs[b_idx] - (s1 - 1 - s1_idx)

        s2_loop = pypto.ceildiv(cur_seq_len, s2_tile)
        loop_size = replace(loop_size, s2_loop=s2_loop)
        bs_ofs = b_idx * s1 + s1_idx
        loop_ofs = LoopOfs(bs_ofs=bs_ofs)
        loop_index = replace(loop_index, s1_idx=s1_idx)
        ctx_params = replace(ctx_params, loop_size=loop_size, loop_ofs=loop_ofs, loop_index=loop_index)

        compute_loop_s1(ctx_params, cur_seq_len, dtype)


def compute_loop_s1(ctx_params, cur_seq_len, dtype):
    """
    Compute attention loop over query sequence positions.

    Args:
        ctx_params: Context parameters
        cur_seq_len: Current sequence length
        dtype: Data type for computation
    """
    n2 = ctx_params.kernel_cfg.n2
    loop_index = ctx_params.loop_index

    for n2_idx in pypto.loop(n2, name="LOOP_n2", idx_name="n2_idx"):
        loop_index = replace(loop_index, n2_idx=n2_idx)
        ctx_params = replace(ctx_params, loop_index=loop_index)
        compute_loop_n2(ctx_params, cur_seq_len, dtype)


def compute_loop_n2(ctx_params, cur_seq_len, dtype):
    """
    Compute attention loop over key/value heads.

    Args:
        ctx_params: Context parameters
        cur_seq_len: Current sequence length
        dtype: Data type for computation
    """
    loop_index = ctx_params.loop_index
    group_loop = ctx_params.loop_size.group_loop

    for group_idx in pypto.loop(group_loop, name="LOOP_group_idx", idx_name="group_idx"):
        loop_index = replace(loop_index, group_idx=group_idx)
        ctx_params = replace(ctx_params, loop_index=loop_index)
        compute_loop_group(ctx_params, cur_seq_len, dtype)


def compute_loop_group(ctx_params, cur_seq_len, dtype):
    """
    Compute attention loop over groups.

    Args:
        ctx_params: Context parameters
        cur_seq_len: Current sequence length
        dtype: Data type for computation
    """
    # Get needed tile cfg
    g_tile = ctx_params.tile_cfg.g_tile
    v1_tile = ctx_params.tile_cfg.v1_tile

    # Get needed kernel params
    group = ctx_params.kernel_cfg.group
    d = ctx_params.kernel_cfg.d
    n1 = ctx_params.kernel_cfg.n1
    s1 = ctx_params.kernel_cfg.s1

    # Get needed loop index params
    loop_index = ctx_params.loop_index
    n2_idx = loop_index.n2_idx
    group_idx = loop_index.group_idx

    # Get needed loop offset params
    loop_ofs = ctx_params.loop_ofs
    bs_ofs = loop_ofs.bs_ofs

    # Get needed loop params
    s2_loop = ctx_params.loop_size.s2_loop

    # Calculate offset for current group
    n1g_ofs = n2_idx * group + group_idx * g_tile
    out_ofs = [bs_ofs, n1g_ofs, 0]
    loop_ofs = replace(loop_ofs, n1g_ofs=n1g_ofs, out_ofs=out_ofs)

    # Initialize temporary tensors for online softmax
    out_update = pypto.tensor([g_tile, d], pypto.DT_FP32, "out_update")
    sum_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "sum_update")
    max_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "max_update")
    temp_update_tensors = TempUpdateTensor(out_update, sum_update, max_update)

    b_idx = ctx_params.loop_index.b_idx
    s1_idx = ctx_params.loop_index.s1_idx
    q_2d = ctx_params.loop_tensors.q_2d

    pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
    qi = pypto.tensor([g_tile, d], dtype, "qi")
    for g_i in range(g_tile):
        qi_row_ofs = b_idx * n1 * s1 + (n1g_ofs + g_i) * s1 + s1_idx
        qi_row = pypto.view(q_2d, [1, d], [qi_row_ofs, 0])
        pypto.assemble(qi_row, [g_i, 0], qi)

    # Loop over sequence tiles
    for s2_idx in pypto.loop(s2_loop, name="LOOP_s2", idx_name="s2_idx", unroll_list=[8, 1]):
        loop_index = replace(loop_index, s2_idx=s2_idx)
        ctx_params = replace(ctx_params, loop_index=loop_index,
                            temp_update_tensors=temp_update_tensors, loop_ofs=loop_ofs)
        compute_loop_s2(ctx_params, cur_seq_len, dtype, qi)


def compute_loop_s2(ctx_params, cur_seq_len, dtype, qi):
    """
    Compute attention loop over sequence tiles.

    Args:
        ctx_params: Context parameters
        cur_seq_len: Current sequence length
        dtype: Data type for computation
    """
    # Get needed tile cfg
    tile_cfg = ctx_params.tile_cfg
    s2_tile = tile_cfg.s2_tile
    v1_tile = tile_cfg.v1_tile
    g_tile = tile_cfg.g_tile

    # Get needed kernel params
    block_size = ctx_params.kernel_cfg.block_size

    # Get needed loop index params
    s2_idx = ctx_params.loop_index.s2_idx

    block_num = s2_tile // block_size
    idx = s2_idx * block_num

    # Calculate actual sequence length in this tile
    actual_s2_tile = (cur_seq_len - s2_idx * s2_tile).min(s2_tile)

    # Get query for current head group
    pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])

    kj_assemble, vj_assemble = assemble_kvj(idx, actual_s2_tile, ctx_params)

    # Compute attention for this tile
    if pypto.cond(pypto.is_loop_begin(s2_idx)):
        compute_first_tile(qi, kj_assemble, vj_assemble, dtype, ctx_params, actual_s2_tile)
    else:
        compute_other_tile(qi, kj_assemble, vj_assemble, dtype, ctx_params, actual_s2_tile)
    # Finalize output on last tile
    if pypto.cond(pypto.is_loop_end(s2_idx)):
        finalize_output(dtype, ctx_params)


def assemble_kvj(idx, actual_s2_tile, ctx_params):
    """
    Assemble K tensor for current tile from paged blocks.

    Args:
        idx: Starting block index for this tile
        ctx_params: Context parameters containing tensors and config

    Returns:
        pypto.Tensor: Assembled K tensor of shape [s2_tile, d]
    """
    # Get needed tensors
    k_2d = ctx_params.loop_tensors.k_2d
    v_2d = ctx_params.loop_tensors.v_2d
    block_table = ctx_params.loop_tensors.block_table

    # Get needed tile cfg
    s2_tile = ctx_params.tile_cfg.s2_tile

    # Get needed kernel params
    block_size = ctx_params.kernel_cfg.block_size
    n2 = ctx_params.kernel_cfg.n2
    d = ctx_params.kernel_cfg.d

    # Get needed loop index
    b_idx = ctx_params.loop_index.b_idx
    n2_idx = ctx_params.loop_index.n2_idx

    block_num = s2_tile // block_size

    # Create assembled tensor
    kj_assemble = pypto.tensor([s2_tile, d], k_2d.dtype, "kj_assemble")
    vj_assemble = pypto.tensor([s2_tile, d], v_2d.dtype, "vj_assemble")

    # Copy blocks from 2D K tensor according to block table
    for i in range(block_num):
        block_idx = block_table[b_idx, idx + i]
        block_idx_valid = block_idx.max(0)
        kj_view = pypto.view(k_2d, [block_size, d], [(block_idx_valid * n2 + n2_idx) * block_size, 0])
        vj_view = pypto.view(v_2d, [block_size, d], [(block_idx_valid * n2 + n2_idx) * block_size, 0])
        pypto.assemble(kj_view, [i * block_size, 0], kj_assemble)
        pypto.assemble(vj_view, [i * block_size, 0], vj_assemble)

    # Set valid shape (may be smaller than allocated size)
    kj_assemble = pypto.view(kj_assemble, [s2_tile, d], [0, 0], valid_shape=[s2_tile, d])

    # Set valid shape to actual sequence length
    vj_assemble = pypto.view(vj_assemble, [s2_tile, d], [0, 0], valid_shape=[actual_s2_tile, d])
    return kj_assemble, vj_assemble


def compute_first_tile(qi, kj_assemble, vj_assemble, dtype, ctx_params, actual_s2_tile):
    """
    Compute attention for the first tile, computes the initial max, sum, and output values.

    Args:
        sij: QK^T scores for current tile
        vj_assemble: V tensor for current tile
        dtype: Data type for computation
        ctx_params: Context parameters
    """
    tile_cfg = ctx_params.tile_cfg
    c1_tile = tile_cfg.c1_tile
    g_tile = tile_cfg.g_tile
    s2_tile = tile_cfg.s2_tile
    c2_tile = tile_cfg.c2_tile
    v2_tile = tile_cfg.v2_tile
    v1_tile = tile_cfg.v1_tile

    softmax_scale = ctx_params.kernel_cfg.softmax_scale
    out_update = ctx_params.temp_update_tensors.out_update
    sum_update = ctx_params.temp_update_tensors.sum_update
    max_update = ctx_params.temp_update_tensors.max_update

    pypto.set_vec_tile_shapes(128, 128)
    key_antiquant_scale = ctx_params.loop_tensors.key_antiquant_scale[ctx_params.loop_index.n2_idx]
    kj_fp32 = pypto.cast(kj_assemble, pypto.DT_FP32, pypto.CastMode.CAST_NONE)
    kj_antiquant_scale_fp32 = pypto.cast(key_antiquant_scale, pypto.DT_FP32, pypto.CastMode.CAST_NONE)
    out_data_fp32 = pypto.mul(kj_fp32, kj_antiquant_scale_fp32)
    kj_assemble_antiquanted = pypto.cast(out_data_fp32, pypto.DT_BF16, pypto.CastMode.CAST_NONE)

    # Compute Q x K^T
    pypto.set_cube_tile_shapes(c1_tile[0], c1_tile[1], c1_tile[2])
    sij = pypto.matmul(qi, kj_assemble_antiquanted, pypto.DT_FP32, a_trans=False, b_trans=True)

    # Set valid shape to actual sequence length
    pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
    sij = pypto.view(sij, [g_tile, s2_tile], [0, 0], valid_shape=[g_tile, actual_s2_tile])

    # Scale scores by softmax scale factor
    sij_scale = pypto.mul(sij, softmax_scale)

    # Compute maximum score for this tile
    tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)

    # Compute exp(scores - max) for numerical stability
    tsub = pypto.sub(sij_scale, tilda_mij)
    tilda_pij = pypto.exp(tsub)
    tilda_pij_fp16 = pypto.cast(tilda_pij, dtype)

    # Initialize sum and max for online softmax
    sum_update[:] = pypto.sum(tilda_pij, dim=-1, keepdim=True)
    max_update[:] = tilda_mij

    pypto.set_vec_tile_shapes(128, 128)
    value_antiquant_scale = ctx_params.loop_tensors.value_antiquant_scale[ctx_params.loop_index.n2_idx]
    vj_fp32 = pypto.cast(vj_assemble, pypto.DT_FP32, pypto.CastMode.CAST_NONE)
    vj_antiquant_scale_fp32 = pypto.cast(value_antiquant_scale, pypto.DT_FP32, pypto.CastMode.CAST_NONE)
    vj_out_data_fp32 = pypto.mul(vj_fp32, vj_antiquant_scale_fp32)
    vj_assemble_antiquanted = pypto.cast(vj_out_data_fp32, pypto.DT_BF16, pypto.CastMode.CAST_NONE)

    # Compute weighted sum of values: exp(QK^T) x V
    pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
    oi_tmp = pypto.matmul(tilda_pij_fp16, vj_assemble_antiquanted, pypto.DT_FP32)

    # Store initial output
    pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
    out_update[:] = oi_tmp


def compute_other_tile(qi, kj_assemble, vj_assemble, dtype, ctx_params, actual_s2_tile):
    """
    Compute attention for subsequent sequence tiles.

    Args:
        sij: QK^T scores for current tile
        vj_assemble: V tensor for current tile
        dtype: Data type for computation
        ctx_params: Context parameters
    """
    softmax_scale = ctx_params.kernel_cfg.softmax_scale
    out_update = ctx_params.temp_update_tensors.out_update
    sum_update = ctx_params.temp_update_tensors.sum_update
    max_update = ctx_params.temp_update_tensors.max_update

    tile_cfg = ctx_params.tile_cfg
    c1_tile = tile_cfg.c1_tile
    g_tile = tile_cfg.g_tile
    s2_tile = tile_cfg.s2_tile
    c2_tile = tile_cfg.c2_tile
    v2_tile = tile_cfg.v2_tile

    pypto.set_vec_tile_shapes(128, 128)
    key_antiquant_scale = ctx_params.loop_tensors.key_antiquant_scale[ctx_params.loop_index.n2_idx]
    kj_fp32 = pypto.cast(kj_assemble, pypto.DT_FP32, pypto.CastMode.CAST_NONE)
    kj_antiquant_scale_fp32 = pypto.cast(key_antiquant_scale, pypto.DT_FP32, pypto.CastMode.CAST_NONE)
    out_data_fp32 = pypto.mul(kj_fp32, kj_antiquant_scale_fp32)
    kj_assemble_antiquanted = pypto.cast(out_data_fp32, pypto.DT_BF16, pypto.CastMode.CAST_NONE)

    pypto.set_cube_tile_shapes(c1_tile[0], c1_tile[1], c1_tile[2])
    sij = pypto.matmul(qi, kj_assemble_antiquanted, pypto.DT_FP32, a_trans=False, b_trans=True)

    # Set valid shape to actual sequence length
    sij = pypto.view(sij, [g_tile, s2_tile], [0, 0], valid_shape=[g_tile, actual_s2_tile])

    sij_scale = pypto.mul(sij, softmax_scale)
    tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)

    # Update global maximum
    max_new = pypto.maximum(max_update, tilda_mij)

    # Compute exp(scores - max_new)
    tsub = pypto.sub(sij_scale, max_new)
    tilda_pij = pypto.exp(tsub)
    tilda_pij_fp16 = pypto.cast(tilda_pij, dtype)
    sum_local = pypto.sum(tilda_pij, dim=-1, keepdim=True)

    # Update sum using online algorithm
    tsub2 = pypto.sub(max_update, max_new)
    max_update[:] = max_new
    update_mul = pypto.exp(tsub2)
    sum_update[:] = sum_update * update_mul + sum_local

    pypto.set_vec_tile_shapes(128, 128)
    value_antiquant_scale = ctx_params.loop_tensors.value_antiquant_scale[ctx_params.loop_index.n2_idx]
    vj_fp32 = pypto.cast(vj_assemble, pypto.DT_FP32, pypto.CastMode.CAST_NONE)
    vj_antiquant_scale_fp32 = pypto.cast(value_antiquant_scale, pypto.DT_FP32, pypto.CastMode.CAST_NONE)
    vj_out_data_fp32 = pypto.mul(vj_fp32, vj_antiquant_scale_fp32)
    vj_assemble_antiquanted = pypto.cast(vj_out_data_fp32, pypto.DT_BF16, pypto.CastMode.CAST_NONE)

    # Update output using online algorithm
    pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
    oi_tmp = pypto.matmul(tilda_pij_fp16, vj_assemble_antiquanted, pypto.DT_FP32)

    # Store output
    pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
    out_update[:] = out_update * update_mul + oi_tmp


def finalize_output(dtype, ctx_params):
    """
    Finalize and write attention output.

    This function divides the accumulated output by the sum to get
    the final attention result and writes it to the output tensor.

    Args:
        dtype: Output data type
        ctx_params: Context parameters
    """

    d = ctx_params.kernel_cfg.d
    group = ctx_params.kernel_cfg.group
    v2_tile = ctx_params.tile_cfg.v2_tile
    g_tile = ctx_params.tile_cfg.g_tile
    out_update = ctx_params.temp_update_tensors.out_update
    sum_update = ctx_params.temp_update_tensors.sum_update
    atten_out = ctx_params.loop_tensors.atten_out

    # Divide by sum to get final attention result
    oi_final = pypto.div(out_update, sum_update, precision_type=pypto.PrecisionType.INTRINSIC)

    # Reshape and cast to output format
    pypto.set_vec_tile_shapes(1, g_tile, 1, d)
    oi_final_4d = pypto.cast(pypto.reshape(oi_final, [1, g_tile, 1, d]), dtype)

    b_idx = ctx_params.loop_index.b_idx
    n2_idx = ctx_params.loop_index.n2_idx
    s1_idx = ctx_params.loop_index.s1_idx
    n2_idx_start = n2_idx * group
    out_ofs = [b_idx, n2_idx_start, s1_idx, 0]
    pypto.assemble(oi_final_4d, out_ofs, atten_out)


@allow_in_graph
def incre_flash_attention_gqa_antiquant(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_antiquant_scale: torch.Tensor,
    value_antiquant_scale: torch.Tensor,
    kv_actual_seqs: torch.Tensor,
    block_table: torch.Tensor,
):
    atten_out = torch.zeros_like(query)
    input_values = [query, key, value, key_antiquant_scale, value_antiquant_scale, kv_actual_seqs, 
                    block_table, atten_out]
    incre_flash_attention_gqa_antiquant_kernel(*input_values)
    return atten_out