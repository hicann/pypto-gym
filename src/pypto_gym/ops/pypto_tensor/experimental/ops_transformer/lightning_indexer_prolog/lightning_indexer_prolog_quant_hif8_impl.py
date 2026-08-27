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
Lightning Indexer Prolog Quantization Module

This module implements the Lightning Indexer Prolog quantization computation
for DeepSeek V32 model. It handles:
- Query computation with dynamic quantization
- Key computation with RmsNorm and RoPE
- Weight computation for indexer attention

Main Functions:
    - lightning_indexer_prolog_quant_compute: Main computation function
    - quant_rms_norm: Quantized RmsNorm implementation
    - prolog_quant: Per-token quantization function
    - quant_rope_2d: 2D RoPE (Rotary Position Embedding) computation
    - rope_3d: 3D RoPE computation

Example:
    See pg_lightning_indexer_prolog_quant_hif8.py for usage examples.
"""

import pypto

SHAPE_DIM_2 = 2
SHAPE_DIM_3 = 3
COS_SIN_DIM = 2
SCATTER_DIM = -2


def quant_rms_norm(x: pypto.Tensor, gamma: pypto.Tensor, dim: int, epsilon: float):
    """Compute quantized RmsNorm operation.

    Applies Rms Normalization with quantization support. The function normalizes
    the input tensor along the specified dimension using mean and variance,
    then applies learnable scale (gamma) parameters.

    Args:
        x: Input tensor to normalize, shape depends on input
        gamma: Scale parameter tensor, shape should match the normalization dimension
        dim: Dimension along which to normalize. Can be -1 (last dimension) or
             len(x.shape) - 1 (last dimension explicitly)
        epsilon: Small constant added to variance to avoid division by zero

    Returns:
        Normalized tensor with the same shape as input x, with scale and shift applied

    Note:
        The function performs normalization in FP32 precision to maintain numerical
        stability, then casts back to the original dtype.
    """
    assert (dim == len(x.shape) - 1) or (dim == -1)
    actual_dim = dim + len(x.shape) if dim < 0 else dim
    x_dtype = x.dtype

    x_fp32 = pypto.cast(x, pypto.DT_FP32)
    # do division first to avoid overflow
    x2 = x_fp32 * x_fp32
    x2_scaled = x2 * (1.0 / x.shape[actual_dim])
    mean_square = pypto.sum(x2_scaled, actual_dim, keepdim=True)

    rms = pypto.sqrt(mean_square + epsilon)
    res32 = pypto.div(x_fp32, rms, pypto.PrecisionType.INTRINSIC)
    gamma32 = pypto.cast(gamma, pypto.DT_FP32)
    return pypto.cast((res32 * gamma32), x_dtype)


def quant_rope_2d(x: pypto.Tensor, cos: pypto.Tensor, sin: pypto.Tensor):
    """Apply 2D Rotary Position Embedding (RoPE) to input tensor.

    Implements RoPE transformation for 2D tensors. RoPE encodes positional
    information by rotating the input tensor using cosine and sine values.

    Args:
        x: Input tensor of shape (t_tile, rope_dim), where t_tile is the
           sequence length and rope_dim is the RoPE dimension
        cos: Cosine values for RoPE, shape (t_tile, rope_dim)
        sin: Sine values for RoPE, shape (t_tile, rope_dim)

    Returns:
        Tensor with RoPE applied, same shape as input x

    Note:
        The function performs rotation in FP32 precision for numerical stability,
        then casts back to the original dtype.
    """
    key_rope_dim = 2
    x_dtype = x.dtype
    t_tile = x.shape[0]
    rope_dim = x.shape[1]
    assert len(x.shape) == key_rope_dim and len(cos.shape) == COS_SIN_DIM and len(sin.shape) == COS_SIN_DIM

    pypto.set_vec_tile_shapes(t_tile, rope_dim)
    cast_cos = pypto.cast(cos, pypto.DT_FP32)
    cast_sin = pypto.cast(sin, pypto.DT_FP32)
    x_view = pypto.cast(x, pypto.DT_FP32)

    x_embed = (x_view * cast_cos) + ((rotate_half(x_view)) * cast_sin)
    res = pypto.cast(x_embed, x_dtype)
    return res


def prolog_quant(x: pypto.Tensor):
    """Perform per-token quantization to HiF8.

    Quantizes the input tensor to HiF8 format using dynamic quantization.
    The quantization scale is computed per-token based on the maximum absolute
    value, ensuring the full HiF8 range [-32768, 32768] is utilized.

    Args:
        input: Input tensor to quantize, can be any shape. Quantization is
               performed along the last dimension per token.

    Returns:
        Tuple of (quantized_tensor, dequant_scale):
            - quantized_tensor: HiF8 quantized tensor, same shape as input
            - dequant_scale: FP32 scale factor for dequantization, shape matches
                            input with last dimension reduced to 1

    Note:
        The quantization process:
        1. Find per-token maximum absolute value
        2. Compute scale = 32768.0 / max_value
        3. Quantize: hif8 = round(input * scale)
        4. Return dequantization scale = 1.0 / scale
    """
    pypto.experimental.set_operation_options(combine_axis=True)

    hif8_max_value = 32768.0
    hif8_one_value = 1.0
    input_fp32 = pypto.cast(x, pypto.DT_FP32, pypto.CastMode.CAST_NONE)

    abs_res = pypto.abs(input_fp32)
    max_value = pypto.amax(abs_res, dim=-1, keepdim=True)

    scale_dequant = max_value * (hif8_one_value / hif8_max_value)
    out_fp32 = pypto.div(input_fp32, scale_dequant, pypto.PrecisionType.INTRINSIC)
    out_hif8 = pypto.cast(out_fp32, pypto.DT_HF8)
    return (out_hif8, scale_dequant)


def rotate_half(input_tensor: pypto.Tensor) -> pypto.Tensor:
    """Rotate half of the tensor dimensions for RoPE computation.

    Splits the last dimension in half and applies rotation transformation:
    [-x2, x1] where x1 is the first half and x2 is the second half.
    This is a key component of RoPE (Rotary Position Embedding).

    Args:
        input_tensor: Input tensor with last dimension divisible by 2

    Returns:
        Rotated tensor with same shape as input, where the first half of
        the last dimension is negated and swapped with the second half

    Raises:
        AssertionError: If the last dimension is not divisible by 2

    Example:
        If input is [a, b, c, d] along last dim, output is [-c, -d, a, b]
    """
    chunk_size = 2
    shape = input_tensor.shape
    shape_size = len(shape)
    assert shape_size >= 1
    assert shape[shape_size - 1] % chunk_size == 0
    shape[shape_size - 1] //= chunk_size
    offset1 = [0] * shape_size
    offset2 = [0] * shape_size
    offset2[shape_size - 1] = shape[shape_size - 1]
    x1 = pypto.view(input_tensor, shape, offset1)
    x2 = pypto.view(input_tensor, shape, offset2)
    return pypto.concat([x2 * (-1.0), x1 + 0.0], -1)


def rope_3d(x: pypto.Tensor, cos: pypto.Tensor, sin: pypto.Tensor) -> pypto.Tensor:
    """Apply 3D Rotary Position Embedding (RoPE) to input tensor.

    Implements RoPE transformation for 3D tensors with shape (t_tile, head_num, rope_dim).
    The RoPE is applied independently to each head using the provided cosine and sine values.

    Args:
        x: Input tensor of shape (t_tile, head_num, rope_dim)
        cos: Cosine values for RoPE, shape (t_tile, rope_dim)
        sin: Sine values for RoPE, shape (t_tile, rope_dim)

    Returns:
        Tensor with RoPE applied, same shape as input x

    Note:
        The function broadcasts cos and sin to match the head dimension,
        then applies rotation: x_rotated = x * cos + rotate_half(x) * sin
    """
    head_num_axis = 1
    head_dim_axis = 2
    assert len(x.shape) == SHAPE_DIM_3 and len(cos.shape) == SHAPE_DIM_2 and len(sin.shape) == SHAPE_DIM_2

    x_dtype = x.dtype
    t_tile = x.shape[0]
    head_num = x.shape[head_num_axis]
    rope_dim = x.shape[head_dim_axis]

    pypto.set_vec_tile_shapes(8, rope_dim)
    cast_cos = pypto.cast(cos, pypto.DT_FP32)
    cast_sin = pypto.cast(sin, pypto.DT_FP32)
    cast_cos = pypto.reshape(cast_cos, [t_tile, 1, rope_dim])
    cast_sin = pypto.reshape(cast_sin, [t_tile, 1, rope_dim])

    pypto.set_vec_tile_shapes(8, head_num, rope_dim)
    x_view = pypto.cast(x, pypto.DT_FP32)

    x_embed = (x_view * cast_cos) + ((rotate_half(x_view)) * cast_sin)
    res = pypto.cast(x_embed, x_dtype)
    return res


@pypto.frontend.jit(
    pass_options={
        # 0 cast_cos/sin, 1 q_dequant, 6 q_quant
        "vec_nbuffer_setting": {0: 2, 1: 2, 6: 8, -2: 1},
        "cube_l1_reuse_setting": {-1: 8},
    },
    runtime_options={"device_sched_mode": 1},
)
def lightning_indexer_prolog_quant(
    x_in: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16, format=pypto.TileOpFormat.TILEOP_ND),
    q_norm_in: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_HF8, format=pypto.TileOpFormat.TILEOP_ND),
    q_norm_scale_in: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32, format=pypto.TileOpFormat.TILEOP_ND),
    w_qb_in: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_HF8, format=pypto.TileOpFormat.TILEOP_ND),
    w_qb_scale_in: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32, format=pypto.TileOpFormat.TILEOP_ND),
    wk_in: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16, format=pypto.TileOpFormat.TILEOP_ND),
    w_proj_in: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16, format=pypto.TileOpFormat.TILEOP_ND),
    gamma_k_in: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16, format=pypto.TileOpFormat.TILEOP_ND),
    cos_idx_rope_in: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16, format=pypto.TileOpFormat.TILEOP_ND),
    sin_idx_rope_in: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16, format=pypto.TileOpFormat.TILEOP_ND),
    hadamard_q_in: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16, format=pypto.TileOpFormat.TILEOP_ND),
    hadamard_k_in: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16, format=pypto.TileOpFormat.TILEOP_ND),
    k_quant_in: pypto.Tensor(
        [pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_HF8, format=pypto.TileOpFormat.TILEOP_ND
    ),
    k_scale_in: pypto.Tensor(
        [pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32, format=pypto.TileOpFormat.TILEOP_ND
    ),
    k_cache_index_in: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_INT64, format=pypto.TileOpFormat.TILEOP_ND),
    k_scale_cache_index_in: pypto.Tensor(
        [pypto.DYNAMIC, pypto.STATIC], pypto.DT_INT64, format=pypto.TileOpFormat.TILEOP_ND
    ),
    q_quant_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_HF8, format=pypto.TileOpFormat.TILEOP_ND),
    q_scale_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32, format=pypto.TileOpFormat.TILEOP_ND),
    k_quant_out: pypto.Tensor(
        [pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_HF8, format=pypto.TileOpFormat.TILEOP_ND
    ),
    k_scale_out: pypto.Tensor(
        [pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32, format=pypto.TileOpFormat.TILEOP_ND
    ),
    weights_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16, format=pypto.TileOpFormat.TILEOP_ND),
):
    """Compute Lightning Indexer Prolog with quantization.

    Main computation function for Lightning Indexer Prolog quantization.
    This function processes input tokens to generate quantized query, key, and weights
    for the indexer attention mechanism. The computation includes:

    1. Query Path:
       - Dequantize q_norm (HIF8) to FP32
       - Apply linear transformation with w_qb
       - Apply RoPE (Rotary Position Embedding)
       - Apply Hadamard transformation
       - Quantize to HIF8 with per-token-head scale

    2. Key Path:
       - Linear transformation with wk
       - RmsNorm normalization
       - Apply RoPE
       - Apply Hadamard transformation
       - Quantize to HIF8 with per-token-head scale
       - Update key cache using scatter_update

    3. Weights Path:
       - Linear transformation with w_proj
       - Normalize by sqrt(head_num * head_dim)
       - Convert to BF16

    Args:
        x_in: Input hidden states tensor, shape (t, h), dtype BF16
        q_norm_in: Quantized query norm tensor, shape (t, q_lora_rank), dtype HIF8
        q_norm_scale_in: Query norm dequantization scale, shape (t, 1), dtype FP32
        w_qb_in: Query projection weight matrix, HIF8 format with ND layout
        w_qb_scale_in: Query weight dequantization scale, shape (1, head_num * head_dim), dtype FP32
        wk_in: Key projection weight matrix, BF16 format with ND layout
        w_proj_in: Weight projection matrix, BF16 format with ND layout
        gamma_k_in: RmsNorm scale parameter for key, shape (1, head_dim), dtype BF16
        cos_idx_rope_in: Cosine values for RoPE, shape (t, rope_head_dim), dtype BF16
        sin_idx_rope_in: Sine values for RoPE, shape (t, rope_head_dim), dtype BF16
        hadamard_q_in: Hadamard transformation matrix for query, shape (head_dim, head_dim), dtype BF16
        hadamard_k_in: Hadamard transformation matrix for key, shape (head_dim, head_dim), dtype BF16
        k_quant_in: Input key cache, shape (block_num, block_size, n_kv, head_dim), dtype HIF8
        k_scale_in: Key cache scale, shape (block_num, block_size, n_kv, 1), dtype FP32
        k_cache_index_in: Cache index for scatter update, shape (t, 1), dtype INT64
        k_scale_cache_index_in: Cache index for scatter update, shape (t, 1), dtype INT64
        q_quant_out: Output quantized query tensor, shape (t, head_num, head_dim), dtype HIF8
        q_scale_out: Output query quantization scale, shape (t, head_num, 1), dtype FP32
        k_quant_out: Output key cache (updated in-place), shape (block_num, block_size, n_kv, head_dim), dtype HIF8
        k_scale_out: Output key cache scale (updated in-place), shape (block_num, block_size, n_kv, 1), dtype FP32
        weights_out: Output weights tensor, shape (t, head_num), dtype BF16

    Note:
        - The function processes tokens in tiles using loop_unroll for optimization
        - All outputs are written in-place using pypto.assemble or scatter_update
        - The computation uses dynamic tiling based on unroll_list
    """
    x_dtype = x_in.dtype
    # 动态轴
    t = x_in.shape[0]
    h = x_in.shape[1]
    q_lora_rank = q_norm_in.shape[1]
    head_num = w_proj_in.shape[1]
    head_dim = hadamard_q_in.shape[0]
    rope_head_dim = cos_idx_rope_in.shape[1]

    unroll_list = [128, 64, 32, 16, 8, 4, 2, 1]
    for t_idx, unroll_length in pypto.loop_unroll(
        0, t, 1, name="IndexerPrologQuantQuantLoop", idx_name="tIdx", unroll_list=unroll_list
    ):
        t_tile = unroll_length
        # 多分档内会将t_tile作为档位，offset无需乘t_tile
        pypto.set_semantic_label("Query-Linear")
        q_norm = pypto.view(q_norm_in, [t_tile, q_lora_rank], [t_idx, 0])
        q_norm_scale = pypto.view(q_norm_scale_in, [t_tile, 1], [t_idx, 0])
        pypto.set_cube_tile_shapes([128, 128], [256, 1024], [128, 128])
        q_proj = pypto.matmul(q_norm, w_qb_in, pypto.DT_FP32)

        pypto.set_semantic_label("Query-Dequant")
        pypto.set_vec_tile_shapes(8, head_num * head_dim)
        q_proj = q_proj * q_norm_scale
        q_proj = q_proj * w_qb_scale_in
        q_bf16 = pypto.cast(q_proj, x_dtype)

        pypto.set_semantic_label("Query-Rope")
        q_bf16 = pypto.reshape(q_bf16, [t_tile, head_num, head_dim])
        q_rope = pypto.view(q_bf16, [t_tile, head_num, rope_head_dim], [0, 0, 0])
        q_nope = pypto.view(q_bf16, [t_tile, head_num, head_dim - rope_head_dim], [0, 0, rope_head_dim])
        rope_cos = pypto.view(cos_idx_rope_in, [t_tile, rope_head_dim], [t_idx, 0])
        rope_sin = pypto.view(sin_idx_rope_in, [t_tile, rope_head_dim], [t_idx, 0])
        q_roped = rope_3d(q_rope, rope_cos, rope_sin)
        q_cat = pypto.concat([q_roped, q_nope], -1)
        pypto.set_vec_tile_shapes(8, head_num, head_dim)
        q_cat_2d = pypto.reshape(q_cat, [t_tile * head_num, head_dim])

        pypto.set_semantic_label("Query-Hadamard")
        pypto.set_cube_tile_shapes([256, 256], [128, 128], [128, 128])
        q_hadamard = pypto.matmul(q_cat_2d, hadamard_q_in, x_dtype)

        pypto.set_semantic_label("Query-Quant")
        pypto.set_vec_tile_shapes(128, head_dim)
        q_res = prolog_quant(q_hadamard)
        pypto.assemble(q_res[0], [t_idx * head_num, 0], q_quant_out)
        pypto.assemble(q_res[1], [t_idx * head_num, 0], q_scale_out)

        pypto.set_semantic_label("Key-Linear")
        pypto.set_cube_tile_shapes([128, 128], [256, 1024], [128, 128])
        x = pypto.view(x_in, [t_tile, h], [t_idx, 0])
        k_proj = pypto.matmul(x, wk_in, x_dtype)

        pypto.set_semantic_label("Key-RmsNorm")
        pypto.set_vec_tile_shapes(128, head_dim)
        k_rms_norm = quant_rms_norm(k_proj, gamma_k_in, -1, 1e-6)

        pypto.set_semantic_label("Key-Rope")
        k_rope = pypto.view(k_rms_norm, [t_tile, rope_head_dim], [0, 0])
        k_nope = pypto.view(k_rms_norm, [t_tile, head_dim - rope_head_dim], [0, rope_head_dim])
        k_roped = quant_rope_2d(k_rope, rope_cos, rope_sin)
        pypto.set_vec_tile_shapes(128, head_dim)
        k_concat = pypto.concat([k_roped, k_nope], -1)

        pypto.set_semantic_label("Key-Hadamard")
        pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
        hadamard_k = pypto.matmul(k_concat, hadamard_k_in, x_dtype)

        pypto.set_semantic_label("Key-Quant")
        pypto.set_vec_tile_shapes(128, head_dim)
        k_res = prolog_quant(hadamard_k)
        k_cache_4d = pypto.reshape(k_res[0], [t_tile, 1, 1, head_dim])
        k_scale_4d = pypto.reshape(k_res[1], [t_tile, 1, 1, 1])

        index = pypto.view(k_cache_index_in, [t_tile, 1], [t_idx, 0])
        scale_index = pypto.view(k_scale_cache_index_in, [t_tile, 1], [t_idx, 0])
        pypto.set_vec_tile_shapes(128, 1, 1, head_dim)
        k_quant_out.move(pypto.scatter_update(k_quant_in, SCATTER_DIM, index, k_cache_4d))
        k_scale_out.move(pypto.scatter_update(k_scale_in, SCATTER_DIM, scale_index, k_scale_4d))

        pypto.set_semantic_label("Weight-Linear")
        pypto.set_cube_tile_shapes([32, 32], [1024, 1024], [32, 32])
        pypto.set_vec_tile_shapes(128, head_num)
        weights = pypto.cast(pypto.matmul(x, w_proj_in, x_dtype), pypto.DT_FP32)
        weights = pypto.cast(pypto.cast(weights * (head_num**-0.5), pypto.DT_BF16), pypto.DT_FP32)
        weights = pypto.cast(weights * (head_dim**-0.5), pypto.DT_BF16)
        pypto.assemble(weights, [t_idx, 0], weights_out)
