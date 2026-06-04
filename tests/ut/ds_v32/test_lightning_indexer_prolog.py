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
from dataclasses import dataclass, field
from typing import List, Set
import logging
import pytest
import pypto
from conftest import duration_estimate


SHAPE_DIM_2 = 2
SHAPE_DIM_3 = 3

NUM_NEG1 = -1
NUM_0 = 0
NUM_1 = 1
NUM_2 = 2
NUM_3 = 3
NUM_4 = 4
NUM_16 = 16
NUM_28 = 28
NUM_32 = 32
NUM_64 = 64
NUM_128 = 128
NUM_256 = 256
NUM_448 = 448
NUM_1024 = 1024
NUM_1536 = 1536
NUM_2048 = 2048
NUM_7168 = 7168


@dataclass
class LightningIndexerPrologTileConfig:
    c1_tile: List[List[int]]
    v1_tile: List[int]
    c2_tile: List[List[int]]
    v2_tile: List[int]
    rope_2d: List[int]
    rope_3d: List[int]
    rope_4d: List[int]


@dataclass
class LightningIndexerPrologParams:
    b: int
    s1: int
    dim: int
    q_lora_rank: int
    head_dim: int
    head_num: int
    rope_head_dim: int
    block_size: int
    block_num: int
    n_kv: int
    s2: int
    tile_bs: int = -1


@dataclass
class LightningIndexerPrologArgs:
    x: pypto.Tensor
    qr: pypto.Tensor
    q_w: pypto.Tensor
    k_w: pypto.Tensor
    proj_w: pypto.Tensor
    ln_w: pypto.Tensor
    ln_b: pypto.Tensor
    cos: pypto.Tensor
    sin: pypto.Tensor
    k_cache: pypto.Tensor
    k_cache_index: pypto.Tensor
    block_table: pypto.Tensor
    query: pypto.Tensor
    weight: pypto.Tensor
    k_cache_out: pypto.Tensor
    tile_config: LightningIndexerPrologTileConfig
    unroll_list: List[int]
    params: LightningIndexerPrologParams


@dataclass
class LightningIndexerPrologBuildConfig:
    b: int = NUM_28
    s1: int = NUM_1
    dim: int = NUM_7168
    rope_head_dim: int = NUM_64
    n_kv: int = NUM_1
    head_dim: int = NUM_128
    head_num: int = NUM_64
    q_lora_rank: int = NUM_1536
    s2_tile: int = NUM_2048
    block_size: int = NUM_128
    block_num: int = NUM_448
    tile_bs: int = NUM_NEG1
    c1_tile: List[List[int]] = field(
        default_factory=lambda: [
            [NUM_16, NUM_16],
            [NUM_256, NUM_256],
            [NUM_128, NUM_128],
        ]
    )
    v1_tile: List[int] = field(
        default_factory=lambda: [NUM_1, NUM_256, NUM_128, NUM_128]
    )
    c2_tile: List[List[int]] = field(
        default_factory=lambda: [
            [NUM_16, NUM_16],
            [NUM_256, NUM_256],
            [NUM_128, NUM_128],
        ]
    )
    v2_tile: List[int] = field(
        default_factory=lambda: [NUM_1, NUM_128, NUM_128, NUM_128]
    )
    rope_2d: List[int] = field(default_factory=lambda: [NUM_128, NUM_256])
    rope_3d_vals: List[int] = field(default_factory=lambda: [NUM_32, NUM_128, NUM_128])
    rope_4d: List[int] = field(
        default_factory=lambda: [NUM_1, NUM_64, NUM_128, NUM_128]
    )


def layer_norm(
    x: pypto.Tensor, weight: pypto.Tensor, bias: pypto.Tensor, dim: int
) -> pypto.Tensor:
    assert dim == (len(x.shape) - 1) or dim == -1
    assert x.dtype == pypto.DT_FP32
    eps = 1e-6
    actual_dim = dim + len(x.shape) if dim < 0 else dim
    x_scaled = x / (x.shape[actual_dim])
    mean = pypto.sum(x_scaled, -1, True)
    diff = x - mean
    squared_diff = diff * diff
    square_diff_scaled = squared_diff / (x.shape[actual_dim])
    var = pypto.sum(square_diff_scaled, -1, True)
    std_var = pypto.sqrt(var + eps)
    res32 = diff / std_var
    weight32 = pypto.cast(weight, pypto.DT_FP32)
    bias32 = pypto.cast(bias, pypto.DT_FP32)
    return res32 * weight32 + bias32


def rotate_half(input_tensor: pypto.Tensor) -> pypto.Tensor:
    shape = input_tensor.shape
    shape_size = len(shape)
    assert shape_size >= 1
    assert shape[shape_size - 1] % NUM_2 == 0
    shape[shape_size - 1] //= NUM_2
    offset1 = [0] * shape_size
    offset2 = [0] * shape_size
    offset2[shape_size - 1] = shape[shape_size - 1]
    x1 = pypto.view(input_tensor, shape, offset1)
    x2 = pypto.view(input_tensor, shape, offset2)
    return pypto.concat([x2 * (-1.0), x1 + 0.0], -1)


def rotate_half_valid_shape(input_tensor: pypto.Tensor) -> pypto.Tensor:
    shape = input_tensor.shape
    shape_size = len(shape)
    assert shape_size >= 1
    assert shape[shape_size - 1] % NUM_2 == 0
    shape[shape_size - 1] //= NUM_2
    offset1 = [0] * shape_size
    offset2 = [0] * shape_size
    offset2[shape_size - 1] = shape[shape_size - 1]
    valid_shape = input_tensor.shape
    valid_shape[shape_size - 1] //= NUM_2
    x1 = pypto.view(input_tensor, shape, offset1, valid_shape=valid_shape)
    x2 = pypto.view(input_tensor, shape, offset2, valid_shape=valid_shape)
    return pypto.concat([x2 * (-1.0), x1 + 0.0], -1)


def rope_3d(
    x: pypto.Tensor,
    cos: pypto.Tensor,
    sin: pypto.Tensor,
    tile_config: LightningIndexerPrologTileConfig,
) -> pypto.Tensor:
    assert (
        len(x.shape) == SHAPE_DIM_3
        and len(cos.shape) == SHAPE_DIM_2
        and len(sin.shape) == SHAPE_DIM_2
    )
    pypto.set_vec_tile_shapes(NUM_1, NUM_32, NUM_128)
    cast_x = pypto.cast(x, pypto.DT_FP32)
    if x.dtype == pypto.DT_FP32:
        cast_x = cast_x + 0.0
    cast_cos = pypto.cast(cos, pypto.DT_FP32)
    cast_sin = pypto.cast(sin, pypto.DT_FP32)
    cast_cos[:] = pypto.reshape(cast_cos, [x.shape[NUM_0], 1, x.shape[NUM_2]])
    cast_sin[:] = pypto.reshape(cast_sin, [x.shape[NUM_0], 1, x.shape[NUM_2]])
    x_valid_shape = x.shape
    x_view = pypto.reshape(
        cast_x,
        [x.shape[NUM_0], x.shape[NUM_1], x.shape[NUM_2] // NUM_2, NUM_2],
        valid_shape=[
            x_valid_shape[NUM_0],
            x_valid_shape[NUM_1],
            x_valid_shape[NUM_2] // NUM_2,
            NUM_2,
        ],
    )
    pypto.set_vec_tile_shapes(NUM_1, NUM_32, NUM_128, NUM_128)
    x_trans = pypto.transpose(x_view, NUM_2, NUM_3)
    x_re_second = pypto.reshape(x_trans, x.shape, valid_shape=x_valid_shape)
    pypto.set_vec_tile_shapes(NUM_1, NUM_32, NUM_128, NUM_128)
    x_embed = x_re_second * cast_cos + rotate_half_valid_shape(x_re_second) * cast_sin
    return pypto.cast(x_embed, x.dtype)


def rope(
    x: pypto.Tensor,
    cos: pypto.Tensor,
    sin: pypto.Tensor,
    tile_config: LightningIndexerPrologTileConfig,
) -> pypto.Tensor:
    assert (
        len(x.shape) == SHAPE_DIM_2
        and len(cos.shape) == SHAPE_DIM_2
        and len(sin.shape) == SHAPE_DIM_2
    )
    seq_size = x.shape[NUM_0]
    d_r = x.shape[NUM_1]
    x_dtype = x.dtype
    pypto.set_vec_tile_shapes(
        tile_config.rope_2d[NUM_0],
        tile_config.rope_2d[NUM_1],
    )
    cast_x = pypto.cast(x, pypto.DT_FP32)
    if x_dtype == pypto.DT_FP32:
        cast_x = cast_x + 0.0
    cast_cos = pypto.cast(cos, pypto.DT_FP32)
    cast_sin = pypto.cast(sin, pypto.DT_FP32)
    x_view = pypto.reshape(cast_x, [1, seq_size, d_r // NUM_2, NUM_2])
    pypto.set_vec_tile_shapes(
        tile_config.rope_4d[0],
        tile_config.rope_4d[1],
        tile_config.rope_4d[2],
        tile_config.rope_4d[3],
    )
    x_trans = pypto.transpose(x_view, NUM_2, NUM_3)
    x_re_second = pypto.reshape(x_trans, [seq_size, d_r])
    pypto.set_vec_tile_shapes(
        tile_config.rope_2d[NUM_0],
        tile_config.rope_2d[NUM_1],
    )
    x_embed = x_re_second * cast_cos + rotate_half(x_re_second) * cast_sin
    return pypto.cast(x_embed, x_dtype)


def setup_lightning_indexer_prolog_config():
    pypto.set_pass_options(
                         cube_l1_reuse_setting={-1: NUM_4},
                         cube_nbuffer_setting={NUM_3: NUM_4})


def build_lightning_indexer_prolog_args(
    cfg: LightningIndexerPrologBuildConfig = LightningIndexerPrologBuildConfig(),
):
    d_bf16 = pypto.DT_BF16
    d_i32 = pypto.DT_INT32

    x = pypto.tensor(dtype=d_bf16, shape=[cfg.b, cfg.s1, cfg.dim], name="x")
    qr = pypto.tensor(
        dtype=d_bf16,
        shape=[cfg.b, cfg.s1, cfg.q_lora_rank],
        name="qr",
    )
    q_w = pypto.tensor(
        dtype=d_bf16,
        shape=[cfg.q_lora_rank, cfg.head_num * cfg.head_dim],
        name="q_w",
    )
    k_w = pypto.tensor(
        dtype=d_bf16,
        shape=[cfg.dim, cfg.head_dim],
        name="k_w",
    )
    proj_w = pypto.tensor(
        dtype=d_bf16,
        shape=[cfg.dim, cfg.head_num],
        name="proj_w",
    )
    ln_w = pypto.tensor(
        dtype=d_bf16,
        shape=[cfg.head_dim],
        name="ln_w",
    )
    ln_b = pypto.tensor(
        dtype=d_bf16,
        shape=[cfg.head_dim],
        name="ln_b",
    )
    cos = pypto.tensor(
        dtype=d_bf16,
        shape=[cfg.b, cfg.s1, cfg.rope_head_dim],
        name="cos",
    )
    sin = pypto.tensor(
        dtype=d_bf16,
        shape=[cfg.b, cfg.s1, cfg.rope_head_dim],
        name="sin",
    )
    k_cache = pypto.tensor(
        dtype=d_bf16,
        shape=[cfg.block_num, cfg.block_size, cfg.n_kv, cfg.head_dim],
        name="k_cache",
    )
    k_cache_index = pypto.tensor(
        dtype=d_i32,
        shape=[cfg.b, cfg.s1],
        name="k_cache_index",
    )
    block_table = pypto.tensor(
        dtype=d_i32,
        shape=[cfg.b, cfg.s2_tile // cfg.block_size],
        name="block_table",
    )

    query = pypto.tensor(
        dtype=d_bf16,
        shape=[cfg.b * cfg.s1, cfg.head_num, cfg.head_dim],
        name="qOut",
    )
    weight = pypto.tensor(
        dtype=d_bf16,
        shape=[cfg.b * cfg.s1, cfg.head_num],
        name="weightOut",
    )
    k_cache_out = pypto.tensor(
        dtype=d_bf16,
        shape=[cfg.block_num, cfg.block_size, cfg.n_kv, cfg.head_dim],
        name="kCacheOut",
    )

    tile_cfg = LightningIndexerPrologTileConfig(
        c1_tile=cfg.c1_tile,
        v1_tile=cfg.v1_tile,
        c2_tile=cfg.c2_tile,
        v2_tile=cfg.v2_tile,
        rope_2d=cfg.rope_2d,
        rope_3d=cfg.rope_3d_vals,
        rope_4d=cfg.rope_4d,
    )

    params = LightningIndexerPrologParams(
        b=cfg.b,
        s1=cfg.s1,
        dim=cfg.dim,
        q_lora_rank=cfg.q_lora_rank,
        head_dim=cfg.head_dim,
        head_num=cfg.head_num,
        rope_head_dim=cfg.rope_head_dim,
        block_size=cfg.block_size,
        block_num=cfg.block_num,
        n_kv=cfg.n_kv,
        s2=cfg.s2_tile,
        tile_bs=cfg.tile_bs,
    )

    unroll_list: List[int] = [1, 2, 4, 8, 16, 32]

    args = LightningIndexerPrologArgs(
        x=x,
        qr=qr,
        q_w=q_w,
        k_w=k_w,
        proj_w=proj_w,
        ln_w=ln_w,
        ln_b=ln_b,
        cos=cos,
        sin=sin,
        k_cache=k_cache,
        k_cache_index=k_cache_index,
        block_table=block_table,
        query=query,
        weight=weight,
        k_cache_out=k_cache_out,
        tile_config=tile_cfg,
        unroll_list=unroll_list,
        params=params,
    )

    meta = {
        "b": cfg.b,
        "s1": cfg.s1,
        "head_num": cfg.head_num,
        "head_dim": cfg.head_dim,
        "rope_head_dim": cfg.rope_head_dim,
        "q_lora_rank": cfg.q_lora_rank,
        "dim": cfg.dim,
        "n_kv": cfg.n_kv,
        "cache": {
            "blockSize": cfg.block_size,
            "block_num": cfg.block_num,
        },
        "tiles": {
            "c1Tile": cfg.c1_tile,
            "v1Tile": cfg.v1_tile,
            "c2Tile": cfg.c2_tile,
            "v2Tile": cfg.v2_tile,
            "rope2D": cfg.rope_2d,
            "rope3D": cfg.rope_3d_vals,
            "rope4D": cfg.rope_4d,
        },
        "dims": {
            "x": [cfg.b, cfg.s1, cfg.dim],
            "qr": [cfg.b, cfg.s1, cfg.q_lora_rank],
            "q_w": [cfg.q_lora_rank, cfg.head_num * cfg.head_dim],
            "k_w": [cfg.dim, cfg.head_dim],
            "proj_w": [cfg.dim, cfg.head_num],
            "ln_w": [cfg.head_dim],
            "ln_b": [cfg.head_dim],
            "cos/sin": [cfg.b, cfg.s1, cfg.rope_head_dim],
            "k_cache": [
                cfg.block_num,
                cfg.block_size,
                cfg.n_kv,
                cfg.head_dim,
            ],
            "k_cache_index": [cfg.b, cfg.s1],
            "block_table": [cfg.b, cfg.s2_tile // cfg.block_size],
            "query(out)": [
                cfg.b * cfg.s1,
                cfg.head_num,
                cfg.head_dim,
            ],
            "weight(out)": [cfg.b * cfg.s1, cfg.head_num],
            "k_cache_out(out)": [
                cfg.block_num,
                cfg.block_size,
                cfg.n_kv,
                cfg.head_dim,
            ],
        },
        "dtype": str(d_bf16),
        "tile_bs": cfg.tile_bs,
        "unrollList": sorted(list(unroll_list)),
    }

    return args, meta
