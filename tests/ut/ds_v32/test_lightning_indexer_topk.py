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
import sys
from dataclasses import dataclass, field
from typing import List, Set, Optional
import logging
import pytest
import pypto

SHAPE_DIM0 = 0
SHAPE_DIM1 = 1
SHAPE_DIM2 = 2
SHAPE_DIM3 = 3

NUM_NEG1 = -1
NUM_0 = 0
NUM_1 = 1
NUM_2 = 2
NUM_3 = 3
NUM_4 = 4
NUM_8 = 8
NUM_16 = 16
NUM_32 = 32
NUM_64 = 64
NUM_100 = 100
NUM_128 = 128
NUM_1024 = 1024
NUM_1127 = 1127
NUM_2048 = 2048
NUM_4096 = 4096
NUM_8192 = 8192
AVOID_FP32_TO_FP16_OVERFLOW_SCALE = 1.0 / 2048.0


@dataclass
class LightningIndexerTileConfig:
    weight_tile: List[int]
    c1_tile: List[List[int]]
    v1_tile: List[int]
    topk_tile: List[int]
    adds_tile: List[int]


@dataclass
class LightningIndexerParams:
    b: int
    s1: int
    index_n1: int
    qk_nope: int
    qk_rope: int
    n2: int
    block_size: int
    block_num: int
    selected_count: int
    is_quant: bool = False


@dataclass
class LightningIndexerInputs:
    query: pypto.Tensor
    key: pypto.Tensor
    weights: pypto.Tensor
    act_seq_key: pypto.Tensor
    block_table: pypto.Tensor
    topk_res: pypto.Tensor
    q_scale: Optional[pypto.Tensor]
    k_scale: Optional[pypto.Tensor]
    tmp_out: Optional[pypto.Tensor]
    topk_value: Optional[pypto.Tensor]
    tile_config: LightningIndexerTileConfig
    unroll_list: Set[int]
    params: LightningIndexerParams


@dataclass
class LightningIndexerBuildConfig:
    b: int = NUM_4
    s1: int = NUM_2
    index_n1: int = NUM_64
    qk_nope: int = NUM_128
    qk_rope: int = NUM_0
    n2: int = NUM_1
    block_size: int = NUM_128
    block_num: int = NUM_1127
    selected_count: int = NUM_2048
    is_quant: bool = True
    c1_tile: List[List[int]] = field(
        default_factory=lambda: [
            [NUM_64, NUM_64],
            [NUM_128, NUM_128],
            [NUM_128, NUM_128],
        ]
    )
    v1_tile: List[int] = field(default_factory=lambda: [NUM_64, NUM_128])
    topk_tile: List[int] = field(default_factory=lambda: [NUM_1, NUM_4096])
    adds_tile: List[int] = field(
        default_factory=lambda: [NUM_1, NUM_1, NUM_1, NUM_4096]
    )


def setup_lightning_indexer_topk_config():
    pypto.set_pass_options(
                         cube_l1_reuse_setting={-1: NUM_32},
                         vec_nbuffer_setting={NUM_NEG1: NUM_16})


def build_lightning_indexer_topk_args(
    cfg: LightningIndexerBuildConfig = LightningIndexerBuildConfig(),
):
    d_bf16 = pypto.DT_FP16
    d_i32 = pypto.DT_INT32
    d_int8 = pypto.DT_INT8
    d_f16 = pypto.DT_FP16

    index_d = cfg.qk_nope + cfg.qk_rope
    max_block_num = NUM_1024

    if cfg.is_quant:
        qk_dtype = d_int8
        scale_dtype = d_f16
    else:
        qk_dtype = d_bf16
        scale_dtype = d_f16

    query = pypto.tensor(
        [cfg.b, cfg.s1, cfg.index_n1, index_d],
        qk_dtype,
        "query",
    )

    key = pypto.tensor(
        [cfg.block_num, cfg.block_size, cfg.n2, index_d],
        qk_dtype,
        "key",
    )

    weights = pypto.tensor(
        [cfg.b, cfg.s1, cfg.index_n1],
        d_bf16,
        "weights",
    )

    act_seq_key = pypto.tensor(
        [cfg.b],
        d_i32,
        "actSeqKey",
    )

    block_table = pypto.tensor(
        [cfg.b, max_block_num],
        d_i32,
        "blockTable",
    )

    topk_res = pypto.tensor(
        [cfg.b, cfg.s1, cfg.n2, cfg.selected_count],
        d_i32,
        "topkRes",
    )

    q_scale = (
        pypto.tensor(
            [cfg.b, cfg.s1, cfg.index_n1, 1],
            scale_dtype,
            "qScale",
        )
        if cfg.is_quant
        else None
    )
    k_scale = (
        pypto.tensor(
            [cfg.block_num, cfg.block_size, cfg.n2, 1],
            scale_dtype,
            "kScale",
        )
        if cfg.is_quant
        else None
    )

    tmp_out = None
    topk_value = None

    tile_cfg = LightningIndexerTileConfig(
        weight_tile=[NUM_64, NUM_128],
        c1_tile=cfg.c1_tile,
        v1_tile=cfg.v1_tile,
        topk_tile=cfg.topk_tile,
        adds_tile=cfg.adds_tile,
    )

    unroll_list: List[int] = [1, 2, 4, 8, 16, 32, 64]

    params = LightningIndexerParams(
        b=cfg.b,
        s1=cfg.s1,
        index_n1=cfg.index_n1,
        qk_nope=cfg.qk_nope,
        qk_rope=cfg.qk_rope,
        n2=cfg.n2,
        block_size=cfg.block_size,
        block_num=cfg.block_num,
        selected_count=cfg.selected_count,
        is_quant=cfg.is_quant,
    )

    args = LightningIndexerInputs(
        query=query,
        key=key,
        weights=weights,
        act_seq_key=act_seq_key,
        block_table=block_table,
        topk_res=topk_res,
        q_scale=q_scale,
        k_scale=k_scale,
        tmp_out=tmp_out,
        topk_value=topk_value,
        tile_config=tile_cfg,
        unroll_list=unroll_list,
        params=params,
    )

    meta = {
        "B": cfg.b,
        "S1": cfg.s1,
        "indexN1": cfg.index_n1,
        "indexD": index_d,
        "N2": cfg.n2,
        "blockSize": cfg.block_size,
        "blockNum": cfg.block_num,
        "maxBlockNum": max_block_num,
        "selectedCount": cfg.selected_count,
        "isQuant": cfg.is_quant,
        "dims": {
            "query": [cfg.b, cfg.s1, cfg.index_n1, index_d],
            "key": [cfg.block_num, cfg.block_size, cfg.n2, index_d],
            "weights": [cfg.b, cfg.s1, cfg.index_n1],
            "actSeqKey": [cfg.b],
            "blockTable": [cfg.b, max_block_num],
            "topkRes": [cfg.b, cfg.s1, cfg.n2, cfg.selected_count],
            "qScale": ([cfg.b, cfg.s1, cfg.index_n1, 1] if cfg.is_quant else None),
            "kScale": (
                [cfg.block_num, cfg.block_size, cfg.n2, 1] if cfg.is_quant else None
            ),
        },
        "tiles": {
            "weightTile": tile_cfg.weight_tile,
            "c1Tile": tile_cfg.c1_tile,
            "v1Tile": tile_cfg.v1_tile,
            "topkTile": tile_cfg.topk_tile,
            "addsTile": tile_cfg.adds_tile,
        },
        "unrollList": sorted(list(unroll_list)),
    }

    return args, meta
