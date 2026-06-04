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
from typing import List
import logging
import pytest
import pypto
from conftest import duration_estimate


SHAPE_DIM_0 = 0
SHAPE_DIM_1 = 1


@dataclass
class SelectedAttentionTileConfig:
    g_tile: int
    s2_tile: int
    c1_tile: List
    v1_tile: List
    c2_tile: List
    v2_tile: List


@dataclass
class SASimpleParams:
    n_q: int
    n_kv: int
    softmax_scale: float
    topk: int
    tile: SelectedAttentionTileConfig


@dataclass
class SAInputs:
    q_nope: pypto.tensor
    q_rope: pypto.tensor
    k_slc: pypto.tensor
    v_slc: pypto.tensor
    kv_slc_act_seqs: pypto.tensor
    attention_out: pypto.tensor
    params: SASimpleParams


@dataclass
class SABuildConfig:
    b: int = 32
    s1: int = 4
    n_q: int = 128
    n_kv: int = 1
    qk_nope_head_dim: int = 512
    qk_rope_head_dim: int = 64
    kv_head_dim: int = 512
    topk: int = 2048
    softmax_scale: float = 1.0 / 24.0
    g_tile: int = 128
    s2_tile: int = 2048
    c1_tile: List[int] = field(
        default_factory=lambda: [[128, 128], [64, 64], [256, 256]]
    )
    v1_tile: List[int] = field(default_factory=lambda: [16, 256])
    c2_tile: List[int] = field(
        default_factory=lambda: [[128, 128], [128, 128], [128, 128]]
    )
    v2_tile: List[int] = field(default_factory=lambda: [64, 128])


def build_selected_args(cfg: SABuildConfig = SABuildConfig()):
    d_type = pypto.DT_FP16
    i32 = pypto.DT_INT32

    q_nope_shape = [cfg.b * cfg.s1 * cfg.n_q, cfg.qk_nope_head_dim]
    q_rope_shape = [cfg.b * cfg.s1 * cfg.n_q, cfg.qk_rope_head_dim]

    k_concat_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
    k_slc_shape = [cfg.b * cfg.s1 * cfg.topk, k_concat_dim]
    v_slc_shape = [cfg.b * cfg.s1 * cfg.topk, cfg.kv_head_dim]

    kv_slc_act_seqs_shape = [cfg.b]

    attention_out_shape = [cfg.b, cfg.s1, cfg.n_q, cfg.qk_nope_head_dim]

    q_nope = pypto.tensor(q_nope_shape, d_type, "qNope")
    q_rope = pypto.tensor(q_rope_shape, d_type, "qRope")
    k_slc = pypto.tensor(k_slc_shape, d_type, "kSlc")
    v_slc = pypto.tensor(v_slc_shape, d_type, "vSlc")
    kv_slc_act_seqs = pypto.tensor(kv_slc_act_seqs_shape, i32, "kvSlcActSeqs")
    attention_out = pypto.tensor(attention_out_shape, d_type, "attentionOut")

    tile = SelectedAttentionTileConfig(
        g_tile=cfg.g_tile,
        s2_tile=cfg.s2_tile,
        c1_tile=cfg.c1_tile,
        v1_tile=cfg.v1_tile,
        c2_tile=cfg.c2_tile,
        v2_tile=cfg.v2_tile,
    )
    params = SASimpleParams(
        n_q=cfg.n_q,
        n_kv=cfg.n_kv,
        softmax_scale=cfg.softmax_scale,
        topk=cfg.topk,
        tile=tile,
    )

    args = SAInputs(
        q_nope=q_nope,
        q_rope=q_rope,
        k_slc=k_slc,
        v_slc=v_slc,
        kv_slc_act_seqs=kv_slc_act_seqs,
        attention_out=attention_out,
        params=params,
    )

    meta = {
        "b": cfg.b,
        "s1": cfg.s1,
        "nQ": cfg.n_q,
        "nKv": cfg.n_kv,
        "dims": {
            "qNope": q_nope_shape,
            "qRope": q_rope_shape,
            "kSlc": k_slc_shape,
            "vSlc": v_slc_shape,
            "kvSlcActSeqs": kv_slc_act_seqs_shape,
            "attentionOut": attention_out_shape,
        },
        "topk": cfg.topk,
        "softmaxScale": cfg.softmax_scale,
        "tiles": {
            "gTile": cfg.g_tile,
            "s2Tile": cfg.s2_tile,
            "c1Tile": cfg.c1_tile,
            "v1Tile": cfg.v1_tile,
            "c2Tile": cfg.c2_tile,
            "v2Tile": cfg.v2_tile,
        },
    }
    return args, meta
