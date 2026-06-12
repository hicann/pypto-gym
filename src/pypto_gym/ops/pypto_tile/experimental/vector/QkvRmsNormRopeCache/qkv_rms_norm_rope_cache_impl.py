# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""PyPTO implementation of QkvRmsNormRopeCache.

This module intentionally lives under ``pypto/custom`` and does not depend on
the AscendC implementation.  The exported ``qkv_rms_norm_rope_cache_wrapper``
matches the public AscendC argument order and returns the three required output
tensors.  The current target implementation supports the INT8 quantized PA_NZ
cache branch used by the two network cases; the Python wrapper only validates
arguments and dispatches.
"""

import collections
from dataclasses import dataclass
from typing import Tuple

import pypto
import torch


@dataclass(frozen=True)
class QkvTileConfig:
    q_group_heads: int
    q_vec_token_tile: int
    kv_vec_token_tile: int
    stitch_function_max_num: int
    device_sched_mode: int


TP4_TILE_CONFIG = QkvTileConfig(
    q_group_heads=16,
    q_vec_token_tile=4,
    kv_vec_token_tile=2,
    stitch_function_max_num=128,
    device_sched_mode=3,
)
TP1_TILE_CONFIG = QkvTileConfig(
    q_group_heads=16,
    q_vec_token_tile=4,
    kv_vec_token_tile=4,
    stitch_function_max_num=128,
    device_sched_mode=3,
)
GENERIC_TILE_CONFIG = QkvTileConfig(
    q_group_heads=16,
    q_vec_token_tile=2,
    kv_vec_token_tile=2,
    stitch_function_max_num=128,
    device_sched_mode=3,
)

TP4_Q_GROUP_HEADS = TP4_TILE_CONFIG.q_group_heads
TP4_Q_VEC_TOKEN_TILE = TP4_TILE_CONFIG.q_vec_token_tile
TP4_KV_VEC_TOKEN_TILE = TP4_TILE_CONFIG.kv_vec_token_tile
TP1_Q_GROUP_HEADS = TP1_TILE_CONFIG.q_group_heads
TP1_Q_VEC_TOKEN_TILE = TP1_TILE_CONFIG.q_vec_token_tile
TP1_KV_VEC_TOKEN_TILE = TP1_TILE_CONFIG.kv_vec_token_tile
GENERIC_Q_GROUP_HEADS = GENERIC_TILE_CONFIG.q_group_heads
GENERIC_Q_VEC_TOKEN_TILE = GENERIC_TILE_CONFIG.q_vec_token_tile
GENERIC_KV_VEC_TOKEN_TILE = GENERIC_TILE_CONFIG.kv_vec_token_tile
GENERIC_KV_GROUP_HEADS = 2
TP4_KERNEL_TILE_CONFIG = [TP4_Q_GROUP_HEADS, TP4_Q_VEC_TOKEN_TILE, TP4_KV_VEC_TOKEN_TILE, GENERIC_KV_GROUP_HEADS, 0]
TP1_KERNEL_TILE_CONFIG = [TP1_Q_GROUP_HEADS, TP1_Q_VEC_TOKEN_TILE, TP1_KV_VEC_TOKEN_TILE, GENERIC_KV_GROUP_HEADS, 0]
GENERIC_KERNEL_TILE_CONFIG = [
    GENERIC_Q_GROUP_HEADS,
    GENERIC_Q_VEC_TOKEN_TILE,
    GENERIC_KV_VEC_TOKEN_TILE,
    GENERIC_KV_GROUP_HEADS,
    1,
]

ScatterContiguousParams = collections.namedtuple(
    "ScatterContiguousParams",
    ["cache_out", "src", "tokens", "num_heads", "dim", "c0", "vec_token_tile"],
)

ScatterIndexedParams = collections.namedtuple(
    "ScatterIndexedParams",
    [
        "cache_in",
        "cache_out",
        "index",
        "src",
        "tokens",
        "group_heads",
        "dim",
        "c0",
        "head_start",
        "vec_token_tile",
    ],
)

QGroupParams = collections.namedtuple(
    "QGroupParams",
    [
        "qkv",
        "q_gamma",
        "cos_fp32",
        "sin_fp32",
        "q_out_out",
        "tokens",
        "num_q",
        "dim",
        "epsilon",
        "group_heads",
        "vec_token_tile",
    ],
)

KvGroupParams = collections.namedtuple(
    "KvGroupParams",
    [
        "qkv",
        "k_gamma",
        "cos_fp32",
        "sin_fp32",
        "index",
        "k_cache",
        "v_cache",
        "k_scale",
        "v_scale",
        "k_cache_out",
        "v_cache_out",
        "tokens",
        "q_size",
        "k_size",
        "num_k",
        "num_v",
        "dim",
        "c0",
        "epsilon",
        "group_heads",
        "vec_token_tile",
    ],
)

QuantParams = collections.namedtuple(
    "QuantParams",
    [
        "qkv",
        "q_gamma",
        "k_gamma",
        "cos",
        "sin",
        "index",
        "k_cache",
        "v_cache",
        "k_scale",
        "v_scale",
        "q_out_out",
        "k_cache_out",
        "v_cache_out",
        "qkv_size",
        "head_nums",
        "epsilon",
        "q_group_heads",
        "q_vec_token_tile",
        "kv_vec_token_tile",
    ],
)

QuantGenericParams = collections.namedtuple(
    "QuantGenericParams",
    QuantParams._fields + ("kv_group_heads",),
)

QKV_RUNTIME_ARG_NAMES = (
    "index", "q_out", "k_cache", "v_cache", "k_scale", "v_scale", "k_offset",
    "v_offset", "qkv_size", "head_nums", "epsilon", "cache_mode", "is_output_qkv",
)
QkvRuntimeArgs = collections.namedtuple("QkvRuntimeArgs", QKV_RUNTIME_ARG_NAMES)


def _pop_cos_sin_args(args: tuple, kwargs: dict) -> tuple[torch.Tensor, torch.Tensor, tuple]:
    values = []
    rest = args
    for name in ("cos", "sin"):
        if rest:
            values.append(rest[0])
            rest = rest[1:]
        elif name in kwargs:
            values.append(kwargs.pop(name))
        else:
            raise TypeError(f"missing required argument: {name}")
    return values[0], values[1], rest


def _parse_qkv_runtime_args(args: tuple, kwargs: dict) -> QkvRuntimeArgs:
    defaults = {
        "k_scale": None,
        "v_scale": None,
        "k_offset": None,
        "v_offset": None,
        "qkv_size": (),
        "head_nums": (),
        "epsilon": 1e-6,
        "cache_mode": "PA_NZ",
        "is_output_qkv": False,
    }
    required = QKV_RUNTIME_ARG_NAMES[:4]
    if len(args) > len(QKV_RUNTIME_ARG_NAMES):
        raise TypeError("too many positional arguments")
    values = dict(defaults)
    for name, value in zip(QKV_RUNTIME_ARG_NAMES, args):
        values[name] = value
    for name in QKV_RUNTIME_ARG_NAMES:
        if name in kwargs:
            values[name] = kwargs.pop(name)
    if kwargs:
        raise TypeError(f"unexpected keyword argument(s): {sorted(kwargs)}")
    missing = [name for name in required if name not in values]
    if missing:
        raise TypeError(f"missing required argument(s): {missing}")
    return QkvRuntimeArgs(*(values[name] for name in QKV_RUNTIME_ARG_NAMES))


def _rotate_half(x: pypto.Tensor) -> pypto.Tensor:
    shape = x.shape
    dim = shape[len(shape) - 1]
    half = dim // 2
    left_shape = list(shape)
    left_shape[len(shape) - 1] = half
    offsets_0 = [0] * len(shape)
    offsets_1 = [0] * len(shape)
    offsets_1[len(shape) - 1] = half
    x1 = pypto.view(x, left_shape, offsets_0)
    x2 = pypto.view(x, left_shape, offsets_1)
    return pypto.concat([x2 * (-1.0), x1 + 0.0], -1)


def _rope_fp32(x: pypto.Tensor, cos_fp32: pypto.Tensor, sin_fp32: pypto.Tensor, vec_token_tile: int) -> pypto.Tensor:
    dtype = x.dtype
    t = x.shape[0]
    n = x.shape[1]
    d = x.shape[2]
    x_fp32 = pypto.cast(x, pypto.DT_FP32)
    cos_3d = pypto.reshape(cos_fp32, [t, 1, d], valid_shape=[t, 1, d])
    sin_3d = pypto.reshape(sin_fp32, [t, 1, d], valid_shape=[t, 1, d])
    pypto.set_vec_tile_shapes(vec_token_tile, n, d)
    return pypto.cast(x_fp32 * cos_3d + _rotate_half(x_fp32) * sin_3d, dtype)


def _rope(x: pypto.Tensor, cos: pypto.Tensor, sin: pypto.Tensor, vec_token_tile: int) -> pypto.Tensor:
    return _rope_fp32(x, pypto.cast(cos, pypto.DT_FP32), pypto.cast(sin, pypto.DT_FP32), vec_token_tile)


def _rms_norm(x: pypto.Tensor, gamma: pypto.Tensor, epsilon: float) -> pypto.Tensor:
    dtype = x.dtype
    dim = x.shape[len(x.shape) - 1]
    gamma_shape = [1] * len(x.shape)
    gamma_shape[len(x.shape) - 1] = dim
    x_fp32 = pypto.cast(x, pypto.DT_FP32)
    gamma_fp32 = pypto.cast(pypto.reshape(gamma, gamma_shape), pypto.DT_FP32)
    square = x_fp32 * x_fp32
    mean = pypto.sum(square, -1, keepdim=True) * (1.0 / dim)
    inv_rms = pypto.div(pypto.full(mean.shape, 1.0, pypto.DT_FP32), pypto.sqrt(mean + epsilon))
    return pypto.cast(x_fp32 * inv_rms * gamma_fp32, dtype)


def _quant_int8(x: pypto.Tensor, scale: pypto.Tensor) -> pypto.Tensor:
    shape = x.shape
    dim = shape[len(shape) - 1]
    scale_shape = [1] * len(shape)
    scale_shape[len(shape) - 2] = shape[len(shape) - 2]
    scale_shape[len(shape) - 1] = dim
    x_fp32 = pypto.cast(x, pypto.DT_FP32, pypto.CastMode.CAST_NONE)
    scale_fp32 = pypto.cast(pypto.reshape(scale, scale_shape), pypto.DT_FP32, pypto.CastMode.CAST_NONE)
    quant_fp32 = x_fp32 / scale_fp32
    quant_int32 = pypto.cast(quant_fp32, pypto.DT_INT32, pypto.CastMode.CAST_RINT)
    quant_fp16 = pypto.cast(quant_int32, pypto.DT_FP16, pypto.CastMode.CAST_ROUND)
    return pypto.cast(quant_fp16, pypto.DT_INT8, pypto.CastMode.CAST_TRUNC, satmode=pypto.SaturationMode.ON)


def _scatter_pa_nz_int8_contiguous_page0(params: ScatterContiguousParams):
    cache_out, src, tokens, num_heads, dim, c0, vec_token_tile = params
    d1_per_head = dim // c0
    c1 = num_heads * d1_per_head
    pypto.set_vec_tile_shapes(vec_token_tile, c1, c0)
    src_3d = pypto.reshape(src, [tokens, c1, c0])
    pypto.set_vec_tile_shapes(vec_token_tile, c1, c0)
    src_c1_t_c0 = pypto.transpose(src_3d, 0, 1)
    pypto.set_vec_tile_shapes(1, c1, vec_token_tile, c0)
    src_pa_nz = pypto.reshape(src_c1_t_c0, [1, c1, tokens, c0])
    pypto.assemble(src_pa_nz, [0, 0, 0, 0], cache_out)


def _scatter_pa_nz_int8_indexed_group(params: ScatterIndexedParams):
    cache_in, cache_out, index, src, tokens, group_heads, dim, c0, head_start, vec_token_tile = params
    d1_per_head = dim // c0
    c1 = group_heads * d1_per_head
    c1_offset = head_start * d1_per_head
    block_num = cache_in.shape[0]
    block_size = cache_in.shape[2]

    pypto.set_vec_tile_shapes(vec_token_tile, c1, c0)
    src_3d = pypto.reshape(src, [tokens, c1, c0])
    pypto.set_vec_tile_shapes(tokens, group_heads * dim)
    src_flat = pypto.reshape(src_3d, [tokens, group_heads * dim])

    cache_group = pypto.view(cache_in, [block_num, c1, block_size, c0], [0, c1_offset, 0, 0])
    pypto.set_vec_tile_shapes(1, block_size, c1, c0)
    cache_block_c1 = pypto.transpose(cache_group, 1, 2)
    pypto.set_vec_tile_shapes(block_num * block_size, group_heads * dim)
    cache_scatter = pypto.reshape(cache_block_c1, [block_num * block_size, group_heads * dim])
    updated = pypto.scatter(cache_scatter, 0, index, src_flat)
    pypto.set_vec_tile_shapes(1, block_size, c1, c0)
    updated_block_c1 = pypto.reshape(updated, [block_num, block_size, c1, c0])
    updated_group = pypto.transpose(updated_block_c1, 1, 2)
    pypto.assemble(updated_group, [0, c1_offset, 0, 0], cache_out)


def _compute_q_grouped(params: QGroupParams):
    qkv, q_gamma, cos_fp32, sin_fp32, q_out_out, tokens, num_q, dim, epsilon, group_heads, vec_token_tile = params
    for head_start in range(0, num_q, group_heads):
        pypto.set_vec_tile_shapes(vec_token_tile, group_heads * dim)
        q_2d = pypto.view(qkv, [tokens, group_heads * dim], [0, head_start * dim])
        pypto.set_vec_tile_shapes(vec_token_tile, group_heads, dim)
        q_3d = pypto.reshape(q_2d, [tokens, group_heads, dim])
        q_norm = _rms_norm(q_3d, q_gamma, epsilon)
        cos_q = pypto.view(cos_fp32, [tokens, dim], [0, 0])
        sin_q = pypto.view(sin_fp32, [tokens, dim], [0, 0])
        q_rope = _rope_fp32(q_norm, cos_q, sin_q, vec_token_tile)
        pypto.set_vec_tile_shapes(vec_token_tile, group_heads * dim)
        q_res = pypto.reshape(q_rope, [tokens, group_heads * dim])
        pypto.assemble(q_res, [0, head_start * dim], q_out_out)


def _compute_kv_grouped_fallback(params: KvGroupParams):
    (
        qkv,
        k_gamma,
        cos_fp32,
        sin_fp32,
        index,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        k_cache_out,
        v_cache_out,
        tokens,
        q_size,
        k_size,
        num_k,
        num_v,
        dim,
        c0,
        epsilon,
        group_heads,
        vec_token_tile,
    ) = params
    for head_start in range(0, num_k, group_heads):
        pypto.set_vec_tile_shapes(vec_token_tile, group_heads * dim)
        k_2d = pypto.view(qkv, [tokens, group_heads * dim], [0, q_size + head_start * dim])
        v_2d = pypto.view(qkv, [tokens, group_heads * dim], [0, q_size + k_size + head_start * dim])
        pypto.set_vec_tile_shapes(vec_token_tile, group_heads, dim)
        k_3d = pypto.reshape(k_2d, [tokens, group_heads, dim])
        v_3d = pypto.reshape(v_2d, [tokens, group_heads, dim])
        k_norm = _rms_norm(k_3d, k_gamma, epsilon)
        cos_k = pypto.view(cos_fp32, [tokens, dim], [0, 0])
        sin_k = pypto.view(sin_fp32, [tokens, dim], [0, 0])
        k_rope = _rope_fp32(k_norm, cos_k, sin_k, vec_token_tile)
        k_scale_group = pypto.view(k_scale, [group_heads, dim], [head_start, 0])
        v_scale_group = pypto.view(v_scale, [group_heads, dim], [head_start, 0])
        pypto.set_vec_tile_shapes(vec_token_tile, group_heads, dim)
        k_quant = _quant_int8(k_rope, k_scale_group)
        v_quant = _quant_int8(v_3d, v_scale_group)
        _scatter_pa_nz_int8_indexed_group(
            ScatterIndexedParams(
                k_cache, k_cache_out, index, k_quant, tokens, group_heads, dim, c0, head_start, vec_token_tile
            )
        )
        _scatter_pa_nz_int8_indexed_group(
            ScatterIndexedParams(
                v_cache, v_cache_out, index, v_quant, tokens, group_heads, dim, c0, head_start, vec_token_tile
            )
        )


def _cos_sin_fp32(
    cos: pypto.Tensor,
    sin: pypto.Tensor,
    static_tokens: int,
    dim: int,
    vec_token_tile: int,
) -> tuple[pypto.Tensor, pypto.Tensor]:
    pypto.set_vec_tile_shapes(vec_token_tile, dim)
    cos_static = pypto.view(cos, [static_tokens, dim], [0, 0])
    sin_static = pypto.view(sin, [static_tokens, dim], [0, 0])
    return pypto.cast(cos_static, pypto.DT_FP32), pypto.cast(sin_static, pypto.DT_FP32)


def _compute_quant(params: QuantParams):
    static_tokens = params.qkv_size[0] * params.qkv_size[1]
    c0 = params.k_cache.shape[3]
    num_q = params.head_nums[0]
    num_k = params.head_nums[1]
    num_v = params.head_nums[2]
    dim = params.qkv_size[3]
    q_size = num_q * dim
    k_size = num_k * dim
    v_size = num_v * dim
    cos_fp32, sin_fp32 = _cos_sin_fp32(params.cos, params.sin, static_tokens, dim, params.kv_vec_token_tile)

    _compute_q_grouped(
        QGroupParams(
            params.qkv, params.q_gamma, cos_fp32, sin_fp32, params.q_out_out,
            static_tokens, num_q, dim, params.epsilon, params.q_group_heads, params.q_vec_token_tile,
        )
    )

    pypto.set_vec_tile_shapes(params.kv_vec_token_tile, k_size)
    k_2d_all = pypto.view(params.qkv, [static_tokens, k_size], [0, q_size])
    v_2d_all = pypto.view(params.qkv, [static_tokens, v_size], [0, q_size + k_size])
    pypto.set_vec_tile_shapes(params.kv_vec_token_tile, num_k, dim)
    k_3d_all = pypto.reshape(k_2d_all, [static_tokens, num_k, dim])
    v_3d_all = pypto.reshape(v_2d_all, [static_tokens, num_v, dim])
    k_norm_all = _rms_norm(k_3d_all, params.k_gamma, params.epsilon)
    cos_all = pypto.view(cos_fp32, [static_tokens, dim], [0, 0])
    sin_all = pypto.view(sin_fp32, [static_tokens, dim], [0, 0])
    k_rope_all = _rope_fp32(k_norm_all, cos_all, sin_all, params.kv_vec_token_tile)

    pypto.set_vec_tile_shapes(params.kv_vec_token_tile, num_k, dim)
    k_quant_all = _quant_int8(k_rope_all, params.k_scale)
    v_quant_all = _quant_int8(v_3d_all, params.v_scale)
    _scatter_pa_nz_int8_contiguous_page0(
        ScatterContiguousParams(
            params.k_cache_out, k_quant_all, static_tokens, num_k, dim, c0, params.kv_vec_token_tile
        )
    )
    _scatter_pa_nz_int8_contiguous_page0(
        ScatterContiguousParams(
            params.v_cache_out, v_quant_all, static_tokens, num_v, dim, c0, params.kv_vec_token_tile
        )
    )


def _compute_quant_generic_fallback(params: QuantGenericParams):
    static_tokens = params.qkv_size[0] * params.qkv_size[1]
    c0 = params.k_cache.shape[3]
    num_q = params.head_nums[0]
    num_k = params.head_nums[1]
    num_v = params.head_nums[2]
    dim = params.qkv_size[3]
    q_size = num_q * dim
    k_size = num_k * dim
    cos_fp32, sin_fp32 = _cos_sin_fp32(params.cos, params.sin, static_tokens, dim, params.kv_vec_token_tile)

    _compute_q_grouped(
        QGroupParams(
            params.qkv, params.q_gamma, cos_fp32, sin_fp32, params.q_out_out,
            static_tokens, num_q, dim, params.epsilon, params.q_group_heads, params.q_vec_token_tile,
        )
    )
    _compute_kv_grouped_fallback(
        KvGroupParams(
            params.qkv, params.k_gamma, cos_fp32, sin_fp32, params.index, params.k_cache, params.v_cache,
            params.k_scale, params.v_scale, params.k_cache_out, params.v_cache_out, static_tokens, q_size,
            k_size, num_k, num_v, dim, c0, params.epsilon, params.kv_group_heads, params.kv_vec_token_tile,
        )
    )


@pypto.frontend.jit(
    runtime_options={
        "stitch_function_max_num": TP4_TILE_CONFIG.stitch_function_max_num,
        "device_sched_mode": TP4_TILE_CONFIG.device_sched_mode,
    },
)
def qkv_rms_norm_rope_cache_quant_kernel_regular(
    qkv: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    q_gamma: pypto.Tensor([pypto.STATIC], pypto.DT_BF16),
    k_gamma: pypto.Tensor([pypto.STATIC], pypto.DT_BF16),
    cos: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    sin: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    index: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT64),
    k_cache: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    v_cache: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    k_scale: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    v_scale: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    q_out_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    k_cache_out: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    v_cache_out: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    qkv_size: list,
    head_nums: list,
    epsilon: float,
    tile_config: list,
):
    pypto.set_vec_tile_shapes(1, 128)
    _compute_quant(
        QuantParams(
            qkv, q_gamma, k_gamma, cos, sin, index, k_cache, v_cache, k_scale, v_scale,
            q_out_out, k_cache_out, v_cache_out, qkv_size, head_nums, epsilon,
            tile_config[0], tile_config[1], tile_config[2],
        )
    )


@pypto.frontend.jit(
    runtime_options={
        "stitch_function_max_num": GENERIC_TILE_CONFIG.stitch_function_max_num,
        "device_sched_mode": GENERIC_TILE_CONFIG.device_sched_mode,
    },
)
def qkv_rms_norm_rope_cache_quant_kernel_generic(
    qkv: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    q_gamma: pypto.Tensor([pypto.STATIC], pypto.DT_BF16),
    k_gamma: pypto.Tensor([pypto.STATIC], pypto.DT_BF16),
    cos: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    sin: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    index: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT64),
    generic_k_cache: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    generic_v_cache: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    generic_k_scale: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    generic_v_scale: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    generic_q_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    generic_k_cache_out: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    generic_v_cache_out: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    generic_qkv_size: list,
    generic_head_nums: list,
    generic_epsilon: float,
):
    pypto.set_vec_tile_shapes(1, 128)
    _compute_quant_generic_fallback(
        QuantGenericParams(
            qkv, q_gamma, k_gamma, cos, sin, index, generic_k_cache, generic_v_cache,
            generic_k_scale, generic_v_scale, generic_q_out, generic_k_cache_out,
            generic_v_cache_out, generic_qkv_size, generic_head_nums, generic_epsilon,
            GENERIC_Q_GROUP_HEADS, GENERIC_Q_VEC_TOKEN_TILE, GENERIC_KV_VEC_TOKEN_TILE,
            GENERIC_KV_GROUP_HEADS,
        )
    )


def _validate_qkv_runtime(qkv: torch.Tensor, runtime: QkvRuntimeArgs) -> None:
    if qkv.dtype != torch.bfloat16:
        raise TypeError("this PyPTO implementation currently supports BF16 qkv only")
    if runtime.k_cache.dtype != runtime.v_cache.dtype:
        raise TypeError("k_cache and v_cache must have the same dtype")
    if runtime.cache_mode != "PA_NZ":
        raise NotImplementedError("this PyPTO implementation currently supports PA_NZ cache_mode only")
    if runtime.is_output_qkv:
        raise NotImplementedError("this PyPTO implementation currently supports is_output_qkv=False only")
    if runtime.k_cache.dtype != torch.int8:
        raise TypeError("current network implementation requires int8 k_cache/v_cache")
    if runtime.k_scale is None or runtime.v_scale is None:
        raise ValueError("k_scale and v_scale are required for int8 cache")
    if runtime.k_offset is not None or runtime.v_offset is not None:
        raise NotImplementedError("asymmetric quantization is not supported yet")


def _select_qkv_tile_config(head_nums: Tuple[int, int, int]) -> list[int]:
    if head_nums[1] > 4 or head_nums[2] > 4 or head_nums[0] > 64:
        return GENERIC_KERNEL_TILE_CONFIG
    if head_nums[0] <= 16:
        return TP4_KERNEL_TILE_CONFIG
    return TP1_KERNEL_TILE_CONFIG


def _expand_qkv_index(index: torch.Tensor, qkv_size: Tuple[int, int, int, int]) -> torch.Tensor:
    dim = int(qkv_size[3])
    return index.reshape(index.numel(), 1).expand(index.numel(), GENERIC_KV_GROUP_HEADS * dim).contiguous()


def qkv_rms_norm_rope_cache_wrapper(
    qkv: torch.Tensor,
    q_gamma: torch.Tensor,
    k_gamma: torch.Tensor,
    *args,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the PyPTO kernel and return ``(q_out, k_cache, v_cache)``."""
    cos, sin, args = _pop_cos_sin_args(args, kwargs)
    runtime = _parse_qkv_runtime_args(args, kwargs)
    _validate_qkv_runtime(qkv, runtime)
    q_out_out = runtime.q_out
    k_cache_out = runtime.k_cache
    v_cache_out = runtime.v_cache
    qkv_size = tuple(runtime.qkv_size)
    head_nums = tuple(runtime.head_nums)
    tile_config = _select_qkv_tile_config(head_nums)
    if tile_config[4] == 1:
        kernel_index = _expand_qkv_index(runtime.index, qkv_size)
        qkv_rms_norm_rope_cache_quant_kernel_generic(
            qkv, q_gamma, k_gamma, cos, sin, kernel_index, runtime.k_cache, runtime.v_cache,
            runtime.k_scale, runtime.v_scale, q_out_out, k_cache_out, v_cache_out,
            list(qkv_size), list(head_nums), float(runtime.epsilon),
        )
    else:
        qkv_rms_norm_rope_cache_quant_kernel_regular(
            qkv, q_gamma, k_gamma, cos, sin, runtime.index, runtime.k_cache, runtime.v_cache,
            runtime.k_scale, runtime.v_scale, q_out_out, k_cache_out, v_cache_out,
            list(qkv_size), list(head_nums), float(runtime.epsilon), list(tile_config),
        )
    return q_out_out, k_cache_out, v_cache_out
