#!/usr/bin/env python3
# coding: utf-8
"""PyPTO implementation of QkvRmsNormRopeCache.

This module intentionally lives under ``pypto/custom`` and does not depend on
the AscendC implementation.  The exported ``qkv_rms_norm_rope_cache_wrapper``
matches the public AscendC argument order and returns the three required output
tensors.  The current target implementation supports the INT8 quantized PA_NZ
cache branch used by the two network cases; the Python wrapper only validates
arguments and dispatches.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

import pypto


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

TP4_Q_GROUP_HEADS = TP4_TILE_CONFIG.q_group_heads
TP4_Q_VEC_TOKEN_TILE = TP4_TILE_CONFIG.q_vec_token_tile
TP4_KV_VEC_TOKEN_TILE = TP4_TILE_CONFIG.kv_vec_token_tile
TP1_Q_GROUP_HEADS = TP1_TILE_CONFIG.q_group_heads
TP1_Q_VEC_TOKEN_TILE = TP1_TILE_CONFIG.q_vec_token_tile
TP1_KV_VEC_TOKEN_TILE = TP1_TILE_CONFIG.kv_vec_token_tile


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


def _scatter_pa_nz_int8_contiguous_page0(
    cache_out: pypto.Tensor,
    src: pypto.Tensor,
    tokens: int,
    num_heads: int,
    dim: int,
    c0: int,
    vec_token_tile: int,
):
    d1_per_head = dim // c0
    c1 = num_heads * d1_per_head
    pypto.set_vec_tile_shapes(vec_token_tile, c1, c0)
    src_3d = pypto.reshape(src, [tokens, c1, c0])
    pypto.set_vec_tile_shapes(vec_token_tile, c1, c0)
    src_c1_t_c0 = pypto.transpose(src_3d, 0, 1)
    pypto.set_vec_tile_shapes(1, c1, vec_token_tile, c0)
    src_pa_nz = pypto.reshape(src_c1_t_c0, [1, c1, tokens, c0])
    pypto.assemble(src_pa_nz, [0, 0, 0, 0], cache_out)


def _compute_q_grouped(
    qkv: pypto.Tensor,
    q_gamma: pypto.Tensor,
    cos_fp32: pypto.Tensor,
    sin_fp32: pypto.Tensor,
    q_out_out: pypto.Tensor,
    tokens: int,
    num_q: int,
    dim: int,
    epsilon: float,
    group_heads: int,
    vec_token_tile: int,
):
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


def _compute_quant(
    qkv: pypto.Tensor,
    q_gamma: pypto.Tensor,
    k_gamma: pypto.Tensor,
    cos: pypto.Tensor,
    sin: pypto.Tensor,
    index: pypto.Tensor,
    k_cache: pypto.Tensor,
    v_cache: pypto.Tensor,
    k_scale: pypto.Tensor,
    v_scale: pypto.Tensor,
    q_out_out: pypto.Tensor,
    k_cache_out: pypto.Tensor,
    v_cache_out: pypto.Tensor,
    qkv_size: list,
    head_nums: list,
    epsilon: float,
    q_group_heads: int,
    q_vec_token_tile: int,
    kv_vec_token_tile: int,
):
    tokens = qkv.shape[0]  # shape scalar: T=48 or 12, dynamic token axis
    static_tokens = qkv_size[0] * qkv_size[1]  # shape scalar: B*S, 48 or 12
    c0 = k_cache.shape[3]  # shape scalar: 32
    num_q = head_nums[0]  # shape scalar: 16 or 64
    num_k = head_nums[1]  # shape scalar: 1 or 4
    num_v = head_nums[2]  # shape scalar: 1 or 4
    dim = qkv_size[3]  # shape scalar: 128
    q_size = num_q * dim  # shape scalar: 2048 or 8192
    k_size = num_k * dim  # shape scalar: 128 or 512
    v_size = num_v * dim  # shape scalar: 128 or 512
    pypto.set_vec_tile_shapes(kv_vec_token_tile, dim)  # tile shape: [2,128]
    cos_static = pypto.view(cos, [static_tokens, dim], [0, 0])  # shape: [T,128]
    sin_static = pypto.view(sin, [static_tokens, dim], [0, 0])  # shape: [T,128]
    cos_fp32 = pypto.cast(cos_static, pypto.DT_FP32)  # shape: [T,128]
    sin_fp32 = pypto.cast(sin_static, pypto.DT_FP32)  # shape: [T,128]

    if num_q <= 16:
        _compute_q_grouped(qkv, q_gamma, cos_fp32, sin_fp32, q_out_out, static_tokens, num_q, dim, epsilon, q_group_heads, q_vec_token_tile)  # shape: [48,16,128] -> [48,2048]
    else:
        _compute_q_grouped(qkv, q_gamma, cos_fp32, sin_fp32, q_out_out, static_tokens, num_q, dim, epsilon, q_group_heads, q_vec_token_tile)  # shape: [12,64,128] -> grouped heads -> [12,8192]

    pypto.set_vec_tile_shapes(kv_vec_token_tile, k_size)  # tile shape: [2,128] or [4,512]
    k_2d_all = pypto.view(qkv, [static_tokens, k_size], [0, q_size])  # shape: [T,Nk*128]
    v_2d_all = pypto.view(qkv, [static_tokens, v_size], [0, q_size + k_size])  # shape: [T,Nv*128]
    pypto.set_vec_tile_shapes(kv_vec_token_tile, num_k, dim)  # tile shape: [2,Nk,128] or [4,Nk,128]
    k_3d_all = pypto.reshape(k_2d_all, [static_tokens, num_k, dim])  # shape: [T,Nk,128]
    v_3d_all = pypto.reshape(v_2d_all, [static_tokens, num_v, dim])  # shape: [T,Nv,128]
    k_norm_all = _rms_norm(k_3d_all, k_gamma, epsilon)  # shape: [T,Nk,128]
    cos_all = pypto.view(cos_fp32, [static_tokens, dim], [0, 0])  # shape: [T,128]
    sin_all = pypto.view(sin_fp32, [static_tokens, dim], [0, 0])  # shape: [T,128]
    k_rope_all = _rope_fp32(k_norm_all, cos_all, sin_all, kv_vec_token_tile)  # shape: [T,Nk,128]

    pypto.set_vec_tile_shapes(kv_vec_token_tile, num_k, dim)  # tile shape: [2,Nk,128] or [4,Nk,128]
    k_quant_all = _quant_int8(k_rope_all, k_scale)  # shape: [T,Nk,128] int8
    v_quant_all = _quant_int8(v_3d_all, v_scale)  # shape: [T,Nv,128] int8
    _scatter_pa_nz_int8_contiguous_page0(k_cache_out, k_quant_all, static_tokens, num_k, dim, c0, kv_vec_token_tile)  # dst shape: [11898,Nk*4,128,32], index=arange(T)
    _scatter_pa_nz_int8_contiguous_page0(v_cache_out, v_quant_all, static_tokens, num_v, dim, c0, kv_vec_token_tile)  # dst shape: [11898,Nv*4,128,32], index=arange(T)


@pypto.frontend.jit(
    runtime_options={
        "stitch_function_max_num": TP4_TILE_CONFIG.stitch_function_max_num,
        "device_sched_mode": TP4_TILE_CONFIG.device_sched_mode,
    },
)
def qkv_rms_norm_rope_cache_quant_kernel_tp4(
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
):
    pypto.set_vec_tile_shapes(1, 128)
    _compute_quant(
        qkv,
        q_gamma,
        k_gamma,
        cos,
        sin,
        index,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        q_out_out,
        k_cache_out,
        v_cache_out,
        qkv_size,
        head_nums,
        epsilon,
        TP4_Q_GROUP_HEADS,
        TP4_Q_VEC_TOKEN_TILE,
        TP4_KV_VEC_TOKEN_TILE,
    )


@pypto.frontend.jit(
    runtime_options={
        "stitch_function_max_num": TP1_TILE_CONFIG.stitch_function_max_num,
        "device_sched_mode": TP1_TILE_CONFIG.device_sched_mode,
    },
)
def qkv_rms_norm_rope_cache_quant_kernel_tp1(
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
):
    pypto.set_vec_tile_shapes(1, 128)
    _compute_quant(
        qkv,
        q_gamma,
        k_gamma,
        cos,
        sin,
        index,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        q_out_out,
        k_cache_out,
        v_cache_out,
        qkv_size,
        head_nums,
        epsilon,
        TP1_Q_GROUP_HEADS,
        TP1_Q_VEC_TOKEN_TILE,
        TP1_KV_VEC_TOKEN_TILE,
    )


def qkv_rms_norm_rope_cache_wrapper(
    qkv: torch.Tensor,
    q_gamma: torch.Tensor,
    k_gamma: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    index: torch.Tensor,
    q_out: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    k_offset: Optional[torch.Tensor] = None,
    v_offset: Optional[torch.Tensor] = None,
    qkv_size: Sequence[int] = (),
    head_nums: Sequence[int] = (),
    epsilon: float = 1e-6,
    cache_mode: str = "PA_NZ",
    is_output_qkv: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the PyPTO kernel and return ``(q_out, k_cache, v_cache)``.

    ``q_out``, ``k_cache`` and ``v_cache`` are treated like AscendC in/out
    tensors.  The public wrapper keeps these arguments in the AscendC order,
    while the private JIT kernel only receives the writable output buffers.
    """
    if qkv.dtype != torch.bfloat16:
        raise TypeError("this PyPTO implementation currently supports BF16 qkv only")
    if k_cache.dtype != v_cache.dtype:
        raise TypeError("k_cache and v_cache must have the same dtype")
    if cache_mode != "PA_NZ":
        raise NotImplementedError("this PyPTO implementation currently supports PA_NZ cache_mode only")
    if is_output_qkv:
        raise NotImplementedError("this PyPTO implementation currently supports is_output_qkv=False only")
    q_out_out = q_out
    k_cache_out = k_cache
    v_cache_out = v_cache
    if k_cache.dtype != torch.int8:
        raise TypeError("current network implementation requires int8 k_cache/v_cache")
    if k_scale is None or v_scale is None:
        raise ValueError("k_scale and v_scale are required for int8 cache")
    if k_offset is not None or v_offset is not None:
        raise NotImplementedError("asymmetric quantization is not supported yet")
    kernel = qkv_rms_norm_rope_cache_quant_kernel_tp4 if head_nums[0] <= 16 else qkv_rms_norm_rope_cache_quant_kernel_tp1
    kernel(
        qkv,
        q_gamma,
        k_gamma,
        cos,
        sin,
        index,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        q_out_out,
        k_cache_out,
        v_cache_out,
        list(qkv_size),
        list(head_nums),
        float(epsilon),
    )
    return q_out_out, k_cache_out, v_cache_out
