# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Torch CPU reference for QkvRmsNormRopeCache.

The reference follows the public AscendC operator contract, but it is written
from the mathematical definition only:
SplitVD -> RMSNorm(q, k) -> half-and-half RoPE(q, k) -> PA_NZ cache scatter.
"""

from __future__ import annotations

import collections
import importlib
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

if TYPE_CHECKING:
    import torch


QkvSizeInfo = collections.namedtuple(
    "QkvSizeInfo",
    ["batch", "seq", "num_qkv", "dim", "num_q", "num_k", "num_v"],
)

QkvNormRopeCacheOutput = collections.namedtuple(
    "QkvNormRopeCacheOutput",
    ["q_new", "k_new", "v_new", "q_extra", "k_extra", "v_extra"],
)


def _check_qkv_size(qkv_size: Sequence[int], head_nums: Sequence[int]) -> QkvSizeInfo:
    if len(qkv_size) != 4:
        raise ValueError(f"qkv_size must be [B, S, Nqkv, D], got {qkv_size}")
    if len(head_nums) != 3:
        raise ValueError(f"head_nums must be [Nq, Nk, Nv], got {head_nums}")
    batch, seq, num_qkv, dim = [int(x) for x in qkv_size]
    num_q, num_k, num_v = [int(x) for x in head_nums]
    if num_qkv != num_q + num_k + num_v:
        raise ValueError(f"Nqkv={num_qkv} must equal Nq+Nk+Nv={num_q + num_k + num_v}")
    if num_k != num_v:
        raise ValueError(f"Nk and Nv must be equal, got {num_k} and {num_v}")
    if dim % 2 != 0:
        raise ValueError(f"RoPE head dim must be even, got {dim}")
    return QkvSizeInfo(batch, seq, num_qkv, dim, num_q, num_k, num_v)


def rms_norm_torch(x: torch.Tensor, gamma: torch.Tensor, epsilon: float) -> torch.Tensor:
    x_fp32 = x.to(torch.float32)
    gamma_fp32 = gamma.to(torch.float32).view(*([1] * (x.dim() - 1)), gamma.numel())
    inv_rms = torch.rsqrt(torch.mean(x_fp32 * x_fp32, dim=-1, keepdim=True) + float(epsilon))
    return (x_fp32 * inv_rms * gamma_fp32).to(x.dtype)


def rope_torch(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dim = x.shape[-1]
    x_fp32 = x.to(torch.float32)
    cos_fp32 = cos.to(torch.float32)
    sin_fp32 = sin.to(torch.float32)
    while cos_fp32.dim() < x_fp32.dim():
        cos_fp32 = cos_fp32.unsqueeze(1)
        sin_fp32 = sin_fp32.unsqueeze(1)
    x1 = x_fp32[..., :dim // 2]
    x2 = x_fp32[..., dim // 2:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x_fp32 * cos_fp32 + rotated * sin_fp32).to(x.dtype)


def quant_to_int8_torch(x: torch.Tensor, scale: torch.Tensor, offset: Optional[torch.Tensor] = None) -> torch.Tensor:
    x_fp32 = x.to(torch.float32)
    scale_fp32 = scale.to(torch.float32).view(*([1] * (x.dim() - 2)), scale.shape[0], scale.shape[1])
    quant_fp32 = x_fp32 / scale_fp32
    if offset is not None:
        offset_fp32 = offset.to(torch.float32).view(*([1] * (x.dim() - 2)), offset.shape[0], offset.shape[1])
        quant_fp32 = quant_fp32 + offset_fp32
    quant_int32 = torch.round(quant_fp32).to(torch.int32)
    return quant_int32.clamp(-128, 127).to(torch.float16).to(torch.int8)


def scatter_pa_nz_torch(cache: torch.Tensor, index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """Scatter [T, N, D] src into PA_NZ cache [BlockNum, N*D//C0, BlockSize, C0]."""
    out = cache.clone()
    block_num, nz_cols, block_size, c0 = out.shape
    tokens, heads, dim = src.shape
    if nz_cols != heads * dim // c0:
        raise ValueError(f"cache dim1={nz_cols} does not match heads*dim//c0={heads * dim // c0}")
    flat = src.reshape(tokens, heads * dim)
    nz_src = flat.reshape(tokens, nz_cols, c0)
    index_cpu = index.to(torch.long).cpu()
    for token_idx, slot in enumerate(index_cpu.tolist()):
        if slot < 0:
            continue
        if slot >= block_num * block_size:
            raise ValueError(f"cache index {slot} is out of range [0, {block_num * block_size})")
        out[slot // block_size, :, slot % block_size, :] = nz_src[token_idx]
    return out


def qkv_rms_norm_rope_cache_golden(  # pylint: disable=huawei-too-many-arguments
    qkv: torch.Tensor,
    q_gamma: torch.Tensor,
    k_gamma: torch.Tensor,
    *args,
    **kwargs,
):
    if cache_mode != "PA_NZ":
        raise ValueError(f"only PA_NZ cache_mode is supported, got {cache_mode}")
    result = _check_qkv_size(qkv_size, head_nums)
    batch = result.batch
    seq = result.seq
    dim = result.dim
    num_q = result.num_q
    num_k = result.num_k
    num_v = result.num_v
    tokens = batch * seq
    if qkv.shape != (tokens, (num_q + num_k + num_v) * dim):
        raise ValueError(f"qkv shape mismatch: got {tuple(qkv.shape)}")

    q_end = num_q * dim
    k_end = q_end + num_k * dim
    q = qkv[:, :q_end].reshape(tokens, num_q, dim)
    k = qkv[:, q_end:k_end].reshape(tokens, num_k, dim)
    v = qkv[:, k_end:].reshape(tokens, num_v, dim)

    q_norm = rms_norm_torch(q, q_gamma, runtime.epsilon)
    k_norm = rms_norm_torch(k, k_gamma, runtime.epsilon)
    q_rope = rope_torch(q_norm, cos.reshape(tokens, dim), sin.reshape(tokens, dim))
    k_rope = rope_torch(k_norm, cos.reshape(tokens, dim), sin.reshape(tokens, dim))

    q_new = q_rope.reshape(tokens, num_q * dim).to(q_out.dtype)
    if k_cache.dtype != importlib.import_module("torch").int8 or v_cache.dtype != importlib.import_module("torch").int8:
        raise ValueError("current reference only supports int8 k_cache/v_cache")
    if runtime.k_scale is None or runtime.v_scale is None:
        raise ValueError("k_scale and v_scale are required when k_cache/v_cache are int8")
    if runtime.k_offset is not None or runtime.v_offset is not None:
        raise ValueError("current reference only supports symmetric quantization")
    k_src = quant_to_int8_torch(k_rope, k_scale, None)
    v_src = quant_to_int8_torch(v, v_scale, None)
    k_new = scatter_pa_nz_torch(k_cache, index, k_src)
    v_new = scatter_pa_nz_torch(v_cache, index, v_src)

    if is_output_qkv:
        return QkvNormRopeCacheOutput(q_new, k_new, v_new, q_new.clone(),
                                      k_rope.reshape(tokens, num_k * dim), v.reshape(tokens, num_v * dim))
    return QkvNormRopeCacheOutput(q_new, k_new, v_new, None, None, None)
