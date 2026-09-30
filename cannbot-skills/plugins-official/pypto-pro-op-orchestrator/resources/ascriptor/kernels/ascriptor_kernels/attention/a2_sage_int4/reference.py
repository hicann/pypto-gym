# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent reference for signed-int4 SAGE decode: the physical input preparation and the
same quantized arithmetic, with its own int4 packing and probability narrowing."""

import math
import torch


D_HEAD = 128
TILE_N = 512
P_SCALE = 127.0
NEG_INF = -99999.0
SOFTMAX_SCALE = 1.0 / math.sqrt(D_HEAD)


def pack_signed_int4(values: torch.Tensor) -> torch.Tensor:
    """Pack the last dimension little-nibble-first into signed int32 carriers."""
    if values.dtype != torch.int32 or values.shape[-1] % 8:
        raise ValueError("signed int4 packing requires int32 values and a multiple-of-eight width")
    if bool(((values < -8) | (values > 7)).any()):
        raise ValueError("signed int4 values must be in [-8, 7]")
    packed = torch.zeros((*values.shape[:-1], values.shape[-1] // 8), dtype=torch.int64)
    for lane in range(8):
        packed |= (values[..., lane::8].to(torch.int64) & 0xF) << (lane * 4)
    return packed.to(torch.int32).contiguous()


def unpack_signed_int4(carriers: torch.Tensor, width: int = D_HEAD) -> torch.Tensor:
    """Decode signed int32 carriers without using an ascriptor runtime codec."""
    if carriers.dtype != torch.int32 or width != carriers.shape[-1] * 8:
        raise ValueError("carrier shape does not match the requested signed-int4 width")
    unsigned = carriers.to(torch.int64) & 0xFFFFFFFF
    lanes = []
    for lane in range(8):
        nibble = (unsigned >> (lane * 4)) & 0xF
        lanes.append(torch.where(nibble >= 8, nibble - 16, nibble).to(torch.int32))
    return torch.stack(lanes, dim=-1).reshape(*carriers.shape[:-1], width).contiguous()


def _quant_int4_per_row(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = values.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8) / 7.0
    raw = torch.round(values / scale).clamp(-8, 7).to(torch.int32)
    return pack_signed_int4(raw), scale.float().contiguous()


def _quant_int4_per_tile(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    bh, sequence, _ = values.shape
    tiles = (sequence + TILE_N - 1) // TILE_N
    raw = torch.empty_like(values, dtype=torch.int32)
    scales = torch.empty((bh, tiles), dtype=torch.float32)
    for tile in range(tiles):
        first = tile * TILE_N
        stop = min(first + TILE_N, sequence)
        block = values[:, first:stop]
        scale = block.abs().amax(dim=(-1, -2)).clamp_min(1.0e-8) / 7.0
        scales[:, tile] = scale
        raw[:, first:stop] = torch.round(block / scale[:, None, None]).clamp(-8, 7).to(torch.int32)
    return pack_signed_int4(raw), scales.contiguous()


def _quant_int8_per_channel(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = values.abs().amax(dim=1, keepdim=True).clamp_min(1.0e-8) / P_SCALE
    raw = torch.round(values / scale).clamp(-128, 127).to(torch.int8)
    return raw.contiguous(), scale.float().contiguous()


def generate_inputs(*, bh: int, sequence: int, seed: int, pattern: str) -> dict:
    """Generate physical ABI tensors, including explicit signed-carrier controls."""
    generator = torch.Generator().manual_seed(seed)
    tiles = (sequence + TILE_N - 1) // TILE_N
    if pattern == "source":
        # Preserve the old B1/H32/Sq1 preparation order. With one query row per
        # head, q_smooth is zero and qm carries that row; signed controls below
        # separately exercise nonzero negative Q/K nibbles.
        q_float = torch.randn((1, bh, 1, D_HEAD), generator=generator)
        k_float = torch.randn((1, bh, sequence, D_HEAD), generator=generator)
        v_float = torch.randn((bh, sequence, D_HEAD), generator=generator)
        qm_float = q_float.mean(dim=(0, 2), keepdim=True)
        q_smooth = (q_float - qm_float).reshape(bh, 1, D_HEAD)
        k_smooth_float = k_float - k_float.mean(dim=(0, 1, 2), keepdim=True)
        k_smooth = k_smooth_float.reshape(bh, sequence, D_HEAD)
        q, scale_q = _quant_int4_per_row(q_smooth)
        k, scale_k = _quant_int4_per_tile(k_smooth)
        v, scale_v = _quant_int8_per_channel(v_float)
        qm = qm_float.reshape(bh, 1, D_HEAD).to(torch.float16).contiguous()
        k_smooth = k_smooth.to(torch.float16).contiguous()
    elif pattern == "signed":
        q_raw = torch.randint(-8, 8, (bh, 1, D_HEAD), generator=generator, dtype=torch.int32)
        k_raw = torch.randint(-8, 8, (bh, sequence, D_HEAD), generator=generator, dtype=torch.int32)
        # Force both signs and a negative high nibble in every carrier family.
        q_raw[..., 0::8] = 7
        q_raw[..., 7::8] = -8
        k_raw[..., 0::8] = 7
        k_raw[..., 7::8] = -8
        q = pack_signed_int4(q_raw)
        k = pack_signed_int4(k_raw)
        scale_q = (torch.rand((bh, 1, 1), generator=generator) * 0.04 + 0.02).float()
        scale_k = (torch.rand((bh, tiles), generator=generator) * 0.04 + 0.02).float()
        v = torch.randint(-127, 128, (bh, sequence, D_HEAD), generator=generator, dtype=torch.int16).to(torch.int8)
        scale_v = (torch.rand((bh, 1, D_HEAD), generator=generator) * 0.02 + 0.005).float()
        qm = (torch.randn((bh, 1, D_HEAD), generator=generator) * 0.25).to(torch.float16)
        k_smooth = (torch.randn((bh, sequence, D_HEAD), generator=generator) * 0.5).to(torch.float16)
    else:
        raise ValueError("pattern must be signed or source")
    return {
        "q": q.contiguous(),
        "k": k.contiguous(),
        "v": v.contiguous(),
        "scale_q": scale_q.contiguous(),
        "scale_k": scale_k.contiguous(),
        "scale_v": scale_v.contiguous(),
        "qm": qm.contiguous(),
        "k_smooth": k_smooth.contiguous(),
    }


def _positive_probability_to_int8(values: torch.Tensor) -> torch.Tensor:
    half = values.to(torch.float16).float()
    return torch.floor(half + 0.5).clamp(-128, 127).to(torch.int8)


def attention(inputs: dict) -> dict:
    """Execute the source physical arithmetic without DSL/simulator imports."""
    q_i4 = unpack_signed_int4(inputs["q"]).float()
    k_i4 = unpack_signed_int4(inputs["k"]).float()
    qm = inputs["qm"].float()
    k_smooth = inputs["k_smooth"].float()
    v = inputs["v"].float()
    bh, _, _ = q_i4.shape
    sequence = k_i4.shape[1]
    row_max = torch.full((bh, 1), NEG_INF, dtype=torch.float32)
    row_sum = torch.zeros((bh, 1), dtype=torch.float32)
    accumulator = torch.zeros((bh, 1, D_HEAD), dtype=torch.float32)

    for first in range(0, sequence, TILE_N):
        stop = min(first + TILE_N, sequence)
        tile = first // TILE_N
        qk = torch.matmul(q_i4, k_i4[:, first:stop].transpose(-1, -2))
        qk_scale = inputs["scale_q"] * inputs["scale_k"][:, tile].reshape(bh, 1, 1)
        smooth = torch.matmul(qm, k_smooth[:, first:stop].transpose(-1, -2)).to(torch.float16).float()
        score = (qk * qk_scale + smooth) * SOFTMAX_SCALE
        tile_max = score.amax(dim=-1)
        next_max = torch.maximum(row_max, tile_max)
        expdiff = torch.ones_like(next_max) if first == 0 else torch.exp(row_max - next_max)
        probability = torch.exp(score - next_max[..., None])
        tile_sum = probability.sum(dim=-1)
        p_int8 = _positive_probability_to_int8(probability * P_SCALE).float()
        pv = torch.matmul(p_int8, v[:, first:stop])
        if first == 0:
            row_sum = tile_sum
            accumulator = pv
        else:
            row_sum = row_sum * expdiff + tile_sum
            accumulator = accumulator * expdiff[..., None] + pv
        row_max = next_max

    out = accumulator * (inputs["scale_v"] / P_SCALE) / row_sum[..., None]
    return {"out": out.float().contiguous(), "rowmax": row_max.float().contiguous(), "rowsum": row_sum.float().contiguous()}


def make_inputs(case):
    """Physical ABI tensors for one case, at the carrier layout the kernel reads.

    The two patterns answer different questions. `source` reproduces the original
    preparation order -- real float Q/K/V quantized per row, per 512-key tile and per V
    channel -- so the numbers are the ones a caller would actually hand over. `signed`
    draws the nibbles directly and forces a +7 and a -8 into every carrier family, which
    is the only way to be sure the sign extension out of a packed int32 is exercised at
    both ends rather than on whatever values a random float happened to produce."""
    p = case["parameters"]
    result = dict(p)
    result.update(generate_inputs(bh=p["BH"], sequence=p["S2"], seed=case["seed"],
                                  pattern=p["pattern"]))
    return result


def reference(inputs):
    return attention(inputs)
