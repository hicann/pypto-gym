# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from collections.abc import Sequence

import torch

from ._common import _require_integer_tensor

FP4_E2M1_MAX_VALUE = 6.0
FP4_E1M2_MAX_VALUE = 1.75


def _normalize_axis(axis: int, ndim: int) -> int:
    if ndim <= 0:
        raise ValueError("FP4 tensors must have at least one dimension")
    normalized = int(axis)
    if normalized < 0:
        normalized += ndim
    if normalized < 0 or normalized >= ndim:
        raise ValueError(f"pack_axis out of range for ndim={ndim}: {axis}")
    return normalized


def _as_uint8_carrier(carrier: torch.Tensor) -> torch.Tensor:
    _require_integer_tensor(carrier, "carrier")
    if carrier.dtype != torch.uint8 and int(carrier.numel()) > 0:
        min_code = int(carrier.min().item())
        max_code = int(carrier.max().item())
        if min_code < 0 or max_code > 0xFF:
            raise ValueError(f"carrier values must be in [0, 255], got min={min_code}, max={max_code}")
    return carrier.to(torch.uint8)


def _logical_shape_tuple(logical_shape: Sequence[int] | None) -> tuple[int, ...] | None:
    if logical_shape is None:
        return None
    out = tuple(int(dim) for dim in logical_shape)
    if len(out) == 0:
        raise ValueError("logical_shape must have at least one dimension")
    if any(dim < 0 for dim in out):
        raise ValueError(f"logical_shape dimensions must be non-negative, got: {out}")
    return out


def _pack_nibbles_to_uint8(nibbles: torch.Tensor, pack_axis: int) -> torch.Tensor:
    if not isinstance(nibbles, torch.Tensor):
        nibbles = torch.as_tensor(nibbles)
    axis = _normalize_axis(pack_axis, nibbles.dim())
    nibbles_u8 = nibbles.to(torch.uint8) & 0x0F

    if nibbles_u8.shape[axis] % 2 != 0:
        pad_shape = list(nibbles_u8.shape)
        pad_shape[axis] = 1
        pad = torch.zeros(pad_shape, dtype=torch.uint8, device=nibbles_u8.device)
        nibbles_u8 = torch.cat([nibbles_u8, pad], dim=axis)

    moved = torch.movedim(nibbles_u8, axis, -1)
    low = moved[..., 0::2]
    high = moved[..., 1::2]
    packed = (low | (high << 4)).contiguous()
    return torch.movedim(packed, -1, axis).contiguous()


def _infer_unpack_shape(carrier: torch.Tensor, logical_shape: Sequence[int] | None, pack_axis: int) -> tuple[tuple[int, ...], int]:
    shape = _logical_shape_tuple(logical_shape)
    if shape is None:
        axis = _normalize_axis(pack_axis, carrier.dim())
        inferred = list(carrier.shape)
        inferred[axis] *= 2
        return tuple(inferred), axis

    axis = _normalize_axis(pack_axis, len(shape))
    expected_carrier_shape = list(shape)
    expected_carrier_shape[axis] = (expected_carrier_shape[axis] + 1) // 2
    if carrier.dim() != len(shape):
        raise ValueError(
            f"carrier ndim does not match logical_shape: carrier ndim={carrier.dim()}, "
            f"logical ndim={len(shape)}"
        )
    for dim, (actual, expected) in enumerate(zip(carrier.shape, expected_carrier_shape, strict=True)):
        if int(actual) != int(expected):
            raise ValueError(
                f"carrier shape mismatch at dim {dim}: expected {tuple(expected_carrier_shape)} "
                f"for logical_shape={shape}, got {tuple(carrier.shape)}"
            )
    return shape, axis


def _unpack_uint8_to_nibbles(
    carrier: torch.Tensor,
    logical_shape: Sequence[int] | None,
    pack_axis: int,
) -> torch.Tensor:
    if not isinstance(carrier, torch.Tensor):
        carrier = torch.as_tensor(carrier)
    carrier_u8 = _as_uint8_carrier(carrier)
    shape, axis = _infer_unpack_shape(carrier_u8, logical_shape, pack_axis)

    moved = torch.movedim(carrier_u8, axis, -1)
    packed_width = moved.shape[-1]
    low = moved & 0x0F
    high = (moved >> 4) & 0x0F
    out = torch.empty((*moved.shape[:-1], packed_width * 2), dtype=torch.uint8, device=moved.device)
    out[..., 0::2] = low
    out[..., 1::2] = high
    out = out[..., : shape[axis]]
    return torch.movedim(out, -1, axis).contiguous()


def _fp4_table(kind: str, device: torch.device) -> torch.Tensor:
    if kind == "e2m1":
        return torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32, device=device)
    if kind == "e1m2":
        return torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75], dtype=torch.float32, device=device)
    raise ValueError(f"unknown FP4 kind: {kind}")


def _fp4_nibbles_to_fp32(nibbles: torch.Tensor, kind: str) -> torch.Tensor:
    nibbles_u8 = nibbles.to(torch.uint8) & 0x0F
    table = _fp4_table(kind, nibbles_u8.device)
    magnitude_code = (nibbles_u8 & 0x07).to(torch.long)
    sign = (nibbles_u8 & 0x08) != 0
    values = table.index_select(0, magnitude_code.reshape(-1)).reshape(nibbles_u8.shape)
    return torch.where(sign, -values, values)


def _round_mode_name(round_mode: object) -> str:
    name = str(getattr(round_mode, "name", round_mode)).strip().upper()
    aliases = {
        "RINT": "CAST_RINT",
        "TO_EVEN": "CAST_RINT",
        "ROUND": "CAST_ROUND",
        "AWAY_FROM_ZERO": "CAST_ROUND",
        "FLOOR": "CAST_FLOOR",
        "CEIL": "CAST_CEIL",
        "TRUNC": "CAST_TRUNC",
    }
    name = aliases.get(name, name)
    supported = {"CAST_RINT", "CAST_ROUND", "CAST_FLOOR", "CAST_CEIL", "CAST_TRUNC"}
    if name not in supported:
        raise ValueError(f"unsupported FP4 round_mode: {round_mode}")
    return name


def _fp32_to_fp4_nibbles(
    values: torch.Tensor,
    kind: str,
    nan_to_zero: bool,
    round_mode: object,
) -> torch.Tensor:
    if not isinstance(values, torch.Tensor):
        values = torch.as_tensor(values)
    x = values.to(torch.float32)
    if not nan_to_zero and bool(torch.isnan(x).any().item()):
        raise ValueError("FP4 has no NaN payload; pass nan_to_zero=True to encode NaN as zero")
    if nan_to_zero:
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)

    table = _fp4_table(kind, x.device)
    magnitude = x.abs()
    hi = torch.searchsorted(table, magnitude)
    max_idx = int(table.numel()) - 1
    hi_clamped = torch.clamp(hi, max=max_idx)
    lo_clamped = torch.clamp(hi - 1, min=0)

    dist_hi = torch.abs(table[hi_clamped] - magnitude)
    dist_lo = torch.abs(table[lo_clamped] - magnitude)
    exact = dist_hi == 0
    mode = _round_mode_name(round_mode)
    if mode == "CAST_RINT":
        tie_to_hi = (lo_clamped % 2) != 0
        rounded = torch.where(
            dist_hi < dist_lo,
            hi_clamped,
            torch.where(dist_hi > dist_lo, lo_clamped, torch.where(tie_to_hi, hi_clamped, lo_clamped)),
        )
    elif mode == "CAST_ROUND":
        rounded = torch.where(dist_hi <= dist_lo, hi_clamped, lo_clamped)
    elif mode == "CAST_FLOOR":
        rounded = torch.where(exact, hi_clamped, torch.where(x < 0, hi_clamped, lo_clamped))
    elif mode == "CAST_CEIL":
        rounded = torch.where(exact, hi_clamped, torch.where(x < 0, lo_clamped, hi_clamped))
    else:  # CAST_TRUNC
        rounded = torch.where(exact, hi_clamped, lo_clamped)
    rounded = rounded.to(torch.uint8)

    negative = torch.signbit(x)
    signed = (rounded | 0x08).to(torch.uint8)
    # Preserve the finite source sign, including negative zero and underflow
    # to zero, yielding the negative-zero payload 0x8 (M10-041).
    return torch.where(negative, signed, rounded)


def fp32_to_fp4_e2m1(
    values: torch.Tensor,
    pack_axis: int = -1,
    nan_to_zero: bool = False,
    round_mode: object = "CAST_ROUND",
) -> torch.Tensor:
    """Encode float32 values to FP4 E2M1 uint8 carriers.

    Two FP4 payloads are packed in each uint8, low nibble first along
    ``pack_axis``. ``round_mode`` accepts the five CANN BF16-to-FP4 modes and
    defaults to ``CAST_ROUND`` (nearest, ties away from zero).
    """
    nibbles = _fp32_to_fp4_nibbles(
        values, "e2m1", nan_to_zero=nan_to_zero, round_mode=round_mode
    )
    return _pack_nibbles_to_uint8(nibbles, pack_axis)


def fp4_e2m1_to_fp32(
    carrier: torch.Tensor,
    logical_shape: Sequence[int] | None = None,
    pack_axis: int = -1,
) -> torch.Tensor:
    """Decode FP4 E2M1 uint8 carriers to float32 values."""
    nibbles = _unpack_uint8_to_nibbles(carrier, logical_shape=logical_shape, pack_axis=pack_axis)
    return _fp4_nibbles_to_fp32(nibbles, "e2m1")


def fp32_to_fp4_e1m2(
    values: torch.Tensor,
    pack_axis: int = -1,
    nan_to_zero: bool = False,
    round_mode: object = "CAST_ROUND",
) -> torch.Tensor:
    """Encode float32 values to FP4 E1M2 uint8 carriers.

    Two FP4 payloads are packed in each uint8, low nibble first along
    ``pack_axis``. ``round_mode`` accepts the five CANN BF16-to-FP4 modes and
    defaults to ``CAST_ROUND`` (nearest, ties away from zero).
    """
    nibbles = _fp32_to_fp4_nibbles(
        values, "e1m2", nan_to_zero=nan_to_zero, round_mode=round_mode
    )
    return _pack_nibbles_to_uint8(nibbles, pack_axis)


def fp4_e1m2_to_fp32(
    carrier: torch.Tensor,
    logical_shape: Sequence[int] | None = None,
    pack_axis: int = -1,
) -> torch.Tensor:
    """Decode FP4 E1M2 uint8 carriers to float32 values."""
    nibbles = _unpack_uint8_to_nibbles(carrier, logical_shape=logical_shape, pack_axis=pack_axis)
    return _fp4_nibbles_to_fp32(nibbles, "e1m2")
