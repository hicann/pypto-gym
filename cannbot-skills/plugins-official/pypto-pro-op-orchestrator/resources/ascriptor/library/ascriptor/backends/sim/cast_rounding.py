# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Rounding of the a2 family's ``vec.cast`` (RFC-0008 §4): the old simulator's ``cast_rounding.py``, mode names as the
IR spells them (``none`` / ``rint`` / ``round`` / ``floor`` / ``ceil`` / ``trunc`` / ``odd``). The goldens recorded from the
old simulator carry exactly these numerics, so the port is literal: float -> float rounds by the mode with the
nearest-away tie rule and the odd rule on fp16, float -> int rounds in fp32 then saturates, int -> float rounds the
exact integer by the mode (int64 element by element through fractions), a narrowing int -> int cast saturates."""

from __future__ import annotations

from fractions import Fraction

import torch


def apply_cast_round(values: torch.Tensor, mode: str) -> torch.Tensor:
    if mode in ("none", "rint"):
        return torch.round(values)
    if mode == "floor":
        return torch.floor(values)
    if mode == "ceil":
        return torch.ceil(values)
    if mode == "round":
        return torch.sign(values) * torch.floor(torch.abs(values) + 0.5)
    if mode == "trunc":
        return torch.trunc(values)
    raise ValueError(f"unsupported cast round mode: {mode}")


def _nextafter(values: torch.Tensor, direction: float) -> torch.Tensor:
    return torch.nextafter(values, torch.full_like(values, direction))


def _f64(values: torch.Tensor) -> torch.Tensor:
    return values if values.dtype == torch.float64 else values.to(torch.float64)


def _round_floor(values: torch.Tensor, dst: torch.dtype) -> torch.Tensor:
    rounded = values.to(dst)
    return torch.where(rounded.to(torch.float64) > _f64(values), _nextafter(rounded, float("-inf")), rounded)


def _round_ceil(values: torch.Tensor, dst: torch.dtype) -> torch.Tensor:
    rounded = values.to(dst)
    return torch.where(rounded.to(torch.float64) < _f64(values), _nextafter(rounded, float("inf")), rounded)


def _round_trunc(values: torch.Tensor, dst: torch.dtype) -> torch.Tensor:
    rounded = values.to(dst)
    rw, work = rounded.to(torch.float64), _f64(values)
    result = torch.where((work > 0) & (rw > work), _nextafter(rounded, float("-inf")), rounded)
    return torch.where((work < 0) & (rw < work), _nextafter(rounded, float("inf")), result)


def _round_nearest_away(values: torch.Tensor, dst: torch.dtype) -> torch.Tensor:
    rounded = values.to(dst)
    lo, hi = _round_floor(values, dst), _round_ceil(values, dst)
    work, lw, hw = _f64(values), lo.to(torch.float64), hi.to(torch.float64)
    tie = torch.isfinite(work) & torch.isfinite(lw) & torch.isfinite(hw) & (lo != hi) & ((work - lw) == (hw - work))
    away = torch.where(values.to(torch.float32) >= 0, hi, lo)
    return torch.where(tie, away, rounded)


def _round_odd(values: torch.Tensor, dst: torch.dtype) -> torch.Tensor:
    if dst != torch.float16:
        raise ValueError("the odd rounding is only supported for float -> half")
    truncated = _round_trunc(values, dst)
    work = values.to(torch.float32)
    inexact = torch.isfinite(work) & (truncated.to(torch.float32) != work)
    bits = truncated.view(torch.int16).to(torch.int32) & 0xFFFF
    odd = torch.where(inexact, bits | 1, bits)
    signed = torch.where(odd >= 0x8000, odd - 0x10000, odd).to(torch.int16)
    return signed.view(torch.float16)


def apply_cast_float_round(values: torch.Tensor, dst: torch.dtype, mode: str) -> torch.Tensor:
    if mode in ("none", "rint"):
        return values.to(dst)
    if mode == "floor":
        return _round_floor(values, dst)
    if mode == "ceil":
        return _round_ceil(values, dst)
    if mode == "round":
        return _round_nearest_away(values, dst)
    if mode == "trunc":
        return _round_trunc(values, dst)
    if mode == "odd":
        return _round_odd(values, dst)
    raise ValueError(f"unsupported cast round mode: {mode}")


def _frac(value: torch.Tensor) -> Fraction:
    n, d = float(value.item()).as_integer_ratio()
    return Fraction(n, d)


def _int_bounds(value: int, dst: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    target = Fraction(value, 1)
    nearest = torch.tensor(value, dtype=dst)
    nf = _frac(nearest)
    if nf == target:
        return nearest, nearest
    if nf < target:
        return nearest, torch.nextafter(nearest, torch.tensor(float("inf"), dtype=dst))
    return torch.nextafter(nearest, torch.tensor(float("-inf"), dtype=dst)), nearest


def _int64_scalar_to_float(value: int, dst: torch.dtype, mode: str) -> torch.Tensor:
    if mode in ("none", "rint"):
        return torch.tensor(value, dtype=dst)
    lo, hi = _int_bounds(value, dst)
    if mode == "floor":
        return lo
    if mode == "ceil":
        return hi
    if mode == "trunc":
        return lo if value >= 0 else hi
    if mode == "round":
        if bool((lo == hi).item()):
            return lo
        target = Fraction(value, 1)
        dlo, dhi = target - _frac(lo), _frac(hi) - target
        if dlo < dhi:
            return lo
        if dhi < dlo:
            return hi
        return hi if value >= 0 else lo
    raise ValueError(f"unsupported cast round mode: {mode}")


def apply_cast_int_to_float(values: torch.Tensor, dst: torch.dtype, mode: str) -> torch.Tensor:
    if values.dtype == torch.int64:
        flat = values.reshape(-1)
        out = torch.empty(flat.shape, dtype=dst)
        for i in range(int(flat.numel())):
            out[i] = _int64_scalar_to_float(int(flat[i].item()), dst, mode)
        return out.reshape(values.shape)
    return apply_cast_float_round(values.to(torch.float64), dst, mode)


def _narrowing_int(src: torch.dtype, dst: torch.dtype) -> bool:
    if src.is_floating_point or dst.is_floating_point or src == torch.bool or dst == torch.bool:
        return False
    try:
        return torch.iinfo(dst).bits < torch.iinfo(src).bits
    except TypeError:
        return False


def _clamp_finite(work: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    finite = torch.isfinite(work)
    if bool(finite.any().item()):
        work[finite] = torch.clamp(work[finite], min=lo, max=hi)
    return work


def apply_cast_dtype(values: torch.Tensor, dst: torch.dtype, mode: str, saturate: bool = False) -> torch.Tensor:
    """``values`` converted to ``dst`` under ``mode`` (the old ``apply_cast_dtype``)."""
    work = values
    dst_float = torch.empty((), dtype=dst).is_floating_point()
    if work.dtype == torch.float32 and dst == torch.float32:
        return apply_cast_round(work, mode).to(dst)
    if work.dtype.is_floating_point and dst_float and work.dtype != dst:
        if saturate:
            info = torch.finfo(dst)
            work = _clamp_finite(work.to(torch.float64), float(info.min), float(info.max))
        return apply_cast_float_round(work, dst, mode)
    if not work.dtype.is_floating_point and dst_float:
        return apply_cast_int_to_float(work, dst, mode)
    if work.dtype.is_floating_point and not dst_float:
        work = apply_cast_round(work.to(torch.float32), mode)
    if not saturate and _narrowing_int(values.dtype, dst):  # a narrowing integer cast saturates on the hardware
        info = torch.iinfo(dst)
        return torch.clamp(work.to(torch.int64), min=int(info.min), max=int(info.max)).to(dst)
    if not saturate:
        return work.to(dst)
    if dst_float:
        info = torch.finfo(dst)
        return _clamp_finite(work.to(torch.float64), float(info.min), float(info.max)).to(dst)
    iinfo = torch.iinfo(dst)
    work = _clamp_finite(work.to(torch.float64), float(iinfo.min), float(iinfo.max))
    work[work == float("inf")] = float(iinfo.max)
    work[work == float("-inf")] = float(iinfo.min)
    work[torch.isnan(work)] = 0.0
    return work.to(dst)


__all__ = ["apply_cast_dtype", "apply_cast_round", "apply_cast_float_round", "apply_cast_int_to_float"]
