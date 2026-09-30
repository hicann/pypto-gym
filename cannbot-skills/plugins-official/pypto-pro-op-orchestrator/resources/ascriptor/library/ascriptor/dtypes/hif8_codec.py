# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Reference encode/decode helpers for Ascend HiFloat8 carrier bytes."""


import functools

import torch

from ._common import _as_uint8_codes

HIF8_POSITIVE_ZERO = 0x00
HIF8_NAN = 0x80
HIF8_POSITIVE_INF = 0x6F
HIF8_NEGATIVE_INF = 0xEF
HIF8_MAX_POSITIVE_NORMAL = 0x6E
HIF8_MAX_NEGATIVE_NORMAL = 0xEE

HIF8_MAX_FINITE_VALUE = 2.0 ** 15
HIF8_OVERFLOW_THRESHOLD = HIF8_MAX_FINITE_VALUE * 1.25
_FP32_FRACTION_BITS = 23
_FP16_FRACTION_BITS = 10
_SSR_RESERVED_BITS = 14
_SSR_T14_MASK = (1 << _SSR_RESERVED_BITS) - 1
_SSR_F14_OFFSET = 1 << (_SSR_RESERVED_BITS - 1)
_HYBRID_ROUND_EXP_THRESHOLD = 4


def _pow2(exponent: torch.Tensor) -> torch.Tensor:
    return torch.pow(torch.full_like(exponent, 2.0, dtype=torch.float32), exponent.to(torch.float32))


def _decode_normal_payload(payload: torch.Tensor, d: int, mantissa_bits: int) -> torch.Tensor:
    mantissa_mask = (1 << mantissa_bits) - 1
    mantissa = (payload & mantissa_mask).to(torch.float32)
    significand = 1.0 + mantissa / float(1 << mantissa_bits)
    if d == 0:
        return significand

    exponent_payload = payload >> mantissa_bits
    se = (exponent_payload >> (d - 1)) & 1
    mag_tail_mask = (1 << (d - 1)) - 1
    magnitude = (1 << (d - 1)) | (exponent_payload & mag_tail_mask)
    exponent = torch.where(se == 0, magnitude, -magnitude).to(torch.float32)
    return _pow2(exponent) * significand


def hif8_to_fp32(codes: torch.Tensor) -> torch.Tensor:
    """Decode Ascend HiFloat8 bit patterns to float32 values.

    HiFloat8 has no negative zero: 0x00 is zero and 0x80 is NaN.
    The two largest absolute normal bit patterns, 0x6f and 0xef, decode to
    positive and negative infinity. A 256-entry table built once per device
    by :func:`_decode_hif8` (the simulator decodes per vector op).
    """
    if not isinstance(codes, torch.Tensor):
        codes = torch.as_tensor(codes)
    codes = _as_uint8_codes(codes)
    return _decode_table(str(codes.device))[codes.to(torch.long)]


@functools.cache
def _decode_table(device: str) -> torch.Tensor:
    return _decode_hif8(torch.arange(256, dtype=torch.uint8, device=torch.device(device)))


def _decode_hif8(codes: torch.Tensor) -> torch.Tensor:
    """The bit-level decoder (see :func:`hif8_to_fp32`)."""
    bits = _as_uint8_codes(codes).to(torch.int16)
    payload = bits & 0x7F
    sign = (bits & 0x80) != 0
    out = torch.zeros(bits.shape, dtype=torch.float32, device=bits.device)

    nan_mask = bits == HIF8_NAN
    if bool(nan_mask.any().item()):
        out[nan_mask] = float("nan")

    inf_mask = payload == HIF8_POSITIVE_INF
    if bool(inf_mask.any().item()):
        out[inf_mask] = torch.where(sign[inf_mask], -torch.inf, torch.inf)

    dml_mask = (payload <= 0x07) & ~nan_mask
    if bool(dml_mask.any().item()):
        mantissa = payload[dml_mask]
        nonzero = mantissa != 0
        if bool(nonzero.any().item()):
            values = _pow2(mantissa[nonzero].to(torch.float32) - 23.0)
            # Shape-agnostic scatter (the flat-index form assumed 1-D `codes` and broke on the
            # [rows, cols] carriers the simulator's L0 hif8 decode passes in): mark the (dml AND
            # nonzero-mantissa) positions in a full-shape mask, then assign by boolean mask like
            # every other branch here.
            dml_nonzero_mask = torch.zeros_like(dml_mask)
            dml_nonzero_mask[dml_mask] = nonzero
            out[dml_nonzero_mask] = values

    layouts: tuple[tuple[int, int, int, int], ...] = (
        (0, 0x78, 0x08, 3),
        (1, 0x70, 0x10, 3),
        (2, 0x60, 0x20, 3),
        (3, 0x60, 0x40, 2),
        (4, 0x60, 0x60, 1),
    )
    for d, prefix_mask, prefix_value, mantissa_bits in layouts:
        mask = ((payload & prefix_mask) == prefix_value) & ~inf_mask
        if bool(mask.any().item()):
            out[mask] = _decode_normal_payload(payload[mask], d, mantissa_bits)

    negative_finite = sign & torch.isfinite(out) & (out != 0.0)
    if bool(negative_finite.any().item()):
        out[negative_finite] = -out[negative_finite]
    return out


def _positive_finite_codebook(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """The finite positive codes sorted by value (built once per device: every encode searches it)."""
    return _positive_codebook(str(device))


@functools.cache
def _positive_codebook(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    codes = torch.arange(128, dtype=torch.uint8, device=torch.device(device))
    values = hif8_to_fp32(codes)
    finite = torch.isfinite(values)
    codes = codes[finite]
    values = values[finite]
    order = torch.argsort(values)
    return values[order], codes[order]


def _round_mode_name(round_mode: object) -> str:
    raw = getattr(round_mode, "name", round_mode)
    if raw is None:
        return "CAST_ROUND"
    text = str(raw).strip()
    upper = text.upper()
    lower = text.lower()
    if upper == "CAST_ROUND" or lower in ("round", "ta", "away_from_zero"):
        return "CAST_ROUND"
    if upper == "CAST_HYBRID" or lower == "hybrid":
        return "CAST_HYBRID"
    raise ValueError("hif8 conversion supports round_mode CAST_ROUND or CAST_HYBRID, got: " + text)


def _hif8_layout_for_exponent(exponent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dot = torch.full_like(exponent, -1, dtype=torch.int64)
    exp_bits = torch.full_like(exponent, 3, dtype=torch.int64)
    frac_bits = torch.zeros_like(exponent, dtype=torch.int64)
    abs_exp = torch.abs(exponent)

    dml = (-22 <= exponent) & (exponent < -15)
    dot = torch.where(dml, torch.zeros_like(dot), dot)

    d0 = exponent == 0
    dot = torch.where(d0, torch.full_like(dot, 1), dot)
    exp_bits = torch.where(d0, torch.zeros_like(exp_bits), exp_bits)
    frac_bits = torch.where(d0, torch.full_like(frac_bits, 3), frac_bits)

    d1 = abs_exp == 1
    dot = torch.where(d1, torch.full_like(dot, 2), dot)
    exp_bits = torch.where(d1, torch.full_like(exp_bits, 1), exp_bits)
    frac_bits = torch.where(d1, torch.full_like(frac_bits, 3), frac_bits)

    d2 = (2 <= abs_exp) & (abs_exp <= 3)
    dot = torch.where(d2, torch.full_like(dot, 4), dot)
    exp_bits = torch.where(d2, torch.full_like(exp_bits, 2), exp_bits)
    frac_bits = torch.where(d2, torch.full_like(frac_bits, 3), frac_bits)

    d3 = (4 <= abs_exp) & (abs_exp <= 7)
    dot = torch.where(d3, torch.full_like(dot, 8), dot)
    exp_bits = torch.where(d3, torch.full_like(exp_bits, 3), exp_bits)
    frac_bits = torch.where(d3, torch.full_like(frac_bits, 2), frac_bits)

    d4 = (8 <= abs_exp) & (abs_exp <= 15)
    dot = torch.where(d4, torch.full_like(dot, 12), dot)
    exp_bits = torch.where(d4, torch.full_like(exp_bits, 4), exp_bits)
    frac_bits = torch.where(d4, torch.full_like(frac_bits, 1), frac_bits)
    return dot, exp_bits, frac_bits


def _scaled_fraction(magnitude: torch.Tensor, exponent: torch.Tensor, source_fraction_bits: int) -> torch.Tensor:
    work = magnitude.to(torch.float64)
    scale = torch.pow(torch.full_like(work, 2.0), -exponent.to(torch.float64))
    one = float(1 << source_fraction_bits)
    return (work * one * scale - one).to(torch.int64)


def _assemble_positive_hif8_code(
    exponent: torch.Tensor,
    hif8_value: torch.Tensor,
    round_up: torch.Tensor,
    frac_bits: torch.Tensor,
) -> torch.Tensor:
    max_frac = (torch.ones_like(frac_bits) << frac_bits) - 1
    carry = round_up & (hif8_value == max_frac)
    hif8_frac = torch.where(round_up & ~carry, hif8_value + 1, hif8_value)
    hif8_frac = torch.where(carry, torch.zeros_like(hif8_frac), hif8_frac)

    exponent = torch.where(carry, exponent + 1, exponent)
    dot, exp_bits, frac_bits = _hif8_layout_for_exponent(exponent)

    out = torch.zeros(exponent.shape, dtype=torch.uint8, device=exponent.device)
    dml = dot == 0
    dml_code = (exponent + 23).to(torch.int64)
    out = torch.where(dml & (exponent > -23), dml_code.to(torch.uint8), out)

    d0 = dot == 1
    d0_code = ((dot << 3) + hif8_frac).to(torch.uint8)
    out = torch.where(d0, d0_code, out)

    normal = dot > 1
    abs_exponent = torch.abs(exponent)
    exponent_tail = abs_exponent - (torch.ones_like(exp_bits) << (exp_bits - 1))
    exponent_code = exponent_tail << frac_bits
    sign_exp = torch.where(exponent < 0, torch.ones_like(exponent), torch.zeros_like(exponent))
    sign_exp = sign_exp << (exp_bits - 1 + frac_bits)
    normal_code = ((dot << 3) + sign_exp + exponent_code + hif8_frac).to(torch.uint8)
    return torch.where(normal, normal_code, out)


def _encode_positive_ssr(magnitude: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
    _, _, frac_bits = _hif8_layout_for_exponent(exponent)
    fraction = _scaled_fraction(magnitude, exponent, _FP32_FRACTION_BITS)

    shift = _FP32_FRACTION_BITS - frac_bits
    hif8_value = fraction >> shift
    remainder = fraction - (hif8_value << shift)
    f14_values = remainder >> (shift - _SSR_RESERVED_BITS)
    t14_values = remainder & _SSR_T14_MASK

    special_dml_midpoint = exponent == -23
    f14_values = torch.where(special_dml_midpoint, (fraction >> 10) + _SSR_F14_OFFSET, f14_values)
    t14_values = torch.where(special_dml_midpoint, fraction & _SSR_T14_MASK, t14_values)
    hif8_value = torch.where(special_dml_midpoint, torch.zeros_like(hif8_value), hif8_value)

    return _assemble_positive_hif8_code(exponent, hif8_value, f14_values >= t14_values, frac_bits)


def _encode_positive_ssr_fp16(
    magnitude: torch.Tensor,
    exponent: torch.Tensor,
    mantissa_lsb: torch.Tensor,
) -> torch.Tensor:
    _, _, frac_bits = _hif8_layout_for_exponent(exponent)
    fraction = _scaled_fraction(magnitude, exponent, _FP16_FRACTION_BITS)

    shift = _FP16_FRACTION_BITS - frac_bits
    hif8_value = fraction >> shift
    remainder = fraction - (hif8_value << shift)
    f2_values = remainder >> (shift - 2)

    special_dml_midpoint = exponent == -23
    f2_values = torch.where(special_dml_midpoint, 2 + (fraction >> 9), f2_values)
    hif8_value = torch.where(special_dml_midpoint, torch.zeros_like(hif8_value), hif8_value)

    t2_values = mantissa_lsb.to(torch.int64) * 2 + 1
    return _assemble_positive_hif8_code(exponent, hif8_value, f2_values >= t2_values, frac_bits)


def _encode_positive_hybrid(magnitude: torch.Tensor, saturate: bool, ssr_encoder, *ssr_args) -> torch.Tensor:
    rounded_pos = _encode_positive_round(magnitude, saturate)
    finite_nonzero = magnitude != 0
    if not bool(finite_nonzero.any().item()):
        return rounded_pos

    exponent = torch.zeros(magnitude.shape, dtype=torch.int64, device=magnitude.device)
    # log2 can round a power-of-two predecessor to an integer; frexp gives its exact binary exponent.
    _, binary_exponent = torch.frexp(magnitude[finite_nonzero].to(torch.float32))
    exponent[finite_nonzero] = binary_exponent.to(torch.int64) - 1
    use_ssr = (
        finite_nonzero
        & (torch.abs(exponent) >= _HYBRID_ROUND_EXP_THRESHOLD)
        & (magnitude < HIF8_OVERFLOW_THRESHOLD)
    )
    if bool(use_ssr.any().item()):
        selected_args = [arg[use_ssr] for arg in ssr_args]
        rounded_pos[use_ssr] = ssr_encoder(magnitude[use_ssr], exponent[use_ssr], *selected_args)
    return rounded_pos


def _encode_positive_round(magnitude: torch.Tensor, saturate: bool) -> torch.Tensor:
    pos_values, pos_codes = _positive_finite_codebook(magnitude.device)
    hi = torch.searchsorted(pos_values, magnitude)
    max_idx = int(pos_values.numel()) - 1
    hi_clamped = torch.clamp(hi, max=max_idx)
    lo_clamped = torch.clamp(hi - 1, min=0)

    dist_hi = torch.abs(pos_values[hi_clamped] - magnitude)
    dist_lo = torch.abs(pos_values[lo_clamped] - magnitude)
    rounded_pos = torch.where(dist_hi <= dist_lo, pos_codes[hi_clamped], pos_codes[lo_clamped])

    overflow_mask = magnitude >= HIF8_OVERFLOW_THRESHOLD
    overflow_code = HIF8_MAX_POSITIVE_NORMAL if saturate else HIF8_POSITIVE_INF
    return torch.where(
        overflow_mask,
        torch.full_like(rounded_pos, overflow_code),
        rounded_pos,
    )


def _apply_hif8_sign_and_specials(
    original: torch.Tensor,
    rounded_pos: torch.Tensor,
    finite_mask: torch.Tensor,
) -> torch.Tensor:
    finite_x = original[finite_mask]
    negative = finite_x < 0
    negative_codes = (rounded_pos.to(torch.int16) | 0x80).to(torch.uint8)
    return torch.where(negative & (rounded_pos != 0), negative_codes, rounded_pos)


def _encode_hif8_specials(
    x: torch.Tensor,
    saturate: bool,
    nan_to_zero: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty(x.shape, dtype=torch.uint8, device=x.device)

    nan_mask = torch.isnan(x)
    if bool(nan_mask.any().item()):
        out[nan_mask] = HIF8_POSITIVE_ZERO if nan_to_zero else HIF8_NAN

    pos_inf = x == torch.inf
    if bool(pos_inf.any().item()):
        out[pos_inf] = HIF8_MAX_POSITIVE_NORMAL if saturate else HIF8_POSITIVE_INF

    neg_inf = x == -torch.inf
    if bool(neg_inf.any().item()):
        out[neg_inf] = HIF8_MAX_NEGATIVE_NORMAL if saturate else HIF8_NEGATIVE_INF

    finite_mask = torch.isfinite(x)
    return out, finite_mask


def fp32_to_hif8(
    values: torch.Tensor,
    saturate: bool = False,
    nan_to_zero: bool = False,
    round_mode: object = None,
) -> torch.Tensor:
    """Encode float32 values as Ascend HiFloat8 uint8 bit patterns.

    By default, finite values use the TA conversion mode (round to nearest with
    ties away from zero). ``round_mode=RoundMode.HYBRID`` or
    ``round_mode="hybrid"`` switches to TA for exponents with ``abs(e) < 4``
    and SSR elsewhere. Overflow maps to infinity by default; set
    ``saturate=True`` to clamp overflow to the largest finite HiF8 value.
    """
    mode_name = _round_mode_name(round_mode)
    if not isinstance(values, torch.Tensor):
        values = torch.as_tensor(values)
    x = values.to(torch.float32)
    out, finite_mask = _encode_hif8_specials(x, saturate, nan_to_zero)
    if not bool(finite_mask.any().item()):
        return out

    magnitude = x[finite_mask].abs()
    if mode_name == "CAST_HYBRID":
        rounded_pos = _encode_positive_hybrid(magnitude, saturate, _encode_positive_ssr)
    else:
        rounded_pos = _encode_positive_round(magnitude, saturate)

    out[finite_mask] = _apply_hif8_sign_and_specials(x, rounded_pos, finite_mask)
    return out


def fp16_to_hif8(
    values: torch.Tensor,
    saturate: bool = False,
    nan_to_zero: bool = False,
    round_mode: object = None,
) -> torch.Tensor:
    """Encode float16 values as Ascend HiFloat8 uint8 bit patterns.

    ``CAST_ROUND`` follows the same TA nearest-away hif8 codebook rounding as
    the float32 helper. ``CAST_HYBRID`` uses TA when ``abs(e) < 4`` and a
    half-specific SSR path elsewhere; the implementation follows CANNSIM rather
    than the buggy boundary cases in the half conversion script.
    """
    mode_name = _round_mode_name(round_mode)
    if not isinstance(values, torch.Tensor):
        values = torch.as_tensor(values)
    x_half = values.to(torch.float16)
    out, finite_mask = _encode_hif8_specials(x_half, saturate, nan_to_zero)
    if not bool(finite_mask.any().item()):
        return out

    finite_half = x_half[finite_mask]
    magnitude = finite_half.abs()
    if mode_name == "CAST_HYBRID":
        mantissa_lsb = (finite_half.view(torch.int16).to(torch.int32) & 1).to(torch.int64)
        rounded_pos = _encode_positive_hybrid(magnitude, saturate, _encode_positive_ssr_fp16, mantissa_lsb)
    else:
        rounded_pos = _encode_positive_round(magnitude, saturate)

    out[finite_mask] = _apply_hif8_sign_and_specials(x_half, rounded_pos, finite_mask)
    return out


hifloat8_to_fp32 = hif8_to_fp32
fp32_to_hifloat8 = fp32_to_hif8
fp16_to_hifloat8 = fp16_to_hif8


if __name__ == "__main__":
    def _hex_codes(codes: torch.Tensor) -> list:
        return [f"0x{int(code):02x}" for code in codes.cpu().flatten()]

    def _values(values: torch.Tensor) -> list:
        return [float(value) for value in values.cpu().flatten()]

    def _show_encode(title: str, values: torch.Tensor, codes: torch.Tensor) -> None:
        print(title)
        print("  input  :", _values(values))
        print("  hif8   :", _hex_codes(codes))
        print("  decoded:", _values(hif8_to_fp32(codes)))

    rng = torch.Generator().manual_seed(20240521)
    fp32_random = torch.where(
        torch.rand(8, generator=rng) < 0.5,
        -torch.ones(8, dtype=torch.float32),
        torch.ones(8, dtype=torch.float32),
    )
    fp32_random *= torch.pow(torch.full((8,), 2.0), torch.rand(8, generator=rng) * 37.0 - 22.0)

    fp32_values = torch.tensor(
        [1.0, 8.0, 16.0, 20.0, 2.0 ** -22, 2.0 ** -16, 32768.0, -16.0],
        dtype=torch.float32,
    )
    fp32_values = torch.cat([fp32_values, fp32_random])
    _show_encode("fp32 -> hif8, CAST_ROUND", fp32_values, fp32_to_hif8(fp32_values))
    _show_encode("fp32 -> hif8, CAST_HYBRID", fp32_values, fp32_to_hif8(fp32_values, round_mode="hybrid"))

    fp16_random = torch.where(
        torch.rand(8, generator=rng) < 0.5,
        -torch.ones(8, dtype=torch.float16),
        torch.ones(8, dtype=torch.float16),
    )
    fp16_random *= torch.pow(torch.full((8,), 2.0, dtype=torch.float16), torch.rand(8, generator=rng) * 24.0 - 22.0)

    fp16_values = torch.tensor(
        [5.960464477539062e-07, 7.748603820800781e-07, 1.1920928955078125e-06, 16.0, -16.0],
        dtype=torch.float16,
    )
    fp16_values = torch.cat([fp16_values, fp16_random])
    _show_encode("fp16 -> hif8, CAST_ROUND", fp16_values, fp16_to_hif8(fp16_values))
    _show_encode("fp16 -> hif8, CAST_HYBRID", fp16_values, fp16_to_hif8(fp16_values, round_mode="hybrid"))

    raw_codes = torch.tensor([0x00, 0x08, 0x41, 0x6F, 0x80, 0xC1], dtype=torch.uint8)
    print("hif8 -> fp32")
    print("  hif8  :", _hex_codes(raw_codes))
    print("  fp32  :", _values(hif8_to_fp32(raw_codes)))
