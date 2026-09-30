# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent bit references for int4, FP4, E8M0 and HiFloat8 TA. Imports no DSL and no codec.

Every table here is built from the format's own definition -- Python integer arithmetic, explicit
exponent and fraction classes, and a sorted list of interval midpoints for the encoder. That
independence is the whole point: main.py compares the library's codecs against these, and a
reference that called the codecs would be comparing the library with itself.
"""

import math
from bisect import bisect_right
from functools import lru_cache

import torch


def int4_reference(values):
    if any(value < -8 or value > 7 for value in values):
        raise ValueError("int4 inputs must be in [-8, 7]")
    words = []
    for start in range(0, len(values), 8):
        word = sum((int(value) & 15) << (4 * index) for index, value in enumerate(values[start : start + 8]))
        words.append(word if word < 2**31 else word - 2**32)
    return words


def fp4_reference(kind, byte):
    positives = (0, 0.5, 1, 1.5, 2, 3, 4, 6) if kind == "e2m1" else (0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75)
    return [
        math.copysign(positives[nibble & 7], -1 if nibble & 8 else 1) for nibble in (byte & 15, byte >> 4)
    ]


@lru_cache(maxsize=1)
def hif8_reference_table():
    """Enumerate exponent/fraction classes without calling an execution codec.

    Adapted from the independent attention reference model, with its algorithm
    and provenance recorded in library-examples.hif8.json. This file remains
    self-contained and imports no attention unit.
    """
    values = [None] * 128
    values[0] = 0.0
    for code in range(1, 8):
        values[code] = math.ldexp(1.0, code - 23)
    classes = ((0, 0, 3, 0x08), (1, 1, 3, 0x10), (2, 3, 3, 0x20), (4, 7, 2, 0x40), (8, 15, 1, 0x60))
    for low, high, fraction_bits, prefix in classes:
        for exponent_sign in (1, -1) if low else (1,):
            offset = (high - low + 1) * (1 << fraction_bits) if exponent_sign < 0 else 0
            for magnitude in range(low, high + 1):
                for fraction in range(1 << fraction_bits):
                    code = prefix + offset + (magnitude - low) * (1 << fraction_bits) + fraction
                    values[code] = math.ldexp(1.0 + fraction / (1 << fraction_bits), exponent_sign * magnitude)
    values[0x6F] = math.inf
    assert all(value is not None for value in values)
    signed = values + [-value for value in values]
    signed[0x80] = math.nan
    return tuple(signed)


def hif8_reference_encode(values, *, saturate=False, nan_to_zero=False):
    """FP32-valued inputs, nearest with ties away; both zero signs encode as 0."""
    levels = sorted(
        (value, code) for code, value in enumerate(hif8_reference_table()[:128]) if math.isfinite(value)
    )
    boundaries = [(left[0] + right[0]) / 2 for left, right in zip(levels[:-1], levels[1:], strict=True)]
    boundaries.append(40960.0)
    codes = [code for _, code in levels] + [0x6F]
    output = []
    for value in values:
        if math.isnan(value):
            output.append(0 if nan_to_zero else 0x80)
            continue
        code = codes[bisect_right(boundaries, abs(value))]
        if saturate and code == 0x6F:
            code = 0x6E
        output.append(code | 0x80 if value < 0 and code else code)
    return output


def hif8_boundary_inputs():
    levels = sorted(value for value in hif8_reference_table()[:128] if math.isfinite(value))
    boundaries = torch.tensor([(a + b) / 2 for a, b in zip(levels[:-1], levels[1:], strict=True)] + [40960.0])
    positive = torch.stack(
        (
            torch.nextafter(boundaries, torch.full_like(boundaries, -math.inf)),
            boundaries,
            torch.nextafter(boundaries, torch.full_like(boundaries, math.inf)),
        )
    ).flatten()
    return torch.cat((positive, -positive, torch.tensor([0.0, -0.0, math.inf, -math.inf, math.nan])))


CARRIERS = 256          # every one-byte code of each format
INT4_VALUES = list(range(-8, 8))


def make_inputs(case):
    """The transport case's operands: the complete carrier byte space, in one order or another."""
    pattern = case["parameters"]["pattern"]
    carriers = torch.arange(CARRIERS, dtype=torch.int32).to(torch.uint8)
    if pattern == "descending":
        carriers = carriers.flip(0)
    inputs = {"x": carriers.reshape(1, CARRIERS).contiguous()}
    validate(inputs)
    return inputs


def validate(inputs):
    x = inputs["x"]
    if not isinstance(x, torch.Tensor) or x.dtype != torch.uint8:
        raise ValueError("the carriers must be uint8")
    if tuple(x.shape) != (1, CARRIERS) or not x.is_contiguous():
        raise ValueError(f"the carriers must be one contiguous row of {CARRIERS}")
    if int(torch.unique(x).numel()) != CARRIERS:
        raise ValueError("every byte value must appear exactly once, or a lost byte could hide")


def reference(inputs):
    """A transport changes nothing."""
    validate(inputs)
    return {"o": inputs["x"].clone()}


def tables():
    """The independent value tables the library's codecs are compared against."""
    return {
        "int4": torch.tensor(int4_reference(INT4_VALUES), dtype=torch.int32),
        "fp4_e2m1": torch.tensor([v for byte in range(CARRIERS)
                                  for v in fp4_reference("e2m1", byte)]),
        "fp4_e1m2": torch.tensor([v for byte in range(CARRIERS)
                                  for v in fp4_reference("e1m2", byte)]),
        # This legacy host helper treats byte 255 as +inf rather than an IEEE MX NaN payload.
        "e8m0": torch.tensor([math.ldexp(1.0, code - 127) for code in range(CARRIERS - 1)]
                             + [math.inf]),
        "hif8": torch.tensor(hif8_reference_table()),
    }
