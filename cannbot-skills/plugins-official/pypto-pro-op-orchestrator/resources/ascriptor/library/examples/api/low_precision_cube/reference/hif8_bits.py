# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent IEEE fields and HiFloat8 value classes; no execution codec import.

The SSR threshold arithmetic is source-width specific. It is audited by the
M10-033 boundary controls and the separately identified 512-byte board probe.
"""

import math
import random

def _positive_values():
    """Enumerate format classes, not the execution codec's decode table."""
    table = {0.0: 0}
    for exponent in range(-22, -15):
        table[math.ldexp(1.0, exponent)] = exponent + 23
    for exponent in range(-15, 16):
        size = abs(exponent).bit_length()
        fraction_bits = 3 if size <= 2 else 5 - size
        base = {0: 8, 1: 16, 2: 32, 3: 64, 4: 96}[size]
        exponent_code = 0 if size == 0 else abs(exponent) - (1 << size - 1) << fraction_bits
        if exponent < 0:
            exponent_code += 1 << size - 1 + fraction_bits
        for fraction in range(1 << fraction_bits):
            code = base + exponent_code + fraction
            if code != 111:
                table[math.ldexp((1 << fraction_bits) + fraction, exponent - fraction_bits)] = code
    return table

POSITIVE = _positive_values()

def _ieee_parts(word, bits):
    (width, exp_bits, bias) = (23, 8, 127) if bits == 32 else (10, 5, 15)
    sign = word >> bits - 1
    field = word >> width & (1 << exp_bits) - 1
    fraction = word & (1 << width) - 1
    if field == (1 << exp_bits) - 1:
        return (sign, 'nan' if fraction else 'inf', None, None, None)
    if field == 0:
        if fraction == 0:
            return (sign, 'zero', None, None, None)
        leading = fraction.bit_length() - 1
        exponent = leading + 1 - bias - width
        significand = fraction << width - leading
    else:
        exponent = field - bias
        significand = 1 << width | fraction
    return (sign, 'finite', exponent, significand, width)

def expected_code(word, bits, mode='hybrid', saturate=False, nan_to_zero=False):
    """Independent IEEE fields, explicit format lattice and rational SSR thresholds."""
    (sign, kind, exponent, significand, width) = _ieee_parts(word, bits)
    if kind == 'nan':
        return 0 if nan_to_zero else 128
    if kind == 'zero':
        return 0
    if kind == 'inf':
        return (110 if saturate else 111) | sign << 7
    value = math.ldexp(significand, exponent - width)
    if value >= 40960:
        code = 110 if saturate else 111
    elif mode != 'hybrid' or abs(exponent) < 4:
        closest = min(POSITIVE, key=lambda point: (abs(point - value), -point))
        code = POSITIVE[closest]
    elif exponent < -23:
        code = 0
    else:
        kept = 0 if exponent < -15 else 2 if abs(exponent) < 8 else 1
        discarded = width - kept
        remainder = significand % (1 << discarded)
        precision = 14 if bits == 32 else 2
        if exponent == -23:
            (lower, step) = (0.0, math.ldexp(1.0, -22))
            probability = significand >> width + 1 - precision
        else:
            step = math.ldexp(1.0, exponent - kept)
            lower = (significand >> discarded) * step
            probability = remainder >> discarded - precision
        threshold = significand - (1 << width) & 16383 if bits == 32 else (word & 1) * 2 + 1
        rounded = lower + step if probability >= threshold else lower
        code = 111 if rounded == 49152 else POSITIVE[rounded]
    return code | sign << 7 if code else 0

def boundary_words(bits):
    (width, exp_bits) = (23, 8) if bits == 32 else (10, 5)
    words = {0, 1, (1 << width) - 1, 1 << width}
    for field in range(1, (1 << exp_bits) - 1):
        word = field << width
        words.update([word - 1, word, word + 1])
    for position in range(width):
        word = 1 << position
        words.update([word - 1, word, word + 1])
    special = (1 << exp_bits) - 1 << width
    words.update([special - 1, special, special + 1, special | 1 << width - 1])
    generator = random.Random(1033 + bits)
    words.update((generator.getrandbits(bits - 1) for _ in range(128)))
    return sorted(words | {word | 1 << bits - 1 for word in words})

def decode_code(code):
    if code == 0x80:
        return math.nan
    sign, magnitude = code >> 7, code & 0x7F
    if magnitude == 0x6F:
        return -math.inf if sign else math.inf
    values = {payload: value for value, payload in POSITIVE.items()}
    return -values[magnitude] if sign else values[magnitude]
