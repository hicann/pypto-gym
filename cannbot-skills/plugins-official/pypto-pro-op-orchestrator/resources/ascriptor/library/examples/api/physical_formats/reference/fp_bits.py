# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Literal packed FP4 fields and IEEE FP8 value lattices; no narrowing codec."""

import bisect
import math
import struct


def fp4_e1m2_code(value):
    # E1M2 has three fraction bits at a fixed quarter-unit spacing. The sign
    # follows the source sign bit, including negative zero (M10-041).
    if not math.isfinite(value):
        raise ValueError('this FP4 unit declares finite BF16 input')
    magnitude = min(7, round(abs(value) * 4))
    sign = struct.unpack('<I', struct.pack('<f', value))[0] >> 31
    return magnitude | sign << 3


def fp8_lattice(kind):
    fraction_bits, bias, last = (3, 7, 126) if kind == 'e4m3' else (2, 15, 123)
    points = []
    for code in range(last + 1):
        exponent, fraction = code >> fraction_bits, code & ((1 << fraction_bits) - 1)
        value = math.ldexp(fraction, 1 - bias - fraction_bits) if exponent == 0 else math.ldexp(
            (1 << fraction_bits) + fraction, exponent - bias - fraction_bits)
        points.append(value)
    return points


LATTICES = {kind: fp8_lattice(kind) for kind in ('e4m3', 'e5m2')}


def fp8_code(value, kind):
    word = struct.unpack('<I', struct.pack('<f', value))[0]
    sign = word >> 31
    magnitude = abs(value)
    if math.isnan(value):
        return 0x7F
    if kind == 'e4m3' and (math.isinf(value) or magnitude > 464):
        return 0x7F
    if kind == 'e5m2' and (math.isinf(value) or magnitude >= 61440):
        return (sign << 7) | 0x7C
    points = LATTICES[kind]
    index = bisect.bisect_left(points, magnitude)
    candidates = range(max(0, index - 1), min(len(points), index + 1))
    code = min(candidates, key=lambda i: (abs(points[i] - magnitude), i & 1))
    return (sign << 7) | code
