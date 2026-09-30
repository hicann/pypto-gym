# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host dtype codecs with lazy tensor dependencies.

Importing codecs or a device facade requires only Python. Calling a codec loads
its implementation and requires the ``ascriptor[torch]`` extra. Carrier layout,
rounding defaults and established aliases match the implementation modules.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

E8M0_BIAS = 127
E8M0_MIN_VALUE = 2.0 ** (-127)


def e8m0_to_fp32(codes: torch.Tensor) -> torch.Tensor:
    'Decode e8m0 scale bytes as ``2 ** (code - 127)``.'
    from .e8m0_fp32 import e8m0_to_fp32 as implementation

    return implementation(codes)



def fp32_to_e8m0(values: torch.Tensor, nan_to_zero: bool=False) -> torch.Tensor:
    'Encode positive float32 scale values to floor-log2 e8m0 scale bytes.\n\ne8m0 has no sign or zero payload. Zero values encode to byte 0, the minimum\nscale value. Negative values are rejected. Positive infinities saturate to\nbyte 255.'
    from .e8m0_fp32 import fp32_to_e8m0 as implementation

    return implementation(values, nan_to_zero)

FP4_E2M1_MAX_VALUE = 6.0
FP4_E1M2_MAX_VALUE = 1.75


def fp32_to_fp4_e2m1(values: torch.Tensor, pack_axis: int=-1, nan_to_zero: bool=False, round_mode: object='CAST_ROUND') -> torch.Tensor:
    'Encode float32 values to FP4 E2M1 uint8 carriers.\n\nTwo FP4 payloads are packed in each uint8, low nibble first along\n``pack_axis``. ``round_mode`` accepts the five CANN BF16-to-FP4 modes and\ndefaults to ``CAST_ROUND`` (nearest, ties away from zero).'
    from .fp4_fp32 import fp32_to_fp4_e2m1 as implementation

    return implementation(values, pack_axis, nan_to_zero, round_mode)



def fp4_e2m1_to_fp32(carrier: torch.Tensor, logical_shape: Sequence[int] | None=None, pack_axis: int=-1) -> torch.Tensor:
    'Decode FP4 E2M1 uint8 carriers to float32 values.'
    from .fp4_fp32 import fp4_e2m1_to_fp32 as implementation

    return implementation(carrier, logical_shape, pack_axis)



def fp32_to_fp4_e1m2(values: torch.Tensor, pack_axis: int=-1, nan_to_zero: bool=False, round_mode: object='CAST_ROUND') -> torch.Tensor:
    'Encode float32 values to FP4 E1M2 uint8 carriers.\n\nTwo FP4 payloads are packed in each uint8, low nibble first along\n``pack_axis``. ``round_mode`` accepts the five CANN BF16-to-FP4 modes and\ndefaults to ``CAST_ROUND`` (nearest, ties away from zero).'
    from .fp4_fp32 import fp32_to_fp4_e1m2 as implementation

    return implementation(values, pack_axis, nan_to_zero, round_mode)



def fp4_e1m2_to_fp32(carrier: torch.Tensor, logical_shape: Sequence[int] | None=None, pack_axis: int=-1) -> torch.Tensor:
    'Decode FP4 E1M2 uint8 carriers to float32 values.'
    from .fp4_fp32 import fp4_e1m2_to_fp32 as implementation

    return implementation(carrier, logical_shape, pack_axis)

HIF8_POSITIVE_ZERO = 0
HIF8_NAN = 128
HIF8_POSITIVE_INF = 111
HIF8_NEGATIVE_INF = 239
HIF8_MAX_POSITIVE_NORMAL = 110
HIF8_MAX_NEGATIVE_NORMAL = 238
HIF8_MAX_FINITE_VALUE = 2.0 ** 15
HIF8_OVERFLOW_THRESHOLD = HIF8_MAX_FINITE_VALUE * 1.25


def hif8_to_fp32(codes: torch.Tensor) -> torch.Tensor:
    'Decode Ascend HiFloat8 bit patterns to float32 values.\n\nHiFloat8 has no negative zero: 0x00 is zero and 0x80 is NaN.\nThe two largest absolute normal bit patterns, 0x6f and 0xef, decode to\npositive and negative infinity. A 256-entry table built once per device\nby :func:`_decode_hif8` (the simulator decodes per vector op).'
    from .hif8_codec import hif8_to_fp32 as implementation

    return implementation(codes)



def fp32_to_hif8(values: torch.Tensor, saturate: bool=False, nan_to_zero: bool=False, round_mode: object=None) -> torch.Tensor:
    'Encode float32 values as Ascend HiFloat8 uint8 bit patterns.\n\nBy default, finite values use the TA conversion mode (round to nearest with\nties away from zero). ``round_mode=RoundMode.HYBRID`` or\n``round_mode="hybrid"`` switches to TA for exponents with ``abs(e) < 4``\nand SSR elsewhere. Overflow maps to infinity by default; set\n``saturate=True`` to clamp overflow to the largest finite HiF8 value.'
    from .hif8_codec import fp32_to_hif8 as implementation

    return implementation(values, saturate, nan_to_zero, round_mode)



def fp16_to_hif8(values: torch.Tensor, saturate: bool=False, nan_to_zero: bool=False, round_mode: object=None) -> torch.Tensor:
    'Encode float16 values as Ascend HiFloat8 uint8 bit patterns.\n\n``CAST_ROUND`` follows the same TA nearest-away hif8 codebook rounding as\nthe float32 helper. ``CAST_HYBRID`` uses TA when ``abs(e) < 4`` and a\nhalf-specific SSR path elsewhere; the implementation follows CANNSIM rather\nthan the buggy boundary cases in the half conversion script.'
    from .hif8_codec import fp16_to_hif8 as implementation

    return implementation(values, saturate, nan_to_zero, round_mode)



def pack_signed_int4_to_int32(values: torch.Tensor) -> torch.Tensor:
    "Pack signed int4 values into int32 carriers, low nibble first.\n\nThe logical values must be in the two's-complement signed int4 range\n``[-8, 7]``. The last dimension is packed in groups of 8 values per int32."
    from .int4_int32 import pack_signed_int4_to_int32 as implementation

    return implementation(values)



def unpack_int32_to_signed_int4(packed: torch.Tensor, logical_k: int) -> torch.Tensor:
    'Unpack low-nibble-first int32 carriers into signed int4 int32 values.'
    from .int4_int32 import unpack_int32_to_signed_int4 as implementation

    return implementation(packed, logical_k)



hifloat8_to_fp32 = hif8_to_fp32
fp32_to_hifloat8 = fp32_to_hif8
fp16_to_hifloat8 = fp16_to_hif8
pack_signed_int4 = pack_signed_int4_to_int32
unpack_signed_int4 = unpack_int32_to_signed_int4
pack_int4_to_int32 = pack_signed_int4_to_int32
unpack_int32_to_int4 = unpack_int32_to_signed_int4


__all__ = ['E8M0_BIAS', 'E8M0_MIN_VALUE', 'FP4_E1M2_MAX_VALUE', 'FP4_E2M1_MAX_VALUE', 'HIF8_MAX_FINITE_VALUE', 'HIF8_MAX_NEGATIVE_NORMAL', 'HIF8_MAX_POSITIVE_NORMAL', 'HIF8_NAN', 'HIF8_NEGATIVE_INF', 'HIF8_OVERFLOW_THRESHOLD', 'HIF8_POSITIVE_INF', 'HIF8_POSITIVE_ZERO', 'e8m0_to_fp32', 'fp32_to_e8m0', 'fp32_to_fp4_e1m2', 'fp32_to_fp4_e2m1', 'fp16_to_hif8', 'fp16_to_hifloat8', 'fp32_to_hif8', 'fp32_to_hifloat8', 'fp4_e1m2_to_fp32', 'fp4_e2m1_to_fp32', 'hif8_to_fp32', 'hifloat8_to_fp32', 'pack_int4_to_int32', 'pack_signed_int4', 'pack_signed_int4_to_int32', 'unpack_int32_to_int4', 'unpack_int32_to_signed_int4', 'unpack_signed_int4']
