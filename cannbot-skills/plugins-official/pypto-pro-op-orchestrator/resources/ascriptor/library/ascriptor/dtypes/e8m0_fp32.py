# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import torch

from ._common import _as_uint8_codes

E8M0_BIAS = 127
E8M0_MIN_VALUE = 2.0 ** -127


def e8m0_to_fp32(codes: torch.Tensor) -> torch.Tensor:
    """Decode e8m0 scale bytes as ``2 ** (code - 127)``."""
    if not isinstance(codes, torch.Tensor):
        codes = torch.as_tensor(codes)
    byte = _as_uint8_codes(codes)
    exponent = byte.to(torch.float32) - float(E8M0_BIAS)
    return torch.pow(torch.full_like(exponent, 2.0), exponent)


def fp32_to_e8m0(values: torch.Tensor, nan_to_zero: bool = False) -> torch.Tensor:
    """Encode positive float32 scale values to floor-log2 e8m0 scale bytes.

    e8m0 has no sign or zero payload. Zero values encode to byte 0, the minimum
    scale value. Negative values are rejected. Positive infinities saturate to
    byte 255.
    """
    if not isinstance(values, torch.Tensor):
        values = torch.as_tensor(values)
    x = values.to(torch.float32)

    if bool((x < 0).any().item()):
        raise ValueError("e8m0 only supports non-negative scale values")
    if not nan_to_zero and bool(torch.isnan(x).any().item()):
        raise ValueError("e8m0 has no NaN payload; pass nan_to_zero=True to encode NaN as byte 0")
    if nan_to_zero:
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)

    safe = torch.clamp(x, min=E8M0_MIN_VALUE)
    exponent = torch.floor(torch.log2(safe))
    code = torch.clamp(exponent + float(E8M0_BIAS), min=0.0, max=255.0)
    return code.to(torch.uint8)
