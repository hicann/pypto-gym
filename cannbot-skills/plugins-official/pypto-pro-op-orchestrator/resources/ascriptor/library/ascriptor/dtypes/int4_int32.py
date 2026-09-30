# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import torch


def pack_signed_int4_to_int32(values: torch.Tensor) -> torch.Tensor:
    """Pack signed int4 values into int32 carriers, low nibble first.

    The logical values must be in the two's-complement signed int4 range
    ``[-8, 7]``. The last dimension is packed in groups of 8 values per int32.
    """
    if not isinstance(values, torch.Tensor):
        values = torch.as_tensor(values)
    if values.dim() == 0:
        raise ValueError("signed int4 values must have at least one dimension")

    values_i32 = values.to(torch.int32)
    if bool(((values_i32 < -8) | (values_i32 > 7)).any().item()):
        raise ValueError("signed int4 values must be in [-8, 7]")

    logical_k = values_i32.shape[-1]
    carrier = (logical_k + 7) // 8
    padded = torch.zeros((*values_i32.shape[:-1], carrier * 8), dtype=torch.int32, device=values_i32.device)
    padded[..., :logical_k] = values_i32

    nibbles = torch.where(padded < 0, padded + 16, padded).to(torch.int64)
    shifts = torch.arange(8, dtype=torch.int64, device=values_i32.device) * 4
    packed = (nibbles.reshape(*values_i32.shape[:-1], carrier, 8) << shifts).sum(dim=-1)
    packed = torch.where(packed >= 2 ** 31, packed - 2 ** 32, packed)
    return packed.to(torch.int32)


def unpack_int32_to_signed_int4(packed: torch.Tensor, logical_k: int) -> torch.Tensor:
    """Unpack low-nibble-first int32 carriers into signed int4 int32 values."""
    if not isinstance(packed, torch.Tensor):
        packed = torch.as_tensor(packed)
    if packed.dim() == 0:
        raise ValueError("packed int32 carriers must have at least one dimension")
    if logical_k < 0:
        raise ValueError(f"logical_k must be non-negative, got: {logical_k}")

    carrier = (logical_k + 7) // 8
    if packed.shape[-1] < carrier:
        raise ValueError(
            f"packed carrier last dimension is too small for logical_k={logical_k}: "
            f"need {carrier}, got {packed.shape[-1]}"
        )

    raw = packed[..., :carrier].to(torch.int64)
    shifts = torch.arange(8, dtype=torch.int64, device=packed.device) * 4
    nibbles = ((raw.unsqueeze(-1) >> shifts) & 0xF).to(torch.int32)
    signed = torch.where(nibbles >= 8, nibbles - 16, nibbles)
    return signed.reshape(*packed.shape[:-1], carrier * 8)[..., :logical_k].contiguous()


pack_signed_int4 = pack_signed_int4_to_int32
unpack_signed_int4 = unpack_int32_to_signed_int4
pack_int4_to_int32 = pack_signed_int4_to_int32
unpack_int32_to_int4 = unpack_int32_to_signed_int4
