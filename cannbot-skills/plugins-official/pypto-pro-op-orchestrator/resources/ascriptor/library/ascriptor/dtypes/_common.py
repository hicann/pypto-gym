# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import torch


def _require_integer_tensor(values: torch.Tensor, name: str) -> None:
    if values.dtype == torch.bool or values.is_floating_point():
        raise TypeError(f"{name} must use an integer dtype, got: {values.dtype}")


def _as_uint8_codes(codes: torch.Tensor) -> torch.Tensor:
    _require_integer_tensor(codes, "codes")
    if codes.dtype != torch.uint8 and int(codes.numel()) > 0:
        min_code = int(codes.min().item())
        max_code = int(codes.max().item())
        if min_code < 0 or max_code > 0xFF:
            raise ValueError(f"codes must be in [0, 255], got min={min_code}, max={max_code}")
    return codes.to(torch.uint8)
