# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Reject runtime local allocation capacity before creating native Tiles."""

from ...ir.types import BufType
from ...passes.util import dim_scalar


def require_static_allocation(op, fold, gap):
    """Runtime valid extents belong to views, not an allocation's capacity."""
    typ = op.results[0].type
    elem = typ.elem if isinstance(typ, BufType) else typ
    if any(fold(dim_scalar(dim)) is None for dim in elem.dims):
        raise gap(op, "dynamic local Tile capacity is unsupported; specialize the allocation "
                  "dimensions or use fixed capacity with bounded runtime valid extents", owner="ours")
    if fold(op.attrs.get("addr", 0)) is None:
        raise gap(op, "dynamic local Tile allocation address is unsupported; "
                  "all preceding local capacities must be static", owner="ours")
