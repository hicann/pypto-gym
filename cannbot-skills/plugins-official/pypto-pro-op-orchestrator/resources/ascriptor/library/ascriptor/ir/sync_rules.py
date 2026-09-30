# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Raw flag channel constraints shared by verification, allocation and models."""

from .core import Ident, Literal, Value
from .types import CellType, PIPES, ScalarType

FLAG_IDS = 8
SET_PIPES = frozenset(PIPES)  # the scalar pipe sets flags too (RFC-0001; the dcci protocol needs S -> MTE3)
WAIT_PIPES = frozenset(PIPES)


def raw_flag_error(source, target, event_id):
    """Return a located caller's error text, or None for a valid/unknown integer ID."""
    source = source.name if isinstance(source, Ident) else source
    target = target.name if isinstance(target, Ident) else target
    if not isinstance(source, str) or source not in SET_PIPES:
        return f"raw flag source pipe must be one of {', '.join(sorted(SET_PIPES))}, got {source!r}"
    if not isinstance(target, str) or target not in WAIT_PIPES:
        return f"raw flag destination pipe must be one of {', '.join(sorted(WAIT_PIPES))}, got {target!r}"
    if source == target:
        return f"raw flag needs distinct source and destination pipes, got {source}->{target}"
    if isinstance(event_id, Literal):
        event_id = event_id.value
    if isinstance(event_id, Value):
        dtype = event_id.type.dtype if isinstance(event_id.type, (CellType, ScalarType)) else None
        if dtype is not None and dtype.kind in ("int", "uint"):
            return None
        return "raw flag ID must be an integer scalar"
    if isinstance(event_id, bool) or not isinstance(event_id, int):
        return "raw flag ID must be an integer scalar"
    if not 0 <= event_id < FLAG_IDS:
        return f"raw flag ID must be in [0, {FLAG_IDS}), got {event_id}"
    return None
