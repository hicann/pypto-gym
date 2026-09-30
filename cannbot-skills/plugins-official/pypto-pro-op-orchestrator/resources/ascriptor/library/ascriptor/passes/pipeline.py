# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The canonical pipeline (RFC-0001 §12) and the ``lower`` entry point."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..ir import Module
from . import (
    addr_alloc,
    autosync,
    cellfold,
    crosssync,
    dce,
    desugar,
    device_lower,
    events,
    gmbuff,
    liveness,
    local_mutex,
    mmad_settle,
    scalar_simplify,
    split_sides,
)
from .manager import Pass, PassManager

PIPELINE: tuple[Pass, ...] = (
    cellfold.PASS_DEF,
    desugar.PASS_DEF,
    device_lower.PASS_DEF,
    gmbuff.PASS_DEF,
    crosssync.PASS_DEF,
    autosync.PASS_DEF,
    events.PASS_DEF,
    addr_alloc.PASS_DEF,
    split_sides.PASS_DEF,
    events.PASS_DEF_RESTAMP,
    dce.PASS_DEF,
    local_mutex.PASS_DEF,
    local_mutex.COALESCE_PASS,
    scalar_simplify.PASS_DEF,
    mmad_settle.PASS_DEF,
    liveness.PASS_DEF,
)


def pass_named(name: str) -> Pass:
    for p in PIPELINE:
        if p.name == name:
            return p
    raise KeyError(name)


def lower(module: Module, *, options: Mapping[str, Any] | None = None, stop_after: str | None = None,
          manager: PassManager | None = None) -> Module:
    """Run the pipeline (or its prefix up to ``stop_after``) on a Surface module."""
    pm = manager or PassManager(PIPELINE, options=options)
    return pm.run(module, stop_after=stop_after)


__all__ = ["PIPELINE", "lower", "pass_named"]
