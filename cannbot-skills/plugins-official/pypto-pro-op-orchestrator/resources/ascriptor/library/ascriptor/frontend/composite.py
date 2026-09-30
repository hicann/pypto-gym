# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate public composite boundaries before compiling their ordinary DSL bodies."""

from __future__ import annotations

import inspect

from ..ir.types import CellType, MemType, ScalarType
from .errors import E_BAD_OPERAND, E_BAD_SHAPE, E_BAD_SIGNATURE, E_UNSUPPORTED
from .values import NO_RIDERS, Dyn


def compile_radix_topk(fc, callee, args, kwargs, node):
    """RFC-0014 admission only; selection itself lives in composites/topk.py."""
    try:
        bound = inspect.signature(callee).bind(*args, **kwargs)
    except TypeError as exc:
        raise fc.err(E_BAD_SIGNATURE, f"radix_topk: {exc}", node) from None
    bound.apply_defaults()
    values = bound.arguments
    if fc.kind != "kernel" or fc.fe.kernel.device not in ("950", "a5"):
        raise fc.err(E_UNSUPPORTED, "radix_topk requires an A5 kernel body", node)
    for name, expected in (("largest", True), ("sorted", False)):
        if values[name] is not expected:
            raise fc.err(E_UNSUPPORTED, f"radix_topk requires the literal {name}={expected}", node)

    roots = []
    for name, dtype, shape in (("src", "f32", (1, 4096)),
                               ("dst_values", "f32", (1, 512)),
                               ("dst_indices", "i32", (1, 512))):
        value = values[name]
        if not isinstance(value, Dyn) or not isinstance(value.type, MemType) or value.type.space != "ub":
            raise fc.err(E_BAD_OPERAND, f"radix_topk {name} must be a UB tensor", node)
        if value.type.dtype.name != dtype:
            raise fc.err(E_BAD_OPERAND, f"radix_topk {name} must have dtype {dtype}", node)
        geom = fc.geom(value, node)
        if (tuple(geom.shape) != shape or tuple(geom.span) != shape
                or any(offset != 0 for offset in geom.offset) or value.riders != NO_RIDERS):
            raise fc.err(E_BAD_SHAPE, f"radix_topk {name} requires a complete UB {dtype}{list(shape)} buffer without offsets or riders", node)
        roots.append(fc.roots.get(value.name, geom.root))
    if len(set(roots)) != 3:
        raise fc.err(E_BAD_OPERAND, "radix_topk input and outputs must have distinct backing allocations", node)

    for name, upper in (("count", 4096), ("k", 512)):
        value = values[name]
        if type(value) is int:
            if not 1 <= value <= upper:
                raise fc.err(E_BAD_OPERAND, f"radix_topk requires 1 <= {name} <= {upper}", node)
        elif not (isinstance(value, Dyn) and isinstance(value.type, (ScalarType, CellType))
                  and value.type.dtype.name == "i32"):
            raise fc.err(E_BAD_OPERAND, f"radix_topk {name} must be an integer constant or INT32 scalar", node)
    if type(values["count"]) is int and type(values["k"]) is int and values["k"] > values["count"]:
        raise fc.err(E_BAD_OPERAND, "radix_topk requires k <= count", node)
    return fc.inline(callee, args, kwargs, node)
