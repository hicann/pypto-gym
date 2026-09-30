# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Data cache clean on a GM tensor parameter (RFC-0015 "Data cache clean")."""
from ...ir import Ident, Value
from ...ir.types import ScalarType
from .lower import Memory

TARGETS = {"system.dcci": ("core.clean_dcache",)}
CACHE_LINES = {0: "SINGLE_CACHE_LINE", 1: "ENTIRE_DATA_CACHE"}  # pl.CacheLine
# pl.DcciDst: AUTO prints CACHELINE_OUT for a tensor; the others keep their names.
DESTINATIONS = {0: "CACHELINE_OUT", 1: "CACHELINE_OUT", 2: "CACHELINE_UB", 3: "CACHELINE_ALL", 4: "CACHELINE_ATOMIC"}
DTYPES = ("i32", "f32", "f16", "bf16")  # GM element types measured on A5


def inside(ctx, value, bound):
    """A static or proven integer in [0, bound)."""
    from .selection import interval

    if not (type(value) is int or isinstance(value, Value) and isinstance(value.type, ScalarType)
            and value.type.dtype.kind in {"int", "uint"}):
        return None
    bounds = interval(ctx, value)
    return bounds is not None and 0 <= bounds[0] and bounds[1] < bound


def convert(ctx, node, args):
    from .memory import gm_window

    o = ctx.o
    attrs = ctx.attrs(node, {"cache_line", "dst"})
    line, target = attrs.get("cache_line", 1), attrs.get("dst", 0)  # Pro's defaults: ENTIRE_DATA_CACHE, AUTO.
    o.need(type(line) is int and line in CACHE_LINES, node, "dcci cache_line must be SINGLE_CACHE_LINE or ENTIRE_DATA_CACHE")
    o.need(type(target) is int and target in DESTINATIONS, node,
           "dcci dst must be AUTO, CACHELINE_OUT, CACHELINE_UB, CACHELINE_ALL or CACHELINE_ATOMIC")
    o.need(len(args) in (1, 2) and isinstance(args[0], Memory), node, "dcci needs a GM tensor and at most one offset")
    memory = args[0]
    o.need(memory.value.type.space == "gm", node, "dcci of tiles and workspaces is unmeasured; only GM tensor parameters are admitted")
    o.need(memory.root is None and memory.nz is None, node, "dcci through a GM view or an NZ-packed parameter is unmeasured")
    o.need(memory.value.type.dtype.name in DTYPES, node,
           "dcci of GM tensors other than INT32, FP32, FP16 and BF16 is unmeasured on A5")
    o.need(len(memory.shape) == 2 and all(type(n) is int for n in memory.shape), node, "dcci needs a static two-dimensional GM tensor")
    rows, cols = memory.shape
    if len(args) == 1:
        window = memory.value
    else:
        offset = args[1]
        if isinstance(offset, tuple):
            o.need(len(offset) == 2, node, "dcci tensor offsets need a row and a column")
            offsets = tuple(ctx.snapshot(value, node) for value in offset)
            proven = all(inside(ctx, value, bound) for value, bound in zip(offsets, (rows, cols), strict=True))
        else:
            element = ctx.snapshot(offset, node)
            proven = inside(ctx, element, rows * cols)
        o.need(proven, node, "dcci offsets must be integers proven inside the GM tensor")
        if not isinstance(offset, tuple):
            offsets = split(ctx, node, element, cols)
        window = gm_window(ctx, node, memory, offsets, (1, 1))
    ctx.emit("core.clean_dcache", node, attrs={"dst": window, "entire_type": Ident(CACHE_LINES[line]),
                                              "dcci_dst": Ident(DESTINATIONS[target])})


def split(ctx, node, element, cols):
    """Row-major (row, column) of a proven element offset, which Pro adds to the data pointer."""
    from .selection import interval

    if type(element) is int:
        return divmod(element, cols)
    low, high = interval(ctx, element)
    if high < cols:
        return 0, element
    row = ctx.emit("scalar.div", node, (element, cols), typ=element.type)
    column = ctx.emit("scalar.mod", node, (element, cols), typ=element.type)
    ctx.bounds[row.name], ctx.bounds[column.name] = (low // cols, high // cols), (0, cols - 1)
    ctx.nonnegative.update((row.name, column.name))
    return row, column
