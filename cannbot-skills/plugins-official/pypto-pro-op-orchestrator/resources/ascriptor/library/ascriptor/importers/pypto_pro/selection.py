# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded value selection over separately owned physical slots."""
from dataclasses import dataclass
from itertools import product

from ...ir import Block, Ident, Value
from ...ir.types import CellType, ScalarType, dtype
from .lower import Memory


@dataclass(frozen=True)
class Choice:
    index: Value
    items: tuple


def interval(ctx, value):
    if type(value) is int:
        return value, value
    if isinstance(value, Value) and isinstance(value.type, CellType):  # A reassigned scalar here (loop_extents).
        return ctx.cell_ranges.get(value.name)
    return ctx.bounds.get(value.name) if isinstance(value, Value) else None


def select(ctx, node, items, index):
    o = ctx.o
    o.need(isinstance(items, (tuple, list)) and items, node, "Expected a nonempty tuple")
    from .slots import Slot, choose
    if isinstance(items[0], Slot):
        return choose(ctx, node, items, index)
    if type(index) is int:
        o.need(0 <= index < len(items), node, "Tuple index is out of range")
        return items[index]
    index = ctx.snapshot(index, node)
    bounds = interval(ctx, index)
    o.need(bounds is not None and 0 <= bounds[0] <= bounds[1] < len(items), node,
           "Dynamic tuple/slot index requires an in-range integer proof")
    if all(isinstance(item, Memory) for item in items):
        first = items[0]
        o.need(all((item.value.type, item.shape, item.valid, item.pitch) ==
                   (first.value.type, first.shape, first.valid, first.pitch) for item in items), node,
               "Dynamic slot descriptors must agree in type, shape and pitch")
        return Choice(index, tuple(items))
    o.need(all(type(item) is int for item in items), node, "Only integer columns and tile slots admit dynamic selection")
    typ = o.scalar_type(node["fields"]["type"])
    o.need(typ.dtype.name in {"i32", "i64"}, node, "Integer selection requires an integer result type")
    limit = 1 << (typ.dtype.bits - 1)
    o.need(all(-limit <= item < limit for item in items), node, "Tuple column exceeds its integer result width")
    result = items[-1]
    for i in reversed(range(len(items) - 1)):
        cond = ctx.emit("scalar.cmp", node, (index, i), typ=ScalarType(dtype("b1")), attrs={"pred": Ident("eq")})
        result = ctx.emit("scalar.select", node, (cond, items[i], result), typ=typ)
    if not isinstance(result, Value):
        return result
    ctx.bounds[result.name] = (min(items), max(items))
    return result


def dispatch(ctx, node, values, action):
    """Expand consumers, not storage; correlated choices use the same branch."""
    axes = {}
    for value in values:
        if isinstance(value, Choice):
            previous = axes.setdefault(value.index, len(value.items))
            ctx.o.need(previous == len(value.items), node, "Correlated selections disagree in depth")
    count = 1
    for depth in axes.values():
        count *= depth
    ctx.o.need(count <= 16, node, "Slot selection expansion exceeds the admitted 16 alternatives")
    for positions in product(*(range(depth) for depth in axes.values())):
        picked = dict(zip(axes, positions, strict=True))
        condition = None
        for index, position in picked.items():
            test = ctx.emit("scalar.cmp", node, (index, position), typ=ScalarType(dtype("b1")), attrs={"pred": Ident("eq")})
            condition = test if condition is None else ctx.emit("scalar.and", node, (condition, test), typ=ScalarType(dtype("b1")))
        child = ctx.child()
        concrete = [value.items[picked[value.index]] if isinstance(value, Choice) else value for value in values]
        action(child, concrete)
        ctx.emit("cf.if", node, (condition,), regions=(Block(tuple(child.ops)), Block(())))


def binary_bounds(ctx, kind, a, b, result):
    left, right = interval(ctx, a), interval(ctx, b)
    bounds = None
    if kind == "Min":
        ctx.upper[result.name] = set().union(*(ctx.upper.get(v.name, set()) | {v.name}
                                              for v in (a, b) if isinstance(v, Value)))
    if kind == "FloorMod" and type(b) is int and b > 0:
        bounds = (0, b - 1)  # The caller proves the nonnegative dividend.
    elif left is not None and right is not None:
        lo, hi = left
        x, y = right
        if kind == "Add":
            bounds = lo + x, hi + y
        elif kind == "Sub":
            bounds = lo - y, hi - x
        elif kind == "Mul":
            candidates = (lo * x, lo * y, hi * x, hi * y)
            bounds = min(candidates), max(candidates)
        elif kind == "FloorDiv" and x > 0:
            bounds = lo // y, hi // x
        elif kind == "Min":
            bounds = min(lo, x), min(hi, y)
        elif kind == "Max":
            bounds = max(lo, x), max(hi, y)
    if bounds is not None and result.type.dtype.name in {"i32", "i64", "u32"}:
        bits = result.type.dtype.bits
        low, limit = (0, 1 << bits) if result.type.dtype.kind == "uint" else (-(1 << (bits - 1)), 1 << (bits - 1))
        if low <= bounds[0] <= bounds[1] < limit:
            ctx.bounds[result.name] = bounds
            if bounds[0] >= 0:
                ctx.nonnegative.add(result.name)
    from .loop_extents import record
    record(ctx, kind, a, b, result)
