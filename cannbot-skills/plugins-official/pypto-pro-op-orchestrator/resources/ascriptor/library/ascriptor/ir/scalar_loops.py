# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Checked finite-loop counter summaries and comparison-guided widening hints."""

from .core import Value
from .scalar_math import limits
from .types import CellType, ScalarType


def trip_count(op, state, bound):
    args = [bound(x, state) for x in op.operands]
    if not all(x is not None and x[0] == x[1] for x in args):
        return None
    start, stop, step = (x[0] for x in args)
    if not step or (stop - start) * step <= 0:
        return None
    count = (abs(stop - start) + abs(step) - 1) // abs(step)
    low, high = limits(op.results[0].type.dtype)
    # Native induction must also reach its exit without integer wraparound.
    return count if all(low <= x <= high for x in (start, stop, step, start + count * step)) else None


def increment(op, name, typ, state, bound):
    if (op.opcode not in ('scalar.add', 'scalar.sub') or len(op.results) != 1
            or op.results[0].type != ScalarType(typ.dtype)):
        return None
    a, b = op.operands
    if op.opcode == 'scalar.add' and getattr(b, 'name', None) == name:
        a, b = b, a
    if getattr(a, 'name', None) != name or a.type != typ or isinstance(getattr(b, 'type', None), CellType):
        return None
    interval = bound(b, state)
    if interval is None or interval[0] != interval[1]:
        return None
    return interval[0] * (-1 if op.opcode == 'scalar.sub' else 1)


def delta(block, name, typ, state, bound, writes, captures=None):
    """Final and prefix deltas on every arm; an update must read the current Cell."""
    lo = hi = bottom = top = 0
    captures = dict(captures or {})
    for op in block.ops:
        step = increment(op, name, typ, state, bound)
        if step is not None:
            captures[op.results[0].name] = step
        part = (0, 0, 0, 0)
        if op.opcode == 'cf.if':
            arms = [delta(r, name, typ, state, bound, writes, captures) for r in op.regions]
            if len(arms) == 1:
                arms.append((0, 0, 0, 0))
            if any(x is None for x in arms):
                return None
            part = min(x[0] for x in arms), max(x[1] for x in arms), min(x[2] for x in arms), max(x[3] for x in arms)
            if any(name in writes(x) for r in op.regions for x in r.walk()):
                captures.clear()
        elif op.regions:
            if any(name in writes(x) for r in op.regions for x in r.walk()):
                return None
        elif name in writes(op):
            step = captures.get(getattr(op.operands[1], 'name', None)) if op.opcode == 'scalar.set' else None
            if step is None:
                return None
            part = step, step, min(0, step), max(0, step)
            captures.clear()
        bottom, top = min(bottom, lo + part[2]), max(top, hi + part[3])
        lo, hi = lo + part[0], hi + part[1]
    return lo, hi, bottom, top


def finite_bounds(op, state, cells, bound, writes):
    count = trip_count(op, state, bound)
    if count is None:
        return {}, {}
    headers, exits = {}, {}
    body = op.regions[0]
    for name in cells.keys() & state.keys():
        initial = state[name]
        if initial is None:
            continue
        change = delta(body, name, cells[name], state, bound, writes)
        if change is None:
            continue
        lo, hi, bottom, top = change
        if lo == hi == 0:
            continue  # use the ordinary fixed point for unchanged/resetting Cells
        header = initial[0] + min(0, (count - 1) * lo), initial[1] + max(0, (count - 1) * hi)
        low, high = limits(cells[name].dtype)
        if low <= header[0] + bottom <= header[1] + top <= high:
            headers[name] = header
            exits[name] = initial[0] + count * lo, initial[1] + count * hi
    return headers, exits


def thresholds(body, state, bound):
    result = {}
    for op in body.walk():
        if op.opcode != 'scalar.cmp':
            continue
        for cell, other in (op.operands, op.operands[::-1]):
            if not isinstance(cell, Value) or not isinstance(cell.type, CellType) or not cell.type.dtype.is_integer:
                continue
            interval = bound(other, state)
            if interval is not None:
                result.setdefault(cell.name, set()).update(x + d for x in interval for d in (-1, 0, 1))
    return result


def widen(old, new, typ, hints):
    low, high = limits(typ.dtype)
    points = {low, high} | {x for x in hints if low <= x <= high}
    return (max(x for x in points if x <= new[0]) if new[0] < old[0] else old[0],
            min(x for x in points if x >= new[1]) if new[1] > old[1] else old[1])
