# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Program-point integer Cell bounds with conservative structured-loop fixed points."""

from collections.abc import Mapping

from .core import Literal, Value
from .registry import REGISTRY
from .scalar_loops import finite_bounds, thresholds, widen
from .scalar_math import limits
from .scalar_range import ScalarRanges
from .types import CellType, ScalarType


def values(obj):
    if isinstance(obj, Value):
        yield obj
    elif isinstance(obj, Mapping):
        for item in obj.values():
            yield from values(item)
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            yield from values(item)


def cell_writes(op):
    """Explicit registry writes, implicit mask counters and conservative call escapes."""
    spec = REGISTRY.get(op.opcode)
    result = set()
    for i, operand in enumerate(op.operands):
        slot = spec.operands[min(i, len(spec.operands) - 1)]
        if slot.access in ("write", "readwrite") or op.opcode in ("cf.call", "simt.launch"):
            result.update(v.name for v in values(operand) if isinstance(v.type, CellType))
    for name, attr in op.attrs.items():
        if spec.attr(name).access in ("write", "readwrite") or op.opcode == "vf.mask_update" and name == "cnt":
            result.update(v.name for v in values(attr) if isinstance(v.type, CellType))
    return result


def snapshot_cell(op):
    if op.opcode != "scalar.add" or len(op.operands) != 2 or not op.results:
        return None
    a, b = op.operands
    if isinstance(a, Literal) and a.value == 0:
        a, b = b, a
    if (isinstance(a, Value) and isinstance(a.type, CellType) and isinstance(b, Literal) and b.value == 0
            and isinstance(op.results[0].type, ScalarType) and a.type.dtype == op.results[0].type.dtype
            and a.type.dtype.is_integer):
        return a
    return None


def fit(interval, typ):
    if interval is None or not isinstance(typ, (ScalarType, CellType)) or not typ.dtype.is_integer:
        return None
    low, high = limits(typ.dtype)
    return interval if low <= interval[0] <= interval[1] <= high else None


def join(left, right, keys):
    if left is None:
        return {k: right.get(k) for k in keys} if right is not None else None
    if right is None:
        return {k: left.get(k) for k in keys}
    return {k: (min(left[k][0], right[k][0]), max(left[k][1], right[k][1]))
            if left.get(k) is not None and right.get(k) is not None else None for k in keys}


class CellRanges:
    """Facts are attached to the original op objects, not just variable names."""

    def __init__(self, function, *, bindings=None, core_ranges=None):
        self.before = {}
        self.snapshots = {}
        self.core_ranges = core_ranges or {}
        self.cells = {v.name: v.type for op in function.walk() for v in op.results if isinstance(v.type, CellType)}
        self.cells.update((p.name, p.type) for p in function.params if isinstance(p.type, CellType))
        state = {}
        for p in function.params:
            value = (bindings or {}).get(p.name)
            state[p.name] = fit((value, value), p.type) if type(value) is int else None
        self.walk(function.body, state, record=True)

    @staticmethod
    def bound(value, state):
        return ScalarRanges(definitions={}, facts=state, allow_cells=True).bounds(value)

    def refine(self, state, comparison, truth):
        if comparison is None or comparison.opcode != "scalar.cmp":
            return state.copy()
        a, b = comparison.operands
        pred = str(comparison.attrs["pred"])
        if not isinstance(a, Value) or not isinstance(a.type, CellType):
            a, b = b, a
            pred = {"lt": "gt", "le": "ge", "gt": "lt", "ge": "le", "eq": "eq", "ne": "ne"}[pred]
        if not isinstance(a, Value) or not isinstance(a.type, CellType):
            return state.copy()
        interval, other = self.bound(a, state), self.bound(b, state)
        if interval is None or other is None:
            return state.copy()
        if not truth:
            pred = {"lt": "ge", "le": "gt", "gt": "le", "ge": "lt", "eq": "ne", "ne": "eq"}[pred]
        lo, hi = interval
        if pred == "eq":
            lo, hi = max(lo, other[0]), min(hi, other[1])
        elif pred == "ne" and other[0] == other[1]:
            constant = other[0]
            lo += lo == constant
            hi -= hi == constant
        elif pred in ("lt", "le"):
            hi = min(hi, other[1] - (pred == "lt"))
        elif pred in ("gt", "ge"):
            lo = max(lo, other[0] + (pred == "gt"))
        if lo > hi:
            return None
        return {**state, a.name: (lo, hi)}

    def loop(self, op, state, record):
        body = op.regions[0]
        written = set().union(*(cell_writes(x) for x in body.walk()))
        carried = self.cells.keys() & state.keys()
        header = state.copy()
        induction = ScalarRanges(definitions={op.results[0].name: op}, facts=state, allow_cells=True).bounds(op.results[0])
        if any(isinstance(x, Value) and isinstance(x.type, CellType) and x.name in written for x in op.operands):
            induction = None
        header[op.results[0].name] = induction
        unhandled_exit = any(x.opcode in ("cf.break", "cf.continue", "cf.return") for x in body.walk())
        finite, exits = {}, {}
        if unhandled_exit:
            for name in written:
                header[name] = None
        else:
            if induction is not None:
                finite, exits = finite_bounds(op, state, self.cells, self.bound, cell_writes)
                header.update(finite)
            hints = thresholds(body, state, self.bound)
            # Interval iteration starts with the entry state (including a possible
            # zero-trip path). Widen only moving endpoints to the type limits.
            # Rechecking the widened body must still prove updates representable;
            # an unguarded overflowing increment will discard the entire fact.
            iteration = 0
            while True:
                after = self.walk(body, header.copy(), record=False)
                merged = join(header, after, carried)
                changed = {k for k in carried if k not in finite and merged[k] != header.get(k)}
                if not changed:
                    break
                for name in changed:
                    old, new = header.get(name), merged[name]
                    if iteration >= 8 and old is not None and new is not None:
                        new = widen(old, new, self.cells[name], hints.get(name, ()))
                    header[name] = new
                iteration += 1
        self.walk(body, header.copy(), record=record)
        out = state.copy()
        for name in carried:
            out[name] = None if unhandled_exit and name in written else exits.get(name, header.get(name))
        return out

    def walk(self, block, state, *, record):
        previous = None
        for op in block.ops:
            if state is None:
                break
            if record:
                self.before[id(op)] = {name: state[name] for name in self.cells.keys() & state.keys()
                                       if state[name] is not None}
            if op.opcode == "cf.for":
                state = self.loop(op, state, record)
            elif op.opcode == "cf.if":
                condition = self.bound(op.operands[0], state)
                # Only the immediately preceding comparison is known to read the
                # current Cell version. An older predicate remains a snapshot.
                comparison = previous if previous is not None and previous.results == op.operands else None
                branches = []
                for arm in (0, 1):
                    incoming = self.refine(state, comparison, arm == 0)
                    if condition is not None and condition[0] == condition[1] and bool(condition[0]) != (arm == 0):
                        incoming = None
                    branches.append(self.walk(op.regions[arm], incoming, record=record)
                                    if incoming is not None and arm < len(op.regions) else incoming)
                state = join(*branches, state.keys())
            elif op.opcode in ("cf.break", "cf.continue", "cf.return"):
                return None
            elif op.regions:
                # Unknown region semantics cannot establish Cell or snapshot facts.
                state = {name: None if name in self.cells else value for name, value in state.items()}
            elif op.opcode == "scalar.cell":
                result = op.results[0]
                state[result.name] = fit(self.bound(op.attrs.get("init", 0), state), result.type)
            elif op.opcode == "scalar.set":
                cell, value = op.operands
                state[cell.name] = fit(self.bound(value, state), cell.type)
            else:
                for result in op.results:
                    if not isinstance(result.type, ScalarType) or not result.type.dtype.is_integer:
                        continue
                    state.pop(result.name, None)
                    interval = self.core_ranges.get(op.opcode)
                    if interval is None:
                        interval = ScalarRanges(definitions={result.name: op}, facts=state, allow_cells=True).bounds(result)
                    state[result.name] = fit(interval, result.type)
                    if record:
                        self.snapshots[result.name] = state[result.name]
                for name in cell_writes(op):
                    state[name] = None
            previous = op
        return state
