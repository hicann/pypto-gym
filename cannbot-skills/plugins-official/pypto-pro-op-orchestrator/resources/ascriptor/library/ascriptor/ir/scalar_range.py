# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Immutable integer bounds; Cell facts require an explicit program point."""

from .core import Literal, Value
from .scalar_math import integer_divmod, limits, rounding
from .types import CellType, ScalarType


class ScalarRanges:
    def __init__(self, function=None, *, definitions=None, facts=None, allow_cells=False):
        self.definitions = definitions if definitions is not None else {
            value.name: op for op in function.walk() for value in op.results}
        self.facts = facts if facts is not None else {}
        self.function = function
        self.allow_cells = allow_cells
        self._snapshots = None

    def bounds(self, value, seen=frozenset()):
        if isinstance(value, Literal):
            value = value.value
        if isinstance(value, (int, bool)):
            return int(value), int(value)
        if (not isinstance(value, Value) or not isinstance(value.type, (ScalarType, CellType))
                or not value.type.dtype.is_integer or isinstance(value.type, CellType) and not self.allow_cells):
            return None
        if value.name in seen:
            return None
        if value.name in self.facts:
            lo, hi = limits(value.type.dtype)
            result = self.facts[value.name]
            return result if result is not None and lo <= result[0] <= result[1] <= hi else None
        if isinstance(value.type, CellType):
            return None  # Cell bounds are valid only at an explicitly supplied program point
        op = self.definitions.get(value.name)
        if op is None:
            return None
        if self.function is not None and any(isinstance(x, Value) and isinstance(x.type, CellType) for x in op.operands):
            from .scalar_flow import CellRanges
            if self._snapshots is None:
                self._snapshots = CellRanges(self.function).snapshots
            return self._snapshots.get(value.name)
        seen = seen | {value.name}
        args = [self.bounds(x, seen) for x in op.operands]
        result = None
        if op.opcode == "scalar.const":
            constant = op.attrs.get("value")
            if isinstance(constant, (int, bool)):
                result = int(constant), int(constant)
        elif op.opcode in ("core.cube_idx", "core.vec_idx", "core.sub_block_idx"):
            result = 0, limits(value.type.dtype)[1]
        elif op.opcode == "scalar.mod" and rounding(op) == "floor" and args[1] is not None:
            lo, hi = args[1]
            if lo == hi and lo > 0:
                result = 0, lo - 1
        elif op.opcode == "scalar.and":
            masks = [a[0] for a in args if a is not None and a[0] == a[1] and a[0] >= 0]
            if masks:
                result = 0, min(masks)
        elif op.opcode == "scalar.cast" and args[0] is not None:
            result = args[0]
        elif op.opcode == "cf.for" and all(a is not None and a[0] == a[1] for a in args):
            start, stop, step = (a[0] for a in args)
            if step and (stop - start) * step > 0:
                last = start + ((abs(stop - start) - 1) // abs(step)) * step
                result = min(start, last), max(start, last)
        elif op.opcode == "scalar.neg" and args[0] is not None:
            result = -args[0][1], -args[0][0]
        elif op.opcode == "scalar.select":
            if args[0] is not None and args[0][0] == args[0][1]:
                result = args[1 if args[0][0] else 2]
            elif args[1] is not None and args[2] is not None:
                result = min(args[1][0], args[2][0]), max(args[1][1], args[2][1])
        elif len(args) == 2 and all(a is not None for a in args):
            (a, b), (c, d) = args
            code = op.opcode
            if code == "scalar.add":
                result = a + c, b + d
            elif code == "scalar.sub":
                result = a - d, b - c
            elif code == "scalar.mul":
                products = (a * c, a * d, b * c, b * d)
                result = min(products), max(products)
            elif code in ("scalar.min", "scalar.max"):
                fn = min if code == "scalar.min" else max
                result = fn(a, c), fn(b, d)
            elif code == "scalar.div" and c == d and c > 0:
                result = (integer_divmod(a, c, rounding(op))[0], integer_divmod(b, c, rounding(op))[0])
            elif code == "scalar.ceil_div" and c == d and c > 0:
                result = -(-a // c), -(-b // c)
            elif code == "scalar.mod" and rounding(op) == "trunc" and c == d and c > 0:
                result = (0 if a >= 0 else -(c - 1), 0 if b <= 0 else c - 1)
            elif code == "scalar.cmp":
                pred = str(op.attrs["pred"])
                yes = {"lt": b < c, "le": b <= c, "gt": a > d, "ge": a >= d,
                       "eq": a == b == c == d, "ne": b < c or d < a}[pred]
                no = {"lt": a >= d, "le": a > d, "gt": b <= c, "ge": b < c,
                      "eq": b < c or d < a, "ne": a == b == c == d}[pred]
                result = (1, 1) if yes else (0, 0) if no else (0, 1)
        if result is None:
            return None
        lo, hi = limits(value.type.dtype)
        return result if lo <= result[0] <= result[1] <= hi else None

    def normalized(self, value, slots):
        interval = self.bounds(value)
        return type(slots) is int and slots > 0 and interval is not None and 0 <= interval[0] <= interval[1] < slots
