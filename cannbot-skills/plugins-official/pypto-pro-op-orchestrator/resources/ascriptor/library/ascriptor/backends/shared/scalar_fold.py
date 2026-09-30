# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Constant-fold a Lowered-IR scalar value against one valuation of the kernel's parameters.

A backend whose tile shapes are compile-time quantities (``pto::Tile`` capacities are template
arguments; `pl` tile types are traced) needs an integer for a dim written as ``%div``, where
``div = scalar.div(tokens, 2)`` and ``tokens`` is a kernel parameter. This folder walks that SSA
chain: leaves are literals or bound parameters, interior nodes are the pure integer ``scalar.*``
ops, and anything else — a cell, a core id, a loop variable, a memory read — folds to ``None``.

Two levels, and the second one was not optional. Value-level folding covers a dim named by a
parameter or derived from one. It does **not** cover a *sliced extent*: ``x[r:r+1, :]`` lowers to
``(r + 1) - r``, which is the constant 1 only if the two occurrences of ``r`` are known to be the
same value — a correlation value-level folding throws away. So :meth:`ScalarFolder.linear` keeps
an affine form, for the same reason ``pypto_pro``'s ``ScalarEnv`` carries one.

The original design note here claimed a C++ target needs only constant folding, because a
``pto::Tile``'s **capacity** must be constant while its **valid region** may be a runtime
expression. That is true of quantities a kernel merely computes with, and false of shapes: the
sliced extent above *is* a tile shape. Measured, the affine form took the fully-static GM ↔ UB
transfers from 141 to 326. What stays out of scope is the rest of ``ScalarEnv`` — interval
arithmetic, loop-variable progressions, core-id ranges — which exists to fold quantities `pl`
must have at trace time and this backend can leave as runtime C++.

Semantics follow the reference interpreter op by op (``backends/sim/interp.py``), because a dim
folded differently here than the interpreter computes it would size a tile the goldens never
used. The values come from the one table every folder shares (``ir.scalar_math.evaluate``):
integer quotient/remainder in the explicit floor/trunc mode (RFC-0001 §6.10), a result outside
its declared type refused rather than carried as an unbounded int, a narrowing cast reduced.
"""

from __future__ import annotations

from typing import Any

from ...ir import Function, Literal, Op
from ...ir.scalar_math import evaluate, limits, rounding

#: opcodes whose result is only known at run time; folding stops here (never guesses).
_RUNTIME = ("core.", "scalar.load", "scalar.cell", "scalar.set", "list.", "cf.")

#: Negative right shift remains outside this folder's qualified domain.
_SIGN_SENSITIVE = frozenset({"shr"})


class ScalarFolder:
    """Folds scalar values of one function against ``bindings`` (parameter name -> int)."""

    def __init__(self, fn: Function, bindings: dict[str, int] | None = None) -> None:
        self.fn = fn
        self.bindings = dict(bindings or {})
        self.defs: dict[str, Op] = {r.name: op for op in fn.body.walk() for r in op.results}
        self._cache: dict[str, int | None] = {}
        self._active: set[str] = set()  # recursion guard: an SSA cycle folds to None

    def fold(self, x: Any) -> int | None:
        """A compile-time int for ``x``, or None when it is not one."""
        if isinstance(x, Literal):
            x = x.value
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, int):
            return x
        if isinstance(x, float):
            return int(x) if x.is_integer() else None
        name = getattr(x, "name", None)
        if not isinstance(name, str):
            return None
        direct = self._value(name)
        if direct is not None:
            return direct
        # Value-level folding alone is not enough for a *sliced extent*. `x[r:r+1, :]` lowers to
        # ``(r + 1) - r``: constant 1, but only if the two occurrences of ``r`` are known to be the
        # same value. Interval arithmetic loses that correlation, so the linear form keeps it —
        # the same reason pypto_pro's ScalarEnv carries one. Without this a one-row transfer looks
        # like a run-time row count and would be given a DYNAMIC valid region it does not need.
        lin = self.linear(x)
        if lin is not None and not lin[1]:
            dt = getattr(getattr(x, "type", None), "dtype", None)
            # exact ints walk the affine form: a constant its declared type cannot hold is no fold
            return lin[0] if dt is None or not dt.is_integer or limits(dt)[0] <= lin[0] <= limits(dt)[1] else None
        return None

    def linear(self, x: Any, _seen: frozenset[str] = frozenset()) -> tuple[int, dict[str, int]] | None:
        """``(constant, {value name: coefficient})`` for ``x`` as an affine form, or None.

        Only the operations that stay affine are walked: add, sub, neg, and multiplication where
        one side is constant. Anything else terminates as an opaque variable, which is sound — a
        variable with a coefficient simply keeps the form non-constant.
        """
        if isinstance(x, Literal):
            x = x.value
        if isinstance(x, bool):
            return (int(x), {})
        if isinstance(x, int):
            return (x, {})
        name = getattr(x, "name", None)
        if not isinstance(name, str):
            return None
        k = self._value(name)
        if k is not None:
            return (k, {})
        if name in _seen:  # an SSA cycle: not an affine form
            return None
        d = self.defs.get(name)
        opaque = (0, {name: 1})  # not decomposable: the variable itself
        if d is None or not d.opcode.startswith("scalar."):
            return opaque
        kind = d.opcode[len("scalar."):]
        if kind not in ("add", "sub", "mul", "neg"):
            return opaque
        seen = _seen | {name}
        parts = [self.linear(o, seen) for o in d.operands]
        if any(p is None for p in parts):
            return opaque
        if kind == "neg" and len(parts) == 1:
            b, cs = parts[0]
            return (-b, {v: -c for v, c in cs.items()})
        if len(parts) != 2:
            return opaque
        (b0, c0), (b1, c1) = parts
        if kind in ("add", "sub"):
            sign = 1 if kind == "add" else -1
            out = dict(c0)
            for v, c in c1.items():
                out[v] = out.get(v, 0) + sign * c
                if out[v] == 0:
                    del out[v]
            return (b0 + sign * b1, out)
        # mul stays affine only when one side is a constant
        if not c1:
            return (b0 * b1, {v: c * b1 for v, c in c0.items()}) if b1 or not c0 else (0, {})
        if not c0:
            return (b0 * b1, {v: c * b0 for v, c in c1.items()}) if b0 or not c1 else (0, {})
        return opaque

    #: forms whose result is **non-decreasing** in the operand being bounded, when the other
    #: operand is a non-negative constant, with the ceiling that follows. `sub` is here for
    #: `a - k` only — `k - a` decreases in `a`, and a lower bound is not tracked.
    _MONOTONE = {
        "add": lambda a, k: a + k,
        "sub": lambda a, k: a - k,
        "mul": lambda a, k: a * k,
        "div": lambda a, k: a // k,
        "ceil_div": lambda a, k: -(-a // k),
    }
    #: the same, but commutative, so the constant may be on either side
    _COMMUTES = frozenset({"add", "mul"})

    def bound(self, x: Any, _depth: int = 0) -> int | None:
        """The largest value ``x`` can take, or None when that is not knowable.

        Weaker than :meth:`fold` and used where a compile-time *capacity* is needed but the exact
        value is not: a ``pto::Tile``'s Rows / Cols must be template arguments even when its valid
        region is set at run time, so a tail-clamped extent needs its ceiling.

        Only forms that carry a ceiling on their face are read, and only ones **non-decreasing** in
        the operand being bounded: a `min`, a `max` whose both sides are bounded, and the constant-
        scaled arithmetic the corpus wraps around a tail clamp. That last part is what makes this
        useful at all — the idiom is `align16(min(tail, BLK))`, which arrives as
        `mul(ceil_div(min(…), 16), 16)`, so a walk that stops at the outermost node stops before
        the `min` that carries the ceiling and refuses a capacity that is written on the face of
        the expression (§8 q5, 197 ops).

        The direction is the whole safety argument. A bound that is wrong *upwards* is merely
        wasteful — a tile larger than the transfer. One that is wrong *downwards* sizes a tile
        smaller than the transfer, and PTO would not catch that until its run-time assert. So the
        division-like forms additionally require a non-negative bound to propagate: the
        interpreter's `div` is a floor, `fold` already refuses negative operands there, and
        `(-1) // 8 == -1` in Python against `0` in C is exactly the disagreement that would move a
        ceiling the wrong way.
        """
        exact = self.fold(x)
        if exact is not None:
            return exact
        name = getattr(x, "name", None)
        if not isinstance(name, str) or _depth > 16:
            return None
        d = self.defs.get(name)
        if d is None or not d.opcode.startswith("scalar."):
            return None
        kind = d.opcode[len("scalar."):]
        if kind == "select" and len(d.operands) == 3:
            bounds = [self.bound(o, _depth + 1) for o in d.operands[1:]]
            return max(bounds) if None not in bounds else None
        if kind == "min":
            # bounded by whichever side has a bound; take the tighter of the two
            sides = [self.bound(o, _depth + 1) for o in d.operands[:2]]
            known = [b for b in sides if b is not None]
            return min(known) if known else None
        if kind == "max":
            # bounded only when *both* sides are: an unbounded side is unbounded above
            sides = [self.bound(o, _depth + 1) for o in d.operands[:2]]
            return max(sides) if len(sides) == 2 and None not in sides else None
        if kind == "align":
            n = d.attrs.get("n")
            a = self.bound(d.operands[0], _depth + 1) if d.operands else None
            if a is None or a < 0 or not isinstance(n, int) or n <= 0:
                return None
            return -(-a // n) * n
        fn = self._MONOTONE.get(kind)
        if fn is None or len(d.operands) != 2:
            return None
        k = self.fold(d.operands[1])
        var = d.operands[0]
        if k is None and kind in self._COMMUTES:
            k, var = self.fold(d.operands[0]), d.operands[1]
        if k is None or k < 0 or (k == 0 and kind in ("div", "ceil_div")):
            return None
        a = self.bound(var, _depth + 1)
        if a is None or (a < 0 and kind in ("div", "ceil_div")):
            return None
        return fn(a, k)

    def _value(self, name: str) -> int | None:
        if name in self._cache:
            return self._cache[name]
        if name in self._active:  # a cycle through a loop-carried value: not a constant
            return None
        if name in self.bindings:
            self._cache[name] = int(self.bindings[name])
            return self._cache[name]
        self._active.add(name)
        try:
            out = self._from_def(self.defs.get(name))
        finally:
            self._active.discard(name)
        self._cache[name] = out
        return out

    def _from_def(self, d: Op | None) -> int | None:
        if d is None:  # a parameter with no binding, or a block argument
            return None
        code = d.opcode
        if code.startswith(_RUNTIME):
            return None
        if not code.startswith("scalar.") or not d.results:
            return None
        kind = code[len("scalar."):]
        dt = getattr(d.results[0].type, "dtype", None)
        if kind == "const":
            v = getattr(d.attrs.get("value"), "value", d.attrs.get("value"))
            if isinstance(v, float) and dt is not None and dt.is_float:
                return int(v) if v.is_integer() else None  # as ``fold`` reads a float literal
            return _int(evaluate(kind, [v], dt))
        args = [self.fold(o) for o in d.operands]
        if any(a is None for a in args):
            return None
        if kind in _SIGN_SENSITIVE and any(a < 0 for a in args):
            return None  # the tree's two spellings disagree on negatives; refuse
        n = d.attrs.get("n")
        if kind == "align" and (not isinstance(n, int) or n <= 0 or len(args) != 1):
            return None
        return _int(evaluate(kind, args, dt, rounding=rounding(d), n=n, pred=d.attrs.get("pred")))


def _int(value: int | bool | None) -> int | None:
    return None if value is None else int(value)
