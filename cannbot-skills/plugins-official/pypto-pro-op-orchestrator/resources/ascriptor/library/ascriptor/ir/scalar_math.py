# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Typed scalar rules shared by evaluation and validation: exact integer division and A5 vf extrema."""

from .core import Literal, Value
from .types import CellType, ScalarType


def rounding(op):
    value = op.attrs.get("rounding", "floor")
    return getattr(value, "name", value)


def integer_divmod(a: int, b: int, mode: str = "floor") -> tuple[int, int]:
    if mode == "floor":
        return divmod(a, b)
    if mode != "trunc":
        raise ValueError(f"integer rounding must be floor or trunc, got {mode!r}")
    q = abs(a) // abs(b)
    if (a < 0) != (b < 0):
        q = -q
    return q, a - q * b


def limits(dtype):
    if dtype.kind == "bool":
        return 0, 1
    if dtype.kind == "int":
        return -(1 << (dtype.bits - 1)), (1 << (dtype.bits - 1)) - 1
    return 0, (1 << dtype.bits) - 1


def wrap(value, dtype):
    """``value`` reduced two's complement into an integer ``dtype``; b1 keeps truth."""
    if dtype.kind == "bool":
        return bool(value)
    value = int(value) % (1 << dtype.bits)
    return value - (1 << dtype.bits) if dtype.kind == "int" and value >> (dtype.bits - 1) else value


_COMPARE = {"lt": lambda a, b: a < b, "le": lambda a, b: a <= b, "gt": lambda a, b: a > b,
            "ge": lambda a, b: a >= b, "eq": lambda a, b: a == b, "ne": lambda a, b: a != b}
_EXACT = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b,
          "and": lambda a, b: a & b, "or": lambda a, b: a | b, "xor": lambda a, b: a ^ b,
          "min": min, "max": max}


def evaluate(code, args, dtype, *, rounding="floor", n=None, pred=None, overflow="refuse"):
    """The value of integer ``scalar.<code>`` over constant ``args`` with result ``dtype``, or None: do not fold.

    The one table every constant folder reads (RFC-0006, integer scalar simplification). None covers what the typed
    domain leaves out (RFC-0001 §6.10, RFC-0015 shifts): a non-integer operand or result, a zero divisor, the
    signed minimum divided by -1, negated or made absolute, a shift count outside [0, width), a left shift of a
    negative operand or past the unsigned width. ``overflow="refuse"`` also returns None when the exact result
    lies outside ``limits(dtype)``, which is what a folder that prints or substitutes the value needs;
    ``"wrap"`` reduces it into ``dtype`` as the device does. ``cast`` reduces in both: narrowing is never the identity.
    """
    if dtype is None or not dtype.is_integer or not all(isinstance(x, (int, bool)) for x in args):
        return None
    a = args[0] if args else None
    b = n if code == "align" else args[1] if len(args) > 1 else None
    low = limits(dtype)[0]
    if code in ("div", "mod", "ceil_div", "align"):
        if not b or dtype.kind == "int" and a == low and b == -1:
            return None
        result = (-(-a // b) * (b if code == "align" else 1) if code in ("ceil_div", "align")
                  else integer_divmod(a, b, rounding)[code == "mod"])
    elif code in ("shl", "shr"):
        if dtype.kind == "bool" or isinstance(a, bool) or isinstance(b, bool) or not 0 <= b < dtype.bits:
            return None
        if code == "shl" and (a < 0 or (a << b) >> dtype.bits):
            return None
        result = a << b if code == "shl" else wrap(a, dtype) >> b  # arithmetic when signed, logical when not
    elif code in ("neg", "abs"):
        if dtype.kind == "int" and a == low:
            return None
        result = -a if code == "neg" else abs(a)
    elif code == "cast":
        return wrap(a, dtype)
    elif code == "not":
        result = (not a) if dtype.kind == "bool" else ~a
    elif code == "cmp":
        compare = _COMPARE.get(str(getattr(pred, "name", pred)))
        result = compare(a, b) if compare is not None and b is not None else None
    elif code == "select":
        result = args[1 if a else 2] if len(args) == 3 else None
    elif code == "const":
        result = a
    else:
        result = _EXACT[code](a, b) if code in _EXACT and b is not None else None
    if result is None:
        return None
    if dtype.kind == "bool" or overflow == "wrap":
        return wrap(result, dtype)
    return int(result) if low <= result <= limits(dtype)[1] else None


def division_error(op, integer):
    if op.opcode not in ("scalar.div", "scalar.mod", "scalar.ceil_div", "scalar.align") or not op.results:
        return None
    if rounding(op) not in ("floor", "trunc"):
        return "integer rounding must be floor or trunc"
    kind = op.results[0].type
    if not isinstance(kind, ScalarType) or not kind.dtype.is_integer:
        return None
    low, high = limits(kind.dtype)
    args = op.operands if op.opcode != "scalar.align" else (*op.operands, op.attrs.get("n"))
    if len(args) != 2:
        return None  # Structural operand diagnostics belong to the registry verifier.
    typed = [arg.type.dtype for arg in args if isinstance(arg, Value) and isinstance(arg.type, (ScalarType, CellType))]
    if typed and kind.dtype not in typed:
        return "integer division result dtype must match an operand; cast operands before widening"
    for arg in args:
        if isinstance(arg, Value) and isinstance(arg.type, (ScalarType, CellType)):
            dt = arg.type.dtype
            if not dt.is_integer or limits(dt)[0] < low or limits(dt)[1] > high:
                return "integer division operands need a lossless common type; use an explicit cast"
        else:
            value = arg.value if isinstance(arg, Literal) else arg
            if not isinstance(value, int) or not low <= value <= high:
                return "integer division literal does not fit the result type; use an explicit cast"
    a, b = (integer(x) for x in args)
    if b == 0:
        return "integer division by zero"
    if kind.dtype.kind == "int" and a == low and b == -1:
        return "integer division quotient overflows: signed minimum / -1"
    if op.opcode == "scalar.align":
        if b is not None and b <= 0:
            return "integer alignment must be positive"
        if a is not None and b is not None and not low <= -(-a // b) * b <= high:
            return "integer alignment result overflows its type"
    return None


def vf_extremum_error(op, kind, family):
    """RFC-0001 §6.16: an f32 ``scalar.max`` / ``scalar.min`` in a ``vf`` function is not valid on an A5 device."""
    result = op.results[0].type if op.opcode in ("scalar.max", "scalar.min") and op.results else None
    if kind != "vf" or family != "a5" or not isinstance(result, ScalarType) or result.dtype.name != "f32":
        return None
    return (f"{op.opcode} on f32 is not valid A5 IR in a vf: A5 has no vector-function spelling for f32 min/max, "
            "and its compiler rejects max() there (\"max() in vector function only supports integer types\", "
            "RFC-0001 §6.16)")
