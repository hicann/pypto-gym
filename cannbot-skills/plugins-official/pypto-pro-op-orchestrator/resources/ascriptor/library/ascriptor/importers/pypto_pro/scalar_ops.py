# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Typed scalar bitwise, shift, extremum, negation and logical-not expressions.

Pro promotes both binary operands with explicit casts and prints C++ operators, so
operand types must already equal the result type; C++17 shift domains are checked
statically here and dynamically by the model.
"""
import math
from fractions import Fraction

from ...ir import Ident, Value
from ...ir.types import CellType, ScalarType

BINARY = {'Min': 'min', 'Max': 'max', 'BitAnd': 'and', 'BitOr': 'or', 'BitXor': 'xor',
          'BitShiftLeft': 'shl', 'BitShiftRight': 'shr'}
UNARY = {'Neg', 'Not', 'BitNot'}
TARGETS = {**{kind: ('scalar.' + name,) for kind, name in BINARY.items()},
           'Neg': ('scalar.neg',), 'BitNot': ('scalar.not',), 'Not': ('scalar.not',)}
INTEGERS = {'i32', 'i64'}


def printed_float(value):
    """The FP32 value of Pro CCE's finite literal spelling, ``std::to_string(double) + "f"``.

    ``%f`` keeps six fractional digits; the compiler then rounds that decimal to binary32.
    """
    if not math.isfinite(value):
        return value  # Printed as NaN/infinity builtins.
    text = f'{value:f}'
    magnitude = abs(Fraction(text))
    if magnitude:
        exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
        exponent -= magnitude < Fraction(2) ** exponent
        quantum = max(exponent - 23, -149)
        magnitude = math.ldexp(round(magnitude / Fraction(2) ** quantum), quantum)
        magnitude = math.inf if magnitude >= 2.0 ** 128 else magnitude
    return -float(magnitude) if text.startswith('-') else float(magnitude)


def scalar(value):
    """The scalar type of a value or a readable cell."""
    if isinstance(value, Value):
        return ScalarType(value.type.dtype) if isinstance(value.type, CellType) else value.type
    return None


def typed(value, typ):
    """A value (or cell) of exactly ``typ`` or a literal inside its domain."""
    if isinstance(value, Value):
        return scalar(value) == typ
    name = typ.dtype.name
    if name == 'f32':
        return type(value) is float
    return type(value) is int and name in INTEGERS and -(1 << (typ.dtype.bits - 1)) <= value < 1 << (typ.dtype.bits - 1)


def binary(ctx, node, kind, a, b, typ):
    o, name = ctx.o, typ.dtype.name
    if kind in {'Min', 'Max'} and name == 'f32':
        # A5 measured IEEE maximum/minimum (RFC-0001 §6.16) for Pro's builtin in vector and cube sections. In a VF
        # body the op is invalid A5 IR, so the module check refuses it at each site.
        o.need(ctx.simt is None, node, 'FP32 scalar min/max in a SIMT function is unmeasured on A5')
    else:
        o.need(name in INTEGERS, node, f'{kind} admits INT32/INT64 scalars')
    o.need(typed(a, typ) and typed(b, typ), node, f'{kind} operands must already have the promoted {name} type')
    if kind in {'BitShiftLeft', 'BitShiftRight'}:
        o.need(not isinstance(b, int) or 0 <= b < typ.dtype.bits, node, 'Constant shift count is outside [0, width)')
        o.need(kind == 'BitShiftRight' or not isinstance(a, int) or a >= 0, node,
               'Left shift of a negative constant is undefined in C++17')
    return ctx.emit('scalar.' + BINARY[kind], node, (a, b), typ=typ)


def unary(ctx, node, kind, a, typ):
    o = ctx.o
    if kind == 'Not':
        o.need(typ.dtype.name == 'b1', node, 'Logical not produces BOOL')
        operand = scalar(a)
        if operand == typ:
            return ctx.emit('scalar.not', node, (a,), typ=typ)
        o.need(isinstance(operand, ScalarType) and operand.dtype.name in INTEGERS | {'f32'}, node,
               'Logical not admits BOOL, INT32/INT64 and FP32 scalars')
        # C++ `!x` on an arithmetic value is `x == 0`; FP32 NaN is nonzero and -0 compares equal.
        zero = 0.0 if operand.dtype.name == 'f32' else 0
        return ctx.emit('scalar.cmp', node, (a, zero), typ=typ, attrs={'pred': Ident('eq')})
    allowed = INTEGERS | ({'f32'} if kind == 'Neg' else set())
    o.need(typ.dtype.name in allowed and typed(a, typ), node, f'{kind} requires a matching scalar of an admitted type')
    return ctx.emit('scalar.neg' if kind == 'Neg' else 'scalar.not', node, (a,), typ=typ)
