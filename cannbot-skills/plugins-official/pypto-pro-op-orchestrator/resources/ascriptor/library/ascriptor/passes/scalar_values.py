# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Typed initializer equalities and quotient/remainder identities."""

from ..ir import Ident, Literal, Value
from ..ir.scalar_math import integer_divmod, limits, rounding
from ..ir.types import CellType, ScalarType


def initializer(op, current, resolve):
    """Capture an equality, never an alias to a live Cell or an implicit cast."""
    cell = op.results[0]
    if not isinstance(cell.type, CellType) or not cell.type.dtype.is_integer:
        return None
    value = op.attrs.get("init", 0)
    if isinstance(value, Value) and isinstance(value.type, CellType):
        value = current.get(value.name)
    value = resolve(value)
    if isinstance(value, Value):
        return value if isinstance(value.type, ScalarType) and value.type.dtype == cell.type.dtype else None
    literal = value.value if isinstance(value, Literal) else value
    lo, hi = limits(cell.type.dtype)
    return Literal(literal) if isinstance(literal, (int, bool)) and lo <= literal <= hi else None


def remainder_identity(op, definitions, ranges, rw):
    """Match a - trunc/floor(a/d)*d without changing any overflowing intermediate."""
    if op.opcode != "scalar.sub":
        return op
    dividend, product = op.operands
    typ = op.results[0].type
    if not isinstance(dividend, Value) or dividend.type != typ or not isinstance(product, Value) or product.type != typ:
        return op  # in particular, live Cells are not immutable dividend snapshots
    mult = definitions.get(product.name)
    if mult is None or mult.opcode != "scalar.mul":
        return op
    for quotient, divisor in (mult.operands, mult.operands[::-1]):
        if not isinstance(quotient, Value) or quotient.type != typ or not isinstance(divisor, Literal):
            continue
        constant = divisor.value
        low, high = limits(typ.dtype)
        if type(constant) is not int or not 0 < constant <= high:
            continue
        division = definitions.get(quotient.name)
        if division is None or division.opcode != "scalar.div" or division.operands != (dividend, divisor):
            continue
        mode = rounding(division)
        minimum, maximum = ranges.bounds(dividend) or (low, high)
        qlo = integer_divmod(minimum, constant, mode)[0]
        qhi = integer_divmod(maximum, constant, mode)[0]
        if not low <= qlo * constant <= qhi * constant <= high:
            continue
        # The identity itself bounds the subtraction result, even when independent
        # interval subtraction would lose correlation. Nonnegative trunc == floor.
        mode = "floor" if minimum >= 0 else mode
        return rw.rewritten(op, "recover typed remainder from quotient and product", opcode="scalar.mod",
                            operands=(dividend, divisor), attrs={"rounding": Ident(mode)})
    return op
