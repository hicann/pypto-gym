# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Preserve declared integer widths at the PyPTO scalar boundary."""

from ...ir import Literal, Value
from ...ir.types import CellType, ScalarType

INTEGER_NAMES = frozenset({"i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64"})


def integer_type(value):
    typ = getattr(value, "type", None)
    if isinstance(typ, (ScalarType, CellType)) and typ.dtype.name in INTEGER_NAMES:
        return typ.dtype
    return None


def supports_cast(op):
    source = op.operands[0]
    literal = source.value if isinstance(source, Literal) else source
    return integer_type(op.results[0]) is not None and (
        integer_type(source) is not None or type(literal) is int)


def vf_divmod_operands(op, ref, pl_dtype):
    """Return typed spellings only when every operand is a <=32-bit integer.

    Wide values remain wide even when a hand-built IR result is narrower.
    Untyped literals inherit the result dtype only when they are representable.
    """
    result = integer_type(op.results[0])
    if result is None or result.bits > 32:
        return None
    operands = []
    for value in op.operands:
        if isinstance(value, Value):
            dtype = integer_type(value)
            if dtype is None or dtype.bits > 32:
                return None
            operands.append(f"pl.cast({ref(value)}, {pl_dtype(dtype.name)})")
        else:
            literal = value.value if isinstance(value, Literal) else value
            signed = result.kind == "int"
            low = -(1 << (result.bits - 1)) if signed else 0
            high = (1 << (result.bits - int(signed))) - 1
            if type(literal) is not int or not low <= literal <= high:
                return None
            operands.append(f"pl.const({literal}, {pl_dtype(result.name)})")
    return tuple(operands)
