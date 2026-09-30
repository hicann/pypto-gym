# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Expand mathematical integer division into native truncating arithmetic."""

from ..ir import LOWERED, Ident, Literal, Module, Rewriter, Value
from ..ir.scalar_math import rounding
from ..ir.fix_bounds import StaticMemory
from ..ir.types import ScalarType, dtype
from .manager import Pass, PassContext

PASS = "integer_division"


def run(module: Module, ctx: PassContext) -> Module:
    rw = Rewriter(module, PASS)
    names = {v.name for f in module.functions for v in f.params}
    names.update(v.name for f in module.functions for op in f.walk() for v in op.results)
    serial = 0
    constants = {}
    for function in module.functions:
        folder = StaticMemory(function)
        for op in function.walk():
            if op.results and op.opcode in ("scalar.div", "scalar.mod", "scalar.ceil_div", "scalar.align"):
                constants[op.id] = folder.integer(op.results[0])

    def rewrite(op):
        nonlocal serial
        code = op.opcode
        if code not in ("scalar.div", "scalar.mod", "scalar.ceil_div", "scalar.align"):
            return None
        kind = op.results[0].type
        if not isinstance(kind, ScalarType) or not kind.dtype.is_integer:
            return None
        if constants.get(op.id) is not None:
            return [rw.rewritten(op, "Fold exact integer arithmetic before native division lowering",
                                 opcode="scalar.const", operands=(), attrs={"value": constants[op.id]})]
        if code in ("scalar.div", "scalar.mod") and rounding(op) == "trunc":
            return None
        if code in ("scalar.div", "scalar.mod") and ctx.option("native_floor_divmod", False):
            # The backend's own `/` and `%` round toward negative infinity for signed integers, which
            # is this op. Expanding it here would print those very operators as the truncating half of
            # a correction the reader then applies a second time - wrong for every negative dividend
            # whose remainder is not zero, and measured so on hardware.
            ctx.explain.note("Native floor division needs no correction sequence", op=op.id)
            return None
        if code == "scalar.mod" and ctx.option("compact_integer_mod", False) and module.attrs.get("scalar_simplify", False):
            # CCE/PTO print the semantic operation through the same typed helper.
            # PyPTO still receives the explicit truncation/correction sequence.
            ctx.explain.note("Keep floor remainder as a compact typed helper call", op=op.id)
            return None
        out = []
        note = "Implement integer rounding with native quotient/remainder without overflowing a+b-1"

        def emit(code, args, *, attrs=None, typ=kind, result=None):
            nonlocal serial
            if result is None:
                while True:
                    serial += 1
                    name = f"idiv_{serial}"
                    if name not in names:
                        names.add(name)
                        break
                result = Value(name, typ)
            if result == op.results[0]:
                operands = tuple(x if isinstance(x, (Value, Literal)) else Literal(x) for x in args)
                out.append(rw.rewritten(op, note, opcode=code, operands=operands, results=(result,), attrs=attrs or {}))
            else:
                out.append(rw.make(code, args, results=(result,), attrs=attrs or {}, from_ops=(op,), note=note, loc=op.loc))
            return result

        a = op.operands[0]
        b = op.attrs["n"] if code == "scalar.align" else op.operands[1]
        unsigned = kind.dtype.kind in ("uint", "bool")
        if unsigned and code in ("scalar.div", "scalar.mod"):
            return [rw.rewritten(op, note, attrs={**op.attrs, "rounding": Ident("trunc")})]
        ceiling = code in ("scalar.ceil_div", "scalar.align")
        native = {"rounding": Ident("trunc")}
        q = emit("scalar.div", (a, b), attrs=native) if code != "scalar.mod" else None
        r = emit("scalar.mod", (a, b), attrs=native)
        boolean = ScalarType(dtype("b1"))
        def compare(a, b, pred):
            return emit("scalar.cmp", (a, b), attrs={"pred": Ident(pred)}, typ=boolean)
        adjust = compare(r, 0, "ne")
        if not unsigned:
            negative_a, negative_b = compare(a, 0, "lt"), compare(b, 0, "lt")
            signs = compare(negative_a, negative_b, "eq" if ceiling else "ne")
            adjust = emit("scalar.and", (adjust, signs), typ=boolean)
        delta = emit("scalar.select", (adjust, b if code == "scalar.mod" else 1, 0))
        result = op.results[0] if code != "scalar.align" else None
        value = emit("scalar.add" if ceiling or code == "scalar.mod" else "scalar.sub",
                     (r if code == "scalar.mod" else q, delta), result=result)
        if code == "scalar.align":
            emit("scalar.mul", (value, b), result=op.results[0])
        ctx.explain.note(note, op=op.id)
        return out

    return rw.rewrite(rewrite)


PASS_DEF = Pass(PASS, run, accepts=LOWERED, produces=LOWERED,
                doc="Make floor/ceiling division explicit over native truncating integer arithmetic")
