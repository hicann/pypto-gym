# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Expose BF16 conversions before printing ordinary A5 scalar instructions."""

from ..ir import LOWERED, Literal, Rewriter, Value
from ..ir.types import ScalarType, dtype
from .manager import Pass

PASS = "bf16_scalar"
F32 = ScalarType(dtype("f32"))
BF16 = ScalarType(dtype("bf16"))


def run(module, ctx):
    if ctx.device.family != 'a5':
        return module
    rw = Rewriter(module, PASS)
    allowed = {op.id for f in module.functions if f.kind == 'func' for op in f.walk()}
    names = {v.name for f in module.functions for v in f.params}
    names.update(v.name for f in module.functions for op in f.walk() for v in op.results)
    serial = 0

    def rewrite(op):
        nonlocal serial
        if op.id not in allowed:
            return None
        out = []
        note = "Materialize BF16 scalar conversion with defined widening and RNE narrowing"

        def kind(value):
            return getattr(getattr(value, 'type', None), 'dtype', None)

        def emit(code, args, typ, result=None):
            nonlocal serial
            if result is None:
                while True:
                    serial += 1
                    name = f'bf16_scalar_{serial}'
                    if name not in names:
                        names.add(name)
                        break
                result = Value(name, typ)
            if result in op.results:
                operands = tuple(x if isinstance(x, (Value, Literal)) else Literal(x) for x in args)
                out.append(rw.rewritten(op, note, opcode=code, operands=operands, results=(result,), attrs={}))
            else:
                out.append(rw.make(code, args, results=(result,), from_ops=(op,), note=note, loc=op.loc))
            return result

        def convert(value, target, result=None):
            source = kind(value)
            if target.name == 'bf16' and source is not None and source.name not in ('bf16', 'f32'):
                value = emit('scalar.cast', (value,), F32)
            if source is not None and source.name == 'bf16' and target.name not in ('bf16', 'f32'):
                value = emit('scalar.cast', (value,), F32)
            return emit('scalar.cast', (value,), ScalarType(target), result)

        code = op.opcode
        result_kind = kind(op.results[0]) if op.results else None
        if code in ('scalar.abs', 'scalar.sqrt') and result_kind == BF16.dtype:
            wide = convert(op.operands[0], F32.dtype)
            computed = emit(code, (wide,), F32)
            convert(computed, BF16.dtype, op.results[0])
        elif code == 'scalar.const' and result_kind == BF16.dtype:
            convert(op.attrs['value'], BF16.dtype, op.results[0])
        elif code == 'scalar.cell' and 'init' in op.attrs:
            value = op.attrs['init']
            source = kind(value)
            if source != result_kind and (source == BF16.dtype or result_kind == BF16.dtype):
                cast = convert(value, result_kind)
                out.append(rw.rewritten(op, note, attrs={**op.attrs, 'init': cast}))
        elif code in ('scalar.set', 'scalar.store'):
            value = op.operands[-1]
            target = kind(op.operands[0])
            source = kind(value)
            if source != target and (source == BF16.dtype or target == BF16.dtype):
                cast = convert(value, target)
                out.append(rw.rewritten(op, note, operands=(*op.operands[:-1], cast)))
        elif code == 'scalar.cast':
            source = kind(op.operands[0])
            if source == BF16.dtype and result_kind not in (BF16.dtype, F32.dtype) or (
                    result_kind == BF16.dtype and source is not None and source not in (BF16.dtype, F32.dtype)):
                convert(op.operands[0], result_kind, op.results[0])
        if out:
            ctx.explain.note(note, op=op.id)
        return out or None

    return rw.rewrite(rewrite)


PASS_DEF = Pass(PASS, run, accepts=LOWERED, produces=LOWERED,
                doc="Materialize ordinary BF16 cell, store and scalar-math conversions")
