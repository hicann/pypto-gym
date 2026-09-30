# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Materialize repeated nontrapping branch slot arithmetic at its common conditional."""

from dataclasses import replace

from ..ir import Block, Literal, Value
from ..ir.scalar_flow import cell_writes, values
from ..ir.scalar_math import limits
from ..ir.types import CellType, ScalarType


def reads(op):
    return {v.name for v in values((op.operands, op.attrs)) if isinstance(v.type, CellType)}


def signature(op):
    return op.opcode, op.operands, op.results[0].type, tuple(sorted((k, str(v)) for k, v in op.attrs.items()))


def eligible(op, available):
    if op.opcode not in ('scalar.mod', 'scalar.div', 'scalar.and') or len(op.results) != 1:
        return False
    typ = op.results[0].type
    if not isinstance(typ, ScalarType) or not typ.dtype.is_integer or len(op.operands) != 2:
        return False
    value, divisor = op.operands
    return (isinstance(value, Value) and value.name in available
            and isinstance(value.type, (ScalarType, CellType)) and value.type.dtype == typ.dtype
            and isinstance(divisor, Literal) and type(divisor.value) is int
            and 0 < divisor.value <= limits(typ.dtype)[1])


def candidates(block, available, changed, result):
    changed = set(changed)
    for op in block.ops:
        if op.opcode in ('cf.call', 'cf.break', 'cf.continue', 'cf.return', 'simt.launch'):
            return available  # conservatively block every subsequent operand
        if op.opcode == 'cf.if':
            arms = [candidates(body, available, changed, result) for body in op.regions]
            changed.update(set().union(*arms))
        elif op.regions:
            return available
        elif eligible(op, available) and op.operands[0].name not in changed:
            result.setdefault(signature(op), []).append(op)
        changed.update(cell_writes(op))
    return changed


def hoist(function, rw, ctx):
    names = {p.name for p in function.params} | {v.name for op in function.walk() for v in op.results}

    def walk(block, available):
        available = set(available)
        output = []
        for op in block.ops:
            if op.opcode == 'cf.if':
                found = {}
                for body in op.regions:
                    candidates(body, available, set(), found)
                for group in found.values():
                    if len(group) < 2:
                        continue
                    sample = group[0]
                    kind = {'scalar.mod': 'slot', 'scalar.div': 'div', 'scalar.and': 'mask'}[sample.opcode]
                    base = f'{sample.operands[0].name}_{kind}_{sample.operands[1].value}'
                    name, index = base, 0
                    while name in names:
                        index += 1
                        name = f'{base}.{index}'
                    names.add(name)
                    value = Value(name, sample.results[0].type)
                    capture = rw.make(sample.opcode, sample.operands, results=(value,), attrs=sample.attrs,
                                      from_ops=group, note='capture nontrapping repeated branch arithmetic')
                    # Keep a fresh comparison adjacent to its conditional so
                    # program-point refinement still sees the current Cell read.
                    anchor = output[-1] if output else None
                    before_test = (anchor is not None and anchor.opcode == 'scalar.cmp'
                                   and anchor.results == op.operands
                                   and sample.operands[0].name not in {v.name for v in anchor.results})
                    output.insert(len(output) - bool(before_test), capture)
                    available.add(name)
                    ctx.explain.note(f'%{name} dominates {len(group)} repeated branch expressions',
                                     op=capture.id, ops=tuple(x.id for x in group), kind='branch-scalar-hoist')
            nested = available | {v.name for v in op.results}
            if op.regions:
                op = replace(op, regions=tuple(walk(r, nested) for r in op.regions))
            output.append(op)
            available.update(v.name for v in op.results)
        return Block(tuple(output))

    return replace(function, body=walk(function.body, {p.name for p in function.params}))
