# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""SIMT atomics, exact scalar math and FP32 classification inside converted SIMT bodies (RFC-0015).

Atomics act on one proven element of SIMT storage and keep Pro's prior-value result. `mul_hi`, `fma`
and `fmod` keep their operand dtype. Classification builtins give 0 or 1, as measured on A5.
"""
from math import prod

from ...ir import Ident, Value
from ...ir.types import MemType, ScalarType, dtype
from .lower import Memory

ATOMICS = {f'simt.atomic_{op}': op for op in ('add', 'sub', 'exch', 'max', 'min', 'and', 'or', 'xor', 'cas', 'inc', 'dec')}
INTEGER = frozenset({'add', 'sub', 'exch', 'max', 'min', 'and', 'or', 'xor', 'cas'})
FLOAT = frozenset({'add', 'sub', 'exch', 'max', 'min', 'cas'})
ADMITTED = {('gm', 'i32'): INTEGER, ('ub', 'i32'): INTEGER, ('gm', 'u32'): INTEGER | {'inc', 'dec'},
            ('gm', 'f32'): FLOAT, ('ub', 'f32'): FLOAT}
MATH = {'simt.mul_hi': 2, 'simt.fma': 3, 'simt.fmod': 2}
CLASSES = ('simt.isnan', 'simt.isinf', 'simt.isfinite')
CALLS = frozenset(ATOMICS) | frozenset(MATH) | frozenset(CLASSES)
TARGETS = {**dict.fromkeys(ATOMICS, ('simt.atomic',)), **{name: (name,) for name in MATH},
           **{name: (name, 'scalar.cmp') for name in CLASSES}}
F32, I32, B1 = ScalarType(dtype('f32')), ScalarType(dtype('i32')), ScalarType(dtype('b1'))


def convert(ctx, node, args):
    ctx.attrs(node)
    name = node['fields']['name']
    if name in ATOMICS:
        return atomic(ctx, node, args, ATOMICS[name])
    if name in MATH:
        return scalar_math(ctx, node, args, name)
    return classification(ctx, node, args, name)


def atomic(ctx, node, args, kind):
    from .selection import interval

    o = ctx.o
    o.need(len(args) == (4 if kind == 'cas' else 3) and isinstance(args[0], Memory), node,
           'SIMT atomics act on one element of SIMT storage')
    storage = args[0].value
    space, name = storage.type.space, storage.type.dtype.name
    element = ScalarType(storage.type.dtype)
    o.need(kind not in {'inc', 'dec'} or name == 'u32', node, 'SIMT atomic inc/dec admit UINT32 counters')
    o.need(kind in ADMITTED.get((space, name), ()), node, f'SIMT atomic {kind} does not admit {name} {space} storage')
    o.need(o.scalar_type(node['fields']['type']) == element, node, 'SIMT atomic result type contradicts its storage')
    index = ctx.snapshot(args[1], node)
    bounds = interval(ctx, index)
    o.need(bounds is not None and bounds[0] >= 0 and bounds[1] < prod(storage.type.dims), node,
           'SIMT element index is not proven inside its storage')
    values = [operand(ctx, node, value, element, 'SIMT atomic') for value in args[2:]]
    # Pro passes (compare, value); the target keeps `src` third and `compare` last. Both printers call
    # atomicCAS(ptr, compare, value).
    return ctx.emit('simt.atomic', node, (storage, index, *reversed(values)) if kind == 'cas' else (storage, index, *values),
                    typ=element, attrs={'op': Ident(kind)})


def scalar_math(ctx, node, args, name):
    o, short = ctx.o, name.removeprefix('simt.')
    o.need(len(args) == MATH[name], node, f'SIMT {short} expects {MATH[name]} operands')
    result = o.scalar_type(node['fields']['type'])
    if name == 'simt.mul_hi':
        o.need(result.dtype.name in {'i32', 'u32'}, node, 'SIMT mul_hi admits two INT32 or two UINT32 operands')
    else:
        o.need(result == F32, node, f'SIMT {short} admits FP32 operands')
    return ctx.emit(name, node, tuple(operand(ctx, node, value, result, f'SIMT {short}') for value in args), typ=result)


def classification(ctx, node, args, name):
    o, short = ctx.o, name.removeprefix('simt.')
    o.need(len(args) == 1 and o.scalar_type(node['fields']['type']) == B1, node, f'SIMT {short} tests one FP32 scalar')
    raw = ctx.emit(name, node, (operand(ctx, node, args[0], F32, f'SIMT {short}'),), typ=I32)
    return ctx.emit('scalar.cmp', node, (raw, 0), typ=B1, attrs={'pred': Ident('ne')})


def operand(ctx, node, value, typ, what):
    """A value of exactly `typ`, or a literal Pro typed to it that keeps its value."""
    o = ctx.o
    if type(value) in (int, float):
        kind, bits = typ.dtype.kind, typ.dtype.bits
        if kind == 'float':
            fits = type(value) is float
        else:
            low, high = (0, 1 << bits) if kind == 'uint' else (-(1 << (bits - 1)), 1 << (bits - 1))
            fits = type(value) is int and low <= value < high
        o.need(fits, node, f'{what} literal changes value in {typ.dtype.name}')
        return ctx.snapshot(value, node, typ)
    value = ctx.snapshot(value, node)
    o.need(isinstance(value, Value) and value.type == typ, node, f'{what} operands must have dtype {typ.dtype.name}')
    return value


def edges(o, ref):
    """Every (parent, field, position, child) reference below `ref`, excluding types and attributes."""
    found, seen, stack = [], set(), [ref]
    while stack:
        node = o.node(stack.pop())
        if node['id'] in seen:
            continue
        seen.add(node['id'])
        for key, value in node['fields'].items():
            if key in {'type', 'kwargs', 'attrs'}:
                continue
            for position, child in enumerate(value if isinstance(value, list) else [value]):
                if isinstance(child, dict) and '$ref' in child:
                    found.append((node, key, position, o.node(child)))
                    stack.append(child)
    return found


def launch_storage(o, source, params):
    """UINT32 GM tensors are admitted only as whole SIMT launch arguments or as GM sources of loads (sort32 identifiers)."""
    names = {p.name for p in params if isinstance(p.type, MemType) and p.type.dtype.name == 'u32'}
    for parent, key, position, child in edges(o, source['fields']['body']) if names else ():
        if child['kind'] == 'Var' and child['fields']['name'] in names:
            call = parent['fields'].get('name') if parent['kind'] == 'Call' and key == 'args' else None
            o.need(call == 'simt.launch' and position >= 3 or call == 'block.load' and position == 1,
                   parent if parent.get('location') else source,
                   'UINT32 storage is admitted only as sort32 identifiers loaded from GM or as SIMT launch storage')
