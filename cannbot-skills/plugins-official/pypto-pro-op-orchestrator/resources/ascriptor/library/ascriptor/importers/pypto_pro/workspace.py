# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Explicit private Tensor ABI parameters mapped to launcher-owned GM scratch."""
from math import prod

from ...ir import Value
from ...ir.types import DimValue, MemType, ScalarType, dtype
from .dynamic import arithmetic, extent_type
from .lower import Memory


def build(ctx, source, variables):
    o = ctx.o
    entries = o.document['abi']['workspace']
    if entries:
        o.need(not o.mixed and o.side == 'vec' and o.document['abi']['block_dim'] == 1, source,
               'Workspace currently requires one vector participant')
    values, layout = [], []
    cursor = 0
    for entry in entries:
        name = entry['parameter']
        o.need(name in variables, source, 'Workspace parameter is absent from the source signature')
        var = variables[name]
        typ, original = o.parameter(var)
        o.need(isinstance(typ, MemType) and name not in o.nz_parameters, var, 'Workspace parameter must be an ND tensor')
        dims = []
        for dim in entry['shape']:
            if type(dim) is int:
                dims.append(dim)
            else:
                tensor = ctx.env.get(dim['tensor'])
                o.need(isinstance(tensor, Memory) and tensor.value.type.space == 'gm', var,
                       'Workspace shape must reference a public tensor')
                dims.append(tensor.shape[dim['axis']])
        for declared, actual in zip(original, dims, strict=True):
            if isinstance(declared, DimValue):
                ctx.env[declared.name] = actual
                o.dynamic_params.pop(declared.name)
            else:
                o.need(type(actual) is int and actual == declared, var,
                       'Workspace contract contradicts a static source dimension')
        before = len(o.emitted)
        numel = arithmetic(ctx, var, 'mul', *dims)
        size = arithmetic(ctx, var, 'mul', numel, typ.dtype.bits // 8)
        if type(cursor) is int:
            offset = (cursor + 31) // 32 * 32
        else:
            offset = ctx.emit('scalar.align', var, (cursor,), typ=ScalarType(dtype('i64')), attrs={'n': 32})
        value = ctx.emit('mem.workspace', var, typ=MemType('ws', typ.dtype, extent_type(dims)),
                         attrs={'name': name, 'numel': numel, 'offset': offset})
        cursor = offset + size if type(offset) is int and type(size) is int else ctx.emit(
            'scalar.add', var, (offset, size), typ=ScalarType(dtype('i64')))
        ctx.env[name] = Memory(value, tuple(dims), tuple(dims))
        values.append(value)
        layout.append({'name': name, 'dims': [v.name if isinstance(v, Value) else v for v in dims],
                       'elem_bytes': typ.dtype.bits // 8, 'alignment': 32})
        o.record(var, before)
    try:
        check_sizes(layout, {}, static_only=True)
    except ValueError as error:
        o.fail(source, str(error))
    return values, layout


def check_sizes(layout, bound, *, static_only=False):
    """Guard all intermediate signed size/align arithmetic before allocation."""
    end = 0
    for entry in layout:
        if static_only and any(isinstance(d, str) for d in entry['dims']):
            continue
        dims = [bound[d] if isinstance(d, str) else d for d in entry['dims']]
        if any(type(d) is not int or d <= 0 for d in dims):
            raise ValueError('Workspace dimensions must be positive integers')
        end = (end + 31) // 32 * 32 + prod(dims) * entry['elem_bytes']
        if end > 2**63 - 1 - 31:
            raise ValueError('Workspace size exceeds checked signed address arithmetic')
    return end
