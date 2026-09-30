# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Canonical runtime dimensions and bounded transfer extents."""
from ...ir import Value
from ...ir.types import DimValue, ScalarType, dtype
from .selection import interval


def parameter_shape(o, var, refs):
    o.need(len(refs) == 2, var, "Only two-dimensional GM parameters are admitted")
    dims = []
    for axis, ref in enumerate(refs):
        node = o.node(ref)
        if node['kind'] == 'ConstInt':
            value = o.literal(ref, var)
            o.need(type(value) is int and value > 0, var, "GM dimensions must be positive")
            dims.append(value)
            continue
        name = f"__pypto_dyn_{var['fields']['name']}_{axis}"
        o.need(node['kind'] == 'Var' and node['fields']['name'] == name, var,
               "Only canonical Pro runtime shape variables are admitted")
        typ = o.node(node['fields']['type'])
        o.need(typ['kind'] == 'ScalarType' and typ['fields']['dtype']['$dtype'] == 'index', var,
               "Runtime dimensions must use Pro INDEX")
        o.dynamic_params[name] = Value(name, o.scalar_type(typ))
        dims.append(DimValue(name))
    return tuple(dims)


def le(ctx, left, right):
    if left == right:
        return True
    a, b = interval(ctx, left), interval(ctx, right)
    if a is not None and b is not None and a[1] <= b[0]:
        return True
    return isinstance(left, Value) and isinstance(right, Value) and right.name in ctx.upper.get(left.name, ())


def positive(ctx, value):
    bounds = interval(ctx, value)
    return bounds is not None and bounds[0] > 0


def extent_type(values):
    return tuple(DimValue(v.name) if isinstance(v, Value) else v for v in values)


def arithmetic(ctx, node, opcode, left, right):
    if type(left) is int and type(right) is int:
        return left * right if opcode == 'mul' else left - right
    return ctx.emit('scalar.' + opcode, node, (left, right), typ=ScalarType(dtype('i64')))
