# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Single-bit CTRL saturation writes and reads; the IR keeps raw bit polarity."""
from ...ir import Ident, Value
from ...ir.types import ScalarType, dtype

BITS = {48: 'float', 50: 'float8', 53: 'int', 59: 'cast', 60: 'global'}
# SaturationFlagMode -> (bit, predicate): FLOAT/FLOAT8/CAST are enabled when their bit is 0.
MODES = {0: ('float', 'eq'), 1: ('float8', 'eq'), 2: ('int', 'ne'), 3: ('cast', 'eq')}
UNSIGNED = "Pro holds CTRL reads in uint64 locals; derived integers must stay provably nonnegative"
TARGETS = {'set_ctrl_spr': ('core.set_sat_flag',), 'get_ctrl_spr': ('core.get_sat_flag',),
           'get_saturation_flag': ('core.get_sat_flag',)}


def convert(ctx, node, args):
    o, name = ctx.o, node['fields']['name']
    if name == 'set_ctrl_spr':
        ctx.attrs(node)
        o.need(len(args) == 3 and all(type(v) is int for v in args[:2]) and args[0] == args[1] and args[0] in BITS, node,
               'Only single raw saturation CTRL bits are admitted')
        value = args[2]
        o.need(type(value) is int or isinstance(value, Value) and isinstance(value.type, ScalarType)
               and value.type.dtype.name in {'i32', 'i64'}, node, 'CTRL bit writes take an INT32/INT64 value')
        # Pro writes (value & 1); the target writes value != 0, so unproven values keep only their low bit.
        if type(value) is int:
            value = bool(value & 1)
        elif not 0 <= ctx.bounds.get(value.name, (-1, 0))[0] <= ctx.bounds.get(value.name, (0, 2))[1] <= 1:
            value = ctx.emit('scalar.and', node, (value, 1), typ=value.type)
        ctx.emit('core.set_sat_flag', node, attrs={'mode': Ident(BITS[args[0]]), 'enable': value})
        return None
    # Pro prints both reads without outer parentheses, so C++ precedence regroups them inside larger expressions.
    o.need(ctx.whole == node['id'], node, 'Pro prints CTRL reads without parentheses; assign the read to a variable first')
    if name == 'get_ctrl_spr':
        ctx.attrs(node)
        o.need(len(args) == 2 and all(type(v) is int for v in args) and args[0] == args[1] and args[0] in BITS, node,
               'Only single raw saturation CTRL bit reads are admitted')
        mode, pred = BITS[args[0]], None
    else:
        choice = ctx.attrs(node, {'mode'}).get('mode')
        o.need(not args and type(choice) is int and choice in MODES, node, 'Unadmitted saturation flag mode')
        mode, pred = MODES[choice]
    flag = ctx.emit('core.get_sat_flag', node, typ=ScalarType(dtype('i32')), attrs={'mode': Ident(mode)})
    typ = o.scalar_type(node['fields']['type'])
    if pred is not None:
        o.need(typ.dtype.name == 'b1', node, 'Saturation flag reads produce BOOL')
        return ctx.emit('scalar.cmp', node, (flag, 0), typ=typ, attrs={'pred': Ident(pred)})
    o.need(typ.dtype.name == 'i64', node, 'CTRL bit reads produce INT64')
    value = ctx.emit('scalar.cast', node, (flag,), typ=typ)
    ctx.bounds[value.name] = (0, 1)
    ctx.nonnegative.add(value.name)
    ctx.unsigned.add(value.name)
    return value


def unsigned_result(ctx, node, operands, result):
    """Propagate Pro's uint64 CTRL locals. Signed and unsigned values agree while every operand and result is nonnegative."""
    from .selection import interval

    ctx.o.need(not any(isinstance(v, Value) and v.name in ctx.unsigned_cells for v in operands), node,
               "A reassigned variable holding a CTRL read has no proven range; bind each read once")
    if not any(isinstance(v, Value) and v.name in ctx.unsigned for v in operands):
        return result
    values = operands if result is None else (*operands, result)
    ctx.o.need(all((type(v) is int or isinstance(v, Value) and v.type.dtype.is_integer)
                   and (interval(ctx, v) or (-1, 0))[0] >= 0 for v in values), node, UNSIGNED)
    if isinstance(result, Value):
        ctx.unsigned.add(result.name)
    return result
