# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Signed INT32 bitwise operations and scalar/register shifts."""
from ...ir import Value
from ...ir.types import RegType, ScalarType, dtype
from .vector_predicates import mask, zeroing

TARGETS = {'vf.and_': ('vf.and',), 'vf.or_': ('vf.or',), 'vf.xor': ('vf.xor',),
           'vf.not_': ('vf.not',), 'vf.shift_left': ('vf.shiftl', 'vf.shiftls'),
           'vf.shift_right': ('vf.shiftr', 'vf.shiftrs')}


def ireg(value):
    return isinstance(value, Value) and value.type == RegType(dtype('i32'))


def convert(ctx, node, args):
    o, name = ctx.o, node['fields']['name']
    attrs = ctx.attrs(node, {'mode', 'dtype'} if name == 'vf.shift_right' else {'mode'})
    zeroing(o, node, attrs)
    unary, shift = name == 'vf.not_', name in {'vf.shift_left', 'vf.shift_right'}
    o.need(len(args) == (3 if unary else 4) and all(ireg(v) for v in args[:2]) and mask(args[-1]),
           node, 'Integer VF import requires INT32 registers and a b32 predicate')
    if 'dtype' in attrs:
        o.need(o.dt(attrs['dtype'], node).name == 'i32', node, 'Shift dtype reinterpretation is not admitted')
    operands = list(args[:-1])
    target = TARGETS[name][0]
    if not unary:
        amount = args[2]
        if shift and not ireg(amount):
            o.need(type(amount) is int and 0 <= amount <= 32767 or isinstance(amount, Value)
                   and amount.type == ScalarType(dtype('i32')), node,
                   'Scalar shift requires an INT32 value or a constant in [0, 32767]')
            # EmitVFShift in the pinned producer explicitly narrows to int16_t.
            operands[2] = ctx.emit('scalar.cast', node, (amount,), typ=ScalarType(dtype('i16')))
            target = TARGETS[name][1]
        else:
            o.need(ireg(amount), node, 'Bitwise and per-lane shift sources must be INT32 registers')
    ctx.emit(target, node, operands, attrs={'mask': args[-1]})
    return None
