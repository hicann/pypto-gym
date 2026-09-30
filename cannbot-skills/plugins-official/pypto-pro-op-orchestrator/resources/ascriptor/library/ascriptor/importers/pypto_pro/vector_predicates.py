# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Typed predicates, b32 selection/broadcast and admitted numeric casts."""
from ...ir import Ident, Value
from ...ir.types import CellType, MaskType, RegType, ScalarType, dtype

COMPARE = {f'vf.{name}': name for name in ('eq', 'ne', 'lt', 'le', 'gt', 'ge')}
PREDICATES = {'vf.update_mask', 'vf.select', 'vf.full', 'vf.astype', *COMPARE}
TARGETS = {'vf.update_mask': ('vf.mask_update',), 'vf.select': ('vf.select',),
           'vf.full': ('vf.dup',), 'vf.astype': ('vf.cast',),
           **{name: ('vf.cmp', 'vf.cmps') for name in COMPARE}}
ROUNDS = {0: 'round', 1: 'rint', 2: 'floor', 3: 'ceil', 4: 'trunc'}
# A5 runs compiled the UINT16 zero fill of Pro's histogram idiom and zeroed the register.
UINT16_ZERO_FILL = True


def reg(value):
    return isinstance(value, Value) and value.type in (RegType(dtype('f32')), RegType(dtype('i32')))


def storage_reg(value):
    return isinstance(value, Value) and isinstance(value.type, RegType) and value.type.dtype.name in {
        'f32', 'i32', 'f16', 'bf16'} and value.type.n == 1


def mask(value):
    return isinstance(value, Value) and value.type == MaskType(32)


def scalar(value, dt):
    return (type(value) in ((int, float) if dt.name == 'f32' else (int,))
            or isinstance(value, Value) and value.type == ScalarType(dt))


def zeroing(o, node, attrs):
    mode = attrs.get('mode', 0)
    o.need(type(mode) is int and mode == 0, node, 'VF import requires ZEROING mode')


def convert(ctx, node, args):
    o, name = ctx.o, node['fields']['name']
    if name == 'vf.update_mask':
        attrs = ctx.attrs(node, {'dtype'})
        dt = o.dt(attrs['dtype'], node) if 'dtype' in attrs else dtype('f32')
        o.need(len(args) == 1 and dt.name in {'f32', 'i32', 'f16', 'bf16', 'u8', 'u16', 'u32'}, node,
               'Only b8/b16/b32 mask updates are admitted')
        count = ctx.snapshot(args[0], node)
        o.need(type(count) is int or isinstance(count, Value) and count.type in
               (ScalarType(dtype('i32')), ScalarType(dtype('i64'))), node, 'Mask count must be an integer scalar')
        snapshot = ctx.emit('scalar.cast', node, (count,), typ=ScalarType(dtype('u32')))
        counter = ctx.emit('scalar.cell', node, typ=CellType(dtype('u32')), attrs={'init': snapshot})
        result = ctx.emit('vf.mask', node, typ=MaskType(dt.bits), attrs={'init': Ident('none')})
        ctx.emit(TARGETS[name][0], node, (result,), attrs={'cnt': counter})
        return result
    if name in COMPARE:
        attrs = ctx.attrs(node, {'cmp_dtype'})
        o.need(len(args) == 4 and mask(args[0]) and reg(args[1]) and mask(args[3]), node,
               'Comparison requires b32 predicates and FP32/INT32 data')
        dst, a, b, pred = args
        if 'cmp_dtype' in attrs:
            o.need(o.dt(attrs['cmp_dtype'], node) == a.type.dtype, node, 'Comparison dtype reinterpretation is not admitted')
        vector = reg(b)
        o.need(b.type == a.type if vector else scalar(b, a.type.dtype), node, 'Comparison source dtype mismatch')
        ctx.emit(TARGETS[name][0 if vector else 1], node, (dst, a, b),
                 attrs={'mode': Ident(COMPARE[name]), 'mask': pred})
        return None
    if name == 'vf.astype':
        attrs = ctx.attrs(node, {'mode', 'layout', 'round_mode', 'saturate'})
        zeroing(o, node, attrs)
        o.need(len(args) == 3 and storage_reg(args[0]) and storage_reg(args[1]) and isinstance(args[2], Value)
               and args[2].type in (MaskType(16), MaskType(32)) and args[0].type != args[1].type, node, 'Numeric casts require admitted registers and a b16/b32 predicate')
        layout, rounding, sat = attrs.get('layout', 0), attrs.get('round_mode', 1), attrs.get('saturate', 0)
        pair = (args[1].type.dtype.name, args[0].type.dtype.name)
        cross = pair in {('f32', 'f16'), ('f16', 'f32'), ('f32', 'bf16'), ('bf16', 'f32')}
        o.need(mask(args[2]) or cross and pair[1] == 'f32', node,
               'Only widening FP16/BF16 casts admit a b16 predicate')
        o.need(cross or set(pair) == {'f32', 'i32'}, node, 'Unadmitted numeric cast pair')
        o.need(type(layout) is int and layout in ((0, 1) if cross else (0,)), node,
               'Cast layout must be ZERO/ONE for cross-width floats, ZERO for same-width casts')
        if cross:
            o.need(type(rounding) is int and rounding == 1 and type(sat) is int and sat == 0, node,
                   'Cross-width float casts require RINT and saturation OFF')
        o.need(type(rounding) is int and rounding in ROUNDS, node, 'Unadmitted cast rounding mode')
        o.need(type(sat) is int and sat in (0, 1), node, 'Invalid cast saturation')
        o.need(args[1].type.dtype.name == 'f32' or sat == 0, node, 'INT32 to FP32 cast has no saturation selector')
        ctx.emit(TARGETS[name][0], node, args[:2], attrs={'mask': args[2], 'layout': Ident('one' if layout else 'zero'),
                 'round': Ident(ROUNDS[rounding]), 'saturate': bool(sat), 'merge': Ident('zeroing')})
        return None
    if name == 'vf.select':
        attrs = ctx.attrs(node, {'mode'})
        zeroing(o, node, attrs)
        o.need(len(args) == 4 and all(reg(v) for v in args[:3]) and mask(args[3])
               and args[0].type == args[1].type == args[2].type, node, 'Select requires matching 32-bit registers')
        ctx.emit(TARGETS[name][0], node, args[:3], attrs={'mask': args[3]})
        return None
    attrs = ctx.attrs(node, {'dtype', 'mode'})
    zeroing(o, node, attrs)
    if args and isinstance(args[0], Value) and args[0].type == RegType(dtype('u16')):
        o.need(UINT16_ZERO_FILL, node, 'UINT16 broadcast awaits native compile-and-zero evidence')
        o.need(len(args) == 3 and type(args[1]) is int and args[1] == 0 and isinstance(args[2], Value)
               and args[2].type == MaskType(16) and ('dtype' not in attrs or o.dt(attrs['dtype'], node) == dtype('u16')), node,
               'UINT16 broadcast admits a literal zero with a b16 predicate')
        ctx.emit(TARGETS['vf.full'][0], node, args[:2], attrs={'mask': args[2]})
        return None
    o.need(len(args) == 3 and reg(args[0]) and mask(args[2]) and scalar(args[1], args[0].type.dtype), node,
           'Broadcast requires a scalar, explicit b32 mask and 32-bit destination')
    if 'dtype' in attrs:
        o.need(o.dt(attrs['dtype'], node) == args[0].type.dtype, node, 'Broadcast dtype mismatch')
    ctx.emit(TARGETS[name][0], node, args[:2], attrs={'mask': args[2]})
    return None
