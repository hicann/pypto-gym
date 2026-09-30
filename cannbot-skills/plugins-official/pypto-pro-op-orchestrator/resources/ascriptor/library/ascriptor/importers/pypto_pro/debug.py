# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Observation-only debug calls: printf, pto_assert and GM tensor dumps."""
import re
from pathlib import PurePosixPath

from ...ir import Block, Value
from ...ir.types import ScalarType, dtype

TARGETS = {'debug.printf': ('debug.print',), 'debug.assert': ('debug.print',), 'debug.dump_tensor': ('debug.dump',)}
CONVERSION = re.compile(r'%[-+ #0]*\d*(?:\.\d+)?([diuxfp])')


def location(node):
    """Pro's `[basename:line]` debug prefix."""
    loc = node['location']
    return f"[{PurePosixPath(loc['file']).name}:{loc['line']}]"


def printed(ctx, node, fmt, values, sources):
    """Scalar arguments for `debug.print`; C and Python spell negative %u/%x differently."""
    o = ctx.o
    specs = CONVERSION.findall(fmt)
    o.need(len(specs) == len(values) and '%%' not in fmt, node, 'Debug format and scalar arguments disagree')
    args = []
    for spec, value, source in zip(specs, values, sources, strict=True):
        o.need(spec != 'p', node, 'Pointer debug conversions have no scalar meaning')
        value = ctx.snapshot(value, node)
        if isinstance(value, Value):
            o.need(isinstance(value.type, ScalarType), node, 'Debug arguments must be scalars')
        if spec in 'ux':
            from .selection import interval

            bounds = interval(ctx, value) if type(value) is not bool else (0, 1)
            o.need(bounds is not None and bounds[0] >= 0, node, 'Unsigned debug conversions need a proven nonnegative argument')
        o.need(spec != 'f' or o.node(source).get('kind') == 'ConstFloat' or isinstance(value, Value)
               and value.type.dtype.name == 'f32', node, 'Float debug conversions need an FP32 scalar')
        args.append(value)
    return args


def convert(ctx, node, args):
    o, name = ctx.o, node['fields']['name']
    sources = node['fields']['args']
    if name == 'debug.printf':
        attrs = ctx.attrs(node, {'format', 'show_location'})
        fmt = attrs.get('format')
        o.need(isinstance(fmt, str) and type(attrs.get('show_location')) is bool, node, 'Expected the pinned printf form')
        values = printed(ctx, node, fmt, args, sources)
        prefix = location(node) + ' ' if attrs['show_location'] else ''
        ctx.emit('debug.print', node, attrs={'fmt': prefix + fmt, 'args': values})
        return None
    if name == 'debug.assert':
        attrs = ctx.attrs(node, {'format', 'show_location', 'condition_text'})
        fmt, text = attrs.get('format'), attrs.get('condition_text')
        o.need(len(args) >= 1 and isinstance(fmt, str) and isinstance(text, str) and type(attrs.get('show_location')) is bool,
               node, 'Expected the pinned pto_assert form')
        # A failed Pro assertion prints and continues; debug.assert would stop the model instead.
        cond = ctx.snapshot(args[0], node, ScalarType(dtype('b1')))
        o.need(isinstance(cond, Value) and cond.type == ScalarType(dtype('b1')), node, 'Assertions need a BOOL condition')
        failed = ctx.emit('scalar.not', node, (cond,), typ=cond.type)
        values = printed(ctx, node, fmt, args[1:], sources[1:]) if fmt else []
        o.need(bool(fmt) or len(args) == 1, node, 'Assertion arguments need a format')
        child = ctx.child()
        prefix = location(node) + ' ' if attrs['show_location'] else ''
        child.emit('debug.print', node, attrs={'fmt': f'{prefix}Assertion failed: {text}\n'})
        if fmt:
            child.emit('debug.print', node, attrs={'fmt': fmt, 'args': values})
        ctx.emit('cf.if', node, (failed,), regions=(Block(tuple(child.ops)), Block(())))
        return None
    if name == 'debug.dump_tensor':
        from .lower import Memory
        from .memory import gm_window

        attrs = ctx.attrs(node, {'dump_flag', 'show_location'})
        o.need(len(args) == 3 and isinstance(args[0], Memory) and args[0].value.type.space == 'gm'
               and type(attrs.get('show_location')) is bool, node, 'Only GM tensor dumps are admitted')
        tensor, offsets, shapes = args
        o.need(tensor.root is None, node, 'Dumps of GM views need a strided dump rule')
        o.need(isinstance(shapes, tuple) and len(shapes) == 2, node, 'Expected a two-dimensional dump window')
        extent = tuple(ctx.snapshot(v, node) for v in shapes)
        # Pro's TPRINT starts with pipe_barrier(PIPE_ALL) before reading GM.
        ctx.emit('sync.barrier', node)
        if attrs['show_location']:
            ctx.emit('debug.print', node, attrs={'fmt': location(node) + ' dump_tensor\n'})
        window = gm_window(ctx, node, tensor, offsets, extent)
        ctx.emit('debug.dump', node, (window,), attrs={'desc': attrs.get('dump_flag', '')})
        return None
    o.fail(node, 'Tile dumps and traps need their own contracts' if name in {'debug.dump_tile', 'debug.trap'}
           else f'Unsupported debug call {name}')
    return None
