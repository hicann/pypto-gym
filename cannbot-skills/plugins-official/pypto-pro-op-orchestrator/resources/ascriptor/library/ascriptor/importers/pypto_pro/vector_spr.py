# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Vector mask SPR writes and reads, register reductions and interleaved register transfers."""
from math import prod

from ...ir import Ident, Value
from ...ir.types import MaskType, RegType, ScalarType, dtype
from .lower import Memory

# Each admitted form follows a rule measured on A5 (RFC-0015). Setting an entry to False refuses the form at
# its source location while its rule is re-examined.
ADMIT = {'spr_read': True, 'group_reduce': True, 'whole_reduce': True, 'interleave': True}
PENDING = {'spr_read': 'SPR mask reads', 'group_reduce': 'Datablock reductions',
           'whole_reduce': 'Whole-register reductions', 'interleave': 'Interleaved transfers'}

WRITES = {'system.set_mask_norm': 'vec.set_mask_normal', 'system.set_vec_mask': 'vec.set_mask',
          'system.reset_mask': 'vec.reset_mask', 'system.set_mask_count': None}
GROUPS = {'vf.reduce_sum': 'vf.cgadd', 'vf.reduce_max': 'vf.cgmax', 'vf.reduce_min': 'vf.cgmin'}
WHOLE = {'vf.reduce_max': 'vf.cmax', 'vf.reduce_min': 'vf.cmin'}
TARGETS = {**{name: (target,) for name, target in WRITES.items() if target}, 'vf.get_mask_spr': ('vf.mask_from_spr',),
           **{f'{name} (datablock)': (target,) for name, target in GROUPS.items()},
           **{f'{name} (whole)': (target,) for name, target in WHOLE.items()},
           'vf.load_align (DINTLV)': ('vf.load_interleave',), 'vf.store_align (INTLV)': ('vf.store_interleave',)}
# Pro block operations measured or known to leave {MASK1, MASK0} alone: descriptor updates and GM stores (A5
# TSTORE). TLOAD, scalar tile access and TileOp helpers are unmeasured, so they end the admitted SPR value.
NEUTRAL = {'block.make_tile', 'block.subview', 'block.set_validshape', 'block.store'}
# VF predicate producers A5 ran between an SPR write and a read without changing it: pset, plt, pand, movp, vcmp_lt.
KEEPS_SPR = {'vf.mask', 'vf.mask_update', 'vf.mask_and', 'vf.mask_from_spr'}
REGISTERS = (RegType(dtype('f32')), RegType(dtype('i32')))
BLOCK_COPY = {'data_copy_mode', 'block_stride', 'repeat_stride', 'post_update'}
# Interleaved transfers A5 ran, per register dtype: DINTLV load dist, INTLV store dists (Pro's INTLV follows the
# register width and has no B16 name), store predicate widths and elements per transfer. Offsets count elements.
INTERLEAVE = {'f32': (22, (5, 6), (32, 16), 128), 'i32': (22, (5, 6), (32, 16), 128), 'f16': (21, (5,), (16,), 256)}


def claims(name, node, args):
    """SPR reads, register max/min and datablock reductions and the two-register aligned transfers."""
    block = node['fields']['kwargs'].get('datablock', False) is not False
    return (name == 'vf.get_mask_spr' or name in GROUPS and block or name in WHOLE and not block
            or name == 'vf.load_align' and len(args) == 4
            or name == 'vf.store_align' and len(args) == 4 and isinstance(args[2], Value) and isinstance(args[2].type, RegType))


def keeps_spr(opcode, attrs):
    return opcode in KEEPS_SPR or opcode == 'vf.cmp' and str(attrs.get('mode')) == 'lt'


def stale(o, reason):
    """An operation that may rewrite the SPR ends the admitted value; later reads report the reason."""
    if o.spr == 'live':
        o.spr = reason


def pending(o, node, key):
    o.need(ADMIT[key], node, f'{PENDING[key]} await native A5 evidence for their reference rule')


def write(ctx, node, args):
    o, name = ctx.o, node['fields']['name']
    o.need(not ctx.vf, node, 'SPR mask writes inside VF sections are not admitted')
    o.need(o.side == 'vec', node, 'SPR mask state exists only on the vector side')
    o.need(WRITES[name] is not None, node, 'Counter mask mode has no c310 instruction to govern; set_mask_count is not admitted')
    o.need(ctx.depth == 0, node, 'SPR mask writes must sit at the top level of the vector section')
    ctx.attrs(node)
    if name == 'system.set_vec_mask':
        args = [ctx.snapshot(v, node) for v in args]
        o.need(len(args) == 2 and all(type(v) is int or isinstance(v, Value) and v.type in
                                      (ScalarType(dtype('i32')), ScalarType(dtype('i64'))) for v in args), node,
               'set_vec_mask operands must be INT32/INT64 scalars or integer literals')
    else:
        o.need(not args, node, f'{name} takes no operands')
    o.need(o.spr_norm or name == 'system.set_mask_norm', node, 'SPR mask writes need set_mask_norm earlier in the vector section')
    attrs = {'high': args[0], 'low': args[1]} if args else None
    ctx.emit(WRITES[name], node, attrs=attrs)
    if name == 'system.set_mask_norm':
        o.spr_norm = True
    else:
        o.spr = 'live'  # C++ converts both halves to uint64: INT32 values sign-extend.


def read(ctx, node, args):
    o = ctx.o
    width = ctx.attrs(node, {'width'}).get('width', 0)
    o.need(not args and type(width) is int and width in (0, 1), node, 'get_mask_spr admits B32 and B16 widths')
    o.need(ctx.top and ctx.depth == 0, node, 'SPR mask reads must sit at the top level of a VF section called at the top level')
    o.need(o.spr != 'unset', node, 'SPR mask read needs a norm-mode SPR write before its VF section')
    o.need(o.spr != 'predicate', node, 'SPR mask read after an unmeasured VF predicate producer is not admitted')
    o.need(o.spr == 'live', node, 'SPR mask read after a tile load, tile-level operation or SIMT launch is not admitted')
    pending(o, node, 'spr_read')
    result = ctx.emit('vf.mask', node, typ=MaskType(16 if width else 32), attrs={'init': Ident('none')})
    ctx.emit('vf.mask_from_spr', node, (result,))
    return result


def reduce(ctx, node, args):
    o, name = ctx.o, node['fields']['name']
    attrs = ctx.attrs(node, {'datablock', 'merge_mode'})
    block = attrs.get('datablock', False) is not False
    o.need(not block or attrs['datablock'] is True, node, 'Reduction datablock flag must be a boolean')
    mode = attrs.get('merge_mode', 0)
    o.need(type(mode) is int and mode == 0, node, 'VF import requires ZEROING mode')
    o.need(len(args) == 3 and all(isinstance(v, Value) for v in args), node,
           'Register reductions need a destination, a source and a predicate')
    dst, src, pred = args
    o.need(dst.type == src.type and dst.type in REGISTERS and pred.type == MaskType(32), node,
           f'{"Datablock" if block else "Whole-register"} reductions require matching FP32/INT32 registers and a b32 predicate')
    pending(o, node, 'group_reduce' if block else 'whole_reduce')
    if block:
        ctx.emit(GROUPS[name], node, (dst, src), attrs={'mask': pred})
    else:  # Pro keeps the native index lane: lane 1 holds the first active lane of the extremum.
        ctx.emit(WHOLE[name], node, (dst, src), attrs={'mask': pred, 'index': True})


def interleaved(o, node, first, second):
    """The measured rule of two matching FP32, INT32 or FP16 registers."""
    o.need(all(isinstance(v, Value) and isinstance(v.type, RegType) and v.type.dtype.name in INTERLEAVE
               for v in (first, second)) and first.type == second.type, node,
           'Interleaved transfers admit matching FP32/INT32/FP16 registers')
    return INTERLEAVE[first.type.dtype.name]


def load(ctx, node, args):
    o = ctx.o
    attrs = ctx.attrs(node, {'dist', *BLOCK_COPY})
    o.need(not set(attrs) & (BLOCK_COPY - {'post_update'}) and attrs.get('post_update', False) is False, node,
           'Interleaved transfers admit no block-copy or post-update attributes')
    first, second, tile, offset = args
    dist, _, _, elements = interleaved(o, node, first, second)
    o.need(first.name != second.name, node, 'Interleaved load needs distinct destinations')
    # An absent distribution follows the register width, as in Pro.
    o.need(attrs.get('dist', dist) == dist, node, 'Interleaved load distribution must match the register width')
    o.need(isinstance(tile, Memory) and tile.value.type.space == 'ub' and tile.pitch is None
           and tile.value.type.dtype == first.type.dtype and type(offset) is int and offset >= 0
           and offset % (elements // 16) == 0 and offset + elements <= prod(tile.shape), node,
           'Interleaved load needs a static aligned offset inside a whole UB tile of the register dtype')
    pending(o, node, 'interleave')
    ctx.emit('vf.load_interleave', node, (first, second, tile.value),
             attrs={'offset': offset, 'mode': Ident(f'dintlv_b{first.type.dtype.bits}')})
    ctx.reads.append(tile.value)


def store(ctx, node, args):
    o = ctx.o
    attrs = ctx.attrs(node, {'dist', *BLOCK_COPY})
    o.need(not set(attrs) & BLOCK_COPY, node, 'Interleaved transfers admit no block-copy or post-update attributes')
    tile, first, second, pred = args
    _, dists, widths, elements = interleaved(o, node, first, second)
    o.need(attrs.get('dist') in dists, node, 'Interleaved store requires an INTLV distribution matching the register width')
    o.need(isinstance(pred, Value) and pred.type in [MaskType(w) for w in widths], node,
           f'Interleaved store predicate must be {" or ".join(f"b{w}" for w in widths)} for {first.type.dtype.name} registers')
    o.need(isinstance(tile, Memory) and tile.value.type.space == 'ub' and tile.pitch is None
           and tile.value.type.dtype == first.type.dtype and prod(tile.shape) >= elements, node,
           f'Interleaved store needs a whole UB tile of the register dtype holding {elements} elements')
    pending(o, node, 'interleave')
    # A5 writes every element pair whatever the predicate (b32 or b16); Pro still passes it to vsts.
    ctx.emit('vf.store_interleave', node, (tile.value, first, second),
             attrs={'offset': 0, 'mode': Ident(f'intlv_b{first.type.dtype.bits}'), 'mask': pred})
    ctx.writes.append(tile.value)


def convert(ctx, node, args):
    name = node['fields']['name']
    if name == 'vf.get_mask_spr':
        return read(ctx, node, args)
    if name in GROUPS:
        return reduce(ctx, node, args)
    return (load if name == 'vf.load_align' else store)(ctx, node, args)
