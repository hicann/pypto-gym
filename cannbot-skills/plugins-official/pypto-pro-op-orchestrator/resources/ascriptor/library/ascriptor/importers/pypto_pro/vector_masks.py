# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Predicate conjunction, predicate spill/fill through UB, and VF memory barriers."""
from math import prod

from ...ir import Ident, Value
from ...ir.types import MaskType
from .lower import Memory

TARGETS = {'vf.mem_bar': ('vf.barrier',), 'vf.and_ (predicates)': ('vf.mask_and',),
           'vf.store_align (predicate)': ('vf.mask_to_ub',), 'vf.load_align (predicate)': ('vf.ub_to_mask',)}
# Pro's MemBarMode enumerators in declaration order (framework/include/ir/op_attr_types.h:194). A mode exports as
# its index, which the CCE emitter casts back and prints by name as the mem_bar tag (backend_cce_vf_ops.cpp:1010).
MEM_BAR_MODES = (('VST_VLD', 'vec_store', 'vec_load'), ('VLD_VST', 'vec_load', 'vec_store'),
                 ('VST_VST', 'vec_store', 'vec_store'), ('VST_LD', 'vec_store', 'scalar_load'),
                 ('VST_ST', 'vec_store', 'scalar_store'), ('VLD_ST', 'vec_load', 'scalar_store'),
                 ('ST_VLD', 'scalar_store', 'vec_load'), ('ST_VST', 'scalar_store', 'vec_store'),
                 ('LD_VST', 'scalar_load', 'vec_store'), ('VV_ALL', 'vec_all', 'vec_all'),
                 ('VS_ALL', 'vec_all', 'scalar_all'), ('SV_ALL', 'scalar_all', 'vec_all'))


def predicate(value):
    return isinstance(value, Value) and isinstance(value.type, MaskType)


def claims(name, args):
    """Whether a call is a predicate overload rather than the register form of the same name."""
    return name == 'vf.mem_bar' or bool(args) and (
        name == 'vf.and_' and predicate(args[0]) or name in {'vf.store_align', 'vf.load_align'} and len(args) == 2
        and any(predicate(v) for v in args))


def convert(ctx, node, args):
    o, name = ctx.o, node['fields']['name']
    if name == 'vf.mem_bar':
        mode = ctx.attrs(node, {'mode'}).get('mode', 0)  # Pro prints VST_VLD for an absent mode.
        o.need(type(mode) is int and 0 <= mode < len(MEM_BAR_MODES), node,
               'VF memory barrier mode must be one of the twelve MemBarMode indices')
        o.need(not args, node, 'VF memory barrier takes no operands')
        _, src, dst = MEM_BAR_MODES[mode]
        ctx.emit('vf.barrier', node, attrs={'src': Ident(src), 'dst': Ident(dst)})
        return None
    if name == 'vf.and_':
        mode = ctx.attrs(node, {'mode'}).get('mode', 0)
        o.need(type(mode) is int and mode == 0, node, 'Predicate and admits ZEROING only')
        o.need(len(args) == 4 and all(predicate(v) and v.type == MaskType(32) for v in args), node,
               'Predicate and requires four b32 predicates')
        ctx.emit('vf.mask_and', node, args[:3], attrs={'mask': args[3]})
        return None
    ctx.attrs(node)
    tile, pred = args if name == 'vf.store_align' else args[::-1]
    # A5 returned b32 and b16 images bit for bit (p6-fused-mask, p7-vf-probes).
    o.need(isinstance(tile, Memory) and tile.value.type.space == 'ub' and tile.pitch is None
           and pred.type in (MaskType(32), MaskType(16)) and prod(tile.shape) * tile.value.type.dtype.bits >= 256, node,
           'Predicate spill/fill needs a b32 or b16 predicate and a whole UB tile of at least 32 bytes')
    if name == 'vf.store_align':
        ctx.emit('vf.mask_to_ub', node, (tile.value, pred))
        ctx.writes.append(tile.value)
    else:
        ctx.emit('vf.ub_to_mask', node, (pred, tile.value))
        ctx.reads.append(tile.value)
    return None
