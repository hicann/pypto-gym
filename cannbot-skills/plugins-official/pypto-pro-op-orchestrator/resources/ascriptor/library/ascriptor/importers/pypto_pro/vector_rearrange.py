# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Register value copies, bitcast snapshots and two-output lane permutations."""
from ...ir import Value
from ...ir.types import RegType
from .vector_predicates import storage_reg

TARGETS = {'vf.move': ('vf.copy',), 'vf.bit_cast': ('vf.reinterpret', 'vf.copy'),
           'vf.interleave': ('vf.interleave',), 'vf.de_interleave': ('vf.deinterleave',)}
CARRIERS = ('u8', 'u16', 'u32')  # VF-local unsigned views of INT32 words; lanes are little-endian bytes.
BITCAST_PAIRS = {('f32', 'i32'), ('i32', 'f32'), ('f16', 'bf16'), ('bf16', 'f16'),
                 *(('i32', u) for u in CARRIERS), *((u, 'i32') for u in CARRIERS)}


def carrier(value):
    return (isinstance(value, Value) and isinstance(value.type, RegType) and value.type.n == 1
            and value.type.dtype.name in CARRIERS)


def convert(ctx, node, args):
    o, name = ctx.o, node['fields']['name']
    if name == 'vf.move':
        ctx.attrs(node)
        o.need(len(args) == 2 and all(storage_reg(v) for v in args) and args[0].type == args[1].type,
               node, 'Only unmasked same-type register move is admitted; masked move requires MERGING')
        ctx.emit(TARGETS[name][0], node, args)
        return None
    attrs = ctx.attrs(node, {'dtype'})
    if name == 'vf.bit_cast':
        o.need(len(args) in (1, 2) and all(storage_reg(v) or carrier(v) for v in args), node,
               'Bitcast requires an admitted source register and optional destination')
        target = o.dt(attrs.get('dtype'), node)
        o.need((args[-1].type.dtype.name, target.name) in BITCAST_PAIRS, node,
               'Bitcast admits FP32/INT32 and FP16/BF16 same-width pairs and INT32 to or from UINT8/UINT16/UINT32')
        typ = RegType(target)
        if len(args) == 2:
            dst = args[0]
            o.need(dst.type == typ, node, 'Bitcast destination contradicts its dtype attribute')
        else:
            dst = ctx.emit('vf.reg', node, typ=typ)
        view = ctx.emit(TARGETS[name][0], node, (args[-1],), typ=typ)
        # Native assignment uses value assignment (or `auto`), not `auto&`.
        ctx.emit(TARGETS[name][1], node, (dst, view))
        return dst if len(args) == 1 else None
    o.need(len(args) == 4 and all(storage_reg(v) for v in args)
           and len({v.type for v in args}) == 1, node, 'Register permutations require four matching register types')
    o.need(args[0] != args[1] and not set(args[:2]).intersection(args[2:]), node,
           'Permutation outputs must be distinct from each other and the inputs')
    if 'dtype' in attrs:
        o.need(o.dt(attrs['dtype'], node) == args[0].type.dtype, node,
               'Permutation dtype reinterpretation needs a separate rule')
    ctx.emit(TARGETS[name][0], node, args)
    return None
