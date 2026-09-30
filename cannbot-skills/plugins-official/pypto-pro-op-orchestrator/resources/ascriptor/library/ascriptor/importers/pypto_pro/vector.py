# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Audited VF mappings; tables are shared with coverage tools."""

from math import prod

from ...ir import Ident, Value
from ...ir.types import MaskType, RegType, ScalarType, dtype
from .lower import Memory

UNARY = {f'vf.{name}': f'vf.{name}' for name in ('abs', 'exp', 'ln', 'log', 'log2', 'log10', 'sqrt', 'neg', 'relu')}
BINARY = {f'vf.{name}': f'vf.{name}' for name in ('add', 'sub', 'mul', 'max', 'min', 'div')}
BINARY['vf.abs_sub'] = 'vf.abssub'
BINARY['vf.prelu'] = 'vf.prelu'
UNARY['vf.pair_reduce_sum'] = 'vf.cpadd'
SCALAR = {f'vf.{name}': f'vf.{name}' for name in ('adds', 'muls', 'maxs', 'mins')}
SCALAR['vf.leaky_relu'] = 'vf.lrelu'
REDUCE = {'vf.reduce_sum': 'vf.cadd'}
ARITHMETIC = {**UNARY, **BINARY, **SCALAR, **REDUCE}


def fp32_scalar(value):
    """An FP32 scalar value, a rounded float literal or an exactly representable integer literal."""
    return (type(value) is float or type(value) is int and abs(value) <= 2**24
            or isinstance(value, Value) and value.type == ScalarType(dtype('f32')))


def arithmetic(ctx, node, args):
    o, name = ctx.o, node['fields']['name']
    allowed = {'merge_mode', 'datablock'} if name in REDUCE else {'mode'}
    attrs = ctx.attrs(node, allowed)
    mode = attrs.get('merge_mode' if name in REDUCE else 'mode', 0)
    o.need(type(mode) is int and mode == 0, node, 'VF import requires ZEROING mode')
    if name in REDUCE:
        o.need(attrs.get('datablock', False) is False, node, 'Datablock reduction requires a separate rule')
    unary = name in UNARY or name in REDUCE
    o.need(len(args) == (3 if unary else 4), node, 'VF arithmetic needs destination, sources and predicate')
    dst, a, mask = args[0], args[1], args[-1]
    o.need(isinstance(dst, Value) and isinstance(a, Value) and dst.type == a.type == RegType(dtype('f32'))
           and isinstance(mask, Value) and mask.type == MaskType(32), node, 'Unsupported VF arithmetic type/mask')
    if not unary:
        b = args[2]
        if name in SCALAR:
            o.need(fp32_scalar(b), node, 'Unsupported VF scalar operand')
        else:
            o.need(isinstance(b, Value) and b.type == dst.type, node, 'VF operand dtype mismatch')
    ctx.emit(ARITHMETIC[name], node, args[:-1], attrs={'mask': mask})


def vector_call(ctx, node, args):
    from . import vector_masks, vector_memory, vector_spr
    from .vector_fused import TARGETS as FUSED
    from .vector_fused import convert as fused
    from .vector_fused_cast import TARGETS as FUSED_CAST
    from .vector_fused_cast import convert as fused_cast
    from .vector_index import TARGETS as INDEX
    from .vector_index import convert as index
    from .vector_indexed import SOURCES as INDEXED
    from .vector_indexed import convert as indexed
    from .vector_integer import TARGETS as INTEGER
    from .vector_integer import convert as integer
    from .vector_predicates import PREDICATES, convert, storage_reg
    from .vector_rearrange import TARGETS as REARRANGE
    from .vector_rearrange import convert as rearrange

    o, name = ctx.o, node["fields"]["name"]
    if vector_spr.claims(name, node, args):
        return vector_spr.convert(ctx, node, args)
    if vector_masks.claims(name, args):
        return vector_masks.convert(ctx, node, args)
    if vector_memory.claims(name, node, args):
        return vector_memory.convert(ctx, node, args)
    if name in FUSED_CAST:
        return fused_cast(ctx, node, args)
    if name in FUSED:
        return fused(ctx, node, args)
    if name in INDEX:
        return index(ctx, node, args)
    if name in INDEXED:
        return indexed(ctx, node, args)
    if name in REARRANGE:
        return rearrange(ctx, node, args)
    if name in INTEGER:
        return integer(ctx, node, args)
    if name in PREDICATES:
        return convert(ctx, node, args)
    if name in ARITHMETIC:
        return arithmetic(ctx, node, args)
    if name in {"vf.create_mask", "vf.mask_reg", "vf.reg_tensor"}:
        attrs = ctx.attrs(node, {"dtype", "pattern"} if name == "vf.create_mask" else {"dtype"})
        dt = o.dt(attrs.get('dtype'), node)
        # UINT8/16/32 registers and b8 predicates are VF-local carriers; storage keeps its four dtypes.
        o.need(not args and dt.name in {'f32', 'i32', 'f16', 'bf16', 'u8', 'u16', 'u32'}, node,
               'VF import requires a supported numeric producer')
        if name != 'vf.reg_tensor':
            patterns = {0: 'all', 1: 'none', **{i+2: f'vl{n}' for i, n in enumerate((1, 2, 3, 4, 8, 16, 32, 64))}}
            if dt.bits in (8, 16):
                patterns[10] = 'vl128'
            pattern = attrs.get('pattern', 0 if name == 'vf.create_mask' else 1)
            o.need(type(pattern) is int and pattern in patterns, node, 'Unsupported VF mask pattern')
            return ctx.emit('vf.mask', node, typ=MaskType(dt.bits), attrs={'init': Ident(patterns[pattern])})
        return ctx.emit('vf.reg', node, typ=RegType(dt))
    ctx.attrs(node)
    if name in {"vf.load_align", "vf.store_align"}:
        o.need(len(args) == 3, node, "Only the basic aligned VF overload is admitted")
        if name == "vf.load_align":
            register, memory, offset = args
            mask = None
        else:
            memory, register, mask = args
            offset = 0
        o.need(isinstance(memory, Memory) and memory.value.type.space == "ub"
               and storage_reg(register) and memory.value.type.dtype == register.type.dtype,
               node, "VF data/mask overload or dtype needs a separate rule")
        element_bytes = register.type.dtype.bits // 8
        lanes, alignment = 256 // element_bytes, 32 // element_bytes
        o.need(type(offset) is int and offset >= 0 and offset % alignment == 0 and offset + lanes <= prod(memory.shape),
               node, "VF full-register access exceeds or misaligns the backing storage")
        o.need(memory.pitch is None or memory.pitch == memory.shape[1], node,
               "VF access to pitched subviews needs a dedicated footprint rule")
        if mask is not None:
            o.need(isinstance(mask, Value) and mask.type == MaskType(register.type.dtype.bits), node, "Store predicate width must match its register")
            ctx.emit("vf.store_cont", node, (memory.value, register), attrs={"offset": offset, "mask": mask})
            ctx.writes.append(memory.value)
        else:
            ctx.emit("vf.load_cont", node, (register, memory.value), attrs={"offset": offset})
            ctx.reads.append(memory.value)
        return None
    o.fail(node, f"No VF instruction conversion for {name}")
    return None
