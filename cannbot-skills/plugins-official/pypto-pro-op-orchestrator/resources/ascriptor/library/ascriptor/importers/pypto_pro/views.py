# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""GM pointer views and same-buffer tile aliases; declarations over one root storage."""
from dataclasses import dataclass
from math import prod

from ...ir import Ident, Value
from ...ir.types import DType, MemType, ScalarType, dtype
from .lower import Memory, cast_bounds
from .selection import binary_bounds, interval

VIEW_TYPES = {'f16', 'bf16', 'f32', 'i32'}
TARGETS = {'ptr.make_tensor': ('mem.view',), 'block.make_tile': ('mem.reinterpret', 'mem.reshape')}


@dataclass(frozen=True)
class Pointer:
    """A GM element pointer: its root parameter, element origin and addressed element type."""
    root: Memory
    origin: int | Value
    dtype: DType


def pointer_call(ctx, node, args):
    handler = {'ptr.make_ptr': make_ptr, 'ptr.addptr': addptr, 'ptr.make_tensor': make_tensor}.get(node['fields']['name'])
    ctx.o.need(handler is not None, node, f"No pointer conversion for {node['fields']['name']}")
    return handler(ctx, node, args)


def gm_pointer(ctx, node, source):
    """The pointer of a GM tensor source. A view gives its own origin and dtype, not its strides or extent:
    Pro CCE reuses the view's registered base pointer, and A5 reads views of views and pointers of views there."""
    o = ctx.o
    o.need(isinstance(source, Memory) and source.value.type.space == 'gm', node,
           'Pointer views need a GM tensor parameter')
    root = source.root or source
    o.need(all(type(n) is int for n in root.shape), node, 'GM views need a static root shape')
    return Pointer(root, source.origin, source.value.type.dtype)


def element_type(ctx, node, root, current):
    o = ctx.o
    attrs = ctx.attrs(node, {'dtype'})
    result = o.dt(attrs['dtype'], node) if 'dtype' in attrs else current
    o.need(result.name in VIEW_TYPES and result.bits == root.value.type.dtype.bits, node,
           'GM views keep the element width of their root')
    return result


def pointer_type(ctx, node, element):
    o = ctx.o
    typ = o.node(node['fields']['type'])
    o.need(typ['kind'] == 'PtrType' and o.dt(typ['fields']['dtype'], node) == element, node,
           'Pointer type disagrees with its operands')


def make_ptr(ctx, node, args):
    ctx.o.need(len(args) == 1, node, 'Expected one pointer source')
    source = args[0] if isinstance(args[0], Pointer) else gm_pointer(ctx, node, args[0])
    result = Pointer(source.root, source.origin, element_type(ctx, node, source.root, source.dtype))
    pointer_type(ctx, node, result.dtype)
    return result


def addptr(ctx, node, args):
    o = ctx.o
    ctx.attrs(node)
    o.need(len(args) == 2 and isinstance(args[0], Pointer), node, 'Pointer arithmetic needs a GM tensor pointer')
    pointer, offset = args
    # A reassigned offset is read where the pointer is computed: Pro CCE copies it into a local (view_mutable_offset on A5).
    offset = ctx.snapshot(offset, node)
    if isinstance(offset, Value):
        o.need(isinstance(offset.type, ScalarType) and offset.type.dtype.name in {'i32', 'i64'}, node,
               'Pointer offsets must be signed integers')
        if offset.type.dtype.name == 'i32':
            wide = ctx.emit('scalar.cast', node, (offset,), typ=ScalarType(dtype('i64')))
            cast_bounds(ctx, offset, wide)
            offset = wide
    else:
        o.need(type(offset) is int, node, 'Pointer offsets must be signed integers')
    pointer_type(ctx, node, pointer.dtype)
    if offset == 0 or pointer.origin == 0:
        origin = pointer.origin if offset == 0 else offset
    elif type(offset) is int and type(pointer.origin) is int:
        origin = pointer.origin + offset
    else:
        origin = ctx.emit('scalar.add', node, (pointer.origin, offset), typ=ScalarType(dtype('i64')))
        binary_bounds(ctx, 'Add', pointer.origin, offset, origin)
    return Pointer(pointer.root, origin, pointer.dtype)


def make_tensor(ctx, node, args):
    o = ctx.o
    o.need(len(args) == 3, node, 'Expected a pointer, a shape and strides')
    source, shape, strides = args
    pointer = source if isinstance(source, Pointer) else gm_pointer(ctx, node, source)
    root = pointer.root
    element = element_type(ctx, node, root, pointer.dtype)
    o.need(isinstance(shape, tuple) and len(shape) == 2 and all(type(n) is int and n > 0 for n in shape), node,
           'Only static two-dimensional GM views are admitted')
    rows, cols = shape
    o.need(isinstance(strides, tuple) and len(strides) in (0, 2) and all(type(s) is int for s in strides), node,
           'GM view strides must be static and match the view rank')
    row_stride, inner = strides or (cols, 1)
    o.need(inner == 1, node, 'GM views need an innermost stride of 1')
    o.need(rows == 1 or row_stride >= cols, node, 'GM view rows must not overlap')
    typ = o.node(node['fields']['type'])
    fields = typ['fields'] if typ['kind'] == 'TensorType' else {}
    view = o.node(fields['tensor_view'])['fields'] if fields.get('tensor_view') else None
    o.need(view is not None and fields['memref'] is None and o.dt(fields['dtype'], node) == element
           and tuple(o.literal(n, node) for n in fields['shape']) == shape
           and tuple(o.literal(s, node) for s in view['stride']) == strides and view['layout']['name'] == 'ND'
           and not view['valid_shape'] and view['ptr'] == node['fields']['args'][0], node,
           'View descriptor disagrees with its operands')
    row_stride = cols if rows == 1 else row_stride  # One row reads no row stride.
    bounds = interval(ctx, pointer.origin)
    o.need(bounds is not None and bounds[0] >= 0 and bounds[1] + (rows - 1) * row_stride + cols <= prod(root.shape), node,
           'GM view declaration must be proven inside its root tensor')
    if pointer.origin == 0 and shape == root.shape and row_stride == cols and element == root.value.type.dtype:
        return root
    value = ctx.emit('mem.view', node, (root.value,), typ=MemType('gm', element, shape),
                     attrs={'shape': list(shape), 'strides': [row_stride, 1], 'offset': pointer.origin})
    return Memory(value, shape, shape, row_stride, root, pointer.origin)


def scalar_address(ctx, node, memory, index):
    """getval/setval print `*((T*)ptr + index)` over a GM view's pointer and `GetValue(index)` over a Vec tile alias:
    a flat element index from the view's origin, or into the alias's elements (view_scalar_probe on A5)."""
    root = memory.root
    if root is None or memory.value.type.space == 'ub':
        return memory.value, index
    element, value = memory.value.type.dtype, root.value
    if element != value.type.dtype:
        value = ctx.emit('mem.reinterpret', node, (value,), typ=MemType('gm', element, value.type.dims))
    index = ctx.snapshot(index, node)
    if memory.origin == 0 or type(index) is int and type(memory.origin) is int:
        return value, memory.origin + index
    wide = ScalarType(dtype('i64'))
    terms = []
    for term in (memory.origin, index):
        if isinstance(term, Value) and term.type != wide:
            ctx.o.need(isinstance(term.type, ScalarType) and term.type.dtype.name == 'i32', node, 'Scalar indices must be signed integers')
            cast = ctx.emit('scalar.cast', node, (term,), typ=wide)
            cast_bounds(ctx, term, cast)
            term = cast
        terms.append(term)
    address = ctx.emit('scalar.add', node, tuple(terms), typ=wide)
    binary_bounds(ctx, 'Add', *terms, address)
    return value, address


@dataclass(frozen=True)
class Fractals:
    """An NZ-declared Vec tile alias: compact NZ fractals of its root's bytes, read only by inserts."""
    memory: Memory


def alias(ctx, node, entry, element, shape, valid, reserved, layout=None, metadata=False, slot=False):
    """A later declaration at an allocation's address re-declares those bytes, starting from its own valid shape.
    Each side of a paired kernel re-declares its own linked root. Pro CCE folds a Vec declaration identical to an
    earlier one without valid-shape metadata into a reference to it, which shares its valid state, and identical
    declarations with metadata stay distinct handles (alias_identical_probe on A5). Mixed metadata, Mat tiles and
    tile group slots were not measured."""
    o = ctx.o
    root = entry['memory']
    source = root.value.type
    o.need(source.space in {'ub', 'l1'}, node, 'Mat and L0 tile aliases other than an NZ/ZN transposition need layout legalization')
    o.need(reserved <= entry['size'] and prod(shape) * element.bits <= prod(root.shape) * source.dtype.bits, node,
           'Tile alias exceeds the elements of its first declaration')
    key, kind = (element, shape, layout), 'slot' if slot else 'metadata' if metadata else 'fold'
    earlier = entry['declared'].get(key)
    if earlier is not None:
        o.need(source.space == 'ub', node, 'Identical Mat re-declarations are unmeasured: A5 measured a shared valid state for Vec tiles')
        o.need('slot' not in (kind, earlier[0]), node,
               'Overlapping physical allocations of tile group slots need alias/backing-storage legalization')
        o.need(kind == earlier[0], node, 'Identical Vec declarations with and without valid-shape metadata are unmeasured')
        if kind == 'fold':
            return earlier[1]
    result = _alias(ctx, node, root, source, element, shape, valid, layout)
    entry['declared'].setdefault(key, (kind, result))
    return result


def _alias(ctx, node, root, source, element, shape, valid, layout):
    o = ctx.o
    if source.space == 'l1':
        # NZ(A[m, n]) and ZN(A.T[n, m]) are the same bytes, so the alias is its root's tile (mat_zn_alias on A5).
        o.need(element == source.dtype and shape[::-1] == root.shape and (layout == 'zn') != root.transposed, node,
               'Mat and L0 tile aliases other than an NZ/ZN transposition need layout legalization')
        return Memory(root.value, shape, valid, root=root, transposed=layout == 'zn')
    o.need(layout != 'nz' or element == source.dtype, node, 'NZ Vec aliases keep their root dtype')
    value, dims = root.value, tuple(root.shape)
    if element != source.dtype:
        o.need(dims[1] * source.dtype.bits % element.bits == 0, node, 'Tile alias rows must hold whole elements')
        dims = (dims[0], dims[1] * source.dtype.bits // element.bits)
        value = ctx.emit('mem.reinterpret', node, (value,), typ=MemType('ub', element, dims))
    if shape != dims:
        value = ctx.emit('mem.reshape', node, (value,), typ=MemType('ub', element, shape), attrs={'shape': list(shape)})
    if layout == 'nz':
        value = ctx.emit('mem.reinterpret', node, (value,), typ=MemType('ub', element, shape, 'nz'), attrs={'layout': Ident('nz')})
        return Fractals(Memory(value, shape, valid, root=root))
    return Memory(value, shape, valid, root=root)
