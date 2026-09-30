# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""VF tile offsets, data-block copies and unaligned load/store chains.

Inside a VF, Pro binds `tile + k` to a name and prints the element pointer `(base + (k))`. Post-updating
accesses share one cursor per printed pointer for the whole section (`GetOrCreateVFTilePtr`), so cursors are
keyed by (window, literal offset) and every position is tracked statically. Block copies follow the A5 rules of
RFC-0001 at every stride; post-updating block copies fold their cursor into static offsets.
"""
from dataclasses import dataclass, field
from math import prod

from ...ir import REGISTRY, Ident, Value
from ...ir.types import MaskType, MemType, UnalignRegType
from .lower import Memory
from .vector_predicates import storage_reg

UNALIGNED = {'vf.load', 'vf.store', 'vf.load_unalign_init', 'vf.unalign_reg_for_store', 'vf.load_unalign_pre',
             'vf.load_unalign', 'vf.store_unalign', 'vf.store_unalign_post'}
CURSOR = ('vf.ub_cursor',)  # A cursor at an offset also slices its window; mem.slice has other owners in the index.
TARGETS = {'vf.load_align (data block)': ('vf.load',), 'vf.store_align (data block)': ('vf.store',),
           'vf.load_unalign_init': ('vf.unalign',), 'vf.unalign_reg_for_store': ('vf.unalign',),
           'vf.load_unalign_pre': ('vf.load_unalign_pre',), 'vf.load_unalign': ('vf.load_unalign', *CURSOR),
           'vf.load': ('vf.unalign', 'vf.load_unalign_pre', 'vf.load_unalign', *CURSOR),
           'vf.store': ('vf.unalign', 'vf.store_unalign', 'vf.store_unalign_post', *CURSOR)}
DTYPES = {'f32', 'i32', 'f16'}  # BF16 stays refused: no fixture or device evidence.
CHAIN_AFTER_ANY_STRIDE = False  # Whether a primed load chain survives a stride other than one register.
UNALIGNED_REGISTERS = 4  # Pro documents at most four unaligned registers per VF section.
BLOCK_ATTRS = {'data_copy_mode', 'block_stride', 'repeat_stride', 'post_update', 'dist'}
CONTROL = 'VF memory access inside VF control flow needs its own cursor and footprint rule'
SHARED = 'A post-update pointer serves either block copies or unaligned access, not both'
UNPRIMED = ('Unaligned load reads an address its register is not primed at: prime it there, '
            'or continue a chain only after a whole-register stride')


@dataclass(frozen=True)
class Address:
    memory: Memory
    offset: int  # Elements of the tile dtype.


@dataclass
class Cursor:
    value: Value
    position: int
    role: str


@dataclass
class Chains:
    cursors: dict = field(default_factory=dict)  # (window name, literal offset or None) -> Cursor
    primed: dict = field(default_factory=dict)  # load register name -> (window name, element) or None
    registers: int = 0


def claims(name, node, args):
    return name in UNALIGNED or name in {'vf.load_align', 'vf.store_align'} and (
        'data_copy_mode' in node['fields']['kwargs'] or any(isinstance(arg, Address) for arg in args))


def address(ctx, node, args):
    o = ctx.o
    ctx.attrs(node)
    memory = args[0] if len(args) == 3 else None
    o.need(isinstance(memory, Memory) and memory.value.type.space == 'ub' and memory.pitch is None
           and memory.valid == memory.shape and args[2] == memory.shape
           and node['fields']['type'] == o.node(node['fields']['args'][0])['fields'].get('type'), node,
           'VF tile offsets need a plain UB tile window')
    o.need(type(args[1]) is int, node, 'VF tile offsets must be integer literals')
    o.need(0 <= args[1] < prod(memory.shape), node, 'VF tile offset lies outside the tile')
    return Address(memory, args[1])


def convert(ctx, node, args):
    o, name = ctx.o, node['fields']['name']
    if name in {'vf.load_align', 'vf.store_align'}:
        return block_copy(ctx, node, args)
    o.need(not ctx.depth, node, CONTROL)
    if name in {'vf.load_unalign_init', 'vf.unalign_reg_for_store'}:
        ctx.attrs(node)
        o.need(not args, node, 'Unaligned register declarations take no operands')
        return register(ctx, node, 'load' if name == 'vf.load_unalign_init' else 'store')
    if name in {'vf.store_unalign', 'vf.store_unalign_post'}:
        return chain_store(ctx, node, args)
    if name == 'vf.load_unalign_pre':
        ctx.attrs(node)
        o.need(len(args) == 2 and loader(ctx, args[0]), node, 'load_unalign_pre needs an unaligned load register and a tile')
        memory, offset, _, _ = place(ctx, node, args[1], unaligned=True)
        ctx.emit('vf.load_unalign_pre', node, (args[0], memory.value), attrs={'offset': offset})
        chains(ctx).primed[args[0].name] = (memory.value.name, offset)
        ctx.reads.append(memory.value)
        return None
    return load(ctx, node, args) if name in {'vf.load', 'vf.load_unalign'} else store(ctx, node, args)


def chains(ctx):
    if ctx.vf_memory is None:
        ctx.vf_memory = Chains()
    return ctx.vf_memory


def place(ctx, node, operand, reg=None, unaligned=False):
    """(window, literal offset, cursor key, register lanes) of a VF tile operand."""
    o = ctx.o
    if isinstance(operand, Address):
        memory, offset, key = operand.memory, operand.offset, operand.offset
    else:
        o.need(isinstance(operand, Memory) and operand.value.type.space == 'ub' and operand.pitch is None
               and operand.valid == operand.shape, node, 'VF memory access needs a plain UB tile or a literal offset of one')
        memory, offset, key = operand, 0, None
    dt = memory.value.type.dtype
    o.need(dt.name in DTYPES, node, 'VF block copies and unaligned access admit FP32, INT32 and FP16 tiles')
    o.need(reg is None or storage_reg(reg) and reg.type.dtype == dt, node, 'VF memory access needs a register of the tile dtype')
    o.need(not unaligned or memory.shape[0] == 1, node, 'Unaligned access admits single-row tiles')
    return memory, offset, (memory.value.name, key), 256 // (dt.bits // 8)


def uniform(ctx, mask, dt):
    """A predicate still holding its static create_mask pattern, whole-block in each 32-byte block."""
    if not (isinstance(mask, Value) and mask.type == MaskType(dt.bits)):
        return False
    pattern, scopes, scope = None, [], ctx
    while scope is not None:  # A branch sees the ops of the contexts around it.
        scopes, scope = scopes + scope.ops, scope.parent
    for op in (inner for top in scopes for inner in top.walk()):
        if mask in op.results and op.opcode == 'vf.mask':
            pattern = str(op.attrs['init'])
        uses = zip(op.operands, REGISTRY.get(op.opcode).operands, strict=False)
        if any(v == mask and use.access in {'write', 'readwrite'} for v, use in uses):
            return False
    lanes = 32 // (dt.bits // 8)
    return pattern in {'all', 'none'} or pattern is not None and pattern.startswith('vl') and int(pattern[2:]) % lanes == 0


def block_copy(ctx, node, args):
    """A5 loads a block at strides other than 1 when any of its predicate bits is set and stores predicate-active
    lanes in lane order; stride 0 puts every block at block 0 (RFC-0001, I035). A stride-1 load prints the lane copy,
    so only whole-block predicates keep it equal to Pro's block test."""
    o, name = ctx.o, node['fields']['name']
    attrs = ctx.attrs(node, BLOCK_ATTRS)
    o.need('data_copy_mode' in attrs, node, 'Contiguous load_align/store_align through a VF tile offset needs its own rule')
    o.need(type(attrs['data_copy_mode']) is int and attrs['data_copy_mode'] in (1, 2), node,
           'Only DATA_BLOCK_LOAD/DATA_BLOCK_COPY select a block copy')
    o.need('dist' not in attrs, node, "Pro's block-copy emitters ignore dist")
    loading = name == 'vf.load_align'
    o.need(len(args) == 3 or not loading and len(args) in (4, 5), node,
           'Block copies take a register, a tile and a predicate; a store may add positional block and repeat strides')
    reg, tile, mask = args[:3] if loading else (args[1], args[0], args[2])
    o.need(len(args) < 4 or 'block_stride' not in attrs, node, 'Block stride is given twice')
    o.need(len(args) < 5 or 'repeat_stride' not in attrs, node, 'Repeat stride is given twice')
    stride = args[3] if len(args) >= 4 else attrs.get('block_stride', 0)  # Pro prints 0 for an absent stride.
    repeat, post = args[4] if len(args) == 5 else attrs.get('repeat_stride'), attrs.get('post_update', False)
    o.need(type(stride) is int, node, 'Block copies need a literal block stride')
    o.need(0 <= stride < 1 << 15, node, 'Block stride must stay below 32768: it fills the upper half of an INT32 config word')
    o.need(type(post) is bool and (repeat is None or post), node,
           'repeat_stride needs post_update=True: without it vsldb drops the stride and vsstb prints it unmeasured')
    o.need(repeat is None or type(repeat) is int and 0 <= repeat < 1 << 16, node,
           'Repeat stride must be a literal below 65536: it fills the lower half of the config word')
    memory, offset, key, _ = place(ctx, node, tile, reg)
    dt = memory.value.type.dtype
    esize = dt.bits // 8
    if post:
        o.need(not ctx.depth, node, 'Post-updating block copies inside VF control flow need a path-sensitive cursor')
        block_cursor = chains(ctx).cursors.setdefault(key, Cursor(None, offset, 'block'))
        o.need(block_cursor.role == 'block', node, SHARED)
        offset = block_cursor.position
    o.need(offset * esize % 32 == 0, node, 'Block copy base violates 32-byte alignment')
    o.need(offset * esize + (7 * stride + 1) * 32 <= prod(memory.shape) * esize, node, 'Block copy footprint exceeds the tile')
    o.need(isinstance(mask, Value) and mask.type == MaskType(dt.bits), node, 'Block copy predicates need the register lane width')
    o.need(not loading or stride != 1 or uniform(ctx, mask, dt), node,
           'A stride-1 block load prints the lane copy but A5 reads whole blocks: its predicate must be an unwritten '
           'whole-block create_mask pattern')
    attributes = {'offset': offset, 'blk_stride': stride, 'mask': mask}
    if loading:
        ctx.emit('vf.load', node, (reg, memory.value), attrs=attributes)
        ctx.reads.append(memory.value)
    else:
        ctx.emit('vf.store', node, (memory.value, reg), attrs=attributes)
        ctx.writes.append(memory.value)
    if post:  # A5 advances the pointer repeat_stride 32-byte blocks after the copy (p7-vf-probes).
        block_cursor.position += (repeat or 0) * (32 // esize)


def register(ctx, node, role):
    state = chains(ctx)
    state.registers += 1
    ctx.o.need(state.registers <= UNALIGNED_REGISTERS, node, 'Pro documents at most four unaligned registers per VF section')
    value = ctx.emit('vf.unalign', node, typ=UnalignRegType(role))
    if role == 'load':
        state.primed[value.name] = None
    return value


def loader(ctx, value):
    return isinstance(value, Value) and value.type == UnalignRegType('load') and value.name in chains(ctx).primed


def cursor(ctx, node, memory, offset, key, role):
    """The section cursor of one printed pointer, created where it is first used."""
    state = chains(ctx)
    if key not in state.cursors:
        window = memory.value
        if key[1] is not None and offset:
            extents = [1, memory.shape[1] - offset]
            window = ctx.emit('mem.slice', node, (window,), typ=MemType('ub', window.type.dtype, tuple(extents)),
                              attrs={'offsets': [0, offset], 'extents': extents})
        state.cursors[key] = Cursor(ctx.emit('vf.ub_cursor', node, (window,), typ=window.type), offset, role)
    found = state.cursors[key]
    ctx.o.need(found.role != 'block', node, SHARED)
    ctx.o.need(found.role == role, node, 'A post-update cursor serves either loads or stores, not both')
    return found


def literal(ctx, node, value, low, high, what):
    ctx.o.need(type(value) is int and low <= value <= high, node, f'Unaligned {what} must be a literal in [{low}, {high}]')
    return value


def load(ctx, node, args):
    o, name = ctx.o, node['fields']['name']
    unified = name == 'vf.load'
    if unified:
        o.need(not node['fields']['kwargs'], node, 'The pinned vf.load emitter ignores count, repeat_stride and post_update')
        o.need(len(args) in (2, 3), node, 'vf.load takes a register, a tile and an optional stride')
        reg, ureg, target, rest = args[0], None, args[1], args[2:]
    else:
        ctx.attrs(node)
        o.need(len(args) in (3, 4) and loader(ctx, args[1]), node,
               'load_unalign needs a register, an unaligned load register and a tile')
        reg, ureg, target, rest = args[0], args[1], args[2], args[3:]
    memory, offset, key, lanes = place(ctx, node, target, reg, unaligned=True)
    size, state, window = prod(memory.shape), chains(ctx), memory.value
    if unified:
        ureg = register(ctx, node, 'load')
    if not rest:
        o.need(offset + lanes <= size, node, 'Unaligned load reads past the tile')
        if unified:
            ctx.emit('vf.load_unalign_pre', node, (ureg, window), attrs={'offset': offset})
        o.need(unified or state.primed[ureg.name] == (window.name, offset), node, UNPRIMED)
        ctx.emit('vf.load_unalign', node, (reg, window, ureg), attrs={'offset': offset})
        primed = None
    else:
        stride = literal(ctx, node, rest[0], 0, size, 'load strides')
        found = cursor(ctx, node, memory, offset, key, 'load')
        o.need(found.position + max(lanes, stride) <= size, node, 'Unaligned load reads or advances past the tile')
        if unified:
            ctx.emit('vf.load_unalign_pre', node, (ureg, found.value), attrs={'offset': 0})
        o.need(unified or state.primed[ureg.name] == (window.name, found.position), node, UNPRIMED)
        ctx.emit('vf.load_unalign', node, (reg, found.value, ureg),
                 attrs={'offset': 0, 'stride': stride, 'post_mode': Ident('update')})
        found.position += stride
        primed = (window.name, found.position) if stride == lanes or CHAIN_AFTER_ANY_STRIDE else None
    state.primed[ureg.name] = primed
    ctx.reads.append(window)


def store(ctx, node, args):
    o = ctx.o
    o.need(not node['fields']['kwargs'], node, 'The pinned vf.store emitter ignores repeat_stride and post_update')
    o.need(len(args) in (2, 3), node, 'vf.store takes a tile, a register and an optional count')
    memory, offset, key, lanes = place(ctx, node, args[0], args[1], unaligned=True)
    count = literal(ctx, node, args[2] if len(args) == 3 else lanes, 1, lanes, 'store counts')
    ureg = register(ctx, node, 'store')
    found = cursor(ctx, node, memory, offset, key, 'store')
    o.need(found.position + count <= prod(memory.shape), node, 'Unaligned store writes past the tile')
    ctx.emit('vf.store_unalign', node, (found.value, args[1], ureg),
             attrs={'offset': 0, 'count': count, 'post_mode': Ident('update')})
    ctx.emit('vf.store_unalign_post', node, (found.value, ureg), attrs={'offset': 0, 'stride': 0, 'post_mode': Ident('update')})
    found.position += count
    ctx.writes.append(memory.value)


def chain_store(ctx, node, args):
    """Refused: NORM vstus/vstas do not compile and vstur/vstar follow the AR count. A chain flushed by
    store_unalign_post(post_update=True) needs an IR extension for the tail A5 keeps across registers and VF calls."""
    o = ctx.o
    ctx.attrs(node, {'post_update'})
    if node['fields']['name'] == 'vf.store_unalign':
        o.need(len(args) == 4, node, 'Three-argument store_unalign (vstur) follows the AR count and may hang')
        o.need(storage_reg(args[1]), node, 'store_unalign admits a data register; mask sources (pstu) need their own rule')
        o.need(node['fields']['kwargs'].get('post_update') is True, node,
               'NORM store_unalign does not compile: its vstus form accepts only POST_UPDATE')
    o.fail(node, 'Chained unaligned stores need an IR extension: A5 keeps an unflushed tail across registers and VF calls, '
                 'and Lowered IR has no pending-store state to prove every chain flushed')
