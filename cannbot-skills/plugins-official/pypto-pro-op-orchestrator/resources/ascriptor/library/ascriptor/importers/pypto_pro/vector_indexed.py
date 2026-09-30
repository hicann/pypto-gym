# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Indexed register/UB gathers and scatters, lane compaction, narrowing pack and byte histograms."""
from ...ir import Ident, Value
from ...ir.types import MaskType, RegType
from .lower import Memory
from .vector_memory import Address

SOURCES = {'vf.gather', 'vf.scatter', 'vf.squeeze', 'vf.unsqueeze', 'vf.pack', 'vf.histograms'}
TARGETS = {'vf.gather (register)': ('vf.gather',), 'vf.gather (NORM)': ('vf.gather_copy',),
           'vf.gather (DATA_BLOCK_LOAD)': ('vf.gatherb',), 'vf.scatter': ('vf.scatter_copy',),
           'vf.squeeze': ('vf.squeeze',), 'vf.unsqueeze': ('vf.unsqueeze',), 'vf.pack': ('vf.pack',),
           'vf.histograms': ('vf.histograms',)}
# vselr casts every operand to its unsigned carrier: the index needs only the data width.
SELECT = {'f32': {'i32', 'u32'}, 'i32': {'i32', 'u32'}, 'f16': {'u16'}, 'u8': {'u8'}}
ELEMENT = {'f32': 'u32', 'f16': 'u16'}  # vgather2/vscatter element index registers
COMPACT = {'f32', 'f16', 'u32', 'u8'}
RANK = {'i32', 'u32', 'u16', 'u8'}
PACK = {('i32', 'u16'), ('u32', 'u16'), ('u16', 'u8')}


def register(value, names):
    return (isinstance(value, Value) and isinstance(value.type, RegType) and value.type.n == 1
            and value.type.dtype.name in names)


def lanes(pred, reg):
    """A one-register predicate with the register's logical lane width."""
    return isinstance(pred, Value) and pred.type == MaskType(max(reg.type.dtype.bits, 8))


def whole_tile(o, node, tile, reg):
    """(tile, element offset): a whole UB tile of the register dtype or a literal `tile + k` of one, whose base
    A5 moves k elements (p7-vf-probes)."""
    memory, offset = (tile.memory, tile.offset) if isinstance(tile, Address) else (tile, 0)
    o.need(isinstance(memory, Memory) and memory.value.type.space == 'ub' and memory.pitch is None
           and memory.value.type.dtype == reg.type.dtype, node,
           'Indexed UB access needs a whole Vec tile of the register dtype')
    o.need(offset * reg.type.dtype.bits % 256 == 0, node, 'Indexed UB access at a tile offset needs 32-byte alignment')
    return memory.value, offset


def convert(ctx, node, args):
    o, name = ctx.o, node['fields']['name']
    if name == 'vf.gather':
        mode = ctx.attrs(node, {'data_copy_mode'}).get('data_copy_mode')
        if len(args) == 3 and not isinstance(args[1], Memory):
            dst, src, index = args
            o.need(mode is None, node, 'Register gather takes no data_copy_mode')
            o.need(register(dst, SELECT) and isinstance(src, Value) and src.type == dst.type
                   and register(index, SELECT[dst.type.dtype.name]), node,
                   'Register gather admits FP32/INT32 data with INT32/UINT32 indices, FP16 with UINT16 and UINT8 with UINT8')
            o.need(dst not in (src, index), node, 'Register gather destination must not alias its source or index')
            ctx.emit('vf.gather', node, (dst, src, index))
            return None
        o.need(len(args) == 4, node, 'UB gather needs a destination, tile, index and predicate')
        dst, tile, index, pred = args
        o.need(mode is None or type(mode) is int and mode in (0, 1), node, 'UB gather admits NORM and DATA_BLOCK_LOAD only')
        o.need(register(dst, ELEMENT), node, 'UB gather admits FP32 and FP16 destinations')
        tile, offset = whole_tile(o, node, tile, dst)
        if mode == 1:
            o.need(register(index, {'u32'}), node, 'DATA_BLOCK_LOAD gather needs UINT32 byte offsets')
        else:
            o.need(register(index, {ELEMENT[dst.type.dtype.name]}), node,
                   'NORM gather needs UINT32 indices for FP32 and UINT16 for FP16; vgather2_bc is not admitted')
        o.need(lanes(pred, dst), node, 'UB gather predicate width must match the destination lanes')
        o.need(dst != index, node, 'UB gather destination must not alias its index')
        ctx.emit('vf.gatherb' if mode == 1 else 'vf.gather_copy', node, (dst, tile, index),
                 attrs={'offset': offset, 'mask': pred})
        ctx.reads.append(tile)
        return None
    if name == 'vf.scatter':
        ctx.attrs(node)
        o.need(len(args) == 4, node, 'Scatter needs a tile, source, index and predicate')
        tile, src, index, pred = args
        o.need(not register(src, {'i32'}), node, 'INT32 scatter prints a signed index cast that CANN rejects (A5-UP-037)')
        o.need(register(src, ELEMENT), node, 'Scatter admits FP32 and FP16 payloads')
        tile, offset = whole_tile(o, node, tile, src)
        o.need(register(index, {ELEMENT[src.type.dtype.name]}), node, 'Scatter needs UINT32 indices for FP32 and UINT16 for FP16')
        o.need(lanes(pred, src), node, 'Scatter predicate width must match the source lanes')
        ctx.emit('vf.scatter_copy', node, (tile, src, index), attrs={'offset': offset, 'mask': pred})
        ctx.writes.append(tile)
        return None
    if name == 'vf.squeeze':
        mode = ctx.attrs(node, {'gather_mode'}).get('gather_mode')
        o.need(type(mode) is int and mode == 1, node, 'Squeeze needs gather_mode=NO_STORE_REG; STORED also writes the AR SPR')
        o.need(len(args) == 3 and register(args[0], COMPACT) and isinstance(args[1], Value) and args[1].type == args[0].type,
               node, 'Squeeze admits matching FP32, FP16, UINT32 and UINT8 registers')
        dst, src, pred = args
        o.need(lanes(pred, dst), node, 'Squeeze predicate width must match the register lanes')
        o.need(dst != src, node, 'Squeeze destination must not alias its source')
        ctx.emit('vf.squeeze', node, (dst, src), attrs={'mask': pred, 'store': False})
        return None
    if name == 'vf.unsqueeze':
        ctx.attrs(node)
        o.need(len(args) == 2 and register(args[0], RANK) and lanes(args[1], args[0]), node,
               'Unsqueeze admits INT32/UINT32/UINT16/UINT8 destinations with a predicate of their lane width')
        ctx.emit('vf.unsqueeze', node, (args[0],), attrs={'mask': args[1]})
        return None
    if name == 'vf.pack':
        part = ctx.attrs(node, {'part'}).get('part', 0)
        o.need(type(part) is int and part in (0, 1), node, 'Pack part must be LOWER or UPPER')
        o.need(len(args) == 2 and all(register(v, {'i32', 'u32', 'u16', 'u8'}) for v in args)
               and (args[1].type.dtype.name, args[0].type.dtype.name) in PACK, node,
               'Pack admits INT32/UINT32 to UINT16 and UINT16 to UINT8 registers')
        ctx.emit('vf.pack', node, tuple(args), attrs={'part': Ident('highest' if part else 'lowest')})
        return None
    attrs = ctx.attrs(node, {'bin_type', 'hist_type'})
    group, mode = attrs.get('bin_type', 0), attrs.get('hist_type', 0)
    o.need(all(type(v) is int and v in (0, 1) for v in (group, mode)), node,
           'Histograms admit BIN0/BIN1 and ACCUMULATE/FREQUENCY')
    o.need(len(args) == 3 and register(args[0], {'u16'}) and register(args[1], {'u8'})
           and isinstance(args[2], Value) and args[2].type == MaskType(8), node,
           'Histograms need a UINT16 destination, UINT8 source and b8 predicate')
    # Counts accumulate onto the destination; native registers start uninitialized.
    o.need(args[0] in ctx.written, node, 'Histogram counts must be written earlier in this VF body')
    ctx.emit('vf.histograms', node, tuple(args[:2]),
             attrs={'mask': args[2], 'bin_group': group, 'mode': Ident('frequency' if mode else 'accumulate')})
    return None
