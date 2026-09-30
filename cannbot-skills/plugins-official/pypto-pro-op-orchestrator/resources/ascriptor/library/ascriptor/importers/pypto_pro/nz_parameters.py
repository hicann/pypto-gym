# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""NZ-packed GM parameters as their physical storage (RFC-0015 NZ-packed GM parameters).

`pl.Tensor[[R, C], T, pl.NZ]` holds ceil(C / C0) column blocks of align16(R) rows of C0 = 32 / sizeof(T) elements:
logical element (r, c) is storage element (c // C0) * align16(R) * C0 + r * C0 + c % C0 (native A5: nz_gm_probe,
nz_padded_probe, nz_quant_store_probe). The parameter becomes the ND GM tensor of that storage,
`gm<T, [ceil(C / C0) * align16(R), C0]>`, with [R, C] in the module metadata. Loads and stores move whole column
blocks of a window inside that storage, padding included (nz_padding_probe); pointers address the storage.
"""
import struct

from ...ir.types import MemType

TARGETS = {'block.load': ('dma.gm_to_l1', 'dma.gm_to_ub.pad'), 'block.store': ('dma.ub_to_gm.pad', 'dma.l0c_to_gm.nz2nz')}
BLOCKS = {'f16': 16, 'i8': 32}  # C0: the elements of one 32-byte block


def declare(o, var, dt, shape, view):
    """The physical type of an NZ-packed parameter; the caller has checked its pointer provenance."""
    o.need(dt.name != 'f32', var, 'FP32 NZ-packed GM tensors hold channel-split 8-element blocks on A5 (nz_store_probe); '
           'no IR store sets channel split')
    o.need(dt.name in BLOCKS, var, 'NZ-packed GM parameters are measured only for FP16 and INT8')
    o.need(all(type(n) is int for n in shape), var, 'NZ-packed GM parameters need static shapes')
    o.need(not view['stride'] and not view['valid_shape'], var, 'NZ-packed GM parameters admit no stride or valid shape')
    o.nz_parameters[var['fields']['name']] = list(shape)
    storage = -(-shape[1] // BLOCKS[dt.name]) * ((shape[0] + 15) // 16 * 16), BLOCKS[dt.name]
    return MemType('gm', dt, storage), storage


def guard(ctx, node, name, args):
    """An NZ-packed parameter anywhere in a call's operands, run-time selections included, is a load source, a store
    destination or a pointer source, or the call refuses."""
    from .lower import Memory
    from .selection import Choice

    def operands(values):
        for value in values:
            if isinstance(value, (tuple, list, Choice)):
                yield from operands(value.items if isinstance(value, Choice) else value)
            else:
                yield value

    def packed(value):
        return isinstance(value, Memory) and value.nz is not None
    if not any(packed(value) for value in operands(args)):
        return
    is_load, is_store = name == 'block.load' and len(args) == 3, name == 'block.store' and len(args) in {3, 4}
    ctx.o.need(not ctx.vf and ctx.simt is None and (is_load and packed(args[1]) or is_store and packed(args[0])
                                                    or name in {'ptr.make_ptr', 'ptr.make_tensor'} and packed(args[0])),
               node, 'NZ-packed GM parameters are admitted only as load sources, store destinations and pointer sources')
    if is_load or is_store:
        tile = args[0] if is_load else args[1]
        ctx.o.need(not isinstance(tile, (tuple, list, Choice)) and not getattr(tile, 'slot', False), node,
                   'NZ-packed GM transfers with slot-buffer or run-time selected tiles are unmeasured')


def scale(ctx, node, ref):
    """Pro passes a store's scale as the FP32 bits of a UINT64 literal; A5 multiplies by it (nz_quant_store_probe)."""
    literal = ctx.o.node(ref)
    bits = literal['fields'].get('value') if literal['kind'] == 'ConstInt' else None
    ctx.o.need(type(bits) is int and 0 <= bits < 1 << 32 and not bits & 0x1FFF, node,
               'Store scales must be static FP32 literals that the 19-bit fixpipe scale holds exactly')
    value = struct.unpack('<f', struct.pack('<I', bits))[0]
    ctx.o.need(0 < value < float('inf'), node, 'Store scales must be positive and finite')
    return value


def window(ctx, node, gm, offsets, extent):
    """The storage rows of a window of whole column blocks inside the padded storage, and its block count."""
    o = ctx.o
    rows, cols = gm.nz
    c0, padded = gm.value.type.dims[1], (rows + 15) // 16 * 16
    height, width = extent
    o.need(isinstance(offsets, tuple) and len(offsets) == 2 and all(type(v) is int for v in offsets), node,
           'NZ-packed GM transfers need static offsets')
    row, col = offsets
    o.need(width % c0 == 0 and col % c0 == 0 and 0 <= row and row + height <= padded and 0 <= col
           and col + width <= -(-cols // c0) * c0, node,
           'NZ-packed GM windows must be whole column blocks inside the padded storage')
    bursts = width // c0
    origin, rows_used = col // c0 * padded + row, (bursts - 1) * padded + height
    if (origin, rows_used) == (0, gm.shape[0]):
        return gm.value, bursts, padded
    value = ctx.emit('mem.slice', node, (gm.value,), typ=MemType('gm', gm.value.type.dtype, (rows_used, c0)),
                     attrs={'offsets': [origin, 0], 'extents': [rows_used, c0]})
    return value, bursts, padded


def fractals(ctx, node, alias, gm):
    """A whole NZ Vec alias of 16-aligned rows holds its column blocks as contiguous bytes from its root's start
    (nz_vec_probe on A5): its shape and those bytes, one UB row per column block."""
    memory = alias.memory
    ctx.o.need(memory.value.type.dtype == gm.value.type.dtype and memory.valid == memory.shape and memory.shape[0] % 16 == 0,
               node, 'NZ-packed GM transfers with NZ Vec aliases need whole 16-aligned aliases of their dtype')
    (height, width), c0, dt = memory.shape, gm.value.type.dims[1], memory.value.type.dtype
    root, count = memory.root.value, height * width
    total = root.type.dims[0] * root.type.dims[1]
    if count < total:
        root = ctx.emit('mem.reshape', node, (root,), typ=MemType('ub', dt, (1, total)), attrs={'shape': [1, total]})
        root = ctx.emit('mem.slice', node, (root,), typ=MemType('ub', dt, (1, count)), attrs={'offsets': [0, 0], 'extents': [1, count]})
    blocks = (-(-width // c0), height * c0)
    return memory.shape, ctx.emit('mem.reshape', node, (root,), typ=MemType('ub', dt, blocks), attrs={'shape': list(blocks)})


def load(ctx, node, tile, gm, offsets):
    """Whole column blocks of a window, copied into an NZ Mat of whole blocks or into an NZ Vec alias."""
    from .lower import Memory
    from .views import Fractals

    o = ctx.o
    ctx.attrs(node)
    if isinstance(tile, Fractals):
        shape, flat = fractals(ctx, node, tile, gm)
        source, bursts, padded = window(ctx, node, gm, offsets, shape)
        height = shape[0]
        ctx.emit('dma.gm_to_ub.pad', node, (flat, source), attrs={
            'n_burst': bursts, 'burst_len_byte': height * 32, 'src_stride_byte': (padded - height) * 32, 'dst_stride': 0})
        return
    o.need(isinstance(tile, Memory) and tile.value.type.space == 'l1' and tile.value.type.layout == 'nz'
           and tile.value.type.dtype == gm.value.type.dtype and not tile.transposed and tile.root is None, node,
           'NZ-packed GM parameters load only into NZ Mat tiles or NZ Vec aliases of their dtype')
    o.need(tile.valid is not None and all(type(v) is int for v in tile.valid), node,
           'Descriptor mutation requires a valid-shape reset before transfer')
    # Valid rows land at the Mat's row pitch in every column block, and valid columns take whole blocks
    # (nz_partial_load_probe on A5, one narrowed axis each).
    height = tile.valid[0]
    o.need(height == tile.shape[0] or tile.valid[1] == tile.shape[1], node,
           'NZ-packed GM loads into Mats narrowed in both rows and columns are unmeasured')
    source, bursts, padded = window(ctx, node, gm, offsets, tile.valid)
    ctx.emit('dma.gm_to_l1', node, (tile.value, source), attrs={
        'n_burst': bursts, 'burst_len': height, 'src_stride': padded - height, 'dst_stride': tile.shape[0] - height})


def store(ctx, node, gm, tile, offsets, factor):
    """An NZ Vec alias back into whole column blocks, or a scaled INT8 Acc store of whole 32-element blocks: one block,
    or, as measured, several from the tensor's padded rows at column 0."""
    from .lower import Memory
    from .views import Fractals

    o = ctx.o
    ctx.attrs(node)
    if isinstance(tile, Fractals):
        o.need(factor is None, node, 'Scaled stores from Vec tiles are unmeasured')
        shape, flat = fractals(ctx, node, tile, gm)
        destination, bursts, padded = window(ctx, node, gm, offsets, shape)
        height = shape[0]
        ctx.emit('dma.ub_to_gm.pad', node, (destination, flat), attrs={
            'n_burst': bursts, 'burst_len_byte': height * 32, 'src_stride': 0, 'dst_stride_byte': (padded - height) * 32})
        return
    o.need(isinstance(tile, Memory) and tile.value.type.space == 'l0c' and gm.value.type.dtype.name == 'i8'
           and factor is not None, node,
           'NZ-packed GM stores are admitted from NZ Vec aliases and as scaled INT8 accumulator stores')
    o.need(tile.valid == tile.shape, node, 'Partial fixpipe windows are not admitted')
    # A5 multiplies by the scale, rounds half to even and saturates (nz_quant_store_probe), and puts block k of a store
    # k * align16(R) * 32 elements after its first (nz_int8_blocks_probe, I039). PTO strides by the store's own padded
    # rows, so Pro compiles several blocks only from the tensor's padded rows.
    o.need(tile.shape[1] <= 32 or tile.shape[0] == (gm.nz[0] + 15) // 16 * 16, node,
           'Scaled INT8 NZ stores of several column blocks from fewer rows than the padded tensor do not compile natively: '
           "Pro's direct TSTORE refuses partial-M stores spanning N fractals (ValidateNZTransfer, nz_int8_strip_probe)")
    destination, _, padded = window(ctx, node, gm, offsets, tile.shape)
    o.need(tile.shape[1] <= 32 or offsets[1] == 0, node,
           'Scaled INT8 NZ stores of several column blocks from a later column are unmeasured: A5 ran one block at '
           '[0, 32] (nz_quant_store_probe) and several blocks at [0, 0] (nz_int8_blocks_probe)')
    ctx.emit('dma.l0c_to_gm.nz2nz', node, (destination, tile.value), attrs={
        'M': tile.shape[0], 'N': tile.shape[1], 'M_pad': padded, 'M_src': tile.shape[0], 'relu': False, 'scale': factor})
