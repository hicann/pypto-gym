# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Descending sort records and merges of descending record runs on whole Vec tiles.

UINT32 Vec tiles are admitted only as sort32 identifiers. Native A5 probes fixed each admitted form (RFC-0015).
"""
from ...ir.types import MemType
from .lower import Memory

TARGETS = {'block.sort32': ('vec.sort32',), 'block.mrgsort': ('vec.mergesort4',),
           'block.mrgsort2': ('vec.mergesort_2seq',)}


def index_storage(dt):
    """UINT32 GM parameters and Vec tiles carry sort32 identifiers or SIMT launch storage (see `guard`)."""
    return dt.name == 'u32'


def guard(ctx, node, args):
    """Refuse UINT32 GM/Vec uses other than a load from GM, the identifier operand of sort32, its valid-shape updates and
    SIMT launch storage.

    SIMT function bodies access UINT32 storage under their own element and atomic rules (simt, simt_exact)."""
    if ctx.simt is not None:
        return
    name = node['fields']['name']
    for position, arg in enumerate(args):
        if isinstance(arg, Memory) and arg.value.type.dtype.name == 'u32':
            ctx.o.need(name in {'block.load', 'block.set_validshape'} or (name == 'block.sort32' and position == 2)
                       or (name == 'simt.launch' and position >= 3), node,
                       'UINT32 storage is admitted only as sort32 identifiers loaded from GM or as SIMT launch storage')


def vec_tiles(ctx, node, args, what):
    ctx.o.need(ctx.o.side == 'vec', node, f'{what} is a vector-side instruction')
    ctx.o.need(all(isinstance(t, Memory) and t.value.type.space == 'ub' for t in args), node, f'{what} needs Vec tiles')


def whole(ctx, node, args, what, count, valid=False):
    """Distinct tiles of their own storage; aliases and subviews need a same-buffer rule. Partial valid extents are the
    caller's rule when `valid` is set."""
    o = ctx.o
    o.need(all(t.root is None for t in args), node, f'{what} through a tile alias needs a same-buffer rule')
    o.need(all(t.pitch is None and (valid or t.valid == t.shape) for t in args) and len({t.value.name for t in args}) == len(args),
           node, f'{what} needs {count} distinct whole tiles')
    o.need(all(t.valid is not None and all(type(v) is int for v in t.valid) for t in args), node,
           f'{what} needs static valid extents')


def row(ctx, node, tile, index):
    columns = tile.shape[1]
    return ctx.emit('mem.slice', node, (tile.value,), typ=MemType('ub', tile.value.type.dtype, (1, columns)),
                    attrs={'offsets': [index, 0], 'extents': [1, columns]})


def sort32(ctx, node, args):
    o = ctx.o
    ctx.attrs(node)
    o.need(o.side == 'vec', node, 'sort32 is a vector-side instruction')
    o.need(len(args) == 3, node, 'sort32 tail groups with a tmp tile are not admitted')
    vec_tiles(ctx, node, args, 'sort32')
    dst, src, idx = args
    o.need(src.value.type.dtype.name == dst.value.type.dtype.name == 'f32', node, 'Only FP32 sort records are admitted')
    o.need(idx.value.type.dtype.name == 'u32', node, 'sort32 identifiers must be UINT32')
    whole(ctx, node, args, 'sort32', 'three', valid=True)
    rows, n = src.shape
    o.need(n % 32 == 0 and 0 < n // 32 <= 255, node, 'sort32 needs 1..255 whole 32-value groups')
    o.need(idx.shape[0] in {1, rows}, node, 'sort32 needs one identifier row, or one per source row')
    o.need(idx.shape[1] == n and tuple(dst.shape) == (rows, 2 * n), node,
           'sort32 needs one identifier per score and two FP32 words per record in every source row')
    # A5 sorts the valid rows, or the whole 32-value groups of the valid width (sort32_valid_rows/columns_probe).
    height, width = src.valid
    o.need(width % 32 == 0 and dst.valid == (height, 2 * width) and idx.valid == (idx.shape[0], width), node,
           'sort32 needs agreeing valid extents: whole groups, their identifiers and their records')
    o.need(height == rows or width == n, node, 'sort32 over both fewer valid rows and fewer valid groups is unmeasured')
    o.need(idx.shape[0] == 1 or src.valid == src.shape, node, 'sort32 with one identifier row per source row is measured '
           'for whole valid extents only')
    if rows == 1:
        ctx.emit('vec.sort32', node, (dst.value, src.value, idx.value), attrs={'repeat': width // 32})
        return None
    for index in range(height):  # Row r restarts at identifier row r, or at the one shared row (sort32_id_rows_probe).
        ids = idx.value if idx.shape[0] == 1 else row(ctx, node, idx, index)
        ctx.emit('vec.sort32', node, (row(ctx, node, dst, index), row(ctx, node, src, index), ids),
                 attrs={'repeat': width // 32})
    return None


def mrgsort(ctx, node, args):
    o = ctx.o
    length = ctx.attrs(node, {'block_len'}).get('block_len')
    o.need(len(args) == 2, node, 'mrgsort needs a destination and a source tile')
    vec_tiles(ctx, node, args, 'mrgsort')
    dst, src = args
    o.need(src.value.type.dtype.name == dst.value.type.dtype.name == 'f32', node, 'Only FP32 merge records are admitted')
    whole(ctx, node, args, 'mrgsort', 'two', valid=True)
    o.need(src.shape[0] == 1 and tuple(dst.shape) == tuple(src.shape), node,
           'mrgsort needs one-row source and destination tiles of equal width')
    o.need(type(length) is int and length > 0 and length % 2 == 0 and length // 2 <= 0xFFFF, node,
           'mrgsort block_len must count whole FP32 records in storage elements')
    width = src.valid[1]  # A5 merges the groups of the valid width (mrgsort_valid_probe).
    o.need(dst.valid == src.valid, node, 'mrgsort needs equal source and destination valid extents')
    o.need(width % (4 * length) == 0 and width // (4 * length) <= 255, node,
           'mrgsort needs 1..255 whole groups of four block_len runs')
    ctx.emit('vec.mergesort4', node, (dst.value, src.value),
             attrs={'length_per_seq': length // 2, 'repeat': width // (4 * length)})
    return None


def mrgsort2(ctx, node, args):
    o = ctx.o
    exhausted = ctx.attrs(node, {'exhausted'}).get('exhausted')
    o.need(len(args) == 4, node, 'Only two-source mrgsort2 merges are measured')
    o.need(exhausted is False, node, 'mrgsort2 exhausted=True pauses at an exhausted list; only False is measured')
    vec_tiles(ctx, node, args, 'mrgsort2')
    dst, first, tmp, second = args  # Pro's graph binding (A5-UP-040), not its documented order.
    o.need(all(t.value.type.dtype.name == 'f32' for t in args), node, 'Only FP32 merge records are admitted')
    whole(ctx, node, args, 'mrgsort2', 'four')
    o.need(all(t.shape[0] == 1 for t in args), node, 'mrgsort2 needs one-row tiles')
    sizes = first.shape[1], second.shape[1]  # Each source merges its own records (mrgsort2_unequal_probe on A5).
    o.need(all(words % 2 == 0 and words // 2 <= 0xFFFF for words in sizes), node,
           'mrgsort2 sources of partial-record length are unmeasured')
    o.need(dst.shape[1] == tmp.shape[1] == sum(sizes), node,
           'mrgsort2 needs destination and scratch tiles as wide as both sources')
    # tmp is scratch whose contents are unspecified after the call; the target neither reads nor writes it.
    ctx.emit('vec.mergesort_2seq', node, (dst.value, first.value, second.value),
             attrs={'size1': sizes[0] // 2, 'size2': sizes[1] // 2})
    return None


def convert(ctx, node, args):
    return {'block.sort32': sort32, 'block.mrgsort': mrgsort, 'block.mrgsort2': mrgsort2}[node['fields']['name']](ctx, node, args)
