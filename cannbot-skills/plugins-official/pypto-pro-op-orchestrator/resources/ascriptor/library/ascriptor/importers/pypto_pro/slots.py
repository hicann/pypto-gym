# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Contiguous tile groups as one slot buffer: `buf<P, n>` selected by `mem.get_buf` (RFC-0015 slot buffers).

A group becomes a buffer only where one allocation per slot cannot hold it: its cursor advances, or it has more
than 16 slots and a runtime selection. Every other group keeps one allocation per slot (P4 storage).
"""
from dataclasses import dataclass

from ...ir import Value
from ...ir.types import BufType, CellType, ScalarType, dtype
from .lower import Memory, Struct

TARGETS = {'GetItemExpr': ('mem.get_buf',)}


@dataclass(eq=False)
class Group:
    depth: int
    buffer: tuple | None = None  # (buf value, slot type, base address, pitch, shape, valid, transposed)


@dataclass(frozen=True)
class Slot:
    """A declared slot of a buffer group; only a selection of the whole group yields a tile."""
    group: Group
    index: int


def plan(o, graph):
    """Map make_tile and cursor struct.create node ids to groups that must become slot buffers, and give the make_tile
    ids of every group's slots: a slot is never folded into another declaration (alias_identical_probe measured
    standalone tiles only)."""
    nodes = graph['nodes']
    assigns = {}
    for n in nodes:
        if n['kind'] == 'AssignStmt':
            assigns.setdefault(o.node(n['fields']['var'])['fields']['name'], []).append(n)

    def value(name):
        rows = assigns.get(name, [])
        return o.node(rows[0]['fields']['value']) if len(rows) == 1 else None

    groups = []
    for row in graph['metadata']['groups']:
        handle = o.node(row['value'])
        elements = [o.node(e) for e in handle['fields']['elements']] if handle['kind'] == 'MakeTuple' else []
        if not elements or any(e['kind'] != 'Var' for e in elements):
            continue
        tiles = value(elements[0]['fields']['name'])
        slots = [value(o.node(e)['fields']['name']) if o.node(e)['kind'] == 'Var' else None
                 for e in (tiles['fields']['elements'] if tiles and tiles['kind'] == 'MakeTuple' else [])]
        creation = value(elements[-1]['fields']['name'])
        if (len(slots) != row['depth'] or any(s is None or s['kind'] != 'Call' or s['fields']['name'] != 'block.make_tile'
                                              for s in slots)
                or creation is None or creation['kind'] != 'Call' or creation['fields']['name'] != 'struct.create'):
            continue
        groups.append((elements[0]['fields']['name'], elements[-1]['fields']['name'], row['depth'], slots, creation))
    members = {s['id'] for _, _, _, slots, _ in groups for s in slots}
    groups = [group for group in groups if group[2] >= 2]
    names = {cursor for _, cursor, _, _, _ in groups}
    advanced = {o.node(n['fields']['args'][0])['fields']['name'] for n in nodes if n['kind'] == 'Call'
                and n['fields']['name'] == 'struct.set' and n['fields']['args']
                and o.node(n['fields']['args'][0])['kind'] == 'Var'} & names

    def static(ref, seen=()):  # Mirrors the values the importer folds to Python integers.
        node = o.node(ref)
        kind, f = node['kind'], node['fields']
        if kind == 'ConstInt':
            return True
        if kind == 'Var':
            source = value(f['name'])
            return f['name'] not in seen and source is not None and static({'$ref': source['id']}, (*seen, f['name']))
        if kind in ('FloorMod', 'FloorDiv'):
            return static(f['left'], seen) and o.node(f['right'])['kind'] == 'ConstInt'
        if kind == 'GetItemExpr':
            base = o.node(f['value'])
            return (base['kind'] == 'Var' and base['fields']['name'] in names - advanced
                    and o.node(f['slice'])['kind'] == 'ConstInt')
        return False

    tiles, cursors = {}, {}
    for tuple_name, cursor_name, depth, slots, creation in groups:
        runtime = any(n['kind'] == 'GetItemExpr' and o.node(n['fields']['value'])['kind'] == 'Var'
                      and o.node(n['fields']['value'])['fields']['name'] == tuple_name and not static(n['fields']['slice'])
                      for n in nodes)
        if cursor_name in advanced or depth > 16 and runtime:
            group = Group(depth)
            tiles.update({s['id']: Slot(group, k) for k, s in enumerate(slots)})
            if cursor_name in advanced:
                cursors[creation['id']] = group
    return tiles, cursors, members


def declare(ctx, node, slot, typ, addr, reserved, shape, valid, transposed, position):
    """Slot 0 allocates the whole buffer; later slots must continue it at the aligned tile pitch."""
    o, group = ctx.o, slot.group
    o.need(not o.mixed, node, 'Slot buffers are admitted only in single-side kernels')
    o.need(position in {'ub', 'l1'}, node, 'Slot buffers are admitted only for Vec and Mat tiles')
    if slot.index == 0:
        value = ctx.emit('mem.alloc', node, typ=BufType(typ, group.depth), attrs={'addr': addr})
        group.buffer = (value, typ, addr, reserved, shape, valid, transposed)
    else:
        buffer = group.buffer
        o.need(buffer is not None and addr == buffer[2] + slot.index * buffer[3]
               and (typ, reserved, shape, valid, transposed) == (buffer[1], *buffer[3:]), node,
               'Slot buffers need contiguous slots of one descriptor at the aligned tile pitch')
    o.roots[(position, addr)] = {'buffer': group, 'size': reserved}
    return slot


def choose(ctx, node, items, index):
    """Select one slot of a buffer; a runtime index must be proven inside the group, never wrapped."""
    from .selection import interval

    o = ctx.o
    group = items[0].group
    o.need(all(isinstance(item, Slot) and item.group is group for item in items)
           and [item.index for item in items] == list(range(group.depth)) and group.buffer is not None, node,
           'Slot-buffer tiles are selected only from their whole group')
    if type(index) is int:
        o.need(0 <= index < group.depth, node, 'Tuple index is out of range')
    else:
        index = ctx.snapshot(index, node)
        bounds = interval(ctx, index)
        o.need(isinstance(index, Value) and isinstance(index.type, ScalarType) and index.type.dtype.kind in ('int', 'uint')
               and bounds is not None and 0 <= bounds[0] <= bounds[1] < group.depth, node,
               'Dynamic tuple/slot index requires an in-range integer proof')
    value, typ, _, _, shape, valid, transposed = group.buffer
    result = ctx.emit('mem.get_buf', node, (value, index), typ=typ)
    return Memory(result, shape, valid, transposed=transposed, slot=True)


def cursor(ctx, node, start):
    """A cursor that advances is a scalar cell starting at depth - 1, as Pro's struct field is."""
    from .loop_extents import cell_range

    cell = ctx.emit('scalar.cell', node, typ=CellType(dtype('i64')))
    ctx.emit('scalar.set', node, (cell, start))
    cell_range(ctx, cell, (start, start))
    return Struct([cell])


def advance(ctx, node, args):
    from .loop_extents import cell_range
    from .selection import interval

    o = ctx.o
    attrs = ctx.attrs(node, {'field'})
    target = args[0] if len(args) == 2 and isinstance(args[0], Struct) and len(args[0].fields) == 1 else None
    cell = target.fields[0] if target is not None else None
    o.need(attrs.get('field') == 'cursor' and isinstance(cell, Value) and isinstance(cell.type, CellType), node,
           'Only slot-buffer cursor updates are admitted')
    value = args[1]
    o.need(type(value) is int or isinstance(value, Value) and value.type == ScalarType(dtype('i64')), node,
           'Tile-group cursors hold INDEX values')
    ctx.emit('scalar.set', node, (cell, value))
    cell_range(ctx, cell, interval(ctx, value))


def cursor_steps(ctx, nodes):
    """(cell, update statements, constant step or None) for each cursor that these statements update."""
    o, found = ctx.o, {}
    for n in nodes:
        call = o.node(n['fields']['expr']) if n['kind'] == 'EvalStmt' else None
        if call is None or call['kind'] != 'Call' or call['fields']['name'] != 'struct.set' or len(call['fields']['args']) != 2:
            continue
        base = o.node(call['fields']['args'][0])
        target = ctx.env.get(base['fields']['name']) if base['kind'] == 'Var' else None
        if isinstance(target, Struct) and len(target.fields) == 1 and isinstance(target.fields[0], Value):
            found.setdefault(target.fields[0], (base['fields']['name'], call, []))[2].append(n)
    result = []
    for cell, (name, call, steps) in found.items():
        update = o.node(call['fields']['args'][1])
        step = None
        if update['kind'] in ('Add', 'Sub'):
            left, right = o.node(update['fields']['left']), o.node(update['fields']['right'])
            read = left['kind'] == 'GetItemExpr' and o.node(left['fields']['value'])['kind'] == 'Var'
            if (read and o.node(left['fields']['value'])['fields']['name'] == name and right['kind'] == 'ConstInt'
                    and o.node(left['fields']['slice'])['kind'] == 'ConstInt' and o.node(left['fields']['slice'])['fields']['value'] == 0):
                step = right['fields']['value'] * (-1 if update['kind'] == 'Sub' else 1)
        result.append((cell, steps, step))
    return result


def join_cursors(ctx, children, bodies):
    """A cursor updated in either branch holds the union of both branch ranges after the join."""
    from .loop_extents import cell_range

    cells = {cell for body in bodies if body for cell, _, _ in cursor_steps(ctx, list(ctx.walk(body)))}
    for cell in cells:
        ranges = [child.cell_ranges.get(cell.name) for child in children]
        cell_range(ctx, cell, None if None in ranges else (min(r[0] for r in ranges), max(r[1] for r in ranges)))
