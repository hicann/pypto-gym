# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Microscaling products: FP8 operands with E8M0 scale planes on the cube side (RFC-0015).

GM planes keep Pro's [rows, groups / 2, 2] shape and are read row-major. ScaleLeft/ScaleRight tiles are
declarations at their data tile's address >> 4. An FP8 Mat-to-Left/Right move and the move of its plane are one
dma.l1_to_l0.mx, which issues both loads: directly in sequence, or in Pro's order measured on A5
(Left data, Right data, Left plane, Right plane), whose middle two instructions touch disjoint storage. The pair
is issued at the plane move, once both sources are evaluated, so runtime-selected Mat slots dispatch together.
A scale Mat may load fewer rows (ZZ) or N (NN) and return to its whole shape, as measured (mx_tail_code,
mx_cut_boxes, mx_nn_tail).
"""
from dataclasses import dataclass

from ...ir import Ident
from ...ir.types import MemType
from .lower import Memory

FP8 = frozenset({'e4m3', 'e5m2'})
PLANES = {'ScaleLeft': ('row_major', 'row_major', 32), 'ScaleRight': ('col_major', 'col_major', 32)}
PLANE_BYTES = 4096  # The ScaleLeft/ScaleRight address domains.
TARGETS = {'block.load (E8M0 plane)': ('dma.gm_to_l1.mx_scale_nd2nz',), 'block.move (FP8 tile + plane)': ('dma.l1_to_l0.mx',),
           'block.matmul_mx': ('cube.mmad.mx',), 'block.matmul_mx_acc': ('cube.mmad.mx',)}
CALLS = {rule.split()[0] for rule in TARGETS}
BINDINGS = {'GetItemExpr', 'MakeTuple', 'Var', 'ConstInt', 'Add', 'Sub', 'Mul', 'FloorDiv', 'FloorMod'}  # Pro hoists these


@dataclass(frozen=True)
class Plane:
    """A declaration-only ScaleLeft/ScaleRight tile, validated by space, address and Pro shape."""
    space: str
    addr: int
    shape: tuple


def storage(dt):
    return dt.name in FP8 or dt.name == 'e8m0'


def operand(value):
    return isinstance(value, Plane) or isinstance(value, Memory) and storage(value.value.type.dtype)


def slots(value):
    """The tiles a move operand may be: itself, or every slot of a runtime selection."""
    from .selection import Choice

    return value.items if isinstance(value, Choice) else (value,)


def guard(ctx, node, args):
    if any(operand(arg) for arg in args):
        name = node['fields']['name']
        ctx.o.need(ctx.o.side == 'cube', node, 'Microscaling loads, moves and products run on the cube side')
        if name == 'block.set_validshape':
            return valid_shape(ctx, node, args)
        ctx.o.need(name in CALLS, node, 'FP8 and E8M0 storage is admitted only in microscaling loads, moves and products')
    return None


def valid_shape(ctx, node, args):
    """Pro's measured reduced loads: a whole scale Mat set to fewer rows (ZZ) or N (NN), exactly one load of it, and
    its whole shape again in the same block. The load's DN builtin writes code 0 in the rest of its last 16-row box
    and leaves later boxes stale (mx_tail_code, mx_cut_boxes); statements that do not name the Mat may intervene
    (mx_cut_statements)."""
    o, scale_tile = ctx.o, args[0]
    o.need(isinstance(scale_tile, Memory) and scale_tile.value.type.dtype.name == 'e8m0', node,
           'A5 measured valid shapes on MX tiles only for E8M0 scale Mats; FP8 and plane tiles stay whole')
    o.need(len(args) == 3 and all(type(v) is int for v in args[1:]), node, 'MX valid shapes must be static')
    valid = tuple(args[1:])
    if scale_tile.valid != scale_tile.shape:
        o.need(valid == scale_tile.shape, node, 'A reduced MX scale Mat must be set back to its whole shape')
        return
    axis = 1 if scale_tile.transposed else 0  # ZZ [rows, groups], NN [groups, N]
    o.need(valid[1 - axis] == scale_tile.shape[1 - axis] and valid[axis] < scale_tile.shape[axis], node,
           'A5 measured reduced MX scale Mats only with every group and fewer rows')
    o.need(not scale_tile.transposed or scale_tile.shape[1] - 16 < valid[1], node,
           'A5 measured reduced NN scale Mats only with fewer N ending in the last 16-N box (mx_nn_tail)')
    loads = 0
    for statement in later(o, node):
        call = o.node(statement['fields']['expr']) if statement['kind'] == 'EvalStmt' else {}
        names = named(ctx, call) if call.get('kind') == 'Call' else []
        if names and isinstance(names[0], Memory) and names[0].value is scale_tile.value:
            if call['fields']['name'] == 'block.set_validshape':
                o.need(loads == 1, call, 'A reduced MX scale Mat needs exactly one load before its whole shape returns')
                return
            o.need(call['fields']['name'] == 'block.load' and loads == 0, call,
                   'A reduced MX scale Mat admits one load and then its whole shape; other uses stay unmeasured')
            loads += 1
            continue
        o.need(not any(isinstance(ctx.env.get(n['fields']['name']), Memory) and ctx.env[n['fields']['name']].value is scale_tile.value
                       for n in ctx.expr_nodes({'$ref': statement['id']}) if n['kind'] == 'Var'), statement,
               'A reduced MX scale Mat admits one load and then its whole shape; other uses stay unmeasured')
    o.fail(node, 'A reduced MX scale Mat needs its whole shape again in the same block')


def routes(node, args):
    name = node['fields']['name']
    return name in {'block.matmul_mx', 'block.matmul_mx_acc'} or name in CALLS and any(operand(arg) for arg in args)


def plane_parameter(o, var, refs):
    shape = tuple(o.literal(ref, var) for ref in refs)
    o.need(len(shape) == 3 and all(type(n) is int and n > 0 for n in shape) and shape[2] == 2, var,
           'E8M0 GM parameters must be static [rows, groups / 2, 2] microscaling planes')
    return shape


def tile(o, node, space, dt, shape, layout):
    """Whole FP8 fractals in Mat (NZ or ZN), Left and Right, and E8M0 planes as ZZ/NN Mats or ScaleLeft/ScaleRight."""
    rows, cols = shape[::-1] if space in {'Right', 'ScaleRight'} or layout in {'nn', 'zn'} else shape
    if dt.name in FP8:
        o.need(space in {'Mat', 'Left', 'Right'} and (layout is None or space == 'Mat' and layout == 'zn')
               and rows % 16 == 0 and cols % 32 == 0, node,
               'FP8 tiles need whole NZ fractals (16-aligned rows, 32-aligned columns) in Mat, Left or Right, or ZN Mats')
    else:
        o.need(dt.name == 'e8m0' and (space in PLANES or space == 'Mat' and layout in {'zz', 'nn'}), node,
               'Scale planes are E8M0 ZZ/NN Mat, ScaleLeft and ScaleRight tiles')
        o.need(rows % 16 == 0 and cols % 2 == 0, node, 'MX scale planes need 16-aligned rows and an even group count')


def address(o, memory):
    return next(addr for (_, addr), entry in o.roots.items() if 'memory' in entry and entry['memory'].value is memory.value)


def later(o, node):
    """The statements after the EvalStmt whose whole expression is ``node``, within its block."""
    if o.mx_next is None:
        o.mx_next = {}
        for seq in [n for n in o.nodes.values() if n['kind'] == 'SeqStmts']:
            statements = [o.node(ref) for ref in seq['fields']['stmts']]
            for i, current in enumerate(statements):
                if current['kind'] == 'EvalStmt':
                    o.mx_next[o.node(current['fields']['expr'])['id']] = statements[i + 1:]
    return o.mx_next.get(node['id'], [])


def binding(o, statement):
    """An assignment that issues nothing: list literals, tile selections and index arithmetic, as Pro hoists them."""
    if statement['kind'] != 'AssignStmt':
        return False
    stack = [statement['fields']['value']]
    while stack:
        node = o.node(stack.pop())
        if node['kind'] not in BINDINGS:
            return False
        stack += [ref for key, value in node['fields'].items() if key != 'type'
                  for ref in (value if isinstance(value, list) else [value]) if isinstance(ref, dict) and '$ref' in ref]
    return True


def named(ctx, call):
    """The values a call's Var operands hold; other expressions give None."""
    refs = [ctx.o.node(ref) for ref in call['fields']['args']]
    return [ctx.env.get(ref['fields']['name']) if ref['kind'] == 'Var' else None for ref in refs]


def moved(ctx, statement):
    """(call, destination, source) of a two-operand move statement, else Nones."""
    call = ctx.o.node(statement['fields']['expr']) if statement['kind'] == 'EvalStmt' else None
    if call is None or call['kind'] != 'Call' or call['fields']['name'] != 'block.move' or len(call['fields']['args']) != 2:
        return None, None, None
    return call, *named(ctx, call)


def partner(ctx, node, side):
    """The plane move statement of a data move: the next one, or Pro's order measured on A5 (mx_split_moves)."""
    o = ctx.o
    if node['id'] in o.mx_split:  # the Right data move of Pro's order
        return o.mx_split.pop(node['id'])
    after = [statement for statement in later(o, node) if not binding(o, statement)][:3]
    moves = [moved(ctx, statement) for statement in after]
    if (side == 'ScaleLeft' and len(moves) == 3 and isinstance(moves[0][1], Memory) and moves[0][1].value.type.space == 'l0b'
            and isinstance(moves[0][2], Memory) and moves[0][2].value.type.dtype.name in FP8
            and [getattr(m[1], 'space', None) for m in moves[1:]] == ['ScaleLeft', 'ScaleRight']):
        o.mx_split[moves[0][0]['id']] = after[2]
        return after[1]
    return after[0] if after else None


def convert(ctx, node, args):
    name = node['fields']['name']
    if name == 'block.load':
        return load(ctx, node, args)
    return move(ctx, node, args) if name == 'block.move' else matmul(ctx, node, args)


def load(ctx, node, args):
    from .memory import memory_call

    o = ctx.o
    o.need(len(args) == 3 and all(isinstance(t, Memory) for t in args[:2]), node, 'Expected typed transfer operands')
    mat, plane = args[:2]
    kwargs = node['fields']['kwargs']
    transposed = kwargs == {'is_transpose': True, 'tile_dims': [0, 1]}
    if 'e8m0' not in (mat.value.type.dtype.name, plane.value.type.dtype.name):
        o.need(mat.value.type.space == 'l1' and (not kwargs or transposed), node,
               'FP8 data loads admit Mat tiles in the default order or with order=[1, 0]')
        return memory_call(ctx, node, args)  # A ZN Mat without order=[1, 0] stays unmeasured there.
    o.need(mat.value.type.space == 'l1' and plane.value.type.space == 'gm' and mat.value.type.dtype == plane.value.type.dtype
           and len(plane.shape) == 3, node, 'MX scale loads read an E8M0 [rows, groups / 2, 2] GM plane into an E8M0 Mat')
    o.need(transposed or kwargs in ({}, {'tile_dims': [0, 1]}), node, 'MX scale loads read plane axes 0 and 1')
    o.need(transposed or not mat.transposed, node, 'A5 reads an NN Mat loaded without order=[1, 0] from a group-major '
           '[groups / 2, rows, 2] plane (mx_group_major), which no target load reads')
    o.need(not transposed or mat.transposed, node, 'A5 reads a ZZ Mat loaded with order=[1, 0] from a group-major '
           '[groups / 2, rows, 2] plane (mx_zz_order), which no target load reads')
    o.need(mat.root is None and plane.root is None, node, 'MX scale loads need whole tiles')
    offsets = args[2]
    o.need(isinstance(offsets, tuple) and len(offsets) == 3 and all(type(v) is int for v in offsets) and offsets[2] == 0,
           node, 'MX scale loads need static [row, group pair, 0] offsets')
    o.need(mat.valid == mat.shape or not mat.transposed or offsets == (0, 0, 0), node,
           'A5 measured reduced NN scale loads only from the plane origin')
    (_, groups), (total, pairs, _), (row, pair_index, _) = mat.value.type.dims, plane.shape, offsets
    rows = mat.valid[1] if mat.transposed else mat.valid[0]  # A reduced Mat loads its valid rows or N.
    o.need(0 <= row and row + rows <= total and 0 <= pair_index and 2 * pair_index + groups <= 2 * pairs, node,
           'Transfer window exceeds the declared GM tensor')
    dt = plane.value.type.dtype
    window = ctx.emit('mem.reshape', node, (plane.value,), typ=MemType('gm', dt, (total, 2 * pairs)),
                      attrs={'shape': [total, 2 * pairs]})
    if (row, pair_index, rows, groups) != (0, 0, total, 2 * pairs):
        window = ctx.emit('mem.slice', node, (window,), typ=MemType('gm', dt, (rows, groups)),
                          attrs={'offsets': [row, 2 * pair_index], 'extents': [rows, groups]})
    ctx.emit('dma.gm_to_l1.mx_scale_nd2nz', node, (mat.value, window),
             attrs={'rows': rows, 'k_groups': groups, 'src_k_groups': 2 * pairs})
    return None


def move(ctx, node, args):
    """A data move checks its operands and finds its plane move; the plane move issues the pair."""
    o = ctx.o
    pending = o.mx_pairs.pop(node['id'], None)
    if pending is not None:
        return pair(ctx, node, args, *pending)
    ctx.attrs(node)
    o.need(len(args) == 2 and isinstance(args[0], Memory) and all(isinstance(v, Memory) for v in slots(args[1])), node,
           'A scale-plane move must follow the FP8 data move of its Left or Right tile, next or in Pro\'s split order')
    dst = args[0]
    side = {'l0a': 'ScaleLeft', 'l0b': 'ScaleRight'}.get(dst.value.type.space)
    for src in slots(args[1]):
        o.need(side is not None and src.value.type.space == 'l1' and src.value.type.dtype == dst.value.type.dtype
               and src.value.type.dtype.name in FP8, node, 'FP8 moves are admitted from a Mat into a Left or Right tile of its dtype')
        o.need(src.shape == dst.shape and src.valid == src.shape and dst.valid == dst.shape and src.root is None
               and dst.root is None, node, 'Only full equal-coordinate Mat-to-Left/Right moves are admitted')
        o.need(not src.slot, node, 'A5 measured MX moves from tile-group slots within the 16-alternative expansion only')
        o.need(not src.transposed or side == 'ScaleLeft', node, 'A5 measured ZN FP8 Mats moving into Left tiles only')
    statement = partner(ctx, node, side)
    call, target, _ = moved(ctx, statement) if statement is not None else (None, None, None)
    o.need(isinstance(target, Plane) and target.space == side, node,
           f'An FP8 Mat-to-Left/Right move needs its {side} plane move next, or Pro\'s order measured on A5: '
           'Left data, Right data, Left plane, Right plane')
    o.mx_pairs[call['id']] = (node, args, side)
    return None


def pair(ctx, node, args, data, operands, side):
    """One dma.l1_to_l0.mx per data and plane slot pair; runtime selections dispatch together (selection)."""
    from .selection import Choice, dispatch

    o = ctx.o
    ctx.attrs(node)
    right = side == 'ScaleRight'
    o.need(len(args) == 2 and isinstance(args[0], Plane) and args[0].space == side and not node['fields']['kwargs']
           and all(isinstance(scale, Memory) and scale.root is None and not scale.slot and scale.value.type.space == 'l1'
                   and scale.value.type.dtype.name == 'e8m0' and scale.transposed == right for scale in slots(args[1])),
           node, f'The {side} plane moves from an E8M0 {"NN" if right else "ZZ"} Mat tile')
    target, dst = args[0], operands[0]
    rows, cols = dst.shape  # Pro coordinates: Left [M, K], Right [K, N].
    shape = (rows // 32, cols) if right else (rows, cols // 32)
    o.need(target.shape == shape and all(scale.shape == shape and scale.valid == shape for scale in slots(args[1])), node,
           f'The {side} plane of a {list(dst.shape)} tile is a whole {list(shape)} E8M0 tile')
    o.need(target.addr == address(o, dst) >> 4, node, f'The {side} plane must sit at its data tile address >> 4')

    def issue(child, values):
        dst, src, _, scale = values
        m, n = src.value.type.dims  # Extents before transpose; a move transposes when exactly one side is typed reversed.
        child.emit('dma.l1_to_l0.mx', data, (dst.value, src.value), attrs={
            'm_src': m, 'n_src': n, 'm_dst': m, 'n_dst': n, 'src_row0': 0, 'src_col0': 0, 'src_is_transpose': right != src.transposed,
            'dst_position': Ident(dst.value.type.space), 'src_mx': scale.value})
        entry = o.ledger.setdefault(data['id'], {'source': data['id'], 'kind': data['kind'], 'target_ids': []})
        entry['target_ids'].append(o.emitted[-1])
        entry['disposition'] = 'translated'

    values = [*operands, *args]
    if any(isinstance(value, Choice) for value in values):
        dispatch(ctx, data, values, issue)
    else:
        issue(ctx, values)
    return None


def matmul(ctx, node, args):
    o, name = ctx.o, node['fields']['name']
    attrs = ctx.attrs(node, {'phase'})
    o.need('phase' not in attrs, node, 'A5 orders a Final-phase store after its MX product through a unit flag (mx_phase); '
           'no target synchronization expresses that flag')
    accumulate = name == 'block.matmul_mx_acc'
    o.need(len(args) == (6 if accumulate else 5) and all(isinstance(t, Memory) for t in args[:-2]), node,
           'Unadmitted MX matmul operands')
    dst, left, right, scale_a, scale_b = args[0], *args[-4:]
    o.need(not accumulate or args[1].value is dst.value, node, 'matmul_mx_acc needs an in-place accumulator')
    types = [t.value.type for t in (dst, left, right)]
    o.need([t.space for t in types] == ['l0c', 'l0a', 'l0b'] and types[0].dtype.name == 'f32'
           and all(t.dtype.name in FP8 for t in types[1:]) and all(t.valid == t.shape for t in (dst, left, right))
           and left.shape[1] == right.shape[0] and left.shape[1] % 64 == 0 and dst.shape == (left.shape[0], right.shape[1]),
           node, 'MX matmul needs whole FP32 Acc [M, N], FP8 Left [M, K] and Right [K, N] tiles with K % 64 == 0')
    (m, k), n = left.shape, right.shape[1]
    for scale, side, data, shape in ((scale_a, 'ScaleLeft', left, (m, k // 32)), (scale_b, 'ScaleRight', right, (k // 32, n))):
        o.need(isinstance(scale, Plane) and scale.space == side and scale.shape == shape and scale.addr == address(o, data) >> 4,
               node, f'The {side} operand is the {list(shape)} plane at its data tile address >> 4')
    ctx.emit('cube.mmad.mx', node, (dst.value, left.value, right.value), attrs={'M': m, 'N': n, 'K': k, 'is_init': not accumulate})
    return None
