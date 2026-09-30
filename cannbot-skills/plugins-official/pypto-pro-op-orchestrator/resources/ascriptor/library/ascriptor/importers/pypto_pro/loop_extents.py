# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Small structural proofs for overflow-safe tiled loops and residual windows."""
from ...ir import Value
from .dynamic import le, positive
from .selection import interval


def expression(ctx, value):
    return ctx.expressions.get(value.name) if isinstance(value, Value) else None


def loop(ctx, iv, lo, hi):
    """Recognize ceil(N/B) written without the overflowing N+B-1 addition."""
    expr = expression(ctx, hi)
    if lo != 0 or expr is None or expr[0] != 'Add' or expr[2] != 1:
        return
    div = expression(ctx, expr[1])
    if div is None or div[0] != 'FloorDiv' or type(div[2]) is not int or div[2] <= 0:
        return
    sub = expression(ctx, div[1])
    if sub is None or sub[0] != 'Sub' or sub[2] != 1 or not positive(ctx, sub[1]):
        return
    if any(interval(ctx, value) is None for value in (hi, expr[1], div[1])):
        return  # Never recover a relation through potentially overflowing arithmetic.
    n, block = sub[1], div[2]
    bounds = interval(ctx, n)
    limit = 1 << (iv.type.dtype.bits - 1)
    if bounds is None or not 0 < bounds[1] < limit:
        return
    ctx.tiled_loops[iv.name] = (n, block)
    ctx.bounds[iv.name] = (0, (bounds[1] - 1) // block)
    ctx.nonnegative.add(iv.name)


def cell_range(ctx, cell, bounds):
    """Keep a reassigned scalar's range at this program point when its dtype holds it."""
    bits = cell.type.dtype.bits
    if bounds is not None and cell.type.dtype.kind == 'int' and -(1 << (bits - 1)) <= bounds[0] <= bounds[1] < 1 << (bits - 1):
        ctx.cell_ranges[cell.name] = bounds
    else:
        ctx.cell_ranges.pop(cell.name, None)


def induction_cells(ctx, child, node, lo, hi):
    """Range each scalar a counted loop reassigns; returns the ranges after the loop.

    A loop without break or continue that steps a scalar once, by `c = c + S` or `c = c - S` at the top level
    of its body, holds entry + S * k before the step of iteration k. Other reassignments give no range."""
    f, after = node['fields'], {}
    body = ctx.o.node(f['body'])['fields']['stmts']
    nodes = list(ctx.walk(f['body']))
    exits = any(n['kind'] in ('BreakStmt', 'ContinueStmt') for n in nodes)
    trips = max(hi - lo, 0) if type(lo) is int and type(hi) is int else None
    assigns = {}
    for n in nodes:
        if n['kind'] == 'AssignStmt':
            assigns.setdefault(ctx.o.node(n['fields']['var'])['fields']['name'], []).append(n)
    updates = []
    for name, steps in assigns.items():
        cell = ctx.cells.get(name)
        if cell is None:
            continue
        value = ctx.o.node(steps[0]['fields']['value'])
        left, right = (ctx.o.node(value['fields'][k]) for k in ('left', 'right')) if value['kind'] in ('Add', 'Sub') else (None, None)
        step = (right['fields']['value'] * (-1 if value['kind'] == 'Sub' else 1)
                if left and left['kind'] == 'Var' and left['fields']['name'] == name and right['kind'] == 'ConstInt' else None)
        updates.append((cell, steps, step))
    from .slots import cursor_steps
    updates.extend(cursor_steps(ctx, nodes))  # Slot-buffer cursors advance through struct.set.
    for cell, steps, step in updates:
        entry = interval(ctx, cell)
        child.cell_ranges.pop(cell.name, None)
        if trips == 0:
            after[cell] = entry
        elif exits or len(steps) != 1 or {'$ref': steps[0]['id']} not in body or None in (step, entry, trips):
            after[cell] = None
        else:
            span = step * (trips - 1)
            cell_range(child, cell, (entry[0] + min(0, span), entry[1] + max(0, span)))
            after[cell] = (entry[0] + step * trips, entry[1] + step * trips)
    return after


def record(ctx, kind, a, b, result):
    ctx.expressions[result.name] = (kind, a, b)
    if kind == 'Mul':
        for iv, block in ((a, b), (b, a)):
            info = ctx.tiled_loops.get(iv.name) if isinstance(iv, Value) else None
            if info is not None and type(block) is int and info[1] == block:
                # binary_bounds must also prove that the multiply cannot overflow.
                if result.name in ctx.bounds:
                    n = info[0]
                    ctx.offset_limits[result.name] = n
    if kind == 'Sub' and isinstance(b, Value) and ctx.offset_limits.get(b.name) == a:
        bounds = interval(ctx, a)
        if bounds is not None and result.type.dtype.name == 'i64':
            ctx.bounds[result.name] = (1, bounds[1])
            ctx.nonnegative.add(result.name)
            ctx.residuals[result.name] = (a, b)


def window(ctx, offset, size, bound):
    offsets = interval(ctx, offset)
    if offsets is None or offsets[0] < 0 or not positive(ctx, size):
        return False
    sizes, bounds = interval(ctx, size), interval(ctx, bound)
    if sizes is not None and bounds is not None and offsets[1] + sizes[1] <= bounds[0]:
        return True
    if offset == 0:
        return le(ctx, size, bound)
    for name, (limit, base) in ctx.residuals.items():
        if base != offset or not le(ctx, limit, bound):
            continue
        if isinstance(size, Value) and (size.name == name or name in ctx.upper.get(size.name, ())):
            return True
    return False
