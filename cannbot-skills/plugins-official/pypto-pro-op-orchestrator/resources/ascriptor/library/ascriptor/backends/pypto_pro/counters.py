# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""vf.mask_update counter cells (I040), kept out of emit.py for its size bound."""
from ...ir import Value
from ...ir.types import CellType, MaskType


def decremented(fn) -> set[str]:
    """Counter cells whose mask_update decrement a read can see: an op after the update in program order, or the
    update itself on the next trip of a loop that does not declare the cell."""
    order = []

    def visit(ops, loops):
        for op in ops:
            order.append((op, loops))
            for region in op.regions:
                visit(region.ops, loops + (op,) if op.opcode == "cf.for" else loops)

    def reads(op, name):
        operands = op.operands[1:] if op.opcode == "scalar.set" else op.operands  # a set writes its first operand
        return any(isinstance(v, Value) and v.name == name for v in operands) or any(v.name == name for v in op.attr_values())

    visit(fn.body.ops, ())
    cells = set()
    for index, (op, loops) in enumerate(order):
        cnt = op.attrs.get("cnt") if op.opcode == "vf.mask_update" else None
        if not isinstance(cnt, Value) or not isinstance(cnt.type, CellType):
            continue
        carried = any(all(r.name != cnt.name for o in loop.walk() for r in o.results) for loop in loops)
        if carried or any(reads(later, cnt.name) for later, _ in order[index + 1:]):
            cells.add(cnt.name)
    return cells


def decrement(printer, op) -> None:
    """plt decrements a counter cell in place (RFC-0001 §6.6), but Pro's update_mask copies its count (A5 job 349):
    after the update, print the saturating decrement of a cell that a later read can see. `c - pl.min(c, n)` ran in
    and out of pl.range bodies (A5 jobs 356, 357); bisheng crashed on `((c - n) if (n < c) else 0)` in one (353, 356)."""
    names = getattr(printer, "_decremented", None)
    if names is None:
        names = printer._decremented = decremented(printer.fn)
    cnt, dst = op.attrs.get("cnt"), op.operands[0]
    if not isinstance(cnt, Value) or cnt.name not in names or not isinstance(dst.type, MaskType):
        return
    cell, lanes = printer.env.ref(cnt), dst.type.lanes
    printer.emit(f"{cell} = ({cell} - pl.min({cell}, {lanes}))")
