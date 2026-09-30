# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""``liveness``: the live range of every on-chip allocation, for the explain output and the backends.

An allocation is live from its first to its last memory access in program order, widened over
every loop that contains an access. The pass records ``live = [first_op_id, last_op_id]`` on each
``mem.alloc`` and a usage summary per memory space; it changes no code (addresses are bump
allocated, RFC-0006 §5 — the ranges show what a reusing allocator would gain).
"""

from __future__ import annotations

from dataclasses import replace

from ..ir import Block, Function, Module, Op, Value
from .manager import Pass, PassContext
from .util import Defs, view_of

PASS = "liveness"


def ranges(f: Function, defs: Defs) -> dict[str, tuple[Op, Op]]:
    flat: list[Op] = []
    loops: list[tuple[int, int]] = []
    first: dict[str, int] = {}
    last: dict[str, int] = {}

    def root_of(v: Value) -> str | None:
        try:
            return view_of(v, defs).root.name
        except Exception:  # noqa: BLE001 - not a memory view
            return None

    def walk(block: Block) -> None:
        for op in block.ops:
            idx = len(flat)
            flat.append(op)
            names = [x for x in op.operands if isinstance(x, Value)] + list(op.attr_values())
            for v in names:
                r = root_of(v)
                if r is not None:
                    first.setdefault(r, idx)
                    last[r] = idx
            for reg in op.regions:
                walk(reg)
            if op.opcode == "cf.for":
                loops.append((idx, len(flat) - 1))

    walk(f.body)
    out: dict[str, tuple[Op, Op]] = {}
    for name, lo in first.items():
        hi = last[name]
        for a, b in loops:
            if a <= lo <= b or a <= hi <= b:
                lo, hi = min(lo, a), max(hi, b)
        out[name] = (flat[lo], flat[hi])
    return out


def run(module: Module, ctx: PassContext) -> Module:
    defs = Defs(module)
    functions = []

    def annotate(f: Function) -> Function:
        """Stamp one function using its own live-range map."""
        live = ranges(f, defs)

        def stamp(block: Block) -> Block:
            out = []
            for op in block.ops:
                if op.opcode == "mem.alloc" and op.results[0].name in live:
                    a, b = live[op.results[0].name]
                    op = replace(op, attrs={**op.attrs, "live": [a.id, b.id]})
                    ctx.explain.note(f"%{op.results[0].name}: live #{a.id}..#{b.id}", op=op.id, kind="live")
                if op.regions:
                    op = replace(op, regions=tuple(stamp(r) for r in op.regions))
                out.append(op)
            return Block(tuple(out))

        return replace(f, body=stamp(f.body))

    for f in module.functions:
        functions.append(annotate(f) if f.kind in ("kernel", "func") else f)
    return Module(module.name, dict(module.attrs), tuple(functions))


PASS_DEF = Pass(PASS, run, accepts="lowered/1", produces="lowered/1", doc="live ranges of on-chip allocations (annotation only)")

__all__ = ["PASS_DEF", "ranges", "run"]
