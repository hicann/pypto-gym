# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""``dce``: drop ops nobody observes.

An op is dead when it has no effect (no ``memory`` / ``sync`` / ``control`` / ``debug`` effect
in the registry, no region) and none of its results is used. Dead scalar arithmetic, views,
allocations, cells that are only written, and control flow whose body became empty are removed
until nothing changes. A cell counts as used when something reads it (an operand other than the
first of ``scalar.set``). Kernel outputs are parameters and never dead.
"""

from __future__ import annotations

from dataclasses import replace

from ..ir import REGISTRY, Block, Function, Module, Op, Value
from .manager import Pass, PassContext

PASS = "dce"
KEEP_EFFECTS = {"memory", "sync", "control", "debug"}


def _uses(f: Function) -> set[str]:
    used: set[str] = set()
    for op in f.walk():
        operands = op.operands[1:] if op.opcode == "scalar.set" else op.operands
        for x in operands:
            if isinstance(x, Value):
                used.add(x.name)
        for v in op.attr_values():
            used.add(v.name)
    return used


def _dead(op: Op, used: set[str]) -> bool:
    if op.regions:
        return op.opcode in ("cf.for", "cf.if") and all(not r.ops for r in op.regions) and not any(r.name in used for r in op.results)
    if op.opcode == "mem.alloc":  # an allocation nobody touches on this side (addresses are assigned before the split)
        return not any(r.name in used for r in op.results)
    spec = REGISTRY.find(op.opcode)
    if spec is None or spec.effects & KEEP_EFFECTS or spec.terminator:
        return False
    if op.opcode == "scalar.set":
        return op.operands[0].name not in used  # type: ignore[union-attr]
    return not any(r.name in used for r in op.results)


def sweep(f: Function, ctx: PassContext | None = None) -> Function:
    removed = 0
    while True:
        used = _uses(f)
        gone: list[Op] = []

        def walk(block: Block) -> Block:
            out = []
            for op in block.ops:
                if op.regions:
                    op = replace(op, regions=tuple(walk(r) for r in op.regions))
                if _dead(op, used):
                    gone.append(op)
                    continue
                out.append(op)
            return Block(tuple(out))

        body = walk(f.body)
        if not gone:
            break
        removed += len(gone)
        f = replace(f, body=body)
    if ctx is not None and removed:
        ctx.explain.note(f"@{f.name}: {removed} dead op(s) removed", kind="dce")
    return f


def run(module: Module, ctx: PassContext) -> Module:
    return Module(module.name, dict(module.attrs), tuple(sweep(f, ctx) for f in module.functions))


PASS_DEF = Pass(PASS, run, accepts="lowered/1", produces="lowered/1", doc="remove effect-free ops whose results are unused")

__all__ = ["PASS_DEF", "run", "sweep"]
