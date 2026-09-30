# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""``split_sides``: one function per core side (RFC-0001 §10 invariant 1, RFC-0006 §4).

A mixed kernel becomes ``@name.cube`` and ``@name.vec`` (``kind = func``, attribute ``side``), both
with the kernel's parameters; ``mode = vec`` / ``cube`` kernels produce one function. Every op goes
to the side its registry entry names; events and barriers go to the side of their pipe; scalar
arithmetic, cells, views, allocations and control flow go to both sides and ``dce`` afterwards
removes what a side does not use. ``region.side`` blocks are copied to their side only.

Vector-only values (D-023): the results of ``core.vec_idx`` / ``core.vec_num`` / ``core.sub_block_idx``
and everything computed from them exist on the vector side only. A cube-side op that consumes one,
or that sits inside a loop or branch decided by one, is an error — the cube core has no such value.
The kernel's attributes (outputs, block_dim, …) move to the module's ``meta`` and the module
becomes ``lowered/1``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..ir import LOWERED, REGISTRY, Block, Function, Ident, Module, Op, Origin, Value
from ..ir.builder import Rewriter
from ..ir.types import EventType
from .util import CUBE_PIPES, VEC_PIPES, op_side
from .manager import Pass, PassContext, PassError

PASS = "split_sides"
VEC_ONLY_SEEDS = ("core.vec_idx", "core.vec_num", "core.sub_block_idx")


def _ident(x: Any) -> str:
    return x.name if isinstance(x, Ident) else str(x)


class _Splitter:
    def __init__(self, f: Function, mode: str, ctx: PassContext) -> None:
        self.f = f
        self.mode = mode
        self.ctx = ctx
        self.vec_only: set[str] = set()  # value names that exist on the vector side only
        self.event_side: dict[str, str] = {}
        for op in f.walk():
            if op.opcode != "sync.event":
                continue
            if "side" in op.attrs:
                self.event_side[op.results[0].name] = _ident(op.attrs["side"])
                continue
            t = op.results[0].type  # a user event: the side owning either of its pipes
            if isinstance(t, EventType):
                pipes = {t.set_pipe, t.wait_pipe}
                if pipes & CUBE_PIPES:
                    self.event_side[op.results[0].name] = "cube"
                elif pipes & VEC_PIPES:
                    self.event_side[op.results[0].name] = "vec"

    def side_of(self, op: Op) -> str:
        """cube | vec | both for one op (regions aside)."""
        if op.opcode in VEC_ONLY_SEEDS:
            return "vec"
        if op.opcode in ("sync.set", "sync.wait", "sync.set_all", "sync.release") and isinstance(op.operands[0], Value):
            s = self.event_side.get(op.operands[0].name)
            if s is not None:
                return s
        if op.opcode == "sync.event":
            return self.event_side.get(op.results[0].name, "both")
        if op.opcode in ("sync.barrier", "sync.set_flag", "sync.wait_flag"):
            pipes = [_ident(op.attrs.get(k)) for k in ("pipe", "src", "dst") if k in op.attrs]
            sides = {s for s in (op_side(replace(op, attrs={"pipe": Ident(p)})) for p in pipes) if s}
            return sides.pop() if len(sides) == 1 else "both"
        if op.opcode.startswith("sync.mutex_"):
            return "both"  # the interpreter / backends pick the side from the flag's kind
        spec = REGISTRY.find(op.opcode)
        if spec is not None and spec.side in ("cube", "vec"):
            return spec.side
        s = op_side(op)
        return s or "both"

    def tainted(self, op: Op) -> bool:
        return any(isinstance(x, Value) and x.name in self.vec_only for x in op.operands) or \
            any(v.name in self.vec_only for v in op.attr_values())

    def split_block(self, block: Block, forced: str | None, vec_region: bool) -> tuple[list[Op], list[Op]]:
        cube: list[Op] = []
        vec: list[Op] = []
        for op in block.ops:
            if op.opcode == "region.side":
                s = _ident(op.attrs["side"])
                if forced is not None and forced != s:
                    raise PassError(PASS, f"region.side {s} #{op.id} ({op.loc}) is nested inside a {forced} region")
                c, v = self.split_block(op.regions[0], s, vec_region or s == "vec")
                cube.extend(c)
                vec.extend(v)
                continue
            side = self.side_of(op)
            taint = self.tainted(op) or vec_region or op.opcode in VEC_ONLY_SEEDS
            if taint:
                if side == "cube":
                    raise PassError(PASS, f"cube-side {op.opcode} #{op.id} ({op.loc}) depends on a vector-only value "
                                    f"(core.vec_idx / vec_num / sub_block_idx or a region decided by one)")
                side = "vec"
                for r in op.results:
                    self.vec_only.add(r.name)
                if op.opcode == "scalar.set" and isinstance(op.operands[0], Value):
                    self.vec_only.add(op.operands[0].name)
            if forced is not None:
                if side not in (forced, "both"):
                    raise PassError(PASS, f"{op.opcode} #{op.id} ({op.loc}) is {side}-side but sits inside a region.side {forced} block")
                side = forced
            if self.mode == "vec" and side == "cube":
                raise PassError(PASS, f"{op.opcode} #{op.id} ({op.loc}) is a cube op in a vec-mode kernel")
            if self.mode == "cube" and side == "vec":
                raise PassError(PASS, f"{op.opcode} #{op.id} ({op.loc}) is a vector op in a cube-mode kernel")
            if op.regions:
                region_vec = taint  # a loop / branch decided by a vector-only value is vector-only as a whole
                parts = [self.split_block(r, forced, vec_region or region_vec) for r in op.regions]
                if side in ("cube", "both") and not region_vec:
                    cube.append(replace(op, regions=tuple(Block(tuple(c)) for c, _ in parts)))
                if side in ("vec", "both"):
                    vec.append(replace(op, regions=tuple(Block(tuple(v)) for _, v in parts)))
                continue
            if side in ("cube", "both"):
                cube.append(op)
            if side in ("vec", "both"):
                vec.append(op)
        return cube, vec


def _renumber(block: Block, rw: Rewriter, taken: set[int]) -> Block:
    """Fresh ids for the ops that also exist on the other side (op ids are unique per module, RFC-0001 §8)."""
    out = []
    for op in block.ops:
        if op.regions:
            op = replace(op, regions=tuple(_renumber(r, rw, taken) for r in op.regions))
        if op.id in taken:
            op = replace(op, id=rw.fresh_id(), origin=op.origin + (Origin(PASS, "moved", (op.id,), "copy on the vector side"),))
        out.append(op)
    return Block(tuple(out))


def run(module: Module, ctx: PassContext) -> Module:
    kernels = [f for f in module.functions if f.kind == "kernel"]
    if len(kernels) != 1:
        raise PassError(PASS, f"expected one kernel, found {len(kernels)}")
    k = kernels[0]
    mode = _ident(module.attrs.get("mode", "mix"))
    sp = _Splitter(k, mode, ctx)
    cube_ops, vec_ops = sp.split_block(k.body, None, False)
    rw = Rewriter(module, PASS)
    functions = [f for f in module.functions if f.kind != "kernel"]
    cube_body, vec_body = Block(tuple(cube_ops)), Block(tuple(vec_ops))
    if mode == "mix":
        vec_body = _renumber(vec_body, rw, {o.id for o in cube_body.walk() if o.id is not None})
    if mode in ("mix", "cube"):
        functions.append(Function("func", f"{k.name}.cube", k.params, {"side": Ident("cube")}, cube_body))
    if mode in ("mix", "vec"):
        functions.append(Function("func", f"{k.name}.vec", k.params, {"side": Ident("vec")}, vec_body))
    meta = {key: v for key, v in k.attrs.items()}
    meta["kernel"] = k.name
    ctx.explain.note(f"@{k.name}: {len(cube_ops)} top-level ops on the cube side, {len(vec_ops)} on the vector side; "
                     f"{len(sp.vec_only)} vector-only value(s)", kind="split")
    attrs = dict(module.attrs)
    attrs["ir"] = LOWERED
    attrs["meta"] = meta
    attrs["next_id"] = rw._next_id
    return Module(module.name, attrs, tuple(functions))


PASS_DEF = Pass(PASS, run, produces=LOWERED, doc="one func per side; scalars and control flow duplicated; vector-only taint checked",
                establishes=("1", "2"))

__all__ = ["PASS_DEF", "run"]
