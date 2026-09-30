# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A5 physical-slot mutex insertion and structured scalar-transparent coalescing."""

from __future__ import annotations

from dataclasses import replace

from ..ir import REGISTRY, Block, Function, Ident, Literal, Module, Op, Rewriter, Value
from ..ir.types import BufType, CellType, MemType, ScalarType, dtype
from .manager import Pass, PassContext, PassError
from .util import Defs, Emit, function_names, literal_or_value, view_of

PLAN = "a5_mutex_plan"
GET = "sync.local_mutex_get"
RELEASE = "sync.local_mutex_release"
OPS = {GET, RELEASE}
LOCAL_SPACES = {"ub", "l1", "l0a", "l0b", "l0c", "bt", "l0amx", "l0bmx"}
PURE = {f"scalar.{name}" for name in (
    "const", "add", "sub", "mul", "and", "or", "xor", "not", "neg", "abs", "min", "max", "cmp", "select", "cast",
)}


def mark(module: Module, body: Block, inside: set[int]) -> Block:
    rw = Rewriter(module, PLAN)

    def walk(block):
        return Block(tuple(rw.rewritten(op, "owned by A5 local mutex autosync")
                           if op.id in inside and not op.regions else
                           replace(op, regions=tuple(walk(r) for r in op.regions)) if op.regions else op
                           for op in block.ops))
    return walk(body)


def _managed(op: Op) -> bool:
    return any(origin.pass_name == PLAN for origin in op.origin)


def memory_values(op: Op) -> list[Value]:
    """Actual memory operands, including complete VF summaries; metadata is not an access."""
    values = []
    if op.opcode == "cf.call":
        for key in ("read", "write"):
            values.extend(op.attrs.get(key, []))
        if not values:
            values.extend(op.operands[1:])
    elif op.opcode == "simt.launch":
        values.extend(op.operands)
    else:
        spec = REGISTRY.get(op.opcode)
        for index, value in enumerate(op.operands):
            operand = spec.operands[min(index, len(spec.operands) - 1)] if spec.operands else None
            if operand is not None and operand.access != "none":
                values.append(value)
        for attr in spec.attrs:
            if attr.access != "none" and attr.name in op.attrs:
                value = op.attrs[attr.name]
                values.extend(value if isinstance(value, (list, tuple)) else [value])
    return [value for value in values if isinstance(value, Value) and isinstance(value.type, (MemType, BufType))]


def insert_function(fn: Function, module: Module, rw: Rewriter, ctx: PassContext) -> Function:
    defs = Defs(Module(module.name, dict(module.attrs), (fn,)))
    names = function_names(fn)
    side = str(fn.attrs["side"])
    needed = set()
    views = {}
    for op in fn.walk():
        if not _managed(op):
            continue
        for value in memory_values(op):
            try:
                view = view_of(value, defs)
            except (TypeError, AssertionError) as error:
                raise PassError("local_mutex", f"cannot resolve #{op.id} memory %{value.name}: {error}") from error
            typ = view.root.type.elem if isinstance(view.root.type, BufType) else view.root.type
            if typ.space in ("gm", "ws"):
                continue
            if typ.space not in LOCAL_SPACES or view.root_op is None or view.root_op.opcode != "mem.alloc":
                raise PassError("local_mutex", f"#{op.id}: unsupported local backing allocation for %{value.name}")
            views[value.name] = view
            needed.add(view.root.name)
    ids = {}
    census = []
    count = 0
    for op in fn.walk():
        if op.opcode != "mem.alloc" or op.results[0].name not in needed:
            continue
        root = op.results[0]
        slots = root.type.slots if isinstance(root.type, BufType) else 1
        ids[root.name] = list(range(count, count + slots))
        count += slots
        census.append(f"%{root.name}: {slots} slots")
        if count > 32:
            raise PassError("local_mutex", f"{side} needs {count} mutex IDs (maximum 32) at #{op.id} {op.loc}; " + "; ".join(census))
        ctx.explain.note(f"{side} %{root.name}: slot mutex IDs {ids[root.name]}", op=op.id, kind="mutex-allocation",
                         side=side, allocation=root.name, mutex_ids=ids[root.name])

    selected = {}

    def selection(value):
        definition = defs.op(value)
        if definition is None or definition.opcode == "mem.alloc":
            return None
        if definition.opcode == "mem.get_buf":
            return selected[value.name]
        if definition.opcode in ("mem.slice", "mem.reinterpret", "mem.reshape"):
            return selection(definition.operands[0])
        raise PassError("local_mutex", f"cannot resolve slot snapshot for %{value.name}")

    def sync(action, ident, anchor, guards):
        return rw.make(action, attrs={"id": ident, "pipe": Ident(pipe_of(anchor)), "side": Ident(side),
                                      "mode": 0, "guards": guards}, from_ops=(anchor,),
                       note="mode-zero ownership of actual physical slots")

    def walk(block):
        output = []
        for original in block.ops:
            op = original
            if op.regions:
                op = replace(op, regions=tuple(walk(region) for region in op.regions))
            if op.opcode == "mem.alloc" and op.results[0].name in ids:
                op = rw.rewritten(op, "one local mutex per physical slot", attrs={**op.attrs, "mutex_ids": ids[op.results[0].name]})
            if op.opcode == "mem.get_buf":
                root = view_of(op.results[0], defs).root.name
                if root in ids:
                    e = Emit(rw, fn, op, names)
                    index = literal_or_value(op.operands[1])
                    size = len(ids[root])
                    # Explicit modulo is also a snapshot when the source index is a Cell.
                    if isinstance(index, int):
                        slot = index % size
                    else:
                        slot = e.fresh("slot")
                        e.pre.append(rw.make("scalar.mod", (index, size), results=(slot,), from_ops=(op,),
                                             note="capture the normalized physical slot"))
                    selected[op.results[0].name] = e.add(ids[root][0], slot)
                    output.extend(e.pre)
                    # Backends may defer the view until a consumer. Both address and
                    # mutex must use this captured selection, not the mutable index.
                    op = rw.rewritten(op, "capture the physical slot and mutex identity together",
                                      operands=(op.operands[0], slot if isinstance(slot, Value) else Literal(slot)))
            accesses = memory_values(op) if _managed(op) else []
            protected = []
            for value in accesses:
                view = views.get(value.name)
                if view is None:
                    continue
                identity = selection(value)
                if identity is None:
                    candidates = ids[view.root.name]
                else:
                    candidates = [identity]
                for ident in candidates:
                    if all(ident != old for old, _ in protected):
                        protected.append((ident, view.root.name))
            gets, releases = [], []
            for index, (ident, root) in enumerate(protected):
                get, release = sync(GET, ident, op, [root]), sync(RELEASE, ident, op, [root])
                # Different dynamic selections of the same ring may coincide at runtime.
                previous = [other for other, other_root in protected[:index] if other_root == root]
                if previous:
                    e = Emit(rw, fn, op, names)
                    condition = None
                    for other in previous:
                        if isinstance(ident, int) and isinstance(other, int):
                            continue
                        comparison = Value(e.fresh("mutex_distinct").name, ScalarType(dtype("b1")))
                        e.pre.append(rw.make("scalar.cmp", (ident, other), results=(comparison,), attrs={"pred": Ident("ne")},
                                             from_ops=(op,), note="deduplicate dynamic slot mutex identities"))
                        if condition is not None:
                            both = Value(e.fresh("mutex_distinct").name, ScalarType(dtype("b1")))
                            e.pre.append(rw.make("scalar.and", (condition, comparison), results=(both,), from_ops=(op,)))
                            condition = both
                        else:
                            condition = comparison
                    output.extend(e.pre)
                    if condition is not None:
                        get = rw.make("cf.if", (condition,), regions=(Block((get,)), Block()), from_ops=(op,), note="acquire unique ID")
                        release = rw.make("cf.if", (condition,), regions=(Block((release,)), Block()), from_ops=(op,), note="release unique ID")
                gets.append(get)
                releases.insert(0, release)
            output.extend((*gets, op, *releases))
        return Block(tuple(output))

    body = walk(fn.body)
    return replace(fn, body=body, attrs={**fn.attrs, "local_mutex_count": count})


def pipe_of(op):
    from .deps import op_pipe
    return op_pipe(op)


def insert(module: Module, ctx: PassContext) -> Module:
    if not module.attrs.get("a5_slot_mutex"):
        return module  # the A2 family plans slot sessions instead, and marks no module for mutexes
    rw = Rewriter(module, "local_mutex")
    functions = tuple(insert_function(fn, module, rw, ctx) if fn.kind == "func" and "side" in fn.attrs else fn
                      for fn in module.functions)
    return Module(module.name, {**module.attrs, "next_id": rw._next_id, "local_mutex_version": 1}, functions)


def coalesce(module: Module, ctx: PassContext) -> Module:
    if not ctx.option("local_mutex_coalesce", True):
        return module
    rw = Rewriter(module, "mutex_coalesce")

    def function(fn):
        definitions = {v.name: op for op in fn.walk() for v in op.results}

        def key(value):
            if isinstance(value, Literal):
                return value.value
            if not isinstance(value, Value):
                return value
            # Never re-evaluate a mutable Cell to prove identity equality.
            if isinstance(value.type, CellType):
                return None
            op = definitions.get(value.name)
            if op and (op.opcode in PURE or op.opcode == "scalar.mod"):
                args = tuple(key(v) for v in op.operands)
                if all(v is not None for v in args):
                    return op.opcode, args, tuple(sorted((k, str(v)) for k, v in op.attrs.items()))
            return ("ssa", value.name)

        def transparent(op):
            if op.regions:
                return False
            if op.opcode in PURE:
                return True
            if op.opcode == "scalar.set":
                # IDs cannot directly reference Cells, and computations remain in place.
                return True
            return False

        def walk(block):
            ops = [replace(op, regions=tuple(walk(region) for region in op.regions)) if op.regions else op for op in block.ops]
            changed = True
            while changed:
                changed = False
                for index, release in enumerate(ops):
                    if release.opcode != RELEASE:
                        continue
                    other = index + 1
                    while other < len(ops) and transparent(ops[other]):
                        other += 1
                    if other == len(ops) or ops[other].opcode != GET:
                        continue
                    get = ops[other]
                    if any(release.attrs[k] != get.attrs[k] for k in ("side", "pipe", "mode")):
                        continue
                    if key(release.attrs["id"]) != key(get.attrs["id"]):
                        continue
                    crossed = [op.id for op in ops[index + 1:other]]
                    ctx.explain.note(f"merge mutex sections: remove #{release.id}/#{get.id} across scalar ops {crossed}",
                                     op=release.id, ops=(release.id, get.id), crossed=crossed, kind="mutex-coalesce")
                    # Preserve a provenance record on the next surviving operation.
                    if other + 1 < len(ops):
                        ops[other + 1] = rw.rewritten(ops[other + 1], f"removed local release #{release.id} / get #{get.id}")
                    del ops[other]
                    del ops[index]
                    changed = True
                    break
            return Block(tuple(ops))
        return replace(fn, body=walk(fn.body))

    return Module(module.name, {**module.attrs, "next_id": rw._next_id}, tuple(function(fn) for fn in module.functions))


PASS_DEF = Pass("local_mutex", insert, accepts="lowered/1", produces="lowered/1",
                doc="A5 per-participant physical-slot mutex IDs and explicit mode-zero sections")
COALESCE_PASS = Pass("mutex_coalesce", coalesce, accepts="lowered/1", produces="lowered/1",
                     doc="merge equal mutex sections within a block across unrelated scalar operations")
