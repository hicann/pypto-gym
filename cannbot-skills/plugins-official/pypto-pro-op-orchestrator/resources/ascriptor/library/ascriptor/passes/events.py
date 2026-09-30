# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""``events``: hardware flag ids for every event (RFC-0005 §3).

A flag id is a resource of one *channel* — the ordered pipe pair ``(set_pipe, wait_pipe)`` — and the
core has eight of them per channel (ids 0..7). An event of depth ``d`` occupies ``d`` ids. Two events
of one channel may share ids when their live ranges do not overlap; a live range runs from the first
to the last use, widened to every loop that contains a use (a set in one iteration is consumed in a
later one) and, for pre-set events, over the whole function (their tokens exist from the start).
Literal ids of ``sync.set_flag`` / ``sync.wait_flag`` are reserved on their channel. Ids are assigned
by interval colouring, lowest free id first, in order of first use; the pass records the plan on each
declaration (``ids`` attribute, first id in the event type) and explains every choice.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..ir import Block, Function, Ident, Module, Op, Value
from ..ir.types import EventType
from ..ir.sync_rules import FLAG_IDS
from .manager import Pass, PassContext, PassError
from .util import retype_values

PASS = "events"
BUDGET = FLAG_IDS


class BudgetError(PassError):
    """A channel needs more flag ids at once than the hardware has. Carries the channel and the demand so that
    autosync's ``auto`` coalescing policy can retry that channel with coalesced hand-offs (D-042)."""

    def __init__(self, message: str, *, channel: tuple[str, str], need: int) -> None:
        super().__init__(PASS, message)
        self.channel = channel
        self.need = need
USES = ("sync.set", "sync.wait", "sync.set_all", "sync.release")


@dataclass
class _Event:
    op: Op
    value: Value
    channel: tuple[str, str]
    depth: int
    preset: int
    first: int = -1
    last: int = -1
    ids: list[int] | None = None


def _pipe(x: object) -> str:
    return x.name if isinstance(x, Ident) else str(x)


def _preset_tokens(op: Op, depth: int) -> int:
    p = op.attrs.get("preset", False)
    if isinstance(p, bool):
        return depth if p else 0
    return int(p)


def plan_function(f: Function, ctx: PassContext) -> tuple[Function, dict[str, list[int]]]:
    events: dict[str, _Event] = {}
    reserved: dict[tuple[str, str], set[int]] = {}
    flat: list[Op] = []
    loops: list[tuple[int, int]] = []  # (first index, last index) of every loop body

    def walk(block: Block) -> None:
        for op in block.ops:
            idx = len(flat)
            flat.append(op)
            if op.opcode == "sync.event":
                t = op.results[0].type
                if not isinstance(t, EventType) or not t.set_pipe or not t.wait_pipe:
                    raise PassError(PASS, "sync.event must declare both set and wait pipes")
                events[op.results[0].name] = _Event(op, op.results[0], (t.set_pipe, t.wait_pipe), t.depth, _preset_tokens(op, t.depth))
            elif op.opcode in USES and isinstance(op.operands[0], Value):
                ev = events.get(op.operands[0].name)
                if ev is not None:
                    ev.first = idx if ev.first < 0 else ev.first
                    ev.last = idx
            elif op.opcode in ("sync.set_flag", "sync.wait_flag"):
                ch = (_pipe(op.attrs["src"]), _pipe(op.attrs["dst"]))
                eid = op.attrs.get("event_id")
                if isinstance(eid, int):
                    reserved.setdefault(ch, set()).add(eid)
                elif isinstance(eid, Value):
                    # Any of the raw channel's IDs may be selected at runtime.
                    # Automatic events must not alias that unknown selection.
                    reserved.setdefault(ch, set()).update(range(BUDGET))
            for r in op.regions:
                walk(r)
            if op.opcode == "cf.for":
                loops.append((idx, len(flat) - 1))

    walk(f.body)
    n = len(flat)
    for ev in events.values():
        if ev.first < 0:
            ctx.explain.note(f"{ev.value} is never used", op=ev.op.id, kind="unused")
            ev.first, ev.last = n, n
            continue
        if ev.preset:
            ev.first, ev.last = 0, n - 1
            continue
        for lo, hi in loops:
            if lo <= ev.first <= hi or lo <= ev.last <= hi:
                ev.first, ev.last = min(ev.first, lo), max(ev.last, hi)
    by_channel: dict[tuple[str, str], list[_Event]] = {}
    for ev in events.values():
        by_channel.setdefault(ev.channel, []).append(ev)
    plan: dict[str, list[int]] = {}
    for ch, evs in by_channel.items():
        evs.sort(key=lambda e: (e.first, e.value.name))
        taken = set(reserved.get(ch, ()))
        free = [i for i in range(BUDGET) if i not in taken]
        active: list[_Event] = []
        peak = len(taken)
        for ev in evs:
            if ev.first >= n:  # unused: ids that never collide with anything
                ev.ids = list(range(ev.depth))
                plan[ev.value.name] = ev.ids
                continue
            for a in list(active):
                if a.last < ev.first:
                    active.remove(a)
                    free.extend(a.ids or [])
            free.sort()
            if len(free) < ev.depth:
                names = ", ".join(f"{a.value} (depth {a.depth}, #{flat[a.first].id}..#{flat[a.last].id})" for a in active + [ev])
                need = sum(a.depth for a in active) + ev.depth + len(taken)
                raise BudgetError(f"pipe channel {ch[0]}->{ch[1]} needs {need} flags at once but the hardware has {BUDGET} (event ids "
                                  f"0-{BUDGET - 1}). Events on this channel: {names}. Reduce the buffer depth, or shorten a buffer's live "
                                  "range so two events stop overlapping.", channel=ch, need=need)
            ev.ids, free = free[: ev.depth], free[ev.depth:]
            active.append(ev)
            peak = max(peak, sum(a.depth for a in active) + len(taken))
            plan[ev.value.name] = ev.ids
            ctx.explain.note(f"{ev.value}: {ch[0]}->{ch[1]} ids {ev.ids}{' preset ' + str(ev.preset) if ev.preset else ''}, "
                             f"live #{flat[ev.first].id}..#{flat[ev.last].id}", op=ev.op.id, kind="ids", ids=ev.ids)
        ctx.explain.note(f"channel {ch[0]}->{ch[1]}: peak {peak} of {BUDGET} flags", kind="channel")
    mapping = {}
    for ev in events.values():
        assert ev.ids is not None
        t = ev.value.type
        assert isinstance(t, EventType)
        mapping[ev.value.name] = Value(ev.value.name, EventType(t.depth, t.set_pipe, t.wait_pipe, ev.ids[0]))
    body = retype_values(f.body, mapping)

    def stamp(block: Block) -> Block:
        out = []
        for op in block.ops:
            if op.opcode == "sync.event" and op.results[0].name in plan:
                op = replace(op, attrs={**op.attrs, "ids": list(plan[op.results[0].name])})
            if op.regions:
                op = replace(op, regions=tuple(stamp(r) for r in op.regions))
            out.append(op)
        return Block(tuple(out))

    return replace(f, body=stamp(body)), plan


def run(module: Module, ctx: PassContext) -> Module:
    functions = []
    for f in module.functions:
        if f.kind in ("kernel", "func") and any(o.opcode == "sync.event" for o in f.walk()):
            f, _ = plan_function(f, ctx)
        functions.append(f)
    return Module(module.name, dict(module.attrs), tuple(functions))


PASS_DEF = Pass(PASS, run, doc="flag ids per (set_pipe, wait_pipe) channel by interval colouring under the 8-id budget", establishes=("4",))
# After split_sides the per-side functions are the real compilation units, and one side's events no longer share
# a channel with the other's: the colouring runs again so every id is assigned against that side's own control
# flow and live ranges.
PASS_DEF_RESTAMP = Pass("events_restamp", run, accepts="lowered/1", produces="lowered/1",
                        doc="re-colour flag ids on the split per-side functions", establishes=("4",))

__all__ = ["BUDGET", "PASS_DEF", "PASS_DEF_RESTAMP", "BudgetError", "plan_function", "run"]
