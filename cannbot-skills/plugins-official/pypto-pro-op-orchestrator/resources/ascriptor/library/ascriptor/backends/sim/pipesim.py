# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The pipe-level simulator (RFC-0006 §7): cycles, synchronisation and hazards of a Lowered module.

Two phases over the same run:

1. **Functional + trace.** The reference interpreter executes the module as before (one lane per
   core side, sub-blocks on their own threads) with a :class:`Tracer` attached: every executed op
   becomes a :class:`Task` — its pipe, its cycle cost from the :mod:`timing` model, the memory
   windows it reads and writes, and for a ``cf.call`` the executed vf ops that give the VF cost.
   The functional result is therefore always the program-order one; the trace is deterministic.

2. **Schedule.** :class:`Scheduler` replays the trace on a machine model: per lane, the scalar
   pipe walks the tasks in issue order and hands each to its pipe's FIFO at the scalar clock; every
   pipe runs its FIFO in order, a task starting at ``max(pipe clock, issue clock, dependencies)``.
   Dependencies are the same-side event tokens (set on one pipe, consumed round-robin on another,
   ``depth`` outstanding at most, pre-set tokens for loop-carried events), the cross-side flags of
   the mutex protocol (visible ``intra_core_sync_latency`` cycles after the signal), the all-core
   collectives and ``pipe_barrier(ALL)``. A wait whose token never arrives is a deadlock, reported
   with the blocked op.

   Every task carries a vector clock (one counter per lane pipe) merged from its pipe's previous
   task, the scalar pipe at issue time and the tokens it consumed. Two tasks that touch overlapping
   bytes of one buffer, at least one writing, are a **hazard** unless the clocks order them — a
   race detector over the executed trace, independent of the cycle numbers. Cross-side ordering
   through the mutex tokens counts; GM conflicts are only checked when asked (cores partition GM by
   convention, and atomics are races on purpose).

The report gives the makespan, per-pipe busy cycles and utilisation, the hazards, the cache-line warnings
(cross-core scalar GM stores into one 64-byte line whose writer does not clean the line after its store, or
whose cross-core publication can run before that store; checked with the GM hazards; I012), the deadlock
(if any) and a Chrome trace of every task for the timeline viewer.
"""

from __future__ import annotations

import heapq
import json
import re
from bisect import bisect_left
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from itertools import product
from pathlib import Path
from typing import Any

from ...devices import load as load_profile
from ...ir import Ident, Module, Op, Value
from .timing import CycleModel, load_model

PIPES = ("S", "MTE1", "MTE2", "MTE3", "M", "V", "FIX")


BLOCK = 32  # bytes: conflict granularity for ordinary reads and on-chip accesses
LINE = 64  # bytes: an A5 data cache line, counted from the start of a GM storage (I012)


@dataclass(frozen=True)
class Access:
    kind: str  # read | write
    key: tuple  # (space, core, base, slot, sub) for on-chip memory, (space, base) for GM
    lo: int  # byte range in the storage (flat), rounded to the access granularity
    hi: int
    r0: int = 0  # rows of the rectangle, when the window is a 2-D rectangle of a 2-D storage
    r1: int = 0
    c0: int = 0  # column byte range within a row, at the same access granularity
    c1: int = 0
    pitch: int = 0  # row pitch in bytes; 0 = flat range only
    name: str = ""
    atomic: bool = False  # an actual SIMT or DMA atomic RMW, not an ordinary load/store
    intervals: tuple[tuple[int, int], ...] = ()  # sorted, disjoint byte runs; lo/hi enclose them

    def overlaps(self, o: Access) -> bool:
        if self.key != o.key or self.lo >= o.hi or o.lo >= self.hi:
            return False
        if self.intervals:
            return any(o._intersects_interval(lo, hi) for lo, hi in self.intervals)
        if o.intervals:
            return any(self._intersects_interval(lo, hi) for lo, hi in o.intervals)
        if self.pitch and self.pitch == o.pitch:
            return self.r0 < o.r1 and o.r0 < self.r1 and self.c0 < o.c1 and o.c0 < self.c1
        if not self.pitch:
            return o._intersects_interval(self.lo, self.hi)
        if not o.pitch:
            return self._intersects_interval(o.lo, o.hi)
        shorter, other = (self, o) if self.r1 - self.r0 <= o.r1 - o.r0 else (o, self)
        return any(other._intersects_interval(row * shorter.pitch + shorter.c0, row * shorter.pitch + shorter.c1)
                   for row in range(shorter.r0, shorter.r1))

    def _intersects_interval(self, lo: int, hi: int) -> bool:
        if self.lo >= hi or lo >= self.hi:
            return False
        if self.intervals:
            index = bisect_left(self.intervals, (hi,)) - 1
            return index >= 0 and self.intervals[index][1] > lo
        if not self.pitch:
            return True
        first = max(self.r0, (lo - self.c1) // self.pitch + 1)
        return first < self.r1 and first * self.pitch + self.c0 < hi


def access_granularity(kind: str, key: tuple, *, atomic: bool = False) -> int:
    """GM stores, including both halves of an atomic RMW, own only their written bytes."""
    return 1 if key[0] in ("gm", "ws") and (kind == "write" or atomic) else BLOCK


def _round_out(lo: int, hi: int, granularity: int = BLOCK) -> tuple[int, int]:
    return lo // granularity * granularity, -(-hi // granularity) * granularity


def interval_accesses(kind: str, key: tuple, spans: Iterable[tuple[int, int]], name: str = "", *, atomic: bool = False) -> list[Access]:
    """Keep a fragmented footprint in one history entry, without filling its holes."""
    merged: list[tuple[int, int]] = []
    for lo, hi in sorted(set(spans)):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    if not merged:
        return []
    return [Access(kind, key, merged[0][0], merged[-1][1], name=name, atomic=atomic,
                   intervals=tuple(merged) if len(merged) > 1 else ())]


def view_accesses(kind: str, key: tuple, shape: tuple[int, ...], strides: tuple[int, ...], origin: int, element_bytes: int, name: str = "") -> list[Access]:
    """Conflict ranges of a resolved view, preserving pitched rectangles and holes.

    Shape/strides/origin use elements of the view dtype. The storage key is shared
    by every view of the allocation; rectangles use absolute rows within that key.
    """
    if len(shape) != len(strides) or any(n <= 0 for n in shape):
        return []
    granularity = access_granularity(kind, key)
    if len(shape) >= 2 and strides[-1] == 1:
        pitch = strides[-2] * element_bytes
        if pitch > 0 and pitch % granularity == 0:
            rectangles = []
            for prefix in product(*(range(n) for n in shape[:-2])):
                start = (origin + sum(i * stride for i, stride in zip(prefix, strides[:-2], strict=True))) * element_bytes
                row, column = divmod(start, pitch)
                c0, c1 = _round_out(column, column + shape[-1] * element_bytes, granularity)
                if c1 > pitch:
                    break
                end_row = row + shape[-2]
                rectangles.append(Access(kind, key, row * pitch + c0, (end_row - 1) * pitch + c1, row, end_row, c0, c1, pitch, name))
            else:
                return rectangles
    # Collapse a contiguous suffix into runs; non-unit/gather strides enumerate
    # only the remaining dimensions. Merge coincident or adjacent conflict ranges.
    suffix, run_elements = len(shape), 1
    while suffix > 0 and strides[suffix - 1] == run_elements:
        suffix -= 1
        run_elements *= shape[suffix]
    spans = []
    for prefix in product(*(range(n) for n in shape[:suffix])):
        start = origin + sum(i * stride for i, stride in zip(prefix, strides[:suffix], strict=True))
        spans.append(_round_out(start * element_bytes, (start + run_elements) * element_bytes, granularity))
    return interval_accesses(kind, key, spans, name)


@dataclass
class Task:
    lane: str  # lane name (core0/cube, core0/vec1 …)
    group: int  # core index
    side: str
    sub: int
    seq: int
    op: Op
    pipe: str
    cost: int
    accesses: list[Access] = field(default_factory=list)
    vf: Any = None  # VfCost for cf.call
    start: int = -1
    end: int = -1
    issued: int = -1
    clock: dict[tuple[str, str], int] = field(default_factory=dict)  # vector clock after the task

    @property
    def label(self) -> str:
        return f"{self.op.opcode} #{self.op.id}"


class Tracer:
    """Collects the tasks of every lane during the functional run (attached to the Machine)."""

    def __init__(self, model: CycleModel) -> None:
        self.model = model
        self.tasks: dict[str, list[Task]] = {}
        self.vf_stack: dict[str, list[list[tuple[Op, int | None, bool]]]] = {}
        self.simt_count: dict[str, int] = {}

    def begin_vf(self, lane: str) -> None:
        self.vf_stack.setdefault(lane, []).append([])

    def record_vf(self, lane: str, op: Op, stride: int | None, folded: bool) -> None:
        st = self.vf_stack.get(lane)
        if st:
            st[-1].append((op, stride, folded))

    def end_vf(self, lane: str) -> list[tuple[Op, int | None, bool]]:
        st = self.vf_stack.get(lane)
        return st.pop() if st else []

    def add(self, task: Task) -> None:
        self.tasks.setdefault(task.lane, []).append(task)

    # -- per-process tracing: a core group's lanes run in a forked child, whose tasks come back through a queue

    def export(self, lanes: list[str]) -> dict[str, list[tuple]]:
        """The tasks of ``lanes`` as plain tuples (ops by id): what a child process sends its parent."""
        return {lane: [(t.group, t.side, t.sub, t.seq, t.op.id, t.pipe, t.cost, t.accesses, t.vf,
                        {"event_id": t.op.attrs["event_id"]} if t.op.opcode in ("sync.set_flag", "sync.wait_flag") else
                        {"id": t.op.attrs["id"]} if t.op.opcode.startswith("sync.local_mutex_") else {})
                       for t in self.tasks.get(lane, [])]
                for lane in lanes}

    def absorb(self, payload: dict[str, list[tuple]], ops: dict[int, Op]) -> None:
        for lane, rows in payload.items():
            self.tasks[lane] = [Task(lane, group, side, sub, seq,
                                    replace(ops[op_id], attrs={**ops[op_id].attrs, **runtime_attrs}) if runtime_attrs else ops[op_id],
                                    pipe, cost, accesses, vf)
                                for group, side, sub, seq, op_id, pipe, cost, accesses, vf, runtime_attrs in rows]


# ------------------------------------------------------------------------------ the scheduler


class Deadlock(RuntimeError):
    pass


@dataclass
class _EventState:
    depth: int
    tokens: deque  # (ready cycle, setter clock)
    sets: int = 0
    waits: int = 0
    channel: tuple[str, str] | None = None
    ids: tuple[int, ...] | None = None
    next_set: int = 0
    pending: deque = field(default_factory=deque)
    history: list[_FlagToken] = field(default_factory=list)


@dataclass
class _FlagToken:
    ready: int
    setter: str
    slot: int
    consumed: int | None = None
    waiter: str | None = None
    preset: bool = False


@dataclass
class _Pipe:
    key: tuple[str, str]
    clock: int = 0
    busy: int = 0
    queue: deque = field(default_factory=deque)
    vc: dict[tuple[str, str], int] = field(default_factory=dict)
    seq: int = 0
    blocked_on: str | None = None


class Scheduler:
    def __init__(self, tracer: Tracer, model: CycleModel, *, check_gm: bool = False) -> None:
        self.model = model
        self.profile = load_profile('a5' if model.device_family == 'a5' else 'a2')
        self.check_gm = check_gm
        # launch order, not the order the tracer met the lanes in (lane timing, or which forked group reported first):
        # every round of the schedule follows it, and with it the report and the Chrome trace
        self.lanes = {lane: tracer.tasks[lane] for lane in sorted(tracer.tasks, key=_launch_order)}
        self.pipes: dict[tuple[str, str], _Pipe] = {}
        self.cursor: dict[str, int] = {}  # lane -> next task to issue
        self.events: dict[tuple[str, str], _EventState] = {}  # (lane, event name)
        self.cross: dict[tuple[int, str, int], deque] = {}  # (group, kind, flag) -> tokens (signal, visible, clock); kind cube->vec per sub
        self.collective: dict[tuple[str, int], deque] = {}
        self.collective_base: dict[tuple[str, int], int] = {}
        self.collective_counts: dict[tuple, int] = {}
        self.hazards: list[str] = []
        self.warnings: list[str] = []
        self.deadlock: str | None = None
        self.done: list[Task] = []
        self.history: dict[tuple, list[tuple[Access, Task]]] = {}
        self.lines: dict[tuple, dict[int, dict[tuple[str, bool], Task]]] = {}  # GM storage -> line -> (lane, atomic) -> first store
        self.warned: set[tuple] = set()
        self.positions: dict[str, dict[int, int]] = {}  # lane -> task identity -> issue order, for the cache-line check
        self.latency = model.intra_core_latency
        self.flag_timeline: dict[str, Any] = {}
        self._flag_timeline_checked = False
        self.local_mutexes: dict[str, list[Task | None]] = {lane: [None] * 32 for lane in self.lanes}
        self.mutex_predecessors: dict[tuple[str, int], Task | None] = {}
        self.mutex_sections: list[tuple[Task, Task]] = []
        for lane in self.lanes:
            for p in PIPES:
                self.pipes[(lane, p)] = _Pipe((lane, p))
            self.cursor[lane] = 0
        self._prepare_local_mutexes()

    def _prepare_local_mutexes(self) -> None:
        """Bind predecessor releases in scalar issue order, never by pipe scheduling order."""
        for lane, tasks in self.lanes.items():
            held: list[Task | None] = [None] * 32
            for task in tasks:
                if not task.op.opcode.startswith("sync.local_mutex_"):
                    continue
                ident = int(task.op.attrs["id"])
                if not 0 <= ident < 32 or task.op.attrs.get("mode") != 0:
                    raise ValueError(f"{lane}: invalid local mutex ID/mode at {task.label}")
                if task.op.opcode == "sync.local_mutex_get":
                    if held[ident] is not None:
                        raise ValueError(f"{lane}: repeated local mutex get {ident} at {task.label}")
                    self.mutex_predecessors[(lane, task.seq)] = self.local_mutexes[lane][ident]
                    held[ident] = task
                else:
                    getter = held[ident]
                    if getter is None or getter.pipe != task.pipe:
                        raise ValueError(f"{lane}: unmatched local mutex release {ident} at {task.label}")
                    self.mutex_sections.append((getter, task))
                    self.local_mutexes[lane][ident] = task
                    held[ident] = None
            if any(task is not None for task in held):
                raise ValueError(f"{lane}: unreleased local mutex at end of trace")

    # -- driving -----------------------------------------------------------------------------

    def run(self) -> None:
        progress = True
        while progress:
            progress = False
            for lane in self.lanes:
                progress |= self._step_scalar(lane)
                for p in PIPES[1:]:
                    progress |= self._step_pipe(self.pipes[(lane, p)])
        pending = [(lane, self.cursor[lane], len(tasks)) for lane, tasks in self.lanes.items() if self.cursor[lane] < len(tasks)]
        queued = [p for p in self.pipes.values() if p.queue]
        if pending or queued:
            where = []
            for lane, i, _ in pending:
                t = self.lanes[lane][i]
                where.append(f"{lane}/S blocked at {t.label} ({t.op.loc})")
            for p in queued:
                t = p.queue[0]
                where.append(f"{p.key[0]}/{p.key[1]} blocked at {t.label} ({t.op.loc}) waiting for {p.blocked_on}")
            self.deadlock = "deadlock: " + "; ".join(where)
        self._check_flag_timeline()

    def _issue_clock(self, lane: str) -> _Pipe:
        return self.pipes[(lane, "S")]

    def _step_scalar(self, lane: str) -> bool:
        """The scalar pipe issues the next task of the lane, or executes it itself when it is a scalar / S-pipe op."""
        tasks = self.lanes[lane]
        i = self.cursor[lane]
        if i >= len(tasks):
            return False
        t = tasks[i]
        s = self.pipes[(lane, "S")]
        if t.pipe == "S":
            ok = self._execute(t, s, issued_by=None)
            if not ok:
                return False
        else:
            t.issued = s.clock
            s.clock += 1  # the issue marker
            t.clock = dict(s.vc)
            self.pipes[(lane, t.pipe)].queue.append(t)
        self.cursor[lane] = i + 1
        return True

    def _step_pipe(self, p: _Pipe) -> bool:
        if not p.queue:
            return False
        t = p.queue[0]
        if not self._execute(t, p, issued_by=t.clock):
            return False
        p.queue.popleft()
        return True

    # -- one task ----------------------------------------------------------------------------

    def _execute(self, t: Task, p: _Pipe, issued_by: dict | None) -> bool:
        oc = t.op.opcode
        start = max(p.clock, t.issued if t.issued >= 0 else 0)
        vc = dict(p.vc)
        if issued_by:
            _merge(vc, issued_by)
        dep_ready = 0
        consumed: list[_FlagToken] = []
        if oc == "sync.local_mutex_get":
            previous = self.mutex_predecessors[(t.lane, t.seq)]
            if previous is not None:
                if previous.end < 0:
                    p.blocked_on = f"local mutex {t.op.attrs['id']} release {previous.label} on {previous.pipe}"
                    return False
                dep_ready = previous.end
                _merge(vc, previous.clock)
        elif oc == "sync.local_mutex_release":
            pass  # p.clock already includes completion of all preceding work on this pipe
        elif oc == "sync.set" or oc == "sync.set_all":
            ev = self._event(t)
            n = ev.depth if oc == "sync.set_all" else 1
            if len(ev.tokens) + n > ev.depth:
                self.hazards.append(f"{t.lane}: {t.label} sets {self._ev_name(t)} while {len(ev.tokens)} token(s) are outstanding "
                                    f"(depth {ev.depth}); a set overtook a wait")
            ready = start + t.cost
            for _ in range(n):
                ev.tokens.append((ready, dict(vc)))
                self._record_set(ev, ready, t)
            ev.sets += n
        elif oc in ("sync.wait", "sync.release"):
            ev = self._event(t)
            n = ev.depth if oc == "sync.release" else 1
            if len(ev.tokens) < n:
                p.blocked_on = f"a set of {self._ev_name(t)}"
                return False
            for _ in range(n):
                ready, setter = ev.tokens.popleft()
                consumed.append(ev.pending.popleft())
                dep_ready = max(dep_ready, ready)
                _merge(vc, setter)
            ev.waits += n
        elif oc == "sync.mutex":
            # the old kernelbase prologue: the consumer side publishes `depth` free tokens before the body
            kind = t.op.attrs.get("kind")
            kind = kind.name if isinstance(kind, Ident) else str(kind)
            flag, depth = int(t.op.attrs.get("id", 0)), int(t.op.attrs["depth"])
            if kind == "vc" and t.side == "cube":
                for sub in (0, 1):
                    q = self.cross.setdefault((t.group, f"cube->vec{sub}", flag), deque())
                    q.extend((0, 0, {}) for _ in range(depth))
            elif kind == "cv" and t.side == "vec":
                q = self.cross.setdefault((t.group, f"vec{t.sub}->cube", flag), deque())
                q.extend((0, 0, {}) for _ in range(depth))
        elif oc.startswith("sync.crosscore."):
            r = self._crosscore(t, oc.removeprefix("sync.crosscore."), start, vc)
            if r is None:
                p.blocked_on = f"the cross-core partner of {oc}"
                return False
            dep_ready = r
        elif oc == "sync.barrier":
            pipe = t.op.attrs.get("pipe", "ALL")
            pipe = pipe.name if isinstance(pipe, Ident) else str(pipe)
            if pipe == "ALL":
                lane = t.lane
                for q in PIPES[1:]:
                    other = self.pipes[(lane, q)]
                    if other.queue:
                        p.blocked_on = f"{q} to drain"
                        return False
                    dep_ready = max(dep_ready, other.clock)
                    _merge(vc, other.vc)
                    other.vc = dict(vc)  # everything after the barrier follows everything before it
        elif oc in ("sync.set_flag", "sync.wait_flag"):
            key = (t.lane, f"flag:{t.op.attrs.get('src')}->{t.op.attrs.get('dst')}#{t.op.attrs.get('event_id')}")
            channel = tuple(str(getattr(t.op.attrs[k], 'name', t.op.attrs[k])) for k in ('src', 'dst'))
            raw_id = t.op.attrs.get('event_id')
            ids = (raw_id,) if isinstance(raw_id, int) else None
            ev = self.events.setdefault(key, _EventState(1, deque(), channel=channel, ids=ids))
            if oc == "sync.set_flag":
                ev.tokens.append((start + t.cost, dict(vc)))
                self._record_set(ev, start + t.cost, t)
            else:
                if not ev.tokens:
                    p.blocked_on = f"set_flag {key[1]}"
                    return False
                ready, setter = ev.tokens.popleft()
                consumed.append(ev.pending.popleft())
                dep_ready = ready
                _merge(vc, setter)
        start = max(start, dep_ready)
        for token in consumed:
            token.consumed, token.waiter = start, f"{t.label} ({t.op.loc})"
        end = start + t.cost
        t.start, t.end = start, end
        p.clock = end
        p.busy += t.cost
        p.seq += 1
        vc[p.key] = p.seq
        p.vc = vc
        p.blocked_on = None
        t.clock = dict(vc)
        self.done.append(t)
        self._check_hazards(t)
        if self.check_gm and t.op.opcode in ("scalar.store", "simt.launch"):
            self._check_lines(t)
        return True

    def _ev_name(self, t: Task) -> str:
        x = t.op.operands[0]
        return x.name if isinstance(x, Value) else str(x)

    def _event(self, t: Task) -> _EventState:
        key = (t.lane, self._ev_name(t))
        ev = self.events.get(key)
        if ev is None:
            x = t.op.operands[0]
            typ = getattr(x, "type", None)
            depth = getattr(typ, "depth", 1) or 1
            channel = (typ.set_pipe, typ.wait_pipe) if typ is not None else None
            ids = (typ.id,) if depth == 1 and getattr(typ, 'id', None) is not None else None
            ev = self.events[key] = _EventState(depth, deque(), channel=channel, ids=ids)
        return ev

    def declare_event(self, lane: str, name: str, depth: int, preset: int, *,
                      channel: tuple[str, str] | None = None, ids: list[int] | tuple[int, ...] | None = None) -> None:
        from ...ir.sync_rules import FLAG_IDS

        binding = tuple(ids) if ids is not None else None
        if not 0 <= preset <= depth:
            raise ValueError(f"event {name}: preset must be in 0..{depth}")
        if binding is not None and (len(binding) != depth or len(set(binding)) != depth
                                    or any(not isinstance(i, int) or not 0 <= i < FLAG_IDS for i in binding)):
            raise ValueError(f"event {name}: allocated flag IDs must be {depth} distinct values in 0..{FLAG_IDS - 1}")
        ev = self.events[(lane, name)] = _EventState(depth, deque(), channel=channel, ids=binding)
        for _ in range(preset):
            ev.tokens.append((0, {}))
            token = _FlagToken(0, f"preset of {name}", ev.next_set % depth, preset=True)
            ev.next_set += 1
            ev.pending.append(token)
            ev.history.append(token)

    @staticmethod
    def _record_set(ev: _EventState, ready: int, task: Task) -> None:
        token = _FlagToken(ready, f"{task.label} ({task.op.loc})", ev.next_set % ev.depth)
        ev.next_set += 1
        ev.pending.append(token)
        ev.history.append(token)

    def _check_flag_timeline(self) -> None:
        """Construction-time FIFO pops do not free a flag before its scheduled wait."""
        if self._flag_timeline_checked:
            return
        self._flag_timeline_checked = True
        physical: dict[tuple[str, str, str, int], list[_FlagToken]] = {}
        missing = []
        checked = bound = 0

        def overlaps(tokens: list[_FlagToken], capacity: int, label: str) -> None:
            active: list[tuple[float, int, _FlagToken]] = []
            for serial, token in enumerate(sorted(tokens, key=lambda item: item.ready)):
                end = token.consumed if token.consumed is not None else float('inf')
                while active and active[0][0] <= token.ready:
                    heapq.heappop(active)
                if len(active) >= capacity:
                    previous = active[0][2]
                    self.hazards.append(
                        f"temporal flag hazard on {label}: {token.setter} completes at cycle {token.ready} "
                        f"with {len(active)} token(s) still outstanding (capacity {capacity}); "
                        f"{previous.setter} is consumed at {previous.consumed} by {previous.waiter}")
                if end > token.ready:
                    heapq.heappush(active, (end, serial, token))

        for (lane, name), ev in self.events.items():
            checked += len(ev.history)
            overlaps(ev.history, ev.depth, f"{lane}/event {name}")
            if ev.channel is None or None in ev.channel or ev.ids is None:
                if ev.history:
                    missing.append(f"{lane}/{name}")
                continue
            for token in ev.history:
                physical.setdefault((lane, *ev.channel, ev.ids[token.slot]), []).append(token)
                bound += 1
        for (lane, src, dst, flag), tokens in physical.items():
            if sum(token.preset for token in tokens) > 1:
                self.hazards.append(f"temporal flag hazard on {lane}/{src}->{dst} flag {flag}: "
                                    "multiple event presets occupy the same physical flag before the body")
            overlaps(tokens, 1, f"{lane}/{src}->{dst} flag {flag}")
        self.flag_timeline = {"checked_tokens": checked, "physical_checked_tokens": bound,
                              "physical_bindings_complete": not missing, "unbound_events": sorted(missing)}

    # -- cross-core ---------------------------------------------------------------------------

    def _crosscore(self, t: Task, kind: str, start: int, vc: dict) -> int | None:
        g, flag = t.group, int(t.op.attrs.get("flag_id", 0))
        if not 0 <= flag <= self.profile.crosscore_id_max:
            raise RuntimeError(f"cross-core flag ID must be in 0..{self.profile.crosscore_id_max}, got {flag}")
        signal = start
        visible = signal + self.latency
        if kind == "cube_ready":
            if any(len(self.cross.get((g, f"cube->vec{sub}", flag), ())) >= self.profile.crosscore_counter_max for sub in (0, 1)):
                raise RuntimeError(f"cross-core flag {flag} exceeds {self.profile.crosscore_counter_max} pending tokens")
            for sub in (0, 1):
                self.cross.setdefault((g, f"cube->vec{sub}", flag), deque()).append((signal, visible, dict(vc)))
            return 0
        if kind == "vec_ready":
            if len(self.cross.get((g, f"vec{t.sub}->cube", flag), ())) >= self.profile.crosscore_counter_max:
                raise RuntimeError(f"cross-core flag {flag} exceeds {self.profile.crosscore_counter_max} pending tokens")
            self.cross.setdefault((g, f"vec{t.sub}->cube", flag), deque()).append((signal, visible, dict(vc)))
            return 0
        if kind == "wait_cube":
            q = self.cross.get((g, f"cube->vec{t.sub}", flag))
            if not q:
                return None
            _, vis, setter = q.popleft()
            _merge(vc, setter)
            return vis
        if kind == "wait_vec":
            qs = [self.cross.get((g, f"vec{sub}->cube", flag)) for sub in (0, 1)]
            if not all(qs):
                return None
            vis = 0
            for q in qs:
                _, v, setter = q.popleft()  # type: ignore[union-attr]
                vis = max(vis, v)
                _merge(vc, setter)
            return vis
        if kind in ("allcube_ready", "allvec_ready", "intracore_allvec_ready"):
            side = "cube" if kind == "allcube_ready" else "vec"
            scope = (side, flag) if kind != "intracore_allvec_ready" else (f"vec@{g}", flag)
            phases = self.collective.setdefault(scope, deque())
            key = (t.lane, "ready")
            n = self.collective_counts.get((scope, *key), 0)
            base = self.collective_base.get(scope, 0)
            if n - base >= self.profile.crosscore_counter_max:
                raise RuntimeError(f"{kind} #{t.op.id}: collective flag {flag} exceeds {self.profile.crosscore_counter_max} pending generations on {t.lane}")
            while len(phases) <= n - base:
                phases.append([])
            phases[n - base].append((key, (signal, dict(vc))))
            self.collective_counts[(scope, *key)] = n + 1
            return 0
        if kind in ("allcube_wait", "allvec_wait", "intracore_allvec_wait"):
            side = "cube" if kind == "allcube_wait" else "vec"
            scope = (side, flag) if kind != "intracore_allvec_wait" else (f"vec@{g}", flag)
            phases = self.collective.setdefault(scope, deque())
            key = (t.lane, "wait")
            n = self.collective_counts.get((scope, *key), 0)
            base = self.collective_base.get(scope, 0)
            members = {lane for lane in self.lanes if (lane.endswith("/cube") if side == "cube" else "/vec" in lane)
                       and (kind != "intracore_allvec_wait" or lane.startswith(f"core{g}/"))}
            if len(phases) <= n - base:
                return None
            arrived = {who[0] for who, _ in phases[n - base] if who[1] == "ready"}
            if not members <= arrived:
                return None
            vis = 0
            for who, (sig, setter) in phases[n - base]:
                if who[1] == "ready":
                    vis = max(vis, sig)
                    _merge(vc, setter)
            self.collective_counts[(scope, *key)] = n + 1
            consumed = min(self.collective_counts.get((scope, member, "wait"), 0) for member in members)
            while base < consumed:
                phases.popleft()
                base += 1
            self.collective_base[scope] = base
            return vis
        return 0

    # -- hazards -----------------------------------------------------------------------------

    def _check_hazards(self, t: Task) -> None:
        for a in t.accesses:
            if a.kind == "clean" or (a.key[0] in ("gm", "ws") and not self.check_gm):
                continue  # a clean_dcache names bytes but stores none: it is no side of a hazard (RFC-0006 §9)
            prev = self.history.setdefault(a.key, [])
            for b, u in prev:
                if a.kind == "read" and b.kind == "read":
                    continue
                if a.atomic and b.atomic:
                    continue
                if not a.overlaps(b):
                    continue
                if _before(u.clock, t.clock):
                    continue
                kind = {("write", "read"): "RAW", ("read", "write"): "WAR", ("write", "write"): "WAW"}[(b.kind, a.kind)]
                self.hazards.append(f"{kind} hazard on %{a.name or b.name} ({a.key[0]}): {u.lane}/{u.pipe} {u.label} ({u.op.loc}) "
                                    f"cycles [{u.start}, {u.end}) and {t.lane}/{t.pipe} {t.label} ({t.op.loc}) cycles [{t.start}, {t.end}) "
                                    "are not ordered by any event, barrier or mutex")
            prev.append((a, t))
            if len(prev) > 256:
                floor = self._floor()
                prev[:] = [(b, u) for b, u in prev if not _before(u.clock, floor)][-128:]

    def _check_lines(self, t: Task) -> None:
        """Warn once per pair of source operations when different cores store into one GM cache line without the
        protocol A5 needs: the writer cleans the line after its store, and its cross-core publication cannot run
        before that store (I012, RFC-0006 §9)."""
        for a in t.accesses:
            if a.kind != "write" or a.key[0] not in ("gm", "ws"):
                continue
            lines = self.lines.setdefault(a.key, {})
            for lo, hi in a.intervals or ((a.lo, a.hi),):
                for line in range(lo // LINE, (hi - 1) // LINE + 1):
                    stores = lines.setdefault(line, {})
                    for (lane, atomic), u in stores.items():
                        pair = (a.key, min(u.op.id, t.op.id), max(u.op.id, t.op.id))
                        if lane == t.lane or (atomic and a.atomic) or pair in self.warned:
                            continue
                        self.warned.add(pair)  # one verdict per pair of source operations, warning or not
                        gaps = self._protocol_gaps(u, t, a.key, line)
                        if gaps:
                            self.warnings.append(
                                f"cache-line warning on %{a.name} ({a.key[0]}): {u.lane} {u.label} ({u.op.loc}) and {t.lane} "
                                f"{t.label} ({t.op.loc}) store into its 64-byte line {line} from different cores; "
                                f"{'; '.join(gaps)}. A5 loses such scalar stores unless every writer cleans the line after "
                                "its store and its cross-core publication cannot run before that store (I012)")
                    stores.setdefault((t.lane, a.atomic), t)

    def _protocol_gaps(self, u: Task, t: Task, key: tuple, line: int) -> list[str]:
        """What the pair misses: the cross-core order between the two stores, then the earlier writer's protocol."""
        if _before(u.clock, t.clock):
            return self._writer_gaps(u, key, line)
        if _before(t.clock, u.clock):
            return self._writer_gaps(t, key, line)
        return [f"no cross-core flag orders {u.lane} {u.label} before {t.lane} {t.label}"]

    def _writer_gaps(self, store: Task, key: tuple, line: int) -> list[str]:
        """Walk the writer's own lane from its store: the clean of that line, then its first cross-core publication
        and whether the storing pipe reaches the publishing one (the same pipe, a barrier, or flags set after the
        store). A publication that the storing pipe does not reach can run before the store on A5 (I012)."""
        missing, reach, issued, cleaned = [], {store.pipe}, set(), False
        for task in self.lanes[store.lane][self._position(store) + 1:]:
            code = task.op.opcode
            if code == "core.clean_dcache":
                cleaned = cleaned or _cleans(task, key, line)
            elif code == "sync.set_flag":
                issued.add((_name(task.op.attrs.get("src")), task.op.attrs.get("event_id")))
            elif code == "sync.wait_flag":
                flag = (_name(task.op.attrs.get("src")), task.op.attrs.get("event_id"))
                if flag in issued and flag[0] in reach:
                    reach.add(_name(task.op.attrs.get("dst")))
            elif code == "sync.barrier" and _name(task.op.attrs.get("pipe"), "ALL") == "ALL":
                reach = set(PIPES)
            elif code.startswith("sync.crosscore.") and code.endswith("_ready"):
                if not cleaned:
                    missing.append(f"{store.lane} does not clean line {line} after {store.label}")
                if task.pipe not in reach:
                    missing.append(f"{store.lane} publishes {task.label} ({task.op.loc}) on {task.pipe} without waiting "
                                   f"for the {store.pipe} pipe of {store.label}")
                return missing
        if not cleaned:
            missing.append(f"{store.lane} does not clean line {line} after {store.label}")
        return [*missing, f"{store.lane} never publishes {store.label} to another core"]

    def _position(self, task: Task) -> int:
        """The task's place in its lane's issue order."""
        index = self.positions.get(task.lane)
        if index is None:
            index = self.positions[task.lane] = {id(x): i for i, x in enumerate(self.lanes[task.lane])}
        return index[id(task)]

    def _floor(self) -> dict:
        """A clock every future task will be ahead of: the minimum over all pipes."""
        out: dict[tuple[str, str], int] = {}
        for p in self.pipes.values():
            for k, v in p.vc.items():
                out[k] = min(out.get(k, v), v)
        return out

    # -- reporting ---------------------------------------------------------------------------

    def report(self) -> dict[str, Any]:
        makespan = max((t.end for t in self.done), default=0)
        pipes = {}
        for key, p in self.pipes.items():
            if p.busy or p.seq:
                pipes[f"{key[0]}/{key[1]}"] = {"busy": p.busy, "end": p.clock, "tasks": p.seq,
                                                "utilisation": round(p.busy / makespan, 4) if makespan else 0.0}
        return {"cycles": makespan, "pipes": pipes, "hazards": list(self.hazards), "warnings": list(self.warnings),
                "deadlock": self.deadlock,
                "tasks": len(self.done), "flag_timeline": dict(self.flag_timeline),
                "local_mutex": {"slots_per_lane": 32, "sections": len(self.mutex_sections),
                                "ids_by_lane": {lane: [i for i, release in enumerate(slots) if release is not None]
                                                for lane, slots in self.local_mutexes.items()},
                                "timing": "modeled, not silicon calibrated"}}

    def chrome_trace(self) -> dict[str, Any]:
        events = []
        for t in self.done:
            if t.cost <= 0 and t.pipe == "S":
                continue
            events.append({"name": t.label, "cat": "sync" if t.op.opcode.startswith("sync.") else "pipe", "ph": "X", "ts": t.start,
                           "dur": max(t.end - t.start, 0), "pid": t.group, "tid": f"{t.lane}/{t.pipe}",
                           "args": {"time_domain": "cycle", "loc": str(t.op.loc), "cost": t.cost, "issued": t.issued}})
            if t.op.opcode.startswith("sync.local_mutex_"):
                events[-1]["args"].update(mutex_id=t.op.attrs["id"], mode=0)
        return {"traceEvents": events, "displayTimeUnit": "ns"}


def _name(value: Any, default: str = "") -> str:
    """The identifier an attribute carries, whichever spelling the module used."""
    return value.name if isinstance(value, Ident) else value if isinstance(value, str) else default


def _cleans(task: Task, key: tuple, line: int) -> bool:
    """Does this ``core.clean_dcache`` cover that cache line? ENTIRE_DATA_CACHE, its default, covers every line."""
    if _name(task.op.attrs.get("entire_type"), "ENTIRE_DATA_CACHE") == "ENTIRE_DATA_CACHE":
        return True
    return any(a.kind == "clean" and a.key == key and any(lo // LINE <= line <= (hi - 1) // LINE
                                                          for lo, hi in a.intervals or ((a.lo, a.hi),))
               for a in task.accesses)


def _merge(a: dict, b: dict) -> None:
    for k, v in b.items():
        if a.get(k, -1) < v:
            a[k] = v


def _before(a: dict, b: dict) -> bool:
    """a happens-before b: every counter of a is at most b's (a is a prefix of b's knowledge)."""
    return all(b.get(k, -1) >= v for k, v in a.items())


def _launch_order(lane: str) -> list:
    """core0/cube, core0/vec0, core0/vec1, core1/cube …: the numbers in a lane name compare as numbers (core2 < core10)."""
    return [int(part) if index % 2 else part for index, part in enumerate(re.split(r"(\d+)", lane))]


# ------------------------------------------------------------------------------ entry point


@dataclass
class SimResult:
    outputs: list[Any]
    report: dict[str, Any]
    scheduler: Scheduler

    @property
    def cycles(self) -> int:
        return int(self.report["cycles"])

    @property
    def hazards(self) -> list[str]:
        return list(self.report["hazards"])

    @property
    def warnings(self) -> list[str]:
        return list(self.report["warnings"])

    def write_trace(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.scheduler.chrome_trace()), encoding="utf-8")


def simulate(module: Module, args: tuple[Any, ...], *, block_dim: int | None = None, timeout: float = 120.0, seed_outputs: bool = False,
             check_gm: bool = False, processes: bool | None = None) -> SimResult:
    """Run a lowered module through the interpreter with tracing, then schedule the trace. ``processes`` as in
    :meth:`Machine.run`: by default every core group traces in its own forked process (the trace is per lane and
    deterministic either way); the schedule is one process."""
    from ...runtime.launch_config import launch_block_dim
    from .interp import Machine
    from .launch import bind_arguments, entry_function

    block_dim = launch_block_dim(module, block_dim, "pipesim")
    model = load_model(module.device or "950")
    tracer = Tracer(model)
    bound = bind_arguments(module, args, seed_outputs=seed_outputs)
    machine = Machine(module, timeout=timeout)
    machine.tracer = tracer
    machine.run(bound, block_dim=block_dim, processes=processes)
    sched = Scheduler(tracer, model, check_gm=check_gm)
    for lane, tasks in sched.lanes.items():  # in launch order: the flag timeline's hazards come out in its order
        for t in tasks:
            if t.op.opcode == "sync.event":
                x = t.op.results[0]
                depth = getattr(x.type, "depth", 1) or 1
                preset = t.op.attrs.get("preset", 0)
                preset = depth if preset is True else int(preset or 0)
                ids = t.op.attrs.get("ids")
                if ids is None and depth == 1 and getattr(x.type, 'id', None) is not None:
                    ids = [x.type.id]
                channel = (getattr(x.type, 'set_pipe', None), getattr(x.type, 'wait_pipe', None))
                sched.declare_event(lane, x.name, depth, preset, channel=channel, ids=ids)
    sched.run()
    outputs = [bound[o.name] for o in entry_function(module).attrs.get("outputs", []) if isinstance(o, Value)]
    return SimResult(outputs, sched.report(), sched)


__all__ = ["Access", "Deadlock", "PIPES", "Scheduler", "SimResult", "Task", "Tracer", "simulate"]
