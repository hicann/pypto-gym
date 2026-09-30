# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""``deps``: the memory dependency graph of a function (RFC-0005 §2.1).

Every op's accesses come from the registry (operand / value-attribute access sets) and the view
geometry (:func:`view_of`): an :class:`Access` is a root tensor, a rectangular window in the root's
coordinates, an optional slot-buffer index, the access kind and the pipe. Two accesses conflict when
the roots match, the slots may coincide and the windows may overlap (exactly when static, conservatively
when a bound is a run-time scalar).

Ordering comes from the structured control flow: inside a block, program order; inside a ``cf.for``,
the body is also compared with itself one or more iterations later (loop-carried edges with a
*distance*). Slot buffers indexed by a counter cell that advances by a constant per iteration rotate
through their slots, so the same buffer conflicts with itself only at the distance that brings the
index back to the same slot — the event depth the old ``DEvent`` / ``TEvent`` / ``QEvent`` encoded.

The graph also answers *is B already ordered after A?* through per-pipe vector clocks (pipes execute
in order; events add cross-pipe edges), which is the transitive reduction ``autosync`` needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from ..ir import REGISTRY, Block, Function, Ident, Literal, Op, Value
from ..ir.types import BufType, CellType, EventType, MemType
from .util import Defs, View, view_of

READ, WRITE = "read", "write"
SCALAR_PIPE = "S"
ALL_PIPES = ("S", "MTE1", "MTE2", "MTE3", "M", "V", "FIX")


# ------------------------------------------------------------------------------ accesses


@dataclass(frozen=True)
class Access:
    op: Op
    root: str
    space: str
    kind: str  # read | write
    lo: tuple[Any, ...]  # per root dim: int or Value (or a deferred marker tuple)
    hi: tuple[Any, ...]  # exclusive
    slot: Any  # None, int or Value
    pipe: str
    whole: bool = False  # the whole root, conservatively
    c0: int = 1  # elements of the view's dtype per 32-byte block (DMA / vector store granularity along the last dim)
    #: The dimensions ``lo`` / ``hi`` are expressed in when a ``mem.reshape`` re-described the root
    #: (``View.dims``), with the windows it re-addressed (``View.origin``); None means the root's own
    #: coordinates. Two windows are comparable only inside one coordinate system with a static
    #: origin, and a reshape is not a loss of precision by itself - the offsets stay exact.
    coords: tuple[Any, ...] | None = None

    def describe(self) -> str:
        return f"{self.kind} %{self.root}" + ("" if self.slot is None else f"[slot {self.slot}]")


def op_pipe(op: Op) -> str:
    p = op.attrs.get("pipe")
    if p is not None:
        return str(p)
    if op.opcode in ("sync.set", "sync.set_all", "sync.wait", "sync.release") and op.operands:
        t = getattr(op.operands[0], "type", None)  # a user event without the attribute: the pipes it was declared with
        if isinstance(t, EventType):
            return t.set_pipe if op.opcode in ("sync.set", "sync.set_all") else t.wait_pipe
    spec = REGISTRY.get(op.opcode)
    return spec.pipe or SCALAR_PIPE


def _mem_value(v: Any) -> bool:
    return isinstance(v, Value) and isinstance(v.type, (MemType, BufType))


def _window(view: View) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    lo = tuple(view.offsets)
    hi = []
    for o, e in zip(view.offsets, view.extents, strict=True):
        if isinstance(o, int) and isinstance(e, int):
            hi.append(o + e)
        else:
            hi.append(("+", o, e))
    return lo, tuple(hi)


def accesses_of(op: Op, defs: Defs, *, pipe: str | None = None) -> list[Access]:
    """The memory accesses of one op (its own operands and value-attributes; regions excluded)."""
    spec = REGISTRY.get(op.opcode)
    pipe = pipe or op_pipe(op)
    out: list[Access] = []

    def add(v: Value, access: str) -> None:
        if not _mem_value(v):
            return
        try:
            view = view_of(v, defs)
        except TypeError:
            return
        whole = False
        if op.opcode == "dma.l1_to_l0" and "m_copy" in op.attrs:
            if v == op.operands[1]:
                view = replace(view, extents=(op.attrs["m_copy"], op.attrs["n_dst"]))
            else:
                whole = True  # the physical L0 tile may exceed the logical re-view
        if view.root_op is not None and view.root_op.opcode == "mem.view":
            # a strided re-description (RFC-0010) aliases its base: anchor the access on the base's
            # root and treat it as the whole tensor - the safe default of RFC-0010 §6
            view = view_of(view.root_op.operands[0], defs)  # type: ignore[arg-type]
            whole = True
        kinds = [READ, WRITE] if access == "readwrite" else [access]
        coords = (view.dims, view.origin) if view.reshaped else None
        lo, hi = _window(view)
        c0 = 64 if view.dtype.bits < 8 else 32 * 8 // view.dtype.bits
        for kind in kinds:
            out.append(Access(op, view.root.name, view.root.type.space if isinstance(view.root.type, MemType) else view.root.type.elem.space,
                              kind, lo, hi, view.slot, pipe, whole=whole, c0=c0, coords=coords))

    if op.opcode == "cf.call":
        listed = False
        for key, kind in (("read", READ), ("write", WRITE)):
            for v in op.attrs.get(key, []) or []:
                if isinstance(v, Value):
                    add(v, kind)
                    listed = True
        if not listed:  # no access lists on the call: every UB view handed to the vf counts as read and written
            for v in op.operands[1:]:
                if isinstance(v, Value) and _mem_value(v):
                    add(v, "readwrite")
        return out
    if op.opcode == "simt.launch":
        for v in op.operands:
            if isinstance(v, Value) and _mem_value(v):
                add(v, "readwrite")
        return out
    ospecs = list(spec.operands)
    for i, v in enumerate(op.operands):
        s = ospecs[min(i, len(ospecs) - 1)] if ospecs else None
        if s is None or s.access == "none" or not isinstance(v, Value):
            continue
        add(v, s.access)
    for a in spec.attrs:
        if a.access != "none" and a.name in op.attrs:
            v = op.attrs[a.name]
            if isinstance(v, Value):
                add(v, a.access)
    return out


# ------------------------------------------------------------------------------ overlap


def _static(x: Any) -> int | None:
    return x if isinstance(x, int) else None


def _same_symbol(a: Any, b: Any) -> bool:
    if isinstance(a, Value) and isinstance(b, Value):
        return a.name == b.name
    if isinstance(a, tuple) and isinstance(b, tuple):
        return a[0] == b[0] and all(_same_symbol(x, y) or (isinstance(x, int) and isinstance(y, int) and x == y) for x, y in zip(a[1:], b[1:], strict=True))
    return False


def intervals_may_overlap(lo1: Any, hi1: Any, lo2: Any, hi2: Any) -> bool:
    """False only when the two half-open intervals are provably disjoint."""
    a, b, c, d = _static(lo1), _static(hi1), _static(lo2), _static(hi2)
    if a is not None and b is not None and c is not None and d is not None:
        return not (b <= c or d <= a)
    # symbolic: identical windows overlap; a static window entirely below a symbolic lower bound is unknown -> overlap
    if _same_symbol(lo1, lo2) and _same_symbol(hi1, hi2):
        return True
    return True


def _round_out(lo: Any, hi: Any, c0: int) -> tuple[Any, Any]:
    """A static interval widened to whole 32-byte blocks: what a DMA burst or a vector store really touches."""
    a, b = _static(lo), _static(hi)
    if a is None or b is None or c0 <= 1:
        return lo, hi
    return a // c0 * c0, -(-b // c0) * c0


def windows_may_overlap(x: Access, y: Access) -> bool:
    if x.whole or y.whole:
        return True
    if x.coords != y.coords:
        return True  # one window counts in a reshaped shape and the other does not: not comparable
    if x.coords is not None and not all(isinstance(o, int) for offsets, _ in x.coords[1] for o in offsets):
        return True  # a reshape of a window at a runtime offset may start elsewhere on another execution
    if len(x.lo) != len(y.lo):
        return True
    n = len(x.lo)
    for i, (a, b, c, d) in enumerate(zip(x.lo, x.hi, y.lo, y.hi, strict=True)):
        if i == n - 1:  # the contiguous dim: compare whole blocks
            a, b = _round_out(a, b, x.c0)
            c, d = _round_out(c, d, y.c0)
        if not intervals_may_overlap(a, b, c, d):
            return False
    return True


# ------------------------------------------------------------------------------ slot counters


@dataclass(frozen=True)
class SlotIndex:
    """A slot-buffer index as ``cell + k`` (``cell`` None: a static index)."""

    cell: str | None
    k: int

    @staticmethod
    def of(x: Any, defs: Defs) -> SlotIndex | None:
        if isinstance(x, int):
            return SlotIndex(None, x)
        if isinstance(x, Value):
            if isinstance(x.type, CellType):
                return SlotIndex(x.name, 0)
            op = defs.op(x)
            if op is not None and op.opcode in ("scalar.add", "scalar.sub") and len(op.operands) == 2:
                a, b = op.operands
                if isinstance(a, Value) and isinstance(a.type, CellType) and isinstance(b, Literal) and isinstance(b.value, int):
                    return SlotIndex(a.name, b.value if op.opcode == "scalar.add" else -b.value)
                if op.opcode == "scalar.add" and isinstance(b, Value) and isinstance(b.type, CellType) and isinstance(a, Literal):
                    return SlotIndex(b.name, int(a.value))
        return None


def counter_steps(loop: Op) -> dict[str, int]:
    """Cells the loop body advances by a constant exactly once per iteration: ``cell -> step``."""
    steps: dict[str, int] = {}
    sets: dict[str, list[Op]] = {}
    for op in loop.regions[0].walk():
        if op.opcode == "scalar.set":
            cell = op.operands[0]
            if isinstance(cell, Value):
                sets.setdefault(cell.name, []).append(op)
    body_top = list(loop.regions[0].ops)  # sets inside nested control flow do not advance uniformly
    for name, ops in sets.items():
        if len(ops) != 1 or ops[0] not in body_top:
            continue
        src = ops[0].operands[1]
        if not isinstance(src, Value):
            continue
        k = SlotIndex.of(src, _DEFS_STUB)  # resolved by the caller through set_defs()
        if k is not None and k.cell == name and k.k != 0:
            steps[name] = k.k
    return steps


class _DefsStub:
    """Filled by :func:`set_defs` so :func:`counter_steps` can resolve ``scalar.add`` operands."""

    defs: Defs | None = None

    def op(self, v: Value) -> Op | None:
        return None if self.defs is None else self.defs.op(v)


_DEFS_STUB = _DefsStub()


def set_defs(defs: Defs) -> None:
    _DEFS_STUB.defs = defs


def slot_distance(x: Access, y: Access, slots: int | None, steps: dict[str, int], defs: Defs) -> tuple[bool, int | None]:
    """(may_alias_same_iteration, carried_distance): whether the two slot indices coincide in one iteration, and the
    smallest positive number of iterations after which x's slot is y's slot (None: never / unknown -> conservative 1)."""
    if x.slot is None and y.slot is None:
        return True, 1
    sx, sy = SlotIndex.of(x.slot, defs), SlotIndex.of(y.slot, defs)
    if sx is None or sy is None or slots is None:
        return True, 1  # unknown index expression: conservative
    if sx.cell is None and sy.cell is None:  # fixed slots: the same one every iteration, so it comes round after one
        return sx.k % slots == sy.k % slots, (1 if sx.k % slots == sy.k % slots else None)
    if sx.cell != sy.cell:
        return True, 1
    step = steps.get(sx.cell)
    if step is None:  # the counter does not advance uniformly: same iteration may alias; next iterations unknown
        return sx.k % slots == sy.k % slots, 1
    same = sx.k % slots == sy.k % slots
    # x at iteration i uses slot (c + i*step + kx) % slots; y at iteration i + d uses (c + (i + d)*step + ky) % slots
    for d in range(1, slots + 1):
        if (d * step + sy.k - sx.k) % slots == 0:
            return same, d
    return same, None


# ------------------------------------------------------------------------------ the graph


@dataclass
class Node:
    op: Op
    index: int  # position in the region's flattened program order
    pipe: str
    path: tuple[tuple[Op, int], ...]  # enclosing (op, region index) chain inside the region
    accesses: list[Access] = field(default_factory=list)


@dataclass(frozen=True)
class Edge:
    kind: str  # RAW | WAR | WAW
    src: Node  # must complete first
    dst: Node
    distance: int  # 0: same iteration of the innermost common loop; d > 0: dst runs d iterations later
    loop: Op | None  # the loop the distance refers to
    root: str


def _kind(x: Access, y: Access) -> str | None:
    if x.kind == WRITE and y.kind == READ:
        return "RAW"
    if x.kind == READ and y.kind == WRITE:
        return "WAR"
    if x.kind == WRITE and y.kind == WRITE:
        return "WAW"
    return None


def _static_int(x: Any) -> int | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, Literal) and isinstance(x.value, int) and not isinstance(x.value, bool):
        return x.value
    return None


def _trip_at_least_one(loop: Op) -> bool:
    """True when the ``cf.for`` provably runs its body at least once (static bounds). A dynamic bound may be an
    empty range — knowledge learned inside the body must then not escape the loop (mirror of the cf.if arms)."""
    if len(loop.operands) < 2:
        return False
    lo, hi = _static_int(loop.operands[0]), _static_int(loop.operands[1])
    step = _static_int(loop.operands[2]) if len(loop.operands) > 2 else 1
    if lo is None or hi is None:
        return False
    if step is None:
        step = 1
    return hi > lo if step > 0 else hi < lo


def _barrier_pipe(op: Op) -> str:
    p = op.attrs.get("pipe", "ALL")
    return p.name if isinstance(p, Ident) else str(p)


def exclusive_arms(a: Node, b: Node) -> bool:
    """True when the two nodes sit in different arms of the same ``cf.if``: they never both run in one pass."""
    for (pa, ia), (pb, ib) in zip(a.path, b.path, strict=False):
        if pa is not pb:
            return False
        if ia != ib:
            return pa.opcode == "cf.if"
    return False


class DepGraph:
    """Nodes in program order and the conflict edges of one block (with its nested regions)."""

    def __init__(self, fn: Function, region: Block, defs: Defs, *, track_gm: bool = True) -> None:
        self.fn = fn
        self.defs = defs
        set_defs(defs)
        self.track_gm = track_gm
        self.nodes: list[Node] = []
        self.by_id: dict[int, Node] = {}
        self.slots: dict[str, int] = {}  # root name -> slots of a buf allocation
        self._steps: dict[int, dict[str, int]] = {}  # loop op id -> counter steps
        self._flatten(region, ())
        self.edges: list[Edge] = []
        self._build_edges()

    # -- flattening ----------------------------------------------------------------------

    def _flatten(self, block: Block, path: tuple[tuple[Op, int], ...]) -> None:
        for op in block.ops:
            node = Node(op, len(self.nodes), op_pipe(op), path)
            node.accesses = [a for a in accesses_of(op, self.defs) if self.track_gm or a.space not in ("gm", "ws")]
            self.nodes.append(node)
            if op.id is not None:
                self.by_id[op.id] = node
            if op.opcode == "cf.for":
                self._steps[op.id or -1] = counter_steps(op)
            for i, region in enumerate(op.regions):
                self._flatten(region, (*path, (op, i)))
        for n in self.nodes:
            for a in n.accesses:
                if a.root not in self.slots:
                    root_op = self.defs.by_name.get(a.root, (None, None))[0]
                    if root_op is not None and root_op.results and isinstance(root_op.results[0].type, BufType):
                        self.slots[a.root] = root_op.results[0].type.slots

    # -- structure queries ----------------------------------------------------------------

    @staticmethod
    def common_loops(a: Node, b: Node) -> list[Op]:
        loops = []
        for (pa, ia), (pb, ib) in zip(a.path, b.path, strict=False):
            if pa is not pb or ia != ib:
                break
            if pa.opcode == "cf.for":
                loops.append(pa)
        return loops

    def innermost_common_loop(self, a: Node, b: Node) -> Op | None:
        loops = self.common_loops(a, b)
        return loops[-1] if loops else None

    # -- edges ----------------------------------------------------------------------------

    def _build_edges(self) -> None:
        by_root: dict[str, list[tuple[Node, Access]]] = {}
        for n in self.nodes:
            for a in n.accesses:
                by_root.setdefault(a.root, []).append((n, a))
        for root, items in by_root.items():
            slots = self.slots.get(root)
            for i, (na, aa) in enumerate(items):
                for nb, ab in items[i + 1:]:
                    if na is nb:
                        continue
                    self._consider(na, aa, nb, ab, slots)
                # loop-carried conflicts of an access with itself / earlier ops in the same loop body
            for i, (na, aa) in enumerate(items):
                for nb, ab in items[: i + 1]:
                    loop = self.innermost_common_loop(na, nb)
                    if loop is None:
                        continue
                    self._consider_carried(na, aa, nb, ab, slots, loop)

    def _consider(self, na: Node, aa: Access, nb: Node, ab: Access, slots: int | None) -> None:
        """Program-order pair: ``na`` before ``nb`` in the flattened order."""
        kind = _kind(aa, ab)
        if kind is None or not windows_may_overlap(aa, ab):
            return
        loop = self.innermost_common_loop(na, nb)
        steps = self._steps.get(loop.id or -1, {}) if loop is not None else {}
        same, dist = slot_distance(aa, ab, slots, steps, self.defs)
        if same and not exclusive_arms(na, nb):
            self.edges.append(Edge(kind, na, nb, 0, None, aa.root))
        if loop is not None and dist is not None and dist > 0:
            # na (iteration i) -> nb (iteration i + dist): only when the later iteration's nb still conflicts
            self.edges.append(Edge(kind, na, nb, dist, loop, aa.root))

    def _consider_carried(self, na: Node, aa: Access, nb: Node, ab: Access, slots: int | None, loop: Op) -> None:
        """``nb`` at or before ``na`` in the body: nb of a later iteration may conflict with na of this one."""
        kind = _kind(aa, ab)
        if kind is None or not windows_may_overlap(aa, ab):
            return
        steps = self._steps.get(loop.id or -1, {})
        same, dist = slot_distance(aa, ab, slots, steps, self.defs)
        if dist is None or dist <= 0:
            return
        if na is nb and kind == "WAW" and dist == 1 and aa.pipe == ab.pipe:
            return  # an op rewriting its own window next iteration on the same pipe is ordered by the pipe
        self.edges.append(Edge(kind, na, nb, dist, loop, aa.root))

    # -- happens-before through pipes and events ------------------------------------------

    def vector_clocks(self, sync_edges: list[tuple[Node, Node]]) -> dict[int, dict[str, int]]:
        """For every node (by index): the latest index per pipe known to complete before it starts, given program
        order on each pipe plus the given synchronisation edges (set after src, wait before dst)."""
        last_on_pipe: dict[str, int] = {}
        clocks: dict[int, dict[str, int]] = {}
        incoming: dict[int, list[Node]] = {}
        for s, d in sync_edges:
            incoming.setdefault(d.index, []).append(s)
        # What an op learns inside a branch arm (pipe order after the arm's ops, waits in the arm) holds only when
        # the arm runs: every arm starts from the state before the ``cf.if``, and so does whatever follows it.
        # A ``cf.for`` whose bounds are not provably non-empty is the same shape with an implicit empty arm: what
        # the body taught must not survive the exit (the body may run zero times), though the body itself — every
        # iteration after the first — keeps it.
        frames: list[tuple[Op, dict[str, int]]] = []
        previous: tuple[tuple[Op, int], ...] = ()
        for n in self.nodes:
            common = 0
            for (pa, ia), (pb, ib) in zip(previous, n.path, strict=False):
                if pa is not pb or ia != ib:
                    break
                common += 1
            for op, _ in reversed(previous[common:]):
                if op.opcode == "cf.if":
                    _, last_on_pipe = frames.pop()
                    last_on_pipe = dict(last_on_pipe)
                elif op.opcode == "cf.for":
                    _, saved = frames.pop()
                    if not _trip_at_least_one(op):
                        last_on_pipe = dict(saved)
            for op, _ in n.path[common:]:
                if op.opcode in ("cf.if", "cf.for"):
                    frames.append((op, dict(last_on_pipe)))
            previous = n.path
            clock: dict[str, int] = {}
            if n.op.opcode == "sync.barrier" and _barrier_pipe(n.op) == "ALL":
                # pipe_barrier(PIPE_ALL): everything issued before completes, everything after starts later
                for p, i in last_on_pipe.items():
                    clock[p] = i
                    for q, j in clocks[i].items():
                        clock[q] = max(clock.get(q, -1), j)
                clocks[n.index] = clock
                for p in ALL_PIPES:  # every pipe, seen or not: whatever comes next on any pipe follows the barrier
                    last_on_pipe[p] = n.index
                continue
            prev = last_on_pipe.get(n.pipe)
            if prev is not None:
                clock.update(clocks[prev])
                clock[n.pipe] = prev
            for s in incoming.get(n.index, []):
                for p, i in self._transferred(s, n, clocks).items():
                    clock[p] = max(clock.get(p, -1), i)
            clocks[n.index] = clock
            last_on_pipe[n.pipe] = n.index
        return clocks

    def _transferred(self, s: Node, d: Node, clocks: dict[int, dict[str, int]]) -> dict[str, int]:
        """What a set after producer ``s`` tells consumer ``d``: ``s`` and everything ``s`` knew — except that inside a
        branch arm ``d`` is outside of, ``s`` may not have run: the set behind the branch then certifies whichever arm
        ran, so the knowledge of other pipes is intersected with that of the last op on the producer's pipe in every
        other arm (an arm without such an op leaves nothing of it). On the producer's own pipe the set certifies more
        than ``s``: it follows the whole branch / loop at the consumer's level, so every op of that pipe up to its end
        that ran is complete (a store after an if / else that produces in both arms is ordered after both)."""
        know = dict(clocks[s.index])
        common = 0
        for (pa, ia), (pb, ib) in zip(s.path, d.path, strict=False):
            if pa is not pb or ia != ib:
                break
            common += 1
        if any(op.opcode == "cf.for" and not _trip_at_least_one(op) for op, _ in s.path[common:]):
            return {}  # the producer sits in a loop the consumer is outside of, and the body may run zero times:
            # the set (lifted to the join) then certifies nothing the consumer's own pipe order does not already know
        end = s.index  # the last node inside the op the set follows (a user's set sits where it is: no more)
        if len(s.path) > common and s.op.opcode not in ("sync.set", "sync.set_all"):
            anchor = s.path[common][0]
            while (end + 1 < len(self.nodes) and len(self.nodes[end + 1].path) > common
                   and self.nodes[end + 1].path[common][0] is anchor):
                end += 1
        for depth in range(len(s.path) - 1, common - 1, -1):
            branch, arm = s.path[depth]
            if branch.opcode != "cf.if":
                continue
            for k in range(len(branch.regions)):
                if k == arm:
                    continue
                last = None
                for m in self.nodes:
                    if (m.pipe == s.pipe and len(m.path) > depth and m.path[depth][0] is branch and m.path[depth][1] == k
                            and all(x is y and i == j for (x, i), (y, j) in zip(m.path[:depth], s.path[:depth], strict=True))):
                        last = m
                if last is None or last.index not in clocks:
                    return {s.pipe: end}
                other = dict(clocks[last.index])
                know = {p: min(i, other[p]) for p, i in know.items() if p in other}
            if len(branch.regions) < 2:
                return {s.pipe: end}
        know[s.pipe] = max(know.get(s.pipe, -1), end)
        return know

    def ordered(self, clocks: dict[int, dict[str, int]], a: Node, b: Node) -> bool:
        """True when ``a`` is known to complete before ``b`` starts."""
        if a.pipe == b.pipe and a.index < b.index:
            return True
        return clocks[b.index].get(a.pipe, -1) >= a.index


__all__ = ["Access", "DepGraph", "Edge", "Node", "SlotIndex", "accesses_of", "counter_steps", "exclusive_arms", "op_pipe", "slot_distance",
           "windows_may_overlap"]
