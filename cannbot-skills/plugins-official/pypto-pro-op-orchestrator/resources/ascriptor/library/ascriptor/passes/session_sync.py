# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""``session_sync``: paired ready/valid slot sessions for the A2 family (RFC-0005 §5).

A session is one channel of one side at one capacity. Its producer pipe hands buffers to its
consumer pipe through two events of equal depth ``W``: ``ready`` forward, ``valid`` reverse with
``W`` preset credits. One *window* of a session is ``valid.wait`` (take a credit), the producer's
work, ``ready.set`` / ``ready.wait`` (publish), the consumer's work, ``valid.set`` (return the
credit). Consecutive work of one pipe merges into the window it is already in, so the flags a
kernel needs follow its role switches rather than its dependency edges, and the depth follows the
declared physical slots rather than a run-ahead replay: a producer takes a credit before it can
publish, so ``W`` outstanding publications is exactly what the reverse window admits (M10-083).

A session holds one such ledger per *depth* of its scope that orders role switches (``Level``): a
producing loop followed by a consuming loop gets a window around each loop at the outer depth,
while the publication inside each loop belongs to the inner depth, whose own credit is what keeps
that publication from setting a flag twice.

Windows never cross a block boundary, so every path through a branch or loop leaves the tokens as
it found them and a skipped arm emits nothing at all -- no cell, guard mirror or acknowledgement
is needed. The dependency graph judges the result instead of planning it: every cross-pipe edge
``deps`` reports must be covered by a window, and an uncovered edge is a located error (§5.9).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any

from ..ir import Block, Function, Ident, Literal, Module, Op, Value
from ..ir.builder import Rewriter
from ..ir.sync_rules import FLAG_IDS
from ..ir.types import CellType, EventType
from .deps import READ, WRITE, DepGraph
from .manager import PassContext, PassError
from .util import SET_PIPES, Defs, function_names, op_side

PASS = "session_sync"
LEDGER_POLICIES = ("capacity", "channel", "allocation")
TERMINATORS = ("cf.return", "cf.break", "cf.continue")
LOCAL_SPACES = {"ub", "l1", "l0a", "l0b", "l0c", "bt", "l0amx", "l0bmx"}
MAX_DEPTH = FLAG_IDS  # a ledger of depth W holds W of the ids its channel has, so W never exceeds them

CLOSED, OPEN, PUBLISHED = "closed", "open", "published"
VALID_WAIT, READY_SET, READY_WAIT, VALID_SET = "valid.wait", "ready.set", "ready.wait", "valid.set"


def ledger_policy(ctx: PassContext) -> str:
    policy = ctx.option("session_ledgers", "capacity")
    if policy not in LEDGER_POLICIES:
        raise PassError(PASS, f"session_ledgers must be one of {', '.join(LEDGER_POLICIES)}")
    return policy


@dataclass(frozen=True)
class Slot:
    """A slot index as ``(cell + offset) % modulus``; ``cell`` None is a fixed index."""

    cell: str | None
    offset: int
    modulus: int | None


@dataclass
class Member:
    """One allocation a session protects."""

    root: str
    slots: int
    cap: int  # the slots a window may keep in flight: `sync_depth` where the author declared one


@dataclass
class Level:
    """One nesting depth of a session's scope, with the windows the items of a block at that depth
    open and the ledger those windows use.

    A block with a role-bearing access of its own runs the protocol at that depth; a block whose
    every role-bearing item hands over inside one of its subtrees orders nothing itself and is
    transparent, so those subtrees keep this depth's ledger and number their windows in one
    sequence however deeply each of them sits. A hand-off written as a producing loop followed by a
    consuming loop therefore gets a window around each loop at the outer depth, while the
    publication inside the producing loop belongs to the inner depth -- the credit of one depth is
    never the credit of another, which is what keeps a nested hand-off from waiting on itself.

    ``rings`` is how many windows of *this* depth pass before a physical slot is written again: two
    windows share a slot only when their counter values differ by a multiple of the ring, so a
    credit count at or below it can never overwrite storage a reader still holds."""

    index: int
    windows: dict[int, int] = field(default_factory=dict)  # access op id -> window number
    published: set[int] = field(default_factory=set)  # consumer ops that follow their window's publication
    marks: list[tuple[int, str, str]] = field(default_factory=list)  # (anchor op id, before|after, kind)
    attribution: list[tuple[int, Op]] = field(default_factory=list)  # (window, op) for the slot proof
    rings: dict[str, int] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    total: int = 0  # windows one execution of this depth opens
    leading: bool = False  # a consumer phase opens the block: its hand-back is the first thing emitted
    publishes: bool = False  # a producer access of this depth publishes here, rather than inside a subtree
    depth: int = 1
    ready: Value | None = None
    valid: Value | None = None

    def capacity(self, members: dict[str, Member]) -> int:
        if self.leading:
            return 1  # the next producer must wait for that hand-back, so no credit may be spare
        ring_depths = [min(m.cap, self.rings.get(root, 1)) for root, m in members.items()]
        return max(1, min(ring_depths + [MAX_DEPTH]))

    def describe(self, members: dict[str, Member]) -> str:
        parts = []
        for root, member in sorted(members.items()):
            text = f"%{root} {member.slots} slot(s)"
            if member.cap != member.slots:
                text += f" capped to {member.cap}"
            parts.append(text + f", ring {self.rings.get(root, 1)}: "
                         + self.reasons.get(root, "a fixed or unreadable slot index admits one window"))
        return "; ".join(parts)


@dataclass
class Session:
    side: str
    producer: str
    consumer: str
    key: tuple[Any, ...]
    members: dict[str, Member] = field(default_factory=dict)
    roles: dict[int, str] = field(default_factory=dict)  # op id -> "P" | "C" (direct accesses)
    scope: tuple[tuple[Op, int], ...] = ()
    levels: dict[int, Level] = field(default_factory=dict)  # nesting depth below the scope -> its windows

    @property
    def name(self) -> str:
        return f"{self.side}:{self.producer}->{self.consumer}"

    @property
    def credits(self) -> bool:
        """The scalar pipe cannot set a flag, so a session it consumes has no reverse channel. It
        needs none: the scalar stream issues every other pipe's work, so it cannot reach the next
        window's publication before it has taken this one's -- one token is outstanding at most."""
        return self.consumer != "S"

    def ordered(self) -> list[Level]:
        return [self.levels[index] for index in sorted(self.levels)]

    def announced(self) -> set[int]:
        """Consumer accesses a publication of this session precedes inside their own block, at
        whatever depth it was made: a publication issued inside a subtree follows, on the producer
        pipe, every producer access of the enclosing window that precedes that subtree."""
        return {op for level in self.levels.values() for op in level.published}


def _literal_trips(loop: Op) -> int | None:
    values = []
    for operand in loop.operands[:3]:
        value = operand.value if isinstance(operand, Literal) else operand
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        values.append(value)
    if len(values) < 2:
        return None
    lo, hi = values[0], values[1]
    step = values[2] if len(values) > 2 else 1
    if step == 0:
        return None
    return max(0, (hi - lo + step - (1 if step > 0 else -1)) // step)


def _common_prefix(paths: list[tuple[tuple[Op, int], ...]]) -> tuple[tuple[Op, int], ...]:
    if not paths:
        return ()
    common = paths[0]
    for path in paths[1:]:
        keep = 0
        for (a, i), (b, j) in zip(common, path, strict=False):
            if a is not b or i != j:
                break
            keep += 1
        common = common[:keep]
    return common


def _splice(block: Block, before: dict[int, list[Op]], after: dict[int, list[Op]]) -> Block:
    out: list[Op] = []
    for op in block.ops:
        if op.regions:
            op = replace(op, regions=tuple(_splice(region, before, after) for region in op.regions))
        out.extend(before.get(op.id, ()))  # type: ignore[arg-type]
        out.append(op)
        out.extend(after.get(op.id, ()))  # type: ignore[arg-type]
    return Block(tuple(out))


class _Planner:
    """Plans every session of one function and rewrites its body."""

    def __init__(self, fn: Function, body: Block, inside: set[int], module: Module, rw: Rewriter, ctx: PassContext) -> None:
        self.fn = fn
        self.body = body
        self.inside = inside
        self.rw = rw
        self.ctx = ctx
        self.names = function_names(fn)
        self.defs = Defs(module)
        self.graph = DepGraph(fn, body, self.defs, track_gm=bool(ctx.option("autosync_gm", False)))
        self.allocs = {op.results[0].name: op for op in body.walk() if op.opcode == "mem.alloc" and op.results}
        self.cells = {op.results[0].name: op for op in body.walk() if op.opcode == "scalar.cell" and op.results}
        self.spaces = {a.root: a.space for n in self.graph.nodes for a in n.accesses}
        self.sessions: list[Session] = []
        self._roles_cache: dict[tuple[int, int], frozenset[str]] = {}
        self._escape_cache: dict[int, frozenset[str]] = {}

    # -- slot indices ---------------------------------------------------------------------

    def slot_of(self, index: Any) -> Slot | None:
        """Resolve a slot index to ``(cell + offset) % modulus``. The corpus spells the modulo of a
        monotonic counter as ``cell & (2**n - 1)`` or ``cell % n`` to keep signed division out of
        the scalar stack, so both are read here; anything else is unreadable."""
        if isinstance(index, Literal):
            index = index.value
        if isinstance(index, int) and not isinstance(index, bool):
            return Slot(None, index, None)
        if not isinstance(index, Value):
            return None
        if isinstance(index.type, CellType):
            return Slot(index.name, 0, None)
        op = self.defs.op(index)
        if op is None or len(op.operands) != 2:
            return None
        left, right = op.operands
        value = right.value if isinstance(right, Literal) else right
        if not isinstance(value, int) or isinstance(value, bool):
            return None
        inner = self.slot_of(left)
        if inner is None or inner.cell is None:
            return None
        if op.opcode == "scalar.add":
            return replace(inner, offset=inner.offset + value)
        if op.opcode == "scalar.sub":
            return replace(inner, offset=inner.offset - value)
        if op.opcode == "scalar.mod" and value > 0:
            return replace(inner, modulus=min(value, inner.modulus or value))
        if op.opcode == "scalar.and" and value > 0 and not (value + 1) & value:  # a 2**n - 1 mask
            return replace(inner, modulus=min(value + 1, inner.modulus or value + 1))
        return None

    def advances(self, op: Op, cell: str) -> int | None:
        """How many times one execution of ``op`` (and its subtree) advances ``cell`` by one.
        None: the count is not a literal, so no ring can be derived from it."""
        total = 0
        if op.opcode == "scalar.set" and op.operands and getattr(op.operands[0], "name", None) == cell:
            source = op.operands[1]
            if not isinstance(source, Value):
                return None
            definition = self.defs.op(source)
            if definition is None or definition.opcode != "scalar.add":
                return None
            base, step = definition.operands
            step = step.value if isinstance(step, Literal) else step
            if getattr(base, "name", None) != cell or step != 1:
                return None
            total += 1
        counts = []
        for region in op.regions:
            inner = 0
            for child in region.ops:
                count = self.advances(child, cell)
                if count is None:
                    return None
                inner += count
            counts.append(inner)
        if counts:
            if op.opcode == "cf.if":
                total += max(counts)  # the arms are alternatives: the widest one bounds a window
            else:
                trips = _literal_trips(op) if op.opcode == "cf.for" else 1
                body = sum(counts)
                if body and trips is None:
                    return None
                total += body * (trips or 0)
        return total

    # -- members and sessions -------------------------------------------------------------

    def member(self, root: str) -> Member:
        slots = self.graph.slots.get(root, 1)
        alloc = self.allocs.get(root)
        cap = slots
        if alloc is not None and "sync_depth" in alloc.attrs:
            cap = min(slots, int(alloc.attrs["sync_depth"]))
        return Member(root, slots, cap)

    def managed_edges(self) -> list[Any]:
        out = []
        for edge in self.graph.edges:
            if edge.src.op.id not in self.inside or edge.dst.op.id not in self.inside:
                continue
            if edge.src.pipe == edge.dst.pipe:
                continue  # one pipe issues in order; overlapping writes stay the author's (RFC-0005 §2.2)
            if edge.src.pipe not in SET_PIPES:
                # whichever role the pair ends up in, the earlier op is the one that signals: a
                # publication forward when it wrote, a hand-back when it read. The scalar pipe can
                # set no flag either way, so this pair stays the author's (RFC-0005 §5.8).
                self.ctx.explain.note(
                    f"{edge.kind} on %{edge.root}: #{edge.src.op.id} {edge.src.op.loc} runs on the {edge.src.pipe} "
                    f"pipe, which cannot set a flag for #{edge.dst.op.id} ({edge.dst.pipe}); order it with sync.barrier",
                    kind="warning", ops=(edge.src.op.id, edge.dst.op.id))
                continue
            if self.side_of(edge) is not None and self.spaces.get(edge.root) in LOCAL_SPACES:
                out.append(edge)
        return out

    def side_of(self, edge: Any) -> str | None:
        """The side a pair belongs to. Scalar ops carry no side of their own -- they issue for both
        -- so one known side names the pair; two different ones are crosssync's question."""
        side, other = op_side(edge.src.op), op_side(edge.dst.op)
        if side is not None and other is not None and side != other:
            return None
        return side or other

    def orient(self, edges: list[Any]) -> dict[tuple[str, str, frozenset[str]], tuple[str, str]]:
        """Which pipe of a pair produces for the other. The data flow decides: a RAW names its
        writer the producer, so an in-place consumer (``muls(x, x, s)``, whose write also raises a
        WAW against the next round's loader) hands its storage back through the same session's
        reverse credit instead of opening a second, opposite one."""
        orientation: dict[tuple[str, str, frozenset[str]], tuple[str, str]] = {}
        for kinds in (("RAW",), ("WAR", "WAW")):
            for edge in edges:
                if edge.kind not in kinds:
                    continue
                side = self.side_of(edge)
                assert side is not None
                key = (side, edge.root, frozenset({edge.src.pipe, edge.dst.pipe}))
                want = ((edge.dst.pipe, edge.src.pipe) if edge.kind == "WAR" else (edge.src.pipe, edge.dst.pipe))
                seen = orientation.get(key)
                if seen is None:
                    orientation[key] = want
                elif seen != want and edge.kind == "RAW":
                    raise PassError(PASS, f"%{edge.root} flows both ways between {seen[0]} and {seen[1]} "
                                          f"(#{edge.src.op.id} {edge.src.op.loc}): one session cannot own both "
                                          "hand-offs; separate the storage or publish it explicitly")
        return orientation

    def derive(self, policy: str) -> None:
        self.edges = edges = self.managed_edges()
        self.orientation = self.orient(edges)
        pairs: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        for (side, root, _), (producer, consumer) in self.orientation.items():
            if producer not in SET_PIPES:  # a WAR whose writer is the scalar pipe: nothing can publish to it
                self.ctx.explain.note(
                    f"%{root}: the {producer} pipe cannot set a flag to publish for {consumer}; order it with sync.barrier",
                    kind="warning", root=root)
                continue
            pairs[(side, producer, consumer)].add(root)

        sessions: dict[tuple[Any, ...], Session] = {}
        for (side, producer, consumer), roots in sorted(pairs.items()):
            for root in sorted(roots):
                member = self.member(root)
                if policy == "channel":
                    key: tuple[Any, ...] = (side, producer, consumer)
                elif policy == "allocation":
                    key = (side, producer, consumer, root)
                else:
                    key = (side, producer, consumer, member.cap)
                session = sessions.setdefault(key, Session(side, producer, consumer, key))
                session.members[root] = member
        self.sessions = [sessions[k] for k in sorted(sessions, key=lambda k: tuple(str(x) for x in k))]
        for session in self.sessions:
            self._classify(session)

    def _classify(self, session: Session) -> None:
        for node in self.graph.nodes:
            if node.op.id not in self.inside:
                continue
            writes = {a.root for a in node.accesses if a.kind == WRITE and a.root in session.members}
            reads = {a.root for a in node.accesses if a.kind == READ and a.root in session.members}
            if node.pipe == session.producer and writes:
                session.roles[node.op.id] = "P"
            elif node.pipe == session.consumer and (reads or writes):
                session.roles[node.op.id] = "C"
        session.scope = _common_prefix([self.graph.by_id[i].path for i in session.roles])

    def roles_in(self, op: Op, session: Session) -> frozenset[str]:
        key = (id(session), op.id or -1)
        cached = self._roles_cache.get(key)
        if cached is None:
            found = {session.roles[op.id]} if op.id in session.roles else set()
            for region in op.regions:
                found |= self.region_roles(region, session)
            cached = self._roles_cache[key] = frozenset(found)
        return cached

    def region_roles(self, region: Block, session: Session) -> frozenset[str]:
        found: set[str] = set()
        for inner in region.ops:
            found |= self.roles_in(inner, session)
        return frozenset(found)

    # -- the window walk ------------------------------------------------------------------

    def plan(self, session: Session) -> None:
        session.levels = {}
        block = self.body if not session.scope else session.scope[-1][0].regions[session.scope[-1][1]]
        self._walk(block, session, 0)
        for level in session.ordered():
            for member in session.members.values():
                self._ring(session, level, member)
            level.depth = level.capacity(session.members) if session.credits else 1

    def _item(self, op: Op, session: Session) -> str | None:
        """What this op is to the session: a producer or consumer access of its own depth, an
        ``internal`` hand-over one region performs on both sides, or nothing."""
        roles = self.roles_in(op, session)
        if not roles:
            return None
        if len(roles) == 2 and op.id in session.roles:
            raise PassError(PASS, f"#{op.id} {op.loc}: {session.name} both produces and consumes in one operation")
        if len(roles) == 2 and any(len(self.region_roles(region, session)) == 2 for region in op.regions):
            return "internal"
        # a branch whose arms hold different roles never runs both, so it is one access of its depth:
        # the producing arm decides the window and the other arm's reads follow an earlier publication
        return "P" if "P" in roles else "C"

    def _walk(self, block: Block, session: Session, index: int) -> None:
        items = [(op, self._item(op, session)) for op in block.ops]
        if not any(item in ("P", "C") for _, item in items):
            # Every hand-off of this block happens inside one of its items, so this block orders
            # nothing of its own and is transparent: its subtrees keep this depth's ledger and number
            # their windows in one sequence, however deeply each of them sits, and how far two of
            # those windows may overlap is what the slot rotation says.
            for op, item in items:
                if item == "internal":
                    for region in op.regions:
                        self._walk(region, session, index)
            return
        level = session.levels.setdefault(index, Level(index))
        state, announced = CLOSED, False
        for op, item in items:
            if state != CLOSED and op.opcode in TERMINATORS:
                self._close(session, level, state, op, "before")  # nothing after it runs on this path
                state = CLOSED
            elif state != CLOSED and self._escapes(op):
                raise PassError(PASS, f"#{op.id} {op.loc}: {session.name} holds an open window across a "
                                      f"conditional {'/'.join(sorted(self._leaves(op)))}, whose closing hand-back "
                                      "would be skipped on the escaping path; end the hand-off before it")
            if item is None:
                level.attribution.append((max(level.total, 1), op))
                continue
            if item == "C" and state == CLOSED:
                # a consumer phase opens the block: it holds the credit this execution starts with,
                # and returning that credit is what the first producer phase has to wait for. Nothing
                # of this depth published it -- an outer window or an earlier execution did
                level.leading = True
                self._open(session, level, op)
                state, announced = PUBLISHED, False
            elif item == "C" and state == OPEN:
                self._publish(level, op, "before")
                state, announced = PUBLISHED, True
            elif item != "C":
                if state == PUBLISHED:
                    self._close(session, level, state, op, "before")
                    state = CLOSED
                if state == CLOSED:
                    self._open(session, level, op)
                    state, announced = OPEN, False
                if item == "internal":
                    # The publication happens inside, at its own depth, but the bound on the
                    # consumer does not: with no forward half here its hand-back is throttled by
                    # nothing and it sets the reverse flag while a token of it is outstanding
                    # (M10-096, RFC-0005 §5.10). Pair this depth behind the hand-over, which is
                    # where the producer's publication of this window has been made.
                    self._publish(level, op, "after")
                    state, announced = PUBLISHED, True
            self._number(op, session, level, level.total, published=announced and item == "C")
            level.attribution.append((level.total, op))
            if item == "internal":
                for region in op.regions:
                    self._walk(region, session, index + 1)
        if state != CLOSED and block.ops:
            self._close(session, level, state, block.ops[-1], "after")

    def _open(self, session: Session, level: Level, anchor: Op) -> None:
        if session.credits:
            self._mark(level, anchor, "before", VALID_WAIT)
        level.total += 1

    def _publish(self, level: Level, anchor: Op, position: str) -> None:
        self._mark(level, anchor, position, READY_SET)
        self._mark(level, anchor, position, READY_WAIT)
        level.publishes = True

    def _close(self, session: Session, level: Level, state: str, anchor: Op, position: str) -> None:
        if not session.credits:
            return  # nothing was taken, so nothing is returned
        if state == OPEN:
            # no consumer ran in this window: publish it anyway, so a reader of a later window is
            # ordered after this producer's work, and hand the credit back behind that publication
            self._publish(level, anchor, position)
        self._mark(level, anchor, position, VALID_SET)

    def _mark(self, level: Level, anchor: Op, position: str, kind: str) -> None:
        assert anchor.id is not None
        level.marks.append((anchor.id, position, kind))

    def _escapes(self, op: Op) -> bool:
        """Whether reaching ``op`` can leave the block without running what follows it, which is
        where a window still open would lose its closing hand-back."""
        return bool(self._leaves(op))

    def _leaves(self, op: Op) -> frozenset[str]:
        found = self._escape_cache.get(id(op))
        if found is None:
            if op.opcode in TERMINATORS:
                return self._escape_cache.setdefault(id(op), frozenset({op.opcode}))
            inner = {kind for region in op.regions for child in region.ops for kind in self._leaves(child)}
            if op.opcode == "cf.for":
                inner -= {"cf.break", "cf.continue"}  # a loop catches its own; only a return climbs out
            found = self._escape_cache[id(op)] = frozenset(inner)
        return found

    def _number(self, op: Op, session: Session, level: Level, window: int, *, published: bool) -> None:
        if op.id in session.roles and window:
            level.windows[op.id] = window
            if published:
                level.published.add(op.id)
        for region in op.regions:
            for inner in region.ops:
                self._number(inner, session, level, window, published=published)

    def _ring(self, session: Session, level: Level, member: Member) -> None:
        """A physical slot returns after ``ring`` windows of this depth. Two windows share a slot
        only when their counter values differ by a multiple of the modulus, so the smallest window
        gap that can accumulate that many advances bounds the reuse from below."""
        def reason(text: str) -> None:
            level.reasons[member.root] = text

        slots = {self.slot_of(a.slot) for i in level.windows for a in self.graph.by_id[i].accesses if a.root == member.root}
        if None in slots or not slots or not level.total:
            return
        cells = {s.cell for s in slots if s is not None}
        if cells == {None}:
            reason("a fixed slot index is the same storage every window")
            return
        if len(cells) != 1 or None in cells:
            reason("several counters select this allocation")
            return
        cell = next(iter(cells))
        assert cell is not None
        init = self.cells.get(cell)
        if init is None or int(init.attrs.get("init", 0)) < 0:
            reason(f"counter %{cell} is not a counter initialised at or above zero")
            return
        modulus = min([s.modulus or member.slots for s in slots if s is not None] + [member.slots])
        if any(self._advance_op(op, cell) and not self._in_scope(op, session) for op in self.body.walk()):
            reason(f"counter %{cell} also advances outside the session's scope")
            return
        # Attribute every advance to the window it sits in, split by the position of this member's
        # writes: between two writes of one physical slot the counter advanced a multiple of the
        # modulus, so the smallest window gap that can accumulate that many advances is the ring.
        before: dict[int, int] = defaultdict(int)
        after: dict[int, int] = defaultdict(int)
        written: set[int] = set()
        for window, op in level.attribution:
            count = self.advances(op, cell)
            if count is None:
                reason(f"an advance of counter %{cell} is not a literal count")
                return
            (after if window in written else before)[window] += count
            if self._writes(op, member.root):
                written.add(window)
        if not written:
            reason(f"%{member.root} is written outside every window")
            return
        total = {w: before[w] + after[w] for w in range(1, level.total + 1)}
        if not any(total.values()):
            reason(f"counter %{cell} never advances inside the session's scope")
            return
        windows = sorted(written)
        repeats = any(op.opcode == "cf.for" for op, _ in session.scope)
        for gap in range(1, modulus * max(level.total, 1) + 1):
            for start in windows:
                stop = start + gap
                if not repeats and stop > level.total:
                    continue  # the scope runs once: a gap past its last window has no second write
                if (stop - 1) % level.total + 1 not in written:
                    continue
                reached = after[start] + before[(stop - 1) % level.total + 1]
                reached += sum(total[(w - 1) % level.total + 1] for w in range(start + 1, stop))
                if reached == 0:
                    # zero is a multiple of the modulus like any other: the counter holds the value
                    # it had, so the rotation separates these two writes by nothing and equal
                    # offsets make them the same slot. Reading only `>= modulus` here granted a
                    # credit per slot to a member written twice per trip, and the second write then
                    # raced the first window's reader (RFC-0005 §5.3).
                    level.rings[member.root] = gap
                    reason(f"%{cell} does not advance over {gap - 1} window(s) of this depth, so the "
                           f"rotation does not separate those writes")
                    return
                if reached >= modulus:
                    level.rings[member.root] = gap
                    reason(f"%{cell} needs {modulus} advance(s) to return a slot, which no {gap - 1} "
                           f"window(s) of this depth can reach")
                    return
        level.rings[member.root] = MAX_DEPTH  # the counter never turns: only the declared depth caps it
        reason(f"%{cell} never advances through {modulus} slot(s) here")

    def _advance_op(self, op: Op, cell: str) -> bool:
        return op.opcode == "scalar.set" and bool(op.operands) and getattr(op.operands[0], "name", None) == cell

    def _in_scope(self, op: Op, session: Session) -> bool:
        node = self.graph.by_id.get(op.id or -1)
        if node is None:
            return False
        return node.path[:len(session.scope)] == session.scope

    def _writes(self, op: Op, root: str) -> bool:
        node = self.graph.by_id.get(op.id or -1)
        if node is not None and any(a.kind == WRITE and a.root == root for a in node.accesses):
            return True
        return any(self._writes(inner, root) for region in op.regions for inner in region.ops)

    # -- the flag budget ------------------------------------------------------------------

    def fit(self) -> None:
        """Eight flag ids per ordered channel (RFC-0005 §5.7). A ledger of depth ``W`` holds ``W`` of
        them, and a pre-set one holds them for the whole function, so the demand counted here is the
        plain sum over a channel's ledgers plus what the author's own events already declare on it --
        no live ranges, which over-counts rather than under-counts. Coarsening merges the two
        smallest capacity classes of the fullest channel, the pair whose overlap is worth the least,
        and re-plans them; `events` keeps the exact colouring and its own budget error."""
        while True:
            demand = self._demand()
            channel = max(demand, key=lambda ch: (demand[ch][0], ch))
            need, ledgers = demand[channel]
            if need <= MAX_DEPTH:
                return
            mergeable = sorted((s for s in self.sessions if channel in self._channels(s)),
                               key=lambda s: (self.width(s), str(s.key)))
            if len(mergeable) < 2:
                raise PassError(PASS, f"channel {channel[0]}->{channel[1]} needs {need} of the {MAX_DEPTH} flag ids: "
                                      f"{'; '.join(ledgers)}. No two ledgers are left to merge; give an allocation fewer "
                                      "slots or a smaller sync_depth, free an explicit event on this channel, or hand one "
                                      "of these buffers over by hand")
            keep, drop = mergeable[0], mergeable[1]
            self.ctx.explain.note(
                f"channel {channel[0]}->{channel[1]} needs {need} of {MAX_DEPTH} flag ids: {keep.name} merges the "
                f"{self.width(drop)}-id ledger over {', '.join('%' + r for r in sorted(drop.members))} into its "
                f"{self.width(keep)}-id one, which orders more than it must and overlaps less",
                kind="session-budget", channel=list(channel), need=need)
            keep.members.update(drop.members)
            self.sessions.remove(drop)
            keep.roles, keep.levels = {}, {}
            self._roles_cache.clear()
            self._classify(keep)
            self.plan(keep)

    def width(self, session: Session) -> int:
        """The flag ids one of the session's channels holds: one per credit of every depth."""
        return sum(level.depth for level in session.levels.values())

    def _channels(self, session: Session) -> tuple[tuple[str, str], ...]:
        forward = (session.producer, session.consumer)
        return (forward, (session.consumer, session.producer)) if session.credits else (forward,)

    def _demand(self) -> dict[tuple[str, str], tuple[int, list[str]]]:
        out: dict[tuple[str, str], tuple[int, list[str]]] = defaultdict(lambda: (0, []))
        for op in self.fn.walk():  # the author's own events hold their ids on the same channels
            if op.opcode == "sync.event" and isinstance(op.results[0].type, EventType):
                event = op.results[0].type
                channel = (event.set_pipe, event.wait_pipe)
                need, ledgers = out[channel]
                out[channel] = (need + event.depth, [*ledgers, f"{op.results[0].name} declares {event.depth}"])
        for session in self.sessions:
            forward, reverse = (session.producer, session.consumer), (session.consumer, session.producer)
            for level in session.ordered():
                for channel, present in ((forward, level.publishes), (reverse, session.credits)):
                    if not present:
                        continue
                    need, ledgers = out[channel]
                    out[channel] = (need + level.depth, [*ledgers, f"{session.name} depth {level.index} holds "
                                                         f"{level.depth} over " + ", ".join("%" + r for r in sorted(session.members))])
        return out or {("", ""): (0, [])}

    # -- emission -------------------------------------------------------------------------

    def fresh(self, base: str) -> str:
        n = 0
        while True:
            n += 1
            name = f"{base}{n}"
            if name not in self.names:
                self.names.add(name)
                return name

    def declare(self, session: Session) -> list[Op]:
        guards = sorted(session.members)
        forward = f"{session.producer.lower()}_{session.consumer.lower()}"
        reverse = f"{session.consumer.lower()}_{session.producer.lower()}"
        from_ops = tuple(self.graph.by_id[i].op for i in sorted(session.roles))
        out = []
        for level in session.ordered():
            tail = "" if len(session.levels) == 1 else f"d{level.index}_"
            if level.publishes:
                level.ready = Value(self.fresh(f"ev_{forward}_ready_{tail}"),
                                    EventType(level.depth, session.producer, session.consumer))
            if session.credits:
                level.valid = Value(self.fresh(f"ev_{reverse}_valid_{tail}"),
                                    EventType(level.depth, session.consumer, session.producer))
            note = (f"{session.name} slot session over {', '.join('%' + g for g in guards)} at depth {level.index}: "
                    f"{level.depth} {'credit(s)' if session.credits else 'forward token, no reverse channel'}, "
                    f"{level.total} window(s)" + (", opening on a consumer phase" if level.leading else "")
                    + ("" if level.publishes else ", publishing inside"))
            for value, preset in ((level.ready, 0), (level.valid, level.depth)):
                if value is None:
                    continue
                attrs: dict[str, Any] = {"name": value.name, "guards": guards, "side": Ident(session.side)}
                if preset:
                    attrs["preset"] = preset
                out.append(self.rw.make("sync.event", (), results=(value,), attrs=attrs, from_ops=from_ops, note=note))
            self.ctx.explain.note(
                " / ".join(v.name for v in (level.ready, level.valid) if v is not None) + f": {note}; "
                + level.describe(session.members), kind="session", session=session.name, depth=level.depth,
                guards=guards, windows=level.total, level=level.index)
        return out

    def insertions(self, session: Session) -> tuple[dict[int, list[Op]], dict[int, list[Op]]]:
        before: dict[int, list[Op]] = defaultdict(list)
        after: dict[int, list[Op]] = defaultdict(list)
        # an outer credit is taken before an inner one and returned after it, so the outermost depth
        # owns the outside of both lists
        for target, levels in ((before, session.ordered()), (after, session.ordered()[::-1])):
            for level in levels:
                for anchor, position, kind in level.marks:
                    if (position == "before") != (target is before):
                        continue
                    op = self.graph.by_id[anchor].op
                    event, pipe = ((level.ready, session.producer) if kind == READY_SET else
                                   (level.ready, session.consumer) if kind == READY_WAIT else
                                   (level.valid, session.producer) if kind == VALID_WAIT else
                                   (level.valid, session.consumer))
                    assert event is not None, f"{session.name} depth {level.index} emits {kind} without its ledger"
                    opcode = "sync.set" if kind in (READY_SET, VALID_SET) else "sync.wait"
                    target[anchor].append(self.rw.make(opcode, (event,), attrs={"pipe": Ident(pipe)}, from_ops=(op,),
                                                       loc=op.loc, note=f"{session.name} depth {level.index} {kind}"))
        return before, after

    # -- verification ---------------------------------------------------------------------

    def check(self) -> None:
        by_pair: dict[tuple[str, str, str, str], Session] = {}
        for session in self.sessions:
            for root in session.members:
                by_pair[(session.side, session.producer, session.consumer, root)] = session
        for edge in self.edges:
            side = self.side_of(edge)
            producer, consumer = self.orientation[(side, edge.root, frozenset({edge.src.pipe, edge.dst.pipe}))]
            session = by_pair.get((side, producer, consumer, edge.root))
            if session is not None:  # a session the scalar-pipe report declined to open
                self._covered(session, edge)

    def _covered(self, session: Session, edge: Any) -> None:
        """A pair is covered when one depth of the session covers it: the depth that holds both ops
        in its windows orders them, and a hand-off nested inside one window is the inner depth's."""
        roles = (session.roles.get(edge.src.op.id), session.roles.get(edge.dst.op.id))
        where = (f"%{edge.root}: #{edge.src.op.id} ({edge.src.pipe}) {edge.src.op.loc} -> "
                 f"#{edge.dst.op.id} ({edge.dst.pipe}) {edge.dst.op.loc}")
        if roles not in (("P", "C"), ("C", "P")):
            raise PassError(PASS, f"{session.name} cannot place the {edge.kind} pair {where}: it holds neither role order")
        refusals = []
        for level in session.ordered():
            refusal = self._uncovered(session, level, edge)
            if refusal is None:
                return
            refusals.append(f"at depth {level.index} {refusal}")
        raise PassError(PASS, f"{session.name} does not cover the {'carried ' if edge.distance else ''}{edge.kind} "
                              f"{'hand-off' if roles[0] == 'P' else 'hand-back'} {where} (RFC-0005 §5.6): "
                              + "; ".join(refusals))

    def _uncovered(self, session: Session, level: Level, edge: Any) -> str | None:
        """Why this depth does not cover the pair, or ``None`` when it does. The producer's work is
        published forward by ``ready`` and the consumer's hand-back travels the reverse credit."""
        src, dst = level.windows.get(edge.src.op.id), level.windows.get(edge.dst.op.id)
        if src is None or dst is None:
            return "one of the two runs outside every window of this depth"
        if session.roles.get(edge.src.op.id) == "P":
            if edge.distance:
                # the reader belongs to a later execution. Every window ends in a publication, whose
                # set the producer pipe issues behind this write and whose wait the consumer pipe
                # takes before any work of a later execution, so the two are in order
                if not any(level.publishes for level in session.levels.values()):
                    return "no window of this session publishes, so a reader of a later execution waits for nothing"
                return None
            # the producer's work precedes every later window's publication on its own pipe, so a
            # consumer in that window or any later one is ordered after it
            if src > dst or (src == dst and edge.dst.op.id not in session.announced()):
                finer = "" if len(session.members) == 1 else (
                    f"; this depth's windows also serve {', '.join('%' + r for r in sorted(session.members) if r != edge.root)}"
                    ', so session_ledgers="allocation" may separate them')
                return "the consumer runs inside the producer's own window phase, before it publishes" + finer
            return None
        assert session.credits, "a hand-back from a pipe that cannot set a flag is reported before a session opens"
        ring = level.rings.get(edge.root, 1)
        if not edge.distance:
            if src >= dst:
                return "the reader and the writer share one window, so no credit separates them"
            return None
        if level.depth > ring:
            return (f"this depth holds {level.depth} credit(s) where %{edge.root} returns a slot after {ring} "
                    "window(s), so the producer could overtake the reader")
        if ring > edge.distance:
            self.ctx.explain.note(
                f"{session.name}: %{edge.root} rotates over {ring} window(s) of depth {level.index} where the graph "
                f"falls back to {edge.distance} iteration(s) (#{edge.src.op.id} -> #{edge.dst.op.id}); the slot proof "
                "carries the credit count", kind="session-ring", session=session.name, root=edge.root, ring=ring,
                distance=edge.distance)
        return None


def plan_function(fn: Function, body: Block, inside: set[int], module: Module, rw: Rewriter, ctx: PassContext) -> Function:
    planner = _Planner(fn, body, inside, module, rw, ctx)
    planner.derive(ledger_policy(ctx))
    for session in planner.sessions:
        planner.plan(session)
    planner.fit()
    planner.check()
    decls: list[Op] = []
    before: dict[int, list[Op]] = defaultdict(list)
    after: dict[int, list[Op]] = defaultdict(list)
    for session in planner.sessions:
        decls.extend(planner.declare(session))
        session_before, session_after = planner.insertions(session)
        for target, source in ((before, session_before), (after, session_after)):
            for anchor, ops in source.items():
                target[anchor].extend(ops)
    return replace(fn, body=Block(tuple(decls) + _splice(body, before, after).ops))


def run(module: Module, ctx: PassContext) -> Module:
    from .autosync import _inline_regions

    rw = Rewriter(module, PASS)
    functions = []
    for fn in module.functions:
        if fn.kind not in ("kernel", "func"):
            functions.append(fn)
            continue
        body, inside = _inline_regions(fn.body)
        functions.append(replace(fn, body=body) if not inside else plan_function(fn, body, inside, module, rw, ctx))
    return Module(module.name, {**module.attrs, "next_id": rw._next_id}, tuple(functions))


__all__ = ["LEDGER_POLICIES", "MAX_DEPTH", "Member", "Session", "Slot", "ledger_policy", "plan_function", "run"]
