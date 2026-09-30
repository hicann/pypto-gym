# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Cross-side hazard checking, for every kernel rather than for every ``auto_sync`` region.

RFC-0005 gives `autosync` one job — *"handles cross-side synchronisation **only through the manual
mutex protocol**"*, with *"**insertion of cross-side sync stays manual**"* — and then lists
cross-side *checking* under the same pass. The first half is a design position and it is the right
one; the second half put a cross-core check inside a pass whose ``run()`` begins:

    body, inside = _inline_regions(f.body)
    if not inside:
        functions.append(replace(f, body=body))
        continue                                   # the whole pass, skipped

``inside`` is the set of ops that were inside a ``region.autosync``. A kernel that synchronises
itself by hand has none, so it got no cross-side checking at all — and a kernel that synchronises
itself by hand is precisely the one whose author is driving the mutex protocol personally. The
kernel that started this carried a single-slot buffer behind a two-credit mutex and answered with a
wrong number rather than a hang; it was caught only because it happened to contain an
``auto_sync`` block.

Nothing else owns this. The canonical pipeline touches mutexes in two places: `gmbuff`, which
checks ``depth <= slots`` but only for a ``GMBuff`` workspace ring and says so (*"GM is not
event-analysed, so this window scan is the only machine check the trust layer gets"*), and
`autosync`. `auto_mutex` — RFC-0005's "later option" — is a pypto_pro backend feature
(`backends/pypto_pro/native_sync.py`, RFC-0013), not a pass, and does not cover cce.

So this pass takes the checking and leaves `autosync` its own job. It runs on every function, it
reads and returns the module unchanged, and it asks the two questions `autosync` asked of the
edges it owned, whose analysis (:func:`cross_side_findings`, :func:`credit_hazard`) now lives here.

**The credits finding is an error; the coverage finding is a warning.** Credits errors because it
speaks only when it can prove the pair unordered — every other cross-core wait the consuming side
could reach in the window silences it — and because no arrangement of events can repair what it
finds: events do not cross sides, and the credit count is the author's declaration. Coverage warns
because the analysis is the conservative one and its false positives are the shapes it cannot see;
``autosync_cross_side`` still selects ``warn`` (the default), ``error`` or ``off``, and ``off``
silences both.

The gate was removed in two steps on purpose. The first ran the analysis over every cross-side
edge of both repositories — 106 units, 51 functions, 1230 edges — and reported rather than
refused, because how much of what a removed gate exposes is a defect rather than a shape the
analysis cannot see is a measurement and not a deduction. Every one of those 1230 edges was
ordered by a mutex pair; the four credits findings were all in one unit that does not lower today
either way; and 50 of the 51 functions declared a region, so the gate had been costing the shipped
corpus nothing. With the backlog measured at zero, refusing became free. The population this
protects is the kernels not yet written.

``crosssync_report`` — the pass option, or the ``ASCRIPTOR_CROSSSYNC_REPORT`` environment
variable for the unit runners, which build their own ``PassManager`` — prints one JSON census line
per function to stderr. That is how the number above was taken, and how to take it again.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

from dataclasses import dataclass

from ..ir import Function, Ident, Module, Op
from .autosync import _inline_regions
from .deps import DepGraph, Edge, Node
from .manager import Pass, PassContext, PassError
from .util import Defs, op_side, trip_count

PASS = "crosssync"


def cross_side_edges(fn, body, defs):
    """Every cross-pipe edge whose two ends run on different sides.

    The same filter `autosync` applies, minus its ``inside`` test — which is the whole change.
    `DepGraph` is already built on the entire function body there (``inside`` is carried
    separately and only decides which pairs that pass will insert events for), so this costs one
    graph per function and no new analysis.

    GM is tracked, which RFC-0005 §3.1 asks for ("over L1 / UB / GM") and this pass could not
    afford until two precision repairs landed (M10-078): a window reached through a `mem.reshape`
    kept its exact offsets instead of collapsing to the whole root, and a carried edge over a
    runtime-moving window stopped being convicted. Measured over both repositories afterwards, GM
    adds 118 cross-side edges and no finding at all; before them the same switch added 298 edges
    and 61 findings, every one of them false. `autosync` keeps its own `autosync_gm` default: it
    inserts events, and a GM dependency is not an event's to order.
    """
    g = DepGraph(fn, body, defs, track_gm=True)
    edges = []
    for e in g.edges:
        if e.src.pipe == e.dst.pipe:
            continue
        src, dst = op_side(e.src.op), op_side(e.dst.op)
        if src is not None and dst is not None and src != dst:
            edges.append(e)
    return g, edges



@dataclass(frozen=True)
class CrossFinding:
    """One cross-side edge's verdict. ``kind``: ``ordered`` | ``coverage`` | ``credits``."""

    kind: str
    edge: Edge
    message: str


# A mutex call and a raw numeric cross-core flag reach the same opcode and share one flag-id space,
# so coverage reads both the same way: `ready` publishes on the producer's side and `free` on the
# consumer's, `lock` waits on the producer's side and `wait` on the consumer's (device_lower._MUTEX
# is this table, spelled as opcodes). RFC-0005 §3.1 asks for "a mutex `ready` -> `wait` pair (or a
# cross-core flag)", and the flag id is what makes a pair a pair.
_MUTEX_ROLE = {"ready": ("publish", 0), "free": ("publish", 1), "lock": ("wait", 0), "wait": ("wait", 1)}
_MUTEX_SIDES = {"vc": ("vec", "cube"), "cv": ("cube", "vec")}  # kind -> (producer, consumer)
_PUBLISH_SIDE = {"sync.crosscore.cube_ready": "cube", "sync.crosscore.vec_ready": "vec"}


def _flag_use(op: Op, g: DepGraph, mutexes: dict[int, tuple[str, int]]) -> tuple[int, str, str] | None:
    """``(flag id, publish | wait, side)`` of one cross-core publish or wait, or None.

    Lowered ops carry ``flag_id`` and the registry already knows their side; a surface
    ``sync.mutex_*`` call is read through its declaration, which is the only place the direction
    lives. Collectives and barriers are not flags and are handled by the caller.
    """
    if op.opcode in _PUBLISH_SIDE:
        return int(op.attrs["flag_id"]), "publish", _PUBLISH_SIDE[op.opcode]
    if op.opcode in _CROSS_WAIT_SIDE:
        return int(op.attrs["flag_id"]), "wait", _CROSS_WAIT_SIDE[op.opcode]
    if op.opcode.startswith("sync.mutex_"):
        role = _MUTEX_ROLE.get(op.opcode.removeprefix("sync.mutex_"))
        decl = g.defs.op(op.operands[0]) if role is not None and op.operands else None
        if decl is None or decl.opcode != "sync.mutex":
            return None
        kind = decl.attrs["kind"]
        sides = _MUTEX_SIDES.get(kind.name if isinstance(kind, Ident) else str(kind))
        if sides is None:
            return None
        return int(decl.attrs["id"]), role[0], sides[role[1]]
    return None


def _cross_window(g: DepGraph, e: Edge) -> list[Node]:
    """What runs between the two ops, in EXECUTION order - which for a carried edge wraps the loop.

    A carried edge's ordering calls sit after the source going round the shared loop and before the
    destination on the next trip, so a forward slice of program order shows the wrong half of the
    cycle: it is the distinction :func:`credit_hazard` already draws, and reading one window for
    both is how the loose test came to accept a hand-back's calls for a hand-off's.
    """
    if e.distance and e.loop is not None:
        tail = [n for n in g.nodes if n.index > e.src.index and any(op is e.loop for op, _ in n.path)]
        head = [n for n in g.nodes if n.index < e.dst.index and any(op is e.loop for op, _ in n.path)]
        return tail + head  # the rest of this trip, then the next trip up to the destination
    lo, hi = sorted((e.src.index, e.dst.index))
    return g.nodes[lo + 1:hi]


def cross_side_findings(g: DepGraph, fn: Function, edges: list[Edge]) -> list[CrossFinding]:
    """Verdicts for cross-side hazard edges, with no policy and nothing raised.

    Two questions per edge. **Coverage**: between the two ops, is one flag published on the
    source's side and then awaited on the destination's? Events cannot cross sides, so nothing
    else can order them. **Credits**: is the hand-back ordered, or does the mutex carry more
    credits than the cycles separating the pair (:func:`credit_hazard`)?

    Coverage is that one sentence and no more: not the roles, not which class declared the flag,
    not the credits. A publish only orders the source if it runs on the source's own side, a wait
    only orders the destination on the destination's, and the two must name the same flag with the
    publish first - which is equally true of ``VcMutex.ready`` and of a bare ``vec_ready(1, ...)``.
    The earlier test asked only whether the window held *some* publish-ish opcode and *some*
    wait-ish one, so one flag's publish covered another flag's wait, either direction covered the
    other, and a carried edge was read over the wrong half of its cycle
    (`docs/rfc/0005-autosync-on-ir.md`, cross-side checking amendment).

    Separated from the planner so that a pass which is not the event inserter can ask the same
    questions. `autosync` owns only the pairs inside a ``region.autosync``; the correctness of a
    cross-side handoff is not conditional on that region existing."""
    mutexes = _mutex_credits(fn)
    out: list[CrossFinding] = []
    for e in edges:
        credits = credit_hazard(g, e, mutexes)
        if credits is not None:
            out.append(CrossFinding("credits", e, credits))
        src_side, dst_side = op_side(e.src.op), op_side(e.dst.op)
        kind = "carried " if e.distance else ""
        here = (f"cross-side {kind}{e.kind} on %{e.root}: #{e.src.op.id} ({e.src.pipe}, {src_side}) -> "
                f"#{e.dst.op.id} ({e.dst.pipe}, {dst_side})")
        uses: list[tuple[int, str, str, Node]] = []
        rendezvous = None
        for n in _cross_window(g, e):
            if n.op.opcode == "sync.barrier" or ".all" in n.op.opcode or ".intracore" in n.op.opcode:
                rendezvous = n  # a whole-side rendezvous orders any pair; see credit_hazard
                break
            use = _flag_use(n.op, g, mutexes)
            if use is not None:
                uses.append((*use, n))
        if rendezvous is not None:
            out.append(CrossFinding("ordered", e, f"{here} ordered by {rendezvous.op.opcode} #{rendezvous.op.id}"))
            continue
        pair = None
        for index, (flag, role, side, node) in enumerate(uses):
            if role != "publish" or side != src_side:
                continue
            awaited = next((n for f, r, s, n in uses[index + 1:] if f == flag and r == "wait" and s == dst_side), None)
            if awaited is not None:
                pair = (flag, node, awaited)
                break
        if pair is not None:
            out.append(CrossFinding("ordered", e, (
                f"{here} ordered by flag {pair[0]}: {pair[1].op.opcode} #{pair[1].op.id} on {src_side} "
                f"then {pair[2].op.opcode} #{pair[2].op.id} on {dst_side}")))
            continue
        if e.distance and not all(_same_bytes_every_cycle(n, e.root) for n in (e.src, e.dst)):
            # A carried edge over a window that moves with a runtime offset is not evidence: `deps`
            # falls back to distance one for any index it cannot read (RFC-0005 §2.1), so it does
            # not know that this trip's bytes are the next trip's. `credit_hazard` declines the
            # same shapes for the same reason, and convicting here asserts a hazard the analysis
            # cannot establish. The same buffer's same-iteration edge is still convicted, so a
            # hand-off with no flag at all is still reported.
            continue
        seen = ", ".join(sorted({f"flag {f} {r}ed on {s}" for f, r, s, _ in uses})) or "nothing"
        out.append(CrossFinding("coverage", e, (
            f"{here.replace(' on %', ' hazard on %')} has no flag published on {src_side} and then awaited on "
            f"{dst_side} between the two ops ({e.src.op.loc} / {e.dst.op.loc}); the window holds {seen}. "
            f"Events cannot cross sides — hand the buffer over with sync.mutex (VcMutex / CvMutex), and keep "
            f"one mutex per hand-off buffer")))
    return out


def credit_hazard(g: DepGraph, e: Edge, mutexes: dict[int, tuple[str, int]]) -> str | None:
    """A hand-back across a mutex is ordered only once the mutex's credits are spent (M10-076).

    A mutex is a counting semaphore: the consumer publishes ``depth`` credits before the body, so the
    producer's ``lock`` of cycle ``i`` consumes credit ``i`` and blocks on the consumer's ``free`` of cycle
    ``i - depth`` — on nothing at all while ``i < depth``. The forward hand-off (``ready`` -> ``wait``) has
    no credits and always orders cycle ``i`` against cycle ``i``, which is why the RAW on a hand-off buffer
    looks synchronised. It is the *hand-back* — the consumer's read of one cycle against the producer's
    write of the next, a WAR either loop-carried or between two written-out cycles — that needs the return
    of the slot, and that edge exists only when ``depth`` is at most the number of cycles ``j`` separating
    the two: a buffer written the same way every cycle separates them by one and wants exactly one credit;
    a buffer rotating over two slots separates them by two and can hold two.

    With more credits than slots the producer takes the next slot while the consumer is still reading the
    one before, and no set of events can repair it — events do not cross sides, and the credit count is the
    author's declaration. So the caller reports it as an error rather than a warning: this returns a message
    only when it can prove the pair unordered, because every other cross-core wait the consuming side could
    reach in the window (a barrier, a collective, another mutex's ``wait``, a ``lock`` whose credits *are*
    spent) is an ordering this analysis does not model, and silences it."""
    if e.distance and e.loop is None:
        return None
    if e.distance:
        trips = trip_count(e.loop)
        if trips is not None and trips <= e.distance:
            return None  # the loop never runs the two trips the hazard needs
        # The wrap-around window: what the two ops have between them going *round* the loop they share,
        # which is where the slot comes back. `nodes` is one flattened program order, so that window is the
        # shared loop's body minus the ops the forward window already covers.
        window = [n for n in g.nodes if (n.index > e.src.index or n.index < e.dst.index)
                  and any(op is e.loop for op, _ in n.path)]
    elif e.src.index < e.dst.index:
        window = g.nodes[e.src.index + 1:e.dst.index]  # two cycles written out: what lies between them
    else:
        return None
    if not all(_same_bytes_every_cycle(n, e.root) for n in (e.src, e.dst)):
        return None  # a window that moves with a runtime offset: `deps` returns the conservative distance one
    side = op_side(e.dst.op)
    locks: dict[int, list[Node]] = {}
    for n in window:
        opcode = n.op.opcode
        if opcode == "sync.barrier":
            return None
        if _CROSS_WAIT_SIDE.get(opcode) != side:
            continue
        flag = int(n.op.attrs.get("flag_id", -1))
        decl = mutexes.get(flag)
        if decl is None or _MUTEX_LOCK_WAIT.get(decl[0]) != opcode:
            return None  # a wait fed by the other side's `ready`, or a collective: it orders the pair
        locks.setdefault(flag, []).append(n)
    if not locks:
        return None
        # How many cycles of one mutex separate the two ops. A carried edge crosses the body `distance` times,
        # and the window shows one body's worth of locks: `distance` times that count is never an under-count.
        # A multi-slot buffer whose rotation `deps` could not read keeps the conservative distance one while the
        # slots really do separate the cycles, so the slot count is a floor: this check must never refuse a
        # program it cannot convict, and a fixed slot of a rotating buffer is left to the pipe model.
    span = max(e.distance, 1, g.slots.get(e.root) or 1)
    spans = {flag: len(ns) * span for flag, ns in locks.items()}
    if any(mutexes[flag][1] <= j for flag, j in spans.items()):
        return None  # that lock does block on a `free` at or before the consumer's cycle
    flag = min(spans, key=lambda f: (mutexes[f][1] - spans[f], f))
    kind, depth = mutexes[flag]
    node, j = locks[flag][0], spans[flag]
    later = "the next cycle" if j == 1 else f"{j} cycles later"
    return (
        f"cross-side {'carried ' if e.distance else ''}{e.kind} hazard on %{e.root}: #{e.src.op.id} "
        f"({e.src.pipe}, {op_side(e.src.op)}) reads it and #{e.dst.op.id} ({e.dst.pipe}, {side}) writes it "
        f"{later} ({e.src.op.loc} / {e.dst.op.loc}), with only the {kind} mutex id={flag} lock #{node.op.id} in "
        f"between. That lock carries {depth} credits, so it blocks on the free of {depth} cycles back rather "
        f"than on the free of the cycle that read %{e.root}: the producer takes the slot again while the "
        f"consumer is still reading it. Name the buffer instead of the number — "
        f"{'CvMutex' if kind == 'cv' else 'VcMutex'}(id, guards=%{e.root}, ...) reads the credits off its slots, "
        f"and this pair wants {j} — or give %{e.root} {depth} slots and hand a different one over each cycle.")


_CROSS_WAIT_SIDE = {"sync.crosscore.wait_vec": "cube", "sync.crosscore.wait_cube": "vec",
                    "sync.crosscore.allcube_wait": "cube", "sync.crosscore.allvec_wait": "vec",
                    "sync.crosscore.intracore_allvec_wait": "vec"}
# The one cross-core wait of a mutex that the consumer pre-fills with credits: the producer's `lock`.
_MUTEX_LOCK_WAIT = {"cv": "sync.crosscore.wait_vec", "vc": "sync.crosscore.wait_cube"}


def _same_bytes_every_cycle(node: Node, root: str) -> bool:
    """Every window ``node`` opens on ``root`` is the same bytes on every execution: static bounds, or the whole
    root. A window whose offset is a runtime value may well be a rotation :mod:`deps` could not read, and the
    carried distance it falls back to (one) is then no evidence about how far apart two cycles are."""
    windows = [a for a in node.accesses if a.root == root]
    return bool(windows) and all(a.whole or all(isinstance(x, int) for x in a.lo + a.hi) for a in windows)


def _mutex_credits(fn: Function) -> dict[int, tuple[str, int]]:
    """``flag id -> (kind, depth)`` of every ``sync.mutex`` the function declares; a reused id keeps the deepest."""
    out: dict[int, tuple[str, int]] = {}
    for op in fn.walk():
        if op.opcode != "sync.mutex":
            continue
        kind = op.attrs["kind"]
        kind = kind.name if isinstance(kind, Ident) else str(kind)
        fid, depth = int(op.attrs["id"]), int(op.attrs["depth"])
        seen = out.get(fid)
        out[fid] = (kind, depth if seen is None or seen[0] != kind else max(depth, seen[1]))
    return out


_REPORT_ENV = bool(os.environ.get("ASCRIPTOR_CROSSSYNC_REPORT"))


def run(module: Module, ctx: PassContext) -> Module:
    policy = str(ctx.option("autosync_cross_side", "warn"))
    report = bool(ctx.option("crosssync_report", False)) or _REPORT_ENV
    if policy == "off" and not report:
        return module
    for f in module.functions:
        if f.kind not in ("kernel", "func"):
            continue
        body, inside = _inline_regions(f.body)
        g, edges = cross_side_edges(f, body, defs := Defs(module))
        if not edges:
            continue
        findings = cross_side_findings(g, f, edges)
        census = Counter(x.kind for x in findings)
        for x in findings:
            if x.kind == "ordered" or policy == "off":
                continue
            if x.kind == "credits" or policy == "error":
                raise PassError(PASS, x.message)
            ctx.explain.note(x.message, op=x.edge.dst.op.id,
                             ops=(x.edge.src.op.id, x.edge.dst.op.id), kind="warning")
        if report:
            newly = Counter(x.kind for x in findings if x.kind != "ordered"
                            and not (x.edge.src.op.id in inside and x.edge.dst.op.id in inside))
            print("[" + PASS + "] " + json.dumps({
                "function": f.name, "edges": len(edges), "regions": bool(inside),
                "ordered": census["ordered"], "coverage": census["coverage"], "credits": census["credits"],
                "newly_visible_coverage": newly["coverage"], "newly_visible_credits": newly["credits"],
            }), file=sys.stderr)
    return module


PASS_DEF = Pass(PASS, run, doc="cross-side mutex coverage and credit checks for every kernel; reads the module, never rewrites it")

__all__ = ["PASS", "PASS_DEF", "CrossFinding", "credit_hazard", "cross_side_edges", "cross_side_findings", "run"]
