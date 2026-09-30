# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""``gmbuff``: check and lower GMBuff workspace rings (RFC-0009).

A ``GMBuff(dtype, shape, slots=N)`` reaches this pass as a ``mem.workspace`` whose result is a
slot buffer (``BufType``) and whose ``gmbuff_*`` attributes carry the ring geometry. The pass
machine-checks the ring algebra that used to live in hand-maintained comments, then rewrites each
slot selection into the exact spelling the proven kernels always used — a ``scalar.mod`` beat and
a masked ``mem.slice`` of the plain ``[cube_num, slots, rows, cols]`` workspace — so every later
pass and backend sees IR it already handles.

The checks (the pass's actual payload):

1. **single beat** — every slot index of one ring must be the same counter plus a static offset
   (producer ``beat``, consumer ``beat - K``). Two independent counters have no analysable phase
   relation; the old two-counter kernels were exactly the ones whose rings could not be verified.
2. **reader lag < slots** — the offset span ``K`` must be strictly smaller than the slot count,
   or the producer's slot ``beat % slots`` aliases the consumer's ``(beat - K) % slots`` on real
   hardware (the v5 comment's invariant: "the ring MUST be strictly larger than the lookahead";
   the simulator does not model mutex depth, so only this check or the NPU can catch it).
3. **mutex cover** — when a ring is touched from both sides, every write must sit inside a
   producer window (``lock`` .. ``ready``) and every read inside a consumer window (``wait`` ..
   ``free``), and each such covering mutex must have ``depth <= slots`` — a deeper mutex admits
   more beats in flight than the ring has slots. GM is still not event-analysed (``autosync_gm``
   off); since M10-078 `crosssync` does judge cross-side GM edges by flag, so this window scan is
   no longer the only machine check on that leg, and it stays the only one that reads the ring.

Beat identity is by value: one ``cf.for`` induction value or one cell. A cell rewritten between
uses still counts as one beat counter — the check is deliberately linear, not a dataflow engine.
"""

from __future__ import annotations

from typing import Any

from ..ir import Module, Op, Value
from ..ir.builder import Rewriter
from ..ir.types import BufType, DimValue, MemType, ScalarType, dtype
from .util import op_side
from .deps import accesses_of
from .manager import Pass, PassContext, PassError
from .util import Defs, literal_or_value

PASS = "gmbuff"
I32 = ScalarType(dtype("i32"))


class _Ring:
    def __init__(self, root_op: Op) -> None:
        self.op = root_op
        self.value = root_op.results[0]
        self.slots = int(root_op.attrs["gmbuff_slots"])
        self.per_core = bool(root_op.attrs.get("gmbuff_per_core"))
        self.dims: list[Any] = list(root_op.attrs["gmbuff_dims"])  # [cube_num?, slots, rows, cols]
        t = self.value.type
        if not isinstance(t, BufType) or not isinstance(t.elem, MemType):
            raise PassError(PASS, "GMBuff ring must contain memory tiles")
        self.elem: MemType = t.elem
        self.new_value = Value(self.value.name, MemType("ws", self.elem.dtype,
                                                        tuple(DimValue(d.name) if isinstance(d, Value) else int(d)
                                                              for d in self.dims)))
        self.get_bufs: list[Op] = []


def _resolve_beat(v: Any, defs: Defs) -> tuple[str | None, int]:
    """A slot index as (beat counter identity, static offset); (None, c) for a constant."""
    off = 0
    for _ in range(64):
        try:
            v = literal_or_value(v)
        except TypeError:
            break
        if isinstance(v, int):
            return None, off + v
        op = defs.op(v)
        if op is None:
            return v.name, off  # a parameter
        if op.opcode in ("cf.for", "scalar.cell", "core.cube_idx", "core.vec_idx"):
            return v.name, off
        if op.opcode in ("scalar.add", "scalar.sub"):
            a, b = (literal_or_value(x) for x in op.operands)
            sign = 1 if op.opcode == "scalar.add" else -1
            if isinstance(b, int):
                off += sign * b
                v = a
                continue
            if op.opcode == "scalar.add" and isinstance(a, int):
                off += a
                v = b
                continue
        return v.name, off  # any other expression: its own identity
    return getattr(v, "name", None), off


def _check_beats(ring: _Ring, defs: Defs) -> tuple[int, int]:
    """The single-beat and reader-lag checks; returns (min offset, max offset)."""
    resolved = [(_resolve_beat(op.operands[1], defs), op) for op in ring.get_bufs]
    bases = {base for (base, _), _ in resolved}
    if len(bases) > 1:
        names = sorted(str(b) for b in bases)
        raise PassError(PASS, f"%{ring.value.name}: slot indexes use {len(bases)} independent counters ({', '.join(names)}); "
                              "a GMBuff ring is indexed by ONE beat counter — the producer with beat, the reader with "
                              "beat - K (RFC-0009 §2.3)")
    offs = sorted(c for (_, c), _ in resolved)
    lo, hi = offs[0], offs[-1]
    if hi - lo >= ring.slots:
        raise PassError(PASS, f"%{ring.value.name}: reader lag {hi - lo} >= slots {ring.slots}: the producer's slot "
                              f"(beat % {ring.slots}) aliases the reader's ((beat - {hi - lo}) % {ring.slots}) on "
                              "hardware. Grow slots= beyond the lookahead (RFC-0009 §1)")
    return lo, hi


# The mutex protocol reaches this pass as crosscore flags: for a cube-producer (cv) mutex,
# lock = wait_vec (take the consumer's credit), ready = cube_ready (publish), wait = wait_cube,
# free = vec_ready; a vec-producer (vc) mutex is the mirror image.
_METHOD = {
    "cv": {"sync.crosscore.wait_vec": "lock", "sync.crosscore.cube_ready": "ready",
           "sync.crosscore.wait_cube": "wait", "sync.crosscore.vec_ready": "free"},
    "vc": {"sync.crosscore.wait_cube": "lock", "sync.crosscore.vec_ready": "ready",
           "sync.crosscore.wait_vec": "wait", "sync.crosscore.cube_ready": "free"},
}
_MUTEX_METHODS = {"sync.mutex_lock": "lock", "sync.mutex_ready": "ready",
                  "sync.mutex_wait": "wait", "sync.mutex_free": "free"}


def _check_cover(fn_ops: list[Op], rings: dict[str, _Ring], defs: Defs, ctx: PassContext) -> None:
    """The mutex-cover check: one forward scan tracking open lock..ready / wait..free windows."""
    touched: dict[str, list[tuple[Op, str, str | None, frozenset, frozenset]]] = {n: [] for n in rings}
    mutexes: dict[Any, tuple[str, int]] = {}  # flag id (or value name) -> (kind, depth)
    for op in fn_ops:
        if op.opcode == "sync.mutex":
            kind = str(op.attrs.get("kind", "cv"))
            info = (kind, int(op.attrs["depth"]))
            mutexes[op.attrs.get("id")] = info
            if op.results:
                mutexes[op.results[0].name] = info
    prod_open: set[Any] = set()
    cons_open: set[Any] = set()

    def move(m: Any, method: str) -> None:
        if method == "lock":
            prod_open.add(m)
        elif method == "ready":
            prod_open.discard(m)
        elif method == "wait":
            cons_open.add(m)
        elif method == "free":
            cons_open.discard(m)

    for op in fn_ops:
        if op.opcode.startswith("sync.crosscore."):
            m = op.attrs.get("flag_id")
            kind = mutexes.get(m, ("cv", 2))[0]
            method = _METHOD.get(kind, _METHOD["cv"]).get(op.opcode)
            if method is not None:
                move(m, method)
            continue
        if op.opcode in _MUTEX_METHODS and op.operands and isinstance(op.operands[0], Value):
            move(op.operands[0].name, _MUTEX_METHODS[op.opcode])
            continue
        if op.opcode.startswith(("mem.", "sync.", "cf.", "region.")):
            continue
        for a in accesses_of(op, defs):
            if a.root in rings:
                touched[a.root].append((op, a.kind, op_side(op), frozenset(prod_open), frozenset(cons_open)))
    for name, ring in rings.items():
        sides = {s for _, _, s, _, _ in touched[name]}
        if not (("cube" in sides or None in sides) and ("vec" in sides or None in sides)):
            continue  # one side only: no cross-side hand-off to cover
        for op, kind, _, prods, conss in touched[name]:
            window, role = (prods, "producer (lock .. ready)") if kind == "write" else (conss, "consumer (wait .. free)")
            if not window:
                raise PassError(PASS, f"%{name}: {kind} at #{op.id} ({op.opcode}, {op.loc}) is outside every mutex "
                                      f"{role} window; a cross-side GMBuff ring is only ordered by its mutexes "
                                      "(RFC-0009 §3)")
            for m in sorted(window, key=str):
                depth = mutexes.get(m, ("cv", 2))[1]
                if depth > ring.slots:
                    raise PassError(PASS, f"%{name}: {kind} at #{op.id} is covered by mutex {m!r} of depth {depth} > "
                                          f"slots {ring.slots}: the mutex admits more beats in flight than the ring "
                                          "holds (RFC-0009 §3)")
        ctx.explain.note(f"%{name}: cross-side ring covered ({len(touched[name])} accesses inside mutex windows)",
                         kind="cover")


def run(module: Module, ctx: PassContext) -> Module:
    defs = Defs(module)
    rings: dict[str, _Ring] = {}
    for f in module.functions:
        for op in f.walk():
            if op.opcode == "mem.workspace" and "gmbuff_slots" in op.attrs:
                rings[op.results[0].name] = _Ring(op)
    if not rings:
        return module

    for f in module.functions:
        fn_ops = list(f.walk())
        mine = {n: r for n, r in rings.items() if any(o is r.op for o in fn_ops)}
        if not mine:
            continue
        for op in fn_ops:
            for i, v in enumerate(op.operands):
                if isinstance(v, Value) and v.name in mine:
                    if op.opcode == "mem.get_buf" and i == 0:
                        mine[v.name].get_bufs.append(op)
                    else:
                        raise PassError(PASS, f"%{v.name}: a GMBuff is only used through slot selection "
                                              f"(ws[beat]); #{op.id} ({op.opcode}) uses it directly")
            for v in op.attr_values():
                if v.name in mine:
                    raise PassError(PASS, f"%{v.name}: a GMBuff is only used through slot selection "
                                          f"(ws[beat]); #{op.id} ({op.opcode}) references it from an attribute")
        for name, ring in mine.items():
            lo, hi = _check_beats(ring, defs)
            ctx.explain.note(f"%{name}: {ring.slots} slot(s), beat offsets {lo}..{hi} (reader lag {hi - lo}), "
                             f"{len(ring.get_bufs)} slot selection(s)", op=ring.op.id, kind="ring")
        _check_cover(fn_ops, mine, defs, ctx)

    # The rewrite target is the exact spelling the proven kernels used: ONE masked slice of the
    # plain [cube_num, slots, rows, cols] workspace per access. A slice of the slot view composes
    # with the slot selection (the slot view's own offsets are zero, so composition is
    # concatenation); the slot view itself is only materialised for consumers that are not
    # slices, keeping GM view chains one level deep — the simulator and the backends support
    # nothing deeper for GM.
    rw = Rewriter(module, PASS)
    names = {v.name for f in module.functions for op in f.walk() for v in op.results}
    counter = [0]

    def fresh(base: str) -> str:
        counter[0] += 1
        while f"{base}{counter[0]}" in names:
            counter[0] += 1
        names.add(f"{base}{counter[0]}")
        return f"{base}{counter[0]}"

    by_get_buf: dict[int, _Ring] = {}
    slot_views: dict[str, tuple[_Ring, Op]] = {}  # get_buf result name -> (ring, get_buf op)
    for ring in rings.values():
        for op in ring.get_bufs:
            assert op.id is not None
            by_get_buf[op.id] = ring
            slot_views[op.results[0].name] = (ring, op)
    sliced_views: set[str] = set()  # slot views consumed by mem.slice (composed away)
    kept_views: set[str] = set()  # slot views with a non-slice consumer (materialised)
    compose: dict[int, str] = {}  # mem.slice op id -> slot view name it composes with
    for f in module.functions:
        for op in f.walk():
            for i, v in enumerate(op.operands):
                if isinstance(v, Value) and v.name in slot_views:
                    if op.opcode == "mem.slice" and i == 0:
                        sliced_views.add(v.name)
                        assert op.id is not None
                        compose[op.id] = v.name
                    elif op.opcode != "mem.get_buf":
                        kept_views.add(v.name)
    scalars: dict[str, tuple[Value, Value | None]] = {}  # slot view name -> (mod, core), named up front
    for view_name, (ring, _) in slot_views.items():
        mod = Value(fresh(f"{ring.value.name}_slot"), I32)
        core = Value(fresh("cube_idx"), I32) if ring.per_core else None
        scalars[view_name] = (mod, core)

    def base_attrs(ring: _Ring, view_name: str) -> tuple[list[Any], list[Any], list[int]]:
        mod, core = scalars[view_name]
        offsets: list[Any] = ([core] if ring.per_core else []) + [mod, 0, 0]
        extents: list[Any] = ([1, 1] if ring.per_core else [1]) + [ring.dims[-2], ring.dims[-1]]
        mask = [0] * (len(offsets) - 2) + [1, 1]
        return offsets, extents, mask

    def rewrite(op: Op) -> Op | list[Op] | None:
        if op.opcode == "mem.workspace" and op.results and op.results[0].name in rings:
            ring = rings[op.results[0].name]
            attrs = {k: v for k, v in op.attrs.items() if not k.startswith("gmbuff_")}
            return rw.rewritten(op, results=(ring.new_value,), attrs=attrs,
                                note=f"GMBuff ring lowered: [{'cube_num, ' if ring.per_core else ''}{ring.slots}, rows, cols]")
        if op.id in by_get_buf:
            ring = by_get_buf[op.id]
            view_name = op.results[0].name
            mod, core = scalars[view_name]
            out: list[Op] = []
            out.append(rw.make("scalar.mod", (op.operands[1], ring.slots), results=(mod,), from_ops=(op,)))
            if core is not None:
                out.append(rw.make("core.cube_idx", (), results=(core,), from_ops=(op,)))
            if view_name in kept_views:  # a whole-slot consumer: materialise the slot view
                offsets, extents, mask = base_attrs(ring, view_name)
                out.append(rw.make("mem.slice", (ring.new_value,), results=(op.results[0],),
                                   attrs={"offsets": offsets, "extents": extents, "mask": mask}, from_ops=(op,)))
            return out
        if op.id in compose:  # a slice of the slot view: one composed root-level slice
            view_name = compose[op.id]
            ring, _ = slot_views[view_name]
            offsets, extents, mask = base_attrs(ring, view_name)
            inner_off = [x for x in op.attrs["offsets"]]
            inner_ext = [x for x in op.attrs["extents"]]
            inner_mask = [int(m) for m in op.attrs.get("mask", [1] * len(inner_off))]
            head = len(offsets) - 2  # the slot view's offsets in the kept dims are zero: concatenate
            offsets = offsets[:head] + inner_off
            extents = extents[:head] + inner_ext
            mask = mask[:head] + inner_mask
            return rw.rewritten(op, operands=(ring.new_value,),
                                attrs={**{k: v for k, v in op.attrs.items() if k not in ("offsets", "extents", "mask")},
                                       "offsets": offsets, "extents": extents, "mask": mask},
                                note="composed with the GMBuff slot selection")
        return None

    return rw.rewrite(rewrite)


PASS_DEF = Pass(PASS, run, doc="check GMBuff ring algebra (one beat, lag < slots, mutex cover) and lower slot "
                               "selections to workspace slices")

__all__ = ["PASS_DEF", "run"]
