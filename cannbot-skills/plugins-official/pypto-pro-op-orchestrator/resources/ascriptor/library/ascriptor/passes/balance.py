# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""``balance``: the token oracle on the IR, and the public synchronisation diagnostic.

Replays every function's sets and waits along every control path and reports a wait on zero
tokens, a set beyond an event's depth, or a final count different from the initial one. It judges
hand-written protocols and generated ones alike, so it is neither planner's property; both
`session_sync` on the A2 family and `local_mutex` on A5 are checked by it, as are the runners of
`docs/diagnosing-sync.md`.
"""

from __future__ import annotations

import os

from ..ir import Block, Function, Module, Op, Value
from .util import trip_count

_BALANCE_DEBUG = bool(os.environ.get("ASCRIPTOR_BALANCE_DEBUG"))  # print the state count whenever it grows (a blow-up's shape)
_SYNC_OPS = ("sync.set", "sync.set_all", "sync.wait", "sync.release")


def check_balance(module: Module, *, rounds: int = 4) -> list[str]:
    """The old test-suite's token oracle on the IR: replay every function's sets and waits along every control path
    (literal loops their exact trip count, unresolved loops ``rounds`` times, each branch arm and the no-arm path
    of an ``if`` without else), counting tokens per event.
    A wait on zero tokens, a set beyond the depth, or a final count different from the initial one is reported.

    One replay per event: an event's balance depends on its own tokens and the decisions of the branches holding
    its sets and waits — never on other events — and the replay walks only the ops that touch those (a state per
    path would otherwise multiply with every condition in the function, which `v8_allhif8` turned into hours)."""
    problems: list[str] = []
    for f in module.functions:
        if f.kind in ("kernel", "func"):
            problems.extend(_balance_of(f, rounds))
    return sorted(set(problems))


def _balance_of(f: Function, rounds: int) -> list[str]:
    """One function's token problems (see :func:`check_balance`)."""
    problems: list[str] = []
    depth: dict[str, int] = {}
    preset: dict[str, int] = {}
    for op in f.walk():
        if op.opcode == "sync.event":
            t = op.results[0].type
            d = getattr(t, "depth", 1) or 1
            depth[op.results[0].name] = d
            p = op.attrs.get("preset", 0)
            preset[op.results[0].name] = d if p is True else int(p or 0)
    if not depth:
        return problems

    def name(op: Op) -> str:
        x = op.operands[0]
        return x.name if isinstance(x, Value) else str(x)

    defining = {r.name: op for op in f.walk() for r in op.results}
    # what every op's subtree touches: the events of the sync ops inside it (the op included)
    events_in: dict[int, set[str]] = {}

    def collect(op: Op) -> set[str]:
        found: set[str] = set()
        if op.opcode in _SYNC_OPS and name(op) in depth:
            found.add(name(op))
        for r in op.regions:
            for o in r.ops:
                found |= collect(o)
        events_in[op.id or -1] = found
        return found

    for op in f.body.ops:
        collect(op)
    # per event: the tested values that decide whether its sets and waits run
    conds_of: dict[str, set[str]] = {n: set() for n in depth}
    for op in f.walk():
        if op.opcode == "cf.if" and isinstance(op.operands[0], Value):
            for n in events_in[op.id or -1]:
                conds_of[n].add(op.operands[0].name)
    # A decision is part of the state only while a later branch can still test the value: it is forgotten after
    # the last branch on it in program order, or, when the value is computed outside a loop holding that branch,
    # after that loop (kept for every iteration). Otherwise the paths multiply with every condition.
    loops_of: dict[int, tuple[int, ...]] = {}

    def nest(block: Block, chain: tuple[int, ...]) -> None:
        for op in block.ops:
            loops_of[op.id or -1] = chain
            for r in op.regions:
                nest(r, (*chain, op.id or -1) if op.opcode == "cf.for" else chain)

    nest(f.body, ())
    last_if = {op.operands[0].name: op for op in f.walk() if op.opcode == "cf.if" and isinstance(op.operands[0], Value)}
    forget: dict[int, list[str]] = {}  # branch or loop id -> decisions forgotten once it has run
    for c, op in last_if.items():
        d = defining.get(c)
        inner = loops_of.get(op.id or -1, ())
        outer = loops_of.get(d.id or -1, ()) if d is not None else ()
        at = inner[len(outer)] if len(outer) < len(inner) and inner[: len(outer)] == outer else (op.id or -1)
        forget.setdefault(at, []).append("if:" + c)

    def check_event(n: str) -> None:
        """Replay one event with its own memo, relevance map, and peak counter."""
        conds = conds_of[n]
        rel: dict[int, bool] = {}  # op id -> the op or something inside it matters to this event

        def mark(op: Op) -> bool:
            inside = any([mark(o) for r in op.regions for o in r.ops])  # every op visited: no short-circuit
            own = (n in events_in.get(op.id or -1, ())
                   or any(r.name in conds for r in op.results))
            rel[op.id or -1] = own or inside
            return own or inside

        for op in f.body.ops:
            mark(op)
        memo: dict[tuple[int, tuple[tuple[str, int], ...]], list[dict[str, int]]] = {}
        peak = [0]

        def run(block: Block, tokens: dict[str, int], path: str) -> list[dict[str, int]]:
            """All states of this event reachable at the end of ``block`` from ``tokens`` (its token count and the
            arms taken by the branches that hold its sets and waits — until the value is computed again).
            A block entered twice in the same state ends in the same states: memoised."""
            key = (id(block), tuple(sorted(tokens.items())))
            hit = memo.get(key)
            if hit is not None:
                return [dict(st) for st in hit]
            states = [dict(tokens)]
            for op in block.ops:
                if not rel.get(op.id or -1):
                    continue  # nothing of this event inside
                merges = op.opcode in ("cf.for", "cf.if")
                nxt: list[dict[str, int]] = []
                for st in states:
                    if any(r.name in conds and "if:" + r.name in st for r in op.results):
                        st = {k: v for k, v in st.items() if k not in {"if:" + r.name for r in op.results}}
                    if op.opcode == "cf.for":
                        cur = [st]
                        count = trip_count(op)
                        count = rounds if count is None else count
                        periods: dict[tuple, int] = {}
                        iteration = 0
                        while iteration < count and cur:
                            # The transition depends only on these states. Skip whole repeated periods,
                            # preserving the exact remainder even for a very large literal trip count.
                            unique = {tuple(sorted(s.items())): s for s in cur}
                            signature = tuple(sorted(unique))
                            previous = periods.get(signature)
                            if previous is not None:
                                period = iteration - previous
                                iteration += (count - iteration) // period * period
                                if iteration == count:
                                    break
                            periods[signature] = iteration
                            cur = [s2 for s1 in unique.values() for s2 in run(op.regions[0], s1, path + f"/for#{op.id}")]
                            iteration += 1
                        nxt.extend(cur)
                    elif op.opcode == "cf.if":
                        c = op.operands[0]
                        key2 = "if:" + c.name if isinstance(c, Value) and c.name in conds else None
                        for arm in (0, 1):  # arm 1 of a branch without else: the no-arm path
                            if key2 is not None and st.get(key2, arm) != arm:
                                continue  # an earlier branch on the same value took the other arm
                            st2 = dict(st) if key2 is None else {**st, key2: arm}
                            nxt.extend(run(op.regions[arm], st2, path + f"/{'then' if arm == 0 else 'else'}#{op.id}")
                                       if arm < len(op.regions) else [st2])
                    elif op.opcode in ("sync.set", "sync.set_all") and name(op) == n:
                        k = depth[n] if op.opcode == "sync.set_all" else 1
                        st = dict(st)
                        st[n] = st.get(n, 0) + k
                        if st[n] > depth[n]:
                            problems.append(f"@{f.name}{path}: {op.opcode} #{op.id} on %{n} makes {st[n]} tokens, depth {depth[n]}")
                        else:
                            nxt.append(st)
                    elif op.opcode in ("sync.wait", "sync.release") and name(op) == n:
                        k = depth[n] if op.opcode == "sync.release" else 1
                        st = dict(st)
                        if st.get(n, 0) < k:
                            problems.append(f"@{f.name}{path}: {op.opcode} #{op.id} on %{n} would hang ({st.get(n, 0)} token(s))")
                        else:
                            st[n] = st.get(n, 0) - k
                            nxt.append(st)
                    else:
                        nxt.append(st)
                if op.id in forget:  # no later branch tests these values: forget the decisions taken on them
                    gone = set(forget[op.id])
                    nxt = [{k: v for k, v in st.items() if k not in gone} for st in nxt]
                    merges = True
                if merges:  # paths joined: merge identical states to keep their number small
                    seen: dict[tuple, dict[str, int]] = {}
                    for st in nxt:
                        seen.setdefault(tuple(sorted(st.items())), st)
                    nxt = list(seen.values())
                states = nxt
                if _BALANCE_DEBUG and len(states) > peak[0]:
                    peak[0] = len(states)
                    varying = sorted({k for st in states for k in st if any(st.get(k) != states[0].get(k) for st in states)})
                    print(f"[balance %{n}] {path} #{op.id} {op.opcode}: {len(states)} states; varying {varying[:10]}", flush=True)
            memo[key] = [dict(st) for st in states]
            return states

        for st in run(f.body, {n: preset[n]}, ""):
            if st.get(n, 0) != preset[n]:
                problems.append(f"@{f.name}: %{n} ends with {st.get(n, 0)} token(s), started with {preset[n]}")

    for n in sorted(depth):
        check_event(n)
    return problems


__all__ = ["check_balance"]
