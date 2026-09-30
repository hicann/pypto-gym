# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Building and rewriting IR (RFC-0001 §9.3).

:class:`Builder` constructs a module op by op with automatic ids and value names; passes use
:class:`Rewriter` to replace ops while recording provenance (``origin``) on everything they touch.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

from .core import SURFACE, Block, FuncRef, Function, Literal, Loc, Module, Op, Origin, Value
from .types import CellType, ScalarType, Type, dtype, parse_type

OperandLike = Any  # Value | Literal | FuncRef | int | float | bool


def _as_type(t: Type | str) -> Type:
    return parse_type(t) if isinstance(t, str) else t


def _as_operand(x: OperandLike) -> Value | Literal | FuncRef:
    if isinstance(x, (Value, Literal, FuncRef)):
        return x
    if isinstance(x, (bool, int, float, complex)):
        return Literal(x)
    raise TypeError(f"cannot use {x!r} as an operand")


def _as_loc(loc: Loc | str | None) -> Loc | None:
    if loc is None or isinstance(loc, Loc):
        return loc
    return Loc.of(loc)


class FunctionBuilder:
    def __init__(self, module: Builder, kind: str, name: str, params: Sequence[tuple[str, Type | str]], attrs: dict[str, Any] | None) -> None:
        self.module = module
        self.kind = kind
        self.name = name
        self.params: dict[str, Value] = {}
        self.attrs: dict[str, Any] = dict(attrs or {})
        self._names: set[str] = set()
        self._stack: list[list[Op]] = [[]]
        for pname, ptype in params:
            v = Value(pname, _as_type(ptype))
            self.params[pname] = v
            self._names.add(pname)

    # -- names and values ---------------------------------------------------------------------

    def p(self, name: str) -> Value:
        return self.params[name]

    def fresh(self, name: str) -> str:
        base = name
        k = 1
        while name in self._names:
            name = f"{base}.{k}"
            k += 1
        self._names.add(name)
        return name

    # -- ops ----------------------------------------------------------------------------------

    def op(self, opcode: str, operands: Iterable[OperandLike] = (), attrs: dict[str, Any] | None = None,
           results: Type | str | Sequence[Type | str] | None = None, names: str | Sequence[str] | None = None,
           loc: Loc | str | None = None, regions: Sequence[Block] = ()) -> Value | tuple[Value, ...] | None:
        """Append an op to the current block; returns its result value(s)."""
        if results is None:
            rtypes: list[Type] = []
        elif isinstance(results, (str, Type)):
            rtypes = [_as_type(results)]
        else:
            rtypes = [_as_type(t) for t in results]
        if names is None:
            base = opcode.split(".", 1)[1].replace(".", "_")
            rnames = [self.fresh(base) for _ in rtypes]
        elif isinstance(names, str):
            rnames = [self.fresh(names)]
        else:
            rnames = [self.fresh(n) for n in names]
        if len(rnames) != len(rtypes):
            raise ValueError("names and results must have the same length")
        vals = tuple(Value(n, t) for n, t in zip(rnames, rtypes, strict=True))
        node = Op(opcode, tuple(_as_operand(x) for x in operands), vals, dict(attrs or {}), self.module.next_id(), _as_loc(loc),
                  (), tuple(regions))
        self._stack[-1].append(node)
        if not vals:
            return None
        return vals[0] if len(vals) == 1 else vals

    def rename_last_result(self, name: str) -> Value:
        """Rename the single result of the op just appended (the frontend names values after
        the Python variable they are assigned to). Nothing can reference it yet."""
        ops = self._stack[-1]
        if not ops or len(ops[-1].results) != 1:
            raise ValueError("rename_last_result needs a just-appended op with one result")
        op = ops[-1]
        old = op.results[0]
        self._names.discard(old.name)
        new = Value(self.fresh(name), old.type)
        ops[-1] = replace(op, results=(new,))
        return new

    @contextmanager
    def block(self):
        """Collect ops into a new block: ``with fb.block() as blk: ...; blk.block``."""
        holder = _BlockHolder()
        self._stack.append([])
        try:
            yield holder
        finally:
            holder.block = Block(tuple(self._stack.pop()))

    @contextmanager
    def region(self, opcode: str, operands: Iterable[OperandLike] = (), attrs: dict[str, Any] | None = None,
               results: Type | str | Sequence[Type | str] | None = None, names: str | Sequence[str] | None = None,
               loc: Loc | str | None = None):
        """An op with one region; the body is what runs inside the ``with``. Yields the result value(s)."""
        pending = _Pending(opcode, list(operands), dict(attrs or {}), results, names, loc)
        pending.id = self.module.next_id()
        vals = self._prepare_results(pending)
        self._stack.append([])
        try:
            yield vals[0] if len(vals) == 1 else (vals if vals else None)
        finally:
            body = Block(tuple(self._stack.pop()))
            self._emit_pending(pending, vals, [body])

    @contextmanager
    def loop(self, lo: OperandLike, hi: OperandLike, step: OperandLike = 1, name: str = "i", loc: Loc | str | None = None):
        with self.region("cf.for", [lo, hi, step], {"name": name}, "i32", name, loc) as i:
            yield i

    @contextmanager
    def if_(self, cond: OperandLike, loc: Loc | str | None = None):
        ib = _IfBuilder(self, cond, loc)
        op_id = self.module.next_id()
        self._stack.append([])
        try:
            yield ib
        finally:
            then_block = Block(tuple(self._stack.pop()))
            regions = [then_block] + ([ib.else_block] if ib.else_block is not None else [])
            self._stack[-1].append(Op("cf.if", (_as_operand(cond),), (), {}, op_id, _as_loc(loc), (), tuple(regions)))

    def cell(self, dt: str, init: OperandLike | None = None, name: str = "v", loc: Loc | str | None = None) -> Value:
        attrs: dict[str, Any] = {}
        if init is not None:
            attrs["init"] = init if isinstance(init, Value) else init
        v = self.op("scalar.cell", (), attrs, CellType(dtype(dt)), name, loc)
        if not isinstance(v, Value):
            raise RuntimeError("scalar.cell must produce one Value")
        return v

    def ret(self, *values: OperandLike, loc: Loc | str | None = None) -> None:
        self.op("cf.return", values, loc=loc)

    # -- internals ----------------------------------------------------------------------------

    def _prepare_results(self, pending: _Pending) -> tuple[Value, ...]:
        if pending.results is None:
            rtypes: list[Type] = []
        elif isinstance(pending.results, (str, Type)):
            rtypes = [_as_type(pending.results)]
        else:
            rtypes = [_as_type(t) for t in pending.results]
        if pending.names is None:
            base = pending.opcode.split(".", 1)[1].replace(".", "_")
            rnames = [self.fresh(base) for _ in rtypes]
        elif isinstance(pending.names, str):
            rnames = [self.fresh(pending.names)]
        else:
            rnames = [self.fresh(n) for n in pending.names]
        return tuple(Value(n, t) for n, t in zip(rnames, rtypes, strict=True))

    def _emit_pending(self, pending: _Pending, vals: tuple[Value, ...], regions: list[Block]) -> None:
        node = Op(pending.opcode, tuple(_as_operand(x) for x in pending.operands), vals, pending.attrs, pending.id,
                  _as_loc(pending.loc), (), tuple(regions))
        self._stack[-1].append(node)

    def finish(self) -> Function:
        if len(self._stack) != 1:
            raise RuntimeError("unbalanced blocks")
        return Function(self.kind, self.name, tuple(self.params.values()), self.attrs, Block(tuple(self._stack[0])))


class _BlockHolder:
    block: Block = Block()


class _Pending:
    def __init__(self, opcode, operands, attrs, results, names, loc) -> None:
        self.opcode, self.operands, self.attrs, self.results, self.names, self.loc = opcode, operands, attrs, results, names, loc
        self.id: int | None = None


class _IfBuilder:
    def __init__(self, fb: FunctionBuilder, cond: OperandLike, loc: Loc | str | None) -> None:
        self.fb = fb
        self.cond = cond
        self.loc = loc
        self.else_block: Block | None = None

    @contextmanager
    def else_(self):
        self.fb._stack.append([])
        try:
            yield
        finally:
            self.else_block = Block(tuple(self.fb._stack.pop()))


class Builder:
    """Build a module: ``b = Builder("k", device="950"); with b.function("kernel", "k", [...]) as fb: ...``."""

    def __init__(self, name: str, *, device: str, mode: str = "mix", ir: str = SURFACE, source: str | None = None, **attrs: Any) -> None:
        self.name = name
        self.attrs: dict[str, Any] = {"device": device, "ir": ir, "mode": mode}
        if source is not None:
            self.attrs["source"] = source
        self.attrs.update(attrs)
        self.functions: list[Function] = []
        self._next_id = 1

    def next_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    @contextmanager
    def function(self, kind: str, name: str, params: Sequence[tuple[str, Type | str]] = (), attrs: dict[str, Any] | None = None):
        fb = FunctionBuilder(self, kind, name, params, attrs)
        yield fb
        self.functions.append(fb.finish())

    def finish(self) -> Module:
        attrs = dict(self.attrs)
        attrs["next_id"] = self._next_id
        return Module(self.name, attrs, tuple(self.functions))


class Rewriter:
    """Rewrite ops in a module on behalf of a pass, recording provenance.

    ``rewrite(fn)`` visits every op (regions included, bottom-up); ``fn(op)`` returns ``None`` to
    keep it, an :class:`Op` or a list of ops to replace it. Ops created through :meth:`make` /
    :meth:`rewritten` get fresh ids and an ``origin`` entry naming this pass.
    """

    def __init__(self, module: Module, pass_name: str) -> None:
        self.module = module
        self.pass_name = pass_name
        self._next_id = int(module.attrs.get("next_id", module.max_id() + 1))

    def fresh_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def make(self, opcode: str, operands: Iterable[OperandLike] = (), results: Sequence[Value] = (), attrs: dict[str, Any] | None = None,
             *, from_ops: Iterable[Op] = (), kind: str = "inserted", note: str | None = None, loc: Loc | None = None,
             regions: Sequence[Block] = ()) -> Op:
        from_ids = tuple(o.id for o in from_ops if o.id is not None)
        if loc is None:
            for o in from_ops:
                if o.loc is not None:
                    loc = o.loc
                    break
        return Op(opcode, tuple(_as_operand(x) for x in operands), tuple(results), dict(attrs or {}), self.fresh_id(), loc,
                  (Origin(self.pass_name, kind, from_ids, note),), tuple(regions))

    def rewritten(self, op: Op, note: str | None = None, **changes: Any) -> Op:
        """The same op with fields changed; keeps its id and appends an origin entry."""
        origin = op.origin + (Origin(self.pass_name, "rewritten", (op.id,) if op.id is not None else (), note),)
        return replace(op, origin=origin, **changes)

    def rewrite(self, fn: Callable[[Op], Op | list[Op] | None]) -> Module:
        functions = tuple(replace(f, body=self._rewrite_block(f.body, fn)) for f in self.module.functions)
        attrs = dict(self.module.attrs)
        attrs["next_id"] = self._next_id
        return Module(self.module.name, attrs, functions)

    def _rewrite_block(self, block: Block, fn: Callable[[Op], Op | list[Op] | None]) -> Block:
        out: list[Op] = []
        for op in block.ops:
            if op.regions:
                op = replace(op, regions=tuple(self._rewrite_block(b, fn) for b in op.regions))
            r = fn(op)
            if r is None:
                out.append(op)
            elif isinstance(r, Op):
                out.append(r)
            else:
                out.extend(r)
        return Block(tuple(out))


def scalar(dt: str) -> ScalarType:
    return ScalarType(dtype(dt))


__all__ = ["Builder", "FunctionBuilder", "Rewriter", "scalar"]
