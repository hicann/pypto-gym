# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared machinery of the passes: use-def index, view geometry, scalar arithmetic emission.

* :class:`Defs` maps every value of a function to its defining op (parameters have none).
* :func:`view_of` folds a chain of ``mem.slice`` / ``mem.get_buf`` / ``mem.reinterpret`` /
  ``mem.reshape`` back to its root allocation (or parameter) with accumulated offsets and extents
  per root dimension — the geometry the old DSL kept on ``Tensor.offset / span / shape`` and the
  old stubs used to infer their DMA parameters. Offsets and extents are ints when static and
  :class:`~ascriptor.ir.Value` (``i32``) when they depend on a scalar computed at run time.
* :class:`Emit` builds the scalar arithmetic a rewritten op needs (``scalar.add`` …) as new ops
  with provenance, folding constants, so a pass never has to hand-write ``Op`` objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from ..ir import REGISTRY, Block, Function, Ident, Literal, Module, Op, Value
from ..ir.builder import Rewriter
from ..ir.types import BufType, DimValue, DType, MemType, ScalarType, dtype

I32 = ScalarType(dtype("i32"))
Scalar = int | Value

SET_PIPES = {"MTE1", "MTE2", "MTE3", "M", "V", "FIX"}  # the scalar pipe cannot set a flag
CUBE_PIPES = {"MTE1", "M", "FIX"}
VEC_PIPES = {"V", "MTE3"}


def op_side(op: Op) -> str | None:
    """cube | vec from the pipe (MTE2 belongs to whichever side the op's registry entry says)."""
    spec = REGISTRY.find(op.opcode)
    if spec is not None and spec.side in ("cube", "vec"):
        return spec.side
    pipe = op.attrs.get("pipe")
    pipe = pipe.name if isinstance(pipe, Ident) else (pipe or (spec.pipe if spec else None))
    if pipe in CUBE_PIPES:
        return "cube"
    if pipe in VEC_PIPES:
        return "vec"
    return None


def is_static(x: Any) -> bool:
    return isinstance(x, (int, bool)) and not isinstance(x, Value)


def trip_count(op: Op) -> int | None:
    """A device loop's literal trip count, or ``None`` when a bound is dynamic."""
    bounds = []
    for operand in op.operands:
        if isinstance(operand, Literal) and isinstance(operand.value, int):
            bounds.append(operand.value)
        elif isinstance(operand, int):
            bounds.append(operand)
        else:
            return None
    lo, hi, step = bounds
    return len(range(lo, hi, step)) if step else None


def as_int(x: Any) -> int | None:
    return int(x) if isinstance(x, (int, bool)) else None


def dim_scalar(d: Any) -> Scalar:
    """A type dimension as a scalar: literal ints stay ints, ``%N`` dims become the i32 value ``%N``."""
    if isinstance(d, DimValue):
        return Value(d.name, I32)
    if isinstance(d, int):
        return d
    raise TypeError(f"dimension {d!r} is not a literal or a scalar value")


def literal_or_value(x: Any) -> Scalar:
    if isinstance(x, Literal):
        return int(x.value)
    if isinstance(x, Value):
        return x
    if isinstance(x, (int, bool)):
        return int(x)
    raise TypeError(f"{x!r} is not a scalar operand")


class Defs:
    """Value name -> (defining op, function) for a whole module; parameters map to (None, function)."""

    def __init__(self, module: Module) -> None:
        self.by_name: dict[str, tuple[Op | None, Function]] = {}
        for f in module.functions:
            for p in f.params:
                self.by_name[p.name] = (None, f)
            for op in f.walk():
                for r in op.results:
                    self.by_name[r.name] = (op, f)

    def op(self, v: Value) -> Op | None:
        return self.by_name.get(v.name, (None, None))[0]

    def function(self, v: Value) -> Function | None:
        return self.by_name.get(v.name, (None, None))[1]


@dataclass(frozen=True)
class View:
    """A window onto a root tensor: geometry in the root's coordinates (RFC-0001 §4.3 views)."""

    root: Value  # mem.alloc / mem.workspace result, or a GM parameter
    root_op: Op | None  # None for parameters
    offsets: tuple[Scalar, ...]  # per root dim, in elements of the *root* dtype scaled to the view dtype
    extents: tuple[Scalar, ...]  # per root dim
    kept: tuple[bool, ...]  # dims present in the view type (False: indexed away, extent 1)
    dtype: DType
    layout: str | None
    slot: Scalar | None = None  # mem.get_buf slot index, if the root is a slot buffer
    reshaped: bool = False  # a mem.reshape sits in the chain: offsets are in the reshaped coordinates
    dims: tuple[Scalar, ...] | None = None  # the full dimensions the offsets refer to (the reshaped ones after mem.reshape)
    gm_strides: tuple[Scalar, ...] | None = None  # mem.view root: explicit element strides per dim (RFC-0010)
    origin: tuple[Any, ...] = ()  # (offsets, dims) of each window a mem.reshape re-addressed, outermost first

    @property
    def shape(self) -> tuple[Scalar, ...]:
        """The full dimensions of the tensor the window is cut from (the old ``Tensor.shape``)."""
        return self.dims if self.dims is not None else root_dims(self.root)

    @property
    def span(self) -> tuple[Scalar, ...]:
        """Extents of the kept dims only (the old ``Tensor.span`` of the view)."""
        return tuple(e for e, k in zip(self.extents, self.kept, strict=True) if k)

    @property
    def sliced_dims(self) -> tuple[int, ...]:
        return tuple(i for i, k in enumerate(self.kept) if k)

    @property
    def rank(self) -> int:
        return len(self.offsets)


def elem_type(t: Any) -> MemType:
    return t.elem if isinstance(t, BufType) else t


def root_dims(root: Value) -> tuple[Scalar, ...]:
    t = elem_type(root.type)
    assert isinstance(t, MemType)
    return tuple(dim_scalar(d) for d in t.dims)


def view_of(v: Value, defs: Defs) -> View:
    """Fold the view chain under ``v`` back to its root."""
    op = defs.op(v)
    t = v.type
    if op is None or op.opcode in ("mem.alloc", "mem.workspace"):
        mt = elem_type(t)
        assert isinstance(mt, MemType), f"{v} is not a memory value"
        dims = tuple(dim_scalar(d) for d in mt.dims)
        return View(v, op, (0,) * len(dims), dims, (True,) * len(dims), mt.dtype, mt.layout)
    if op.opcode == "mem.get_buf":
        base = view_of(op.operands[0], defs)  # type: ignore[arg-type]
        return View(base.root, base.root_op, base.offsets, base.extents, base.kept, base.dtype, base.layout,
                    slot=literal_or_value(op.operands[1]), reshaped=base.reshaped, dims=base.dims, origin=base.origin)
    if op.opcode == "mem.slice":
        base = view_of(op.operands[0], defs)  # type: ignore[arg-type]
        offs = [literal_or_value(x) for x in op.attrs["offsets"]]
        exts = [literal_or_value(x) for x in op.attrs["extents"]]
        mask = [bool(m) for m in op.attrs.get("mask", [True] * len(offs))]
        # the slice is expressed in the base view's kept dims; map them onto the root dims
        kept_idx = [i for i, k in enumerate(base.kept) if k]
        new_off = list(base.offsets)
        new_ext = list(base.extents)
        new_kept = list(base.kept)
        for j, i in enumerate(kept_idx):
            new_off[i] = add_static(base.offsets[i], offs[j])
            new_ext[i] = exts[j]
            new_kept[i] = mask[j]
        return View(base.root, base.root_op, tuple(new_off), tuple(new_ext), tuple(new_kept), base.dtype, base.layout,
                    slot=base.slot, reshaped=base.reshaped, dims=base.dims, gm_strides=base.gm_strides, origin=base.origin)
    if op.opcode == "mem.reinterpret":
        base = view_of(op.operands[0], defs)  # type: ignore[arg-type]
        mt = elem_type(t)
        assert isinstance(mt, MemType)
        layout = mt.layout if "layout" in op.attrs else base.layout
        old_bits, new_bits = max(base.dtype.bits, 8), max(mt.dtype.bits, 8)
        if old_bits == new_bits:  # same element width: the window keeps a view's strides too (RFC-0010 §10)
            return replace(base, dtype=mt.dtype, layout=layout)
        last = max(i for i, k in enumerate(base.kept) if k)
        offs, exts = list(base.offsets), list(base.extents)
        dims = list(base.shape)
        offs[last] = scale_static(offs[last], old_bits, new_bits)
        exts[last] = scale_static(exts[last], old_bits, new_bits)
        dims[last] = scale_static(dims[last], old_bits, new_bits)
        return View(base.root, base.root_op, tuple(offs), tuple(exts), base.kept, mt.dtype, layout, base.slot, base.reshaped, tuple(dims),
                    origin=base.origin)
    if op.opcode == "mem.reshape":
        base = view_of(op.operands[0], defs)  # type: ignore[arg-type]
        mt = elem_type(t)
        assert isinstance(mt, MemType)
        dims = tuple(dim_scalar(d) for d in mt.dims)
        # a new coordinate system at the window's first element (RFC-0010 §10): the origin names that window
        return View(base.root, base.root_op, (0,) * len(dims), dims, (True,) * len(dims), mt.dtype, mt.layout, base.slot, reshaped=True, dims=dims,
                    origin=(*base.origin, (base.offsets, base.dims)))
    if op.opcode == "mem.view":
        # a strided GM re-description (RFC-0010): a new root - the base's geometry is not rectangular
        # in the view's coordinates; consumers read the strides, deps re-anchors to the base itself
        mt = elem_type(t)
        assert isinstance(mt, MemType)
        dims = tuple(dim_scalar(d) for d in mt.dims)
        return View(v, op, (0,) * len(dims), dims, (True,) * len(dims), mt.dtype, mt.layout,
                    gm_strides=tuple(literal_or_value(s) for s in op.attrs["strides"]))
    if op.opcode == "list.item":
        mt = elem_type(t)
        assert isinstance(mt, MemType)
        dims = tuple(dim_scalar(d) if not isinstance(d, DimValue) or True else d for d in mt.dims)
        return View(v, op, (0,) * len(dims), dims, (True,) * len(dims), mt.dtype, mt.layout)
    raise TypeError(f"{v} ({op.opcode} #{op.id}) is not a memory view")


def add_static(a: Scalar, b: Scalar) -> Scalar | tuple:
    """``a + b`` folded when both are ints; otherwise a deferred sum marker resolved by :class:`Emit`."""
    if isinstance(a, int) and isinstance(b, int):
        return a + b
    if isinstance(a, int) and a == 0:
        return b
    if isinstance(b, int) and b == 0:
        return a
    return ("+", a, b)


def scale_static(x: Any, old_bits: int, new_bits: int) -> Any:
    if isinstance(x, int):
        return x * old_bits // new_bits
    if old_bits > new_bits:
        return ("*", x, old_bits // new_bits)
    return ("//", x, new_bits // old_bits)


class Emit:
    """Scalar arithmetic for a rewrite: constants fold, dynamic terms become ``scalar.*`` ops collected in ``pre``
    (to be placed before the op being rewritten). Names are fresh within the function."""

    def __init__(self, rw: Rewriter, fn: Function, anchor: Op, names: set[str] | None = None) -> None:
        self.rw = rw
        self.anchor = anchor
        self.pre: list[Op] = []
        # the name set is shared by every Emit of one function (rewrite_module passes it) so fresh names stay unique
        self._names = names if names is not None else function_names(fn)

    def fresh(self, base: str = "t") -> Value:
        n = 0
        while True:
            n += 1
            name = f"{base}.{n}"
            if name not in self._names:
                self._names.add(name)
                return Value(name, I32)

    def value(self, x: Any) -> Scalar:
        """Resolve deferred markers produced by :func:`add_static` / :func:`scale_static`."""
        if isinstance(x, tuple):
            op, a, b = x
            a, b = self.value(a), self.value(b)
            return {"+": self.add, "*": self.mul, "//": self.div}[op](a, b)
        return x

    def _binop(self, opcode: str, a: Scalar, b: Scalar, fold: Any) -> Scalar:
        a, b = self.value(a), self.value(b)
        if isinstance(a, int) and isinstance(b, int):
            return fold(a, b)
        res = self.fresh(opcode.split(".")[-1])
        self.pre.append(self.rw.make(opcode, (a, b), results=(res,), from_ops=(self.anchor,), kind="inserted",
                                     note="scalar arithmetic of the lowered op's parameters", loc=self.anchor.loc))
        return res

    def add(self, a: Scalar, b: Scalar) -> Scalar:
        a, b = self.value(a), self.value(b)
        if isinstance(a, int) and a == 0:
            return b
        if isinstance(b, int) and b == 0:
            return a
        return self._binop("scalar.add", a, b, lambda x, y: x + y)

    def sub(self, a: Scalar, b: Scalar) -> Scalar:
        a, b = self.value(a), self.value(b)
        if isinstance(b, int) and b == 0:
            return a
        return self._binop("scalar.sub", a, b, lambda x, y: x - y)

    def mul(self, a: Scalar, b: Scalar) -> Scalar:
        a, b = self.value(a), self.value(b)
        if (isinstance(a, int) and a == 1):
            return b
        if (isinstance(b, int) and b == 1):
            return a
        if (isinstance(a, int) and a == 0) or (isinstance(b, int) and b == 0):
            return 0
        return self._binop("scalar.mul", a, b, lambda x, y: x * y)

    def div(self, a: Scalar, b: Scalar) -> Scalar:
        a, b = self.value(a), self.value(b)
        if isinstance(b, int) and b == 1:
            return a
        return self._binop("scalar.div", a, b, lambda x, y: x // y)

    def ceil_div(self, a: Scalar, b: Scalar) -> Scalar:
        a, b = self.value(a), self.value(b)
        if isinstance(b, int) and b == 1:
            return a
        return self._binop("scalar.ceil_div", a, b, lambda x, y: -(-x // y))

    def min(self, a: Scalar, b: Scalar) -> Scalar:
        return self._binop("scalar.min", a, b, min)

    def max(self, a: Scalar, b: Scalar) -> Scalar:
        return self._binop("scalar.max", a, b, max)

    def align(self, a: Scalar, n: int) -> Scalar:
        a = self.value(a)
        if isinstance(a, int):
            return -(-a // n) * n
        return self.mul(self.ceil_div(a, n), n)

    def prod(self, xs: list[Scalar]) -> Scalar:
        acc: Scalar = 1
        for x in xs:
            acc = self.mul(acc, x)
        return acc


def function_names(fn: Function) -> set[str]:
    return {p.name for p in fn.params} | {r.name for op in fn.walk() for r in op.results}


def rewrite_module(module: Module, rw: Rewriter, visit: Any) -> Module:
    """Visit every op of every function bottom-up. ``visit(op, function, emit_factory)`` returns ``None`` to keep the
    op, an :class:`Op`, or a list of ops replacing it; ``emit_factory(anchor)`` gives an :class:`Emit` whose fresh
    names are unique in that function. The result carries the rewriter's ``next_id``."""
    functions = []

    def rewrite_function(f: Function) -> Function:
        """Rewrite one function with names scoped to that function."""
        names = function_names(f)

        def factory(anchor: Op) -> Emit:
            return Emit(rw, f, anchor, names)

        def walk(block: Block) -> Block:
            out: list[Op] = []
            for op in block.ops:
                if op.regions:
                    op = replace(op, regions=tuple(walk(b) for b in op.regions))
                r = visit(op, f, factory)
                if r is None:
                    out.append(op)
                elif isinstance(r, Op):
                    out.append(r)
                else:
                    out.extend(r)
            return Block(tuple(out))

        return replace(f, body=walk(f.body))

    for f in module.functions:
        functions.append(rewrite_function(f))
    attrs = dict(module.attrs)
    attrs["next_id"] = rw._next_id
    return Module(module.name, attrs, tuple(functions))


def retype_values(block: Any, mapping: dict[str, Value]) -> Any:
    """Replace every operand / result / attribute value whose name is in ``mapping`` (regions included); used when
    a pass refines a value's type (an event that received its flag ids, a buffer that received its address)."""
    def fix(x: Any) -> Any:
        if isinstance(x, Value) and x.name in mapping:
            return mapping[x.name]
        if isinstance(x, list):
            return [fix(y) for y in x]
        if isinstance(x, tuple):
            return tuple(fix(y) for y in x)
        return x

    def walk(b: Block) -> Block:
        out = []
        for op in b.ops:
            attrs = {k: fix(v) for k, v in op.attrs.items()}
            op = replace(op, operands=tuple(fix(x) for x in op.operands), results=tuple(fix(r) for r in op.results), attrs=attrs,
                         regions=tuple(walk(r) for r in op.regions))
            out.append(op)
        return Block(tuple(out))

    return walk(block)


def attr_scalar(x: Scalar) -> Any:
    """A scalar as an ``int|value`` attribute."""
    return x


def ident(name: str) -> Ident:
    return Ident(name)


@dataclass
class OpList:
    """Ops produced by one rewrite: the pre-ops then the replacement(s)."""

    pre: list[Op] = field(default_factory=list)
    ops: list[Op] = field(default_factory=list)

    def all(self) -> list[Op]:
        return self.pre + self.ops


__all__ = ["CUBE_PIPES", "I32", "SET_PIPES", "VEC_PIPES", "Defs", "Emit", "OpList", "Scalar", "View", "add_static", "as_int",
           "attr_scalar", "dim_scalar", "elem_type", "function_names", "ident", "is_static", "literal_or_value", "op_side",
           "retype_values", "rewrite_module", "root_dims", "scale_static", "trip_count", "view_of"]
