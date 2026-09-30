# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tensor semantics of the frontend: views, riders, ``<<=`` between memories, explicit DMA and cube ops.

Every tensor value carries a compile-time *geometry* (declared shape, view offset, view span,
which dims were sliced) so the explicit stubs can infer their burst parameters exactly as the
old ``cube.py`` / ``vec/datamove.py`` did, and riders (``.T``, ``.relu()``, ``.requant()``,
``.subblk()``, ``.nz()``, ``.single()``) so the next ``<<=`` can pick the instruction. Plain
``<<=`` between memories compiles to the generic ``dma.copy`` with the riders as attributes; the
device lowering (M4) selects the instruction from the types, as the old ``__ilshift__`` did.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

from ..ir import Ident, Literal
from ..ir.types import (
    BufType,
    CellType,
    Dim,
    DimValue,
    DType,
    MemType,
    Type,
    is_scalar_int,
)
from . import dsl
from .errors import E_BAD_COPY, E_BAD_OPERAND, E_BAD_SHAPE, E_BAD_SIGNATURE, E_UNSUPPORTED
from .values import Dyn, ElemOffset, Img2col, Img2colWindow

LOCAL_SPACES = ("l1", "l0a", "l0b", "l0c", "ub", "bt")


@dataclass
class Geom:
    """Geometry of a tensor value: the old ``shape / offset / span / slice_mask``."""

    shape: tuple[Any, ...]
    offset: tuple[Any, ...]
    span: tuple[Any, ...]
    mask: tuple[bool, ...]
    root: str

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def sliced(self) -> list[int]:
        return [i for i, m in enumerate(self.mask) if m]


def _c0(dt: DType) -> int:
    return 64 if dt.bits < 8 else 32 // (dt.bits // 8)


def _bytes(dt: DType) -> int:
    return max(dt.bits, 8) // 8


class MemRules:
    """Mixin of :class:`FunctionCompiler`."""

    # -- geometry -------------------------------------------------------------------------------

    def geom(self, v: Any, node: ast.AST) -> Geom:
        if isinstance(v, ElemOffset):
            v = v.base
        if not (isinstance(v, Dyn) and isinstance(v.type, MemType)):
            raise self.err(E_BAD_OPERAND, f"expected a tensor, got {v!r}", node)
        g = self.geoms.get(v.name)
        if g is None:
            dims = tuple(self.dim_value(d, node) for d in v.type.dims)
            g = Geom(dims, (0,) * len(dims), dims, (True,) * len(dims), v.name)
            self.geoms[v.name] = g
        return g

    def shape_of(self, v: Any, node: ast.AST) -> tuple[Any, ...]:
        return self.geom(v, node).shape

    def span_of(self, v: Any, node: ast.AST) -> tuple[Any, ...]:
        return self.geom(v, node).span

    def offset_of(self, v: Any, node: ast.AST) -> tuple[Any, ...]:
        return self.geom(v, node).offset

    def _dim_type(self, d: Any) -> Dim:
        if isinstance(d, int):
            return d
        if isinstance(d, Dyn):
            return DimValue(d.name)
        raise TypeError(f"not a dimension: {d!r}")

    def arith(self, op: ast.operator, a: Any, b: Any, node: ast.AST) -> Any:
        """Static-when-possible scalar arithmetic on dims."""
        if isinstance(a, int) and isinstance(b, int) and not isinstance(a, bool):
            if isinstance(op, ast.Add):
                return a + b
            if isinstance(op, ast.Sub):
                return a - b
            if isinstance(op, ast.Mult):
                return a * b
            if isinstance(op, ast.FloorDiv):
                return a // b if b else 0
            raise TypeError(f"unsupported dim arithmetic {type(op).__name__}")
        if isinstance(op, ast.Mult) and (a == 1 or b == 1):
            return b if a == 1 else a
        if isinstance(op, (ast.Add, ast.Sub)) and b == 0:
            return a
        if isinstance(op, ast.Add) and a == 0:
            return b
        return self.binop(op, a, b, node)

    def ceil_div(self, a: Any, b: Any, node: ast.AST) -> Any:
        if isinstance(a, int) and isinstance(b, int):
            return -(-a // b)
        return self.emit("scalar.ceil_div", (a, b), {}, self._promote(a, b, node), None, node)

    def prod(self, xs: list[Any], node: ast.AST) -> Any:
        acc: Any = 1
        for x in xs:
            acc = self.arith(ast.Mult(), acc, x, node)
        return acc

    # -- allocation and views ----------------------------------------------------------------------

    def rule_tensor(self, args: list[Any], kwargs: dict[str, Any], node: ast.AST, slots: int | None) -> Any:
        if len(args) > 5:
            raise self.err(E_BAD_OPERAND, "Tensor and xBuff allocations take at most five positional arguments", node)
        dt = self.need_dtype(args[0] if args else kwargs.get("dtype"), node)
        shape = args[1] if len(args) > 1 else kwargs.get("shape")
        pos = args[2] if len(args) > 2 else kwargs.get("position", dsl.Position.L1)
        if not isinstance(pos, dsl.EnumValue) or pos.family != "Position":
            raise self.err(E_BAD_OPERAND, "the third argument of Tensor is a Position", node)
        if pos.name == "gm":
            raise self.err(E_UNSUPPORTED, "GM tensors are kernel parameters; allocate on-chip tensors only", node)
        if not isinstance(shape, (list, tuple)):
            raise self.err(E_BAD_SHAPE, "shape must be a list", node)
        dims: list[Dim] = []
        vals: list[Any] = []
        for d in shape:
            d = self.rvalue(d, node)
            if isinstance(d, int) and not isinstance(d, bool):
                dims.append(d)
                vals.append(d)
            elif isinstance(d, Dyn) and (is_scalar_int(d.type) or (isinstance(d.type, CellType) and d.type.dtype.is_integer)):
                dims.append(DimValue(d.name))
                vals.append(d)
            else:
                raise self.err(E_BAD_SHAPE, f"shape entries must be ints or integer scalars, got {d!r}", node)
        if len(dims) != 2:
            raise self.err(E_BAD_SHAPE, f"on-chip tensors are two-dimensional, got {len(dims)} dims", node)
        layout = args[4] if len(args) > 4 else kwargs.get("layout")
        layout_name = "nz" if pos.name in ("l1", "l0a", "l0b", "l0c") else None
        if isinstance(layout, dsl.EnumValue) and layout.name in ("nz", "nd"):
            layout_name = layout.name
        t: Type = MemType(pos.name, dt, tuple(dims), layout_name)
        if slots is not None:
            t = BufType(t, slots)  # type: ignore[arg-type]
        name = (args[3] if len(args) > 3 else kwargs.get("name")) or ("buf" if slots else "t")
        attrs = {"name": name} if (len(args) > 3 or kwargs.get("name")) else {}
        if "sync_depth" in kwargs:
            sync_depth = kwargs["sync_depth"]
            if slots is None:
                raise self.err(E_BAD_OPERAND, "sync_depth belongs to DBuff/TBuff/QBuff, not Tensor", node)
            if sync_depth is not None:
                if isinstance(sync_depth, bool) or not isinstance(sync_depth, int):
                    raise self.err(E_BAD_OPERAND, "sync_depth must be a static positive int", node)
                if not 1 <= sync_depth <= slots:
                    raise self.err(E_BAD_OPERAND, f"sync_depth must be in 1..{slots} for this {slots}-slot buffer", node)
                attrs["sync_depth"] = sync_depth
        res = self.emit("mem.alloc", (), attrs, t, name, node)
        self.roots[res.name] = res.name
        if slots is None:
            self.geoms[res.name] = Geom(tuple(vals), (0, 0), tuple(vals), (True, True), res.name)
        return res

    def get_buf(self, base: Dyn, index: Any, node: ast.AST) -> Dyn:
        t = base.type
        assert isinstance(t, BufType)
        res = self.emit("mem.get_buf", (base, index), {}, t.elem, "slot", node)
        dims = tuple(self.dim_value(d, node) for d in t.elem.dims)
        self.geoms[res.name] = Geom(dims, (0,) * len(dims), dims, (True,) * len(dims), res.name)
        self.roots[res.name] = self.roots.get(base.name, base.name)
        return res

    def slice_tensor(self, base: Dyn, parts: list[ast.expr], node: ast.AST) -> Any:
        """``t[a:b, c:d]`` / ``gm[i, a:b, :]`` -> ``mem.slice``; ``t[k]`` on a local tensor -> element offset."""
        t = base.type
        assert isinstance(t, MemType)
        g = self.geom(base, node)
        if len(parts) == 1 and not isinstance(parts[0], ast.Slice) and (t.is_local or self.kind == "simt"):
            return ElemOffset(base, self.rvalue(self.ev(parts[0]), node))
        if len(parts) != g.rank:
            raise self.err(E_BAD_SHAPE, f"index has {len(parts)} dimension(s), tensor has {g.rank}", node)
        if base.riders.transpose:
            raise self.err(E_UNSUPPORTED, "a transposed view cannot be sliced; slice first, then .T", node)
        offsets: list[Any] = []
        extents: list[Any] = []
        mask: list[bool] = []
        for i, p in enumerate(parts):
            if isinstance(p, ast.Slice):
                if p.step is not None:
                    raise self.err(E_UNSUPPORTED, "slice steps are not supported", node)
                lo = self.rvalue(self.ev(p.lower), node) if p.lower is not None else 0
                hi = self.rvalue(self.ev(p.upper), node) if p.upper is not None else g.span[i]
                ext = hi if (p.lower is None) else self.arith(ast.Sub(), hi, lo, node)
                mask.append(True)
            else:
                lo = self.rvalue(self.ev(p), node)
                ext = 1
                mask.append(False)
            offsets.append(lo)
            extents.append(ext)
        if not t.is_local and sum(mask) == 0:
            raise self.err(E_UNSUPPORTED, "index at least one dimension of a GM tensor with a slice", node)
        if not t.is_local and sum(mask) > 2:
            raise self.err(E_UNSUPPORTED, "at most two dimensions of a GM tensor may be sliced", node)
        identity = all(m and (isinstance(o, int) and o == 0) and (e is g.span[i] or (isinstance(e, int) and e == g.span[i]))
                       for i, (o, e, m) in enumerate(zip(offsets, extents, mask, strict=True)))
        if identity:
            return base
        abs_off = tuple(self.arith(ast.Add(), g.offset[i], offsets[i], node) for i in range(g.rank))
        kept = [i for i, m in enumerate(mask) if m] if not t.is_local else list(range(g.rank))
        view_dims = tuple(self._dim_type(extents[i]) for i in kept)
        view = MemType(t.space, t.dtype, view_dims, t.layout)
        attrs: dict[str, Any] = {"offsets": [self.attr(o) for o in offsets], "extents": [self.attr(e) for e in extents]}
        if not all(mask):
            attrs["mask"] = [int(m) for m in mask]
        res = self.emit("mem.slice", (base.plain(),), attrs, view, "view", node)
        self.roots[res.name] = self.roots.get(base.name, base.name)
        self.geoms[res.name] = Geom(g.shape, abs_off, tuple(extents), tuple(mask), g.root)
        return Dyn(res.value, base.riders)

    def reinterpret_view(self, src: Dyn, dt: DType, name: str, node: ast.AST) -> Dyn:
        t = src.type
        assert isinstance(t, MemType)
        g = self.geom(src, node)
        packed = dt.bits < 8
        if t.is_local and t.space == "l0c":
            raise self.err(E_UNSUPPORTED, "L0C tensors cannot be reinterpreted", node)
        ratio_num, ratio_den = _c0(dt), _c0(t.dtype)
        shape, span, offset = list(g.shape), list(g.span), list(g.offset)
        attrs: dict[str, Any] = {}
        if t.is_local and (ratio_num != ratio_den):
            axis = 0 if (packed and src.riders.transpose) else (g.rank - 1)
            for arr in (shape, span, offset):
                arr[axis] = self.arith(ast.FloorDiv(), self.arith(ast.Mult(), arr[axis], ratio_num, node), ratio_den, node)
            attrs["packed_axis"] = axis
            attrs["shape"] = [self.attr(x) for x in shape]
        elif not t.is_local and _bytes(dt) != _bytes(t.dtype):
            raise self.err(E_UNSUPPORTED, f"a GM tensor can only be reinterpreted to a dtype of the same width ({t.dtype} -> {dt})", node)
        dims = tuple(self._dim_type(d) for d in (span if t.is_local else [span[i] for i in g.sliced] or span))
        if not t.is_local:
            dims = t.dims
        view = MemType(t.space, dt, dims, t.layout)
        res = self.emit("mem.reinterpret", (src.plain(),), attrs, view, name or "view", node)
        self.roots[res.name] = self.roots.get(src.name, src.name)
        self.geoms[res.name] = Geom(tuple(shape), tuple(offset), tuple(span), g.mask, g.root)
        return Dyn(res.value, src.riders)

    def reshape_view(self, src: Dyn, shape: Any, name: str, node: ast.AST) -> Dyn:
        t = src.type
        assert isinstance(t, MemType)
        if t.is_local:
            raise self.err(E_UNSUPPORTED, "reshape applies to GM tensors", node)
        g = self.geom(src, node)
        if any(not (isinstance(o, int) and o == 0) for o in g.offset) or not all(g.mask):
            raise self.err(E_UNSUPPORTED, "only a whole GM tensor can be reshaped (not a slice)", node)
        if not isinstance(shape, (list, tuple)):
            raise self.err(E_BAD_SHAPE, "reshape needs a list of dims", node)
        vals = [self.rvalue(d, node) for d in shape]
        for d in vals:
            if not (isinstance(d, int) or (isinstance(d, Dyn) and (is_scalar_int(d.type) or isinstance(d.type, CellType)))):
                raise self.err(E_BAD_SHAPE, f"reshape dims must be ints or integer scalars, got {d!r}", node)
        if all(isinstance(d, int) for d in vals) and all(isinstance(d, int) for d in g.shape) and \
                self.prod(list(vals), node) != self.prod(list(g.shape), node):
            raise self.err(E_BAD_SHAPE, f"reshape {list(g.shape)} -> {vals} changes the element count", node)
        view = MemType(t.space, t.dtype, tuple(self._dim_type(d) for d in vals), t.layout)
        res = self.emit("mem.reshape", (src.plain(),), {"shape": [self.attr(d) for d in vals]}, view, name or "view", node)
        self.roots[res.name] = self.roots.get(src.name, src.name)
        self.geoms[res.name] = Geom(tuple(vals), (0,) * len(vals), tuple(vals), (True,) * len(vals), g.root)
        return Dyn(res.value)

    def strided_view(self, src: Dyn, args: list[Any], kwargs: dict[str, Any], node: ast.AST) -> Dyn:
        """``gm.view(shape, strides=None, offset=0)`` (RFC-0010): the same GM bytes re-described with a
        new shape, explicit element strides and an element offset - zero instructions, DMA-only
        consumption. On a whole GM parameter or workspace the quantities may be runtime scalars; a
        view of a view / slice / reshape / reinterpret **composes**: the chain folds to one
        ``mem.view`` on the root at compile time (over a strided window everything must be static and
        the new rows must tile the source rows; over a contiguous window the fold is a displacement
        and runtime scalars stay allowed). Rank is 1 to 4 (adjacent contiguous dims merge; a rank > 2
        residue needs a unit innermost stride and only the GM->UB NDDMA read path moves it). A
        non-unit innermost stride is legal but read-only - NDDMA is the one engine that gathers
        per-element; the write engines burst contiguous runs and refuse at lowering. Nothing is
        negative, GMList members stay out (maintainer rulings 2026-08-30)."""
        t = src.type
        assert isinstance(t, MemType)
        if t.is_local:
            raise self.err(E_UNSUPPORTED, ".view() applies to GM tensors", node)
        if t.dtype.bits < 8:
            raise self.err(E_UNSUPPORTED, ".view() of a packed sub-byte tensor is not supported - declare the "
                                          "packed plane as its u8 carrier and view that (the carrier byte is the "
                                          "hardware's own addressing unit; the corpus convention for fp4/int4 planes)", node)
        d = self._producer(src.name)
        if d is not None and d.opcode == "list.item":
            raise self.err(E_UNSUPPORTED, ".view() of a GMList member is not supported (maintainer ruling)", node)
        composing = d is not None and d.opcode != "mem.workspace"
        if not composing:
            g = self.geom(src, node)
            if any(not (isinstance(o, int) and o == 0) for o in g.offset) or not all(g.mask):
                raise self.err(E_UNSUPPORTED, "only a whole GM tensor can be viewed (slice the view instead)", node)
        shape = args[0] if args else kwargs.get("shape")
        if not isinstance(shape, (list, tuple)) or not 1 <= len(shape) <= 4:
            raise self.err(E_BAD_SHAPE, "view needs a shape list of one to four dims", node)
        shape_v = [self.rvalue(x, node) for x in shape]
        strides = args[1] if len(args) > 1 else kwargs.get("strides")
        if strides is None:  # row-major contiguous over the new shape
            strides_v = [1]
            for d in reversed(shape_v[1:]):
                strides_v.insert(0, self.arith(ast.Mult(), strides_v[0], d, node))
        else:
            if not isinstance(strides, (list, tuple)) or len(strides) != len(shape):
                raise self.err(E_BAD_SHAPE, "view strides need one entry per shape dim", node)
            strides_v = [self.rvalue(x, node) for x in strides]
        offset = args[2] if len(args) > 2 else kwargs.get("offset", 0)
        offset_v = self.rvalue(offset, node)
        for x in (*shape_v, *strides_v, offset_v):
            if not (isinstance(x, int) or (isinstance(x, Dyn) and (is_scalar_int(x.type) or isinstance(x.type, CellType)))):
                raise self.err(E_BAD_OPERAND, f"view dims, strides and offset must be ints or integer scalars, got {x!r}", node)
            if isinstance(x, int) and x < 0:
                raise self.err(E_BAD_OPERAND, "view shape, strides and offset must be non-negative (maintainer ruling)", node)
        if not (isinstance(strides_v[-1], int) and strides_v[-1] >= 1):
            raise self.err(E_UNSUPPORTED, "the innermost view stride must be a static positive int (it selects between "
                                          "the burst path and the per-element NDDMA read path)", node)
        # canonicalize rank > 2 only: merge adjacent contiguous dims until the window is 2D or
        # genuinely deeper - never below the user's declared rank (the result type keeps its shape)
        while len(shape_v) > 2 and all(isinstance(x, int) for x in (shape_v[-1], shape_v[-2], strides_v[-1], strides_v[-2])) \
                and strides_v[-2] == shape_v[-1] * strides_v[-1]:
            shape_v[-2:] = [shape_v[-2] * shape_v[-1]]
            strides_v[-2:] = [strides_v[-1]]
        if len(shape_v) > 2 and strides_v[-1] != 1:
            raise self.err(E_UNSUPPORTED, "a rank > 2 view needs a unit innermost stride", node)
        if composing:
            root, off1, sr1, C1, R1 = self._linear_window(src, node)
            statics = all(isinstance(x, int) for x in (*shape_v, *strides_v, offset_v))
            if statics:
                reach = offset_v + sum((s - 1) * st for s, st in zip(shape_v, strides_v)) + 1
                if reach > R1 * C1:
                    raise self.err(E_BAD_SHAPE, f"view reaches element {reach - 1} of the {R1 * C1}-element source window", node)
            if sr1 == C1 and R1 == 1:  # a contiguous source window: the fold is a plain displacement
                offset_v = self.arith(ast.Add(), off1, offset_v, node)
            else:
                if not statics:
                    raise self.err(E_UNSUPPORTED, "composing .view() over a strided window needs static shape, strides and offset", node)
                if len(shape_v) > 2:
                    raise self.err(E_UNSUPPORTED, "a rank > 2 view of a strided window is not supported (view the root instead)", node)
                inner_reach = (shape_v[-1] - 1) * strides_v[-1] + 1
                if offset_v % C1 + inner_reach > C1:
                    raise self.err(E_UNSUPPORTED, f"the view's rows cross a row of the source window (start column "
                                                  f"{offset_v % C1}, reach {inner_reach} in {C1}-column rows)", node)
                if len(shape_v) == 2:
                    if strides_v[0] % C1:
                        raise self.err(E_UNSUPPORTED, f"the view's row stride ({strides_v[0]}) must be a whole number of "
                                                      f"source rows ({C1} columns each) to compose", node)
                    strides_v = [strides_v[0] // C1 * sr1, strides_v[1]]
                offset_v = off1 + offset_v // C1 * sr1 + offset_v % C1
            src = root
        g = self.geom(src, node)
        if all(isinstance(x, int) for x in (*shape_v, *strides_v, offset_v)) and all(isinstance(x, int) for x in g.shape):
            numel = self.prod(list(g.shape), node)
            last = offset_v + sum((int(s) - 1) * int(st) for s, st in zip(shape_v, strides_v)) + 1
            if last > numel:
                raise self.err(E_BAD_SHAPE, f"view reaches element {last - 1} of a {numel}-element tensor", node)
        view = MemType(t.space, t.dtype, tuple(self._dim_type(x) for x in shape_v), t.layout)
        attrs = {"shape": [self.attr(x) for x in shape_v], "strides": [self.attr(x) for x in strides_v], "offset": self.attr(offset_v)}
        res = self.emit("mem.view", (src.plain(),), attrs, view, kwargs.get("name") or "view", node)
        self.roots[res.name] = self.roots.get(src.name, src.name)
        self.geoms[res.name] = Geom(tuple(shape_v), (0,) * len(shape_v), tuple(shape_v), (True,) * len(shape_v), g.root)
        return Dyn(res.value)

    def _static_int(self, x: Any, what: str, node: ast.AST) -> int:
        if isinstance(x, Literal):
            x = x.value
        if isinstance(x, int) and not isinstance(x, bool):
            return int(x)
        raise self.err(E_UNSUPPORTED, f"composing .view() needs static {what}, got {x!r}", node)

    def _linear_window(self, v: Dyn, node: ast.AST) -> tuple[Dyn, int, int, int, int]:
        """Fold the GM view chain under ``v`` to ``(root, offset, pitch, cols, rows)`` - a static 2D
        window in the root's element stream; ``pitch == cols and rows == 1`` marks a contiguous run.
        A same-width GM ``mem.reinterpret`` folds away; ``mem.reshape`` re-addresses its window's
        storage as one contiguous run from its first element, as the printers do (RFC-0010 §10);
        ``mem.view`` and ``mem.slice`` contribute their geometry (RFC-0010 §9 phase 2)."""
        d = self._producer(v.name)
        if d is None or d.opcode == "mem.workspace":
            g = self.geom(v, node)
            if not all(isinstance(x, int) for x in g.shape):
                raise self.err(E_UNSUPPORTED, "composing .view() over a symbolic-shaped tensor is not supported", node)
            n = 1
            for x in g.shape:
                n *= int(x)
            return v, 0, n, n, 1
        if d.opcode == "list.item":
            raise self.err(E_UNSUPPORTED, ".view() of a GMList member is not supported (maintainer ruling)", node)
        if d.opcode == "mem.reshape":
            root, off, _, cols, rows = self._linear_window(Dyn(d.operands[0]), node)
            return root, off, rows * cols, rows * cols, 1  # a strided window's rows become one run from its start
        if d.opcode == "mem.reinterpret":
            st, dt = d.operands[0].type, v.type
            assert isinstance(st, MemType) and isinstance(dt, MemType)
            if max(st.dtype.bits, 8) != max(dt.dtype.bits, 8):
                raise self.err(E_UNSUPPORTED, "composing .view() over a width-changing reinterpret is not supported", node)
            return self._linear_window(Dyn(d.operands[0]), node)
        if d.opcode == "mem.view":
            shape = [self._static_int(x, "view shape", node) for x in d.attrs["shape"]]
            strides = [self._static_int(x, "view strides", node) for x in d.attrs["strides"]]
            off = self._static_int(d.attrs.get("offset", 0), "view offset", node)
            root, base_off, bp, bc, br = self._linear_window(Dyn(d.operands[0]), node)
            if not (bp == bc and br == 1):
                raise self.err(E_UNSUPPORTED, "unexpected strided base under a view", node)  # emitted views sit on roots
            off += base_off
            if strides[-1] != 1:
                raise self.err(E_UNSUPPORTED, "composing .view() over a non-unit-innermost view is not supported", node)
            if len(shape) > 2:
                raise self.err(E_UNSUPPORTED, "composing .view() over a rank > 2 view is not supported", node)
            if len(shape) == 1 or strides[0] == shape[-1]:
                n = shape[0] * (shape[1] if len(shape) == 2 else 1)
                return root, off, n, n, 1
            return root, off, strides[0], shape[1], shape[0]
        if d.opcode == "mem.slice":
            if any(not bool(m) for m in d.attrs.get("mask", []) or []):
                raise self.err(E_UNSUPPORTED, "composing .view() over an indexed slice is not supported", node)
            steps = d.attrs.get("steps")
            if steps and any(not (isinstance(s, int) and s == 1) for s in steps):
                raise self.err(E_UNSUPPORTED, "composing .view() over a stepped slice is not supported", node)
            offs = [self._static_int(x, "slice offsets", node) for x in d.attrs["offsets"]]
            exts = [self._static_int(x, "slice extents", node) for x in d.attrs["extents"]]
            root, base_off, bp, bc, br = self._linear_window(Dyn(d.operands[0]), node)
            st = d.operands[0].type
            assert isinstance(st, MemType)
            if len(offs) == 1:
                return root, base_off + offs[0], exts[0], exts[0], 1
            if len(offs) != 2:
                raise self.err(E_UNSUPPORTED, "composing .view() supports rank-1 and rank-2 slices only", node)
            if bp == bc and br == 1:  # contiguous base: the slice indexes the declared 2D coordinates
                dims = [x if isinstance(x, int) else None for x in (self.dim_value(dd, node) for dd in st.dims)]
                if any(x is None for x in dims) or len(dims) != 2:
                    raise self.err(E_UNSUPPORTED, "composing .view() over a slice of a symbolic or higher-rank tensor is not supported", node)
                pitch = dims[1]
            else:  # a strided window: the slice indexes the window's own (rows, cols)
                pitch = bp
            return root, base_off + offs[0] * pitch + offs[1], pitch, exts[1], exts[0]
        raise self.err(E_UNSUPPORTED, f".view() cannot compose over {d.opcode}", node)

    def _producer(self, name: str) -> Any:
        for frame in self.fb._stack:  # noqa: SLF001 - the builder's op frames, innermost last
            for op in frame:
                for r in op.results:
                    if r.name == name:
                        return op
        return None

    def layout_view(self, src: Dyn, layout: str, node: ast.AST) -> Dyn:
        t = src.type
        assert isinstance(t, MemType)
        if t.space != "ub":
            raise self.err(E_UNSUPPORTED, f".{layout}() applies to UB tensors", node)
        if t.layout == layout:
            return src
        view = MemType(t.space, t.dtype, t.dims, layout)
        res = self.emit("mem.reinterpret", (src.plain(),), {"layout": Ident(layout)}, view, "view", node)
        self.roots[res.name] = self.roots.get(src.name, src.name)
        self.geoms[res.name] = self.geom(src, node)
        return Dyn(res.value, src.riders)

    def rule_workspace(self, args: list[Any], kwargs: dict[str, Any], node: ast.AST) -> Any:
        dt = self.need_dtype(args[0] if args else kwargs.get("dtype"), node)
        shape = args[1] if len(args) > 1 else kwargs.get("shape")
        name = (args[2] if len(args) > 2 else kwargs.get("name")) or "ws"
        if not isinstance(shape, (list, tuple)):
            raise self.err(E_BAD_SHAPE, "split_workspace needs a shape list", node)
        vals = [self.rvalue(d, node) for d in shape]
        numel = self.prod(list(vals), node)
        t = MemType("ws", dt, tuple(self._dim_type(d) for d in vals))
        res = self.emit("mem.workspace", (), {"name": name, "numel": self.attr(numel)}, t, name, node)
        self.roots[res.name] = res.name
        self.geoms[res.name] = Geom(tuple(vals), (0,) * len(vals), tuple(vals), (True,) * len(vals), res.name)
        return res

    def rule_gmbuff(self, args: list[Any], kwargs: dict[str, Any], node: ast.AST) -> Any:
        """``GMBuff(dtype, shape, slots=N, name='', per_core=True)``: a GM workspace ring (RFC-0009).

        Allocates ``[cube_num?, slots, *shape]`` workspace elements but types the result as a slot
        buffer of ``shape`` tensors, so ``ws[beat]`` selects a slot. The ``gmbuff`` pass checks the
        ring algebra and rewrites the slot selections into the plain workspace + ``var_mod`` views.
        """
        dt = self.need_dtype(args[0] if args else kwargs.get("dtype"), node)
        shape = args[1] if len(args) > 1 else kwargs.get("shape")
        if not isinstance(shape, (list, tuple)) or len(shape) != 2:
            raise self.err(E_BAD_SHAPE, "GMBuff needs a two-dimensional slot shape list", node)
        slots = args[2] if len(args) > 2 else kwargs.get("slots")
        if not isinstance(slots, int) or isinstance(slots, bool) or slots < 1:
            raise self.err(E_BAD_OPERAND, "GMBuff slots must be a static positive int", node)
        name = (args[3] if len(args) > 3 else kwargs.get("name")) or "gmws"
        per_core = kwargs.get("per_core", True)
        if not isinstance(per_core, bool):
            raise self.err(E_BAD_OPERAND, "GMBuff per_core must be a static bool", node)
        shape_vals = [self.rvalue(d, node) for d in shape]
        vals: list[Any] = [slots, *shape_vals]
        if per_core:
            from ..ir.types import DTYPES, ScalarType

            core_n = self.emit("core.cube_num", (), {}, ScalarType(DTYPES["i32"]), None, node)
            vals.insert(0, core_n)
        numel = self.prod(list(vals), node)
        elem = MemType("ws", dt, tuple(self._dim_type(d) for d in shape_vals))
        t = BufType(elem, slots)
        attrs = {"name": name, "numel": self.attr(numel), "gmbuff_slots": slots,
                 "gmbuff_per_core": int(per_core), "gmbuff_dims": [self.attr(v) for v in vals]}
        res = self.emit("mem.workspace", (), attrs, t, name, node)
        self.roots[res.name] = res.name
        return res

    # -- attributes and methods ---------------------------------------------------------------------

    def tensor_attribute(self, base: Dyn, attr: str, node: ast.AST) -> Any:
        t = base.type
        assert isinstance(t, MemType)
        g = self.geom(base, node)
        if attr == "T":
            if not t.is_local and g.rank != 2 and len(g.sliced) != 2:
                raise self.err(E_UNSUPPORTED, ".T needs a two-dimensional view", node)
            if t.is_local and t.space not in ("l1", "l0c", "l0a", "l0b"):
                raise self.err(E_UNSUPPORTED, f".T applies to L1 / L0C tensors, not {t.space}", node)
            return base.with_riders(transpose=not base.riders.transpose)
        if attr == "shape":
            return list(g.shape)
        if attr == "span":
            return list(g.span)
        if attr == "offset":
            return list(g.offset)
        if attr == "dtype":
            from .compiler import _dtype_name

            return _dtype_name(t.dtype)
        if attr == "position":
            return getattr(dsl.Position, t.space.upper())
        if attr == "name":
            return base.name
        if attr == "is_transpose":
            return base.riders.transpose
        from .values import BoundMethod

        return BoundMethod(base, attr)

    def tensor_method(self, obj: Dyn, name: str, args: list[Any], kwargs: dict[str, Any], node: ast.AST) -> Any:
        t = obj.type
        assert isinstance(t, MemType)
        if name == "reinterpret":
            dt = self.need_dtype(args[0] if args else kwargs.get("target_dtype", kwargs.get("dtype")), node)
            return self.reinterpret_view(obj, dt, (args[1] if len(args) > 1 else kwargs.get("name")) or "", node)
        if name == "reshape":
            shape = args[0] if args else kwargs.get("shape")
            if len(args) > 1 and not isinstance(args[0], (list, tuple)):
                shape = list(args)
            return self.reshape_view(obj, shape, kwargs.get("name") or "", node)
        if name == "flatten":
            g = self.geom(obj, node)
            start = args[0] if args else kwargs.get("start_dim", 0)
            end = args[1] if len(args) > 1 else kwargs.get("end_dim", -1)
            if end < 0:
                end += g.rank
            merged = list(g.shape[:start]) + [self.prod(list(g.shape[start:end + 1]), node)] + list(g.shape[end + 1:])
            return self.reshape_view(obj, merged, kwargs.get("name") or "", node)
        if name in ("nz", "nd"):
            return self.layout_view(obj, name, node)
        if name == "view":
            return self.strided_view(obj, args, kwargs, node)
        if name == "requant":
            if t.space != "l0c":
                raise self.err(E_UNSUPPORTED, ".requant() applies to L0C tensors", node)
            scale = self.rvalue(args[0] if args else kwargs.get("scale", 1.0), node)
            offset = self.rvalue(args[1] if len(args) > 1 else kwargs.get("offset", 0), node)
            hybrid = bool(args[2] if len(args) > 2 else kwargs.get("hif8_hybrid", False))
            return obj.with_riders(scale=scale, offset=offset, hif8_hybrid=hybrid)
        if name == "subblk":
            if t.space != "l0c":
                raise self.err(E_UNSUPPORTED, ".subblk() applies to L0C tensors", node)
            return obj.with_riders(subblk=self.rvalue(args[0] if args else kwargs.get("sub_block_id"), node))
        if name == "relu":
            if t.space != "l0c":
                raise self.err(E_UNSUPPORTED, ".relu() applies to L0C tensors (vector math lives in vf functions)", node)
            return obj.with_riders(relu=True)
        if name in ("single", "brcb", "upsample", "downsample", "unpack", "unpack4"):
            if t.space != "ub":
                raise self.err(E_UNSUPPORTED, f".{name}() applies to UB tensors", node)
            return obj.with_riders(mode=name)
        if name == "set_shape":
            raise self.err(E_UNSUPPORTED, "set_shape is not supported; allocate the tensor with the shape it needs", node)
        raise self.err(E_UNSUPPORTED, f"{obj} has no method {name!r}", node)

    # -- <<= between memories ------------------------------------------------------------------------

    def copy_memory(self, dst: Dyn, src: Dyn, node: ast.AST) -> None:
        """``tensor <<= tensor``: the generic copy with the riders as attributes (old ``__ilshift__``)."""
        dt, st = dst.type, src.type
        assert isinstance(dt, MemType) and isinstance(st, MemType)
        attrs: dict[str, Any] = {}
        r = src.riders
        if dst.riders.transpose:
            raise self.err(E_BAD_COPY, "a transposed view cannot be the target of <<=", node)
        if r.transpose:
            if not ((dt.space == "l1" and not st.is_local and dt.layout == "nz") or (dt.space == "ub" and not st.is_local and dt.layout != "nz")
                    or (st.space == "l1" and dt.space in ("l0a", "l0b")) or (not dt.is_local and st.space == "l0c")):
                raise self.err(E_BAD_COPY, f"a transposed source is only supported for gm -> l1 (nz), gm -> ub, l1 -> l0a/l0b and l0c -> gm, not {st.space} -> {dt.space}", node)
            attrs["transpose"] = True
        if st.space == "l0c":
            if r.relu:
                attrs["relu"] = True
            if r.scale is not None and (r.hif8_hybrid or not (isinstance(r.scale, (int, float)) and r.scale == 1)):
                attrs["scale"] = self.attr(r.scale)
            if r.offset is not None and not (isinstance(r.offset, int) and r.offset == 0):
                attrs["offset"] = self.attr(r.offset)
            if r.hif8_hybrid:
                attrs["hif8_hybrid"] = True
            if dt.space == "ub":
                if r.subblk is not None:
                    attrs["dual_mode"] = Ident("single")
                    attrs["sub_block_id"] = self.attr(r.subblk)
                elif r.has_requant:
                    raise self.err(E_BAD_COPY, "an L0C -> UB store with a requant/relu rider writes one vector subblock; pick it with .subblk(0) / .subblk(1)", node)
                else:
                    attrs["dual_mode"] = Ident("splitm")
                    if dt.dtype != st.dtype:
                        raise self.err(E_BAD_COPY, f"an L0C -> UB split copy keeps the dtype ({st.dtype} -> {dt.dtype} needs .subblk() and a requant rider)", node)
        elif r.has_requant or r.subblk is not None:
            raise self.err(E_BAD_COPY, "requant / relu / subblk riders apply to L0C sources", node)
        if not dt.is_local and self.atomic is not None:
            attrs["atomic"] = self.atomic
        if dt.space in ("l0a", "l0b"):
            if st.space != "l1":
                raise self.err(E_BAD_COPY, f"L0A / L0B are loaded from L1, not {st.space}", node)
            span = self.span_of(src, node)
            self.geoms[dst.name] = Geom(tuple(span), (0, 0), tuple(span), (True, True), self.geom(dst, node).root)
            self.l0_transposed[dst.name] = r.transpose
            attrs.pop("transpose", None)
            if r.transpose:
                attrs["transpose"] = True
        self.emit("dma.copy", (dst.plain(), src.plain()), attrs, None, None, node)

    # -- explicit DMA / cube instructions ------------------------------------------------------------

    def iv(self, v: Any, node: ast.AST) -> Any:
        return self.attr(self.rvalue(v, node))

    def rule_sort(self, rule: str, callee: Any, args: list[Any], kwargs: dict[str, Any], node: ast.AST) -> Any:
        """``sort32(dst, src, idx, repeat=None)``, ``mergesort4(dst, src, length_per_seq, repeat=1)`` and
        ``mergesort_2seq(dst, src1, src2, size1, size2)`` over fp32 (score, index) records in UB; a missing ``repeat``
        of ``sort32`` is the number of 32-score groups in ``src`` (the old vec/sort.py)."""
        if self.kind != "kernel":
            raise self.err(E_UNSUPPORTED, f"{callee.__name__} is only available in kernel functions", node)
        a, kw = list(args), dict(kwargs)

        def arg(i: int, key: str, default: Any = None) -> Any:
            return a[i] if len(a) > i else kw.get(key, default)

        def ub(i: int, what: str) -> Dyn:
            v = arg(i, what)
            if isinstance(v, ElemOffset):
                v = v.base
            if not (isinstance(v, Dyn) and isinstance(v.type, MemType) and v.type.space == "ub"):
                raise self.err(E_BAD_OPERAND, f"{callee.__name__}: {what} must be a UB tensor, got {v!r}", node)
            if v.type.dtype.name != ("u32" if what == "idx" else "f32"):
                raise self.err(E_BAD_OPERAND, f"{callee.__name__}: {what} must be {'uint32' if what == 'idx' else 'fp32'}, got {v.type.dtype}", node)
            return v

        if rule == "vec.sort32":
            dst, src, idx = ub(0, "dst"), ub(1, "src"), ub(2, "idx")
            repeat = arg(3, "repeat")
            if repeat is None:
                numel = 1
                for d in self.shape_of(src, node):
                    numel = self.arith(ast.Mult(), numel, d, node)
                repeat = self.arith(ast.FloorDiv(), numel, 32, node)
            return self.emit(rule, (dst.plain(), src.plain(), idx.plain()), {"repeat": self.iv(repeat, node)}, None, None, node)
        if rule == "vec.mergesort4":
            dst, src = ub(0, "dst"), ub(1, "src")
            attrs = {"length_per_seq": self.iv(arg(2, "length_per_seq"), node), "repeat": self.iv(arg(3, "repeat", 1), node)}
            return self.emit(rule, (dst.plain(), src.plain()), attrs, None, None, node)
        dst, src1, src2 = ub(0, "dst"), ub(1, "src1"), ub(2, "src2")
        attrs = {"size1": self.iv(arg(3, "size1"), node), "size2": self.iv(arg(4, "size2"), node)}
        return self.emit(rule, (dst.plain(), src1.plain(), src2.plain()), attrs, None, None, node)

    def rule_dma(self, rule: str, callee: Any, args: list[Any], kwargs: dict[str, Any], node: ast.AST) -> Any:
        if self.kind != "kernel":
            raise self.err(E_UNSUPPORTED, f"{callee.__name__} is only available in kernel functions", node)
        op, _, variant = rule.partition(":")
        a = list(args)
        kw = dict(kwargs)

        def arg(i: int, key: str, default: Any = None) -> Any:
            return a[i] if len(a) > i else kw.get(key, default)

        def need_tensor(v: Any, spaces: tuple[str, ...], what: str) -> Dyn:
            if isinstance(v, ElemOffset):
                v = v.base
            if not (isinstance(v, Dyn) and isinstance(v.type, MemType) and (v.type.space in spaces or ("gm" in spaces and not v.type.is_local))):
                raise self.err(E_BAD_OPERAND, f"{callee.__name__}: {what} must be a {'/'.join(spaces).upper()} tensor, got {v!r}", node)
            return v

        dst, src = arg(0, "dst"), arg(1, "src")
        name = op.removeprefix("dma.")
        if name == "gm_to_l1.pad" or name == "gm_to_ub.pad":
            dst = need_tensor(dst, ("l1",) if name.startswith("gm_to_l1") else ("ub",), "dst")
            src = need_tensor(src, ("gm",), "src")
            n_burst, burst, sstride = arg(2, "n_burst"), arg(3, "burst_len_element"), arg(4, "src_stride_element")
            if n_burst is None or burst is None or sstride is None:
                n_burst, burst, sstride = self.infer_gm_transfer(src, node)
            dstride = arg(5, "dst_stride")
            if dstride is None:
                if name == "gm_to_ub.pad":
                    g = self.geom(src, node)
                    dstride = 0 if g.rank == 1 else self.arith(ast.FloorDiv(), self.arith(ast.Sub(), self.shape_of(dst, node)[1], burst, node), _c0(dst.type.dtype), node)
                else:
                    dstride = 0
            size = _bytes(src.type.dtype)
            attrs = {"n_burst": self.iv(n_burst, node), "burst_len_byte": self.iv(self.arith(ast.Mult(), burst, size, node), node),
                     "src_stride_byte": self.iv(self.arith(ast.Mult(), sstride, size, node), node), "dst_stride": self.iv(dstride, node)}
            pad = arg(6, "pad")
            if pad is not None:
                if name != "gm_to_ub.pad":
                    raise self.err(E_BAD_OPERAND, f"{callee.__name__}: an explicit pad value is only "
                                                  "wired for gm_to_ub_pad", node)
                if not isinstance(pad, (int, float, bool)) or isinstance(pad, Dyn):
                    raise self.err(E_BAD_OPERAND, f"{callee.__name__}: pad must be a literal (the SPR "
                                                  "takes the value's bit pattern, folded at compile time)", node)
                attrs["pad"] = pad
            self.emit(op, (dst.plain(), src.plain()), attrs, None, None, node)
            return None
        if name == "ub_to_gm.pad":
            dst = need_tensor(dst, ("gm",), "dst")
            src = need_tensor(src, ("ub",), "src")
            n_burst, burst, dstride_e = arg(2, "n_burst"), arg(3, "burst_len_element"), arg(5, "dst_stride_element")
            if n_burst is None or burst is None or dstride_e is None:
                n_burst, burst, dstride_e = self.infer_gm_transfer(dst, node)
            sstride = arg(4, "src_stride")
            if sstride is None:
                g = self.geom(dst, node)
                sstride = 0 if g.rank == 1 else self.arith(ast.FloorDiv(), self.arith(ast.Sub(), self.shape_of(src, node)[1], burst, node), _c0(src.type.dtype), node)
            size = _bytes(src.type.dtype)
            attrs = {"n_burst": self.iv(n_burst, node), "burst_len_byte": self.iv(self.arith(ast.Mult(), burst, size, node), node),
                     "src_stride": self.iv(sstride, node), "dst_stride_byte": self.iv(self.arith(ast.Mult(), dstride_e, size, node), node)}
            if self.atomic is not None:
                attrs["atomic"] = self.atomic
            self.emit(op, (dst.plain(), src.plain()), attrs, None, None, node)
            return None
        if name in ("gm_to_l1.nd2nz", "gm_to_l1.dn2nz"):
            dst = need_tensor(dst, ("l1",), "dst")
            src = need_tensor(src, ("gm",), "src")
            M, N, N_src = arg(2, "M"), arg(3, "N"), arg(4, "N_src")
            if M is None or N is None or N_src is None:
                g = self.geom(src, node)
                sl = g.sliced
                if len(sl) == 2:
                    if name.endswith("nd2nz"):
                        M, N, N_src = g.span[sl[0]], g.span[sl[1]], g.shape[sl[1]]
                    else:
                        M, N, N_src = g.span[sl[1]], g.span[sl[0]], g.shape[sl[1]]
                elif len(sl) == 1:
                    M, N, N_src = (1, g.span[sl[0]], g.shape[sl[0]]) if name.endswith("nd2nz") else (g.span[sl[0]], 1, g.shape[sl[0]])
                else:
                    raise self.err(E_BAD_SHAPE, f"{callee.__name__}: the GM source must be sliced", node)
            M_dst = arg(5, "M_dst")
            if M_dst is None:
                M_dst = self.shape_of(dst, node)[0]
            self.emit(op, (dst.plain(), src.plain()), {"M": self.iv(M, node), "N": self.iv(N, node), "N_src": self.iv(N_src, node), "M_dst": self.iv(M_dst, node)},
                      None, None, node)
            return None
        if name == "gm_to_l1":
            dst = need_tensor(dst, ("l1",), "dst")
            src = need_tensor(src, ("gm",), "src")
            n_burst, burst, sstride, dstride = arg(2, "n_burst"), arg(3, "burst_len"), arg(4, "src_stride"), arg(5, "dst_stride")
            if burst is None:
                g = self.geom(src, node)
                burst = self.ceil_div(self.arith(ast.Mult(), self.prod(list(g.span), node), _bytes(src.type.dtype), node), 32, node)
            self.emit(op, (dst.plain(), src.plain()), {"n_burst": self.iv(n_burst if n_burst is not None else 1, node), "burst_len": self.iv(burst, node),
                                                       "src_stride": self.iv(sstride or 0, node), "dst_stride": self.iv(dstride or 0, node)}, None, None, node)
            return None
        if name == "gm_to_l1.mx_scale":
            dst = need_tensor(dst, ("l1",), "dst")
            src = need_tensor(src, ("gm",), "src")
            row_tile_idx = arg(2, "row_tile_idx", 0)
            k_blocks = arg(3, "k_blocks")
            src_k_blocks = arg(4, "src_k_blocks")
            k_block_idx = arg(5, "k_block_idx", 0)
            if k_blocks is None:
                k_blocks = self.arith(ast.FloorDiv(), self.shape_of(dst, node)[1], 2, node)
            if src_k_blocks is None:
                src_k_blocks = k_blocks
            b = self.arith(ast.Add(), self.arith(ast.Mult(), row_tile_idx, src_k_blocks, node), k_block_idx, node)
            view = self.slice_tensor_values(src, [(b, k_blocks), (0, 32)], node)
            self.emit("dma.gm_to_l1.pad", (dst.plain(), view.plain()), {"n_burst": self.iv(k_blocks, node), "burst_len_byte": 32, "src_stride_byte": 0, "dst_stride": 0},
                      None, None, node)
            return None
        if name == "gm_to_l1.mx_scale_nd2nz":
            dst = need_tensor(dst, ("l1",), "dst")
            src = need_tensor(src, ("gm",), "src")
            rows, k_groups, src_k_groups = arg(2, "rows"), arg(3, "k_groups"), arg(4, "src_k_groups")
            g = self.geom(src, node)
            if rows is None and k_groups is None and src_k_groups is None:
                rows, k_groups, src_k_groups = g.span[0], g.span[1], g.shape[1]
            if src_k_groups is None:
                src_k_groups = k_groups
            self.emit(op, (dst.plain(), src.plain()), {"rows": self.iv(rows, node), "k_groups": self.iv(k_groups, node), "src_k_groups": self.iv(src_k_groups, node)},
                      None, None, node)
            return None
        if name == "set_constant_to_l1":
            tensor = need_tensor(arg(0, "tensor"), ("l1",), "tensor")
            val = self.rvalue(arg(1, "val"), node)
            n_blocks = arg(2, "n_blocks")
            if n_blocks is None:
                sh = self.shape_of(tensor, node)
                n_blocks = self.arith(ast.FloorDiv(), self.arith(ast.Mult(), sh[0], sh[1], node), _c0(tensor.type.dtype), node)
            self.emit(op, (tensor.plain(),), {"val": self.attr(val), "n_blocks": self.iv(n_blocks, node)}, None, None, node)
            return None
        if name in ("l1_to_l0", "l1_to_l0.mx"):
            dst = need_tensor(dst, ("l0a", "l0b"), "dst")
            src = need_tensor(src, ("l1",), "src")
            base = 3 if name.endswith("mx") else 2
            g = self.geom(src, node)
            m_dst = arg(base, "m_dst")
            n_dst = arg(base + 1, "n_dst")
            m_src = arg(base + 2, "m_src")
            n_src = arg(base + 3, "n_src")
            attrs: dict[str, Any] = {
                "m_dst": self.iv(m_dst if m_dst is not None else g.span[0], node), "n_dst": self.iv(n_dst if n_dst is not None else g.span[1], node),
                "m_src": self.iv(m_src if m_src is not None else g.shape[0], node), "n_src": self.iv(n_src if n_src is not None else g.shape[1], node),
                "src_row0": self.iv(g.offset[0], node), "src_col0": self.iv(g.offset[1], node), "dst_position": Ident(dst.type.space),
            }
            if src.riders.transpose:
                attrs["src_is_transpose"] = True
            if name.endswith("mx"):
                src_mx = need_tensor(arg(2, "src_mx"), ("l1",), "src_mx")
                gm = self.geom(src_mx, node)
                attrs["src_mx"] = src_mx.plain()
                attrs["src_mx_row0"] = self.iv(gm.offset[0], node)
                attrs["src_mx_col0"] = self.iv(gm.offset[1], node)
                attrs["src_mx_offset_element"] = self.iv(arg(base + 4, "src_mx_offset_element", 0), node)
            span = (m_dst if m_dst is not None else g.span[0], n_dst if n_dst is not None else g.span[1])
            self.geoms[dst.name] = Geom(tuple(span), (0, 0), tuple(span), (True, True), self.geom(dst, node).root)
            self.l0_transposed[dst.name] = src.riders.transpose
            self.emit(op, (dst.plain(), src.plain()), attrs, None, None, node)
            return None
        if name == "l1_to_l0.img2col":
            dst = need_tensor(dst, ("l0a",), "dst")
            win = src
            if not isinstance(win, Img2colWindow):
                raise self.err(E_BAD_OPERAND, "l1_to_l0a_img2col needs an img2col window: img2col(...)[m0:m0+tm, k0:k0+tk] or .window(...)", node)
            v = win.view
            conv: dsl.Conv2D = v.conv
            attrs = {"h": self.iv(v.h, node), "w": self.iv(v.w, node), "c": self.iv(v.c, node), "c0": _c0(v.fmap.type.dtype), "kh": conv.kh, "kw": conv.kw,
                     "pad_l": conv.pad[0], "pad_r": conv.pad[1], "pad_t": conv.pad[2], "pad_b": conv.pad[3], "stride_h": conv.stride[0], "stride_w": conv.stride[1],
                     "dil_h": conv.dilation[0], "dil_w": conv.dilation[1], "m0": self.iv(win.m0, node), "k0": self.iv(win.k0, node),
                     "m_ext": self.iv(win.m_ext, node), "k_ext": self.iv(win.k_ext, node), "dst_position": Ident("l0a")}
            self.geoms[dst.name] = Geom((win.m_ext, win.k_ext), (0, 0), (win.m_ext, win.k_ext), (True, True), self.geom(dst, node).root)
            self.l0_transposed[dst.name] = False
            self.emit(op, (dst.plain(), v.fmap.plain()), attrs, None, None, node)
            return None
        if name == "l1_to_bt":
            dst = need_tensor(dst, ("bt",), "dst")
            src = need_tensor(src, ("l1",), "src")
            n = arg(2, "n")
            if n is None:
                sh = self.shape_of(dst, node)
                n = self.arith(ast.Mult(), sh[0], sh[1], node)
            self.emit(op, (dst.plain(), src.plain()), {"n": self.iv(n, node)}, None, None, node)
            return None
        if name in ("l0c_to_gm.nz2nd", "l0c_to_gm.nz2dn", "l0c_to_gm.nz2nz", "l0c_to_l1", "l0c_to_ub"):
            src = need_tensor(src, ("l0c",), "src")
            if name == "l0c_to_ub":
                dst = need_tensor(dst, ("ub",), "dst")
            elif name == "l0c_to_l1":
                dst = need_tensor(dst, ("l1",), "dst")
            else:
                dst = need_tensor(dst, ("gm",), "dst")
            gs = self.geom(src, node)
            gd = self.geom(dst, node)
            attrs = {}
            r = src.riders
            if name == "l0c_to_gm.nz2nz":
                m, n = gs.shape
                m_pad = arg(2, "m_pad")
                if m_pad is None:
                    m_pad = self.arith(ast.FloorDiv(), gd.shape[0], self.arith(ast.FloorDiv(), n, max(16, _c0(dst.type.dtype)), node), node)
                attrs.update({"M": self.iv(m, node), "N": self.iv(n, node), "M_pad": self.iv(m_pad, node), "M_src": self.iv(gs.shape[0], node)})
                rider_pos = 3
            elif name == "l0c_to_gm.nz2nd":
                M, N, N_dst, M_src = arg(2, "M"), arg(3, "N"), arg(4, "N_dst"), arg(5, "M_src")
                if M is None or N is None or N_dst is None:
                    sl = gd.sliced
                    if len(sl) == 2:
                        M, N, N_dst = gd.span[sl[0]], gd.span[sl[1]], gd.shape[sl[1]]
                    elif len(sl) == 1:
                        M, N, N_dst = 1, gd.span[sl[0]], gd.shape[sl[0]]
                    else:
                        raise self.err(E_BAD_SHAPE, "l0c_to_gm_nz2nd: the GM destination must be sliced", node)
                attrs.update({"M": self.iv(M, node), "N": self.iv(N, node), "N_dst": self.iv(N_dst, node), "M_src": self.iv(M_src if M_src is not None else gs.shape[0], node)})
                rider_pos = 6
            elif name == "l0c_to_gm.nz2dn":
                M, N, M_dst, M_src = arg(2, "M"), arg(3, "N"), arg(4, "M_dst"), arg(5, "M_src")
                attrs.update({"M": self.iv(M if M is not None else gs.shape[0], node), "N": self.iv(N if N is not None else gs.shape[1], node),
                              "M_dst": self.iv(M_dst if M_dst is not None else gd.shape[-1], node), "M_src": self.iv(M_src if M_src is not None else gs.shape[0], node)})
                rider_pos = 6
            elif name == "l0c_to_l1":
                M, N, M_dst, M_src = arg(2, "M"), arg(3, "N"), arg(4, "M_dst"), arg(5, "M_src")
                attrs.update({"M": self.iv(M if M is not None else gd.span[0], node), "N": self.iv(N if N is not None else gd.span[1], node),
                              "M_dst": self.iv(M_dst if M_dst is not None else gd.shape[0], node), "M_src": self.iv(M_src if M_src is not None else gs.shape[0], node)})
                relu = arg(6, "relu", False) or r.relu
                if relu:
                    attrs["relu"] = True
                self.emit(op, (dst.plain(), src.plain()), attrs, None, None, node)
                return None
            else:  # l0c_to_ub
                M, N, N_dst, M_src = arg(2, "M"), arg(3, "N"), arg(4, "N_dst"), arg(5, "M_src")
                dual = arg(6, "dual_mode", dsl.DualMode.SPLITM)
                dual_name = dual.name if isinstance(dual, dsl.EnumValue) else {0: "single", 1: "splitm", 2: "splitn"}[int(dual)]
                sub = arg(7, "sub_block_id", None)
                if sub is None and r.subblk is not None:
                    sub = r.subblk
                if dual_name == "single" and sub is None:
                    raise self.err(E_BAD_SIGNATURE, "l0c_to_ub in SINGLE mode needs sub_block_id (0 / 1)", node)
                if M is None or N is None:
                    if dual_name == "single":
                        M, N = gd.span[0], gd.span[1]
                    elif dual_name == "splitm":
                        M, N = self.arith(ast.Mult(), gd.span[0], 2, node), gd.span[1]
                    else:
                        M, N = gd.span[0], self.arith(ast.Mult(), gd.span[1], 2, node)
                attrs.update({"M": self.iv(M, node), "N": self.iv(N, node), "N_dst": self.iv(N_dst if N_dst is not None else gd.shape[1], node),
                              "M_src": self.iv(M_src if M_src is not None else gs.shape[0], node), "dual_mode": Ident(dual_name)})
                if sub is not None:
                    attrs["sub_block_id"] = self.iv(sub, node)
                rider_pos = 8
            relu = arg(rider_pos, "relu", False) or r.relu
            scale = arg(rider_pos + 1, "scale", None)
            offset = arg(rider_pos + 2, "offset", None)
            hybrid = arg(rider_pos + 3, "hif8_hybrid", False) or r.hif8_hybrid
            scale = self.rvalue(scale if scale is not None else (r.scale if r.scale is not None else 1.0), node)
            offset = self.rvalue(offset if offset is not None else (r.offset if r.offset is not None else 0), node)
            if relu:
                attrs["relu"] = True
            if hybrid or not (isinstance(scale, (int, float)) and scale == 1):
                attrs["scale"] = self.attr(scale)
            if not (isinstance(offset, int) and offset == 0):
                attrs["offset"] = self.attr(offset)
            if hybrid:
                attrs["hif8_hybrid"] = True
            if not dst.type.is_local and self.atomic is not None:
                attrs["atomic"] = self.atomic
            self.emit(op, (dst.plain(), src.plain()), attrs, None, None, node)
            return None
        if name in ("ub_to_l1.nd2nz", "ub_to_l1.nz"):
            dst = need_tensor(dst, ("l1",), "dst")
            src = need_tensor(src, ("ub",), "src")
            gd, gs = self.geom(dst, node), self.geom(src, node)
            m_dst, n_dst, m_src, n_src = arg(2, "m_dst"), arg(3, "n_dst"), arg(4, "m_src"), arg(5, "n_src")
            attrs = {"m_dst": self.iv(m_dst if m_dst is not None else gd.shape[0], node), "n_dst": self.iv(n_dst if n_dst is not None else gd.span[1], node),
                     "m_src": self.iv(m_src if m_src is not None else gs.span[0], node), "n_src": self.iv(n_src if n_src is not None else gs.span[1], node),
                     "dst_row0": self.iv(gd.offset[0], node), "dst_col0": self.iv(gd.offset[1], node)}
            if name.endswith("nd2nz"):
                N_src = arg(6, "N_src")
                attrs["N_src"] = self.iv(N_src if N_src is not None else gs.shape[1], node)
            else:
                attrs["M_src"] = self.iv(gs.shape[0], node)
                attrs["src_row0"] = self.iv(gs.offset[0], node)
                attrs["src_col0"] = self.iv(gs.offset[1], node)
            self.emit(op, (dst.plain(), src.plain()), attrs, None, None, node)
            return None
        if name in ("ub_to_l1", "ub_to_ub"):
            dst = need_tensor(dst, ("l1",) if name == "ub_to_l1" else ("ub",), "dst")
            src = need_tensor(src, ("ub",), "src")
            gd, gs = self.geom(dst, node), self.geom(src, node)
            c0 = _c0(dst.type.dtype)
            n_burst, burst, sstride, dstride = arg(2, "n_burst"), arg(3, "burst_len"), arg(4, "src_stride"), arg(5, "dst_stride")
            width = gd.span[1]
            attrs = {"n_burst": self.iv(n_burst if n_burst is not None else gd.span[0], node),
                     "burst_len": self.iv(burst if burst is not None else self.ceil_div(width, c0, node), node),
                     "src_stride": self.iv(sstride if sstride is not None else self.ceil_div(self.arith(ast.Sub(), gs.shape[1], width, node), c0, node), node),
                     "dst_stride": self.iv(dstride if dstride is not None else self.ceil_div(self.arith(ast.Sub(), gd.shape[1], width, node), c0, node), node)}
            self.emit(op, (dst.plain(), src.plain()), attrs, None, None, node)
            return None
        if name == "gm_to_ub.nd":
            dst = need_tensor(dst, ("ub",), "dst")
            src = need_tensor(src, ("gm",), "src")
            if variant == "transpose":
                g = self.geom(src, node)
                sl = g.sliced
                if len(sl) == 2:
                    rows, cols, row_stride = g.span[sl[0]], g.span[sl[1]], g.shape[sl[1]]
                elif len(sl) == 1 and sl[0] == g.rank - 1:
                    rows, cols, row_stride = 1, g.span[sl[0]], g.shape[sl[0]]
                else:
                    rows, cols, row_stride = g.span[sl[0]], 1, g.shape[1]
                attrs = {"dim": 2, "loop_src_stride": [self.iv(row_stride, node), 1], "loop_dst_stride": [1, self.iv(self.shape_of(dst, node)[1], node)],
                         "loop_size": [self.iv(rows, node), self.iv(cols, node)]}
                self.emit("dma.gm_to_ub.nd", (dst.plain(), src.plain()), attrs, None, None, node)
                return None
            lists = {k: kw.get(k) for k in ("loop_src_stride", "loop_dst_stride", "loop_size", "loop_left_pad", "loop_right_pad")}
            for i, k in enumerate(("loop_src_stride", "loop_dst_stride", "loop_size")):
                if lists[k] is None and len(a) > 2 + i:
                    lists[k] = a[2 + i]
            attrs = {"dim": len(lists["loop_size"])}
            for k, v in lists.items():
                if v is not None:
                    attrs[k] = [self.iv(x, node) for x in v]
            for k in ("constant_value", "nearest_value_mode", "config_left_pad", "config_right_pad", "fence"):
                if k in kw:
                    attrs[k] = kw[k]
            self.emit("dma.gm_to_ub.nd", (dst.plain(), src.plain()), attrs, None, None, node)
            return None
        raise self.err(E_UNSUPPORTED, f"no compile rule for {rule!r}", node)

    def slice_tensor_values(self, base: Dyn, ranges: list[tuple[Any, Any]], node: ast.AST) -> Dyn:
        """A slice built from (offset, extent) pairs (for composite stubs)."""
        t = base.type
        assert isinstance(t, MemType)
        g = self.geom(base, node)
        offsets = [o for o, _ in ranges]
        extents = [e for _, e in ranges]
        view = MemType(t.space, t.dtype, tuple(self._dim_type(e) for e in extents), t.layout)
        res = self.emit("mem.slice", (base.plain(),), {"offsets": [self.attr(o) for o in offsets], "extents": [self.attr(e) for e in extents]}, view, "view", node)
        self.roots[res.name] = self.roots.get(base.name, base.name)
        self.geoms[res.name] = Geom(g.shape, tuple(self.arith(ast.Add(), g.offset[i], offsets[i], node) for i in range(g.rank)), tuple(extents),
                                    (True,) * g.rank, g.root)
        return Dyn(res.value)

    def infer_gm_transfer(self, gm: Dyn, node: ast.AST) -> tuple[Any, Any, Any]:
        """(n_burst, burst_len_element, stride_element) of a padded GM transfer (old ``_infer_gm_transfer``)."""
        g = self.geom(gm, node)
        if g.rank == 1:
            return 1, g.span[0], 0
        if g.rank == 2:
            return g.span[0], g.span[1], self.arith(ast.Sub(), g.shape[1], g.span[1], node)
        sl = g.sliced
        if len(sl) == 1:
            d = sl[0]
            burst = self.prod(list(g.span[d + 1:]), node)
            return g.span[d], burst, self.arith(ast.Sub(), self.prod(list(g.shape[d + 1:]), node), burst, node)
        if len(sl) == 2:
            d0, d1 = sl
            burst = self.prod(list(g.span[d1:]), node)
            return g.span[d0], burst, self.arith(ast.Sub(), self.prod(list(g.shape[d0 + 1:]), node), burst, node)
        raise self.err(E_BAD_SHAPE, "a GM transfer needs one or two sliced dimensions", node)

    # -- cube shortcuts -------------------------------------------------------------------------------

    def rule_matmul(self, args: list[Any], kwargs: dict[str, Any], node: ast.AST) -> Any:
        if len(args) < 3:
            raise self.err(E_BAD_SIGNATURE, "matmul(dst, a, b, ...)", node)
        dst, a, b = args[:3]
        for v, what in ((dst, "dst"), (a, "a"), (b, "b")):
            if not (isinstance(v, Dyn) and isinstance(v.type, MemType)):
                raise self.err(E_BAD_OPERAND, f"matmul: {what} must be a tensor", node)
        attrs: dict[str, Any] = {}
        for k in ("m", "n", "k", "splitn", "splitk"):
            if k in kwargs and kwargs[k] is not None:
                attrs[k] = self.iv(kwargs[k], node)
        if a.riders.transpose:
            attrs["a_transpose"] = True
        if b.riders.transpose:
            attrs["b_transpose"] = True
        if "bias" in kwargs and kwargs["bias"] is not None:
            attrs["bias"] = kwargs["bias"].plain()
        init = kwargs.get("is_init", True)
        if isinstance(init, Dyn):
            cond = self.as_bool(init, node)
            with self.fb.if_(self._operand(cond, node), loc=self.loc(node)) as ifb:
                self.emit("cube.matmul", (dst.plain(), a.plain(), b.plain()), {**attrs, "init": True}, None, None, node)
                with ifb.else_():
                    self.emit("cube.matmul", (dst.plain(), a.plain(), b.plain()), {**attrs, "init": False}, None, None, node)
            return None
        attrs["init"] = bool(init)
        return self.emit("cube.matmul", (dst.plain(), a.plain(), b.plain()), attrs, None, None, node)

    def rule_matmul_mx(self, args: list[Any], kwargs: dict[str, Any], node: ast.AST) -> Any:
        if len(args) < 5:
            raise self.err(E_BAD_SIGNATURE, "matmul_mx(dst, a, b, scale_a, scale_b, ...)", node)
        dst, a, b, sa, sb = args[:5]
        attrs: dict[str, Any] = {}
        for k in ("m", "n", "k", "splitn", "splitk"):
            if k in kwargs and kwargs[k] is not None:
                attrs[k] = self.iv(kwargs[k], node)
        if a.riders.transpose:
            attrs["a_transpose"] = True
        if b.riders.transpose:
            attrs["b_transpose"] = True
        if kwargs.get("bias") is not None:
            attrs["bias"] = kwargs["bias"].plain()
        attrs["init"] = bool(kwargs.get("is_init", True))
        return self.emit("cube.matmul_mx", (dst.plain(), a.plain(), b.plain(), sa.plain(), sb.plain()), attrs, None, None, node)

    def rule_conv2d(self, args: list[Any], kwargs: dict[str, Any], node: ast.AST) -> Any:
        def arg(i: int, key: str, default: Any = None) -> Any:
            return args[i] if len(args) > i else kwargs.get(key, default)

        l0c, fmap, weight, conv = arg(0, "l0c"), arg(1, "fmap"), arg(2, "weight"), arg(3, "conv")
        if not isinstance(conv, dsl.Conv2D):
            raise self.err(E_BAD_OPERAND, "conv2d needs a Conv2D", node)
        attrs: dict[str, Any] = {"h": self.iv(arg(4, "h"), node), "w": self.iv(arg(5, "w"), node), "c": self.iv(arg(6, "c"), node), "cout": self.iv(arg(7, "cout"), node),
                                 "kh": conv.kh, "kw": conv.kw, "stride_h": conv.stride[0], "stride_w": conv.stride[1], "dil_h": conv.dilation[0], "dil_w": conv.dilation[1],
                                 "pad_l": conv.pad[0], "pad_r": conv.pad[1], "pad_t": conv.pad[2], "pad_b": conv.pad[3]}
        m0 = arg(8, "m0", 0)
        if not (isinstance(m0, int) and m0 == 0):
            attrs["m0"] = self.iv(m0, node)
        tile_k = arg(9, "tile_k")
        if tile_k is not None:
            attrs["tile_k"] = self.iv(tile_k, node)
        bias = arg(10, "bias")
        if bias is not None:
            attrs["bias"] = bias.plain()
        return self.emit("cube.conv2d", (l0c.plain(), fmap.plain(), weight.plain()), attrs, None, None, node)

    def rule_img2col(self, args: list[Any], kwargs: dict[str, Any], node: ast.AST) -> Img2col:
        def arg(i: int, key: str) -> Any:
            return args[i] if len(args) > i else kwargs.get(key)

        fmap = arg(0, "fmap")
        if not (isinstance(fmap, Dyn) and isinstance(fmap.type, MemType) and fmap.type.space == "l1"):
            raise self.err(E_BAD_OPERAND, "img2col needs an L1 feature map", node)
        conv = arg(1, "conv")
        if not isinstance(conv, dsl.Conv2D):
            raise self.err(E_BAD_OPERAND, "img2col needs a Conv2D", node)
        return Img2col(fmap.plain(), conv, self.rvalue(arg(2, "h"), node), self.rvalue(arg(3, "w"), node), self.rvalue(arg(4, "c"), node))

    def img2col_subscript(self, view: Img2col, parts: list[ast.expr], node: ast.AST) -> Img2colWindow:
        if len(parts) != 2 or not all(isinstance(p, ast.Slice) for p in parts):
            raise self.err(E_UNSUPPORTED, "an img2col view is indexed with two slices [m0:m0+tm, k0:k0+tk]", node)
        vals = []
        for p in parts:
            lo = self.rvalue(self.ev(p.lower), node) if p.lower is not None else 0
            if p.upper is None:
                raise self.err(E_UNSUPPORTED, "img2col slices need explicit bounds", node)
            hi = self.rvalue(self.ev(p.upper), node)
            vals.append((lo, self.arith(ast.Sub(), hi, lo, node)))
        (m0, m_ext), (k0, k_ext) = vals
        return view.window(m0, k0, m_ext, k_ext)

    def rule_mmad(self, rule: str, args: list[Any], kwargs: dict[str, Any], node: ast.AST) -> Any:
        if self.kind != "kernel":
            raise self.err(E_UNSUPPORTED, "mmad is only available in kernel functions", node)

        def arg(i: int, key: str, default: Any = None) -> Any:
            return args[i] if len(args) > i else kwargs.get(key, default)

        dst, a, b = arg(0, "dst"), arg(1, "src_a"), arg(2, "src_b")
        for v, sp, what in ((dst, "l0c", "dst"), (a, "l0a", "src_a"), (b, "l0b", "src_b")):
            if not (isinstance(v, Dyn) and isinstance(v.type, MemType) and v.type.space == sp):
                raise self.err(E_BAD_OPERAND, f"mmad: {what} must be an {sp.upper()} tensor", node)
        ga, gb, gd = self.geom(a, node), self.geom(b, node), self.geom(dst, node)
        ta, tb = self.l0_transposed.get(a.name, False), self.l0_transposed.get(b.name, False)
        M = arg(3, "M")
        N = arg(4, "N")
        K = arg(5, "K")
        if M is None:
            M = ga.shape[1] if ta else ga.shape[0]
        if N is None:
            N = gb.shape[1] if tb else gb.shape[0]
        if K is None:
            K = ga.shape[0] if ta else ga.shape[1]
        attrs: dict[str, Any] = {"M": self.iv(M, node), "N": self.iv(N, node), "K": self.iv(K, node), "is_init": bool(arg(6, "is_init", True)),
                                 "dst_row0": self.iv(gd.offset[0], node), "dst_col0": self.iv(gd.offset[1], node),
                                 "dst_rows": self.iv(gd.shape[0], node), "dst_cols": self.iv(gd.shape[1], node)}
        bias = arg(7, "bias")
        if bias is not None:
            attrs["bias"] = bias.plain()
        unit_flag = arg(8, "unit_flag", 0)
        if rule == "cube.mmad" and not (isinstance(unit_flag, int) and unit_flag == 0):
            attrs["unit_flag"] = self.iv(unit_flag, node)
        return self.emit(rule, (dst.plain(), a.plain(), b.plain()), attrs, None, None, node)
