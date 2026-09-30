# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""View geometry for the printer: a value's window folded to its root allocation and a byte offset.

The passes' :func:`~ascriptor.passes.util.view_of` folds ``mem.slice`` chains to root coordinates; the
printer needs one more thing — the *address* of a window — so this module keeps the same chain walk and
turns the coordinates into a byte offset with the layout rules the old handlers used:

* ND (every GM window, ND locals): row-major over the dims the offsets refer to.
* NZ locals (L1 / L0A / L0B / L0C / UB tiles): a fractal row is ``c0`` elements (16 for L0C, else 32 bytes),
  a fractal column holds ``align16(rows)`` rows in L1 / L0 (the L0 loads and the fixpipe address columns in
  16-row units) but exactly ``rows`` rows in UB (a ``.nz()`` view packed by vector code keeps whatever row
  stride it was packed with — v8_allhif8 packs P with a 33-row stride to dodge bank conflicts, and its
  ``ub_to_l1_nz`` passes that stride as ``M_src``), so element ``(r, c)`` sits at
  ``(c // c0) * S * c0 + r * c0 + c % c0`` with ``S`` the column height of the space.
* Packed 4-bit dtypes count carrier bytes: the last axis is halved, a fractal row is 32 carrier bytes.

``mem.reinterpret {tile}`` and ``mem.reshape`` start a new coordinate system at the window's current
start; a plain ``mem.reinterpret`` rescales the last axis to the new element width.

Offsets are small expression trees (``int | Value | (op, a, b)``) rendered by :func:`cexpr` once the
printer knows the C names of the scalar values.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ...ir import Op, Value
from ...ir.types import BufType, DType, MemType
from ...passes.util import Defs, dim_scalar, literal_or_value

Expr = Any  # int | Value | tuple[str, Expr, Expr]

NZ_SPACES = ("l1", "l0a", "l0b", "l0c", "ub")


# --------------------------------------------------------------------------- expression trees


def add(a: Expr, b: Expr) -> Expr:
    if isinstance(a, int) and isinstance(b, int):
        return a + b
    if isinstance(a, int) and a == 0:
        return b
    if isinstance(b, int) and b == 0:
        return a
    return ("+", a, b)


def mul(a: Expr, b: Expr) -> Expr:
    if isinstance(a, int) and isinstance(b, int):
        return a * b
    if (isinstance(a, int) and a == 0) or (isinstance(b, int) and b == 0):
        return 0
    if isinstance(a, int) and a == 1:
        return b
    if isinstance(b, int) and b == 1:
        return a
    return ("*", a, b)


def floordiv(a: Expr, b: int) -> Expr:
    if isinstance(a, int):
        return a // b
    if b == 1:
        return a
    return ("//", a, b)


def mod(a: Expr, b: int) -> Expr:
    if isinstance(a, int):
        return a % b
    if b == 1:
        return 0
    return ("%", a, b)


def align16(a: Expr) -> Expr:
    if isinstance(a, int):
        return (a + 15) // 16 * 16
    return ("align16", a, 16)


def prod(xs: list[Expr]) -> Expr:
    out: Expr = 1
    for x in xs:
        out = mul(out, x)
    return out


def cexpr(e: Expr, name: Callable[[Value], str]) -> str:
    """Render an offset tree as a C expression (``name`` maps scalar values to their C identifiers)."""
    if isinstance(e, bool):
        return "1" if e else "0"
    if isinstance(e, int):
        return str(e) if e >= 0 else f"({e})"
    if isinstance(e, Value):
        return name(e)
    op, a, b = e
    if op == "align16":
        return f"AlignUp({cexpr(a, name)}, 16)"
    sym = {"+": "+", "*": "*", "//": "/", "%": "%"}[op]
    return f"({cexpr(a, name)} {sym} {cexpr(b, name)})"


def div_exact(e: Expr, k: int) -> Expr | None:
    """``e / k`` when the tree proves the division exact (integers, products carrying a multiple of ``k``,
    sums of such); ``None`` when it cannot — the printer then falls back to a byte-address window."""
    if k == 1:
        return e
    if isinstance(e, bool) or isinstance(e, Value):
        return None
    if isinstance(e, int):
        return e // k if e % k == 0 else None
    op, a, b = e
    if op == "*":
        if isinstance(b, int) and b % k == 0:
            return mul(a, b // k)
        if isinstance(a, int) and a % k == 0:
            return mul(a // k, b)
        qa = div_exact(a, k)
        if qa is not None:
            return mul(qa, b)
        qb = div_exact(b, k)
        return None if qb is None else mul(a, qb)
    if op == "+":
        qa, qb = div_exact(a, k), div_exact(b, k)
        return None if qa is None or qb is None else add(qa, qb)
    return None


def expr_values(e: Expr) -> list[Value]:
    if isinstance(e, Value):
        return [e]
    if isinstance(e, tuple):
        return expr_values(e[1]) + expr_values(e[2])
    return []


# --------------------------------------------------------------------------- geometry


@dataclass
class Geo:
    root: Value  # mem.alloc / mem.workspace / parameter / list.item
    space: str
    dtype: DType  # the window's dtype
    layout: str | None
    dims: tuple[Expr, ...]  # dims the offsets refer to, in elements of ``dtype`` (packed: logical elements)
    offsets: tuple[Expr, ...]
    kept: tuple[bool, ...]
    slot: Expr | None  # slot index into a slot buffer root
    base_bytes: Expr  # byte offset of the current coordinate system's origin from the root
    strides: tuple[Expr, ...] | None = None  # mem.view (RFC-0010): element strides of the coordinate system


def _elem_type(t: Any) -> MemType:
    return t.elem if isinstance(t, BufType) else t


def _scale(x: Expr, old_bits: int, new_bits: int) -> Expr:
    if old_bits == new_bits:
        return x
    if old_bits > new_bits:
        return mul(x, old_bits // new_bits)
    return floordiv(x, new_bits // old_bits)


def fold(v: Value, defs: Defs) -> Geo:
    """Fold the view chain under ``v`` to its root."""
    op = defs.op(v)
    if op is None or op.opcode in ("mem.alloc", "mem.workspace", "list.item"):
        mt = _elem_type(v.type)
        if not isinstance(mt, MemType):
            raise TypeError(f"{v} is not a memory value")
        dims = tuple(dim_scalar(d) for d in mt.dims)
        return Geo(v, mt.space, mt.dtype, mt.layout, dims, (0,) * len(dims), (True,) * len(dims), None, 0)
    if op.opcode == "mem.get_buf":
        g = fold(op.operands[0], defs)  # type: ignore[arg-type]
        g.slot = literal_or_value(op.operands[1])
        return g
    if op.opcode == "mem.slice":
        g = fold(op.operands[0], defs)  # type: ignore[arg-type]
        offs = [literal_or_value(x) for x in op.attrs["offsets"]]
        mask = [bool(m) for m in op.attrs.get("mask", [True] * len(offs))]
        kept_idx = [i for i, k in enumerate(g.kept) if k]
        if len(kept_idx) != len(offs):
            raise ValueError(f"mem.slice #{op.id}: {len(offs)} offsets for a rank-{len(kept_idx)} view")
        new_off = list(g.offsets)
        new_kept = list(g.kept)
        for j, i in enumerate(kept_idx):
            new_off[i] = add(g.offsets[i], offs[j])
            new_kept[i] = mask[j]
        g.offsets = tuple(new_off)
        g.kept = tuple(new_kept)
        return g
    if op.opcode == "mem.reinterpret":
        g = fold(op.operands[0], defs)  # type: ignore[arg-type]
        mt = _elem_type(v.type)
        assert isinstance(mt, MemType)
        layout = mt.layout if "layout" in op.attrs else g.layout
        if "tile" in op.attrs:
            rows, cols = (literal_or_value(x) for x in op.attrs["tile"])
            return _rebase(g, mt.dtype, layout, (rows, cols))
        if mt.dtype == g.dtype:
            g.layout = layout
            return g
        old_bits, new_bits = max(g.dtype.bits, 8), max(mt.dtype.bits, 8)
        kept_idx = [i for i, k in enumerate(g.kept) if k]
        last = kept_idx[-1] if kept_idx else len(g.dims) - 1
        offs, dims = list(g.offsets), list(g.dims)
        offs[last] = _scale(offs[last], old_bits, new_bits)
        dims[last] = _scale(dims[last], old_bits, new_bits)
        g.offsets, g.dims, g.dtype, g.layout = tuple(offs), tuple(dims), mt.dtype, layout
        return g
    if op.opcode == "mem.view":
        # a strided GM re-description (RFC-0010): a new coordinate system displaced by the element
        # offset; the strides live in the lowered DMA's descriptor attrs, the printer only needs the base
        g = fold(op.operands[0], defs)  # type: ignore[arg-type]
        mt = _elem_type(v.type)
        assert isinstance(mt, MemType)
        dims = tuple(dim_scalar(d) for d in mt.dims)
        off = literal_or_value(op.attrs.get("offset", 0))
        strides = tuple(literal_or_value(s) for s in op.attrs["strides"])
        base = add(add(g.base_bytes, elem_bytes_offset(g)), mul(off, max(mt.dtype.bits, 8) // 8))
        return Geo(g.root, g.space, mt.dtype, mt.layout, dims, (0,) * len(dims), (True,) * len(dims), g.slot, base,
                   strides=strides)
    if op.opcode == "mem.reshape":
        g = fold(op.operands[0], defs)  # type: ignore[arg-type]
        mt = _elem_type(v.type)
        assert isinstance(mt, MemType)
        dims = tuple(dim_scalar(d) for d in mt.dims)
        return _rebase(g, mt.dtype, mt.layout, dims)
    raise TypeError(f"{v} ({op.opcode} #{op.id}) is not a memory view")


def _rebase(g: Geo, dtype: DType, layout: str | None, dims: tuple[Expr, ...]) -> Geo:
    """Start a new coordinate system at the window's current start."""
    base = add(g.base_bytes, elem_bytes_offset(g))
    return Geo(g.root, g.space, dtype, layout, dims, (0,) * len(dims), (True,) * len(dims), g.slot, base)


L1_FP32_ZZ = False
"""c220 only: the cube stores fp32 L1 tiles as ZZ (16-row bands of 16x8 fractals, RFC-0008 §5), so a
fp32 L1 window's element offset follows the ZZ formula instead of NZ. The printer sets this per module
(views.set_l1_fp32_zz); c310 keeps NZ. Every other dtype stays NZ — including the 32-bit integers,
which this used to fold as ZZ because it tested the element size rather than the dtype."""


def set_l1_fp32_zz(on: bool) -> None:
    global L1_FP32_ZZ
    L1_FP32_ZZ = bool(on)


def c0_of(space: str, dtype: DType) -> int:
    """Elements per fractal row of an NZ tile (carrier bytes for packed dtypes)."""
    if space == "l0c":
        return 16
    if dtype.bits < 8:
        return 32
    return 32 // (dtype.bits // 8)


def elem_bytes_offset(g: Geo) -> Expr:
    """Byte offset of the window's start from its coordinate system's origin."""
    esz = max(g.dtype.bits, 8) // 8
    packed = g.dtype.bits < 8
    offs, dims = list(g.offsets), list(g.dims)
    if not dims:
        return 0
    if g.layout == "nz" and g.space in NZ_SPACES and len(dims) == 2:
        rows = dims[0] if g.space == "ub" else align16(dims[0])  # the fractal column height (module docstring)
        row, col = offs
        if packed:
            # fractal row = 32 carrier bytes = 64 logical elements
            colblk = floordiv(col, 64)
            rem = floordiv(mod(col, 64), 2)
            return add(add(mul(colblk, mul(rows, 32)), mul(row, 32)), rem)
        c0 = c0_of(g.space, g.dtype)
        # `esz == 4` here also caught i32 / u32, which are NZ: the offset was folded with the ZZ
        # formula for a tile the emitter writes and reads as NZ. The dtype decides, not its width.
        if L1_FP32_ZZ and g.space == "l1" and g.dtype.name == "f32":
            # ZZ (c220 fp32): element (r, c) of a [R, C] tile sits at
            # NZ address formula: (r/16)*(align16(C)*16) + (c/8)*128 + (r%16)*8 + (c%8)
            cols16 = align16(dims[1])
            elems = add(add(mul(floordiv(row, 16), mul(cols16, 16)), mul(floordiv(col, 8), 128)),
                        add(mul(mod(row, 16), 8), mod(col, 8)))
            return mul(elems, esz)
        colblk = floordiv(col, c0)
        rem = mod(col, c0)
        elems = add(add(mul(colblk, mul(rows, c0)), mul(row, c0)), rem)
        return mul(elems, esz)
    if g.strides is not None:  # a mem.view coordinate system: offsets step by the explicit strides
        total_s: Expr = 0
        for off, st in zip(offs, g.strides, strict=True):
            total_s = add(total_s, mul(off, st))
        return mul(total_s, esz)
    if packed:
        dims[-1] = floordiv(dims[-1], 2)
        offs[-1] = floordiv(offs[-1], 2)
        esz = 1
    total: Expr = 0
    for i, off in enumerate(offs):
        total = add(total, mul(off, prod(dims[i + 1:])))
    return mul(total, esz)


def byte_offset(g: Geo) -> Expr:
    return add(g.base_bytes, elem_bytes_offset(g))


def nz_offset_bytes(space: str, dtype: DType, row: Expr, col: Expr, rows_total: Expr) -> Expr:
    """Byte offset of element ``(row, col)`` inside an NZ tile of ``rows_total`` rows (DMA tile coordinates
    such as ``dst_row0 / dst_col0``)."""
    g = Geo(None, space, dtype, "nz", (rows_total, 0), (row, col), (True, True), None, 0)  # type: ignore[arg-type]
    return elem_bytes_offset(g)


def nd_offset_bytes(dtype: DType, offsets: tuple[Expr, ...], dims: tuple[Expr, ...]) -> Expr:
    g = Geo(None, "gm", dtype, None, dims, offsets, (True,) * len(dims), None, 0)  # type: ignore[arg-type]
    return elem_bytes_offset(g)


def root_op(v: Value, defs: Defs) -> Op | None:
    return defs.op(fold(v, defs).root)


__all__ = ["Expr", "Geo", "fold", "byte_offset", "elem_bytes_offset", "nz_offset_bytes", "nd_offset_bytes", "cexpr", "div_exact",
           "expr_values", "add", "mul", "floordiv", "mod", "align16", "prod", "c0_of", "root_op"]
