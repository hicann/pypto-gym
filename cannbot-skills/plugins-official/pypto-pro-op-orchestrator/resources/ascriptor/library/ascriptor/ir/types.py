# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Types of the ascriptor IR (RFC-0001 §4).

Everything here is immutable data with one text spelling; ``parse_type(str(t)) == t``.
Dimensions may be integer literals, references to scalar values in scope (``%M``), ``?`` for
ragged list members, or products of these — the only shape arithmetic the IR admits (D-014).
"""

from __future__ import annotations

from dataclasses import dataclass

from .lexer import ParseError, Token, TokenStream, tokenize

# --------------------------------------------------------------------------- scalar dtypes


@dataclass(frozen=True)
class DType:
    name: str
    bits: int
    kind: str  # bool | int | uint | float | complex

    def __str__(self) -> str:
        return self.name

    @property
    def is_integer(self) -> bool:
        return self.kind in ("int", "uint", "bool")

    @property
    def is_float(self) -> bool:
        return self.kind == "float"


_DTYPE_LIST = [
    DType("b1", 8, "bool"),
    DType("i8", 8, "int"), DType("u8", 8, "uint"), DType("i16", 16, "int"), DType("u16", 16, "uint"),
    DType("i32", 32, "int"), DType("u32", 32, "uint"), DType("i64", 64, "int"), DType("u64", 64, "uint"),
    DType("f16", 16, "float"), DType("bf16", 16, "float"), DType("f32", 32, "float"),
    DType("e4m3", 8, "float"), DType("e5m2", 8, "float"), DType("hif8", 8, "float"),
    DType("fp4_e2m1", 4, "float"), DType("fp4_e1m2", 4, "float"), DType("e8m0", 8, "float"),
    DType("i4", 4, "int"),
    DType("c32", 32, "complex"), DType("c64", 64, "complex"),
]
DTYPES: dict[str, DType] = {d.name: d for d in _DTYPE_LIST}


def dtype(name: str) -> DType:
    try:
        return DTYPES[name]
    except KeyError:
        raise KeyError(f"unknown dtype {name!r}; known: {sorted(DTYPES)}") from None


# --------------------------------------------------------------------------- dimensions


@dataclass(frozen=True)
class DimValue:
    """A dimension given by a scalar value in scope, spelled ``%name``."""

    name: str

    def __str__(self) -> str:
        return f"%{self.name}"


@dataclass(frozen=True)
class Ragged:
    """``?``: a per-member dimension of a ``gmlist``, read from the list descriptor at runtime."""

    def __str__(self) -> str:
        return "?"


@dataclass(frozen=True)
class Product:
    factors: tuple[int | DimValue, ...]

    def __str__(self) -> str:
        return "*".join(str(f) for f in self.factors)


Dim = int | DimValue | Ragged | Product


def dim_is_static(d: Dim) -> bool:
    return isinstance(d, int)


def dim_values(d: Dim) -> set[str]:
    if isinstance(d, DimValue):
        return {d.name}
    if isinstance(d, Product):
        return {f.name for f in d.factors if isinstance(f, DimValue)}
    return set()


def dims_str(dims: tuple[Dim, ...]) -> str:
    return "[" + ", ".join(str(d) for d in dims) + "]"


# --------------------------------------------------------------------------- types

MEM_SPACES = ("gm", "gmlist", "l1", "l0a", "l0b", "l0c", "ub", "bt", "ws")
LOCAL_SPACES = ("l1", "l0a", "l0b", "l0c", "ub", "bt")
LAYOUTS = ("nz", "nd")
PIPES = ("MTE1", "MTE2", "MTE3", "M", "V", "FIX", "S")
MASK_WIDTHS = (8, 16, 32, 64)


class Type:
    """Base class; concrete types are the frozen dataclasses below."""

    def __str__(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError


@dataclass(frozen=True)
class ScalarType(Type):
    dtype: DType

    def __str__(self) -> str:
        return self.dtype.name


@dataclass(frozen=True)
class MemType(Type):
    space: str
    dtype: DType
    dims: tuple[Dim, ...]
    layout: str | None = None

    def __post_init__(self) -> None:
        if self.space not in MEM_SPACES:
            raise ValueError(f"unknown memory space {self.space!r}")
        if self.layout is not None and self.layout not in LAYOUTS:
            raise ValueError(f"unknown layout {self.layout!r}")
        if any(isinstance(d, Ragged) for d in self.dims) and self.space != "gmlist":
            raise ValueError("'?' dimensions are only allowed in gmlist member types")

    @property
    def is_local(self) -> bool:
        return self.space in LOCAL_SPACES

    @property
    def rank(self) -> int:
        return len(self.dims)

    def __str__(self) -> str:
        s = f"{self.space}<{self.dtype}, {dims_str(self.dims)}"
        if self.layout is not None:
            s += f", {self.layout}"
        return s + ">"


@dataclass(frozen=True)
class BufType(Type):
    elem: MemType
    slots: int

    def __str__(self) -> str:
        return f"buf<{self.elem}, {self.slots}>"


@dataclass(frozen=True)
class RegType(Type):
    dtype: DType
    n: int = 1

    def __post_init__(self) -> None:
        if type(self.n) is not int or self.n not in (1, 2):
            raise ValueError(f"register count must be 1 or 2, got {self.n!r}")

    @property
    def lanes(self) -> int:
        return 256 * self.n * 8 // max(self.dtype.bits, 8)

    def __str__(self) -> str:
        return f"reg<{self.dtype}, {self.n}>"


@dataclass(frozen=True)
class MaskType(Type):
    width: int
    n: int = 1

    def __post_init__(self) -> None:
        if self.width not in MASK_WIDTHS:
            raise ValueError(f"mask width must be one of {MASK_WIDTHS}, got {self.width}")
        if type(self.n) is not int or self.n not in (1, 2):
            raise ValueError(f"mask register count must be 1 or 2, got {self.n!r}")
        if self.n == 2 and self.width not in (32, 64):
            raise ValueError("two-register masks require b32 or b64 logical lanes")

    @property
    def lanes(self) -> int:
        return 256 * self.n * 8 // self.width

    def __str__(self) -> str:
        return f"mask<b{self.width}{', 2' if self.n == 2 else ''}>"


@dataclass(frozen=True)
class UnalignRegType(Type):
    """The 32-byte alignment shift register of an unaligned load or store chain.

    ``role`` is ``load`` or ``store``: AscendC spells them ``UnalignRegForLoad`` /
    ``UnalignRegForStore`` (both aliases of the CCE ``vector_align``), and a chain's pre / post
    ops must use a register of the matching role.
    """

    role: str

    def __post_init__(self) -> None:
        if self.role not in ("load", "store"):
            raise ValueError(f"unalignreg role must be load or store, got {self.role!r}")

    def __str__(self) -> str:
        return f"unalignreg<{self.role}>"


@dataclass(frozen=True)
class CellType(Type):
    dtype: DType

    def __str__(self) -> str:
        return f"cell<{self.dtype}>"


@dataclass(frozen=True)
class EventType(Type):
    depth: int
    set_pipe: str | None = None
    wait_pipe: str | None = None
    id: int | None = None

    def __str__(self) -> str:
        parts = [str(self.depth)]
        if self.set_pipe is not None or self.wait_pipe is not None or self.id is not None:
            parts += [self.set_pipe or "_", self.wait_pipe or "_", "_" if self.id is None else str(self.id)]
        return "event<" + ", ".join(parts) + ">"


@dataclass(frozen=True)
class FlagType(Type):
    id: int | None = None

    def __str__(self) -> str:
        return "flag" if self.id is None else f"flag<{self.id}>"


def is_scalar_int(t: Type) -> bool:
    return isinstance(t, ScalarType) and t.dtype.is_integer and t.dtype.kind != "bool"


# --------------------------------------------------------------------------- parsing


def parse_type(text: str) -> Type:
    ts = TokenStream(tokenize(text))
    t = parse_type_tokens(ts)
    if not ts.at("eof"):
        raise ts.error(f"trailing tokens after type: {ts.cur.text!r}")
    return t


def _parse_int(ts: TokenStream) -> int:
    tok = ts.expect("int")
    return int(tok.text, 0)


def parse_dim(ts: TokenStream) -> Dim:
    factors: list[int | DimValue] = []
    while True:
        if ts.at("int"):
            factors.append(_parse_int(ts))
        elif ts.at("value"):
            factors.append(DimValue(ts.advance().text[1:]))
        elif ts.at_punct("?") and not factors:
            ts.advance()
            return Ragged()
        else:
            raise ts.error(f"expected a dimension, found {ts.cur.text!r}")
        if ts.at_punct("*"):
            ts.advance()
            continue
        break
    if len(factors) == 1:
        return factors[0]
    return Product(tuple(factors))


def parse_dims(ts: TokenStream) -> tuple[Dim, ...]:
    ts.expect_punct("[")
    dims: list[Dim] = []
    if not ts.at_punct("]"):
        dims.append(parse_dim(ts))
        while ts.accept("punct", ","):
            dims.append(parse_dim(ts))
    ts.expect_punct("]")
    return tuple(dims)


def parse_type_tokens(ts: TokenStream) -> Type:
    head = ts.expect("ident").text
    if head in DTYPES:
        return ScalarType(DTYPES[head])
    if head in MEM_SPACES:
        ts.expect_punct("<")
        dt = ts.expect("ident").text
        if dt not in DTYPES:
            raise ts.error(f"unknown dtype {dt!r}")
        ts.expect_punct(",")
        dims = parse_dims(ts)
        layout = None
        if ts.accept("punct", ","):
            layout = ts.expect("ident").text
        ts.expect_punct(">")
        try:
            return MemType(head, DTYPES[dt], dims, layout)
        except ValueError as exc:
            raise ts.error(str(exc)) from None
    if head == "buf":
        ts.expect_punct("<")
        elem = parse_type_tokens(ts)
        if not isinstance(elem, MemType) or not elem.is_local:
            raise ts.error("buf element must be an on-chip memory type")
        ts.expect_punct(",")
        slots = _parse_int(ts)
        ts.expect_punct(">")
        return BufType(elem, slots)
    if head == "reg":
        ts.expect_punct("<")
        dt = ts.expect("ident").text
        if dt not in DTYPES:
            raise ts.error(f"unknown dtype {dt!r}")
        n = 1
        if ts.accept("punct", ","):
            n = _parse_int(ts)
        ts.expect_punct(">")
        try:
            return RegType(DTYPES[dt], n)
        except ValueError as exc:
            raise ts.error(str(exc)) from None
    if head == "mask":
        ts.expect_punct("<")
        w = ts.expect("ident").text
        n = _parse_int(ts) if ts.accept("punct", ",") else 1
        ts.expect_punct(">")
        if not (w.startswith("b") and w[1:].isdigit() and int(w[1:]) in MASK_WIDTHS):
            raise ts.error(f"mask width must be b8/b16/b32/b64, got {w!r}")
        try:
            return MaskType(int(w[1:]), n)
        except ValueError as exc:
            raise ts.error(str(exc)) from None
    if head == "unalignreg":
        ts.expect_punct("<")
        role = ts.expect("ident").text
        ts.expect_punct(">")
        try:
            return UnalignRegType(role)
        except ValueError as exc:
            raise ts.error(str(exc)) from None
    if head == "cell":
        ts.expect_punct("<")
        dt = ts.expect("ident").text
        if dt not in DTYPES:
            raise ts.error(f"unknown dtype {dt!r}")
        ts.expect_punct(">")
        return CellType(DTYPES[dt])
    if head == "event":
        ts.expect_punct("<")
        depth = _parse_int(ts)
        set_pipe = wait_pipe = None
        eid = None
        if ts.accept("punct", ","):
            set_pipe = _parse_pipe_or_blank(ts)
            ts.expect_punct(",")
            wait_pipe = _parse_pipe_or_blank(ts)
            ts.expect_punct(",")
            if ts.at("ident", "_"):
                ts.advance()
            else:
                eid = _parse_int(ts)
        ts.expect_punct(">")
        return EventType(depth, set_pipe, wait_pipe, eid)
    if head == "flag":
        fid = None
        if ts.accept("punct", "<"):
            fid = _parse_int(ts)
            ts.expect_punct(">")
        return FlagType(fid)
    raise ts.error(f"unknown type {head!r}")


def _parse_pipe_or_blank(ts: TokenStream) -> str | None:
    tok = ts.expect("ident")
    if tok.text == "_":
        return None
    if tok.text not in PIPES:
        raise ParseError(f"{tok.line}:{tok.col}: unknown pipe {tok.text!r}")
    return tok.text


def type_values(t: Type) -> set[str]:
    """Names of the scalar values a type refers to through its dimensions."""
    if isinstance(t, MemType):
        out: set[str] = set()
        for d in t.dims:
            out |= dim_values(d)
        return out
    if isinstance(t, BufType):
        return type_values(t.elem)
    return set()


__all__ = [
    "DType", "DTYPES", "dtype", "Dim", "DimValue", "Ragged", "Product", "dim_is_static", "dim_values", "dims_str",
    "Type", "ScalarType", "MemType", "BufType", "RegType", "MaskType", "UnalignRegType", "CellType", "EventType",
    "FlagType", "MEM_SPACES", "LOCAL_SPACES", "LAYOUTS", "PIPES", "MASK_WIDTHS", "is_scalar_int", "parse_type",
    "parse_type_tokens", "parse_dims", "parse_dim", "type_values", "Token", "TokenStream", "tokenize", "ParseError",
]
