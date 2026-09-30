# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The IR data model (RFC-0001 §3, §5, §7, §8): modules, functions, blocks, ops, values.

Everything is a frozen dataclass; passes build new trees through :mod:`ascriptor.ir.builder`.
Attribute values are plain data: ``int | float | bool | str | Ident | Value | list | dict``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from .types import Type

SURFACE = "surface/1"
LOWERED = "lowered/1"
FUNCTION_KINDS = ("kernel", "vf", "simt", "func")
MODES = ("mix", "vec", "cube")
SIDES = ("cube", "vec")


@dataclass(frozen=True)
class Loc:
    """Source positions, innermost first: ``loc("a.py:12:4" <- "a.py:40:8")``."""

    chain: tuple[str, ...]

    @classmethod
    def of(cls, *positions: str) -> Loc:
        return cls(tuple(positions))

    def __str__(self) -> str:
        return "loc(" + " <- ".join(f'"{p}"' for p in self.chain) + ")"


@dataclass(frozen=True)
class Origin:
    """One provenance entry: which pass did what to this op, and from which ops."""

    pass_name: str
    kind: str  # inserted | rewritten | moved | expanded
    from_ids: tuple[int, ...] = ()
    note: str | None = None


@dataclass(frozen=True)
class Value:
    """An SSA value ``%name`` with its type. Identity is the name within a function."""

    name: str
    type: Type

    def __str__(self) -> str:
        return f"%{self.name}"


@dataclass(frozen=True)
class Literal:
    """An untyped literal operand; its type comes from the op's registry pattern."""

    value: int | float | bool | complex

    def __str__(self) -> str:
        return literal_str(self.value)


@dataclass(frozen=True)
class FuncRef:
    name: str

    def __str__(self) -> str:
        return f"@{self.name}"


@dataclass(frozen=True)
class Ident:
    """A bare identifier in attribute position: an enum-like constant such as ``nz`` or ``MTE2``."""

    name: str

    def __str__(self) -> str:
        return self.name


Operand = Value | Literal | FuncRef
AttrValue = int | float | bool | str | Ident | Value | list | dict


def literal_str(v: int | float | bool | complex) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, complex):  # spelled as a call so the text form needs no new token
        return f"complex({literal_str(v.real)}, {literal_str(v.imag)})"
    if isinstance(v, float):
        if v != v:
            return "nan"
        if v in (float("inf"), float("-inf")):
            return "inf" if v > 0 else "-inf"
        s = repr(v)
        return s if ("." in s or "e" in s or "E" in s) else s + ".0"
    return str(v)


@dataclass(frozen=True)
class Op:
    opcode: str
    operands: tuple[Operand, ...] = ()
    results: tuple[Value, ...] = ()
    attrs: Mapping[str, Any] = field(default_factory=dict)
    id: int | None = None
    loc: Loc | None = None
    origin: tuple[Origin, ...] = ()
    regions: tuple[Block, ...] = ()

    @property
    def namespace(self) -> str:
        return self.opcode.split(".", 1)[0]

    def walk(self) -> Iterator[Op]:
        yield self
        for block in self.regions:
            yield from block.walk()

    def attr_values(self) -> Iterator[Value]:
        """Every Value referenced from the attribute tree (optional value inputs, RFC-0001 §6)."""
        yield from _values_in(self.attrs)


def _values_in(obj: Any) -> Iterator[Value]:
    if isinstance(obj, Value):
        yield obj
    elif isinstance(obj, list):
        for x in obj:
            yield from _values_in(x)
    elif isinstance(obj, Mapping):
        for x in obj.values():
            yield from _values_in(x)


@dataclass(frozen=True)
class Block:
    ops: tuple[Op, ...] = ()

    def walk(self) -> Iterator[Op]:
        for op in self.ops:
            yield from op.walk()

    def __len__(self) -> int:
        return len(self.ops)


@dataclass(frozen=True)
class Function:
    kind: str
    name: str
    params: tuple[Value, ...] = ()
    attrs: Mapping[str, Any] = field(default_factory=dict)
    body: Block = field(default_factory=Block)

    def walk(self) -> Iterator[Op]:
        yield from self.body.walk()


@dataclass(frozen=True)
class Module:
    name: str
    attrs: Mapping[str, Any] = field(default_factory=dict)
    functions: tuple[Function, ...] = ()

    @property
    def ir(self) -> str:
        return str(self.attrs.get("ir", SURFACE))

    @property
    def level(self) -> str:
        return self.ir.split("/", 1)[0]

    @property
    def device(self) -> str | None:
        d = self.attrs.get("device")
        return None if d is None else str(d)

    def function(self, name: str) -> Function:
        for f in self.functions:
            if f.name == name:
                return f
        raise KeyError(name)

    def walk(self) -> Iterator[Op]:
        for f in self.functions:
            yield from f.walk()

    def max_id(self) -> int:
        return max((op.id for op in self.walk() if op.id is not None), default=0)


__all__ = [
    "SURFACE", "LOWERED", "FUNCTION_KINDS", "MODES", "SIDES", "Loc", "Origin", "Value", "Literal", "FuncRef", "Ident",
    "Operand", "AttrValue", "Op", "Block", "Function", "Module", "literal_str",
]
