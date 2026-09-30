# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The op registry (RFC-0001 §6): the single source of truth for every opcode.

An :class:`OpSpec` says what an op is (level, function kinds, side, pipe, devices), what it takes
(operands with type patterns and read/write access, attributes with types), what it produces
(result patterns), whether it owns regions, and which old ``easyasc`` instruction names it
replaces (``legacy``). The verifier, autosync, the sim router and the capability matrices are
all derived from these records; nothing else may hard-code an opcode's shape.

Type patterns are type spellings with variables: a capital letter in dtype position binds a
dtype (``T``), a lowercase identifier in dimension position binds a dimension (``r``), ``*``
matches anything in its position, and the pseudo-spaces ``local`` / ``l0`` / ``mem`` match a
family of memory spaces. Standalone ``scalar`` / ``int`` / ``float`` / ``bool`` match scalar
types by class; a standalone capital letter binds a whole type; ``value`` matches any type.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .lexer import TokenStream, tokenize
from .types import (
    DTYPES,
    LOCAL_SPACES,
    MEM_SPACES,
    BufType,
    CellType,
    Dim,
    DType,
    EventType,
    FlagType,
    MaskType,
    MemType,
    RegType,
    ScalarType,
    Type,
    UnalignRegType,
)

LEVELS = ("surface", "lowered", "both")
SIDES = ("cube", "vec", "any", "both")
KINDS = ("kernel", "vf", "simt")
ACCESS = ("read", "write", "readwrite", "none")
ATTR_TYPES = ("int", "float", "bool", "str", "ident", "value", "list", "dims", "any", "type")

# --------------------------------------------------------------------------- patterns


@dataclass(frozen=True)
class Pat:
    """A parsed type pattern. ``kind`` selects the matcher; ``args`` holds its sub-patterns."""

    kind: str
    args: tuple[Any, ...] = ()

    def __str__(self) -> str:
        return self.args[-1] if self.kind == "src" else self.kind


def parse_pattern(text: str) -> Pat:
    ts = TokenStream(tokenize(text))
    p = _parse_pat(ts)
    if not ts.at("eof"):
        raise ts.error(f"trailing tokens in pattern {text!r}")
    return Pat("src", (p, text))


def _parse_pat(ts: TokenStream) -> Pat:
    if ts.at_punct("*"):
        ts.advance()
        return Pat("any")
    head = ts.expect("ident").text
    if head == "value":
        return Pat("any")
    if head in ("scalar", "int", "float", "bool"):
        return Pat("scalar_class", (head,))
    if head in DTYPES:
        return Pat("scalar", (head,))
    if len(head) == 1 and head.isupper():
        return Pat("typevar", (head,))
    if head in MEM_SPACES or head in ("local", "l0", "mem"):
        ts.expect_punct("<")
        dt = _parse_dtype_pat(ts)
        dims: Any = Pat("any")
        layout: Any = Pat("any")
        if ts.accept("punct", ","):
            dims = _parse_dims_pat(ts)
            if ts.accept("punct", ","):
                layout = _parse_layout_pat(ts)
        ts.expect_punct(">")
        return Pat("mem", (head, dt, dims, layout))
    if head == "buf":
        ts.expect_punct("<")
        elem = _parse_pat(ts)
        slots: Any = Pat("any")
        if ts.accept("punct", ","):
            slots = _parse_num_pat(ts)
        ts.expect_punct(">")
        return Pat("buf", (elem, slots))
    if head == "reg":
        ts.expect_punct("<")
        dt = _parse_dtype_pat(ts)
        n: Any = Pat("any")
        if ts.accept("punct", ","):
            n = _parse_num_pat(ts)
        ts.expect_punct(">")
        return Pat("reg", (dt, n))
    if head == "mask":
        w: Any = Pat("any")
        if ts.accept("punct", "<"):
            if ts.at_punct("*"):
                ts.advance()
            else:
                w = Pat("num", (int(ts.expect("ident").text[1:]),))
            ts.expect_punct(">")
        return Pat("mask", (w,))
    if head == "unalignreg":
        role: Any = Pat("any")
        if ts.accept("punct", "<"):
            if ts.at_punct("*"):
                ts.advance()
            else:
                role = Pat("role", (ts.expect("ident").text,))
            ts.expect_punct(">")
        return Pat("unalignreg", (role,))
    if head == "cell":
        ts.expect_punct("<")
        dt = _parse_dtype_pat(ts)
        ts.expect_punct(">")
        return Pat("cell", (dt,))
    if head == "event":
        if ts.accept("punct", "<"):
            while not ts.at_punct(">"):
                ts.advance()
            ts.expect_punct(">")
        return Pat("event")
    if head == "flag":
        if ts.accept("punct", "<"):
            while not ts.at_punct(">"):
                ts.advance()
            ts.expect_punct(">")
        return Pat("flag")
    raise ts.error(f"unknown pattern head {head!r}")


def _parse_dtype_pat(ts: TokenStream) -> Pat:
    if ts.at_punct("*"):
        ts.advance()
        return Pat("any")
    name = ts.expect("ident").text
    if name in DTYPES:
        return Pat("dtype", (name,))
    if name in ("int", "float", "bool"):
        return Pat("dtype_class", (name,))
    if len(name) == 1 and name.isupper():
        return Pat("dtypevar", (name,))
    raise ts.error(f"bad dtype pattern {name!r}")


def _parse_num_pat(ts: TokenStream) -> Pat:
    if ts.at_punct("*"):
        ts.advance()
        return Pat("any")
    if ts.at("int"):
        return Pat("num", (int(ts.advance().text, 0),))
    name = ts.expect("ident").text
    return Pat("dimvar", (name,))


def _parse_dims_pat(ts: TokenStream) -> Pat:
    if ts.at_punct("*"):
        ts.advance()
        return Pat("any")
    ts.expect_punct("[")
    items: list[Pat] = []
    if not ts.at_punct("]"):
        items.append(_parse_dim_pat(ts))
        while ts.accept("punct", ","):
            items.append(_parse_dim_pat(ts))
    ts.expect_punct("]")
    return Pat("dims", tuple(items))


def _parse_dim_pat(ts: TokenStream) -> Pat:
    if ts.at_punct("*"):
        ts.advance()
        return Pat("any")
    if ts.at_punct("?"):
        ts.advance()
        return Pat("ragged")
    if ts.at("int"):
        return Pat("num", (int(ts.advance().text, 0),))
    name = ts.expect("ident").text
    return Pat("dimvar", (name,))


def _parse_layout_pat(ts: TokenStream) -> Pat:
    if ts.at_punct("*"):
        ts.advance()
        return Pat("any")
    name = ts.expect("ident").text
    if len(name) == 1 and name.isupper():
        return Pat("layoutvar", (name,))
    return Pat("layout", (name,))


_SPACE_CLASSES = {"local": set(LOCAL_SPACES), "l0": {"l0a", "l0b"}, "mem": set(MEM_SPACES), "gm": {"gm", "ws"}}  # workspaces are GM memory


def match(pattern: Pat, t: Type, env: dict[str, Any]) -> bool:
    """Unify ``t`` against ``pattern``, extending ``env`` (variable name -> binding)."""
    if pattern.kind == "src":
        return match(pattern.args[0], t, env)
    k = pattern.kind
    if k == "any":
        return True
    if k == "typevar":
        name = "type:" + pattern.args[0]
        if name in env:
            return env[name] == t
        env[name] = t
        return True
    if k == "scalar":
        return isinstance(t, ScalarType) and t.dtype.name == pattern.args[0]
    if k == "scalar_class":
        return isinstance(t, ScalarType) and _dtype_in_class(t.dtype, pattern.args[0])
    if k == "mem":
        space, dt, dims, layout = pattern.args
        if not isinstance(t, MemType):
            return False
        if space in _SPACE_CLASSES:
            if t.space not in _SPACE_CLASSES[space]:
                return False
        elif t.space != space:
            return False
        return _match_dtype(dt, t.dtype, env) and _match_dims(dims, t.dims, env) and _match_layout(layout, t.layout, env)
    if k == "buf":
        elem, slots = pattern.args
        return isinstance(t, BufType) and match(elem, t.elem, env) and _match_num(slots, t.slots, env)
    if k == "reg":
        dt, n = pattern.args
        return isinstance(t, RegType) and _match_dtype(dt, t.dtype, env) and _match_num(n, t.n, env)
    if k == "mask":
        (w,) = pattern.args
        return isinstance(t, MaskType) and _match_num(w, t.width, env)
    if k == "unalignreg":
        (role,) = pattern.args
        return isinstance(t, UnalignRegType) and (role.kind == "any" or t.role == role.args[0])
    if k == "cell":
        (dt,) = pattern.args
        return isinstance(t, CellType) and _match_dtype(dt, t.dtype, env)
    if k == "event":
        return isinstance(t, EventType)
    if k == "flag":
        return isinstance(t, FlagType)
    raise AssertionError(f"unhandled pattern kind {k}")


def _dtype_in_class(d: DType, cls: str) -> bool:
    if cls == "scalar":
        return True
    if cls == "int":
        return d.kind in ("int", "uint")
    if cls == "float":
        return d.kind == "float"
    if cls == "bool":
        return d.kind == "bool"
    return False


def _match_dtype(p: Pat, d: DType, env: dict[str, Any]) -> bool:
    if p.kind == "any":
        return True
    if p.kind == "dtype":
        return d.name == p.args[0]
    if p.kind == "dtype_class":
        return _dtype_in_class(d, p.args[0])
    if p.kind == "dtypevar":
        name = "dtype:" + p.args[0]
        if name in env:
            return env[name] == d
        env[name] = d
        return True
    raise AssertionError(p)


def _match_num(p: Pat, n: int, env: dict[str, Any]) -> bool:
    if p.kind == "any":
        return True
    if p.kind == "num":
        return n == p.args[0]
    if p.kind == "dimvar":
        return _bind_dim(p.args[0], n, env)
    raise AssertionError(p)


def _bind_dim(name: str, d: Dim, env: dict[str, Any]) -> bool:
    key = "dim:" + name
    if key in env:
        return env[key] == d
    env[key] = d
    return True


def _match_dims(p: Pat, dims: tuple[Dim, ...], env: dict[str, Any]) -> bool:
    if p.kind == "any":
        return True
    items = p.args
    if len(items) != len(dims):
        return False
    for ip, d in zip(items, dims, strict=True):
        if ip.kind == "any":
            continue
        if ip.kind == "ragged":
            if d.__class__.__name__ != "Ragged":
                return False
        elif ip.kind == "num":
            if d != ip.args[0]:
                return False
        elif ip.kind == "dimvar":
            if not _bind_dim(ip.args[0], d, env):
                return False
        else:
            raise AssertionError(ip)
    return True


def _match_layout(p: Pat, layout: str | None, env: dict[str, Any]) -> bool:
    if p.kind == "any":
        return True
    if p.kind == "layout":
        return layout == p.args[0]
    if p.kind == "layoutvar":
        key = "layout:" + p.args[0]
        if key in env:
            return env[key] == layout
        env[key] = layout
        return True
    raise AssertionError(p)


# --------------------------------------------------------------------------- specs


@dataclass(frozen=True)
class OperandSpec:
    name: str
    pattern: str
    access: str = "read"
    variadic: bool = False
    doc: str = ""

    def __post_init__(self) -> None:
        if self.access not in ACCESS:
            raise ValueError(f"bad access {self.access!r} for operand {self.name}")
        object.__setattr__(self, "_pat", parse_pattern(self.pattern))

    @property
    def pat(self) -> Pat:
        return self._pat  # type: ignore[attr-defined]


@dataclass(frozen=True)
class AttrSpec:
    name: str
    type: str = "any"  # one of ATTR_TYPES, or a '|'-union such as "int|value"
    required: bool = False
    default: Any = None
    pattern: str | None = None  # for value-typed attrs: the type pattern the value must match
    access: str = "none"  # for value-typed attrs that read or write a buffer (e.g. mask, bias)
    doc: str = ""

    def __post_init__(self) -> None:
        for t in self.type.split("|"):
            if t not in ATTR_TYPES:
                raise ValueError(f"bad attr type {t!r} for attr {self.name}")
        object.__setattr__(self, "_pat", parse_pattern(self.pattern) if self.pattern else None)

    @property
    def pat(self) -> Pat | None:
        return self._pat  # type: ignore[attr-defined]

    @property
    def types(self) -> tuple[str, ...]:
        return tuple(self.type.split("|"))


@dataclass(frozen=True)
class ResultSpec:
    name: str
    pattern: str
    doc: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "_pat", parse_pattern(self.pattern))

    @property
    def pat(self) -> Pat:
        return self._pat  # type: ignore[attr-defined]


@dataclass(frozen=True)
class OpSpec:
    name: str
    level: str = "both"
    kinds: frozenset[str] = frozenset({"kernel"})
    side: str = "any"
    pipe: str | None = None
    operands: tuple[OperandSpec, ...] = ()
    attrs: tuple[AttrSpec, ...] = ()
    results: tuple[ResultSpec, ...] = ()
    regions: tuple[str, ...] = ()
    devices: frozenset[str] | None = None  # None: every device
    dtypes: frozenset[str] | None = None  # None: whatever the patterns admit
    terminator: bool = False
    effects: frozenset[str] = frozenset()  # memory | sync | debug | control
    legacy: tuple[str, ...] = ()
    doc: str = ""

    def __post_init__(self) -> None:
        if "." not in self.name:
            raise ValueError(f"opcode {self.name!r} must be namespaced (ns.name)")
        if self.level not in LEVELS:
            raise ValueError(f"{self.name}: bad level {self.level!r}")
        if self.side not in SIDES:
            raise ValueError(f"{self.name}: bad side {self.side!r}")
        if not self.kinds or not self.kinds <= set(KINDS):
            raise ValueError(f"{self.name}: bad kinds {sorted(self.kinds)}")
        names = [o.name for o in self.operands] + [a.name for a in self.attrs] + [r.name for r in self.results]
        if len(names) != len(set(names)):
            raise ValueError(f"{self.name}: duplicate operand/attr/result names")
        if any(o.variadic for o in self.operands[:-1]):
            raise ValueError(f"{self.name}: only the last operand may be variadic")

    @property
    def namespace(self) -> str:
        return self.name.split(".", 1)[0]

    def attr(self, name: str) -> AttrSpec | None:
        for a in self.attrs:
            if a.name == name:
                return a
        return None

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name, "level": self.level, "kinds": sorted(self.kinds), "side": self.side, "pipe": self.pipe,
            "operands": [{"name": o.name, "pattern": o.pattern, "access": o.access, "variadic": o.variadic} for o in self.operands],
            "attrs": [{"name": a.name, "type": a.type, "required": a.required, "default": a.default, "pattern": a.pattern, "access": a.access} for a in self.attrs],
            "results": [{"name": r.name, "pattern": r.pattern} for r in self.results],
            "regions": list(self.regions), "devices": None if self.devices is None else sorted(self.devices),
            "dtypes": None if self.dtypes is None else sorted(self.dtypes), "terminator": self.terminator,
            "effects": sorted(self.effects), "legacy": list(self.legacy), "doc": self.doc,
        }


class Registry:
    def __init__(self) -> None:
        self._ops: dict[str, OpSpec] = {}
        self._legacy: dict[str, str] = {}
        self.retired: dict[str, str] = {}  # old name -> reason it has no successor

    def register(self, spec: OpSpec) -> OpSpec:
        if spec.name in self._ops:
            raise ValueError(f"opcode {spec.name!r} registered twice")
        for old in spec.legacy:
            if old in self._legacy:
                raise ValueError(f"legacy name {old!r} claimed by both {self._legacy[old]} and {spec.name}")
            self._legacy[old] = spec.name
        self._ops[spec.name] = spec
        return spec

    def retire(self, old: str, reason: str) -> None:
        if old in self._legacy:
            raise ValueError(f"{old!r} is both mapped to {self._legacy[old]} and retired")
        self.retired[old] = reason

    def __contains__(self, name: str) -> bool:
        return name in self._ops

    def __len__(self) -> int:
        return len(self._ops)

    def get(self, name: str) -> OpSpec:
        try:
            return self._ops[name]
        except KeyError:
            raise KeyError(f"unknown opcode {name!r}") from None

    def find(self, name: str) -> OpSpec | None:
        return self._ops.get(name)

    def all(self) -> list[OpSpec]:
        return [self._ops[k] for k in sorted(self._ops)]

    def namespaces(self) -> dict[str, list[OpSpec]]:
        out: dict[str, list[OpSpec]] = {}
        for spec in self.all():
            out.setdefault(spec.namespace, []).append(spec)
        return out

    def successor(self, old_name: str) -> str | None:
        return self._legacy.get(old_name)

    def legacy_names(self) -> Mapping[str, str]:
        return dict(self._legacy)

    def to_json(self) -> str:
        return json.dumps({"ir": "1", "ops": [s.to_json() for s in self.all()], "retired": self.retired}, indent=1)


REGISTRY = Registry()


def load_builtin_ops() -> Registry:
    """Import the built-in op tables (idempotent) and return the registry."""
    from . import ops  # noqa: F401  (registers on import)

    return REGISTRY


__all__ = [
    "Pat", "parse_pattern", "match", "OperandSpec", "AttrSpec", "ResultSpec", "OpSpec", "Registry", "REGISTRY",
    "load_builtin_ops", "LEVELS", "SIDES", "KINDS", "ACCESS", "ATTR_TYPES",
]


def _unused(_: Iterable[Any]) -> None:  # keeps the Iterable import honest for type checkers
    return None
