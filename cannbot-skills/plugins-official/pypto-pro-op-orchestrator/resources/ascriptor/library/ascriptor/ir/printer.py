# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Text form of the IR (RFC-0001 §9.1): ``parse_module(print_module(m)) == m``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .core import Block, FuncRef, Function, Ident, Literal, Module, Op, Origin, Value, literal_str
from .registry import REGISTRY


def quote(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def format_attr_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return literal_str(v)
    if isinstance(v, str):
        return quote(v)
    if isinstance(v, Ident):
        return v.name
    if isinstance(v, Value):
        return str(v)
    if isinstance(v, FuncRef):
        return str(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(format_attr_value(x) for x in v) + "]"
    if isinstance(v, Mapping):
        return format_attrs(v)
    raise TypeError(f"cannot print attribute value of type {type(v).__name__}: {v!r}")


def format_attrs(attrs: Mapping[str, Any]) -> str:
    if not attrs:
        return "{}"
    return "{ " + ", ".join(f"{k} = {format_attr_value(v)}" for k, v in attrs.items()) + " }"


def format_origin(origin: tuple[Origin, ...]) -> list[dict[str, Any]]:
    out = []
    for o in origin:
        d: dict[str, Any] = {"pass": o.pass_name, "kind": Ident(o.kind), "from": list(o.from_ids)}
        if o.note is not None:
            d["note"] = o.note
        out.append(d)
    return out


def format_operand(x: Any) -> str:
    if isinstance(x, (Value, Literal, FuncRef)):
        return str(x)
    raise TypeError(f"bad operand {x!r}")


def format_op_line(op: Op) -> str:
    parts: list[str] = []
    if op.results:
        parts.append(", ".join(str(r) for r in op.results) + " =")
    parts.append(op.opcode + "(" + ", ".join(format_operand(x) for x in op.operands) + ")")
    if op.results:
        types = [str(r.type) for r in op.results]
        parts.append(": " + (types[0] if len(types) == 1 else "(" + ", ".join(types) + ")"))
    attrs = dict(op.attrs)
    if op.origin:
        attrs["origin"] = format_origin(op.origin)
    if attrs:
        parts.append(format_attrs(attrs))
    if op.id is not None:
        parts.append(f"#{op.id}")
    if op.loc is not None:
        parts.append(str(op.loc))
    return " ".join(parts)


def _region_names(op: Op) -> list[str]:
    spec = REGISTRY.find(op.opcode)
    names = list(spec.regions) if spec else []
    while len(names) < len(op.regions):
        names.append(f"region{len(names)}")
    return names


def print_block(block: Block, indent: int, out: list[str], source: list[str] | None = None) -> None:
    pad = "  " * indent
    for op in block.ops:
        if source is not None and op.loc is not None:
            line = _source_line(op.loc.chain[0], source)
            if line is not None:
                out.append(f"{pad};; {line}")
        head = pad + format_op_line(op)
        if not op.regions:
            out.append(head)
            continue
        names = _region_names(op)
        out.append(head + " {")
        for i, region in enumerate(op.regions):
            if i > 0:
                out.append(f"{pad}}} {names[i]} {{")
            print_block(region, indent + 1, out, source)
        out.append(pad + "}")


def _source_line(pos: str, source: list[str]) -> str | None:
    try:
        line_no = int(pos.rsplit(":", 2)[1])
        return source[line_no - 1].strip()
    except (IndexError, ValueError):
        return None


def print_function(f: Function, out: list[str], source: list[str] | None = None) -> None:
    params = ", ".join(f"{p}: {p.type}" for p in f.params)
    head = f"{f.kind} @{f.name}({params})"
    if f.attrs:
        head += " " + format_attrs(f.attrs)
    out.append(head + " {")
    print_block(f.body, 1, out, source)
    out.append("}")


def print_module(m: Module, *, source: str | None = None) -> str:
    """Print a module. With ``source`` (the DSL file's text) every op is preceded by its source line."""
    out: list[str] = []
    head = f"module @{m.name}"
    if m.attrs:
        head += " " + format_attrs(m.attrs)
    out.append(head)
    lines = source.splitlines() if source is not None else None
    for f in m.functions:
        out.append("")
        print_function(f, out, lines)
    return "\n".join(out) + "\n"


__all__ = ["print_module", "print_function", "print_block", "format_op_line", "format_attrs", "format_attr_value", "quote"]
