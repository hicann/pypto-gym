# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""JSON form of the IR (RFC-0001 §9.2). Same structure as the text form; types keep their text spelling."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .core import Block, FuncRef, Function, Ident, Literal, Loc, Module, Op, Origin, Value
from .parser import ModuleScope, Scope
from .types import parse_type


def _attr_to_json(v: Any) -> Any:
    if isinstance(v, bool) or isinstance(v, (int, float, str)):
        return v
    if isinstance(v, Ident):
        return {"ident": v.name}
    if isinstance(v, Value):
        return {"value": v.name}
    if isinstance(v, FuncRef):
        return {"func": v.name}
    if isinstance(v, (list, tuple)):
        return [_attr_to_json(x) for x in v]
    if isinstance(v, Mapping):
        return {"attrs": {k: _attr_to_json(x) for k, x in v.items()}}
    raise TypeError(f"cannot serialise attribute {v!r}")


def _attr_from_json(v: Any, scope: Scope) -> Any:
    if isinstance(v, dict):
        if "ident" in v:
            return Ident(v["ident"])
        if "value" in v:
            val = scope.lookup(v["value"])
            if val is None:
                raise ValueError(f"undefined value %{v['value']} in attribute")
            return val
        if "func" in v:
            return FuncRef(v["func"])
        if "attrs" in v:
            return {k: _attr_from_json(x, scope) for k, x in v["attrs"].items()}
        raise ValueError(f"bad attribute object {v!r}")
    if isinstance(v, list):
        return [_attr_from_json(x, scope) for x in v]
    return v


def _operand_to_json(x: Any) -> Any:
    if isinstance(x, Value):
        return {"value": x.name}
    if isinstance(x, Literal):
        if isinstance(x.value, complex):
            return {"lit": [x.value.real, x.value.imag], "complex": True}
        return {"lit": x.value}
    if isinstance(x, FuncRef):
        return {"func": x.name}
    raise TypeError(f"bad operand {x!r}")


def _op_to_json(op: Op) -> dict[str, Any]:
    d: dict[str, Any] = {
        "op": op.opcode,
        "results": [{"name": r.name, "type": str(r.type)} for r in op.results],
        "operands": [_operand_to_json(x) for x in op.operands],
        "attrs": {k: _attr_to_json(v) for k, v in op.attrs.items()},
    }
    if op.id is not None:
        d["id"] = op.id
    if op.loc is not None:
        d["loc"] = list(op.loc.chain)
    if op.origin:
        d["origin"] = [{"pass": o.pass_name, "kind": o.kind, "from": list(o.from_ids), "note": o.note} for o in op.origin]
    if op.regions:
        d["regions"] = [_block_to_json(b) for b in op.regions]
    return d


def _block_to_json(b: Block) -> list[dict[str, Any]]:
    return [_op_to_json(op) for op in b.ops]


def to_json(m: Module) -> dict[str, Any]:
    return {
        "ir": m.ir,
        "name": m.name,
        "attrs": {k: _attr_to_json(v) for k, v in m.attrs.items()},
        "functions": [
            {
                "kind": f.kind, "name": f.name,
                "params": [{"name": p.name, "type": str(p.type)} for p in f.params],
                "attrs": {k: _attr_to_json(v) for k, v in f.attrs.items()},
                "body": _block_to_json(f.body),
            }
            for f in m.functions
        ],
    }


def _op_from_json(d: dict[str, Any], scope: Scope) -> Op:
    operands = []
    for x in d.get("operands", []):
        if "value" in x:
            v = scope.lookup(x["value"])
            if v is None:
                raise ValueError(f"undefined value %{x['value']}")
            operands.append(v)
        elif "lit" in x:
            v = x["lit"]
            operands.append(Literal(complex(*v) if x.get("complex") else v))
        elif "func" in x:
            operands.append(FuncRef(x["func"]))
        else:
            raise ValueError(f"bad operand {x!r}")
    results = tuple(Value(r["name"], parse_type(r["type"])) for r in d.get("results", []))
    attrs = {k: _attr_from_json(v, scope) for k, v in d.get("attrs", {}).items()}
    loc = Loc(tuple(d["loc"])) if d.get("loc") else None
    origin = tuple(Origin(o["pass"], o["kind"], tuple(o.get("from", [])), o.get("note")) for o in d.get("origin", []))
    regions: list[Block] = []
    if d.get("regions"):
        inner = Scope(scope)
        for r in results:
            inner.define(r)
        regions = [_block_from_json(b, inner) for b in d["regions"]]
    else:
        for r in results:
            scope.define(r)
    return Op(d["op"], tuple(operands), results, attrs, d.get("id"), loc, origin, tuple(regions))


def _block_from_json(ops: list[dict[str, Any]], parent: Scope) -> Block:
    scope = Scope(parent)
    return Block(tuple(_op_from_json(d, scope) for d in ops))


def from_json(d: dict[str, Any]) -> Module:
    functions = []
    for f in d.get("functions", []):
        scope = Scope()
        params = tuple(Value(p["name"], parse_type(p["type"])) for p in f.get("params", []))
        for p in params:
            scope.define(p)
        attrs = {k: _attr_from_json(v, scope) for k, v in f.get("attrs", {}).items()}
        functions.append(Function(f["kind"], f["name"], params, attrs, _block_from_json(f.get("body", []), scope)))
    scope = ModuleScope(functions)
    attrs = {k: _attr_from_json(v, scope) for k, v in d.get("attrs", {}).items()}
    if "ir" not in attrs and "ir" in d:
        attrs["ir"] = d["ir"]
    return Module(d["name"], attrs, tuple(functions))


__all__ = ["to_json", "from_json"]
