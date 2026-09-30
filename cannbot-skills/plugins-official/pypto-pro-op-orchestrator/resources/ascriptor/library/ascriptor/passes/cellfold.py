# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""``cellfold``: forward read-only cells to their init values.

The old DSL spelled every derived scalar as a ``Var`` (``half_dim = Var(dim // 2)``), so the
ported corpus is full of cells that are initialised once and never ``scalar.set`` again. Such a
cell is an SSA value wearing a mutable costume, and the costume has a cost: a type dimension
``[%half_dim, %dim]`` names the cell, and every pass that rebuilds a reference from the name
(``dim_scalar``) assumes an i32 *scalar* — a cell-typed dimension came back mistyped and the
module stopped verifying after device_lower (the delta_h family). Folding the cell away removes
the mismatch at the source: dimensions end up naming the init's scalar value, which is exactly
what the rebuilders assume.

A cell is folded when it has an ``init`` and no ``scalar.set`` anywhere in its function, and its
init chain resolves to a non-cell value (an init naming another folded cell is chased; one naming
a *mutable* cell keeps the reader unfolded — forwarding it would re-introduce a cell-typed name in
scalar positions). Literal inits are left alone: some operand patterns require a value, and a
literal-init cell in a dimension has no known instance in the corpus. The init dominates the cell
declaration, and the declaration dominates every use, so forwarding never moves a value across a
definition. References are rewritten everywhere a name can hide: operands, the attribute tree,
result/parameter types, and ``MemType`` dimensions (``DimValue`` / ``Product`` factors, through
``BufType.elem``); the folded ``scalar.cell`` declarations are dropped.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..ir import Block, Function, Module, Op, Value
from ..ir.types import BufType, CellType, DimValue, MemType, Product, Type
from .manager import Pass, PassContext

PASS = "cellfold"


def _fold_map(f: Function) -> dict[str, Value]:
    """cell name -> the non-cell scalar Value its init chain resolves to."""
    written: set[str] = set()
    init: dict[str, Any] = {}
    for op in f.walk():
        if op.opcode == "scalar.set":
            written.add(op.operands[0].name)  # type: ignore[union-attr]
        elif op.opcode == "vf.mask_update":
            cnt = op.attrs.get("cnt")  # the plt POST_UPDATE form decrements the counter IN PLACE (a hardware
            if isinstance(cnt, Value):  # write scalar.set never spells): the cell must survive as a variable
                written.add(cnt.name)
        elif op.opcode == "scalar.cell":
            init[op.results[0].name] = op.attrs.get("init")
    fold: dict[str, Value] = {}
    for name, v in init.items():
        if name in written or not isinstance(v, Value):
            continue
        seen = {name}
        while isinstance(v, Value) and isinstance(v.type, CellType):
            if v.name in written or v.name in seen or not isinstance(init.get(v.name), Value):
                v = None  # a mutable / literal-init / cyclic source: keep this reader unfolded
                break
            seen.add(v.name)
            v = init[v.name]
        if isinstance(v, Value) and not isinstance(v.type, CellType):
            fold[name] = v
    return fold


def _subst_dim(d: Any, fold: dict[str, Value]) -> Any:
    if isinstance(d, DimValue) and d.name in fold:
        return DimValue(fold[d.name].name)
    if isinstance(d, Product):
        return Product(tuple(DimValue(fold[x.name].name) if isinstance(x, DimValue) and x.name in fold else x
                             for x in d.factors))
    return d


def _subst_type(t: Type, fold: dict[str, Value]) -> Type:
    et = t.elem if isinstance(t, BufType) else t
    if isinstance(et, MemType):
        dims = tuple(_subst_dim(d, fold) for d in et.dims)
        if dims != et.dims:
            et = replace(et, dims=dims)
            return replace(t, elem=et) if isinstance(t, BufType) else et
    return t


def _subst_value(v: Value, fold: dict[str, Value]) -> Value:
    r = fold.get(v.name)
    name, t = (r.name, r.type) if r is not None else (v.name, v.type)
    return Value(name, _subst_type(t, fold))


def _subst_attr(a: Any, fold: dict[str, Value]) -> Any:
    if isinstance(a, Value):
        return _subst_value(a, fold)
    if isinstance(a, list):
        return [_subst_attr(x, fold) for x in a]
    if isinstance(a, tuple):
        return tuple(_subst_attr(x, fold) for x in a)
    if isinstance(a, dict):
        return {k: _subst_attr(x, fold) for k, x in a.items()}
    return a


def fold_function(f: Function, ctx: PassContext | None = None) -> Function:
    fold = _fold_map(f)
    if not fold:
        return f

    def walk(block: Block) -> Block:
        out = []
        for op in block.ops:
            if op.opcode == "scalar.cell" and op.results[0].name in fold:
                continue
            out.append(replace(
                op,
                operands=tuple(_subst_value(x, fold) if isinstance(x, Value) else x for x in op.operands),
                results=tuple(Value(r.name, _subst_type(r.type, fold)) for r in op.results),
                attrs={k: _subst_attr(v, fold) for k, v in op.attrs.items()},
                regions=tuple(walk(r) for r in op.regions),
            ))
        return Block(tuple(out))

    if ctx is not None:
        ctx.explain.note(f"@{f.name}: {len(fold)} read-only cell(s) forwarded to their init", kind="cellfold")
    params = tuple(Value(p.name, _subst_type(p.type, fold)) for p in f.params)
    return replace(f, params=params, body=walk(f.body))


def run(module: Module, ctx: PassContext) -> Module:
    return Module(module.name, dict(module.attrs), tuple(fold_function(f, ctx) for f in module.functions))


PASS_DEF = Pass(PASS, run, accepts="surface/1", produces="surface/1",
                doc="forward read-only cells to their init values (cells out of type dimensions)")

__all__ = ["PASS_DEF", "fold_function", "run"]
