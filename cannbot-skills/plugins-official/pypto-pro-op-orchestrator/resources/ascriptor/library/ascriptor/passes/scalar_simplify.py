# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Conservative integer scalar cleanup with immutable snapshot preservation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace

from ..ir import REGISTRY, Block, Ident, Literal, Module, Rewriter, Value
from ..ir.scalar_flow import CellRanges, cell_writes
from ..ir.scalar_math import evaluate, limits, rounding
from ..ir.scalar_range import ScalarRanges
from ..ir.types import CellType, DimValue, Product, ScalarType, Type
from .manager import Pass, PassContext
from .scalar_hoist import hoist, reads
from .scalar_values import initializer, remainder_identity

PURE = {"scalar." + name for name in (
    "const", "add", "sub", "mul", "neg", "abs", "and", "or", "xor", "not", "min", "max", "cmp", "select", "cast",
)}
PURE_VIEWS = {"mem.get_buf", "mem.slice", "mem.reinterpret", "mem.reshape", "mem.view"}
ARITHMETIC = PURE | {"scalar.div", "scalar.mod", "scalar.ceil_div", "scalar.align"}


def _names(obj):
    if isinstance(obj, (Value, DimValue)):
        yield obj.name
        if isinstance(obj, Value):
            yield from _names(obj.type)
    elif isinstance(obj, Mapping):
        for value in obj.values():
            yield from _names(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _names(value)
    elif is_dataclass(obj):
        for field in fields(obj):
            yield from _names(getattr(obj, field.name))


def _integer(value):
    return isinstance(value, Literal) and isinstance(value.value, (int, bool)) or (
        isinstance(value, Value) and isinstance(value.type, (ScalarType, CellType)) and value.type.dtype.is_integer)


def _literal(value):
    return value.value if isinstance(value, Literal) else None


def _evaluate(op):
    """Exact evaluation only within the declared integer domain."""
    code = op.opcode.removeprefix("scalar.")
    args = [op.attrs.get("value")] if code == "const" else [_literal(x) for x in op.operands]
    dt = op.results[0].type.dtype
    value = evaluate(code, args, dt, rounding=rounding(op), n=op.attrs.get("n"), pred=op.attrs.get("pred"))
    # the pass folds a cast only where it is the identity; a narrowing one keeps its op (RFC-0006)
    return None if code == "cast" and dt.kind != "bool" and value != args[0] else value


def simplify_function(fn, rw, ctx):
    fn = hoist(fn, rw, ctx)
    aliases, constants, definitions = {}, {}, {}
    bindings = ctx.option("scalar_bindings", {}) if fn.kind == "func" else {}
    for p in fn.params:
        value = bindings.get(p.name)
        if (isinstance(p.type, ScalarType) and p.type.dtype.is_integer and type(value) is int
                and limits(p.type.dtype)[0] <= value <= limits(p.type.dtype)[1]):
            constants[p.name] = value
    core_ranges = ctx.option("scalar_core_ranges", {}) if fn.kind == "func" else {}
    cell_ranges = CellRanges(fn, bindings=bindings, core_ranges=core_ranges)
    facts = {name: interval for name, interval in cell_ranges.snapshots.items() if interval is not None}
    ranges = ScalarRanges(definitions=definitions, facts=facts)
    type_names = set(_names(tuple(p.type for p in fn.params)))
    for op in fn.walk():
        type_names.update(_names(tuple(v.type for v in op.results)))

    def resolve(value, literal=True):
        if not isinstance(value, Value) or isinstance(value.type, CellType):
            return value
        while value.name in aliases:
            value = aliases[value.name]
        return Literal(constants[value.name]) if literal and value.name in constants else value

    def attr(value, literal=True):
        if isinstance(value, Value):
            result = resolve(value, literal)
            return result.value if isinstance(result, Literal) else result
        if isinstance(value, list):
            return [attr(x, literal) for x in value]
        if isinstance(value, tuple):
            return tuple(attr(x, literal) for x in value)
        if isinstance(value, dict):
            return {k: attr(v, literal) for k, v in value.items()}
        return value

    def attribute(spec, name, value):
        types = set(spec.attr(name).types)
        literal = bool(types & {"int", "bool", "list", "dims", "any"})
        if (types <= {"bool", "value"} and isinstance(value, Value)
                and getattr(getattr(value.type, "dtype", None), "name", None) != "b1"):
            literal = False  # preserve an integer Value's explicit truth conversion
        return attr(value, literal)

    def affine(value, typ, seen=frozenset(), *, bounded=False):
        if isinstance(value, Literal):
            return (int(value.value), {}) if isinstance(value.value, (int, bool)) else None
        if not isinstance(value, Value) or value.type != typ:
            return None  # excludes Cells, floating operands and implicit width changes
        if value.name in seen:
            return None
        op = definitions.get(value.name)
        opaque = (0, {value.name: 1})
        if op is None or op.opcode not in {"scalar.add", "scalar.sub", "scalar.mul", "scalar.neg"}:
            return opaque
        if bounded and ranges.bounds(value) is None:
            return opaque
        parts = [affine(x, typ, seen | {value.name}, bounded=bounded) for x in op.operands]
        if any(p is None for p in parts):
            return opaque
        if op.opcode == "scalar.neg":
            k, terms = parts[0]
            return -k, {v: -c for v, c in terms.items()}
        (ka, ta), (kb, tb) = parts
        if op.opcode == "scalar.mul":
            if ta and tb:
                return opaque
            k, terms = (kb, ta) if not tb else (ka, tb)
            return ka * kb, {v: c * k for v, c in terms.items() if c * k}
        sign = -1 if op.opcode == "scalar.sub" else 1
        terms = dict(ta)
        for value, coefficient in tb.items():
            terms[value] = terms.get(value, 0) + sign * coefficient
            if not terms[value]:
                del terms[value]
        return ka + sign * kb, terms

    def cancels_to_one_term(form, result):
        """Is this affine form one OTHER existing value, with nothing added to it?

        The opaque form of a value is itself, which satisfies everything else here and rebuilds to nothing.
        """
        if form is None or form[0] or len(form[1]) != 1:
            return False
        (name, coefficient), = form[1].items()
        return coefficient == 1 and name != result.name

    def key(op):
        return (op.opcode, str(op.results[0].type), op.operands,
                tuple(sorted((k, str(v)) for k, v in op.attrs.items())))

    def alias_to(op, value, kind):
        aliases[op.results[0].name] = value
        ctx.explain.note(f"%{op.results[0].name} reuses %{value.name}", op=op.id, kind=kind)
        return None

    def rebuild(op, form, output, common):
        """Reconstruct one-variable affine arithmetic only with a no-overflow proof.

        The proof guards the arithmetic a reconstruction ADDS, so the two forms that add none - a bare
        alias, and the one op the expression already is - do not need it and must not cost an op.
        """
        if form is None or len(form[1]) != 1:
            return op
        offset, terms = form
        name, coefficient = next(iter(terms.items()))
        if name == op.results[0].name:
            return op
        typ = op.results[0].type
        value = next((v for v in definitions[name].results if v.name == name), None) if name in definitions else None
        if value is None:
            value = next((v for v in fn.params if v.name == name), None)
        if value is None or value.type != typ:
            return op
        if coefficient == 1 and not offset:
            # `(a + v) - a` is `v` for every pair of operands: the cancelled term is the only place the
            # original could have overflowed, and the alias keeps no arithmetic at all. Requiring a range
            # for `v` here would refuse the cancellation exactly where the range analysis is one-sided.
            return alias_to(op, value, "scalar-affine")
        bounds = ranges.bounds(value)
        lo, hi = limits(typ.dtype)
        if bounds is None or not lo <= coefficient <= hi or not lo <= offset <= hi:
            return op
        products = (bounds[0] * coefficient, bounds[1] * coefficient)
        if not (lo <= min(products) <= max(products) <= hi
                and lo <= min(products) + offset <= max(products) + offset <= hi):
            return op
        if coefficient == -1 and offset:
            # `c - x` is one subtraction. Spelling it `x * -1` and then `+ c` is two ops and one extra
            # name, and that name is what `offset_base.N` was: an artefact of this reconstruction.
            operands = (Literal(offset), value)
            return op if op.opcode == "scalar.sub" and op.operands == operands else rw.rewritten(
                op, "reconstruct bounded integer affine expression", opcode="scalar.sub", operands=operands, attrs={})
        if coefficient != 1:
            mult = rw.rewritten(op, "reconstruct bounded integer affine expression", opcode="scalar.mul",
                                operands=(value, Literal(coefficient)), attrs={})
            if not offset:
                return op if (op.opcode, op.operands, op.attrs) == (mult.opcode, mult.operands, mult.attrs) else mult
            old = common.get(key(mult))
            if old is None and coefficient < 0:
                # `c - k*x`: the program usually already holds `k*x`, because that is the offset the
                # extent was measured from. Reusing it is one subtraction; building `x * -k` instead
                # computes a second product of the same two operands and needs a name for it.
                positive = rw.rewritten(op, "reconstruct bounded integer affine expression", opcode="scalar.mul",
                                        operands=(value, Literal(-coefficient)), attrs={})
                shared = common.get(key(positive))
                if shared is not None:
                    operands = (Literal(offset), shared.results[0])
                    return op if op.opcode == "scalar.sub" and op.operands == operands else rw.rewritten(
                        op, "reconstruct bounded integer affine expression", opcode="scalar.sub",
                        operands=operands, attrs={})
            if old is None:
                return op  # a fresh base is an op the expression did not have; keep what is written
            value = old.results[0]
        if not offset:
            return alias_to(op, value, "scalar-affine")
        operands = (value, Literal(offset))
        return op if op.opcode == "scalar.add" and op.operands == operands else rw.rewritten(
            op, "reconstruct bounded integer affine expression", opcode="scalar.add", operands=operands, attrs={})

    def total(op):
        if op.opcode in PURE:
            return True
        if op.opcode not in {"scalar.div", "scalar.mod"}:
            return False
        # Positive constant divisors cannot trap for any representable dividend.
        divisor = _literal(op.operands[1])
        typ = op.results[0].type
        return (isinstance(typ, ScalarType) and typ.dtype.is_integer and type(divisor) is int
                and 0 < divisor <= limits(typ.dtype)[1])

    def walk(block, incoming=None):
        output, common, current = [], dict(incoming or {}), {}
        for original in block.ops:
            spec = REGISTRY.get(original.opcode)
            operands = tuple(resolve(current.get(x.name, x)) if original.opcode in ARITHMETIC
                             and isinstance(x, Value) and isinstance(x.type, CellType) else resolve(x)
                             for x in original.operands)
            attrs = {k: attribute(spec, k, v) for k, v in original.attrs.items()}
            op = original
            if operands != op.operands or attrs != op.attrs:
                op = rw.rewritten(op, "propagate immutable integer values", operands=operands, attrs=attrs)
                if operands != tuple(resolve(x) for x in original.operands):
                    ctx.explain.note("read the current immutable initializer", op=op.id, kind="cell-initializer-forward")
            for result in op.results:
                definitions[result.name] = op
                if op.opcode in core_ranges:
                    facts[result.name] = core_ranges[op.opcode]
            if op.opcode in core_ranges and core_ranges[op.opcode][0] == core_ranges[op.opcode][1]:
                op = rw.rewritten(op, "specialize effective launch geometry", opcode="scalar.const", operands=(),
                                  attrs={"value": core_ranges[op.opcode][0]})
            if op.regions:
                condition = _literal(op.operands[0]) if op.opcode == "cf.if" else None
                if isinstance(condition, (int, bool)):
                    arm = 0 if condition else 1
                    selected = op.regions[arm] if arm < len(op.regions) else Block(())
                    body, common = walk(selected, common)
                    output.extend(rw.rewritten(child, "selected arm of a proven constant conditional") for child in body.ops)
                    current.clear()
                    ctx.explain.note(f"retain only arm {arm} of constant conditional", op=op.id, kind="constant-branch")
                    continue
                branches = [walk(r, common if op.opcode == "cf.if" else None) for r in op.regions]
                # Only already-dominating captures may survive a branch join.
                # Child definitions cannot escape, and loops start a fresh epoch.
                common = {k: v for k, v in common.items()
                          if op.opcode == "cf.if" and all(table.get(k) is v for _, table in branches)}
                op = replace(op, regions=tuple(body for body, _ in branches))
                current.clear()
            written = cell_writes(op)
            if written:
                common = {k: previous for k, previous in common.items() if not reads(previous) & written}
                current.clear()
            if op.opcode.startswith("cf.") and op.opcode != "cf.if" or op.opcode == "simt.launch":
                # A call cannot rewrite a value, only a cell, and `cell_writes` treats every cell it is
                # handed as written - so an expression that reads no cell at all is the same value on
                # both sides of it. That is what keeps one `beat & 1` across a vector function call.
                common = {k: previous for k, previous in common.items()
                          if op.opcode == "cf.call" and not reads(previous)}
                current.clear()
            eligible = (op.opcode in ARITHMETIC and len(op.results) == 1
                        and isinstance(op.results[0].type, ScalarType) and op.results[0].type.dtype.is_integer
                        and all(_integer(x) for x in op.operands))
            if eligible:
                result = op.results[0]
                rewritten = remainder_identity(op, definitions, ranges, rw)
                if rewritten is not op:
                    ctx.explain.note("quotient/product subtraction is remainder", op=op.id, kind="quotient-remainder")
                    op = rewritten
                definitions[result.name] = op
                # Same-type identities must not turn captured Cell reads into late reads.
                identity = None
                if op.opcode in {"scalar.add", "scalar.sub", "scalar.mul"}:
                    a, b = op.operands
                    if _literal(b) == (1 if op.opcode == "scalar.mul" else 0):
                        identity = a
                    elif op.opcode in {"scalar.add", "scalar.mul"} and _literal(a) == (1 if op.opcode == "scalar.mul" else 0):
                        identity = b
                elif op.opcode in {"scalar.min", "scalar.max"}:
                    a, b = op.operands
                    left, right = ranges.bounds(a), ranges.bounds(b)
                    if left is not None and right is not None:
                        if left[1] <= right[0]:
                            identity = a if op.opcode == "scalar.min" else b
                        elif right[1] <= left[0]:
                            identity = b if op.opcode == "scalar.min" else a
                if (isinstance(identity, Value) and identity.type == result.type and result.name not in type_names):
                    aliases[result.name] = identity
                    ctx.explain.note(f"%{result.name} reuses %{identity.name}", op=op.id, kind="scalar-identity")
                    continue
                if op.opcode in {"scalar.mod", "scalar.and"} and result.name not in type_names:
                    value, operand = op.operands
                    constant = _literal(operand)
                    divisor = constant if op.opcode == "scalar.mod" else constant + 1 if type(constant) is int else None
                    eligible_wrap = (rounding(op) == "floor" if op.opcode == "scalar.mod" else
                                     type(divisor) is int and divisor > 0 and divisor & (divisor - 1) == 0)
                    if (eligible_wrap and isinstance(value, Value) and value.type == result.type
                            and ranges.normalized(value, divisor)):
                        aliases[result.name] = value
                        ctx.explain.note(f"%{result.name} reuses normalized %{value.name}", op=op.id,
                                         kind="redundant-slot-wrap")
                        continue
                    interval = cell_ranges.before.get(id(original), {}).get(getattr(value, "name", None))
                    if (eligible_wrap and isinstance(value, Value) and isinstance(value.type, CellType)
                            and value.type.dtype == result.type.dtype and type(divisor) is int and divisor > 0
                            and interval is not None and 0 <= interval[0] <= interval[1] < divisor):
                        op = rw.rewritten(op, "capture a Cell proven normalized at this program point",
                                          opcode="scalar.add", operands=(value, Literal(0)), attrs={})
                        definitions[result.name] = op
                        facts[result.name] = interval
                        ctx.explain.note(f"%{result.name} captures %{value.name} in {interval}", op=op.id,
                                         kind="normalized-cell-snapshot")
                known = _evaluate(op)
                interval = ranges.bounds(result)
                if known is None and interval is not None and interval[0] == interval[1]:
                    known = interval[0]
                if known is None and op.opcode in {"scalar.add", "scalar.sub", "scalar.mul", "scalar.neg"}:
                    definitions[result.name] = op
                    linear = affine(result, result.type)
                    if linear is not None and not linear[1] and limits(result.type.dtype)[0] <= linear[0] <= limits(result.type.dtype)[1]:
                        known = linear[0]
                form = None
                if known is None and op.opcode in {"scalar.div", "scalar.mod"}:
                    divisor = _literal(op.operands[1])
                    dividend = ranges.bounds(op.operands[0])
                    value = op.operands[0]
                    if isinstance(value, Value) and isinstance(value.type, CellType) and value.type.dtype == result.type.dtype:
                        dividend = cell_ranges.before.get(id(original), {}).get(value.name)
                    if type(divisor) is int and divisor > 0 and dividend is not None:
                        linear = affine(op.operands[0], result.type, bounded=True)
                        if linear is not None and linear[0] % divisor == 0 and all(c % divisor == 0 for c in linear[1].values()):
                            if op.opcode == "scalar.mod":
                                known = 0
                            else:
                                form = linear[0] // divisor, {v: c // divisor for v, c in linear[1].items()}
                        elif (dividend[0] >= 0 and rounding(op) == "floor"
                              and (op.opcode == "scalar.div" or divisor & (divisor - 1))):
                            op = rw.rewritten(op, "nonnegative operands need no floor correction",
                                              attrs={**op.attrs, "rounding": Ident("trunc")})
                            captured = cell_ranges.snapshots.get(result.name)
                            if captured is not None:
                                facts[result.name] = captured
                            ctx.explain.note(f"%{result.name}: dividend is nonnegative at this read", op=op.id,
                                             kind="nonnegative-divmod")
                if known is None:
                    if form is None and op.opcode in {"scalar.add", "scalar.sub", "scalar.mul", "scalar.neg"}:
                        form = affine(result, result.type, bounded=True)
                        if not cancels_to_one_term(form, result):
                            # The bounded form needs a range for every intermediate, because the value it
                            # rebuilds must not have wrapped on the way. A term that CANCELS is exact
                            # whether or not it wrapped, so the unbounded form decides that one case -
                            # which is how `(a + v) - a` survives `v` having only a one-sided bound.
                            exact = affine(result, result.type)
                            if cancels_to_one_term(exact, result):
                                form = exact
                    op = rebuild(op, form, output, common)
                    if op is None:
                        continue
                if known is not None:
                    if result.type.dtype.name == "b1":
                        known = bool(known)
                    constants[result.name] = known
                    if op.opcode != "scalar.const" or op.attrs.get("value") != known:
                        op = rw.rewritten(op, "fold exact integer expression", opcode="scalar.const", operands=(), attrs={"value": known})
                        ctx.explain.note(f"%{result.name} = {known}", op=op.id, kind="constant-fold")
                elif op.opcode == "scalar.mod" and rounding(op) == "floor":
                    divisor = _literal(op.operands[1])
                    if isinstance(divisor, int) and divisor > 0 and divisor & (divisor - 1) == 0:
                        op = rw.rewritten(op, "power-of-two floor remainder is the low-bit mask, including negative dividends",
                                          opcode="scalar.and", operands=(op.operands[0], Literal(divisor - 1)), attrs={})
                        ctx.explain.note(f"%{result.name}: remainder by {divisor} becomes mask {divisor - 1}", op=op.id, kind="remainder-mask")
                elif op.opcode == "scalar.select" and _literal(op.operands[0]) is not None:
                    value = op.operands[1 if _literal(op.operands[0]) else 2]
                    if isinstance(value, Value) and value.type == result.type:
                        op = rw.rewritten(op, "constant scalar selection", opcode="scalar.cast", operands=(value,), attrs={})
                if total(op) and result.name not in type_names:
                    signature = key(op)
                    previous = common.get(signature)
                    if previous is not None:
                        aliases[result.name] = previous.results[0]
                        ctx.explain.note(f"%{result.name} reuses %{previous.results[0].name}", op=op.id,
                                         ops=(previous.id, op.id), kind="scalar-cse")
                        continue
                    common[signature] = op
            elif (op.opcode not in PURE_VIEWS | {"scalar.const", "scalar.store", "scalar.cell", "scalar.set"}
                  and not op.opcode.startswith(("cf.", "vf.", "cube.", "dma.", "mem.", "sync."))):
                common.clear()  # only immutable scalar expressions cross unrelated effects
            for result in op.results:
                definitions[result.name] = op
            if op.opcode == "scalar.cell":
                value = initializer(op, current, resolve)
                if value is not None:
                    current[op.results[0].name] = value
            output.append(op)
        return Block(tuple(output)), common

    body, _ = walk(fn.body)

    def typed(obj):
        if isinstance(obj, DimValue):
            # A dimension names a scalar in scope. An alias target is an operand of the value it
            # replaces, so it is in scope wherever that value was, and the dimension follows it: this
            # is what lets a value a type mentions be aliased away at all, the same way a proven
            # constant already leaves the type as an integer.
            name, seen = obj.name, set()
            while name in aliases and isinstance(aliases[name], Value) and name not in seen:
                seen.add(name)
                name = aliases[name].name
            value = constants.get(name)
            if type(value) is int:
                return value
            return obj if name == obj.name else DimValue(name)
        if isinstance(obj, (Value, Type, Product)):
            changes = {field.name: typed(getattr(obj, field.name)) for field in fields(obj)}
            return replace(obj, **changes)
        if isinstance(obj, Mapping):
            return {k: typed(v) for k, v in obj.items()}
        if isinstance(obj, tuple):
            return tuple(typed(v) for v in obj)
        if isinstance(obj, list):
            return [typed(v) for v in obj]
        return obj

    def specialize(block):
        output = []
        for op in block.ops:
            changes = {"operands": typed(op.operands), "results": typed(op.results), "attrs": typed(op.attrs),
                       "regions": tuple(specialize(r) for r in op.regions)}
            output.append(rw.rewritten(op, "specialize proven constant dimensions", **changes)
                          if any(getattr(op, key) != value for key, value in changes.items()) else op)
        return Block(tuple(output))

    body = specialize(body)
    # Only remove scalar operations; type-only references are real uses too.
    while True:
        used = set(_names(fn.attrs)) | set(_names(tuple(p.type for p in fn.params)))
        for op in body.walk():
            used.update(_names((op.operands, op.attrs, tuple(v.type for v in op.results))))

        def prune(block, used=used):
            output = []
            for op in block.ops:
                if (total(op) and op.results and isinstance(op.results[0].type, ScalarType)
                        and op.results[0].type.dtype.is_integer and all(_integer(x) for x in op.operands)
                        and not any(v.name in used for v in op.results)):
                    ctx.explain.note(f"remove unused scalar #{op.id}", op=op.id, kind="scalar-dce")
                    continue
                output.append(replace(op, regions=tuple(prune(r) for r in op.regions)) if op.regions else op)
            return Block(tuple(output))

        cleaned = prune(body)
        if cleaned == body:
            break
        body = cleaned
    return replace(fn, body=body, params=tuple(typed(p) for p in fn.params))


def run(module: Module, ctx: PassContext) -> Module:
    enabled = ctx.option("scalar_simplify", module.attrs.get("scalar_simplify", True))
    if type(enabled) is not bool:
        raise ValueError("scalar_simplify must be a bool")
    if not enabled:
        return Module(module.name, {**module.attrs, "scalar_simplify": False}, module.functions)
    rw = Rewriter(module, "scalar_simplify")
    functions = []
    for fn in module.functions:
        while True:
            simplified = simplify_function(fn, rw, ctx)
            if simplified == fn:
                break
            fn = simplified
        functions.append(fn)
    return Module(module.name, {**module.attrs, "scalar_simplify": True, "next_id": rw._next_id}, tuple(functions))


PASS_DEF = Pass("scalar_simplify", run, accepts="lowered/1", produces="lowered/1",
                doc="fold integer constants, share pure scalar expressions and retain snapshot semantics")
