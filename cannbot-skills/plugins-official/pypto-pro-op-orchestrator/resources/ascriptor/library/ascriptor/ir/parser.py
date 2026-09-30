# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Parser for the ``.ascrip`` text form (RFC-0001 §9.1)."""

from __future__ import annotations

from typing import Any

from .core import Block, FuncRef, Function, Ident, Literal, Loc, Module, Op, Origin, Value
from .lexer import ParseError, TokenStream, tokenize
from .types import Type, parse_type_tokens


class Scope:
    def __init__(self, parent: Scope | None = None) -> None:
        self.parent = parent
        self.values: dict[str, Value] = {}

    def lookup(self, name: str) -> Value | None:
        s: Scope | None = self
        while s is not None:
            if name in s.values:
                return s.values[name]
            s = s.parent
        return None

    def define(self, v: Value) -> None:
        self.values[v.name] = v


class ModuleScope(Scope):
    """Resolve unqualified metadata references without importing callee-local names."""

    def __init__(self, functions: list[Function]) -> None:
        super().__init__()
        self.ambiguous: set[str] = set()
        for fn in functions:
            if fn.kind not in ("kernel", "func"):
                continue
            for value in (*fn.params, *(result for op in fn.walk() for result in op.results)):
                previous = self.values.get(value.name)
                if previous is not None and previous != value:
                    self.ambiguous.add(value.name)
                self.define(value)

    def lookup(self, name: str) -> Value | None:
        if name in self.ambiguous:
            raise ParseError(f"ambiguous module metadata value %{name}: declarations have different types")
        return super().lookup(name)


class Parser:
    def __init__(self, text: str) -> None:
        self.ts = TokenStream(tokenize(text))

    # -- module -----------------------------------------------------------------------------

    def parse_module(self) -> Module:
        ts = self.ts
        ts.expect("ident", "module")
        name = ts.expect("funcref").text[1:]
        attrs_at = ts.i if ts.at_punct("{") else None
        if attrs_at is not None:
            depth = 0
            while True:
                if ts.at("eof"):
                    raise ts.error("unexpected end of input inside module attributes")
                token = ts.advance()
                if token.kind == "punct":
                    depth += (token.text == "{") - (token.text == "}")
                if depth == 0:
                    break
        functions: list[Function] = []
        while not ts.at("eof"):
            functions.append(self.parse_function())
        end = ts.i
        attrs = {}
        if attrs_at is not None:
            ts.i = attrs_at
            attrs = self.parse_attrs(ModuleScope(functions))
            ts.i = end
        return Module(name, attrs, tuple(functions))

    def parse_function(self) -> Function:
        ts = self.ts
        kind_tok = ts.expect("ident")
        if kind_tok.text not in ("kernel", "vf", "simt", "func"):
            raise ParseError(f"{kind_tok.line}:{kind_tok.col}: expected a function kind, found {kind_tok.text!r}")
        name = ts.expect("funcref").text[1:]
        scope = Scope()
        params: list[Value] = []
        ts.expect_punct("(")
        if not ts.at_punct(")"):
            params.append(self._parse_param(scope))
            while ts.accept("punct", ","):
                params.append(self._parse_param(scope))
        ts.expect_punct(")")
        for p in params:
            scope.define(p)
        attrs: dict[str, Any] = {}
        if ts.at_punct("{") and self._brace_is_attrs(has_regions=True):
            attrs = self.parse_attrs(scope)
        body = self.parse_block(scope)
        return Function(kind_tok.text, name, tuple(params), attrs, body)

    def _parse_param(self, scope: Scope) -> Value:
        ts = self.ts
        name = ts.expect("value").text[1:]
        ts.expect_punct(":")
        t = parse_type_tokens(ts)
        v = Value(name, t)
        scope.define(v)
        return v

    # -- blocks and ops ---------------------------------------------------------------------

    def parse_block(self, parent: Scope) -> Block:
        ts = self.ts
        ts.expect_punct("{")
        scope = Scope(parent)
        ops: list[Op] = []
        while not ts.at_punct("}"):
            if ts.at("eof"):
                raise ts.error("unexpected end of input inside a block")
            ops.append(self.parse_op(scope))
        ts.expect_punct("}")
        return Block(tuple(ops))

    def parse_op(self, scope: Scope) -> Op:
        ts = self.ts
        result_names: list[str] = []
        if ts.at("value"):
            result_names.append(ts.advance().text[1:])
            while ts.accept("punct", ","):
                result_names.append(ts.expect("value").text[1:])
            ts.expect_punct("=")
        opcode_tok = ts.expect("ident")
        opcode = opcode_tok.text
        if "." not in opcode:
            raise ParseError(f"{opcode_tok.line}:{opcode_tok.col}: opcode {opcode!r} must be namespaced")
        ts.expect_punct("(")
        operands: list[Any] = []
        if not ts.at_punct(")"):
            operands.append(self.parse_operand(scope))
            while ts.accept("punct", ","):
                operands.append(self.parse_operand(scope))
        ts.expect_punct(")")
        # Result types and attributes may come in either order when hand-written; the printer
        # emits `: type` first.
        types: list[Type] = []
        attrs: dict[str, Any] = {}
        for _ in range(2):
            if ts.at_punct(":") and not types:
                ts.advance()
                if ts.accept("punct", "("):
                    types.append(parse_type_tokens(ts))
                    while ts.accept("punct", ","):
                        types.append(parse_type_tokens(ts))
                    ts.expect_punct(")")
                else:
                    types.append(parse_type_tokens(ts))
            elif ts.at_punct("{") and not attrs and self._brace_is_attrs(has_regions=True):
                attrs = self.parse_attrs(scope)
        if len(types) != len(result_names):
            raise ParseError(f"{opcode_tok.line}:{opcode_tok.col}: {len(result_names)} result(s) but {len(types)} type(s)")
        results = tuple(Value(n, t) for n, t in zip(result_names, types, strict=True))
        op_id = None
        if ts.at("opid"):
            op_id = int(ts.advance().text[1:])
        loc = self.parse_loc() if ts.at("ident", "loc") else None
        origin = self._pop_origin(attrs)
        # Results of an op with regions are visible only inside the regions (induction values);
        # results of a plain op are visible after it in the enclosing block.
        regions: list[Block] = []
        if ts.at_punct("{"):
            inner = Scope(scope)
            for r in results:
                inner.define(r)
            regions.append(self.parse_block(inner))
            while ts.at("ident") and ts.peek().kind == "punct" and ts.peek().text == "{":
                ts.advance()  # region name; the registry gives the canonical name
                regions.append(self.parse_block(inner))
        else:
            for r in results:
                scope.define(r)
        return Op(opcode, tuple(operands), results, attrs, op_id, loc, origin, tuple(regions))

    def _brace_is_attrs(self, has_regions: bool) -> bool:
        """At '{': decide between an attribute block and a region/body block by lookahead."""
        ts = self.ts
        nxt = ts.peek(1)
        if nxt.kind == "ident" and ts.peek(2).kind == "punct" and ts.peek(2).text == "=":
            return True
        if nxt.kind == "punct" and nxt.text == "}":
            after = ts.peek(2)
            return after.kind == "punct" and after.text == "{"  # `{} {` -> empty attrs then a block
        return False

    def parse_operand(self, scope: Scope) -> Any:
        ts = self.ts
        if ts.at("value"):
            tok = ts.advance()
            v = scope.lookup(tok.text[1:])
            if v is None:
                raise ParseError(f"{tok.line}:{tok.col}: use of undefined value {tok.text}")
            return v
        if ts.at("funcref"):
            return FuncRef(ts.advance().text[1:])
        if ts.at("int"):
            return Literal(int(ts.advance().text, 0))
        if ts.at("float"):
            return Literal(float(ts.advance().text))
        if ts.at("ident", "true") or ts.at("ident", "false"):
            return Literal(ts.advance().text == "true")
        if ts.at("ident", "complex"):
            ts.advance()
            ts.expect_punct("(")
            re = float(ts.advance().text)
            ts.expect_punct(",")
            im = float(ts.advance().text)
            ts.expect_punct(")")
            return Literal(complex(re, im))
        raise ts.error(f"expected an operand, found {ts.cur.text!r}")

    def parse_loc(self) -> Loc:
        ts = self.ts
        ts.expect("ident", "loc")
        ts.expect_punct("(")
        chain = [self._unquote(ts.expect("string").text)]
        while ts.accept("arrow"):
            chain.append(self._unquote(ts.expect("string").text))
        ts.expect_punct(")")
        return Loc(tuple(chain))

    # -- attributes -------------------------------------------------------------------------

    def parse_attrs(self, scope: Scope) -> dict[str, Any]:
        ts = self.ts
        ts.expect_punct("{")
        attrs: dict[str, Any] = {}
        while not ts.at_punct("}"):
            key = ts.expect("ident").text
            ts.expect_punct("=")
            attrs[key] = self.parse_attr_value(scope)
            if not ts.accept("punct", ","):
                break
        ts.expect_punct("}")
        return attrs

    def parse_attr_value(self, scope: Scope) -> Any:
        ts = self.ts
        if ts.at("int"):
            return int(ts.advance().text, 0)
        if ts.at("float"):
            return float(ts.advance().text)
        if ts.at("string"):
            return self._unquote(ts.advance().text)
        if ts.at("value"):
            tok = ts.advance()
            v = scope.lookup(tok.text[1:])
            if v is None:
                raise ParseError(f"{tok.line}:{tok.col}: use of undefined value {tok.text} in attribute")
            return v
        if ts.at("funcref"):
            return FuncRef(ts.advance().text[1:])
        if ts.at_punct("["):
            ts.advance()
            items: list[Any] = []
            if not ts.at_punct("]"):
                items.append(self.parse_attr_value(scope))
                while ts.accept("punct", ","):
                    items.append(self.parse_attr_value(scope))
            ts.expect_punct("]")
            return items
        if ts.at_punct("{"):
            return self.parse_attrs(scope)
        if ts.at("ident"):
            name = ts.advance().text
            if name == "true":
                return True
            if name == "false":
                return False
            return Ident(name)
        raise ts.error(f"expected an attribute value, found {ts.cur.text!r}")

    @staticmethod
    def _unquote(s: str) -> str:
        body = s[1:-1]
        return body.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")

    @staticmethod
    def _pop_origin(attrs: dict[str, Any]) -> tuple[Origin, ...]:
        raw = attrs.pop("origin", None)
        if raw is None:
            return ()
        out = []
        for d in raw:
            kind = d.get("kind")
            out.append(Origin(str(d.get("pass")), kind.name if isinstance(kind, Ident) else str(kind),
                              tuple(int(x) for x in d.get("from", [])), d.get("note")))
        return tuple(out)


def parse_module(text: str) -> Module:
    p = Parser(text)
    m = p.parse_module()
    return m


__all__ = ["parse_module", "Parser", "ParseError"]
