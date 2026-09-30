# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tokenizer for the ``.ascrip`` text form and for type spellings (RFC-0001 §9.1).

One lexer serves the module parser, the type parser and the pattern parser so that a type
spelled inside an op line and a type spelled on its own tokenize identically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Order matters: longer punctuation first, floats before ints.
_TOKEN_RE = re.compile(
    r"""
    (?P<ws>[ \t\r]+)
  | (?P<nl>\n)
  | (?P<comment>;[^\n]*)
  | (?P<string>"(?:[^"\\]|\\.)*")
  | (?P<value>%[A-Za-z_][A-Za-z0-9_.]*)
  | (?P<funcref>@[A-Za-z_][A-Za-z0-9_.]*)
  | (?P<opid>\#[0-9]+)
  | (?P<float>-?(?:[0-9]+\.[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?|-?[0-9]+[eE][-+]?[0-9]+|-?inf|nan)
  | (?P<int>-?0[xX][0-9a-fA-F]+|-?[0-9]+)
  | (?P<ident>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)
  | (?P<arrow><-)
  | (?P<punct>[()\[\]{}<>,=:*?])
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class Token:
    kind: str  # ident | value | funcref | opid | int | float | string | punct | arrow | eof
    text: str
    line: int
    col: int

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Token({self.kind}, {self.text!r}, {self.line}:{self.col})"


class LexError(ValueError):
    pass


def tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    line, line_start, pos = 1, 0, 0
    n = len(text)
    while pos < n:
        m = _TOKEN_RE.match(text, pos)
        if m is None:
            raise LexError(f"{line}:{pos - line_start + 1}: unexpected character {text[pos]!r}")
        kind = m.lastgroup
        if kind is None:
            raise LexError(f"{line}:{pos - line_start + 1}: unrecognized token")
        s = m.group(kind)
        if kind == "nl":
            line += 1
            line_start = m.end()
        elif kind not in ("ws", "comment"):
            tokens.append(Token(kind, s, line, pos - line_start + 1))
        pos = m.end()
    tokens.append(Token("eof", "", line, pos - line_start + 1))
    return tokens


class TokenStream:
    """Cursor over a token list with the few lookahead helpers recursive descent needs."""

    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.i = 0

    @property
    def cur(self) -> Token:
        return self.tokens[self.i]

    def peek(self, k: int = 1) -> Token:
        j = min(self.i + k, len(self.tokens) - 1)
        return self.tokens[j]

    def at(self, kind: str, text: str | None = None) -> bool:
        t = self.cur
        return t.kind == kind and (text is None or t.text == text)

    def at_punct(self, text: str) -> bool:
        return self.at("punct", text)

    def advance(self) -> Token:
        t = self.cur
        if t.kind != "eof":
            self.i += 1
        return t

    def accept(self, kind: str, text: str | None = None) -> Token | None:
        if self.at(kind, text):
            return self.advance()
        return None

    def expect(self, kind: str, text: str | None = None) -> Token:
        if not self.at(kind, text):
            want = f"{kind} {text!r}" if text is not None else kind
            raise self.error(f"expected {want}, found {self.cur.kind} {self.cur.text!r}")
        return self.advance()

    def expect_punct(self, text: str) -> Token:
        return self.expect("punct", text)

    def error(self, message: str) -> ParseError:
        t = self.cur
        return ParseError(f"{t.line}:{t.col}: {message}")


class ParseError(ValueError):
    pass
