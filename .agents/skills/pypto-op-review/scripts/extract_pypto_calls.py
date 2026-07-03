#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
"""
List PyPTO API call sites in a Python file (line number + call expression).

Used for line-by-line / op-by-op review when debugging PyPTO-specific errors.
Resolves: import pypto; pypto.matmul(...), pypto.frontend.jit(...), import pypto as p; p.view(...)
Also records: from pypto import matmul; matmul(...)  (tagged as short-import).

Usage:
  python3 extract_pypto_calls.py <path/to/kernel.py>
  python3 extract_pypto_calls.py <path/to/kernel.py> --json
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Set


@dataclass
class CallSite:
    line: int
    col: int
    call: str
    kind: str  # "attr" | "short_import"


def _attr_chain(func: ast.AST) -> Optional[List[str]]:
    if isinstance(func, ast.Name):
        return [func.id]
    if isinstance(func, ast.Attribute):
        inner = _attr_chain(func.value)
        if inner is None:
            return None
        return inner + [func.attr]
    return None


class _Finder(ast.NodeVisitor):
    def __init__(self) -> None:
        self.pypto_roots: Set[str] = set()
        self.short_from_pypto: Set[str] = set()
        self.sites: List[CallSite] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "pypto":
                self.pypto_roots.add(alias.asname or "pypto")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "pypto" and node.names:
            for alias in node.names:
                if alias.name == "*":
                    continue
                self.short_from_pypto.add(alias.asname or alias.name)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        chain = _attr_chain(func)
        if chain and chain[0] in self.pypto_roots:
            call_str = ".".join(chain)
            self.sites.append(
                CallSite(
                    line=node.lineno,
                    col=node.col_offset,
                    call=f"{call_str}(...)",
                    kind="attr",
                )
            )
        elif isinstance(func, ast.Name) and func.id in self.short_from_pypto:
            self.sites.append(
                CallSite(
                    line=node.lineno,
                    col=node.col_offset,
                    call=f"{func.id}(...)",
                    kind="short_import",
                )
            )
        self.generic_visit(node)


def extract_pypto_calls(source: str, path: str) -> List[CallSite]:
    tree = ast.parse(source, filename=path)
    f = _Finder()
    f.visit(tree)
    return sorted(f.sites, key=lambda s: (s.line, s.col))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Extract pypto call sites for op-by-op debugging.")
    ap.add_argument("path", type=Path, help="Python source file")
    ap.add_argument("--json", action="store_true", help="JSON lines output")
    args = ap.parse_args()

    p = args.path
    if not p.is_file():
        logging.error("Not a file: %s", p)
        sys.exit(1)

    src = p.read_text(encoding="utf-8", errors="replace")
    try:
        sites = extract_pypto_calls(src, str(p))
    except SyntaxError as e:
        logging.error("SyntaxError: %s", e)
        sys.exit(2)

    if args.json:
        logging.info(json.dumps([asdict(s) for s in sites], indent=2))
        return

    logging.info("# %s — %d pypto-related call site(s)\n", p, len(sites))
    for i, s in enumerate(sites, 1):
        tag = f"[{s.kind}]"
        logging.info("%4d  L%5d  %-14s  %s", i, s.line, tag, s.call)


if __name__ == "__main__":
    main()
