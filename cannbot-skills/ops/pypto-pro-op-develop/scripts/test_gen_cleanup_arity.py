#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Arity tests for gen_cleanup._arity_blockers.

The check used to pattern-match one spelling, `(x,) = fn(...)`, and treat every other
tuple-returning call as "not unpacked". That refused the ordinary multi-output form
`a, b = fn(...)`, so no legitimate multi-output L1/fusion delivery could pass cleanup --
and nothing tested it. These cases pin 0/1/2/N returns against 0/1/2/N unpack widths.

Run::

    python3 test_gen_cleanup_arity.py
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import tempfile
from pathlib import Path

_LOGGER = logging.getLogger("test_gen_cleanup_arity")

_HERE = Path(__file__).resolve().parent


def _load_gen_cleanup():
    spec = importlib.util.spec_from_file_location("gen_cleanup", _HERE / "gen_cleanup.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


GC = _load_gen_cleanup()

OP = "demo"
FN = f"{OP}_golden_cpu"

# Golden bodies keyed by how many values they return.
GOLDENS = {
    0: f"def {FN}(x):\n    print(x)\n",                       # no return value at all
    1: f"def {FN}(x):\n    return x + 1\n",
    2: f"def {FN}(x):\n    return x + 1, x + 2\n",
    3: f"def {FN}(x):\n    return x + 1, x + 2, x + 3\n",
}

# Return shapes the first rewrite got wrong: a list literal read as one value, and
# returns whose arity is not statically knowable read as one value rather than unknown.
LIST_2 = f"def {FN}(x):\n    return [x + 1, x + 2]\n"
DELEGATES = f"def {FN}(x):\n    return _ref(x)\n"
CALL_RETURN = f"def {FN}(x):\n    return torch.max(x, dim=-1)\n"
NESTED_HELPER = (
    f"def {FN}(x):\n"
    f"    def h(v):\n"
    f"        return v * 2\n"
    f"    return h(x), h(x + 1)\n"
)


def _blockers(golden_src: str, delivery: str) -> list[str]:
    with tempfile.TemporaryDirectory() as raw:
        out_dir = Path(raw)
        (out_dir / f"{FN}.py").write_text(golden_src, encoding="utf-8")
        arity_blockers = getattr(GC, "_arity_blockers")
        return arity_blockers(OP, delivery, out_dir)


CASES: list[tuple[str, int, str, bool]] = [
    # (label, golden return arity, delivery snippet, expect a blocker)

    # --- the regression this fix is for: legitimate multi-output deliveries ---
    ("2 returns unpacked into 2", 2, f"a, b = {FN}(x)\n", False),
    ("2 returns unpacked into 2, parenthesised", 2, f"(a, b) = {FN}(x)\n", False),
    ("3 returns unpacked into 3", 3, f"a, b, c = {FN}(x)\n", False),
    # A multi-output golden bound to one bare name is NOT fine -- see the
    # "never indexed" cases below. Two earlier cases asserted the opposite and
    # encoded the assumption that a single name "is always legal, whatever the
    # arity"; that dropped a real refusal, so they are gone rather than relaxed.

    # --- the original known defect must still be caught ---
    ("1 return unpacked as 1-tuple", 1, f"(a,) = {FN}(x)\n", True),
    ("1 return unpacked into 2", 1, f"a, b = {FN}(x)\n", True),

    # --- genuine mismatches in the other direction ---
    ("2 returns unpacked into 3", 2, f"a, b, c = {FN}(x)\n", True),
    ("3 returns unpacked into 2", 3, f"a, b = {FN}(x)\n", True),

    # --- legitimate single-output ---
    ("1 return bound to one name", 1, f"expected = {FN}(x)\n", False),

    # --- starred targets are elastic, never a mismatch ---
    ("3 returns with starred target", 3, f"a, *rest = {FN}(x)\n", False),

    # --- no call site at all ---
    ("2 returns, golden never called", 2, "expected = torch.zeros(4)\n", False),

    # --- a multi-output golden bound to one bare name must still be refused ---
    ("2 returns bound to one name, never indexed", 2, f"g = {FN}(x)\n_assert_precision(out, g)\n", True),
    ("3 returns bound to one name, never indexed", 3, f"g = {FN}(x)\n_assert_precision(out, g)\n", True),
]

# Cases keyed on a raw golden body rather than an arity.
RAW_CASES: list[tuple[str, str, str, bool]] = [
    # list literal unpacks like a tuple -- must NOT be refused
    ("list-literal return unpacked into 2", LIST_2, f"a, b = {FN}(x)\n", False),
    ("list-literal return unpacked into 3", LIST_2, f"a, b, c = {FN}(x)\n", True),

    # arity not statically knowable -- stay silent rather than guess either way
    ("delegated return unpacked into 2", DELEGATES, f"a, b = {FN}(x)\n", False),
    ("delegated return bound to one name", DELEGATES, f"g = {FN}(x)\n", False),
    ("torch call return unpacked into 2", CALL_RETURN, f"a, b = {FN}(x)\n", False),

    # a nested helper's returns must not be mistaken for the golden's own
    ("nested helper, 2-tuple return, 1-wide unpack", NESTED_HELPER, f"(a,) = {FN}(x)\n", True),
    ("nested helper, 2-tuple return, 2-wide unpack", NESTED_HELPER, f"a, b = {FN}(x)\n", False),

    # a single-name binding the delivery clearly treats as a tuple is fine
    ("2 returns bound to one name then indexed", GOLDENS[2], f"g = {FN}(x)\n_assert_precision(out, g[0])\n", False),
    ("2 returns bound to one name then unpacked", GOLDENS[2], f"g = {FN}(x)\na, b = g\n", False),

    # single-value golden bound to one name stays legal
    ("1 return bound to one name, asserted", GOLDENS[1], f"g = {FN}(x)\n_assert_precision(out, g)\n", False),
]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    failures: list[str] = []

    for label, arity, delivery, want_blocker in CASES:
        got = _blockers(GOLDENS[arity], delivery)
        ok = bool(got) == want_blocker
        if not ok:
            failures.append(
                f"{label}: expected {'a blocker' if want_blocker else 'no blocker'}, got {got}"
            )
        (_LOGGER.info if ok else _LOGGER.error)("%s %s", "PASS" if ok else "FAIL", label)

    for label, golden_src, delivery, want_blocker in RAW_CASES:
        got = _blockers(golden_src, delivery)
        ok = bool(got) == want_blocker
        if not ok:
            failures.append(
                f"{label}: expected {'a blocker' if want_blocker else 'no blocker'}, got {got}"
            )
        (_LOGGER.info if ok else _LOGGER.error)("%s %s", "PASS" if ok else "FAIL", label)

    # A golden with no return value is indeterminate; stay silent rather than guess.
    got = _blockers(GOLDENS[0], f"a, b = {FN}(x)\n")
    ok = got == []
    if not ok:
        failures.append(f"golden with no return: expected silence, got {got}")
    (_LOGGER.info if ok else _LOGGER.error)(
        "%s golden with no return stays silent", "PASS" if ok else "FAIL")

    # Inconsistent return arity within the golden is also indeterminate.
    inconsistent = f"def {FN}(x):\n    if x:\n        return x\n    return x, x\n"
    got = _blockers(inconsistent, f"a, b = {FN}(x)\n")
    ok = got == []
    if not ok:
        failures.append(f"inconsistent returns: expected silence, got {got}")
    (_LOGGER.info if ok else _LOGGER.error)(
        "%s inconsistent golden returns stay silent", "PASS" if ok else "FAIL")

    # An unreadable delivery must not raise.
    got = _blockers(GOLDENS[2], "a, b = (\n")
    ok = got == []
    if not ok:
        failures.append(f"unparseable delivery: expected silence, got {got}")
    (_LOGGER.info if ok else _LOGGER.error)(
        "%s unparseable delivery stays silent", "PASS" if ok else "FAIL")

    for problem in failures:
        _LOGGER.warning("\n%s", problem)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
