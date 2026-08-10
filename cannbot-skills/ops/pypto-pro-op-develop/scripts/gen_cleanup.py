#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
"""Pro L1 cleanup — from the last staged file to delivery test_{op}.py.

Produces test_{op}.py from the final staged file (test_{op}_module<suffix_N>.py):
  1. Copy the file content
  2. Rename wrapper function: {op}_wrapper_module<suffix_N> → {op}_wrapper
  3. Change test comparison source: {op}_golden_stage<suffix_N> → {op}_golden_cpu

The staged file chain (module1.py / module12.py / module123.py / ...) is preserved
as audit artifacts — this script does NOT modify or delete them.

Usage::

    python gen_cleanup.py --op softmax \\
        --final-impl custom/softmax/modules/test_softmax_module123.py \\
        --final-suffix 123 \\
        --spec custom/softmax/SPEC.md \\
        --golden custom/softmax/softmax_golden_cpu.py \\
        --out-dir custom/softmax

    python gen_cleanup.py --self-test
"""

from __future__ import annotations

import argparse
import ast
import logging
import sys
from pathlib import Path

_LOGGER = logging.getLogger("gen_cleanup")


def integrate_test(
    op: str,
    final_impl_src: str,
    final_suffix: str,
) -> str:
    """Produce test_{op}.py from the final staged file.

    - Rename wrapper: {op}_wrapper_module{suffix_N} → {op}_wrapper
    - Change golden import: {op}_golden_stage{suffix_N} → {op}_golden_cpu
    - Change golden function calls: {op}_golden_stage{suffix_N}(...) → {op}_golden_cpu(...)
    """
    src = final_impl_src

    # Rename wrapper function definition and calls
    old_wrapper = f"{op}_wrapper_module{final_suffix}"
    new_wrapper = f"{op}_wrapper"
    src = src.replace(old_wrapper, new_wrapper)

    # Change golden import: from modules.softmax_golden_stage123 import ... → from softmax_golden_cpu import ...
    # Pattern: from modules.{op}_golden_stage{suffix} import {op}_golden_stage{suffix}
    old_golden_module = f"modules.{op}_golden_stage{final_suffix}"
    new_golden_module = f"{op}_golden_cpu"
    src = src.replace(old_golden_module, new_golden_module)

    # Also handle: from {op}_golden_stage{suffix} import ... (without modules. prefix)
    old_golden_module_alt = f"{op}_golden_stage{final_suffix}"
    src = src.replace(
        f"from {old_golden_module_alt} import",
        f"from {new_golden_module} import",
    )

    # Change golden function name in calls and imports
    old_golden_fn = f"{op}_golden_stage{final_suffix}"
    new_golden_fn = f"{op}_golden_cpu"
    src = src.replace(old_golden_fn, new_golden_fn)

    # Fix the op-dir walk-up for the file's new depth.
    #
    # The staged file lives at custom/<op>/modules/, so it reaches the operator
    # directory with `.parent.parent`. The delivery file lives one level higher,
    # at custom/<op>/, where that same expression yields custom/ -- a directory
    # holding no golden. This survives `python custom/<op>/test_<op>.py` only
    # because script mode puts custom/<op>/ on sys.path[0] anyway; import the
    # module instead and the golden import raises ModuleNotFoundError. Rewrite
    # the depth rather than leaning on that accident.
    # Both spellings of the two-level walk-up occur in the wild; rewriting only
    # one leaves the other silently pointing at custom/.
    src = src.replace(
        "Path(__file__).resolve().parent.parent",
        "Path(__file__).resolve().parent",
    )
    src = src.replace(
        "Path(__file__).resolve().parents[1]",
        "Path(__file__).resolve().parent",
    )

    header = (
        f"# Delivery kernel for {op} (consolidated from modules/test_{op}_module{final_suffix}.py).\n"
        f"# Exports {op}_wrapper. Same kernel logic as the final staged file.\n"
        f"# Staged file chain preserved as audit artifacts.\n"
    )
    return header + src


def _self_test() -> int:
    op = "test_op"
    suffix = "12"
    staged_src = (
        f"from modules.test_op_golden_stage12 import test_op_golden_stage12\n"
        f"\n"
        f"def test_op_wrapper_module12(x):\n"
        f"    return _kernel(x)\n"
        f"\n"
        f"def test_case():\n"
        f"    out = test_op_wrapper_module12(x)\n"
        f"    golden = test_op_golden_stage12(x)\n"
        f"    _assert_precision(out, golden)\n"
    )
    result = integrate_test(op, staged_src, suffix)
    ok = (
        "test_op_wrapper(x)" in result  # wait, the call should be renamed too
        or "test_op_wrapper_module12" not in result
    ) and (
        "test_op_golden_stage12" not in result
    ) and (
        "test_op_golden_cpu" in result
    ) and (
        "from test_op_golden_cpu import" in result or "from modules.test_op_golden_cpu" in result
    )
    # Actually check more carefully
    ok_wrapper = "test_op_wrapper_module12" not in result and "def test_op_wrapper(" in result
    ok_golden = "test_op_golden_stage12" not in result and "test_op_golden_cpu" in result
    ok_import = "from test_op_golden_cpu import" in result
    passed = ok_wrapper and ok_golden and ok_import
    _LOGGER.info(
        "%s self-test: wrapper=%s golden=%s import=%s",
        "PASS" if passed else "FAIL",
        ok_wrapper, ok_golden, ok_import,
    )
    if not passed:
        _LOGGER.info("=== output ===\n%s", result)
    return 0 if passed else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Pro L1 cleanup: generate test_{op}.py from the final staged file"
    )
    ap.add_argument("--op", required=False, help="Operator name")
    ap.add_argument("--final-impl", type=Path, help="modules/test_{op}_module<suffix_N>.py")
    ap.add_argument("--final-suffix", help="cumulative suffix of the final module, e.g. 123")
    ap.add_argument("--spec", type=Path, help="Path to SPEC.md (reserved, unused in current impl)")
    ap.add_argument("--golden", type=Path, help="Path to {op}_golden_cpu.py (reserved, unused in current impl)")
    ap.add_argument("--out-dir", type=Path, default=Path("."), help="Output directory (custom/<op>)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    if args.self_test:
        return _self_test()
    if not (args.op and args.final_impl and args.final_suffix):
        ap.error("--op, --final-impl, --final-suffix required (or --self-test)")
    if not args.final_impl.exists():
        _LOGGER.error("final staged file not found: %s", args.final_impl)
        return 1
    src = args.final_impl.read_text(encoding="utf-8")
    result = integrate_test(args.op, src, args.final_suffix)

    # A rename cannot reconcile a return-arity difference between the two
    # goldens, and getting it wrong fails every precision test at once with a
    # ValueError that looks nothing like a cleanup problem. Refuse rather than
    # emit a file that cannot run.
    blockers = _arity_blockers(args.op, result, args.out_dir)
    if blockers:
        for line in blockers:
            _LOGGER.error("%s", line)
        _LOGGER.error(
            "refusing to write %s -- fix the staged unpack or the golden, then re-run",
            args.out_dir / f"test_{args.op}.py",
        )
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"test_{args.op}.py"
    out_path.write_text(result, encoding="utf-8")
    _LOGGER.info("wrote %s (from %s)", out_path, args.final_impl)
    return 0


def _own_returns(fn: ast.AST) -> list[ast.Return]:
    """Return statements belonging to `fn` itself, not to anything nested inside it.

    ``ast.walk`` descends into nested ``def``s, so a local helper's returns used to land in
    the same set as the golden's own: a golden with `def h(v): return v*2` and
    `return h(x), h(x+1)` looked self-inconsistent, the guard returned None, and it
    silently stopped checking the very case it exists to catch.
    """
    found: list[ast.Return] = []

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                continue           # a different scope; its returns are not the golden's
            if isinstance(child, ast.Return):
                found.append(child)
            visit(child)

    visit(fn)
    return found


def _return_shape(value: ast.AST) -> tuple[int, bool] | None:
    """Classify one return expression: (arity, is_multi), or None for "cannot tell".

    Three-way on purpose. Treating "not a tuple literal" as "one value" was wrong in both
    directions: `return [h, s]` unpacks into two targets but read as one, so an ordinary
    `a, b = fn(...)` delivery was falsely refused; and `return _ref(x)` or
    `return torch.max(x, dim=-1)` could be any arity, so claiming either answer is a guess.
    """
    if isinstance(value, (ast.Tuple, ast.List)):
        if any(isinstance(e, ast.Starred) for e in value.elts):
            return None            # elastic width
        return (len(value.elts), True)
    if isinstance(value, (ast.BinOp, ast.UnaryOp, ast.Compare, ast.Constant, ast.Subscript)):
        return (1, False)          # arithmetic, comparison, indexing: one value
    return None                    # Call / Name / Attribute / IfExp / comprehension: unknown


def _golden_return_shape(golden_path: Path, fn_name: str) -> tuple[int, bool] | None:
    """What fn_name returns, as (arity, returns_multiple_values).

    A bare value is (1, False); an N-tuple or N-element list is (N, True). The multi flag
    matters separately from the arity: `(x,) = fn(...)` has width 1 and so "matches" a bare
    return numerically, but Python still unpacks dim 0 of the returned tensor.

    None means "cannot tell" -- unreadable file, absent function, a return whose arity is
    not statically knowable, or returns that disagree with each other. Callers must stay
    silent on None rather than guess.
    """
    try:
        tree = ast.parse(golden_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn_name:
            returns = [r for r in _own_returns(node) if r.value is not None]
            if not returns:
                return None
            shapes = {_return_shape(r.value) for r in returns}
            if len(shapes) != 1:
                return None        # inconsistent across returns
            return shapes.pop()    # may itself be None: an unknowable return
    return None


def _name_is_treated_as_multi(tree: ast.AST, name: str) -> bool:
    """True if the delivery ever indexes `name` or unpacks it into several targets.

    Evidence that the author knows the bound value is a tuple. Without any such use, a
    name bound from a multi-output golden is fed straight into the precision comparison as
    if it were a tensor, which is the case the old check refused.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == name:
            return True
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name) and node.value.id == name:
            if any(isinstance(t, (ast.Tuple, ast.List)) for t in node.targets):
                return True
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Name) and node.iter.id == name:
            return True
    return False


def _delivery_unpacks(result: str, fn_name: str) -> list[tuple[int, bool, int]]:
    """Every `<target> = fn_name(...)` in the delivery, as (width, target_is_tuple, lineno).

    `target_is_tuple` separates `(a,) = fn(x)` from `a = fn(x)`: both bind one thing, but
    the first unpacks and the second does not, and only the first is wrong for a
    single-value golden.

    A single-name binding is omitted entirely when the delivery shows it knows the value
    may be a tuple (indexing it, unpacking it later, iterating it). Otherwise it is
    reported with width 1 so a multi-output golden bound to one bare name is still
    refused -- treating every single-name binding as "always legal" dropped that check:
    `g = fn(x)` then `_assert_precision(out, g)` compares a tensor against a tuple and
    fails every precision test, which cleanup used to catch before writing the file.
    """
    try:
        tree = ast.parse(result)
    except SyntaxError:
        return []

    def calls_golden(value: ast.AST) -> bool:
        if not isinstance(value, ast.Call):
            return False
        func = value.func
        if isinstance(func, ast.Name):
            return func.id == fn_name
        if isinstance(func, ast.Attribute):
            return func.attr == fn_name
        return False

    found: list[tuple[int, bool, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not calls_golden(node.value):
            continue
        for target in node.targets:
            if isinstance(target, (ast.Tuple, ast.List)):
                # A starred target makes the width elastic, so it can never mismatch.
                if any(isinstance(e, ast.Starred) for e in target.elts):
                    continue
                found.append((len(target.elts), True, node.lineno))
            elif isinstance(target, ast.Name) and not _name_is_treated_as_multi(tree, target.id):
                found.append((1, False, node.lineno))
    return found


def _arity_blockers(op: str, result: str, out_dir: Path) -> list[str]:
    """Report a mismatch between how the delivery unpacks the golden and what it returns.

    Compares real arities via the AST. The previous version pattern-matched the single
    spelling `(x,) = fn(...)` and treated every other tuple-returning call as "not
    unpacked", so the ordinary multi-output form `a, b = fn(...)` was refused and no
    legitimate multi-output delivery could get through cleanup.
    """
    fn = f"{op}_golden_cpu"
    shape = _golden_return_shape(out_dir / f"{fn}.py", fn)
    if shape is None:
        return []          # cannot read it -- say nothing rather than guess
    returns, returns_tuple = shape

    blockers: list[str] = []
    for width, target_is_tuple, lineno in _delivery_unpacks(result, fn):
        if not returns_tuple and not target_is_tuple:
            continue           # one value bound to one name
        if returns_tuple and target_is_tuple and width == returns:
            continue           # multi-value return, matching unpack width
        if returns_tuple and not target_is_tuple:
            blockers += [
                f"{fn} returns {returns} values, but the delivery binds them to a single "
                f"name at line {lineno} without ever indexing or unpacking it.",
                "  The bound name goes straight into the precision comparison, which then",
                "  compares a tensor against a tuple and fails every case. Unpack it, or",
                "  index the element being compared.",
            ]
            continue
        if not returns_tuple:
            blockers += [
                f"{fn} returns a bare value, but the delivery unpacks it into "
                f"{width} target(s) at line {lineno}.",
                "  This unpacks dim 0 of the tensor and fails every precision test with",
                "  'ValueError: too many values to unpack'. The staged golden returned a",
                "  tuple; the cpu golden does not.",
            ]
        else:
            blockers += [
                f"{fn} returns {returns} values, but the delivery unpacks it into "
                f"{width} target(s) at line {lineno}.",
                "  Match the unpack width to the golden's return arity.",
            ]
    return blockers


if __name__ == "__main__":
    sys.exit(main())
