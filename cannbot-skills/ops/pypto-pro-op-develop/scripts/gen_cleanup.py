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
import logging
import re
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
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"test_{args.op}.py"
    out_path.write_text(result, encoding="utf-8")
    _LOGGER.info("wrote %s (from %s)", out_path, args.final_impl)
    return 0


if __name__ == "__main__":
    sys.exit(main())
