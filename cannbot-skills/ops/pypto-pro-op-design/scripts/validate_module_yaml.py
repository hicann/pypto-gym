#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
"""Validate custom/<op>/module_interfaces.yaml wiring.

Codifies the validity rules for the Pro Module contract:

  0. The file is a non-empty mapping declaring op/module_count/modules, with at
     least one module. Without this every later rule is satisfied vacuously by an
     empty file -- which is how a generator error message redirected into
     module_interfaces.yaml once passed the mandatory Stage-3 gate.

  1. Every inputs[*].source == "primary" name exists in primary_inputs.
  2. Every inputs[*].source == "module_j" has j < current module id, and the
     referenced name exists in module_j.outputs (no forward/self reference).
  3. Every final_outputs[*].source == "module_j" has j <= module_count, and the
     referenced name exists in module_j.outputs.

  `module_j` is the canonical spelling (see SKILL.md and gen_module_interfaces.py).
  The pre-rename `phase_j` spelling is also accepted for backward compatibility.
  4. No no-op module (every module has at least 1 output).
  5. Shape expressions parse using only + - * // and name/int tokens.
  6. dtype strings are from the allowed vocabulary.
  7. module_count equals len(modules).

Usage::

    python validate_module_yaml.py custom/<op>/module_interfaces.yaml [--json]
    python validate_module_yaml.py --self-test

Exit code: 0 if valid, 1 if any rule fails (or on load/parse error).
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import logging
import re
import sys
from pathlib import Path

_LOGGER = logging.getLogger("validate_module_yaml")

DTYPE_VOCAB = {
    "float32", "float16", "bfloat16", "int32", "int64", "bool", "int",
    # Pro integer/float subtypes (pl.DT_UINT8/INT8/UINT16/INT16/UINT32/UINT64/FP8*):
    "uint8", "int8", "uint16", "int16", "uint32", "uint64", "float8",
}
# Canonical spelling is `module_<j>` -- that is what SKILL.md documents and what
# gen_module_interfaces.py emits. `phase_<j>` is the pre-rename spelling and is still
# accepted so that artifacts written before the rename keep validating; accepting both
# cannot invalidate anything already on disk, which is exactly what the previous
# `^phase_(\d+)$`-only regex did to every artifact produced from the current generator.
_MODULE_CANONICAL_PREFIX = "module"
_MODULE_RE = re.compile(r"^(?:module|phase)_(\d+)$")
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.USub, ast.UAdd, ast.Constant,
)


def _shape_dims(shape) -> list[str]:
    if isinstance(shape, (list, tuple)):
        return [str(d) for d in shape]
    s = str(shape).strip().strip("[]")
    return [d.strip() for d in s.split(",") if d.strip()]


def _dim_parses(expr: str) -> bool:
    if expr.lstrip("-").isdigit():
        return True
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return False
        if isinstance(node, ast.Constant) and not isinstance(node.value, int):
            return False
    return True


def _validate_shape_and_dtype(entry, location, errors):
    for dimension in _shape_dims(entry.get("shape", "")):
        if not _dim_parses(dimension):
            errors.append(f"rule5: {location} shape dim '{dimension}' not parseable")
    dtype = entry.get("dtype")
    if dtype is not None and dtype not in DTYPE_VOCAB:
        errors.append(f"rule6: {location} dtype '{dtype}' not in {sorted(DTYPE_VOCAB)}")


def _validate_primary_inputs(primary_inputs, errors):
    for pi in primary_inputs:
        if not isinstance(pi, dict):
            continue
        name = pi.get("name")
        _validate_shape_and_dtype(pi, f"primary input '{name}'", errors)


def _index_outputs(modules, errors):
    outputs_by_id: dict[int, set] = {}
    for module in modules:
        phase_id = module.get("id")
        outputs = module.get("outputs", []) or []
        if not outputs:
            errors.append(f"rule4: module {phase_id} has no outputs (no-op module)")
        names = set()
        for out in outputs:
            name = out.get("name")
            names.add(name)
            _validate_shape_and_dtype(out, f"module {phase_id} output '{name}'", errors)
        outputs_by_id[phase_id] = names
    return outputs_by_id


def _validate_module_input(phase_id, phase_input, primary, outputs_by_id, errors):
    source = phase_input.get("source")
    name = phase_input.get("name")
    if source == "primary":
        if name not in primary:
            errors.append(
                f"rule1: module {phase_id} input '{name}' source=primary not in primary_inputs"
            )
        return
    source_match = _MODULE_RE.match(str(source or ""))
    if not source_match:
        errors.append(
            f"rule2: module {phase_id} input '{name}' has invalid source '{source}' "
            f"(expected 'primary' or '{_MODULE_CANONICAL_PREFIX}_<j>')"
        )
        return
    source_id = int(source_match.group(1))
    if not (isinstance(phase_id, int) and source_id < phase_id):
        errors.append(
            f"rule2: module {phase_id} input '{name}' references "
            f"module_{source_id} (must be < {phase_id})"
        )
    elif name not in outputs_by_id.get(source_id, set()):
        errors.append(
            f"rule2: module {phase_id} input '{name}' not produced by module_{source_id}"
        )


def _validate_module_inputs(modules, primary, outputs_by_id, errors):
    for module in modules:
        phase_id = module.get("id")
        for pi in module.get("inputs", []) or []:
            _validate_module_input(phase_id, pi, primary, outputs_by_id, errors)


def _validate_final_outputs(final_outputs, module_count, outputs_by_id, errors):
    for fo in final_outputs:
        name = fo.get("name")
        source = fo.get("source")
        source_match = _MODULE_RE.match(str(source or ""))
        if not source_match:
            errors.append(
                f"rule3: final_output '{name}' has invalid source '{source}' "
                f"(expected '{_MODULE_CANONICAL_PREFIX}_<j>')"
            )
            continue
        source_id = int(source_match.group(1))
        if source_id > module_count:
            errors.append(
                f"rule3: final_output '{name}' references module_{source_id} > N={module_count}"
            )
        elif name not in outputs_by_id.get(source_id, set()):
            errors.append(f"rule3: final_output '{name}' not produced by module_{source_id}")


def validate(spec: dict) -> list[str]:
    errors: list[str] = []
    # rule0 first: every rule below is satisfied vacuously by a spec that declares nothing.
    # An empty or truncated file yields modules == [] and module_count == 0, so rule7's
    # 0 == len([]) held and the whole check reported PASS. That is how a generator error
    # message redirected into module_interfaces.yaml passed the mandatory Stage-3 gate and
    # the architect advanced to Stage 4 with an error string as the module contract.
    if not isinstance(spec, dict) or not spec:
        return ["rule0: module_interfaces.yaml is empty or is not a mapping"]
    missing = [key for key in ("op", "module_count", "modules") if key not in spec]
    if missing:
        errors.append(f"rule0: missing required top-level key(s): {', '.join(missing)}")
    if not (spec.get("modules") or []):
        errors.append("rule0: modules is empty; a module contract needs at least one module")
    if errors:
        return errors    # nothing below can say anything useful about a spec this broken

    primary_inputs = spec.get("primary_inputs", [])
    primary = {item.get("name") for item in primary_inputs if isinstance(item, dict)}
    _validate_primary_inputs(primary_inputs, errors)
    modules = spec.get("modules", []) or []
    outputs_by_id = _index_outputs(modules, errors)
    _validate_module_inputs(modules, primary, outputs_by_id, errors)
    module_count = spec.get("module_count", 0)
    _validate_final_outputs(
        spec.get("final_outputs", []) or [], module_count, outputs_by_id, errors
    )
    # Check that module_count matches the actual number of modules
    if module_count != len(modules):
        errors.append(
            f"rule7: module_count={module_count} != len(modules)={len(modules)}"
        )
    return errors


def _load(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyYAML not available; cannot read YAML file.") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


_VALID = {
    "schema_version": 1,
    "op": "test_op",
    "module_count": 2,
    "has_cross_core": False,
    "is_fusion": True,
    "primary_inputs": [{"name": "x", "shape": "[M, N]", "dtype": "float16"}],
    "modules": [
        {"id": 1, "name": "p1", "section": "vector",
         "golden_steps": ["step1"],
         "inputs": [{"name": "x", "source": "primary"}],
         "outputs": [{"name": "h1", "shape": "[M, N]", "dtype": "float16"}],
         "golden_stage_fn": "test_op_golden_stage1"},
        {"id": 2, "name": "p2", "section": "vector",
         "golden_steps": ["step2"],
         # Canonical spelling, matching SKILL.md and gen_module_interfaces.py. The
         # fixture had been flipped to `phase_1`/`phase_2` to satisfy a validator that
         # accepted only the pre-rename spelling -- which hid the fact that nothing the
         # generator produces could pass. _MODULE_RE now accepts both, so the fixture
         # tracks the documented contract again and `phase_<j>` keeps working; see the
         # legacy_phase_spelling and rule2_unknown_prefix self-test cases below.
         "inputs": [{"name": "h1", "source": "module_1"}],
         "outputs": [{"name": "y", "shape": "[M, N]", "dtype": "float16"}],
         "golden_stage_fn": "test_op_golden_stage12"},
    ],
    "final_outputs": [{"name": "y", "source": "module_2"}],
}


def _self_test() -> int:
    cases: list[tuple[str, dict, str]] = []
    cases.append(("valid", _VALID, ""))

    c1 = copy.deepcopy(_VALID)
    c1["modules"][0]["inputs"][0]["name"] = "zzz"
    cases.append(("rule1_unknown_primary", c1, "rule1"))

    # These two must use a *valid* module reference, or they trip the invalid-source
    # branch and pass without ever exercising the forward-ref and out-of-range
    # branches they exist to cover.
    c2 = copy.deepcopy(_VALID)
    c2["modules"][0]["inputs"][0] = {"name": "y", "source": "module_2"}
    cases.append(("rule2_forward_ref", c2, "rule2"))

    c3 = copy.deepcopy(_VALID)
    c3["final_outputs"][0]["source"] = "module_9"
    cases.append(("rule3_out_of_range", c3, "rule3"))

    # Backward compatibility: the pre-rename `phase_<j>` spelling must still validate,
    # so artifacts written before the rename keep passing. This replaces a case that
    # asserted the opposite -- that canonical `module_1` was invalid.
    c2b = copy.deepcopy(_VALID)
    c2b["modules"][1]["inputs"][0]["source"] = "phase_1"
    c2b["final_outputs"][0]["source"] = "phase_2"
    cases.append(("legacy_phase_spelling", c2b, ""))

    # A prefix that is neither canonical nor legacy is still rejected, so relaxing
    # _MODULE_RE did not turn rule2 into a no-op.
    c2c = copy.deepcopy(_VALID)
    c2c["modules"][1]["inputs"][0]["source"] = "stage_1"
    cases.append(("rule2_unknown_prefix", c2c, "rule2"))

    c4 = copy.deepcopy(_VALID)
    c4["modules"][0]["outputs"] = []
    cases.append(("rule4_noop_module", c4, "rule4"))

    c5 = copy.deepcopy(_VALID)
    c5["modules"][0]["outputs"][0]["shape"] = "[M ** 2, N]"
    cases.append(("rule5_bad_shape", c5, "rule5"))

    c6 = copy.deepcopy(_VALID)
    c6["modules"][0]["outputs"][0]["dtype"] = "float128"
    cases.append(("rule6_bad_dtype", c6, "rule6"))

    c7 = copy.deepcopy(_VALID)
    c7["module_count"] = 3
    cases.append(("rule7_count_mismatch", c7, "rule7"))

    # rule0: a spec that declares nothing must not pass vacuously.
    cases.append(("rule0_empty_mapping", {}, "rule0"))

    c0 = copy.deepcopy(_VALID)
    c0["modules"] = []
    c0["module_count"] = 0
    cases.append(("rule0_no_modules", c0, "rule0"))

    c0b = copy.deepcopy(_VALID)
    del c0b["module_count"]
    cases.append(("rule0_missing_key", c0b, "rule0"))

    bad = 0
    for name, spec, want in cases:
        errs = validate(spec)
        if not want:
            ok = not errs
        else:
            ok = any(e.startswith(want) for e in errs)
        bad += 0 if ok else 1
        _LOGGER.error("%s %s: %s", "PASS" if ok else "FAIL", name,
                     errs if errs else "no errors")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml", nargs="?", type=Path)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    if args.self_test:
        return _self_test()
    if not args.yaml:
        ap.error("yaml path required (or --self-test)")
    spec = _load(args.yaml)
    errors = validate(spec or {})
    if args.json:
        _LOGGER.info(json.dumps(
            {"status": "PASS" if not errors else "FAIL", "violations": errors}, indent=2))
    elif errors:
        _LOGGER.error("FAIL:")
        for e in errors:
            _LOGGER.info("  - %s", e)
    else:
        _LOGGER.info("PASS")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
