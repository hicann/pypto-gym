#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
"""Validate custom/<op>/module_interfaces.yaml wiring.

Codifies the seven validity rules for the Pro Module contract:

  1. Every inputs[*].source == "primary" name exists in primary_inputs.
  2. Every inputs[*].source == "phase_j" has j < current module id, and the
     referenced name exists in phase_j.outputs (no forward/self reference).
  3. Every final_outputs[*].source == "phase_j" has j <= module_count, and the
     referenced name exists in phase_j.outputs.
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
_MODULE_RE = re.compile(r"^phase_(\d+)$")
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
            f"rule2: module {phase_id} input '{name}' has invalid source '{source}'"
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
            errors.append(f"rule3: final_output '{name}' has invalid source '{source}'")
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

    c2 = copy.deepcopy(_VALID)
    c2["modules"][0]["inputs"][0] = {"name": "y", "source": "module_2"}
    cases.append(("rule2_forward_ref", c2, "rule2"))

    c3 = copy.deepcopy(_VALID)
    c3["final_outputs"][0]["source"] = "module_9"
    cases.append(("rule3_out_of_range", c3, "rule3"))

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

    bad = 0
    for name, spec, want in cases:
        errs = validate(spec)
        if not want:
            ok = not errs
        else:
            ok = any(e.startswith(want) for e in errs)
        bad += 0 if ok else 1
        _LOGGER.info("%s %s: %s", "PASS" if ok else "FAIL", name,
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
        _LOGGER.info("FAIL:")
        for e in errors:
            _LOGGER.info("  - %s", e)
    else:
        _LOGGER.info("PASS")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
