# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check the final, Ascriptor-shaped handoff beside Scriptor's working files."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

from .common import ContractError, confined, digest, read_json


REQUIRED = ("golden_cpu.py", "wrapper.py", "test.py", "DESIGN.md", "REPORT.md")
CASE_FIELDS = ("name", "kernel", "input_shapes", "input_dtypes",
               "output_shapes", "output_dtypes", "params")
WORK_ONLY = {".tmp", ".scriptor", "scriptor", "generated", "reports", "prototype",
             "SPEC.md", "GOLDEN_VALIDATION.json", "PRO_MATERIAL_INDEX.md",
             "EXPLORE_REPORT.md", "KB_SELECTION.json", "DESIGN_BINDINGS.json",
             "module_interfaces.yaml", ".DS_Store", "__pycache__", "test_results.json"}

_WRAPPER_TRACE_RUNNER = '''import functools, importlib, inspect, json, runpy, sys
from pathlib import Path
result_path, trace_path, special_json = sys.argv[1:]
special_cases = json.loads(special_json)
if special_cases:
    import torch
calls = []
special_calls = []
wrapper = importlib.import_module("wrapper")
for name, function in tuple(vars(wrapper).items()):
    if inspect.isfunction(function) and function.__module__ == "wrapper" and not name.startswith("_"):
        def traced(label, original):
            signature = inspect.signature(original)
            @functools.wraps(original)
            def call(*args, **kwargs):
                calls.append(label)
                if special_cases:
                    try:
                        bound = signature.bind_partial(*args, **kwargs).arguments
                    except TypeError:
                        bound = {}
                    for case in special_cases:
                        observed = {}
                        for input_name in case["input_special_values"]:
                            value = bound.get(input_name)
                            is_list = case.get("input_is_list", {}).get(input_name, False)
                            members = value if is_list and isinstance(value, (list, tuple)) else [value]
                            shapes = case["input_shapes"][input_name] if is_list else [case["input_shapes"][input_name]]
                            if (not members or len(members) != len(shapes)
                                    or any(not isinstance(t, torch.Tensor) or list(t.shape) != shape
                                           or str(t.dtype).removeprefix("torch.") != case["input_dtypes"][input_name]
                                           for t, shape in zip(members, shapes))):
                                break
                            counts = {"-inf": 0, "+inf": 0, "nan": 0,
                                      "device": "npu" if all(t.device.type == "npu" for t in members) else "non-npu"}
                            for member in members:
                                sample = member.detach().cpu()
                                counts["-inf"] += int(torch.isneginf(sample).sum().item())
                                counts["+inf"] += int(torch.isposinf(sample).sum().item())
                                counts["nan"] += int(torch.isnan(sample).sum().item())
                            observed[input_name] = counts
                        else:
                            special_calls.append({"case": case["name"], "inputs": observed})
                return original(*args, **kwargs)
            return call
        setattr(wrapper, name, traced(name, function))
try:
    sys.argv = ["test.py", "--output", result_path]
    runpy.run_path("test.py", run_name="__main__")
finally:
    Path(trace_path).write_text(json.dumps({"calls": calls, "special_inputs": special_calls}))
'''


def _file(root: Path, relative: str) -> Path:
    if (root / relative).is_symlink():
        raise ContractError(f"delivery file must be a non-empty regular file: {relative}")
    path = confined(root, relative, must_exist=True)
    if not path.stat().st_size:
        raise ContractError(f"delivery file must be a non-empty regular file: {relative}")
    return path


def _literal_cases(path: Path) -> list[dict]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assignments = [node.value for node in tree.body
                   if isinstance(node, ast.Assign) and any(
                       isinstance(target, ast.Name) and target.id == "CASES" for target in node.targets)]
    if len(assignments) != 1:
        raise ContractError("test.py must declare one literal CASES list")
    try:
        cases = ast.literal_eval(assignments[0])
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as exc:
        raise ContractError("test.py CASES must be literal data") from exc
    if not isinstance(cases, (list, tuple)) or not cases or any(not isinstance(row, dict) for row in cases):
        raise ContractError("test.py CASES must be a non-empty list of case objects")
    return list(cases)


def _check_public_wrapper_call(path: Path) -> None:
    """Reject an entry that only prints case records without using the packaged router."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules = {alias.asname or "wrapper" for node in tree.body if isinstance(node, ast.Import)
               for alias in node.names if alias.name == "wrapper"}
    symbols = {alias.asname or alias.name for node in tree.body if isinstance(node, ast.ImportFrom)
               and node.module == "wrapper" for alias in node.names}
    calls = (node.func for node in ast.walk(tree) if isinstance(node, ast.Call))
    if not any((isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                and func.value.id in modules) or (isinstance(func, ast.Name) and func.id in symbols)
               for func in calls):
        raise ContractError("delivery test.py must call its packaged public wrapper")


def check_delivery(op_dir: Path, delivery_dir: Path, export_index: dict | None = None) -> dict:
    """Check a slim package against the selected work-area export and SPEC P0 cases."""
    if delivery_dir.is_symlink():
        raise ContractError("delivery root cannot be a symlink")
    op_dir, delivery_dir = op_dir.resolve(), delivery_dir.resolve()
    if (op_dir.parent.name != "custom" or delivery_dir.parent.name != "delivery"
            or op_dir.name != delivery_dir.name or op_dir.parent.parent != delivery_dir.parent.parent):
        raise ContractError("use paired custom/<op>/ work and delivery/<op>/ package roots")
    if not delivery_dir.is_dir():
        raise ContractError("delivery/<op>/ package is missing")
    for path in delivery_dir.rglob("*"):
        if path.is_symlink():
            raise ContractError(f"delivery package contains a symlink: {path.relative_to(delivery_dir)}")
        if any(part in WORK_ONLY for part in path.relative_to(delivery_dir).parts):
            raise ContractError(f"delivery package contains a work-only artifact: {path.relative_to(delivery_dir)}")
    for relative in REQUIRED:
        _file(delivery_dir, relative)
    golden = ast.parse((delivery_dir / "golden_cpu.py").read_text(encoding="utf-8"))
    functions = {node.name for node in golden.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if not ({"make_inputs", "reference"} <= functions or
            any(name.endswith("_golden_cpu") for name in functions)):
        raise ContractError("golden_cpu.py needs independent reference entry points")
    for node in ast.walk(golden):
        modules = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                   else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
        if any(name.split(".", 1)[0] in {"ascriptor", "torch_npu"} for name in modules):
            raise ContractError("golden_cpu.py must not import ascriptor or torch_npu")

    export_file = _file(op_dir, "generated/export.json")
    index = export_index if export_index is not None else read_json(export_file)
    if not isinstance(index, dict) or not isinstance(index.get("source_hashes"), dict):
        raise ContractError("generated/export.json has no source identity")
    state_path = op_dir / ".scriptor/state.json"
    if state_path.is_file():
        state = read_json(state_path)
        selected = state.get("delivery_sync_mode")
        if selected is not None:
            if state.get("current_stage") != "done":
                raise ContractError("fixed-mode Scriptor delivery requires completed acceptance")
            if index.get("sync_mode") != selected:
                raise ContractError("delivery export differs from the recorded synchronization mode")
            if selected == "manual" and state.get("manual_requested_by_user") is not True:
                raise ContractError("manual delivery lacks an explicit user request")
            from .exporter import verify_recorded_emissions
            verify_recorded_emissions(op_dir, selected)
    for relative, expected in index["source_hashes"].items():
        if digest(_file(op_dir, relative)) != expected:
            raise ContractError(f"delivery export source changed: {relative}")
    report = (delivery_dir / "REPORT.md").read_text(encoding="utf-8")
    export_sha = digest(export_file)
    final_export = re.findall(r"^Final export SHA-256: ([0-9a-f]{64})$", report, re.M)
    if final_export != [export_sha]:
        raise ContractError("REPORT.md must identify exactly the selected final export.json SHA-256")
    final_sources = re.findall(r"^Final source SHA-256 \((scriptor/[^)]+\.py)\): ([0-9a-f]{64})$", report, re.M)
    required_sources = {relative: expected for relative, expected in index["source_hashes"].items()
                        if relative.startswith("scriptor/") and relative.endswith(".py")}
    if len(final_sources) != len(required_sources) or dict(final_sources) != required_sources:
        raise ContractError("REPORT.md final source SHA-256 fields differ from the selected DSL")

    rows = _literal_cases(delivery_dir / "test.py")
    _check_public_wrapper_call(delivery_dir / "test.py")
    exported = index.get("cases")
    if not isinstance(exported, list) or not exported:
        raise ContractError("generated/export.json has no cases")
    by_name = {case.get("name"): case for case in exported if isinstance(case, dict)}
    if len(by_name) != len(exported):
        raise ContractError("export has duplicate or invalid case names")
    seen = set()
    used_kernels = set()
    for row in rows:
        if any(key not in row for key in CASE_FIELDS):
            raise ContractError(f"test.py CASES needs {', '.join(CASE_FIELDS)}")
        name = row["name"]
        if not isinstance(name, str) or name not in by_name or name in seen:
            raise ContractError(f"delivery test.py CASES has an unknown or repeated case: {name!r}")
        seen.add(name)
        case = by_name[name]
        directory = case.get("directory")
        manifest = read_json(_file(op_dir, f"generated/{directory}/manifest.json"))
        entry = manifest.get("entry")
        kernel = row["kernel"]
        if not isinstance(kernel, str) or not kernel.startswith("kernels/") or not kernel.endswith(".py"):
            raise ContractError(f"delivery case must select a kernels/*.py source: {name}")
        expected = {"input_shapes": case["input_shapes"],
                    "input_dtypes": case.get("input_dtypes", index["input_dtypes"]),
                    "output_shapes": case["output_shapes"],
                    "output_dtypes": case.get("output_dtypes", index["output_dtypes"]),
                    "params": case["params"]}
        if any(row[key] != value for key, value in expected.items()):
            raise ContractError(f"delivery test.py CASES differs from the selected export: {name}")
        for category in ("input", "output"):
            expected_lists = {key: flag for key, flag in index.get(f"{category}_is_list", {}).items() if flag}
            declared_lists = {key: flag for key, flag in row.get(f"{category}_is_list", {}).items() if flag}
            if declared_lists != expected_lists or any(type(flag) is not bool for flag in row.get(f"{category}_is_list", {}).values()):
                raise ContractError(f"delivery test.py TensorList types differ from the selected export: {name}")
            if row.get(f"{category}_members", {}) != case.get(f"{category}_members", {}):
                raise ContractError(f"delivery test.py TensorList members differ from the selected export: {name}")
        if row.get("input_special_values", {}) != case.get("input_special_values", {}):
            raise ContractError(f"delivery test.py special inputs differ from the selected export: {name}")
        if entry not in case.get("files", {}):
            raise ContractError(f"export has no kernel entry hash for {name}")
        if digest(_file(delivery_dir, kernel)) != case["files"][entry]:
            raise ContractError(f"delivery kernel differs from the selected export: {kernel}")
        used_kernels.add(kernel)
    if seen != set(by_name):
        raise ContractError(f"delivery test.py CASES omits exported cases: {sorted(set(by_name) - seen)}")
    kernels = delivery_dir / "kernels"
    actual_kernels = {str(path.relative_to(delivery_dir)) for path in kernels.rglob("*") if path.is_file()} if kernels.is_dir() else set()
    if actual_kernels != used_kernels:
        raise ContractError("kernels/ must contain exactly the exported sources reached by test.py")
    return {"status": "STRUCTURE_PASS", "delivery_root": str(delivery_dir), "cases": len(rows),
            "case_records": rows, "kernel_sources": sorted(used_kernels), "export_sha256": export_sha,
            "source_hashes": index["source_hashes"],
            "scope": "structure, identity and declared domain only; run isolated test.py for device evidence"}


def run_delivery_test(op_dir: Path, delivery_dir: Path, structure: dict, *, timeout: float = 3600) -> dict:
    """Execute only the packaged files in a fresh directory on the current device host."""
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 86400:
        raise ContractError("delivery test timeout must be in (0, 86400] seconds")
    op_dir, delivery_dir = op_dir.resolve(), delivery_dir.resolve()
    evidence = op_dir / "reports/delivery-check" / str(time.time_ns())
    evidence.mkdir(parents=True, exist_ok=False)
    source_files = {str(path.relative_to(delivery_dir)): digest(path)
                    for path in delivery_dir.rglob("*") if path.is_file()}
    with tempfile.TemporaryDirectory(prefix="scriptor-delivery-") as raw:
        stage = Path(raw) / delivery_dir.name
        shutil.copytree(delivery_dir, stage)
        jit_work = Path(raw) / "jit-work"
        jit_work.mkdir()
        result_file = stage / "test_results.json"
        trace_file = stage / "wrapper_calls.json"
        env = dict(os.environ)
        for name in ("PYTHONPATH", "CANNBOT_CONFIG_ROOT", "ASCRIPTOR_BOARD", "ASCRIPTOR_BOARDS"):
            env.pop(name, None)
        # PyPTO-Pro otherwise creates build/ below cwd and makes a valid slim package
        # look mutated. Keep JIT and incidental tempfile output outside the package.
        env["ASCEND_WORK_PATH"] = str(jit_work)
        env["TMPDIR"] = raw
        special_cases = [row for row in structure["case_records"] if row.get("input_special_values")]
        try:
            run = subprocess.run([sys.executable, "-c", _WRAPPER_TRACE_RUNNER,
                                  str(result_file), str(trace_file), json.dumps(special_cases)],
                                 cwd=stage, env=env, capture_output=True, text=True,
                                 timeout=timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or b""
            stderr = exc.stderr or b""
            (evidence / "test.stdout").write_bytes(stdout.encode() if isinstance(stdout, str) else stdout)
            (evidence / "test.stderr").write_bytes(stderr.encode() if isinstance(stderr, str) else stderr)
            raise ContractError(f"isolated delivery test timed out; inspect {evidence}") from exc
        (evidence / "test.stdout").write_text(run.stdout)
        (evidence / "test.stderr").write_text(run.stderr)
        if run.returncode != 0 or not result_file.is_file():
            raise ContractError(f"isolated delivery test failed or wrote no test_results.json; inspect {evidence}")
        shutil.copy2(result_file, evidence / "test_results.json")
        if not trace_file.is_file():
            raise ContractError(f"isolated delivery test wrote no wrapper call trace; inspect {evidence}")
        shutil.copy2(trace_file, evidence / "wrapper_calls.json")
        try:
            result = read_json(result_file)
        except (ValueError, OSError) as exc:
            raise ContractError(f"isolated delivery result is invalid; inspect {evidence}") from exc
        observed = result.get("cases")
        expected = {row["name"]: row["kernel"] for row in structure["case_records"]}
        trace = read_json(trace_file)
        calls = trace.get("calls") if isinstance(trace, dict) else None
        if not isinstance(calls, list) or len(calls) < len(expected):
            raise ContractError(f"isolated delivery test did not call the packaged wrapper for every case; inspect {evidence}")
        observations = trace.get("special_inputs", [])
        def matches(observation, case):
            if observation.get("case") != case["name"]:
                return False
            for name, required in case["input_special_values"].items():
                counts = observation.get("inputs", {}).get(name, {})
                if (counts.get("device") != "npu"
                        or any((counts.get(value, 0) > 0) != (value in required)
                               for value in ("-inf", "+inf", "nan"))):
                    return False
            return True

        for case in special_cases:
            if not any(matches(item, case) for item in observations if isinstance(item, dict)):
                raise ContractError(f"isolated delivery test did not pass original special inputs through the NPU wrapper: {case['name']}; inspect {evidence}")
        if (result.get("status") != "PASS" or not isinstance(observed, list)
                or len(observed) != len(expected) or result.get("case_count") != len(expected)):
            raise ContractError(f"isolated delivery result lacks complete PASS cases; inspect {evidence}")
        actual = {row.get("name"): row for row in observed if isinstance(row, dict)}
        if (len(actual) != len(expected) or set(actual) != set(expected)
                or any(actual[name].get("status") != "PASS" or actual[name].get("kernel") != kernel
                       for name, kernel in expected.items())):
            raise ContractError(f"isolated delivery result case/kernel routing differs from CASES; inspect {evidence}")
        stage_files = {str(path.relative_to(stage)): digest(path)
                       for path in stage.rglob("*") if path.is_file()
                       and "__pycache__" not in path.parts and path not in (result_file, trace_file)}
        if stage_files != source_files:
            raise ContractError(f"delivery files changed during isolated test; inspect {evidence}")
    return {**structure, "status": "PASS", "device_test_cases": len(expected),
            "test_result_sha256": digest(evidence / "test_results.json"),
            "evidence_dir": str(evidence.relative_to(op_dir)),
            "package_hashes": source_files,
            "scope": "structure, selected-export identity, SPEC P0 cases and isolated package test on this host"}
