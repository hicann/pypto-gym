# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check the CPU Golden's P0 ABI before a Scriptor Pro bootstrap is sealed."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from .common import ContractError, confined, digest


_DRIVER = r'''
import contextlib
import importlib.util
import json
import sys

import torch

sys.path.insert(0, sys.argv[2])
from scriptorlib.public_io import validate_public_io, validate_input_values, normalize_outputs, decode_params
source = sys.argv[1]
contract = json.load(sys.stdin)
sys.path.insert(0, str(__import__("pathlib").Path(source).parent))
spec = importlib.util.spec_from_file_location("scriptor_cpu_golden_preflight", source)
module = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(sys.stderr):
    spec.loader.exec_module(module)

factory = getattr(module, "_make_inputs", None)
golden = getattr(module, contract["op_name"] + "_golden_cpu", None)
if not callable(factory) or not callable(golden):
    raise ValueError("Scriptor CPU Golden needs _make_inputs(device) and <op>_golden_cpu")
with contextlib.redirect_stdout(sys.stderr), torch.no_grad():
    raw = factory(torch.device("cpu"))
if (len(contract["p0_cases"]) == 1 and isinstance(raw, tuple) and len(raw) == 2
        and isinstance(raw[0], (list, tuple)) and isinstance(raw[1], dict)):
    raw = [(contract["p0_cases"][0]["name"], raw[0], raw[1])]
if not isinstance(raw, (list, tuple)) or len(raw) != len(contract["p0_cases"]):
    raise ValueError("CPU Golden _make_inputs must return every SPEC P0 case exactly once")
by_name = {}
for item in raw:
    if (not isinstance(item, (list, tuple)) or len(item) != 3
            or not isinstance(item[0], str) or not isinstance(item[1], (list, tuple))
            or not isinstance(item[2], dict) or item[0] in by_name):
        raise ValueError("CPU Golden _make_inputs has an invalid or repeated case")
    by_name[item[0]] = item[1], item[2]
if set(by_name) != {case["name"] for case in contract["p0_cases"]}:
    raise ValueError("CPU Golden _make_inputs case IDs differ from SPEC")

inputs, outputs = contract["inputs"], contract["outputs"]
for case in contract["p0_cases"]:
    name = case["name"]
    args, _kwargs = by_name[name]
    if len(args) != len(inputs):
        raise ValueError(f"{name}: CPU Golden input count differs from SPEC")
    for tensor, meta in zip(args, inputs):
        key = meta["name"]
        expected_dtype = case.get("input_dtypes", {}).get(key, meta["dtype"])
        try:
            members = validate_public_io(tensor, name=key, shape=case["input_shapes"][key],
                dtype=expected_dtype, is_list=meta.get("is_list", False), device_type="cpu")
            validate_input_values(members, name=key, is_list=meta.get("is_list", False),
                value_range=meta.get("value_range"),
                special_values=case.get("input_special_values", {}).get(key, []))
        except ValueError as exc:
            raise ValueError(f"{name}: CPU Golden input {key}: {exc}") from exc
    with contextlib.redirect_stdout(sys.stderr), torch.no_grad():
        result = golden(*args, **decode_params(case["params"]))
    values = normalize_outputs(result, outputs)
    for meta in outputs:
        key = meta["name"]
        value = values[key]
        expected_dtype = case.get("output_dtypes", {}).get(key, meta["dtype"])
        try:
            validate_public_io(value, name=key, shape=case["output_shapes"][key],
                dtype=expected_dtype, is_list=meta.get("is_list", False), device_type="cpu")
        except ValueError as exc:
            observed_dtype = str(getattr(value, "dtype", type(value).__name__)).removeprefix("torch.")
            raise ValueError(f"{name}: CPU Golden output {key} shape/dtype/device differs from SPEC "
                             f"(dtype {observed_dtype} != {expected_dtype}): {exc}") from exc
print(json.dumps({"status": "PASS", "cases": len(by_name)}))
'''


def verify_cpu_golden(op_dir: Path, contract: dict, *, timeout: float = 600) -> dict:
    """Run the real CPU Golden once per declared P0 case in an isolated process."""
    source = confined(op_dir, f"{contract['op_name']}_golden_cpu.py", must_exist=True)
    env = dict(os.environ)
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    try:
        run = subprocess.run([sys.executable, "-c", _DRIVER, str(source), str(Path(__file__).resolve().parents[1])],
                             input=json.dumps(contract), capture_output=True, text=True,
                             cwd=op_dir, env=env, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise ContractError("CPU Golden P0 shape/dtype preflight timed out") from exc
    if run.returncode:
        reason = run.stderr.strip().splitlines()[-1] if run.stderr.strip() else "unknown error"
        raise ContractError(f"CPU Golden P0 shape/dtype preflight failed: {reason}")
    try:
        result = json.loads(run.stdout)
    except (ValueError, TypeError) as exc:
        raise ContractError("CPU Golden P0 preflight produced no valid result") from exc
    if result != {"status": "PASS", "cases": len(contract["p0_cases"])}:
        raise ContractError("CPU Golden P0 preflight did not cover every SPEC case")
    return {**result, "cpu_sha256": digest(source), "spec_sha256": digest(op_dir / "SPEC.md")}
