# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Export declared cases and a standalone PyPTO-Pro wrapper from the installed compiler."""
from __future__ import annotations

import sys
from pathlib import Path

from .sources import activate
from .common import ContractError, atomic_json, confined, digest, load_module, read_json
from .workflow import read_spec
from .public_io import validate_public_io, validate_input_values, json_safe


def task_module(op_dir):
    source = confined(op_dir, "scriptor/task.py", must_exist=True)
    sys.path.insert(0, str(source.parent))
    return load_module(source, "scriptor_task_" + digest(source)[:12])


def source_hashes(op_dir):
    files = [op_dir / "SPEC.md", *op_dir.glob("scriptor/**/*.py"), *op_dir.glob("scriptor/**/*.json")]
    return {str(p.relative_to(op_dir)): digest(p) for p in sorted(files)
            if p.is_file() and "__pycache__" not in p.parts}


def verify_recorded_emissions(op_dir, sync_mode):
    """Reject ad hoc PyPTO emissions that bypassed the frozen delivery mode."""
    reports = op_dir / "reports"
    if not reports.is_dir():
        return
    for path in reports.rglob("manifest.json"):
        if path.is_symlink():
            raise ContractError(f"PyPTO-Pro evidence is a symlink: {path.relative_to(op_dir)}")
        metadata = read_json(path)
        pypto = metadata.get("pypto", {}) if isinstance(metadata, dict) else {}
        recorded = pypto.get("sync_mode") if isinstance(pypto, dict) else None
        if recorded is None:
            continue
        if recorded != sync_mode:
            raise ContractError(f"PyPTO-Pro evidence used {recorded} instead of {sync_mode}: "
                                f"{path.relative_to(op_dir)}; preserve the violation evidence")
        source = confined(path.parent, metadata.get("entry"), must_exist=True).read_text(encoding="utf-8")
        decorator = f"@pl.jit(auto_mutex={sync_mode == 'auto_mutex'})"
        if decorator not in source:
            raise ContractError(f"PyPTO-Pro evidence decorator differs from {sync_mode}: {path.relative_to(op_dir)}")


def make_case(task, case, contract):
    import torch
    result = task.make_case(case)
    required = {"kernel", "args", "input_indices", "output_indices", "block_dim"}
    if not isinstance(result, dict) or not required <= set(result):
        raise ContractError(f"make_case must return {sorted(required)}")
    arguments = result["args"]
    if not isinstance(arguments, (list, tuple)):
        raise ContractError("case args must be in the typed kernel signature order")
    for category in ("input", "output"):
        mapping = result[f"{category}_indices"]
        declared = contract[f"{category}s"]
        if set(mapping) != {item["name"] for item in declared}:
            raise ContractError(f"{category}_indices does not match SPEC")
        for item in declared:
            index = mapping[item["name"]]
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(arguments):
                raise ContractError(f"invalid argument index for {item['name']}")
            value = arguments[index]
            expected_dtype = case.get(f"{category}_dtypes", {}).get(item["name"], item["dtype"])
            try:
                members = validate_public_io(value, name=item["name"],
                    shape=case[f"{category}_shapes"][item["name"]], dtype=expected_dtype,
                    is_list=item.get("is_list", False))
                if category == "input":
                    validate_input_values(members, name=item["name"],
                        is_list=item.get("is_list", False), value_range=item["value_range"],
                        special_values=case.get("input_special_values", {}).get(item["name"], []))
            except ValueError as exc:
                raise ContractError(str(exc)) from exc
    if isinstance(result["block_dim"], bool) or not isinstance(result["block_dim"], int) or result["block_dim"] < 1:
        raise ContractError("block_dim must be a positive integer")
    workspace_initialization = result.get("workspace_initialization", "empty")
    if workspace_initialization != "empty":
        raise ContractError("workspace_initialization must be empty; initialize required values in the kernel")
    output_initialization = result.get("output_initialization", {})
    if (not isinstance(output_initialization, dict)
            or set(output_initialization) - {item["name"] for item in contract["outputs"]}
            or any(mode != "empty" for mode in output_initialization.values())):
        raise ContractError("output_initialization must be empty; initialize required values in the kernel")
    return result


LAUNCHER = Path(__file__).with_name("public_io.py").read_text(encoding="utf-8").split("\ndef validate_input_values(", 1)[0] + '\n' + '''# Generated by CANNBot. Runtime dependencies: torch, pypto and pypto_pro only.
import importlib.util
import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CACHE = {}

def launch(inputs, params):
    import torch
    import pypto
    index = json.loads((_HERE / "export.json").read_text())

    def _matches(case):
        if canonical_params(case["params"]) != canonical_params(params):
            return False
        dtypes = case.get("input_dtypes", index["input_dtypes"])
        for name, shape in case["input_shapes"].items():
            if name not in inputs:
                return False
            try:
                validate_public_io(inputs[name], name=name, shape=shape, dtype=dtypes[name],
                                   is_list=index.get("input_is_list", {}).get(name, False))
            except ValueError:
                return False
        return True

    matches = [c for c in index["cases"] if _matches(c)]

    if not matches:
        raise ValueError("no exported PyPTO-Pro case for the supplied shape/dtype/parameters")
    case = matches[0]
    identity = {k: v for k, v in case.items() if k not in {"name", "directory"}}
    if any({k: v for k, v in match.items() if k not in {"name", "directory"}} != identity
           for match in matches[1:]):
        raise ValueError("ambiguous exported implementations for the supplied shape/dtype/parameters")
    first = next(iter(inputs.values()))
    if isinstance(first, (list, tuple)):
        first = first[0]  # a tensor-list input: take the first member's device/dtype
    for value in inputs.values():
        members = value if isinstance(value, (list, tuple)) else (value,)
        if any(member.device != first.device for member in members):
            raise ValueError("all public inputs must share a device")
    outputs = {}
    for name, shape in case["output_shapes"].items():
        alias = case["output_aliases"].get(name)
        if alias:
            outputs[name] = inputs[alias]
        else:
            if case["output_initialization"].get(name, "empty") != "empty":
                raise ValueError("generated wrapper requires kernel-owned output initialization")
            dtype = getattr(torch, case.get("output_dtypes", index["output_dtypes"])[name])
            outputs[name] = ([torch.empty(member_shape, dtype=dtype, device=first.device) for member_shape in shape]
                             if index.get("output_is_list", {}).get(name, False)
                             else torch.empty(shape, dtype=dtype, device=first.device))
    directory = _HERE / case["directory"]
    manifest = json.loads((directory / "manifest.json").read_text())
    if case["name"] not in _CACHE:
        spec = importlib.util.spec_from_file_location("_scriptor_export_" + case["name"], directory / manifest["entry"])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CACHE[case["name"]] = getattr(module, manifest["kernel"])
    args = []
    for binding in case["arguments"]:
        if binding["kind"] == "input":
            value = inputs[binding["name"]]
        elif binding["kind"] == "input_member":
            value = inputs[binding["name"]][binding["index"]]
        elif binding["kind"] == "output":
            value = outputs[binding["name"]]
        elif binding["kind"] == "output_member":
            value = outputs[binding["name"]][binding["index"]]
        elif binding["kind"] == "workspace":
            if binding.get("initialization", "empty") != "empty":
                raise ValueError("generated wrapper requires kernel-owned workspace initialization")
            value = torch.empty(binding["bytes"], dtype=torch.uint8, device=first.device)
        else:
            value = decode_scalar(binding["value"])
        if binding.get("reshape"):
            value = value.reshape(binding["reshape"])
        args.append(value)
    with pypto.options(pass_options={"enable_slice": False}):
        _CACHE[case["name"]][None, int(manifest["block_dim"])](*args)
    values = [outputs[name] for name in index["output_names"]]
    return values[0] if len(values) == 1 else tuple(values)
'''


def export(config, op_dir, *, sync_mode="auto_mutex"):
    if sync_mode not in {"manual", "auto_mutex"}:
        raise ContractError("sync_mode must be manual or auto_mutex")
    state_path = op_dir / ".scriptor/state.json"
    if not state_path.is_file():
        raise ContractError("initialize Scriptor state before exporting a delivery candidate")
    state = read_json(state_path)
    selected = state.get("delivery_sync_mode")
    if selected == "manual" and state.get("manual_requested_by_user") is not True:
        raise ContractError("manual export requires an explicit user request recorded at init")
    if selected is not None and sync_mode != selected:
        raise ContractError(f"export sync_mode={sync_mode} differs from the user-authorized delivery mode {selected}")
    if selected is not None:
        verify_recorded_emissions(op_dir, selected)
    root, sources = activate(config)
    contract = read_spec(config, op_dir)
    task = task_module(op_dir)
    generated = op_dir / "generated"
    generated.mkdir(exist_ok=True)
    index = {"schema": "cannbot.scriptor-export/1", "source_id": sources["source_id"],
             "source_hashes": source_hashes(op_dir), "sync_mode": sync_mode, "cases": [],
             "input_names": [x["name"] for x in contract["inputs"]],
             "input_is_list": {x["name"]: x.get("is_list", False) for x in contract["inputs"]},
             "output_names": [x["name"] for x in contract["outputs"]],
             "output_is_list": {x["name"]: x.get("is_list", False) for x in contract["outputs"]},
             "input_dtypes": {x["name"]: x["dtype"] for x in contract["inputs"]},
             "output_dtypes": {x["name"]: x["dtype"] for x in contract["outputs"]}}
    for case in contract["p0_cases"]:
        data = make_case(task, case, contract)
        kernel = data["kernel"]
        from ascriptor.backends.pypto_pro import emit_module, module_manifest
        from ascriptor.runtime.opexec import lower_kernel
        from .tensorlist_abi import validate_signature
        module = lower_kernel(kernel)
        # Task indices refer to the original signature, before list expansion or scalar folding.
        logical = [p for p in module_manifest(module, block_dim=data["block_dim"])["params"]
                   if p["kind"] != "workspace"]
        bindings = validate_signature(module, data)
        list_shapes = {}
        public_members = {"input": {}, "output": {}}
        for category in ("input", "output"):
            for item in contract[f"{category}s"]:
                name = item["name"]
                param = logical[data[f"{category}_indices"][name]]
                if (param["kind"] == "list") != item.get("is_list", False):
                    raise ContractError(f"SPEC TensorList type differs from the kernel signature: {name}")
                if item.get("is_list", False):
                    shapes = case[f"{category}_shapes"][name]
                    if param["ir_name"] in list_shapes and list_shapes[param["ir_name"]] != shapes:
                        raise ContractError(f"aliased TensorList shapes disagree: {name}")
                    public_members[category][name] = shapes
                    list_shapes[param["ir_name"]] = shapes
        artifacts = emit_module(module, block_dim=data["block_dim"],
                                entry=contract["op_name"] + "_kernel", bindings=bindings,
                                sync_mode=sync_mode, lists=list_shapes or None)
        from .nonfinite_scalars import materialize_nonfinite_scalars
        materialize_nonfinite_scalars(artifacts)
        # The compiler's diagnostic JSON may contain IEEE specials. The standalone
        # contract keeps JSON strict and spells scalar values as reserved strings.
        import json
        artifacts.metadata.update(json_safe(artifacts.metadata))
        for filename, content in list(artifacts.files.items()):
            if filename.endswith(".json"):
                artifacts.files[filename] = (json.dumps(json_safe(json.loads(content)), indent=2,
                                                       allow_nan=False) + "\n").encode()
        artifacts.files["manifest.json"] = (json.dumps(artifacts.metadata, indent=2, allow_nan=False) + "\n").encode()
        directory = confined(generated, case["name"])
        directory.mkdir(parents=True, exist_ok=True)
        for name, content in artifacts.files.items():
            target = confined(directory, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        input_by_ir, output_by_ir = {}, {}
        for name, position in data["input_indices"].items():
            input_by_ir[logical[position]["ir_name"]] = name
        for name, position in data["output_indices"].items():
            output_by_ir[logical[position]["ir_name"]] = name
        arguments = []
        for param in artifacts.metadata["params"]:
            ir_name = param.get("ir_name", param["name"])
            member_of = param.get("member_of")
            if param["kind"] == "workspace":
                binding = {"kind": "workspace", "bytes": artifacts.metadata["workspace_bytes"],
                           "initialization": data.get("workspace_initialization", "empty")}
            elif member_of and member_of in input_by_ir:
                binding = {"kind": "input_member", "name": input_by_ir[member_of],
                           "index": int(ir_name.rsplit("#", 1)[1])}
            elif member_of and member_of in output_by_ir:
                binding = {"kind": "output_member", "name": output_by_ir[member_of],
                           "index": int(ir_name.rsplit("#", 1)[1])}
            elif ir_name in input_by_ir:
                binding = {"kind": "input", "name": input_by_ir[ir_name]}
            elif ir_name in output_by_ir:
                binding = {"kind": "output", "name": output_by_ir[ir_name]}
            elif param["kind"] == "scalar":
                # Both synchronization modes fold original scalars into the source.
                # Only the backend's explicit const_scalars remain launch arguments.
                continue
            else:
                raise ContractError(f"unmapped exported parameter: {ir_name}")
            reshape = artifacts.metadata.get("pypto", {}).get("reshape", {}).get(param["name"])
            if reshape:
                binding["reshape"] = reshape
            arguments.append(binding)
        for value in artifacts.metadata.get("pypto", {}).get("const_scalars", []):
            arguments.append({"kind": "constant", "value": value["value"]})
        aliases = {name: input_by_ir[ir] for ir, name in output_by_ir.items() if ir in input_by_ir}
        initialization = data.get("output_initialization", {})
        if set(initialization) - set(index["output_names"]) or any(v != "empty" for v in initialization.values()):
            raise ContractError("output initialization must be empty; initialize required values in the kernel")
        case_dtypes = {
            "input_dtypes": case.get("input_dtypes", index["input_dtypes"]),
            "output_dtypes": case.get("output_dtypes", index["output_dtypes"]),
        }
        index["cases"].append({**case, **case_dtypes, "directory": case["name"], "arguments": arguments,
                               "output_aliases": aliases, "output_initialization": initialization,
                               "input_members": public_members["input"], "output_members": public_members["output"],
                               "kernel": artifacts.metadata["kernel"],
                               "files": {name: digest(directory / name) for name in artifacts.files}})
    atomic_json(generated / "export.json", index)
    (generated / "launch.py").write_text(LAUNCHER)
    parameters = [*index["input_names"], *[f"{key}={value!r}" for key, value in contract["default_params"].items()]]
    inputs = ", ".join(f"{name!r}: {name}" for name in index["input_names"])
    params = ", ".join(f"{name!r}: {name}" for name in contract["default_params"])
    wrapper = ("# Generated by CANNBot; change the DSL and export again.\n"
               "from generated.launch import launch\n\n"
               f"def {contract['op_name']}_wrapper({', '.join(parameters)}):\n"
               f"    return launch({{{inputs}}}, {{{params}}})\n")
    target = op_dir / f"test_{contract['op_name']}.py"
    if target.exists() and not target.read_text().startswith("# Generated by CANNBot"):
        raise ContractError("existing public wrapper is not exporter-owned; preserve it and adapt explicitly")
    target.write_text(wrapper)
    return {"source_id": sources["source_id"], "sync_mode": sync_mode, "cases": len(index["cases"]),
            "export": "generated/export.json", "wrapper": target.name,
            "scope": "declared P0 cases; additional domains require corresponding export/validation"}


def verify_export(config, op_dir):
    index = read_json(confined(op_dir, "generated/export.json", must_exist=True))
    if index["source_hashes"] != source_hashes(op_dir):
        raise ContractError("DSL or SPEC changed after export; export the candidate again")
    if index["source_id"] != read_json(config / "scriptor-install.json")["source_id"]:
        raise ContractError("export belongs to another ascriptor source identity")
    if index.get("sync_mode", "manual") not in {"manual", "auto_mutex"}:
        raise ContractError("invalid exported synchronization mode")
    for case in index["cases"]:
        for category in ("input", "output"):
            expected_members = {name: case[f"{category}_shapes"][name]
                                for name, flag in index.get(f"{category}_is_list", {}).items() if flag}
            if case.get(f"{category}_members", {}) != expected_members:
                raise ContractError(f"exported TensorList {category} members differ from declared shapes")
        directory = confined(op_dir / "generated", case["directory"])
        for name, expected in case["files"].items():
            if digest(confined(directory, name, must_exist=True)) != expected:
                raise ContractError("generated PyPTO-Pro file was edited after export")
        metadata = read_json(confined(directory, "manifest.json", must_exist=True))
        if metadata.get("pypto", {}).get("sync_mode", "manual") != index.get("sync_mode", "manual"):
            raise ContractError("export synchronization mode differs from the backend manifest")
        source = confined(directory, metadata["entry"], must_exist=True).read_text(encoding="utf-8")
        decorator = f"@pl.jit(auto_mutex={index.get('sync_mode', 'manual') == 'auto_mutex'})"
        if decorator not in source:
            raise ContractError("exported kernel decorator differs from the selected synchronization mode")
    return index
