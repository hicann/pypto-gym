# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run independent checks against the DSL and the standalone PyPTO-Pro delivery."""
from __future__ import annotations

import csv
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

from .sources import activate
from .common import ContractError, atomic_json, confined, digest, finite, load_module, read_json
from .exporter import make_case, task_module, verify_export
from .public_io import normalize_outputs, validate_public_io, decode_params
from .workflow import artifact_hashes, evaluate_report, read_spec, subject


class EnvironmentUnavailable(RuntimeError):
    pass


def summarize_checks(cases, names):
    """A check passes only if it actually passed for every declared case."""
    result = {}
    for name in names:
        statuses = [case.get("checks", {}).get(name, "NOT_RUN") for case in cases.values()]
        if not statuses:
            result[name] = "NOT_RUN"
        elif "FAIL" in statuses:
            result[name] = "FAIL"
        elif "BLOCKED" in statuses:
            result[name] = "BLOCKED"
        elif any(status != "PASS" for status in statuses):
            result[name] = "NOT_RUN"
        else:
            result[name] = "PASS"
    return result


def runtime_options(config, overrides=None):
    path = config / "scriptor.local.json"
    result = read_json(path) if path.exists() else {}
    if not isinstance(result, dict) or set(result) - {"board", "boards_file", "timeout"}:
        raise ContractError("scriptor.local.json accepts board, boards_file and timeout")
    result.update({k: v for k, v in (overrides or {}).items() if v is not None})
    if "board" not in result and os.environ.get("ASCRIPTOR_BOARD"):
        result["board"] = os.environ["ASCRIPTOR_BOARD"]
    if "boards_file" not in result and os.environ.get("ASCRIPTOR_BOARDS"):
        result["boards_file"] = os.environ["ASCRIPTOR_BOARDS"]
    result.setdefault("timeout", 300)
    finite(result["timeout"], "runtime timeout", positive=True)
    return result


def _board(options):
    if not options.get("board"):
        return None
    from ascriptor.runtime.board import Board
    board = Board.from_config(options["board"], Path(options["boards_file"]) if options.get("boards_file") else None)
    board.require_local("Scriptor verification")
    return board


def _delivery_copy(op_dir, destination, contract):
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(op_dir / "generated", destination / "generated", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "build"))
    shutil.copy2(op_dir / f"test_{contract['op_name']}.py", destination)


def _execute(directory, argv, options, *, profile=False):
    """Execute on this device machine; never transfer runs over SSH."""
    _board(options)  # Validate an explicitly selected device describes this machine.
    timeout = options["timeout"]
    import importlib.util
    if importlib.util.find_spec("pypto_pro") is None:
        raise EnvironmentUnavailable("no local PyPTO-Pro; run Scriptor on the device machine")
    command = list(argv)
    if profile:
        if not shutil.which("msprof"):
            raise EnvironmentUnavailable("msprof is required for device-side latency measurement")
        command = ["msprof", "--output=./prof", "--aic-metrics=PipeUtilization", "--task-time=on", *command]
    result = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=timeout)
    (directory / "execution.log").write_text(result.stdout + "\n" + result.stderr)
    if profile:
        sources = list((directory / "prof").glob("PROF_*/mindstudio_profiler_output/op_summary_*.csv"))
        if sources:
            (directory / "op_summary.csv").write_text("\n".join(p.read_text() for p in sources))
    return result.returncode


def _evidence(op_dir, path):
    return {"path": str(path.relative_to(op_dir)), "sha256": digest(path)}


def _cpu_argument(value):
    """A public input/output as CPU tensors: one tensor, or a list for a TensorList."""
    if isinstance(value, (list, tuple)):
        return [member.detach().cpu().clone() for member in value]
    return value.detach().cpu().clone()


def _expected(golden, contract, case, data):
    inputs = [_cpu_argument(data["args"][data["input_indices"][item["name"]]])
              for item in contract["inputs"]]
    result = getattr(golden, contract["op_name"] + "_golden_cpu")(*inputs, **decode_params(case["params"]))
    try:
        return normalize_outputs(result, contract["outputs"])
    except ValueError as exc:
        raise ContractError(f"CPU Golden {exc}") from exc


def _precision_engine(config, contract):
    tolerance = contract["tolerance"]
    if "policy" not in tolerance:
        return None, {"policy": "legacy_tolerance", "tolerance": tolerance}
    if tolerance != {"policy": "pro_scheme_a"}:
        raise ContractError("Scheme A requires tolerance={policy: pro_scheme_a}, without threshold overrides")
    relative = "skills/pypto-pro-op-develop/scripts/precision_compare.py"
    path = confined(config, relative, must_exist=True)
    identity = {"path": relative, "sha256": digest(path)}
    engine = load_module(path, "scriptor_pro_precision")
    return engine, {"policy": "pro_scheme_a", "engine": identity}


def _compare(actual, expected, contract, case, *, engine=None, details=None):
    import torch
    names = {item["name"] for item in contract["outputs"]}
    if set(actual) != names or set(expected) != names:
        raise AssertionError("actual/Golden output set differs from SPEC")
    scheme_a = contract["tolerance"].get("policy") == "pro_scheme_a"
    if scheme_a and engine is None:
        raise ContractError("Scheme A requires the installed Pro precision engine")
    errors = {}
    for item in contract["outputs"]:
        name, is_list = item["name"], item.get("is_list", False)
        dtype = case.get("output_dtypes", {}).get(name, item["dtype"])
        try:
            values = validate_public_io(actual[name], name=name, shape=case["output_shapes"][name],
                                        dtype=dtype, is_list=is_list)
            references = validate_public_io(expected[name], name=name, shape=case["output_shapes"][name],
                                            dtype=dtype, is_list=is_list)
        except ValueError as exc:
            raise AssertionError(f"actual/Golden output: {exc}") from exc
        member_errors, member_details = [], []
        for index, (value, reference) in enumerate(zip(values, references)):
            label = f"{name}[{index}]" if is_list else name
            value, reference = value.detach().cpu(), reference.detach().cpu()
            if scheme_a:
                passed, summary = engine.check_precision(value, reference)
                member_details.append({"passed": passed, "summary": summary})
                if not passed:
                    raise AssertionError(f"Scheme A precision failed for {label}: {summary}")
                member_errors.append(summary)
            elif value.is_floating_point():
                # Matching NaNs and same-sign infinities are valid results. The
                # maximum-error metric uses finite positions only, keeping JSON strict.
                torch.testing.assert_close(value, reference, check_dtype=False,
                                           equal_nan=True, **{key: contract["tolerance"][key]
                                                              for key in ("atol", "rtol")})
                mask = torch.isfinite(value) & torch.isfinite(reference)
                error = float((value[mask].double() - reference[mask].double()).abs().max()) if bool(mask.any()) else 0.0
                member_errors.append(error)
            else:
                integer_atol = contract["tolerance"].get("integer_atol", 0)
                if reference.is_floating_point() or reference.is_complex() or reference.is_quantized:
                    raise AssertionError(f"integer/boolean Golden must retain its dtype: {label}")
                if torch.equal(value, reference):
                    member_errors.append(0)
                elif integer_atol and value.dtype != torch.bool and value.element_size() <= 4:
                    # Widen first: int8 subtraction wraps at the quantization limits.
                    error = int((value.to(torch.int64) - reference.to(torch.int64)).abs().max())
                    if error > integer_atol:
                        raise AssertionError(f"integer output exceeds integer_atol={integer_atol}: {label}, max_abs_error={error}")
                    member_errors.append(error)
                else:
                    raise AssertionError(f"integer/boolean output must exactly match Golden: {label}")
        errors[name] = member_errors if is_list else member_errors[0]
        if details is not None and scheme_a:
            details[name] = {"members": member_details} if is_list else member_details[0]
    return errors


def _read_outputs(directory):
    """Rebuild one public value per name, retaining list member order."""
    import torch
    metadata = read_json(directory / "metadata.json")
    def tensor(item, fallback):
        path = confined(directory, item.get("file", fallback), must_exist=True)
        raw = bytearray(path.read_bytes())
        return torch.frombuffer(raw, dtype=getattr(torch, item["dtype"])).reshape(item["shape"])
    return {name: ([tensor(member, f"{name}.{i}.bin") for i, member in enumerate(item["members"])]
                   if item.get("kind") == "list" else tensor(item, f"{name}.bin"))
            for name, item in metadata.items()}


def _metric(op_dir, source, candidate_hash, *, metric, case, value, unit, basis, label):
    return {"metric": metric, "case": case, "value": value, "unit": unit,
            "basis": basis, "source": label, "artifact_hash": candidate_hash,
            "evidence_file": str(source.relative_to(op_dir)), "evidence_sha256": digest(source)}


def _profile(csv_path, kernel, *, warmup=5, repeat=32):
    if not csv_path.is_file():
        raise EnvironmentUnavailable("device profiling produced no op_summary CSV")
    with csv_path.open() as stream:
        rows = list(csv.DictReader(stream))
    selected = [r for r in rows if kernel.lower() in " ".join(
        str(r.get(key, "")) for key in ("Name", "Op Name", "OP Name", "Task Name")).lower()
                and r.get("Task Duration(us)") not in (None, "", "Task Duration(us)")]
    if len(selected) != repeat:
        raise ContractError(f"profiler has {len(selected)} identifiable kernel rows; expected exactly {repeat}; resolve multi-task/extra-launch timing before reporting latency")
    kept = selected[warmup:]
    durations = [float(r["Task Duration(us)"]) for r in kept]
    if not durations or any(not (x > 0) for x in durations):
        raise ContractError("profiler durations are invalid")
    result = {"latency_us": min(durations), "median_us": statistics.median(durations),
              "warmup": warmup, "repeat": repeat, "kept": len(kept)}
    values = [float(r["aiv_vec_ratio"]) for r in kept if r.get("aiv_vec_ratio") not in (None, "", "N/A")]
    if values:
        result["vector_pipe_utilization_pct"] = statistics.mean(values) * 100
    return result


def check(config, op_dir, stage, *, options=None):
    activate(config)
    options = runtime_options(config, options)
    contract = read_spec(config, op_dir)
    precision_engine, precision_policy = _precision_engine(config, contract)
    candidate_hash = subject(op_dir)
    hashes = artifact_hashes(op_dir)
    run = op_dir / "reports/runs" / f"{stage}-{time.time_ns()}"
    run.mkdir(parents=True)
    result = {"schema": "cannbot.scriptor-check/1", "stage": stage,
              "artifact_hash": candidate_hash, "artifact_hashes": hashes,
              "checks": {"spec": "PASS"}, "metrics": [], "evidence": [], "cases": {},
              "verdict": "PASS", "failures": [], "execution_policy": "hardware_first",
              "precision_policy": precision_policy}
    if stage == "prepare":
        for kind, filename in (("golden_cpu", f"{contract['op_name']}_golden_cpu.py"),
                               ("golden_npu", f"{contract['op_name']}_golden.py")):
            directory = run / kind
            directory.mkdir()
            for p in (op_dir / "SPEC.md", *op_dir.glob("*_golden*.py")):
                shutil.copy2(p, directory / p.name)
            try:
                if kind == "golden_cpu":
                    p = subprocess.run([sys.executable, filename], cwd=directory, capture_output=True,
                                       text=True, timeout=options["timeout"],
                                       env={**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"})
                    (directory / "execution.log").write_text(p.stdout + "\n" + p.stderr)
                    code = p.returncode
                else:
                    code = _execute(directory, [sys.executable, filename], options)
                result["checks"][kind] = "PASS" if code == 0 else "FAIL"
                result["evidence"].append(_evidence(op_dir, directory / "execution.log"))
                if code:
                    result["failures"].append({"check": kind, "exit_code": code})
            except EnvironmentUnavailable as exc:
                result["checks"][kind] = "BLOCKED"
                result["failures"].append({"check": kind, "reason": str(exc)})
    else:
        try:
            export_index = verify_export(config, op_dir)
        except ContractError as exc:
            # A failed or stale export is a real candidate failure, not device
            # evidence. Retain it so the normal verifier can reject the round.
            state = read_json(op_dir / ".scriptor/state.json")
            result["sync_mode"] = state.get("delivery_sync_mode", "manual")
            result["checks"].update({name: "NOT_RUN" for name in
                ("functional", "pipesim", "pypto_hardware", "standalone", "latency")})
            result["checks"]["export"] = "FAIL"
            result["verdict"] = "FAIL"
            result["failures"].append({"check": "export", "reason": str(exc)})
            diagnostic = run / "export-preflight-error.json"
            atomic_json(diagnostic, {"error": type(exc).__name__, "message": str(exc),
                                     "device_execution": "NOT_RUN"})
            result["evidence"].append(_evidence(op_dir, diagnostic))
            atomic_json(run / "checks.json", result)
            return {"report": str((run / "checks.json").relative_to(op_dir)),
                    "verdict": "FAIL", "checks": result["checks"], "failures": result["failures"]}
        result["sync_mode"] = export_index.get("sync_mode", "manual")
        result["diagnostics"] = {name: {"status": "NOT_RUN", "reason": "Use a reduced, explicitly scoped probe after a hardware issue; full-case simulation is not automatic."}
                                 for name in ("functional", "pipesim")}
        task = task_module(op_dir)
        golden = load_module(op_dir / f"{contract['op_name']}_golden_cpu.py", "scriptor_cpu_golden")
        import torch
        execution_checks = ("functional", "pipesim", "pypto_hardware", "standalone", "latency")
        for case in contract["p0_cases"]:
            case_result = {"checks": {name: "NOT_RUN" for name in execution_checks}}
            statuses = case_result["checks"]
            directory = run / case["name"]
            directory.mkdir()
            try:
                data = make_case(task, case, contract)
                _delivery_copy(op_dir, directory, contract)
                shutil.copy2(Path(__file__).resolve().parents[1] / "standalone_driver.py", directory)
                inputs = []
                for item in contract["inputs"]:
                    value = data["args"][data["input_indices"][item["name"]]]
                    dtype = case.get("input_dtypes", {}).get(item["name"], item["dtype"])
                    if isinstance(value, (list, tuple)):
                        members = []
                        for k, member in enumerate(value):
                            member = member.detach().cpu().contiguous()
                            filename = f"{item['name']}.{k}.bin"
                            (directory / filename).write_bytes(member.view(torch.uint8).numpy().tobytes())
                            members.append({"shape": list(member.shape), "file": filename})
                        inputs.append({"name": item["name"], "kind": "list", "dtype": dtype,
                                       "members": members})
                    else:
                        value = value.detach().cpu().contiguous()
                        filename = f"{item['name']}.bin"
                        (directory / filename).write_bytes(value.view(torch.uint8).numpy().tobytes())
                        inputs.append({"name": item["name"], "shape": list(value.shape), "dtype": dtype,
                                       "file": filename})
                atomic_json(directory / "input.json", {"inputs": inputs,
                    "input_names": [x["name"] for x in contract["inputs"]],
                    "output_names": [x["name"] for x in contract["outputs"]],
                    "outputs": contract["outputs"],
                    "wrapper_file": f"test_{contract['op_name']}.py", "wrapper": contract["op_name"] + "_wrapper",
                    "params": case["params"], "repeat": 32})
                code = _execute(directory, [sys.executable, "standalone_driver.py"], options, profile=True)
                result["evidence"].append(_evidence(op_dir, directory / "execution.log"))
                if code:
                    raise AssertionError(f"standalone PyPTO-Pro process exited {code}; inspect execution.log")
                actual = _read_outputs(directory / "output")
                expected = _expected(golden, contract, case, data)
                case_result["precision"] = {}
                case_result["pypto_hardware"] = _compare(actual, expected, contract, case,
                    engine=precision_engine, details=case_result["precision"])
                statuses["pypto_hardware"] = "PASS"
                statuses["standalone"] = "PASS"
                profile_path = directory / "op_summary.csv"
                profile = _profile(profile_path, contract["op_name"] + "_kernel")
                statuses["latency"] = "PASS"
                result["evidence"].append(_evidence(op_dir, profile_path))
                case_result["profile"] = profile
                result["metrics"].append(_metric(op_dir, profile_path, candidate_hash,
                    metric="latency_us", case=case["name"], value=profile["latency_us"], unit="us",
                    basis="hardware", label="msprof.Task Duration(us).min_after_warmup"))
                if "vector_pipe_utilization_pct" in profile:
                    result["metrics"].append(_metric(op_dir, profile_path, candidate_hash,
                        metric="vector_pipe_utilization_pct", case=case["name"],
                        value=profile["vector_pipe_utilization_pct"], unit="%", basis="hardware",
                        label="msprof.aiv_vec_ratio.mean_after_warmup"))
            except EnvironmentUnavailable as exc:
                for name in ("pypto_hardware", "standalone", "latency"):
                    if statuses[name] != "PASS":
                        statuses[name] = "BLOCKED"
                result["failures"].append({"case": case["name"], "check": "hardware", "reason": str(exc)})
            except Exception as exc:
                failing = "latency" if statuses["pypto_hardware"] == "PASS" else "pypto_hardware"
                statuses[failing] = "FAIL"
                result["failures"].append({"case": case["name"], "check": failing, "reason": str(exc)})
            result["cases"][case["name"]] = case_result
        result["checks"].update(summarize_checks(result["cases"], execution_checks))
    if subject(op_dir) != candidate_hash:
        raise ContractError("checks changed production artifacts; verifier checks must be read-only")
    if precision_engine is not None:
        identity = precision_policy["engine"]
        if digest(confined(config, identity["path"], must_exist=True)) != identity["sha256"]:
            raise ContractError("Pro precision engine changed during checks")
    decisive = [value for key, value in result["checks"].items() if key not in {"functional", "pipesim"}]
    if "FAIL" in decisive:
        result["verdict"] = "FAIL"
    elif any(status != "PASS" for status in decisive):
        result["verdict"] = "BLOCKED"
    atomic_json(run / "checks.json", result)
    return {"report": str((run / "checks.json").relative_to(op_dir)), "verdict": result["verdict"],
            "checks": result["checks"], "failures": result["failures"]}


def seal(config, op_dir, checks_path, review_path, output, *, recommendation="reject"):
    checks_file = confined(op_dir, checks_path, must_exist=True)
    review_file = confined(op_dir, review_path, must_exist=True)
    checks, review = read_json(checks_file), read_json(review_file)
    candidate = subject(op_dir)
    if checks.get("artifact_hash") != candidate or review.get("artifact_hash") != candidate:
        raise ContractError("checks or semantic review belong to another candidate")
    required = {"formula", "reference_independent", "domain_coverage"}
    if checks["stage"] != "prepare":
        required |= {"single_runtime_kernel", "host_boundary", "dsl_export_correspondence"}
    if not required <= set(review.get("checks", {})):
        raise ContractError(f"semantic review needs: {sorted(required)}")
    semantic_pass = all(review["checks"][key] == "PASS" for key in required)
    if not isinstance(review.get("findings"), list) or not review["findings"]:
        raise ContractError("semantic review must include concrete findings and source locations")
    report = {**checks, "schema": "cannbot.scriptor-report/1", "reviewer": "pypto-pro-scriptor-verifier",
              "source_id": read_json(config / "scriptor-install.json")["source_id"],
              "check_report": checks_path, "review_report": review_path,
              "recommendation": recommendation, "semantic_review": review,
              "evidence": [*checks["evidence"], _evidence(op_dir, checks_file), _evidence(op_dir, review_file)]}
    report["checks"]["semantic_review"] = "PASS" if semantic_pass else "FAIL"
    if not semantic_pass:
        report["verdict"] = "FAIL"
    extra = op_dir / "reports/observations.json"
    if extra.exists():
        observations = read_json(extra)
        if (observations.get("artifact_hash") != candidate or
                review.get("external_metrics_sha256") != digest(extra)):
            raise ContractError("external observations need a current independent review hash")
        evidence = {str(extra.relative_to(op_dir)): _evidence(op_dir, extra)}
        for metric in observations["metrics"]:
            path = confined(op_dir, metric["evidence_file"], must_exist=True)
            source = read_json(path)
            evidence[metric["evidence_file"]] = _evidence(op_dir, path)
            if metric["basis"] == "hardware" and (
                    source.get("check_report") != _evidence(op_dir, checks_file) or not source.get("evidence")):
                raise ContractError("hardware metrics require this check report and raw evidence")
            for item in source.get("evidence", []):
                raw = confined(op_dir, item["path"], must_exist=True)
                if digest(raw) != item["sha256"]:
                    raise ContractError("external metric raw evidence changed")
                evidence[item["path"]] = item
        report["evidence"].extend(evidence.values())
        report["metrics"] = [*report["metrics"], *observations["metrics"]]
    elif review.get("external_metrics_sha256") is not None:
        raise ContractError("review names a missing external observation file")
    report["criteria"] = evaluate_report(config, op_dir, report)
    if report["stage"] == "accept" and report["criteria"]["status"] not in {"PASS", "NOT_REQUESTED"}:
        report["verdict"] = "FAIL" if report["criteria"]["status"] == "FAIL" else "BLOCKED"
    atomic_json(confined(op_dir, output), report)
    return {"report": output, "verdict": report["verdict"], "criteria": report["criteria"]}
