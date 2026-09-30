# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Scriptor state transitions, including direct handoff from Pro development."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import shutil
import time

from .common import (ContractError, atomic_json, confined, digest, finite,
                     json_pointer, load_module, object_digest, read_json)
from .golden_preflight import verify_cpu_golden
from .policy import resolve_policy, stop_reason

WORKFLOW = "pypto-pro-op-scriptor"
STAGES = ("prepare", "implement", "optimize", "accept")
FINAL_FIELDS = ("final_report", "final_report_sha256", "final_report_snapshot", "completed_at", "stop_reason",
                "validation_status", "criteria_status", "outcome", "verdict", "sync_trial")


def spec_module(config):
    return load_module(config / "skills/pypto-pro-intent-understand/scripts/validate_spec.py",
                       "scriptor_spec_validator")


def read_spec(config, op_dir):
    module = spec_module(config)
    contract = module._validate(module._extract((op_dir / "SPEC.md").read_text()))
    if contract.get("perf_target") is not None and contract.get("exit_criteria") is None:
        raise ContractError("a legacy perf_target requires explicit exit_criteria with metric, unit and cases; do not silently drop the user's target")
    return contract


def criteria_module(config):
    return load_module(config / "skills/pypto-pro-intent-understand/scripts/exit_criteria.py",
                       "scriptor_exit_criteria")


def artifact_hashes(op_dir):
    selected = set()
    for pattern in ("SPEC.md", "*_golden.py", "*_golden_cpu.py", "test_*.py",
                    "scriptor/**/*.py", "scriptor/**/*.json", "generated/**/*.py", "generated/**/*.json"):
        selected.update(op_dir.glob(pattern))
    result = {}
    for path in sorted(selected):
        if not path.is_file() or "__pycache__" in path.parts or "build" in path.parts:
            continue
        relative = str(path.relative_to(op_dir))
        confined(op_dir, relative, must_exist=True)
        result[relative] = digest(path)
    return result


def subject(op_dir):
    return object_digest(artifact_hashes(op_dir))


def evaluate_report(config, op_dir, report):
    observations = report.get("metrics", [])
    for row in observations:
        if row.get("artifact_hash") != report["artifact_hash"]:
            raise ContractError("metric belongs to a different candidate")
        source = confined(op_dir, row["evidence_file"], must_exist=True)
        if digest(source) != row["evidence_sha256"]:
            raise ContractError("metric source changed after collection")
        if "pointer" in row:
            value = json_pointer(read_json(source), row["pointer"])
            finite(value, "observed metric")
            if value != row.get("value"):
                raise ContractError("external metric value does not match its JSON pointer")
    return criteria_module(config).evaluate_exit_criteria(
        read_spec(config, op_dir).get("exit_criteria"), observations)


def required_checks(stage, execution_policy=None):
    if execution_policy not in (None, "hardware_first"):
        raise ContractError("unknown execution policy")
    if stage == "prepare":
        return {"spec", "golden_cpu", "golden_npu", "semantic_review"}
    if execution_policy == "hardware_first":
        return {"pypto_hardware", "standalone", "latency", "semantic_review"}
    # Historical reports retain the checks required when they were produced.
    return {"functional", "pipesim", "pypto_hardware", "semantic_review"} | (
        {"standalone", "latency"} if stage == "accept" else set())


def load_report(config, op_dir, relative, stage, *, passing=True, validation_only=False):
    report = read_json(confined(op_dir, relative, must_exist=True))
    if report.get("schema") != "cannbot.scriptor-report/1" or report.get("stage") != stage:
        raise ContractError(f"expected a {stage} verifier report")
    if report.get("reviewer") != "pypto-pro-scriptor-verifier":
        raise ContractError("report requires the independent scriptor verifier")
    if validation_only and (stage != "accept" or not passing):
        raise ContractError("validation-only loading is restricted to final acceptance")
    if passing and not validation_only and report.get("verdict") != "PASS":
        raise ContractError(f"verifier did not pass: {report.get('verdict')}")
    if report.get("artifact_hashes") != artifact_hashes(op_dir) or report.get("artifact_hash") != subject(op_dir):
        raise ContractError("verifier report is stale or belongs to another candidate")
    install = read_json(config / "scriptor-install.json")
    if report.get("source_id") != install["source_id"]:
        raise ContractError("verifier report belongs to another ascriptor source identity")
    for evidence in report.get("evidence", []):
        source = confined(op_dir, evidence["path"], must_exist=True)
        if digest(source) != evidence["sha256"]:
            raise ContractError("verifier evidence changed")
    required = required_checks(stage, report.get("execution_policy"))
    checks = report.get("checks", {})
    if passing and any(checks.get(key) != "PASS" for key in required):
        raise ContractError(f"required checks did not pass: {sorted(required)}")
    if not report.get("evidence"):
        raise ContractError("verifier report has no execution evidence")
    raw = read_json(confined(op_dir, report.get("check_report", ""), must_exist=True))
    if (raw.get("schema") != "cannbot.scriptor-check/1" or raw.get("stage") != stage or
            raw.get("artifact_hash") != report["artifact_hash"]):
        raise ContractError("sealed report has no matching raw adapter check")
    if raw.get("execution_policy") != report.get("execution_policy"):
        raise ContractError("sealed report changed the execution policy")
    if raw.get("sync_mode") != report.get("sync_mode"):
        raise ContractError("sealed report changed the synchronization mode")
    if raw.get("precision_policy") != report.get("precision_policy"):
        raise ContractError("sealed report changed the precision policy")
    if passing and raw.get("verdict") != "PASS":
        raise ContractError("raw execution checks did not pass")
    if any(checks.get(key) != value for key, value in raw.get("checks", {}).items()):
        raise ContractError("sealed report changed an execution verdict")
    native_metrics = raw.get("metrics", [])
    if report.get("metrics", [])[:len(native_metrics)] != native_metrics:
        raise ContractError("sealed report changed an execution metric")
    review = report.get("semantic_review", {})
    if review.get("artifact_hash") != report["artifact_hash"] or not review.get("findings"):
        raise ContractError("sealed report has no current semantic review")
    review_path = report.get("review_report")
    if review_path and read_json(confined(op_dir, review_path, must_exist=True)) != review:
        raise ContractError("sealed report changed its semantic review")
    external_hash = review.get("external_metrics_sha256")
    if external_hash is not None:
        if not review_path:
            raise ContractError("external metrics lack a bound semantic review")
        extra = confined(op_dir, "reports/observations.json", must_exist=True)
        observations = read_json(extra)
        if (digest(extra) != external_hash or observations.get("artifact_hash") != report["artifact_hash"] or
                report.get("metrics", []) != [*native_metrics, *observations.get("metrics", [])]):
            raise ContractError("sealed report changed its reviewed external metrics")
    elif review_path and report.get("metrics", []) != native_metrics:
        raise ContractError("sealed report contains unreviewed external metrics")
    if (validation_only and "cases" in raw) or (passing and stage != "prepare" and report.get("execution_policy") == "hardware_first"):
        cases = raw.get("cases")
        expected_cases = {case["name"] for case in read_spec(config, op_dir)["p0_cases"]}
        if not isinstance(cases, dict) or set(cases) != expected_cases or report.get("cases") != cases:
            raise ContractError("final per-case results do not match the declared cases and raw execution")
        if any(any(row.get("checks", {}).get(key) != "PASS" for key in required - {"semantic_review"})
               for row in cases.values()):
            raise ContractError("final per-case required checks did not pass")
    report["criteria"] = evaluate_report(config, op_dir, report)
    if validation_only:
        expected = {"PASS": "PASS", "NOT_REQUESTED": "PASS", "FAIL": "FAIL", "UNKNOWN": "BLOCKED"}
        if report.get("verdict") != expected.get(report["criteria"]["status"]):
            raise ContractError("final verdict does not match independently recomputed user criteria")
    return report


def _freeze(op_dir, contract):
    names = ["SPEC.md", f"{contract['op_name']}_golden.py", f"{contract['op_name']}_golden_cpu.py"]
    return {name: digest(confined(op_dir, name, must_exist=True)) for name in names}


def _check_frozen(op_dir, state):
    for relative, expected in state.get("frozen", {}).items():
        if digest(confined(op_dir, relative, must_exist=True)) != expected:
            raise ContractError(f"frozen input changed: {relative}; use rollback_prepare before updating the contract")


def _reset_stages(state, stage):
    state["current_stage"] = stage
    state["stages"] = {name: "in_progress" if name == stage else "pending" for name in STAGES
                       if name != "prepare" or state.get("entry_mode") != "from_pro"}


BOOTSTRAP_FILES = ("SPEC.md", "PRO_MATERIAL_INDEX.md", "EXPLORE_REPORT.md",
                   "DESIGN.md", "DESIGN_BINDINGS.json", "module_interfaces.yaml",
                   "GOLDEN_VALIDATION.json")


def _bootstrap_hashes(config, op_dir):
    contract = read_spec(config, op_dir)
    checker = load_module(config / "skills/pypto-pro-golden-generate/scripts/validate_golden_once.py",
                          "scriptor_golden_receipt")
    valid, reason = checker.check_receipt(op_dir, expected_op=contract["op_name"])
    if not valid:
        raise ContractError(f"Pro Golden receipt is missing or stale: {reason}")
    selection = [path for path in op_dir.glob("**/KB_SELECTION.json")
                 if not path.is_relative_to(op_dir / "reports")
                 and not path.is_relative_to(op_dir / ".scriptor")]
    if not selection or (op_dir / "KB_SELECTION.json" in selection and len(selection) != 1):
        raise ContractError("Pro KB selection is missing or has mixed flat/split layouts")
    files = [*BOOTSTRAP_FILES, f"{contract['op_name']}_golden.py",
             f"{contract['op_name']}_golden_cpu.py",
             *(str(path.relative_to(op_dir)) for path in sorted(selection))]
    return {name: digest(confined(op_dir, name, must_exist=True)) for name in files}


def bootstrap_check(config, op_dir):
    """Seal the completed Plan/Golden/Design inputs before prototype authoring."""
    contract = read_spec(config, op_dir)
    if (op_dir / "prototype").exists() or (op_dir / f"test_{contract['op_name']}.py").exists():
        raise ContractError("Pro prototype already exists; bootstrap-check must precede prototype authoring. "
                            "Use bootstrap-restart to archive it before resealing changed inputs")
    state_path = op_dir / ".scriptor/state.json"
    if state_path.exists() and read_json(state_path).get("current_stage") != "upstream":
        raise ContractError("workflow is already initialized; use state status")
    hashes = _bootstrap_hashes(config, op_dir)
    target = op_dir / ".scriptor/receipts/pro-bootstrap.json"
    pending_path = op_dir / ".scriptor/bootstrap-restart.json"
    if target.exists():
        receipt = read_json(target)
        if receipt.get("hashes") != hashes:
            raise ContractError("Pro bootstrap inputs changed after the pre-prototype receipt; "
                                "use bootstrap-restart with a reason and rerun the prototype")
        if pending_path.exists():
            pending = read_json(pending_path)
            if receipt.get("restart") != pending:
                raise ContractError("bootstrap restart is pending; do not reuse the previous receipt")
            _verify_bootstrap_restart(op_dir, pending)
            pending_path.unlink()
    else:
        restart = read_json(pending_path) if pending_path.exists() else None
        if restart is not None:
            _verify_bootstrap_restart(op_dir, restart)
        cpu_probe = verify_cpu_golden(op_dir, contract)
        if (cpu_probe["cpu_sha256"] != hashes[f"{contract['op_name']}_golden_cpu.py"]
                or cpu_probe["spec_sha256"] != hashes["SPEC.md"]
                or _bootstrap_hashes(config, op_dir) != hashes):
            raise ContractError("Pro bootstrap inputs changed during CPU Golden P0 preflight")
        atomic_json(target, {"schema": "cannbot.pro-bootstrap/1", "hashes": hashes,
                             "completed_at": time.time(),
                             "cpu_golden_probe": cpu_probe,
                             **({"restart": restart} if restart else {})})
        pending_path.unlink(missing_ok=True)
    return {"receipt": str(target.relative_to(op_dir)), "receipt_sha256": digest(target),
            "hashes": hashes,
            "cpu_golden_probe": read_json(target).get("cpu_golden_probe"),
            "restart": read_json(target).get("restart")}


def _verify_bootstrap_restart(op_dir, restart):
    if (not isinstance(restart, dict) or restart.get("schema") != "cannbot.pro-bootstrap-restart/1"
            or not str(restart.get("reason", "")).strip()
            or not isinstance(restart.get("archive"), str)
            or not isinstance(restart.get("previous_receipt_sha256"), str)
            or not isinstance(restart.get("had_prototype"), bool)):
        raise ContractError("invalid bootstrap restart record")
    archive = confined(op_dir, restart.get("archive", "") + "/pro-bootstrap.json", must_exist=True)
    if not archive.parent.is_relative_to((op_dir / "reports/bootstrap-restarts").resolve()):
        raise ContractError("bootstrap restart archive must stay under reports/bootstrap-restarts")
    if digest(archive) != restart.get("previous_receipt_sha256"):
        raise ContractError("previous bootstrap receipt changed after archival")
    if read_json(confined(archive.parent, "restart.json", must_exist=True)) != restart:
        raise ContractError("bootstrap restart record differs from its archive")


def bootstrap_restart(config, op_dir, reason):
    """Archive a pre-init seal and prototype before changing its frozen inputs."""
    if not isinstance(reason, str) or not reason.strip():
        raise ContractError("bootstrap-restart requires a concrete reason")
    state_path = op_dir / ".scriptor/state.json"
    if state_path.exists() and read_json(state_path).get("current_stage") != "upstream":
        raise ContractError("workflow is initialized; use rollback_prepare instead")
    target = op_dir / ".scriptor/receipts/pro-bootstrap.json"
    if target.is_symlink() or not target.is_file():
        raise ContractError("bootstrap-restart requires an existing regular bootstrap receipt")
    previous = read_json(target)
    if previous.get("schema") != "cannbot.pro-bootstrap/1" or not isinstance(previous.get("hashes"), dict):
        raise ContractError("bootstrap-restart requires a valid previous receipt")
    if (op_dir / ".scriptor/bootstrap-restart.json").exists():
        raise ContractError("a bootstrap restart is already pending")
    archive = op_dir / "reports/bootstrap-restarts" / str(time.time_ns())
    archive.mkdir(parents=True, exist_ok=False)
    shutil.copy2(target, archive / "pro-bootstrap.json")
    selections = [path for path in op_dir.glob("**/KB_SELECTION.json")
                  if not path.is_relative_to(op_dir / "reports")
                  and not path.is_relative_to(op_dir / ".scriptor")]
    for source in [*(op_dir / relative for relative in BOOTSTRAP_FILES),
                   *op_dir.glob("*_golden.py"), *op_dir.glob("*_golden_cpu.py"), *selections]:
        if source.is_file():
            snapshot = archive / "inputs-before-restart" / source.relative_to(op_dir)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, snapshot)
    sources = [(source, archive / source.name) for source in op_dir.glob("test_*.py")]
    sources += [(op_dir / "prototype", archive / "prototype"),
               (op_dir / "build", archive / "build"),
               (op_dir / "reports/pro-bootstrap.md", archive / "pro-bootstrap.md"),
               (op_dir / "reports/prototype-logs", archive / "prototype-logs")]
    had_prototype = any(source.exists() for source, _ in sources
                        if source.parent == op_dir and (source.name.startswith("test_")
                                                        or source.name in {"prototype", "build"}))
    restart = {"schema": "cannbot.pro-bootstrap-restart/1", "reason": reason.strip(),
               "archive": str(archive.relative_to(op_dir)),
               "previous_receipt_sha256": digest(target), "had_prototype": had_prototype,
               "started_at": time.time()}
    atomic_json(archive / "restart.json", restart)
    moved = []
    pending = op_dir / ".scriptor/bootstrap-restart.json"
    try:
        for source, destination in sources:
            if source.exists():
                shutil.move(str(source), str(destination))
                moved.append((source, destination))
        atomic_json(pending, restart)
        target.unlink()
    except Exception:
        pending.unlink(missing_ok=True)
        for source, destination in reversed(moved):
            shutil.move(str(destination), str(source))
        raise
    return {"status": "RESTART_PENDING", "reason": restart["reason"],
            "archive": restart["archive"], "had_prototype": had_prototype,
            "next": "repair Plan/Golden/Design, self-validate Golden, run bootstrap-check, "
                    "then rerun and report the complete Pro prototype before init"}


def _verify_bootstrap(config, op_dir):
    receipt = read_json(confined(op_dir, ".scriptor/receipts/pro-bootstrap.json", must_exist=True))
    if receipt.get("schema") != "cannbot.pro-bootstrap/1" or receipt.get("hashes") != _bootstrap_hashes(config, op_dir):
        raise ContractError("Pro bootstrap receipt is stale; complete Plan, Golden and Design before the prototype")
    cpu_probe = receipt.get("cpu_golden_probe")
    if cpu_probe is not None:
        contract = read_spec(config, op_dir)
        if (not isinstance(cpu_probe, dict) or cpu_probe.get("status") != "PASS"
                or cpu_probe.get("cases") != len(contract["p0_cases"])
                or cpu_probe.get("cpu_sha256") != digest(op_dir / f"{contract['op_name']}_golden_cpu.py")
                or cpu_probe.get("spec_sha256") != digest(op_dir / "SPEC.md")):
            raise ContractError("CPU Golden P0 shape/dtype preflight receipt is stale")
    if (op_dir / ".scriptor/bootstrap-restart.json").exists():
        raise ContractError("bootstrap restart is still pending")
    restart = receipt.get("restart")
    if restart is not None:
        _verify_bootstrap_restart(op_dir, restart)
        if restart["had_prototype"]:
            report = op_dir / "reports/pro-bootstrap.md"
            if (not report.is_file() or report.stat().st_mtime <= receipt["completed_at"]
                    or f"Bootstrap receipt SHA-256: {digest(op_dir / '.scriptor/receipts/pro-bootstrap.json')}"
                    not in report.read_text(encoding="utf-8")
                    or not (op_dir / "prototype").is_dir()):
                raise ContractError("restarted bootstrap needs a fresh Pro prototype, full rerun "
                                    "and pro-bootstrap.md naming the new receipt before init")


def _enter_from_pro(config, op_dir, state):
    contract = read_spec(config, op_dir)
    _verify_bootstrap(config, op_dir)
    state["frozen"] = _freeze(op_dir, contract)
    state["frozen"]["GOLDEN_VALIDATION.json"] = digest(op_dir / "GOLDEN_VALIDATION.json")
    _reset_stages(state, "implement")
    state.pop("blocker", None)


def _checkpoint(op_dir, name):
    hashes = artifact_hashes(op_dir)
    target = op_dir / ".scriptor/checkpoints" / name
    target.mkdir(parents=True, exist_ok=False)
    for relative in hashes:
        copy = target / relative
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(op_dir / relative, copy)
    atomic_json(target / "checkpoint.json", hashes)
    return {"path": str(target.relative_to(op_dir)), "artifact_hash": object_digest(hashes)}


def restore_checkpoint(op_dir, checkpoint):
    directory = confined(op_dir, checkpoint["path"])
    hashes = read_json(directory / "checkpoint.json")
    if object_digest(hashes) != checkpoint["artifact_hash"]:
        raise ContractError("checkpoint metadata changed")
    for relative, expected in hashes.items():
        if digest(confined(directory, relative, must_exist=True)) != expected:
            raise ContractError(f"checkpoint artifact changed: {relative}")
    for relative in set(artifact_hashes(op_dir)) - set(hashes):
        confined(op_dir, relative).unlink()
    for relative in hashes:
        target = confined(op_dir, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(directory / relative, target)


@contextmanager
def _locked(op_dir):
    directory = op_dir / ".scriptor"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "state.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def _state_identity(state, installation, action):
    # Keep the v2 ledger and its counting rules; extensions are optional.
    if state.get("schema") == "cannbot.scriptor-state/1":
        raise ContractError("this ledger was written against the retired ascriptor bundle installer; "
                            "start a new request against the installed sources")
    if state.get("schema") != "cannbot.scriptor-state/2":
        raise ContractError("unsupported state schema; explicit migration is required")
    if state.get("workflow") != WORKFLOW or (
            state["source_id"] != installation["source_id"] and action not in {"restart_optimization", "migrate_sources"}):
        raise ContractError("workflow or source identity changed; explicit migration is required")


def optimization_stop(state, now=None):
    """Use the same policy for execution and its read-only next-step hint."""
    if not state["optimization"]["enabled"]:
        return "not_requested"
    return stop_reason(state["optimization"], len(state["rounds"]),
                       criteria_met=state.get("last_criteria", {}).get("met", False),
                       elapsed=(time.time() if now is None else now) - state["optimization_started_at"],
                       no_improvement=state["no_improvement"])


def _receipt_key(op_dir, relative):
    report = read_json(confined(op_dir, relative, must_exist=True))
    raw = confined(op_dir, report["check_report"], must_exist=True)
    return object_digest({"path": str(raw.relative_to(op_dir)), "sha256": digest(raw)})


def _archive_report(op_dir, relative):
    """Retain an exact receipt even when a verifier reuses its output filename."""
    source = confined(op_dir, relative, must_exist=True)
    expected = digest(source)
    target = confined(op_dir, f".scriptor/receipts/{expected}.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(source, target)
    if digest(target) != expected:
        raise ContractError("archived verifier report changed")
    return str(target.relative_to(op_dir))


def _duplicate_round_report(config, op_dir, state, relative):
    requested = confined(op_dir, relative, must_exist=True)
    requested_hash = digest(requested)
    report = read_json(requested)
    raw_path = confined(op_dir, report["check_report"], must_exist=True)
    receipt = _receipt_key(op_dir, relative)
    for row in state["rounds"]:
        if not row.get("finished_at") or not row.get("report"):
            continue
        recorded = row.get("report_sha256")
        historical = confined(op_dir, row.get("report_snapshot", row["report"]))
        if not recorded and report.get("artifact_hash") != row.get("artifact_hash"):
            # The v1 ledger itself proves this is a different candidate, even if
            # an old mutable output path no longer contains its former report.
            continue
        if not historical.is_file():
            raise ContractError("settled verifier report is unavailable; duplicate comparison cannot be verified")
        historical_hash = digest(historical)
        if recorded and historical_hash != recorded:
            raise ContractError("settled verifier report changed")
        previous = read_json(historical)
        comparison = recorded or historical_hash
        previous_receipt = row.get("receipt_key") or _receipt_key(op_dir, str(historical.relative_to(op_dir)))
        same_execution = raw_path == confined(op_dir, previous["check_report"])
        if requested_hash != comparison and receipt != previous_receipt and not same_execution:
            continue
        if not recorded:
            # v1 had no report digest. Do not invent a historical integrity claim.
            raise ContractError("legacy v1 report already settled; it cannot settle another round")
        if requested_hash != recorded:
            raise ContractError("execution receipt already settled with a different verifier report")
        if report.get("artifact_hash") != row.get("artifact_hash"):
            raise ContractError("settled verifier report identity changed")
        for evidence in report.get("evidence", []):
            if digest(confined(op_dir, evidence["path"], must_exist=True)) != evidence["sha256"]:
                raise ContractError("settled verifier evidence changed")
        evaluate_report(config, op_dir, report)
        return True
    return False


def _accept_outcome(state, report):
    status = report["criteria"]["status"]
    if status not in {"PASS", "NOT_REQUESTED"}:
        stopped = next((event for event in reversed(state["history"])
                        if event["action"] == "finish_optimization"), None)
        if not stopped:
            raise ContractError("final outcome requires a recorded legal optimization stop")
        historical = stopped.get("criteria", {"met": state.get("stop_reason") == "criteria_met"})
        stop_state = {**state, "last_criteria": historical}
        expected = optimization_stop(stop_state, stopped["time"])
        if expected is None and stopped.get("reason") == "user_stop":
            expected = "user_stop"
        if not expected or state.get("stop_reason") != expected:
            raise ContractError("unmet or unknown user criteria require a recorded legal optimization stop")
    return {"validation_status": "PASS", "criteria_status": status,
            "verdict": report["verdict"],
            "outcome": {"PASS": "accepted", "NOT_REQUESTED": "accepted",
                        "FAIL": "target_unmet", "UNKNOWN": "target_unknown"}[status]}


def verify_sync_trial(config, op_dir, state):
    from .exporter import verify_export
    selected = state.get("delivery_sync_mode")
    if selected is not None:
        if selected not in {"auto_mutex", "manual"}:
            raise ContractError("invalid recorded delivery synchronization mode")
        if selected == "manual" and state.get("manual_requested_by_user") is not True:
            raise ContractError("manual delivery requires an explicit user request")
        if state.get("sync_trial"):
            raise ContractError("fixed delivery synchronization mode cannot use the legacy closeout trial")
        if verify_export(config, op_dir).get("sync_mode") != selected:
            raise ContractError("final export differs from the user-authorized delivery synchronization mode")
        return
    trial = state.get("sync_trial", {})
    if trial.get("status") not in {"passed", "failed", "blocked"}:
        raise ContractError("complete and report the auto_mutex closeout trial before final acceptance")
    if trial.get("selected_mode") != verify_export(config, op_dir).get("sync_mode", "manual"):
        raise ContractError("final synchronization mode differs from the recorded trial selection")
    evidence_records = list(trial.get("evidence", []))
    if trial.get("report_snapshot"):
        snapshot = confined(op_dir, trial["report_snapshot"], must_exist=True)
        if digest(snapshot) != trial["report_sha256"]:
            raise ContractError("synchronization trial report changed")
        evidence_records.extend(read_json(snapshot).get("evidence", []))
    elif not evidence_records:
        raise ContractError("synchronization trial has no recorded evidence")
    for evidence in evidence_records:
        if digest(confined(op_dir, evidence["path"], must_exist=True)) != evidence["sha256"]:
            raise ContractError("synchronization trial diagnostics changed")


def _require_delivery_mode(op_dir, state, report):
    selected = state.get("delivery_sync_mode")
    if selected is not None and report.get("sync_mode") != selected:
        raise ContractError(f"verifier report sync_mode must be {selected} for this delivery")
    if selected is not None:
        from .exporter import verify_recorded_emissions
        verify_recorded_emissions(op_dir, selected)


def transition(config, project, request, *, actor="build"):
    if actor not in {"build", WORKFLOW, "pypto-pro-op-orchestrator"}:
        raise ContractError("only the primary agent can transition scriptor state")
    if not isinstance(request, dict) or not all(isinstance(request.get(key), str) and request[key] for key in ("action", "opDir")):
        raise ContractError("every state request requires action and opDir (custom/<op>), including later transitions")
    op_dir = confined(project, request["opDir"])
    if not op_dir.is_relative_to((project / "custom").resolve()):
        raise ContractError("operator directory must be under the project's custom directory")
    if (op_dir / ".orchestrator_state.json").exists():
        raise ContractError("operator already belongs to the original PyPTO-Pro workflow")
    action = request["action"]
    state_path = op_dir / ".scriptor/state.json"
    if action == "status":
        if not state_path.is_file():
            raise ContractError("initialize the workflow before transitioning")
        state = read_json(state_path)
        _state_identity(state, read_json(config / "scriptor-install.json"), action)
        return state
    op_dir.mkdir(parents=True, exist_ok=True)
    with _locked(op_dir):
        installation = read_json(config / "scriptor-install.json")
        now = time.time()
        delivery_mode = None
        if action == "init":
            entry = request.get("entry_mode", "fresh")
            if entry not in {"fresh", "optimize_existing", "from_pro"}:
                raise ContractError("unknown workflow entry mode")
            state = read_json(state_path) if state_path.exists() else None
            if state is not None:
                _state_identity(state, installation, action)
                if (entry != "from_pro" or state.get("entry_mode") != "from_pro"
                        or state["current_stage"] != "upstream"):
                    raise ContractError("workflow already initialized; use status or resume its current stage")
                if ("delivery_sync_mode" in state and request.get("delivery_sync_mode", state["delivery_sync_mode"])
                        != state["delivery_sync_mode"]):
                    raise ContractError("resuming Pro must preserve its recorded delivery synchronization mode")
            if entry == "from_pro" and state is None:
                delivery_mode = request.get("delivery_sync_mode", "auto_mutex")
                if delivery_mode not in {"auto_mutex", "manual"}:
                    raise ContractError("delivery_sync_mode must be auto_mutex or manual")
                if delivery_mode == "manual" and request.get("manual_requested_by_user") is not True:
                    raise ContractError("manual delivery requires an explicit user request")
                if delivery_mode == "auto_mutex" and request.get("manual_requested_by_user") not in (None, False):
                    raise ContractError("manual_requested_by_user applies only to manual delivery")
            project_policy = read_json(config / "scriptor.json") if (config / "scriptor.json").exists() else {}
            overrides = dict(request.get("optimization", {}))
            if entry == "optimize_existing":
                overrides["enabled"] = True
            policy = resolve_policy(overrides, state["optimization"] if state else project_policy.get("optimization", {}),
                                    from_pro=entry == "from_pro")
            state = state or {"schema": "cannbot.scriptor-state/2", "workflow": WORKFLOW,
                     "source_id": installation["source_id"], "entry_mode": entry,
                     "current_stage": "prepare", "stages": {s: "pending" for s in STAGES},
                     "optimization": policy,
                     "rounds": [], "round_active": False, "no_improvement": 0,
                     "history": [], "created_at": now}
            state["optimization"] = policy
            if entry == "from_pro":
                if delivery_mode is not None:
                    state["delivery_sync_mode"] = delivery_mode
                    state["manual_requested_by_user"] = delivery_mode == "manual"
                _enter_from_pro(config, op_dir, state)
            else:
                state["stages"]["prepare"] = "in_progress"
        else:
            if not state_path.exists():
                raise ContractError("initialize the workflow before transitioning")
            state = read_json(state_path)
            _state_identity(state, installation, action)
            if action not in {"rollback_prepare", "restart_optimization", "migrate_sources"}:
                _check_frozen(op_dir, state)
            current = state["current_stage"]
            if current == "upstream" and action not in {"blocked", "update_optimization", "migrate_sources"}:
                raise ContractError("finish Pro development, then init with entry_mode=from_pro")
            if action == "complete_prepare":
                if current != "prepare":
                    raise ContractError("prepare is not the current stage")
                load_report(config, op_dir, request["report"], "prepare")
                state["frozen"] = _freeze(op_dir, read_spec(config, op_dir))
                state["stages"]["prepare"] = "completed"
                state["current_stage"] = "implement"
                state["stages"]["implement"] = "in_progress"
            elif action == "complete_implement":
                if current != "implement":
                    raise ContractError("implement is not the current stage")
                report = load_report(config, op_dir, request["report"], "candidate")
                _require_delivery_mode(op_dir, state, report)
                snapshot = _archive_report(op_dir, request["report"])
                state["best"] = _checkpoint(op_dir, f"baseline-{len(state['history'])}")
                state["best_report"] = snapshot
                state["last_criteria"] = report["criteria"]
                state["stages"]["implement"] = "completed"
                state["current_stage"] = "optimize"
                state["stages"]["optimize"] = "in_progress"
                state["optimization_started_at"] = now
            elif action == "begin_round":
                if current != "optimize" or not state["optimization"]["enabled"] or state["round_active"]:
                    raise ContractError("cannot begin an optimization round in this state")
                reason = optimization_stop(state, now)
                if reason:
                    raise ContractError(f"optimization must finish: {reason}")
                state["rounds"].append({"number": len(state["rounds"]) + 1, "started_at": now,
                                        "before_hash": subject(op_dir)})
                state["round_active"] = True
            elif action == "finish_round":
                if _duplicate_round_report(config, op_dir, state, request["report"]):
                    return state
                if current != "optimize" or not state["round_active"]:
                    raise ContractError("no optimization round is active")
                report = load_report(config, op_dir, request["report"], "candidate", passing=False)
                _require_delivery_mode(op_dir, state, report)
                passed = report["verdict"] == "PASS"
                if passed:
                    load_report(config, op_dir, request["report"], "candidate")
                snapshot = _archive_report(op_dir, request["report"])
                keep = passed and (report.get("recommendation") == "keep" or
                                   (report["criteria"]["met"] and state["optimization"].get("stop_on_criteria", True)))
                if keep:
                    state["best"] = _checkpoint(op_dir, f"round-{len(state['history'])}-{len(state['rounds'])}")
                    state["best_report"] = snapshot
                    state["no_improvement"] = 0
                else:
                    state["no_improvement"] += 1
                state["last_criteria"] = report["criteria"] if passed else {"met": False, "status": "FAIL"}
                state["rounds"][-1].update(report=request["report"], verdict=report["verdict"],
                                          accepted=keep, artifact_hash=report["artifact_hash"], finished_at=now,
                                          report_snapshot=snapshot,
                                          report_sha256=digest(op_dir / snapshot),
                                          receipt_key=_receipt_key(op_dir, request["report"]))
                state["round_active"] = False
            elif action == "select_best":
                if current != "optimize" or state["round_active"]:
                    raise ContractError("select a measured candidate only between optimization rounds")
                if not str(request.get("reason", "")).strip():
                    raise ContractError("candidate selection requires the ranking reason")
                selected_path = confined(op_dir, request["report"], must_exist=True)
                selected_sha = digest(selected_path)
                if not any(row.get("finished_at") and row.get("verdict") == "PASS"
                           and row.get("report_sha256") == selected_sha for row in state["rounds"]):
                    raise ContractError("candidate selection requires an unchanged settled PASS report")
                # Restore the candidate and its bound observations before this action.
                # Revalidate all evidence; selecting it does not create another execution
                # receipt, rewrite the earlier keep/reject decision, or consume a round.
                report = load_report(config, op_dir, request["report"], "candidate")
                _require_delivery_mode(op_dir, state, report)
                snapshot = _archive_report(op_dir, request["report"])
                state["best"] = _checkpoint(op_dir, f"selected-{len(state['history'])}")
                state["best_report"] = snapshot
                state["last_criteria"] = report["criteria"]
                state["no_improvement"] = 0
            elif action == "finish_optimization":
                if current != "optimize" or state["round_active"]:
                    raise ContractError("optimization is not ready to finish")
                reason = optimization_stop(state, now)
                if not reason and request.get("reason") == "user_stop":
                    reason = "user_stop"
                if not reason:
                    raise ContractError("no active optimization exit condition is satisfied")
                restore_checkpoint(op_dir, state["best"])
                state["stop_reason"] = reason
                state["stages"]["optimize"] = "skipped" if reason == "not_requested" else "completed"
                state["current_stage"] = "accept"
                state["stages"]["accept"] = "in_progress"
            elif action == "begin_sync_trial":
                from .exporter import source_hashes, verify_export
                if state.get("delivery_sync_mode") is not None:
                    raise ContractError("new delivery policy fixes the selected mode; do not generate a second synchronization mode")
                if current != "accept" or state.get("sync_trial"):
                    raise ContractError("the single synchronization trial starts once during acceptance")
                if subject(op_dir) != state["best"]["artifact_hash"]:
                    raise ContractError("start the synchronization trial from the selected manual checkpoint")
                if verify_export(config, op_dir).get("sync_mode", "manual") != "manual":
                    raise ContractError("start the synchronization trial from manual emission")
                state["sync_trial"] = {"status": "in_progress", "started_at": now,
                    "manual_checkpoint": dict(state["best"]), "source_hashes": source_hashes(op_dir)}
            elif action == "complete_sync_trial":
                from .exporter import source_hashes, verify_export
                trial = state.get("sync_trial", {})
                if current != "accept" or trial.get("status") != "in_progress":
                    raise ContractError("no synchronization trial is active")
                if source_hashes(op_dir) != trial["source_hashes"]:
                    raise ContractError("the synchronization trial must preserve the selected DSL and contract")
                report = load_report(config, op_dir, request["report"], "candidate", passing=False)
                if report.get("execution_policy") != "hardware_first" or report.get("sync_mode") != "auto_mutex":
                    raise ContractError("the trial requires an auto_mutex hardware report")
                passed = report["verdict"] == "PASS"
                if passed:
                    load_report(config, op_dir, request["report"], "candidate")
                    if verify_export(config, op_dir).get("sync_mode") != "auto_mutex":
                        raise ContractError("the passing trial must describe auto_mutex emission")
                snapshot = _archive_report(op_dir, request["report"])
                # Preserve existing user criteria; do not invent a relative-speedup threshold.
                keep = passed and report["criteria"]["status"] in {"PASS", "NOT_REQUESTED"}
                if keep:
                    reason = "auto_mutex passed hardware checks and declared acceptance criteria"
                elif passed:
                    reason = f"auto_mutex hardware checks passed, but declared criteria are {report['criteria']['status']}; retained manual"
                else:
                    failures = report.get("failures", [])
                    details = "; ".join(f"{row.get('case', '')}/{row.get('check', '')}: {row.get('reason', row.get('exit_code', 'failed'))}"
                                        for row in failures[:3])
                    details = details or ", ".join(f"{key}={value}" for key, value in report["checks"].items()
                                                   if value in {"FAIL", "BLOCKED"}) or report["verdict"]
                    reason = f"auto_mutex unavailable: {details}; retained manual"
                status = "passed" if keep else "blocked" if report["verdict"] == "BLOCKED" or report["criteria"]["status"] == "UNKNOWN" else "failed"
                trial.update(status=status, finished_at=now,
                             selected_mode="auto_mutex" if keep else "manual", reason=reason,
                             verdict=report["verdict"], criteria=report["criteria"], failures=report.get("failures", []),
                             report_snapshot=snapshot, report_sha256=digest(op_dir / snapshot),
                             artifact_hash=report["artifact_hash"])
                if keep:
                    state["best"] = _checkpoint(op_dir, f"sync-{len(state['history'])}")
                    state["best_report"] = snapshot
                    state["last_criteria"] = report["criteria"]
                else:
                    restore_checkpoint(op_dir, trial["manual_checkpoint"])
            elif action == "abort_sync_trial":
                trial = state.get("sync_trial", {})
                if current != "accept" or trial.get("status") != "in_progress":
                    raise ContractError("no synchronization trial is active")
                status = request.get("status", "failed")
                if status not in {"failed", "blocked"} or not str(request.get("reason", "")).strip():
                    raise ContractError("record the native trial failure or blocker and its actual reason")
                paths = request.get("evidence")
                if not isinstance(paths, list) or not paths or any(not isinstance(p, str) for p in paths):
                    raise ContractError("a failed emission or blocked trial requires actual diagnostic files")
                evidence = []
                for relative in paths:
                    source = confined(op_dir, relative, must_exist=True)
                    if not source.is_relative_to(op_dir / "reports") or not source.is_file() or not source.stat().st_size:
                        raise ContractError("trial diagnostics must be nonempty files under reports")
                    evidence.append({"path": relative, "sha256": digest(source)})
                trial.update(status=status, finished_at=now, selected_mode="manual", reason=request["reason"],
                             evidence=evidence, artifact_hash=subject(op_dir))
                restore_checkpoint(op_dir, trial["manual_checkpoint"])
            elif action == "complete_accept":
                if current != "accept":
                    raise ContractError("accept is not the current stage")
                report = load_report(config, op_dir, request["report"], "accept", validation_only=True)
                _require_delivery_mode(op_dir, state, report)
                if state.get("delivery_sync_mode") is not None or report.get("execution_policy") == "hardware_first":
                    verify_sync_trial(config, op_dir, state)
                if report["artifact_hash"] != state["best"]["artifact_hash"]:
                    raise ContractError("final report does not describe the selected checkpoint")
                state.update(_accept_outcome(state, report))
                state["stages"]["accept"] = "completed"
                state["current_stage"] = "done"
                state["final_report"] = request["report"]
                state["final_report_snapshot"] = _archive_report(op_dir, request["report"])
                state["final_report_sha256"] = digest(op_dir / state["final_report_snapshot"])
                state["completed_at"] = now
            elif action == "update_optimization":
                if current not in {"upstream", "prepare", "implement", "optimize"}:
                    raise ContractError("start a new optimization request after finalization")
                if not str(request.get("reason", "")).strip():
                    raise ContractError("record the user's prompt change when updating a budget")
                state["optimization"] = resolve_policy(request.get("optimization", {}), state["optimization"])
            elif action == "migrate_sources":
                if (request.get("from_source_id") != state["source_id"] or
                        request.get("to_source_id") != installation["source_id"]):
                    raise ContractError("source migration requires the exact current and installed target identities")
                if state["source_id"] == installation["source_id"]:
                    raise ContractError("source migration requires a different installed source identity")
                if state["round_active"]:
                    raise ContractError("finish the active optimization round before migrating sources")
                if not str(request.get("reason", "")).strip():
                    raise ContractError("source migration requires the user's upgrade reason")
                if state.get("entry_mode") == "from_pro":
                    _check_frozen(op_dir, state)
                previous = json.loads(json.dumps({k: v for k, v in state.items() if k != "source_migrations"}))
                state.setdefault("source_migrations", []).append({
                    "from_source_id": state["source_id"], "to_source_id": installation["source_id"],
                    "time": now, "reason": request["reason"], "previous_state": previous})
                state.update(source_id=installation["source_id"], frozen={}, current_stage="prepare",
                             last_criteria={"met": False})
                _reset_stages(state, "upstream" if current == "upstream" else "prepare")
                if state.get("entry_mode") == "from_pro" and current != "upstream":
                    _enter_from_pro(config, op_dir, state)
                for key in ("best", "best_report", "blocker", *FINAL_FIELDS):
                    state.pop(key, None)
            elif action == "restart_optimization":
                if current not in {"done", "accept"} or state["round_active"]:
                    raise ContractError("only a finalized optimization may start a new request")
                if not str(request.get("reason", "")).strip():
                    raise ContractError("a new optimization request requires its user-provided reason")
                from_pro = state.get("entry_mode") == "from_pro"
                if from_pro:
                    _check_frozen(op_dir, state)
                state.setdefault("prior_runs", []).append({key: state.get(key) for key in (
                    "source_id", "rounds", "best", "best_report", "frozen", *FINAL_FIELDS)})
                overrides = dict(request.get("optimization", {}))
                overrides["enabled"] = True
                project_policy = read_json(config / "scriptor.json") if (config / "scriptor.json").exists() else {}
                state["optimization"] = resolve_policy(overrides, project_policy.get("optimization", {}), from_pro=from_pro)
                state.update(source_id=installation["source_id"], entry_mode="from_pro" if from_pro else "optimize_existing", frozen={},
                             rounds=[], round_active=False, no_improvement=0, current_stage="prepare",
                             last_criteria={"met":False})
                _reset_stages(state, "prepare")
                if from_pro:
                    _enter_from_pro(config, op_dir, state)
                for key in ("best", "best_report", "blocker", *FINAL_FIELDS):
                    state.pop(key, None)
            elif action == "rollback_prepare":
                if not str(request.get("reason", "")).strip():
                    raise ContractError("rollback requires an explicit reason")
                state["frozen"] = {}
                _reset_stages(state, "upstream" if state.get("entry_mode") == "from_pro" else "prepare")
                state["round_active"] = False
                state["last_criteria"] = {"met": False}
                for key in ("best", "best_report", "blocker", *FINAL_FIELDS):
                    state.pop(key, None)
            elif action == "blocked":
                if not str(request.get("reason", "")).strip():
                    raise ContractError("blocked requires the actual external blocker")
                state["blocker"] = request["reason"]
            else:
                raise ContractError(f"unknown state action: {action}")
        state["history"].append({"action": action, "time": now,
                                 "report": request.get("report"), "reason": request.get("reason"),
                                 **({"criteria": state["last_criteria"]} if action == "finish_optimization" else {})})
        state["updated_at"] = now
        atomic_json(state_path, state)
        return state
