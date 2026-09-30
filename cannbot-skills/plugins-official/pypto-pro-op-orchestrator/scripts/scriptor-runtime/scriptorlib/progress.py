# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compact next steps and final views derived from the workflow and its evidence."""
from __future__ import annotations

import csv
import io
import json
import os
import shlex
import tempfile

from .common import ContractError, atomic_json, confined, digest, object_digest, read_json


def state_summary(config, project, op_dir, state):
    from .workflow import optimization_stop

    stage = state["current_stage"]
    active = state["rounds"][-1] if state["round_active"] else None
    installation = read_json(config / "scriptor-install.json")
    cli = [installation["python"], str(config / "scriptor/scripts/scriptor.py"), "--config-root", str(config)]
    relative = str(op_dir.relative_to(project))
    result = {"current_stage": stage, "optimization": state["optimization"],
              "rounds_used": len(state["rounds"]), "round_active": state["round_active"],
              "active_round": active["number"] if active else None,
              "best_checkpoint": state.get("best", {}).get("path"),
              "criteria_status": state.get("criteria_status", state.get("last_criteria", {}).get("status")),
              "criteria_from": "accept" if "criteria_status" in state else "last_candidate",
              "state_path": str(op_dir / ".scriptor/state.json")}
    if "delivery_sync_mode" in state:
        result["delivery_sync_mode"] = state["delivery_sync_mode"]
    for key in ("entry_mode", "validation_status", "stop_reason", "outcome", "verdict", "final_report"):
        if key in state:
            result[key] = state[key]
    if stage == "upstream":
        request = {"action": "init", "entry_mode": "from_pro", "opDir": relative}
        result.update(next_action="init", next_request=request,
                      requires=f"Resume Pro development using {relative}/reports/pro-bootstrap.md, then hand off; no prepare verifier.",
                      next_command=shlex.join(cli + ["state", "--project", str(project), "--request-json", json.dumps(request)]))
        return result
    if stage == "done":
        final = confined(op_dir, "reports/final/final.md")
        result["default_report"] = "reports/final/final.md"
        result["delivery_note"] = ("This is completed Scriptor acceptance. Verify the selected candidate and final report, "
                                   "then run delivery-check against the final delivery/<op>/ package and custom/<op>/ work area.")
        if _current_views(op_dir, state):
            result.update(next_action="read_final", next_command=shlex.join(["cat", str(final)]))
        else:
            result.update(next_action="report", next_command=shlex.join(cli + ["report", "--op-dir", str(op_dir)]))
        if "outcome" not in state:
            result["note"] = "Legacy completion: read the regenerated report for revalidated outcomes; status does not migrate history."
        return result

    report_stage = None
    if stage == "accept" and state.get("delivery_sync_mode") is None and (op_dir / "generated/export.json").is_file():
        trial = state.get("sync_trial", {})
        if not trial:
            result.update(next_action="begin_sync_trial", next_request={"action": "begin_sync_trial", "opDir": relative},
                          requires="Preserve the validated manual checkpoint, then export with --sync-mode auto_mutex and check it on hardware.")
            result["next_command"] = shlex.join(cli + ["state", "--project", str(project), "--request-json",
                                                       json.dumps(result["next_request"], separators=(",", ":"))])
            return result
        if trial.get("status") == "in_progress":
            result.update(next_action="complete_sync_trial", next_request={"action": "complete_sync_trial", "opDir": relative,
                          "report": "reports/sealed/sync-trial.json"},
                          requires="Independent native candidate report; for an emission failure, abort_sync_trial with the reason and diagnostic paths.")
            result["next_command"] = shlex.join(cli + ["state", "--project", str(project), "--request-json",
                                                       json.dumps(result["next_request"], separators=(",", ":"))])
            return result
    if stage == "optimize" and not active:
        reason = optimization_stop(state)
        action = "finish_optimization" if reason else "begin_round"
        if reason:
            result["stop_condition"] = reason
    else:
        action, report_stage = {"prepare": ("complete_prepare", "prepare"),
                                "implement": ("complete_implement", "candidate"),
                                "optimize": ("finish_round", "candidate"),
                                "accept": ("complete_accept", "accept")}[stage]
    request = {"action": action, "opDir": relative}
    if report_stage:
        # This is the conventional example path, not a claim that verification ran.
        request["report"] = f"reports/sealed/{report_stage}.json"
        result["requires"] = f"Current independent {report_stage} verifier report; use the verifier's returned path in report."
    result.update(next_action=action, next_request=request,
                  next_command=shlex.join(cli + ["state", "--project", str(project), "--request-json",
                                                json.dumps(request, separators=(",", ":"), ensure_ascii=False)]))
    return result


def _current_views(op_dir, state):
    """Do not direct an agent to an older or partially regenerated summary."""
    try:
        index = read_json(confined(op_dir, "reports/final/final.json", must_exist=True))
        if not isinstance(index, dict):
            return False
        return (index.get("state_sha256") == object_digest(state) and
                all(digest(confined(op_dir, f"reports/final/{name}", must_exist=True)) == index["views"][name]
                    for name in ("final.md", "final.csv")))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_final(config, op_dir, state=None):
    """Regenerate views under the state lock without changing an acceptance fact."""
    from .workflow import _locked, _state_identity

    ledger = op_dir / ".scriptor/state.json"
    if not ledger.is_file():
        raise ContractError("initialize the workflow before generating final results")
    with _locked(op_dir):
        current = read_json(ledger)
        _state_identity(current, read_json(config / "scriptor-install.json"), "report")
        if state is not None and current != state:
            raise ContractError("workflow changed before final results could be generated; read status")
        return _write_final(config, op_dir, current)


def _write_final(config, op_dir, state):
    from .workflow import _accept_outcome, _check_frozen, criteria_module, load_report, read_spec, required_checks, verify_sync_trial
    if state["current_stage"] != "done":
        raise ContractError("final results require a completed acceptance transition")
    _check_frozen(op_dir, state)
    report_path = confined(op_dir, state["final_report"], must_exist=True)
    report_sha256 = digest(report_path)
    if state.get("final_report_sha256") and report_sha256 != state["final_report_sha256"]:
        raise ContractError("the completed verifier report changed")
    report = load_report(config, op_dir, state["final_report"], "accept", validation_only=True)
    if report.get("execution_policy") == "hardware_first":
        verify_sync_trial(config, op_dir, state)
    if state["best"]["artifact_hash"] != report["artifact_hash"]:
        raise ContractError("final report does not describe the selected checkpoint")
    final_state = {**state, **_accept_outcome(state, report)}
    contract = read_spec(config, op_dir)
    cases = report.get("cases", {})
    if cases and set(cases) != {row["name"] for row in contract["p0_cases"]}:
        raise ContractError("final per-case results do not cover exactly the declared cases")
    required = required_checks("accept", report.get("execution_policy")) - {"semantic_review"}
    if any(any(row.get("checks", {}).get(key) != "PASS" for key in required) for row in cases.values()):
        raise ContractError("final per-case validation is incomplete")
    leaves = criteria_module(config).validate_exit_criteria(contract.get("exit_criteria"))
    evaluated = {row["id"]: row for row in report["criteria"]["conditions"]}
    rows = []
    for case in contract["p0_cases"]:
        metrics = [row for row in report.get("metrics", []) if row["case"] == case["name"]]
        latency = [row["value"] for row in metrics if row["metric"] == "latency_us"
                   and row["unit"] == "us" and row["basis"] == "hardware"]
        legacy_min = [row["value"] for row in metrics if row["metric"] == "latency_us"
                      and row["source"] == "msprof.Task Duration(us).min_after_warmup"
                      and row["unit"] == "us" and row["basis"] == "hardware"]
        latency = legacy_min or latency
        conditions = [{**leaf, "result": evaluated[leaf["id"]]} for leaf in leaves if case["name"] in leaf["cases"]]
        rows.append({"case": case["name"], "input_shapes": case["input_shapes"],
                     "output_shapes": case["output_shapes"], "params": case["params"],
                     "dtypes": {
                         **case.get("input_dtypes", {item["name"]: item["dtype"] for item in contract["inputs"]}),
                         **case.get("output_dtypes", {item["name"]: item["dtype"] for item in contract["outputs"]}),
                     },
                     "checks": cases.get(case["name"], {}).get("checks", {key: "UNKNOWN" for key in required}),
                     "latency_us": latency[0] if len(latency) == 1 else None,
                     "criteria": conditions, "metrics": metrics})
    started = next((event["time"] for event in reversed(state["history"])
                    if event["action"] in {"init", "restart_optimization"}), state["created_at"])
    result = {"schema": "cannbot.scriptor-final/1",
              "validation_status": final_state["validation_status"], "criteria_status": final_state["criteria_status"],
              "stop_reason": state["stop_reason"], "outcome": final_state["outcome"], "verdict": report["verdict"],
              "candidate": state["best"], "source_id": state["source_id"],
              "optimization": state["optimization"], "rounds_used": len(state["rounds"]),
              "synchronization": (
                  {"status": "fixed_delivery_policy", "selected_mode": state["delivery_sync_mode"],
                   "manual_requested_by_user": state.get("manual_requested_by_user", False)}
                  if "delivery_sync_mode" in state else state.get("sync_trial", {"status": "legacy_not_recorded"})),
              "execution_policy": report.get("execution_policy", "legacy_simulation_first"),
              "diagnostics": report.get("diagnostics", {}),
              "criteria": report["criteria"], "criteria_tree": contract.get("exit_criteria"),
              "coverage": {"declared": len(rows), "verified": len(cases),
                           "per_case_details": "available" if cases else "unavailable_in_legacy_report"}, "cases": rows,
              "time": {"workflow_wall_time_s": state.get("completed_at", state.get("updated_at", started)) - started,
                       "author_wall_time_s": None, "stage_durations_s": None,
                       "limitations": "No host lifecycle events or execution-phase duration events were collected."},
              "evidence": [{"path": state["final_report"], "sha256": report_sha256}, *report["evidence"]],
              "fixture_evidence": bool(report.get("fixture_evidence", False)),
              "state_compatibility": "v1_optional_extensions",
              "report_integrity": "completion_digest_verified" if state.get("final_report_sha256")
                                  else "legacy_v1_current_evidence_revalidated"}
    destination = confined(op_dir, "reports/final")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=["case", "input_shapes", "output_shapes", "dtypes", "params",
                                               "functional", "pipesim", "pypto_hardware", "standalone",
                                               "latency_us", "criteria"])
    writer.writeheader()
    for row in rows:
        writer.writerow({"case": row["case"], "latency_us": row["latency_us"],
                         **{key: row["checks"].get(key, "NOT_RUN") for key in ("functional", "pipesim", "pypto_hardware", "standalone")},
                         **{key: json.dumps(row[key], ensure_ascii=False, sort_keys=True)
                            for key in ("input_shapes", "output_shapes", "dtypes", "params", "criteria")}})
    csv_text = output.getvalue()
    limit = state["optimization"]["max_iterations"]
    budget = f"{len(state['rounds'])}/{limit if limit is not None else 'unlimited'}"
    lines = [f"# {contract['op_name']} result", "",
             f"Validation: {final_state['validation_status']}; criteria: {final_state['criteria_status']}; "
             f"outcome: {final_state['outcome']}; overall verdict: {report['verdict']}.", "",
             f"Stop reason: {state['stop_reason']}; optimization rounds: {budget}.", "",
             f"Candidate: `{state['best']['artifact_hash']}`; DSL: `scriptor/`; Scriptor export: `generated/`.",
             "Final user delivery is the separate `delivery/<op>/` package after Scriptor acceptance and delivery-check; `custom/<op>/` keeps work evidence.", "",
             f"Verifier report: [{state['final_report']}]({os.path.relpath(report_path, destination)}).", ""]
    if state.get("sync_trial"):
        trial = state["sync_trial"]
        lines.extend([f"Synchronization: {trial['selected_mode']}; auto_mutex trial: {trial['status']}.",
                      f"Reason: {_cell(trial['reason'])}", ""])
        evidence = trial.get("evidence", [])
        if trial.get("report_snapshot"):
            evidence = [{"path": trial["report_snapshot"], "sha256": trial["report_sha256"]}]
        result["evidence"].extend(evidence)
        lines.extend([f"Trial evidence: [{_cell(row['path'])}]({os.path.relpath(op_dir / row['path'], destination)})."
                      for row in evidence] + [""])
    elif "delivery_sync_mode" in state:
        lines.extend([f"Synchronization: {state['delivery_sync_mode']} (fixed delivery policy).",
                      "Manual selected by explicit user request." if state.get("manual_requested_by_user")
                      else "Auto mutex is the default; no manual variant was generated.", ""])
    if report.get("diagnostics"):
        lines.extend(["Full-case functional/pipesim execution: NOT_RUN; reduced diagnostic probes are separate evidence.", ""])
    if result["fixture_evidence"]:
        lines.extend(["**DETERMINISTIC PROTOCOL FIXTURE: no accelerator was executed.**", ""])
    lines.extend(["| Case | Functional | Hardware | Latency (us) | User conditions |",
                  "|---|---|---|---:|---|"])
    for row in rows:
        conditions = "; ".join(f"{c['id']}: {c['result']['status']} ({c['aggregation']}, scope={','.join(c['cases'])})"
                                for c in row["criteria"]) or "NOT_REQUESTED"
        lines.append(f"| {row['case']} | {row['checks'].get('functional', 'NOT_RUN')} | {row['checks']['pypto_hardware']} | "
                     f"{row['latency_us'] if row['latency_us'] is not None else 'UNKNOWN'} | {conditions} |")
    if leaves:
        lines.extend(["", f"User condition logic: `{_condition_logic(contract['exit_criteria'])}`.", "",
                      "| Condition / metric | Case scope / aggregation | Observed | Target | Result | Basis / source |",
                      "|---|---|---|---|---|---|"])
        for leaf in leaves:
            tested = evaluated[leaf["id"]]
            values = tested.get("values", [])
            observed = ", ".join(f"{case}={value:g}" for case, value in zip(leaf["cases"], values))
            if tested["status"] == "UNKNOWN":
                observed = "Missing/ambiguous: " + ", ".join(tested["missing_or_ambiguous_cases"])
            elif leaf["aggregation"] != "each":
                observed += f"; {leaf['aggregation']}={tested['tested_values'][0]:g}"
            target = f"{leaf['operator']} {leaf['threshold']} {leaf['unit']}"
            cells = [f"{leaf['id']} / {leaf['metric']}", f"{', '.join(leaf['cases'])} / {leaf['aggregation']}",
                     observed, target, tested["status"], f"{leaf['basis']} / {leaf['source']}"]
            lines.append("| " + " | ".join(_cell(value) for value in cells) + " |")
    lines.extend(["", "Condition results retain their original case scope and all/any or aggregation semantics.",
                  f"Workflow wall time: {result['time']['workflow_wall_time_s']:.3f} s. "
                  "Author time and execution-phase durations: unknown.", "",
                  "Read this summary by default; final.json provides detailed fields and final.csv is for export.", ""])
    if not cases:
        lines.extend(["Legacy report has no per-case validation detail; UNKNOWN cells do not claim a new execution.", ""])
    # Write the index last: status can recognize a complete set for this state.
    _atomic_text(destination / "final.csv", csv_text)
    _atomic_text(destination / "final.md", "\n".join(lines))
    result["state_sha256"] = object_digest(state)
    result["views"] = {name: digest(destination / name) for name in ("final.md", "final.csv")}
    atomic_json(destination / "final.json", result)
    return {"status": "written", "default_report": "reports/final/final.md",
            **{name: str((destination / name).relative_to(op_dir))
                                    for name in ("final.json", "final.md", "final.csv")}}


def _condition_logic(tree):
    if "id" in tree:
        return tree["id"]
    mode = next(iter(tree))
    return f"{mode}({', '.join(_condition_logic(child) for child in tree[mode])})"


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
