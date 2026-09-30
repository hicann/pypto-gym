#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Reproducible CLI for the same installed adapter used by the OpenCode plugin."""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
from pathlib import Path
import sys

from scriptorlib.common import (ContractError, atomic_json, confined, config_root,
                               digest, finite, json_pointer, parse_json, read_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    state = sub.add_parser("state", formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Read or advance the workflow. Default output includes the next request and command.",
        epilog='''Requests (report paths are relative to the operator directory):
  --request-json '{"action":"init","entry_mode":"from_pro","opDir":"custom/example"}'
  --request-json '{"action":"status","opDir":"custom/example"}'
  --request-json '{"action":"finish_optimization","opDir":"custom/example"}'
  --request-json '{"action":"complete_accept","opDir":"custom/example","report":"reports/sealed/accept.json"}'
Add --full only when the summary lacks a field needed for the current task.
--request reads a JSON file; --request-json takes the JSON object directly.''')
    state.add_argument("--project", type=Path, default=Path.cwd())
    request = state.add_mutually_exclusive_group(required=True)
    request.add_argument("--request", type=Path)
    request.add_argument("--request-json")
    state.add_argument("--actor", default="build", help="CLI reproduction only; OpenCode supplies its trusted caller")
    state.add_argument("--full", "--full-state", dest="full_state", action="store_true",
                       help="Return the complete state ledger instead of the compact view")
    for name in ("bootstrap-check", "bootstrap-restart", "export", "snapshot", "check", "seal", "observe", "report", "delivery-check"):
        command = sub.add_parser(name)
        command.add_argument("--op-dir", type=Path, required=True)
        if name == "bootstrap-restart":
            command.add_argument("--reason", required=True,
                                 help="why the sealed Pro inputs must be revised before Scriptor init")
        elif name == "export":
            command.add_argument("--sync-mode", choices=("manual", "auto_mutex"), default="auto_mutex")
        elif name == "delivery-check":
            command.add_argument("--delivery-dir", type=Path, required=True,
                                 help="slim runnable package at delivery/<op>/")
            command.add_argument("--structure-only", action="store_true",
                                 help="diagnose package structure without claiming device acceptance")
            command.add_argument("--timeout", type=float, default=3600,
                                 help="maximum seconds for isolated package test.py (default: 3600)")
        elif name == "check":
            command.add_argument("--stage", choices=("prepare", "candidate", "accept"), required=True,
                                 help="candidate/accept for Scriptor mode; prepare is legacy compatibility only")
            command.add_argument("--board")
            command.add_argument("--boards-file")
            command.add_argument("--timeout", type=float)
        elif name == "seal":
            command.add_argument("--checks", required=True)
            command.add_argument("--review", required=True)
            command.add_argument("--output", required=True)
            command.add_argument("--recommendation", choices=("keep", "reject"), default="reject")
        elif name == "observe":
            command.add_argument("--request", type=Path, required=True,
                                 help="Metric descriptors with evidence_file and JSON pointer, not invented values")
    args = parser.parse_args()
    try:
        config = config_root(args.config_root)
        # Keep compiler diagnostics on stderr; the stdout contract is always one JSON result.
        with contextlib.redirect_stdout(sys.stderr):
            if args.command == "doctor":
                from scriptorlib.sources import activate
                root, manifest = activate(config)
                from ascriptor.runtime import compile_kernel  # noqa: F401
                result = {"source_id": manifest["source_id"], "source_import": str(root / "library/ascriptor"),
                          "source_root": str(root), "agent_router": str(root / "agent/zh-CN/ROUTER.md"),
                          "source_emission": True, "python": sys.executable,
                          "modules_available": {name: importlib.util.find_spec(name) is not None
                                                for name in ("torch", "numpy", "pypto_pro", "torch_npu")},
                          "scope": "PyPTO-Pro import/source capabilities only; ACLNN packaging template is excluded; run check for device evidence"}
            elif args.command == "state":
                from scriptorlib.workflow import transition
                data = read_json(args.request) if args.request else parse_json(args.request_json)
                full_state = transition(config, args.project.resolve(), data, actor=args.actor)
                op_dir = confined(args.project.resolve(), data["opDir"])
                results = None
                if data["action"] == "complete_accept":
                    from scriptorlib.progress import write_final
                    try:
                        results = write_final(config, op_dir, full_state)
                    except (ContractError, OSError, ValueError, KeyError) as exc:
                        results = {"status": "unavailable", "reason": str(exc)}
                if args.full_state or data.get("detail") == "full":
                    result = full_state
                else:
                    from scriptorlib.progress import state_summary
                    result = state_summary(config, args.project.resolve(), op_dir, full_state)
                if results is not None:
                    result = {**result, "results": results}
            else:
                op_dir = args.op_dir.resolve()
                if args.command == "bootstrap-check":
                    from scriptorlib.workflow import bootstrap_check
                    result = bootstrap_check(config, op_dir)
                elif args.command == "bootstrap-restart":
                    from scriptorlib.workflow import bootstrap_restart
                    result = bootstrap_restart(config, op_dir, args.reason)
                elif args.command == "export":
                    from scriptorlib.exporter import export
                    result = export(config, op_dir, sync_mode=args.sync_mode)
                elif args.command == "snapshot":
                    from scriptorlib.workflow import artifact_hashes, subject
                    result = {"artifact_hash": subject(op_dir), "artifact_hashes": artifact_hashes(op_dir)}
                elif args.command == "check":
                    from scriptorlib.runner import check
                    result = check(config, op_dir, args.stage, options={"board": args.board,
                                   "boards_file": args.boards_file, "timeout": args.timeout})
                elif args.command == "seal":
                    from scriptorlib.runner import seal
                    result = seal(config, op_dir, args.checks, args.review, args.output,
                                  recommendation=args.recommendation)
                elif args.command == "report":
                    from scriptorlib.progress import write_final
                    result = write_final(config, op_dir)
                elif args.command == "delivery-check":
                    from scriptorlib.delivery import check_delivery, run_delivery_test
                    from scriptorlib.exporter import verify_export
                    package = args.delivery_dir.resolve()
                    structure = check_delivery(op_dir, package, verify_export(config, op_dir))
                    result = structure if args.structure_only else run_delivery_test(
                        op_dir, package, structure, timeout=args.timeout)
                else:
                    from scriptorlib.workflow import subject
                    candidate = subject(op_dir)
                    records = []
                    for descriptor in read_json(args.request):
                        source = confined(op_dir, descriptor["evidence_file"], must_exist=True)
                        data = read_json(source)
                        if data.get("artifact_hash") != candidate:
                            raise ContractError("external metric source is not bound to this candidate")
                        pointer = descriptor["pointer"]
                        value = json_pointer(data, pointer)
                        finite(value, "observed metric")
                        records.append({key: descriptor[key] for key in
                                        ("metric", "case", "unit", "basis", "source", "evidence_file", "pointer")})
                        records[-1].update(value=value, artifact_hash=candidate, evidence_sha256=digest(source))
                    observations = op_dir / "reports/observations.json"
                    atomic_json(observations, {"artifact_hash": candidate, "metrics": records})
                    result = {"observations": len(records), "path": "reports/observations.json",
                              "sha256": digest(observations)}
    except Exception as exc:
        # No full traceback/locals: board configuration is never part of a public report.
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
