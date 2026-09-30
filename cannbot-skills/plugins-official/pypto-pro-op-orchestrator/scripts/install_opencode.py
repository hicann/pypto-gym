#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Install the PyPTO-Pro entry point and its Scriptor mode in OpenCode."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ORCHESTRATOR = Path(__file__).resolve().parents[1]
RUNTIME = ORCHESTRATOR / "scripts/scriptor-runtime"
sys.path.insert(0, str(RUNTIME))

from scriptorlib.common import ContractError, atomic_json, digest, read_json
from scriptorlib.sources import MANIFEST_NAME, verify

SKILLS = ORCHESTRATOR.parents[1] / "ops"
MARKER = "<!-- cannbot:pypto-pro-router:1 -->"
PRO_SKILLS = (
    "pypto-pro-docs-search", "pypto-pro-environment-check", "pypto-pro-golden-generate",
    "pypto-pro-intent-understand", "pypto-pro-material-explore", "pypto-pro-op-design",
    "pypto-pro-op-develop", "pypto-pro-op-perf-tune", "pypto-pro-op-plan",
    "pypto-pro-precision-debug", "pypto-pro-cann-delivery",
)
SCRIPTOR_SKILLS = tuple(f"pypto-pro-scriptor-{part}" for part in ("develop", "optimize", "verify"))


def render(text, config):
    return text.replace("$CANNBOT_CONFIG_ROOT", str(config))


def _known_root(target, config, alias_config=None):
    if not target.exists() and not target.is_symlink():
        return
    existing = target.read_text(encoding="utf-8")
    receipt = config / "scriptor-install.json"
    if receipt.exists():
        expected = read_json(receipt).get("root_prompt_sha256")
        if expected and digest(target) == expected:
            return
    # Preserve project-owned instructions; accept only the current managed template.
    pro = (ORCHESTRATOR / "AGENTS.md").read_text(encoding="utf-8")
    known = {pro, render(pro, config)}
    # macOS /var and /private/var can name the same installation. Accept only
    # the exact managed template rendered with the caller's equivalent path.
    if alias_config is not None and alias_config.resolve() == config:
        known.add(render(pro, alias_config))
    if existing in known:
        return
    raise ContractError("project AGENTS.md contains unmanaged instructions; merge the Pro entry point explicitly")


def _copy_tree(source, destination, *, config=None):
    if not source.is_dir():
        raise ContractError(f"required installation resource is missing: {source}")
    if destination.is_symlink():
        destination.unlink()
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        if any(part in {"__pycache__", ".pytest_cache", "node_modules"} for part in path.parts):
            continue
        target = destination / path.relative_to(source)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink():
                target.unlink()
            if config is not None and path.suffix == ".md":
                target.write_text(render(path.read_text(encoding="utf-8"), config), encoding="utf-8")
            else:
                shutil.copy2(path, target)


def install(project, python=None, *, global_level=False):
    lexical_project = project.absolute()
    project = project.resolve()
    config = project if global_level else project / ".opencode"
    target = project / "AGENTS.md"
    project.mkdir(parents=True, exist_ok=True)
    _known_root(target, config, lexical_project if global_level else lexical_project / ".opencode")
    previous = read_json(config / "scriptor-install.json") if (config / "scriptor-install.json").exists() else {}
    python = python or previous.get("python") or sys.executable
    probe = subprocess.run([python, "-c", "import sys; raise SystemExit(sys.version_info < (3,10))"],
                           capture_output=True, text=True)
    if probe.returncode:
        raise ContractError("the selected existing Python must be >=3.10")
    for name in PRO_SKILLS + SCRIPTOR_SKILLS:
        if not (SKILLS / name / "SKILL.md").is_file():
            raise ContractError(f"missing declared Skill: {name}")
    # The compiler stays in this checkout. OpenCode only receives workflow resources.
    source_root = (ORCHESTRATOR / "resources/ascriptor").resolve()
    verify(source_root)
    manifest = read_json(source_root / MANIFEST_NAME)
    for name in PRO_SKILLS + SCRIPTOR_SKILLS:
        _copy_tree(SKILLS / name, config / "skills" / name, config=config)
    _copy_tree(SKILLS / "pypto-pro-op-kb", config / "pypto-pro-op-kb")
    _copy_tree(ORCHESTRATOR / "references", config / "references", config=config)
    _copy_tree(ORCHESTRATOR / "agents", config / "agents", config=config)
    _copy_tree(ORCHESTRATOR / "hooks/opencode", config / "plugins")
    _copy_tree(ORCHESTRATOR / "hooks/pypto-pro-op-lint", config / "hooks/pypto-pro-op-lint")
    _copy_tree(RUNTIME, config / "scriptor/scripts")
    # OpenCode installs config/package.json dependencies with Bun at startup.
    package_path = config / "package.json"
    package = read_json(package_path) if package_path.exists() else {"private": True}
    if not isinstance(package, dict) or not isinstance(package.get("dependencies", {}), dict):
        raise ContractError(f"invalid OpenCode dependency manifest: {package_path}")
    package.setdefault("dependencies", {}).setdefault("@opencode-ai/plugin", "^1.0.0")
    atomic_json(package_path, package)
    content = render((ORCHESTRATOR / "AGENTS.md").read_text(encoding="utf-8"), config)
    workflows = config / "workflows"
    workflows.mkdir(exist_ok=True)
    (workflows / "pypto-pro-op-orchestrator.md").write_text(content, encoding="utf-8")
    (config / "agents/pypto-pro-op-orchestrator.md").write_text(content, encoding="utf-8")
    entry = MARKER + "\n" + render((ORCHESTRATOR / "AGENTS.md").read_text(encoding="utf-8"), config)
    if target.is_symlink():
        target.unlink()
    target.write_text(entry, encoding="utf-8")
    python_path = Path(python).absolute()
    # Resolving a venv symlink loses its pyvenv.cfg and changes the environment.
    if not (python_path.parent.parent / "pyvenv.cfg").is_file():
        python_path = python_path.resolve()
    receipt = {"schema": "cannbot.scriptor-install/1", "source_id": manifest["source_id"],
               "source_root": str(source_root), "python": str(python_path),
               "sources_index_sha256": manifest["index_sha256"],
               "sources_manifest_sha256": digest(source_root / MANIFEST_NAME),
               "root_prompt_sha256": digest(target), "host": "opencode",
               "workflows": ["pypto-pro-op-orchestrator"],
               "skills": list(PRO_SKILLS + SCRIPTOR_SKILLS)}
    atomic_json(config / "scriptor-install.json", receipt)
    # The adapter must resolve imports to the checked-in compiler source.
    command = [python, str(config / "scriptor/scripts/scriptor.py"), "--config-root", str(config), "doctor"]
    result = subprocess.run(command, text=True, capture_output=True, cwd=project)
    if result.returncode:
        raise ContractError(f"installed adapter preflight failed:\n{result.stderr or result.stdout}")
    return {"config_root": str(config), "source_id": manifest["source_id"],
            "workflows": receipt["workflows"], "doctor": json.loads(result.stdout)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("level", nargs="?", choices=("project", "global"), default="project")
    parser.add_argument("tool", nargs="?", choices=("opencode",), default="opencode")
    parser.add_argument("install_path", nargs="?", type=Path)
    parser.add_argument("--python", help="Existing Python environment; preserve the installed choice on refresh")
    args = parser.parse_args()
    project = args.install_path or (Path.home() / ".config/opencode" if args.level == "global" else Path.cwd())
    try:
        result = install(project, args.python, global_level=args.level == "global")
    except (ContractError, OSError, ValueError) as exc:
        parser.exit(1, f"scriptor installation failed: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
