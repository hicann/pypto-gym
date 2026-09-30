# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Verify the self-contained Ascriptor source snapshot shipped with this plugin."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
from urllib.parse import unquote

from .common import ContractError, canonical, digest, object_digest, read_json

ROOTS = ("library", "agent", "kernels")
MANIFEST_NAME = "sources.json"
INDEX_NAME = "sources-index.json"
VIEW_NAME = "product-view.json"
IGNORED_DIRS = {"__pycache__", ".pytest_cache", ".hypothesis", ".ruff_cache",
                ".mypy_cache", ".venv", ".vscode", ".idea", ".ddtui",
                ".worktrees", ".envs", "build", "dist", "tmp"}
IGNORED_FILES = {"boards.json", "machine_specs.md", ".DS_Store"}
IGNORED_SUFFIXES = {".pyc", ".pyo", ".log", ".bin", ".npy", ".npz", ".pt", ".pth"}
LINK = re.compile(r"(?<!!)\[[^\]\n]+\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
CODE_PATH = re.compile(r"\x60((?:library|agent|kernels)/[A-Za-z0-9_./-]+)\x60")
COMMAND = re.compile(r"\bpython(?:3)?\s+((?:\.\.?/)?(?:[A-Za-z0-9_-]+/)+[A-Za-z0-9_.-]+\.py)\b")


def _safe(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or ".." in path.parts or "\\" in name:
        raise ContractError(f"unsafe snapshot member: {name!r}")
    return str(path)


def _files(root: Path) -> dict:
    files = {}
    for owner in ROOTS:
        directory = root / owner
        if not directory.is_dir() or directory.is_symlink():
            raise ContractError(f"missing source directory: {owner}")
        for parent, dirs, names in os.walk(directory):
            base = Path(parent)
            dirs[:] = sorted(name for name in dirs if name not in IGNORED_DIRS
                             and not name.startswith("tmp_") and not name.endswith(".egg-info"))
            for name in dirs:
                if (base / name).is_symlink():
                    raise ContractError(f"snapshot needs regular directories: {base / name}")
            for name in sorted(names):
                if (name in IGNORED_FILES or name.startswith("tmp_")
                        or Path(name).suffix in IGNORED_SUFFIXES):
                    continue
                path = base / name
                relative = _safe(path.relative_to(root).as_posix())
                if path.is_symlink() or not path.is_file():
                    raise ContractError(f"snapshot needs regular files: {relative}")
                files[relative] = {"sha256": digest(path), "size": path.stat().st_size}
    return files


def _relative_target(document: str, target: str) -> str | None:
    target = unquote(target.split("#", 1)[0].split("?", 1)[0])
    if not target or target.startswith(("/", "#")) or ":" in target.split("/", 1)[0]:
        return None
    parts = []
    for part in (PurePosixPath(document).parent / target).parts:
        if part == "..":
            if not parts:
                return None
            parts.pop()
        elif part != ".":
            parts.append(part)
    return "/".join(parts)


def check_references(root: Path, files: dict) -> None:
    """Check navigation, executable inline commands, and declared JSON evidence paths."""
    names = set(files) | {MANIFEST_NAME, INDEX_NAME, VIEW_NAME}
    directories = {str(parent) for name in names for parent in PurePosixPath(name).parents
                   if str(parent) != "."}
    errors = []
    for document in sorted(name for name in names if name.endswith(".md")):
        content = (root / document).read_text(encoding="utf-8")
        navigation = re.sub(r"(?ms)^ {0,3}(?:\x60{3,}|~{3,}).*?^ {0,3}(?:\x60{3,}|~{3,})[^\n]*$", "", content)
        navigation = re.sub(r"\x60+[^\x60\n]*\x60+", "", navigation)
        for match in LINK.finditer(navigation):
            target = _relative_target(document, match.group(1))
            if target and target not in names and target not in directories:
                errors.append(f"{document}: missing link {target}")
        for match in CODE_PATH.finditer(content):
            target = match.group(1).rstrip("/.,")
            if target not in names and target not in directories:
                errors.append(f"{document}: missing inline path {target}")
        for match in COMMAND.finditer(content):
            command = match.group(1)
            if command.startswith("../"):
                target = _relative_target(document.split("/", 1)[0] + "/README.md", command)
            elif command.startswith("./"):
                target = _relative_target(document, command)
            else:
                target = document.split("/", 1)[0] + "/" + command
            alternatives = {owner + "/" + command.removeprefix("./") for owner in ROOTS}
            alternatives.add(command)
            if target and target not in names and not (alternatives & names):
                errors.append(f"{document}: missing command {command} ({target})")
    for owner, document in (("library", "library/examples/api/index.json"),
                            ("kernels", "kernels/index.json")):
        for row in read_json(root / document)["entries"]:
            target = owner + "/" + row["path"]
            if target not in directories:
                errors.append(f"{document}: missing JSON directory {target}")
    navigation = read_json(root / "agent/index/kernels.json")
    for row in navigation["candidates"]:
        if row["source"] not in directories:
            errors.append(f"agent/index/kernels.json: missing JSON directory {row['source']}")
    api_manifest = read_json(root / "library/docs/api/manifest.json")
    for row in api_manifest["entries"]:
        for key in ("declaration", "example"):
            if key in row:
                target = "library/" + row[key]
                if target not in names:
                    errors.append(f"library/docs/api/manifest.json: missing JSON path {key}={target}")
        for use in row.get("usage", []):
            target = "library/" + use["path"]
            if target not in names:
                errors.append(f"library/docs/api/manifest.json: missing JSON usage {target}")
        reference = row.get("reference", "").split(":", 1)[0]
        if reference.endswith((".py", ".md", ".json")) and "/" in reference:
            target = "library/" + reference
            if target not in names:
                errors.append(f"library/docs/api/manifest.json: missing JSON reference {target}")
    if errors:
        raise ContractError("snapshot has unresolved references:\n" + "\n".join(errors[:40])
                            + (f"\n... {len(errors) - 40} more" if len(errors) > 40 else ""))


def expected(root: Path) -> tuple[dict, dict]:
    view = read_json(root / VIEW_NAME)
    files = dict(sorted(_files(root).items()))
    check_references(root, files)
    pyproject = (root / "library/pyproject.toml").read_text(encoding="utf-8")
    version_match = re.search(r'^version\s*=\s*"(\d+\.\d+\.\d+)"\s*$', pyproject, re.M)
    if version_match is None:
        raise ContractError("library pyproject declares no source version")
    version = version_match.group(1)
    tree_sha256 = object_digest(files)
    source_id = f"ascriptor-{version}-{hashlib.sha256(canonical({'tree': tree_sha256, 'view': object_digest(view)})).hexdigest()[:16]}"
    index = {"schema": "cannbot.ascriptor-sources-index/1", "source_id": source_id,
             "version": version, "files": files}
    index_bytes = json.dumps(index, ensure_ascii=False, indent=2).encode() + b"\n"
    manifest = {"schema": "cannbot.ascriptor-sources/1", "source_id": source_id,
                "version": version, "provenance": {"kind": "self-contained-snapshot",
                                           "tree_sha256": tree_sha256,
                                           "description": "These indexed files are the complete delivered source identity; no external repository or wheel is required."},
                "file_count": len(files), "total_bytes": sum(row["size"] for row in files.values()),
                "index": INDEX_NAME, "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
                "product_view": {"path": VIEW_NAME, "sha256": object_digest(view)},
                "roots": {owner: owner for owner in ROOTS}}
    return manifest, index


def refresh(root: Path, *, check: bool = False) -> dict:
    root = root.resolve()
    manifest, index = expected(root)
    actual = [(MANIFEST_NAME, manifest), (INDEX_NAME, index)]
    rendered = {name: (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
                for name, value in actual}
    differing = [name for name in rendered if not (root / name).is_file()
                 or (root / name).read_bytes() != rendered[name]]
    if not check:
        for name, content in rendered.items():
            (root / name).write_bytes(content)
    return {"source_id": manifest["source_id"], "file_count": manifest["file_count"],
            "synchronized": not differing, "differing": differing}


def verify(root: Path, manifest: dict | None = None) -> dict:
    root = root.resolve()
    stored = read_json(root / MANIFEST_NAME)
    if manifest is not None and manifest != stored:
        raise ContractError("source manifest differs from the expected snapshot")
    result = refresh(root, check=True)
    if not result["synchronized"]:
        raise ContractError(f"source integrity check failed: {result['differing']}")
    extras = {path.name for path in root.iterdir()
              if path.name not in {*ROOTS, VIEW_NAME, INDEX_NAME, MANIFEST_NAME, *IGNORED_DIRS}
              and not path.name.startswith("tmp_")
              and path.name not in IGNORED_FILES
              and path.suffix not in IGNORED_SUFFIXES}
    if extras:
        raise ContractError(f"unexpected snapshot root entries: {sorted(extras)}")
    return read_json(root / INDEX_NAME)


def activate(config: Path) -> tuple[Path, dict]:
    receipt = read_json(config / "scriptor-install.json")
    if receipt.get("schema") != "cannbot.scriptor-install/1":
        raise ContractError("refresh the OpenCode Scriptor configuration for this source snapshot")
    root = Path(receipt["source_root"])
    if not root.is_absolute():
        raise ContractError("source_root must be an absolute source checkout path")
    root = root.resolve()
    manifest = read_json(root / MANIFEST_NAME)
    if receipt.get("source_id") != manifest["source_id"] or receipt.get("sources_index_sha256") != manifest["index_sha256"]:
        raise ContractError("OpenCode receipt selects another source snapshot")
    verify(root, manifest)
    existing = sys.modules.get("ascriptor")
    if existing and not Path(existing.__file__).resolve().is_relative_to(root):
        raise ContractError("ascriptor was imported from a different source before activation")
    sys.path.insert(0, str(root / "library"))
    import ascriptor
    if not Path(ascriptor.__file__).resolve().is_relative_to(root):
        raise ContractError("ascriptor import does not resolve to the source snapshot")
    return root, manifest
