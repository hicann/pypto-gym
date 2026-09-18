#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Validate PyPTO-Pro planning artifacts against the supplied devkit and KB."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Optional


OPS_ROOT = Path(__file__).resolve().parents[2]
MATERIAL_ROOT = OPS_ROOT / "pypto-pro-material-explore"
PLACEHOLDER_RE = re.compile(r"\{[^{}\r\n]+\}")
LOGGER = logging.getLogger(__name__)


class PlanConfigurationError(RuntimeError):
    pass


def _read(path: Path, label: str) -> str:
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read {label}: {error}") from error
    if not text.strip():
        raise ValueError(f"{label} is empty")
    return text


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path) if path.is_file() else None
    if spec is None or spec.loader is None:
        raise PlanConfigurationError(f"missing checker dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise PlanConfigurationError(f"cannot load {path}: {error}") from error
    return module


def _check_spec(op_dir: Path, validator: Any):
    try:
        path = op_dir / "SPEC.md"
        text = _read(path, "SPEC.md")
        op_name = validator.load_spec_contract(path)["op_name"]
        errors = [] if op_name == op_dir.name else [
            f"op_name={op_name!r}, directory={op_dir.name!r}"
        ]
        matches = list(re.finditer(r"(?m)^## kernel 契约补充\s*$", text))
        if len(matches) != 1 or not text[matches[0].end():].strip():
            errors.append("kernel contract section must occur once with content")
        return op_name, errors
    except (AttributeError, KeyError, OSError, ValueError) as error:
        return op_dir.name, [str(error)]


def _check_index(op_dir: Path, devkit: Path, material: Any):
    try:
        expected, _ = material.build_index(devkit, material.manifest_path())
    except Exception as error:
        raise PlanConfigurationError(f"cannot build canonical material index: {error}") from error
    try:
        actual = _read(op_dir / "PRO_MATERIAL_INDEX.md", "PRO_MATERIAL_INDEX.md")
    except ValueError as error:
        return expected, [str(error)]
    return expected, [] if actual == expected else ["PRO_MATERIAL_INDEX.md is stale"]


def _frontmatter(text: str) -> dict[str, str]:
    """Read the template's flat single-line scalars, not general YAML."""
    match = re.match(r"\A---\r?\n(.*?)\r?\n---(?:\r?\n|\Z)", text, re.DOTALL)
    if match is None:
        raise ValueError("EXPLORE_REPORT.md has no leading frontmatter")
    result = {}
    for line in match.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, separator, value = line.partition(":")
        key = key.strip()
        if not separator or not key or key in result:
            raise ValueError(f"invalid frontmatter line: {line!r}")
        scalar = re.fullmatch(
            r'''("(?:[^"\\]|\\.)*"|'(?:[^']|'')*'|[^'"#].*?)(?:[ \t]+#.*)?''',
            value.strip(),
        )
        if scalar is None:
            raise ValueError(f"invalid frontmatter scalar: {line!r}")
        value = scalar.group(1)
        if value.startswith('"'):
            value = json.loads(value)
        elif value.startswith("'"):
            value = value[1:-1].replace("''", "'")
        if not value.strip():
            raise ValueError(f"empty frontmatter value: {key}")
        result[key] = value
    return result


def _file_paths(text: str) -> set[str]:
    """Extract backtick file paths, retaining suffixes and possible anchors."""
    return {
        path for path in re.findall(r"`([^`\r\n]+)`", text)
        if re.match(r"^\S*[/\\]", path) and re.search(r"[^/\\]\.[^/\\#]+(?:#.*)?$", path)
    }


def _path_error(label: str, paths: list[str], reason: str) -> str:
    shown = ", ".join(paths[:8])
    remainder = f" (+{len(paths) - 8} more)" if len(paths) > 8 else ""
    return f"{label} {reason} ({len(paths)}): {shown}{remainder}"


def _check_citations(label: str, prefix: str, cited: set[str], allowed: set[str]) -> list[str]:
    errors = []
    if label != "API":
        missing = []
        for source in allowed - cited:
            if source.startswith(prefix) and not any(path.startswith(source + "#") for path in cited):
                missing.append(source)
        if missing:
            errors.append(_path_error(label, sorted(missing), "paths not covered in report"))
    # Match anchors against complete indexed names; filenames may contain '#'.
    extra = []
    for path in cited - allowed:
        if not any(path.startswith(source + "#") for source in allowed):
            extra.append(path)
    if extra:
        errors.append(_path_error(label, sorted(extra), "referenced paths absent from material index"))
    return errors


def check_report(op_dir: Path, op_name: str, index: str) -> list[str]:
    """Check an explore report against the template and canonical material index."""
    try:
        report = _read(op_dir / "EXPLORE_REPORT.md", "EXPLORE_REPORT.md")
        template = _read(MATERIAL_ROOT / "templates/explore_report.md", "report template")
        actual_meta, template_meta = _frontmatter(report), _frontmatter(template)
    except ValueError as error:
        return [str(error)]
    errors = []
    if actual_meta.get("schema_version") != template_meta.get("schema_version"):
        errors.append("frontmatter schema_version does not match template")
    if actual_meta.get("op_name") != op_name:
        errors.append(f"frontmatter op_name={actual_meta.get('op_name')!r}, expected {op_name!r}")
    if not actual_meta.get("feasibility") or "{" in actual_meta["feasibility"]:
        errors.append("frontmatter feasibility is missing")
    expected = [line for line in template.splitlines() if line.startswith("## ")]
    actual = [line for line in report.splitlines() if line.startswith("## ")]
    if actual != expected:
        errors.append("level-2 headings must exactly match template order")
    remaining = sorted(set(PLACEHOLDER_RE.findall(template)) & set(PLACEHOLDER_RE.findall(report)))
    if remaining:
        errors.append("unresolved template placeholders: " + ", ".join(remaining[:8]))
    paths = _file_paths(report)
    categories = (("sample", "pro_ops/"), ("guide", "docs/guide/"), ("API", "docs/pypto_pro/api/"))
    rooted = r"(?:/|\.\.?/|[A-Za-z]:/|\$(?:\w+|\{[^}]+\})/)"
    allowed = _file_paths(index)
    # Check devkit citations and required sample/guide coverage across the report.
    # Reading and applicability follow the Material Skill and semantic review.
    # Directories and commands are not evidence; supplementary KB references
    # are governed by the KB contract, not the devkit material index.
    for label, prefix in categories:
        cited = set()
        for path in paths:
            before, found, _ = path.replace("\\", "/").partition(prefix)
            # Preserve spaces in paths, but exclude command prefixes and
            # separate rooted arguments such as `/usr/bin/cat /cache/...`.
            if not found:
                continue
            if before and (not before.endswith("/") or not re.match(rooted, before)):
                continue
            if re.search(r"\s+" + rooted, before):
                continue
            cited.add(path)
        errors.extend(_check_citations(label, prefix, cited, allowed))
    return errors


def _mapping(kb_root: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            _read(kb_root / "topology-map.json", "topology-map.json"),
            object_pairs_hook=_unique,
        )
        if not isinstance(value, dict) or not isinstance(value.get("contract"), dict):
            raise ValueError("contract must be an object")
        contract = value["contract"]
        property_keys = contract.get("property_keys")
        version = contract.get("contract_version")
        if not isinstance(version, int) or isinstance(version, bool):
            raise ValueError("contract_version must be an integer")
        if not isinstance(property_keys, list) or not all(
            isinstance(key, str) and key for key in property_keys
        ):
            raise ValueError("property_keys must be non-empty strings")
        if len(property_keys) != len(set(property_keys)):
            raise ValueError("property_keys must be unique")
        if not isinstance(value.get("topologies"), dict):
            raise ValueError("topologies must be an object")
        return value
    except (KeyError, TypeError, ValueError) as error:
        raise PlanConfigurationError(f"invalid topology-map.json: {error}") from error


def _refs(refs: Any, namespace: str, kb_root: Path) -> list[str]:
    if not isinstance(refs, list):
        return [f"{namespace} references must be a list"]
    errors, seen = [], set()
    namespace_root = (kb_root / namespace).resolve()
    for index, ref in enumerate(refs):
        label = f"{namespace}[{index}]"
        if not isinstance(ref, dict):
            errors.append(f"{label} must be an object")
            continue
        path, reason = ref.get("path"), ref.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            errors.append(f"{label}.reason must be non-empty")
        if not isinstance(path, str):
            errors.append(f"{label}.path must be a string")
            continue
        pure = PurePosixPath(path)
        if pure.is_absolute() or ".." in pure.parts or pure.parts[:1] != (namespace,):
            errors.append(f"{label}.path escapes {namespace}/: {path}")
            continue
        if path in seen:
            errors.append(f"duplicate reference path: {path}")
        seen.add(path)
        target = (kb_root / path).resolve()
        try:
            target.relative_to(namespace_root)
        except ValueError:
            errors.append(f"{label}.path resolves outside {namespace}/: {path}")
            continue
        if not target.is_file():
            errors.append(f"referenced file does not exist: {path}")
            continue
        try:
            digest = "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest()
        except OSError as error:
            errors.append(f"cannot hash {path}: {error}")
            continue
        if ref.get("sha256") != digest:
            errors.append(f"stale sha256: {path}")
    return errors


def _check_routing_fields(data: dict[str, Any], mapping: dict[str, Any], label: str) -> list[str]:
    errors = []
    topologies = data.get("topologies")
    if not isinstance(topologies, list):
        errors.append(f"{label} topologies must be a list")
    else:
        invalid = [x for x in topologies if not isinstance(x, str) or x not in mapping["topologies"]]
        if invalid or len(topologies) != len(set(x for x in topologies if isinstance(x, str))):
            errors.append(f"{label} has unknown, duplicate, or non-string topologies: {invalid!r}")
    properties = data.get("properties")
    if not isinstance(properties, dict):
        errors.append(f"{label} properties must be an object")
    else:
        unknown = sorted(set(properties) - set(mapping["contract"]["property_keys"]))
        if unknown:
            errors.append(f"{label} has unknown properties: {unknown}")
    return errors


def check_kb(op_dir: Path, kb_root: Path, op_name: str) -> list[str]:
    """Check selection layouts and contents against the supplied KB's current contract."""
    mapping = _mapping(kb_root)
    contract = mapping["contract"]
    paths = sorted(op_dir.rglob("KB_SELECTION.json"))
    allowed = [path for path in paths if len(path.relative_to(op_dir).parts) in (1, 2)]
    errors = [f"nested selection: {path}" for path in paths if path not in allowed]
    flat = op_dir / "KB_SELECTION.json"
    if flat in allowed and len(allowed) > 1:
        errors.append("flat and split layouts cannot be mixed")
    if not allowed:
        errors.append("no KB_SELECTION.json found")
    required = {
        "schema_version", "op", "class_id", "topologies", "properties",
        "optional_patterns", "required_constraints", "no_matching_pattern",
    }
    for path in allowed:
        label = path.relative_to(op_dir).as_posix()
        try:
            data = json.loads(_read(path, label), object_pairs_hook=_unique)
            if not isinstance(data, dict):
                raise ValueError("root must be an object")
        except ValueError as error:
            errors.append(f"{label}: {error}")
            continue
        missing = sorted(required - set(data))
        if missing:
            errors.append(f"{label} missing fields: {', '.join(missing)}")
        expected_class = "." if path == flat else path.parent.name
        if data.get("op") != op_name or data.get("class_id") != expected_class:
            errors.append(f"{label} op/class_id does not match location")
        if data.get("schema_version") != contract["contract_version"]:
            errors.append(f"{label} schema_version does not match topology-map")
        if "topology" in data:
            errors.append(f"{label} uses forbidden legacy topology")
        errors.extend(_check_routing_fields(data, mapping, label))
        patterns = data.get("optional_patterns")
        errors.extend(f"{label}: {error}" for error in _refs(patterns, "patterns", kb_root))
        errors.extend(
            f"{label}: {error}"
            for error in _refs(data.get("required_constraints"), "constraints", kb_root)
        )
        flag = data.get("no_matching_pattern")
        if not isinstance(flag, bool) or isinstance(patterns, list) and flag is not (not patterns):
            errors.append(f"{label} no_matching_pattern is invalid")
    return errors


def validate(op_dir: Path, devkit: Path, kb_root: Path):
    for path, label in ((op_dir, "operator"), (devkit, "devkit"), (kb_root, "KB root")):
        if not path.is_dir():
            raise PlanConfigurationError(f"{label} directory does not exist: {path}")
    intent = _module(
        "_plan_spec", OPS_ROOT / "pypto-pro-intent-understand/scripts/validate_spec.py",
    )
    material = _module(
        "_plan_material", MATERIAL_ROOT / "scripts/build_material_index.py",
    )
    op_name, spec_errors = _check_spec(op_dir, intent)
    index, index_errors = _check_index(op_dir, devkit, material)
    return [
        ("SPEC", spec_errors),
        ("INDEX", index_errors),
        ("REPORT", check_report(op_dir, op_name, index)),
        ("KB", check_kb(op_dir, kb_root, op_name)),
    ]


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op-dir", required=True)
    parser.add_argument("--devkit", required=True)
    parser.add_argument("--kb-root", required=True)
    args = parser.parse_args(argv)
    try:
        results = validate(*(
            Path(value).resolve() for value in (args.op_dir, args.devkit, args.kb_root)
        ))
    except PlanConfigurationError as error:
        LOGGER.error("ERROR: %s", error)
        return 2
    failed = False
    for name, errors in results:
        failed = failed or bool(errors)
        for error in errors:
            LOGGER.info("[FAIL] %s: %s", name, error)
        if not errors:
            LOGGER.info("[PASS] %s", name)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
