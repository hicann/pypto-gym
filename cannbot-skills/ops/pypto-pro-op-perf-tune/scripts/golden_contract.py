#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Golden report and performance-case contracts used by Stage 5."""

import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional


def _positive_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


class GoldenProtocol(NamedTuple):
    """校验通过的 Golden 协议字段；校验失败时除 error 外均为 None。"""
    device_id: Optional[int]
    iterations: Optional[int]
    warmup: Optional[int]
    repeats: Optional[int]
    seed: Optional[int]
    error: Optional[str]


def _golden_protocol_values(payload: Dict[str, Any]) -> GoldenProtocol:
    """校验 Golden 合同的协议字段；返回 GoldenProtocol（失败时带 error）。"""

    def required_int(name, minimum):
        value = payload.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"Golden {name} must be an integer >= {minimum}")
        return value

    try:
        device_id = required_int("device_id", 0)
        iterations = required_int("iterations", 1)
        warmup = required_int("warmup", 0)
        repeats = required_int("repeats", 1)
        seed = required_int("seed", 0)
    except ValueError as error:
        return GoldenProtocol(None, None, None, None, None, str(error))
    if iterations != 1:
        return GoldenProtocol(None, None, None, None, None, "Stage 5 Golden iterations must be exactly 1")
    if seed != 42:
        return GoldenProtocol(None, None, None, None, None, "Golden seed must be 42")
    if payload.get("timing_scope") != "all_golden_npu_kernels":
        return GoldenProtocol(None, None, None, None, None, "Golden timing_scope must be all_golden_npu_kernels")
    if payload.get("aggregation") != "median_of_repeat_totals":
        return GoldenProtocol(None, None, None, None, None, "Golden aggregation must be median_of_repeat_totals")
    source_error = _source_file_record_error(payload.get("golden_source"), "Golden source")
    if source_error:
        return GoldenProtocol(None, None, None, None, None, source_error)
    return GoldenProtocol(device_id, iterations, warmup, repeats, seed, None)


def _golden_cases(payload: Dict[str, Any], repeats: int, iterations: int):
    """校验 Golden 合同的 cases 列表并逐项复算；返回 (cases, error)。"""
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        return None, "Golden cases must be a non-empty list"
    cases = []
    seen_ids = set()
    for index, record in enumerate(raw_cases):
        if not isinstance(record, dict):
            return None, f"Golden cases[{index}] must be an object"
        case_id = record.get("id")
        shape = record.get("shape")
        dtype = record.get("dtype")
        if not isinstance(case_id, str) or not case_id.strip():
            return None, f"Golden cases[{index}].id must be non-empty"
        case_id = case_id.strip()
        if case_id in seen_ids:
            return None, f"Golden contains duplicate case id {case_id!r}"
        seen_ids.add(case_id)
        if not isinstance(shape, str) or not shape.strip():
            return None, f"Golden cases[{index}].shape must be non-empty"
        if not isinstance(dtype, str) or not dtype.strip():
            return None, f"Golden cases[{index}].dtype must be non-empty"
        raw_samples = record.get("raw_repeat_e2e_us")
        if (
            not isinstance(raw_samples, list)
            or len(raw_samples) != repeats
            or not all(_positive_number(value) for value in raw_samples)
        ):
            return None, (
                f"Golden case {case_id!r} must contain {repeats} positive finite raw repeats"
            )
        expected_median = float(statistics.median(float(value) for value in raw_samples))
        median_total = record.get("median_total_us")
        per_iteration = record.get("per_iteration_e2e_us")
        if not _positive_number(median_total) or not math.isclose(
            float(median_total), expected_median, rel_tol=1e-9, abs_tol=1e-6
        ):
            return None, f"Golden case {case_id!r} median is not reproducible"
        if not _positive_number(per_iteration) or not math.isclose(
            float(per_iteration), expected_median / iterations,
            rel_tol=1e-9, abs_tol=1e-6,
        ):
            return None, (
                f"Golden case {case_id!r} per-iteration E2E is not reproducible"
            )
        cases.append((case_id, shape.strip(), dtype.strip(), float(per_iteration)))
    return cases, None


def parse_golden_contract(out_dir: Path):
    """Load and fully validate the machine-readable Golden collection."""
    contract_path = out_dir / "GOLDEN_PERF_REPORT.json"
    if not contract_path.exists():
        return None, None, "GOLDEN_PERF_REPORT.json not found"
    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return None, None, f"cannot load GOLDEN_PERF_REPORT.json: {error}"
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return None, None, "GOLDEN_PERF_REPORT.json must use schema_version=1"

    protocol = _golden_protocol_values(payload)
    if protocol.error:
        return None, None, protocol.error
    device_id = protocol.device_id
    iterations = protocol.iterations
    warmup = protocol.warmup
    repeats = protocol.repeats
    seed = protocol.seed
    cases, cases_error = _golden_cases(payload, repeats, iterations)
    if cases_error:
        return None, None, cases_error

    return cases, {
        "schema_version": 1,
        "device_id": device_id,
        "iterations": iterations,
        "warmup": warmup,
        "repeats": repeats,
        "seed": seed,
        "timing_scope": payload["timing_scope"],
        "aggregation": payload["aggregation"],
        "golden_source": payload["golden_source"],
        "case_manifest_source": payload.get("case_manifest_source"),
    }, None


def source_file_record(path: Path) -> Dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"case source is not a regular file: {resolved}")
    raw = resolved.read_bytes()
    return {
        "path": str(resolved),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


def _source_file_record_error(record: Any, label: str) -> Optional[str]:
    """Recompute a recorded raw-file identity so evidence stays verifiable."""
    if not isinstance(record, dict):
        return f"{label} source record is missing"
    raw_path = record.get("path")
    expected_sha256 = record.get("sha256")
    expected_size = record.get("size_bytes")
    if not isinstance(raw_path, str) or not raw_path:
        return f"{label} source path is missing"
    if not re.fullmatch(r"[0-9a-f]{64}", str(expected_sha256 or "")):
        return f"{label} source sha256 is invalid"
    if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size < 0:
        return f"{label} source size_bytes is invalid"
    try:
        path = Path(raw_path)
        if not path.is_absolute():
            return f"{label} source path is not absolute"
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            return f"{label} source is not a regular file: {resolved}"
        raw = resolved.read_bytes()
    except OSError as error:
        return f"cannot reload {label} source: {error}"
    if str(resolved) != raw_path:
        return f"{label} source path is not canonical: {raw_path}"
    if len(raw) != expected_size:
        return f"{label} source size mismatch"
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        return f"{label} source sha256 mismatch"
    return None


def _golden_contract_record_error(record: Any) -> Optional[str]:
    """Revalidate the frozen Golden JSON contract recorded for target decisions."""
    if not isinstance(record, dict):
        return "Golden contract record is missing"
    error = _source_file_record_error(record, "Golden contract")
    if error:
        return error
    try:
        path = Path(record["path"])
        cases, metadata, parse_error = parse_golden_contract(path.parent)
    except (KeyError, OSError, ValueError) as error:
        return f"cannot reload Golden contract: {error}"
    if parse_error:
        return parse_error
    if not cases:
        return "Golden contract has no cases"
    expected = record.get("protocol")
    protocol_keys = (
        "schema_version", "device_id", "iterations", "warmup",
        "repeats", "seed", "timing_scope", "aggregation",
    )
    actual = {}
    for key in protocol_keys:
        actual[key] = metadata[key]
    if expected != actual:
        return "Golden contract protocol record mismatch"
    return None


def performance_case_source_error(record: Any) -> Optional[str]:
    """Validate the manifest source and its optional exact-id Golden join."""
    error = _source_file_record_error(record, "performance case")
    if error:
        return error
    if record.get("type") != "case_manifest":
        return "performance case source must be a case_manifest"
    if record.get("schema_version") != 1:
        return "performance case manifest schema_version is not 1"

    diagnostic = record.get("golden_diagnostic")
    if not isinstance(diagnostic, dict):
        return "golden_diagnostic source record is missing"
    status = diagnostic.get("status")
    if status not in {"not_provided", "joined"}:
        return f"invalid golden_diagnostic status: {status!r}"
    if status == "not_provided":
        if record.get("golden_contract") is not None:
            return "Golden contract cannot be present when Golden was not provided"
        return None

    iterations = diagnostic.get("iterations")
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations <= 0:
        return "Golden diagnostic iterations must be a positive integer"
    contract = record.get("golden_contract")
    if not isinstance(contract, dict):
        return "joined Golden target evidence is missing its machine contract"
    error = _golden_contract_record_error(contract)
    if error:
        return error
    try:
        _, metadata, parse_error = parse_golden_contract(Path(contract["path"]).parent)
    except (KeyError, OSError, ValueError) as load_error:
        return f"cannot reload Golden contract relationship: {load_error}"
    if parse_error:
        return parse_error
    expected_manifest = {
        key: record.get(key)
        for key in ("path", "sha256", "size_bytes", "schema_version")
    }
    if metadata.get("case_manifest_source") != expected_manifest:
        return "Golden contract does not bind to this performance case manifest"
    return None


def parse_case_manifest(raw_path: str):
    """Parse the explicit Stage 5 performance-case manifest.

    The manifest defines which existing runner cases are measured. Golden
    JSON is joined separately when it exactly covers the manifest.
    """
    if not raw_path or not str(raw_path).strip():
        return None, None, "--case-manifest must name a non-empty path"
    try:
        path = Path(raw_path).expanduser().resolve(strict=True)
        source_record = source_file_record(path)
        payload = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, ValueError) as error:
        return None, None, f"cannot load case manifest {raw_path!r}: {error}"

    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return None, None, "case manifest must be an object with schema_version=1"
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        return None, None, "case manifest cases must be a non-empty list"

    cases = []
    seen_ids = set()
    case_functions = {}
    for index, item in enumerate(raw_cases):
        if not isinstance(item, dict):
            return None, None, f"case manifest cases[{index}] must be an object"
        case_id = item.get("id")
        shape = item.get("shape")
        dtype = item.get("dtype")
        for field, value in (("id", case_id), ("shape", shape), ("dtype", dtype)):
            if not isinstance(value, str) or not value.strip():
                return None, None, (
                    f"case manifest cases[{index}].{field} must be a non-empty string"
                )
        case_id = case_id.strip()
        if case_id in seen_ids:
            return None, None, f"case manifest contains duplicate case id {case_id!r}"
        seen_ids.add(case_id)
        test_function = item.get("test_function")
        if test_function is not None:
            if not isinstance(test_function, str) or not re.fullmatch(
                    r"test_[A-Za-z_][A-Za-z0-9_]*", test_function.strip()):
                return None, None, (
                    f"case manifest cases[{index}].test_function must name a no-argument "
                    "test_* function"
                )
            case_functions[case_id] = test_function.strip()
        cases.append((case_id, shape.strip(), dtype.strip(), None))

    source_record.update({
        "type": "case_manifest",
        "schema_version": 1,
        "case_functions": case_functions,
        "golden_diagnostic": {"status": "not_provided"},
    })
    return cases, source_record, None


def _validated_unique_case_map(cases, source_name):
    case_map = {}
    for case in cases:
        case_id = str(case[0]).strip()
        if not case_id:
            return None, f"{source_name} case ids must be non-empty"
        if case_id in case_map:
            return None, f"{source_name} contains duplicate case id {case_id!r}"
        case_map[case_id] = case
    return case_map, None


def _canonical_shape(text: str):
    """Normalize simple or named shape text without evaluating arbitrary code."""
    compact = re.sub(r"\s+", "", str(text))
    groups = re.findall(
        r"(?:(?P<name>[A-Za-z_][A-Za-z0-9_]*):)?[\[(](?P<dims>[^\])]+)[\])]",
        compact,
    )
    if not groups and compact in {"()", "[]", "scalar"}:
        return ((None, ()),)
    normalized = []
    for name, group in groups:
        try:
            dims = tuple(int(part) for part in re.split(r"[,xX]", group) if part)
        except ValueError:
            return None
        if any(dim <= 0 for dim in dims):
            return None
        normalized.append((name or None, dims))
    return tuple(normalized) if normalized else None


def _canonical_dtype(text: str):
    aliases = {
        "half": "float16", "fp16": "float16",
        "float": "float32", "fp32": "float32",
        "double": "float64", "fp64": "float64",
        "bf16": "bfloat16",
    }
    tokens = re.findall(
        r"(?:(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*)?"
        r"(?:torch\.)?(?P<dtype>bfloat16|float16|float32|float64|half|float|double|"
        r"fp16|fp32|fp64|bf16|int8|uint8|int16|uint16|int32|uint32|int64|uint64|bool)",
        str(text).lower(),
    )
    return tuple(
        (name or None, aliases.get(dtype, dtype)) for name, dtype in tokens
    ) or None


def golden_case_metadata_error(manifest_case, golden_case):
    case_id = manifest_case[0]
    manifest_shape = _canonical_shape(manifest_case[1])
    golden_shape = _canonical_shape(golden_case[1])
    manifest_dtype = _canonical_dtype(manifest_case[2])
    golden_dtype = _canonical_dtype(golden_case[2])
    if manifest_shape is None or golden_shape is None or manifest_shape != golden_shape:
        return f"case {case_id!r} shape mismatch between manifest and Golden contract"
    if manifest_dtype is None or golden_dtype is None or manifest_dtype != golden_dtype:
        return f"case {case_id!r} dtype mismatch between manifest and Golden contract"
    return None


def _join_golden_contract(
    out_dir: Path, cases: list, source_record: Dict[str, Any], contract_path: Path
):
    """按 exact id 把 Golden 合同联接到 manifest；返回 (cases, source_record, iterations, error)。"""
    golden_cases, metadata, golden_error = parse_golden_contract(out_dir)
    if golden_error:
        return None, None, None, golden_error
    iterations = metadata["iterations"]
    golden_map, map_error = _validated_unique_case_map(
        golden_cases, "GOLDEN_PERF_REPORT.json"
    )
    if map_error:
        return None, None, None, map_error
    manifest_ids = [case[0] for case in cases]
    missing = sorted(set(manifest_ids) - set(golden_map))
    extra = sorted(set(golden_map) - set(manifest_ids))
    if missing or extra:
        return None, None, None, (
            "case id mismatch between --case-manifest and GOLDEN_PERF_REPORT.json: "
            f"missing_in_golden={missing}, extra_in_golden={extra}"
        )
    for manifest_case in cases:
        metadata_error = golden_case_metadata_error(
            manifest_case, golden_map[manifest_case[0]]
        )
        if metadata_error:
            return None, None, None, metadata_error
    contract_manifest = metadata.get("case_manifest_source")
    if contract_manifest != {
        key: source_record[key]
        for key in ("path", "sha256", "size_bytes", "schema_version")
    }:
        return None, None, None, (
            "Golden contract was not collected from this exact case manifest"
        )

    cases = [
        (case_id, shape, dtype, golden_map[case_id][3])
        for case_id, shape, dtype, _ in cases
    ]
    source_record["golden_diagnostic"] = {
        "status": "joined",
        "iterations": iterations,
    }
    contract_record = source_file_record(contract_path)
    contract_record["protocol"] = {}
    for key in (
        "schema_version", "device_id", "iterations", "warmup",
        "repeats", "seed", "timing_scope", "aggregation",
    ):
        contract_record["protocol"][key] = metadata[key]
    source_record["golden_contract"] = contract_record
    return cases, source_record, iterations, None


def resolve_performance_cases(out_dir: Path, args=None):
    """Load the required manifest and optionally join its exact Golden JSON."""
    manifest_path = getattr(args, "case_manifest", None) if args is not None else None
    if not manifest_path:
        return None, None, None, "--case-manifest is required"

    cases, source_record, error = parse_case_manifest(manifest_path)
    if error:
        return None, None, None, error

    contract_path = out_dir / "GOLDEN_PERF_REPORT.json"
    if not contract_path.exists():
        return cases, source_record, None, None
    return _join_golden_contract(out_dir, cases, source_record, contract_path)
