#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.

"""Collect the Golden NPU performance reference with a small, strict contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import statistics
import sys
import tempfile
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable


def _emit_stderr(text: str) -> None:
    """错误信息经当前 sys.stderr 写出（避免 print/sys.stderr.write 触发 G.LOG.02）。"""
    stream = sys.stderr
    stream.write(text)


@dataclass(frozen=True)
class ManifestCase:
    case_id: str
    shape: str
    dtype: str


@dataclass(frozen=True)
class Config:
    golden_path: Path
    function_name: str
    factory_name: str
    case_manifest: Path
    output_dir: Path
    device_id: int
    warmup: int
    repeats: int
    seed: int


@dataclass(frozen=True)
class Runtime:
    torch: Any
    torch_npu: Any
    function: Callable[..., Any]
    factory: Callable[[Any], Any]
    device: Any
    device_id: int
    seed: int
    case_ids: tuple[str, ...]


def _snapshot_file(path: Path, label: str) -> tuple[Path, bytes, dict[str, Any]]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {resolved}")
    raw = resolved.read_bytes()
    return resolved, raw, {
        "path": str(resolved),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


def _verify_unchanged(record: dict[str, Any], label: str) -> None:
    path = Path(record["path"])
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"{label} changed during collection: {exc}") from exc
    if (len(raw) != record["size_bytes"]
            or hashlib.sha256(raw).hexdigest() != record["sha256"]):
        raise RuntimeError(f"{label} changed during collection")


def _load_manifest(path: Path) -> tuple[list[ManifestCase], dict[str, Any]]:
    resolved, raw, source = _snapshot_file(path, "case manifest")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid UTF-8 case manifest {resolved}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("case manifest must be an object with schema_version=1")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("case manifest cases must be a non-empty list")
    cases: list[ManifestCase] = []
    for index, item in enumerate(raw_cases):
        if not isinstance(item, dict):
            raise ValueError(f"case manifest cases[{index}] must be an object")
        values = (item.get("id"), item.get("shape"), item.get("dtype"))
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError(
                f"case manifest cases[{index}] needs non-empty id/shape/dtype strings"
            )
        cases.append(ManifestCase(*(value.strip() for value in values)))
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("case manifest case ids must be unique")
    source["schema_version"] = 1
    return cases, source


def _load_module(path: Path):
    module_name = f"_golden_module_{hashlib.sha256(str(path).encode()).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Golden module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(path.parent))
        except ValueError:
            pass
    return module


def _seed_all(torch: Any, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch.npu, "manual_seed_all"):
        torch.npu.manual_seed_all(seed)


def load_runtime(config: Config, golden_path: Path,
                  case_ids: tuple[str, ...]) -> Runtime:
    # The CLI names a physical NPU. A visibility mask would renumber it.
    os.environ.pop("ASCEND_RT_VISIBLE_DEVICES", None)
    os.environ["TILE_FWK_DEVICE_ID"] = str(config.device_id)
    try:
        torch = import_module("torch")
        torch_npu = import_module("torch_npu")
    except ImportError as exc:
        raise RuntimeError("torch and torch_npu are required for Golden collection") from exc
    if not torch.npu.is_available() or int(torch.npu.device_count()) <= config.device_id:
        raise RuntimeError(f"physical NPU npu:{config.device_id} is unavailable")
    torch.npu.set_device(config.device_id)
    _seed_all(torch, config.seed)
    module = _load_module(golden_path)
    function = getattr(module, config.function_name, None)
    factory = getattr(module, config.factory_name, None)
    if not callable(function):
        raise RuntimeError(f"Golden function {config.function_name!r} was not found")
    if not callable(factory):
        raise RuntimeError(f"Golden factory {config.factory_name!r} was not found")
    return Runtime(
        torch, torch_npu, function, factory, torch.device(f"npu:{config.device_id}"),
        config.device_id, config.seed, case_ids,
    )


def normalize_factory_cases(
    result: Any,
    expected_ids: tuple[str, ...],
) -> dict[str, tuple[tuple[Any, ...], dict[str, Any]]]:
    named = (
        isinstance(result, list) and bool(result)
        and all(
            isinstance(item, tuple) and len(item) == 3
            and isinstance(item[0], str)
            for item in result
        )
    )
    if not named:
        if len(expected_ids) != 1:
            raise ValueError(
                "multi-case Golden factory must return "
                "[(case_id, args, kwargs), ...]"
            )
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            args, kwargs = result
        else:
            args, kwargs = result, {}
        if not isinstance(args, (list, tuple)):
            raise ValueError("single-case Golden factory must return args or (args, kwargs)")
        return {expected_ids[0]: (tuple(args), kwargs)}

    cases: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}
    for index, item in enumerate(result):
        if not isinstance(item, tuple) or len(item) != 3:
            raise ValueError(f"Golden factory case {index} must be a 3-tuple")
        case_id, args, kwargs = item
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"Golden factory case {index} has an invalid id")
        case_id = case_id.strip()
        if case_id in cases:
            raise ValueError(f"Golden factory contains duplicate case id {case_id!r}")
        if not isinstance(args, (list, tuple)) or not isinstance(kwargs, dict):
            raise ValueError(
                f"Golden factory case {case_id!r} needs list/tuple args and dict kwargs"
            )
        cases[case_id] = (tuple(args), kwargs)
    return cases


def prepare_invocation(runtime: Runtime, case_id: str) -> Callable[[], Any]:
    _seed_all(runtime.torch, runtime.seed)
    cases = normalize_factory_cases(runtime.factory(runtime.device), runtime.case_ids)
    actual_ids = set(cases)
    expected_ids = set(runtime.case_ids)
    if actual_ids != expected_ids:
        raise ValueError(
            "Golden factory case ids must exactly match the manifest: "
            f"missing={sorted(expected_ids - actual_ids)}, "
            f"extra={sorted(actual_ids - expected_ids)}"
        )
    args, kwargs = cases[case_id]
    return lambda: runtime.function(*args, **kwargs)


def _create_profiler(torch_npu: Any, output_dir: Path):
    experimental = getattr(torch_npu.profiler, "_ExperimentalConfig")(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    )
    actions = torch_npu.profiler.ProfilerAction

    def one_invocation(step: int):
        return actions.RECORD_AND_SAVE if step == 0 else actions.NONE

    return torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.NPU,
            torch_npu.profiler.ProfilerActivity.CPU,
        ],
        schedule=one_invocation,
        with_stack=False,
        record_shapes=False,
        profile_memory=True,
        experimental_config=experimental,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            str(output_dir), analyse_flag=True,
        ),
    )


def _kernel_details_path(profile_dir: Path) -> Path:
    candidates = [path for path in profile_dir.rglob("kernel_details.csv") if path.is_file()]
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one kernel_details.csv under {profile_dir}, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def sum_kernel_durations(path: Path, device_id: int) -> float:
    total = 0.0
    count = 0
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"Device_id", "Duration(us)"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"kernel CSV missing required columns {sorted(required)}")
        for row_number, row in enumerate(reader, start=2):
            raw_device = str(row.get("Device_id", "")).strip()
            if not re.fullmatch(r"\d+", raw_device) or int(raw_device) != device_id:
                raise ValueError(
                    f"kernel CSV row {row_number} is not from physical NPU {device_id}"
                )
            try:
                duration = float(row["Duration(us)"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid kernel duration at row {row_number}"
                ) from exc
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError(
                    f"kernel duration at row {row_number} must be finite and positive"
                )
            total += duration
            count += 1
    if count == 0 or not math.isfinite(total) or total <= 0:
        raise ValueError("kernel CSV contains no positive NPU kernel durations")
    return total


def profile_invocation(runtime: Runtime, invocation: Callable[[], Any],
                        repeat_dir: Path) -> float:
    """Profile one already-prepared invocation in its own profiler region."""
    repeat_dir.mkdir(parents=True, exist_ok=False)
    runtime.torch_npu.npu.synchronize()
    profiler = _create_profiler(runtime.torch_npu, repeat_dir)
    with profiler:
        invocation()
        runtime.torch_npu.npu.synchronize()
        profiler.step()
    return sum_kernel_durations(
        _kernel_details_path(repeat_dir), runtime.device_id,
    )


def safe_case_dir_name(case_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", case_id).strip("._-") or "case"
    digest = hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:12]
    return f"{slug[:40]}-{digest}"


def collect_case(runtime: Runtime, case: ManifestCase, output_dir: Path,
                  warmup: int, repeats: int) -> dict[str, Any]:
    for _ in range(warmup):
        invocation = prepare_invocation(runtime, case.case_id)
        invocation()
        runtime.torch_npu.npu.synchronize()

    case_root = output_dir / "prof" / "golden-reference" / safe_case_dir_name(case.case_id)
    case_root.mkdir(parents=True, exist_ok=True)
    collection_dir = Path(tempfile.mkdtemp(prefix="collection-", dir=case_root))
    samples = []
    for repeat in range(1, repeats + 1):
        invocation = prepare_invocation(runtime, case.case_id)
        samples.append(profile_invocation(
            runtime, invocation, collection_dir / f"repeat-{repeat:03d}",
        ))
    median_total = float(statistics.median(samples))
    return {
        "id": case.case_id,
        "shape": case.shape,
        "dtype": case.dtype,
        "raw_repeat_e2e_us": samples,
        "median_total_us": median_total,
        "per_iteration_e2e_us": median_total,
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Golden NPU Performance Reference",
        "",
        f"- **Device ID**: {payload['device_id']}",
        "- **Iterations**: 1",
        f"- **Warmup**: {payload['warmup']}",
        f"- **Repeats**: {payload['repeats']}",
        f"- **Seed**: {payload['seed']}",
        "",
        "## Performance Summary",
        "",
        "| case | Input Shape | dtype | Median E2E (us) |",
        "|------|-------------|-------|-----------------|",
    ]
    for case in payload["cases"]:
        lines.append(
            f"| {case['id']} | {case['shape']} | {case['dtype']} | "
            f"{case['median_total_us']:.6f} |"
        )
    lines.extend(["", "## Raw Repeats", ""])
    for case in payload["cases"]:
        values = ", ".join(f"{value:.6f}" for value in case["raw_repeat_e2e_us"])
        lines.append(f"- `{case['id']}`: [{values}] us")
    lines.extend([
        "",
        "All positive NPU kernel durations emitted by the Golden callable are included.",
        "Input preparation and warmup are outside each formal profiler region.",
        "",
    ])
    return "\n".join(lines)


def _atomic_write(path: Path, content: str) -> None:
    descriptor, raw_temp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        os.replace(temp_path, path)
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def _write_reports(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(output_dir / "GOLDEN_PERF_REPORT.md", _render_markdown(payload))
    _atomic_write(
        output_dir / "GOLDEN_PERF_REPORT.json",
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )


def collect(config: Config) -> dict[str, Any]:
    if config.device_id < 0 or config.seed < 0 or config.warmup < 0:
        raise ValueError("device/seed/warmup must be non-negative and repeats must be positive")
    if config.repeats <= 0:
        raise ValueError("device/seed/warmup must be non-negative and repeats must be positive")
    golden_path, _, golden_source = _snapshot_file(config.golden_path, "Golden source")
    cases, manifest_source = _load_manifest(config.case_manifest)
    runtime = load_runtime(config, golden_path, tuple(case.case_id for case in cases))
    collected = [
        collect_case(
            runtime, case, config.output_dir.resolve(), config.warmup, config.repeats,
        )
        for case in cases
    ]
    _verify_unchanged(golden_source, "Golden source")
    _verify_unchanged(manifest_source, "case manifest")
    payload = {
        "schema_version": 1,
        "golden_source": golden_source,
        "case_manifest_source": manifest_source,
        "device_id": config.device_id,
        "iterations": 1,
        "warmup": config.warmup,
        "repeats": config.repeats,
        "seed": config.seed,
        "timing_scope": "all_golden_npu_kernels",
        "aggregation": "median_of_repeat_totals",
        "cases": collected,
    }
    _write_reports(config.output_dir.resolve(), payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect the Golden NPU performance reference contract",
    )
    parser.add_argument("golden", type=Path, help="Generated *_golden.py")
    parser.add_argument("--function", required=True, help="Golden function name")
    parser.add_argument("--factory", required=True, help="Named-case input factory")
    parser.add_argument("--case-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", required=True, type=int)
    parser.add_argument("--warmup", required=True, type=int)
    parser.add_argument("--repeats", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args(argv)
    try:
        collect(Config(
            args.golden, args.function, args.factory, args.case_manifest,
            args.output_dir, args.device, args.warmup, args.repeats, args.seed,
        ))
        return 0
    except Exception as exc:
        _emit_stderr(f"ERROR: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
