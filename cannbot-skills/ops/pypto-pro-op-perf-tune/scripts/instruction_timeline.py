#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Pure parsing and validation helpers for performance instruction timelines."""

import contextlib
import hashlib
import json
import sqlite3
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Tuple


def trace_decimal(value: Any, field: str) -> Decimal:
    """Parse one exported timestamp without losing sub-microsecond digits."""
    if isinstance(value, bool) or value is None:
        raise ValueError(f"trace event {field} must be a finite number")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"trace event {field} must be a finite number") from error
    if not parsed.is_finite():
        raise ValueError(f"trace event {field} must be a finite number")
    return parsed


def trace_events(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        payload = payload.get("traceEvents")
    if not isinstance(payload, list):
        raise ValueError("exported timeline must be a JSON event list or traceEvents object")
    return [event for event in payload if isinstance(event, dict)]


def pipe_class(instruction: str) -> str:
    """Map a BIU instruction label to a diagnostic pipe family."""
    token = str(instruction).strip().upper()
    if token.startswith("MTE") or token.startswith("FIX"):
        return "movement"
    if token.startswith(("VEC", "CUBE", "MAC")) or token == "M":
        return "compute"
    if token in {"SU", "SCALAR", "SCALARLDST"} or token.startswith("SCALAR"):
        return "scalar"
    if token.startswith(("WAIT", "BARRIER", "SYNC")):
        return "wait"
    return "unknown"


def decimal_text(value: Decimal) -> str:
    return format(value, "f")


def merged_decimal_intervals(
        intervals: List[Tuple[Decimal, Decimal]]) -> List[Tuple[Decimal, Decimal]]:
    """Return the union of positive intervals in deterministic order."""
    merged: List[List[Decimal]] = []
    for start, end in sorted(intervals):
        if start >= end:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        elif end > merged[-1][1]:
            merged[-1][1] = end
    return [(start, end) for start, end in merged]


def decimal_interval_intersection_duration(
        left: List[Tuple[Decimal, Decimal]],
        right: List[Tuple[Decimal, Decimal]]) -> Decimal:
    """Measure the intersection of two already-merged interval unions."""
    total = Decimal(0)
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        left_start, left_end = left[left_index]
        right_start, right_end = right[right_index]
        overlap_start = max(left_start, right_start)
        overlap_end = min(left_end, right_end)
        if overlap_start < overlap_end:
            total += overlap_end - overlap_start
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return total


def _find_target_window(
    events: List[Dict[str, Any]], exact_op_name: str
) -> Tuple[Dict[str, Any], Decimal, Decimal]:
    """定位唯一的目标 task 事件并返回其时间窗。"""
    targets = [
        event for event in events
        if event.get("ph") == "X" and event.get("name") == exact_op_name
    ]
    if len(targets) != 1:
        raise ValueError(
            f"expected exactly one target task event for {exact_op_name!r}; "
            f"found {len(targets)}"
        )
    target = targets[0]
    target_start = trace_decimal(target.get("ts"), "ts")
    target_duration = trace_decimal(target.get("dur"), "dur")
    if target_duration <= 0:
        raise ValueError("target task duration must be positive")
    return target, target_start, target_start + target_duration


def _collect_lanes(events: List[Dict[str, Any]]) -> Dict[Tuple[Any, Any], str]:
    """收集 Group 前缀的 BIU 监控 lane 元数据。"""
    lane_names: Dict[Tuple[Any, Any], str] = {}
    for event in events:
        args = event.get("args")
        lane_name = args.get("name") if isinstance(args, dict) else None
        is_meta_event = event.get("ph") == "M" and event.get("name") == "thread_name"
        is_group_lane = isinstance(lane_name, str) and lane_name.startswith("Group")
        if is_meta_event and is_group_lane:
            lane_names[(event.get("pid"), event.get("tid"))] = lane_name
    if not lane_names:
        raise ValueError("exported timeline contains no Group BIU lane metadata")
    return lane_names


def _attribute_events(
    events: List[Dict[str, Any]],
    lane_names: Dict[Tuple[Any, Any], str],
    target_start: Decimal,
    target_end: Decimal,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[Any, Any], List[Dict[str, Any]]]]:
    """把 Group lane 内的指令事件裁剪归因到目标时间窗。"""
    attributed = []
    by_lane: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = {}
    for event in events:
        lane_key = (event.get("pid"), event.get("tid"))
        if event.get("ph") != "X" or lane_key not in lane_names:
            continue
        start = trace_decimal(event.get("ts"), "ts")
        duration = trace_decimal(event.get("dur"), "dur")
        if duration <= 0:
            continue
        end = start + duration
        clipped_start = max(start, target_start)
        clipped_end = min(end, target_end)
        if clipped_start >= clipped_end:
            continue
        record = {
            "lane": lane_names[lane_key],
            "pid": event.get("pid"),
            "tid": event.get("tid"),
            "instruction": str(event.get("name", "")),
            "pipe_class": pipe_class(str(event.get("name", ""))),
            "start_offset_us": decimal_text(clipped_start - target_start),
            "duration_us": decimal_text(clipped_end - clipped_start),
            "args": event.get("args") if isinstance(event.get("args"), dict) else {},
            "_start": clipped_start,
            "_end": clipped_end,
        }
        attributed.append(record)
        by_lane.setdefault(lane_key, []).append(record)
    if not attributed:
        raise ValueError("no BIU instruction events intersect the exact target task window")
    return attributed, by_lane


def _raw_pair_counts(
    lane_events: List[Dict[str, Any]]
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """统计单个 lane 内的原始事件对重叠次数（含全局累计）。"""
    raw_lane_pair_counts: Dict[str, int] = {}
    for left_index, left in enumerate(lane_events):
        for right in lane_events[left_index + 1:]:
            left_class = left["pipe_class"]
            right_class = right["pipe_class"]
            if left_class == right_class or "unknown" in (left_class, right_class):
                continue
            overlap_start = max(left["_start"], right["_start"])
            overlap_end = min(left["_end"], right["_end"])
            if overlap_start >= overlap_end:
                continue
            pair = "+".join(sorted((left_class, right_class)))
            raw_lane_pair_counts[pair] = raw_lane_pair_counts.get(pair, 0) + 1
    return raw_lane_pair_counts, dict(raw_lane_pair_counts)


def _lane_overlap_pairs(
    lane_events: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, Decimal]]:
    """计算单个 lane 的去重管道家族重叠对；返回 (pairs, 全局对计数, 时长增量)。"""
    raw_lane_pair_counts, lane_global_counts = _raw_pair_counts(lane_events)
    family_intervals: Dict[str, List[Tuple[Decimal, Decimal]]] = {}
    for event in lane_events:
        family = event["pipe_class"]
        if family != "unknown":
            family_intervals.setdefault(family, []).append(
                (event["_start"], event["_end"])
            )
    merged_families = {
        family: merged_decimal_intervals(intervals)
        for family, intervals in family_intervals.items()
    }
    lane_pairs = []
    duration_delta: Dict[str, Decimal] = {}
    classes = sorted(merged_families)
    for left_index, left_class in enumerate(classes):
        for right_class in classes[left_index + 1:]:
            pair = "+".join(sorted((left_class, right_class)))
            overlap = decimal_interval_intersection_duration(
                merged_families[left_class], merged_families[right_class]
            )
            if overlap <= 0:
                continue
            lane_pairs.append({
                "pipe_pair": pair,
                "raw_event_pairs": raw_lane_pair_counts.get(pair, 0),
                "deduplicated_overlap_us": decimal_text(overlap),
            })
            duration_delta[pair] = duration_delta.get(pair, Decimal(0)) + overlap
    return lane_pairs, lane_global_counts, duration_delta


def _lane_overlaps(
    by_lane: Dict[Tuple[Any, Any], List[Dict[str, Any]]],
    lane_names: Dict[Tuple[Any, Any], str],
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, Decimal]]:
    """在同一个 lane 内计算管道家族区间的去重重叠。"""
    overlap_by_lane = []
    raw_overlap_pair_counts: Dict[str, int] = {}
    overlap_pair_durations: Dict[str, Decimal] = {}
    for lane_key, lane_events in sorted(
        by_lane.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))
    ):
        lane_pairs, lane_global_counts, duration_delta = _lane_overlap_pairs(
            lane_events
        )
        for pair, count in lane_global_counts.items():
            raw_overlap_pair_counts[pair] = raw_overlap_pair_counts.get(pair, 0) + count
        for pair, duration in duration_delta.items():
            overlap_pair_durations[pair] = (
                overlap_pair_durations.get(pair, Decimal(0)) + duration
            )
        if lane_pairs:
            overlap_by_lane.append({
                "lane": lane_names[lane_key],
                "pid": lane_key[0],
                "tid": lane_key[1],
                "pairs": lane_pairs,
            })
    return overlap_by_lane, raw_overlap_pair_counts, overlap_pair_durations


def _count_events(
    attributed: List[Dict[str, Any]]
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """统计 pipe 家族与指令名的事件数，并清理内部时间戳字段。"""
    event_counts: Dict[str, int] = {}
    instruction_counts: Dict[str, int] = {}
    for record in attributed:
        family = record["pipe_class"]
        event_counts[family] = event_counts.get(family, 0) + 1
        instruction = record["instruction"] or "<empty>"
        instruction_counts[instruction] = instruction_counts.get(instruction, 0) + 1
        record.pop("_start")
        record.pop("_end")
    return event_counts, instruction_counts


class TimelinePayloadInput(NamedTuple):
    """时间线 payload 的组装输入（目标窗口、车道与统计证据）。"""
    exact_op_name: str
    target: Dict[str, Any]
    target_start: Decimal
    target_end: Decimal
    lane_names: Dict[Tuple[Any, Any], str]
    attributed: List[Dict[str, Any]]
    overlap_by_lane: List[Dict[str, Any]]
    raw_overlap_pair_counts: Dict[str, int]
    overlap_pair_durations: Dict[str, Decimal]
    event_counts: Dict[str, int]
    instruction_counts: Dict[str, int]


def _timeline_payload(inputs: TimelinePayloadInput) -> Dict[str, Any]:
    """组装时间线分析的返回字典（结构与字段名保持稳定）。"""
    return {
        "schema_version": 1,
        "target_op_name": inputs.exact_op_name,
        "attribution_status": "unique_exact_target_window",
        "target_task": {
            "pid": inputs.target.get("pid"),
            "tid": inputs.target.get("tid"),
            "start_us": decimal_text(inputs.target_start),
            "duration_us": decimal_text(inputs.target_end - inputs.target_start),
        },
        "evidence_scope": "sampled_representative_biu_lanes",
        "monitored_lanes": [
            {"pid": key[0], "tid": key[1], "name": inputs.lane_names[key]}
            for key in sorted(inputs.lane_names, key=lambda item: (str(item[0]), str(item[1])))
        ],
        "pipe_event_counts": dict(sorted(inputs.event_counts.items())),
        "instruction_event_counts": dict(sorted(inputs.instruction_counts.items())),
        "same_lane_pipe_interval_overlap": {
            "raw_event_pair_counts": dict(sorted(inputs.raw_overlap_pair_counts.items())),
            "raw_event_pair_count_semantics": (
                "overlapping source-event pairs before same-family interval union"
            ),
            "pair_overlap_us": {
                key: decimal_text(value)
                for key, value in sorted(inputs.overlap_pair_durations.items())
            },
            "pair_overlap_semantics": (
                "sum across lanes of each lane's deduplicated pipe-family union intersection"
            ),
            "lanes": inputs.overlap_by_lane,
        },
        "events": inputs.attributed,
        "pipeline_evidence_status": "requires_tile_dag_correlation",
        "completion_eligible": False,
        "limitations": [
            "instruction profiling perturbs timing and does not replace formal compare",
            "BIU events have no Op Name or tile id; attribution uses the unique task window",
            "same-lane interval overlap does not alone prove adjacent-tile double buffering",
            "monitored BIU lanes are representative samples and not proof for every core",
        ],
    }


def analyze_instruction_timeline(payload: Any, exact_op_name: str) -> Dict[str, Any]:
    """Attribute sampled BIU events to one exact target task window.

    BIU events carry neither an Op Name nor a tile id. This parser therefore
    requires one exact target task, clips Group-lane events to its time window,
    and only counts overlap within one ``(pid, tid)`` lane. The result remains
    supporting evidence until it is correlated with the current Tile DAG.
    """
    if not exact_op_name:
        raise ValueError("an exact target Op Name is required for timeline attribution")
    events = trace_events(payload)
    target, target_start, target_end = _find_target_window(events, exact_op_name)
    lane_names = _collect_lanes(events)
    attributed, by_lane = _attribute_events(
        events, lane_names, target_start, target_end
    )
    overlap_by_lane, raw_overlap_pair_counts, overlap_pair_durations = _lane_overlaps(
        by_lane, lane_names
    )
    event_counts, instruction_counts = _count_events(attributed)

    return _timeline_payload(
        TimelinePayloadInput(
            exact_op_name, target, target_start, target_end, lane_names,
            attributed, overlap_by_lane, raw_overlap_pair_counts,
            overlap_pair_durations, event_counts, instruction_counts,
        )
    )


def find_unique_prof_dir(root: Path) -> Path:
    prof_dirs = sorted(path for path in root.glob("PROF_*") if path.is_dir())
    if len(prof_dirs) != 1:
        raise RuntimeError(f"expected exactly one PROF directory; found {len(prof_dirs)}")
    return prof_dirs[0]


def exported_traces(prof_dir: Path) -> Dict[Path, Tuple[int, int, str]]:
    """Snapshot trace identity; content hash catches same-path exports."""
    snapshots: Dict[Path, Tuple[int, int, str]] = {}
    for path in sorted(
            (prof_dir / "mindstudio_profiler_output").glob("msprof_*.json")):
        if not path.is_file() or path.is_symlink():
            continue
        resolved = path.resolve(strict=True)
        before = resolved.stat()
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        after = resolved.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"exported trace changed while being fingerprinted: {resolved}")
        snapshots[resolved] = (after.st_size, after.st_mtime_ns, digest.hexdigest())
    return snapshots


def new_exported_trace(
        prof_dir: Path, before: Dict[Path, Tuple[int, int, str]]) -> Path:
    after = exported_traces(prof_dir)
    removed = sorted(set(before) - set(after), key=str)
    changed = sorted(
        (path for path, fingerprint in after.items() if before.get(path) != fingerprint),
        key=str,
    )
    if removed or len(changed) != 1:
        raise RuntimeError(
            "expected explicit timeline export to create or change exactly one msprof JSON; "
            f"changed={len(changed)}, removed={len(removed)}"
        )
    return changed[0]


def load_exported_trace(path: Path) -> Any:
    """Load exactly one UTF-8 JSON value from a safe exported trace file."""
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"exported timeline is missing or unsafe: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot load exactly one exported timeline JSON: {error}") from error


def validate_biu_database(path: Path) -> Dict[str, Any]:
    required_columns = {
        "group_id", "core_type", "block_id", "instruction",
        "timestamp", "duration", "checkpoint_info",
    }
    try:
        with contextlib.closing(
            sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        ) as connection:
            table_info = connection.execute(
                "PRAGMA table_info(BiuInstrStatus)"
            ).fetchall()
            columns = set()
            for row in table_info:
                columns.add(str(row[1]))
            missing = sorted(required_columns - columns)
            if missing:
                raise RuntimeError(f"biu_perf.db is missing columns: {missing}")
            row_count = int(connection.execute(
                "SELECT COUNT(*) FROM BiuInstrStatus"
            ).fetchone()[0])
    except sqlite3.Error as error:
        raise RuntimeError(f"cannot validate biu_perf.db: {error}") from error
    if row_count <= 0:
        raise RuntimeError("biu_perf.db contains no instruction events")
    return {"row_count": row_count, "columns": sorted(columns)}
