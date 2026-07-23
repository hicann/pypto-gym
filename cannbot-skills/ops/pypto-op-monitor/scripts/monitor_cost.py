#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""pypto-op-monitor — cost extractor for a PyPTO Agent Team run.

The metrics extractor behind the skill: it parses the orchestrator transcript plus
every subagent transcript, charges each wall-clock gap to reasoning, tool execution,
subagent dispatch/lifecycle, setup, user-wait, or idle, and rolls up the token usage
recorded in each assistant event (input / cache read / cache write / output) and the
Read/Write/Edit file footprint (calls, lines, edit ±lines, distinct files) — per agent,
per prompt, per stage, and run-wide.

Two transcript sources are supported behind one report; the default is Claude Code when run
under Claude Code (CLAUDECODE=1), otherwise opencode — override with --source:
  * Claude Code — on-disk JSONL (orchestrator + subagents/*.jsonl); per-event gaps are
    charged to the time buckets.
  * opencode    — `opencode export <sessionID>` JSON with explicit per-message and
    per-tool start/end timing; subagents are child sessions joined via the parent's
    `task` tool metadata.
Both parsers share one record shape and one tool classifier, so every downstream section
is source-agnostic.

The text report presents seven sections, in this order, each carrying BOTH a time and
a token cost (input / cache / output / total) alongside wall and active time:
    1 overall costs                4 reasoning (per agent)
    2 per-stage costs              5 tool calls (per agent)
    3 per-agent costs              6 file interactions (per agent)
                                   7 agent lifecycle (per agent)
`--json` emits the same data (including the per-prompt drill-down) for the skill to
analyse. Stdlib only, Python 3.8+.
See references/log-schema.md for the on-disk layout, timing model, and stage map.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger("pypto-op-monitor")

# --- workflow model -------------------------------------------------------------

STAGE_TITLE = {
    1: "Planning", 2: "Algorithm", 3: "Architecture", 4: "Design",
    5: "Construction", 6: "Verification", 7: "Optimization",
}
ROLE_DEFAULT_STAGE = {
    "pypto-op-planner": 1, "pypto-op-mathematician": 2, "pypto-op-architect": 3,
    "pypto-op-designer": 4, "pypto-op-coder": 5, "pypto-op-debugger": 5,
    "pypto-op-optimizer": 7,
}
SUPPORT = "support"
ORCH = "orchestrator"
# Reasoning / setup gaps longer than this are treated as idle (agent paused, user away
# between turns) rather than model generation. Tool-execution gaps are NEVER capped — a
# verify Bash can legitimately run 30+ min. Override with --idle-threshold.
DEFAULT_IDLE_THRESHOLD_S = 600.0
# Sentinel prompt id for usage/reads seen before any user turn is established.
PREAMBLE = "__preamble__"
OC_SESSION_PREFIX = "ses_"           # opencode session ids look like ses_XXXXXXXX
DEFAULT_OPENCODE_DB = os.path.join(
    os.path.expanduser("~"), ".local", "share", "opencode", "opencode.db")
# Token buckets carried per assistant event: Claude's message.usage keys → short names.
USAGE_FIELDS = (
    ("input_tokens", "input"),
    ("output_tokens", "output"),
    ("cache_read_input_tokens", "cache_read"),
    ("cache_creation_input_tokens", "cache_creation"),
)

# --- tool vocabulary (one classifier, one spec per source) ----------------------
# Claude Code and opencode name the same underlying tools differently (TitleCase vs
# lowercase). A SourceSpec is a source's vocabulary; classify_tool maps any tool name to
# its cost category. Dispatch tools spawn subagents, human tools wait for a reply,
# read/write/edit tools represent file interactions, and all other tools are plain execution.


@dataclass(frozen=True)
class SourceSpec:
    """Tool-name vocabulary for one transcript source."""

    dispatch: set
    human: set
    read: set
    write: set
    edit: set


CLAUDE_SPEC = SourceSpec(
    dispatch={"Task", "Agent"}, human={"AskUserQuestion"},
    read={"Read"}, write={"Write"}, edit={"Edit", "MultiEdit", "NotebookEdit"})
OPENCODE_SPEC = SourceSpec(
    dispatch={"task"}, human=set(), read={"read"}, write={"write"}, edit={"edit"})


def classify_tool(name, spec):
    """Map a tool name to its cost category within a source's vocabulary:
    'dispatch' / 'human' / 'read' / 'write' / 'edit', or 'tool' for plain tool-exec.
    """
    for cat in ("dispatch", "human", "read", "write", "edit"):
        if name in getattr(spec, cat):
            return cat
    return "tool"


@dataclass
class AgentContext:
    """Identity/workflow context for one transcript (who produced it, where it sits)."""

    role: str
    agent_type: str
    description: str
    depth: object
    stage: object
    module: object


@dataclass
class DispatchIndex:
    """Cross-transcript join tables filled while parsing.

    use maps a tool_use id to (tool name, timestamp) of the spawning Agent/Task call;
    res maps a tool_use id to its result timestamp. Together they reconnect each
    subagent to the dispatch call that produced it (used by compute_lifecycle).
    """

    use: dict = field(default_factory=dict)
    res: dict = field(default_factory=dict)


# --- helpers --------------------------------------------------------------------


def _init_logging():
    """Route INFO (the report itself) to stdout and WARNING+ (diagnostics) to stderr,
    both with a bare message format so the emitted text and streams stay unchanged.
    """
    if logger.handlers:
        return
    logger.setLevel(logging.INFO)
    logger.propagate = False
    out = logging.StreamHandler(sys.stdout)
    out.setFormatter(logging.Formatter("%(message)s"))
    out.addFilter(lambda record: record.levelno < logging.WARNING)
    err = logging.StreamHandler(sys.stderr)
    err.setLevel(logging.WARNING)
    err.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(out)
    logger.addHandler(err)


def default_projects_root():
    return os.path.join(os.path.expanduser("~"), ".claude", "projects")


def encode_project_dir(path):
    return re.sub(r"[^A-Za-z0-9]", "-", path)


def parse_ts(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def fmt_dur(seconds):
    if seconds is None:
        return "    n/a"
    s = max(0.0, float(seconds))
    if s < 60:
        return f"{s:5.1f}s"
    total = int(round(s))
    m, sec = divmod(total, 60)
    if m < 60:
        return f"{m:d}m{sec:02d}s"
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m{sec:02d}s"


def fmt_tokens(n):
    """Compact human token/line count: 812, 72.1K, 20.92M, 1.30B."""
    if n is None:
        return "n/a"
    n = float(n)
    if n < 1000:
        return f"{int(round(n))}"
    if n < 1_000_000:
        return f"{n / 1e3:.1f}K"
    if n < 1_000_000_000:
        return f"{n / 1e6:.2f}M"
    return f"{n / 1e9:.2f}B"


def fmt_clock(dt):
    return "--:--:--" if dt is None else dt.astimezone(timezone.utc).strftime("%H:%M:%S")


def short_role(agent_type):
    return (agent_type or "unknown").replace("pypto-op-", "")


def _first_text(content):
    """Best-effort natural-language text of a user turn's initiating event."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                return b.get("text") or ""
    return ""


def _label(s, n=60):
    return " ".join(str(s or "").split())[:n]


def _short_path(fp, n=46):
    parts = str(fp).split("/")
    short = "/".join(parts[-2:]) if len(parts) > 2 else str(fp)
    return short if len(short) <= n else "…" + short[-(n - 1):]


def _patch_counts(patch):
    """Added / removed line counts from a structuredPatch (Edit/Write diff hunks)."""
    if not isinstance(patch, list):
        return 0, 0
    lines = []
    for hunk in patch:
        if isinstance(hunk, dict):
            lines.extend(hunk.get("lines") or [])
    return (
        sum(line.startswith("+") for line in lines),
        sum(line.startswith("-") for line in lines),
    )


def _blank_ops():
    """Per-file interaction counters (reads/writes/edits + their size metrics)."""
    return {"reads": 0, "writes": 0, "edits": 0,
            "read_lines": 0, "read_bytes": 0, "write_lines": 0, "write_bytes": 0,
            "creates": 0, "updates": 0, "added": 0, "removed": 0}


def _new_prompt(pid, source, text, ts, order):
    return {
        "prompt_id": pid, "source": source, "text": _label(text),
        "first_ts": ts, "order": order,
        "tokens": {k: 0 for _, k in USAGE_FIELDS},
        "read_calls": 0, "write_calls": 0, "edit_calls": 0,
        "read_lines": 0, "write_lines": 0, "edit_added": 0, "edit_removed": 0,
        "files": set(),
        "assistant_events": 0,
    }


def resolve_stage(agent_type, description):
    desc = description or ""
    m = re.search(r"Stage\s+(\d+)", desc, re.IGNORECASE)
    stage = int(m.group(1)) if m else ROLE_DEFAULT_STAGE.get(agent_type, SUPPORT)
    mod = re.search(r"\bM(\d+)\b", desc)
    return stage, ("M" + mod.group(1) if mod else None)


def stage_sort_key(key):
    if key == ORCH:
        return (-1, 0)
    if isinstance(key, int):
        return (0, key)
    return (1, 0)


def stage_header(key):
    if key == ORCH:
        return "Orchestrator (main session)"
    if isinstance(key, int):
        return f"Stage {key} · {STAGE_TITLE.get(key, '?')}"
    return "Support (nested helpers)"


def agent_label(m):
    """Compact per-agent label for the token/read tables (stage + role + module)."""
    if m["role"] == ORCH:
        return "orchestrator"
    st = m["stage"]
    tag = f"S{st}" if isinstance(st, int) else "supp"
    mod = f"/{m['module']}" if m["module"] else ""
    return f"{tag} {m['role']}{mod}"


# --- shared record model (both sources fill this identical per-transcript record) ---

def _new_record(path, ctx):
    """Blank per-transcript record. Identical skeleton for every source (Claude /
    opencode) so the downstream aggregation and rendering stay source-agnostic.
    """
    return {
        "path": path, "role": ctx.role, "agent_type": ctx.agent_type,
        "description": ctx.description, "depth": ctx.depth, "stage": ctx.stage,
        "module": ctx.module,
        "start": None, "end": None, "wall_s": None, "active_s": 0.0,
        "reasoning_s": 0.0, "setup_s": 0.0, "toolexec_s": 0.0, "dispatch_wait_s": 0.0,
        "user_wait_s": 0.0, "idle_s": 0.0,
        "tool_time": defaultdict(float), "tool_calls": defaultdict(int),
        "dispatch_calls": 0, "human_wait_calls": 0, "assistant_events": 0,
        "thinking_blocks": 0, "events": 0,
        # token + file-interaction dimensions
        "tokens": {k: 0 for _, k in USAGE_FIELDS},
        "read_calls": 0, "write_calls": 0, "edit_calls": 0,
        "file_ops": defaultdict(_blank_ops),
        "prompts": {}, "prompt_order": [],
    }


def _finalize_record(rec):
    """Close out a parsed record: compute wall/active, freeze the defaultdicts, and roll
    per-file counters into the reads/writes/edits summaries + the per-prompt list. Shared
    by every source so their records are byte-for-byte comparable downstream.
    """
    if rec["start"] and rec["end"]:
        rec["wall_s"] = (rec["end"] - rec["start"]).total_seconds()
    rec["active_s"] = rec["reasoning_s"] + rec["toolexec_s"] + rec["setup_s"]
    rec["tool_time"] = dict(rec["tool_time"])
    rec["tool_calls"] = dict(rec["tool_calls"])
    rec["file_ops"] = dict(rec["file_ops"])
    fo = rec["file_ops"].values()
    rec["files_touched"] = len(rec["file_ops"])
    rec["reads"] = {
        "calls": rec["read_calls"], "files": sum(1 for v in fo if v["reads"]),
        "lines": sum(v["read_lines"] for v in fo), "bytes": sum(v["read_bytes"] for v in fo),
    }
    rec["writes"] = {
        "calls": rec["write_calls"], "files": sum(1 for v in fo if v["writes"]),
        "lines": sum(v["write_lines"] for v in fo), "bytes": sum(v["write_bytes"] for v in fo),
        "creates": sum(v["creates"] for v in fo), "updates": sum(v["updates"] for v in fo),
    }
    rec["edits"] = {
        "calls": rec["edit_calls"], "files": sum(1 for v in fo if v["edits"]),
        "added": sum(v["added"] for v in fo), "removed": sum(v["removed"] for v in fo),
    }
    rec["tokens_total"] = sum(rec["tokens"].values())
    # freeze prompts into an ordered list; convert distinct-file sets to counts
    prompts_list = []
    for pid in rec["prompt_order"]:
        pk = dict(rec["prompts"][pid])
        pk["files"] = len(pk["files"])
        pk["tokens_total"] = sum(pk["tokens"].values())
        prompts_list.append(pk)
    rec["prompts_list"] = prompts_list
    return rec


def _ensure_bucket(rec, pid, source, text, ts):
    """Get (or open) the per-turn (per-prompt) bucket for prompt id `pid`."""
    if pid not in rec["prompts"]:
        rec["prompts"][pid] = _new_prompt(pid, source, text, ts, len(rec["prompt_order"]))
        rec["prompt_order"].append(pid)
    return rec["prompts"][pid]


def _add_tokens(rec, bucket, add):
    """Fold one assistant event's token counts into the record and its turn bucket."""
    for key, val in add.items():
        rec["tokens"][key] += val
        bucket["tokens"][key] += val


def _charge_gap(rec, seconds, threshold, work_key):
    """Charge a positive gap to real work (`work_key`), or to idle beyond the threshold."""
    if seconds <= 0:
        return
    rec["idle_s" if seconds > threshold else work_key] += seconds


def _apply_read(by, bucket, fp, lines, nbytes):
    by["reads"] += 1
    by["read_lines"] += lines
    by["read_bytes"] += nbytes
    bucket["read_lines"] += lines
    bucket["files"].add(fp)


def _apply_write(by, bucket, fp, content, created):
    lines = len(content.splitlines())
    by["writes"] += 1
    by["write_lines"] += lines
    by["write_bytes"] += len(content)
    by["creates" if created else "updates"] += 1
    bucket["write_lines"] += lines
    bucket["files"].add(fp)


def _apply_edit(by, bucket, fp, added, removed):
    by["edits"] += 1
    by["added"] += added
    by["removed"] += removed
    bucket["edit_added"] += added
    bucket["edit_removed"] += removed
    bucket["files"].add(fp)


# --- parse: Claude Code transcript (JSONL; gaps inferred between events) ---------


@dataclass
class ClaudeParseState:
    id2name: dict = field(default_factory=dict)
    previous_ts: object = None
    prompt_id: object = None
    idle_threshold: float = DEFAULT_IDLE_THRESHOLD_S


@dataclass(frozen=True)
class ClaudeBlockContext:
    record: dict
    prompt_id: object
    timestamp: object
    index: object
    state: ClaudeParseState


def _jsonl_events(path):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _index_claude_block(block, context):
    if not isinstance(block, dict):
        return None
    block_type = block.get("type")
    if block_type == "thinking":
        context.record["thinking_blocks"] += 1
        return None
    if block_type == "tool_result":
        result_id = block.get("tool_use_id")
        if result_id and context.timestamp is not None:
            context.index.res[result_id] = context.timestamp
        return result_id
    if block_type != "tool_use":
        return None
    name, tool_id = block.get("name"), block.get("id")
    if tool_id:
        context.state.id2name[tool_id] = name
        if context.timestamp is not None:
            context.index.use[tool_id] = (name, context.timestamp)
    category = classify_tool(name, CLAUDE_SPEC)
    if category in ("read", "write", "edit"):
        context.record[category + "_calls"] += 1
        bucket = _ensure_bucket(
            context.record, context.prompt_id, "startup", "", context.timestamp,
        )
        bucket[category + "_calls"] += 1
    return None


def _index_claude_blocks(content, context):
    result_ids = []
    for block in content if isinstance(content, list) else ():
        result_id = _index_claude_block(block, context)
        if result_id:
            result_ids.append(result_id)
    return result_ids


def _apply_claude_usage(rec, usage, active_pid, ts):
    if not isinstance(usage, dict):
        return
    bucket = _ensure_bucket(rec, active_pid, "startup", "", ts)
    _add_tokens(rec, bucket, {dst: usage.get(src) or 0 for src, dst in USAGE_FIELDS})
    bucket["assistant_events"] += 1


def _apply_claude_file_result(rec, result, active_pid, ts):
    if not isinstance(result, dict):
        return
    if isinstance(result.get("file"), dict):
        bucket = _ensure_bucket(rec, active_pid, "startup", "", ts)
        file_data = result["file"]
        file_path = file_data.get("filePath") or "?"
        _apply_read(rec["file_ops"][file_path], bucket, file_path,
                    int(file_data.get("numLines") or 0), len(file_data.get("content") or ""))
        return
    file_path = result.get("filePath") or "?"
    if "oldString" in result and "filePath" in result:
        bucket = _ensure_bucket(rec, active_pid, "startup", "", ts)
        added, removed = _patch_counts(result.get("structuredPatch"))
        _apply_edit(rec["file_ops"][file_path], bucket, file_path, added, removed)
    elif ("content" in result and "filePath" in result
          and result.get("type") in ("create", "update")):
        bucket = _ensure_bucket(rec, active_pid, "startup", "", ts)
        _apply_write(rec["file_ops"][file_path], bucket, file_path,
                     result.get("content") or "", result.get("type") == "create")


def _update_record_span(rec, ts):
    if ts is None:
        return
    rec["start"] = ts if rec["start"] is None else min(rec["start"], ts)
    rec["end"] = ts if rec["end"] is None else max(rec["end"], ts)


def _charge_claude_result(rec, name, seconds, include_time=True):
    category = classify_tool(name, CLAUDE_SPEC)
    if category == "dispatch":
        rec["dispatch_calls"] += 1
        rec["dispatch_wait_s"] += seconds
    elif category == "human":
        rec["human_wait_calls"] += 1
        rec["user_wait_s"] += seconds
    else:
        rec["tool_calls"][name] += 1
        if include_time:
            rec["toolexec_s"] += seconds
            rec["tool_time"][name] += seconds


def _charge_claude_gap(rec, result_ids, is_assistant, ts, state):
    if state.previous_ts is not None and ts is not None:
        gap = max(0.0, (ts - state.previous_ts).total_seconds())
        if is_assistant:
            _charge_gap(rec, gap, state.idle_threshold, "reasoning_s")
        elif result_ids:
            share = gap / len(result_ids)
            for result_id in result_ids:
                _charge_claude_result(rec, state.id2name.get(result_id, "unknown"), share)
        else:
            _charge_gap(rec, gap, state.idle_threshold, "setup_s")
    elif result_ids:
        for result_id in result_ids:
            _charge_claude_result(
                rec, state.id2name.get(result_id, "unknown"), 0.0, include_time=False,
            )
    if ts is not None:
        state.previous_ts = ts


def parse_transcript(jsonl_path, ctx, index, idle_threshold=DEFAULT_IDLE_THRESHOLD_S):
    """Parse one Claude Code JSONL transcript into the shared time/token/file record;
    mutates the index join tables.
    """
    rec = _new_record(jsonl_path, ctx)
    if not (jsonl_path and os.path.exists(jsonl_path)):
        return _finalize_record(rec)

    state = ClaudeParseState(idle_threshold=idle_threshold)
    for event in _jsonl_events(jsonl_path):
        rec["events"] += 1
        timestamp = parse_ts(event.get("timestamp"))
        message = event.get("message") or {}
        content = message.get("content") if isinstance(message, dict) else None
        usage = message.get("usage") if isinstance(message, dict) else None
        prompt_id = event.get("promptId")
        if prompt_id is not None:
            _ensure_bucket(
                rec, prompt_id, event.get("promptSource"), _first_text(content), timestamp,
            )
            state.prompt_id = prompt_id
        active_pid = state.prompt_id if state.prompt_id is not None else PREAMBLE
        is_assistant = event.get("type") == "assistant"
        block_context = ClaudeBlockContext(
            rec, active_pid, timestamp, index, state,
        )
        result_ids = _index_claude_blocks(content, block_context)
        if is_assistant:
            rec["assistant_events"] += 1
        _apply_claude_usage(rec, usage, active_pid, timestamp)
        _apply_claude_file_result(
            rec, event.get("toolUseResult"), active_pid, timestamp,
        )
        _update_record_span(rec, timestamp)
        _charge_claude_gap(rec, result_ids, is_assistant, timestamp, state)

    return _finalize_record(rec)


# --- parse: opencode session (JSON export; explicit per-message / per-tool timing) ---

def opencode_export(session_id, opencode_bin="opencode"):
    """Return the parsed JSON of `opencode export <session_id>`.

    opencode streams the export to stdout; a large session overruns the OS pipe buffer
    and a piped capture truncates at a 64K/96K boundary, so we redirect to a temp file
    (the shape a shell `>` redirect produces) and read it back.
    """
    fd, path = tempfile.mkstemp(suffix=".json", prefix="ocexport-")
    try:
        with os.fdopen(fd, "wb") as fh:
            subprocess.run([opencode_bin, "export", session_id], stdout=fh,
                           stderr=subprocess.DEVNULL, check=True)
        with open(path, "rb") as fh:
            return json.loads(fh.read().decode("utf-8"))
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _oc_ts(ms):
    """opencode timestamps are epoch milliseconds → tz-aware datetime."""
    if ms is None:
        return None
    try:
        return datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None


def _oc_diff_counts(diff):
    """Added / removed line counts from an opencode edit `metadata.diff` (unified diff)."""
    added = removed = 0
    for ln in (diff or "").splitlines():
        if ln.startswith("+++") or ln.startswith("---"):
            continue
        if ln.startswith("+"):
            added += 1
        elif ln.startswith("-"):
            removed += 1
    return added, removed


def _oc_read_lines(output):
    """Lines returned by an opencode `read`: count the `N: ` line-number prefixes inside
    the tool output's <content> block (opencode does not report a line count separately).
    """
    if not isinstance(output, str):
        return 0
    m = re.search(r"<content>(.*?)</content>", output, re.S)
    body = m.group(1) if m else output
    return len(re.findall(r"(?m)^\s*\d+:", body))


@dataclass
class OpenCodeParseState:
    timestamps: list = field(default_factory=list)
    previous_end: object = None
    prompt_id: object = None
    index: object = None


def _open_opencode_prompt(rec, info, parts, created, state):
    if info.get("role") == "user":
        text = next(
            (part.get("text") or "" for part in parts if part.get("type") == "text"), "",
        )
        state.prompt_id = info.get("id") or PREAMBLE
        _ensure_bucket(rec, state.prompt_id, "user", text, created)
    return state.prompt_id if state.prompt_id is not None else PREAMBLE


def _apply_opencode_usage(rec, role, info, active_pid, created):
    if role != "assistant":
        return
    rec["assistant_events"] += 1
    tokens = info.get("tokens") or {}
    cache = tokens.get("cache") or {}
    bucket = _ensure_bucket(rec, active_pid, "startup", "", created)
    _add_tokens(rec, bucket, {
        "input": tokens.get("input") or 0,
        "output": (tokens.get("output") or 0) + (tokens.get("reasoning") or 0),
        "cache_read": cache.get("read") or 0,
        "cache_creation": cache.get("write") or 0,
    })
    bucket["assistant_events"] += 1


@dataclass(frozen=True)
class OpenCodeToolEvent:
    name: str
    category: str
    duration: float
    state_data: dict
    start: object
    end: object


def _charge_opencode_tool(rec, event, index):
    if event.category == "dispatch":
        rec["dispatch_wait_s"] += event.duration
        rec["dispatch_calls"] += 1
        child = (event.state_data.get("metadata") or {}).get("sessionId")
        if child and event.start and event.end:
            index.use[child] = ("task", event.start)
            index.res[child] = event.end
    elif event.category == "human":
        rec["user_wait_s"] += event.duration
        rec["human_wait_calls"] += 1
    else:
        rec["toolexec_s"] += event.duration
        rec["tool_time"][event.name] += event.duration
        rec["tool_calls"][event.name] += 1


def _apply_opencode_file(rec, category, state_data, active_pid, created):
    file_path = (state_data.get("input") or {}).get("filePath")
    if not file_path or category not in ("read", "write", "edit"):
        return
    rec[category + "_calls"] += 1
    bucket = _ensure_bucket(rec, active_pid, "startup", "", created)
    bucket[category + "_calls"] += 1
    by_file = rec["file_ops"][file_path]
    if category == "read":
        output = state_data.get("output")
        _apply_read(by_file, bucket, file_path, _oc_read_lines(output), len(output or ""))
    elif category == "write":
        content = (state_data.get("input") or {}).get("content") or ""
        created_file = not (state_data.get("metadata") or {}).get("exists")
        _apply_write(by_file, bucket, file_path, content, created_file)
    else:
        added, removed = _oc_diff_counts((state_data.get("metadata") or {}).get("diff"))
        _apply_edit(by_file, bucket, file_path, added, removed)


def _process_opencode_parts(rec, parts, active_pid, created, state):
    tools_duration = 0.0
    for part in parts:
        if part.get("type") == "reasoning":
            rec["thinking_blocks"] += 1
            continue
        if part.get("type") != "tool":
            continue
        name = part.get("tool")
        state_data = part.get("state") or {}
        timing = state_data.get("time") or {}
        start, end = _oc_ts(timing.get("start")), _oc_ts(timing.get("end"))
        duration = max(0.0, (end - start).total_seconds()) if start and end else 0.0
        state.timestamps.extend(value for value in (start, end) if value)
        tools_duration += duration
        category = classify_tool(name, OPENCODE_SPEC)
        event = OpenCodeToolEvent(
            name, category, duration, state_data, start, end,
        )
        _charge_opencode_tool(rec, event, state.index)
        _apply_opencode_file(rec, category, state_data, active_pid, created)
    return tools_duration


def parse_opencode_session(session_json, ctx, index, idle_threshold=DEFAULT_IDLE_THRESHOLD_S):
    """Parse one `opencode export` session into the shared record shape; mutates the index
    join tables (parent `task` tool → child session).
    """
    rec = _new_record("opencode:" + (session_json.get("info", {}) or {}).get("id", "?"), ctx)
    state = OpenCodeParseState(index=index)
    for msg in session_json.get("messages") or []:
        rec["events"] += 1
        info = msg.get("info") or {}
        role = info.get("role")
        mt = info.get("time") or {}
        created, completed = _oc_ts(mt.get("created")), _oc_ts(mt.get("completed"))
        parts = msg.get("parts") or []
        state.timestamps.extend(value for value in (created, completed) if value)
        active_pid = _open_opencode_prompt(rec, info, parts, created, state)
        if state.previous_end is not None and created is not None:
            _charge_gap(
                rec, (created - state.previous_end).total_seconds(),
                idle_threshold, "setup_s",
            )
        _apply_opencode_usage(rec, role, info, active_pid, created)
        tools_duration = _process_opencode_parts(rec, parts, active_pid, created, state)
        if role == "assistant" and created and completed:
            _charge_gap(rec, (completed - created).total_seconds() - tools_duration,
                        idle_threshold, "reasoning_s")
        state.previous_end = completed or created or state.previous_end

    if state.timestamps:
        rec["start"], rec["end"] = min(state.timestamps), max(state.timestamps)
    return _finalize_record(rec)


# --- discovery ------------------------------------------------------------------

def find_sessions(project_path):
    out = []
    for sub in glob.glob(os.path.join(project_path, "*", "subagents")):
        sid = os.path.basename(os.path.dirname(sub))
        try:
            mtime = os.path.getmtime(sub)
        except OSError:
            mtime = 0
        out.append((sid, sub, mtime))
    out.sort(key=lambda t: t[2], reverse=True)
    return out


def load_run(project_path, session_id, subagents_dir, index,
             idle_threshold=DEFAULT_IDLE_THRESHOLD_S):
    """Parse the main transcript + every subagent of one session."""
    records = []
    main_jsonl = os.path.join(project_path, session_id + ".jsonl")
    if os.path.exists(main_jsonl):
        main_ctx = AgentContext(ORCH, ORCH, "orchestrator main session", 0, ORCH, None)
        records.append(parse_transcript(main_jsonl, main_ctx, index, idle_threshold))
    for meta_path in sorted(glob.glob(os.path.join(subagents_dir, "*.meta.json"))):
        base = meta_path[: -len(".meta.json")]
        try:
            with open(meta_path, encoding="utf-8") as meta_fh:
                meta = json.load(meta_fh)
        except (OSError, json.JSONDecodeError):
            meta = {}
        atype = meta.get("agentType", "unknown")
        desc = meta.get("description", "")
        depth = meta.get("spawnDepth")
        stage, module = resolve_stage(atype, desc)
        ctx = AgentContext(short_role(atype), atype, desc, depth, stage, module)
        r = parse_transcript(base + ".jsonl", ctx, index, idle_threshold)
        r["agent_id"] = os.path.basename(base)
        r["tool_use_id"] = meta.get("toolUseId")
        records.append(r)
    return records


def _opencode_tool_parts(session_json):
    for message in session_json.get("messages") or []:
        yield from (
            part for part in message.get("parts") or []
            if part.get("type") == "tool"
        )


def _oc_child_ids(session_json):
    """Child session ids spawned by this session, in dispatch order (parent `task`
    tool → state.metadata.sessionId).
    """
    child_ids = []
    for part in _opencode_tool_parts(session_json):
        if part.get("tool") != "task":
            continue
        child = (part.get("state", {}).get("metadata") or {}).get("sessionId")
        if child:
            child_ids.append(child)
    return child_ids


def load_opencode_run(root_session_id, index, idle_threshold=DEFAULT_IDLE_THRESHOLD_S,
                      opencode_bin="opencode"):
    """Parse an opencode run: the root session as orchestrator, plus every descendant
    session reached through `task`-tool dispatch (BFS, so any nesting is covered).
    """
    root = opencode_export(root_session_id, opencode_bin)
    root_ctx = AgentContext(ORCH, ORCH, "orchestrator main session", 0, ORCH, None)
    records = [parse_opencode_session(root, root_ctx, index, idle_threshold)]
    seen = {root_session_id}
    queue = [(csid, root, 1) for csid in _oc_child_ids(root)]
    while queue:
        csid, _parent, depth = queue.pop(0)
        if csid in seen:
            continue
        seen.add(csid)
        try:
            sj = opencode_export(csid, opencode_bin)
        except Exception as exc:                                  # noqa: BLE001
            logger.warning("[pypto-op-monitor] opencode export %s failed: %s", csid, exc)
            continue
        info = sj.get("info") or {}
        atype = info.get("agent") or "unknown"
        desc = info.get("title") or ""
        stage, module = resolve_stage(atype, desc)
        ctx = AgentContext(short_role(atype), atype, desc, depth, stage, module)
        r = parse_opencode_session(sj, ctx, index, idle_threshold)
        r["agent_id"] = csid
        r["tool_use_id"] = csid          # the child's own id joins it to the parent task
        records.append(r)
        queue.extend((gc, sj, depth + 1) for gc in _oc_child_ids(sj))
    return records


def list_opencode_sessions(db_path=DEFAULT_OPENCODE_DB):
    """List top-level opencode sessions (no parent) with their child count and last-active
    time, newest first. Reads the opencode sqlite store read-only (stdlib sqlite3).
    """
    if not os.path.exists(db_path):
        return None
    con = sqlite3.connect("file:%s?mode=ro" % db_path, uri=True, timeout=5)
    try:
        rows = con.execute(
            "SELECT id, parent_id, title, agent, time_updated FROM session").fetchall()
    finally:
        con.close()
    kids = Counter(r[1] for r in rows if r[1])
    tops = [(r[0], r[2], r[3], r[4]) for r in rows if not r[1]]
    tops.sort(key=lambda r: r[3] or 0, reverse=True)
    return [{"id": sid, "title": title, "agent": agent,
             "children": kids.get(sid, 0), "updated_ms": upd}
            for sid, title, agent, upd in tops]


# --- aggregation ----------------------------------------------------------------

def compute_lifecycle(records, index, tol=5.0):
    """Join each subagent to its spawning dispatch call — Claude `meta.toolUseId` or the
    opencode parent `task` tool, both keyed through `index` — to derive parent-observed
    dispatch duration and, for synchronous dispatch only, spawn+close overhead (async
    dispatch is labelled not-measurable rather than a misleading 0.0s).
    """
    rows = []
    for r in records:
        tid = r.get("tool_use_id")
        if not tid:
            continue
        use = index.use.get(tid)
        end = index.res.get(tid)
        dispatch_s = (end - use[1]).total_seconds() if (use and end) else None
        self_wall = r["wall_s"]
        if dispatch_s is None or self_wall is None:
            mode, overhead = "unknown", None
        elif dispatch_s >= self_wall - tol:
            mode, overhead = "sync", max(0.0, dispatch_s - self_wall)
        else:
            mode, overhead = "async", None   # parent did not block; overhead N/A
        rows.append({
            "agent_id": r.get("agent_id"), "role": r["role"], "module": r["module"],
            "stage": r["stage"], "description": r["description"], "mode": mode,
            "dispatch_s": dispatch_s, "self_wall_s": self_wall, "overhead_s": overhead,
            "active_s": r["active_s"],
            "tokens": r["tokens"], "tokens_total": r["tokens_total"],
        })
    return rows


def _group_records_by_stage(records):
    by_stage = defaultdict(list)
    for record in records:
        by_stage[record["stage"]].append(record)
    return by_stage


def _rollup_tools(records):
    tool_time = defaultdict(float)
    tool_calls = defaultdict(int)
    for record in records:
        for name, duration in record["tool_time"].items():
            tool_time[name] += duration
        for name, count in record["tool_calls"].items():
            tool_calls[name] += count
    rollup = sorted(
        ((name, tool_time[name], tool_calls[name]) for name in tool_calls),
        key=lambda item: item[1], reverse=True,
    )
    return tool_calls, rollup


def _rollup_tokens_and_files(records):
    tokens = {k: 0 for _, k in USAGE_FIELDS}
    file_by_path = defaultdict(_blank_ops)
    for record in records:
        for k in tokens:
            tokens[k] += record["tokens"][k]
        for file_path, values in record["file_ops"].items():
            aggregate = file_by_path[file_path]
            for key, value in values.items():
                aggregate[key] += value
    return tokens, file_by_path


def _file_rollup(records, file_by_path):
    values = file_by_path.values()
    reads = {
        "calls": sum(record["reads"]["calls"] for record in records),
        "files": sum(1 for value in values if value["reads"]),
        "lines": sum(value["read_lines"] for value in values),
        "bytes": sum(value["read_bytes"] for value in values),
    }
    writes = {
        "calls": sum(record["writes"]["calls"] for record in records),
        "files": sum(1 for value in values if value["writes"]),
        "lines": sum(value["write_lines"] for value in values),
        "bytes": sum(value["write_bytes"] for value in values),
        "creates": sum(value["creates"] for value in values),
        "updates": sum(value["updates"] for value in values),
    }
    edits = {
        "calls": sum(record["edits"]["calls"] for record in records),
        "files": sum(1 for value in values if value["edits"]),
        "added": sum(value["added"] for value in values),
        "removed": sum(value["removed"] for value in values),
    }
    top_files = sorted(
        file_by_path.items(),
        key=lambda kv: (kv[1]["reads"] + kv[1]["writes"] + kv[1]["edits"],
                        kv[1]["read_lines"]), reverse=True)[:8]
    return reads, writes, edits, top_files


def _trim_record_end(record, run_end):
    if not record["end"] or record["end"] <= run_end:
        return
    record["end"] = run_end
    if record["start"]:
        record["wall_s"] = (record["end"] - record["start"]).total_seconds()


def _run_window(records):
    starts = [record["start"] for record in records if record["start"]]
    sub_ends = [
        record["end"] for record in records
        if record["end"] and record["stage"] != ORCH
    ]
    all_ends = [record["end"] for record in records if record["end"]]
    run_start = min(starts) if starts else None
    run_end = max(sub_ends) if sub_ends else (max(all_ends) if all_ends else None)
    wall = (run_end - run_start).total_seconds() if (run_start and run_end) else None
    if run_end is not None:
        for record in records:
            _trim_record_end(record, run_end)
    return run_start, run_end, wall


def _stage_row(key, members):
    return {
        "stage": key, "members": members, "agent_count": len(members),
        "reasoning_s": sum(member["reasoning_s"] for member in members),
        "toolexec_s": sum(member["toolexec_s"] for member in members),
        "setup_s": sum(member["setup_s"] for member in members),
        "active_s": sum(member["active_s"] for member in members),
        "wall_s": sum(member["wall_s"] or 0 for member in members),
        "tokens": {
            key: sum(member["tokens"][key] for member in members)
            for _, key in USAGE_FIELDS
        },
        "tokens_total": sum(member["tokens_total"] for member in members),
        "reads_calls": sum(member["reads"]["calls"] for member in members),
        "writes_calls": sum(member["writes"]["calls"] for member in members),
        "edits_calls": sum(member["edits"]["calls"] for member in members),
    }


def _stage_rows(by_stage):
    rows = []
    for key in sorted(by_stage, key=stage_sort_key):
        members = sorted(
            by_stage[key],
            key=lambda member: member["start"] or datetime.max.replace(tzinfo=timezone.utc),
        )
        rows.append(_stage_row(key, members))
    return rows


def _scalar_totals(records):
    keys = (
        "reasoning_s", "toolexec_s", "setup_s", "dispatch_wait_s",
        "user_wait_s", "idle_s", "human_wait_calls", "assistant_events",
        "thinking_blocks",
    )
    return {key: sum(record[key] for record in records) for key in keys}


def build_report(records, index):
    by_stage = _group_records_by_stage(records)
    tool_calls, tool_rollup = _rollup_tools(records)
    tokens, file_by_path = _rollup_tokens_and_files(records)
    reads, writes, edits, top_files = _file_rollup(records, file_by_path)
    scalar = _scalar_totals(records)
    run_start, run_end, wall = _run_window(records)
    stages = _stage_rows(by_stage)

    lifecycle = compute_lifecycle(records, index)
    life_vals = [x["overhead_s"] for x in lifecycle if x["mode"] == "sync"]
    async_count = sum(1 for x in lifecycle if x["mode"] == "async")

    return {
        "stages": stages,
        "tool_rollup": tool_rollup,
        "top_files": top_files,
        "totals": {
            "agent_count": len([r for r in records if r["role"] != ORCH]),
            "transcripts": len(records),
            "reasoning_s": scalar["reasoning_s"], "toolexec_s": scalar["toolexec_s"],
            "setup_s": scalar["setup_s"], "dispatch_wait_s": scalar["dispatch_wait_s"],
            "user_wait_s": scalar["user_wait_s"], "idle_s": scalar["idle_s"],
            "human_wait_calls": scalar["human_wait_calls"],
            "assistant_events": scalar["assistant_events"],
            "thinking_blocks": scalar["thinking_blocks"],
            "wall_s": wall, "start": run_start, "end": run_end,
            "tool_calls_total": sum(tool_calls.values()),
            "tokens": tokens, "tokens_total": sum(tokens.values()),
            "reads": reads, "writes": writes, "edits": edits,
            "files_touched": len(file_by_path),
        },
        "lifecycle": sorted(lifecycle,
                            key=lambda x: (x["overhead_s"] is None, -(x["overhead_s"] or 0))),
        "lifecycle_summary": {
            "joined": len(lifecycle),
            "sync": len(life_vals), "async": async_count,
            "overhead_total_s": sum(life_vals) if life_vals else 0.0,
            "overhead_avg_s": (sum(life_vals) / len(life_vals)) if life_vals else None,
        },
    }


# --- rendering ------------------------------------------------------------------

def _top_tools(rec, k=3):
    items = sorted(rec["tool_time"].items(), key=lambda x: x[1], reverse=True)[:k]
    return ", ".join(f"{n} {fmt_dur(t)}" for n, t in items) if items else "-"


_TOKEN_HEADER = f"{'in':>7} {'cache':>8} {'out':>7} {'total':>8}"
_REPORT_RULE = "=" * 110


def _token_cells(tokens, total):
    cache = (tokens.get("cache_read") or 0) + (tokens.get("cache_creation") or 0)
    return (f"{fmt_tokens(tokens['input']):>7} {fmt_tokens(cache):>8} "
            f"{fmt_tokens(tokens['output']):>7} {fmt_tokens(total):>8}")


def _render_preamble(lines, totals, session_label, location, active_total):
    lines.extend([
        _REPORT_RULE,
        "PyPTO Agent Team — Run Cost Monitor  (time · tokens · file-interactions)",
        f"session : {session_label}", f"location: {location}",
    ])
    if totals["start"] and totals["end"]:
        lines.append(f"window  : {fmt_clock(totals['start'])} → {fmt_clock(totals['end'])} UTC "
                     f"(wall {fmt_dur(totals['wall_s'])} · active {fmt_dur(active_total)})")
    lines.append(f"scope   : {totals['transcripts']} transcripts "
                 f"(1 orchestrator + {totals['agent_count']} subagents)")
    lines.append(_REPORT_RULE)


def _render_overall(lines, totals, active_total):
    tokens = totals["tokens"]
    lines.extend(["", "1  OVERALL COSTS"])
    lines.append(f"    wall {fmt_dur(totals['wall_s'])} (run: start→last stage)    active agent-time "
                 f"{fmt_dur(active_total)}  =  reason {fmt_dur(totals['reasoning_s'])} + tools "
                 f"{fmt_dur(totals['toolexec_s'])} + setup {fmt_dur(totals['setup_s'])}")
    lines.append(f"    tokens {fmt_tokens(totals['tokens_total'])} total  =  input "
                 f"{fmt_tokens(tokens['input'])} + cache-rd {fmt_tokens(tokens['cache_read'])} "
                 f"+ cache-wr {fmt_tokens(tokens['cache_creation'])} + output "
                 f"{fmt_tokens(tokens['output'])}")
    lines.append(f"    file-ops {totals['reads']['calls']}r / {totals['writes']['calls']}w / "
                 f"{totals['edits']['calls']}e over {totals['files_touched']} files  "
                 f"(rd {fmt_tokens(totals['reads']['lines'])} ln · "
                 f"wr {fmt_tokens(totals['writes']['lines'])} ln · ed "
                 f"+{fmt_tokens(totals['edits']['added'])}/-{fmt_tokens(totals['edits']['removed'])} ln)")
    lines.append(f"    {totals['tool_calls_total']} tool calls    {totals['assistant_events']} "
                 f"generation segments    {totals['thinking_blocks']} persisted thinking blocks")
    lines.append(f"    excluded non-work: user_wait {fmt_dur(totals['user_wait_s'])} "
                 f"({totals['human_wait_calls']} human prompts) + idle {fmt_dur(totals['idle_s'])} + "
                 f"dispatch_wait {fmt_dur(totals['dispatch_wait_s'])} (overlaps children)")
    lines.append("    note: cache-rd is cumulative context re-reads (cheap rate), re-counted every "
                 "call — read input+output for genuinely fresh tokens.")


def _render_stages(lines, report, totals, active_total, file_ops_total):
    lines.extend(["", "2  PER-STAGE COSTS"])
    lines.append(f"    {'stage':<28} {'ag':>3} {'reason':>8} {'tools':>8} {'wall':>9} {'active':>8} "
                 f"{'rd/wr/ed':>9}  {_TOKEN_HEADER}")
    for stage in report["stages"]:
        file_ops = f"{stage['reads_calls']}/{stage['writes_calls']}/{stage['edits_calls']}"
        lines.append(f"    {stage_header(stage['stage'])[:28]:<28} {stage['agent_count']:>3} "
                     f"{fmt_dur(stage['reasoning_s']):>8} {fmt_dur(stage['toolexec_s']):>8} "
                     f"{fmt_dur(stage['wall_s']):>9} {fmt_dur(stage['active_s']):>8} "
                     f"{file_ops:>9}  {_token_cells(stage['tokens'], stage['tokens_total'])}")
    lines.append(f"    {'-' * 28} {'-' * 3} {'-' * 8} {'-' * 8} {'-' * 9} {'-' * 8} {'-' * 9}  {'-' * 32}")
    lines.append(f"    {'TOTAL':<28} {totals['transcripts']:>3} {fmt_dur(totals['reasoning_s']):>8} "
                 f"{fmt_dur(totals['toolexec_s']):>8} {fmt_dur(totals['wall_s']):>9} "
                 f"{fmt_dur(active_total):>8} {file_ops_total:>9}  "
                 f"{_token_cells(totals['tokens'], totals['tokens_total'])}")
    lines.append("    stage 'wall' = Σ member spans (they overlap); TOTAL 'wall' = run span "
                 "(first activity → last stage; excludes the orchestrator session's idle tail).")


def _render_agents(lines, members, totals, active_total, file_ops_total):
    lines.extend(["", "3  PER-AGENT COSTS   (active = reasoning + tool-exec + setup)"])
    lines.append(f"    {'agent / stage':<24} {'start':>8} {'wall':>9} {'active':>8} {'calls':>5} "
                 f"{'rd/wr/ed':>9}  {_TOKEN_HEADER}")
    for member in members:
        file_ops = (f"{member['reads']['calls']}/{member['writes']['calls']}/"
                    f"{member['edits']['calls']}")
        lines.append(f"    {agent_label(member)[:24]:<24} {fmt_clock(member['start']):>8} "
                     f"{fmt_dur(member['wall_s']):>9} {fmt_dur(member['active_s']):>8} "
                     f"{sum(member['tool_calls'].values()):>5} {file_ops:>9}  "
                     f"{_token_cells(member['tokens'], member['tokens_total'])}")
    lines.append(f"    {'-' * 24} {'-' * 8} {'-' * 9} {'-' * 8} {'-' * 5} {'-' * 9}  {'-' * 32}")
    lines.append(f"    {'TOTAL':<24} {'':>8} {fmt_dur(totals['wall_s']):>9} "
                 f"{fmt_dur(active_total):>8} {totals['tool_calls_total']:>5} "
                 f"{file_ops_total:>9}  {_token_cells(totals['tokens'], totals['tokens_total'])}")
    lines.append("    wall = per-agent calendar span (agents overlap); active = real compute.")


def _render_reasoning(lines, members, totals, active_total):
    lines.extend(["", "4  REASONING   (reasoning = model-generation latency; the thinking proxy)"])
    lines.append(f"    {'agent / stage':<24} {'wall':>9} {'active':>8} {'reason':>8} {'segs':>5} "
                 f"{'avg/seg':>8} {'think':>5}  {_TOKEN_HEADER}")
    for member in members:
        segments = member["assistant_events"]
        average = member["reasoning_s"] / segments if segments else 0.0
        lines.append(f"    {agent_label(member)[:24]:<24} {fmt_dur(member['wall_s']):>9} "
                     f"{fmt_dur(member['active_s']):>8} {fmt_dur(member['reasoning_s']):>8} "
                     f"{segments:>5} {fmt_dur(average):>8} {member['thinking_blocks']:>5}  "
                     f"{_token_cells(member['tokens'], member['tokens_total'])}")
    average = (totals["reasoning_s"] / totals["assistant_events"]
               if totals["assistant_events"] else 0.0)
    lines.append(f"    {'-' * 24} {'-' * 9} {'-' * 8} {'-' * 8} {'-' * 5} {'-' * 8} {'-' * 5}  {'-' * 32}")
    lines.append(f"    {'TOTAL':<24} {fmt_dur(totals['wall_s']):>9} {fmt_dur(active_total):>8} "
                 f"{fmt_dur(totals['reasoning_s']):>8} {totals['assistant_events']:>5} "
                 f"{fmt_dur(average):>8} {totals['thinking_blocks']:>5}  "
                 f"{_token_cells(totals['tokens'], totals['tokens_total'])}")
    lines.append("    'out' tokens are the reasoning product; `thinking` blocks rarely persist to "
                 f"disk. + setup/injection {fmt_dur(totals['setup_s'])}.")


def _render_tools(lines, report, members, totals, active_total):
    lines.extend(["", "5  TOOL CALLS   (tool-exec time; subagent dispatch excluded → Agent Lifecycle)"])
    lines.append(f"    {'agent / stage':<24} {'wall':>9} {'active':>8} {'tool-exec':>9} {'calls':>5}  "
                 f"{_TOKEN_HEADER}   top tools")
    for member in members:
        calls = sum(member["tool_calls"].values())
        if calls:
            lines.append(f"    {agent_label(member)[:24]:<24} {fmt_dur(member['wall_s']):>9} "
                         f"{fmt_dur(member['active_s']):>8} {fmt_dur(member['toolexec_s']):>9} "
                         f"{calls:>5}  {_token_cells(member['tokens'], member['tokens_total'])}"
                         f"   {_top_tools(member)}")
    lines.append(f"    {'-' * 24} {'-' * 9} {'-' * 8} {'-' * 9} {'-' * 5}  {'-' * 32}")
    lines.append(f"    {'TOTAL':<24} {fmt_dur(totals['wall_s']):>9} {fmt_dur(active_total):>8} "
                 f"{fmt_dur(totals['toolexec_s']):>9} {totals['tool_calls_total']:>5}  "
                 f"{_token_cells(totals['tokens'], totals['tokens_total'])}")
    lines.extend(["", "    run-wide by tool   (calls · total · avg · share of tool-exec):"])
    lines.append(f"      {'tool':<20} {'calls':>6} {'total':>9} {'avg':>8}  share")
    denominator = totals["toolexec_s"] or 1.0
    for name, duration, calls in report["tool_rollup"]:
        average = duration / calls if calls else 0
        share = duration / denominator * 100
        bar = "█" * int(round(share / 5))
        lines.append(f"      {name[:20]:<20} {calls:>6} {fmt_dur(duration):>9} "
                     f"{fmt_dur(average):>8}  {share:4.1f}% {bar}")


def _render_files(lines, report, members, totals, active_total):
    lines.extend(["", "6  FILE INTERACTIONS   (Read / Write / Edit)"])
    lines.append(f"    {'agent / stage':<22} {'wall':>9} {'active':>8} {'rd':>3} {'rd-ln':>6} "
                 f"{'wr':>3} {'wr-ln':>6} {'ed':>3} {'ed ±lines':>11} {'files':>5}  {_TOKEN_HEADER}")
    for member in members:
        calls = member["reads"]["calls"] + member["writes"]["calls"] + member["edits"]["calls"]
        if not calls:
            continue
        edits = f"+{fmt_tokens(member['edits']['added'])}/-{fmt_tokens(member['edits']['removed'])}"
        lines.append(f"    {agent_label(member)[:22]:<22} {fmt_dur(member['wall_s']):>9} "
                     f"{fmt_dur(member['active_s']):>8} {member['reads']['calls']:>3} "
                     f"{fmt_tokens(member['reads']['lines']):>6} {member['writes']['calls']:>3} "
                     f"{fmt_tokens(member['writes']['lines']):>6} {member['edits']['calls']:>3} "
                     f"{edits:>11} {member['files_touched']:>5}  "
                     f"{_token_cells(member['tokens'], member['tokens_total'])}")
    edits = f"+{fmt_tokens(totals['edits']['added'])}/-{fmt_tokens(totals['edits']['removed'])}"
    lines.append(f"    {'-' * 22} {'-' * 9} {'-' * 8} {'-' * 3} {'-' * 6} {'-' * 3} {'-' * 6} "
                 f"{'-' * 3} {'-' * 11} {'-' * 5}  {'-' * 32}")
    lines.append(f"    {'TOTAL':<22} {fmt_dur(totals['wall_s']):>9} {fmt_dur(active_total):>8} "
                 f"{totals['reads']['calls']:>3} {fmt_tokens(totals['reads']['lines']):>6} "
                 f"{totals['writes']['calls']:>3} {fmt_tokens(totals['writes']['lines']):>6} "
                 f"{totals['edits']['calls']:>3} {edits:>11} {totals['files_touched']:>5}  "
                 f"{_token_cells(totals['tokens'], totals['tokens_total'])}")
    lines.append(f"    writes: {totals['writes']['creates']} new + {totals['writes']['updates']} overwrite · "
                 "'files' = distinct paths (per-agent rows re-count shared files; TOTAL run-wide).")
    if report["top_files"]:
        lines.append("    most-touched files (rd/wr/ed):")
        for file_path, values in report["top_files"]:
            lines.append(f"      {values['reads']:>2}r {values['writes']:>2}w {values['edits']:>2}e  "
                         f"{_short_path(file_path)}")


def _render_lifecycle(lines, report, totals):
    summary = report["lifecycle_summary"]
    lines.extend(["", "7  AGENT LIFECYCLE   "
                  "(dispatch + spawn/close overhead; self-wall = the agent's own wall)"])
    lines.append(f"    orchestrator parent-blocked dispatch wait (Σ): "
                 f"{fmt_dur(totals['dispatch_wait_s'])} (overlaps child wall; excluded from active)")
    if not summary["joined"]:
        lines.append("    (no dispatch tool calls joined — single-agent or pre-orchestrator run)")
        return
    lines.append(f"    {'agent / stage':<24} {'mode':>6} {'dispatch':>9} {'self-wall':>9} "
                 f"{'overhead':>9} {'active':>8}  {_TOKEN_HEADER}")
    for item in report["lifecycle"]:
        overhead = fmt_dur(item["overhead_s"]) if item["overhead_s"] is not None else "n/a"
        lines.append(f"    {agent_label(item)[:24]:<24} {item['mode']:>6} "
                     f"{fmt_dur(item['dispatch_s']):>9} {fmt_dur(item['self_wall_s']):>9} "
                     f"{overhead:>9} {fmt_dur(item['active_s']):>8}  "
                     f"{_token_cells(item['tokens'], item['tokens_total'])}")
    overhead_note = (f" · avg {fmt_dur(summary['overhead_avg_s'])}/agent"
                     if summary["sync"] else "")
    lines.append(f"    dispatches joined {summary['joined']} (sync {summary['sync']} · "
                 f"async {summary['async']}); sync spawn+close overhead Σ "
                 f"{fmt_dur(summary['overhead_total_s'])}{overhead_note}")
    if summary["async"]:
        lines.append("    async dispatch = parent did not block → overhead not measurable this "
                     "way (child cost is its own per-agent wall).")


def render_text(report, session_label, location):
    """Render the seven source-agnostic cost sections."""
    totals = report["totals"]
    members = [member for stage in report["stages"] for member in stage["members"]]
    active_total = totals["reasoning_s"] + totals["toolexec_s"] + totals["setup_s"]
    file_ops_total = (f"{totals['reads']['calls']}/{totals['writes']['calls']}/"
                      f"{totals['edits']['calls']}")
    lines = []
    _render_preamble(lines, totals, session_label, location, active_total)
    _render_overall(lines, totals, active_total)
    _render_stages(lines, report, totals, active_total, file_ops_total)
    _render_agents(lines, members, totals, active_total, file_ops_total)
    _render_reasoning(lines, members, totals, active_total)
    _render_tools(lines, report, members, totals, active_total)
    _render_files(lines, report, members, totals, active_total)
    _render_lifecycle(lines, report, totals)
    lines.extend(["", _REPORT_RULE])
    lines.append(f"    {totals['transcripts']} transcripts · {fmt_tokens(totals['tokens_total'])} "
                 f"tokens · active {fmt_dur(active_total)} of {fmt_dur(totals['wall_s'])} "
                 f"wall · file-ops {file_ops_total} over {totals['files_touched']} files")
    lines.append(_REPORT_RULE)
    return "\n".join(lines)


def _iso(timestamp):
    return timestamp.astimezone(timezone.utc).isoformat() if timestamp else None


def _round_metric(value):
    return round(value, 3) if value is not None else None


def _prompt_json(prompt):
    return {
        "index": prompt["order"] + 1, "prompt_id": prompt["prompt_id"],
        "source": prompt["source"], "text": prompt["text"],
        "assistant_generation_segments": prompt["assistant_events"],
        "tokens": {**prompt["tokens"], "total": prompt["tokens_total"]},
        "file_interactions": {
            "files": prompt["files"],
            "reads": {"calls": prompt["read_calls"], "lines": prompt["read_lines"]},
            "writes": {"calls": prompt["write_calls"], "lines": prompt["write_lines"]},
            "edits": {
                "calls": prompt["edit_calls"], "added": prompt["edit_added"],
                "removed": prompt["edit_removed"],
            },
        },
    }


def _agent_json(member):
    tools = sorted(member["tool_time"].items(), key=lambda item: item[1], reverse=True)
    by_use = sorted(
        member["file_ops"].items(),
        key=lambda item: item[1]["reads"] + item[1]["writes"] + item[1]["edits"],
        reverse=True,
    )
    return {
        "agent_id": member.get("agent_id"), "role": member["role"],
        "agent_type": member["agent_type"], "description": member["description"],
        "spawn_depth": member["depth"], "stage": member["stage"],
        "module": member["module"],
        "start": _iso(member["start"]), "end": _iso(member["end"]),
        "wall_seconds": _round_metric(member["wall_s"]),
        "reasoning_seconds": _round_metric(member["reasoning_s"]),
        "toolexec_seconds": _round_metric(member["toolexec_s"]),
        "setup_seconds": _round_metric(member["setup_s"]),
        "dispatch_wait_seconds": _round_metric(member["dispatch_wait_s"]),
        "user_wait_seconds": _round_metric(member["user_wait_s"]),
        "idle_seconds": _round_metric(member["idle_s"]),
        "assistant_generation_segments": member["assistant_events"],
        "thinking_blocks": member["thinking_blocks"],
        "tool_cost": [
            {"tool": name, "calls": member["tool_calls"].get(name, 0),
             "total_seconds": _round_metric(duration)}
            for name, duration in tools
        ],
        "tokens": {**member["tokens"], "total": member["tokens_total"]},
        "file_interactions": {
            "files_touched": member["files_touched"],
            "reads": member["reads"], "writes": member["writes"],
            "edits": member["edits"],
            "by_file": [{"file": file_path, **values} for file_path, values in by_use],
        },
        "prompts": [_prompt_json(prompt) for prompt in member["prompts_list"]],
    }


def _totals_json(totals):
    return {
        "agent_count": totals["agent_count"], "transcripts": totals["transcripts"],
        "wall_seconds": _round_metric(totals["wall_s"]),
        "reasoning_seconds": _round_metric(totals["reasoning_s"]),
        "toolexec_seconds": _round_metric(totals["toolexec_s"]),
        "setup_seconds": _round_metric(totals["setup_s"]),
        "dispatch_wait_seconds": _round_metric(totals["dispatch_wait_s"]),
        "user_wait_seconds": _round_metric(totals["user_wait_s"]),
        "idle_seconds": _round_metric(totals["idle_s"]),
        "human_wait_calls": totals["human_wait_calls"],
        "assistant_generation_segments": totals["assistant_events"],
        "thinking_blocks": totals["thinking_blocks"],
        "tool_calls_total": totals["tool_calls_total"],
        "tokens": {**totals["tokens"], "total": totals["tokens_total"]},
        "file_interactions": {
            "files_touched": totals["files_touched"],
            "reads": totals["reads"], "writes": totals["writes"], "edits": totals["edits"],
        },
        "start": _iso(totals["start"]), "end": _iso(totals["end"]),
    }


def _stage_json(stage):
    return {
        "stage": stage["stage"], "agent_count": stage["agent_count"],
        "reasoning_seconds": _round_metric(stage["reasoning_s"]),
        "toolexec_seconds": _round_metric(stage["toolexec_s"]),
        "setup_seconds": _round_metric(stage["setup_s"]),
        "active_seconds": _round_metric(stage["active_s"]),
        "wall_seconds": _round_metric(stage["wall_s"]),
        "tokens": {**stage["tokens"], "total": stage["tokens_total"]},
        "tokens_total": stage["tokens_total"],
        "file_interactions": {
            "reads": stage["reads_calls"], "writes": stage["writes_calls"],
            "edits": stage["edits_calls"],
        },
        "agents": [_agent_json(member) for member in stage["members"]],
    }


def _lifecycle_json(item):
    return {
        "agent_id": item["agent_id"], "role": item["role"], "module": item["module"],
        "stage": item["stage"], "description": item["description"], "mode": item["mode"],
        "dispatch_seconds": _round_metric(item["dispatch_s"]),
        "self_wall_seconds": _round_metric(item["self_wall_s"]),
        "overhead_seconds": _round_metric(item["overhead_s"]),
        "active_seconds": _round_metric(item["active_s"]),
        "tokens": {**item["tokens"], "total": item["tokens_total"]},
    }


def to_jsonable(report, session_label, location):
    totals = report["totals"]
    summary = report["lifecycle_summary"]
    return {
        "session": session_label, "location": location,
        "totals": _totals_json(totals),
        "tool_cost": [
            {"tool": name, "calls": count, "total_seconds": _round_metric(duration),
             "avg_seconds": _round_metric(duration / count if count else 0)}
            for name, duration, count in report["tool_rollup"]
        ],
        "top_files": [{"file": file_path, **values}
                      for file_path, values in report["top_files"]],
        "lifecycle_summary": {
            "joined": summary["joined"], "sync": summary["sync"], "async": summary["async"],
            "overhead_total_seconds": _round_metric(summary["overhead_total_s"]),
            "overhead_avg_seconds": _round_metric(summary["overhead_avg_s"]),
        },
        "stages": [_stage_json(stage) for stage in report["stages"]],
        "lifecycle": [_lifecycle_json(item) for item in report["lifecycle"]],
    }


# --- main -----------------------------------------------------------------------

def _emit(report, session_label, location, as_json):
    if as_json:
        logger.info("%s", json.dumps(to_jsonable(report, session_label, location),
                                     indent=2, ensure_ascii=False))
    else:
        logger.info("%s", render_text(report, session_label, location))


def _running_under_claude_code():
    """True when this process was spawned by Claude Code, which sets CLAUDECODE=1 in the
    environment of every process it spawns; opencode does not set it. Detecting Claude Code
    positively is the reliable direction, so opencode is the default fallback.
    """
    return os.environ.get("CLAUDECODE") == "1"


def _detect_source(args):
    if args.source in ("claude", "opencode"):
        return args.source
    # An explicit opencode session id is unambiguous.
    if args.session and str(args.session).startswith(OC_SESSION_PREFIX):
        return "opencode"
    # Otherwise detect Claude Code positively (it sets CLAUDECODE=1); everything else —
    # opencode, or no harness at all — falls back to opencode.
    if _running_under_claude_code():
        return "claude"
    return "opencode"


def _run_claude(args):
    project = args.project or encode_project_dir(os.getcwd())
    project_path = os.path.join(args.projects_root, project)
    if not os.path.isdir(project_path):
        logger.error("[pypto-op-monitor] project dir not found: %s", project_path)
        avail = sorted(os.path.basename(d) for d in glob.glob(os.path.join(args.projects_root, "*"))
                       if os.path.isdir(d))
        for a in avail:
            logger.error("    %s", a)
        return 2

    sessions = find_sessions(project_path)
    if args.list_sessions:
        if not sessions:
            logger.info("[pypto-op-monitor] no sessions with subagents under %s", project_path)
            return 0
        for sid, sub, mtime in sessions:
            n = len(glob.glob(os.path.join(sub, "*.meta.json")))
            stamp = datetime.fromtimestamp(mtime, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            logger.info("%s  %3d agents  last-active %sZ", sid, n, stamp)
        return 0

    if not sessions:
        logger.error("[pypto-op-monitor] no subagent logs found under %s", project_path)
        logger.error("[pypto-op-monitor] (a session only gets a subagents/ dir once the orchestrator "
                     "dispatches its first agent)")
        return 1

    index = DispatchIndex()
    if args.all_sessions:
        records = []
        for sid, sub, _ in sessions:
            records.extend(load_run(project_path, sid, sub, index, args.idle_threshold))
        session_label, location = f"ALL ({len(sessions)} sessions)", project_path
    else:
        if args.session:
            match = [s for s in sessions if s[0] == args.session]
            if not match:
                logger.error("[pypto-op-monitor] session %s has no subagents", args.session)
                return 1
            sid, sub, _ = match[0]
        else:
            sid, sub, _ = sessions[0]
        records = load_run(project_path, sid, sub, index, args.idle_threshold)
        session_label, location = sid, sub

    _emit(build_report(records, index), session_label, location, args.json)
    return 0


def _run_opencode(args):
    if args.list_sessions:
        sess = list_opencode_sessions(args.opencode_db)
        if sess is None:
            logger.error("[pypto-op-monitor] opencode db not found: %s", args.opencode_db)
            return 2
        if not sess:
            logger.info("[pypto-op-monitor] no opencode sessions in %s", args.opencode_db)
            return 0
        for s in sess:
            stamp = datetime.fromtimestamp((s["updated_ms"] or 0) / 1000.0,
                                           timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            logger.info("%s  %3d children  %-24s last-active %sZ  %s", s["id"], s["children"],
                        (s["agent"] or "-")[:24], stamp, (s["title"] or "")[:48])
        return 0

    root = args.session
    if not root:
        sess = list_opencode_sessions(args.opencode_db) or []
        runs = [s for s in sess if s["children"] > 0] or sess
        if not runs:
            logger.error("[pypto-op-monitor] no --session given and no opencode run discoverable "
                         "(pass --session ses_... or check --opencode-db)")
            return 1
        root = runs[0]["id"]

    index = DispatchIndex()
    if args.all_sessions:
        sess = list_opencode_sessions(args.opencode_db) or []
        roots = [s["id"] for s in sess if s["children"] > 0] or [s["id"] for s in sess]
        records = []
        for rid in roots:
            records.extend(load_opencode_run(rid, index, args.idle_threshold, args.opencode_bin))
        session_label, location = f"ALL ({len(roots)} opencode runs)", "opencode:" + args.opencode_db
    else:
        records = load_opencode_run(root, index, args.idle_threshold, args.opencode_bin)
        session_label, location = root, "opencode:" + root

    _emit(build_report(records, index), session_label, location, args.json)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Time + token + file-interaction cost extractor for a PyPTO Agent Team run "
                    "(Claude Code or opencode transcripts).")
    p.add_argument("--source", choices=("auto", "claude", "opencode"), default="auto",
                   help="transcript source (default: auto — opencode for a ses_... session; else "
                        "Claude Code when run under Claude Code (CLAUDECODE=1), otherwise opencode)")
    p.add_argument("--projects-root", default=default_projects_root(),
                   help="[claude] logs root (default: ~/.claude/projects)")
    p.add_argument("--project", default=None,
                   help="[claude] project dir under projects-root (default: derived from cwd)")
    p.add_argument("--session", default=None,
                   help="session id (claude: default most-recent with subagents; "
                        "opencode: the run's root ses_... id)")
    p.add_argument("--all-sessions", action="store_true",
                   help="aggregate every session (claude: under the project; opencode: every run)")
    p.add_argument("--list-sessions", action="store_true",
                   help="list sessions/runs, then exit")
    p.add_argument("--json", action="store_true", help="emit JSON instead of a text report")
    p.add_argument("--idle-threshold", type=float, default=DEFAULT_IDLE_THRESHOLD_S,
                   help="seconds; reasoning/setup gaps longer than this are counted as idle "
                        f"rather than work (default {int(DEFAULT_IDLE_THRESHOLD_S)}). Tool "
                        "execution gaps are never capped.")
    p.add_argument("--opencode-bin", default="opencode",
                   help="[opencode] opencode executable used for `opencode export` (default: opencode)")
    p.add_argument("--opencode-db", default=DEFAULT_OPENCODE_DB,
                   help="[opencode] sqlite store used by --list-sessions / run discovery "
                        f"(default: {DEFAULT_OPENCODE_DB})")
    args = p.parse_args(argv)
    _init_logging()

    if _detect_source(args) == "opencode":
        return _run_opencode(args)
    return _run_claude(args)


if __name__ == "__main__":
    raise SystemExit(main())
