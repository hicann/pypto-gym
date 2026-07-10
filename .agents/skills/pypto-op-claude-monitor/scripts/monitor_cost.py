#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""pypto-op-claude-monitor — cost extractor for a PyPTO Agent Team run.

The metrics extractor behind the skill: it parses the orchestrator transcript plus
every subagent transcript, charges each wall-clock gap to reasoning, tool execution,
subagent dispatch/lifecycle, setup, user-wait, or idle, and rolls up the token usage
recorded in each assistant event (input / cache read / cache write / output) and the
Read/Write/Edit file footprint (calls, lines, edit ±lines, distinct files) — per agent,
per prompt, per stage, and run-wide.

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
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger("pypto-op-claude-monitor")

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
DISPATCH_TOOLS = {"Task", "Agent"}
# Tools whose "execution time" is really a human answering — charge to user_wait,
# never to compute tool-exec.
HUMAN_WAIT_TOOLS = {"AskUserQuestion"}
# File-interaction tools. Calls are counted by name (a call with no result still counts);
# the read/write/edit size metrics are resolved from the toolUseResult shape.
FILE_READ_TOOLS = {"Read"}
FILE_WRITE_TOOLS = {"Write"}
FILE_EDIT_TOOLS = {"Edit", "MultiEdit", "NotebookEdit"}
# Reasoning / setup gaps longer than this are treated as idle (agent paused, user away
# between turns) rather than real model generation. Tool-execution gaps are NEVER capped
# here — a verify Bash can legitimately run 30+ min. Override with --idle-threshold.
DEFAULT_IDLE_THRESHOLD_S = 600.0
# The token buckets carried in each assistant event's message.usage; mapped to short keys.
USAGE_FIELDS = (
    ("input_tokens", "input"),
    ("output_tokens", "output"),
    ("cache_read_input_tokens", "cache_read"),
    ("cache_creation_input_tokens", "cache_creation"),
)
# Sentinel prompt id for usage/reads seen before any user turn is established.
PREAMBLE = "__preamble__"


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
    added = removed = 0
    if isinstance(patch, list):
        for hunk in patch:
            for ln in (hunk.get("lines") or []) if isinstance(hunk, dict) else []:
                if ln.startswith("+"):
                    added += 1
                elif ln.startswith("-"):
                    removed += 1
    return added, removed


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


# --- core: parse one transcript -------------------------------------------------

def parse_transcript(jsonl_path, ctx, index, idle_threshold=DEFAULT_IDLE_THRESHOLD_S):
    """Parse one transcript into a time/token/read record; mutates the index join tables."""
    rec = {
        "path": jsonl_path, "role": ctx.role, "agent_type": ctx.agent_type,
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
    if not (jsonl_path and os.path.exists(jsonl_path)):
        rec["file_ops"] = {}
        rec["prompts_list"] = []
        rec["files_touched"] = 0
        rec["reads"] = {"calls": 0, "files": 0, "lines": 0, "bytes": 0}
        rec["writes"] = {"calls": 0, "files": 0, "lines": 0, "bytes": 0,
                         "creates": 0, "updates": 0}
        rec["edits"] = {"calls": 0, "files": 0, "added": 0, "removed": 0}
        rec["tokens_total"] = 0
        return rec

    def ensure_bucket(pid, source, text, ts):
        if pid not in rec["prompts"]:
            rec["prompts"][pid] = _new_prompt(pid, source, text, ts, len(rec["prompt_order"]))
            rec["prompt_order"].append(pid)
        return rec["prompts"][pid]

    id2name = {}            # tool_use id -> tool name (within this transcript)
    prev_ts = None
    cur_pid = None          # promptId of the turn currently in flight
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec["events"] += 1
            ts = parse_ts(ev.get("timestamp"))
            typ = ev.get("type")
            msg = ev.get("message") or {}
            content = msg.get("content") if isinstance(msg, dict) else None
            usage = msg.get("usage") if isinstance(msg, dict) else None
            pid = ev.get("promptId")
            tur = ev.get("toolUseResult")

            # a user event carrying a promptId opens (or continues) a turn
            if pid is not None:
                ensure_bucket(pid, ev.get("promptSource"), _first_text(content), ts)
                cur_pid = pid
            active_pid = cur_pid if cur_pid is not None else PREAMBLE

            # index tool_use / tool_result; collect block stats; count Read calls
            result_ids = []
            is_assistant = (typ == "assistant")
            if isinstance(content, list):
                for b in content:
                    if not isinstance(b, dict):
                        continue
                    bt = b.get("type")
                    if bt == "tool_use":
                        name = b.get("name")
                        tid = b.get("id")
                        if tid:
                            id2name[tid] = name
                            if ts is not None:
                                index.use[tid] = (name, ts)
                        if name in FILE_READ_TOOLS:
                            rec["read_calls"] += 1
                            ensure_bucket(active_pid, "startup", "", ts)["read_calls"] += 1
                        elif name in FILE_WRITE_TOOLS:
                            rec["write_calls"] += 1
                            ensure_bucket(active_pid, "startup", "", ts)["write_calls"] += 1
                        elif name in FILE_EDIT_TOOLS:
                            rec["edit_calls"] += 1
                            ensure_bucket(active_pid, "startup", "", ts)["edit_calls"] += 1
                    elif bt == "tool_result":
                        rid = b.get("tool_use_id")
                        if rid:
                            result_ids.append(rid)
                            if ts is not None:
                                index.res[rid] = ts
                    elif bt == "thinking":
                        rec["thinking_blocks"] += 1
            if is_assistant:
                rec["assistant_events"] += 1

            # token usage → per-record and per-prompt
            if isinstance(usage, dict):
                bucket = ensure_bucket(active_pid, "startup", "", ts)
                for src_key, dst_key in USAGE_FIELDS:
                    v = usage.get(src_key) or 0
                    rec["tokens"][dst_key] += v
                    bucket["tokens"][dst_key] += v
                bucket["assistant_events"] += 1

            # file-interaction footprint (read / write / edit) → per-record + per-prompt.
            # The op is resolved from the toolUseResult shape (Write also carries a
            # structuredPatch, so check its oldString-free create/update shape distinctly).
            if isinstance(tur, dict):
                op = fp = None
                if isinstance(tur.get("file"), dict):                       # Read
                    op, f = "read", tur["file"]
                    fp = f.get("filePath") or "?"
                    lines_n, bytes_n = int(f.get("numLines") or 0), len(f.get("content") or "")
                elif "oldString" in tur and "filePath" in tur:             # Edit
                    op, fp = "edit", tur.get("filePath") or "?"
                    added_n, removed_n = _patch_counts(tur.get("structuredPatch"))
                elif "content" in tur and "filePath" in tur \
                        and tur.get("type") in ("create", "update"):        # Write
                    op, fp = "write", tur.get("filePath") or "?"
                    wc = tur.get("content") or ""
                    lines_n, bytes_n = len(wc.splitlines()), len(wc)
                if op:
                    by = rec["file_ops"][fp]
                    bucket = ensure_bucket(active_pid, "startup", "", ts)
                    bucket["files"].add(fp)
                    if op == "read":
                        by["reads"] += 1
                        by["read_lines"] += lines_n
                        by["read_bytes"] += bytes_n
                        bucket["read_lines"] += lines_n
                    elif op == "write":
                        by["writes"] += 1
                        by["write_lines"] += lines_n
                        by["write_bytes"] += bytes_n
                        by["creates" if tur.get("type") == "create" else "updates"] += 1
                        bucket["write_lines"] += lines_n
                    else:  # edit
                        by["edits"] += 1
                        by["added"] += added_n
                        by["removed"] += removed_n
                        bucket["edit_added"] += added_n
                        bucket["edit_removed"] += removed_n

            # time bookkeeping
            if ts is not None:
                if rec["start"] is None or ts < rec["start"]:
                    rec["start"] = ts
                if rec["end"] is None or ts > rec["end"]:
                    rec["end"] = ts

            # charge the gap (prev -> this) to the right bucket
            if prev_ts is not None and ts is not None:
                gap = (ts - prev_ts).total_seconds()
                if gap < 0:
                    gap = 0.0
                if is_assistant:
                    # model generation; an over-threshold "reasoning" gap is really idle
                    if gap > idle_threshold:
                        rec["idle_s"] += gap
                    else:
                        rec["reasoning_s"] += gap
                elif result_ids:
                    # split the gap across results in this event (usually exactly one)
                    share = gap / len(result_ids)
                    for rid in result_ids:
                        name = id2name.get(rid, "unknown")
                        if name in DISPATCH_TOOLS:
                            rec["dispatch_wait_s"] += share
                            rec["dispatch_calls"] += 1
                        elif name in HUMAN_WAIT_TOOLS:
                            # time spent waiting on a human, not compute
                            rec["user_wait_s"] += share
                            rec["human_wait_calls"] += 1
                        else:
                            # real tool execution — never capped (verifies run long)
                            rec["toolexec_s"] += share
                            rec["tool_time"][name] += share
                            rec["tool_calls"][name] += 1
                else:
                    # gap before a plain user prompt / attachment; big ones are human idle
                    if gap > idle_threshold:
                        rec["idle_s"] += gap
                    else:
                        rec["setup_s"] += gap
            else:
                # still count the call even if untimed
                for rid in result_ids:
                    name = id2name.get(rid, "unknown")
                    if name in DISPATCH_TOOLS:
                        rec["dispatch_calls"] += 1
                    elif name in HUMAN_WAIT_TOOLS:
                        rec["human_wait_calls"] += 1
                    else:
                        rec["tool_calls"][name] += 1
            if ts is not None:
                prev_ts = ts

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


# --- aggregation ----------------------------------------------------------------

def compute_lifecycle(records, index, tol=5.0):
    """Join each subagent to its spawning Agent/Task call (via meta.toolUseId) to derive
    parent-observed dispatch duration and, for synchronous dispatch only, spawn+close
    overhead (async dispatch is labelled not-measurable rather than a misleading 0.0s).
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


def build_report(records, index):
    by_stage = defaultdict(list)
    for r in records:
        by_stage[r["stage"]].append(r)

    # run-wide tool rollup (regular tools only; dispatch handled separately)
    tool_time = defaultdict(float)
    tool_calls = defaultdict(int)
    for r in records:
        for name, t in r["tool_time"].items():
            tool_time[name] += t
        for name, c in r["tool_calls"].items():
            tool_calls[name] += c

    # run-wide token + file-interaction rollup
    tokens = {k: 0 for _, k in USAGE_FIELDS}
    file_by_path = defaultdict(_blank_ops)
    for r in records:
        for k in tokens:
            tokens[k] += r["tokens"][k]
        for fp, v in r["file_ops"].items():
            a = file_by_path[fp]
            for kk in v:
                a[kk] += v[kk]
    fv = file_by_path.values()
    reads_t = {"calls": sum(r["reads"]["calls"] for r in records),
               "files": sum(1 for v in fv if v["reads"]),
               "lines": sum(v["read_lines"] for v in fv), "bytes": sum(v["read_bytes"] for v in fv)}
    writes_t = {"calls": sum(r["writes"]["calls"] for r in records),
                "files": sum(1 for v in fv if v["writes"]),
                "lines": sum(v["write_lines"] for v in fv), "bytes": sum(v["write_bytes"] for v in fv),
                "creates": sum(v["creates"] for v in fv), "updates": sum(v["updates"] for v in fv)}
    edits_t = {"calls": sum(r["edits"]["calls"] for r in records),
               "files": sum(1 for v in fv if v["edits"]),
               "added": sum(v["added"] for v in fv), "removed": sum(v["removed"] for v in fv)}
    top_files = sorted(
        file_by_path.items(),
        key=lambda kv: (kv[1]["reads"] + kv[1]["writes"] + kv[1]["edits"],
                        kv[1]["read_lines"]), reverse=True)[:8]

    reasoning_total = sum(r["reasoning_s"] for r in records)
    toolexec_total = sum(r["toolexec_s"] for r in records)
    setup_total = sum(r["setup_s"] for r in records)
    dispatch_wait_total = sum(r["dispatch_wait_s"] for r in records)
    user_wait_total = sum(r["user_wait_s"] for r in records)
    idle_total = sum(r["idle_s"] for r in records)
    human_wait_calls = sum(r["human_wait_calls"] for r in records)
    assistant_events = sum(r["assistant_events"] for r in records)
    thinking_blocks = sum(r["thinking_blocks"] for r in records)

    starts = [r["start"] for r in records if r["start"]]
    ends = [r["end"] for r in records if r["end"]]
    wall = (max(ends) - min(starts)).total_seconds() if starts and ends else None

    stages = []
    for key in sorted(by_stage, key=stage_sort_key):
        members = sorted(by_stage[key],
                         key=lambda a: a["start"] or datetime.max.replace(tzinfo=timezone.utc))
        stages.append({
            "stage": key, "members": members, "agent_count": len(members),
            "reasoning_s": sum(m["reasoning_s"] for m in members),
            "toolexec_s": sum(m["toolexec_s"] for m in members),
            "setup_s": sum(m["setup_s"] for m in members),
            "active_s": sum(m["active_s"] for m in members),
            "wall_s": sum(m["wall_s"] or 0 for m in members),
            "tokens": {k: sum(m["tokens"][k] for m in members) for _, k in USAGE_FIELDS},
            "tokens_total": sum(m["tokens_total"] for m in members),
            "reads_calls": sum(m["reads"]["calls"] for m in members),
            "writes_calls": sum(m["writes"]["calls"] for m in members),
            "edits_calls": sum(m["edits"]["calls"] for m in members),
        })

    lifecycle = compute_lifecycle(records, index)
    life_vals = [x["overhead_s"] for x in lifecycle if x["mode"] == "sync"]
    async_count = sum(1 for x in lifecycle if x["mode"] == "async")

    return {
        "stages": stages,
        "tool_rollup": sorted(((n, tool_time[n], tool_calls[n]) for n in tool_calls),
                              key=lambda x: x[1], reverse=True),
        "top_files": top_files,
        "totals": {
            "agent_count": len([r for r in records if r["role"] != ORCH]),
            "transcripts": len(records),
            "reasoning_s": reasoning_total, "toolexec_s": toolexec_total,
            "setup_s": setup_total, "dispatch_wait_s": dispatch_wait_total,
            "user_wait_s": user_wait_total, "idle_s": idle_total,
            "human_wait_calls": human_wait_calls,
            "assistant_events": assistant_events, "thinking_blocks": thinking_blocks,
            "wall_s": wall, "start": min(starts) if starts else None,
            "end": max(ends) if ends else None,
            "tool_calls_total": sum(tool_calls.values()),
            "tokens": tokens, "tokens_total": sum(tokens.values()),
            "reads": reads_t, "writes": writes_t, "edits": edits_t,
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


def render_text(report, session_label, location):
    """Seven sections in report order, each carrying BOTH a time and a token cost:
    overall costs, per-stage costs, per-agent costs, then the per-agent reasoning /
    tool-call / file-interaction / lifecycle breakdowns. Every per-agent and per-stage
    row shows wall + active time and an input/cache/output/total token split; the
    overall section keeps the full cache-read vs cache-write split. Per-prompt detail
    lives in --json.
    """
    t = report["totals"]
    members_flat = [m for st in report["stages"] for m in st["members"]]
    active_t = t["reasoning_s"] + t["toolexec_s"] + t["setup_s"]

    def cache_of(tk):
        return (tk.get("cache_read") or 0) + (tk.get("cache_creation") or 0)

    th = f"{'in':>7} {'cache':>8} {'out':>7} {'total':>8}"          # token-block header

    def tc(tk, total):                                             # token-block cells
        return (f"{fmt_tokens(tk['input']):>7} {fmt_tokens(cache_of(tk)):>8} "
                f"{fmt_tokens(tk['output']):>7} {fmt_tokens(total):>8}")

    rule = "=" * 110
    lines = [rule,
             "PyPTO Agent Team — Run Cost Monitor  (time · tokens · file-interactions)",
             f"session : {session_label}",
             f"location: {location}"]
    if t["start"] and t["end"]:
        lines.append(f"window  : {fmt_clock(t['start'])} → {fmt_clock(t['end'])} UTC "
                     f"(wall {fmt_dur(t['wall_s'])} · active {fmt_dur(active_t)})")
    lines.append(f"scope   : {t['transcripts']} transcripts "
                 f"(1 orchestrator + {t['agent_count']} subagents)")
    lines.append(rule)

    # ---- 1  OVERALL COSTS --------------------------------------------------------
    tk = t["tokens"]
    lines.append("")
    lines.append("1  OVERALL COSTS")
    lines.append(f"    wall {fmt_dur(t['wall_s'])} (calendar span)    active agent-time "
                 f"{fmt_dur(active_t)}  =  reason {fmt_dur(t['reasoning_s'])} + tools "
                 f"{fmt_dur(t['toolexec_s'])} + setup {fmt_dur(t['setup_s'])}")
    lines.append(f"    tokens {fmt_tokens(t['tokens_total'])} total  =  input {fmt_tokens(tk['input'])} "
                 f"+ cache-rd {fmt_tokens(tk['cache_read'])} + cache-wr {fmt_tokens(tk['cache_creation'])} "
                 f"+ output {fmt_tokens(tk['output'])}")
    lines.append(f"    file-ops {t['reads']['calls']}r / {t['writes']['calls']}w / {t['edits']['calls']}e "
                 f"over {t['files_touched']} files  (rd {fmt_tokens(t['reads']['lines'])} ln · "
                 f"wr {fmt_tokens(t['writes']['lines'])} ln · ed +{fmt_tokens(t['edits']['added'])}"
                 f"/-{fmt_tokens(t['edits']['removed'])} ln)")
    lines.append(f"    {t['tool_calls_total']} tool calls    {t['assistant_events']} generation "
                 f"segments    {t['thinking_blocks']} persisted thinking blocks")
    lines.append(f"    excluded non-work: user_wait {fmt_dur(t['user_wait_s'])} "
                 f"({t['human_wait_calls']} human prompts) + idle {fmt_dur(t['idle_s'])} + "
                 f"dispatch_wait {fmt_dur(t['dispatch_wait_s'])} (overlaps children)")
    lines.append("    note: cache-rd is cumulative context re-reads (cheap rate), re-counted every "
                 "call — read input+output for genuinely fresh tokens.")

    # ---- 2  PER-STAGE COSTS ------------------------------------------------------
    fo_t = f"{t['reads']['calls']}/{t['writes']['calls']}/{t['edits']['calls']}"
    lines.append("")
    lines.append("2  PER-STAGE COSTS")
    lines.append(f"    {'stage':<28} {'ag':>3} {'reason':>8} {'tools':>8} {'wall':>9} {'active':>8} "
                 f"{'rd/wr/ed':>9}  {th}")
    for st in report["stages"]:
        fo = f"{st['reads_calls']}/{st['writes_calls']}/{st['edits_calls']}"
        lines.append(f"    {stage_header(st['stage'])[:28]:<28} {st['agent_count']:>3} "
                     f"{fmt_dur(st['reasoning_s']):>8} {fmt_dur(st['toolexec_s']):>8} "
                     f"{fmt_dur(st['wall_s']):>9} {fmt_dur(st['active_s']):>8} {fo:>9}  "
                     f"{tc(st['tokens'], st['tokens_total'])}")
    lines.append(f"    {'-' * 28} {'-' * 3} {'-' * 8} {'-' * 8} {'-' * 9} {'-' * 8} {'-' * 9}  {'-' * 32}")
    lines.append(f"    {'TOTAL':<28} {t['transcripts']:>3} {fmt_dur(t['reasoning_s']):>8} "
                 f"{fmt_dur(t['toolexec_s']):>8} {fmt_dur(t['wall_s']):>9} {fmt_dur(active_t):>8} "
                 f"{fo_t:>9}  {tc(t['tokens'], t['tokens_total'])}")
    lines.append("    stage 'wall' = Σ member spans (they overlap); TOTAL 'wall' = calendar span.")

    # ---- 3  PER-AGENT COSTS ------------------------------------------------------
    lines.append("")
    lines.append("3  PER-AGENT COSTS   (active = reasoning + tool-exec + setup)")
    lines.append(f"    {'agent / stage':<24} {'start':>8} {'wall':>9} {'active':>8} {'calls':>5} "
                 f"{'rd/wr/ed':>9}  {th}")
    for m in members_flat:
        active = m["active_s"]
        fo = f"{m['reads']['calls']}/{m['writes']['calls']}/{m['edits']['calls']}"
        lines.append(f"    {agent_label(m)[:24]:<24} {fmt_clock(m['start']):>8} "
                     f"{fmt_dur(m['wall_s']):>9} {fmt_dur(active):>8} "
                     f"{sum(m['tool_calls'].values()):>5} {fo:>9}  {tc(m['tokens'], m['tokens_total'])}")
    lines.append(f"    {'-' * 24} {'-' * 8} {'-' * 9} {'-' * 8} {'-' * 5} {'-' * 9}  {'-' * 32}")
    lines.append(f"    {'TOTAL':<24} {'':>8} {fmt_dur(t['wall_s']):>9} {fmt_dur(active_t):>8} "
                 f"{t['tool_calls_total']:>5} {fo_t:>9}  {tc(t['tokens'], t['tokens_total'])}")
    lines.append("    wall = per-agent calendar span (agents overlap); active = real compute.")

    # ---- 4  REASONING ------------------------------------------------------------
    lines.append("")
    lines.append("4  REASONING   (reasoning = model-generation latency; the thinking proxy)")
    lines.append(f"    {'agent / stage':<24} {'wall':>9} {'active':>8} {'reason':>8} {'segs':>5} "
                 f"{'avg/seg':>8} {'think':>5}  {th}")
    for m in members_flat:
        active = m["active_s"]
        segs = m["assistant_events"]
        avg = m["reasoning_s"] / segs if segs else 0.0
        lines.append(f"    {agent_label(m)[:24]:<24} {fmt_dur(m['wall_s']):>9} {fmt_dur(active):>8} "
                     f"{fmt_dur(m['reasoning_s']):>8} {segs:>5} {fmt_dur(avg):>8} "
                     f"{m['thinking_blocks']:>5}  {tc(m['tokens'], m['tokens_total'])}")
    avg_seg = (t["reasoning_s"] / t["assistant_events"]) if t["assistant_events"] else 0.0
    lines.append(f"    {'-' * 24} {'-' * 9} {'-' * 8} {'-' * 8} {'-' * 5} {'-' * 8} {'-' * 5}  {'-' * 32}")
    lines.append(f"    {'TOTAL':<24} {fmt_dur(t['wall_s']):>9} {fmt_dur(active_t):>8} "
                 f"{fmt_dur(t['reasoning_s']):>8} {t['assistant_events']:>5} {fmt_dur(avg_seg):>8} "
                 f"{t['thinking_blocks']:>5}  {tc(t['tokens'], t['tokens_total'])}")
    lines.append("    'out' tokens are the reasoning product; `thinking` blocks rarely persist to "
                 f"disk. + setup/injection {fmt_dur(t['setup_s'])}.")

    # ---- 5  TOOL CALLS -----------------------------------------------------------
    lines.append("")
    lines.append("5  TOOL CALLS   (tool-exec time; subagent dispatch excluded → Agent Lifecycle)")
    lines.append(f"    {'agent / stage':<24} {'wall':>9} {'active':>8} {'tool-exec':>9} {'calls':>5}  "
                 f"{th}   top tools")
    for m in members_flat:
        calls = sum(m["tool_calls"].values())
        if calls == 0:
            continue
        active = m["active_s"]
        lines.append(f"    {agent_label(m)[:24]:<24} {fmt_dur(m['wall_s']):>9} {fmt_dur(active):>8} "
                     f"{fmt_dur(m['toolexec_s']):>9} {calls:>5}  {tc(m['tokens'], m['tokens_total'])}"
                     f"   {_top_tools(m)}")
    lines.append(f"    {'-' * 24} {'-' * 9} {'-' * 8} {'-' * 9} {'-' * 5}  {'-' * 32}")
    lines.append(f"    {'TOTAL':<24} {fmt_dur(t['wall_s']):>9} {fmt_dur(active_t):>8} "
                 f"{fmt_dur(t['toolexec_s']):>9} {t['tool_calls_total']:>5}  "
                 f"{tc(t['tokens'], t['tokens_total'])}")
    lines.append("")
    lines.append("    run-wide by tool   (calls · total · avg · share of tool-exec):")
    lines.append(f"      {'tool':<20} {'calls':>6} {'total':>9} {'avg':>8}  share")
    tot = t["toolexec_s"] or 1.0
    for name, ttime, calls in report["tool_rollup"]:
        avg = ttime / calls if calls else 0
        share = ttime / tot * 100
        bar = "█" * int(round(share / 5))
        lines.append(f"      {name[:20]:<20} {calls:>6} {fmt_dur(ttime):>9} "
                     f"{fmt_dur(avg):>8}  {share:4.1f}% {bar}")

    # ---- 6  FILE INTERACTIONS ----------------------------------------------------
    lines.append("")
    lines.append("6  FILE INTERACTIONS   (Read / Write / Edit)")
    lines.append(f"    {'agent / stage':<22} {'wall':>9} {'active':>8} {'rd':>3} {'rd-ln':>6} "
                 f"{'wr':>3} {'wr-ln':>6} {'ed':>3} {'ed ±lines':>11} {'files':>5}  {th}")
    for m in members_flat:
        if m["reads"]["calls"] == 0 and m["writes"]["calls"] == 0 and m["edits"]["calls"] == 0:
            continue
        active = m["active_s"]
        ed = f"+{fmt_tokens(m['edits']['added'])}/-{fmt_tokens(m['edits']['removed'])}"
        lines.append(f"    {agent_label(m)[:22]:<22} {fmt_dur(m['wall_s']):>9} {fmt_dur(active):>8} "
                     f"{m['reads']['calls']:>3} {fmt_tokens(m['reads']['lines']):>6} "
                     f"{m['writes']['calls']:>3} {fmt_tokens(m['writes']['lines']):>6} "
                     f"{m['edits']['calls']:>3} {ed:>11} {m['files_touched']:>5}  "
                     f"{tc(m['tokens'], m['tokens_total'])}")
    ed_t = f"+{fmt_tokens(t['edits']['added'])}/-{fmt_tokens(t['edits']['removed'])}"
    lines.append(f"    {'-' * 22} {'-' * 9} {'-' * 8} {'-' * 3} {'-' * 6} {'-' * 3} {'-' * 6} "
                 f"{'-' * 3} {'-' * 11} {'-' * 5}  {'-' * 32}")
    lines.append(f"    {'TOTAL':<22} {fmt_dur(t['wall_s']):>9} {fmt_dur(active_t):>8} "
                 f"{t['reads']['calls']:>3} {fmt_tokens(t['reads']['lines']):>6} "
                 f"{t['writes']['calls']:>3} {fmt_tokens(t['writes']['lines']):>6} "
                 f"{t['edits']['calls']:>3} {ed_t:>11} {t['files_touched']:>5}  "
                 f"{tc(t['tokens'], t['tokens_total'])}")
    lines.append(f"    writes: {t['writes']['creates']} new + {t['writes']['updates']} overwrite · "
                 f"'files' = distinct paths (per-agent rows re-count shared files; TOTAL run-wide).")
    if report["top_files"]:
        lines.append("    most-touched files (rd/wr/ed):")
        for fp, v in report["top_files"]:
            lines.append(f"      {v['reads']:>2}r {v['writes']:>2}w {v['edits']:>2}e  {_short_path(fp)}")

    # ---- 7  AGENT LIFECYCLE ------------------------------------------------------
    ls = report["lifecycle_summary"]
    lines.append("")
    lines.append("7  AGENT LIFECYCLE   (dispatch + spawn/close overhead; self-wall = the agent's own wall)")
    lines.append(f"    orchestrator parent-blocked dispatch wait (Σ): {fmt_dur(t['dispatch_wait_s'])} "
                 f"(overlaps child wall; excluded from active)")
    if not ls["joined"]:
        lines.append("    (no dispatch tool calls joined — single-agent or pre-orchestrator run)")
    else:
        lines.append(f"    {'agent / stage':<24} {'mode':>6} {'dispatch':>9} {'self-wall':>9} "
                     f"{'overhead':>9} {'active':>8}  {th}")
        for x in report["lifecycle"]:
            oh = fmt_dur(x["overhead_s"]) if x["overhead_s"] is not None else "n/a"
            lines.append(f"    {agent_label(x)[:24]:<24} {x['mode']:>6} "
                         f"{fmt_dur(x['dispatch_s']):>9} {fmt_dur(x['self_wall_s']):>9} "
                         f"{oh:>9} {fmt_dur(x['active_s']):>8}  {tc(x['tokens'], x['tokens_total'])}")
        overhead_note = (f" · avg {fmt_dur(ls['overhead_avg_s'])}/agent" if ls["sync"] else "")
        lines.append(f"    dispatches joined {ls['joined']} (sync {ls['sync']} · async {ls['async']}); "
                     f"sync spawn+close overhead Σ {fmt_dur(ls['overhead_total_s'])}{overhead_note}")
        if ls["async"]:
            lines.append("    async dispatch = parent did not block → overhead not measurable this "
                         "way (child cost is its own per-agent wall).")

    lines.append("")
    lines.append(rule)
    lines.append(f"    {t['transcripts']} transcripts · {fmt_tokens(t['tokens_total'])} tokens · "
                 f"active {fmt_dur(active_t)} of {fmt_dur(t['wall_s'])} wall · "
                 f"file-ops {fo_t} over {t['files_touched']} files")
    lines.append(rule)
    return "\n".join(lines)


def to_jsonable(report, session_label, location):
    def iso(dt):
        return dt.astimezone(timezone.utc).isoformat() if dt else None

    def rnd(x):
        return round(x, 3) if x is not None else None

    def prompt_json(p):
        return {
            "index": p["order"] + 1, "prompt_id": p["prompt_id"],
            "source": p["source"], "text": p["text"],
            "assistant_generation_segments": p["assistant_events"],
            "tokens": {**p["tokens"], "total": p["tokens_total"]},
            "file_interactions": {
                "files": p["files"],
                "reads": {"calls": p["read_calls"], "lines": p["read_lines"]},
                "writes": {"calls": p["write_calls"], "lines": p["write_lines"]},
                "edits": {"calls": p["edit_calls"], "added": p["edit_added"],
                          "removed": p["edit_removed"]},
            },
        }

    def agent_json(m):
        tools = sorted(m["tool_time"].items(), key=lambda x: x[1], reverse=True)
        by_use = sorted(m["file_ops"].items(),
                        key=lambda kv: kv[1]["reads"] + kv[1]["writes"] + kv[1]["edits"],
                        reverse=True)
        return {
            "agent_id": m.get("agent_id"), "role": m["role"],
            "agent_type": m["agent_type"], "description": m["description"],
            "spawn_depth": m["depth"], "stage": m["stage"], "module": m["module"],
            "start": iso(m["start"]), "end": iso(m["end"]),
            "wall_seconds": rnd(m["wall_s"]),
            "reasoning_seconds": rnd(m["reasoning_s"]),
            "toolexec_seconds": rnd(m["toolexec_s"]),
            "setup_seconds": rnd(m["setup_s"]),
            "dispatch_wait_seconds": rnd(m["dispatch_wait_s"]),
            "user_wait_seconds": rnd(m["user_wait_s"]),
            "idle_seconds": rnd(m["idle_s"]),
            "assistant_generation_segments": m["assistant_events"],
            "thinking_blocks": m["thinking_blocks"],
            "tool_cost": [
                {"tool": n, "calls": m["tool_calls"].get(n, 0), "total_seconds": rnd(tt)}
                for n, tt in tools
            ],
            "tokens": {**m["tokens"], "total": m["tokens_total"]},
            "file_interactions": {
                "files_touched": m["files_touched"],
                "reads": m["reads"], "writes": m["writes"], "edits": m["edits"],
                "by_file": [{"file": fp, **v} for fp, v in by_use],
            },
            "prompts": [prompt_json(p) for p in m["prompts_list"]],
        }

    t = report["totals"]
    out = {
        "session": session_label, "location": location,
        "totals": {
            "agent_count": t["agent_count"], "transcripts": t["transcripts"],
            "wall_seconds": rnd(t["wall_s"]),
            "reasoning_seconds": rnd(t["reasoning_s"]),
            "toolexec_seconds": rnd(t["toolexec_s"]),
            "setup_seconds": rnd(t["setup_s"]),
            "dispatch_wait_seconds": rnd(t["dispatch_wait_s"]),
            "user_wait_seconds": rnd(t["user_wait_s"]),
            "idle_seconds": rnd(t["idle_s"]),
            "human_wait_calls": t["human_wait_calls"],
            "assistant_generation_segments": t["assistant_events"],
            "thinking_blocks": t["thinking_blocks"],
            "tool_calls_total": t["tool_calls_total"],
            "tokens": {**t["tokens"], "total": t["tokens_total"]},
            "file_interactions": {
                "files_touched": t["files_touched"],
                "reads": t["reads"], "writes": t["writes"], "edits": t["edits"],
            },
            "start": iso(t["start"]), "end": iso(t["end"]),
        },
        "tool_cost": [
            {"tool": n, "calls": c, "total_seconds": rnd(tt),
             "avg_seconds": rnd(tt / c if c else 0)}
            for n, tt, c in report["tool_rollup"]
        ],
        "top_files": [
            {"file": fp, **v} for fp, v in report["top_files"]
        ],
        "lifecycle_summary": {
            "joined": report["lifecycle_summary"]["joined"],
            "sync": report["lifecycle_summary"]["sync"],
            "async": report["lifecycle_summary"]["async"],
            "overhead_total_seconds": rnd(report["lifecycle_summary"]["overhead_total_s"]),
            "overhead_avg_seconds": rnd(report["lifecycle_summary"]["overhead_avg_s"]),
        },
        "stages": [],
        "lifecycle": [
            {"agent_id": x["agent_id"], "role": x["role"], "module": x["module"],
             "stage": x["stage"], "description": x["description"], "mode": x["mode"],
             "dispatch_seconds": rnd(x["dispatch_s"]),
             "self_wall_seconds": rnd(x["self_wall_s"]),
             "overhead_seconds": rnd(x["overhead_s"]),
             "active_seconds": rnd(x["active_s"]),
             "tokens": {**x["tokens"], "total": x["tokens_total"]}}
            for x in report["lifecycle"]
        ],
    }
    for st in report["stages"]:
        out["stages"].append({
            "stage": st["stage"],
            "agent_count": st["agent_count"],
            "reasoning_seconds": rnd(st["reasoning_s"]),
            "toolexec_seconds": rnd(st["toolexec_s"]),
            "setup_seconds": rnd(st["setup_s"]),
            "active_seconds": rnd(st["active_s"]),
            "wall_seconds": rnd(st["wall_s"]),
            "tokens": {**st["tokens"], "total": st["tokens_total"]},
            "tokens_total": st["tokens_total"],
            "file_interactions": {
                "reads": st["reads_calls"], "writes": st["writes_calls"],
                "edits": st["edits_calls"],
            },
            "agents": [agent_json(m) for m in st["members"]],
        })
    return out


# --- main -----------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        description="Time + token + file-interaction cost extractor for a PyPTO Agent Team run.")
    p.add_argument("--projects-root", default=default_projects_root())
    p.add_argument("--project", default=None,
                   help="project dir under projects-root (default: derived from cwd)")
    p.add_argument("--session", default=None,
                   help="session id (default: most recently active with subagents)")
    p.add_argument("--all-sessions", action="store_true",
                   help="aggregate every session under the project")
    p.add_argument("--list-sessions", action="store_true",
                   help="list sessions that have subagent logs, then exit")
    p.add_argument("--json", action="store_true", help="emit JSON instead of a text report")
    p.add_argument("--idle-threshold", type=float, default=DEFAULT_IDLE_THRESHOLD_S,
                   help="seconds; reasoning/setup gaps longer than this are counted as idle "
                        f"rather than work (default {int(DEFAULT_IDLE_THRESHOLD_S)}). Tool "
                        "execution gaps are never capped.")
    args = p.parse_args(argv)
    _init_logging()

    project = args.project or encode_project_dir(os.getcwd())
    project_path = os.path.join(args.projects_root, project)
    if not os.path.isdir(project_path):
        logger.error("[pypto-op-claude-monitor] project dir not found: %s", project_path)
        avail = sorted(os.path.basename(d) for d in glob.glob(os.path.join(args.projects_root, "*"))
                       if os.path.isdir(d))
        for a in avail:
            logger.error("    %s", a)
        return 2

    sessions = find_sessions(project_path)
    if args.list_sessions:
        if not sessions:
            logger.info("[pypto-op-claude-monitor] no sessions with subagents under %s", project_path)
            return 0
        for sid, sub, mtime in sessions:
            n = len(glob.glob(os.path.join(sub, "*.meta.json")))
            stamp = datetime.fromtimestamp(mtime, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            logger.info("%s  %3d agents  last-active %sZ", sid, n, stamp)
        return 0

    if not sessions:
        logger.error("[pypto-op-claude-monitor] no subagent logs found under %s", project_path)
        logger.error("[pypto-op-claude-monitor] (a session only gets a subagents/ dir once the orchestrator "
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
                logger.error("[pypto-op-claude-monitor] session %s has no subagents", args.session)
                return 1
            sid, sub, _ = match[0]
        else:
            sid, sub, _ = sessions[0]
        records = load_run(project_path, sid, sub, index, args.idle_threshold)
        session_label, location = sid, sub

    report = build_report(records, index)
    if args.json:
        logger.info("%s", json.dumps(to_jsonable(report, session_label, location),
                                     indent=2, ensure_ascii=False))
    else:
        logger.info("%s", render_text(report, session_label, location))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
