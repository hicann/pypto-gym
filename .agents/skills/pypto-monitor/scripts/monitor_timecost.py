#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""pypto-monitor — time-cost breakdown for a PyPTO Agent Team run.

Parses the orchestrator transcript plus every subagent transcript and charges each
wall-clock gap to reasoning, tool execution, subagent dispatch/lifecycle, setup,
user-wait, or idle. Stdlib only, Python 3.8+.
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

logger = logging.getLogger("pypto-monitor")

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
# Reasoning / setup gaps longer than this are treated as idle (agent paused, user away
# between turns) rather than real model generation. Tool-execution gaps are NEVER capped
# here — a verify Bash can legitimately run 30+ min. Override with --idle-threshold.
DEFAULT_IDLE_THRESHOLD_S = 600.0


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


def fmt_clock(dt):
    return "--:--:--" if dt is None else dt.astimezone(timezone.utc).strftime("%H:%M:%S")


def short_role(agent_type):
    return (agent_type or "unknown").replace("pypto-op-", "")


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


# --- core: parse one transcript -------------------------------------------------

def parse_transcript(jsonl_path, ctx, index, idle_threshold=DEFAULT_IDLE_THRESHOLD_S):
    """Parse one transcript into a time-cost record; mutates the index dispatch-join tables."""
    rec = {
        "path": jsonl_path, "role": ctx.role, "agent_type": ctx.agent_type,
        "description": ctx.description, "depth": ctx.depth, "stage": ctx.stage,
        "module": ctx.module,
        "start": None, "end": None, "wall_s": None,
        "reasoning_s": 0.0, "setup_s": 0.0, "toolexec_s": 0.0, "dispatch_wait_s": 0.0,
        "user_wait_s": 0.0, "idle_s": 0.0,
        "tool_time": defaultdict(float), "tool_calls": defaultdict(int),
        "dispatch_calls": 0, "human_wait_calls": 0, "assistant_events": 0,
        "thinking_blocks": 0, "events": 0,
    }
    if not (jsonl_path and os.path.exists(jsonl_path)):
        return rec

    id2name = {}            # tool_use id -> tool name (within this transcript)
    prev_ts = None
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
            content = (ev.get("message") or {}).get("content")

            # index tool_use / tool_result; collect block stats
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
    rec["tool_time"] = dict(rec["tool_time"])
    rec["tool_calls"] = dict(rec["tool_calls"])
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
            "stage": key, "members": members,
            "reasoning_s": sum(m["reasoning_s"] for m in members),
            "toolexec_s": sum(m["toolexec_s"] for m in members),
            "wall_s": sum(m["wall_s"] or 0 for m in members),
        })

    lifecycle = compute_lifecycle(records, index)
    life_vals = [x["overhead_s"] for x in lifecycle if x["mode"] == "sync"]
    async_count = sum(1 for x in lifecycle if x["mode"] == "async")

    return {
        "stages": stages,
        "tool_rollup": sorted(((n, tool_time[n], tool_calls[n]) for n in tool_calls),
                              key=lambda x: x[1], reverse=True),
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
    t = report["totals"]
    lines = []
    lines.append("=" * 96)
    lines.append("PyPTO Agent Team — Time-Cost Monitor (tool calls + thinking, full run)")
    lines.append(f"session : {session_label}")
    lines.append(f"location: {location}")
    if t["start"] and t["end"]:
        lines.append(f"window  : {fmt_clock(t['start'])} → {fmt_clock(t['end'])} UTC "
                     f"(wall {fmt_dur(t['wall_s'])})")
    lines.append(f"scope   : {t['transcripts']} transcripts "
                 f"(1 orchestrator + {t['agent_count']} subagents)")
    lines.append("=" * 96)

    # ---- §A per-agent time cost --------------------------------------------------
    lines.append("")
    lines.append("§A  PER-AGENT TIME COST  (wall = reasoning + tool-exec + setup)")
    for st in report["stages"]:
        lines.append("")
        lines.append(f"  {stage_header(st['stage'])}   "
                     f"[Σwall {fmt_dur(st['wall_s'])} · Σreason {fmt_dur(st['reasoning_s'])} "
                     f"· Σtools {fmt_dur(st['toolexec_s'])}]")
        lines.append(f"    {'start':>8} {'wall':>8} {'reason':>8} {'tools':>8} "
                     f"{'calls':>5} {'role':<13} {'mod':<4}  top tools / description")
        for m in st["members"]:
            label = (m["description"] or "")[:30] if m["role"] == ORCH else \
                f"{_top_tools(m)}  · {(m['description'] or '')[:22]}"
            lines.append(
                f"    {fmt_clock(m['start']):>8} {fmt_dur(m['wall_s']):>8} "
                f"{fmt_dur(m['reasoning_s']):>8} {fmt_dur(m['toolexec_s']):>8} "
                f"{sum(m['tool_calls'].values()):>5} {m['role'][:13]:<13} "
                f"{(m['module'] or '-'):<4}  {label}")
            if m["role"] == ORCH:
                pad = f"    {'':>8} {'':>8} {'':>8} {'':>8} {'':>5} {'':<13} {'':<4}  "
                if m["dispatch_wait_s"] > 0:
                    lines.append(pad + f"(dispatch_wait {fmt_dur(m['dispatch_wait_s'])} over "
                                 f"{m['dispatch_calls']} dispatches — overlaps child time, §D)")
                if m["user_wait_s"] > 0 or m["idle_s"] > 0:
                    lines.append(pad + f"(user_wait {fmt_dur(m['user_wait_s'])} on "
                                 f"{m['human_wait_calls']} prompts + idle {fmt_dur(m['idle_s'])} "
                                 f"— excluded from work)")

    # ---- §B tool-call time cost --------------------------------------------------
    lines.append("")
    lines.append("§B  TOOL-CALL TIME COST  (run-wide, all transcripts; dispatch excluded → §D)")
    lines.append(f"    {'tool':<20} {'calls':>6} {'total':>9} {'avg':>8}  share")
    tot = t["toolexec_s"] or 1.0
    for name, ttime, calls in report["tool_rollup"]:
        avg = ttime / calls if calls else 0
        share = ttime / tot * 100
        bar = "█" * int(round(share / 5))
        lines.append(f"    {name[:20]:<20} {calls:>6} {fmt_dur(ttime):>9} "
                     f"{fmt_dur(avg):>8}  {share:4.1f}% {bar}")
    lines.append(f"    {'-' * 20} {'-' * 6} {'-' * 9}")
    lines.append(f"    {'TOTAL tool-exec':<20} {t['tool_calls_total']:>6} "
                 f"{fmt_dur(t['toolexec_s']):>9}")

    # ---- §C thinking / reasoning -------------------------------------------------
    lines.append("")
    lines.append("§C  THINKING / REASONING TIME COST")
    avg_seg = (t["reasoning_s"] / t["assistant_events"]) if t["assistant_events"] else 0
    lines.append(f"    model-generation (reasoning) total : {fmt_dur(t['reasoning_s'])}")
    lines.append(f"    across assistant generation segments: {t['assistant_events']} "
                 f"(avg {fmt_dur(avg_seg)}/segment)")
    lines.append(f"    explicit `thinking` blocks persisted : {t['thinking_blocks']} "
                 f"(rarely written to disk — generation latency above is the time proxy)")
    lines.append(f"    setup / injection overhead           : {fmt_dur(t['setup_s'])}")
    lines.append(f"    excluded as non-work: user_wait {fmt_dur(t['user_wait_s'])} "
                 f"({t['human_wait_calls']} human prompts) · idle {fmt_dur(t['idle_s'])} "
                 f"(gaps > idle-threshold)")

    # ---- §D subagent lifecycle ---------------------------------------------------
    ls = report["lifecycle_summary"]
    lines.append("")
    lines.append("§D  SUBAGENT LIFECYCLE COST  (dispatch + spawn/close overhead)")
    lines.append(f"    parent-blocked dispatch wait (Σ)     : {fmt_dur(t['dispatch_wait_s'])} "
                 f"(in-band time parents spent on Agent/Task results; overlaps child wall)")
    if not ls["joined"]:
        lines.append("    (no dispatch tool calls joined — single-agent or pre-orchestrator run)")
    else:
        lines.append(f"    dispatches joined: {ls['joined']}  "
                     f"(synchronous/blocking: {ls['sync']} · asynchronous/background: {ls['async']})")
        if ls["sync"]:
            lines.append("    spawn+close overhead (sync only = parent-observed dispatch − child "
                         "self-wall):")
            lines.append(f"      Σ {fmt_dur(ls['overhead_total_s'])} | avg "
                         f"{fmt_dur(ls['overhead_avg_s'])}/agent  "
                         f"(spawn+teardown combined — no separate 'close' event exists)")
            lines.append("      highest-overhead dispatches:")
            lines.append(f"      {'overhead':>9} {'dispatch':>9} {'self-wall':>9}  role/module · what")
            for x in [r for r in report["lifecycle"] if r["mode"] == "sync"][:5]:
                tag = f"{x['role']}/{x['module'] or '-'}"
                lines.append(f"      {fmt_dur(x['overhead_s']):>9} {fmt_dur(x['dispatch_s']):>9} "
                             f"{fmt_dur(x['self_wall_s']):>9}  {tag} · {(x['description'] or '')[:30]}")
        if ls["async"]:
            lines.append(f"    {ls['async']} dispatch(es) were non-blocking (background): the parent "
                         f"did not wait, so spawn/close overhead is not measurable this way —")
            lines.append("      each child's cost is its own §A wall (orchestrator stayed free to "
                         "work meanwhile).")

    # ---- §E totals ---------------------------------------------------------------
    active = t["reasoning_s"] + t["toolexec_s"] + t["setup_s"]
    lines.append("")
    lines.append("=" * 96)
    lines.append(f"TOTALS  wall {fmt_dur(t['wall_s'])} (calendar span) | "
                 f"active agent-time {fmt_dur(active)} "
                 f"(reason {fmt_dur(t['reasoning_s'])} + tools {fmt_dur(t['toolexec_s'])} "
                 f"+ setup {fmt_dur(t['setup_s'])})")
    lines.append(f"        {t['tool_calls_total']} tool calls | "
                 f"{t['assistant_events']} generation segments | "
                 f"{t['agent_count']} subagents")
    lines.append(f"        non-work (excluded): user_wait {fmt_dur(t['user_wait_s'])} + "
                 f"idle {fmt_dur(t['idle_s'])} + dispatch_wait {fmt_dur(t['dispatch_wait_s'])} "
                 f"(overlaps children)")
    lines.append("note: agents run concurrently, so summed agent-time exceeds the wall span; "
                 "dispatch_wait overlaps child time and is excluded from active.")
    lines.append("=" * 96)
    return "\n".join(lines)


def to_jsonable(report, session_label, location):
    def iso(dt):
        return dt.astimezone(timezone.utc).isoformat() if dt else None

    def rnd(x):
        return round(x, 3) if x is not None else None

    def agent_json(m):
        tools = sorted(m["tool_time"].items(), key=lambda x: x[1], reverse=True)
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
            "start": iso(t["start"]), "end": iso(t["end"]),
        },
        "tool_cost": [
            {"tool": n, "calls": c, "total_seconds": rnd(tt),
             "avg_seconds": rnd(tt / c if c else 0)}
            for n, tt, c in report["tool_rollup"]
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
             "overhead_seconds": rnd(x["overhead_s"])}
            for x in report["lifecycle"]
        ],
    }
    for st in report["stages"]:
        out["stages"].append({
            "stage": st["stage"],
            "reasoning_seconds": rnd(st["reasoning_s"]),
            "toolexec_seconds": rnd(st["toolexec_s"]),
            "wall_seconds": rnd(st["wall_s"]),
            "agents": [agent_json(m) for m in st["members"]],
        })
    return out


# --- main -----------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        description="Time-cost breakdown (tool calls + thinking) for a PyPTO Agent Team run.")
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
        logger.error("[pypto-monitor] project dir not found: %s", project_path)
        avail = sorted(os.path.basename(d) for d in glob.glob(os.path.join(args.projects_root, "*"))
                       if os.path.isdir(d))
        for a in avail:
            logger.error("    %s", a)
        return 2

    sessions = find_sessions(project_path)
    if args.list_sessions:
        if not sessions:
            logger.info("[pypto-monitor] no sessions with subagents under %s", project_path)
            return 0
        for sid, sub, mtime in sessions:
            n = len(glob.glob(os.path.join(sub, "*.meta.json")))
            stamp = datetime.fromtimestamp(mtime, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            logger.info("%s  %3d agents  last-active %sZ", sid, n, stamp)
        return 0

    if not sessions:
        logger.error("[pypto-monitor] no subagent logs found under %s", project_path)
        logger.error("[pypto-monitor] (a session only gets a subagents/ dir once the orchestrator "
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
                logger.error("[pypto-monitor] session %s has no subagents", args.session)
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
