---
name: pypto-monitor
description: "Time-cost profiler for a PyPTO Agent Team operator-generation run. Claude Code only — reads its on-disk transcript layout (~/.claude/projects/<project>/<session>/); does not work under opencode. Parses the orchestrator plus every dispatched subagent transcript and breaks wall-clock time down by agent, 7-stage workflow, and tool (Bash/Read/Edit/Write/Skill/state_transition/...), separating reasoning/thinking, tool calls, subagent dispatch and spawn/close overhead, and human-wait vs real compute. Triggers: monitor agents, agent/stage/tool time cost, thinking time, dispatch/state-transition cost, profile the operator run, time breakdown."
---

# pypto-monitor

Read-only time-cost profiler for a **PyPTO Agent Team** operator-generation run. It reads the
orchestrator transcript **and** every subagent transcript on disk and breaks the run's
wall-clock time down by agent, stage, tool call, and thinking — including the cost of
dispatching/closing subagents and state transitions. It never touches operator artifacts,
`MEMORY.md`, or `.orchestrator_state.json`, so it's safe on finished or in-progress runs.

> **Claude Code only.** Depends on Claude Code's on-disk transcript layout; it does not work
> under opencode, which stores session state in an incompatible format.

## What it measures

Each transcript's wall time is partitioned (additively, no overlap) into buckets:

| Bucket | Meaning |
|---|---|
| **reasoning** | Model-generation latency — the proxy for *thinking* time (explicit `thinking` blocks are rarely persisted). |
| **tool-exec** (per tool) | Time a tool ran, keyed by tool name (`Bash`, `Read`, `Edit`, `Skill`, `state_transition`, …), paired by tool_use `id` → tool_result `tool_use_id`. |
| **dispatch_wait** | In-band time a parent spent on `Agent`/`Task` results. Overlaps the child's own time, so it's reported separately, never inside tool-exec. |
| **user_wait** | Time waiting on a human (`AskUserQuestion`). Excluded from work. |
| **idle** | Reasoning/setup gaps longer than `--idle-threshold`. Excluded from work. |
| **setup** | Small prompt/attachment injection gaps. |

Active agent-time = reasoning + tool-exec + setup; `dispatch_wait` / `user_wait` / `idle` are
excluded so a multi-day orchestrator span isn't mistaken for compute.

## Report sections

- **§A Per-agent time cost** — every agent (incl. orchestrator) grouped by stage: wall, reasoning, tool-exec, calls, top tools.
- **§B Tool-call time cost** — run-wide per-tool rollup: calls, total, avg, share.
- **§C Thinking / reasoning** — total generation time, segments, persisted thinking blocks, plus excluded user_wait / idle.
- **§D Subagent lifecycle** — dispatch wait, and spawn+close overhead (dispatch − child self-wall) for sync dispatches; async ones are flagged not-measurable rather than faked.
- **§E Totals** — wall span vs. active agent-time, with non-work called out.

## Data source

```
~/.claude/projects/<project-dir>/
    <session-id>.jsonl                 # orchestrator (main) transcript
    <session-id>/subagents/
        agent-<id>.meta.json           # {agentType, description, toolUseId, spawnDepth}
        agent-<id>.jsonl               # subagent transcript (one event per line)
```

`meta.toolUseId` joins each subagent back to the `Agent`/`Task` call that spawned it (how §D
derives dispatch/overhead). Full schema, timing model, and stage/role map:
`references/log-schema.md`.

## How to run

Stdlib-only (Python 3.8+), no dependencies.

```bash
# Full breakdown for the most recent run of the CURRENT project (cwd auto-detected)
python3 .claude/skills/pypto-monitor/scripts/monitor_timecost.py

# List sessions that have agent logs (newest first)
python3 .claude/skills/pypto-monitor/scripts/monitor_timecost.py --list-sessions

# A specific session / machine-readable output
python3 .claude/skills/pypto-monitor/scripts/monitor_timecost.py --session <session-id>
python3 .claude/skills/pypto-monitor/scripts/monitor_timecost.py --json
```

| Flag | Default | Purpose |
|---|---|---|
| `--project=<dir>` | encoded from `cwd` | Project dir under the projects root. |
| `--session <id>` | most-recently-active | Which session to report on. |
| `--all-sessions` | off | Aggregate all sessions of the project. |
| `--list-sessions` | off | List sessions with agent logs, then exit. |
| `--idle-threshold <s>` | `600` | Reasoning/setup gaps beyond this count as idle, not work. Tool-exec gaps are never capped. |
| `--projects-root <p>` | `~/.claude/projects` | Override the logs root. |
| `--json` | off | Structured JSON instead of the text report. |

> **Gotcha:** Claude Code project dirs start with `-`, which `argparse` reads as a flag. Use
> `--project=<name>` (equals form), or omit `--project` to auto-detect from `cwd`.

## Reading it honestly

- **Active agent-time ≫ wall span** is normal — agents run concurrently. §E shows both.
- **`dispatch_wait` overlaps children** and is excluded from active time.
- **Thinking time is a proxy** (generation latency); the persisted-block count is shown for transparency.
- **`user_wait` / `idle`** are human/away time, not compute; tune `--idle-threshold` for legitimately long reasoning.

## Files

- `scripts/monitor_timecost.py` — the analyzer (text report and `--json`).
- `references/log-schema.md` — log layout, timing model, dispatch join, stage/role map.
