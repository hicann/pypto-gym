---
name: pypto-op-monitor
description: "Cost profiler for a PyPTO harness operator-generation run on Claude Code or opencode: breaks down time (wall, active) and tokens (input, cache, output) by stage, agent, reasoning, tool call, file interaction (read/write/edit), and lifecycle."
---

# pypto-op-monitor

Read-only cost profiler for a **PyPTO Agent Team** operator-generation run. `monitor_cost.py`
parses the orchestrator transcript and every subagent transcript, partitions each one's wall
time into reasoning / tool-exec / dispatch / setup / user-wait / idle, and rolls up token usage
and the Read/Write/Edit file footprint — per agent, per prompt, per stage, and run-wide. It
never touches operator artifacts, `MEMORY.md`, or `.orchestrator_state.json`, so it is safe on
finished or in-progress runs.

**Two sources, one report.** Claude Code (on-disk JSONL) and opencode (`opencode export`) are
both supported. The source defaults to Claude Code when run under Claude Code (`CLAUDECODE=1`),
otherwise opencode; a `ses_…` id or `--source` forces a choice. Every section and
`--json` field is identical across both — source-specific mechanics live in
[`references/log-schema.md`](references/log-schema.md).

## How the skill works

1. Resolve the run: a Claude session id / project, or an opencode run's root `ses_…` id.
2. Run `monitor_cost.py` once — text to read, `--json` to analyse.
3. Present the numbers in the **report template** order below. Fill every `<…>` straight from
   the script — do **not** recompute. Layer analysis (ranking, bottlenecks, recommendations) on
   top of those numbers, never in place of them.

## What it measures

Each transcript's wall time is partitioned additively (no overlap) into buckets:

| Bucket | Meaning |
|---|---|
| **reasoning** | Model-generation latency — the proxy for *thinking* time. |
| **tool-exec** (per tool) | Time a tool ran, keyed by tool name. |
| **dispatch_wait** | Time a parent spent on a subagent dispatch result. Overlaps the child → reported separately. |
| **user_wait** | Time waiting on a human. Excluded from work. |
| **idle** | Reasoning/setup gaps longer than `--idle-threshold`. Excluded from work. |
| **setup** | Small prompt/attachment injection gaps. |

**Active** agent-time = reasoning + tool-exec + setup. **Wall** is the operator-generation
window — first activity → the last stage finishing — so the orchestrator session's idle tail
after the run (it can stay open for hours/days) is excluded. `dispatch_wait` / `user_wait` /
`idle` are excluded from active.

Two count dimensions roll up from the same events: **token cost** (input · cache-read ·
cache-write · output · total, per assistant event) and **file interactions** (Read/Write/Edit
calls · lines · edit ±lines · distinct files). Per-agent, per-prompt and per-stage sums each
reconcile to the run total, so nothing is double-counted. The overall section splits cache-read
vs cache-write; per-agent/per-stage tables use one `cache` column.

## How to run

Stdlib-only (Python 3.8+), no dependencies. `MON` below is the script under the active harness's
skills tree — `.claude/skills/pypto-op-monitor/scripts/monitor_cost.py` under **Claude Code**,
`.agents/skills/pypto-op-monitor/scripts/monitor_cost.py` under **opencode**.

```bash
python3 MON                    # most recent run of the current harness/project
python3 MON --list-sessions    # list runs (newest first), then exit
python3 MON --session <id>     # a specific run — Claude UUID or opencode ses_…
python3 MON --json             # machine-readable output
```

| Flag | Default | Purpose |
|---|---|---|
| `--source <auto\|claude\|opencode>` | `auto` | `auto`: opencode for a `ses_…` id; else Claude Code when run under Claude Code (`CLAUDECODE=1`), otherwise opencode. |
| `--session <id>` | most-recent | Claude session UUID, or an opencode run's root `ses_…` id. |
| `--all-sessions` | off | Aggregate every run (Claude: under the project; opencode: every run). |
| `--list-sessions` | off | List runs, then exit. |
| `--idle-threshold <s>` | `600` | Reasoning/setup gaps beyond this count as idle. Tool-exec gaps are never capped. |
| `--json` | off | Structured JSON instead of the text report. |
| `--project=<dir>` / `--projects-root <p>` | from `cwd` / `~/.claude/projects` | *[claude]* Project dir / logs root. |
| `--opencode-bin <p>` / `--opencode-db <p>` | `opencode` / `~/.local/share/opencode/opencode.db` | *[opencode]* Export executable / discovery store. |

> **Gotcha:** Claude project dirs start with `-` (argparse reads it as a flag). Use the equals
> form `--project=<name>`, or omit it to auto-detect from `cwd`.

## Report template

Present the result in **this** order (matching the script's sections). Fill every `<…>` from the
output; omit a stage/agent row that never ran and drop whichever dispatch line does not apply.
Keep wall + active time and the token split (input / cache / output / total) in every table.
`--json` adds a per-prompt drill-down for multi-turn transcripts.

```markdown
### `<op-name>` harness run — session `<session-id>`

**Shape:** 1 orchestrator + `<N>` subagents · Stage `<first>`→`<last>` · wall `<wall>` · active `<active>`.

#### 1 · Overall costs

| Metric | Value |
|---|---|
| Wall / active | `<wall>` run-span · **`<active>`** active (reason `<r>` · tools `<t>` · setup `<s>`) |
| Tokens | **`<total>`** = input `<in>` · cache-rd `<crd>` · cache-wr `<cwr>` · output `<out>` |
| File interactions | `<rd>`r / `<wr>`w / `<ed>`e (`+<add>/-<del>` ln) over `<files>` files |
| Tool calls | `<n>` · top tool **`<tool>` = `<share>`** of tool-exec |
| Generation | `<segs>` segments · `<blocks>` persisted thinking blocks |
| Excluded non-work | `<user_wait>` user_wait + `<idle>` idle + `<dispatch_wait>` dispatch |

#### 2 · Per-stage costs

| Stage | Reason | Tools | Wall | Active | input | cache | output | total | rd/wr/ed |
|---|---|---|---|---|---|---|---|---|---|
| `<n · name>` | `<r>` | `<t>` | `<wall>` | `<active>` | `<in>` | `<cache>` | `<out>` | `<total>` | `<r/w/e>` |
| … | | | | | | | | | |

*(rank stages heaviest-first by tools + reason)*

#### 3 · Per-agent costs

| Agent / stage | Wall | Active | input | cache | output | total | rd/wr/ed |
|---|---|---|---|---|---|---|---|
| `<label>` | `<wall>` | `<active>` | `<in>` | `<cache>` | `<out>` | `<total>` | `<r/w/e>` |
| … | | | | | | | |

*(the few costliest agents — not all N)*

#### 4 · Reasoning

| Agent / stage | Wall | Active | Reason | segs | output | total |
|---|---|---|---|---|---|---|
| `<label>` | `<wall>` | `<active>` | `<reason>` | `<segs>` | `<out>` | `<total>` |

*(heaviest reasoners; output tokens are the reasoning product)*

#### 5 · Tool calls

| Agent / stage | Wall | Active | Tool-exec | calls | total | top tools |
|---|---|---|---|---|---|---|
| `<label>` | `<wall>` | `<active>` | `<toolexec>` | `<calls>` | `<total>` | `<tools>` |

Run-wide: top tool **`<tool>` = `<share>`** of tool-exec.

#### 6 · File interactions

| Agent / stage | Wall | Active | rd | wr | ed (±lines) | files | total |
|---|---|---|---|---|---|---|---|
| `<label>` | `<wall>` | `<active>` | `<rd>` | `<wr>` | `<ed> (+<a>/-<d>)` | `<files>` | `<total>` |

Most-touched: `<file = Nr/Mw/Ke>`.

#### 7 · Agent lifecycle

| Agent / stage | Mode | Dispatch | Self-wall | Overhead | Active | total |
|---|---|---|---|---|---|---|
| `<label>` | `<async|sync>` | `<dispatch>` | `<self-wall>` | `<overhead|n/a>` | `<active>` | `<total>` |

Dispatch: `<all N async (orchestrator never blocked) | sync overhead Σ <x>>`.

#### Analysis & recommendations

- **Bottleneck:** `<stage / agent / tool that dominates active time, with the number>`.
- **Tokens:** heaviest agent/stage `<= tokens>`; cache-rd `<crd>` is cumulative re-reads — fresh
  work is input+output `<in+out>`.
- **File churn:** most-touched `<file = Nr/Mw/Ke>` — a file re-read or edited many times signals
  context churn or a hot spot (e.g. an operator's `MEMORY.md` updated across every stage).
- **Wall vs. active:** `<wall>` run-span but only `<active>` of real work — the rest was
  `<user_wait + idle>` of human/idle time.
- **Recommendation:** `<one or two concrete, data-driven next steps>` (e.g. cut repeated Bash
  verify re-runs, reduce `MEMORY.md` edit churn, parallelize a serial stage, trim a long tail).
```

Close by offering a next step: drill into an agent, request the `--json` per-prompt breakdown,
or re-run against another session (`--list-sessions`).

## Reading it honestly

- **Active ≫ wall** is normal — agents run concurrently; the overall section shows both.
- **`dispatch_wait` overlaps children** and is excluded from active.
- **Thinking time is a proxy** (generation latency); the persisted-block count is shown for transparency.
- **`user_wait` / `idle`** are human/away time, not compute; tune `--idle-threshold` for genuinely long reasoning.
- **`cache-rd` dominates by design** — cached context is re-read at the cheap rate every call, so it is cumulative, not fresh input; read `input` + `output` for new tokens. No dollar cost is computed.
- **File interactions cover `Read`/`Write`/`Edit` only** (not `Bash cat`, attachments, or skill loads). A call counts even if it errors, so `calls ≥ files`; a `Write` is classified create vs overwrite.
- **Claude vs. opencode timing** — same buckets, different sourcing: Claude *infers* reasoning/tool time from event gaps, opencode reads *explicit* per-message/per-tool spans (reasoning = message span − tool time inside it). opencode tool names are lowercase and reasoning tokens fold into `output`; sums reconcile to opencode's own totals exactly.

## Files

- `scripts/monitor_cost.py` — the metrics extractor (text report and `--json`).
- `references/log-schema.md` — per-source log layout, timing model, dispatch join, stage/role map.
