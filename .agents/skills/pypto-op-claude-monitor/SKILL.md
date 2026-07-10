---
name: pypto-op-claude-monitor
description: "Cost profiler for a PyPTO Agent Team operator-generation run: breaks time (wall / active) and tokens (input / cache / output) down by stage, agent, reasoning, tool call, file interaction (Read/Write/Edit), and lifecycle. Triggers: monitor agents, agent/stage/tool time cost, thinking time, token cost per agent/stage, input/output/cache tokens, file interactions/reads/writes/edits, dispatch/lifecycle/state-transition cost, profile the operator run, time/token/cost breakdown."
---

# pypto-op-claude-monitor

Read-only cost profiler for a **PyPTO Agent Team** operator-generation run. `monitor_cost.py`
is the metrics extractor: it parses the orchestrator transcript **and** every subagent
transcript on disk, charges each wall-clock gap to reasoning / tool-exec / dispatch / setup /
human-wait / idle, and rolls up the token usage (input / cache read / cache write / output) and
the `Read` / `Write` / `Edit` file footprint — per agent, per prompt, per stage, and run-wide.
It never touches operator artifacts, `MEMORY.md`, or `.orchestrator_state.json`, so it is safe on
finished or in-progress runs.

> **Claude Code only.** Reads Claude Code's on-disk transcript layout
> (`~/.claude/projects/<project>/<session>/`); it does not work under opencode, which stores
> session state in an incompatible format.

## How the skill works

1. Resolve the target run (a session id, or the most recent run of a project).
2. Run `monitor_cost.py` once — text for reading, `--json` for analysis.
3. Present the numbers to the user in the **report template** order below. Fill every `<…>`
   straight from the script's output — do **not** recompute. Any *analysis* (ranking, ratios,
   churn/bottleneck calls, recommendations) is layered on top of those numbers, never instead of them.

## What it measures

Each transcript's wall time is partitioned (additively, no overlap) into buckets:

| Bucket | Meaning |
|---|---|
| **reasoning** | Model-generation latency — the proxy for *thinking* time. |
| **tool-exec** (per tool) | Time a tool ran, keyed by tool name (`Bash`, `Read`, `Edit`, …), paired tool_use `id` → tool_result. |
| **dispatch_wait** | In-band time a parent spent on `Agent`/`Task` results. Overlaps the child, so reported separately. |
| **user_wait** | Time waiting on a human (`AskUserQuestion`). Excluded from work. |
| **idle** | Reasoning/setup gaps longer than `--idle-threshold`. Excluded from work. |
| **setup** | Small prompt/attachment injection gaps. |

**Active** agent-time = reasoning + tool-exec + setup; **wall** is the calendar span;
`dispatch_wait` / `user_wait` / `idle` are excluded from active. On top of the time model two
count dimensions are rolled up from the same event stream:

| Dimension | Source | Fields |
|---|---|---|
| **token cost** | each assistant event's `message.usage` | input · cache-read · cache-write · output · total |
| **file interactions** | `Read` / `Write` / `Edit` `toolUseResult` | rd/wr/ed calls · lines read/written · edit ±lines · distinct files |

Every section carries both a time cost (wall + active) and a token cost. Per-agent, per-prompt and
per-stage sums each reconcile to the run total (and rd/wr/ed counts match the tool tally), so
nothing is double-counted.

## Report sections (script output)

Seven sections **in this order**, each with a time cost (wall + active) and a token cost
(input / cache / output / total). Per-agent sections add one row per agent (incl. orchestrator):

1. **Overall costs** — wall vs. active (reason + tools + setup), the full input/cache-read/cache-write/output/total split, file-ops over N files, tool-call & generation-segment counts, excluded non-work.
2. **Per-stage costs** — per 7-stage bucket: reason · tools · wall · active · rd/wr/ed · tokens, + TOTAL.
3. **Per-agent costs** — start · wall · active · tool-calls · rd/wr/ed · tokens, + TOTAL.
4. **Reasoning** — reasoning time · generation segments · avg/seg · thinking blocks · tokens.
5. **Tool calls** — tool-exec time · calls · tokens · top tools; plus a run-wide per-tool rollup (calls · total · avg · share).
6. **File interactions** — Read/Write/Edit calls · lines read/written · edit ±lines · distinct files · tokens; plus most-touched files.
7. **Agent lifecycle** — dispatch · self-wall · spawn/close overhead (sync only) · active · tokens; orchestrator dispatch-wait Σ; async flagged not-measurable.

The **overall** section splits cache-read vs cache-write; per-agent/per-stage tables use one
`cache` column. `--json` adds the per-prompt drill-down (token + file interactions per turn) for
multi-turn transcripts.

## Data source

```
~/.claude/projects/<project-dir>/
    <session-id>.jsonl                 # orchestrator (main) transcript
    <session-id>/subagents/
        agent-<id>.meta.json           # {agentType, description, toolUseId, spawnDepth}
        agent-<id>.jsonl               # subagent transcript (one event per line)
```

`meta.toolUseId` joins each subagent back to the `Agent`/`Task` call that spawned it (how the
agent-lifecycle section derives dispatch/overhead). Full schema, timing model, and stage/role map:
`references/log-schema.md`.

## How to run

Stdlib-only (Python 3.8+), no dependencies.

```bash
# Full breakdown for the most recent run of the CURRENT project (cwd auto-detected)
python3 .claude/skills/pypto-op-claude-monitor/scripts/monitor_cost.py

# List sessions that have agent logs (newest first)
python3 .claude/skills/pypto-op-claude-monitor/scripts/monitor_cost.py --list-sessions

# A specific session / machine-readable output
python3 .claude/skills/pypto-op-claude-monitor/scripts/monitor_cost.py --session <session-id>
python3 .claude/skills/pypto-op-claude-monitor/scripts/monitor_cost.py --json
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

## Report template

Present the result to the user in **this** order — the same order as the report sections. Fill
every `<…>` from the script's output; omit a stage/agent row that never ran and drop whichever
dispatch line (async/sync) does not apply. Every table keeps wall + active time and the token
types (input / cache / output / total).

```markdown
### `<op-name>` harness run — session `<session-id>`

**Shape:** 1 orchestrator + `<N>` subagents · Stage `<first>`→`<last>` · wall `<wall>` · active `<active>`.

#### 1 · Overall costs

| Metric | Value |
|---|---|
| Wall / active | `<wall>` calendar · **`<active>`** active (reason `<r>` · tools `<t>` · setup `<s>`) |
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
- **Wall vs. active:** `<wall>` calendar but only `<active>` of real work — the rest was
  `<user_wait + idle>` of human/idle time.
- **Recommendation:** `<one or two concrete, data-driven next steps>` (e.g. cut repeated Bash
  verify re-runs, reduce `MEMORY.md` edit churn, parallelize a serial stage, trim a long tail).
```

Close by offering a next step: drill into a specific agent, request the per-prompt breakdown from
`--json`, or re-run against another session (`--list-sessions`).

## Reading it honestly

- **Active agent-time ≫ wall span** is normal — agents run concurrently; the overall section shows both.
- **`dispatch_wait` overlaps children** and is excluded from active time.
- **Thinking time is a proxy** (generation latency); the persisted-block count is shown for transparency.
- **`user_wait` / `idle`** are human/away time, not compute; tune `--idle-threshold` for legitimately long reasoning.
- **`cache-rd` dominates by design** — cached context is re-read (at the cheap cache-read rate) every call, so it is cumulative, not fresh input; read `input` + `output` for new tokens. Per-agent/per-stage `cache` = cache-read + cache-write; the overall section splits them. No dollar cost is computed.
- **File interactions cover `Read` / `Write` / `Edit` only** — `Bash cat`, attachments, and skill loads are not counted. Calls are counted by tool name; size metrics (lines, edit ±lines) come from the result, so an errored call has a call but no size (`calls ≥ op-files`). Edit ±lines come from `structuredPatch` hunks; a `Write` is classified new (create) vs overwrite (update).

## Files

- `scripts/monitor_cost.py` — the metrics extractor (text report and `--json`).
- `references/log-schema.md` — log layout, timing model, dispatch join, stage/role map.
