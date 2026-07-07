# Subagent log schema, timing model & stage-resolution heuristics

Reference for `scripts/monitor_timecost.py`. Read this before extending the analyzer.

## 1. On-disk layout

Claude Code stores the orchestrator's transcript and one transcript pair per dispatched
subagent under the active session:

```
~/.claude/projects/
  <project-dir>/                         # cwd with every non-alphanumeric char → '-'
    <session-id>.jsonl                   # the MAIN (orchestrator) transcript
    <session-id>/
      subagents/
        agent-<id>.meta.json             # static metadata about the dispatched agent
        agent-<id>.jsonl                 # the subagent's full transcript, one event per line
      memory/                            # (unrelated to this skill)
```

- `<project-dir>` = working directory with every char not in `[A-Za-z0-9]` replaced by `-`.
  The leading `/` becomes a leading `-`, which is why `--project` must be passed as
  `--project=<name>` (a bare `-`-prefixed value is parsed as a flag by `argparse`).
- A `subagents/` directory appears only after the orchestrator dispatches its first agent.
- The analyzer parses the **main transcript as a first-class pseudo-agent** (role
  `orchestrator`, stage `orchestrator`) because the lifecycle tool calls — dispatching
  subagents (`Agent`/`Task`), `state_transition`, `Skill`, `AskUserQuestion` — live there.

## 2. `agent-<id>.meta.json`

| Field | Type | Example | Use |
|---|---|---|---|
| `agentType` | string | `pypto-op-coder` | Team role → default stage. |
| `description` | string | `Stage 5 M1: code latent→QKV` | Primary source for stage + module. |
| `toolUseId` | string | `toolu_01Dwg…` | The `Agent`/`Task` tool-call id that spawned this subagent — **the join key for §D lifecycle**. |
| `spawnDepth` | int | `1` | 1 = dispatched by the orchestrator; ≥2 = nested helper. |

## 3. `agent-<id>.jsonl` / main transcript

Newline-delimited JSON, one transcript event per line. Relevant keys: `timestamp`, `type`
(`user` / `assistant` / `attachment`), `message.content`.

- **`timestamp`** — ISO-8601 UTC, millisecond precision, `Z` suffix
  (`2026-06-24T10:17:41.240Z`). Parse via `fromisoformat(ts.replace("Z","+00:00"))`.
- **`message.content`** — when a list, each block has a `type`:
  - `text` — natural-language output.
  - `tool_use` — `{type, id, name, input}`. A tool invocation.
  - `tool_result` — `{type, tool_use_id, content}` (inside a `user` event). Matches a
    `tool_use.id` by `tool_use_id`.
  - `thinking` — extended-thinking block (rarely persisted; see §6).

Parallel tool calls appear as several `tool_use` blocks/events followed by their results;
**id-pairing makes attribution robust regardless of interleaving**.

## 4. Timing model (the core of the analyzer)

Events are chronological. For each consecutive pair the **gap** `t_i − t_{i-1}` is charged
to a bucket by the *later* event `E_i`. The buckets partition the transcript's wall time
with no overlap:

| `E_i` is… | charged to |
|---|---|
| `assistant` event, gap ≤ idle-threshold | **reasoning** (model generation; the thinking proxy) |
| `assistant` event, gap > idle-threshold | **idle** (paused, not real generation) |
| `user` with tool_result whose tool ∈ `{Agent, Task}` | **dispatch_wait** (waiting on a child) |
| `user` with tool_result whose tool ∈ `{AskUserQuestion}` | **user_wait** (waiting on a human) |
| `user` with tool_result, any other tool | **tool-exec[name]** (never capped) |
| `user` (plain prompt) / `attachment`, gap ≤ idle-threshold | **setup** |
| `user` (plain prompt) / `attachment`, gap > idle-threshold | **idle** (human away between turns) |

Key rules and why:

- **Tool-exec gaps are never capped.** A verify `Bash` (NPU compile+run) legitimately runs
  30+ minutes; capping it would corrupt the dominant, real cost.
- **`AskUserQuestion` is human-wait, not compute.** A single answer gap can span hours or
  days; routing it to `user_wait` (not tool-exec) keeps §B sane.
- **Idle cap applies only to reasoning/setup gaps** (default 600s, `--idle-threshold`).
  Real model generation segments are seconds-to-minutes; a multi-hour "reasoning" gap is an
  agent paused or a user away, so it's reclassified as idle. Subagents run unattended and
  essentially never trip this — idle is almost entirely an orchestrator phenomenon.
- **Active agent-time** = reasoning + tool-exec + setup. `dispatch_wait`, `user_wait`,
  `idle` are excluded (the first overlaps children; the others are not compute).

Per-tool execution time is the sum of the gaps charged to that tool. With parallel calls
these per-tool sums can exceed wall time (overlap) at the run level — same busy-vs-wall
caveat as concurrent agents.

## 5. Subagent lifecycle (§D) and dispatch modes

`meta.toolUseId` is the id of the `Agent`/`Task` `tool_use` that spawned the subagent. The
analyzer builds **global** id→start and id→end indexes across *all* transcripts (the
spawning call may live in the main transcript, or in another subagent for nested helpers),
then for each subagent:

```
dispatch_s = end(toolUseId) - start(toolUseId)     # parent-observed duration
self_wall  = subagent transcript span
```

Dispatch is then classified:

- **synchronous** (`dispatch_s ≥ self_wall − tol`): parent blocked until the child returned.
  `overhead = dispatch_s − self_wall` = **spawn + close overhead** (typically ~5s/agent).
  There is no separate "close" event, so spawn and teardown are reported combined.
- **asynchronous / background** (`dispatch_s < self_wall − tol`): the `Agent` call returned
  immediately and the result arrived later; the parent did not block. Overhead is **not
  measurable this way** and is reported as such — never as a misleading `0.0s`. The child's
  cost is its own §A wall.

Both modes occur in real runs, so the analyzer handles and labels both.

## 6. Stage resolution

Each agent → one stage bucket (`orchestrator`, `1`–`7`, or `support`):

1. **Explicit wins:** `description` matching `Stage\s+(\d+)` (case-insensitive). Handles
   redesign loops (`Stage 4 redesign…`) and the verifier (legitimately in Stages 4/5/6/7).
2. **Role fallback** by `agentType`:

   | `agentType` | Default stage |
   |---|---|
   | `pypto-op-planner` | 1 — Planning |
   | `pypto-op-mathematician` | 2 — Algorithm |
   | `pypto-op-architect` | 3 — Architecture |
   | `pypto-op-designer` | 4 — Design |
   | `pypto-op-coder` | 5 — Construction |
   | `pypto-op-debugger` | 5 — Construction |
   | `pypto-op-optimizer` | 7 — Optimization |
   | `pypto-op-verifier` | *(no default — relies on explicit description)* |
   | `Explore`, `general-purpose`, other | `support` |

   The verifier has no role-default on purpose (genuinely stage-ambiguous); an un-labeled
   verifier lands in `support` rather than being misattributed.
3. **Module token:** `\bM(\d+)\b` from the description (`M1`, `M12`, …) exposes the Stage-5
   per-module coder → verifier → debugger cycles. `null` when absent.

`support` holds `spawnDepth ≥ 2` helpers and any agent whose stage can't be resolved; their
time counts in totals but isn't charged to a single workflow stage.

## 7. Known limitations (report these honestly)

- **Thinking time is a proxy** (model-generation latency). Explicit `thinking` blocks are
  rarely written to disk, so the persisted-block count is shown for transparency but is not
  the time source.
- **Idle-threshold is a heuristic.** The 600s default cleanly separates human-away gaps from
  real generation in observed runs, but a genuinely long single reasoning burst beyond the
  threshold would be miscounted as idle. Tune with `--idle-threshold`.
- **Clock is UTC `HH:MM:SS` (no date).** A multi-day / multi-operator session shows a wall
  span of many hours; read it together with `user_wait` + `idle` and the per-agent starts.
- **`description`-dependent stage accuracy.** Mislabeled dispatches fall back to the role map
  or `support`.
- **Wall-clock only.** No CPU/token cost dimension; durations include time spent blocked on
  tools and sub-dispatches.
