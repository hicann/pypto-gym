# Log schema, timing model & stage resolution — `monitor_cost.py` reference

Read before extending the extractor. It **supports two transcript sources** — Claude Code
(on-disk JSONL, §2) and opencode (`opencode export` JSON, §3). Both parse into one **shared
record** (`parse_transcript` / `parse_opencode_session`, closing through `_finalize_record`) via
one **shared tool classifier** (`classify_tool` over a per-source `SourceSpec`), so the cost model
(§1), aggregation, rendering, and `--json` are identical for both. Each source section only states
how that source supplies the shared concepts.

## 1. The shared cost model

Every transcript — orchestrator or subagent, either source — becomes one record whose wall time
is partitioned additively (no overlap) into six time buckets, plus two count dimensions (tokens,
file interactions). The main transcript is parsed as a pseudo-agent in its own right (role & stage
`orchestrator`) because the lifecycle calls — subagent dispatch, `Skill`, human prompts — live
there.

### 1.1 Time buckets

| Bucket | Meaning | In active? |
|---|---|---|
| **reasoning** | model-generation latency (the thinking proxy) | yes |
| **tool-exec[name]** | time a tool ran, per tool name | yes |
| **setup** | small prompt/attachment injection gaps | yes |
| **dispatch_wait** | time a parent spent on a subagent-dispatch result (overlaps the child) | no |
| **user_wait** | time waiting on a human | no |
| **idle** | a reasoning/setup gap longer than `--idle-threshold` | no |

**Active** = reasoning + tool-exec + setup. The rest are excluded — `dispatch_wait` overlaps the
child; `user_wait` / `idle` are not compute. Rules (shared, however a source measures durations):

- **Tool-exec is never capped.** A verify `Bash` (NPU compile+run) legitimately runs 30+ min;
  capping would corrupt the dominant real cost.
- **Human-wait is not compute.** A single answer can span hours, so it goes to `user_wait`, never
  tool-exec, keeping the tool rollup sane. (Claude's `AskUserQuestion`; opencode has no in-band
  equivalent, so its `user_wait` is normally 0.)
- **The idle cap applies only to reasoning/setup gaps** (default 600s, `--idle-threshold`). Real
  generation is seconds-to-minutes; a multi-hour "reasoning" gap is a pause/away, reclassified
  idle. Subagents run unattended and essentially never trip it — idle is almost entirely an
  orchestrator phenomenon.
- **Per-tool sums can exceed wall time** under parallel calls (overlap), same as concurrent agents.

**Run wall = the operator-generation window:** first activity → the last dispatched agent
finishing. The orchestrator SESSION can outlive the run (left open, reused); that idle tail is not
counted, and the root transcript's own span is clipped to the window so its per-agent wall matches
the run.

### 1.2 Token accounting

Four buckets per assistant event, summed per agent and per prompt (both reconcile exactly to the
run total): **input** (fresh prompt tokens), **cache_read** (cached context re-read this call —
cheap and **cumulative**), **cache_write** (tokens written to cache), **output** (generated).
`total = input + cache_read + cache_write + output`. `cache_read` dwarfs the rest — the growing
context is re-read every call, not new data; read `input + output` for genuinely fresh tokens. No
dollar cost is computed.

**Prompt grouping.** A *prompt* is one user turn. The parser tracks the in-flight turn id and
attributes each assistant event (and its file-tool calls) to it, keeping the turn's first text as
a short label. Anything before the first turn falls into a `__preamble__` sentinel (nothing is
dropped). A subagent usually has one prompt (its dispatch) → per-prompt == per-agent; a subagent
messaged mid-run gets one prompt per message.

### 1.3 File interactions — Read / Write / Edit

Per agent and per prompt: rd/wr/ed **calls**, **lines read/written**, **edit lines ±**, and
**distinct files** (union across the three ops). A **call is counted even if it errors**, while
size metrics need a successful result — hence `calls ≥ files`. A `Write` is classified create vs
overwrite. A run-wide per-path map feeds the "most-touched files" list (a file edited/re-read many
times is a context-churn signal, e.g. an operator's `MEMORY.md`). Only these three tools count —
`Bash cat`, attachments, and skill loads do not. Edit ± lines come from a structured patch/diff,
not a semantic diff.

### 1.4 Stage resolution

Each agent → one stage bucket (`orchestrator`, `1`–`7`, or `support`), identically for both
sources:

1. **Explicit wins:** description matching `Stage\s+(\d+)` (case-insensitive) — handles redesign
   loops (`Stage 4 redesign…`) and the verifier (legitimately in Stages 4–7).
2. **Role fallback** by agent type:

   | Agent type | Default stage |
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

3. **Module token:** `\bM(\d+)\b` from the description (`M1`, `M12`, …) exposes the Stage-5
   per-module coder → verifier → debugger cycles; `null` when absent.

`support` holds `spawnDepth ≥ 2` helpers and any agent whose stage can't be resolved; their time
counts in totals but isn't charged to a workflow stage. The verifier has no role-default on purpose
(genuinely stage-ambiguous) → an unlabeled verifier lands in `support`, not misattributed.

### 1.5 Subagent lifecycle & dispatch modes

Each subagent is joined to the call that spawned it through a **global** id→start / id→end index
across *all* transcripts (the call may live in the main transcript or, for nested helpers, another
subagent). Per subagent:

```
dispatch_s = end(join_id) - start(join_id)    # parent-observed duration
self_wall  = subagent transcript span
```

- **synchronous** (`dispatch_s ≥ self_wall − tol`): parent blocked; `overhead = dispatch_s −
  self_wall` = spawn+close overhead (~5s; no separate close event, so combined).
- **asynchronous** (`dispatch_s < self_wall − tol`): the dispatch returned immediately, the result
  arrived later, the parent did not block. Overhead is **not measurable this way** — reported as
  such, never a misleading `0.0s`. The child's cost is its own per-agent wall.

Both modes occur in real runs. The join id is `meta.toolUseId` (Claude, §2.2) or the child session
id (opencode, §3.3); `compute_lifecycle` is identical for both.

## 2. Source: Claude Code (on-disk JSONL)

### 2.1 Layout

```
~/.claude/projects/
  <project-dir>/                     # cwd with every non-[A-Za-z0-9] char → '-'
    <session-id>.jsonl               # the MAIN (orchestrator) transcript
    <session-id>/subagents/
      agent-<id>.meta.json           # static metadata about the dispatched agent
      agent-<id>.jsonl               # the subagent's transcript, one event per line
```

The leading `/` of `<project-dir>` becomes a leading `-`, which is why `--project` must be passed
as `--project=<name>` (a bare `-`-prefixed value is parsed as a flag by `argparse`). A `subagents/`
directory appears only after the orchestrator dispatches its first agent.

### 2.2 `agent-<id>.meta.json`

| Field | Example | Use |
|---|---|---|
| `agentType` | `pypto-op-coder` | team role → default stage (§1.4) |
| `description` | `Stage 5 M1: code latent→QKV` | primary source for stage + module |
| `toolUseId` | `toolu_01Dwg…` | the `Agent`/`Task` tool-call id that spawned this subagent — the lifecycle join key (§1.5) |
| `spawnDepth` | `1` | 1 = dispatched by the orchestrator; ≥2 = nested helper (→ `support`) |

### 2.3 Transcript events → the shared model

Newline-delimited JSON, one event per line. Keys: `timestamp` (ISO-8601 UTC ms, `Z` suffix; parse
via `fromisoformat(ts.replace("Z","+00:00"))`), `type` (`user`/`assistant`/`attachment`),
`message.content`, `message.usage`, `promptId`/`promptSource`, top-level `toolUseResult`.
`message.content` blocks: `text`; `tool_use` `{id, name, input}`; `tool_result`
`{tool_use_id, content}` (in a `user` event, matched to a `tool_use.id`); `thinking` (rarely
persisted). Id-pairing makes attribution robust regardless of parallel interleaving.

- **Timing** — events are chronological; the gap `t_i − t_{i-1}` is charged to a bucket by the
  *later* event `E_i`:

  | `E_i` is… | charged to |
  |---|---|
  | `assistant`, gap ≤ idle-threshold | **reasoning** |
  | `assistant`, gap > idle-threshold | **idle** |
  | `user` with tool_result, tool ∈ `{Agent, Task}` | **dispatch_wait** |
  | `user` with tool_result, tool ∈ `{AskUserQuestion}` | **user_wait** |
  | `user` with tool_result, any other tool | **tool-exec[name]** (never capped) |
  | `user` (plain prompt) / `attachment`, gap ≤ idle-threshold | **setup** |
  | `user` (plain prompt) / `attachment`, gap > idle-threshold | **idle** |

- **Tokens** — from each assistant event's `message.usage`: `input_tokens` → input,
  `cache_read_input_tokens` → cache_read, `cache_creation_input_tokens` → cache_write,
  `output_tokens` → output. (`usage` also carries an `iterations` breakdown; only the top-level
  totals are used, which equal the final iteration.)
- **Files** — the op is resolved from the following event's `toolUseResult` *shape* (independent of
  the tool name); calls are counted from the `tool_use` block by name:

  | Op | `toolUseResult` shape | Size metrics |
  |---|---|---|
  | **read** | `.file = {filePath, content, numLines, …}` | lines = `numLines`, bytes = `len(content)` |
  | **write** | `{type: create\|update, filePath, content, structuredPatch}` | lines = `len(content.splitlines())`, bytes = `len(content)`; `type` → create vs overwrite |
  | **edit** | `{filePath, oldString, newString, structuredPatch}` | added/removed = `+`/`-` lines across `structuredPatch` hunks |

  Disambiguated by distinctive keys: a `.file` dict → read; an `oldString` → edit; a `content` +
  `type ∈ {create, update}` → write.
- **Lifecycle join** — `meta.toolUseId` (§1.5).

## 3. Source: opencode (`opencode export <sessionID>`)

opencode has no on-disk tree; `opencode export <id>` prints one JSON object `{info, messages[]}`.

> **Export via a temp file, not a pipe.** opencode streams the export to stdout; a large session
> overruns the OS pipe buffer and a piped capture (`subprocess … capture_output`) truncates at a
> 64K/96K boundary with `rc=0`. `opencode_export()` redirects to a temp file (a shell `>` redirect)
> and reads it back.

### 3.1 Run assembly

A run's subagents are **child sessions**, not files. The root `ses_…` session is the orchestrator;
each `task` tool part in it carries `state.metadata.sessionId` = a child session id.
`load_opencode_run` BFS-walks these (exporting each), covering any nesting. A child's `info.agent`
is its team role and `info.title` (e.g. `Stage 5: … (@pypto-op-coder subagent)`) feeds the same
stage resolution (§1.4). `--list-sessions` reads the opencode sqlite store read-only (`session`
table, `parent_id IS NULL`) for top-level runs + child counts.

### 3.2 Events → the shared model

| Concept | opencode source |
|---|---|
| event | one entry in `messages[]` = `{info, parts[]}`; `info.role` ∈ `user` / `assistant` |
| timestamp | `info.time.created` / `.completed`, `state.time.start` / `.end` — **epoch ms** (÷1000) |
| generation segment | one `assistant` message |
| **reasoning** | assistant message span (`completed − created`) **minus** the tool time inside it; idle-capped past the threshold |
| **tool-exec[name]** | Σ per `tool` part `state.time.end − start`, keyed by lowercase name (`read`/`bash`/…) |
| **dispatch_wait** | Σ `task` part durations |
| **setup / idle** | gap between one message's end and the next message's start (≤ / > idle-threshold) |
| thinking blocks | count of `reasoning` parts |
| wall | span of all message + tool timestamps |

**Tokens** — each `assistant` message carries `info.tokens = {input, output, reasoning,
cache:{read, write}}`: input → input, `output + reasoning` → output (reasoning tokens are generated
output; folding them in makes `total` reconcile), `cache.read` → cache_read, `cache.write` →
cache_write. Summing per-message equals opencode's own `session.info.tokens` **exactly** (verified).

**Files** — from `tool` parts by name (`read`/`write`/`edit`), metrics from `state`:

| Op | Size metrics (path = `state.input.filePath`) |
|---|---|
| **read** | lines = count of `N:` prefixes in the `<content>` block of `state.output`; bytes = `len(output)` |
| **write** | lines = `len(input.content.splitlines())`, bytes = `len(content)`; `state.metadata.exists` → overwrite (true) vs create (false) |
| **edit** | added/removed = `+`/`-` lines of the unified diff in `state.metadata.diff` (skipping `+++`/`---` headers) |

### 3.3 Lifecycle join

The parent's `task` tool gives `state.metadata.sessionId` (child) and `state.time.{start,end}`. The
extractor sets each child record's `tool_use_id` = its own session id and populates the global
dispatch index (`use[child] = ("task", start)`, `res[child] = end`) from the parent, so the
unchanged `compute_lifecycle` (§1.5) derives dispatch duration, mode, and overhead as for Claude.

## 4. Known limitations (report these honestly)

- **Thinking time is a proxy** (generation latency). Explicit `thinking`/`reasoning` blocks are
  rarely persisted, so the block count is shown for transparency but is not the time source.
- **Idle-threshold is a heuristic.** The 600s default cleanly separates human-away gaps from real
  generation in observed runs, but a genuinely long single reasoning burst beyond it would be
  miscounted as idle. Tune with `--idle-threshold`.
- **Stage accuracy is description-dependent.** Mislabeled dispatches fall back to the role map or
  `support`.
- **Durations are wall-clock, not CPU** — they include time blocked on tools and sub-dispatches.
