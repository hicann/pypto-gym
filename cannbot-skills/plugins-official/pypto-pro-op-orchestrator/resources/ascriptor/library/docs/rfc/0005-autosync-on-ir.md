# RFC-0005: Automatic synchronisation on the IR

Status: implemented (`passes/deps.py`, `passes/autosync.py`, `passes/session_sync.py`,
`passes/local_mutex.py`, `passes/events.py`). A5 uses local slot mutexes (below) and family `a2`
uses slot sessions (§5); §2 is the shared analysis, and where §5 differs from the original
edge-planning design, §5 is what the code does. RFC-0006 places the passes in the pipeline.
Depends on: RFC-0001 §6 (access sets), §7 (structured control flow), §10 (pipes on lowered ops).

## A5 local buffer mutex policy (2026-09-15)

On A5, `auto_sync` regions use mode-zero buffer ownership. There is no strategy
option and no event fallback: family `a2` uses the slot sessions of §5 instead,
and the edge-driven event planner is removed. An allocation has one implicit
mutex identity per physical slot, local to each AIC or individual AIV participant.
All banks share that participant's 32-ID budget. Views inherit their backing
slot's identity; dynamic slot selection and its mutex ID are captured together.
Duplicate operands acquire one identity once. Exhaustion is a located error,
never implicit wrapping or aliasing. Desugared scratch participates too.
IDs follow allocation order in the side's IR, including the default two-slot
L0A and two-slot L0B buffers created by matmul desugaring. Their four IDs count
against the same 32-ID limit before any backend runs; typed views inherit those
IDs. No separate high-ID reservation or backend scratch allocator exists.

Every actual managed memory operation acquires its distinct local identities
before execution and releases them afterwards, always with mode zero. VF buffer
accesses are protected at the call boundary. Metadata-only view operations do
not acquire locks. Cross-core ownership, explicit events/barriers and GM
protocols remain distinct. A5 ignores `sync_depth`: four physical slots still
have four identities even when the declaration carries `sync_depth=2`, which is
a window cap for the event-based family only (§5.3, M10-082). No throttle or
acknowledgement protocol exists on either route any more.

`sync.local_mutex_get/release` are explicit Lowered IR effects. Side-local ID
assignment follows address allocation and side splitting. A dedicated IR pass
may remove a same-side, same-pipe, same-mode, equal-ID release/get pair within
one Block, crossing only proven unrelated scalar operations. Those operations
remain in order. No branch/loop boundary, synchronization, memory access or
unknown effect is crossed. Mutable slot changes require an equality proof;
already captured immutable IDs are stable. This is critical-section merging,
not a same-pipe memory-completion barrier.

The pipe simulator binds local mutex predecessors in scalar issue order and
transfers completion/vector-clock dependencies at release retirement. It must
not model a race to acquire an initially free host lock. Every execution lane
has its own 32-entry state; runtime IDs survive trace export and import.
PyPTO-Pro copies these allocation IDs into native Tile groups and comments out
local mutex instructions when native `auto_mutex` is enabled; see [RFC-0013](0013-pypto-native-synchronization.md).
CCE and PTO-ISA emit the explicit mode-zero calls.
Functional pairing checks, dynamic hazards, native compilation and silicon
execution are separate evidence. Mutex timing is a model estimate.

## 1. Why redo it

The old `asc_autosync.py` (~1400 lines) works on an instruction list with three hand-written
tables (read / write field names per pipe, 141 explicit vector opnames, "classic pairs"), scans
`start_loop` / `end_loop` markers for control flow, aliases buffers by name, and handles cross-side
synchronisation only through the manual mutex protocol. Every table is a copy of knowledge the
registry now holds, and every special case is a consequence of not having a dependency graph.

## 2. Two layers

### 2.1 `deps`: the memory dependency graph

Built per kernel function — both sides in one graph, before `split_sides` (D-029) — from data the IR already carries:

* **access sets** from the registry — every operand and value-attribute is `read`, `write` or
  `readwrite`; no per-op table in the pass;
* **aliasing** from the explicit view ops — `mem.slice`, `mem.get_buf`, `mem.reinterpret` chains
  give every value a root allocation plus offset / extent intervals; two accesses conflict when the
  roots match and the intervals overlap (exactly when static, conservatively when an offset is a
  scalar value);
* **control flow** from `cf.for` / `cf.if` regions — program order inside a block, and for loops the
  body analysed cyclically so that **loop-carried** edges (iteration *i* writes what iteration *i + d*
  reads) get a distance *d*;
* **slot buffers** — `buf<T, n>` indexed by a cell that advances by one per iteration (the double /
  triple / quad buffering idiom) rotate through *n* slots, so accesses to the same buffer conflict only
  at distance *n*: the edge distance is the event depth the old `SEvent / DEvent / TEvent / QEvent`
  encoded by hand.

Edges carry kind (RAW / WAR / WAW), the two pipes, the distance, and the op ids. The graph is reused by
DCE, liveness / `addr_alloc` (interference), the sim's hazard checker (dynamic trace vs static graph),
and `ascriptor explain`.

### 2.2 `autosync`: from the graph to a protocol

Only conflicts between *different pipes on the same side* need synchronisation (same-pipe ops
execute in order). What the pass makes of them depends on the device family: A5 takes mode-zero
slot mutexes (above), family `a2` takes paired `ready` / `valid` slot sessions (§5). Both read the
same graph, and in both the graph judges rather than plans: the protocol follows the structure of
the code — the physical slots a buffer declares, and the points where one pipe hands work to
another — and every edge of the graph must then be shown to be covered by it (§5.9). The design
this RFC started from planned each edge on its own and reduced the result transitively through
per-pipe vector clocks; §5 records what that cost and why it was replaced. The clocks remain in
`deps` for the coverage question and for `crosssync`.

`events` (a separate pass) assigns channel ids per `(set_pipe, wait_pipe)` by interval colouring
under the hardware limit.

**"Same-pipe ops execute in order" is an ordering of *issue*, not of *landing*, and it is the
kernel author's job where two writes on one pipe overlap in memory.** Autosync inserts nothing
between them, by construction — there is no cross-pipe edge to place an event on — and the board
does not order them for you. Two cases are board-proved. Two DMA writes to overlapping GM are
wrong in 5 of 5 runs without `bar_mte3()` and right in 5 of 5 with it, every wrong element holding
the *first* store's value (D-065; that probe and its scan script stayed in the old repository, and
`examples/api/gm_views/` is the live shape). An L1 fill overlapping an L1 load —
`set_constant_to_l1` over a whole slot, then a copy into a prefix of it — is wrong in 4 of 7
loaded runs, and here **`bar_mte2()` between them does not fix
it** (3 of 4 still wrong): a `create_cbuf_matrix` fill and a copy are not ordered against each other
by a pipe barrier either. The repair for that shape is therefore structural rather than a barrier —
write disjoint regions, or contract only the rows you loaded with `k=` and never write the rest
(D-214).

## 3. Coverage the old pass could not have

* a fixed slot index (`l0c_qk[0]` of a three-slot buffer, one slot per
  role) is the same slot every iteration: its carried distance is one, not the slot count — a
  first version said three, and `v8_allhif8`'s fixed L0C slots ran three iterations unguarded.
* every op in the registry, including `cf.call` (callee access sets), `simt.launch`, `list.*`,
  atomic modes, and value-attributes such as masks and bias tiles;
* **cross-side checking** — *moved to the `crosssync` pass; see the amendment below.* Insertion of
  cross-side sync stays manual (the mutex placement is a pipelining decision), with automatic
  insertion as a later option, and that half remains this pass's position;
* the M2 reference interpreter (Surface level) is the oracle: M4's pipe-level sim executes the lowered
  module with a hazard checker; no hazard on the corpus = sufficient, and the event count per kernel
  compared with the old output = leaner.

### 3.1 Amendment: cross-side checking belongs to its own pass

Listing cross-side checking here put a cross-core check inside the intra-core event inserter, and
it inherited a trigger condition that has nothing to do with it. This pass begins by inlining
`region.autosync` and returns immediately when a function contains none — so a kernel that drives
the mutex protocol by hand, the one whose author is personally responsible for the credits, was
checked not at all. Nothing else covered it either: `gmbuff` checks `depth <= slots` for a `GMBuff`
workspace ring only, and `auto_mutex` (the "later option" above) is a pypto_pro backend feature
(RFC-0013), not a pass, and does not reach cce.

The checking therefore moves to **`crosssync`**, a read-only pass placed beside `gmbuff`:

    ... → device_lower → gmbuff → crosssync → autosync → events → ...

It asks the same two questions, of every cross-side edge of every function:

* **coverage**: every RAW / WAR edge between the cube and the vector side over L1 / UB / GM must be
  covered by a mutex `ready` → `wait` pair (or a cross-core flag), naming both op ids and the buffer;
* **credits**: a hand-BACK — the consumer's read of one cycle against the producer's write of the
  next — needs the other edge, `free` → `lock`, and that one is a counting semaphore: `lock` of cycle
  `i` blocks on `free` of cycle `i - depth`. It orders the pair only when `depth` is at most the
  number of cycles between the two, which is the number of slots the buffer rotates over. More
  credits than slots is a defect, and is reported only where no other cross-core wait could order
  the pair (M10-076).

It tracks GM, as the coverage question above names it. That was unaffordable until two precision
repairs in `deps` landed with M10-078: a window reached through a `mem.reshape` collapsed to the
whole root, although its offsets were exact and `View.dims` already recorded which shape they
count in; and a carried edge over a window that moves with a runtime offset was convicted,
although `deps` falls back to distance one for any index it cannot read (§2.1) and therefore does
not know the two trips touch the same bytes — `credit_hazard` already declined those. Measured
over both repositories: before, GM added 298 cross-side edges and 61 findings, every one false
(180 of the spurious edges were one kernel writing disjoint tiles of one output from both sides);
after, it adds 118 edges and no finding. The conservative answers that remain are two windows in
different coordinate systems, and a carried edge the analysis cannot place.

The analysis is one function (`crosssync.cross_side_findings`); `autosync` no longer checks these
edges at all, and only records them so that it does not try to insert events across a side.
`crosssync` refuses a credits finding and warns a coverage one, under the existing
`autosync_cross_side` option (`warn` | `error` | `off`).

The gate was removed in two steps. The first ran the analysis over every cross-side edge of both
repositories and reported rather than refused, because how much of what a removed gate exposes is
a defect rather than a shape the analysis cannot see is a measurement, not a deduction. The
measurement: 106 units, 51 functions, **1230 cross-side edges, every one of them ordered**; four
credits findings, all in one unit that does not lower today either way; and 50 of the 51 functions
declaring a region, so the gate had been costing the shipped corpus nothing. With the backlog
measured at zero, refusing was free, and the second step took it. `crosssync_report` (option, or
`ASCRIPTOR_CROSSSYNC_REPORT`) prints the census again.

## 4. What the IR already provides, and what may be added

Access sets, explicit views, structured loops and cells are in place (M1). The only candidate
addition is a canonical marker for slot-counter cells (a `cf.for` iteration index the counter is
tied to) if the induction analysis in §2.1 proves fragile; decide when writing `deps`.

## 5. As implemented: A2/A3 slot sessions (2026-09-16)

Device family `a2` — A2 and A3 — plans `auto_sync` regions as **slot sessions**: one paired
`ready` / `valid` window per producer/consumer role switch, whose capacity is the physical slot
count of the buffers it protects. Family `a5` uses the local mutexes above. There is no strategy
option and no event fallback; both routes share `deps`, `crosssync`, `events`, `check_balance`
and the pipe simulator.

This replaces the edge-driven event planner, in which the unit of protection was one hazard edge
and each direction of a slot ring was planned and sized on its own. Removed with it: depth
inference by run-ahead replay, acknowledgement synthesis, armed cells, conditioned mirrors, the
`autosync_coalesce` and `autosync_vacuous_carried` policies, carried distance as depth, the
proved L0 paired lease / duplex stream / split-K ring, the L1 physical-slot lease and the
`local_ready` bookkeeping pass. Their reasoning and measurements stay in Git history and in the
receipts of the defects that produced them (M10-008, M10-068, M10-071, M10-073, M10-074, M10-082,
M10-083); a removed mechanism's receipt remains historical evidence for the source it ran on, and
none of it transfers to this planner.

### 5.1 Identities

* **family** — a backing allocation (a `mem.alloc` root). Views, slices, `mem.reinterpret` and
  `mem.reshape` inherit their root's family, exactly as on the A5 mutex route.
* **ledger** — one `ready` / `valid` event pair. An event of depth *W* occupies *W* flag ids of
  its channel and rotates its tokens through them, so a ledger costs *W* ids forward and *W*
  reverse. A depth that publishes declares **both** halves whenever the session has credits,
  including a depth whose publications are all made inside a subtree: half a ledger is not a
  ledger, because each half is what bounds the other side (§5.5), and a depth carrying the reverse
  half alone bounds nothing — M10-096 and
  §5.10. The one window whose publication is not made at its own depth is §5.4's leading consumer
  phase, which has no producer work of that depth to anchor a `ready.set` behind; §5.3 caps such a
  depth at one credit, and the corpus census of §5.10 finds no depth that consists solely of it.
* **session** — a ledger with a producer pipe *P* and a consumer pipe *C*. The pipe pairs are
  derived from the dependency graph: a pair exists when `deps` reports a cross-pipe conflict on a
  member of that ledger. No table of opcodes, pipes or "classic pairs" is written down, which is
  the one thing §1 objected to that a fixed session table would have brought back.
* **window** — one execution of a session's critical section.
* **scope** — the innermost `Block` that contains every access of that session.
* **depth** — a nesting level of that scope at which role switches are ordered. A block whose own
  items switch roles runs the protocol itself and owns a ledger; a block whose every role-bearing
  item hands over inside one of its subtrees orders nothing of its own and is transparent, so its
  subtrees keep the enclosing depth's ledger and number their windows in one sequence, however
  deeply each of them sits. A session therefore has one ledger per depth that opens windows, and
  the credit of one depth is never the credit of another — which is what keeps a hand-off nested
  inside another window from waiting on itself.

### 5.2 Ledgers merge by capacity class

By default one ledger serves every allocation of one side, one channel and one capacity
(`session_ledgers = "capacity"`). Merging is conservative: the members of one ledger are ordered
against each other as well, never less than their own dependencies require, and the cost is
overlap rather than correctness. Capacity classes stay apart because a one-slot `Tensor` inside
the ledger of a two-slot `DBuff` would cap the pair at one credit and retire that double
buffering to save one flag id. `"channel"` merges the classes too, and `"allocation"` gives each
allocation its own ledger; both exist for controlled comparison and neither changes the protocol.

What the default costs is measured, not assumed, and the
receipt holds the census: a minority of
(kernel, session, allocation) triples hold fewer credits at every depth than that allocation
declares slots, in equal part because the ring analysis cannot read an induction variable indexing
a group buffer and because a merge whose members live in different loops lifts the session's scope
above the loop that advances their counters. `"allocation"` recovers some of them by spending
events and taking the busiest channel to all eight of its ids; `"channel"` spends fewer and loses
more. Recovering that headroom would buy overlap in the currency the busiest channels have least
of, which is why the default stays and the headroom is recorded rather than spent — the whole
overlap price against the retired planner is +3.3 % of the model cycles of one measured case.

### 5.3 Capacity, declared on both directions

Per member and per depth: its physical slot count, capped by its `sync_depth`
(M10-082) and by its **ring** at that depth — the number of windows of that depth after which the member's
physical slot is written again. Two windows share a slot only when the counter that indexes it has
advanced a multiple of the modulus, so the ring is the smallest window gap that can accumulate
that many advances, searched exactly over the advances attributed to each window. **Zero is such a
multiple**: the rotation separates two windows the counter does not advance between by nothing at
all, and at equal offsets they are the same slot, so the ring is that gap however small — a member
written twice with one advance at the end of the body therefore admits one credit, not the two its
slot count suggests. The test is on the accumulated advance alone, so it is conservative where two
windows write *different* offsets of one counter (`buf[c]` against `buf[c + 1]`, the hand-written
software pipeline): those are distinct slots, and reading their congruence rather than the advance
would keep the credit. No catalogue kernel writes that shape, so the exact rule stays unwritten
until one does. A member whose
slot index is not a counter the analysis can read (a fixed index, several counters, a value loaded
at run time, an advance that is not a literal) has ring one. The ledger's capacity `W` at that
depth is the smallest member value, and at least one.

A depth whose block **opens in a consumer phase** carries one credit whatever the slots say. That
consumer holds the credit the execution starts with, and returning it is what the first producer
phase has to wait for, so a spare credit would let the producer past a reader it has to follow.

`ready` (P to C) declares depth `W` with preset zero; `valid` (C to P) declares depth `W` with
preset `W`. The two directions describe one window of `W` credits, which is the invariant
M10-083 requires: producing a slot consumes one reverse credit before it publishes one forward
token, so the publications that can be outstanding are exactly the credits the reverse window
admits, and neither direction is inferred independently of the other.

A desugared split-K `_subk` body is an ordinary instance of the rule. Its window loads `_l0a` and
`_l0b` on MTE1 and reads them in one MMAD on M; both are two-slot rings written once per window,
so the ledger declares `ready` depth two (MTE1 to M, preset zero) and `valid` depth two (M to
MTE1, preset two) — the protocol M10-083's checkpoint had to prove structurally for that one
shape, now derived for every shape. What M10-083 requires is the *pairing*, not the number: where
two windows of a depth write the same slot because the counter does not advance between them, the
same derivation pairs them at one credit, and a kernel with two such regions per step gets exactly
that (`tests/passes/test_splitk_ready_depth.py`). No name prefix, desugar origin or fragment count enters the
rule, so the M10-081 M-pipe settle barrier inside such a body no longer disqualifies it.

### 5.4 Insertion: one window per role switch

Walk each session's scope block in program order. A **producer access** is an op that writes a
member on `P`; a **consumer access** is an op that reads one on `C`, or writes one there in a
WAR / WAW relation. Each item of the block is one of four things to the session: a producer, a
consumer, an **internal** hand-over that a single region of it performs on both sides, or
transparent. A branch whose arms hold different roles never runs both, so it is one access of its
own depth — the producing arm decides the window, and the other arm's reads follow a publication
that has already been made. The window has three states — closed, open (a credit is held),
published:

* producer or internal access, closed → emit `valid.wait` on `P` before it; open.
* producer access, open → nothing. **Consecutive producer work merges into one window**, which is
  where this planner's flag economy comes from.
* producer or internal access, published → emit `valid.set` on `C`, then `valid.wait` on `P`; open.
* consumer access, open → emit `ready.set` on `P` and `ready.wait` on `C` before it; published.
* consumer access, published → nothing; consecutive consumer work merges too.
* an internal access → the publication it needs is made inside it, at its own depth (§5.1), but
  the bound on the consumer is not: emit `ready.set` on `P` and `ready.wait` on `C` **after** it,
  then published; the walk descends into its regions. The consumer therefore cannot reach this
  window's hand-back before the producer has published the window, which is the half §5.5's
  argument needs and M10-096 was the absence of.
* consumer access, closed → the block opens in a consumer phase: emit `valid.wait` on `P` before
  it and treat it as published, so that the first producer phase closes that credit before taking
  it again. This is the only window whose publication was not made at this depth, and §5.3 caps
  the depth at one credit for it.
* end of a block that opened a window: published → `valid.set` on `C`; open → `ready.set`,
  `ready.wait`, `valid.set`, an empty handover that publishes the producer's work for a reader of a
  later window and returns the credit it took.

Every other op is transparent — another pipe's work, scalar computation, metadata views, explicit
user events and barriers. A window therefore balances on every path: a skipped arm or a zero-trip
loop emits nothing at all, and no token survives a block execution. A `cf.break` or `cf.continue`
reachable while a window is open is a located error, because its closure would be skipped; a
terminator that ends the block closes the window before itself, and a loop catches its own break.

Where several depths hold windows, the outer credit is taken before the inner one and returned
after it. A hand-off written as a producing loop followed by a consuming loop is the shape this
serves: the outer depth's window spans each loop, so the consumer's reads are inside the window
that produced for them and its hand-back follows them, while the publication inside each loop —
one per trip, throttled by the inner credit so no flag is set twice — belongs to the inner depth.

### 5.5 Why no run-ahead analysis, acknowledgement, cell or mirror is needed

A hardware flag must never be set while its token is outstanding. The legacy planner enforced
that by bounding the producer's run-ahead: it replayed the placed sets and waits with loops
unrolled, raised depths to the bound, and synthesised a reverse acknowledgement event where the
replay found nothing that throttled the set pipe. A set lifted behind a branch that might skip its
producer had no bound at all, which is what armed cells repaired, and a consumer behind a branch
the set could not see is what conditioned mirrors repaired. M10-008 records that this phase model
was approximate.

A session needs none of it. Its producer takes one of `W` credits before every window and its
consumer returns one after it, so at most `W` publications are outstanding at any time, on every
path, whatever the branches do: **the reverse channel is the acknowledgement**. A hoisted set is
safe with no work in front of it, because it cannot overtake the credit it already consumed; a
hoisted wait only consumes; and the depth is the declared capacity rather than a replay result.
M10-008 is closed with the mechanism that carried it.

That argument is symmetric and needs both halves of the ledger, which is why §5.1 declares both at
every depth. The reverse half bounds the producer: it cannot open window *N+W* before the consumer
has returned window *N*'s credit. The forward half bounds the consumer: it cannot reach window
*N*'s hand-back before the producer has published *N*. Take the forward half away and the second
bound goes with it — the consumer's returns are then throttled by nothing, it sets the reverse flag
while a token of it is still outstanding, and the hardware rule at the head of this section is
broken by the very channel that was meant to enforce it. That was M10-096.

That phase model owned five IR attributes — `armed_cell`, `armed_when`, `armed_arms` and `mirror`
on `cf.if`, `ack_of` on `sync.event` — and one rule in the `events` pass: an event the token oracle
found able to end a block execution with a token in flight had its flag id pinned to the function
end, because reusing that id hands the leftover token to the next event, which hung the
`attn_backward` MTE3->V channel on silicon (D-066, aicore abort 507015). **Both retire with the
planner that wrote them (2026-09-17).** Nothing writes those attributes any more, the frontend
exposes no surface that would let an author write one, and `check_balance` no longer evaluates
guard cells. The rule is not weakened: a window belongs to one block execution and closes at its
end, so no token survives one (§5.4), and A5 declares no events at all. A hand-authored protocol
that does leave a token is a balance error naming the event, not a pinned id, and the catalogue
leaves none: `check_balance` comes back empty in every model gate run the receipt records, and the
old corpus's `balance_xfail` table, which once excused such a tail, is not part of this checkout at
all. Were one ever accepted, the pinning
returns as one condition on the oracle's report, and its removal here is provably inert: every
catalogue lowering is byte-identical without it (§5.9).

### 5.6 Refused shapes

Each refusal names both ops, the family, the channel and the distance, and is an error rather
than a silently weaker plan:

* a cross-pipe pair no depth of the session holds in its windows — one op inside a window and the
  other outside every window of that depth, at every depth. The message lists what each depth
  refused, which is how the shape is read back: an access the walk reached at no depth that also
  sequences the other.
* a cross-pipe RAW whose consumer runs inside the producer's own window phase, before it publishes.
  A consumer in the producing arm of a branch is this shape; the diagnostic names
  `session_ledgers="allocation"` when a finer ledger would separate the two.
* a WAR / WAW hand-back whose reader and writer share one window, since no credit separates them.
* a member that flows both ways between two pipes — a RAW in each direction. One session cannot own
  both hand-offs, and the refusal names the storage to separate or publish by hand.
* an op that both produces and consumes one member in a single operation on two pipes.
* `break` / `continue` reachable while a window is open (§5.4).
* more than eight ids on one ordered channel after the deterministic coarsening of §5.7.

Reported rather than refused, because no arrangement of events repairs them and the author's
`sync.barrier` does: a pair whose earlier op runs on the scalar pipe, which can set no flag in
either role, and a WAR whose writer is the scalar pipe, for which nothing can publish.

### 5.7 Ids and the budget

Eight flag ids per ordered pipe pair. The planner counts its demand before it emits — the sum of
`W` over every depth of every ledger on a channel, plus what the author's own events declare
there, with no live ranges, so it over-counts rather than under-counts — and coarsens
deterministically until it fits: the two narrowest ledgers of the fullest channel merge, the pair
whose overlap is worth the least, and are re-planned. Only then does it emit. A remaining overflow
is a located error carrying the ledger and slot census, never a clipped depth. The `events` pass keeps its
interval colouring per channel, widening live ranges over loops, pinning pre-set events to the
whole function and reserving literal `set_flag` / `wait_flag` ids; its budget error remains.

### 5.8 What a session does not order

Unchanged from the edge planner, and unchanged by merging: cross-side pairs are never events
(each event records the `side` of its pipes and `split_sides` routes it; cube-to-vector ownership
is `crosssync`'s question and, on the A2 family, travels through a GM workspace); GM windows are
not an event's to order; two writes that overlap in memory on one pipe are ordered in issue, not
in landing, and remain the author's business with the two board-proved cases of §2.2 and the
C220 exceptions (M10-081's M settle, the V-V barrier); ops outside every `auto_sync` region are
the author's, reported as a warning when nothing orders them.

### 5.9 Verification

1. **Retention** — the regressions re-read the emitted protocol from the rewritten body rather
   than the planner's tables: `tests/passes/test_session_sync.py` reads back each declared ledger
   as `(set pipe, wait pipe, depth, preset, guards)` and the order of a window's four marks around
   the work they enclose — `valid.wait` … writes … `ready.set` / `ready.wait` … reads …
   `valid.set` — over the shapes of §5.4 and §5.6, and the token replay of 3 shows each of them
   equal on every path. The pass itself re-reads nothing after it emits: it emits from the marks
   its walk placed, and coverage (2) is checked against that walk before a single op is inserted.
2. **Coverage** — every managed cross-pipe RAW / WAR / WAW edge `deps` reports must be covered at
   some depth of its session: a forward pair by the publication of the window holding both ops,
   a hand-back by a credit whose capacity is at most the rotation distance. A pair whose reader
   belongs to a **later execution** of the scope is covered by the publication that closes the
   writer's own window: the producer pipe issues that set behind the write, and the consumer pipe
   takes its wait before any work of a later execution, so the two are in order whatever the
   distance — which is why no delayed-handoff analysis appears here. An uncovered edge is a
   located error, not an omission.
3. `check_balance` replays the tokens along every path and stays the public diagnostic.
4. The pipe simulator's happens-before hazards, deadlock report and physical flag occupancy per
   (lane, channel, id) remain the oracle; the functional interpreter remains the arithmetic
   oracle. Model cycles are a model estimate, separate from silicon.
5. A corpus census over the A2/A3 catalogue compares events, ids per channel, sets and waits,
   lowering time and model cycles against the planner recorded in Git history, and is the evidence
   for the overlap this design trades for flag economy.

### 5.10 Amendment: a ledger is a pair at every depth (2026-09-18)

As first implemented, `session_sync` declared the forward half only at a depth that publishes at
its own level and the reverse half wherever the session has credits, so a depth whose publications
are all made inside a subtree carried the reverse half alone. Its counts balanced, and §5.4's walk
emitted nothing wrong; what it emitted was half a ledger, which §5.1 never authorised and §5.5's
argument cannot be built from. M10-096
records the consequence and the census: **52** such depths over the A2/A3 corpus's 82 lowerings,
and **70 of 70** of their hand-backs reachable on a path that waits for nothing, against **518 of
542** bounded where the depth does carry both halves.

The correction is the one §5.1 already implied: an internal access emits the forward pair after
itself (§5.4). Two measurements decided it against the cheaper alternative of dropping the reverse
half instead, and both are held by probes named in the defect record:

* Dropping the reverse half is not available. At such a depth the forward *ordering* is the inner
  pair's, so the credit's remaining duty is the cross-window WAR, and §5.3's ring already decides
  where that bites: **32** of the 52 depths have a member whose slot index is fixed — the same
  storage is written every window — so the credit is the only ordering between the producer's next
  write and the consumer's current read. It survives only as a per-depth exemption, each carrying
  the proof this section makes for the machinery it retired.
* Emitting the forward half fits the budget of §5.7. Over the **22** lowerings that carry such a
  depth, adding `W` ids to each forward channel leaves none over the eight; the tightest becomes
  `a2_backward_stage` MTE2->MTE1 at 4 + 2 = **6**. The channels that need it are not the corpus's
  busiest.

Collapsing such a depth into its inner one is not a candidate at all: §5.1 already makes a block
with no role-bearing item of its own transparent and gives it no ledger, so a depth that publishes
inside necessarily carries producer work of its own, and §5.9's coverage rule is what that work
needs ordered.

After the correction the same census reports **0** of 336 depths carrying the reverse half alone,
the count of depths itself unchanged, so the repair adds a half and neither opens nor closes a
window. That 0 also answers the shape §5.1 names as the exception: a depth consisting solely of a
leading consumer phase would still carry the reverse half alone, and the corpus holds none. It is
constructible in principle — a session whose producer works only at an outer depth — so it stays a
named residual rather than a proved impossibility.

The price was expected to be overlap, in the currency §5.2 measures: windows that ran ahead of
their publication no longer do. Measured against the same control tree, over 76 case runs of the
seven units on both devices, it is not: model cycles move by **+0.00 % to +0.05 %** per case and by
**+0.013 %** summed. The shape of the change says why. Each case gains a near-constant 4 to 13
cycles rather than a proportion of its run, so what is paid is the issue of a few more sync
instructions and no window overlap is lost — the inner publication was already there, and the outer
`ready.wait` the repair adds is satisfied by the time the consumer reaches it on the common path.
Model cycles remain a model estimate (§5.9.4). The two units whose control run is itself the
failure, `a2_mha_bf16` and `a2_block32_causal`, contribute only the cases that ran before it.
