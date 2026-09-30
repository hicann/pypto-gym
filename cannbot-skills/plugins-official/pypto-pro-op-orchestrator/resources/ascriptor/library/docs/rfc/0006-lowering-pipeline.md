# RFC-0006: The lowering pipeline and the pipe-level simulator

Status: implemented (M4). Code: `ascriptor/passes/`, `ascriptor/backends/sim/pipesim.py`,
`ascriptor/backends/sim/timing/`. Depends on: RFC-0001 (IR, §10 Lowered invariants, §12 pass
order), RFC-0005 (autosync design), D-018 (value-sized allocations), D-023 (vector-only values),
D-027 (`range` loops).

## 1. Purpose

Turn a verified Surface module into a Lowered module every backend can print (RFC-0001 §2): the
cube shortcuts expanded, every `<<=` a device instruction, every buffer hazard guarded by an event
with a hardware flag id, every allocation at an address, the kernel split into one function per
core side. The same Lowered module feeds the pipe-level simulator, which is the milestone's oracle:
a module is correctly synchronised when the simulator finds no hazard and no deadlock on the
recorded inputs and the outputs match the goldens bit for bit.

## 2. The pass manager

`passes/manager.py`. A `Pass` is a `Module -> Module` function with the IR level it accepts and
produces and the Lowered invariants it establishes; `PassManager(PIPELINE).run(module,
stop_after=)` verifies the input, runs the passes in order, verifies after each one, and keeps every
intermediate module (`after(name)`) and every `Explain` note (`explanations(op_id)`). Ops a pass
creates or rewrites go through `Rewriter` and carry an `origin` entry; decisions worth reading go
through `ctx.explain.note(...)`. `ascriptor dump-ir --after PASS [--explain]` prints a stage;
`ascriptor explain --op N` prints an op's origin chain and every note that mentions it.

Passes must not depend on Python object identity across runs: every module is immutable and every
pass returns a new one; `module.attrs["next_id"]` hands the id counter from pass to pass.

## 3. Order and contracts

### Integer scalar simplification

`scalar_simplify` runs after local mutex coalescing. It folds representable
integer/bool constants, simplifies same-type integer affine expressions,
shares pure expressions within a basic block and into dominated `if` arms, and removes unused pure scalar
results. It never substitutes a mutable Cell for a previously captured scalar,
shares a live Cell expression across a write or loop boundary, reassociates floating arithmetic,
or folds a narrowing cast as an identity. Immutable integer expressions may be
shared across unrelated vector, memory and synchronization operations without
moving those operations. Mutable writes invalidate expressions reading that Cell;
a call is a write to every Cell it is passed, and invalidates nothing else, because a call
cannot rewrite a value. Affine
reconstruction requires representable intermediate values, with loop/core ranges
and integer widths retained; unknown overflow behavior prevents reconstruction. That proof
guards the arithmetic a reconstruction adds, so it is not required of a form that adds none:
an expression whose affine form is one other value with no offset is that value, whatever its
range, since the cancelled term is the only place the original could have overflowed. A
reconstruction must not raise the operation count - it may replace or reuse operations the
function already holds, and never materializes a new named base to state a scaled form. A
type-dimension reference follows an alias to its target, which is in scope wherever the aliased
value was; a value no reference reaches, in text or in a type, is then removed. Rewriter
provenance and explain notes retain the reason for each change.

These integer rules bind every constant folder, not only this pass: the backends'
own folders and a printer's parameter binding read the same table
(`ir/scalar_math.py::evaluate`) and answer with the reference interpreter's value
or with nothing. A result outside the declared type, a shift count outside
`[0, width)`, a zero divisor, the signed minimum negated or divided by -1, and a
non-integer operand are all "nothing": the device wraps or faults, so the exact
integer is not the program's value and must not reach generated source or a tile
dimension. A caller that needs the wrapped value asks for it explicitly; static
memory bounds and the model do, printers do not.

An outer captured expression may be reused in a child branch before any write,
call or unsupported effect invalidates it. After the branch, only outer captures
that remain valid in every arm survive; child definitions never escape their arm.
Loop interval widening sends only a changing endpoint to the integer type limit.
Stable bounds survive only if the widened body still proves every update
representable. Thus a guarded increasing counter can retain its nonnegative
lower bound, while a potentially overflowing increment cannot. Program-point
nonnegative Cell bounds permit native positive-divisor div/mod without signed
floor correction; immutable results retain their captured bounds after Cell writes.

Finite counted loops may bound a Cell using a checked per-iteration delta summary.
The trip count, induction exit and every intermediate counter update must fit
their declared integer types; stale update snapshots, unknown writes and unsupported
exits do not qualify. Loop comparisons provide widening thresholds (including the
neighbor of a reset boundary), which remain proposals checked by the body transfer.
Captured integer ranges feed arithmetic, min/max, comparison and constant-branch
simplification without giving the live Cell a function-wide value.

Expression sharing invalidates entries according to the Cells they directly read;
immutable operands retain their old captured value after an underlying Cell write.
Calls, loops and unsupported effects remain barriers. Repeated positive-constant
div/mod and mask expressions in child branches may be materialized before their
common conditional when their operands dominate that point and no intervening
write changes a read Cell. These operations are nontrapping for every representable
input. No memory or synchronization operation is moved. Proven constant conditionals
retain only the selected arm, preserving its operation order and rewrite provenance.

Positive power-of-two floor remainders become low-bit masks, including negative
dividends under the target's two's-complement integer representation. CCE/PTO
may print other floor remainders through a shared typed `FloorMod` helper;
its remainder/sign correction avoids overflow and preserves the division
domain in RFC-0001. PyPTO specializes an export-local IR copy with integer argument
bindings and the effective launch geometry before division expansion. Exact
multiples and proven nonnegative divisions need no signed floor correction.
Unproven divisions retain explicit integer lowering. A final scalar
cleanup follows native arithmetic expansion where enabled.

`scalar_simplify=False` in PassManager options preserves the expanded diagnostic
form. Compact output remains the default. Constant selection, alias propagation
and dead-code removal must not change memory operations or mutex call order.

`autosync` selects the ownership of one device family and no option stands between the two: on
family `a5` it marks mutex ownership, and after `dce` `local_mutex` assigns side-local slot IDs
and inserts explicit operations, which `mutex_coalesce` merges before liveness analysis;
intermediate ownership markers are provenance, not a claim that runtime synchronization is already
inserted. On family `a2` it dispatches to `session_sync`, which plans the paired ready/valid slot
sessions of RFC-0005 §5. `events` colours whatever `sync.event` declarations either route leaves,
the author's own included.

```
verify → cellfold → desugar → device_lower → gmbuff → crosssync → autosync → events → addr_alloc → split_sides
       → events_restamp → dce → local_mutex → mutex_coalesce → scalar_simplify → mmad_settle → liveness → verify
```

This differs from RFC-0001 §12 (which put `split_sides` before `autosync`): three passes moved
in front of the split, for reasons the split itself imposes (D-029):

* `autosync` sees both sides in one graph, so cross-side hazards (`FIX -> V` through UB on a5,
  `MTE3 -> MTE1` through L1) are found and checked against the mutex protocol;
* `events` colours channels that are naturally side-local (`(set_pipe, wait_pipe)` pairs never
  straddle a side), so the order does not matter for ids, but the event declarations must exist
  before the split assigns them a side;
* `addr_alloc` runs once on the kernel so the two sides agree on every address — on a5 the cube
  core writes vector UB (`dma.l0c_to_ub`) and the vector core writes L1 (`dma.ub_to_l1`).

| pass | reads | writes | establishes (RFC-0001 §10) |
|---|---|---|---|
| `desugar` | `cube.matmul`, `cube.matmul_mx`, `cube.conv2d` | slot views of `_l0a` / `_l0b` / `_btbuf`, `dma.l1_to_l0(.mx/.img2col)`, `dma.l1_to_bt`, `cube.mmad(.mx)`, counter bumps, tile loops | no shortcut ops |
| `device_lower` | `dma.copy`, `sync.mutex_*` | one `dma.*` instruction per copy (types + riders), `sync.crosscore.*` per mutex method | 5 |
| `autosync` | `region.autosync` | `sync.event` / `set` / `wait`; regions gone | 3 |
| `events` | `sync.event` uses | `ids` per declaration, `EventType.id` | 4 |
| `addr_alloc` | `mem.alloc`, `mem.workspace` | `addr` / `offset` (int or scalar value + arithmetic) | 6 |
| `split_sides` | the kernel | `@k.cube`, `@k.vec` (`kind = func`, `side`), module `meta` | 1, 2 |
| `events_restamp` | actual per-side event uses | final per-side physical flag IDs | 4 |
| `dce` | everything | effect-free unused ops removed per side | — |
| `mmad_settle` | `cube.mmad` into one L0C | `sync.barrier(pipe = M)` before an accumulate whose producer has not settled (A2 family only) | — |
| `liveness` | memory accesses | `live = [first, last]` on allocations | — |

`mmad_settle` is last before `liveness` because the hazard it repairs is a property of the final
lowered order, and because a barrier inserted there cannot disturb an event, address or mutex
decision already made. It asks the same analysis the hardware lint reports with, so a quiet lint
after the pipeline is a measurement of the repair rather than two implementations agreeing; the
rule it enforces, and why the split-K expansion's own settle stays where it is, are RFC-0008 §5.

The `local_ready` pass that used to sit between `events_restamp` and `dce` removed the armed
bookkeeping of the edge planner, and retired with it (RFC-0005 §5): a slot session emits no guard
cell to prove redundant, and no drain to prove unreachable. Every synchronization operation the
passes after the split keep therefore keeps its original OpID, operands, pipe and assigned event
ID, and no scalar simplification can change an ID allocation or a loop-carried ownership
protocol.

## 4. desugar

The shortcuts become the instruction stream the old `easyasc.shortcuts` emitted (`tmp/research/
shortcuts_old.md` was the reference): `matmul` in its `nosplit` / `splitn` / `splitk` variants,
`matmul_mx` with the scale-tile offsets, `conv2d` with `img2col` loads and a static or dynamic K
loop. Two deliberate differences (D-028):

* the kernel-owned L0 scratch buffers are **byte carriers** (`buf<l0a<u8, [128, 256]>, 2>`, one
  32 KB slot each) and each use re-views a slot as the tile the load produces
  (`mem.reinterpret {tile = [rows, cols]}`), so an fp32 `[256, 32]` or an fp4 `[128, 512]` tile is
  a plain typed window, and the reference interpreter needs no layout tricks;
* the tile loops are `cf.for` device loops (D-027); a static conv K loop is unrolled as the old
  shortcut did, because the first tile carries the bias and `is_init`.

The scratch buffers and their counters are created once per function on first use, at the top of
the body. Dynamic `is_init` never reaches the pass: the frontend already splits it into a
`cf.if` with two shortcut ops.

For ordinary (unpacked) matmul operands, explicit `m`, `n` and `k` select the
logical source windows as well as the MMAD dimensions. Desugaring must stage
`[m, k]` in L0A and `[n, k]` in L0B, with source axes exchanged for transposed
loads. The original L1 allocation dimensions still determine the source pitch.
Split-N and split-K apply their subwindows to the same logical extents; any
fractal padding of a K copy must preserve the MMAD row-block pitch. Loading the
full allocation and merely changing MMAD K is invalid on C220: a copy at K128
and consumption at K73 use different L0A row-block pitches (M10-021).
Packed int4 carrier loads and MX scale/packed-axis units retain their separate
contracts; this ordinary-operand correction does not reinterpret carrier counts
as logical elements.

A byte scratch carrier's two array dimensions are not typed matrix dimensions.
When a backend needs static typed L0 capacity for a runtime staging extent, it
must prove a sufficient bound in the destination element units. It must not
substitute one raw-byte array axis for an unknown typed axis or silently clamp
the MMAD extent to that guessed capacity. A backend without such a proof must
reject that form at its source location; static branches or a proven bounded
extent remain possible representations. This requirement follows from the
shared byte-slot representation above and does not change its physical size.
PyPTO explicitly [rejects runtime local capacity](0013-pypto-native-synchronization.md#static-local-capacity);
a proven constant capacity with runtime valid extents is a separate supported form.

## 5. device_lower

Per `dma.copy`, the instruction is chosen from the memory spaces, layouts, ranks and riders exactly
as the old `Tensor.__ilshift__` tables did (`tmp/research/tensor_sync_flow.md` §1.3–1.4), and its
parameters are computed from the view geometry (`passes/util.py::view_of`: root, offsets, extents,
full shape) with scalar arithmetic emitted in front of the op when a dimension is a run-time value.
`sync.mutex_lock / ready / wait / free` become the cross-core primitives of the old `CvMutex` /
`VcMutex` methods with the declaration's pipes (a5 vector release pipe `V`).

C310 ordinary byte transposes need paired16-row source fractals (M10-030).
`device_lower` records `m_copy = ceil(m_dst / 32) * 32` when the logical row
count is not already known to satisfy this constraint. The optional Lowered
attribute preserves logical `m_dst`/`n_dst` and MMAD extents while making the
physical access explicit. Source column accesses retain32-byte granularity.
The declared L1 allocation and L0 slot must contain the expanded load; known
insufficient allocations are source-located errors and dynamic bounds are
preconditions checked by the Lowered simulator. Dependencies and pipe traces include the extra
source rows and destination bytes. Logical padding values remain unspecified.
The printer selects this recorded extent; it must not invent hidden overreads.
Packed/MX loads and other devices retain their separate geometry contracts.

## 6. autosync and events

RFC-0005 as implemented; the amendments are recorded there (§5). In short: `autosync` marks the
operations a `region.autosync` owns and hands them to the planner the device family selects, of
which there are two and no option between them. On A5 they become physical-slot mutexes
(`local_mutex`, RFC-0005 "A5 local buffer mutex policy"). On A2 and A3 they become paired
`ready` / `valid` slot sessions (`session_sync`, RFC-0005 §5): one ledger pair per
`(side, channel, capacity class)` and block depth, one window per producer/consumer role switch
with consecutive same-role work merged, and each ledger's credits the physical slots its window
rotates through. One dependency graph per function is still built from registry access sets and
view geometry, but it judges the emitted plan instead of placing events; `crosssync` keeps the
cross-side checks against the mutex protocol. The edge planner this replaced — vector clocks,
run-ahead bound, acknowledgement synthesis, armed cells, conditioned mirrors — is gone with the IR
attributes that carried it (§5.5). `events` assigns ids per channel by interval colouring under the
8-id budget (loops widen a live range, pre-set events live for the whole function, literal
`set_flag` / `wait_flag` ids are reserved).

## 7. addr_alloc

One bump cursor per space, never freed: the model every corpus kernel was written for (the old
AscendC `TPipe::InitBuffer`; slot buffers explicit). Alignment 32 B (UB, L1, BT), 512 B (L0A, L0B),
1 KB (L0C). Static sizes give static addresses checked against the device capacities; a value-sized
allocation (D-018) gets its size and address as scalar ops placed before it, and every later
allocation of that space is dynamic too — the simulator checks those against the capacity at run
time (the interpreter's `mem.alloc` reads `addr`). Workspaces are bumped in bytes without
alignment, as `split_workspace` did.

## 8. split_sides and dce

Every op goes to the side the registry names; events and barriers to the side of their pipe (an
event's side is recorded by autosync, or inferred from its pipes for user events); scalar
arithmetic, cells, views, allocations and control flow to both sides; `region.side` bodies to
their side only. Vector-only values (D-023) taint everything computed from them and every loop or
branch they decide; a cube op that consumes one is an error naming the op. The vector copy of a
duplicated op gets a fresh id with an origin entry (`moved`) pointing at the original — op ids stay
unique per module, and the interpreter keys storage by allocation *name*, which both sides share.
`dce` then removes what a side never observes: effect-free ops with unused results, cells only
written, allocations nobody touches (their addresses were assigned before the split), loops and
branches that became empty.

## 9. The pipe-level simulator

`backends/sim/pipesim.py` runs a Lowered module in two phases.

**Trace.** The reference interpreter executes the module as before (one forked process per core
group with a thread per lane, GM in shared memory; every lane a thread of one process with
`processes=False`) with a `Tracer` attached — a child sends its lanes' tasks back through a queue
(ops by id), and since a lane's trace is program order it is the same either way; per-process
tracing took `pfa_fd` from 44 s to 4 s: every executed kernel-level op becomes a `Task` — its pipe, the
cycle cost from the timing model, the memory windows it reads and writes (registry access sets, the
resolved `MemRef` rectangles keyed by core / space / allocation / slot / sub-block), and for a
`cf.call` the executed vf ops that give the VF cost. Functional results are the program-order ones
and the trace is deterministic; the interpreter stays the single source of semantics.

**Lane turns.** The lanes of one process take turns: a lane runs until it waits for another lane (a
cross-core flag or mutex, a collective, the SIMT launch gate) and takes the turn back when its wait
ends. Torch lets go of the interpreter lock for every tensor op, so lanes that ran at once handed it
to one another at every op, and switching threads took most of a run: MHA preload with 8 heads on
8 cores took 442 s in threads (25 million context switches, more system time than user time) and
15.3 s forked; taking turns it takes 14.4 s and 3.0 s, and on one core 23.4 s becomes 3.7 s. The
turn decides when a lane runs, never what it computes: outputs, cycles, hazards and the trace's
tasks are unchanged. The stall detector judges a wait before the lane asks for the turn back, so a
lane waiting for its turn counts as running; past the limit a lane leaves its wait without the turn
and reports at once; once a lane fails, the others leave at their next op instead of each running
on to its next wait.

**Schedule.** Per lane, the scalar pipe walks the tasks in issue order and hands each to its pipe's
FIFO at the scalar clock (one marker cycle); a pipe runs its FIFO in order, a task starting at
`max(pipe clock, issue clock, dependencies)`. Dependencies: same-side event tokens (FIFO per event,
`depth` outstanding at most, pre-set tokens for carried events), the cross-side flags of the mutex
protocol (visible `intra_core_sync_latency` = 200 cycles after the signal), the all-core collectives,
`pipe_barrier(ALL)` (drains the lane's pipes). Lanes are visited in launch order - core by number,
cube before vec0 and vec1 - not in the order the trace met them (lane timing, or which forked group
reported first), so the report with its hazards and the Chrome trace are the same bytes in threads
and in forked core groups. No progress with work left is a **deadlock**, reported
with every blocked op and what it waits for. The functional run underneath detects its own
deadlocks the same way across every lane and process: when every lane alive sits in a wait and
none has left one for half a second, the run stops at once with the blocked op instead of at the
timeout (`v8_allhif8`'s mutex deadlock is reported in a second, not after five minutes).

**Flag occupancy in scheduled time (M10-069).** Matching an event's FIFO tokens
while constructing a schedule is not proof that its physical flag can hold them.
A wait can be assigned a future start time behind a long transfer; removing its
token from the construction-time queue must not make that flag available to a
set that completes before the wait actually starts. Record each token's set
completion and matched wait start, then check their half-open occupancy intervals.
This deliberately uses the earliest consumption time; equal-time release and
reuse do not form an overlap. Preset tokens start at cycle zero, and the existing
`set_all` / `release` operation counts are expanded into their individual tokens.

When Lowered declarations supply allocated IDs, occupancy is checked for each
`(participant, source pipe, destination pipe, flag ID)`, including different
event names that reuse an ID and raw flags with their executed IDs. Event slot
selection follows the declared ID list: presets advance the set cursor, and
sets and waits advance their respective cursors. Cube, Vector0 and Vector1 are
different participants. A low-level diagnostic trace without a complete ID map
retains logical-event capacity checks and reports the missing physical binding;
it must not claim complete physical-flag coverage.

Overlapping occupancy is a synchronization hazard through the existing report
API. The checker must not delay a set, add implicit backpressure, change FIFO
pairing, or modify functional outputs or vector-clock dependencies to make an
invalid schedule pass. This checks the executed, timed schedule, not every
possible hardware timing, and not the planner's own capacity rule (RFC-0005
§5.3), which is a separate judge: it reports a hazard only where the simulated
cycle ranges of two accesses overlap, so a capacity one window too wide can pass
here and still race on silicon.
It does not identify a native kernel failure's cause without a matched control.

**Hazards.** (What the run itself checks: an integer index register is read with its declared signedness,
so a `u8` register holding `arange << 2` gathers bytes 0..252, not a signed view of them.) Every task carries a
vector clock merged from its pipe's previous task, the scalar
pipe at issue time and the tokens it consumed. Two tasks touching overlapping bytes of one buffer,
at least one writing, are a hazard unless the clocks order them — a happens-before race detector
over the executed trace, independent of the cycle numbers (the old checker compared modelled cycle
windows and only five ownership pairs). Cross-side ordering through mutex tokens counts. GM
conflicts are checked only on request (`check_gm`): cores partition GM by convention and atomics are
races on purpose.

GM and workspace writes use their exact byte footprints. Sharing a 32-byte block does not
make two disjoint writes a WAW conflict: rounding the destination out to a whole block invents
bytes the store does not modify. This applies to DMA, scalar and SIMT stores, including both
halves of an atomic RMW on its destination. Ordinary reads and on-chip accesses retain their
32-byte conflict granularity. Unordered accesses that actually overlap still report hazards;
the atomic/atomic exemption and synchronization ordering rules are unchanged.
For padded DMA stores, the destination origin, burst count, payload bytes and destination
stride determine the footprint; a carrier view's extent neither truncates nor widens a burst.
Keep disjoint byte runs in a complete footprint record so a large scatter cannot evict its
own early intervals from the bounded hazard history.

M10 SIMT trace correction: a launch records the actual scalar loads, stores and atomic RMW
addresses executed by all its participating threads, coalesced at the access granularity above
without filling holes between written bytes. Passing a pointer does not make its whole allocation read/write. Atomic accesses
are marked separately: two atomic RMW accesses may overlap, while an unordered atomic access
and an ordinary read/write still form a hazard. The launch remains one scheduled V task, so
this checks dependencies between launches/lanes; it is not a within-launch thread-race checker.

M10 footprint correction: a contiguous slice of only the final GM dimension is one row burst,
including when earlier dimensions were indexed away. GM/UB transfer selection must not interpret
it as one padded burst per element. Trace footprints retain the resolved view's storage offset,
strides, rows and columns after reshape/reinterpret or explicit GM view creation. A pitched
rectangle cannot be replaced by a contiguous numel interval: that both invents overlaps and
misses later rows. Preserve the access granularity above and verify genuine-overlap negative
controls when refining the footprint model.

**Cache lines (I012).** On A5, GM *scalar* stores by different cores into one 64-byte cache line survive
only when each writer cleans the line after its store and its cross-core publication cannot run before that
store. Four AIVs writing four elements of one line kept every store with both (jobs 363, 366, 20 launches
each), and lost stores as soon as one was missing: three of four without the dcci, two of four when the MTE3
publication did not wait for the storing scalar pipe, and two or three of four when nothing ordered the
writers, whatever dcci did — before or after the store, one line or the entire cache (jobs 310–312, 338).
Only the producer's dcci counts: dropping the consumer's kept every store. A publication issued on the
storing pipe needs no wait, and the dcci itself needs no wait for the storing pipe when they share one, which
is all the measured stores on a pipe other than S: every DMA (MTE3) variant kept all four writes, including
the fully concurrent one (job 363). Exact footprints cannot show this. So with `check_gm` the scheduler also
records which cores' `scalar.store` tasks and SIMT launches wrote each line, counting 64 bytes from the start
of each GM storage (the measured data pointers were 64-byte aligned), and walks the writer's own lane from
the store: the first `core.clean_dcache` that covers the line, then the first cross-core publication and
whether the storing pipe reaches its pipe (the same pipe, a `bar_all`, or raw flags set after the store that
chain one pipe to the other). A write by a second core adds one warning per pair of source operations to
`report["warnings"]` unless the two stores are ordered by the cross-core chain and the earlier writer has
both the clean and the publication order. The warning names both operations and what is missing. Warnings
are not hazards: two atomic writes do not raise one, and outputs do not change. DMA stores measured clean
and are not checked. `core.clean_dcache` stores nothing: it records the bytes it names so the check can see
which line was cleaned, and those accesses take part in no hazard.

**Timing.** `timing/cycle_model.py` applies the old simulator's per-instruction formulas
(`a5_cycle_model.json`, unchanged) to the new opcodes: DMA `overhead + bytes / bandwidth`, `mmad`
`67 + M16 · N16 · Kc0 / macs_per_cycle` by operand width, L0 loads `ceil(bytes / 128)` + head
overhead, VF `46 + max(busiest issue pipe, longest latency, 0.46 × register critical path)` + head
overhead from the executed vf ops (issue intervals and bank-conflict penalties per op, scalar ops
on constants free), sync markers one cycle. Padded GM→L1 loads, `img2col` and `l1_to_bt` are
costed by bytes instead of the old 11-cycle fallback (D-030).

The report has the makespan, per-pipe busy cycles and utilisation, the hazards, warnings and deadlock;
`SimResult.write_trace` writes a Chrome trace (`ascriptor sim --trace`).

## 10. Gate (M4)

`tests/test_corpus_lowering.py`: every tracked kernel lowers through the whole pipeline and its
events balance on every control path (`autosync.check_balance`, the old token-replay oracle on
the IR — one test per kernel), and its first recorded case replays bit for bit on the pipe-level
simulator with no hazard and no deadlock (a second test per kernel, so a heavy kernel's balance
replay and simulation run on different workers). Known gaps are listed under `lowering_xfail` in `tests/kernels/a5/corpus.json` with a
reason, and an entry that starts passing fails the suite. The suite runs under pytest-xdist
(`-n 6` in `pyproject.toml`; a heavy replay takes ~2 GB, which bounds the workers on a 16 GB box):
every kernel's replay is one single-threaded Python process, so the gate takes kernels ÷ workers. With per-event balance replay, per-process tracing and the
cached HiF8 codec the heavy attention kernels take seconds each (`pfa_fd_hif8` had taken 5½
minutes, `v8_allhif8`'s balance replay longer than that). Unit tests: `tests/passes/`,
`tests/sim/test_pipesim.py` (including the negative control: with the autosync regions stripped the
simulator reports the missing hand-offs).

## 11. Open questions

1. ~~Event sharing across blocks~~ — done (D-037), under the edge planner: same-iteration pairs
   shared an event across the blocks of one loop body and pairs with one wait point folded into
   one. A slot session shares by `(side, channel, capacity class)` instead (RFC-0005 §5.2), and an
   event set in a loop body and waited after the loop is what a window of the enclosing depth is.
2. `addr_alloc` never reuses addresses; `liveness` records what a reusing allocator would gain.
3. SIMT cost counts the ops thread 0 executes; the old simulator counted statements while any
   thread was active.


### MX scale blocks during split-K

An MX scale tile stores each16-row by two-scale-column block in32 physical
bytes. A split-K iteration starting at K offset `i` selects scale byte offset
`(i // 64) * 32` within each row tile, preserving the original source K pitch.
This rule applies to either operand regardless of its data transpose. Lowering
must use the original scale Value with an explicit packed offset; an ordinary
UInt8 tensor column slice is not that physical MX scale address. M10-044 fixes
the nontransposed split-K branches and retains per-op provenance/explanation.

### Reuse normalized slot indices

A read-only integer range analysis recognizes immutable constants, positive
floor remainders, nonnegative bit masks and lossless integer casts. Cells do not
inherit function-wide bounds from their initializer. A separate program-point
analysis propagates integer Cell writes, joins branches, and checks a loop-header
fixed point. Its branch refinement requires a fresh comparison; integer overflow,
unknown writes and unhandled loop exits discard affected facts. Widening checks
comparison-guided candidate bounds, and counted-loop summaries use proven trip
counts and representable updates. Thus an increment/reset
counter can be in `[0, slots-1]` at a buffer access and reach `slots` before reset.
Consumers may omit a second wrap only
when this analysis proves `0 <= index < slots`; names and provenance notes are
not proofs. Scalar cleanup removes redundant same-type remainder/mask operations
using that fact while preserving the original immutable snapshot.

CCE may index the buffer's physical slot array directly. PTO-ISA may omit
`SlotOf`, and PyPTO may select a Tile group with the existing named scalar.
Native division expansion preserves value identity, so a backend may consult
the pre-expansion scalar graph for those same immutable definitions. Derived
indices without a proof keep normalization and PyPTO's expression-materialization
workaround. Alias views reuse the captured index; a later Cell write cannot
change which slot a previous view denotes.

A proven redundant Cell remainder/mask becomes a same-width immutable snapshot
at the original definition, never a live Cell alias. Repeated pure integer reads
within a block and its dominated `if` arms may share that snapshot until a write,
call or loop boundary. A branch join retains only outer snapshots valid in every arm.
Nontrapping constant-divisor operations may also share a result when a range is
unknown. Implicit writes such as VF mask-counter updates invalidate the cache.
Backends rederive snapshot bounds from program-point analysis; provenance text
does not establish a bound. PyPTO preserves its upstream loop-carry workaround
when materializing these snapshots.

Within a straight-line initialization window, a same-width integer Cell may be
known equal to an immutable initializer. Pure scalar reads can use that value;
the Cell declaration and later writes remain independent. Any write, call or
control boundary ends forwarding, and no initializer fact enters a loop body or
branch. Cell-to-Cell initializer copies retain their native snapshot protection.
For same-typed immutable `a` and a positive constant `d`, `a - (a // d) * d`
may become remainder only when the quotient rounding is preserved and its product
is representable. Unknown overflowing products or implicit casts prevent this rule.

### Scalar cleanup after backend specialization

Memory descriptors, vector operations, memory accesses and synchronization do not
invalidate a common expression of immutable integer operands in the same Block.
Control boundaries, calls and mutable writes still do. Cells are never commoned
as though they were immutable snapshots. Only scalar expressions are shared;
memory/vector/synchronization operations retain their original order.

PyPTO kernel-side and VF scalar emission retain typed statement records until physical
Tile declarations and specialization are resolved. A backend cleanup computes
liveness from executable Python syntax after native local mutex operations
have been omitted.
It removes unused pure integer definitions and unobserved integer Cell-update
cycles. Empty, effect-free finite loops/branches can then disappear. Loads,
stores, calls, synchronization and unsupported statement forms remain observable.
This is emitter cleanup, not deletion of shared IR synchronization: CCE/PTO still
need the arithmetic used by their actual mutex instructions.

Single-use integer definitions may inline into an adjacent same-type Cell
initialization when their operands are immutable, an in-place add/sub update,
or an adjacent direct boolean `if` test. Explicit casts and snapshot guards are
kept. Shared initializer expressions remain materialized once and may receive an
`initial_<cell>` name. Cross-Cell copies remain materialized for native loop-carry correctness. No memory/control boundary
or mutable write is crossed. Named buffer indices and alias snapshots stay
materialized at their original point. Cleanup records original op IDs and reasons
in artifact metadata. `scalar_simplify=False` preserves uncleaned scalar output;
it does not restore omitted native mutex operations or comments. Empty emitted
sections remain syntactically valid. Floating arithmetic is not reassociated or inlined.

Generated Python locals use one namespace per function, with global helper names
reserved. Cube and Vec sections of one kernel share a namespace. Collisions use
stable numeric suffixes; unrelated VFs do not rename each other's locals. Unnamed
RegList declarations inherit their DSL assignment name for element-name hints;
an explicit name takes precedence. Mandatory integer casts and the deliberate
Cell snapshot spelling that prevents upstream loop-carry aliasing remain intact.
Kernel-side negative Cell sentinels retain their runtime spelling even when
constant-branch elimination removes all writes: PyPTO must not eagerly reject a
constant negative address in a runtime-guarded, unexecuted memory operation.
