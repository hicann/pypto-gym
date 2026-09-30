# Pipes, events and buffer ownership

`Position` identifies storage; `Pipe` identifies an execution stream. Pipe
members are immutable enum-like values: `repr(Pipe.MTE2)` is `Pipe.MTE2`,
and equality with the string `"MTE2"` is false. The old `PipeType` constructor
and identity-based public contract are gone.

| Stream | Typical operation |
| --- | --- |
| MTE2 | GM input transfer |
| MTE1 | L1-to-L0 operand staging |
| M | cube computation |
| FIX | L0C result publication |
| V | vector computation |
| MTE3 | UB publication to GM/L1 |
| S | scalar scheduling and supported wait locations |
| ALL | local all-pipe barrier domain |

Actual opcode/device classification comes from the current IR registry and
lowering. A pipe label alone does not establish a dependency. A single-pipe
barrier and a local `barrier(Pipe.ALL)` also do not rendezvous separate vector
participants or hand ownership between cube and vector sides.

## Local buffers and events

`DBuff`, `TBuff` and `QBuff` have two, three and four physical slots. Indexing
uses the corresponding slot modulo. `SEvent`, `DEvent`, `TEvent` and `QEvent`
provide one through four local event credits. Storage capacity and event depth
are separate properties; extra credits cannot make premature reuse safe.

The reverse case is explicit too. `DBuff`, `TBuff` and `QBuff` accept
`sync_depth=N` when an allocation needs more physical history slots than the
producer/consumer pipeline should keep in flight. The physical ring and its
modulo index remain unchanged. On A2/A3 the cap is the credit count of the slot
session that protects the allocation: both directions of its ledger declare
depth N, so the producer may keep N windows in flight and no more, and where one
ledger serves several allocations the tightest declared cap applies.
`1 <= sync_depth <= slots` is checked by both the frontend and IR verifier.
The setting never rewrites hand-authored events or cross-side mutex credit
counts; a too-tight schedule must still pass pipe simulation and hardware.
On A5, slot mutexes replace this event-budget fallback: `sync_depth` does not
reduce the mutex window. Each physical slot has its own ID.

The [explicit event-ring unit](../../examples/api/event_depths)
fills a batch of UB slots before storing it to GM. Readiness moves from MTE2
to MTE3. A reverse availability event starts with one token per slot and
returns that token after the consumer finishes. Three batches wrap the ring
twice and end with one partial batch. The availability event finishes with
its initial count restored. Removing readiness produces an actual UB hazard
for each tested depth, which the negative regression rejects.

`with auto_sync():` inserts supported same-side synchronization based on
memory dependencies. It does not assign cross-side ownership. The separate
[GM ring](../../examples/api/buffer_ring) combines local autosync with an
explicit MTE3-to-MTE2 publication event. Its default Python `check()` performs
functional simulation; use `check(launcher="pipesim")` or the public
`check --launcher pipesim` command to check lowered events and GM hazards.

On A5 the default strategy wraps actual on-chip memory accesses in mode-zero
`get_buf`/`rls_buf` pairs. Tensor/xBuff slots receive implicit IDs in 0..31,
shared by all views of the same physical slot. Each AIC and each AIV instance
has an independent 32-ID namespace shared across banks. More than 32 required
identities is a compile error. Dynamic buffer addresses and IDs use the same
captured slot value. A VF's buffer accesses are protected at its call boundary.
An IR pass merges equal release/get pairs in one Block across proven unrelated
scalar operations; it does not cross memory effects, synchronization or
control-flow boundaries. Same-pipe memory completion still needs the applicable
barrier. Explicit events and cross-core handoffs remain intact.

A2/A3 have no mutex instruction. There, `auto_sync` plans paired `ready`/`valid`
slot sessions instead: one window per producer/consumer role switch, its depth
the physical slots the window rotates through, capped by `sync_depth`
([RFC-0005 §5](../rfc/0005-autosync-on-ir.md#5-as-implemented-a2a3-slot-sessions-2026-09-16)).
`session_ledgers` selects how coarsely allocations share one event pair
(`capacity`, the default, `channel` or `allocation`); `local_mutex_coalesce=False`
provides the unmerged A5 control.

Bulk `setall()`/`release()` operate on the event's whole depth; their balance
depends on the surrounding credit protocol. Generated object construction
and teardown can also seed/drain tokens and must be inspected when implementing
a backend. The baseline rings pair individual publications/consumptions and
restore their initial credit state. Explicit numeric flags also require a
supported source/destination pipe pair and caller-owned flag identity.

## Cross-side ownership

`VcMutex` hands vector-produced storage to the cube; `CvMutex` hands
cube-produced storage to vector consumers. `lock` obtains a producer slot,
`ready` publishes it, `wait` acquires it on the consumer side, and `free`
returns it after the last required read. Producer/consumer end pipes must
cover the operations that actually touch the shared storage.

All four calls are required, and one cycle runs them in that order:

1. producer `lock` — take a slot, blocking until a credit is available;
2. producer writes the slot — the copy or drain that fills the handed-over buffer;
3. producer `ready` — publish it; `ready` of cycle *i* orders `wait` of cycle *i*;
4. consumer `wait` — acquire the published slot;
5. **the consuming instructions** — every read of that buffer belongs here;
6. consumer `free` — return the credit, after the LAST such read.

Step 5 is the one a hand-written handshake omits. A `wait` followed straight by
`free` returns the slot before anything read it, and a read placed outside the
pair is ordered by nothing at all, however many mutex calls the loop contains —
events cannot cross sides, so no amount of `auto_sync` repairs it. Keep the
counts balanced on each side too: a `ready` that no `wait` consumes leaves a
token behind, and a missing `free` or `ready` stalls the other side's next call,
which the functional simulator reports as a deadlock naming the mutex, the
awaited call and both token counters. It reports that within half a second of every
live lane blocking. A run that merely reaches its time limit while a lane is still
computing is a `SimTimeout` instead: a slow case or a loaded host, cured by a larger
`timeout` and never by editing the handshake.

The running order is not always the source order. When the producer's buffer has
a later reader than the publishing copy, its `free` belongs after that reader —
in the [A5 roundtrip](../../examples/api/cube_vector_roundtrip)'s
`roundtrip` the L1 tile is freed only once the cube has drained the product that
consumed it, so `vector_to_cube.free()` sits after `cube_to_vector.ready()`.
Guard one hand-off buffer per mutex; two directions need two mutexes.

One of `depth` or `guards` is required, and both are keyword-only. The credit
count is a property of the buffer the mutex guards — one credit per slot that
buffer rotates through, times the number of cycles the mutex runs per rotation —
so no default is right for an unknown buffer, and the old default of 2 was a
wrong answer waiting for a single-slot buffer. Name the buffer instead:
`CvMutex(0, guards=ub_score, ...)` reads the slot count out of the type. Writing both is how a mutex that
cycles more than once per rotation says so; a lint compares them and warns when
the credits exceed the slots, since taking a slot back while the consumer still
reads it is a wrong answer rather than a hang. `guards` also records which
buffer the handoff belongs to, which `autosync` otherwise has to infer.

The [A2/A3 bridge](../../examples/api/cube_vector_bridge) uses GM because
this family does not have A5's direct L0C-to-UB route. Both vector participants
read disjoint 16-row halves before the GM slot returns. The designated A3
hardware result is recorded separately from the copied CPU/source checks.

The [A5 roundtrip](../../examples/api/cube_vector_roundtrip) uses
both directions: vector compact-NZ data enters L1, then the cube publishes
its result directly to each vector UB. Two-slot buffers and depth-two mutexes
survive three distinct beats. Autosync orders each side; the two mutexes
establish the two ownership transfers.

Collective `allcube_*`, `allvec_*` and intra-core helpers need an explicit
participant/cohort contract. A one-core teaching case is not evidence for an
arbitrary multi-core barrier. A collective wait that a member never publishes is
reported by the functional simulator as the same deadlock, naming the scope and flag.
Event balance, hazard/deadlock checks and hardware execution remain distinct
acceptance stages.
