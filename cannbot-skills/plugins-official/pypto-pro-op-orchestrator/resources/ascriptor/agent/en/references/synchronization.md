# Ownership, events and reuse

Same-side dependencies and cross-side ownership are separate. `auto_sync()` inserts
same-side ordering; a cube/vector handoff needs the explicit protocol accepted by
the selected device. A5 direct on-chip handoffs and A2-family workspace bridges have
different capabilities. Look up their exact public signatures rather than transposing
a pattern between facades. The four calls of that protocol, and where a consumer
belongs inside them, are on [cross-side handoff](cross-side-handoff.md); this page is
the lifetime and declaration side of the same mutex.

Describe every reusable slot by producer, first consumer, last consumer, reuse point
and the pipe that actually retires the resource. A forward publish event orders
producer to consumer; a separate loop-carried dependency may be needed before the
producer overwrites storage still read by that consumer.

**A `set()` covers only the producer work issued before it.** An event guarding two buffers has to
be set after both are written, not between them: raised in the middle, the handshake still looks
complete — one set, one wait, balanced, right pipes — while the consumer is ordered against the
first transfer alone. Functional simulation cannot see it, and the values can be bit-exact; the
hazard list is the only signal.

`Tensor`, `DBuff`, `TBuff` and `QBuff` supply one, two, three and four local slots.
Determine the maximum simultaneously live storage roles from the schedule. Multiplying
overlapping beats by simultaneous roles is a conservative bound when all those roles
overlap, not a universal minimum. Prove disjoint lifetimes before reusing one slot.
Event/mutex credit depth does not allocate extra storage. Index rotation and credits
must agree, and different lifetime families need distinct counters.

A mutex now has to say which: one of `depth=` or `guards=` is required and both are
keyword-only, because the credit count belongs to the buffer being handed over and no
default is right for a buffer the declaration never names. Prefer `guards=<that buffer>`,
which reads the count off its slots and cannot drift from it; write `depth=` as well only
when the mutex cycles more than once per rotation, and a lint will say so if the credits
exceed the slots. A mutex with more credits than slots lets the producer retake a slot the
consumer is still reading — a wrong answer, not a hang.

The compiler now checks that for every kernel, not only for one containing an `auto_sync`
region. The check used to live inside the pass that inserts intra-core events, which returns
immediately when a function declares no region — so a kernel synchronised entirely by hand,
the one whose author is personally running the mutex protocol, was checked least. It is the
`crosssync` pass now: more credits than the cycles separating a hand-back is an error, an
uncovered cross-side edge is a warning, and `autosync_cross_side=off` silences both when you
want the pipe model to show you the race instead.

On the A2 family, `auto_sync` plans every same-side hand-off as one protocol rather than one per
dependency: a paired `ready` / `valid` slot session, whose window opens where the producing pipe
starts writing and closes after the consuming pipe's last read, and whose credit count is the
physical slots that window rotates through, capped by `sync_depth`
([RFC-0005 §5](../../../library/docs/rfc/0005-autosync-on-ir.md#5-as-implemented-a2a3-slot-sessions-2026-09-16)).
Consecutive work of one pipe merges into the open window, so the flags a kernel spends follow its
role switches and not its edge count. Nothing here needs a lease, a fragment enumeration or a
hand-raised depth: the protocol the retired planner had to prove for shared L0A/L0B scratch is what
every hand-off now gets, and the earlier lease options are gone with it. What the author still owns
is unchanged — cross-side ownership, GM and workspace protocols, overlapping writes on one pipe,
and the scalar pipe as a producer, which can set no flag.

Two habits still pay off. Declare the slots the schedule actually rotates through, because that
count *is* the credit count; and keep one counter per lifetime family, because a session reads the
rotation from that counter. Where a shape cannot be planned — a consumer that runs before the
producer publishes in the same window, a `break` across an open window, a member that flows both
ways between two pipes — the pass refuses with both operations located, and the repair is an
explicit event pair or a schedule where producer and consumer share one window.

Also check the rotation the planner actually reads: on the A2 family it is what the credits are
counted from, and where it reads nothing the session falls back to one credit. A `cf.for`
induction value is not read as a slot index (on runtime `765e4ec` it also fell back to a
conservative one-iteration distance), and wrapping it in a read-only `Var` is folded away. The
recorded counter control (`docs/migration/fragments/for-iv-slot-distance-20260909/README.md`)
uses an explicit Cell initialized before the loop and incremented once at body
end, so the existing analysis recognizes the two-slot rotation. The arithmetic
and storage remain unchanged. This is a scoped source workaround, not a reason
to raise event depths by hand or assume every multi-buffer already overlaps.

Inspect the surviving IR predicates, Cell writes and canonical allocation roots
when a wait spans unrelated work. M10-068
shows how distinct allocations can be conservatively grouped across branches;
separating them still depends on alias and flag-budget proofs. Descriptor copies
also need their assignment-time values preserved through native lowering.
[M10-070](../../../library/docs/upstream.md#a5-up-036)
records a native loop-backedge snapshot failure despite correct Surface and
Lowered results. Its kernel integer-copy adaptation is scoped; it does not
justify float identity arithmetic or prove every upstream scalar form correct.

For delayed stage `d`, write the consumed work item and the warmup/drain guards
explicitly. A producer running one iteration ahead normally requires a final drain
iteration. The last downstream reader can extend the original source's lifetime.
Do not free a handoff when an intermediate stage finishes if a later stage still
uses its storage. VF local loads/stores may also need `vf_barrier`; kernel-side
events alone do not describe all VF memory ordering.

For a guarded carried event, distinguish a token still pending from the previous
iteration from the current producer's condition. A skipped producer can still owe
a wait; the first iteration may owe none, and the drain must consume exactly the
remaining state. Check zero, one and several iterations plus alternating skipped
producer paths. A declared event depth does not prove the maximum outstanding count.

Use the [evidence table](../common-language.md#evidence) for synchronization conclusions. Inspect op provenance and physical accesses. Keep the
critical section narrow only after identifying the real last reader. An unexplained
warning remains an open issue; no arbitrary retry budget prevents an authorized fix.

Check token lifetime using actual set completion and matched wait start, rather
than the order in which a simulator constructs its queues. A future wait has
not yet freed the physical flag. M10-069
checks overlapping lifetimes by participant, ordered pipe channel and allocated
ID, including reuse by different event names. Balanced counts and no memory
hazard do not replace this check; missing physical bindings remain incomplete
evidence. The separate historical
[shared-publication analysis](../../../library/docs/rfc/0005-autosync-on-ir.md#55-why-no-run-ahead-analysis-acknowledgement-cell-or-mirror-is-needed) was retired
with the edge planner; the timeline checker qualifies its own model scope.

An edited Lowered IR or emitted artifact is a diagnostic control. Qualify the
normal source compiler separately, retain its runtime and artifact identity,
and test the generated native code. Byte-equivalent emission preserves an
observation's relationship to the source; it does not relabel an earlier
hardware run. Compare actual stage-interval unions and complete outputs before
attributing a timing change to a removed dependency.

Use a case where the same active core reuses its slots more than once. For M-tiled
attention an applicable stress condition is
`ceil(BH * ceil(S1 / TILE_M) / active_core_count) > 1`.
For a recurrent project use its actual chunk ownership formula or a supported
one-core run. Multicore smoke with only one item per core is not a reuse test.

Accepted source owners **once the defect is the library's**: `ascriptor/passes/`,
`ascriptor/frontend/rules_sync.py`, `ascriptor/backends/sim/pipesim.py`; the library's
`docs/diagnosing-sync.md` and `docs/rfc/0006-lowering-pipeline.md` own detailed
diagnostics/model contracts. Hand-written events are not on that list: a kernel that declares its
own `SEvent`s owns their placement, and a reported hazard is about the kernel until the lowered
IR shows otherwise.
Use [debug](../playbooks/debug.md) to trace a missing edge or an incorrect model.

A complete derived example is the [CVC lifetime table](cube-vector-cube.md#concrete-lifetime-table).
For other stage sequences use [generic scheduling](pipeline-model.md): recompute
all work mappings, last readers, capacity and drain after a graph change.
