# Derive a multi-stage buffered pipeline

Use for repeated mixed computation: CVC, VCV, CVCV, VCVC and longer graphs.
The stage names select examples, not a fixed buffer recipe. Start from
[preflight](authoring-preflight.md) and fill [the pipeline plan](../../templates/pipeline-plan.md).
For a performance decision also read [Roofline](roofline.md).

## Define the graph and work

Name every stage, its resource/participant, inputs, outputs, mathematical
precision and work mapping. An edge can connect the same item, a previous item,
several producer items, or an invariant shared input. Include every reader,
including consumers several stages later and saved scalar state.

Separate global item identity, core assignment, local pipeline round and each
stage's item index. Derive the number of items for each active core, including
idle cores and uneven partitions. Increasing the core count may leave only one
item per core and remove all opportunities for cross-item lookahead.

Two cube stages share M and associated transfer resources; several VF stages
share their participating vector resources. Dependencies inside one item
remain valid when different items overlap. A recurrence can restrict the
advance of one stage while leaving independent input movement available.

## Select a legal schedule

For one-to-one stages, describe stage s at local round t by `k_s = t - d_s` and
the guard `0 <= k_s < N_s`. Select d_s from dependencies, resources and capacity;
it need not equal the stage number. If stages change tile granularity, use an
explicit mapping `k_s = f_s(t)` and describe its rate, join and valid inputs.
No generic delay table alone proves such a graph executable.

### Group consecutive stage pairs

For one-to-one alternating stages, group adjacent pairs: group g handles item
`t-g`, and the final group may contain one stage. With zero-based stage indices,
`d_s = s // 2`. A one-item CVCV lookahead is:

```text
for i in range(N + 1):
    if i < N:
        C1(i); V1(i)
    if i > 0:
        C2(i - 1); V2(i - 1)
```

Each call denotes its side's work and required synchronization. A mixed kernel
[splits into cube/vector sides](../../../library/ascriptor/passes/split_sides.py);
dependencies, resources and events determine when each stage starts. Sharing
an `if` does not create a completion barrier. Select buffers by item identity
and derive capacity/reuse from actual last-reader completion events.

| Graph | Grouped mapping | What needs checking |
|---|---|---|
| C1 → V → C2 | C1(t), V(t), C2(t-1) | V output survives until the next group's C2; C1/C2 share cube resources |
| V1 → C → V2 | V1(t), C(t), V2(t-1) | C output survives until the next group's V2; both VF stages share vector resources |
| C1 → V1 → C2 → V2 | C1(t), V1(t), C2(t-1), V2(t-1) | Two interleaved groups, distinct live versions and last-reader synchronization; one drain round |
| V1 → C1 → V2 → C2 | V1(t), C1(t), V2(t-1), C2(t-1) | Both handoff directions, final cube operand lifetime and shared resources |
| C1 → V1 → C2 → V2 → C3 | C1(t), V1(t), C2(t-1), V2(t-1), C3(t-2) | Three group-specific item indices and a complete drain |

For G groups, guard each group with `0 <= t-g < N` and traverse
`range(N + G - 1)` for startup and drain. These are logical work groupings;
verify actual overlap in the trace. Recheck dependencies and capacity for
cross-item state, extra readers, changed rates or round barriers.

Order same-round producers/consumers and resource users explicitly. A producer
must not block waiting for capacity that only a later, now-unreachable consumer
can return. Changing credit depth without changing storage and schedule does
not solve that cycle. Consider consumer-first issue, a shorter lookahead or
distinct storage when necessary.

Conceptual plan expansion is:

```text
for each pipeline round, including startup and drain:
    visit stages in the proved issue order
    compute this stage's own work mapping
    if that work is valid and all dependencies are satisfied:
        acquire capacity, perform/publish work, retire required readers
```

This is a reasoning recipe, not dynamic stage-dispatch syntax for the DSL.
Implement supported static/native control flow and verify the emitted IR.
The [CVC tutorial](cube-vector-cube.md) maps the recipe to actual source.

## Derive physical storage and credit

For each edge and role, mark the protected write, all readers, last physical
reader's pipe and the release that authorizes overwrite. The required storage
is the maximum number of simultaneously live physical versions under the
proposed schedule. Include aliasing, padded footprint, subblock pitch and
long-lived invariant data when totaling memory capacity.

Count against the proved issue/completion partial order, including reuse
constraints; the difference between logical round labels alone is insufficient.
If the schedule allows publishing a new item before the previous item's last
reader retires, both versions must fit. If two earlier versions can still be
live, three slots are needed to admit that publication without stalling.
Earlier retirement or a capacity wait can change the live count and schedule;
stage count alone cannot choose DBuff/TBuff/QBuff. What the derivation yields is a
sufficient depth, not a minimal one: a smaller depth that accepts different stalls
may also be legal, and nothing above rules one out.

Use the producing item's identity for its slot and the consuming item's
identity for that same version. Do not index delayed reads by the current
producer counter. Keep different lifetime families independent. A K/V input
used again by a later cube stage stays live until that later read retires.
Recurrence metadata such as rescale factors must follow its consuming item.

Event/mutex credits describe permissions and maximum outstanding work; they
do not allocate bytes. Initial credits, lock/ready/wait/free, physical rotation
and final drain must match the protocol. Publish producer completion; return
capacity after the last reader, potentially on a different pipe. Protect by the actual
storage role rather than the stage boundary: acquiring a handoff capacity ahead of work
that does not need it is legal and costs the independent computation it blocks. Read
[synchronization](synchronization.md) for the current APIs and reverse reuse edges.

## Keep scalar metadata with its payload

A delayed consumer may need both a tensor and the scale, mask, length or
normalizer produced for that tensor. Protect those values with the same item
identity even when they use different physical slot families. Keeping the
payload in two slots while overwriting one shared scale can pass a constant
input and still corrupt a later nonuniform item.

Treat separate recurrences separately: the next producer's reduction state
may advance while the previous consumer's output state waits, provided its
metadata remains intact. Check item transitions as well as slot wraps. The
[attention topic](attention-authoring.md) applies this to online max/sum,
delayed rescale and PV accumulation; it does not eliminate the output update
stage merely because the source has a lookahead loop.

## Finish the schedule

For each stage derive its valid rounds; the loop ends after the last required
consumer finishes. `N+1` is one one-step implementation, not a universal drain.
With nonuniform delays/rates, compute each final valid item explicitly.
Check that every required stage handles each item exactly once, including
state updates and stores, and that unused first/last-round stages are skipped.

For guarded producers, a skipped producer does not erase an older pending
token. Account for previous work separately. For joins, wait for all required
producers, and release shared input only after every consumer retires. For
recurrences, preserve the exact arithmetic/rounding order. A proposed schedule
that exceeds storage, lacks a supported handoff, or violates a recurrence must
be revised or carry a precise, evidence-backed restriction.

## Validate the actual schedule

1. Check independent numerical outputs and storage/precision ABI. Include one
   item, depth boundaries, multiple wraps on one active core, uneven partitions
   and supported tails. Zero work is tested only when in the contract.
2. Check lowered hazards, deadlock and event balance. Retain negative controls
   for early overwrite, missing drain and delayed-state indexing. A missing final
   drain is the reason the numerical control cannot be dropped: it passes the hazard
   and deadlock checks and simply omits an output item.
3. Compare a matched serial implementation when isolating scheduling. Use
   fresh processes if module caching could mix candidate/control sources.
4. Map actual compute task intervals back to stage and item, with source/IR
   identities. For an intra-core overlap claim compare the corresponding
   cube/vector participants on that core; exclude DMA and synchronization.
5. Take the union of vector-lane intervals, intersect with the cube intervals,
   then union overlaps before summing. A sum across simultaneous lanes doubles
   the claimed overlap. Source order and a positive DMA overlap do not prove
   cross-item compute overlap.
6. If deployed speed matters, measure hardware separately with a matched
   protocol. A legal multi-buffer program can still be serialized by waits;
   more overlap can coexist with removable traffic or lower overall performance.

A CVC schedule can read, in source order, as though it produces C1 early and still overlap a
different pair of stages in the trace. That is why predicted stage labels are checked against
a trace instead of read off the source. Obtain that trace yourself: run
[the mixed-pipeline demo](../../../kernels/ascriptor_kernels/tutorials/mixed_pipeline) under
`--launcher pipesim` on a `cvc_pipeline_*` case and map the intervals back to stage and item.
A one-item case may have zero overlap and extra startup cost legitimately.

## Transfer the method to a new graph

Recompute mappings, resource order, last readers, storage and drain after
adding/reordering a stage, changing tile rates or adding a consumer. Preserve
the original formula and precision boundaries. Start with the nearest
[pattern](patterns.md), then test at least one configuration not copied from
that example. Generalization means deriving and validating the changed graph;
repeating the same fixed delay and depth on every graph does not establish it.

For an additional late consumer, the
[residual teaching demo](../../../kernels/ascriptor_kernels/tutorials/late_reader) retains P
across its early and late readers: the activation reads P immediately, the residual add reads
it again after the second matmul, and the slot's lifetime is set by the last reader. Its
three-slot P ring and independent two-slot families illustrate graph-specific lifetimes and a
different drain. Run it under `--launcher pipesim`, not just `sim` — the functional simulator
gives each launch its own storage and will accept a lifetime the real pipeline would violate.
