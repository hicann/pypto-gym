# Optimize a correct kernel

Read [common language](../common-language.md) once per new context and
[preflight](../references/authoring-preflight.md) before material changes. Freeze
the objective, allowed changes, comparison scope and stop conditions before searching.

Start from a unit validated on the target device against its independent reference.
Follow [hardware first](../runtime-and-maintenance.md#hardware-first); use small, low-core
model probes after an observed issue. PyPTO-Pro optimization keeps the selected delivery
auto_mutex mode; generate manual only when the user explicitly requests it, without an extra
closeout variant. Fix failures through [debug](debug.md). State the metric the task needs:
board latency for deployed performance, or explicitly labeled modeled cycles for a
scheduling study. Functional-interpreter wall time is neither measurement.

## Cost before a change

Fill [the performance note](../../templates/performance-analysis.md) using
[Roofline](../references/roofline.md). Record cube MACs/FLOPs, separate vector
and conversion work, unique and requested bytes, reuse counts, tile/capacity,
allowed/active cores and work per core, model resource costs and critical waits.
State clock/bandwidth sources, cache scope and unknown fields.
Before each tuning change, complete the template's before/after dispatch,
allocation and repeated-work tables. Sum every slot before emission, and record
both call counts and processed rows/bytes: smaller tiles can increase calls
without reducing arithmetic. Preserve historical results, development acquisitions
and final measurements as separate evidence with their actual identities.
For attention tasks, use [the attention topic](../references/attention-authoring.md)
for relevant state/layout examples; other tasks follow the shared rules below.

| Investigation | Question |
|---|---|
| Work allocation | How much independent work exists; what limits active cores; how many items remain per core? |
| Reuse | Which inputs are invariant; how often are they reloaded; can they stay resident? |
| Tile, capacity and layout | Are primitive granularity, physical pitch, buffered versions and resident storage compatible? |
| Pipeline and dependencies | What waits; what independent work can advance; which reader retires a slot? |
| Stage internals | Is this stage on the critical path; what instruction, movement or scalar work affects total time? |

This is an analysis order, not a requirement to modify every dimension. Preserve
task restrictions. One-core scheduling and whole-device optimization answer
different questions. For mixed stages derive the work indices, buffer lifetimes
and drain using [generic scheduling](../references/pipeline-model.md) and the
[CVC tutorial](../references/cube-vector-cube.md).
For a slot or window rejection, start with the
[paired API checks](../../../library/examples/api/cube_vector_roundtrip#slot-and-window-boundary-checks)
and retain the exact parameters, located diagnostic and backend/stage outcome.

1. Record source/contract/library revisions, target device/backend/toolchain, shapes,
   core count, warmup/repeat counts, command and metric. Check that imports and artifacts
   come from the source being measured. Use distinct scratch paths.
2. Measure the full composition and relevant stages under one protocol. Read modeled
   critical-path/pipe occupancy only from the selected timing model; do not mix its
   cycles with board microseconds or a different model revision.
3. Pick one supported change: remove redundant traffic, keep an intermediate on chip,
   choose a legal tile, separate overlapping buffer roles, or overlap independent
   loads with real ownership edges. A layout marker does not perform data packing.
4. For a reduction split, define the partial layout, merge owner/mechanism, launch
   topology, visibility, initialization and changed arithmetic order before coding.
   A functional atomic result does not establish a legal board merge path.
5. Preserve the precision/alias/shape contract. Check correctness, tails and same-core
   slot reuse before measuring the same cases again. Roll back a change that violates
   the contract or whose speedup disappears under comparable settings.
6. Report before/after distributions and meaningful limitations. Occupancy or bandwidth
   saturation identifies a bottleneck; it does not prove all traffic is necessary.

Use the unit's `profile` entry: it checks correctness first and measures the same
implementation with its declared warmup/repeat policy. A missing board measurement
stays unmeasured. Do not publish historical fastest-variant claims as current results.
A new optimization task follows its own explicitly agreed input domain and comparison scope.

Record useful findings with the canonical kernel's support/comparison metadata.
Agent guidance links there. Update the canonical unit first; when it has a library
teaching copy, export and validate that copy using the canonical revision and content digest.

## Explain the result and stop

Each experiment retains five lines: bottleneck evidence; hypothesis; expected
space; precision/layout/ownership constraints; result and remaining limitation.
For scheduling-only claims match arithmetic, VF body, layout, transfers, tile
and cores. Residency, grid and tiling experiments record their changed work and
total benefit. Recompute the cost picture after every material change.

For a requested compute-overlap claim, use actual stage/item intervals on the
same core, excluding DMA/synchronization and unioning simultaneous vector lanes.
Prefetch speedup alone does not establish cross-item compute overlap. One item
per core demonstrates distribution; multiple items per core demonstrate reuse.

Retain sample counts, warmup, distributions, raw records, cache conditions and
participant denominators. Invalid all-zero total cycles for expected compute
cannot establish utilization; one unused pipe can legitimately be zero. Keep
valid latency separate from unavailable counters. Model ratios are not HBM
bandwidth efficiency, and uncalibrated multi-core model contention cannot
predict board scaling. Do not multiply residency and multi-core speedups.

Stop according to the agreed criteria and budget. Report remaining headroom
separately; do not add an overlap/utilization gate after success. Preserve a
[checkpoint](../../templates/workflow-checkpoint.md) across context handoffs.
