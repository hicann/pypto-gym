# Performance analysis: <case / candidate>

This is an analysis note, not a runtime API or a new unit schema.
See the completed MLA case in [English](../en/references/mla-cost-case.md) or
[中文](../zh-CN/references/mla-cost-case.md) for layout, split-KV and row-tile comparisons.
The MHA worked arithmetic in [English](../en/references/mha-cost-case.md) /
[中文](../zh-CN/references/mha-cost-case.md) distinguishes reconstruction from unavailable
historical source/profile evidence and from current measured results.

Objective: correctness / constrained scheduling / deployed latency.
Contract and artifact identities: <source, library, backend, comparison>.
Allowed changes: <grid, tile, transfer volume, layout, arithmetic boundaries>.
Exit criteria and budget: <agreed before search; report headroom separately>.

## Fill before each tuning change

Keep a before/after record for every measured tile, launch, residency or scheduling
change. Fill numeric values and their derivation before emission; update the
same record with accuracy and timing afterwards. Use `N/A` for an absent phase
and `UNKNOWN` for missing evidence. Source-assigned work is distinct from measured
core utilization. Split producers and merge owners need separate denominators.

| Dispatch / repeated work | Before | Candidate | Formula / source / unit |
|---|---|---|---|
| Total logical output rows | | | e.g. `B*SQ*Nq` |
| Q-head → KV-head mapping / row-group membership | | | Shared K/V requires the same mapped KV head and batch |
| K/V numerical relation / actual storage alias | | | Record separately; count an alias once but retain all readers |
| Logical / physical M; KV tile N | | | Include useful and padded rows |
| Row groups / producer items / KV splits | | | Splits repeat row ownership |
| Device / allowed / launched Cube and Vector cores | | | Read actual launch metadata |
| Source-assigned active cores / measured active cores | | | Keep unavailable counters UNKNOWN |
| Busiest Cube: items / distinct rows / producer row-visits | | | Repeated partial rows are not new output rows |
| Busiest merge Vector: items / output rows | | | Separate from producer distribution |
| KV iterations per producer / global / busiest core | | | Include empty partitions separately |
| Full KV rereads per batch / equivalent reads on busiest core | | | State the row grouping and residency scope |
| Q and RopeQ requested bytes | | | Include rereads across KV splits |
| K and RopeK requested bytes | | | Separate from unique storage and HBM bytes |
| Independent V requested bytes | | | Do not inherit an MLA-specific V=K assumption |
| Score / product FIX publications and bytes | | | Say whether one call splits to two Vectors |
| P publications and bytes | | | Count both participants |
| Max/sum and output-state updates | | | Record VF calls **and** row updates |
| Partial-state GM publications / bytes | | | Include initialized padding |
| Merge items / parts; state and product loads / bytes | | | Include collective completion and output stores |

One row below describes one allocation family. Record `shape × dtype × slots`,
physical pitch/padding and the resulting bytes for both candidates. Count a
view or typed alias once with its backing allocation, and retain the last reader
that prevents reuse. A smaller logical window does not shrink its physical carrier.

| Role / storage | Before: shape, dtype, slots, padding → bytes | Candidate: shape, dtype, slots, padding → bytes | Lifetime / alias owner |
|---|---|---|---|
| Q / L1 | | | |
| RopeQ / L1 | | | |
| K / L1 | | | Include PV's last read when K serves as V |
| Independent V / L1 | | | Retain delayed PV's last operand-staging reader |
| RopeK / L1 | | | |
| P / L1 | | | |
| Score / L0C | | | |
| Product / L0C | | | Note sequential alias versus overlapping generations |
| Implicit matmul operands / L0A and L0B | | | Slot reservation can exceed a typed operand view |
| Score / UB per Vector | | | |
| Product / UB per Vector | | | |
| P / UB per Vector | | | Include physical NZ pitch |
| Max, sum, rescale / UB per Vector | | | Each role and its slot count separately |
| Output accumulator / UB per Vector | | | |
| Output staging / UB per Vector | | | |
| Merge metadata / UB per Vector | | | |
| Partial output, max, sum / private GM | | | State who writes and who merges |

| Storage total | Before bytes | Candidate bytes | Available bytes / source; remaining space |
|---|---|---|---|
| L1 per Cube | | | |
| L0A / L0B per Cube, separately | | | |
| L0C per Cube | | | |
| UB per Vector | | | |
| Private GM workspace | | | |

If emission rejects a candidate, retain the exact operation, source location,
parameters, backend/version and diagnostic. Link an owner reproduction before
turning that failure into a support rule; do not generalize one window failure
to every subview. Recalculate all slots before trying a larger tile.

## Complete the evidence

| Result | Before | Candidate | Evidence / limitation |
|---|---|---|---|
| Hardware latency, samples and collection protocol | | | List every acquisition, median and min/max; preserve the task's strict or non-strict comparison |
| Independent accuracy / input and output contract | | | |
| Runtime kernel count | | | |
| Stage/item evidence: staging, QK, softmax, PV, sync/loop | | | Label model intervals separately from measured hardware; do not add overlapping intervals |
| Confirmed bottleneck | UNKNOWN unless established | UNKNOWN unless established | Name the measurement that establishes it |
| Hypothesis and predicted work change | | | State changed dimensions and remaining alternatives |

| Quantity | Value / unit | Source, scope and uncertainty |
|---|---|---|
| Actual and allowed cores / work per core | | |
| Cube work by operand precision / vector work / conversions | | |
| Unique input/output bytes | | |
| Requested bytes by boundary and read/write direction | | |
| Input reuse counts / per-core copies / padded work | | |
| Resident data + live buffer versions by memory level | | |
| Measured L2/HBM bytes | UNKNOWN unless measured | |
| Nominal / sustained compute and bandwidth | | |
| Access pattern / cache / frequency evidence | | |
| Fixed-work resource bound / dependency bound | | |
| Bound after a justified removable-work change | | |
| Actual time distribution and valid samples | | |

Label each fact as measured, source-inspected, recomputed from stated parameters,
or reported with raw evidence unavailable. A successful arithmetic reconstruction
does not restore deleted source or profiles. Record current and control source,
artifact, runtime and input identities separately, even when their shapes match.

Experiment: <one hypothesis>.
Evidence: <heavy resource and critical waits>.
Expected space: <work removed or latency hidden, with units and assumptions>.
Implementation constraints: <precision, layout, ownership, lifetime>.
Matched control: <what is fixed; intentionally changed dimensions>.
Result: <correctness, changed work, actual timing and remaining limitation>.
Stop: <criteria met / budget exhausted / specific external blocker>.

Unknown HBM traffic cannot be replaced by GM requests to report HBM efficiency.
A pipe ratio requires a valid denominator; two simultaneous lanes are not two
wall-clock intervals. Model cycles, nominal compute estimates and board time
retain separate labels.
