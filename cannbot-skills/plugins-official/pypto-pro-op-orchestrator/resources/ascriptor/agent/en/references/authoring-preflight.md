# Preflight before a kernel candidate

Read after [common language](../common-language.md) and the selected playbook,
before implementation or a material architecture change. Fill the existing
[authoring contract](../../templates/authoring-contract.md) once; infer ordinary
choices from the formula, current declarations and examples rather than asking
a repeated questionnaire.

## Required decisions

| Decision | Concrete result before editing |
|---|---|
| Observable semantics | Formula, casts/rounding, reduction order, aliasing, non-finite behavior, supported shape domain |
| Delivery | One runtime launch or agreed composition; exact permitted host work |
| ABI | Typed GM dimensions/dtypes, scalar binding, explicit output allocation and returned values |
| Work ownership | Available/allowed cores, partition formula, vector participants, work items per active core |
| Storage | Logical extent, instruction footprint, physical pitch/alignment, slots and capacity by memory level |
| Initialization | First accumulation versus later updates, padded lanes, output seeding and in-place rules |
| Precision | Every materialized/lossy boundary, independent reference and comparison budget |
| Synchronization | Producers, all readers, last-reader pipe, publication and reuse dependencies |
| Objective | Correctness, constrained scheduling study or open performance optimization |
| Verification | Full-workload hardware first; issue-driven reduced shapes/cores; matched controls and stop conditions |

Runtime ABI and the frontend's static subset belong to the current library;
do not copy old `shape_bindings`, generated-name or constant-condition rules.
Read [authoring](../../../library/docs/api/authoring.md) when specializing or
using helpers/control flow. No facade name alone establishes a supported form.

## Triggered reading

Read the first column's target, then expand only when the named feature is present.
Do not recursively preload every link; return here when implementation exposes a new boundary.
Find signatures in the [reference](../../../library/docs/api/reference.md) or names in the [manifest](../../../library/docs/api/manifest.json).

| Current boundary | Read first | Expand when |
|---|---|---|
| Ordinary vector row/tail | [Short memory checklist](memory-and-tails.md#vector-tail) and one same-dtype example | Packed, NZ, subviews or slots need their specific boundaries |
| Register/VF, mask | Used operations in [Registers API](../../../library/docs/api/registers.md) | New distributions or mask state need the matching example |
| Reduction/broadcast | [Reduction section](numerical-patterns.md#reason-reduction); `cadd` leaves the result in lane 0, and putting it back across the lanes is `ub_to_reg_single` in the [distribution table](../../../library/docs/api/registers.md) | Changed accumulation, sensitive casts or comparison budgets need [precision](precision.md) |
| Packed/cast/rounding | For packed writes, [derive the footprint](memory-and-tails.md#packed-writeback); look up the format in [Formats API](../../../library/docs/api/formats.md) | Separate carrier placement, store distribution and owned bytes; use [precision](precision.md) for rounding/saturation |
| SIMT | [Ownership checklist](patterns.md#simt-start) and the used operation in [SIMT API](../../../library/docs/api/simt.md) | Derive participant/thread ownership, atomic contributors and rendezvous scope |
| Sort/topk | [Record checklist](patterns.md#sort-start) and the used operation in [Sorting API](../../../library/docs/api/sorting.md) | Fix output order/ties, record footprint and score–ID pairing |
| DMA/view/padding | Used transfer in [Storage API](../../../library/docs/api/storage.md) | Non-contiguous or changed pitch needs the physical-view section |
| Indexed or per-row access (dynamic subscript, `Var.GetValueFrom`) | [Indexing and slicing](../../../library/docs/api/storage.md#indexing-and-slicing) and the [scalar section](../../../library/docs/api/authoring.md#scalar-values-and-memory) | A padded or clamped index needs its own legal-but-masked value |
| Cube, bias, MX | Used family in [Cube API](../../../library/docs/api/cube.md) | Cross-side publication triggers synchronization |
| Cube drain into the vector side (`l0c_to_ub`, `ub <<= l0c`) | [Device facts](facts-device.md#the-drain-into-the-vector-side-has-no-safe-default) and the operation in [Cube API](../../../library/docs/api/cube.md#draining-l0c-into-the-vector-side) | A converting or requantising drain is forced to `SINGLE`; a reduction across M is the only plain copy that needs it |
| Reused slots, cross-side or multiple readers | [Cross-side handoff](cross-side-handoff.md) for the calls, [Synchronization](synchronization.md) and [Synchronization API](../../../library/docs/api/synchronization.md) for lifetimes | Repeated mixed graphs add the [pipeline method](pipeline-model.md) |
| Performance objective / multiple launches | [Roofline](roofline.md) / [decomposition](../playbooks/decompose.md) | Follow the task's corresponding playbook |
| Launch/IR/returned values | [Execution API](../../../library/docs/api/execution.md) | Identity checks distinguish canonical device IDs from facade aliases |

## Implementation and handoff

Settle tiling and dataflow before coding, then build a minimal complete path and validate
each added stage under the task's execution procedure. For each operation, cast, buffer,
synchronization edge and data movement, be able to identify the required semantics,
precision boundary, physical storage need or producer/consumer dependency. Record the
non-obvious choices in the task contract; copied code is not a justification. Existing
canonical units remain the change target when the task is to repair or optimize them.

Repeated mixed pipelines need legal multi-buffer/lookahead, per-stage indices and drain.
Check startup/drain with one item, repeated slot wraps on active cores, valid tails and full output.
Separate schedules preserving arithmetic/layout/traffic from experiments changing reuse, grid or tiling.
Report conclusions through the [evidence table](../common-language.md#evidence), with agreed criteria
and remaining headroom; do not add a new success threshold. Before handoff, record versions,
read paths, invariants, failures and the next boundary in a scratch [checkpoint](../../templates/workflow-checkpoint.md).
