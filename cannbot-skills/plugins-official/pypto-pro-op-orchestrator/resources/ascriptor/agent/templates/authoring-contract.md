# Authoring contract: <kernel name>

Status: draft. Subject: <the demo folder or task this contract governs>.
This is a task record, kept in ignored `tmp/<task>/`; it is not a repository artifact
and there is no machine contract file to validate it against. A kernel that lands in the
gallery lands as a folder of exactly `kernel.py`, `reference.py`, `main.py` and
`metadata.json` -- see [kernels/AGENTS.md](../../kernels/AGENTS.md) for what each owns,
and `python tools/build_index.py --check` there for what is enforced.

## Observable behavior

Write the exact formula, operation/cast order, constants, non-finite behavior and
permitted input domain. State one runtime kernel or the explicit multi-launch topology.
List permitted host work; distinguish reference-only math from production execution.

| Name | Input/output/state | Shape/symbols | Dtype/layout/strides | Allocation/alias/initialization |
|---|---|---|---|---|
| <name> | <role> | <shape> | <storage> | <owner and rules> |

## Implementation boundary

Device/backend and supported core counts: <scope>.

Objective: <correctness / constrained scheduling / deployed latency>.
Allowed changes: <grid, tile, transfer volume, layout and arithmetic boundaries>.
Required evidence and stop criteria: <frozen before search; model/board scope>.
Pipeline applicability: <work per active core, candidate or specific limiting dependency/resource>.
For mixed repeated stages attach the stage/work/lifetime/credit tables from
`pipeline-plan.md`; for performance work attach `performance-analysis.md`.
Report criteria met and remaining headroom separately.

| Storage | Logical shape | Instruction footprint | Physical allocation | Last reader / reuse edge |
|---|---|---|---|---|
| <buffer> | <live extent> | <bytes and alignment> | <shape/slots> | <owner/pipe> |

Tail/empty-input behavior: <valid extents, initialization, reduction masks, writes>.
Workspace and saved state: <producer, consumers, lifetime, layout, version, recomputation>.

## Numerical and verification contract

| Boundary/output | Accumulation/cast/rounding | Comparison | Tolerance and reason |
|---|---|---|---|
| <name> | <order> | <exact/bit/numeric> | <rule> |

Deterministic seeds/cases: <small, normal, tail, multiple tiles, repeated slots,
initialization and precision-sensitive boundaries>.
Validator negative controls: <zero, sign, missing/extra output, dtype/shape,
single-stage corruption and non-finite cases with expected rejection>.
Independent reference: <the folder's own `reference.py`, which never imports ascriptor,
and the domain invariants it asserts>.
Commands and dependencies: <the `main.py` invocations this contract is checked by --
launcher, backend, case -- and what each is expected to do, including expected failure>.
Evidence: <emit / vendor compile / functional / pipesim / board, separately>.
Remaining semantic ambiguity: <none, or precise unresolved behavior and evidence>.

Before acceptance, replace every placeholder and run the folder alone from scratch:
copy it somewhere outside the repository and run `python main.py --list` and then every
case, so a hidden import from the tree around it fails there rather than later. A filled
table alone is not a passing gate, and nothing in the delivered folder records that it
passed -- this contract and the run output are where that lives.
