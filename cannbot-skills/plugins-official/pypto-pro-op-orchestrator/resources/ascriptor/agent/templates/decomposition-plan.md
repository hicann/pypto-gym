# Decomposition: <algorithm>

Status and version: <draft/verified and contract revision>.
Original formula/reference: <unit-local independent implementation>.
Reason for launch topology: <requested or contract-permitted boundary>.

## DAG and stage ABI

| Stage | Inputs / producer | Outputs / consumers | Shape, dtype, layout | Initialized extent / alias | Completion and last reader |
|---|---|---|---|---|---|
| <stage> | <names> | <names> | <ABI> | <rules> | <ownership> |

Every public output has a producer; every internal input has exactly one producer.
Validate unique nodes, acyclic dependencies, complete returns and no dangling output.
Contract representation uses the kernels owner's schema, not this narrative table.

## Saved state

| State | Meaning / version | Producer / consumer | Shape / dtype / layout | Allocation, lifetime, initialization | Local preparation |
|---|---|---|---|---|---|
| <state> | <contract> | <stages> | <ABI> | <owner> | <generated/recomputed> |

A backward unit includes the required forward-state generation. No sibling project
or recorded data may supply a hidden dependency.
For recurrent state, state whether each boundary is pre-chunk or post-chunk and
whether chunks are stored in chronological or reverse order. Distinguish named
forward variants from backward preparation when their rounding seams differ.

## Precision

`plan_tolerance`: <full staged and implemented result versus original formula, reason>.
`implementation_tolerance`: <implemented stage/composition versus staged refs, reason>.

| Stage/edge | Source precision | Result precision | Accumulation/rounding/saturation | Per-stage override / reason |
|---|---|---|---|---|
| <name> | <dtype> | <dtype> | <order> | <budget> |

## Verification and handoff

Record generated seeds/cases, each leaf comparison, full composition versus original
reference, DAG validation, CPU isolation, installed dependency versions and commands.
After implementation, add functional/pipesim and actual NPU-path results separately.
Reject all-zero, sign-flipped, missing, malformed and individually corrupted stage
outputs. Keep justified original pointwise limits; add a norm bound only with a
specific validator counterexample and valid-result margin.
Contract corrections retain evidence and a versioned change; expected outputs are
not adjusted merely to make a failed implementation pass.
