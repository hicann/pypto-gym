# Debug a kernel

Read [common language](../common-language.md) in a new context and
[preflight](../references/authoring-preflight.md) before implementing or materially changing a kernel.

Start from the observed result. When it came from a device, follow
[hardware-first diagnosis](../runtime-and-maintenance.md#hardware-first). When there is no device
— a handover to review, a workstation, a card that was never assigned — start at the stage that
can see the symptom, which for a hazard is `pipesim`, and say in the report that device execution
is unestablished rather than treating its absence as a pass.
Before sim/pipesim, reduce shape and active cores while retaining the suspected tail,
reuse or inter-core dependency. Record why that small probe still exercises the issue.
Preserve a generated failing case and an independent reference, then find the first
incorrect boundary. A simulator failure can be a library defect. You may read and
repair the new library, including simulator internals, within the authorized task.

Treat warnings affecting correctness, synchronization or support scope as diagnostic
evidence even when execution returns successfully. Retain the message, source/op location,
version and triggering case; determine whether it exposes a kernel error, an unsupported
form or a limitation in the diagnostic itself. Repair the owning layer or document why
the warning does not apply to this case with contract/source evidence and a focused check.
Until resolved, report the affected validation as unverified. Suppressing a warning or
obtaining matching output alone does not resolve a synchronization or support warning.

| Symptom | Next evidence | Owning layer |
|---|---|---|
| Wrong on all cases | Formula, typed ABI, transpose and cast order | Unit / frontend |
| Only tails fail | Physical accesses and first masked reduction | Unit / DMA / vector model |
| One tile passes, reuse fails | [Synchronization](../references/synchronization.md): last readers, slot rotation, event provenance | Unit / passes |
| Functional pass, hazard or deadlock | [Synchronization](../references/synchronization.md): lowered op IDs, event balance and overlapping accesses | Unit first, then passes / pipe model |
| Compile gap | Located op and emitted artifact | Backend / lowering |
| Board differs | First differing stage and minimal board experiment | Kernel / backend / model |

From the accepted library checkout, inspect the fixed smoke or substitute the failing
kernel's existing source path and symbol:

```bash
ascriptor dump-ir examples/api/axpb::axpb --after all --explain
```

Read the reported op ID before using `ascriptor explain PATH::NAME --op N`.
`PATH::NAME` and `N` are substitutions from that dump, not literal runnable arguments.
Every emitted CCE statement's `// #N` connects generated code to its IR origin.

Use the [white-box simulator guide](../references/simulator-white-box.md) for handler
lookup, tensor/register-state tracing, assertions and temporary instrumentation.
Record model-derived rules separately from silicon measurements. Numerical execution
is not timing evidence; an event-balanced trace may still contain a memory hazard.
For predicate fields, online MX subnormals, scalar roots or SIMT zero signs, use the
current [precision boundaries](../references/precision.md) and their owner contracts,
regressions and evidence to choose the owning layer and retain the supported domain.

For a short-row VF store, inspect the emitted distribution and `PAT_ALL` versus
the explicit half predicate before trusting the view shape. M057
showed a last-slot overwrite that both models had silently clipped. The repaired
model rejects active allocation overruns; the kernel must still choose its intended
write mask. For a low-precision numeric mismatch, compare the pre-cast value and
the declared mathematical target separately: M058
restores a source-defined FP32 target, and the [V8 P-stage demo](../../../kernels/ascriptor_kernels/attention/a5_v8_p_stage)
shows the other half — its `main.py` states the measured one-ULP native-`exp` budget for that
stage, and `reference_candidates` in its `reference.py` turns that interval into a set of
admissible HiFloat8 codes instead of a widened tolerance. Neither observation authorizes an
arbitrary tolerance change, and the budget is that stage's on A5, not a general native-exp
accuracy claim.

Reduce the failure to one operation/view, one output or one stage while keeping the
failing initialization and reuse. Add outputs back only after the first boundary is
understood. Do not repeatedly increase timeouts or guess at casts to hide a race.
Investigate the first actor exception before treating a later wait as a missing token.

Open a [defect record](../../templates/defect.md) promptly for a suspected library
issue. Repair the smallest owning layer, update its specification when wrong, add a
generated regression and rerun the original failure. Keep unaffected boundary cases.
Do not weaken the reference, silently discard active out-of-bounds accesses or alter timing constants to make
an incorrect kernel pass. Finish through maintain, including the fixed
version and removal of obsolete workarounds.
