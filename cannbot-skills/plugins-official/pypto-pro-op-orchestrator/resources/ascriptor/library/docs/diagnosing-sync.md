# Diagnose synchronization with generated inputs

Start from a deterministic reference and the first failing operation or stage. The
current [A5 API example](../examples/api/axpb) is self-contained: one vector core
computes `o = 2*x + y` for contiguous FP32 `(1,64)` tensors. With the accepted installed
`ascriptor[sim]` environment, copy that file into an otherwise empty scratch directory,
unset `PYTHONPATH`, and run its actual entries:

```bash
python axpb.py reference --output tmp/sync/reference
python axpb.py check --launcher sim --output tmp/sync/functional
python axpb.py check --launcher pipesim --output tmp/sync/pipe
ascriptor dump-ir axpb.py::axpb --after all --explain
```

The pipe entry lowers Surface IR, checks token balance, enables GM hazard checking
and compares actual returned outputs. `--backend` does not make this host simulation
execute vendor code. These commands need no archived source or recorded expected data.

For complete control over initialization, assertions and trace output, copy the
workflow [recipe](../../agent/templates/pipe_axpb.py) beside `axpb.py` and run
`python pipe_axpb.py --output tmp/sync/trace`. Simulators run from real files, including
the main guard; do not paste the launch into stdin. The bounded integer-valued cases
make the independent Torch formula exact and include positive, negative and zero values.

The implementation entry points are `ascriptor.passes.PassManager`,
`ascriptor.passes.PIPELINE`, `ascriptor.passes.autosync.check_balance` and
`ascriptor.backends.sim.pipesim.simulate`. They are available in the installed package
for diagnosis; they are not authoring facade exports or an `OpExec` launcher name.
`seed_outputs=True` carries the explicit poison into output storage; consume
`result.outputs` instead of assuming every launcher mutates the input placeholder.
`processes=False` keeps this small investigation within one test process.

The trace contains scheduled operation IDs, source locations, pipe tracks and modeled
cycles. Read each event's `time_domain`; viewer display-unit metadata does not convert
cycles to board time. Numerical equality, balanced tokens and absence of modeled
hazards are separate observations. None establishes vendor compilation or hardware
correctness, and balanced events alone cannot establish memory safety.

## Interpret the three checks

* **Numerical comparison** checks all named outputs, shape/dtype, initialized extent
  and the contract's exact/bitwise/numeric budget. Generate fresh reference values for
  each case. Include a bad-output control rather than trusting a permissive tolerance.
* **Token balance** replays each function's event paths. Resolved literal loops use
  their exact trip counts, including zero and reverse ranges. Unresolved loops use
  bounded `rounds` (default four), and branch alternatives are explored. It reports an
  empty wait, token overflow or incomplete drain. This bounded symbolic exploration
  does not prove every runtime trip count; retain real repeated-slot and boundary cases.
  The literal-loop regression preserves valid
  two-slot loops and genuine overflow/empty-wait controls.
* **Pipe simulation** schedules traced accesses with per-lane pipe order and explicit
  events/barriers/mutexes. Enable `check_gm=True` when public/global workspace accesses
  matter. An overlapping write without a modeled happens-before edge is a hazard;
  blocked work with no progress is a deadlock. Physical footprints and atomic effects
  must themselves be correct for that conclusion to be meaningful.

The scheduler treats a lane's same-pipe FIFO as ordered. That model assumption does
not prove a device instruction has completed its writeback or landing before a later
same-pipe instruction consumes it. Preserve device-required barriers, including VF
local-memory ordering where applicable, and compare emitted instructions and a scoped
board experiment when this distinction is disputed. Separate lanes sharing a pipe
name are not one FIFO. A model pass is not permission to remove a required hardware
landing dependency.

## Find the first incorrect boundary

1. Keep the failing seed, initialization, tail, alias, same-core reuse and independent
   reference while reducing the case. The first actor exception can cause downstream
   waits to fail; diagnose it before retrying a timeout.
2. Use `dump-ir --after PASS --explain`, then
   `ascriptor explain axpb.py::axpb --op ID` with an actual reported ID. Inspect its
   source location, provenance, operands, physical layout and memory effect. Generated
   CCE statements carry `// #N` for the same correlation. Do not guess an operation ID.
3. For a hazard, identify RAW/WAR/WAW, both actors/pipes, allocation and physical byte
   intervals. Same-side autosync regions and cross-side ownership are separate. Inspect
   `ascriptor/passes/deps.py`, the planner of that family (`session_sync.py` on A2/A3,
   `local_mutex.py` on A5), `events.py` and the relevant simulator handler before
   concluding a missing edge is a kernel error. A warning outside an
   autosync region remains actionable; widening a region is justified only when it
   expresses the intended ownership.
4. Trace producer completion, first/last consumer and the reuse edge for each slot.
   Slot count, a ledger's credits and the rotating index are distinct: on A2/A3 the credits
   are `min(slots, sync_depth, ring)`, not the slot count (RFC-0005 §5.3). Count warmup,
   body and drain tokens on each participating side/sub-block. A fixed slot index can
   be reused every iteration; multiple slots do not automatically increase its lifetime.
5. For deadlock, distinguish cross-side mutex participation from same-side event tokens.
   Inspect the specific protocol's preset/publish/consume/free semantics, not a universal
   meaning assigned to the names ready/valid. Preserve the participation requirements
   of both vector sub-blocks for the selected handoff. For a cross-side WRONG RESULT
   rather than a deadlock, count the mutex's credits against the hand-off buffer's slots:
   `depth` is how many cycles the producer may run ahead and is always written out, so one
   mutex cycled twice over one tile needs `depth=1`. `crosssync` refuses the mismatch where it
   can prove nothing else orders the two cycles
   ([M10-076](api/synchronization.md#cross-side-ownership)); where something else
   might, `pipesim` is the oracle — an unordered hand-back shows up as a WAR hazard
   between the consumer's `cf.call` and the producer's `dma.l0c_to_ub`.
6. Follow the disputed opcode through `ascriptor/backends/sim/interp.py`, `dma_ops.py`,
   `vec_ops.py`, `vf_ops.py` or `pipesim.py`. Temporary state logging/assertions are
   allowed in an authorized successor source checkout. Record the revision/import
   origin; inspect one allocation/view/actor transition rather than dumping machine
   configuration or unrelated tensors.
7. Reduce the rule to a counterexample and nearest valid/invalid controls. Repair the
   owning layer, retain real physical conflicts, remove exploratory instrumentation
   or justify a maintained diagnostic, and rerun the original failing case. Never
   clip accesses, change expected values or alter cycle costs to pass a wrong kernel.

The DMA/trace defect separates contiguous
row descriptors, pitched rectangles and genuine 32-byte conflicts. The
[shared-publication investigation](rfc/0005-autosync-on-ir.md#55-why-no-run-ahead-analysis-acknowledgement-cell-or-mirror-is-needed) is closed on
the A2 family with the phase model that carried it, and the hand-written publication protocols it
documents remain correct. For SIMT footprints/atomic
classification or a missing scalar store, consult the current status and tested artifact
in [M10-012](rfc/0006-lowering-pipeline.md#9-the-pipe-level-simulator) and
M10-014. A source candidate or an open defect
does not establish a released fix in the installed package.

The M10-069 timeline repair uses full-depth drain controls in
the event regression. Recorded execution-count
differences for `sync.release` with partial presets still need focused
model/emitter investigation. Full-depth passes do not qualify partial-preset drains.

## Inspect the inferred reuse distance

A legal double buffer may still receive a conservative one-iteration dependency.
In runtime `765e4ec`, `cellfold` removes read-only `Var(tick)` aliases before
`autosync`, while the dependency analysis recognizes explicit advancing Cells
and does not infer the step of an i32 `cf.for` induction value. The
bounded counter comparison (`docs/migration/fragments/for-iv-slot-distance-20260909/README.md`)
preserves equivalent outputs and records the different inferred K-buffer distance,
actual last-reader order and native development measurements. An explicit Cell
with one unconditional increment makes that rotation visible to the existing
analysis. This documents a source workaround; the runtime inference is unchanged.
It is separate from M10-008's unsafe publication sharing and from the M10-069 and
M10-070 correctness repairs.

## Current diagnostic inputs

Generate inputs and independent references at run time. Do not use a stored
corpus or replayed result as device acceptance. Keep the failing source,
minimal reproducer and bounded model/device logs with the task.

Record current defects with expected/actual behavior, source-located reproduction,
generated controls, model-derived versus silicon-measured evidence and the tested
package/source version. Specifications, implementation and regressions belong to the
library; a complete algorithm and its precision/launch contract belong to its unit.
