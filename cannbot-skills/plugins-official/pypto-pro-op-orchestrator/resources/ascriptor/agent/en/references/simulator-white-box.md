# Explore the simulator precisely

You may inspect simulator handlers, trace state, add temporary logging/assertions
and fix a confirmed defect in a source checkout of the new library. Internal APIs
are readable and editable; they are not ordinary public facade exports. Record the
checkout revision and actual import path so the experiment identifies its model.

The functional interpreter evaluates values from Surface IR. Pipe simulation runs
Lowered IR, records functional accesses and schedules pipe/event dependencies to
detect hazards and deadlocks. Modeled cycles use a versioned cost model. None alone
establishes CANN compilation or a silicon measurement.

Start with the generated-input [pipe recipe](../runtime-and-maintenance.md), which
shows lowering, balance/GM checks, actual output comparison and trace saving in a
complete real file. Keep that reference and reduced case while following the handlers.

1. Preserve deterministic input generation, output initialization and an independent
   reference. Reduce to the first failing operation or stage without removing the
   tail, aliasing, non-finite value or repeated-slot condition that triggers it.
2. Dump the IR with `--after all --explain` for lowered questions. Record operation ID,
   opcode, source location, origin chain, dtype, operands, attributes and scope. Use
   `explain --op` for the actual ID. Inspect the relevant generated `// #N` statement
   when the emitter or vendor compiler is implicated.
3. Locate that opcode in the following implementation owners, then follow only the
   handler and helpers that compute the disputed behavior:

| Question | Owner in the library checkout |
|---|---|
| Arguments, shape symbols, output seeding | `ascriptor/backends/sim/launch.py` |
| Dispatch, memory views, registers, masks, actor state | `ascriptor/backends/sim/interp.py` |
| DMA byte extents, padding, layout transfers | `ascriptor/backends/sim/dma_ops.py` |
| A2 tensor-vector repeats/strides | `ascriptor/backends/sim/vec_ops.py` |
| A5 VF register operations | `ascriptor/backends/sim/interp.py`, `ascriptor/backends/sim/vf_ops.py` |
| Rounding/saturation | `ascriptor/backends/sim/cast_rounding.py`, `cast_saturation.py` |
| Access overlap, event tokens, vector-clock hazards | `ascriptor/backends/sim/pipesim.py` |
| Analytical cycle parameters | `ascriptor/backends/sim/timing/` |

From the library checkout, these read-only commands locate actual definitions:

```bash
rg -n 'class MemRef|class RegRef|class MaskRef|class Machine|def run_block' ascriptor/backends/sim/interp.py
rg -n 'class Access|class Scheduler|def _check_hazards|def simulate' ascriptor/backends/sim/pipesim.py
```

4. Trace before/after state for one op. Useful fields include core/side/sub-block,
   `MemRef.base/slot/offsets/extents/gm_strides/view_offset`, backing allocation,
   resolved byte indices, `RegRef.bytes/dtype/valid`, `MaskRef.bits` and a full
   `MaskRef.physical()` snapshot, plus the selected round mode. The logical mask view
   alone omits fine bits that pack/unpack may retain. For scheduling, record
   `Task.op.id/pipe/accesses/clock`, consumed tokens and
   last readers. Do not dump unrelated full tensors or machine configuration.
5. Add assertions at the first disputed transition. Write a small executable file
   under ignored `tmp/<task>/`; never launch a simulator runner from stdin. Use unique
   output directories for each case, stage and input call, so a later call cannot
   overwrite earlier evidence; keep the assigned process budget. Capture the first actor exception before
   interpreting downstream wait failures. A timeout retry is diagnostic only when
   execution cost, rather than a missing dependency, is the remaining hypothesis.
6. Write the derived rule, revision, source location and exact reproducer/result.
   Label it **model-derived**. If hardware behavior matters, the lead schedules the
   minimal board experiment with the same generated reference and records a separate
   **silicon-measured** result, toolchain and domain.
7. When hardware and the model disagree, retain both observations and minimize the
   case. Compare the declared contract, independent reference, model and emitted
   instruction, then repair the layer that violates the intended contract. Never clip accesses,
   change expected values or alter timing costs merely to pass a bad kernel. Remove
   exploratory instrumentation or retain a justified maintained diagnostic; ship the
   smallest fix, regression, defect status and fixed version.

The maintained follow-up is library maintenance.
Temporary investigation is distinct from a supported final fix. A suspected model
defect is enough to open a record; certainty is not required before preserving evidence.

Use maintained failures to practice the complete investigation. These cases were
derived from independently generated M10 project inputs; they need no recorded data:

| Case and first disputed boundary | Narrow correction and regression |
|---|---|
| M10-006: a final-axis FP32 slice should fill one `[1,32]` UB row, but lowered DMA emitted 32 padded scalar bursts | Inspect the DMA descriptor before execution. Preserve one contiguous 128-byte burst on both load/store. Compare Surface and Lowered outputs with generated `arange` values. |
| M10-006: reshaped pitched GM rectangles produced false overlap reports and could omit later rows | Inspect resolved offsets, shape, strides and covered intervals. Keep disjoint rectangles, a true overlap, stride holes, later-row accesses and different pitches. The 32-byte conflict granularity remains unchanged. |
| M10-007: two-slot fill/drain loops pass numerical and pipe execution but the static checker reports overflow | The checker must evaluate literal `range(2)` as two iterations, not its symbolic representative count. Keep zero/reverse ranges, a genuine third-token overflow, an empty wait, an incomplete drain and long-loop termination controls. |
| [M10-008](../../../library/docs/rfc/0005-autosync-on-ir.md#55-why-no-run-ahead-analysis-acknowledgement-cell-or-mirror-is-needed): one ready event is shared by key and delayed score publication | The edge-planner defect is historical; A2/A3 slot sessions and A5 local mutexes replaced that planner. The dedicated score `SEvent(Pipe.V, Pipe.MTE3)` and reuse guards remain valid explicit protocols. Their passing pipe suites qualify only those project scopes. |
| [M10-051](../../../library/docs/api/mask-write-semantics.md): inactive predicate bits and pack/unpack disagree with native output | Inspect logical lanes and all 256 physical bits. The diagnosed native contract corrects the old model/reference formulas; keep nonzero inactive destinations, both pack halves and alias controls. Unobserved producer fine bits and b64 raw fields remain outside the measurement. |
| M10-053: Surface rounds a scalar root through a cell, while Lowered exposes an unrounded value | Round `scalar.sqrt` at its declared result type. The historical `sqrtf` link failure belonged to native instruction selection. Preserve independent dynamic-input references; ordinary CCE/PTO BF16 conversion and sqrt were subsequently repaired in M10-055, while PyPTO restrictions remain upstream gaps. |
| M10-054: native rounding changes negative-zero bits while shared IR/model/reference already agree | Repair the target wrapper's zero-result sign handling. Keep nonzero/NaN bits and the existing `fmod` intermediate unchanged; model success and actual repaired board qualification are separate evidence. |
| M10-057: a 128-half store through the final 64-half row silently passed both models | In `Interp.op_vf_store_cont`, trace the backing allocation, composed origin/offset, distribution and selected active destinations. Validate every absolute address before writing, then use those same addresses. Keep exact-end, LOWHALF, empty-mask, sparse-lane, negative-base and no-partial-write controls; the short view never supplies an implicit mask. |
| M10-060: the new bounds guard exposes a V2/V4 packed tail store extending from byte 8,128 to 8,256 in an 8,192-byte tile | A full HiFloat8 ZERO carrier packs to 128 bytes, while the row owns 64. Apply the explicit HiFloat8 LOWHALF mask in the two affected tail helpers and preserve the complete 64-byte guard. Keep valid 128-half state stores and FP32 pack4 paths unchanged. Retain the original matrix failure; this is a kernel footprint repair, distinct from M040 and M059. |

From a checkout with an accepted environment containing the relevant repairs:

```bash
python -c 'import ascriptor; print(ascriptor.__version__, ascriptor.__file__)'
python -m pytest -q -n 0 tests/runtime/test_dma_footprints.py tests/passes/test_balance_loops.py
```

The first file's maintained tests
separate descriptor formation, values and physical overlap. The second file's
tests separate literal token
conservation from unresolved loop bounds and replay termination. For installed-wheel acceptance, copy
the selected tests into an ignored directory outside the source package, unset
`PYTHONPATH`, run with the assigned immutable environment and record its actual import
origin. A source-checkout pass and an installed-package pass are separate observations.

During diagnosis, add temporary assertions immediately before the descriptor or
access interval is consumed, and log one operation's resolved state. First show the
counterexample under the affected version; then change the owning rule and show both
the original case and the negative controls. Never shrink a real physical conflict:
two cores writing distinct four-byte values can still conflict within the same
32-byte tracked block. Correct ownership or ordering when that is the actual cause.
