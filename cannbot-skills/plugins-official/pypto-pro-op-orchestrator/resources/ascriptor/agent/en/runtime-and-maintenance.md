# Run, inspect and maintain a kernel

The calls you make while writing, and how a script reaches a card, are on
[running what you are writing](references/development-execution.md); this page is the procedure around them.
Use this snapshot's [sources.json](../../sources.json) to select its source identity. The four facades
own authoring declarations; `ascriptor.runtime` owns execution and source compilation. Backend
selection is explicit per compile/run and does not mutate a process-global target.

| Operation | Current entry | What its success establishes |
|---|---|---|
| Independent expected values | A folder's `reference.py`, which imports no facade, compiler, backend or simulator | Generated inputs and reference invariants, without importing the simulator |
| Functional execution | `OpExec(entry, launcher="sim")` | Supported arithmetic and Surface semantics |
| Lowered pipe execution | `OpExec(entry, launcher="pipesim")` | Dynamic event/hazard/deadlock checks under the model; any of the three raises |
| Source emission | `compile_kernel(entry, backend=..., block_dim=..., bindings=...)` or `ascriptor compile` | A backend can express this Lowered IR |
| Local CANN execution | `OpExec(..., launcher="aclnn")` | Vendor build and the declared device run |
| CANN simulator | `OpExec(..., launcher="cannsim")` | Vendor simulator execution, separately from host pipesim |
| This machine's card | `OpExec(..., launcher="board")`, run on the box | Vendor build and actual execution here |
| This machine's PyPTO-Pro | `OpExec(..., launcher="pypto")`, run on the box | Generated PyPTO-Pro sources executed on this machine's card |

`compile_kernel` has no `output=`: it returns an `Artifacts` whose `files` mapping holds the
generated text, and where that text lands is the caller's business. The `--output` in the commands
further down belongs to a unit runner's CLI, not to this API.

<a id="hardware-first"></a>
## Hardware first; simulation for diagnosis

For a hardware kernel task, compile and test the complete declared shapes on the
assigned idle, locked device first, comparing against an independent reference
and measuring the target workload. Static IR/footprint inspection may precede the
launch. Full-shape functional simulation or pipesim is not a prerequisite.

A workload whose device iteration costs hours may use a full-shape model pass
before the device run. Record the measured per-case device cost and the resulting
ordering decision. This changes the order only: complete declared shapes still
need independent reference comparison on the assigned device for acceptance.

After a correctness or performance issue, construct a smaller diagnostic case
that preserves the suspected cause: dtype/casts, layout/tails, initialization,
slot reuse or the relevant inter-core handoff. Run sim/pipesim from a real file
with a bounded timeout and the fewest valid cores. Regenerate the kernel/launch
for that core count; preserve required Cube/Vector participants, cross-core
interactions and enough iterations to wrap reused slots. Record original and
probe shapes/core counts plus what the reduction preserves and cannot establish.

Recheck the fix on the original full workload and deployment core count. A reduced
model cannot certify its correctness or supply its latency/utilization metrics.
Without hardware, report the blocker. This forbids one move: presenting a full-shape
simulator run as the hardware acceptance that was asked for, or reaching for one instead of
reporting that the device was unavailable. It does not forbid running the full shape under
`sim` and `pipesim` — that is the ordinary authoring loop of
[running what you are writing](references/development-execution.md), it costs seconds, and
what it establishes is named in the [evidence table](common-language.md#evidence). Keep the
deliberately bounded smoke/probe for diagnosis. Explicit model-only research or simulator
maintenance follows its separately agreed scope.

<a id="where-each-step-runs"></a>
## Where each step runs

Run source inspection, lowering and model checks from this checked-in snapshot.
Run a device check on the machine that owns the assigned card, with the same
snapshot on `PYTHONPATH`. Its local `ASCRIPTOR_BOARDS` entry must set `"local": true`;
connection fields are rejected. Generate the inputs and independent reference in
the same process that executes and compares the device result. Keep outputs and
locks in an isolated local workspace. Record the interpreter, imported source path,
source ID, device and complete workload with the result.

[The execution API](../../library/docs/api/execution.md#running-the-whole-unit-on-the-device-machine)
shows the local invocation. A model result never substitutes for hardware acceptance.

<a id="sync-closeout"></a>
## PyPTO-Pro delivery synchronization policy

New deliveries use `sync_mode="auto_mutex"` from their first candidate through the
final package. Verify the emitted `@pl.jit(auto_mutex=True)`, manifest and hardware
result. The low-level `OpExec` default remains manual, so pass the delivery mode explicitly:

```python
delivery = OpExec(kernel, launcher="pypto", sync_mode="auto_mutex")
manual = OpExec(kernel, launcher="pypto", sync_mode="manual")  # explicit user request only
```

Manual keeps explicit backend synchronization mapping, and the DSL may still use
`auto_sync`; generate it only for an explicit user request. Auto_mutex delegates local
locks to PyPTO while preserving the compiler's cross-core protocols and barriers.
The [native policy](../../library/docs/rfc/0013-pypto-native-synchronization.md)
explains the alias forms it cannot support. Validate the selected native artifact
on the same device, full shapes and declared precision/performance criteria. If
emission, compilation, correctness or hardware checks fail, preserve the source
location, logs and blocker; do not fall back to manual. Compilation alone is insufficient.

The source tree supports applicable source emission without tensor dependencies. Numerical
entries use declared host extras. CANN, drivers and vendor PTO/PyPTO tooling remain external.
No folder in either owner declares a supported backend/launcher combination, so run the
combination and read what it says. Either way, a requested
combination must execute that path or report a gap. An emitter's success does not establish vendor
compilation, and PyPTO's support is not described by a historical two-item unsupported list.

DMA padding literals use standard-library host encoding in the base install.
For exact rounding and NaN bits, consult the [owner contract and regression](../../library/docs/api/storage.md#base-install-padding-literals).
Host literal encoding, device arithmetic and VF conversion retain separate scopes.

Inside a copy of `examples/api/axpb` -- four files, nothing else -- these commands inspect the same
independently checked example without CANN:

```bash
python main.py
python main.py --launcher pipesim
ascriptor compile kernel.py::axpb --backend cce -o tmp/runtime/emit
ascriptor dump-ir kernel.py::axpb --after all --explain
```

`main.py --launcher pipesim` generates its inputs, runs the canonical lowering pipeline, checks
event balance, enables GM hazards and compares returned outputs. For trace/state investigation,
copy the accepted library's `examples/api/axpb` into an otherwise empty scratch directory and put
the following [recipe](../templates/pipe_axpb.py) inside it, saved as `pipe_axpb.py`: it imports the
folder's own `kernel` module, which is why it sits beside it rather than next to it. Execute the real
file with this snapshot's `library/` on `PYTHONPATH`. No archive or recorded expected
tensors are used.

```bash
python pipe_axpb.py --output tmp/runtime/trace
```

`PassManager`, `check_balance` and `simulate` below are source implementation
entry points for investigation; they are not facade exports or an `OpExec` launcher.
The fixed domain is one A5 core, contiguous FP32 `(1,64)` inputs and separate outputs.
Bounded integer-valued inputs make `2*x+y` exact; NaN output seeding detects unwritten
lanes. The reference expression is independent of the kernel and simulator.

Every case must have an empty balance/hazard list, no deadlock, exact returned values,
unchanged inputs and two rejected bad outputs. Trace/schedule files are written before
the result assertions; an interpreter exception may occur before a result exists.
Inspect the first exception in that case. Trace event `time_domain` is `cycle`; a viewer's
display-unit metadata does not turn modeled cycles into silicon time.

`check_balance` uses exact trip counts for resolved literal loops; unresolved loops use
bounded `rounds` (default four). A passing bounded exploration is not proof for all
symbolic iteration counts. The pipe scheduler orders a lane's same-pipe FIFO, which does
not establish every device instruction's writeback/landing requirement. Preserve required
device barriers and validate disputed hardware behavior separately. See the library's
[synchronization diagnosis](../../library/docs/diagnosing-sync.md) for the next steps.

The CLI reports actual operation IDs and source origins. Select an ID from that dump before
using `ascriptor explain kernel.py::axpb --op ID`; `ID` is a value to replace, not a fixed example
number. Generated statements carry `// #N` to connect a compiler diagnostic with Lowered IR.
Backend bindings are explicit when a backend requires compile-time specialization; typed GM
shape symbols establish runtime dimensions without scalar-value inference.

The runtime stages generated projects, harnesses and data beneath `out_dir`. Source hashes
control rebuilding. Actual outputs come from the launcher return, and initialized outputs need
the explicit seeding contract. Remote directories have per-output identities. Input/output
files in temporary runtime scratch are transport artifacts, not a required recorded dataset.

Hardware access comes from private configuration, never public prose or source. The operator
checks the selected device with `npu-smi` before a run. The runtime's advisory `flock` coordinates
participating runs; it does not perform the old repeated idle-sampling protocol. Use the configured
environment, one assigned visible device and owned workspace. A timeout is evidence to diagnose;
inspect the first build/run failure before retrying. The retired script/protection override flags
are not supported setup for this product.

For a compiler or simulator repair, first read the relevant library RFC and amend it in the same
change if its contract is wrong. Locate the owning layer:

| Disputed behavior | Current source owner |
|---|---|
| Facade names, signatures and DSL lowering | `ascriptor/a*.py`, `frontend/dsl.py`, `frontend/rules_*.py` |
| Types, operands, accesses and verification | `ascriptor/ir/` |
| Dependencies, events, layouts, resource allocation | `ascriptor/passes/` |
| CCE printing and hardware wrappers | `ascriptor/backends/cce/` |
| PTO ISA or PyPTO printing | `ascriptor/backends/pto_isa/`, `ascriptor/backends/pypto_pro/` |
| Runtime binding, projects, harnesses, launchers | `ascriptor/runtime/` |
| Values, physical accesses and scheduling model | [Simulator white-box owners](references/simulator-white-box.md) |

New operations need truthful types, read/write effects, scopes and device support in the registry,
frontend rules, applicable lowering, model semantics and backends. Do not add a handwritten
legacy autosync session or inline backend escape. Passes use `Rewriter` provenance and explain
decisions; unsupported printing names the source-located operation. Verify the changed semantics
with generated positive and negative cases, then run the affected integration gates.

The library owns small compiler/runtime regressions and defects, kernels owns complete algorithms
and their cases, and agent owns workflows and bilingual routes. No historical test count proves
a successor gate. Exact/bitwise comparison is appropriate for exact contracts; nonzero budgets
need precision evidence and wrong-output controls. Use [precision](references/precision.md) and
maintenance for the detailed repair and evidence protocol.
