# Staged multi-phase Cube matmul on A5: precision and performance gates

Use this reference when changing a staged multi-phase Cube matmul on A5. It
collects target-version facts and turns unvalidated alternatives into explicit
gates. Re-run every hardware probe after changing CANN, PyPTO-Pro, or the A5
subtype.

Start an A/B record from the
[staged matmul checklist](../../pypto-pro-op-perf-tune/references/general-knowledge/staged-matmul-ab-protocol.md).
Use the [stage-task flatten template](../../pypto-pro-op-perf-tune/templates/stage-task-flatten.py.tmpl)
for ownership and tail work, and the
[tiling-key template](../../pypto-pro-op-perf-tune/templates/tiling-key-resource-specialization.py.tmpl)
when the Cube resource graph must vary by shape.

## Keep facts, hypotheses, and decisions separate

Record each result as one of:

- **Target fact:** reproduced on the stated CANN, PyPTO-Pro, and physical A5.
- **Candidate:** compiles, simulates, or has a plausible numerical argument but
  has not passed the target-device gate.
- **Decision:** accepted only after the official precision checker and comparable
  device timing both pass.

CPU replay, static inspection, JIT generation, and an internal `max_abs` limit
are useful diagnostics. None of them is an A5 performance result or an official
precision pass.

## Materialize a BF16 RHS as ordinary FP16 GM before FP16 Cube

On the tested DAV_3510/CANN 9.2 stack, Cube does not perform an implicit
BF16-to-FP16 conversion:

- loading BF16 GM directly into an FP16 Mat tile fails C++ compilation because
  the source and destination types conflict;
- FP16 Left with BF16 Right is rejected by the matmul dtype static assertion;
- a BF16 Vec -> FP16 Vec -> NZ -> Mat `insert` probe can compile, but the same
  insert route used by the actual RHS-consuming stages produced systematic
  errors. A compiling
  cast/insert probe therefore does not establish a valid physical Cube RHS.

Use this staged route instead:

1. Let the designated AIV work partition load BF16 GM in its declared layout.
2. Cast BF16 to FP16 in a Vec tile with the documented Tile-level cast.
3. Store the result into a disjoint, ordinary FP16 GM workspace.
4. Complete the required cross-core visibility protocol before AIC reads it.
5. Let Cube load FP16 GM into the normal RHS Mat layout; keep both matmul
   operands FP16 and accumulate in FP32.

The workspace is not free. Record its maximum bytes, conversion read/write
traffic, whether it overlaps independent Cube work, writer ownership, and the
barrier reached by every launched AIV/AIC lane. Do not treat a local-memory
insert as equivalent to a normal GM load.

## Audit FP16 range; a random sample is not a range proof

FP16 has a much narrower exponent range than BF16. Converting a finite BF16 RHS
to FP16 can overflow to infinity, underflow, or lose low-order bits even when
the sample generator happened to produce only small magnitudes.

For every FP16 candidate, record bounds or measurements for:

- the original input, weight, gamma, and normalized/intermediate values;
- the BF16 value immediately before conversion and FP16 value immediately
  after conversion;
- maximum magnitude, smallest nonzero magnitude, and NaN/Inf counts;
- the accumulation bound and any cancellation-sensitive stage;
- every contract shape and every declared value-domain category.

A conditional bound is useful only when its premise is part of the declared
contract. If the premise is merely true for the data you happened to generate,
label the route range-risk and keep a BF16/FP32-safe fallback or reject it.
Never infer the full input distribution from a sample of it.

## Treat FP32 K128 direct as a different reduction tree

An FP32 K128 direct Cube contraction is a candidate when both operands are
ordinary, same-dtype FP32 GM values and the API accepts the exact layout and
shape. It is not numerically interchangeable with two K64 partials followed by
a VF add: the reduction grouping and rounding points differ.

Promote K128 direct only after all of these gates pass:

1. Run an isolated microprobe on the target A5 and current CANN/PyPTO-Pro.
2. Feed A and B the same deterministic source values and compare the FP32 stage
   output before any BF16 conversion.
3. Exercise normal, small-value, cancellation, tail, and relevant special-value
   cases, then run the unmodified official checker.
4. Confirm generated code, operand dtype/layout, K extent, phase, and unique
   writer ownership.
5. Warm up and measure repeated device time under the same lock and environment;
   include any operand-widening traffic in the comparison.

Compilation or a mean/max error improvement alone does not approve a changed
reduction tree.

A completed A/B is also a caution against over-attribution.  Where a
three-residual K64 chain, a K128 direct chain and a native-FP32 terminal
contraction all retained nearly the same small-region failures on one output
while the remaining outputs passed, that result rejects the stage boundary
under test as the dominant cause for those inputs; it does not bless the
alternatives.  Before changing a consumer reduction again, capture the
producer's FP32 output and test whether the error is already present.  Stop
adding BF16 residual terms when the next residual is exactly zero.

## Gate Fixpipe F322BF16 `CAST_RINT` at the terminal boundary

Fixpipe F322BF16 with `CAST_RINT` is a candidate only for a terminal FP32-to-BF16
protocol boundary. Use it only when there is no later FP32 consumer, the FIX
path has one unambiguous writer, and the Final phase and tail ownership are
proved.

Require the following evidence:

1. Inspect generated code and confirm F322BF16 and `CAST_RINT`; do not infer the
   rounding mode from an API name.
2. Probe exact ties and adjacent representable values against Torch BF16
   round-to-nearest-even behavior.
3. A/B the same FP32 source against the established vector
   `pl.cast(..., CAST_RINT)` boundary.
4. Check aligned and non-aligned valid extents, untouched sentinel space, and
   all official contract shapes.
5. Run the complete official precision checker and comparable device timing.

`CAST_ROUND` used to construct BF16 residual terms has different observed tie
behavior and is not a substitute for the final BF16 protocol conversion. A
successful terminal-stage micro-test establishes conversion semantics only; it
does not establish that K128 direct plus Fixpipe preserves the whole operator.

## Use the three-region checker as the release gate

The benchmark first tries overall MERE/MARE. If that fails, it partitions every
element into normal, small-value, and cancellation regions, and all three
regions must pass. A single normal-region mismatch can fail when the
same-precision reference is clean.

Internal `max_abs`, mean error, mismatch counts, or a locally chosen tolerance
are triage signals. They cannot replace the official MERE/MARE and three-region
result because they do not encode the reference-error allowance or region
membership. Preserve the exact official checker and report per-region counts
for both the candidate and its reference. Also check NaN positions before
interpreting zero-looking error metrics.

See [precision constraints](../constraints/precision.md) for the checker model
and special-value behavior.

## Hold the A5 lock by inode, not by process appearance

`/proc/locks` identifies a lock by device and inode rather than pathname. On
Linux kernels that expose blocked requests there, a waiter line is rendered
with `->`; the held record has no arrow. A waiting `flock` wrapper is not the
holder, and killing that waiter does not release the current lock.

Before using NPU0, resolve the agreed device lock file to its device/inode and:

1. identify the held record and holder PID, not merely a process named
   `flock`;
2. inspect `/proc/<pid>/fd` and the process tree when parent/child FD inheritance
   is unclear;
3. acquire a dedicated FD synchronously before environment setup or device
   work, for example:

   ```bash
   exec {dev_lock_fd}>"$NPU0_LOCK"   # the lock file agreed for this device
   flock "$dev_lock_fd"
   ```

4. keep that shell and FD alive through all child processes, device
   synchronization, log collection, and cleanup;
5. release with `flock -u "$dev_lock_fd"` or close the FD only after every child
   and device job is complete.

Do not background the board command and allow the holder shell to exit. If a
holder looks stale, first prove its PID, command, inode, children, and device
state.  A stale-looking holder is still an external process: obtain explicit
owner/user authorization before signaling it.  Terminating it is not a
handoff: a queued waiter may acquire the inode immediately, so enqueue the
replacement run normally rather than assuming it will be next.

## Specialize resource graphs with `tiling_key`, not a runtime shape branch

A runtime condition that selected two versus three residual Cube chains passed
Python checks and JIT generation, then triggered A5 device error `161002` on the
first launch and poisoned later matmuls in that process. Parser acceptance is
not evidence that runtime control flow may change TileGroup rotation,
auto-mutex edges, operand loads, accumulator use, or phase topology.

Use `tiling_key` when one JIT/wrapper launch must select different static
resource graphs. Inspect every key's IR and generated code. The tested probe
folded the key before resource scheduling and removed it from each IR, avoiding
the unsafe runtime resource branch.

This is a resource-safety result only. `tiling_key` does not prove numerical
equivalence, residual sufficiency, or performance. Alternate keys and shapes in
one A5 process to catch stale state, then apply the full precision and timing
gates independently to every key.

## Prefer exact-order operand reuse before changing a reduction tree

If adjacent outputs share the same Left residuals, make them one static task,
load those residuals once, and compute each output against its own Right tile.
The acceptance invariant is per output: the K-block order, residual order,
`Partial`/`Final` phases, accumulator, and store boundary must be identical to
the frozen schedule.  Do not add the two outputs together or extend their K
extent merely to make reuse easier.

For every such pair schedule, record:

- before/after task, matmul, Left load/move, Right load/move, and store counts;
- an AST or generated-code proof that each output's numerical sequence is
  unchanged;
- a one-output fallback for an odd tail;
- that every `TileGroup.next()` acquisition and use stays in one branch/helper
  scope; and
- target CCE, official precision, and repeated device timing for each tiling
  key.

A tested small-key pair schedule kept its task and matmul counts unchanged,
kept all Right traffic unchanged, and halved the Left loads and moves.
This is a scheduling hypothesis until measured timings confirm it.
Use
[`cube-shared-left-output-pair.py.tmpl`](../../pypto-pro-op-perf-tune/templates/cube-shared-left-output-pair.py.tmpl)
as the implementation skeleton.

Measured A5 data bounds that hypothesis.  Sharing one Left tile across two
outputs and one Right tile across two row tiles kept the whole contract domain
correct, but the aggregate barely moved: the small row-tile cases each improved
by single-digit microseconds while the largest row case regressed by several
times that.  A four-output small-shape reuse schedule came out behind the exact
pair schedule.
The lesson is shape-specific: output reuse reduces Left traffic, row reuse
reduces Right traffic, and larger static task bodies can lose to scheduling
granularity.  Preserve both exact schedules and select by a static tiling key
only after repeated target timing; do not extrapolate a single row count to
shapes you have not measured.

A later exact four-output wave makes the promotion rule sharper.  Static
analysis cut that stage's logical L1 bytes by roughly a tenth and its task
descriptors by rather more, but the slowest physical core still executed the
same number of output chains.  A target A5 B/A/A/B run then returned a
geometric-mean time ratio less than half a percent below the control, so the
candidate was rejected.  Treat logical L1 bytes, issue counts, and task
descriptors as upside bounds, not performance results.  If the critical
per-core output chain is unchanged, require repeated target A5 ABBA against the
frozen exact source; never promote from static counts, simulator timing, or a
profiler trace alone.

The same final source reproduced full-domain correctness in two further repeat
runs, whose aggregate timings differed slightly from each other.  Treat a
spread of that size as run-to-run variation; select schedules from repeated
case-level timing, never from an aggregate delta smaller than the spread you
have observed.

## VF is a strong candidate for vector math in this chain, not a default

The authoritative rule is the conditional one in
[vector authoring constraints](../constraints/vec.md): pick the level that is
correct and supported by the installed API, and when both are legal, measure
both on the target. **No operator family, shape, or prior benchmark makes
either level the universal default** -- including this page. An earlier revision
here said "prefer VF ... switching to Tile needs extra justification", which
inverts that rule; where the two disagreed, `vec.md` governs.

Within that rule, VF is usually the candidate worth measuring first for FP32
vector arithmetic in a staged chain -- normalization, elementwise residual work,
RoPE, fixed-tree additions, compensated addition -- because it keeps the widened
arithmetic chain explicit and avoids paying SIMT-style machinery for operations
that map naturally to vector lanes. Tile-level operations are required for
movement, layout conversion and dtype cast; “VF path” never means forcing an
unsupported cast into VF.

VF has hard limits:

- it is not a default replacement for dense Cube matmul; an all-VF FP32 dot
  needs profiler and precision evidence before use;
- lane masks, tails, and register validity are explicit--tensor `valid_shape`
  does not automatically predicate VF registers;
- insert `vf.mem_bar(VST_VLD)` where a VF store must become visible to a later
  VF load or Tile-level consumer; an auto-mutex or cross-core barrier does not
  replace an intra-AIV memory barrier;
- AIV subblocks may share MTE resources, so give independent GM traffic to the
  designated lane unless a target probe proves concurrency safe;
- every launched lane must reach unconditional MIX/event barriers even when its
  data tail is empty;
- VF does not provide cross-core ownership or global visibility by itself. Use
  a proved GM workspace and synchronization protocol.

Keep a dense contraction on Cube unless measurement establishes a better
target-specific alternative. Keep vector math on VF unless a documented API
and target measurement justify moving it.

## Gate an Acc-to-Vec event bridge on liveness and tail correctness

For an A5 `DualModeSplitM` mailbox, a one-block primitive probe and a completed
`block_dim=28` launch are necessary but not sufficient.  Stage A must reproduce
the production physical-core task mapping and per-lane event order, then pass
two independent checks in the same bounded run:

- **liveness:** launch and synchronize return before the watchdog without an
  AIC/AIV device error;
- **tail correctness:** aligned and odd `valid_m` cases are bit-exact, both AIV
  lanes cover exactly the logical rows, and the invalid-row sentinel remains
  untouched.

Use the aligned SplitM extent when calculating lane ownership:
`aligned=(valid_m+1)//2*2`, `v0=aligned//2`, lane 0 owns `[0,v0)`, and lane 1
owns `[v0,valid_m)`.  Both lanes must still execute the same unconditional
event sequence when one owns zero rows.

A production-shaped probe demonstrated why these are separate gates: two fresh
runs returned normally at the full block count, yet both mismatched most of the
output elements with a large `max_abs` at `valid_m=1`.  Stage A was rejected and
the run did not enter full operator validation.  Apply the same stop rule
to every bridge: a timeout, device error, mismatch, sentinel overwrite, or
missing required tail immediately rejects the candidate.  Keep the GM + MIX
barrier fallback and do not spend a full precision/performance run on it.

If the full schedule seeds an event after one MIX barrier and reuses its token
after another MIX barrier, reproduce that exact barrier topology in Stage A.
Balanced set/wait counts and a barrier-free event probe do not prove token
lifetime through `sync_all`; the current plain-section API and backend do not
provide that guarantee.  A composition with balanced counts, UB `0x1F400` and
all 32 mutex ids audited still returned A5 error `507014` before producing an
output.  Close that candidate after the first device error.  Do not fresh-retry
the same SHA or promote it; require a new protocol/SHA and a dedicated
seed→wave→MIX barrier→second-wave probe first.

## Gate physical TileTypes, live slots, and Vec scratch ownership

Before changing TN or operand dtype, validate the physical TileType chain.
`pl.move` requires compatible physical shapes; setting an N256 Right tile's
`valid_shape` to N128 does not make an N128 Mat source compatible.  An offset
selects a destination-sized rectangle from a larger source and cannot insert a
narrow source into a wider destination.

If hard stages require incompatible narrow/wide or BF16/FP16 types, use
separate typed groups and prove their same-address reuse with unconditional MIX
barriers.  Count live values inside each helper: output-pair and 2x2 schedules
that consume `r0` and `r1` together require two physical RHS slots.  A one-slot
group compiles but aliases the operands; auto-mutex cannot repair insufficient
storage.

Apply the same ownership rule to AIV scratch.  A bare `make_tile` inside a
repeated load -> cast -> store loop is outside TileGroup rotation and can be
overwritten by the next iteration before an asynchronous consumer completes.
Capture the packed GM value and every residual workspace before Cube reads
them.  Use the
[`typed-tilegroup-stage-reuse.py.tmpl`](../../pypto-pro-op-perf-tune/templates/typed-tilegroup-stage-reuse.py.tmpl)
skeleton and record both address intervals and mutex-slot intervals.

## Pass the native reference to small/cancellation gates

Keep three outputs for precision validation: high-precision oracle, candidate,
and target-dtype native reference.  Pass the native output to the
comparator.  Without it, the small-value and cancellation reference counts can
collapse to zero and locally reject results the reference implementation
accepts. When the native path cannot be reproduced, label those counters
diagnostic-only and do not claim a real failure from them.

Apply the same rule to the candidate's input factory.  Where an input-factory
helper imports a device accessor such as `_get_device` from the canonical NPU
helper module, that module is not interchangeable with the CPU oracle: copying
the oracle under the helper's expected name fixes the module lookup but then
fails, because the oracle carries no device accessor.  Copy the NPU helper
under the name the factory imports, keep the oracle under a separate module
name, and load both roles explicitly.  Before locking, import the helper and
assert the device accessor exists without allocating an NPU Tensor; keep the
first real input-factory call inside the device lock.

## Quantify residual pruning before implementing it on Cube

A wider residual dtype does not automatically make a single matmul term
sufficient.  Where one FP16-rounded normalized term replaced two in a
normalization stage, it failed both dependent outputs across every tested
case/seed combination, while the two-term form passed.  The single-term
reconstruction error was under `2e-3`, but the following contractions amplified
it beyond the native-aware comparator threshold.

Sparse second-term correction was not a useful compromise in that measured
model.  Correcting fixed, evenly distributed subsets of the K128 blocks -- up
to most of them -- still failed the dependent outputs for both a one-row and a
many-row case.  Therefore, count downstream comparator failures rather than
approving a residual plan from local reconstruction error alone.  Keep the
per-K residual order explicit and reject partial-correction schedules before
paying the A5 implementation cost when the CPU arithmetic model already fails.

Two FP16 terms are not by themselves a cancellation guarantee.  A CPU screen
kept the proposed per-K128 `t1 -> t2` order, rounded the producing stage's RHS
to FP16, then fed the existing two-BF16-term consumer chain.  Ordinary random
inputs passed, but in-range weights constructed to cancel one output failed at
an intermediate row count, reporting tens to hundreds of small-value errors
against zero in the BF16-native reference.  Adding a third BF16 residual to the
consumer stage cleared both constructed cases and kept the dependent output
passing.  This is a mathematical screen, not an A5 result, but it justifies a
shape-specialized safety key; do not slow the wide path until target
measurements establish that the extra residual term is needed there.

An A5 gate later showed why the ordinary CPU screen is only a priority filter.
A combined BF16 full-A/TN256 small branch and FP16-two large branch passed its
one-row case, but failed multi-row small/cancellation gates even though every
normal-domain error count was zero.  The large cases reported hundreds to more
than a thousand small-value errors, and the small TN256 branch also failed a
dependent output on multi-row shapes.  Reject the whole specialization before
profiling; do not infer that a mathematically plausible two-residual model or
an exact-looking task reorder preserves the target Cube reduction ABI.

The absolute random gate also needs a production control when the candidate is
intended to preserve that ABI.  Three independently staged scheduling
candidates -- consumer-stage output pairing, and TN256 applied to one tiling
key and then to both -- produced byte-identical aggregate comparator records on
the same seeds, passing and failing exactly the same cases in the same
small/cancellation regions.  Their source hashes were distinct and verified on
the target.  This pattern is
not evidence that all three schedules introduced the same error; it shows the
absolute synthetic domain cannot isolate a candidate delta by itself.

For such candidates, capture candidate and the already accepted production
control in separate processes and separate working/build directories, then
compare every output element exactly.  Separate processes matter because
PyPTO-Pro derives the artifact directory from the JIT function name; two
same-name variants loaded in one process or directory can overwrite or reuse
artifacts and invalidate both correctness and timing A/B.  Continue to retain
the absolute random result as range-risk evidence, but only blame the schedule
after a controlled differential mismatch.

Do not approve a wider output tile from task-count arithmetic alone.  On A5, a
TN256 schedule preserved every output bit-for-bit across the whole contract
domain and halved the producing stage's nominal output-task count, yet an
isolated ABBA run was several percent slower in geometric mean over
representative cases.  Adding exact four-output reuse upstream and output
pairing downstream recovered only part of that loss; the combined schedule
remained slower.  A full L0B/L0C tile,
fewer independent tasks, and a longer per-task live range can dominate fewer
DSL calls.  Require isolated candidate/control processes and repeated device
times before composing individually plausible scheduling deltas.

Small exact reuse deltas can still be worth keeping when they pass a stricter
whole-domain monotonicity gate.  Combining four upstream destinations per token
load with two independent downstream outputs per Left move preserved every
output bit across the whole contract domain.  A full-domain ABBA then showed
candidate time no worse than the control on every case, for a geometric-mean
gain of under two percent.  That is too small to infer a target from, but it is
a defensible production increment because the exactness, shape coverage, and
per-case sign all agree.  In contrast, a row-count specialization whose
representative measurements stay within about one percent of alternating
controls should be treated as noise and rejected even if one large case looks
slightly faster.

The isolated differential harness has a two-location helper precondition.
When candidate and control live in different directories and each source
imports a sibling golden helper, copy the canonical NPU helper beside both
sources.  Assert both files exist before acquiring the device lock.  A
missing sibling helper fails before NPU allocation and must never be recorded
as a candidate correctness result.

Flattening an AIV stage must be judged on the full shape distribution, not only
the smallest case.  In one A5 run, replacing the old row-tile owner with
four-row groups distributed over every raw AIV worker, plus flattening the
small-row heads and chunks of the downstream vector stages, preserved every
output bit-for-bit across the whole contract domain.  A full B/A/A/B profile
cut geometric-mean time by several percent against the exact control, with
every case but one improving and that one moving by well under a tenth of a
percent.  The largest gains were on shapes where the old outer row-tile loop
activated only a few physical cores.

Keep the numerical payload inside each group unchanged and prove `(core,
subblock) -> raw_worker` ownership exhaustively.  Use a fallback for regimes
where the existing schedule already fills the machine.  The rule is: change
ownership first, not the reduction tree, and profile every contract shape
before composing it
with Cube reuse deltas.

Composition needs its own full-domain gate even when every delta has already
passed alone.  Combining the exact Cube-stage reuse with the verified AIV-stage
flatten preserved every output bit-for-bit across the whole contract domain.
Its full B/A/A/B geometric-mean time improved on the prior exact control by
roughly a tenth, and every individual case improved.  This all-case sign
check is stronger evidence than adding the two earlier headline speedups:
composition can change stage balance, cache pressure, and barrier wait time.

Conversely, target compilation is not a runtime proof for a multi-wave MIX
pipeline.  An SSA-isolated two-head-wave variant fixed the generated C++
undeclared identifiers and compiled on A5, then returned device error 507015 at
the first synchronize without producing an output.  The concrete cause was a
terminal-stage ownership arithmetic error: the output's chunk count was fixed
by its head count, hidden width and vector width, but the two waves were
assigned chunk counts that summed past it while each wave's own head extent
implied fewer.  Audit both per-wave coverage and
the total physical tensor extent; disjoint waves can still be jointly out of
bounds.  Require a bounded selected-case launch before full exactness, then
rerun all-case differential and performance gates for the final composition.

After correcting those extents, the same candidate compiled, launched and
synchronized its selected cases, then matched an isolated exact control
bit-for-bit on every output.  This turns the arithmetic diagnosis into a
controlled
fix, but it is still only the selected-case gate; full-domain exactness and
performance remain separate acceptance requirements.

The final composition subsequently passed those remaining gates.  It matched
the exact control bit-for-bit on every output across the whole contract domain,
then completed a multi-sample B/A/A/B profile.  Relative to the preceding exact
schedule its geometric mean improved by a few percent, with all but two cases
improving and those two moving by under a fifth of a percent.  This is the
evidence needed to promote
a corrected multi-wave schedule: selected launch establishes bounds/liveness,
full-domain differential establishes numerical ABI, and an alternating
whole-domain profile establishes that pipeline overlap survives composition.

## Minimum acceptance record

For each A/B decision, preserve:

- source SHA, target/CANN/PyPTO-Pro identity, physical NPU, and lock holder
  evidence;
- the single changed variable, exact stage equation, dtype boundaries, and
  reduction tree;
- generated-code evidence for matmul dtype/layout, resource specialization,
  phase, and terminal conversion mode;
- workspace bytes, ownership, barriers, tails, sentinels, and non-finite audit;
- intermediate FP32 comparison plus overall and three-region results;
- warmup policy, raw repeated device times, baseline definition, and conversion
  overhead;
- explicit accept/reject reason without extrapolating a sample to the full
  input distribution.

Do not claim a speedup or numerical safety until the corresponding evidence is
in this record.

## Hidden K-partition count is a resource-graph choice

Do not generalize a four-part K reduction by deleting a wrapper divisibility
guard.  When the hidden extent is smaller than the assumed partition width, a
graph written as `n_parts = hidden // width` runs zero producing iterations,
while the consuming stage still reads the full set of GM buffers it was
allocated -- uninitialized, because nothing wrote them.  Using ceil instead
makes the upper partitions address K tiles outside the input.  Both are
deterministic correctness defects even though the wrapper accepts the shape.

Use a separate static hidden graph.  Accumulate only the legal K tiles, in
their original order, into one FP32 accumulator; never write or load absent GM
partials; and create the missing zero terms locally in UB before each consumer.
Keep hard MIX barriers unconditional.  For cancellation-sensitive small-head
paths, carry the already validated residual chain through every stage that
consumes it, rather than assuming ordinary random cases cover near-zero
values.

The acceptance matrix must include the actual batch/sequence pair, not only the
same flattened row count.  At minimum: a below-partition-width hidden extent at
both the narrowest and a wide head count, a second row-tile row, independent
epsilons, zero input, an aligned control whose hidden extent is a clean
multiple of the partition width, and constructed narrow/wide cancellation
inputs.  Pass both the FP64 oracle and BF16-native
reference to the real three-region comparator.  Gate the frozen reference path in
a separate process with full-domain candidate-vs-production bit-exact capture;
an absolute random control can fail production itself and must not be
misattributed to a delta that only appears off-sample.
