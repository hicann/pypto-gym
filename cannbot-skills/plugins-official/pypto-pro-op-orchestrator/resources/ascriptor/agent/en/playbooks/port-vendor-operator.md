# Port a vendor operator

You hold a vendor C++ operator — AscendC sources, a host tiling function, an infershape — and
the target is an Ascriptor kernel, usually on a device family the upstream declares unsupported.
Read [common language](../common-language.md) once, then this page, then
[preflight](../references/authoring-preflight.md) around the code path you end up with.

This is not [migrate](migrate.md), which moves a project that already lives in this workspace,
and not [import](import-pypto-pro.md), which turns PyPTO Pro source into Lowered IR. It is
[authoring](author.md) with an upstream implementation as evidence, and the work that page does
not cover is at both ends: before the first line of DSL, and after the first correct run.

## Source-only handoff

For a new A5 PyPTO-Pro task limited to source-case functional/precision validation, when the caller
has official AscendC source (normally A2/A3), its host/tiling/infershape and a runnable source case,
select `workflow_mode=source_ascendc_case`. Load the installed `pypto-pro-scriptor-develop` skill
and follow its `references/ascendc-source-functional-precision.md` link; the OpenCode location is
`$CANNBOT_CONFIG_ROOT/skills/pypto-pro-scriptor-develop/references/ascendc-source-functional-precision.md`.
If the skill is unavailable, report the missing resource rather than guessing a path.
Do not dispatch `pypto-pro-op-develop` for this entry: that
skill assumes a frozen PyPTO-Pro `DESIGN`, `DESIGN_BINDINGS` and Module contract. The source-led
route first derives a read-only `SOURCE_CASE_CONTRACT.json` and `SOURCE_MAPPING.json`, then uses
`pypto-pro-scriptor-develop` to write the A5 DSL and `pypto-pro-scriptor-verify` to independently
check the real NPU result. Missing source semantics or a runnable case is an input gap, not a
reason to invent a formal contract.

The source-led route has one-case scope unless a case matrix is supplied. It must leave
`FUNCTIONAL_RESULT.json`, `PRECISION_RESULT.json`, `RUN_MANIFEST.json` and `ISSUES.md`; a
performance result is separate and cannot turn a missing functional or precision check green.
Existing formal Scriptor state/contracts and requests for the full Scriptor or optimization flow
keep their original entry. This route does not advance formal state or replace its acceptance,
delivery or performance gates; separately requested performance work remains outstanding.

## 1. Fix the snapshot, name the target, read the support table

Record the upstream revision, every file's complete SHA-256 and the actual UTC read time in the
[migration record](../../templates/migration-record.md). The upstream is read-only evidence: it
cannot become a runtime import, and it cannot supply the reference.

**Name the delivered backend and launcher, and keep them apart from the controls.** A port is
delivered on one backend — for A5 that is normally PyPTO-Pro, emitted with
`compile_kernel(entry, backend="pypto_pro")` and run with `OpExec(..., launcher="pypto")`. CCE and
PTO ISA are *controls*: they adjudicate a failure and they catch a form only one backend prints.
A CCE pass is never the delivery, and a case that runs under CCE but not under the delivered
backend is an open item, not a green one.

Read the upstream's own product support table before anything else. When it marks your target
unsupported there is **no upstream implementation to compare against** — no correctness oracle on
that part, and no performance baseline either. The mathematics, and the upstream running on a
device it does support (step 9), are what is left.

## 2. Recover the semantics from three sources

One source is not enough, and they fail differently:

| Source | What only it tells you |
|---|---|
| Documentation | The formula, the dtypes and the constraint list that bounds the input domain |
| **infershape** | The output shape — and whether it depends on input *values* rather than shapes |
| **Kernel body** | The arithmetic order, the accumulation width and the rounding mode |

Where they disagree the kernel is what shipped; record the disagreement rather than choosing
quietly. Two properties of the arithmetic are observable and neither is stated in any document:

- **The rounding mode.** AscendC `CAST_ROUND` breaks ties away from zero, `torch.Tensor.to`
  breaks them to even. `RoundMode.AWAY_FROM_ZERO` is the match;
  [precision](../references/precision.md) governs the rest of the boundary.
- **The reduction order.** fp32 addition is not associative, so a tree fold and a serial chain
  give different last bits, and any result sitting near a rounding boundary moves. Read the
  reduction out of the kernel body rather than assuming a sum.

Both cost about one ULP on a handful of elements, which is precisely what a tolerance wide
enough to pass would absorb. Aim the reference at bitwise agreement so that neither can hide.

Exit with a torch reference that never imports ascriptor and a `domain()` that raises on every
case the upstream does not accept.

<a id="domain"></a>
## 3. Declare the domain, then cover its corners

The upstream's constraint list is the domain you inherit, and `domain()` is where it becomes
executable. The cases then have to reach its **corners**, because a list that samples the middle
passes everything and still ships a kernel that raises on a shape the constraints admit. Three
kinds of corner go missing by default:

- **The extremes of each constraint**, especially the largest. That is where a tiling plan stops
  fitting, and finding it at delivery means re-deriving the dataflow rather than widening a
  number. Measured on one port: the largest admitted corner did not fit the first plan at all,
  and closing it took a second shape of group rather than a larger budget.
- **The code path a value takes, which is not the value.** Reaching a constraint's extreme is not
  the same as reaching the path that extreme exercises: on the same port the largest dimension
  routed to the *narrow* tiling arm, so the wide arm — the one every smaller shape used — had no
  case at all, and was wrong. After the tiling makes a choice, ask which arm each case took and
  whether any arm has none.
- **The edges of the work partition**: fewer work items than cores, a group that is not full, a
  tile whose live width is not a whole number of registers.
- **The values, not just the shapes**: an input that contributes no output at all, a ragged
  batch, the shortest sequence that still produces one row.

Take the shapes the upstream's own unit tests and examples use — they are free, and they are what
its author considered representative — then add the corners those miss. Each case says in one
line what it is the only case to exercise.

## 4. Separate the semantics from that part's implementation

Write the two-column table before any DSL. The test for each row is whether it survives a change
of chip. A streaming overlap state machine, a UB residency plan, a 32-byte block broadcast trick
and a per-core tiling table are implementation; the formula, the cast boundaries, the reduction
order and the output domain are semantics. **Do not transliterate.**

Permitted host work is bounded by what the upstream host already does. When its tiling function
reads an input's values to derive offsets, deriving the same offsets on the host is alignment
with the upstream rather than a shortcut — and it can keep a data-dependent quantity off the
device entirely. Anything past that boundary is a separate agreement and belongs in the
[authoring contract](../../templates/authoring-contract.md) with its reason.

## 5. Re-derive for the target

Four questions, answered before code and costed with [roofline](../references/roofline.md):

1. Which compute units. The upstream's choice reflects its part, not yours; a vector-only
   operator on a part whose cube sits idle deserves the comparison, and so does the reverse.
2. The core partition and the grid, against this card's actual counts.
3. How data-dependence is resolved — for PyPTO-Pro this is a hard constraint, step 7.
4. What the upstream's most complex machinery actually buys. Check it against the cases the
   upstream's own tests exercise: machinery that exists for a parameter regime those cases never
   enter is machinery you do not owe the port.

## 6. Probe the primitives before writing the kernel

A short script that exercises each unfamiliar mechanism once, at one tiny shape, through emit and
`sim`, costs minutes and moves every surprise ahead of the design instead of into a finished
kernel. Keep it; a failure later is diagnosed against it.

<a id="emit-gate"></a>
## 7. Emit before you build on the design

`compile_kernel(entry, backend=...)` takes seconds, needs no card and needs no vendor toolchain —
[the page's first rule](../references/pypto-pro.md#emit-first). A `PyptoGap` names its owner, and
the owner decides the next move: `ours` is a contract and is not to be worked around, `upstream`
is reported at the source span, `unmapped` is neither until someone checks.

Three parts of that contract reach a port before anything else does:

- **Tile allocation dimensions must be compile-time.** A runtime scalar may be an address or a
  loop count; it may not be an extent. Reading a geometry field out of a metadata tensor and
  staging that many rows is refused by design — [RFC 0013](../../../library/docs/rfc/0013-pypto-native-synchronization.md).
  A bound scalar parameter is not a runtime value: the backend specialises per valuation.
- **A partial transfer into a wider tile stacks two validshapes** and has no spelling. Carry the
  room a whole-register access needs in a **spare row** rather than in spare columns, and choose
  the tiling so every group is the same constant width.
- `break` and `continue` have no line in the printer; guard the loop body instead.

Then check the surface on all three backends, so a form that only one of them prints is known
now rather than at delivery.

**A clean emit is not a compile.** The vendor toolchain sees what the emitter wrote, and a form
that is legal but large fails there instead: straight-line code from an unrolled schedule spills
(measured once at 35 vector slots against a 6144-byte VF stack, where the same arithmetic as a
small device loop fit easily). Prefer a loop over a static container of compile-time parameters —
it keeps the strides constant without unrolling the work, and without the device division that
would pull in a supplement patch.

## 8. Walk the evidence ladder, keeping the stages apart

`sim`, then `pipesim`, then the card **on the delivered backend**, under
[hardware first](../runtime-and-maintenance.md#hardware-first) and the placement in
[where each step runs](../runtime-and-maintenance.md#where-each-step-runs). Emission, vendor
compilation and device execution are three claims, and
[the evidence table](../common-language.md#evidence) keeps them apart. Aim for bitwise agreement:
a port reproduces an operator, so a tolerance wide enough to hide a rounding mode has hidden one.

**The simulators do not model every hazard, and neither does the backend's synchronization.**
Two of these were only ever visible on a card:

- A UB store inside a `@vf` is not visible to a later load without
  `vf_barrier(VfPipe.STORE, VfPipe.LOAD)`. One kernel that staged values and read them back
  passed every case under `sim` and `pipesim` and returned every case wrong on the card.
- A software pipeline's prefetch, left running past the end of its loop, is **a write with no
  reader** — and the next pipeline's prologue is a second write to the same buffer. The
  generated locks were correct for every dependency that exists; a write-after-write with
  nothing between them is not one. On the card the stale transfer won, and the first row of
  every group after the first was computed from the previous group's data. **Guard the
  epilogue**: prefetch only while there is an iteration left to consume it.

When a card disagrees with both simulators, run the control backend before blaming the delivered
one. In the second case CCE failed the same way, and that is what said the defect was ours.

Run the same cases through a control backend on the same card. A PyPTO-Pro failure is a porting
defect only when CCE passes that case, and a single backend has hidden a silent wrong answer
before — but the control is diagnosis, not delivery. Finish with the
[delivery synchronization policy](../runtime-and-maintenance.md#sync-closeout) and report the mode actually requested and validated.

<a id="oracle"></a>
## 9. The upstream oracle, when a card for it exists

Optional, and worth one pass when the upstream supports a part you can reach: run the vendor
operator there and compare its real output against the reference from step 2. Check the installed
CANN first — when the operator is already in `libopapi.so` or bound by the vendor's torch
extension, there is nothing to build.

It turns "we read the sources correctly" from inference into evidence, and it is the only check
that reaches semantics no document states. Measured once: the reference matched on five of six
cases and disagreed on four of 9216 elements of the sixth, which was the reduction order of
step 2 — every stage below hardware had agreed, because the kernel carried the same mistake as
the reference it was compared against. **A disagreement here is a finding about your reading, not
a defect in the vendor**: reproduce the differing elements several ways and let the arithmetic say
which reading the vendor implements before changing anything.

**Only code travels.** Ship the reference and the seeds, generate the inputs and the vendor's
output on that box, compare there, and bring back the corrected reference and the numbers — never
input or output tensors. The reasons are the ones in
[running what you are writing](../references/development-execution.md): a seeded generator is not
byte-identical across machines, and the data exists only to be compared with itself.

<a id="fast-enough"></a>
## 10. Make it fast enough, and say what that was measured against

A port that reproduces the arithmetic and runs at a fraction of what the part can do is half a
port. [Optimize](optimize.md) owns the method; four things are specific to a port.

**There may be no baseline, so state what there is instead.** When the upstream does not support
your target there is no vendor number to beat. What remains is the memory-bound floor for the
bytes the kernel actually requests — [roofline](../references/roofline.md), and the library owns
the bandwidth assumption — plus the port's own before and after on one card. Report the floor as
a sensitivity scenario, never as a measured HBM efficiency.

**Measure with the profiler, not with the harness.** The authoring `OpExec` round trip was
measured at about 7.6 s per launch and *identical* for a 0.01 MB case and a 2.9 MB one: host wall
time says nothing about the kernel. `ASCRIPTOR_PROFILE=1 ASCRIPTOR_REPEAT=N` wraps the run and
`ascriptor.runtime.perf` reads device-side `Task Duration(us)`; the recipe is in
[the library's perf page](../../../library/docs/perf.md).

**Size the working set above L2, or you are measuring L2.** One A5 part's L2 is 112 MiB
(`l2_size` in its platform config) and ordinary demo shapes fit in it many times over. Measured:
with a 302 MB input the first profiled launch and the median of the rest agree to within 1%, so
the number is about HBM; with a 2.8 MB input the two cannot be told apart and the comparison is
against the wrong memory entirely. Report the first profiled launch separately from the later
ones — their ratio is what L2 is worth for this access pattern.

**Check the launch width before touching the code, and let the measurement choose the work.**
Measured on one port: the delivered `block_dim` was 8 on a card whose 28 blocks the kernel could
fill, and raising it was 3.48x for a launch parameter. Then the per-pipe ratios said what to do
next — with nothing above 0.53 the kernel was dependency-bound rather than pipe-bound, so the
next token's load never overlapped this one's arithmetic; double-buffering it was 1.92x and took
both pipes to 0.89–0.99. Meanwhile the re-read that an overlapped window costs, which looked like
the obvious target, was free: the wider shape requested 1.91x the bytes for 1.24x the time at
identical unique traffic, because the re-read rows were still in L2, so the scheme that would
have removed those bytes was never written. An optimization the measurement deletes from the
plan is worth as much as one it justifies — and **a pipelining change buys a new correctness
obligation**, so it goes back through step 8 rather than shipping on its speedup.

## 11. Land it

The four-file demo folder and its rules are [the kernels contract](../../../kernels/AGENTS.md);
[admission](../references/development-execution.md#admission) says when to write the other three
files. Provenance has no field in `metadata.json` — the upstream revision and the hand edits go in
a comment at the top of `kernel.py`, and the disposition table stays in the task's
`tmp/<task>/`. A backend refusal is recorded as the `# pypto_pro:` comment in that demo's
`main.py` and nowhere else.

Report the delivered backend, the cases and which corners of [the domain](#domain) they cover,
the stages of step 8 separately, and the measurement of [step 10](#fast-enough) with what it was
measured against. A limit you chose to keep belongs in `do_not_copy_when` **with the measurement
that made it a choice**; a limit you have not measured is an open item, not a documented one.
