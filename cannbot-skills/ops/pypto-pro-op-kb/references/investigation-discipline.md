# Investigation discipline

Domain-neutral rules distilled from work that went wrong before it went right.
Each one below cost at least a full build–measure–revert cycle on a real
operator. They apply to any measurement-driven debugging or optimisation, not
only to kernels.

---

## 1. Never paraphrase the thing that judges you

If a harness, gate, or scorer decides whether your work is correct, **import and
run its code**; do not reimplement its rule from prose — not from documentation,
not from your own notes, not from a design document that describes it correctly.

A hand-written copy of a benchmark's accuracy rule reported **0/20 on kernels
that were partly fine**. It was wrong in four independent ways at once, and each
was individually plausible: a threshold off by the factor the real code applies,
a table row taken from the neighbouring dtype, a whole region of the rule
missing, and a reference input the real code supplies that the copy did not.

The failure mode is not carelessness. Prose descriptions of a rule are lossy,
and every gap is filled with a plausible guess that then reads as intentional.
Load the real thing and fail loudly if it is unavailable — a silent fallback to
a local approximation reintroduces exactly the bug.

**Corollary:** judge with the harness's own verdict, not a statistic derived
from it. A summary number can move non-monotonically against the thing it
summarises — one benchmark's headline error metric is a max dominated by the
smallest reference value, so a strictly *more* accurate implementation measured
a *worse* number.

## 2. Match the statistic to the question

A statistic answers the question it was built for, not the one you are asking.

A diagnostic reported a stage as "1.2× the reference" and it was believed twice.
The metric measured *relative* error over the tensor's **bulk** (deliberately
excluding near-zero elements, which is correct for describing a tensor). The
failures were *absolute* errors in the **tail**. The stage was in fact 2.8×
worse than the reference, and two cycles were spent rewriting an innocent stage.

Before drawing a conclusion from any aggregate, state which population it covers
and check that the failures live in that population.

## 3. A negative result is a statement about a configuration

An experiment showed a candidate fix changed nothing. That was recorded as
"refuted" — correctly, at the time. But it had been measured while a different,
upstream error was 1.5× larger and swamped the effect. Once that upstream stage
was fixed, the refuted candidate became the dominant remaining term.

> **Record the conditions a negative result was obtained under, and re-open it
> when whatever dominated it changes.** "Refuted" is not a permanent property.

Write the confounder into the note, not just the conclusion: *"no effect, but
measured while X was 1.5× larger"* is a note that ages correctly. *"Refuted"*
is not.

## 4. Cost a fix in the score's own units before building it

A change that trades one dimension for another can make the metric worse than
doing nothing. Compute both sides first, in whatever unit actually decides the
outcome.

Worked example: a fix that converts the last failing case of twenty gains
`(0.3 + 0.5·score)/N ≈ +1.7 points` and costs ~2× runtime across all twenty,
`≈ −5 points`. **Net −3.3.** Shipping 19/20 with the arithmetic recorded was
correct; chasing the case count would have lowered the score it was meant to
raise.

"It fixes the failure" is not sufficient justification. Neither is "it is
faster".

## 5. Back up before restructuring anything that passes

Every restructure of working code is a candidate revert. Copy the file first —
it costs one command.

Two of three restructures attempted on one operator were reverted after
measurement, and the backups are the only reason the passing state survived. Do
this even when — especially when — the change is well-reasoned; the ones that
get reverted are rarely the ones that looked doubtful.

## 6. Reason to size candidates, measure to choose among them

Error models, traffic arithmetic, and roofline estimates are worth doing: they
rank what to try and say when to stop. They are not evidence.

On one operator, several rounds of confident modelling ("~2e-3 relative, well
under threshold") were contradicted by the device every time it was asked. The
models that mattered were all decided by measurement; the two conclusions acted
on *without* one were both wrong. Symmetrically, when a measurement predicted a
2× gain and delivered 10%, the gap was itself informative — it said the traffic
was already cache-resident.

**Where the choice is a plan rather than an implementation, that measurement is
often free.** Replaying a dataflow in a host framework, graded by the real
comparator, chose between five precision schemes across three problem sizes on a
laptop with no accelerator. Only the implementation of the winner needed
hardware.

## 7. Report the number that is true, especially when it is worse

A scoring helper credited failing cases with a base term they had not earned,
inflating a result from 57.95 to 62.46. Finding and fixing that lowered the
headline number by 4.5 points — and is the reason the rest of the numbers in
that report can be trusted.

State plainly what was not achieved and why, separating "not done" from "not
possible". A target derived from a published lower bound may be arithmetically
unreachable, in which case it is a specification problem and no amount of
engineering will close it — say so once, with the check that demonstrates it,
rather than repeatedly attempting it.

## 8. Measure the bound before pushing on the target

Before spending another round on "make it faster", establish what the achievable
number *is*. Two techniques, in increasing order of cost:

**A sibling bound.** Time every case at a *simpler operator's measured time* on the
same shapes and tiling, and score that. It is an empirical zero-work reference that
already carries every real overhead a synthetic probe would have to guess at.
Measured: pricing one activation's entire compute body at zero — every case at a
6-operation sibling's measured time — gave a ceiling of 86.4 against a target of
80 and a current 71.0. Cheap, and more trustworthy than a probe you have to build,
wherever a sibling shares the tiling.

**A floor probe.** A kernel that moves exactly the operator's contractual bytes and
nothing else, with every data-movement lesson applied. It bounds from below.

Two rules about what these bounds mean:

- **A ceiling derived from the levers you enumerated is a statement about your
  list.** One operator produced two arithmetically sound impossibility arguments,
  at 75 and then 72, both built by summing the improvements that had been thought
  of. A floor probe measured the real bound at **88.66** — the "ceiling" was 17
  points low.
- **A floor probe does not price the primitives the semantics require.** The same
  88.66 was measured *without performing the operator's cross-lane scan*, and every
  cross-lane primitive on that part costs 15–20 ns. The floor was necessary, not
  sufficient. **Pair a floor probe with the cost of the operations the operator
  cannot avoid**, or it will promise headroom that no expressible dataflow reaches.

State which kind of bound you have. A sibling bound bounds *any* implementation; a
floor probe bounds data movement only; a bound assembled from a list bounds nothing.

## 9. A restatement drifts toward the conclusion you already hold

A number that is measured once gets *restated* many times — in a summary, a
commit message, a handoff, a later report. Each restatement is an opportunity for
it to become the number that best supports the conclusion already reached, and
that drift is invisible because every version traces back to a real measurement.

Measured: an operator sat at a per-case sum of 8.42, its target needed 12.00, and
a bound that priced its entire compute body at zero was 14.58. This was restated
as **"the target needs 82 % of the distance to a kernel that does no work."**
82 % is the fraction *of the bound* (12.00 / 14.58). The fraction of remaining
*distance* is (12.00 − 8.42) / (14.58 − 8.42) = **58 %**, and 82 % of the distance
would be a different, materially harder target. The phrasing had drifted to the
harder-sounding reading — the one that better justified stopping — and it survived
three restatements including into a published document.

The same class, seen twice more: a figure derived as `3 × overall − a − b` from
rounded printed means (arithmetically sound, inputs rounded, off by 1.2); and a
ratio republished as 14.6× that measurement later put at 5–7×, after it had
already been propagated to three other pieces of work.

**Practice.** Carry the *inputs* alongside the conclusion, not just the conclusion
— "8.42 now, 12.00 needed, 14.58 ceiling" cannot drift, while "82 %" can. Recompute
a derived figure at the point of restatement rather than copying the previous
sentence. Mark a derived number as derived, and re-measure it before spending on
it. Be most suspicious when a restatement makes your existing conclusion *easier*
to defend.

## 10. A finding that lives on one branch is not a finding

When several investigations run in parallel, each keeps its own copy of the shared
record, and each appends to it from wherever its copy happened to end. The result
is not one record with conflicts — it is *n* records that each look complete and
internally consistent, so nothing signals that they disagree.

Measured, across nine parallel operator investigations sharing one findings
document and one knowledge base: entries **#22–#30 named a different finding
depending on which copy you read**; two knowledge-base pages existed only on the
branch that wrote them and had never reached the shared branch at all; three more
were stale on the shared branch, one of them missing the measured primitive costs
that superseded its own advice. A summary claiming four delivered pages was
correct on no branch — three were reachable.

**Practice.**

- Merge the shared record on a cadence, not at the end. The cost of merging grows
  with divergence, and the cost of *not* merging is that later investigations
  re-derive what an earlier one already measured.
- **Do not resolve this class of drift by taking the longer file.** Verified
  counter-example: the shared copy held 27 lines that the longest branch copy
  lacked. Merge three-way against each file's actual merge base, and treat a
  branch that *extended* an entry as an amendment to that entry rather than as a
  new one.
- Where each investigation appends its own entry to a shared registry, the
  conflict is append-vs-append and the resolution is a **union**. Picking a side
  silently drops one participant's work, and no test will catch it — the artifact
  is simply built without it.
- A checker that verifies every page is *reachable* from the index earns its keep
  here: one page was written, never registered, and would have merged in as
  unreferenced.

**Serializing access to the one scarce resource.** Parallel investigations
share not just a record but a board, and three operational facts govern that:

- **`flock(1)` does not exist on macOS.** Take the same advisory lock from
  Python instead — `python3 -c` calling `fcntl.flock(fd, LOCK_EX)` acquires the
  identical lock `flock(1)` would, so a mixed-platform fleet still serializes
  correctly against one lockfile. A guard that silently no-ops on the
  controller's platform is worse than none: every agent believes it holds the
  lock.
- **Launch board-side work detached** — `nohup setsid` — so that a local tool
  timeout kills the local waiter and not the remote job. A foreground remote
  command couples the remote job's lifetime to a local timeout that knows
  nothing about it.
- **Never kill a board task that is already past its authentication prompt.**
  Killing the local side orphans the remote process, which keeps holding the
  NPU; recovering it here has required a manual reboot. Wait for it, or reattach
  to it — the detached launch above is what makes reattaching possible.

## 11. Before claiming a capability is absent, run the absence gate

"The framework cannot express X" is a much stronger claim than "I have not
found a composition that expresses X", and collapsing the second into the first
has a measured price: two separate efforts independently concluded a mask could
not be converted between element widths, designed their kernels around the
absence, and each carried avoidable cost for several rounds — the capability
existed, filed under a name neither search reached
([mask-width conversion](../constraints/vec-mask-width.md)).

Before promoting a candidate's failure into a capability claim, record all of:

1. **the API-surface search** — where was looked, and for which names. The mask
   converter was missed because every function with `mask` in its name was
   audited, correctly, and the converter is not named `mask`;
2. **installed-signature evidence** — the signature and docstring in the
   installed package, not the memory of one;
3. **the nearest composable primitive considered**, and why the composition
   fails;
4. **a minimal probe result** — the smallest kernel that would use it;
5. **generated-code or board evidence** when the gap may sit below the Python
   surface.

The evidence order for this repo's stack: `$PYPTO_DEVKIT_DIR/docs` first, then
the installed `pypto_pro` sources, then the generated kernel, then the board.

**Two traps that make step 1 return a confident, empty answer:**

- **`find` and Glob do not follow symlinks by default, and
  `$PYPTO_DEVKIT_DIR/pro_ops` is a symlink.** Without `-L` the official-sample
  sweep returns **zero** files, and "I searched every sample and found nothing"
  becomes an absence claim built on a search that examined nothing. This has
  actually happened here. Use `find -L`, and sanity-check the hit count against
  the sample manifest before drawing a conclusion from a miss. The same applies
  to skills, which are installed as symlinks — invoke known script paths
  directly rather than globbing for them.
- **A documentation dtype/capability table is not a whitelist.** `vf.astype`'s
  table lists four conversions and never mentions BF16, yet FP32↔BF16 and
  UINT16→UINT32 both work and both appear in an official sample. An absence
  claim resting on a doc table has not yet met requirement 2 — check the
  installed signature and the samples before promoting a table's silence into a
  capability claim. See
  [pypto-pro-dsl-limitations-a5.md](pypto-pro-dsl-limitations-a5.md) §Tier 3
  for the tables known to under-report.

## 12. Instance-dependent constants must be verified by exhaustion, not by sampling

A design whose constants depend on the case — per-instance tile geometry,
per-shape address maps, ownership boundaries computed from a runtime extent —
cannot be validated by a hand-built table of representative rows. One operator's
Stage 3 failed **four consecutive reviews** here, and all four were the same
failure: *the argument sampled one end of the parameter space and asserted the
whole of it*. A reviewer reading a plausible table cannot see which rows are
missing.

It converged only when the requirement changed from "show a table" to "show a
script". The script must:

1. **Call the shipped predicate**, not a copy of it transcribed into the test.
   A reimplementation validates the transcription, which is not the thing that
   ships. This is the same rule as §1.
2. **Enumerate the whole declared parameter space**, not a chosen subset —
   that is the entire point.
3. Assert, per instance, at least: **capacity** (does it fit), **row
   alignment**, and **pairwise address-range non-overlap**.
4. Carry a **negative control**: with one constraint removed, the script must
   *report* a violation. An assertion never observed to fail is not known to
   have teeth, and a green exhaustive sweep from a toothless script is more
   dangerous than no sweep, because it reads as proof.

**Why address-range non-overlap has to be listed separately.** The defect that
survived three rounds was found by that check alone. It was not a budget
overflow, so the capacity assertion passed it; the allocations fit, they simply
**overlapped**. Nothing in the DSL catches this — there is no allocator, and
alignment is validated while overlap is not, so it trips **no `static_assert`
and no runtime error** and corrupts silently
([framework-findings §10](pypto-pro-framework-findings.md)). Capacity and
overlap are independent properties; checking the first does not sample the
second.

Related, in a different register: **grade a test run by the number of
comparisons that ran**, not by its summary line. Expected verdict lines =
`Σ (modes per test function)`; a shortfall means something short-circuited, and
an all-pass summary counts functions that did not raise rather than
comparisons that executed. Count the verdict lines and compare against the
expected total before reading the summary as coverage.

If the gate is incomplete, report **"the current candidate has not found a
valid composition"** — a statement that invites the next search — not "the DSL
cannot do this", a statement that ends it. The distinction is load-bearing
downstream: a capability claim reroutes designs, files upstream asks, and (in
this repo's flow) triggers a `capability_gap` verdict that an independent
verifier must then spend a cycle re-litigating.

## 13. A green control proves nothing until it has been shown to go red

A probe reports "no mismatches". That is evidence only if the same run also
demonstrates that the checker *can* report mismatches. Three probes in one
session returned clean and none of them meant anything:

- A `-inf` regression case computed its first chunk from a 5 % rule that, at
  the planned chunk size, never covered the first chunk at all. The case was
  built to exercise one adjudication and exercised nothing.
- A stale-lane diagnostic used a shape with a single column block, so the tile
  was never reused and the lane it was written to catch could not go stale.
- A cross-core store-hazard probe derived its expectation and its skip-control
  from the *same* flag, so the two moved together. With the control engaged,
  kernel and expectation agreed trivially, and zero mismatches was the
  arithmetically correct answer.

The third is the general form, and the least visible: **coupling the oracle to
the thing under test**. A wrong shape is caught by reading the shape; a shared
flag is caught only by asking what the control would have to emit if the
implementation were broken, and then checking that it does.

### 13.0 The negative control has its own mechanism, and that mechanism can fail

One level below §13. A checker was validated by perturbing the source it guards
and confirming it went red. The perturbation appeared to pass — and the *test* was
broken, not the checker: the `sed` pattern matched 20-space indentation while the
code under test now sat at 24, so it edited nothing. A no-op perturbation produces
a green result that looks exactly like a checker correctly accepting valid input.

> **A negative control must prove it changed something before its result carries
> information.** Diff the perturbed artifact against the original, or assert the
> edit count, and only then read the verdict.

The same session produced two related failures in the same script, both of which
make a naive checker lie while looking healthy. Anchoring a block comparison on
the first occurrence of its opening line, when that line occurs four times, diffed
one section against a different one and printed 80 lines that read as catastrophic
deviation. And widening an alignment window to absorb length changes made the
*next* section's lines score as deletions — 16 phantom deletions, including
another module's preamble. Both were caught only because someone read the diff
instead of the exit code.

So state the expectation as what a correct implementation must produce,
unconditionally — never as a function of the switch being tested — and make the
control's required output a number written down in advance. `checker-is-live:
YES (6240 mismatches)` before the verdict line, and a verdict that is void
without it.

The same asymmetry applies to geometry. In one operator the boundary under test
sat at `(m_off + 32)·N·esize`; at `m_off = 0` that is always a multiple of the
beat, so a probe built there is *structurally* incapable of exposing the hazard.
A clean result at `m_off = 0` says the construction was invalid, not that the
path is safe. When a probe's whole purpose is to reach one narrow condition,
compute the condition and assert you have reached it before you interpret the
outcome — and when a superseded run's control turns out to have been vacuous,
mark that run inadmissible in the record rather than leaving a green line
behind for someone to cite.

### 13.1 The probe that fails for a reason you did not enumerate is the valuable one

A corollary from the same session, and the reason a failed probe is not a
wasted one. A three-round probe of a table-lookup mechanism failed every round,
each time on the author's own defect:

1. a missing `from __future__ import annotations`, so an annotation was
   evaluated eagerly — a pure host error that taught nothing;
2. a bare `range()` rejected by the parser — which surfaced the substantive
   finding that **no in-kernel loop can carry a constant index**, a constraint
   nobody had enumerated and which reshaped the construction;
3. `mul` refusing a bool operand — which eliminated the phi-free arithmetic
   alternative outright.

Only the first was waste. Rounds 2 and 3 each returned a real constraint, and
the one that mattered most arrived through a failure mode that was not on
anyone's list of things to test.

The practical consequence is about how a failed probe is reported. "The probe
failed, re-queuing" discards the finding; "the probe failed *because* X, which
we did not know" keeps it. Before dismissing a round as your own bug, ask
whether the bug was possible only because of a platform rule you had not
written down — and if so, that rule is the result, whatever the probe was
originally asking.

### 13.2 A buffer-lifetime fix is unproven until one core reuses the buffer

§13 says a probe built at `m_off = 0` is *structurally* incapable of exposing an
alignment hazard, so a clean result there says the construction was invalid. The
same argument has a second form, and it applies to every synchronization,
slot-lifetime, or buffer-rotation fix: **a shape where each active core handles
exactly one work item cannot exercise reuse at all**, so passing it is not
evidence the fix works.

The mechanism is the ring. A rotating slot family is only wrong once the ring
*wraps* — once a producer comes back around to a slot a consumer has not
finished with. A run with one work item per core never wraps, so an undersized
slot family, a premature free credit, and a correctly sized one all behave
identically. The bug is not merely unobserved; it is unreachable. That is how it
reaches the board.

**Compute the reuse factor and assert it before interpreting the result.**
Generally, `ceil(work_items / active_cores) > 1`. For the M-tiled attention shape
this rule was written against, that is

```text
ceil(BH * ceil(S1 / TILE_M) / core_count) > 1
```

and it is worth writing out because the arithmetic is easy to lose: raising `BH`
does nothing if `core_count` rose with it, and a "bigger" test case chosen by
total element count can easily have a *lower* reuse factor than the small one it
replaced.

**Why this is worth a rule of its own here.** This KB already holds two defects
in exactly the class the gate catches — `mutex_ids` capped at 32 simultaneous
slots, and a slot counter stepping by the wrong stride, which aliases two logical
buffers onto one physical buffer while the event machinery still issues two
credits ([framework-findings](pypto-pro-framework-findings.md), A5 probe
session). Neither is visible at one work item per core. And the composition law
in [../patterns/buffer-reuse-lifetime.md](../patterns/buffer-reuse-lifetime.md)
is only *checkable* on a wrapping shape.

So: when a fix is to a lifetime, a mutex, a rotation, or a cross-core handoff,
the regression shape is part of the fix. State the reuse factor next to the
verdict, the way §13 requires the control's mismatch count next to it. "It passes"
with a reuse factor of 1 is the same non-statement as "no mismatches" from a
checker never shown to go red.

One rule travels with it, in the same register: **the ownership checker's
warnings are correctness signals.** When the machinery that manages buffer ownership
for you emits a warning and the run passes anyway, you have a passing result with
an unresolved warning, which is not a passing result. Read it before shipping.

## 14. A gate identified by its path is not identified

A Stage 3 close-out reported `"status": "PASS"` on the module-contract
validator. Running the validator that worktree actually installs, against the
delivered files, unchanged:

```
FAIL:
  - rule2: module 2 input 'mm_i32' has invalid source 'phase_1'
  - rule3: final_output 'out' has invalid source 'phase_2'
```

Both reports were honest readings of a real execution. There were **two
different files with the same name**:

| realpath | `_MODULE_RE` | state |
|---|---|---|
| `<codex worktree>/cannbot-skills/.../validate_module_yaml.py` | `^module_(\d+)$` | uncommitted local change |
| `<main repo>/cannbot-skills/.../validate_module_yaml.py` | `^phase_(\d+)$` | committed |

The task supplied `CANNBOT_CONFIG_ROOT` pointing at the worktree, and
`.claude/skills/…` symlinks to the local copy — so the correct handle was in
hand. The author instead copied a hardcoded absolute path out of a CLAUDE.md
example, which is written against the main repo, and ran the other file.

Then the damage compounded in a way worth tracing, because **it starts with a
red gate, not a green one**. The wrong validator rejected a correct YAML. The
author took that failure at face value, concluded the design document's
statement "the validator accepts `module_N` and rejects `phase_N`" was
backwards, "corrected" the document, and rewrote the YAML to match. The original
statement had been right and the original YAML had been right. One read of the
wrong copy inverted a correct document, broke a passing artifact, and
manufactured a "root cause" that did not exist — all while every individual step
was a faithful report of something that really happened.

Trusting a measurement over a document is the right instinct and was the trap:
the measurement was of the wrong file.

Defenses, in order of value:

- **Resolve tools through the config root or the symlink, never by copying an
  absolute path out of boilerplate.** In a sibling worktree the same relative
  path is a different file. This project's CLAUDE.md warns that skills are
  installed as symlinks and forbids `Glob`/`find` for locating them; the same
  hazard reaches hardcoded example paths, which are written against one repo and
  read from another.
- **Make the gate print its own identity.** The fix adopted here has the runner
  emit `realpath`, `mtime` and the deciding source line (`_MODULE_RE = …`)
  before the verdict, so "which gate ran" is in the transcript rather than in
  the author's belief.
- **Prove the gate is live in the same run.** A negative control — flip the
  value the gate keys on and require it to go red — turns "it passed" into "it
  passed and it can fail." Here: `flipped to 'phase_' -> rc=1 detected=True`.
  This is §13 applied to gates rather than probes, and it converts a
  wrong-copy execution from silent into loud.
- **Paste raw output, not a verdict word.** "PASS" is a summary the author
  produces; captured stdout is one the tool produces, and it carries the
  identity lines above.
- **Never edit the gate to admit the artifact.** Check `git status` on the
  validator whenever a failing gate turns green — the cheap way to pass a gate
  is always available and always wrong.

A note on how this was caught, since it generalizes. The delivery it arrived in
was *unusually* rigorous — an exhaustive layout script with eight negative
controls, each demonstrated going red in the same execution. The discipline was
fully present in the author's mind and simply not attached to this particular
artifact. Rigor does not diffuse across the deliverables of one task; it has to
be applied to each gate individually, and a report that is scrupulous in one
section is not thereby trustworthy in another.

### 14.0 Hashing HEAD when the knowledge lives in the working tree

A third instance of the same wrong-copy error, in a form that produces a
*plausible* artifact rather than an error.

A `KB_SELECTION.json` recorded `constraints/precision.md` at the hash of its
**git HEAD blob**. The working tree differed, and the uncommitted delta was
precisely the section — "integer pre-bias must precede float widening" — that
the operator's own recorded invariant was derived from. So the selection
attested to a version of the page that *did not contain the knowledge the class
consumed*, and every downstream check that compares "the hash on record" to
"the hash of the file" would have to read the working tree to notice. A sibling
entry in the same file had been hashed from the working tree, so the
inconsistency was internal to one artifact.

The rule that would have caught it: **a content hash must be taken from the
bytes that were actually read.** If a reference is consulted by opening the file,
hash the file; `git show HEAD:<path>` is a different document whenever the tree
is dirty, and this repository is dirty by design in several worktrees.

Three instances now, in three forms — a gate resolved to the wrong copy (§14), a
document "corrected" against the wrong copy (§14, compounding), and a hash taken
from the wrong copy here. The common shape is that **a path is not an identity
when more than one tree is in play**, and none of the three announced itself as
a wrong-copy error at the point of failure.

### 14.2 Run the artifact you are shipping, once, before it costs anything

A delivery failed **every case with a compile error**. The post-mortem first
blamed a toolchain version gap between the build host and the board. That was wrong, and the truth is duller: **the submitted file had never
been executed anywhere.** It reproduced the failure on the first board run,
byte-for-byte, in seconds.

The gap opened in a sequence where every individual step was defensible:

1. a gate ran against the delivery file and passed 15 of 15 checks;
2. a genuine improvement was then made — one added test case, one path fix — and
   the delivery file was regenerated;
3. the new file was committed and submitted **without being run**, because the
   gate's PASS was still fresh in mind and the change was "only test code."

Step 2 invalidated step 1's evidence and nobody re-ran anything. The change
really was confined to test code, and the kernel really was byte-identical — both
of which were verified, and neither of which was the point. **The regenerated
file had never met a compiler.**

Two rules, and the second is the one that was missing:

- **A gate verdict names a hash.** When the artifact changes, the verdict is
  about the old hash and says nothing about the new one, however small the diff.
  Re-run or re-verify; do not reason about which parts of a file the change
  "couldn't have affected."
- **Before any irreversible spend, execute the exact artifact.** Not the staged
  source it came from, not the version the gate saw — the bytes being shipped.
  One run, one command. It costs seconds, against losing the whole delivery.

A note on how the wrong post-mortem happened, since it wasted a round. The
version difference between build host and board was real and independently
documented, so it was *available* as an explanation and it fit the shape of the
failure. Reaching for it meant the cheap decisive test — run the shipped file on
the board — went unrun for another cycle. **When a plausible systemic cause and
a trivial local check are both on the table, run the check first**; a systemic
story that survives the check is worth much more than one that pre-empted it.

### 14.1 An audit that runs every line finds what reading every line does not

The same author, asked to re-verify, replaced a prose walkthrough of the
design's evidence section with a script that resolved and executed all 18 rows
of it. That audit immediately caught two errors the author had read past more
than once: a cited document that does not exist under the cited name
(`vf.astype.md`; the real file is `…/type_conversion/astype.md`), and a
numeric bound off by 16,320 (`264,241,216` written for `127²×16384 =
264,257,536`).

Neither is the kind of error careful reading catches, because both *look*
right — a plausible filename, a number of the correct magnitude. Citations and
constants are exactly the content that should be checked mechanically, and
"0 documentary rows" is a stronger claim about a document than any amount of
proofreading.

One caveat from the same run: the first audit reported 31 of 41 references
unresolvable, and 30 of those were the resolver's own false negatives (short-form
names, KB-root-relative paths, a source root declared at the top of the
document). An audit's first red is as likely to indict the audit as the
artifact — repair the resolver, then re-confirm it still rejects a fabricated
path before believing the green.

## 15. The premise is the thing to test first, not the candidates within it

Three optimisation directions were dispatched on one kernel in one session. All three
premises were wrong, and each was refuted by a measurement that cost far less
than the work it prevented:

1. *"The deficit is prefill traffic."* A pipe-utilisation profile showed
   `aic_mac 0.08-0.18` and `aiv_time == kernel duration` — the Cube idle, the
   vector core always critical, GM traffic never binding. Two candidates modelled
   at 2.04x and 1.45x were aimed at an idle resource.
2. *"Band A's cost is per-KV-column iteration overhead, so pack columns."*
   Refuted from the **same profile that produced the attribution**, by tabulating
   ns per loop-iteration across a 35x range and finding it monotone — the
   signature of fixed cost.
3. *"`TQ_PREFILL=128` gives 2x."* It gives 1.45x, because the bound expression
   `kv_lim_i = S_kv - S + (i+1)*TQ` contains `TQ`, so a previously shipped lever
   had already consumed those iterations.

The pattern: each premise was inherited from a document (a DESIGN section, a
prior agent's report) that was internally rigorous but rested on an unmeasured
assumption about *what the binding resource is*. Rigour downstream of a wrong
premise produces confident wrong direction, and it is indistinguishable from
rigour downstream of a right one.

**Rule.** Before commissioning work against a model, spend one measurement on
the model's own load-bearing assumption. "Which resource is saturated" is
usually one profile away and it invalidates or confirms every candidate at once.
Prefer that to adjudicating candidates more carefully within the model.

**Corollary — a rejected candidate must be re-examined when the premise changes.**
`TQ_PREFILL=128` had been rejected for "zero traffic benefit". Once traffic was
measured not to be the constraint, that rejection carried no information, and the
candidate turned out to be both the cheapest to build (zero extra UB — the tile
groups were *already* declared at 64 lanes and narrowed at runtime) and the best
remaining. Rejections inherit the premise they were made under; re-derive them
rather than trusting the earlier verdict. The same re-examination found the
recorded capacity obstacle ("L0A over by 32,768 B") was arithmetic for a
*different* candidate's double-buffered tile, not this one's single slot.

## 16. Fixing an underflow is not fixing the empty case

A data-dependent loop bound was found by review to go negative
(`min(128, 0+64+128-512) = -320`) and was clamped at zero. The clamp was correct
and insufficient: the zero-iteration loop it produced **hung the AI Core**,
because a consumer outside the bound waited on a handshake its producer, inside
the bound, never performed. The pre-clamp code had been slow but correct — it ran
the full loop with every lane masked.

Two transferable pieces:

- **A range fix has three cases, not two.** Negative, empty, and non-empty.
  Clamping merges the first into the second; if the second is unsafe, the fix
  moves the defect rather than removing it. Enumerate all three explicitly.
- **Localize a hang by flooring one participant at a time.** Flooring only the
  suspected loop, while deliberately *leaving* zero bounds on the other three
  sites carrying the same expression, both proved which site owned the hang and
  refuted the competing "the primitive mishandles an empty range" theory in one
  run. A positive control — reverting the fix and confirming the hang returns
  with the same log signature — is what separates "I changed something and it
  stopped" from "I fixed the cause".

Related: a safety argument of the form "no in-contract input can reach this" must
be re-derived per tile size when the threshold contains the tile size. Here the
condition is `S >= S_kv + QTile`, and one contract case sits exactly on it,
protected only by carrying `is_causal=False`.
