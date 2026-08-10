# Measure and tune a correct kernel

Use this playbook only after correctness passes on the target platform. PyPTO-Gym owns
kernel correctness and reproducible profiling; packaging, scoring formulas and submission
contracts belong to whichever external driver consumes the kernel.

1. Record the device, architecture, CANN/PyPTO versions, kernel commit, shapes, dtypes,
   launch geometry and profiler command.
2. Establish a baseline for the same inputs and numerical contract.
3. Warm up both paths and collect repeated measurements under the same conditions.
4. Retain the profiler artifact and parser output used for every claim.
5. Change one implementation factor, re-run correctness and profile again.
6. Accept the change only when the selected target metric improves without violating the
   numerical contract.

Use
[`../../ops/pypto-pro-op-perf-tune/SKILL.md`](../../pypto-pro-op-perf-tune/SKILL.md)
for collection and analysis. Apply its target-specific roofline references only after target
detection confirms the matching architecture.

Do not treat a utilization threshold, externally reported score or result from another
shape/platform/version as a universal performance target.

## Traps that silently invalidate a measurement

Each of these can produce a plausible number rather than an error.

- **Stale profiler cache.** The evaluator archives each run under
  `reports/prof_data/<level>/<op>/<case>/` and locates `kernel_details.csv` by
  taking the first `os.listdir` entry rather than the newest — and the
  `trace_view.json` path is derived from that CSV's directory, so it inherits the
  same staleness. Left uncleaned, a score freezes at the kernel version that
  produced the surviving archive. **Current cann-bench clears it itself**:
  `perf_eval.py` calls `_clean_prof_dir_contents` before the first attempt and
  between retries, and the PyPTO-Pro child removes leftover timestamp
  subdirectories. Do not add a routine `rm -rf` for it. Keep the tell-tale
  instead — an overall score reproducing to ~1e-13 across independent runs is
  not something real hardware timing does; on a checkout predating that cleanup,
  clear the directory by hand.
- **Skipped rebuild.** The runner skips building when a wheel is already in
  `dist/`. Clear `dist/` and `build/` when the kernel changed.
- **Wall-clock fallback.** Running without performance collection times the host
  call, including dispatch and framework overhead, and is largely insensitive to
  kernel changes. Do not compare such numbers against profiled ones.
- **Host-side timing in the wrapper.** The benchmark times the wrapper, so every
  tensor operation there becomes a measured device kernel. Synchronising inside
  the wrapper additionally charges the whole stream's wait to the measurement.
- **A wrapper op that runs on the dev box can be absent on the eval runner.**
  The eval container ships its own CANN, and its operator inventory is not a
  superset of the dev box's: observed runner `950pr` = Ascend950PR_957c with
  CANN 9.1.0 (docker cake-ci CANN 9.0.0, torch_npu 2.10.0.post4) against a dev
  box on CANN 9.2.0. A submission whose wrapper called `.to(torch.float32)`
  scored 7/20 there: **every** fp16/bf16 case raised `RuntimeError: call
  aclnnInplaceCopy failed, error code is 561103` with `Config_Error(EZ1013):
  ... aclnnInplaceCopy_1_CastAiCore cannot be found` at the wrapper's cast
  line, while every fp32 case — which never took that path — passed
  (job_cb400c1718cb, 2026-08-06; full per-case JSON archived beside the op).
  The failure signature is easy to misread as a broken kernel or a polluted
  runner; the tell is that the failing set partitions exactly by which cases
  execute the host-side op. **Treat every aclnn-dispatching host op in the
  wrapper as correctness-critical, not merely as measured time** — the
  boundary in `constraints/wrapper-boundary.md` is also the compatibility
  boundary, because a kernel-only submission depends on nothing from the
  runner's op inventory. Version skew cuts the other way too: constructs the
  dev toolchain accepts may not exist for the eval toolchain, so keep the
  submission's dependency surface (host ops *and* DSL constructs) to what the
  eval environment is known to provide.
- **A dispatcher that enumerates public signatures fails the hidden set, and the
  error reads like a kernel bug.** A `transpose` submission routed on an exact
  `(rank, dtype)` table built from the 20 public cases: public scored 20/20 at
  72.40, and the hidden set failed **20 of 80** — every failure the same
  `no transpose class for signature [[3, 'float32']]`-style `ValueError` from
  the dispatcher, never reaching a kernel. The hidden set legitimately spans
  ranks and dtypes the public cases never show (here ranks 2-8 x fp16/bf16/fp32/
  int8/int16/int32/int64). The fix cost **ten lines**: route an unmatched
  signature to the generic class instead of raising, since each class's kernel
  already generalised — hidden went to **80/80** and public *rose* to 73.33
  (job_e813540ee6f2, 2026-08-07). Two rules follow: **an unmatched input must
  fall through to a working path, never to an exception** (the same
  no-host-guard rule as `constraints/wrapper-boundary.md`, applied to
  dispatch), and **a public-only score cannot detect this class of defect at
  all** — the dispatch table is exactly as complete as the case list it was
  written from. Test dispatch against the *declared* domain, not the observed
  cases.
- **The local copy of a submission is not what scored — download the archive.**
  A resubmission built from the on-disk package directory scored **0/20** with
  `jit() got unexpected keyword argument(s): timeout` on every case, while the
  *same* operator's earlier submission had scored 70.73. Downloading the
  archive that actually scored (`/api/jobs/<id>/submission/download`) settled
  it in one grep: the shipped file had **no** `timeout=` kwarg; the local copy
  had acquired one after that submission, from another session. Everything
  derived from the local copy inherited a kwarg the evaluator's pypto rejects,
  and it failed at kernel-compile entry before any case ran. Two rules:
  **diff against the downloaded archive, not the local directory, before
  claiming "kernel byte-identical to what scored"**, and note that a *sibling*
  kwarg may be fine — `compile_timeout=` is accepted on the same fleet, so
  "it's just a timeout hint" is not a safe assumption about which spelling
  survives. Also drop `dist/` from the archive: a prebuilt wheel can be
  installed in place of the source you just edited.
- **A uniform slowdown across every case is not automatically the runner — check
  the launch width.** A resubmission measured **1.900x median / 1.997x max**
  slower than the prior one across 77 shared cases, and the first reading
  (same runner, ten hours apart) invited a contention story. Re-measuring on a
  *second* runner killed that story outright: the two runners agreed to
  **median 1.000, p90 1.007**. The slowdown was in the code — the local copy
  had `block_dim = min(get_platform_info().core_num, row_blocks)` where the
  archive that scored used `vector_core_num`, so the kernel launched on 28 of
  56 vector cores. The ratio distribution was the tell it was not
  environmental: **bimodal at 1.02 and 1.99** (cases whose `row_blocks` fell
  below 28 were unaffected), where contention is flat. Procedure: before
  attributing a uniform slowdown to a runner, (a) re-measure on a different
  runner — agreement to ~1% indicts the code, and (b) read the *shape* of the
  per-case ratios, not just the median. Corrects an earlier version of this
  entry that recorded the contention reading as fact.
  See the `vector_core_num` entry in
  [../references/pypto-pro-framework-findings.md](../references/pypto-pro-framework-findings.md).
- **Board contention.** A byte-identical package has been measured 12.9–15.9x
  slower at a busy time of day. **Ship a control variant in every comparison
  run**; if the control moves by tens of percent, the run is noise and must be
  repeated rather than interpreted.
- **A degraded device, disguised by a healthy-looking reference.** Devices on a
  shared board degrade independently, and the vendor op being compared against
  is not a health check: on one contended device a vector-only A5 kernel ran
  **15x** slower while the CANN built-in beside it moved under 2% — the
  built-in was not competing for the same vector cores. When a number
  collapses, sweep every visible `/dev/davinci*` before believing any
  per-kernel conclusion. (Measured under another DSL on the same silicon; the
  trap is board-level, not DSL-level.)
- **Timing the wrong device.** `torch_npu.npu.synchronize()` and
  `torch_npu.npu.Event` act on the **current** device, not on the device the
  tensors live on. A harness with a `--device` flag must call
  `torch_npu.npu.set_device(index)` before timing, or it silently reports
  launch overhead instead of kernel time.
- **The `trace_view` metric strategy zeroes an eager-mode pypto_pro
  submission.** `--perf-metric-strategy trace_view` takes its time from
  tilefwk/PYPTO-named trace events and nothing else, with no fallback
  (`TraceViewStrategy` / `parse_tilefwk_metrics` in
  `src/kernel_eval/base/perf_strategy.py`: the name filter accepts only
  `tilefwk`/`PYPTO`). Eager-mode pypto_pro launches appear as plain `AI_CORE`
  events, so the strategy finds nothing, the run is flagged as suspected CPU
  fallback (anti-cheat), and the operator scores **0 with accuracy passing** —
  while `kernel_details.csv` from the same run holds the real device rows.
  Measured 2026-08-06, identical submissions with only the strategy changed:
  `foreach_addcdiv_scalar` 0.00 vs 121.67, `apply_rotary_pos_emb` 0.00 vs
  80.78. **Choose the strategy from the implementation type**: tilefwk-event-
  emitting implementations (PYPTO graph mode) need `trace_view`; eager-mode
  pypto_pro needs the framework default (`kernel_details`). The older
  "trace_view is mandatory for PyPTO" guidance applies to the former only.
- **`trace_view` is *structurally* unreachable on the isolated-subprocess
  path** — a second, independent mechanism producing the same zero.
  `pypto_pro_child.py` hardcodes `profiler_level` to `ProfilerLevel.Level1`,
  while `trace_view` requires **Level2** (`bench_registry.py:64`), and
  `--profiler-level` is plumbed only into `perf_eval.py` and
  `mc2_distributed_runner.py` — **it never reaches the isolated child**. So the
  artifact `trace_view` needs is not merely unmatched, it is never produced.
  Symptoms: `elapsed_us = 0.0`, `score_error_code:
  'no_npu_kernel_detected'`, composite score zeroed by anti-cheat, on a kernel
  that ran correctly. No flag combination fixes this from the CLI. *Inference,
  flagged as such:* the evaluation server runs the same unpatched child, so it
  should behave identically — that part is read from shared source, not
  measured server-side, and is worth reporting upstream rather than designing
  around.
- **A `pl.jit` kwarg the target runtime does not accept kills the submission
  with no attributable error.** The evaluation server's pypto_pro raised
  `TypeError: jit() got unexpected keyword argument(s): timeout` on a module
  our own box compiles fine. Since the submission imports its kernel module at
  package import, the exception fired inside the evaluator's import and
  surfaced only as `staged_rc_1_missing_report` — a successful wheel build
  followed by nothing, reproducible on three different runners and easy to
  misread as pool flakiness. **Grep the shipped module for jit kwargs before
  packaging**, and make the forwarder import the kernel module *lazily*
  (inside the entry function): it converts an opaque infra-stage death into a
  per-case traceback you can read. Removing the kwarg took the operator from a
  hard server failure to 82.51. (foreach_addcdiv_scalar, 2026-08-07.)
- **Provenance.** Confirm the profiler output actually contains the submitted
  kernel's rows, and that vendor-library rows total zero. A result produced by a
  fallback path is not a result.
- **Stale binary under an unchanged kernel name.** pypto keys its build directory
  on the kernel's `co_name`, so an edited body compiled under the same name can be
  served the previous binary. The run then reproduces **byte-identically**, which
  reads as "my change did nothing" and is indistinguishable from it. Name kernels
  by a hash of their own source, and make sure the hash covers **what the compiler
  sees** — a generator that hashed only the rendered `@pl.jit` wrapper left the
  guard inert for exactly the edits it existed to catch, because the wrapper
  template interpolates the vector function's *name*, not its source. Hash the
  header, the called body and the wrapper together.
- **A per-case-scored benchmark is not optimized by a mean.** An optimisation
  measured at **−2.23% mean over the 13 cases it touched, with no case slower**,
  scored **−0.22 on the server** (69.6082 → 69.3886, same archive, same runner,
  and server readings reproduce to ~0.01 so that is signal). The mean was real;
  it was the wrong statistic. `OperatorScore` sums a *per-case* term, and the
  four smallest cases regressed 4.5–11% while the large ones gained 1–2% — the
  losses cost −0.095 of summed score against +0.017 of wins. Convert every
  candidate result into **per-case score deltas weighted the way the grader
  weights them** before deciding to ship; a percentage on wall time, however
  carefully controlled, does not answer the question being scored.
- **On a degraded board the *sign* flips first for fixed-overhead-dominated
  cases.** The same change measured −0.79%, −2.35%, −0.62%, −2.83% on four small
  cases locally and **+11.1%, +11.1%, +9.3%, +4.5%** on the server. Those four
  are the ones where a fixed per-launch cost dominates, and a board that is slow
  on fixed-overhead-heavy paths inflates the variable part they are being
  compared against. A same-session ratio with a null control is a real defence
  for the *large* cases and buys much less than it appears to for the small
  ones — so when a change trades a fixed cost against a variable one, get the
  small cases measured somewhere trustworthy before shipping.
- **A board can degrade by multiples between days, and the drift check will not
  tell you.** Measured 2026-08-07: byte-identical shipped code, cached binaries,
  the same script, the same metadata and the same card read OperatorScore
  **70.02 one day and 61.21 the next** — mean SOL 0.400 → 0.224, every one of
  twenty cases slower, the DMA-bound strided cases worst (9.79x, 8.71x, 5.83x)
  against about 2x on the contiguous ones. A descriptor probe on that second day
  read 6.81 where the recorded value was 0.488, a **14x** miss.
  **The before/after control drift check passed on both days** — 19.2% and 10.6%,
  both inside the 20% band — because it compares a control to itself *within*
  the run and therefore cannot see a floor that moved between runs. Two defences
  actually work, and they are cheap: (a) re-run one **recorded absolute** (a
  probe whose ns/element you have written down) and refuse to interpret anything
  until it reproduces; (b) shape the experiment as a **same-session contrast**
  with an untouched arm, so the arm you did not change reads the noise floor for
  you. On the day above, (b) still licensed a 2.2% A/B result — the changed arm
  moved −0.62…−3.27% on all 13 cases while the untouched arm sat within ±0.1% on
  6 of 7 — while (a) correctly voided an absolute pricing decision.
  A within-run *ratio* is not automatically safe either: the same pair of
  descriptor variants was 3.03x apart on the good day and 14.8x apart on the bad
  one.
- **A partial profiler export under-reports deterministically.** Where a helper
  divides surviving rows by an expected step count, a dropped launch scales the
  answer by `rows/expected` — two rows of five read as 40 % of the true time.
  **This survives every defence built for noise**: it reproduces to ~1 %, so a
  second pass confirms it, and the control variant passes because the control was
  not affected. It once produced a clean, control-checked 3.67× for a change worth
  1.59×. Only an invariant catches it: **print the profiler row count next to every
  timing, and have something compare it to what was expected.** A printed
  invariant nobody checks is not a check.
- **A device fault reported as a total accuracy failure.** A wedged accelerator can
  let a run complete and report `TOTAL 0/N passed` — no timeout, no hang, a clean
  verdict that reads as a catastrophic regression and invites a revert. The tell is
  that **no comparison ran**: failures arrive as `RuntimeError` from device
  synchronise/copy entry points rather than as an error metric against a threshold.
  See the discrimination procedure in `pypto-pro-environment-check`. **Before acting
  on a total failure, require positive evidence that the comparison executed.**
- **A green run the public data cannot fail.** An optimisation that removes an
  operation can validate 20/20 and still be wrong, when the public cases cannot
  reach the inputs the removed operation was handling. One case generator lays
  special values in contiguous blocks of `numel//20`; where that block size is even
  and the reduction is 2 wide, no public row ever mixes `+inf` with a finite
  element, so a collapse that breaks exactly that pairing passes everything. **When
  an optimisation deletes an operation, ask what it was doing for the special
  values, then check whether the public data can even reach that case.**

- **`PYTHONPATH=src` *replaces* the overlay instead of extending it, and a
  correct kernel becomes 20/20 compile failures.** The documented invocation
  `env PYTHONPATH=src python -m kernel_eval.cli …` discards whatever
  `env_setup.sh` put on the path, so the eval child imports the **older**
  `pypto_pro` from the conda environment rather than the overlay. That older
  codegen passes a `MaskReg` predicate into an int32 `vand`, and every case
  dies with `error: no matching function for call to 'vand'` /
  `no known conversion from 'pto::MaskReg' (aka 'vector_bool') to
  'vector_s32'`. The failure reads as a broken kernel — it is a resolution
  problem, and the kernel is fine. **Always append, with absolute paths and the
  overlay first:**

  ```bash
  PYTHONPATH="$OVERLAY:$BENCH/src" python -m kernel_eval.cli eval ...
  ```

  and **verify what actually resolved** before trusting any verdict:

  ```bash
  python -c "import pypto_pro; print(pypto_pro.__file__)"
  ```

  This is a specific instance of the whole-run-fails-at-once signature: when
  every case fails identically at *compile*, suspect resolution before code.
- **`--source-dir` force-reinstalls into the shared environment.** It runs
  `pip install --force-reinstall --no-deps`
  (`package_manager.py:474-508`), which **evicts the previously installed
  operator's submission**. Evaluating two operators alternately in one conda
  environment therefore measures whichever was installed last, with no warning.
  Re-install before each evaluation, or give each operator its own environment.
- **Bare-metal single-card device binding is broken, and the workaround needs
  its own proof.** `evaluator.py:1102` writes the **physical** chip id into
  `ASCEND_RT_VISIBLE_DEVICES` *and* passes the same integer to `set_device()`.
  But `ASCEND_RT_VISIBLE_DEVICES` names an absolute physical chip while
  renumbering the visible set from 0 — the two agree **only at 0**. So
  `--device-id 4` raises `RuntimeError 107001`, and passing 0 to dodge that is
  overwritten back to physical npu0. It never shows on k8s, where the device
  plugin has already renumbered the allocated card to physical 0. Workaround:
  keep an inherited value that is already a single chip
  (`ASCEND_RT_VISIBLE_DEVICES=4` with `--device-id 0`) — and because the
  binding cannot be trusted, **prove in the log which physical card ran the
  work**, via an HBM fingerprint or the `npu-smi` process table. A workaround
  for a device-selection bug that does not verify device selection is not a
  workaround.
- **Score the run, not the summary line.** A suite of 20 test functions in
  which some functions sweep several modes must print **more than 20** accuracy
  verdicts. The expected count is `Σ (modes per function)`, and any shortfall
  means a function short-circuited — returned early, skipped a mode, reused a
  previous result. **The "20/20 PASS" summary is exactly what hides this**: it
  counts functions that did not raise, not comparisons that ran. Compute the
  expected line count before the run and check the actual against it, the same
  way the profiler-row-count invariant above catches a partial export.

## Choosing the metric

Prefer the profiler's device-side end-to-end figure over host wall-clock; on a
representative case the two have differed by roughly a factor of two on the same
kernel, and only the former reflects what the kernel did. On CANN Bench the
metric *strategy* is itself implementation-conditional — see the `trace_view`
trap above before passing `--perf-metric-strategy`.

Report kernel time and wrapper/device-library time separately so an apparent kernel
improvement cannot hide a slower public callable.

**`kernel_details` is a device measurement, not a host-overhead fallback.**
Some older evaluation guidance describes the default `kernel_details` caliber as host wall-clock including
Python dispatch and framework overhead, and mandates `trace_view` as the only
true device figure. **Neither half of that holds on this checkout**, and no
page in this KB has ever claimed it — the stale text is in the skill only. `KernelDetailsStrategy` (`perf_strategy.py:590-608`) takes
the **median** of the device kernel `Duration` column in `kernel_details.csv` —
a device-side measurement carrying no host or framework time. Verified on a case
whose CSV held five rows: their median equalled the scoring `elapsed_us`
exactly, and the file contained only that kernel's rows with zero non-kernel
rows. Meanwhile `TraceViewStrategy` (`:724`) carries a
`PendingDeprecationWarning` stating that `KernelDetailsStrategy`'s metadata will
subsume `aicore_e2e` and that it — the default — should be used instead.

Two consequences. The `kernel_details` number does **not** need a
host-overhead discount applied to it. And the "`trace_view` or the score is
meaningless" framing is obsolete on this checkout; combined with the
structural unreachability of `trace_view` on the isolated-subprocess path
above, `kernel_details` is frequently the *only* caliber available.

What does not change: **state the caliber in the report, every time.** Two
numbers from different strategies are not comparable, the default has already
shifted once, and a figure whose caliber is unrecorded cannot be re-checked
later.

## BenchSite server operations

Operational facts about the remote submission service, recorded 2026-08.
They are service-side and can change without notice; when an interaction
contradicts one, believe the service and update this list.

- **Names differ by direction.** A submission's `selected_operators` takes
  snake_case function names (`swi_glu`) while `get_benchmark_task` takes
  CamelCase operator names (`SwiGlu`). Neither errors on the other's
  spelling; the mismatch surfaces as an empty selection.
- **`infra_error` jobs are auto-refunded** — the job record's charges/refunds
  fields show it. Do not re-budget an infrastructure failure as spent quota.
- **Quota is per-token and daily**: base 10/day plus an earnable bonus,
  resetting at 16:00Z. Plan sweeps against the reset time, not the calendar
  day.
- **At most 3 active jobs per user** (`active_job_cap`); a fourth submission
  is rejected, not queued.
- **A runner claim can lag the queue by over an hour under pool congestion.**
  A job sitting unclaimed is not stuck; check pool load before resubmitting —
  a resubmission spends quota and joins the same queue.
- **Runners are not interchangeable for timing.** `multi-0` and `multi-2` have
  produced consistent numbers; `multi-1` has produced uniform ~15x slowdowns
  (every case, same factor — the fingerprint of a contended or degraded box;
  see the degraded-device trap above), and `simen-0` runs ~10% conservative.
  Compare timings only within one trusted runner, and record which runner
  produced every number.
