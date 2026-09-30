# Diagnosing hardware disagreements: the board against the simulators

How to work a kernel that passes its independent generated reference on the interpreter
and is wrong, or faulting, on the board. Distilled from the M5 support-surface work (the 64-bit scatters,
the alignment sweeps, the mxfp8 online cast). The oracles and what each one is blind to, the tools,
the isolation ladder, the traps met so far, and the practicalities of the shared box. Repository
language applies (English); the box is only ever "the a5 shared box of `machine_specs.md`".
Synchronisation problems (hazards, deadlocks, autosync's decisions) have their own page,
[`diagnosing-sync.md`](diagnosing-sync.md).

## 0. The three oracles, and what each one cannot see

| oracle | executes | blind to |
|---|---|---|
| the reference interpreter (`--launcher sim`) | the IR, element-exactly, every lane in Python | everything below the IR: alignment, the 256-byte register window, the DMA's 32-byte row pitch, instruction timing, what the compiler's macros expand to |
| cannsim (`--launcher cannsim`) | the printed CCE on a functional model, instruction by instruction, in program order | timing inside a `@vf` (a load's result is always visible to the next instruction), over-reads past a tile (it reads the bytes and moves on), the AI core exception of an unaligned access (it performs the access) |
| the board (`--launcher board`) | the hardware for the selected case/toolchain/device | other cases, devices and software versions; output differences or execution errors still require diagnosis with probes |

What follows from the table: cannsim agreeing with the interpreter proves that the printed
instructions compute the right function *when executed one after the other*; only the board proves
the timing and the addressing. So cannsim is the cheap first filter (a printing bug shows there), the
board is the decider, and when the two disagree the cause is in cannsim's blind spots — that column
is the hypothesis list.

## 1. The tools

| tool | use |
|---|---|
| `python run.py check --case ID --device DEVICE --launcher board --output SCRATCH` in a complete unit, run on the box | generate inputs and an independent reference; compare using the unit contract, preserving failed jobs and their exact source/version identities |
| `tools/diag/probe.py PROBE.py --launcher board [--variant NAME …]` | a probe kernel — any file with `inputs()`, optionally `patch(files, variant)` and `report(name, outputs)` — run on the interpreter and on the launcher, outputs diffed per element; `patch` edits the printed artifact before the build (an instruction variant, a register dump) |
| the lints — `ascriptor check` / `compile`, the `[lint]` lines of every `OpExec` launch | the traps already known, before any run; also check the generated unit's initialization and output-poison contract |
| `<out_dir>/board_run.log`, `board_build.log`, `board_harness_build.log` | pulled back (redacted) after every board run: the AI core exception (`aclrtSynchronizeStream … 507035`) and the box toolchain's compile errors |
| `<out_dir>/project/build.log` (cannsim / aclnn) | bisheng's errors — `no matching function for call to 'vcvt'` is a form the c310 header does not have, a `static_assert` names an unsupported mode |
| `ascriptor compile k.py::name -o dir`, `ascriptor explain k.py::name --op N` | the artifact to read (every statement carries `// #<op id>`) and the op's source line |

`<out_dir>` is the `OpExec` directory under the unit's selected scratch output, or the
diagnostic probe's explicit `--out-dir`. Probe interpreter comparisons diagnose a model/device
disagreement; they do not replace the unit's independent acceptance reference. No recorded
tensor data or old source checkout is required by the successor workflow.

## 2. The workflow

### 2.1 Read the symptom

* **`error`, `507035` in `board_run.log`** — an AI core exception: a whole-register access that is
  not 32-byte aligned (D-051), an address outside the tile, a DMA descriptor the hardware refuses.
  The lints name the first kind; for the rest a probe with a dynamic offset parameter finds where
  it starts (the D-051 sweep: fp32 accesses at bytes 4, 8 and 16 fault, 0 and 32 run).
* **`diff` on cannsim and on the board alike** — a printing bug: the wrong intrinsic, operand order,
  a macro that expands into more stores than intended (D-050's two-register `vsts`). cannsim
  reproduces it; work there, one cannsim at a time (~4.5 GB each).
* **`diff` on the board only** — timing or addressing that cannsim does not model (§0). The board
  decides; cannsim can only confirm that a variant computes the right function.
* **zeros, or another tile's bytes, in an output** — a store window past the tile, a DMA landing
  past the rows (D-050, D-053), or bytes the kernel never writes. Generate nonzero inputs and
  compare poisoned versus explicitly seeded outputs; inspect every declared written region.
* **a rounding-level difference** (`ok-tol` with `--rtol 1e-4 --atol 1e-5`) is the expected verdict
  for fp32 cube accumulation, fp32 `cadd`, `vsqrt` / `vln` / `vdiv` and the complex64 division
  (RFC-0007 §6). It is a finding only when a kernel that was bit-exact stops being so.

### 2.2 Isolate: the ladder

Climb only as far as the symptom needs. Build/run time depends on the current toolchain and
case; record actual elapsed time. The lead schedules hardware and budgets any cannsim process
separately rather than assuming enough shared memory for concurrent jobs.

1. **Lints first.** `ascriptor compile` on the kernel; every warning is a candidate cause.
2. **Original against modified, same generated case, on the board.** Preserve the source,
   seed, parameters, reference and comparison rule for both artifacts. Run each in a separate
   output directory; do not overwrite the first failure while investigating the change.
3. **One stage at a time.** A probe kernel that runs the real `@vf` (import it from the kernel's
   module) with its inputs staged from GM and its UB outputs copied to GM: the vector stage of a
   cube kernel, one `@vf` of many. The interpreter's run of the same probe is the reference and
   `probe.py` diffs them; a stage that is right in isolation moves the suspicion to what
   surrounds it (the loop, the previous iteration's registers, the other side).
4. **Dump the registers.** In `patch(files, variant)`, insert a store of the register after the
   suspect instruction into a spare region of an output tile — `vsts((vector_u32&)r, (__ubuf__
   uint32_t*)(out + 512), 0, NORM_B32, pset_b32(PAT_ALL));` — guarded (`if (r == 0 && g == 1)`)
   so one iteration is captured, and size the output for it. The dumped lanes against what the
   interpreter would hold at that point say *which* instruction's result is wrong, not only which
   output.
5. **Vary the instruction.** Same hook, one change per variant, `--variant a --variant b`: the
   alternative form (`vlds` for `vsldb`), an idempotent op after it (`vor r, r, r`), a barrier, a
   mask. A variant that fixes the output is a hypothesis, not yet an explanation.
6. **Minimal reproduction.** A probe of a handful of `@vf`s that each hold one hypothesis about
   the suspect instruction — back-to-back, in a loop, followed by an ALU consumer, in the
   original's exact prologue — so one build answers all of them and the rule can be stated
   (`tools/diag/probes/vsldb_hazard.py`, §3).
7. **Sweep the parameter.** A `Var` for the offset / stride / count so one build covers the range
   that matters (the alignment sweep: `k` in bytes, the fault appears at the first unaligned
   value).
8. **Compare with the reference implementation.** How does AscendC (or the old framework's backend)
   emit the same op? Read its expansion in the CANN headers, then run its *literal* form as a variant
   next to the printed one — a pass there turns "the hardware does X" into "our printed form differs
   from theirs in Y", and Y is usually small (D-057: the register's own overload instead of the
   unsigned carrier cast).

The ladder does not only isolate. Read backwards, it attributes — which matters most when the
first differing stage sits inside your own kernel and "compiler blind spot" and "pattern to
avoid" look identical from the output. Three rungs are the controls that tell them apart:

- **Rung 3 separates an instruction from its surroundings.** A `@vf` that is right in isolation
  and wrong in place puts the fault in the composition — the loop, the previous iteration's
  registers, the other side — not in the printed instruction. No library change helps that one.
- **Rung 8 is the printer's control.** When the literal AscendC form of the same op passes where
  the printed one fails, "the hardware does X" has become "our printed form differs in Y", and
  Y belongs to the backend.
- **A second backend is the third control.** Run the same case through `cce` and through
  `pto_isa`. One failing while the other passes puts the fault in whichever printer is alone;
  both failing the same way while the model passes puts it below both, in the lowering or in a
  device behaviour the model does not carry.

A hardware-only mismatch with none of these three established is a hypothesis with no owner yet.
Name the control you ran; if you ran none, say the attribution is open rather than implying one.

### 2.3 Lock it in

A hardware finding is not done when the kernel passes. It is done when: the printer change has a
test that pins the printed text (`tests/backends/test_cce.py`); the trap is a lint when it can be
named statically (`ascriptor/ir/lint.py`, `tests/ir/test_lint.py`) or an interpreter
`HardwareWarning` when only the run-time address knows; generated positive/negative controls
retain the actual boundary; the decision is in `decisions.md` or a linked defect with the
evidence (which probe, which launcher, which bytes); `tools/cce_support.py` and the table in §4 carry
it; and the kernels that use the same instruction were re-run (`v8_allhif8` / `v8_p_path` for the
strided block stores after D-052).

## 3. Worked example: the `vsldb` hazard (D-052)

The mxfp8 online cast (`float_to_mxfp8_online_cast_matmul`) produced its second K group from the
first group's data on the board, bit-exact on cannsim.

1. *Symptom.* `diff` on the board only, in the fp8 output's second group of every row; the scale
   plane was right. So the vector stage, not the cube.
2. *Lints.* One: the packed scale read loaded a whole register from a 128-byte tile (fixed first —
   a masked block load of the four blocks that exist; it was not the cause).
3. *Original against modified.* The committed kernel and the lint-fixed one both wrong: not a
   regression of the fix.
4. *One stage.* A probe kernel running the real `@vf` with GM outputs: wrong on the board, right on
   the interpreter and cannsim — the vector loop itself.
5. *Register dump.* After the second iteration's block load of the source row: lanes 0..31 held the
   *first* group's values, lanes 32..63 the rest of the first load's 256-byte window; the scale
   register loaded right after it was correct. The load's result was not what its consumers saw.
6. *Variants.* An idempotent `vor(r, r, r)` right after the `vsldb`: right. `vlds(…, NORM)` +
   `vand` under the mask instead of `vsldb`: right. Both accepted as the printed form (D-052).
7. *Minimal reproduction.* See below.

### The minimal reproduction (D-055)

The five textbook shapes all passed on the board with bare `vsldb` — two back-to-back loads, a
loop, a loop pair, a load followed by an ALU consumer, the kernel's prologue with a one-point
store of a load register — so the additive approach stopped there and the subtractive one took
over: the real loop body with one piece removed per variant, six bodies per build, the board
deciding. Four builds:

| round | variants (bare `vsldb`) | verdicts |
|---|---|---|
| 1 | the real body; f32 store instead of the fp8 cast; `sub` instead of `div`; loads + `div` by a constant + cast + store; loads + `div` + store; the body without its tail | fail, fail, fail, **pass**, **pass**, fail — the cast, the divider and the tail are irrelevant |
| 2 | no `vand` + `vcmax` (the one-point store takes the u32 load itself); no staging round-trip (a float reduce feeds `div`); no `vmaxs` + `dup`; the two loads swapped | **pass**, **pass**, fail, fail — and the swapped order still loses the f32 register: not "the first of two loads" |
| 3 | the minimal candidate (loads, `vand`, `vcmax`, one-point store, barrier, broadcast load, `add`); the reduce replaced by three chained `vand`s; the reduce stored aside and the round-trip carrying the load's lane 0 | fail, fail, **pass** — not the reduce unit |
| 4 | only A bare (B as `vlds`); the pre-barrier store taking B's lane 0 with the ALU result stored after A's use; the ALU-sourced one-point store with no barrier and no broadcast load | fail, **pass**, fail — one `vsldb` is enough, the barrier round-trip is not needed |
| 5 | the six-instruction body with a `vf_barrier` at four places: `LOAD→STORE` before the store, `VEC_ALL→VEC_ALL` after the load, `VEC_ALL→VEC_ALL` before the store, `LOAD→STORE` after the load | fail, fail, fail, fail — not a missing barrier (D-056) |
| 6 | A loaded through AscendC's own `RegTensor<float>` + `Reg::DataCopy<DATA_BLOCK_COPY>` (`kernel_operator.h` included first) next to the printed form | **pass**, fail — AscendC's form is immune although it expands to one bare `vsldb` |
| 7 | `RegTensor` outside the loop + `DataCopy`; a plain `vector_f32` inside the loop with the carrier `vsldb`; `RegTensor<float>` with a bare `vsldb` | **pass**, fail, **pass** — neither the wrapper nor the declaration |
| 8 | the float overload into the plain `vector_f32` (unsigned and signed config); `RegTensor<uint32_t>` read back as f32; the carrier overload with an unsigned config | **pass**, **pass**, fail, fail — the reinterpret-cast around the intrinsic is the cause (D-057) |

The six-instruction reproduction: `vsldb A; vsldb B; vand; vcmax; vsts (one point, the reduce);
vadd A + 1; vsts` — from the second loop iteration on, A is the previous iteration's data. Lining
up every pass and fail, the one consistent predicate is the **last `vsts` before A's first
consumer**: an ALU result there (the reduce, a `vand` chain) fails; a load-produced register
there (B's lane 0, even right after the ALU-sourced store) passes; consuming A before any store
passes. Reading: that store retires the block load's pending write-back early and the consumer
stops waiting — in the core's scoreboard or in the compiler's schedule is open, since the
toolchain's `llvm-objdump` decodes none of the c310 vector ISA. No memory barrier closes it — `VLD_VST` or `VV_ALL` right after the load or right before the store
fail identically (D-056): `mem_bar` orders UB accesses, not a register's write-back. The cause turned out to be
ours: the printer issued block copies through the unsigned carrier, `vsldb((vector_u32&)a, (__ubuf__ uint32_t*)p,
…)` for a `vector_f32 a` — a reinterpret-cast of the register object around the intrinsic — and AscendC's
`LoadAlign<float, DATA_BLOCK_COPY>`, which is the same bare `vsldb` in the float overload, is immune (rounds 6–8,
D-057). Strided block copies now print in the register's own type with no guard; the contiguous `vlds` form
(D-052) and `vor A, A, A` after a carrier-typed load remain the fallbacks for dtypes without a native form. `tools/diag/probes/vsldb_hazard.py` keeps the reproduction
and its five bounding siblings as one build, with the expected verdicts in its docstring, so the
next toolchain is a single command to check.

## 4. Traps met so far

| symptom | cause | caught now by | decision |
|---|---|---|---|
| `507035` on a `ub[1] <<= reg` | a whole-register access at byte 4: the load / store units address 32-byte blocks | lint + interpreter `HardwareWarning` | D-051 |
| the next tile zeroed after a 64-bit store | the compiler header's two-register `vsts` issues a second 256-byte store | one 32-bit `vsts` of the interleaved halves | D-050 |
| garbage in the upper lanes; a neighbour's bytes in an output | an unmasked 256-byte register window past the tile | lint | D-051 |
| an extremum's index appearing in lane 1 | `vcmax` / `vcmin` leave it there on every width | the printer masks to lane 0 | D-050 |
| complex32 products 7.8e-3 off | fp16 arithmetic rounds twice; the goldens round once | c32 mul / div computed in fp32 | D-049 |
| bisheng `static_assert` on a merging f32 → f16 `vcvt` | the form is zeroing-only on c310 | `CceGap` with the fix (`c310.MERGE_REJECTED`) | D-051 |
| the second K group computed from the first's data | a block copy issued through the unsigned carrier (a reinterpret-cast of the register around `vsldb`) loses its value at the first use when the last `vsts` before it stores an ALU result (§3) | strided block copies in the register's own overload, `vlds` + `vand` at stride 1, `vor` only after a carrier-typed load; `tools/diag/probes/vsldb_hazard.py` re-checks in one build | D-052, D-055, D-057 |
| a `[16, 2]` uint8 tile back with one row and a neighbour's bytes | every DMA burst lands on a 32-byte-pitched row | `lint_lowered` — printed as `[lint]` on stderr before the run: read it, a probe's own staging tile fell into it twice | D-053 |
| an empty or unknown generated case selection | the wrong unit/case was requested; no kernel ran | the unit runner rejects missing cases and references | RFC-0012 |
| `no matching function for call to 'vcvt'` | a cast pair the c310 header has no overload for (u32 → f32 seen in a probe) | bisheng's `build.log` | open |
| a column-gather view back with ~70% wrong bytes, no fault | NDDMA rows land on the UB port's 32-byte blocks: a 48-byte destination row silently corrupts its neighbours | device_lower refuses statically ("pad the tile row"); the D-053 lint had flagged the row width before the run | D-085 |
| `507035` at launch from a kernel that only touches L1 via DMA | `mode="vec"` launches AIV-only, and the AIV side has no L1 path; every corpus kernel that stages through L1 is `mix` | nothing yet — a lint candidate (flag L1 tensors in a `vec`-mode kernel); met in the D-084 probes | D-084 |

Probe hygiene that cost runs: fp8 outputs must be `torch.float8_e4m3fn` tensors (`.view(torch.uint8)`
before `.numpy()`); an artifact regex must match the pointer form the printer uses today (`(__ubuf__
T*)name` and `(__ubuf__ T*)(name + (expr))` both occur); a probe's own staging tile obeys the same
rules as the kernel's (D-053 was found in a probe's `[16, 2]` output).

## 5. Practicalities on the shared box

* Budget cannsim separately (historical processes used about4.5 GB). The lead serializes board
  work after checking the selected device's health/process state and uses an advisory lock.
* The unit runner takes the box's CANN tree from the **local** environment rather than from
  `boards.json`, because the project is built on this machine: export `ASCEND_HOME_PATH`
  (or set the local board's `cann_path`) next to `ASCRIPTOR_BOARDS`, or the run fails with `BuildError: cann_path is not set and ASCEND_HOME_PATH is empty` — which reads like
  a broken box and is not one.
* Record local setup or launcher failures separately from numerical failures. Preserve
  the original job and do not compare an output that was never produced. Use bounded
  waits and report progress during a long device run.
* Everything below the assigned workspace, through the workspace lock (`runtime/board.py`); never
  another user's processes; never the safety override.
* Read `board_run.log` before comparing bytes: after a fault the output buffer is whatever the
  harness had, and a "difference" there means nothing.
* Machine details stay in `machine_specs.md` / `boards.json` — never in a tracked file, a commit
  message, a probe's name or a log excerpt pasted into a document (review before sharing).

## 6. Record what you found

The decision entry carries the evidence chain (symptom, the rung that isolated it, the variant that
fixed it, the minimal reproduction and the rule as far as it is known — say what is *not*
established); RFC-0007 §2 / §6 the printed form; `howto.md` "Read the warnings" the new warning;
`status.md` the gate numbers and the open item; this page's §4 the trap. A finding that changes what
the interpreter should model (D-053's row pitch) is an open item in `status.md`, not a silent change
to the reference.
