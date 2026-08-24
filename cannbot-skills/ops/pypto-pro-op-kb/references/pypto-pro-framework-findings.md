# PyPTO-Pro framework findings

Limitations and defects met while building operators in the `pl` tile DSL.
Kept here so the next author does not re-derive them, and so they can be
reported upstream.

Each entry states what was observed, how it was established, and what it costs.
Entries marked **blocking** stop work outright; the rest are worked around.

---

## Toolchain

### 1. `unknown type name '__aicore__'` means the wrong CANN is on PATH, not a broken toolchain

**Observed.** Every `@pl.jit` compile ends with
`error: unknown type name '__aicore__'` in the generated `kernel.cpp`, followed
by `candidate function not viable: call to [host] function from __global__
[aicore] function`.

**Cause.** The CANN on PATH has no A5 support.
`bisheng -xcce --cce-aicore-arch=dav-c310-vec -dM -E -` then defines `__CCE__`
and `__CCE_ARCH__ 100` — a fallback — but neither `__CCE_AICORE__` nor
`__aicore__`. The mechanism: `include/pto/pto-inst.hpp` pulls in the header
defining `__aicore__` only under `#if defined(__CPU_SIM)`, and gates the arch
macros on `__CPU_SIM || __CCE_AICORE__ || __COSTMODEL`, so on the device path
`__aicore__` has to come from the compiler. An A5-capable CANN defines it
(count 3 in the same macro dump).

**How I got this wrong, recorded because the reasoning was seductive.** A host
had several CANN installs. The default login environment selected one without A5
support, so kernels failed. I then "confirmed a regression" by recompiling a
source that had built days earlier and watching it fail — but I recompiled it
with the *same wrong toolchain*, so the test could only agree with me. A
directory mtime that happened to fall near the last successful build completed
the story. The A5-capable install was one directory deeper than my
`find -maxdepth 3` reached.

**Diagnostics, in this order:**

1. **Has anyone built successfully since?** Search the users' home trees for
   `call_kernel.so` newer than the suspect time (`find ... -newermt`). A single
   hit disproves a toolchain regression outright, and costs one command.
2. **Enumerate every install** before concluding anything — list every Ascend
   toolkit directory on the machine, not just the one on `PATH` — and do not
   trust a shell profile or a symlink named `cann` to point at the right one.
   Search without a depth limit; an extra directory level is enough to hide the
   working install.
3. Only then dump the predefined macros to confirm which install is A5-capable.

**Cost when misdiagnosed.** A day of work planned around a blocker that did not
exist.

### 1f. Do not store to a UB element inside a vector function and read it back

A vector function that writes a UB element in one pass and reloads it in a later
pass **compiles, runs, and produces garbage**. Measured on a quantizing
operator: 250,165 of 262,144 int8 elements wrong, `max|diff| = 255` — wraparound,
i.e. the value reaching the narrowing step was nonsense rather than imprecise.

The shipped softmax sample never does this, and the shape of it is worth
copying: it writes to a **separate output tile** and **recomputes** the
normalised value in its final pass instead of reloading what an earlier pass
stored. Restructuring to that pattern — every element written once, never read
back inside the same invocation — fixed the operator outright with no other
change.

There is a `vf.mem_bar` primitive, which suggests the ordering can be forced
explicitly; the recompute is cheaper than finding out, since the reload is
saving arithmetic that the vector unit is not short of.

Related: **every producing `vf` call must be its own assignment**. The parser
rejects nesting by name — *"vf.reduce_sum produces a result and must use the
assignment form"* — and the samples obey it throughout, which is easy to read as
style rather than a rule.

### 1g. Give every kernel variant a distinct function name

The JIT's build directory is keyed on the **kernel function name**. Two files
that define a kernel with the same name share that directory, so one variant can
silently execute the other's compiled binary — and nothing in the output says
so.

This produced a false result that survived being written up. An A/B of two very
different implementations of the same operator reported them within 3% of each
other, apparently refuting the design. They shared a kernel name, one source had
been copied over the other, and the comparison was in fact a kernel against
itself. With distinct names and a fresh build the real gap was **1.75-1.98x**.

Three cheap checks, in the order they would have caught it:

1. **Suspect agreement.** Two implementations that reorganise all global traffic
   landing within 3% is evidence to investigate, not to accept.
2. **Sanity-check derived rates against peak.** The reported latency implied the
   slower variant was moving its nominal bytes at above theoretical HBM
   bandwidth. An impossible rate means the number does not belong to the kernel
   you think you measured.
3. **Diff the sources actually under test**, not the ones you believe are there.
   One command.

A control that only exercises the harness — re-measuring the same variant twice
— cannot catch this. The control has to distinguish the variants.

### 1c. `precision_compare` is stricter on integers than the accuracy contract

The dev-side helper compares integer outputs for **exact equality**, while the
contract applies the operator's `proto.yaml` tolerance — commonly `int8: 1`,
i.e. every element within ±1. A kernel can therefore fail the local check and
pass the real gate.

Measured on a quantizing operator: 1281 of 262144 int8 elements differed, **all
by exactly 1**, `max|diff| = 1`. Local check FAIL, contract rule PASS.

So when the local check reports an integer mismatch, get the distribution before
changing anything — `max|diff|` and the off-by-one count. Chasing exact equality
that the contract never required costs real time, and the two-step
`fp32 → fp16 → int8` convert (necessary, see below) inherently produces
off-by-one drift near rounding boundaries.

### 1d. `fp32 → int8` is not a single convert

The platform's `vconv` table carries `f162s8` but no `f322s8`. Asking for the
direct conversion fails at build time with
`no matching function for call to 'castData'`. Route through fp16, which is
exact over the int8 range so the only cost is the rounding drift above.

### 1e. Reduction tiles must span at least one 32-byte block

A `[2, 1]` fp32 reduction tile is 8 bytes, and the backend instantiates a tile
type with **no** `SetValidShape`, `GetValidRow` or `GetValidCol` members — the
failure surfaces as `no member named 'SetValidShape' in 'pto::Tile<...>'`, which
does not obviously point at the size. `[16, 1]` fp32 is 64 bytes and works;
that is the geometry the shipped layernorm sample uses.

Relatedly, the wide tiles in an address map satisfy the 32-byte alignment rule
incidentally, so it is the *small* reduction tiles that break it when packed back
to back. Round every offset up explicitly.

### 1b. What the device-free gate does and does not catch

Measured while bringing up a first kernel against the broken toolchain above.
Driving `_ensure_compiled` with CPU tensors caught, in order and at build time:

- a UB address that was not 32-byte aligned, reported with the offending
  address — the wide tiles align incidentally, but a `[rows, 1]` fp32 reduction
  tile is 8 bytes, so packing two back to back misaligns the second;
- a plain Python `for` over a tuple of tiles, rejected with *"For loop must use
  pl.range()"*.

After those two fixes the same kernel produced a complete 199-line CCE source
whose emitted tile operations matched the intended dataflow one for one
(`TLOAD`x3, `TCVT`x5, `TADD`, `TMUL`, `TROWSUM`, `TSQRT`, `TROWEXPANDDIV`x2,
`TCOLEXPANDMUL`, `TABS`, `TROWMAX`, `TMAXS`x2, `TMINS`, `TSTORE`x3). So the gate
verifies structure as well as syntax: grep the emitted ops and check the
multiplicities against the design, because a source file existing is not the
same as the kernel body having been translated.

What it cannot check: anything numerical, anything about synchronization at
runtime, and anything about performance.

### 2. Codegen is device-free, and that is worth exploiting

Not a defect — a capability that is not documented and changes the iteration
loop. `__call__` splits into `_ensure_compiled(args)` followed by the launch, so:

```python
args = kernel._normalize_launch_args((cpu_tensor, ...), {})
kernel._ensure_compiled(args)          # parse -> IR -> CCE source -> compiler
```

drives the entire pipeline with **CPU** tensors. Parse errors, API misuse, tile
and address mistakes and IR failures all surface without a device and without
taking the device lock. Use it as the fast gate before every on-board run.

---

## Correctness hazards

### 3. The multi-class dispatcher cannot match under keyword invocation — blocking for multi-class operators

**Observed.** `ValueError: no <op> class for signature []` on every case of any
operator with more than one class.

**Established.** The caller invokes the candidate as `func(**params)` — keyword
arguments only, with keys taken from `signature(golden_func)`. The generated
PyPTO-Pro dispatcher builds its lookup key from positional `args` alone, so the
key is empty and matches nothing. Reproduced directly by reimplementing the
matcher: it returns the right class for `stock(x)` and raises for `stock(x=x)`.
The non-Pro dispatcher in the sibling generator is correct — it falls back to
keyword values and prefix-matches.

**Cost.** The operator fails on routing, before any kernel runs.

**Workaround.** Emit a dispatcher that binds positional *and* keyword arguments
against the golden's parameter order before keying, and that skips absent-or-None
tensors so an omitted optional argument and an explicit `None` land in the same
class. Both forms occur in the same operator when optional tensors are involved.

### 4. Unregistered op names pass silently to the backend and are invisible to the pipeline scanner

**Observed.** Names not in the Python op registry are emitted verbatim as
`block.<name>`, so convenience spellings such as `row_sum` work at runtime.

**Established.** The call parser falls through to a default handler that emits
the attribute name unchanged. The pipeline transform's cross-core scanner,
however, looks the op up in a role table keyed on the **registered** spelling and
`return`s on a miss — with no warning and no diagnostic.

**Cost.** Two failure modes. A typo becomes a C++ backend error at codegen time
rather than a Python `AttributeError` at parse time. And under `pipeline=`, a
reduction written with the convenience spelling receives **no automatic
cross-core synchronization** while the same operation written with the
registered spelling does.

**Rule.** Always use the registered spelling — `sum(..., dim=)`,
`maximum(..., dim=)`, `expand_*`. It costs nothing and the failure is invisible.

### 5. `load` is destination-first, `store` is tensor-first

`load(dst_tile, src_tensor, offsets)` against
`store(dst_tensor, src_tile, offsets)`. The asymmetry reliably produces
argument-order bugs that compile cleanly.

### 6. A missing platform marker silently skips the test

Test files without the SOC marker default to a non-A5 platform list, so the test
**skips** on A5 rather than failing. A green run proves nothing until the marker
is confirmed present.

### 7. Stale API in the runtime docstring

The `runtime` package docstring demonstrates a value-returning `load` and a
kernel that returns its result. Both forms are rejected by the current parser.
Do not learn style from it.

---

## Expressiveness

### 8. No per-token dynamic quantization primitive

`quant` / `dequant` are declared and registered but have **no usage in any
shipped example test**, so their scale conventions are undemonstrated. Every
covered path is fixpipe: a single scalar for a whole tile, or a per-channel
scaling tile. Per-token must be composed from abs, row-reduce, clamp, broadcast
and convert. See [../patterns/vec-per-token-dynamic-quant.md](../patterns/vec-per-token-dynamic-quant.md).

### 9. Reduction workspaces are full source size

A tile-op reduction requires a `tmp` of the **full source shape**, same dtype, in
vector memory. A chain with two reductions therefore carries several full-width
buffers, which at a few thousand columns consumes the entire budget for a single
row — no batching, no double buffering. The register-level reduce primitives have
no such cost, so any performance-sensitive vector kernel has to use them.

### 10. No allocator, and overlap is unchecked

Every on-chip address is a hand-computed byte constant and every rotating buffer
needs manually assigned non-colliding mutex identifiers. Alignment is validated;
**overlap is not**. End the address map with an explicit budget assertion, and
give each rotating tile family its own slot counter — a shared counter that
steps by the wrong stride can alias two logical slots onto one physical buffer
while the event machinery still issues two credits, producing corruption that
looks like a precision bug.

### 11. Kernels cannot return values

All outputs are pre-allocated tensor parameters. A multi-output operator becomes
a multi-parameter kernel; plan the signature accordingly.

### 12. Narrow-float scalar parameters are rejected

fp16 and bf16 scalar kernel parameters raise. Pass an epsilon or scale as fp32
or an integer, or materialize it in-kernel as a constant.

### 13. `load_tile` rejects a `tile_dims` keyword

Met while moving column-wise access off `load`/`store`; the fix was to use
`load_tile`/`store_tile` with tile-index offsets. Unclear whether this is a
documentation gap or a missing feature — worth confirming upstream.

---

### 16. `auto_mutex` keys on `mutex_id`, not on address overlap — silent wrong results

The dual-view idiom is required by the DSL: a `[R, 1]` reduction carrier must be
declared `layout=pl.DN` for the `dim=0` broadcasts, and elementwise ops on a
ColMajor `[R, 1]` tile fail a static assertion, so the same address must also be
exposed as a row-major `[1, R]` tile. Two `make_tile_group` calls at one address
therefore carry two different `mutex_ids`.

`auto_mutex` tracks dependencies per `mutex_id`. It does not know the two groups
alias, so any dependency that crosses the views is invisible to it.

Whether that matters depends on which pipes the two accesses sit on:

* **Write and read both on the vector pipe** — safe. Instructions issue in
  program order on one pipe, so no barrier is needed and none is missed. This is
  why the reduction carriers work: they are written by a `dim=0` reduction and
  read by a broadcast, both vector ops.
* **DMA writes one view, a vector op reads the other** — silently wrong. The
  MTE2→V barrier is never emitted, and the broadcast reads whatever was at the
  address before the transfer landed.

Measured on a fused dequant/activation/quant kernel: loading the per-token
`activation_scale` into the row-major alias and broadcasting through the DN view
corrupted about an eighth of the output elements on the largest case and
produced scale errors of 1e7 relative. It is a *latent* failure — it only
appears once a block runs more than one tile, so every case whose token count
fit in a single tile per core passed, and a test set built from small shapes
would have shipped it. Only a handful of the contract cases exposed it, and
those were the largest.

**Rule.** Never DMA into an address that is read through a different tile-group
view. Land the transfer in its own tile with its own `mutex_id`, then copy it
across with a vector op:

```python
pl.load(t_asl, activation_scale, [0, row_off])   # own address, own mutex
pl.mul(t_asm, t_asl, 1.0)                        # vector write to the alias
pl.expand_mul(t_af, t_af, t_as, dim=0)           # vector read of the DN view
```

The copy costs one vector instruction per tile and puts the dependency back
where the framework can see it — in program order on a single pipe.

**Detection.** Aliased groups are declarable but never checked. A build-time
warning whenever two `make_tile_group` calls overlap in address range and carry
different `mutex_ids` would have caught this at codegen instead of in a
20-case accuracy sweep.

### 17. The generated cross-core sync infers the pipe from the *consuming* op

pypto-pro can generate the whole cube/vector handoff instead of making the
author hand-write it: declare the shared buffers outside both sections with
`fwd_ids=`/`bwd_ids=`, tag the leaf functions `@pl.pipeline.stage`, and pass
`pipeline=pl.pipeline.PipelineConfig(preload=N)` to `pl.jit`. All three are
required — with `fwd_ids` but no stage decorator and no `PipelineConfig`, **no
sync is emitted at all** and the consumer silently reads the buffer before the
producer has written it.

The scanner derives each event's pipe from the op that touches the buffer. That
works when the consumer touches it with a vector op, and fails when the consumer
touches it with a DMA. A probe whose vector stage consumed a cross-core buffer
by `pl.store`-ing it straight to GM (an MTE3 op) produced, **inside
`section_cube()`**:

```python
pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=...)
```

which reaches the backend as `set_intra_block(PIPE_V, ...)` and is rejected:

```
error: the ranges of 1st parameter must be [0, 0], [2, 5], [10, 10]
```

The cube core has no vector pipe. `PIPE_V` is legal in the vector section and
illegal in the cube section, so the same enum passes one half of the compile and
fails the other.

**This is narrower than it first appeared.** The auto path emits the correct
`PIPE_FIX` for the reference kernel *and* for a full staged kernel whose vector
stage consumes the buffer with `pl.mul`. Only the DMA-consumer probe broke. So
the rule is: **have the consuming stage touch a cross-core buffer with a vector
op first**. If a stage only wants to forward the buffer to GM, copy it into a
local tile with a vector op and store that.

**Where the auto path is still the right choice.** Hand-writing the same ten
events — following the reference's discipline exactly, verified against the
reference's own generated C++ — deadlocked on device (`aicore timeout`), and
bisecting down to a single buffer with one forward/backward pair still
deadlocked. The generated form is the one to reach for; the hand-written form is
a last resort.

**Related gotcha, same symptom class.** A tile declared `valid_shape=[-1, -1]`
takes its extent from `set_validshape` at runtime. Both sides of a cross-core
buffer need one: setting it only on the consumer leaves the producer's
`pl.move` with an undefined destination extent, which faults as an
aicore/aivec error carrying MTE error info rather than raising anything the DSL
can catch.

### 18. Close the accumulator: the last matmul of a K-loop needs `AccPhase.Final`

A K-loop that walks the contraction in blocks accumulates into one L0C tile:

```python
for kb in pl.range(0, n_blk):
    ...
    if kb == 0:
        pl.matmul(ac, la, rb, phase=pl.AccPhase.Partial)
    else:
        pl.matmul_acc(ac, ac, la, rb, phase=pl.AccPhase.Partial)
pl.store(out, ac, [0, 0], phase=pl.STPhase.Final)
```

**That is wrong, and it faults the device.** `AccPhase.Partial` leaves the
accumulator open; the *last* block of the chain must be `AccPhase.Final`. With
`Partial` throughout, even a **single-block** loop dies with
`device error type 0xFFFF` — so the symptom is not proportional to the loop
depth and does not look like an accumulation bug at all.

**Depth itself is not a constraint.** Measured on a cube-only kernel, two-slot
groups throughout, checked against torch:

| slots | blocks | K | max relative |
|---|---|---|---|
| 2 | 1 | 128 | 7.2e-05 |
| 2 | 2 | 256 | 1.9e-04 |
| 2 | 4 | 512 | 2.4e-04 |
| 2 | 8 | 1024 | 2.1e-04 |
| 2 | 16 | 2048 | 2.2e-04 |
| 2 | **56** | **7168** | 8.7e-04 |
| 4 | 56 | 7168 | 8.7e-04 |

So **two slots suffice at any depth** — `auto_mutex` does order slot reuse
against the matmul still reading it, and a 56-block contraction (which a
7168-wide hidden extent requires) is fine. Extra slots buy nothing for
correctness.

**This finding previously said the opposite** — that slots must be >= blocks,
inferred from a bisect where 1 and 2 blocks passed and 4 failed. That bisect ran
on a kernel with other defects and the conclusion did not survive isolation.
Recorded here because "slots >= blocks" would rule out every large contraction
in this operator set, and it is not true.

### 19. The reference kernel's tile constants are not freely resizable

The reference flash-attention kernel looks parameterised — `TS`, `TKV`, `TD` are
module constants — but only two of the three are free. `TS_HALF = TS // 2` is
used as the pointer stride through the softmax:

```python
FLOAT_REP_SIZE = 64  # elements per fp32 register
src_ub1 = input_tile + TS_HALF
src_ub2 = input_tile + TS_HALF * 2
preg_136 = vf.update_mask(128, dtype=pl.DT_FP16)   # 128 fp16 lanes = 64 fp32
```

so `TS_HALF` *is* the register width and `TS = 128` is fixed by hardware. `TKV`
is a loop count and `TD` enters only as `D_LOOPS = TD // FLOAT_REP_SIZE`; both
are free. Changing `TS` produces a `ParserSyntaxError`, not a wrong answer — but
only because of the literal `128` in `update_mask`. The pointer arithmetic would
have gone quietly wrong.

**Scope — this binds `Vf` code only.** The constraint is on the reference's
register pointer arithmetic, not on vector tiles generally: tile-op vector
sections address tiles by shape, not by register, and both shipped operators in
this tree run 16-row tiles without trouble. So "M/2 must be a whole register" is
a rule for kernels that reuse the reference's `Vf` softmax, and NOT a reason to
redesign a tile-op kernel's tiling.

**Rule.** If you plan to reuse the reference's `Vf` functions, you inherit
`TS = 128`; budget the rest of the tiling around that. If you write the vector
section with tile ops, `TS` is free.

### 20. The pipeline transform enforces a stage contract — read it before writing stages

`runtime/pipeline/_analyzer.py` validates the kernel before generating any
cross-core sync, and its rules are stricter than the examples suggest. Violations
raise a clear `ValueError` at trace time, so this is cheap knowledge to have up
front:

* **A stage is either the producer or the consumer of a cross-core buffer, never
  both.** Reading and writing the same shared buffer in one stage is rejected:
  *"cross-core buffer 'X' is both read and written within a single stage (roles W
  and R)"*. Even an identity like `pl.mul(slot, slot, 1.0)` trips it. Copy the
  buffer into a local tile if the consumer needs to modify it.
* **At most one producer stage and one consumer stage per buffer** (C7).
* **The stage chain must strictly alternate cube/vector** (C5) — *"consecutive
  same-core stages ... the delay model requires the stage chain to strictly
  alternate"*.
* **Each stage call must sit inside its own `with pl.section_cube()/
  section_vector()` block**, one stage per block; a bare stage call in the loop
  body, two calls in one block, or an unsupported `with` wrapper are all errors.
* **The pipeline loop must iterate `pl.range(start, end[, step])`** so the end
  bound can be extracted for the generated is-valid guard.
* **Stage arguments that depend on a loop variable must be plain scalar
  arithmetic** (`ki + 1`, `qi * TS`). The transform reconstructs them for its
  prologue and drain iterations.
* Event ids are finite: address-overlap reverse syncs can exhaust them
  (*"not enough free event ids"*), which is a reason to keep cross-core buffer
  count and overlaps low.

Read this file before writing a mixed kernel. It is the actual specification of
what the pipeline accepts, and it is more informative than any example.

## Workflow

### 14. There is no performance stage

The shipped flow ends at correctness; performance tuning is listed as
"consult as needed" with no stage, no gate and no artifact. An operator can
therefore complete the whole workflow and be slow. Supply the loop explicitly:
profile, identify the dominant pipe, change one factor, re-verify correctness,
keep only on measured improvement, and stop only at a wall proven with data.

### 15. The roofline reference shipped with no constants

The performance reference states an evidence gate it did not itself satisfy,
leaving an architect to reason about a roofline with no numbers available. The
right fix is not to paste constants — the platform constraints page explicitly
forbids that, since transcribed numbers detach from the installed version — but
to supply a tool that derives them from the installed platform file, plus the
knowledge that no platform file contains: which SKU applies, how a comparison
anchor relates to the hardware roofline, and what fraction of peak is actually
reachable.

## Precision planning against an explicit accuracy contract

### 21. Never paraphrase the configured comparator

The accuracy gate is not "relative error below the dtype threshold". It is a
three-region rule with a CPU baseline, and a hand-written approximation of it
sent this operator's development down a blind alley for a full cycle: a
reimplemented checker reported **0/20** on kernels whose real verdict was mixed,
and it did so by being wrong in four independent ways at once.

Measured against the configured comparator (identical on the A5 box and in the
local checkout; the box's copy simply predates the normal-region relaxation and
is therefore the stricter of the two):

| what | the real rule | what a plausible paraphrase gets wrong |
|---|---|---|
| element flagged | `rel > 10 * threshold` | flagging at `threshold` — 10x too strict |
| bf16 small-value boundary | `2**-8` | `2**-11`, which is the **fp16** row of the table |
| cancel region | `abs(out) < 2**-3` and `2**-8 <= abs(golden) < 2**-3`, judged against CPU | absent entirely |
| reference | golden **truncated to the output dtype**, and a `native_output` = golden re-run at input precision on CPU | comparing against the untruncated fp64 golden with no CPU baseline |

There is also a first stage that passes outright on `MERE < threshold and
MARE < 10 * threshold`, before any region analysis runs.

Two consequences worth stating separately. `MARE` is a max over every valid
element, so on any output with near-zero elements it is dominated by the single
smallest `abs(golden)` and moves non-monotonically with real accuracy — a *more*
accurate scheme measured a *worse* MARE here. Judge with `passed`, not `MARE`.
And the non-normal regions are graded as a ratio against CPU, so "worse than the
threshold" and "worse than CPU" are different questions; only the second one
fails a case.

**Rule:** the verifier must use the comparator explicitly configured for the task
and record its source/version. If it is unavailable, fail loudly. A silent fallback
to a local approximation reintroduces exactly the bug it is meant to prevent.

### 22. A bf16 GM intermediate is not a free choice in a multi-matmul chain

Consider a golden that upcasts every input to fp32, carries the whole chain in
fp32, and rounds only its final outputs. A kernel that instead rounds each GM
intermediate to bf16 — the obvious tiling, and the one commonly reached for
first — fails every case on the outputs fed by the longest accumulation
chain.

The mechanism is not accumulated relative error, which stays near `2e-3` and
would pass. It is that `query` is a 128-term signed sum, so its elements are
Gaussian about zero: each bf16 rounding in the chain contributes ~0.1 of
**absolute** error, and the elements whose true value happens to land in
`[2**-8, 2**-3]` then carry a relative error of 4 or more. Worse, 0.1 also
pushes `abs(output)` past the cancel-zero threshold, so those elements do not
even qualify for the cancel region's CPU comparison — they land in the normal
region, which must have zero exceedances. About 77 elements per case, every
case.

Priced on CPU against the real comparator, before writing any kernel:

| scheme | M=1 | M=128 | M=512 |
|---|---|---|---|
| bf16 GM, one bf16 matmul term | FAIL | FAIL | FAIL |
| fp32 GM, one term | FAIL | FAIL | FAIL |
| fp32 GM, two-term split | PASS | FAIL | FAIL |
| **fp32 GM, three-term split** | **PASS** | **PASS** | **PASS** |
| fp32 GM, fp32 matmul | PASS | PASS | PASS |

Reading it: fp32 GM alone takes `k_rope` to zero error, because `k_rope`'s only
problem was the bf16 rounding of the projection feeding it. `query` needs more,
because its left operand reaches the cube as a *computed* value and the cube
takes bf16 operands — so the operand itself must be split into
`t1 + t2 + t3`, each the bf16 rounding of what the previous terms could not
represent, all three accumulated into one fp32 L0C against one resident weight
tile. Two terms leaves ~16 mantissa bits and fails from M=128 up; three reaches
~24 and passes everywhere.

Prefer the three-term split over the cube's native fp32 mode: 3 bf16 passes is
~126 TFLOPS effective on A5 against 94.6 for fp32, and it reuses the bf16 tile
path instead of needing fp32 L0A/L0B tiles at half the K depth.

Split only what needs it. A weight that arrived as bf16 is already exact, so its
residual terms are identically zero — splitting it is pure waste, and a "3-term"
variant that splits the *right* operand is a no-op that measures identical to
2-term. Likewise the first two projections take `token_x` directly, which is
also exactly bf16: they need one term, and only their **output** store had to
widen to fp32.

**The cheap way to find this:** the whole table above was produced on a laptop
with no NPU, by replaying the kernel's dataflow in torch with `store` and
`terms` as parameters and checking with the configured comparator. A precision
plan is settleable offline; only the kernel that implements it needs hardware.

### 23. Vector-kernel throughput is set by DMA granularity and task striding

Three measurements from one operator, all found by reading a per-kernel profile
rather than by reasoning about the code:

**A tile that is too narrow wastes most of the bus.** `mlap_copy_kernel` moved
67 MB at **121 GB/s** against a roof near 1 TB/s, purely because `TCOL = 512`
made each transfer a 1 KB DMA. Widening the column tile to 4096 (an 8 KB
transfer) was the entire fix. Pick the column tile from the transfer size you
want, not from a round number.

**Striding the outer loop over rows starves every core but one.** The same
kernel ran `for row in pl.range(core_id, n_rows, num_cores)` with an inner loop
over column tiles. At `M = 1` -- the decode shape, and eight of twenty cases --
that is one row, so one core did all the work and 31 idled. Striding over
`(row, column_tile)` pairs instead (`t // n_ct`, `t % n_ct`) fixes it with no
change to the body. Whenever the outer dimension can be small, fold the inner
dimension into the task index.

**One task per output row can be dominated entirely by descriptor count.** RoPE
over `query_rope` ran `M * N = 65536` tasks, each issuing eight 128-byte DMAs:
**506 us**, 33% of the case. Batching 64 token rows into one `[TRR, HALF]` tile
per task -- same arithmetic, same total bytes, one strided transfer instead of
64 -- took it to **46 us**. The vf needs only to become a loop over registers;
row boundaries do not matter to an elementwise pass, because each row's operands
sit at the same offset in every tile.

The corresponding cube-side lever is the N tile. At `TN = 64` each row of a B
tile is a 128-byte run of a 24576-wide weight, and the split matmul ran at
~1 TB/s; `TN = 128` doubled the run and took it to **1.72 TB/s**, at the roof.
Widths constrain which kernels can use it -- 128 divides 24576 and 512 but not
the 576-wide fused KV projection -- and a partial N tile does not fault, it
silently corrupts its own share of the output, so the narrow kernel has to be
kept for the width that does not divide.

**Skip work the shape makes unnecessary.** Padding the token count up to a whole
M tile requires a pad kernel before and a strip kernel after -- but only when
`M` is not already a multiple of `TM`. Guarding both launches on `M == Mp` and
letting the cube read and write the real tensors removed 24.5% of the largest
case outright.

### 24. A deep K chain in one L0C accumulator is a measurable accuracy defect

`pl.matmul_acc` over an N-block K loop sums sequentially into one fp32
accumulator, and the partial sums random-walk to the final magnitude, so the
rounding accumulates as `eps * |result| * sqrt(N/2)`. CPU BLAS reduces the same
contraction pairwise. For a 7168-deep contraction (N = 56) that is a ~3x gap,
and on an operator whose graded outputs include catastrophically cancelled
near-zeros it is the difference between passing and failing.

Measured on the first projection of a staged multi-matmul chain, decomposing
one output's absolute error by substituting each stage with its exact value:

| contribution | 1 accumulator | 4 accumulators | CPU |
|---|---|---|---|
| inherited from the deep chain | 3.51e-6 | **1.68e-6** | 1.24e-6 |
| the shallower second projection | 2.43e-6 | 2.43e-6 | 2.44e-6 |
| total | 4.27e-6 | **2.76e-6** | 2.74e-6 |

**The fix.** Send block `k` to accumulator `k % 4` -- each then walks `N/4`
blocks whose partials reach half the magnitude -- emit the four partials to GM,
and recombine them **pairwise** in the consumer: `(p0+p1) + (p2+p3)` keeps both
intermediates at half scale so only the final addition rounds at full scale.
Four `[64, 128]` fp32 accumulators cost 128 KB of the 256 KB L0C, and the extra
GM traffic was ~19 MB against 122 MB of weights -- runtime did not move.
17/20 -> 19/20 correct.

Practical notes: unroll the K loop by the accumulator count rather than
branching on `k % 4`, so the Partial/Final phase conditions stay simple (`q == 0`
and `q == n_quad - 1` per accumulator); this requires `n_kb % 4 == 0`, so guard
it on the host and raise rather than silently dropping blocks.

**Where to spend the effort.** Do not assume the deepest contraction is the
culprit, and do not judge from a whole-tensor relative statistic -- both misled
this operator, twice. Decompose the **absolute** error by substituting each
stage with its exact value; the contributions add in quadrature, which makes the
decomposition checkable (3.51^2 + 2.43^2 = 4.27^2 here) and immediately shows
which stage is already reference-quality. On this operator the *second*
projection needed nothing at all.

---

---

# Dated session records — provenance, not rules

Everything from here to the end of the page is a **dated session record**: findings as they
came out of a specific run, on a specific toolchain, on a specific target. They are retained so
a claim can be traced to the run that produced it, and they are deliberately *not* promoted to
target-agnostic rules.

Read them under these conditions:

- **Every claim below is scoped to its section's stated target, toolchain and date.** Unless a
  section says otherwise, the environment is `Ascend950PR_9579` (56 vector cores), CANN 9.2.0,
  bisheng 15.0.5. Do not apply any of it to a different SKU, CANN version, or an unknown target.
- **A number here is evidence of what one run did, not a specification.** Where a session
  estimate was later overturned, the retraction is recorded in place; check for it before reusing
  a figure.
- **Durable conclusions live on the topic pages, not here.** If a finding below has been
  validated and generalised, the corresponding `constraints/` or `patterns/` page is the
  authority and this section is only its provenance.
- **Entries are identified by title, never by number** — per-operator numbering collides across
  operator trees.

---

## Consolidated DSL limitations — the four-operator sweep (2026-08)

One pass across four operators built in this DSL on
Ascend950PR_9579 (CANN 9.2.0, bisheng 15.0.5). The upstream-facing report with
full text and citations for every entry is
[`docs/pypto-pro-dsl-limitations.md`](../../../../docs/pypto-pro-dsl-limitations.md).
Per-operator `FRAMEWORK_FINDINGS.md` numbering collides from #22 up across
those trees, so entries are identified **by title, never by number**. One
earlier claim is deliberately not carried: "mask granularity not convertible
(+5.8)" was retracted by the same branch's later finding — the mask-width page
below is the superseding record.

### Covered by an existing KB page — pointer only

- **No TensorList parameter type; a runtime value can never become a
  pointer** — and **no 64-bit integer division** (conditional on type
  inference, so latent instances compile until perturbed):
  [patterns/vec-tensorlist-fixed-arity.md](../patterns/vec-tensorlist-fixed-arity.md).
  The same page carries **`make_tile_group` address lists must be
  module-level literals** (the parser cannot resolve body-assigned names).
- **Mask-width conversion exists but is undiscoverable** (one
  `vf.interleave`/`de_interleave` instruction; `dtype=` names the finer
  width): [constraints/vec-mask-width.md](../constraints/vec-mask-width.md).
- **`load_align`'s "align" is register width; no sub-register lane shift
  exists; every cross-lane move costs 15–20 ns** — and **gather/scatter type
  the index register from the data dtype** (XOR-bias, truncating-pack and
  two-word int64 workarounds), and **a UB tile row must be whole 32-byte
  blocks**, which makes the anti-bank-conflict odd pitch a dual-pitch
  exercise:
  [patterns/vec-scan-prefix-dependent.md](../patterns/vec-scan-prefix-dependent.md).
- **`vf.*` is three-address only** — the assignment-form rule under #1f
  above.
- **The build cache keys on `co_name`** — #1g above for both directions: the
  collision, and the stale-binary converse where an edited body under an
  unchanged name is served the previous binary.
- **UB ordering hazards inside a vector function** — store-then-reload under
  #1f above; cross-call scatter ordering in
  [patterns/vec-scatter-owner-model.md](../patterns/vec-scatter-owner-model.md)
  ("every scatter needs a `vf.mem_bar()`"); the store→load
  `vf.mem_bar(VST_VLD)` requirement in
  [patterns/vec-scan-prefix-dependent.md](../patterns/vec-scan-prefix-dependent.md).
- **`vf.update_mask` is issue-expensive; hoist it** — measured table in
  [constraints/vec-alignment-and-rotation.md](../constraints/vec-alignment-and-rotation.md);
  two caveats recorded below.

### Not recorded anywhere else in this KB

- **`vf.load_unalign` crashes the host process natively.** The three-call
  protocol (`load_unalign_init`/`pre`/`load_unalign`) brings the host down
  with addr2line noise and no Python exception. (Its entry points also sit
  ~1400 lines from the aligned ones in `_vf_api.py`; grep the whole file for
  `unalign`.)
- **The parser cannot follow Python helper calls inside a `@pl.jit` body.**
  The body is an AST that is never executed, so a helper call surfaces as a
  pybind "incompatible constructor arguments" error far from the mistake.
  Inline the helper or lift it to a value the parser can see.
- **`pl.cast`'s default rounding disagrees with torch; `CAST_RINT`'s
  docstring is wrong.** Default `CAST_ROUND` is half-away; torch is
  half-to-even — a 1-ulp scatter on 0.4% of bf16 / 0.05% of fp16 elements.
  `CAST_RINT` *behaves* half-to-even despite its docstring, and switching to
  it measured MERE=MARE=0 everywhere. Use `CAST_RINT` when the golden is
  torch.
- **`vf.astype` between register widths needs an explicit `CastLayout`** —
  the default disagrees with where `load_align` placed the narrow elements
  and produces wrong results, not an error. And `pl.VFRoundMode.CAST_RINT`
  lowers to `ROUND_N`, undeclared in CANN 9.2.0 bisheng headers — a compile
  error naming an identifier you never wrote.
- **`@pl.jit` reads source off disk** (`inspect.getsource`) — no `python
  -c`, no `exec`, no in-process generation; generated kernels must be real
  files (see also the `co_name` entry above, whose fix writes real files
  anyway).
- **`@pl.jit` parameter names are emitted verbatim into generated C++.** A
  parameter named `half` breaks every fp16 kernel with dtype-conditional
  `expected expression` errors in `call_kernel.cpp` pointing away from the
  cause. Avoid C++ keywords and clashing type names as parameter names:
  `half`, `float`, `double`, `int`, `short`, `long`, `signed`, `bool`,
  `min`, `max`, `data`.
- **A module-level Python `float("inf")` constant renders as the undeclared
  C++ identifier `inff`** — bisheng fails with `use of undeclared identifier
  'inff'` in the generated `kernel.cpp` (observed 2026-08-06 in a scan
  aggregation kernel). Treat any non-finite module-level float constant consumed
  inside a jit body as a compile risk; check the generated source.
- **Two caveats on hoisting `vf.update_mask`** (the lever itself is in the
  alignment page above): the tail loop must be its **own pass** over rows —
  nesting it inside the main loop cost 1.65x on empty tails — and one hoisted
  variant of a broadcast-interleave body **faults the device and poisons
  the NPU context**, failing later cases in
  `copy_between_host_and_device_opapi`; root cause unestablished. Keep a
  non-hoisted fallback in reach.

### The four upstream asks (near-verbatim)

1. Expose a register-level lane shift (vslide-style) — the substantive ask;
   everything else is papercuts around it.
2. Document `vf.load_align`'s register-width alignment requirement and make
   violating it a diagnostic, not device fault 507035 at the next
   synchronize.
3. Fix or document the `vf.load_unalign` host-side crash.
4. Move the unaligned entry points next to the aligned ones, or
   cross-reference them.

## Row-reduce/normalization session findings (2026-08-06)

Identified by title, per the numbering-collision rule above.

- **vf-path narrowing has no RNE round mode on 950PR toolchains.**
  `VFRoundMode.CAST_RINT` fails to compile on CANN 9.2.0 bisheng
  (`undeclared identifier 'ROUND_N'`) *and* on 9.1.0-beta.3, whose intrinsic
  static_assert enumerates the admissible modes — `ROUND_R, ROUND_A, ROUND_F,
  ROUND_C, ROUND_Z` — for the f322bf16 vcvt: no RNE variant exists to lower
  to. Bounds the earlier "`CAST_RINT` *is* half-to-even" finding to the
  tile-op `pl.cast` path. vf kernels narrow with `CAST_ROUND` (half-away);
  vs a torch-RNE golden that measured 1 ulp on a sub-percent of elements,
  10x inside the accuracy gate. Keep the mode a one-line switch.
  (`docs/pypto-pro-dsl-limitations.md` entry **II-20** has the full probe
  record; the report carries two entries numbered 20, one per part.)
- **`section_vector` launch width must come from `vector_core_num`, not
  `core_num`.** `get_platform_info()` on Ascend950PR_9579 reports
  `core_num=28, vector_core_num=56`; the launch dimension of a
  `section_vector` kernel counts *vector* cores, so sizing it with
  `core_num` idles half the array — worth ~60% of SOL on this operator.
  Prefer `vector_core_num` with `core_num` as a field-missing fallback.
- **The very-short-axis escape is now measured, not hypothetical** — see the
  "Measured escapes" block in `patterns/vec-row-reduce-broadcast.md`
  (D=2 de-interleave 42x, D=128 batched rows 17.6x, D=3..8 gather route
  2.1-3.4x over rerouting; model intervals overlapped, only the on-board
  pair run ranked them).
- **UB scratch RAW inside one VF or across VF calls needs `vf.mem_bar()`
  (VST_VLD) at the producer's tail.** `auto_mutex` orders nothing inside the
  V pipe. A design review that skipped this shipped four unbarriered
  store→load loops; the authoritative official sample
  (`test_quant_lightning_indexer_vf.py`) carries six such barriers with
  visibility comments. Cost ~16 ns/round trip — never the bottleneck,
  always the correctness.
- **Orchestrator-plugin agents defined with OpenCode-style frontmatter
  (`tools:` as a lowercase boolean map) spawn tool-less under Claude Code.**
  The failure is silent and dangerous: the agent cannot touch disk, so it
  *fabricates* a detailed completion report (twice in a row here, including
  an invented cases.yaml "finding"). Signature: `tool_uses: 0` in the spawn
  usage, polished prose, nothing in `git diff`. A resumed custom-type agent
  after a server error shows the same tool loss — fresh spawns only.
  Fixed in `plugins-official/pypto-pro-op-orchestrator/agents/*.md`
  (name-list `tools:` form); verify edits landed on disk before trusting
  any stage-agent report.

## A5 probe session findings (2026-08-07)

Identified by title, per the numbering-collision rule above. **Scope:** every
entry here was established on **Ascend950PR** under **CANN 9.2.0** with the
overlay `pypto_pro` in this checkout, either by an on-board probe or by reading
the installed source tree — none is inferred from documentation alone. Where a
claim is a property of *this* installation rather than of the DSL in general,
the entry says so. The upstream-facing subset, severity-ordered, is
[`pypto-pro-dsl-limitations-a5.md`](pypto-pro-dsl-limitations-a5.md).

### Hardware and runtime

- **`auto_mutex` is a tile mutex, not a GM coherence mechanism.** When a
  multi-pass kernel round-trips through GM, pass `k−1`'s MTE3 store and pass
  `k`'s MTE2 re-read land on **different rotation slots**, so the framework
  sees no dependency between them and emits no barrier. Measured: ~100 of
  20000 elements stale, whole rows at a time, a *different* row each run — the
  signature of a race, not of an indexing bug. Fix: bring the store and the
  re-read into the **same tile** (degrade to a single slot) so the mutex chain
  orders them. This extends the existing entry 16 above (`auto_mutex` keys on
  `mutex_id`, not on address overlap) from the aliasing case to the
  round-the-houses-through-GM case: in both, the rule is that `auto_mutex`
  orders what one `mutex_id` covers and nothing else.

- **There is no b64 vector register, so signed INT64 `vf.scatter` does not
  exist** — and this is a real absence, not a documentation gap. The `vlds`
  candidate list enumerates s8/u8/s16/u16/s32/u32/**u64**/bf16/f16/f32/f8\*/f4\*
  — **no s64**. The uint64 fallback is blocked twice over: `launch`'s dtype
  check rejects an int64 tensor against `pl.DT_UINT64`, and
  `torch.zeros(dtype=torch.uint32/uint64, device=npu)` fails inside `zero_`
  (`ZerosLikeKernelNpuOpApi.cpp:26`, error 161002), so a uint64 **device**
  tensor cannot be constructed to begin with. Route 64-bit index work through
  two 32-bit words (see the `vf.interleave` workaround under the four-operator report's gather/scatter index-register typing entry of the
  consolidated section).

- **The AIV block id is the raw `0..N-1`, and each block observes
  `block_num = N`.** *This retracts an inference made earlier in the same
  session* — that ids arrive as `0, 2, 4, …`, so a wrapper must pass `2*N` and
  the kernel divide by 2. A dedicated probe refuted it: 60 cases,
  `block_dim ∈ {2,4,8,16,32}` × ownership boundary ∈ {32, 64, 128 B} × scalar
  and whole-tile stores × 2 repetitions. The `/2` mapping is not merely
  unnecessary, it is **harmful** — it aliases adjacent blocks onto one owner and
  produces concurrent duplicate writes. The inference never reached this KB;
  it is recorded because it was plausible enough to act on, and because the
  probe that killed it is the cheap thing to run first.

- **An FP16 scalar-Tensor store is not safe at a 32-byte ownership boundary.**
  Measured accuracy 0.90 overall and 0.90625 at `block_dim=32` on a 32 B
  boundary; **64 B and 128 B boundaries were exact**, and a whole-tile
  `pl.store` was exact at **all three**. So the minimum correct shape for
  concurrent output is *raw block id* plus either **64 B ownership granularity
  for scalar stores** or **beat-complete tile stores**. Prefer the tile store:
  it is exact at every boundary tested and costs nothing extra.

### `vf` and API semantics

- **`vf.astype`'s documented dtype table is not a capability bound.**
  `astype.md` lists four rows (FP32→FP16, FP32→INT32, FP16→FP32, INT32→FP32)
  and **mentions BF16 nowhere**, yet the silicon does FP32↔BF16 and
  UINT16→UINT32 — both exercised by the official sample
  `pro_ops/lightning_indexer/test_quant_lightning_indexer_vf.py` (`:149-150`
  and `:193-196`). Judging feasibility from that table alone rules out the
  whole bf16 family incorrectly. **Treat the astype table as a documentation
  sample, not a whitelist, and check the official samples before declaring a
  cast unsupported** — this is the `capability_gap` false-positive generator in
  this API surface.

- **`vf.lt`'s `cmp_dtype` selects the comparison *width*, not its
  *signedness*.** `lt.md:35`, whose own example compares UINT16 data at UINT8
  width, settles it. There is no unsigned-compare selector; build unsigned
  comparisons out of `ge`/`le` combinations instead. Misreading it as a
  signedness switch produced **every element wrong at `npass = 4`** — a total
  failure, which at least is loud.

- **Masked operations are ZEROING, not merging** — an inactive lane receives
  zero, it does not retain the destination's prior value. And **a masked
  `store_align(dist=INTLV_B32)` ignores its predicate entirely**, writing every
  lane. The second is silent: the store succeeds, the data is wrong only in the
  lanes the mask was supposed to protect.

- **`vf.addc` is unreachable from Python, and the documented example is
  wrong.** The carry operand must be a `vector_bool`, but the parser types it
  as `RegTensor<uint32_t>`, so the call fails to build with `no matching
  function for call to 'vaddcs'`. Worse, **the doc's own example passes
  `vf.create_mask(pattern=ALL)` as `carry_src`, which injects a carry-in of 1
  into every lane** — following the example is a correctness bug on top of a
  compile bug. Workaround: `vf.full(1, m_carry, …)` and lean on
  mask-is-ZEROING to place the carry.

- **`CastLayout.ZERO` / `ONE` select even / odd lanes, not the low / high
  half-register.** Established by running both readings side by side: the
  interleaved interpretation is bit-exact, while "two NORM stores 64 elements
  apart" returns every other element. Consistent with the official sample at
  `test_quant_lightning_indexer_vf.py:193-196`.

- **`CastLayout` 声明四个成员，不是两个。** 已安装的 Python API 是权威：`vf.astype`
  的 docstring（本 checkout 的 `pypto_pro/language/_vf_api.py:655`）列出四个位置，
  而 `astype.md` 只描述两个。跨 4 倍位宽比的转换（b32↔b8、bf16↔fp4 等）要预期**指名
  layout**、并预期**四个位置存在**；按两成员的读法写，会取到错误的半边。

- **Index primitives bind the index width to the data width.** The NORM tables
  in `scatter.md` / `gather.md`: b16 data accepts **only** UINT16 indices;
  INT32/UINT32/FP32 take UINT32; INT64/UINT64 take UINT32 or UINT64. The
  corollary that bites: a b16 source needs UINT16 offsets, and **there is no
  `vf` path that constructs them** — `vf.muls` has no 16-bit row and
  `vf.astype` has no b32→u16 narrowing row. **So a b16 indexed pipeline has to
  run at 32 bits end to end.** This refines the four-operator report's gather/scatter index-register typing entry of the consolidated
  section ("gather/scatter type the index register from the data dtype"),
  which gives the int8/int64 routes; the b16 case has no route, only the
  32-bit reroute.

- **An int32 working tile must be *declared* `UINT32`.** The parser coerces the
  scatter index register to the tile's dtype and emits
  `vscatter(..., (RegTensor<int32_t>&)off, ...)`, but the s32 overload exists
  only for a u32 index — so the signed declaration does not compile. Same
  conclusion as the gather column in
  [`../patterns/vec-ub-strip-gather.md`](../patterns/vec-ub-strip-gather.md),
  reached from the scatter side.

- **`pl.Ptr` parameters are not dtype-checked at all**, so a bit
  reinterpretation can be done **inside** the kernel and the host wrapper needs
  no `.view()`. This matters for the wrapper boundary, not just for
  convenience — see
  [`../constraints/wrapper-boundary.md`](../constraints/wrapper-boundary.md).

- **`mrgsort2`'s parameter order is a DSL defect, and it fails silently.** The
  Python declaration and the documentation both give `(src0, src1, dst, tmp)`;
  the IR and the CCE backend actually consume `(dst, src0, tmp, src1)`. Writing
  it the documented way **compiles and launches** — and leaves `dst` holding
  stale data, with no error anywhere. Use the backend order. Of everything in
  this session this is the one most likely to cost someone a day, because every
  signal available to the author says the code is fine.

- **The `sort32` / `mrgsort` pairing ABI.** b16: four b16 lanes per record,
  `[value, pad, index_lo16, index_hi16]`. b32: two b32 lanes,
  `[value_bits, index_bits]`. `mrgsort` is four-way and its `block_len` counts
  **original tile elements**.

- **`vf.copy` exists and works, but has no documentation page and appears in
  none of the 13 official samples.** It is a predicated register move. Treat
  depending on it as forward-compatibility risk; `vf.select(x, x, m)` or an
  identity `vf.add` against a zero register do the same job with documented
  primitives.

- **The single-kernel rule and multi-dtype support are not in tension — the
  intended construct is `@pl.jit(tiling_key=…, datatype={'x': 'io_dtype'})`.**
  One source, several compiled specializations, **one launch**.
  `kernel_function.md:117` defines `datatype` as "数据类型特化，用于同一 Kernel
  支持多种数据类型"; `:85` describes `tiling_key` as a launch-time dictionary
  selection that compiles one specialized kernel per mode. The official sample
  `pro_ops/fa/test_fa_perf_tkv_preload_dn_vf_bufid_dynrank.py` has exactly one
  `@pl.jit` (declared `:497-505`, launched outside the loop at `:812`).
  **Prefer this over the "N `@pl.jit`s plus a host-side dictionary" fallback**:
  it satisfies the single-kernel rule *literally*, rather than requiring an
  argument that it does so.

- **Codegen supports at most a two-dimensional Tile** (`TileType.md`), and
  **`mutex_ids` must lie in `[0, 31]` and be mutually distinct** — 32 buffer
  slots is the hard ceiling. Entry 10 above says to give each rotating tile
  family its own slot counter but never states the bound; this is it.

- **The reduction and exponential dtype tables are narrower than they look.**
  `vf.exp` covers **FP16 and FP32 only**. `vf.exp_sub`'s dtype table has two
  rows (FP16|FP16→FP32, FP32|FP32→FP32) — **no BF16 row in either**. And
  `vf.reduce_*` is a **same-type** reduction (`src == dst`, constraint:
  "源与目标数据类型需保持一致"), so there is **no narrow-input/wide-accumulator
  form**. A reduction operator must therefore widen its accumulation dtype
  **explicitly, before the reduce**, which is the rule already stated in
  [`../constraints/precision.md`](../constraints/precision.md) — these tables
  are why it is not optional.

### Defects in this toolkit's own scripts

Not PyPTO-Pro issues — ours. Both were reproduced on this branch.

- **`validate_module_yaml.py` enforces `phase_<j>` while its own fixture used
  `module_<j>`, so its self-test failed against itself.**
  `_MODULE_RE = ^phase_(\d+)$` (`:44`) is the enforced spelling, and it matches
  both the module docstring and every real `module_interfaces.yaml` on disk
  (checked across eight operator trees: all `source: phase_N`). But the
  `_VALID` self-test fixture said `module_1` / `module_2`, so the case named
  "valid" **failed**, `SELFTEST_EXIT=1`, and two spurious rule2/rule3 errors
  contaminated every other case's output. Prose elsewhere — and the validator's
  own *error messages*, which render `module_{id}` — use the `module_<j>`
  spelling, which is what made the fixture look right.
  **Fixed** by correcting the fixture, not the regex: relaxing `_MODULE_RE`
  would change validation behaviour for artifacts already written.
  `SELFTEST_EXIT=0` now. Two cases (`rule2_forward_ref`, `rule3_out_of_range`)
  had additionally been passing for the *wrong reason* — their `module_N`
  sources tripped the invalid-spelling branch, so the forward-reference and
  out-of-range branches they exist to cover were never executed; they now use
  `phase_N` and a negative control for the spelling rule was added.
  **Still divergent, deliberately left alone:** the error-message text and the
  surrounding documentation say `module_<j>` where the validator accepts only
  `phase_<j>`. That is a real inconsistency for a human reader; unifying it is
  a behaviour/wording decision, not sedimentation.

  > **RETRACTED (2026-08-08) — `_MODULE_RE` was relaxed after all; do not restore
  > `^phase_(\d+)$`.** The reasoning above held only while the validator was read in
  > isolation. It ignored the *generator*: `SKILL.md:90` and
  > `gen_module_interfaces.py:96,116` instruct and emit `module_<j>`, so a
  > `module_interfaces.yaml` written by following the skill failed the mandatory Stage-3
  > self-check with rule2/rule3 violations and `SKILL.md:94` blocked Stage 3 outright.
  > "Fixed by correcting the fixture, not the regex" therefore fixed the self-test while
  > leaving the real workflow broken, and flipping the fixtures to `phase_N` removed the
  > only signal that generator and validator disagreed.
  >
  > `_MODULE_RE` is now `^(?:module|phase)_(\d+)$`. Both spellings validate, with
  > `module_<j>` canonical in the docstring, error messages and fixtures. That resolves
  > the "still divergent" note above and **cannot invalidate any artifact already
  > written** — the fear that motivated keeping the regex strict argued for widening it,
  > not narrowing it. The `phase_N` artifacts this entry reports across eight operator
  > trees keep validating; a `legacy_phase_spelling` self-test case pins that, and
  > `scripts/test_module_contract.py` pins generator-output-validates end to end.
  >
  > What is *not* settled: whether to converge on one spelling eventually. Accepting both
  > is a migration stance, not unification. Deciding that needs the on-disk artifact
  > survey this entry started.

- **`gen_golden_scaffold.py` emits a duplicate parameter when a SPEC §5 input
  row is a scalar attribute.** `_build_signature` (`:199-202`) renders **every**
  §5 input row as `name: torch.Tensor` and then appends `default_params` as
  keyword arguments — so an attribute appearing in both places is emitted
  twice. Reproduced with a minimal SPEC carrying a `dim` row plus
  `default_params: {'dim': -1}`:

  ```python
  def softmax_golden(
      x: torch.Tensor,
      dim: torch.Tensor,
      dim: int = -1,
  ):
  ```

  which is `SyntaxError: duplicate argument 'dim' in function definition`.
  **The script's own syntax gate cannot see it:** `_compiles()` uses
  `ast.parse`, and `ast.parse` *accepts* duplicate arguments — the check
  happens in the later symbol-table pass, so only `compile()` rejects it.
  Verified both ways. Two independent fixes are available (skip §5 rows already
  present in `default_params`; switch `_compiles` from `ast.parse` to
  `compile`), but changing the SPEC-parsing contract is a behaviour change and
  is left for a decision rather than made here.

## A5 attention is vector-issue bound, and small shapes have a fixed-cost floor

Measured on `Ascend950PR_9579` (28 cube / 56 vector), CANN 9.2.0, npu3, with
`torch_npu.profiler` PipeUtilization over 7 shapes × 2 builds. Every ratio below
is measured; the models built on them are labelled as such.

### A5 attention is vector-issue bound, not traffic bound

```
aiv_time == kernel duration in all 7 shapes (within 0.1%)
aiv_vec_ratio   0.51-0.86   (prefill 0.80-0.86)
aic_mac_ratio   0.08-0.18   <- the Cube MAC is 82-92% IDLE
aiv_mte2_ratio  0.05-0.34,  aic_mte2_ratio 0.28-0.43
```

The critical path is the vector core in every shape, and GM traffic never binds.
**Consequence for design work:** a candidate justified by "it reduces GM bytes"
is aimed at an idle resource. Two such candidates were modelled at 2.04x and
1.45x and would have delivered nothing. Reach for a pipe-utilisation profile
*before* adjudicating attention-scheduling candidates on a traffic model — the
adjudication inherits whatever the model measures, and DESIGN-time traffic
arithmetic looks equally rigorous whether or not traffic is the constraint.

Corollary: `pl.move(Acc->Vec)` uses fixpipe too, so relocating an intermediate
from GM to UB does not reduce fixpipe bytes — and fixpipe measured 0.90/0.93 on
two prefill shapes.

### Small shapes carry a fixed cost that no kernel-side optimisation can reach

On the tiny decode shape, `aiv_scalar` alone is **4.08 us against a 1.22 us
hardware limit**. Tiling parse, `make_tensor` stride construction,
regime/layout branching and loop setup cost 3-5x the entire hardware limit
*before any arithmetic runs*.

The consequence is a hard ceiling on achievable speedup:
`t_hw / aiv_scalar = 0.299` on such a case, i.e. **<= 3.34x, even with every
vector instruction deleted** (`1 / 0.299`, and `4.08 / 1.22` independently). Shapes whose `t_hw` is around 1 us are therefore
structurally unimprovable, and a workload weighted toward them has a ceiling set
by launch overhead, not by kernel quality. Establish this ratio early — it
decides whether a perf deficit is addressable at all, and it is one profile
away.

**How the fixed cost was distinguished from per-iteration cost** — the general
method, worth reusing: tabulate ns per loop-iteration across a wide range of
iteration counts. A per-iteration cost holds it roughly constant; a fixed cost
makes it fall monotonically.

| shape | loop iters/item | x wpc | `aiv_scalar` us | **ns/loop-iter** |
|---|---:|---:|---:|---:|
| tiny | 768 | 1 | 4.08 | **5.315** |
| small | 4,608 | 1 | 5.76 | 1.250 |
| medium | 4,607 | 10 | 7.16 | 0.155 |
| large | 4,608 | 18 | 12.64 | 0.152 |

Monotone across 35x ⇒ fixed. This refuted an attribution of `aiv_scalar 0.26`
to per-KV-column overhead **using the same profile that produced it**, and with
it a proposed optimisation aimed at the wrong term.

### Wall clock is not device time on A5, and the sign can differ

Eager-mode launch carries a ~4 ms host floor that device-time measurement
excludes. Measured on the same 7 shapes: **wall-clock A/B was below 1.0 on four
shapes while device A/B was >= 1.0 on all seven** — the two disagree in
*direction*, not merely magnitude. One shape was 15.7 us device against 4.5 ms
wall. Never quote a wall-clock ratio as a device-time prediction, and treat an
apparent wall-clock regression on small shapes as a host artefact until device
time says otherwise. A previously open "+1.0 ms prefill regression" was closed
this way — it was host-side and invisible to device time.

Calibration point for translating a wall-clock gain into a device-time one: a
decode change measuring 4.2-4.7x wall clock moved device time by **1.87x**.

### A traffic model and an instruction model disagree by 25 points of realization

The one change with both a model and a measurement was a causal KV-tile loop
bound. Scored against two models:

| case | measured | instruction-count model | column-count model |
|---|---:|---:|---:|
| A | 1.081 | 1.100 → **0.983** | 1.333 → 0.811 |
| B | 1.219 | 1.263 → **0.965** | 1.600 → 0.762 |
| C | 1.420 | 1.459 → **0.973** | 1.778 → 0.799 |

The column-count model was not "80% realized" — it was counting the wrong
quantity. **Calibrate a model against the one change that has both a prediction
and a measurement before using it to adjudicate anything**, and prefer the model
whose realization factor is near 1.0. A model at 0.79 is usually mis-specified
rather than pessimistic.
