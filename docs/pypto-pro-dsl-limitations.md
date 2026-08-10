# PyPTO-Pro DSL limitations and improvement requests

Consolidated from building four CANN Bench operators end-to-end in the
PyPTO-Pro `pl` / `vf` tile DSL: `foreach_addcdiv_scalar`, `swi_glu`,
`apply_rotary_pos_emb`, and `cummin`. Hardware and toolchain:
Ascend950PR_9579, CANN 9.2.0, bisheng/clang 15.0.5.

Every entry states what was observed, how it was established, and what it
cost; each ends with the fix that would have prevented it. Citations of the
form `[FF #N tree:lines]` point into `tools/cann_bench_a5/FRAMEWORK_FINDINGS.md`
on the named operator branch (the per-operator working trees; entries #22 and
up are numbered per-session there and collide, so entries here are identified
by title). Links into `cannbot-skills/ops/pypto-pro-op-kb/` point at the knowledge-base pages
that carry the full workaround.

One earlier claim is deliberately **not** carried: "mask granularity is not
convertible" (with a projected +5.8 OperatorScore for fixing it) was retracted
by the same branch's later finding — see entry 3, which supersedes it.

**A later session added a second report, and the two do not overlap.**
[`cannbot-skills/ops/pypto-pro-op-kb/references/pypto-pro-dsl-limitations-a5.md`](../cannbot-skills/ops/pypto-pro-op-kb/references/pypto-pro-dsl-limitations-a5.md)
covers findings from an A5 probe session (2026-08-07): `mrgsort2`'s
API/backend argument-order disagreement, the predicate ignored by a masked
interleaved `store_align`, `auto_mutex`'s scope, scalar-store granularity, the
b64 register absence, and several documentation tables that read as capability
whitelists and are not. It is ordered by severity rather than by discovery, and
it indexes this file's 20 entries by title rather than restating them. A reader
reporting upstream wants both.

## Limitations

**1. No TensorList parameter type.** `runtime/jit.py:354-370` recognises
`TENSOR` / `PTR` / `TILING` / `SCALAR` only, argument arity must match the
declaration exactly, and `ptr.make_ptr` requires a `PtrType` — so a runtime
value (a `data_ptr()` read from a tensor) can never become a pointer, and a
Python list of tensors cannot be passed at all. This forces generated
bucket ladders of unrolled parameters (`L = 1..64` reaches 385 declared
parameters at `L = 64`, which does compile and launch). The host-loop
alternative measured **6.46x** device time at `L = 64`. Fix: a real
list-of-tensors parameter kind, or allow `pl.Ptr` construction from a runtime
address. Full pattern:
[`cannbot-skills/ops/pypto-pro-op-kb/patterns/vec-tensorlist-fixed-arity.md`](../cannbot-skills/ops/pypto-pro-op-kb/patterns/vec-tensorlist-fixed-arity.md).

**2. No 64-bit integer division; scalars are int32-only in practice.**
`Div of bitwidth greater than 32 not supported` — and the failure is
conditional on type inference (for example a `pl.min` result feeding a
divisor widens it), so a latent instance compiles until something unrelated
perturbs the types. Workaround: restructure the loop nest to keep divisor
arithmetic in int32. Fix: lower 64-bit division, or diagnose it at parse time
instead of deep in codegen. [FF #15 cummin:364-386]

**3. Mask-width conversion exists but is undiscoverable.** (Supersedes the
retracted "not convertible" claim.) `vf.interleave` / `vf.de_interleave` with
`dtype=` naming the *finer* width re-space a `MaskReg`'s read stride — the
conversion is one instruction, measured correct on 128/128 lanes with
`DT_UINT16` and wrong on 32/128 with `DT_UINT32`. Two operators independently
audited every function with `mask` in its name and concluded conversion was
impossible; the converter is filed under advanced computation. The real
symptom stands: an fp16-width mask (1 bit per 16-bit lane) fed to an int32
`select` reads the wrong bits — values correct, indices garbage. Converting
did not pay on the measured operator (the narrow-dtype penalty was a
tile-pitch bank conflict, found separately), and the residual cost of the
fp32-widening workaround is two extra UB passes. Fix: name or alias a
mask-width conversion (`vf.mask_astype`), and document the MaskReg read-stride
model next to `vf.select`. Full analysis:
[`cannbot-skills/ops/pypto-pro-op-kb/constraints/vec-mask-width.md`](../cannbot-skills/ops/pypto-pro-op-kb/constraints/vec-mask-width.md).
[cummin FF #28 :750-804; #25 :621-660]

**4. `vf.load_align`'s "align" means register width (256 B); violation is a
bare device fault.** No sub-register lane shift is expressible:
`store_align` + `load_align` at an offset of −k lanes faults (507035) even
when every shift is 32-byte aligned and a `vf.mem_bar` sits between the store
and the load. `vf.shift_left/right` are bit shifts within lanes, not lane
moves. Measured per 64-lane register: gather ~20 ns, scatter ~18 ns,
barriered UB round trip ~16 ns, versus under 1 ns for an aligned load/store
and ~0.3 ns for arithmetic — every way of moving data across lanes costs
15–20 ns, every way of not moving it costs under 1. Log-depth scans die on
this (measured floor 1.537 ns/element/core versus 0.942 shipped). Fix: a
vslide-style register lane shift; a compile-time alignment diagnostic instead
of the device fault. [cummin FF #30 :828-916]

**5. `vf.load_unalign` crashes the host process natively.** The three-call
protocol (`load_unalign_init` / `load_unalign_pre` / `load_unalign`) brings
the host process down with pages of addr2line noise and no Python exception.
Fix or document. [cummin FF :901-909]

**6. The unaligned load entry points sit ~1400 lines from the aligned ones**
(`_vf_api.py:1565-1589` versus `:134`). Two readers independently concluded
no unaligned load exists. Move them adjacent or cross-reference. [cummin FF
:860-867]

**7. The parser cannot resolve body-assigned names in `make_tile_group`
address lists** (`name 'C' is not defined`), so tile geometry must be
module-level literals. Fix: constant-fold body-level integer arithmetic, or
issue a diagnostic that names the restriction. [cummin DESIGN.md §6.5]

**8. The parser cannot follow Python helper calls inside a `@pl.jit` body.**
The body is parsed as an AST, never executed, so a helper function call that
would have produced valid arguments surfaces instead as a pybind
"incompatible constructor arguments" error far from the actual mistake. Fix:
diagnose the unresolvable call at parse time. [FF #6 :127-146]

**9. `vf.*` is three-address only — no expression nesting.** Every producing
call must use the assignment form ("vf.reduce_sum produces a result and must
use the assignment form", F00002). Rediscovered independently in two
operators because the samples obey it silently, which reads as style rather
than a rule. Fix: document as a language rule, or auto-flatten nested
expressions. [FF #11 :243-260]

**10. `pl.cast`'s default rounding disagrees with torch, and the `CAST_RINT`
docstring is wrong.** The default `CAST_ROUND` rounds half away from zero;
torch rounds half to even, producing a 1-ulp scatter (measured 0.4% of bf16
elements, 0.05% of fp16). `CAST_RINT`'s docstring describes something else,
but it *behaves* half-to-even — which is why it matches torch: switching to
it took MERE and MARE to 0 everywhere. Fix: default to RINT or document the
divergence; fix the docstring. [FF #14; arpe FF #23 :536-571]

**11. `vf.astype` between register widths needs an undeclared `CastLayout`,
and `CAST_RINT` lowers to an undeclared enum.** The layout default disagrees
with where `load_align` actually placed the narrow elements, producing wrong
results rather than an error; and `pl.VFRoundMode.CAST_RINT` lowers to
`ROUND_N`, which the CANN 9.2.0 bisheng headers do not declare — a compile
error naming an identifier the user never wrote. Fix: document the layout
contract next to `vf.astype`; ship headers that declare what the DSL emits.
[cummin FF :789-802; DESIGN §6.4]

**12. The build cache is keyed on the kernel's `co_name`.** An edited body
compiled under an unchanged function name can be served the previous binary,
so the run reproduces **byte-identically** — indistinguishable from "my
change was inert" after the fact. Fix: key the cache on a hash of the
rendered source; error on one name with two bodies. Mitigation until then:
stamp a source-hash into every kernel name. [FF #17 :387-418]

**13. `@pl.jit` reads source off disk via `inspect.getsource`.** Kernels
cannot be built from `python -c`, `exec`, or any in-process generation —
launch fails with `OSError: could not get source code`. Fix: accept a source
string, or register generated source with `linecache`. [FF #5 :107-126]

**14. `vf.gather` / `vf.scatter` type the index register from the *data*
dtype, while the underlying intrinsics take `vector_u32` regardless.** Fatal
for int32 payloads ("no known conversion ... vector_u32" at C++ compile).
Workarounds, all measured working: keep payloads in UINT32 tiles and use an
order-preserving XOR `0x80000000` key (compare biased, select raw); widen
int8 through int16 with a **truncating** `vf.pack` (the saturating cast
corrupts −1 to 127); emit int64 as two UINT32 words via `vf.interleave`. Two
zero-scoring cases went to 0.593 / 0.687 with these. Fix: type the index
register `uint32` independent of the payload dtype. [FF #21 :479-495; #26
:662-692; DESIGN §6.1]

**15. A UB tile row must be a whole number of 32-byte blocks**
(`pto_tile.hpp:1444`), so an odd *element* pitch is inexpressible — and an
odd pitch is THE anti-bank-conflict tool for strided gathers. Measured:
gather stride 64 fp32 = 51.0 µs versus stride 65 = 7.0 µs (~5–7x for any
power-of-two stride). Under the 32-byte-block rule the only legal
conflict-free pitches are odd multiples of 32 bytes, and no single pitch
serves both a 4-byte and a 2-byte tile (the sets `8·odd` and `16·odd`
elements are disjoint) — forcing a dual-pitch layout bridged by `pl.cast`.
The wrong pitch held six cases at ~6 GB/s/job against 26.5 achievable. Fix:
allow an arbitrary declared pitch (padding internally), or surface the legal
pitch set in the API. [cummin FF #27 :693-749]

**16. `@pl.jit` parameter names are emitted verbatim into generated C++.** A
tensor parameter named `half` breaks every fp16 kernel — the generated
`(half*)` casts in `call_kernel.cpp` become `expected expression` errors that
are dtype-conditional and point away from the cause. Same class: `float`,
`double`, `int`, `short`, `long`, `signed`, `bool`, `min`, `max`, `data`.
Fix: reject C++ reserved and clashing identifiers at parse time, or mangle
parameter names (`p_` prefix). [swiglu FF #22 :496-535]

**17. UB memory ordering inside a vector function is undocumented in two
places.** (a) Back-to-back `vf.scatter` calls to the same UB address are not
ordered **across calls** (the documented rule covers only within one call):
measured 3/256 mismatches without `vf.mem_bar()` and 0/256 with, at ~1/1000
on realistic draws — passes public cases, loses hidden ones. (b) A UB store
followed by a load of the same address needs
`vf.mem_bar(mode=pl.MemBarMode.VST_VLD)` or the vector core faults. Fix:
document the cross-call ordering model; auto-insert or diagnose the missing
barrier. [FF #10 :208-242; :875-882]

**18. `vf.update_mask` is issue-expensive, placement dominates, and the
framework's own reference generator had it wrong.** Hoisting the mask out of
the per-register loop recovered 550.75 → 398.06 µs (the DMA floor) on a 67M
index fill; applied to the reference generator's exp kernels it moved mean
SOL 0.300 → 0.961 across 20/20 cases. Two caveats from the same lever on
another operator: the tail loop must be its **own pass** over rows (nesting
it cost 1.65x on empty tails), and one hoisted variant
(`rope_bcast_interleaved_float32`) **faults the device and poisons the NPU
context** — later cases fail in `copy_between_host_and_device_opapi`; root
cause unestablished. Fix: compiler hoist of loop-invariant `update_mask`;
triage the fault. [cummin FF #29 :805-827; arpe FF #25 :621-659]

**19. A module-level Python `float("inf")` constant renders as the undeclared
C++ identifier `inff`.** A kernel that consumes a module-level
`float("inf")` in a `vf`/`vdup` context generates `kernel.cpp` containing a
bare `inff`, and bisheng fails with `use of undeclared identifier 'inff'`
(kernel.cpp:87; cummin `cummin_agg_float32`, 2026-08-06) — the renderer
appears to append the fp32 literal suffix `f` to the Python repr `inf`
instead of emitting a real C++ spelling for the value. Until fixed, treat
any non-finite module-level float constant referenced inside a jit body as a
compile risk and check the generated `kernel.cpp`. Fix: emit `INFINITY` /
`std::numeric_limits<float>::infinity()` for non-finite float literals.

**20. `vf.astype` narrowing has no round-to-nearest-even mode on 950PR —
`CAST_RINT` fails to compile on both available toolchains.** On CANN 9.2.0
bisheng the enum lowers to the undeclared identifier `ROUND_N`; on CANN
9.1.0-beta.3 the same kernel hits the intrinsic's own static_assert, whose
text settles the question: *"The 4th argument of this vcvt (f322bf16) can
only be: ROUND_R, ROUND_A, ROUND_F, ROUND_C, ROUND_Z"* — the f32→bf16 (and
f32→f16, probed separately) vcvt simply has no RNE variant to lower to.
This bounds the earlier tile-op finding ("`pl.cast` `CAST_RINT` behaves
half-to-even, use it when the golden is torch"): that remains true on the
tile-op lowering path, but a **vf-path kernel cannot narrow with RNE at
all**. The only viable vf narrowing is `CAST_ROUND` (half-away); measured
against a torch-RNE golden on rms_norm this cost 1 ulp on a sub-percent of
bf16/fp16 elements — MARE 7.812e-3 against a 7.81e-2 gate, a 10× margin.
Keep the round mode a single-point switch so a future toolchain that gains
RNE needs a one-line change. (rms_norm, dual-toolchain compile probes on
Ascend950PR_9579, 2026-08-06.)

**21. `pl.jit` keyword arguments are not portable across pypto_pro builds,
and rejecting one kills the submission with no attributable error.** Our
development box accepts `@pl.jit(auto_mutex=True, timeout=600)` (the timeout
raises the compile budget on large unrolls); the evaluation server's
pypto_pro raises `TypeError: jit() got unexpected keyword argument(s):
timeout` at `runtime/jit.py:1309`. Because a submission normally imports its
kernel module at package import time, the exception fires inside the
evaluator's own import and surfaces only as a stage-level
`staged_rc_1_missing_report` — a successful wheel build followed by nothing,
on every runner. It cost three diagnostic submissions to localise, and was
only readable after making the forwarder import the kernel module lazily so
the failure landed inside a case. Fix, in order of preference: keep the
accepted-kwarg set stable across builds, or ignore-with-warning an unknown
kwarg rather than raising, or at minimum name the accepted set in the error.
Consumers: ship only the kwargs the target runtime is known to accept, grep
the emitted module before packaging, and prefer a lazy import in the
forwarder so a runtime mismatch is reported per case rather than as an
infrastructure failure. (foreach_addcdiv_scalar, cannbench 950PR pool,
2026-08-07.)

## Upstream asks, in decreasing order of value

Recorded near-verbatim from the cummin branch [FF :911-936]; the per-entry
"Fix:" lines above are the complete list, these four are the ones that would
have changed the outcome:

1. **Expose a register-level lane shift (vslide-style).** The substantive
   ask; everything else above is papercuts around it.
2. **Document `vf.load_align`'s register-width alignment requirement**, and
   make violating it a diagnostic rather than device fault 507035 at the next
   synchronize.
3. **Fix or document the `vf.load_unalign` host-side crash.**
4. **Move the unaligned entry points next to the aligned ones**, or
   cross-reference them.
5. **Either add an RNE narrowing mode to `vf.astype` (if any future ISA
   revision admits it) or make `CAST_RINT` on the vf path a front-end
   diagnostic** — today it surfaces as `use of undeclared identifier
   'ROUND_N'` (9.2.0) or an intrinsic static_assert (9.1.0-beta.3) deep in
   generated C++, far from the Python line that chose the mode (entry 20).
