# RFC-0008: The a2 family (c220) — milestone M9

Status: draft, 2026-08-28. Decisions: D-061.

## 1. What the a2 family is

The a2 family is Ascend 910B (`dav-2201`, arch **c220**): devices `b1` … `b4`
(`ascriptor/devices/profiles/b*.json`; a3 — Ascend 910_93 — shares the ISA and gets its own
facade later). The facade is `ascriptor.a2` with `DEVICE = "b3"`: the old repository's
default (`easyasc/a2.py`) and the box named in the git-ignored `machine_specs.md`.

What differs from a5 (c310), as the old repository's kernels and references record it
(historical A2 constraints and device facts):

| a5 (c310) | a2 (c220) |
|---|---|
| vector core programmed with `@vf` register functions (`vf.*`) and `@simt` | vector core programmed with the **tensor-vector ISA** (`vec.*`: 42 registered ops + the mask state ops), no registers, no SIMT |
| `l0c_to_ub`, `ub_to_l1*`: cube <-> vec through on-chip memories | neither exists: cube -> vec is `l0c_to_gm_nz2nd` -> GM workspace -> `gm_to_ub_pad`, vec -> cube is `ub_to_gm_pad` -> GM -> `gm_to_l1_nd2nz` |
| 32 cube / 64 vec cores, UB 256 KB, L0C 256 KB, BT 4 KB | 20 (b3) or 24 (b1, b2) cube cores, two vec sub-blocks per cube core (`GetSubBlockIdx`, `GetVecIdx = 2 * cube + sub_block`), UB 192 KB per sub-block, L0C 128 KB, BT 512 B (one slot = 64 fp32 bias values) |
| cross-core waits on any pipe | cross-core waits (`wait_cube` / `wait_vec` / `all*_wait`) and mutex start pipes on `PIPE_S` only |
| `mmad_mx`, mx scales, fp4 / fp8 cube formats | `mmad` on fp16 / bf16 / int8 / int4 (`DT.int4` through int32 carriers), no mx |
| `gm_to_ub_pad` bursts unbounded | `n_burst` is a 12-bit field (0 … 4095) |
| `Div<float>` IEEE | `Div<float>` carries a deterministic 1-ulp correction (the board); `compare(x, x, NE)` is unordered-false |

## 2. Method (D-061)

Two sources, one per layer, as the maintainer directed:

* **Semantics** (the interpreter, RFC-0003 replays): the old simulator's vector pipe
  (`easyasc/simulator/pipe_vec.py`, `VPipe`) with `constraints/vec.md` as the written spec —
  the mask model (256 lanes per vec core, the active prefix `256 / sizeof(dtype)`, normal vs
  counter mode), the repeat / block-stride / repeat-stride traversal (8 blocks of 32 bytes per
  repeat), masked write-back for element-wise ops, the sentinel / zero rules of the reductions,
  the mask-ignoring ops (`select`, `compare*`, `gather*`, `scatter`, the sort family, `brcb`),
  the packed-bit control tensors of `compare` / `select`. The auto-inference of `repeat` and the
  strides from a view's `(span, shape)` is the frontend's job (the old stubs' `infer_repeat` /
  `infer_strides`), so the IR always carries explicit attributes.
* **Emission** (the cce backend): for every op, start from the old repository's **AscendC
  handler** (`easyasc/targets/ascendc/asc_handlers/*.py` — the API and parameter mapping the a2
  kernels were verified with on the board), follow CANN's `dav_c220` implementation of that API
  (`<cann>/x86_64-linux/asc/impl/basic_api/dav_c220/kernel_operator_*_impl.h`) down to the CCE
  intrinsic, and print the intrinsic. `arch/c220.py` records the AscendC name next to every
  intrinsic so the trace can be re-walked (the a5 rule: CANN's own implementation is the ground
  truth, `docs/diagnosing-hardware.md` rung 8).

The validation ladder is the a5 one minus T2: M3 gate (the ported corpus compiles and
verifies), the replay of goldens recorded from the old simulator (RFC-0003), M4 gate (lowering +
the pipe-level simulator with the a2 cycle model), T1 (print + bisheng `dav-c220`), then
straight to T3 on the assigned A2 card machine. **There is no T2 on a2**: the
local cannsim's SoC table (`<cann>/python/site-packages/cannsim/soc_info.py`) supports
`Ascend950` only — the 910B camodel directories under `tools/simulator/` are not wired into the
tool — and the maintainer directed to skip the simulator stage rather than chase it
(2026-08-29).

## 3. Phases

* **A — frontend, interpreter, corpus gates.** `ascriptor.a2`; the `vec.*` DSL names with the
  old signatures (positional strides, `count=` / `count_per_rep=` keyword modes); the frontend
  rule (`rules_vec.py`) with the auto-inference; the interpreter's vector pipe; the a2 corpus
  ported into `tests/kernels/a2/` (`tools/port_kernel.py`) with goldens recorded from the old
  simulator (`tools/record_goldens.py`); `tests/test_corpus_a2.py`, the replay and lowering
  gates parameterised by device; `tests/kernels/a2/corpus.json` with the exclusions.
* **B — the c220 cce backend and the launchers.** `arch/c220.py`; the printer's `vec.*`, DMA
  and cube branches for c220; the entry (`KERNEL_TYPE_MIX_AIC_1_2` / AIV-only, `ASCEND_IS_AIC` /
  `ASCEND_IS_AIV` split, cross-core sync); the aclnn project for `ascend910b`; a `boards.json`
  entry for the a2 box (no cannsim launcher: T2 does not exist on a2).
* **C — the support surface.** One sample kernel per feature under `tests/kernels/a2/samples/`
  (D-045) through T1, cannsim and the board, and the a2 rows of `docs/cce-support.md`.

## 4. The vector ISA model

The frontend (`frontend/rules_vec.py`, the DSL names in `frontend/dsl_vec.py`) and the interpreter
(`backends/sim/vec_ops.py`, the cast numerics in `backends/sim/cast_rounding.py`) share one model,
the old simulator's:

* **Operands.** UB windows only; the dtype allow-lists are the old stubs' (`add` / `sub` / `mul`:
  f32, f16, i32; `div`: f32, f16; `vmax`: + i16; `vand` / `vor` / `vnot`: i16, u16; the unary
  transcendental ops: f32, f16; the shifts: 16- and 32-bit integers; `dup`: f32, f16, i32, u32;
  the reductions: f32, f16; `compare*` sources f32 / f16 / i16 / i32 into an i8 / u8 bit tensor;
  `select` f32 / f16 / i16 / i32 with a u8 bit tensor; `gather*` / `scatter` u32 byte offsets;
  `transdata5hd` b16). `muladddst` takes `(f32, f32)`, `(f16, f16)` or `(f32, f16)` for
  `(dst, src)`.
  Mixed `muladddst(f32, f16, f16)` consumes64logical lanes per repeat on
  every operand. Source strides still count32-byte blocks of half elements;
  only the low64mask bits apply. Bounds, masked writes and block/repeat strides
  follow that same lane mapping (M10-036).
* **Repeat and strides.** An instruction walks `repeat` repeats of 8 blocks of 32 bytes; the
  block strides and repeat strides are in blocks. The frontend infers what the call leaves out:
  `repeat = CeilDiv(numel(span), 256 / sizeof)` from the destination (the source for the
  reductions, `compare` and `scatter`; the wider dtype for `cast`; `numel(src) / 8` for `brcb`);
  a view whose row is exactly 8 blocks gets `(blk 1, rep = row_of_allocation / C0)`, a one-block
  row is the broadcast layout `(blk 0, rep = row_of_allocation / C0)` — a `[M, 8]` fp32 tile is 8
  aliases of one block per repeat, not dense storage — any other row `(1, 8)`, and a single
  matched row `rep 0`. `cast` pairs a repeat stride of 8 on the wider side with `8 * C0_narrow /
  C0_wide` on the narrower one. The IR carries every value explicitly.
* **Sticky SPRs at launch.** A launch begins with accumulation off on both sides and, on the vector
  side, every mask lane on in normal mode — a state each side **establishes** rather than inherits,
  because these SPRs survive a kernel boundary and nothing on this family resets them between
  launches. A body whose first mask-dependent instruction comes before its own `set_mask`, or whose
  GM store comes before any `atomic.begin`, would otherwise take what the previous kernel left
  (M10-095). The vendor establishes the same state, and does it at entry rather than exit for the
  same reason, but only for a mix op and only on the cube core.
* **The mask.** 256 lanes per vector lane; the active
  prefix is `256 / sizeof(dtype)` lanes (the wider dtype for `cast`) and it is the same for every
  repeat.
  `set_mask(high, low)` writes lanes 0..127 (`low` first), `set_mask_by_count(n)` a prefix,
  `reset_mask()` all ones. Element-wise ops, `dup`, `cast` and `muladddst` keep a masked-off
  lane's old value; `cadd` / `cgadd` / `cpadd` add 0 for it, `cmax` / `cgmax` see -inf, `cmin` /
  `cgmin` +inf, and a reduction whose lanes are all off leaves its result alone (`cpadd` always
  writes the pair sums); `brcb`, `compare*`, `select`, `gather*`, `scatter`, `transdata5hd` and
  the sort family ignore the mask.
* **Counter mode (D-062).** `count=` and `count_per_rep=` are **attributes of the instruction**,
  not machine state: the IR carries no vector-mode register, so source order and execution order
  can never disagree — the old stubs' source-order mode tracker, its lazily-emitted switches and
  its automatic `bar_v` (the part of a2 the maintainer flagged as the most troublesome) have no
  successor in the IR. A `count=n` op processes the first `n` elements, contiguous, no strides,
  no mask (`dup` keeps `blk 1 / rep 8`, the form AscendC's counted `Duplicate` hardwires); a
  `count_per_rep=n` op runs under its own `n`-lane prefix mask. After either, the mask register
  is all ones again — the bracket CANN's own dav_c220 Level-2 calls emit (`set_mask_count`;
  `set_vector_mask(0, n)`; the instruction; `set_mask_norm`; `set_vector_mask(-1, -1)`), with
  **no barrier**: the mask SPR writes dispatch in order with the vector instructions on the V
  queue, so ordering is free. The interpreter executes each op by its own attributes; the c220
  backend materialises the SPR sequence around each op (the per-op bracket first — hoisting a
  shared bracket over a run of same-mode ops is a later, measured optimisation with the whole
  CFG in view). Explicit `set_mask` / `set_mask_by_count` / `reset_mask` remain real ops on the
  lane's mask register; `set_mask_normal()` compiles but is a no-op for the interpreter.
* **Numerics.** Half and bf16 compute in fp32 and round on the store; `rec` is `1 / x`, `lrelu`
  `x > 0 ? x : x * s`, `axpy` `dst + src * s`, `muladddst` `dst + src1 * src2`; the bitwise ops
  work on the bit pattern of any width; a left shift moves the full fixed-width pattern (the
  sign can flip), a right shift is arithmetic for signed and logical for unsigned tensors
  (`round_en` adds the shifted-out bit for signed); `compare` treats NaN as unordered-false,
  `NE` included; `cast` rounds by the mode (`round` nearest-away, `rint`, `floor`, `ceil`,
  `trunc`, `odd` on fp16), float -> int rounds in fp32 then saturates, a narrowing int -> int
  saturates; `compare` packs its bits LSB-first, 32 bytes per repeat; `gather` / `scatter`
  offsets are bytes (`start_idx` added), `gather_block` moves 8 32-byte blocks per repeat.
* **Footprints.** An op that addresses past the end of the allocation its window lives in is an
  interpreter error (the sim-hidden overrun into the neighbouring tile); a strided op's footprint
  counts only the lanes the mask enables, a block op's every block.

## 5. The cube path on c220: fractal layouts, entry and launch

**Fractal layouts (the maintainer's note, 2026-08-28).** `mmad` on c220 and on c310 accept
different L0 fractal formats, which is why the old framework has four L0 layouts — NZ, ZN, ZZ,
NN. The rules, from the old simulator (`easyasc/simulator/pipe_cube.py`) and the AscendC handler
(`asc_handlers/cube.py`):

| step | c220 (a2) | c310 (a5) |
|---|---|---|
| `gm_to_l1_nd2nz` | fp32: ND -> **ZZ** on L1 (`GM2L1_ND2ZZ`); other dtypes: ND -> NZ | ND -> NZ |
| `l1_to_l0` to L0A | fp32 L1 (ZZ): ZZ -> **ZZ** (`L0ZZ2ZZ`; transposed source: ZZ -> **NN**, `L0ZZ2NN`); other dtypes: NZ -> **ZZ** (`L0NZ2ZZ`; transposed: NZ -> NN) | NZ -> NZ (transposed: NZ -> ZN) |
| `l1_to_l0` to L0B | fp32 L1: ZZ -> NZ (`L0ZZ2NZ`; transposed: ZZ -> ZN); other dtypes: NZ -> NZ (transposed: NZ -> ZN) | NZ -> NZ (transposed: NZ -> ZN) |
| `mmad` operands | L0A = ZZ, L0B = NZ | L0A = NZ, L0B = NZ |
| `l1_to_l0_img2col` | L0A materialised as ZZ | NZ |
| `l0c_to_l1` | not for an fp32 destination | any |
| `mmad` int4 | `DT.int4` through int32 carriers on both L0 sides, int32 L0C, an optional fused bias through BT, the old simulator's B-device tail-transfer quirk | not an a5 format |
| L0C | NZ, C0 = 16 | NZ, C0 = 16 |

The interpreter keeps L1 / L0 tiles **logical** (ND) as it does for a5 — a fractal layout is
how the bytes sit in the on-chip memory, and no `mmad` result depends on it — so phase A needs
no change there; the layouts are the c220 backend's business (phase B): `device_lower` records the
L0 format of every `l1_to_l0` (`zz` / `nn` for L0A, `nz` / `zn` for L0B, the fp32 ZZ L1 source) and
the printer maps each to its `load_cbuf_to_ca` / `load_cbuf_to_cb` form (the AscendC `LoadData2D`
/ `LoadData2dTranspose` the old handler emitted), and `gm_to_l1_nd2nz` of fp32 to the ND -> ZZ copy.
The one place the interpreter meets a fractal layout is a plain `gm_to_l1` of data the kernel
pre-packed (the conv weights, `transdata_w_to_fractal`): those bytes are the device's L1 format,
and the replay of the a2 conv kernels will say whether the a5 model (logical tiles) already
covers them (open item until the corpus replays).

**Short accumulation settle (M10-081).** Fresh M16 tests on both 910B3 and
910_93 establish that two MMADs updating one L0C are not interlocked by
M-pipe program order alone: the following `is_init=False` MMAD may read before
the previous writeback settles. An M-pipe barrier between them was measured to be
sufficient for FP32. The A2-family split-K expansion therefore emits that barrier
after every fragment, and a lowered correctness lint catches hand-written
same-L0C MMAD streams without an M/ALL barrier.

The rule admitted FP32 accumulators only until 2026-09-18, when that restriction
was withdrawn as unjustified: the interlock the hardware lacks is a writeback, and
a writeback does not read the accumulator's dtype. Replaying the corpus with the
dtype condition removed reports eighteen `is_init=False` MMADs that destine an L0C
slot an earlier unsettled MMAD wrote, not one of them FP32, one of them in the
kernel of M10-095. The
device-family limit stays, and stays a measurement boundary rather than a claim:
c220 was measured not to interlock this RAW, other families were not, and each
carries its own ordering policy. The rule does not apply to the independently
scheduled L0 operand ownership flags.

A settle emitted **within** a split-K expansion is narrower than the hazard, which
is the second thing the census showed: a hand-written K split — two `matmul` calls
accumulating one L0C slot, or one call in a loop whose back edge carries no barrier
— received nothing on any dtype, and eight catalogue kernels are written that way.
The rule is the hardware's and not the author's, so **the pipeline owns it**: the
`mmad_settle` pass (RFC-0006) inserts `sync.barrier(pipe = M)` immediately before
every accumulate that the lowered-IR analysis finds reading an unsettled L0C, on
this family and no other. Before the accumulate rather than after the producer,
because one placement then covers the straight-line pair, the loop back edge and
the branch join.

The split-K expansion keeps emitting its own settle, and the pass is silent where
it already did the job. That is deliberate rather than tidy: `desugar`'s placement
after the whole MMAD branch is what M10-081 measured bitwise on both boards, and
moving it to buy one insertion point would spend that evidence for nothing. The
lint remains the check — it is what the pass asks — and after the pipeline it has
nothing left to report on an A2-family kernel, which is how the repair is measured.

**Entry and launch.** Filled in as phase B lands (the old kernelbase's a2 entry:
`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2 | AIV_ONLY | AIC_ONLY)`, the `ASCEND_IS_AIC` /
`ASCEND_IS_AIV` split, `get_subblockid`, cross-core sync through `ffts_cross_core_sync`).

## 6. Status

* 2026-08-28: RFC drafted; the registry already carries the 42 a2 ops (`ascriptor/ir/ops/vec.py`,
  `devices=A2`) and the mask state ops; the interpreter and the printer implement none of them
  yet; `ascriptor/backends/sim/timing/a2_cycle_model.json` is the old repository's table.
* 2026-08-29 (phase A validated): the reference interpreter replays the old simulator's recordings
  bit for bit — 84/107 kernels compile (the 23 exclusions are documented in
  `tests/kernels/a2/corpus.json`), and every replayable kernel matches every recorded case
  (`docs/corpus_a2.md`); 218 small cases are tracked under `tests/goldens/functional`. Fixes on the
  way there: masked-off tail blocks past a UB allocation are never read or written (`_vsrc_block`
  / `_vmasked_write`), workspace dims may be core-count queries or write-once Vars (the host
  tiling rule's mirror), dynamic row counts use free shape symbols, `matmul_quant_fp32_ubyte` is
  the unsigned twin of the byte kernel (the old kernel was dtype-polymorphic in `z`), and the sort
  family reproduces the old simulator's unstable tie order (how silicon vbitsort orders ties is a
  T3 question).
* 2026-08-29 (phase B printer): `vec.*` prints through `backends/cce/emit_vec.py` (the `C220Vec`
  mixin on `FnPrinter`, tables in `arch/c220.py`); `ModulePrinter.arch` selects c310 / c220 from
  the module's device; the D-062 bracket materialises per counted / count_per_rep op; a2 GM->L1
  (ND) lowers to the plain 32-B block copy (c220 has no byte-granular L1 pad DMA); 75 corpus
  kernels print end to end (`tests/backends/test_cce_a2.py`), one is `cce_xfail` (atomic fixpipe
  store).
* 2026-08-29 (T1: bisheng dav-c220): `tensorutils_cce.h` carries the c220 sections behind
  `__CCE_AICORE__ == 220` (D-064) — the DataCopyPad `align_b8/b16/b32` forms, the classic block
  copies shared verbatim, `gm_to_l1_nd2nz` (b8 / b16 native; float = the old GM2L1_ND2ZZ 16-row
  bands feeding the cube's ZZ layout), the eight `l1_to_l0` layout bodies of §5 on raw
  `load_cbuf_to_ca/cb(_transpose)` (exact argument types decide the polymorphic builtin's
  overload), `mmad` / `mmad_bias` shared (the 10-argument `mad` is spelled identically on both
  arches), the 12-argument fixpipe with `set_nd_para` / `set_quant_pre` and the c220
  `fixpipe_quant` arms, FFTS cross-core (pair 0x2 / all 0x0 / intra 0x1; waits block the scalar
  pipe), `SetMaskCount` on the real builtin, the VA-register `transdata5hd` and the
  vbitsort / vmrgsort4 sort family. **69 corpus kernels compile through bisheng on both c220
  units**; `tests/backends/test_cce_a2.py` compiles six canonical kernels (vec / counted /
  matmul / quant / sort / cross-core) when the local toolchain is present. Remaining header
  items: the conv im2col load (five kernels `cce_xfail`, needs its own dav_c220 load3d trace),
  the c220 `set_atomic_*` family (`arch/c220.py` tables it; no corpus kernel outside the
  xfailed atomic-fixpipe one needs it yet), `l0c_to_l1`.

* 2026-08-29 (T3: first board runs, the a2 / 910B3 box): the OpExec board launcher works end to
  end on the a2 machine (boards.json entry, aliases b1..b4; the aclnn ascend910b project builds
  on the box; the injected `matmul::clearWorkspace` of the 910B mix wrapper is shimmed). Verdicts:
  `matmul_quant_fp32_byte` (fixpipe QF322B8_PRE, signed saturation), `fma_mixed_kernel`
  (vmla f32 <- f16) and `to_hifx_kernel` (the counted / count_per_rep brackets of D-062, per-block
  cmax reductions) are **bit-exact**; `exp_general` and the muls->adds->exp probe sit at the
  hardware-vexp ULP level (2.4e-07, ok-tol at rtol 1e-4 / atol 1e-5); `matmul_half_basic` shows
  the known fp16 mad accumulation difference (1.9e-06, ok-tol — the old framework's number on the
  same box). Two silicon findings drove fixes: the **V-V hazard barrier** (D-065 — the c220 vector
  pipe does not interlock UB accesses between its own instructions; the printer now materialises
  `pipe_barrier(PIPE_V)` between colliding vector ops, the data half of the old auto `bar_v`) and
  **exact argument types** for every intrinsic (the polymorphic builtins resolve by argument
  pattern; some patterns belong to older chips). Diagnosis probes live as samples
  (`probe_tiling_echo`, `probe_src_addr`, `probe_vadds_chain`) with tracked goldens.

* 2026-08-29 (header completion — atomics, l0c_to_l1, im2col): the c220 `set_atomic_*` family
  lands as its own arm of the atomic section (identical raw SPR spellings on both arches; the
  c220 arm follows CANN's order — the op SPR first, then the dtype — and carries no
  `ASCEND_IS_AIV` guard, because the c220 units compile separately and the SPR applies on
  whichever core issues the store: AIV for UB->GM, **AIC for fixpipe stores**). The printer
  brackets an atomic L0C->GM fixpipe with `SetAtomic<kind><T>()` / `SetAtomicNone()` on c220
  (`_fixpipe_atomic`), which un-xfails the attn_backward kernel; `l0c_to_l1` prints on the
  12-argument `copy_matrix_cc_to_cbuf` (mapping B.12 — dstStride in 32-B units, fp32->fp32 on
  the channel-split leg); the conv im2col load prints on `img2colv2_cbuf_to_ca` (mapping B.11 —
  c220 load3dv2 materialises the whole [m_ext, k_ext] window in one call after FMATRIX /
  PADDING, no c310 repeat-SPR dance; bf16 goes through the half spelling as in CANN), which
  un-xfails the five conv kernels; `set_constant_to_l1` moved to the shared region (CANN's
  dav_3510 and dav_c220 InitL1BufferCal are byte-identical). `cce_xfail` is now **empty**: 78
  kernels print and **78/78 compile through bisheng on both c220 units**; the canonical bisheng
  gate adds `conv_half_basic` (img2colv2 + FMATRIX).

* 2026-08-29 (T3 full-corpus board sweep + the fp32 ZZ window fix): `tools/run_cases.py
  --launcher board` over every tracked a2 golden (77 rows). **66 kernels with a valid board run
  all pass**: 46 bit-exact, 20 at hardware ULP levels (vexp 2.4e-07; fp32 mad 1e-6..1e-5; fp16
  outputs flip 1 ULP — 9.766e-04 at [1,2), 3.125e-02 at [32,64), so fp16 kernels judge at rtol
  1e-3). The sweep caught one real printer bug: **fp32 L1 window offsets folded as NZ while the
  c220 cube stores fp32 L1 tiles as ZZ** (§5) — an offset-0 load hid it (nosplit passed), every
  fp32 splitk / splitn kernel read the wrong K/N sub-block (diffs up to 6.2e+02). Fix:
  `views.elem_bytes_offset` grows a ZZ branch — element (r, c) of a [R, C] fp32 L1 tile sits at
  `(r/16)*(align16(C)*16) + (c/8)*128 + (r%16)*8 + (c%8)` — armed per module by the printer
  (`set_l1_fp32_zz`, c220 only; c310 keeps NZ). Board after the fix: splitk 7.4e+01 -> 3.8e-06,
  splitn 2.9e+01 -> 5.7e-06, kmkn_splitn 6.2e+02 -> 1.1e-05 (the nosplit level). Remaining rows:
  the conv M-tail semantics note below (2), the attn_backward hang note above (1), the
  pipeline-protocol lowering_xfails (3), and coverage artifacts of excluded kernels that still
  have goldens (int4, div probes — `run_cases` walks the golden index, not corpus.json). A
  kernel killed on the box can leave the next run failing `aclrtSynchronizeStream` 507015
  (task-abort residue): bilinear_interp hit it and is bit-exact solo.

* 2026-08-29 (the attn_backward hang, root-caused and fixed — D-066): the maintainer's question
  ("does the old framework hang too?") settled the frame: the old framework's build of the same
  kernel passes on the same box in 3.6 s (bf16 1 ULP) — and it emits NO flags at all (same-core
  sync is pure PipeBarrier there; its FFTS sequence is line-for-line identical to ours). A
  barrier-ized Event::set/wait build of OUR artifacts also passes with correct numerics, so the
  data path and FFTS were innocent; per-pipe-pair bisection on silicon converged on the MTE3->V
  valid group. Root cause: the events pass coloured flag ids by static live range, but a valid
  hand-off under run-ahead compensation can end a (per-side) block execution with a token still
  in flight — the balance oracle's finding, previously read as conservative noise. A later event
  reusing the id consumed the leftover token and the channel deadlocked (aicore abort 507015).
  Fix: the colouring pins every oracle-flagged event's id to the function end, and the colouring
  re-runs after split_sides (`events_restamp`) — pre-split the two sides' books balance each
  other out and the tail is invisible. Board after the fix: **attn_backward passes** (gq/gk at
  the bf16 1-ULP level — the old framework's exact number — gv bit-exact). The balance checker's
  warning stays (the tail is real); the flag-id plan now respects it. Both the run-ahead
  compensation that produced such a tail and the pinning that guarded it retired with the edge
  planner on 2026-09-17 (RFC-0005 §5.5); this entry is why the rule existed.

### Settled

* **`sort_rows`: silicon orders ties differently from the old simulator, the sort itself is right,
  and the comparison now says which.** The T3 question §6 left open. On the eight-card sweep the
  kernel's two outputs split cleanly: the sorted **values are bit-identical**, all 163840 of them,
  and only the index output differs — by **4 entries, in two adjacent pairs, each a swap** (row 19
  columns 3691/3692 carry 1128, 209 in the golden and 209, 1128 on the board; row 27 columns
  3721/3722 the same shape). Equal values, permuted indices: `vbitsort` breaks a tie the other way
  round. No `rtol`/`atol` can express that — a bound compares magnitudes, and index 209 against
  index 1128 has none — so the entry states what the output **means** instead
  (`board_tolerance` … `"outputs": {"1": {"index_gather": {"input": 0, "values": 0}}}`, D-215): the
  index output passes only if gathering `x` by it reproduces the recorded values **bit for bit**
  and each row selects the same multiset of positions the golden selected. That is a stricter check
  than the bitwise one it replaces, not a looser one — it leaves exactly one freedom, the order
  among equal keys — and it is what a widened bound could never have been. Both bounds stay 0.
  If a consumer ever needs a stable order the kernel must break ties itself; none does.

### Open items

* ~~**conv M-tail rows differ from the old simulator on silicon**~~ — **closed 2026-09-05 (D-223):
  the interpreter now slides the window there too, and the mask is gone.** For output rows
  m >= OH*OW (a tile's M padding, the height an NZ output plane must have) the old simulator writes
  zeros and `load3dv2` does not: it keeps sliding the window by the same formula, so a tail row whose
  window still lands inside the feature map produces real data. Modelling that is one term — the
  `m_idx < ho * wo` guard leaves `_im2col_window` and `op_cube_conv2d`, and the remaining `ih` / `iw`
  bounds give exactly "partly inside produces data, fully outside reads zero". Measured on both
  arches with the interpreter's own output compared against the board's directly: a5
  `conv_half_dilation` 7.6e-06, a2 `conv_half_dilation` 7.6e-06, a2 `conv_half_large` 3.05e-05 — fp32
  cube accumulation order alone, over EVERY row, where board-against-golden on the tail had been
  1.235e+01 and 4.412e+01. The three goldens are re-recorded from this repository's interpreter
  (`tools/rerecord_from_interp.py`, with the reason stamped in each manifest) and the
  `rows_valid_per_period` masks that skipped those rows are deleted, so the tail is now checked
  rather than excused. No corpus entry uses `rows_valid_per_period` any more; the mechanism stays for
  a region that genuinely cannot be compared.

* **Software-pipelined user events** — no corpus kernel is blocked any more, the planner
  limitation itself stands. The hand-rolled slot protocols of the block32_causal / full_pfa_mha /
  flash_attn_full family placed sets and waits under runtime pipeline guards that the run-ahead
  bound (D-034) cannot statically pair. The resolution was to give the kernels a construct the
  planner does understand rather than to teach the planner those guards: RFC-0009's GMBuff rings
  plus autosync (D-067, D-068) — the sixteen hand-event kernels are rewritten and board-verified,
  `lowering_xfail` is empty, and the one armed tail token at kernel end was drained by autosync's
  guarded final drain (D-072), which emptied `balance_xfail` too; a slot session keeps it empty
  structurally, because its window closes at the end of the block that opened it (RFC-0005 §5.4).
  The silicon hang this once caused is resolved (D-066: the tail token forbade flag-id reuse).
  Should a kernel ever need the original spelling, what it would take is a pipeline-slot construct
  the planner understands; there is no run-ahead bound left to refine.
* **Zero-trip loops in the happens-before model**: knowledge learned inside a `cf.for` body whose
  bounds are not provably non-empty no longer escapes the loop, and an accepted pair whose
  producer sits inside such a loop transfers nothing across its exit (`passes/deps.py`,
  found by `addn_tensor_list` case-0 — a one-member list runs the accumulate loop zero times).
* **Range-aware V-V hazard keys**: the D-065 tracker collides at allocation-root granularity;
  two vector ops touching disjoint ranges of one UB root still get a `pipe_barrier(PIPE_V)`.
  Correct, conservative; refine only if board profiles show it costing cycles.

## 7. M10 A3 facade and board validation (D-242, 2026-09-06)

The maintainer requires A3 to reuse the A2 kernel family and independent references, with device
selection as the authoring difference. Add the A3 facade using the same tensor-vector DSL and
shared algorithms; do not maintain a duplicate A3 kernel tree. The selected A3 profile still
supplies its device/build identity and the runtime uses the corresponding toolchain/configuration.

Validate generated inputs against self-contained references on the exact A3 server entry the
maintainer selected in ignored `machine_specs.md`, using ignored launcher configuration. Machine
access values must not appear in this RFC. Record A3 emission, compilation and hardware results
separately, including relevant tail, buffer reuse and cross-side cases from the A2-family contract.
No recorded golden archive is required (RFC-0003 §9). A2 historical board results do not count as
an A3 run. Configuration correspondence was checked in the M10 planning review; `ascriptor/a3.py`
is implemented, and real A3 execution is recorded per feature — the slot sessions' own A3 board
runs are in their receipt.

### INT4 fused bias (2026-09-23)

An int4 `matmul` may carry `bias=`. The bias is moved L1 -> BT and read by the initialising
`mmad`, so a split-K matmul adds it exactly once; `tests/passes/test_a2_int4_fused_bias.py` owns
that lowering and `int4_bias` in the canonical set compiles it with bisheng for both sides.
Qualified on an A2 card in both shapes: single-chunk at M=N=K=64, and `k=128, splitk=64` against
an independent reference, the unfused result plus the bias, and the single-chunk fused result —
all exact, and a bias applied once per chunk would have shown as exactly one extra bias.

It took a day to get there and the path is worth keeping, because none of the four obstacles was
the one it looked like, and two of them were separate defects sitting underneath this one.

**The specialisation was missing.** `tensorutils_cce.h` specialised `mmad` for int4 operands -- it
calls `mad_s4` -- and had no matching `mmad_bias`, so the generic template printed a `mad` call
that bisheng rejected at the header. Nothing above the backend objected: the frontend, the IR,
`desugar`, the reference interpreter (which computes the sum correctly) and the printer all
accepted the form, and the first refusal came from the vendor compiler. A sim-only or pipesim-only
check would have called the kernel correct.

**Its shape had to come from the compiler, not the headers.** The two bias-table `mad_s4`
declarations that are easy to find put `BTAddr` second and sit under `__NPU_ARCH__ == 5102` and
`__DAV_L310_EFF__ / __DAV_L311__`, neither of them c220; building on them fails with "parameters
too many". An arity sweep against bisheng says the c220 builtin takes **eleven** arguments with the
bias-table address **fourth**, after both operands: `mad_s4(matrixC, matrixA, matrixB, BTAddr, M,
K, N, unitFlag, isWeightOffset, ctrlMatrixC, initMatrixC)`.

**The first board qualification failed, and not because of the bias.** A minimal probe loading the
int4 carriers straight from GM was wrong in exactly half its M rows *with and without* a bias --
so the control failed too, and a control that fails qualifies nothing. That probe turned out not
to be about int4 at all, and not about the bias: the cube keeps **fp32** L1 tiles in the ZZ
fractal layout and everything else in NZ, and the backend chose ZZ by `sizeof(T) == 4`. An int4
operand's carriers are an int32 tile, so it was written, read and address-folded as ZZ -- needing
`align16(rows) * align16(cols) * width` -- while `addr_alloc` and the model had it as NZ and
reserved half that. Two adjacent tiles overlapped and the one loaded first lost its upper half,
invisibly below a card, because D-022 makes every DMA a logical window copy. See
[A2-B32-ND2NZ-L1-OVERWRITE](../defects/A2-B32-ND2NZ-L1-OVERWRITE.md), closed by testing the dtype
instead of its width at the three sites that had it backwards.

With that fixed, the fused form qualifies on an A2 card at M=N=K=64: an fp16 control,
both damage axes, the fused result against an independent reference, the unfused result against
that reference minus the bias, and the fused result against the unfused plus the bias -- **all
exact, 0 of 4096 elements wrong**.

**And the split-K qualification failed too, again in the control.** Extending the probe to
`splitk=` found the fused result wrong *and* its no-bias control wrong, at both 8 and 16 carriers
per chunk, while fp16 split-K was exact. That was a third defect, under the second: `k` is logical
int4 elements, but an int4 operand's L1 tile, its L0 slot and the `i4` reinterpret are counted in
int32 carriers of 8, and the split-K path sliced L1 in logical elements. Only `cube.mmad` stayed
right, because it takes `K` directly — which is why the emitted sequence read plausibly against
the shipped kernel's. `desugar` now divides the chunk width and the window start by 8 for an int4
operand, exactly as the `mx` path already did with its own packing factor.

Nothing on a card had ever run that combination. No kernel in either repository passes `splitk=`
with int4 operands: `a2_int4_tail` writes its own K loop in carriers, `a2_sage_int4` uses
`splitn=`, and the only `int4` + `splitk=` call sites are the canonical backend cases — compiled,
never executed. **A form that only ever compiles has no board result**, the same way a form that
never compiles does not, and this RFC had already learned the second half that morning.

The lesson that outlived the feature is the shape all three obstacles shared. Between 2026-09-23
and the same day's qualification `desugar` refused the form outright, and the refusal was correct
at the time: the specialisation did not exist. What made it expensive was that the *refusal* was
then kept alive for a borrowed reason — every available control shared an unrelated defect — and
the way out, twice, was to stop treating the int4 path as the subject and run an fp16 tile through
the identical harness. A control that fails does not convict the subject; it says the harness is
untrustworthy and the subject is still untested.

### INT4 byte boundary (2026-09-20)

C220 `mad_s4` consumes `2 * ceil(K/2)` logical nibbles. On measured A3
silicon, odd K with a high padding nibble of -1 in both operands adds 1;
with 7 it adds 49. Zero padding and even K controls match exact products.
The functional and lowered models must retain this physical byte boundary.
A logical-K algorithm accepting arbitrary carrier padding must clear unused
nibbles on the device before MMAD. The public `a2_int4_tail` unit does so in
private GM storage, followed by explicit cache clean before MTE2; it keeps
all original padding controls and the exact INT64 reference. A2 confirmation
and complete repaired-unit qualification are tracked in M10-099.
