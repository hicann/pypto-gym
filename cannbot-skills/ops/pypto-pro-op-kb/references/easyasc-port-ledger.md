# EasyASC port ledger

EasyASC is a sibling Ascend DSL (`easyascv2`) with its own agent knowledge base, developed
against the same silicon this repository targets — A2/A3 and A5 (C310). Its knowledge was
swept and, where it transfers, moved onto these pages. This file is the inventory: every
source item, what happened to it, and why.

**Coverage is the point.** The rows that say *not ported* carry as much information as the
rows that say ported — they record that an item was examined and rejected for a stated
reason, so nobody re-derives the same rejection.

## The two rules the port runs under

**1. Cross-DSL claims are hypotheses until this stack says otherwise.** Every ported claim
carries one of two provenances, and it is visible in the text on the page:

- a **PyPTO-Pro anchor** — an installed `pypto_pro` source line, a
  `$PYPTO_DEVKIT_DIR/docs/pypto_pro/` page, a `$PYPTO_DEVKIT_DIR/pro_ops/` sample, or a
  measurement recorded in this KB; or
- the marker **`未在 PyPTO-Pro 上验证——由 EasyASC 移植的假设`** plus a named probe that
  would settle it.

A cross-DSL statement written in this KB's ordinary declarative voice would be read by the
next agent as measured here. That is the failure this rule exists to prevent.

**2. "Simulator passes, board fails" needs translating, because there is no simulator here.**
A large share of EasyASC's highest-value material is framed against its Python simulator.
PyPTO-Pro has no equivalent, so that framing does not transfer literally. What transfers is
the **class**: *the cheap check passes and the board fails* — where the cheap check here is a
successful compile, a host/eager replay, or a small-shape run. Items whose only content was
"the simulator models this differently" are marked not-applicable rather than ported.

## Provenance states

| State | Meaning |
|---|---|
| **ported (earlier)** | landed in the first sweep, commit `ce6251f` |
| **ported (made durable)** | written in a session that ended before committing; committed as `2d69e4b` |
| **ported (this session)** | added in the sweep this ledger was written for |
| **already held** | this KB reached the same rule independently; EasyASC adds corroboration at most |
| **not ported** | examined and rejected — reason given |
| **conflict** | both sides recorded, neither promoted |

---

## 1. Ported earlier — `ce6251f`

| EasyASC source | Landed in | Provenance |
|---|---|---|
| branchless authoring inside a vector function | `ops/pypto-pro-op-develop/references/vf-reduction-perf.md` § "Do not branch inside a vector function" | ported (earlier) |
| per-call overhead and loop form | same file, § "Per-call overhead and loop form" | ported (earlier) |
| load-issue cost model | same file, § "Load-issue cost model" — labelled a hypothesis with a re-probe recipe | ported (earlier), unverified |
| capability-absence gate (six required records) | [`investigation-discipline.md`](investigation-discipline.md) §11 | ported (earlier), credentialed by a local +8 OperatorScore miss |
| wall-vs-plateau evidence standard | `ops/pypto-pro-op-perf-tune/references/a5-roofline-and-levers.md` § "When the local levers plateau" | ported (earlier) |
| structural levers; degraded-device and current-device timing traps | same file; [`../playbooks/benchmark-scoring.md`](../playbooks/benchmark-scoring.md) § traps | ported (earlier) |
| store-side UB bank serialisation (odd 32-byte-block pitch) | [`../constraints/vec-alignment-and-rotation.md`](../constraints/vec-alignment-and-rotation.md) | ported (earlier) |
| CN/EN hardware and tiling glossary | [`terminology.md`](terminology.md) | ported (earlier), DSL-neutral terms only |
| gather prices the same as an ordinary load (0.995–1.001x) | — | **not ported**: contradicted by this DSL's own measurement of `vf.gather` at 20–35x an aligned load at real dependency depth ([`../patterns/vec-scan-prefix-dependent.md`](../patterns/vec-scan-prefix-dependent.md)) |

## 2. Ported but uncommitted, made durable — `2d69e4b`

| EasyASC source | Landed in | Provenance |
|---|---|---|
| `a5.md` — transposing dense→fractal read alignment cliff (~40 GB/s vs ~1.7 TB/s off a 32-byte source plane) | [`../constraints/memory-layout.md`](../constraints/memory-layout.md) | ported (made durable), **unverified** + named probe |
| `a5.md` — dense→fractal copy rounds its destination C0 stride to `align16(M_dst)` | same page | ported (made durable), **unverified** |
| `a5.md` §4 — on A5 the cube↔vec handoff stays on chip; no GM bridge needed | [`../constraints/sync-stitch.md`](../constraints/sync-stitch.md) | **conflict**, recorded not adjudicated — it contradicts a repeated local measurement |
| `a5.md` §5 — prefer the on-chip `L0C → L1 → L0` cube→cube re-feed | same page | ported (made durable) as a **checked absence**: `pl.move`'s space table (`pypto_pro/language/_api.py:196-206`) has no `Acc → Mat` row |
| `a5.md` §12 — per-token scalar staging needs one 32-byte UB row per token | [`../constraints/vec-alignment-and-rotation.md`](../constraints/vec-alignment-and-rotation.md) | ported (made durable), **unverified**; sits beside a local measurement it corroborates |
| `a5.md` §6.2 — `DIST_NORM_B8` writes its full 256-byte register under a prefix mask | same page | ported (made durable); the mask-ignored half is measured here, the **sizing** consequence is unverified |
| `a5.md` §8 — an 8-bit register→UB scatter consumes only even source lanes | [`../patterns/vec-scatter-owner-model.md`](../patterns/vec-scatter-owner-model.md) §10 | ported (made durable), **unverified**; write-side mirror of a rule held here on the read side |
| `a5.md` — gather index width bound to data width; 8-bit gather zero-extends to b16 | [`../patterns/vec-ub-strip-gather.md`](../patterns/vec-ub-strip-gather.md) | **already held** — carried as cross-DSL corroboration, which is what makes it a CANN-level rather than a lowering property |
| `a5.md` §6.1 — a BF16→FP4 cast requires one of `RegLayout.ZERO/ONE/TWO/THREE` (`vcvt` `P0..P3`) | [`pypto-pro-framework-findings.md`](pypto-pro-framework-findings.md) § EasyASC cross-port | ported (made durable) with a **PyPTO-Pro anchor**: the prediction was checked and `_vf_api.py:655` declares four `CastLayout` members where the docs show two |

---

## 3. Ported this session

| EasyASC source | Landed in | Provenance |
|---|---|---|
| `facts-authoring.md:147` / `constraints/a5.md` §1 — keep a `@vf` scan's loop-carried accumulator in registers; a UB round trip failed a random-input hardware probe **even with `vf_barrier(STORE, LOAD)`** | [`../patterns/vec-scan-prefix-dependent.md`](../patterns/vec-scan-prefix-dependent.md) | ported, **unverified**; the barrier-is-not-enough half is the new content — this KB already required the barrier |
| `simulator-datamove-footprint-guards.md:19,52-55` — burst footprint is `(n_burst-1)*step + burst_len`; GM slice coverage is `1 + Σ(span_i-1)*stride_i`, **not** `Π spans` | [`../constraints/memory-layout.md`](../constraints/memory-layout.md) | ported as **shape arithmetic**, which is DSL-independent; feeds the overlap check that [`investigation-discipline.md`](investigation-discipline.md) §12 already mandates |
| `facts-authoring.md:180` / `patterns/online-softmax-tail.md` — use the finite sentinel `-1.0e30`, not `float("-inf")` | [`../patterns/online-softmax-tail.md`](../patterns/online-softmax-tail.md) | ported with a **PyPTO-Pro anchor**: a module-level `float("inf")` renders as the undeclared C++ identifier `inff` and fails to compile here ([`pypto-pro-dsl-limitations-a5.md`](pypto-pro-dsl-limitations-a5.md) #19) |
| `patterns/online-softmax-tail.md:262` / `a5-mixed-pipeline.md:228` — the tail mask must land **before** `rowmax`, not in the p-domain; padded zeros inflate the running max and corrupt `alpha` and `l` | [`../patterns/online-softmax-tail.md`](../patterns/online-softmax-tail.md) | ported as **ordering** guidance on a page that is already conceptual-only; the failure chain is the new content |
| `authoring-preflight.md:58-61`, `pitfall-records.md:23` — same-core reuse stress gate: `ceil(BH*ceil(S1/TILE_M)/core_count) > 1`; "a passing single-tile simulation is not a reuse proof" | [`investigation-discipline.md`](investigation-discipline.md) §13.2 | ported as a **shape precondition**, mechanically checkable; generalises §13's existing geometry rule to reuse |
| `patterns/buffer-slot-lifetime.md` — required slots ≥ (overlapped beats) × (simultaneously live roles); and the reason a wrong credit count is silent (prologue and epilogue stay balanced) | [`../patterns/buffer-reuse-lifetime.md`](../patterns/buffer-reuse-lifetime.md) | ported as a **sizing law**, API dropped; connects to the local finding that a wrong slot stride aliases two buffers while the event machinery still issues two credits |
| `authoring-preflight.md:62-63` — treat the ownership checker's warnings as correctness signals; resolve the model or make a concrete proposal, do not waive them | [`investigation-discipline.md`](investigation-discipline.md) §13.2, closing paragraph | ported **generalised** — EasyASC names its own `auto_sync` warnings; the transferable form is that a passing result with an unresolved ownership warning is not a passing result |
| `simulator-datamove-footprint-guards.md:64-66` — do not clamp an out-of-range view to the parent size with `min(...)`; it hides the real out-of-bounds and re-surfaces as an unrelated footprint failure | [`../constraints/memory-layout.md`](../constraints/memory-layout.md), with the extent formulas | ported as a **diagnostic** rule; it is about how a bounds check should fail, which is DSL-independent |
| `facts-device-runtime.md` — a BT slot holds 512 fp32/int32 elements on A5 (64 on A2/A3), and a shortcut-matmul or conv bias must fit one slot | [`../constraints/arch-a5.md`](../constraints/arch-a5.md) | ported as a **derived design bound** (tile N/Cout ≤ 512), not as a capacity number. **Half-anchored**: `bt_size=4096` is in the SKU platform file; the two-slot structure that halves 1024 elements to 512 is the unverified half |
| `facts-device-runtime.md` — the 12-bit `n_burst` field (`[0, 4095]`, 4096 silently no-ops) is a C220 restriction A5 does **not** inherit | [`../constraints/arch-a5.md`](../constraints/arch-a5.md) | ported as a **negative** result — it says do not budget for it on A5 |
| `facts-device-runtime.md` — 950 and 950pr share the C310 instruction and codegen family | [`../constraints/arch-a5.md`](../constraints/arch-a5.md) | ported, **unverified**; it is the reason a 950 finding is worth testing on 950PR at all |

## 4. Conflicts

| Subject | EasyASC | Here | Outcome |
|---|---|---|---|
| A5 UB capacity | 256 KB, 216 KB when the kernel contains SIMT | 248 KB, stated twice in the installed tutorial for Ascend 950PR/950DT | **Resolved in our favour.** `ub_size=253952` (= 248 KB exactly) in `950PR_957x.ini`, and identically in `950DT_957x`, `950PR_958x`, `950DT_958x`. Recorded at [`../constraints/arch-a5.md`](../constraints/arch-a5.md) together with *how* it was settled: the search that first concluded "no source decides this" had grepped `python/pypto_pro`, which has no UB constant, and not the platform `.ini` that arch-a5's own primary-source list names |
| cube↔vec must round-trip GM on A5 | no — the handoff stays on chip | the hand-written per-tile fused construct never ran; the generated pipeline path does | [`../constraints/sync-stitch.md`](../constraints/sync-stitch.md) — reconciled into a three-way statement rather than a winner |
| `splitk` / `splitn` minimum 32 | `decomposition-primitives.md` P5 calls 32 a hard minimum; two other files in the same corpus call it a tuning heuristic | — | **not ported in either form.** The source contradicts itself, so there is nothing to port; a reader who needs the bound must measure it |

## 5. Not ported — and why

Grouped by the reason, because the reasons repeat.

### 5a. The construct does not exist in PyPTO-Pro

| EasyASC item | Why not |
|---|---|
| `Var(existing_var)` is a fresh alias, not a snapshot — the C++ backend may fold it into its uses so a "saved" value reads the new one, and **the simulator snapshots it so this passes every simulator case and only fails on board** | No PyPTO-Pro construct has this shape. There is no re-binding assignment operator on a kernel value (`language/typing/scalar.py` exposes `Scalar`; there is no `__ilshift__` or equivalent). The hazard needs an aliasing declaration form the DSL does not offer |
| nonzero L0C row offset: simulator passes, hardware leaves part of L0C uninitialized | Not expressible. `pl.matmul(dst_tile, lhs, rhs)` and `pl.matmul_acc` take a whole accumulator Tile (`language/_api.py:569,581`); there is no destination-offset parameter, so a matmul cannot target a nonzero row offset in the first place |
| MX matmul geometry (`splitk % 64`, `splitn % 16`) | No MX surface here — `matmul_mx`, `MxType` and any `_mx` entry point are absent from the installed `pypto_pro` |
| `CvMutex` / `VcMutex` spellings, `depth=`, `DBuff`/`TBuff`/`QBuff`, `auto_sync()`, `@vf`/`@simt`/`@func` decorators, `GMTensor` subscripting, `OpExec`, `sim_print` | EasyASC API surface. PyPTO-Pro's equivalents (`make_tile_group(auto_mutex=True)`, sections, `@pl.pipeline.stage`, `vf.mem_bar`) are different objects with different semantics. The *shape* of the mutex-depth argument was ported (§3); the API was not |
| A5 pipe legality details — CrossCore set/ready rejecting `Pipe.S`/`Pipe.ALL`, cube-side ready not using `Pipe.MTE3`, barrier side ownership | Stated against EasyASC's explicit per-pipe event API. PyPTO-Pro does not expose hand-placed CrossCore pipe events at this level; there is no call site at which the rule would be checked |
| pipe→op mapping table (`gm_to_l1_nd2nz` → MTE2, …) | The pipe *names* are already in [`terminology.md`](terminology.md); the op names are EasyASC's. PyPTO-Pro documents its own pipe per API page, which is the version-correct source |
| A2 `SelectMode.TENSOR_SCALAR` silently rewrites NaN lanes | A2-only, and an EasyASC API name |

### 5b. A2/A3-only, explicitly not inherited by A5

| EasyASC item | Why not |
|---|---|
| `DataCopyPad` at `n_burst = 4096` silently does no work | A2/A3 C220 restriction; the source itself states A5 does not inherit it. Carried onto [`../constraints/arch-a5.md`](../constraints/arch-a5.md) only as the negative statement |
| A2 fixed sub-block split (`sb_row = sb*HALF_M`, ownership pinned to rows [0:64)/[64:128)) vs the A5 compact split | Both are EasyASC sub-block authoring models. PyPTO-Pro does not expose the vec sub-block as an addressable half in this form |
| A2/A3 capacity rows of the device table | Not our target |

### 5c. Already held here, usually with a stronger local anchor

| EasyASC item | What we already have |
|---|---|
| matmul accumulation is fp32 except integer paths (int32); L0C is not implicitly zeroed, so dropping the init flag accumulates the previous tile | [`pypto-pro-framework-findings.md`](pypto-pro-framework-findings.md) §18 — measured here in the local spelling: the last matmul of a K loop needs `AccPhase.Final`, and `Partial` throughout faults the device |
| local buffers stay full-tile sized; `valid_*` applies only at GM boundaries | [`../constraints/tail-validshape.md`](../constraints/tail-validshape.md), plus the local `compact=1` carve-out, which is more specific than the EasyASC rule and contradicts it for compact accumulators |
| decomposition legality tiers P / L / A / F | [`decomposition-primitives.md`](decomposition-primitives.md) — the same four tiers as exact / lossy / algorithmic / forbidden |
| UB second dimension must be 32-byte aligned | [`../constraints/vec-alignment-and-rotation.md`](../constraints/vec-alignment-and-rotation.md), measured here |
| "do not infer a device rule from a sample alone" | [`../constraints/sync-stitch.md`](../constraints/sync-stitch.md): "These samples demonstrate their recorded environment only"; and [`investigation-discipline.md`](investigation-discipline.md) §11's evidence order |
| `@vf` hardware overlaps store / load / compute streams even when the body reads sequentially | `ops/pypto-pro-op-develop/references/vf-reduction-perf.md` § "Scratch stores need a barrier before the next vector load", against the local `vf.mem_bar` (`language/_vf_api.py:242`). Only the *simulator treats the barrier as a no-op* half is untranslatable |
| `c220 vgatherb` needs uint16 operands for 16-bit data | [`../patterns/vec-ub-strip-gather.md`](../patterns/vec-ub-strip-gather.md) index-width rule. The "simulator accepts `half`, the on-box compiler rejects it" half has no local counterpart |
| bounded search budget — three candidates, at most two fixes at one boundary, counters in a state file | The orchestrator state machine already bounds this (`max_cycles_per_module`, the `capability_gap` → verifier adjudication path) |
| signed zero survives into FP4 (a negative rounding to zero emits nibble `0x8`) | Already recorded, explicitly as unverified, in [`pypto-pro-framework-findings.md`](pypto-pro-framework-findings.md) § EasyASC cross-port, alongside the cast-mask-width claim from the same section |
| a5pr = 28 cube / 56 vector cores | Independently measured here (`vector_core_num=56`); a cross-check, not a port |

### 5d. Numbers this KB is not allowed to transcribe

[`../constraints/arch-a5.md`](../constraints/arch-a5.md) states the rule: platform capacities are
read from the installed platform files at design time and **not duplicated into this KB**,
because a duplicated number detaches from the version that produced it.

| EasyASC item | Why not |
|---|---|
| the device capacity table (cube/vec core counts, L0A/L0B/L0C, L1, BT bytes, UB per SKU across a2/a3/a5/a5pr) | Transcription ban. What was portable from it is the *derived* content — the BT-slot bound and the `n_burst` negative, both in §3. **It was checked anyway**, row by row against `950PR_957x.ini`, because a table that can be checked cheaply should be: it is right about L0A/L0B, L0C, L1, BT, the 28/56 split and A2/A3's 192 KB UB, and wrong only about A5 UB. Usable as a cross-check; not as a source |
| estimator quick-models: L0C double-buffer bytes, `MAX_L0C_TILE_ELEMENTS_DBUF = 32*1024`, the "stable large-K point" `TILE_M=128, TILE_N=256, TILE_K=256, SPLIT_K=64` | Another DSL's estimator constants, unmeasured here. [`investigation-discipline.md`](investigation-discipline.md) §6 is explicit that a model ranks candidates and is not evidence; a transcribed tuning point would be read as a starting configuration |
| NZ panel footprint `ceil(N/C0)*stride_m*C0`, ZZ footprint `ceil(M/row_block)*align_N*row_block` | Layout-specific and partly in vocabulary this KB does not use (`ZZ`). The general principle — a strided footprint is not the product of its extents — was ported in §3 in its DSL-independent form |

### 5e. Process content belonging to the other project

EasyASC's workflow-state, repository-maintenance, and pattern-page-contract playbooks, its
`tools/` scripts, and its `refs/` decomposition layout are 100% that project's process. This
repository has its own equivalents (the four-stage orchestrator, `CONTRACT.md`, the pattern
index).

Its rule to **run simulator tests sequentially**, because concurrent simulator processes can
corrupt each other, has no counterpart — there is no simulator. The nearest live analogue,
serializing parallel agents against one board, is already held at
[`investigation-discipline.md`](investigation-discipline.md) §10.

---

## Re-reading this ledger later

A row that says *unverified* is an invitation, not a closed question. Each such claim carries
a named probe on its own page; running one converts the row to a measurement or deletes the
section. If you settle one, update both the page and this ledger — a ledger that still lists a
retired hypothesis is how a refuted claim gets re-adopted.
