# Synchronization and section handoff

## Rule

Use `make_tile_group(..., auto_mutex=True)` for rotation and intra-kernel
buffer ownership that the target API documents as auto-managed. Do not add
manual synchronization to the same managed dependency without evidence.

Cross-section or AIC/AIV handoff is not implied by `auto_mutex`. When the data
flow crosses engines or sub-blocks, copy the producer/consumer event sequence
from a matching official example for the installed SDK and validate it on the
target.

## On A5: which cube+vector fusion construct to reach for

Two different constructs are covered here, and only one of them hangs. Decide in
this order — the same order the adjudication further down this page arrives at:

1. **Try the generated pipeline path first.** `fwd_ids`/`bwd_ids` plus
   `@pl.pipeline.stage` plus `PipelineConfig` **does work** here, measured on the
   reference kernel and on a full MLA kernel
   ([framework-findings §17](../references/pypto-pro-framework-findings.md)).
2. **Fall back to single-sided launches with GM intermediates** when the fused
   form fails. This is the proven escape, not evidence that fusion is impossible.
3. **Do not hand-write a per-tile cross-core event sequence inside one `@pl.jit`.**
   That is the construct that hangs; see below.

### The construct that does not run: hand-written per-tile handoff

Measured, repeatedly. A single `@pl.jit` holding a `section_cube()` and a
`section_vector()` with a **hand-written** cross-core handoff on every tile
compiles and then dies with `aicore timeout`. One attention kernel was taken
through **thirteen hypotheses and nine mutation-ladder rungs** on that construct
and never ran once, even though every ingredient passed in isolation — the cube
half (including the transposed NT load and a dual-accumulator contraction) and
the register-level softmax were each proven separately. The reference Ascend-C
implementation of a sibling operator hit the identical wall on the identical
construct. This says nothing about the generated pipeline path in step 1.

### The proven fallback: decompose into single-sided launches

Each kernel is cube-only or vector-only and contains *no* cross-core
synchronisation at all; ordering comes from the launch boundary. Two operators
were rescued this way:

| operator | fused | single-sided |
|---|---|---|
| attention prolog | 0/20 | 11 launches → 19/20 |
| attention | never ran | 3 launches → 20/20, correct on the second device run |

The cost is the intermediates round-tripping through GM — up to 268 MB of score
matrix on the largest attention case — and that is real. It is also affordable
far more often than it looks, because the benchmark pays `0.3` per accurate case
*before* any performance term: a slow correct kernel scores, a fast one that
never runs does not. Reserve this for after the generated pipeline path in step 1
has been tried; once on the GM split, only revisit fusion if the split form is
already correct and profiling says the GM round trip dominates.

The wrapper constraint that comes with it: the benchmark times the wrapper, so
there must be **no launch inside a host loop** — a fixed number of launches,
each looping internally. See
[wrapper-boundary.md](wrapper-boundary.md).

## Review sequence

1. Draw each producer→consumer dependency and the memory space carrying it.
2. Mark which dependencies are covered by tile-group mutexes.
3. For every remaining dependency, cite the official API/example that defines
   the required pipe and event.
4. Allocate event IDs without overlap across simultaneously active pipelines.
5. Match physical slot count to the maximum number of in-flight work items.
6. Validate correctness before adding preload depth or extra buffering.

## Evidence

- auto-managed rotating Vec groups:
  [softmax_impl.py](../examples/samples/softmax/softmax_impl.py)
- retained cube→vector handoff with embedded correctness test:
  [fused_matmul_add_impl.py](../examples/samples/fused_matmul_add/fused_matmul_add_impl.py)
- retained vector→cube handoff with embedded correctness test:
  [vec_cube_abs_sqrt_matmul_impl.py](../examples/samples/vec_cube_abs_sqrt_matmul/vec_cube_abs_sqrt_matmul_impl.py)

These samples demonstrate their recorded environment only. The installed
`$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/` documentation and official samples
remain authoritative for pipe/event signatures.

---

## An unresolved conflict: must A5 cube↔vec round-trip GM?

Recorded, not adjudicated. A sibling DSL (EasyASC) targeting the same A5 silicon
states the opposite of this page's headline advice, and both statements rest on
work that was actually run. Leaving the disagreement visible is more useful than
picking a winner, because the two claims are about **different objects**.

| | claim | basis |
|---|---|---|
| this page, above | a fused single-launch cube+vector kernel with per-tile handoff does not run on A5; decompose into single-sided launches with GM intermediates | measured here, repeatedly — 13 hypotheses, 9 mutation-ladder rungs, never ran once |
| EasyASC `agent/references/constraints/a5.md` §4 | on A5 the cube↔vec handoff **stays on chip**; no GM workspace bridge is required, unlike the A2 model | asserted as an A5-vs-A2 authoring consequence; that page cites no probe for this specific line |

**They are reconcilable, and the reconciliation is the actionable part.** The
disagreement is not about whether the silicon has on-chip cube↔vec paths — it
does, and PyPTO-Pro exposes them: `pl.move`'s documented space table
(`pypto_pro/language/_api.py:196-206` in this checkout) carries `Acc (L0C) → Vec
(UB)` on the fix pipe and `Vec (UB) → Mat (L1)` on mte3. What failed here was one
*construct*: a hand-written per-tile cross-core event sequence inside a single
`@pl.jit`. And this KB's own
[framework-findings §17](../references/pypto-pro-framework-findings.md) records
that the **generated** path — `fwd_ids`/`bwd_ids` plus `@pl.pipeline.stage` plus
`PipelineConfig` — does work, on the reference kernel and on a full MLA kernel.

So the honest three-way statement is:

1. on-chip cube↔vec handoff **exists** and is reachable (both DSLs agree);
2. the **hand-written** per-tile event sequence hangs here (measured);
3. the **generated** pipeline path works here (measured, §17) — and is the thing
   to try before falling back to GM intermediates.

Read the "decompose into single-sided launches" advice above as *the proven
escape when the fused form fails*, not as evidence that fusion is impossible.
**Nothing here has been re-measured against EasyASC's claim** — it remains a
cross-DSL assertion about the same silicon, and the experiment that would settle
it is a PyPTO-Pro kernel built on the generated pipeline path at the tile
granularity the fused attempt used.

## Cube→cube re-feed: EasyASC's on-chip route has no `pl.move` spelling here

**未在 PyPTO-Pro 上验证——由 EasyASC 移植的假设 (unverified on PyPTO-Pro — an
assumption ported from EasyASC), and the PyPTO-Pro side of it is a checked
absence rather than a checked presence.**

EasyASC (`constraints/a5.md` §5) tells an author that when a later cube matmul
consumes an earlier one's result, the direct on-chip route
`mmad → L0C → l0c_to_l1 → l1_to_l0 → mmad` should be preferred, and the detour
`L0C → UB → L1` avoided as pure traffic and synchronization with no added
capability. It adds that its `auto_sync()` does **not** create the `FIX → MTE1`
edge between the L0C→L1 copy and the next consumer, so that edge must be fenced
by hand.

**In PyPTO-Pro the recommended route does not appear to exist.** `pl.move`'s
documented space table (`pypto_pro/language/_api.py:196-206`) lists exactly one
path out of `Acc`:

```
Acc (L0C) → Vec (UB)     fix
Mat (L1)  → Left (L0A)   mte1
Mat (L1)  → Right (L0B)  mte1
Mat (L1)  → Vec (UB)     v
Vec (UB)  → Mat (L1)     mte3
```

There is **no `Acc → Mat` row**. The only on-chip way back from an accumulator to
L1 is therefore `Acc → Vec → Mat` — precisely the detour EasyASC says to avoid.

Two consequences, and one of them is a warning about this very entry:

- **Design consequence.** Do not plan a cube→cube on-chip re-feed around an
  `L0C → L1` move. Budget either the `Acc → Vec → Mat` round trip (two moves, two
  spaces, and a vector-pipe touch that the fix→mte3 sequence has to be ordered
  against) or a GM intermediate.
- **Do not promote this into a capability claim without running the absence
  gate.** A missing row in one docstring table is weaker evidence than it looks —
  this KB has a standing finding that
  [documentation tables under-report](../references/investigation-discipline.md)
  (`vf.astype`'s table omits BF16 while the silicon does it). Before recording
  "PyPTO-Pro cannot re-feed L0C to L1", check the installed `_api.py` for other
  spellings, grep the official samples under `$PYPTO_DEVKIT_DIR/pro_ops/` with
  `find -L`, and build the minimal two-matmul probe.

The `FIX → MTE1` fencing half of EasyASC's advice is untested here and has no
obvious PyPTO-Pro counterpart to test, since the edge it fences is on the path
that appears to be absent.
