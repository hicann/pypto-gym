# Vector tile alignment, buffer rotation, and sync — measured rules

Constraints that are cheap to satisfy and expensive to discover. Each was found by a
run that looked like something else: a shape-specific kernel bug, a bandwidth ceiling, an
arithmetic error, a device fault. They were measured on an Ascend 950-family A5 target with
the pypto-pro DSL unless a section says otherwise; each entry names the diagnostic that
identifies it.

Transferred from an earlier pypto-pro working set (`.agents/pypto-pro-op-kb/pitfall-records.md`,
2026-07-21 … 07-23) rather than re-derived. Provenance is kept because none of these are
in the installed API documentation, and a reader who doubts one should be able to find the
original run.

---

## `vf` tile width must be a multiple of 64 lanes, not 8

**Symptom.** A `@pl.vector_function` kernel using `vf.load_align` / `vf.store_align` over
`n_regs = ceil(D/64)` registers faults with **`507035`** (device error type 3) at sync — for
any `D` whose 8-aligned width is not also a 64-multiple. `D = 2049` rounds to 2056 and
faults; `D = 768 / 1024 / 2048` are already 64-multiples and pass, which masks it.

**Cause.** `load_align` / `store_align` move a **full 64-lane register**. The last register
starts at `(n_regs-1)*64` and runs off the end of a tile whose width is not a 64-multiple.

**Rule.** Size the tile to `align64(D)`. Then count UB slots honestly: double-buffered in and
out is **four** slots, so at 256 KB of UB a single vf tile needs
`4 * TR * align64(D) * 4 ≤ 262144`, i.e. `align64(D) ≤ 2048` at `TR = 8`.

## Reduction tiles: rows must be a multiple of 8

**Symptom.** softmax at `N = 2048` with `TR = 4` fails to compile — bisheng
`static_assert "Rows must be 32 bytes align"` (`pto_tile.hpp:1523`).

**Cause.** The `[TR, 1]` `DN` reduction tile at `TR = 4` is 16 bytes, not 32-byte aligned.
Eight fp32 rows is exactly 32 bytes.

**Rule — the constraint binds the *declared* extent, not the working one.** Declare the
reduction tile `[roundup(TR, 8), 1]` and narrow it with `pl.set_validshape`. That satisfies the
assertion while leaving the effective `TR` free, including 1.

Two operators reached this independently and it is the remedy both landed on:
`rms_norm` verified identical fp32 rounding at effective `TR` = 1, 2, 3, 4, 8, 16; `softmax`
declared `[8, 1]` and narrowed, which took every public last-axis case from a two-pass kernel
to single-pass. Original observation held at `TR = 16` (N=256) and `TR = 8` (N=1024).

> **Corrected.** This page first said to "reduce the buffer count rather than dropping TR
> below 8". That advice is wrong and was propagated before it was tested: at a long
> normalisation axis **no buffer count rescues the tile** — 8 rows of a D=8192 fp32 row is
> 256 KB against 248 KB of UB, so the tile is oversized before buffering is considered.
> Over-declaring is the escape; reducing buffers is not.

> **The assert's wording is misleading, and here is the rule in the compiler's own words.**
> `softmax` reported that `pto_tile.hpp:1523`'s *"Rows must be 32 bytes align"* actually
> constrains the **Cols** extent. `scatter` then hit the row-major form and the compiler printed
> the predicate itself:
>
> ```
> Tile<Vec, __bf16, 2048, 8, RowMajor, ...>
> pto_tile.hpp:1444: RowMajor + NoneBox => Cols * sizeof(T) % 32 == 0
> ```
>
> So the reliable statement is **`Cols * sizeof(T) % 32 == 0`** — the *contiguous* extent in
> bytes, whichever axis that is for the layout. For a `[TR, 1]` `DN` tile it is `TR`; for a
> row-major tile it is the column count. Two operators reached this from opposite layouts.
>
> It is a hard `static_assert` and it binds **every** tile of that shape, so in a mixed-dtype
> kernel the **narrowest** staging tile sets the floor: `scatter` charged its 2-byte staging tile
> against the width budget, which shrank `TB` and took working bf16 rungs from 16 columns to 8,
> regressing 14/20 to 8/20 before anything improved. Budget the narrow tile first.

## `.current()` does not advance the rotation — `.next()` does

**Symptom.** Adding a buffer changes performance by less than 2%, and multi-buffering looks
worthless.

**Cause.** `pl.make_tile_group(..., mutex_ids=[a, b])` declares N buffers, but rotation only
happens through **`.next()`**. `.current()` returns the same buffer every time, so a kernel
declaring 2–4 buffers and calling `.current()` in its loop is **single-buffered** — paying the
capacity and getting none of the overlap.

**Rule.** `.next()` for every *streaming* operand; `.current()` **only** to re-read an operand
that is deliberately resident across an inner loop after `.next()` loaded it outside.

**Why it is worth checking first.** Inert rotation produced two confident, wrong conclusions
in one session — "MTE2 is bandwidth-bound, pipelining cannot help", and "K=128 double-buffered
is worse than K=256 single" (75.8 vs 108.1 MMAC/µs). With real rotation K=128 won decisively
at 179.9 MMAC/µs. **Diagnostic: if adding a buffer moves performance under 2%, grep the loop
body for `.current()` before concluding anything about the pipe.**

Related correctness trap: an operand written once per outer iteration and read across the
whole inner loop needs a *second* buffer once the outer loop pipelines, or the next row's move
races the current row's reads.

## `auto_mutex` does not sync a bare `make_tile` that is *computed*

**Symptom.** A layernorm computes `maxdiff ≈ 5–7` — wrong, with no error raised — while its
normalization alone is exact.

**Cause.** `auto_mutex` tracks cross-op read-after-write hazards **only for
`make_tile_group`**. A bare `pl.make_tile` written by one op and read by a later one is not
synced, and the reader sees stale data. A *loaded* bare tile happened to work; a *computed*
one did not, which is what makes this look like an arithmetic bug rather than a sync bug.

**Rule.** Use `make_tile_group` for any tile in a cross-op dependency, even single-buffered —
`addrs=[A], mutex_ids=[k]`, call `.next()` once. Reserve bare `make_tile` for a transient
reduction scratch or its row-major alias.

## `vf.update_mask` per register costs 2.4x–6.5x — hoist it

**Symptom.** A `vf` kernel whose body is a handful of arithmetic ops runs several times
slower than its byte count implies, and profiles as vector-issue-bound rather than
bandwidth-bound.

**Cause.** The common register-loop idiom recomputes the lane mask every iteration:

```python
for r in pl.range(0, n_regs):
    valid = pl.min(LANES, n - r * LANES)
    m = vf.update_mask(valid, dtype=pl.DT_FP32)     # scalar min + mask update, per register
    ...
```

Only the *last* register of a tile ever needs a partial mask, so `n_regs - 1` of these are
pure overhead — and on a short body they dominate.

**Rule.** Build one `vf.create_mask(pattern=pl.MaskPattern.ALL, ...)` outside the loop and
use it for every register.

**Whether the tail path can then be dropped depends on what the inactive lanes' output is
used for, and getting this wrong corrupts results silently.**

> **The criterion: does an inactive lane's output take part in ADDRESSING?**
>
> * **No** — the lane's value is only stored. Dropping the tail path is safe: those lanes
>   compute garbage, and `pl.set_validshape` bounds the DMA so `pl.store` never transfers
>   them. This is the elementwise case.
> * **Yes** — the lane's value is used to decide *where* something is written or read
>   (an index for a scatter or gather, an offset, a pointer). **Keep a real tail
>   iteration.** Garbage in a value is discarded by the store; garbage in an *index*
>   addresses an arbitrary position in the tile, and nothing downstream rejects it — the
>   kernel passes its gate and writes to the wrong place.
>
> When in doubt, hoist the full-mask case and keep the tail iteration. That is correct in
> both branches and costs nothing when the tail is empty.

**Dropping the tail** is safe **only** when the tile width is a multiple of 64 lanes — this qualifies the no-tail branch above, not the keep-the-tail advice; keeping the tail is safe at any width. See the first rule on
this page. `load_align` moves a whole register, so with a non-64-multiple width the extra
lanes run off the tile and fault at `507035` instead of computing harmless garbage.

**Measured**, swi_glu on Ascend950PR_9579, same body, two separately `stamped()` names in
one module so neither could be served the other's binary:

| case | shape, dtype | per-register mask | hoisted | ratio |
|---|---|---|---|---|
| 3 | 4096x8192 bf16 | 91.06 us | 14.01 us | **6.5x** |
| 18 | 2x1023x4096 fp32 | 19.87 us | 3.85 us | 5.2x |
| 15 | 512x4096 fp32 | 6.58 us | 1.50 us | 4.4x |
| 11 | 3x7x13x1018 fp32 | 2.52 us | 0.93 us | 2.7x |
| 8 | 1538x1537 fp32, dim 0 | 8.36 us | 3.50 us | 2.4x |
| 5 | 2039x65520 fp32 | 655.86 us | 649.88 us | 1.01x |

**The gain scales with how far the kernel is from the DRAM roof, which is why it is easy to
dismiss.** Case 5 moves 801 MB and is bandwidth-bound at 1.23 TB/s, so the mask costs it 1%;
every case that fits cache pays the full 2.4–6.5x. A kernel tuned only on its largest case
will conclude the mask is free. cummin measured this lever at 28% on a scan body of ~20 ops;
on a 4-op elementwise body the same fix is worth an order of magnitude more, because the
overhead is fixed per register and the body it is amortised over is not.

**Cost, for the case where dropping the tail is legal at all** (inactive lanes' output is
only stored — see the criterion above; if it takes part in addressing, keep the tail
regardless of what these numbers say). An explicitly masked tail register (hoisted mask for
the full registers, one `update_mask` for the remainder) measured within noise of the no-tail
form — 3.65 vs 3.85 us on case 18, 1.54 vs 1.50 on case 15 — **so keeping the tail costs
essentially nothing.** These numbers were once written as "the tail path buys nothing and
costs a branch", which read as an argument for removing it; measured, it is an argument that
removing it is not worth the risk.

## Power-of-two *store* strides serialise on UB banks too — pad to an odd block count

**Symptom.** A kernel whose strided or scatter UB **stores** land at a power-of-two row
stride (32 / 64 / 128 elements) runs tens of percent slower on the board than its byte
count implies — while any simulator that models UB as flat memory shows a bit-exact,
fast kernel. Board-only, ~30% in the originating case.

**Cause.** UB is banked, on the store path as on the load path. The *gather*-side rule and
its pypto-pro measurements (stride 64 = 51.0 µs vs 65 = 7.0 µs) are in
[vec-scan-prefix-dependent.md](../patterns/vec-scan-prefix-dependent.md); the store side
was measured under EasyASC on the same silicon: padding an NZ fractal-row store stride from
64 to 65 rows took the kernel from **167.67 µs to 128.05 µs**, with the pad row excluded on
the copy that publishes the tile onward (the publishing copy takes a separate source
stride, so the downstream tile stays compact).

**Rule.** The same pitch rule as the gather side, applied to the tile a scatter/strided
store writes into: make the row pitch an **odd multiple of 32 bytes** — a tile row must be
a whole number of 32-byte blocks (`pto_tile.hpp:1444`), so an odd *element* count is not
the tool here — and exclude the pad at the publishing copy via the valid window rather
than carrying it onward. Board-validate the timing of any NZ / strided UB store; this
class is invisible to flat-UB simulation.

## A masked *continuous* store may round its active lanes up to a whole 32-byte block

EasyASC board measurement on the same silicon; **unverified for `vf.store_align` — treat
as a hypothesis and run the probe below before relying on either behavior.**

A masked continuous store rounded its active lane count up to a whole 32-byte block
(`align8(n)` lanes at b32), so the spill lands past the active lanes. That is harmless
while rows are written in increasing order — it lands on data the loop has yet to write —
and wrong the moment a row spans several lane groups with the groups as the outer loop,
because the last group of row `r` then lands on the already-written first group of row
`r+1`. The *scatter* store on the same hardware was lane-exact: its mask is lane-wise and
it writes exactly the active lanes.

**Probe** (one build): store `n < 8` fp32 lanes under mask at a row boundary with the next
row pre-filled with a sentinel, then read the sentinel back. If it was overwritten, masked
`vf.store_align` rounds up too — order the writes so the spill lands on not-yet-written
data, or store the boundary through `vf.scatter`.

---

## The one that is already fixed here, kept for its diagnostic

Factory-produced kernels collide in pypto's build cache, which is keyed on `co_name`: the
first shape is bit-correct and every later one returns garbage (`maxdiff ~3`, not a crash),
and reversing the call order flips which shape "works". `k.__name__ = ...` does not help —
`co_name` is fixed at `def` — and `exec` breaks `inspect.getsource`. Code-generate each kernel
to a real file under a unique `def` name.

Use a source-derived unique kernel name; see "Stale binary under an unchanged kernel name" in
[benchmark-scoring.md](../playbooks/benchmark-scoring.md) for the quieter converse, where an
edited body under an unchanged name is served a *stale* binary and the run comes back
byte-identical. (This pointer previously named a per-operator findings file that is not in this
repo, cited by an entry number -- which this KB's own convention forbids, because the numbering
collides across operator trees. Cite in-repo pages by title.) The diagnostic is worth
keeping: **an unchanged module-level reference kernel passing at the same shape where a
factory kernel fails is what exposes it.**

One myth was busted in the same investigation and should not be reintroduced: `inf`/`nan` do
**not** hang fixed-trip `vf` reductions. That symptom was the `co_name` collision's
out-of-bounds access, misattributed.

---

## Per-token scalar staging wants one whole 32-byte row per token

**Cross-DSL: EasyASC board measurement on Ascend 950. 未在 PyPTO-Pro 上验证——由
EasyASC 移植的假设 (unverified on PyPTO-Pro).** It does **not** override the
PyPTO-Pro measurement it sits next to; read that one first.

**What this KB already measured, and which stays authoritative.** An FP16
scalar-Tensor store is not safe at a 32-byte ownership boundary — accuracy 0.90
overall and 0.90625 at `block_dim=32`, while **64 B and 128 B boundaries were
exact** and a whole-tile `pl.store` was exact at all three
([framework-findings, A5 probe session](../references/pypto-pro-framework-findings.md)).
The conclusion there is: prefer the beat-complete tile store.

**What EasyASC adds, from the staging side rather than the store side.** Its
`constraints/a5.md` §12 reports that repeatedly moving *one scalar per token*
between GM and UB — `m > 1`, `n == 1` — into a compact `[1, L]` or `[L, 1]` UB
layout **can misaddress later rows on hardware while its Python simulator
passes**. Its prescription is to give each token one physical 32-byte UB row and
use only the first element of it:

```
scalar_ub : [L, C0]          # C0 = 32 B / sizeof(dtype); 16 for bf16, 8 for fp32
                             # logical slice stays [L, 1]; the backing stays [L, C0]
```

EasyASC is explicit that this is a **burst-pitch** requirement — a property of the
repeated one-element transfer — and *not* a general ban on unaligned scalar
addressing, which it says is fine for an isolated access.

**Why it is worth carrying.** It is the same 32-byte quantum as the measured
PyPTO-Pro finding above, reached from a different direction (staging pitch rather
than concurrent-store ownership), by a different DSL, on the same silicon. Two
independent arrivals at "give the scalar a whole beat" is the kind of agreement
that should raise your prior before you spend a board session. It also predicts
something the PyPTO-Pro measurement did not test: that the hazard survives even
with a single writer, because it is about burst addressing rather than about two
cores sharing a beat.

**Named probe:** stage `L` per-token fp32 scalars into a `[L, 1]` UB tile and into
an `[L, 8]` tile whose column 0 carries the payload, read both back, and compare
against the host values at `L` large enough to need several bursts. A clean `[L,
1]` result retires this section; a mismatch that appears only past the first burst
confirms it. Note the control requirement — the probe is void unless the same run
shows the checker can report a mismatch
([investigation-discipline §13](../references/investigation-discipline.md)).

## A distribution-mode store writes the whole register, mask or no mask

Independent corroboration, from a second DSL, of a silent failure this KB already
records — and a constructive consequence the original entry does not state.

**Already measured here.** A masked `vf.store_align(dist=INTLV_B32)` **ignores its
predicate entirely**, writing every lane; the store succeeds and the data is wrong
only in the lanes the mask was supposed to protect
([framework-findings, A5 probe session](../references/pypto-pro-framework-findings.md)).

**The corroboration.** EasyASC's `constraints/a5.md` §6.2 reports, on the same
silicon, that its `DIST_NORM_B8` store "writes the full 256-byte register even
when only a prefix mask is active". Same shape of defect, a different DSL, a
different distribution mode, and a *different* mask (a prefix rather than a
scatter pattern). Two arrivals make this look like a property of the
distribution-mode store path rather than an artifact of one lowering.

**The consequence worth acting on, which is about allocation rather than
correctness.** EasyASC states it as a sizing rule: the scratch a masked
distribution store targets must be allocated for the **whole register**, 256
bytes, not for the active lane count. That is the constructive form — the
PyPTO-Pro entry tells you the data will be wrong; this tells you the buffer will
also be too small, which is the failure that shows up as corruption of whatever
was allocated next to it.

**未在 PyPTO-Pro 上验证 for the sizing rule specifically** (the mask-is-ignored
half *is* measured here). To verify: place a sentinel immediately after a scratch
tile sized to the active lane count, issue a masked distribution store into it,
and read the sentinel back.
