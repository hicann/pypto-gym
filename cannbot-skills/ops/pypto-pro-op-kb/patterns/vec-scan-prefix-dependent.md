# Prefix-dependent scan on the vector unit

**Use when** element `i` of the output depends on elements `0..i` along one
axis — cumulative min/max/sum/product, running argmin, `torch.cummin` and
friends. Topology `scan` in [topology-map.json](../topology-map.json).

**Validation:** validated skeleton for both dataflows below. Both were built,
compiled and recorded bit-exact on Ascend950PR_9579; see the
[retained validation record](../examples/validation-records.md). The
performance numbers quoted are measured on that SKU and are *not* a target for
another one without renormalising by vector-core count.

**Which parts apply to which scan.** This page was written from a *running-extremum* scan
(cumulative min/max, and the index that comes with it), so three of its sections are
specific to that family and do not apply to a plain associative scan:

| Section | Applies to |
|---|---|
| dataflow choice, lane/pitch layout, traffic floor, job count | **every** prefix-dependent scan |
| seeding and NaN ordering | only where the combine has a **non-total order** (min/max) |
| the index output and its tie-break | only **arg-variants** that emit a position |
| "the contraction rewrite does not transfer" | every combine except `+` on the values (`cumsum`) |

**If your combine is `+` over the values themselves (`cumsum`), read the contraction
rewrite first** — it is a different and simpler
construction, and this KB has a validated sample for it at
[`examples/kernel-index.md`](../examples/kernel-index.md) (`cumsum_matmul_impl.py`), which
rewrites the scan as `x @ U` on the cube unit, `U` upper-triangular ones.

**The criterion is linearity over (+, x), not invertibility** -- an earlier version of this
paragraph said invertibility and listed `cumprod` and `logcumsumexp` alongside `cumsum`,
which is wrong and would send an implementer to a sample that computes a different function.
`x @ U` sums; for `x = [2,3,4]` it gives `[2,5,9]`, which is `cumsum` and is neither
`cumprod` `[2,6,24]` nor `logcumsumexp`. `cumprod` has a log-domain rewrite only where every
element is strictly positive -- a single zero or negative kills it -- and `logcumsumexp` is
not a matmul at all. **Both of those belong on this page's own dataflows**, as does min-plus.

pypto has **no scan primitive and no register-to-register lane shuffle**.
`vf.shift_left/right` are bitwise. `pypto-pro-op-kb/examples/samples/vector_kernels/
cumsum_matmul_impl.py:4` states outright that `pl` has no scan/cumsum op and
rewrites cumsum as `x @ U` on the cube unit — that trick does not transfer to
min-plus and produces no index, so a vector scan has to be built from compare
and select.

## The decision that matters: which axis reaches the lanes

Everything follows from one question — *is the scan axis contiguous in
memory?* View the tensor as `[outer, red, inner]` where `red = shape[dim]` and
`inner = prod(shape[dim+1:])`.

### A. `inner > 1` — the scan axis is strided. Take it.

The inner axis maps straight to vf lanes and the scan is a running fold over
`red`, one row at a time. There is **no cross-lane movement at all**:

```
best, best_idx  <- row 0 (verbatim; see "seeding" below)
for r in 1..red-1:
    x   = load(row r)
    c   = combine_predicate(x, best)       # e.g. vf.le(x, best, m)
    best     = vf.select(x,   best,     c)
    best_idx = vf.select(cur, best_idx, c)
    store(row r) <- best, best_idx
```

Carry `best`/`best_idx` in **UB tiles, not registers**, so a `red` of any
length runs in one pass with no workspace. The tile group holding them must be
**single-buffered**: a rotating `make_tile_group` hands the next iteration a
different buffer and silently loses the carry. That bug produced wrong values
*and* wrong indices on every strided shape while the contiguous path passed,
which is a useful signature — a dual-topology operator failing exactly one
class usually means loop-carried state, not maths.

### B. `inner == 1` — the scan axis is contiguous. Put *rows* on the lanes.

A register holding 64 consecutive elements of one row cannot produce a prefix
without cross-lane movement, and the only cross-lane op is `vf.gather`, whose
source is UB rather than a register — so a Hillis-Steele step costs a UB round
trip. Instead, give each lane a **different row**:

```
tile[ROWS, WP] <- one 2-D DMA of x[r0:r0+ROWS, c0:c0+W]     # efficient
g = vf.arange(0, dtype=DT_UINT32); g = vf.muls(g, WP, mu)   # lane L -> row L
for j in 0..W-1:
    x = vf.gather(tile, g, m)        # column j across all ROWS rows, one op
    ... same compare/select as A ...
    vf.scatter(vtile, best, g, m)
    g = vf.adds(g, 1, mu)            # advance to the next column
```

Measured ~10 vf ops per 64 elements against ~36+ for a UB-round-tripping
log-depth prefix.

**But price `vf.gather` and `vf.scatter` before you build on them.** At the real
dependency depth they cost ~20 ns and ~18 ns per 64-lane register against well
under 1 ns for `vf.load_align` / `vf.store_align` — **20 to 35x**, not the
small factor a two-op microbenchmark suggests. **Where an inner loop crosses
lanes once or twice per register, UB addressing dominates it.** Measured on a
contiguous scan whose inner loop is one gather and two scatters per 64 elements:
**83% of the loop is addressing, 17% is the arithmetic** — at an identical op
count either way. The same arithmetic over contiguous accesses runs 6x cheaper. Isolate them by
swapping one at a time in a loop that keeps the real dependency depth, not in a
loop with two ops and nothing to overlap.

The consequence is **not** "do the transpose in the DMA" — that was measured
and it loses. A `[ROWS, 1]` strided descriptor costs 0.488 ns/element/core at
`ROWS = 64` against 0.109 for the bulk one, and the transpose needs three
passes of them, so it lands at 1.51 against 0.942 for the gather/scatter form.
`ROWS = 256` is cheaper per element but quarters the job count, which is the
other binding constraint, for a net loss.

The consequence is: **the arithmetic is ~5% of the cost, so do not transpose at
all.** That reopens the log-depth prefix — but only if a lane shift can be
made cheap, and on this hardware there are exactly two spellings of one:

* **`vf.gather`** — measured, ~20 ns per register. A six-step prefix built on
  it costs **2.68 ns/element/core** against 0.942 for the gather-and-scatter
  transpose form. Dead.
* **`vf.load_align` at a shifted offset** — **does not work.** It faults the
  vector core (error 507035) even when every shift is 32-byte aligned, and an
  explicit `vf.mem_bar(mode=pl.MemBarMode.VST_VLD)` between the store and the
  load does not help. "Align" means register-width, and every Hillis-Steele
  shift (1, 2, 4, 8, 16, 32) is smaller than a register.
* **`vf.load_unalign`** and **`vf.store_unalign`** — two three-call protocols
  that shift on opposite sides (in on the load, out on the store). Both exist;
  note the load trio sits ~1400 lines from `load_align` in the API file, so
  grep the whole file for `unalign` rather than the neighbourhood of the
  aligned call.
* **A second tile group at a 32-byte-offset base address** reaches shifts of
  8, 16 and 32 lanes with an ordinary `load_align`, because tile bases are
  32-byte aligned = 8 fp32 lanes. It cannot reach 1, 2 or 4, so it only helps a
  blocked scan whose intra-8 step comes from somewhere else.

**Not candidates, despite the names:** `vf.shift_left` / `vf.shift_right` are
bitwise *within* each lane (`dst[i] = src[i] << shift`), not lane movement.

**Measured outcome: the log-depth family is dead on this hardware, and the
cheapest way to find that out is to price the round trip with no shift in it
at all.** Six store/load round trips carrying no shift cost **1.537
ns/element/core** against 0.942 for the gather-and-scatter transpose form — so
the floor beneath every shift spelling already loses, and neither unaligned
spelling needed measuring.

The reason is a trap worth stating on its own: **a store into UB followed by a
read of the same address inside a vector function requires
`vf.mem_bar(mode=pl.MemBarMode.VST_VLD)`** — omit it and the kernel faults the
vector core — and the barrier, not the store or the load, is what costs.

Per 64-lane register on this SKU: `vf.gather` ~20 ns, `vf.scatter` ~18 ns, a
barriered UB round trip ~16 ns, an aligned `load_align`/`store_align` under
1 ns, the arithmetic ~0.3 ns. **Every way of moving data across lanes costs
15-20 ns; every way of not moving it costs under 1.** Design accordingly: pick
the dataflow that touches UB non-contiguously the fewest times per element, and
do not expect a cheaper spelling to exist.

**The padded pitch is not cosmetic.** `WP` must not be a power of two.
Measured, same loop, 4M elements:

| gather stride (elements, fp32) | time |
|---|---|
| 64 | **51.0 us** |
| 65 | 7.0 us |
| 68 | 9.8 us |
| 72 | 15.9 us |

A stride of 64 fp32 is 256 B and lands all 64 lanes on one UB bank — ~5-7x the
cost of a well-chosen stride, and by far the largest single effect in this
topology. Note a tile row must also
be a whole number of 32-byte blocks (`pto_tile.hpp:1444`), so the pitch has to
be the smallest legal **odd multiple of 32 bytes** — not simply `W + 1`.

**Pad the pitch of the tile the gather indexes, in *that tile's* dtype.** This
is easy to get wrong and expensive: a narrow input widened to fp32 is gathered
from the *work* tile, so it is the work tile's pitch that matters. 80 elements
is 160 B as fp16 (5x32, odd, fine) and 320 B as fp32 (10x32, even,
conflicted) — worth 1.5-2x on every narrow case when it is wrong. No single
pitch serves both widths: conflict-free needs `pitch/8` odd for a 4-byte tile
and `pitch/16` odd for a 2-byte one, and `8*odd` and `16*odd` are disjoint. Use
different pitches for the native and work tiles; `pl.cast` bridges them through
`valid_shape`.

## Seeding: never fold row 0 against a sentinel

For min/max the obvious seed is `+inf`/`-inf`. It is wrong wherever NaN must
participate, because no sentinel is smaller than NaN under the order these
operators need. Store row 0 verbatim with index 0 and start the fold at row 1.

## Parallelism is the trap, not the maths

Both dataflows above parallelise over something the scan axis is *not*:

* A gives `outer x ceil(inner/chunk)` jobs. Every `dim == 0` case has
  `outer == 1`, so make the chunk width a **runtime argument** and widen the
  grid until it reaches a few times the core count. The tile stays a
  compile-time shape; only the chunk the job covers changes.

  But the floor matters as much as the target, because each job issues one DMA
  per reduction row and a narrow chunk buys jobs with transfer size. Swept over
  a 3.2 GB case: a 128-element floor gave 128 jobs and 263 GB/s, 512 gave 32
  jobs and **604 GB/s**, 2048 gave 8 jobs and 487. Transfer size wins until the
  job count collapses, and it collapses at a different place for each shape, so
  the policy needs two stages — hold the wide floor unless it starves the array
  outright.
* B gives `ceil(outer/ROWS)` jobs, and achieved bandwidth is almost exactly
  **linear in job count** until it saturates at the core count — measured
  ~26.5 GB/s per job for 4-byte dtypes and ~17 for narrow, across five cases
  spanning 5 to 47 jobs. Every low-`outer` case is starved, not slow, and that
  single number is enough to predict what any fix is worth before writing it.
  A 1-D input (`outer == 1`) is the extreme: **one lane of one core**, measured
  63 ms for 1M elements against a 6 us budget.

The fix is a two-pass segmented scan, and there are two forms of it. **Check
which one your shapes admit before costing the work**, because the cheap form
is usually inapplicable.

*Cheap form.* View `[outer, red]` as `[outer*P, red/P]` with `ROWS % P == 0`,
so a job holds whole rows and every segment a lane needs is inside it: no
cross-core carry, no workspace, no barrier. **It requires `P` to divide
`red`.** That is a far narrower condition than it looks, because **real reduction
extents are overwhelmingly odd or prime** — check it against your own shapes
before designing for it. Measured across one task's ten job-starved cases, only
two admitted any `P > 1`; the other eight had `red` of 1023, 769, 513, 4001,
1000003, 1013, 2049, 2049, every one odd. And the two that qualified had the
highest `outer` of the group — i.e. they needed it least. Reduction extents in real case
lists are routinely prime.

*General form.* Put `ROWS` consecutive **rows** on the lanes at a fixed segment
`s`. Jobs become `ceil(outer/ROWS) * P` and the tile `x[r0:r0+ROWS, s*T :
s*T+W]` stays rectangular for any `T`, with a short final segment. The carry
now crosses jobs, so pass A writes each segment's aggregate to a small
`[outer, P]` GM workspace and pass B reads it back — two launches, still no
barrier, since the launches serialise. Cost is one extra read of the input
(~17% of the traffic for a 2-byte dtype) plus the workspace.

Neither form helps a 1-D input. With `P <= ROWS` the job count is at most
`outer`, so a 1-D case goes from one *lane* of one core to `ROWS` lanes of one
core and no further; only `P > ROWS` moves it off a single core.

## Index outputs

If the operator emits an index alongside the value:

* **Keep indices in int32.** They cannot be carried in fp32 unless the axis is
  provably below 2^24 — and pypto has a gap that pushes you toward fp32
  anyway: `vf.gather`/`vf.scatter` cast the index register to the *data* dtype
  while the intrinsic wants `vector_u32`, so an int32 data payload does not
  compile. The order-preserving workaround is `u = x XOR 0x80000000` into
  uint32, which makes the data dtype uint32 and the cast correct.
* **int64 indices cost one instruction, not two.** Allocate the output as
  int64, alias it `Tensor.view(torch.int32)` on the host (zero-copy), and emit
  both words with a single `vf.store_align(..., lo, zeros, m,
  dist=pl.StoreDist.INTLV_B32)`. The high word is zero whenever the index fits
  in int32.
* **Predicate widths are convertible, but check whether it buys anything.** If
  the values are 16-bit and the indices 32-bit, a `vf.le` on the values yields
  a 128-lane b16 predicate while the index `select` wants a 64-lane b32 one.
  The converter is `vf.interleave` — a unified op whose MaskReg overload
  re-spaces predicate bits, with `dtype=` naming the **finer** width:
  `lo, hi = vf.interleave(p16, p16, dtype=pl.DT_UINT16)`. Measured correct on
  128/128 lanes. **Verify the premise before spending the instruction.**
  Measured: this was expected to remove a whole fp32 round trip and instead ran
  *slower* on every target case, because the round trip was not what cost — a UB
  bank conflict on the work tile's pitch was, and an earlier fix had already
  collected the value by that route.

* **Ties are usually graded exactly.** `thresholds.py` gives int64 a threshold
  of 0 and `compare.py` routes integers to exact equality, so a tie-break
  convention that is off by one fails the case outright regardless of how
  right the values are.

  **Establish the tie-break from the golden, never from the prose.** Where an arg-scan
  emits a position, the reference implementation and the written specification disagree
  often enough that this is worth treating as the default assumption rather than the
  exception — measured instance: one task's `desc.md` said the **first** index wins while
  its own golden returns the **last**. Run the golden on a deliberately tied input and read
  the answer off it.

## Measure the traffic floor before arguing about feasibility

The most useful half-hour on this operator was a kernel that moves exactly the
contractual bytes and does **nothing else** — no scan, no compare, no
cross-tile state, no gather — timed at every case size and fed through the
benchmark's own scoring. It is a hard lower bound for any correct
implementation and it costs one NPU round trip.

It overturned two separate impossibility arguments that had been built on
traffic arithmetic. **Traffic arithmetic is a real law — and still bounds nothing
useful while you are far from the bound it states.** Measured: an operator whose
outputs genuinely are 6x its input bytes at a 2-byte dtype, which really is fixed
once the index dtype is declared, shipped a kernel 2 to 34x off that floor. Traffic
was never what limited it. A ceiling derived from the levers you have enumerated is not a ceiling; it
is a statement about your list.

The floor table, sorted by ratio-to-floor rather than by score, is also a far
better work queue than anything derived from SOL: it separates "starved of
jobs" from "slow per element", which look identical in a score column.

## Two things that dominate the time, neither of them the scan

1. **`vf.update_mask` in a hot loop costs 3.5x.** (update_mask, arange, store)
   over 67M elements: 550.75 us. Hoisting the mask: 398.06 us — which equals
   the same loop with no vf at all (397.65 us). Create masks once per vector
   function and handle tails with tile padding and `valid_shape`.
2. **Know which bandwidth regime you are in.** On this SKU a streaming copy
   reaches ~1.2 TB/s, but a working set under ~130 MB reaches 4–5 TB/s. Small
   and medium cases are therefore **vector-issue-bound**, and op count — not
   traffic — is what to minimise. Check before tuning tiles.

## Keep the recurrence in registers — the barrier is not the whole story

**未在 PyPTO-Pro 上验证——由 EasyASC 移植的假设 (unverified on PyPTO-Pro — an
assumption ported from EasyASC).** Board-probed under EasyASC on Ascend 950. It is
carried here because it **narrows** a rule this KB already states, and narrowing a
rule in the unsafe direction is the kind of correction that is expensive to
discover for yourself.

**What this KB already says.** A scratch store must be fenced before the next
vector load reads it
(`ops/pypto-pro-op-develop/references/vf-reduction-perf.md` § "Scratch stores need
a barrier before the next vector load"). The local fence is `vf.mem_bar`, whose
modes are `VST_VLD`, `VST_VST` and `VV_ALL`
(`pypto_pro/language/_vf_api.py:242-252`). Read on its own, that section implies
the barrier is *sufficient* — put the fence in and the round trip is safe.

**What EasyASC reports, and it is the stronger claim.** For a scan, cumsum, or
row-wise recurrence inside a vector function, keep the loop-carried state in
registers. Specifically (`constraints/a5.md` §1, `facts-authoring.md:147`):

- **Do not use one UB tensor as both the per-element base-value source and the
  prefix / reverse-prefix destination inside the same vector-function loop.** A
  `vf_barrier(STORE, LOAD)` "orders local memory streams, but it does not turn
  that UB read/write alias into scalar program order on hardware." The fence is
  present and the aliased recurrence is still wrong.
- **Separate source and destination UB tensors are not established as equivalent
  to register accumulation either.** That is the part worth carrying: the obvious
  fix for the aliasing rule — stage the base values in one tensor, write results
  to another — **also failed the random-input hardware probe.** EasyASC's retained
  position is to treat a staged UB recurrence as a hardware-validated exception,
  never as a default.

**Why the input distribution is load-bearing here.** The failure was seen on
*random* inputs. A scan probed on a ramp, a constant, or small integers can be
right by construction — the recurrence's wrong intermediate and its right one
coincide for structured data far more often than they do for random data. This is
[investigation-discipline §13](../references/investigation-discipline.md) in its
data form: a green probe on a friendly input is not a green probe. Any test of a
staged recurrence has to run random inputs, and has to show the checker going red.

**Design consequence, in this DSL's terms.** Where the scan state can live in
`vf` registers across the loop, keep it there and do not stage it through UB for
readability or for register pressure without measuring. Where the base vector
genuinely must be staged, use distinct source and destination tiles — that
remains the least-bad staged form — and treat the result as unvalidated until it
has been run on the board against random inputs.

**Named probe to settle it here:** one `vf` scan over a tile, built three ways —
register-carried accumulator; one UB tile as both source and destination with
`vf.mem_bar(VST_VLD)` between; separate source and destination UB tiles with the
same fence — all three run against a CPU reference on uniform random input at a
length that needs several loop iterations. If the two staged forms match the
register form bit-for-bit, this section is retired for PyPTO-Pro; if either
diverges, it is promoted to a measurement.

## Where this was built

The [retained validation record](../examples/validation-records.md) covers both dataflows
and four dtypes; the original generator was not retained as a reusable KB artifact.
`probe/probe_{bw,ops,scan,stride}.py` (every measurement above),
`DESIGN.md` §5 for the honest performance postmortem.
