# Index-driven writes: the owner model

**Topology:** `gather-scatter-indexing`. Use when an index tensor decides *where*
an output element is written (scatter, `index_add`, `index_put`, one-hot
construction, embedding-gradient accumulation), as opposed to where an input is
read from (`gather`, `index_select`, embedding lookup — the same decomposition
applies, but the write-race section does not).

Validation: **validated skeleton for the inner-tile layout**, fp32 update mode,
measured bit-exact against torch on Ascend950PR_9579. Conceptual for the
owner-batch and rank-1 layouts.

**Evidence:** [retained validation record](../examples/validation-records.md), which keeps
the fp32 inner-tile result and its explicit exclusions separate from the conceptual layouts.

**Read the scope literally.** The record validates **the owner model and the inner-tile
addressing**, in fp32. It does **not** validate narrow-dtype accumulation: `float16` `add`
misses the gate on MARE while its MERE stays small, which is accumulation order, not
addressing. Nor does a bit-exactness figure recorded against a **golden** transfer to the
kernel — a golden settled offline before any kernel existed says nothing about the kernel
that later consumes it. Match every claim to the artifact it was measured on.

---

## 1. Factor about the indexed axis, then pick an owner

For an operator that writes `y[..., index[i], ...]`, split every tensor about the
indexed axis `d` into three factors:

    outer = prod(shape[:d])      K = shape[d]      inner = prod(shape[d+1:])

so the tensor is `[outer, K, inner]` and the operation is

    y[o, index[o, u, i], i]  <-  f(y[...], updates[o, u, i])

**An owner is one `(o, i)` coordinate.** Every output element belongs to exactly
one owner, and two owners can never write the same element, because the index
only ever moves an element along `K`. That single observation buys three things
at once:

* **no cross-core write race** — owners are the unit of parallel decomposition,
  so no atomics, no locks, no duplicate-write arbitration between cores;
* **a determinate tie-break for duplicate indices** — an owner consumes update
  positions `u = 0, 1, ..., U-1` in ascending order, which reproduces torch's
  last-write-wins exactly (verified bit-exact across ranks 1-8, dims 0-7, five
  reduce modes and five dtypes);
* **a rank-agnostic kernel** — rank appears only in the host-side product that
  computes `outer`/`K`/`inner`, so one kernel covers rank 1 through 8 and every
  legal `dim` without a shape guard.

Do not guard on rank or on a shape. Guard on the relationship — `indices.rank ==
data.rank`, `indices.shape[a] <= data.shape[a]` — because the declared range is
routinely wider than the cases you can see, and has been observed wider than the
documentation.

## 2. `vf.scatter` writes UB, so the whole indexed axis must be resident

`vf.scatter(base_ptr, src, index, mask)` writes register lanes to UB at
**element** offsets from a tile base. The index may land anywhere in `[0, K)`, so
the tile has to hold the entire `K` axis for the owners it is processing. That
gives the constraint that decides the whole design:

    lanes x K x sizeof(accumulator) <= UB working budget

On A5 (UB 253952 B) with a ~128 KB working budget and fp32, that is
`lanes <= 32768 / K`. `K = 512` still fills a 64-lane register; `K = 8192` leaves
four lanes. **This does not improve with a cleverer tiling** — it is inherent to
holding the indexed axis on-chip — so price it before designing, and expect
large-`K` shapes to be scatter-issue-bound rather than bandwidth-bound.

## 3. Three layouts, chosen by `inner`

| `inner` | UB layout | lane meaning | scatter offset |
|---|---|---|---|
| large | `[K, W]`, `W = min(inner, budget/K)` | inner positions | `idx*W + lane` |
| 1 | `[R, K]`, `R` owners batched | owners | `r*K + idx[r, u]` |
| 1, and `outer == 1` | `[Kc]`, one core-slice of `K` | update positions | `idx - k0`, masked to the slice |

`vf.scatter` documents that **two lanes writing the same address in one call is
undefined** ("index 中的值必须唯一"). The first two layouts satisfy that by
construction and not by luck:

* inner-tile: `0 <= lane < W`, so the low part of `idx*W + lane` separates every
  lane whatever the indices do;
* owner-batch: lanes are distinct owners `r`, and `idx < K`, so `r*K + idx` never
  collides.

The rank-1 layout has no non-indexed axis left, so its lanes must be update
positions, and two of them **can** carry the same index. That case needs an
explicit duplicate-resolution step; do not let the general argument paper over
it. A random index draw makes duplicates rare, so a test set will usually not
catch a mistake here.

## 4. Reduce modes are a masked read-modify-write, not a `select`

`vf.gather` **does** permit duplicate indices, so the general form of one update
step is

    cur = vf.gather(y_tile, off, m)
    ... combine cur with the update ...
    vf.scatter(y_tile, new, off, m)

**Every scatter needs a `vf.mem_bar()` after it.** Two `vf.scatter`
instructions to the same UB address are *not* ordered relative to each other,
so the ascending-`u` traversal that guarantees last-write-wins is only
guaranteed on paper unless a barrier separates the writes. Measured on
Ascend950PR_9579 with `K=4, U=16` so that every column carried duplicates:
plain scatter left **3 of 256** elements holding the earlier write; the same
kernel with `vf.mem_bar()` after each scatter was **0 of 256**. On realistic
index draws the error rate is around 1 in 1000 — low enough to survive a casual
check and still fail on inputs you never inspect, which is precisely why it
belongs in a pattern page.

For `amax` / `amin` the combine collapses into the mask itself, which is both
faster and avoids `vf.select` entirely:

    wm  = vf.or_(vf.gt(upd, cur, m), vf.ne(upd, upd, m))   # amax
    vf.scatter(y_tile, upd, off, wm)

The `vf.ne(upd, upd, m)` term is not decoration: it makes a NaN update win, which
is what `torch.maximum` does. A comparison-only formulation drops NaNs silently.

This matters on A5 because **int64 has no VF arithmetic and no `vf.select`** —
every `Intrinsic_vectorized_*` entry in the platform ini stops at 32 bits — but
it does have VF compare and VF move. So the masked form gives int64 `amin`/`amax`
and `update` for free, and only int64 `add`/`multiply` are left without a vector
path.

## 5. Accumulate wider than you store

Whether a reduce mode needs a wider accumulator is a measurement, not a guess,
and the intuitive answer was wrong here: `multiply` in fp32 was bit-exact, while
**`add` in fp16 failed the gate** (MARE 1.61e-01 against a 9.77e-03 limit) and
needed an fp32 accumulator to become bit-exact. torch's CPU `scatter_add_`
widens half to its `acc_type`; matching the reference's accumulation width is
part of the contract. Settle it offline by replaying the reference and grading
with the real comparator before spending device time.

Integer payloads are graded at threshold **0** — exact equality — so never route
one through a float accumulator to reuse a float code path.

## 6. Index dtype is width-matched to the data

`vf.scatter`/`vf.gather` want a **16-bit** index for b16 data, 32-bit for b32,
32- or 64-bit for b64. For narrow dtypes that caps the working tile at 65536
elements, which is tighter than UB alone would allow — so the b16 budget is set
by the index width, not by the memory.

An int64 *index* forces the scalar fallback in both reference tileops. Cast
indices down to 32 bits inside the kernel; the value range is bounded by `K`.

## 7. Do not reach for `pl.scatter`

The tile-level `pl.scatter(out, src, idx)` with FP32 data and INT32 indices — its
own documentation's recommended combination — dispatches to a **per-element
scalar loop**, not to hardware `vscatter`. Only a 2-byte index reaches the
vector path. Use the register-level `vf.scatter` inside `@pl.vector_function`
and keep control of the addressing.

## 8. Failure signatures

| Symptom | Cause |
|---|---|
| device error 507035, no diagnostic | a scatter offset landed outside the tile. Clamp offsets with `vf.maxs`/`vf.mins` while bringing a kernel up: a probe that faults tells you nothing, a probe that returns a wrong number tells you where |
| every element wrong, including a bare load→store echo | tiles built with `pl.make_tile` instead of `pl.make_tile_group`; `auto_mutex` synchronises tile *groups*, so raw tiles get no MTE↔V ordering at all |
| ~40 % of elements wrong | the UB row stride was guessed. With `set_validshape` narrowing a tile, rows stay at the **declared** width, not the valid one |
| a handful of elements wrong, always where an index repeats | no `vf.mem_bar()` between scatters to the same address (§4) |
| `ParserSyntaxError ... incompatible constructor arg` | a `pl.TileType` was built in a Python helper called from the kernel body. The body is parsed, not executed — construct tile types inline |
| `OSError: could not get source code` | the kernel module was run through `exec`. `@pl.jit` resolves kernels by `inspect.getsource` and needs a real file |

## 9. Related

* [constraints/memory-layout.md](../constraints/memory-layout.md) — the address
  table this pattern's UB budget has to appear in, and the rule that shared
  `mutex_ids` do not make two byte ranges safe to alias.
* [constraints/precision.md](../constraints/precision.md) — the accumulation
  and integer-threshold rules §5 depends on.
* [constraints/wrapper-boundary.md](../constraints/wrapper-boundary.md) — the
  `outer`/`K`/`inner` factorisation is host-side *metadata* arithmetic over
  shapes, and the reshape it implies is a pure view of a contiguous tensor.
  Anything that moves data stays in the kernel.
