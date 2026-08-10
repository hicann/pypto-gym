# Row reduction followed by row broadcast

## Applies when

Each row independently reduces its last dimension to one scalar and then uses
that scalar across the same row. Softmax, row normalization, and row scaling
have this topology.

## Dataflow

For stable softmax over a `[rows, columns]` tile:

1. reduce each row to `row_max`;
2. broadcast and subtract `row_max`;
3. apply `exp`;
4. reduce each row to `row_sum`;
5. broadcast and divide by `row_sum`.

The reduction result has logical shape `[rows, 1]`. Partial row or column tiles
must carry a valid shape through every reduction and broadcast. Exact tile
types, memory placement, and synchronization must match the installed SDK.

## Decision checks

- The reduced axis is the last axis and the scalar is consumed by the same row.
- Rows are independent and can be distributed without cross-row state.
- Padding cannot participate in `max`, `sum`, or the final store.
- The selected implementation surface (tile operation or vector function) has
  documented reduction, predicate, and tail semantics for the target.

Do not use this pattern for a global reduction, a reduction shared across rows,
or a reduction streamed across multiple score chunks. The last case uses the
conceptual recurrence in [online-softmax-tail.md](online-softmax-tail.md).

## Sizing the row tile — two alignment rules, one of which has to be escaped

Both come from the same `pto_tile.hpp` static assertion: for a NoneBox tile, a
**RowMajor** tile's `Cols * sizeof(T)` and a **ColMajor** tile's
`Rows * sizeof(T)` must be multiples of 32. They land on opposite dimensions of
this pattern.

* The data tile `[TR, W]` is RowMajor, so **`W * itemsize` must be a multiple of
  32** — W a multiple of 8 for fp32, 16 for fp16/bf16. A ladder of multiples of
  16 satisfies all three at once.
* The reduction tile `[TR, 1]` that a `dim=0` reduce writes carries
  `layout=pl.DN`, which is ColMajor, so **`TR * 4` must be a multiple of 32** —
  TR a multiple of 8.

The second rule, taken literally, makes a long normalisation axis
unimplementable: eight rows of a D=8192 fp32 row is a 256 KB tile against a
248 KB UB, before any second buffer. **Over-declare the reduction tile instead.**
Declare it `[roundup(TR, 8), 1]` and narrow it at runtime; the declared row count
is what the assertion sees, the valid shape is what the reduction touches, and
the data tile keeps its true size.

```python
rt  = pl.TileType(shape=[8, 1], dtype=pl.DT_FP32, layout=pl.DN, valid_shape=[-1, -1])
rmt = pl.TileType(shape=[1, 8], dtype=pl.DT_FP32, valid_shape=[-1, -1])   # same address
...
pl.set_validshape(rr, [vr, 1])          # vr <= TR, and TR need not be a multiple of 8
pl.sum(rr, sq, oo, dim=0)               # oo doubles as the workspace
pl.set_validshape(rm, [1, vr])
pl.div(rm, rm, cols); pl.add(rm, rm, eps); pl.rsqrt(rm, rm)   # row-major alias
pl.expand_mul(oo, ii, rr, dim=0)        # row scale, read back through the DN alias
pl.expand_mul(oo, oo, gam, dim=1)       # a per-column vector broadcasts on dim=1
```

The `[1, TR]` row-major alias at the same address is not optional: the scalar
div/add/rsqrt chain faults at runtime on the DN layout.

Two allocator facts that cost a device round trip each: every Vec tile address
must be a multiple of 32, and one small allocation (a `[1, D]` coefficient row at
D=2 is 4 bytes) shifts every later tile off the grid, with the error naming the
offending address rather than the allocation that moved it; and a tile whose last
dimension is small enough — `[2048, 2]` bf16 is 4 bytes of Cols — cannot be
declared at all, which removes "re-view a packed buffer at a D-element row pitch"
from the toolbox for a very short axis.

## Where the time goes

Measured on `Ascend950PR_9579` for `rms_norm`, which is this pattern with the
`max`/`sub`/`exp` steps removed:

* **The fp32 form is memory-bound and the arithmetic is free.** At a 268 MB
  working set a pure copy ran 217.0 us and the full reduce-and-broadcast chain
  219.7 us. Five `(W, TR)` tilings of that same tensor spanned 0.8%. Tile
  geometry decides UB capacity and, at that size, nothing measurable.
* **The widen-compute-narrow form is compute-bound, by its vector-pass ratio.**
  fp32 runs 4 full-tile passes per 4 GM bytes; fp16/bf16 runs 6 fp32-width passes
  per 2 GM bytes — 3x the vector work per byte, and 3x the measured slowdown
  (1.8–4.1 TB/s against 1.0–1.2). The lever there is removing a pass, not a
  buffer.
* **Achieved bandwidth tracks the working set against the 128 MiB L2**, not the
  tiling: 5715 / 3309 / 1237 GB/s at 33 / 107 / 268 MB. Quote a bandwidth number
  with its working set or it does not transfer.

## When the reduced axis is very short

Below about 32 bytes per row this pattern degrades badly and the fix is not
inside it. A `[TR, W]` tile with `valid_shape [TR, D]` issues TR DMA bursts of
`D * itemsize` bytes, and the strided-load knee is 128 bytes per row; at D=2 bf16
a *pure copy* measured 117 GB/s, and the reduce-and-broadcast chain on top runs
every vector op over 2-element rows. Packing several rows per tile row makes the
load contiguous but destroys the per-row reduction, and the compact re-view that
would restore it is blocked by the Cols alignment rule above. Record the cost
rather than assuming a tiling escape exists.

**Measured escapes now exist for the vf form** (rms_norm, Ascend950PR_9579,
2026-08-06, 20/20 public + 19/19 hidden-guess): keep the reduction in
*registers* instead of re-viewing tiles — load rows contiguously and restore
per-row sums with in-register shuffles, one kernel family per D band.
- D=2: de-interleave even/odd lanes (`DINTLV_B32` for fp32; `vf.astype`
  half-register split for b16), one row per lane — case `[1000003, 2]` bf16
  went 412.5 µs → **9.7 µs** (t_hw 2.50, SOL 0.757).
- D=128: batch 64 rows per tile with a two-instruction pass-1
  (`mul`+`mul_add_dst`) — case `[104448, 128]` fp32 went 753.15 µs →
  **42.8 µs** (SOL 0.608).
- 3≤D≤8: contiguous flat load + `vf.gather` column collect over 8-row groups.
  The gather cost model (~20 ns/register) predicted parity with rerouting to
  the D=128 family; measurement settled it: gather route wins 2.1–3.4×
  ([1e6,3] fp32 127.7 vs 274.9 µs; [1e6,5] fp32 132.5 vs 274.9; [1e6,3] bf16
  133.9 vs 449.8). Model intervals overlapped — only the on-board pair run
  could rank them; keep that measurement step when porting.
Every scratch store→load between these shuffles needs `vf.mem_bar()`
(VST_VLD): auto_mutex orders nothing inside the V pipe, and the omission reads
clean until a prime-width case goes stale.

## Validation status

The retained [softmax implementation](../examples/samples/softmax/softmax_impl.py)
is the **validated skeleton** for the tile-operation form. A vector-function
form is a separate implementation choice and must be validated against the
target SDK's current API.

The sizing rules, the over-declared reduction tile and the measurements above
are preserved in the [retained validation record](../examples/validation-records.md): 20/20 cases
passing at TR ranging 1..128. The over-declaration is verified at TR = 1, 2, 3,
4, 8 and 16 producing the same fp32 rounding as the legal TR=8 form, not merely a
close one. The "very short axis" section's degradation is measured; its
register-shuffle escapes above are also measured (on branch `bench/rms_norm-a5`,
`custom/rms_norm/{DESIGN.md,test_rms_norm.py}`, verifier-rerun 2026-08-06) —
port the family structure, not the numbers.
