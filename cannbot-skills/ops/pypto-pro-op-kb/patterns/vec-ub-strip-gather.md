# UB strip gather — index-driven movement on a vector-only target

**Topology:** `gather-scatter-indexing`
**Status:** validated on Ascend950PR_9579 (CANN 9.2.0), 12 of 12 dtype
combinations bit-exact; the fp32 case measured end to end.
**Evidence:** [retained validation record](../examples/validation-records.md), scoped to
the target, dtype matrix and result stated above.

## When this applies

An index tensor drives which element of a source tensor each output element
takes, along one axis, with the other axes carried through — `torch.gather`,
`GatherElements`, `take_along_dim`, and the read half of `scatter`/`index_put`.
It does **not** apply to `tf.gather`-style whole-row selection along an axis;
that is a segment copy, not this.

## The constraint that determines everything

**`vf.gather` reads UB only.** There is no GM random-access primitive, and
pypto-pro has no SIMT, so an approach that indexes GM per element does not
translate here at all. Every element a tile needs must already be in UB.

## The shape

Flatten any rank about the indexed axis `k` into three axes:

    x   -> [Px, X, W]     X = x.shape[k], W = x's inner extent
    idx -> [P,  G, V]     G = index.shape[k], V = index's inner extent
    y[p, g, v] = x[premap(p), idx[p, g, v], wmap(v)]

For one outer position and one inner column block, the elements any output can
reach are the **strip** `x[px, :, w0:w0+C]` — the entire indexed axis, `C`
columns wide. Hold it in UB as an `[Xs, C]` tile and the source for output
`(g, c)` is at UB element offset

    idx * C + c

which `vf.gather` takes directly. Three vf ops plus the gather, for every rank
and every `k`:

```python
# HOISTING THIS IS ONLY LEGAL WHEN C divides L. See the premise below.
lanec = vf.and_(vf.arange(0, dtype=pl.DT_INT32), cmask, mi)   # hoisted: c term
...
idx = vf.load_align(t_idx, base)
off = vf.muls(idx, C, mi)
off = vf.add(off, lanec, mi)
y   = vf.gather(t_src, off, md)
vf.store_align(t_out + base, y, md)
```

The `c` term looks free, and it is — **but only under a load-bearing premise**.
Step `r` covers `L` consecutive positions of the row-major `[GT, C]` output
tile starting at position `r*L`, so the column a lane writes is

    c = (r*L + lane) mod C

The general expression is therefore `arange(r*L, r*L+L) & (C-1)`, which depends
on `r`. It collapses to the hoisted, `r`-independent `arange(0, L) & (C-1)`
**exactly when `C` is a power of two that divides `L`** — then `r*L mod C == 0`
and the `r*L` term vanishes.

**This premise is load-bearing, and violating it fails silently.** When
`C > L` the premise does not hold and the hoisted form writes the wrong
columns with no error: at `L = 32, C = 64`, `r*32 mod 64` alternates `0, 32,
0, 32…`, so every odd step writes its whole register **32 columns off**. The
output is fully populated and plausibly shaped; only the column mapping is
wrong. Nothing in the DSL, the tile geometry or a capacity assertion catches
it, because no address goes out of range — the writes land inside the tile, at
the wrong place.

**So, before hoisting, assert `C <= L and L % C == 0 and (C & (C-1)) == 0`.**
If `C > L` — which the `Irun < C` second tile family in the next section
deliberately creates, with `C = Irun` — do not hoist: recompute `lanec` per
step from `arange(r*L, r*L+L) & (C-1)`, or choose `L` as a multiple of `C`.

*(The measured 12/12 dtype sweep below ran entirely in the `C <= L` regime, so
it does not cover this case. Recorded 2026-08-07.)*

## Sizing it — the measured knee

The strip load is a strided GM read of `Xs` rows, `C` elements each, at the
source's inner stride. Measured on 9579 with the moved volume held fixed so row
length is the only variable:

| row bytes | 32 | 64 | **128** | 256 | 512 | >=1024 |
|---|---|---|---|---|---|---|
| % of contiguous bandwidth | 31 | 44 | **88** | 96 | 98 | 100 |

**Target `C * esize >= 128 bytes.** Since `Xs * C * esize <= UB_budget`, that
caps the strip at about `budget / 128` rows — roughly 1024 for 128 KiB. An
indexed axis longer than that takes **passes**, not a narrower strip: below
128 B the cost roughly doubles per halving, and two passes at 128 B beat one at
32 B by a wide margin.

The kernel is DMA-bound at these sizes, not gather-bound, which is what makes
extra passes and half-empty gather lanes affordable.

## Four things that fail silently

1. **`C > inner extent` writes nothing.** `ncb = ceil(Irun / C)` is zero when
   `Irun < C`, so the kernel produces no output and reports success. Every 1-D
   case is `Irun = 1`. Use a second tile family for `Irun < 32/esize`: the
   `(indexed, inner)` block is contiguous when the inner extent is that small,
   so the strip becomes a flat `[1, Xs*Irun]` run — the best possible DMA — and
   the same offset expression works with `C = Irun`.
2. **The b16 index window is narrower than UB.** A b16 gather dst takes a
   **UINT16** index, and the offset must be produced by `vf.pack`, which
   truncates without saturating or erroring. `Xs * C > 65536` returns plausible
   wrong data. Make it a tiling invariant, not a runtime check.
3. **An out-of-range gather offset is undefined, not masked.** When splitting
   the indexed axis into passes, clamp `idx - pass_lo` into `[0, Xs-1]`
   *before* the gather and mask the *store*. Masking after the gather is reading
   whatever that address held.
4. **Pass loop placement decides the traffic.** Passes must sit **outside** the
   output-row-chunk loop, or the strip is re-read once per row chunk and source
   traffic multiplies. Since `pl.store` has no per-element mask, the output tile
   is read back from GM for passes after the first — `(npass-1)` extra reads and
   writes of the *output*, which is far cheaper than re-reading the *source*.

## Dtypes — every spelling is forced, none is a preference

`vf.gather` and `vf.load_align` derive a register's C++ type from the **tile's**
dtype and hand it to an intrinsic that wants another. Four spellings do not
build at all; the diagnostics and the routes around them are in
[`references/pypto-pro-framework-findings.md`](../references/pypto-pro-framework-findings.md).
The short form:

| payload | declare as | narrowing |
|---|---|---|
| fp32 | `DT_FP32` | — |
| int32 | **`DT_UINT32`** | — (the signed spelling will not build) |
| fp16 / bf16 | `DT_FP16` / `DT_BF16` | offsets in INT32, `vf.pack(dtype=DT_UINT16)` |
| int8 | **`DT_UINT8`**, widened to `DT_UINT16` in UB by `pl.cast` | `vf.pack(dtype=DT_UINT8)` — must truncate, never saturate |
| int64 | **`DT_UINT32`, inner extent doubled** | index duplicated by `vf.interleave(idx, idx)` |
| index int64 / int8 | `pl.cast` to an INT32 tile in UB | — |

Two of those are correctness, not convenience. The int8 narrowing must truncate:
the b16 gather result carries the byte zero-extended, so `-1` arrives as
`0x00FF` and a saturating cast returns `127`. And int64 needs no 64-bit gather —
viewed as UINT32 with the inner extent doubled, the source word for output word
`p` is `idx[p/2]*2C + (p mod 2C)`, the same expression with each index value
duplicated.

## Coverage the cases you can see will not exercise

The map from index coordinates to source coordinates is the identity only under
two conditions, and the draws you can inspect routinely satisfy both while the
declared range does not:

* **prefix identity** — extents agree on every axis before `k` except the
  outermost. When it fails, decode the mixed radix **on device** from a small
  block table (`px = sum_j ((o / t_inner_j) mod size_j) * x_stride_j`), once per
  outer position. A host-precomputed table is `O(P)` and `P` is unbounded.
* **inner identity** — extents agree on every axis after `k+1`. When it fails,
  the differing trailing axes fold into the outer loop and the contiguous run
  that remains becomes the inner extent. The three-axis view absorbs this with
  no new kernel code; only the host tiling changes.

## Correctness signal

**Bitwise equality, for every dtype including floats.** This topology moves
data; it computes nothing. Any nonzero error is an indexing bug, and a
tolerance-based gate cannot distinguish a correct permutation of NaNs from a
wrong one — benchmark suites for these operators reliably include an all-NaN
case and an inf case precisely because they are cheap to generate. Compare
floats through `.view()` on an integer of the same width so NaN payloads and
signed zeros are compared as bits, and keep the official gate as the formal
second verdict rather than the first.
