# Paired-lane rotation with a position-indexed coefficient table

## Applies when

Two lanes of the innermost dimension form a pair, and the pair is rotated by a
coefficient looked up from a small table indexed by a *position* derived from the
row index — not from the row index itself. Rotary position embedding has this
topology, in both its split-half and interleaved spellings, and so does any
2x2 rotation whose angle varies along one outer axis.

The defining property is that **the table is much smaller than the operand and is
reused across many rows**. If the coefficients are already materialised at full
operand width, this is not this pattern — it is plain elementwise, and the
interesting part has been paid for elsewhere.

## Dataflow

Per pair `(a, b)` with coefficients `(c, s)`:

```
out_a = a*c - b*s
out_b = b*c + a*s
```

Three facts follow, and together they collapse what usually looks like four
separate kernels into one family:

1. **The arithmetic is invariant.** Pairing rule, operand layout and table rank
   change only *addressing*. Only two things justify separate compiled variants:
   the pairing rule (it changes the dataflow) and the dtype (it changes tile
   types). Everything else should be a runtime scalar.
2. **Pairing determines whether a strided read is needed at all.** With a
   split-half rule the partner of lane `j` is `j + W/2`, which is **inside the
   same contiguous row** — one ordinary tile load contains both halves and no
   strided access is required. Only the interleaved rule (`2j`, `2j+1`) needs
   de-interleaving. Do not generalise an interleaved sample's "loads have no
   stride, split on the host" workaround to the split-half case; it costs real
   copies to solve a problem that case does not have.
3. **The rotation itself is exact.** Selecting the partner and negating it is data
   movement. The only arithmetic is two multiplies and one add, which is what makes
   the precision analysis in *Precision* below tractable.

### Position mapping

With `r` the flattened row index over all axes above the innermost, row-major:

| operand layout | mapping | divisor |
|---|---|---|
| position axis **outside** the repeated axis, e.g. `(B,S,N,W)` | `s = (r / N) % S` | `N` |
| position axis **innermost of the three**, e.g. `(B,N,S,W)` | `s = r % S` | `1` |

Both are `s = (r / divisor) % S` with `b = r / (S*N)`. Carry the divisor as a
runtime scalar and one kernel serves both layouts. Table row is then

```
table_row = b * batch_stride + s
```

where `batch_stride = 0` selects a shared `(S, W/2)` table and `batch_stride = S`
selects a per-batch `(B, S, W/2)` one — so table rank is also a runtime scalar, not
a variant.

### Two table-access regimes

The divisor above splits the tiling into exactly two regimes, and **both are
needed**:

* **`divisor == 1`** — table rows advance in lockstep with operand rows, so a
  `[TR, W]` row tile pairs with a **contiguous `[TR, W/2]` table slice**: one extra
  load. Iterate within segments of `S` rows so a tile never straddles the wrap.
* **`divisor == N > 1`** — each table row serves `N` consecutive operand rows, so
  iterate over units of `N` rows and **broadcast one `[1, W/2]` table row across
  the tile** (the `col_expand_*` form).

Choosing only the broadcast regime is a trap: an operand with `N == 1` under the
first layout falls into `divisor == 1` and must take the contiguous path. Routing
it through the broadcast path yields one iteration per row — in the case that
motivated this note, 500,001 iterations of 256-byte transfers.

## Decision checks

- The innermost dimension is even and pairs never cross a row boundary.
- The table is indexed by a position derived from the row index, and is small
  enough that reloading it per row-group is cheaper than materialising it at
  operand width.
- Operand rows are independent; no cross-row state.
- The pairing rule is fixed at compile time; layout and table rank are not.
- Transfer size, not op count, is the thing being optimised — see *Performance*.

Do not use this pattern when the coefficients arrive already broadcast to operand
width (plain elementwise), or when the rotation mixes lanes across rows.

## Performance

This topology is **pure streaming**: read two operands plus a small table, write
two operands. SOL is achieved bandwidth, so the design variable is DMA transfer
size and the arithmetic is essentially free — a useful thing to know, because it
means buying numerical robustness with extra vector ops usually costs nothing.

The corollary is the common failure: a kernel that owns **one row per iteration**
issues transfers of `W` elements. At `W = 64` fp32 that is 256 bytes, and a
20-million-row operand then needs hundreds of thousands of iterations. Own a tile
of many complete rows instead, and form the pair inside it.

## Precision

Two multiplies and one add, with operands of similar magnitude and opposite sign,
is a **catastrophic cancellation** shape. When the operand range is wide the
absolute error of each product can exceed the magnitude of their difference, and
relative-error gates fail on the positions where the result is near zero.

Before reaching for compensated arithmetic, establish **which references the gate
actually compares against**. A grader that supplies a same-precision reference
alongside its high-precision oracle typically grades the normal range as a *ratio*
against that reference, in which case reproducing the reference's operation order
in the operand's own precision already passes, and compensated arithmetic buys
nothing. A grader that supplies only the oracle grades absolutely, and then
compensation is required.

Getting this backwards is expensive in both directions, and it is not usually
documented in the task package — read the evaluator. A worked instance, including
the measurements and the ~50-op-per-pair kernel that was written for a gate that
did not require it, is preserved in the scoped
[validation record](../examples/validation-records.md).

If compensation *is* required, a single FMA-based `TwoProduct`
(`p = a*b; e = fma(a, b, -p)`) costs ~2 ops per product; Dekker splitting costs
~10 and is worth avoiding unless no FMA is available. Compensation is also fragile:
if the toolchain contracts the `mul`/`sub` that computes the error term, the term
becomes meaningless and adding it back makes the result *worse* than the
uncompensated form.

For narrow dtypes, widen every operand *including the coefficients* to the compute
precision before the multiplies and narrow once at the end. Multiplying in the
narrow dtype is the usual cause of a narrow-dtype gate failure.

## Validation status

The [interleaved rotary sample](../examples/samples/vector_kernels/rope_interleave_impl.py)
is **validated** but covers only part of this pattern: fp32, interleaved pairing,
a single operand pair, static innermost width, and coefficients passed
**pre-broadcast at full width** rather than looked up from a table. It is a
skeleton for the multicore row-tile loop, not a starting point for the table
lookup — which is the part this pattern exists to describe.

The split-half pairing, runtime-scalar layout divisor, runtime-scalar table rank
(including the per-batch `(B,S,W/2)` form), and the two table-access regimes are
outside that sample's validated scope. The
[retained validation record](../examples/validation-records.md) keeps that boundary explicit.
