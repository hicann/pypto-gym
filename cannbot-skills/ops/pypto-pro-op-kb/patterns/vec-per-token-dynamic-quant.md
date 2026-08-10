# Per-token dynamic quantization

## Applies when

Each row must be quantized by a scale derived from that same row — typically
`scale = rowmax(|v|) / 127` followed by `round(v / scale)` to int8 — and the
scale is itself an output because a downstream operator needs it to dequantize.

Fused norm-then-quant and activation-then-quant operators have this topology.

## There is no primitive for this

`pl.quant` / `pl.dequant` exist in the API surface but have **no usage anywhere
in the shipped example tests**, so their scale-tile conventions are
undemonstrated. Every quantization path with test coverage is a fixpipe
attachment on the drain out of the accumulator, and neither form is per-token:

| available | granularity |
|---|---|
| `store(..., pre_quant_scalar=...)` / `move(..., pre_quant_scalar=...)` | **one scalar for the whole tile** (a "dynamic" variant of this still means one scalar, supplied at launch rather than at compile time) |
| `store(..., fp_tile=<Scaling-space tile>)` | **per output channel** — varies along the channel axis, not the row axis |

Per-token therefore has to be composed by hand. Treat that as the expected
shape of the work, not as a sign of a wrong approach.

## Dataflow

Per row block, with the row's values in a resident wide-precision buffer `V`:

1. `abs` then reduce each row to `amax` — a row reduction, so
   [vec-row-reduce-broadcast.md](vec-row-reduce-broadcast.md) applies;
2. clamp: `amax = max(amax, floor)` where `floor` is a small positive constant;
3. `scale = amax / 127` and **store it as an output**;
4. broadcast `scale` back along the row and divide;
5. round to nearest, clamp to the integer range, convert to int8.

**Step order is load-bearing.** Clamping after the division instead of before
rescales the all-zero row by the divisor. Write the clamp exactly where the
reference puts it and check the reference rather than reasoning about it.

## Register accumulation, not tile reductions

The tile-op reductions each require a `tmp` workspace **of the full source
shape**, same dtype, in vector memory. A chain that reduces twice — once for the
norm and once for the amax — therefore carries several full-width buffers, and at
a few thousand columns that is the whole vector-memory budget for a *single row*:
no row batching and no double buffering, on an operator whose performance comes
entirely from keeping the loader ahead.

The register-level reduce primitives accumulate in registers and need no
workspace. Use them for the shipped kernel. The tile-op form is still worth
writing first as a correctness increment, because it is shorter and sits close to
a known-good sample.

## Precision

- **The wide-precision buffer must actually be wide enough for the squares, not
  just the values.** Holding intermediate squares in a 16-bit float overflows
  once inputs approach that format's maximum: the square exceeds the format's
  range long before the value does. This is a representation failure, not an
  accumulation one, so it is not fixed by widening only the accumulator. A format
  with the same exponent range but fewer mantissa bits is a different question and
  needs its own measurement.
- **Do not re-derive the quantized value from a published narrow-precision
  intermediate.** If the operator also emits the pre-quantization tensor at input
  precision, that tensor is rounded; feeding it back into the scale computation
  injects error into an output whose threshold is usually much tighter than the
  int8 one. Keep the wide-precision copy, or recompute from the original inputs.
- Cheaper reformulations — reciprocal-and-multiply instead of divide, reciprocal
  square root instead of square-root-then-divide — are usually affordable here,
  but "usually" is not a licence. Price them against the reference before
  adopting: the currency is the count of elements whose integer result moves,
  and the budget is that **no** element may move by more than the operator's
  integer tolerance.

See [../constraints/precision.md](../constraints/precision.md) for what the gate
measures, especially the integer path and the non-finite rules — a row whose
scale is non-finite propagates to every element of that row's output.

## Validation

`conceptual only`. The composition above follows from the available primitives
and from measured precision results, but no retained runnable implementation
demonstrates it yet. Confirm each call's signature against the installed API
documentation before use, and add a link here once a validated implementation
exists.
