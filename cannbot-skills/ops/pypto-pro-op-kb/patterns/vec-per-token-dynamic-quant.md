# Per-token dynamic quantization

## Applies when

Each row must be quantized by a scale derived from that same row — typically
`scale = rowmax(|v|) / 127` followed by `round(v / scale)` to int8 — and the
scale is itself an output because a downstream operator needs it to dequantize.

Fused norm-then-quant and activation-then-quant operators have this topology.

## `pl.quant` covers the final conversion, not scale derivation

The fixed-revision official API document defines symmetric `pl.quant` for an FP32
source and INT8 destination, using an FP32 per-row multiplication factor shaped
`[rows, 1]` at that revision:
`out = clamp(round(src * quant_scale), -128, 127)`. See the
[official API document at that revision](https://gitcode.com/cann/pypto/blob/0ed5148c029e7e22d8a8134d210d183daa0f22ad/docs/zh/pypto_pro/api/SIMD-API/operation/quantization/quant.md).
The same revision's support table lists Ascend 950PR and 950DT and marks A2 and
A3 unsupported. Confirm the installed API and target before selecting this path.

`pl.quant` performs only the final multiply, round, clamp and integer conversion.
It does not derive `amax`, protect an all-zero row, or emit the scale required by
the operator contract. Those steps remain explicit kernel work. The official
example also loads a prepared scale from global memory; it does not validate
producing the scale and consuming it with `pl.quant` in the same kernel.

## Dataflow

Per row block, with the row's values in a resident wide-precision buffer `V`:

1. `abs` then reduce each row to `amax` — a row reduction, so
   [vec-row-reduce-broadcast.md](vec-row-reduce-broadcast.md) applies;
2. clamp: `clamped_amax = max(amax, floor)` where `floor` is a small positive constant;
3. compute `scale_out = clamped_amax / 127` and **store it as the downstream
   dequantization scale**;
4. derive `quant_scale` in the FP32 order required by the operator reference;
   for `round(V / scale_out)`, it is `1 / scale_out` (algebraically
   `127 / clamped_amax`, but the two FP32 evaluation orders need not be bit-identical);
5. when the installed `pl.quant` contract supports the FP32-to-INT8 path, pass
   `quant_scale` using its documented per-row shape;
   otherwise use an installed-API-supported explicit broadcast, divide, round,
   clamp and conversion path, or report the design as unsupported.

Do not pass `scale_out` directly to `pl.quant`: the API quantizes as
`round(src * quant_scale)`, this page's reference quantizes as
`round(src / scale_out)`, and the downstream operator dequantizes as
`q * scale_out`. If the operator contract uses the opposite scale convention,
name and derive the two quantities from that contract instead of reusing an
ambiguous `scale` variable.

**Step order is load-bearing.** Clamp before deriving either scale: an all-zero
row otherwise produces `scale_out = 0` and division by zero, and clamping later
cannot repair it. Follow the operator reference's order and verify it directly.

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

`conceptual only`. The interface facts above cite a fixed-revision official source,
but no reviewable artifact in this KB demonstrates deriving the scale and
consuming it with `pl.quant` in the same kernel. Confirm the installed API and
target, then retain a runnable implementation and a scope-matching passing
result before promoting this pattern.
