# Precision and cast boundaries

## Rule

Treat dtype, accumulation type, rounding, saturation, scaling, and tolerance as
part of the operator contract. Do not change one for performance without
updating the golden and acceptance criteria.

## Checklist

1. Record input, intermediate, accumulator, and output dtypes.
2. Confirm each cast/quant/dequant API and rounding mode in the installed
   documentation.
3. Keep numerically sensitive reductions and transcendental operations in the
   contract's required accumulation precision; use FP32 unless a narrower path
   is explicitly documented and validated. Register-level reduction primitives
   are typically same-type, so they cannot widen on your behalf — see
   "Widening a narrow-dtype reduction" below.
4. For quantization, make scale, rounding, clamp range, and dequantization
   placement explicit in both kernel and golden.
5. Derive tolerances from the project standard for the actual output dtype and
   algorithm; never copy a tolerance from an unrelated sample.
6. Validate boundary values, tails, and representative random inputs.
7. In a chain of matmuls, keep every GM intermediate **fp32** unless the golden
   itself rounds there, and split any computed matmul operand into bf16 residual
   terms — see "Chained matmuls" below.
8. Bound the depth of any single L0C accumulation chain — see "Deep K chains".

## Widening a narrow-dtype reduction

A reduction over FP16/BF16 inputs can overflow even when every input element and
the final mathematical result are both representable, because the *intermediate*
values are not. Squaring is the common case: an operand near the top of the
format's range squares to roughly the square of that range, far past the
format's maximum. The narrow type reaches `inf` mid-chain and the reduction
returns `inf` from finite, in-range inputs.

Confirm the following against the installed docs for the target SDK, but as of
the current one:

- The register-level reduce primitives (`vf.reduce_*`) are **same-type**: each
  source dtype pairs with the identical destination dtype ("源与目标数据类型需
  保持一致"). There is no narrow-in / wide-accumulate reduce to lean on. The
  widening is yours to write.
- **The transcendentals are narrower still, and BF16 is the gap.** `vf.exp`
  covers **FP16 and FP32 only**, and `vf.exp_sub`'s dtype table has exactly two
  rows — FP16|FP16→FP32 and FP32|FP32→FP32. **Neither has a BF16 row.** A bf16
  softmax or normalization therefore has no native narrow path at all: widen to
  FP32 before the exponential, not merely before the reduce. (Measured against
  the installed API docs on Ascend950PR / CANN 9.2.0.)
- The register-level cast is
  `vf.astype(src, preg, *, dtype=..., layout=..., round_mode=..., saturate=...)`.
  The tile-level cast is `pl.cast(out, src, *, mode=...)`, where the target dtype
  comes from `out`'s dtype. **There is no `vf.cast`.**
- **Both conversion APIs are platform-gated.** Their docs carry a
  「产品支持情况」 section, and support is not universal across Ascend
  generations. Confirm availability for the *detected* target before a design
  depends on either one — see [arch-a5.md](arch-a5.md) for target detection.
  Fail closed: if the widening mechanism is unsupported on the target, the
  design does not silently keep the narrow chain; it finds a supported
  conversion path or is escalated as unsupported.

### Widen before the operation that overflows, not before the reduce

Placing the cast immediately ahead of the reduction is not sufficient when a
multiply, square, or running accumulation feeds it: that operation has already
produced `inf` in the narrow type, and widening `inf` yields `inf`. The order
that holds is **widen → compute → reduce**, with the whole chain in the wide
type, narrowing again only at the store so the output dtype still matches the
contract:

```python
x_wide = vf.astype(x_reg, preg, dtype=pl.DT_FP32)
squared = vf.mul(x_wide, x_wide, preg)
total   = vf.reduce_sum(squared, preg)
```

Note that a widening cast changes how many elements fit in a register: the wider
destination holds fewer, so covering one full narrow register takes more than
one call. Read the API's layout and element-count tables rather than assuming a
1:1 mapping.

**`layout` selects interleaved lanes, not a contiguous half.** An earlier
version of this page said `CastLayout.ZERO`/`ONE` choose "which half of the
source" is consumed. That is wrong, and it was corrected by running both
readings side by side on Ascend950PR: the **even/odd lane** interpretation is
bit-exact, while treating them as low/high halves — two NORM stores 64 elements
apart — returns every other element.

The official sample settles it independently, in its own variable names —
`pro_ops/lightning_indexer/test_quant_lightning_indexer_vf.py:193-196`:

```python
c0_even = vf.astype(cout0, preg_b16, layout=pl.CastLayout.ZERO, dtype=pl.DT_UINT32)
c0_odd  = vf.astype(cout0, preg_b16, layout=pl.CastLayout.ONE,  dtype=pl.DT_UINT32)
```

So `ZERO` takes the even lanes and `ONE` the odd ones, and reassembling a full
narrow register means **interleaving** the two results, not concatenating them.
(The same file at `:149-150` uses the identical `ZERO`/`ONE` pairing for
`dtype=pl.DT_BF16` — which is also the evidence that the `vf.astype` doc
table's silence on BF16 is a documentation gap, not a capability bound.)

### A scale factor is not a substitute

Pre-scaling shifts the representable window; it cannot widen it. Compare the
ratio between the largest and smallest intermediate the operator must represent
against the input format's exponent span. When the requirement is larger, no
scale factor exists and widening is the only correct option.

Decide this at design time, in the cast-boundary chain, from the case's declared
value range and reduction length — not after a precision failure. The cost is
one extra vector op per tile plus wide-type registers along the reduction chain;
pay it where the range analysis says it is needed and keep the narrow type
everywhere else.

## Chained matmuls: fp32 intermediates and residual-split operands

When the golden upcasts its inputs once and carries the whole chain in fp32,
rounding each GM intermediate back to the narrow dtype is not a small
approximation — it can fail the gate outright.

The mechanism matters because it is invisible in a relative-error estimate. An
output that is a long *signed* sum is Gaussian about zero, so each narrow
rounding in the chain contributes a roughly constant **absolute** error, and the
elements whose true value happens to land near zero then carry enormous relative
error. Worse, that same absolute error pushes `|output|` past the cancel-region
threshold, so those elements do not even qualify for the cancel region's
CPU comparison — they land in the normal region, which must have zero
exceedances. Measured on a staged multi-matmul chain: ~0.1 absolute error on one
output, ~77 elements
per case, every case failing.

Two rules follow, and they are separate:

- **GM intermediates carry fp32.** This alone took one RoPE output from failing
  to *zero* error, because its only problem was the narrow rounding of the
  projection feeding it.
- **A computed matmul operand is split into bf16 residual terms.** The cube
  takes narrow operands, so an fp32 intermediate cannot be fed to it directly;
  split it as `t1 + t2 + t3`, each the bf16 rounding of what the previous terms
  could not represent, and accumulate all of them into one fp32 L0C against one
  resident weight tile. Measured: two terms (~16 mantissa bits) passes at M=1
  and fails from M=128 up; **three terms (~24 bits) passes at every shape.**

Split only what needs it. A weight that arrived as bf16 is already exact, so its
residual terms are identically zero — a "3-term" variant that splits the *right*
operand is a no-op that measures identical to 2-term. Likewise a projection
consuming the raw narrow input needs one term; only its **output store** widens.

Prefer the residual split to the cube's native fp32 mode: three bf16 passes is
~126 TFLOPS effective on A5 against 94.6 for fp32, and it reuses the narrow tile
path instead of needing fp32 L0A/L0B at half the K depth. (fp32 matmul *is*
supported and exact — verified — so it remains available where accuracy
dominates and the operand cannot be split.)

## Deep K chains: one L0C accumulator is an accuracy defect

`matmul_acc` over an N-block K loop sums sequentially into one fp32 accumulator.
The partial sums random-walk to the final magnitude, so rounding accumulates as
`eps · |result| · sqrt(N/2)`, while CPU BLAS reduces the same contraction
pairwise. For a 7168-deep contraction (N = 56) that is a ~3x gap — and on an
operator whose graded outputs include cancelled near-zeros it decides pass/fail.

**Fix:** send block `k` to accumulator `k % 4`, so each walks `N/4` blocks whose
partials reach half the magnitude; emit the four partials to GM and recombine
them **pairwise** in the consumer — `(p0+p1) + (p2+p3)` keeps both intermediates
at half scale so only the final addition rounds at full scale. Four `[64, 128]`
fp32 accumulators cost 128 KB of a 256 KB L0C.

Measured on the first projection of such a chain: the inherited error fell 3.51e-6 →
1.68e-6 (2.09x), bringing the chain to parity with the CPU reference
(2.76e-6 vs 2.74e-6) at **unchanged runtime** — the extra GM traffic was ~19 MB
against 122 MB of weights. 17/20 → 19/20.

Unroll the K loop by the accumulator count rather than branching on `k % 4`, so
the Partial/Final phase conditions stay simple (`q == 0` and `q == n_quad - 1`
per accumulator). That requires `n_kb % 4 == 0`; guard it on the host and raise
rather than silently dropping blocks.

## What the benchmark gate actually measures

Designing against "MERE below threshold" misreads the gate. `utils/compare.py`
applies it in two stages, and the second one is where most surprises live.

**Stage 1.** Overall `MERE < threshold` and `MARE < 10 * threshold` passes
outright.

**Stage 2.** Otherwise every element is sorted into one of three regions, each
judged separately, and **all three must pass**:

| region | membership | rule |
|---|---|---|
| normal | everything else | zero errors permitted **when the same-precision reference is clean**; otherwise at most 2x the reference's error count |
| small-value | `\|golden\| < 2^-11` fp16, `2^-8` bf16, `2^-14` fp32 | at most `2x max(reference_errors, 1)` |
| cancellation | `\|output\|` near zero while `\|golden\|` sits between the small-value and cancel boundaries | same 2x rule |

So a run can report `MARE` an order of magnitude over threshold and still pass,
because the offenders all landed in the cancellation region and the rule there
is relative to the reference rather than absolute. And the converse: a *single*
normal-region element is fatal when the reference is clean.

Consequences that change design decisions:

- **Find out whether your reference is clean.** The "same-precision reference"
  is the task golden run at input precision. When a golden promotes to fp32
  internally regardless of input dtype, that reference is identical to the fp64
  one, the reference error count is zero, and the normal region tolerates
  nothing. Check before assuming slack exists.
- **Do not delete a compensated accumulator to save vector ops without measuring
  the cancellation class.** Mean error barely moves while the *count* crosses the
  pass boundary. Reordering does not help either — cancellation is a
  conditioning problem, not a summation-length one.
- **The cheap pre-board check** is to replay the kernel's exact arithmetic on CPU
  against a higher-precision golden and count threshold exceedances per region,
  next to the same counts for the reference. That ratio is what the gate compares.
  A CPU-side oracle that replays the accuracy gate does this, including
  the deterministic input generation.

### Integer outputs are not covered by any of that

Integer outputs take an absolute-difference path: **every** element must satisfy
`|out - golden| <= threshold`, with no proportional fallback. The default
threshold for integer dtypes is `0`; an operator's `proto.yaml` may relax it via
`precision_thresholds` (commonly `int8: 1`). Read the code, not the operator
description — descriptions have been observed to state a tolerated *fraction* of
mismatches where the implementation tolerates none.

A ±1 int8 tolerance does absorb a round-half-to-even versus round-half-away
difference between torch and the hardware convert, but only if nothing else has
already consumed the margin.

### Non-finite values are compared before any error arithmetic

**NaN positions must match exactly** or the case fails outright, reporting
`MERE=0, MARE=0` alongside a position-mismatch message. Inf on one side only is
saturated to the dtype maximum and comparison continues; both-inf-same-sign
counts as a match and is excluded from the statistics.

Two traps:

- **A NaN guard that lets Inf through turns Inf into NaN.** The obvious `x == x`
  test rejects NaN but **passes for ±Inf**, so a sum that overflows to Inf leaves
  an Inf residual and `total - residual = Inf - Inf = NaN` where the reference is
  Inf. Scale by zero first — multiplying by `0.0` maps every finite value to zero
  and both Inf and NaN to NaN — then compare and select.
- **`float -> int` conversion of a non-finite is undefined in C++** and differs
  between hosts and between host and device. Measure it on the evaluation host
  before designing the path; a wrong guess misses by up to 128 LSB on an int8
  output where every element must land within ±1.

Special-value cases are generated **deterministically**, not randomly: a
`[-inf, inf]` range yields ordinary random values with the first 5% set to `-inf`
and the last 5% to `+inf`, and `[nan, nan]` yields random values under a 50% NaN
mask. They are therefore reproducible offline — but `torch.randn` never reaches
these paths, so they need their own sweep. Note also that such a range applies to
*every* input it is declared for, so a non-finite can enter through a scale or
gamma vector rather than through the obvious data input.

## Evidence

- FP32 softmax and independent golden:
  [softmax_impl.py](../examples/samples/softmax/softmax_impl.py) and
  [softmax_golden.py](../examples/samples/softmax/softmax_golden.py)
- explicit scale/round/clamp quantized matmul:
  [matmul_quant_int8_impl.py](../examples/samples/matmul_quant_int8/matmul_quant_int8_impl.py)
- validated BF16 contraction:
  [bf16_matmul_operand_reuse_impl.py](../examples/samples/bf16_matmul_operand_reuse/bf16_matmul_operand_reuse_impl.py)

Use `$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/` as the primary source for supported
dtype pairs and API semantics in the current SDK.
