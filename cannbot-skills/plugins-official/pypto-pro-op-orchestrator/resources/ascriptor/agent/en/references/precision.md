# Preserve the arithmetic contract

Record the complete cast, accumulation, saturation, rounding and parenthesization
path. `(key * beta) * exp(g)` may differ from `key * (beta * exp(g))` after bf16/fp32
rounding. Moving a cast across a reduction or a saved-state boundary is a semantic
decision, even when the final output dtype is unchanged.

Floating cube paths commonly accumulate in fp32 and integer cube paths in int32;
the actual declared operation/dtype combination is authoritative. Preserve bias on
the initialization tile and preserve every intended output/state downcast. A reference
for a half workspace followed by a matmul must model that half boundary. Do not use
Torch integer truncation as a substitute for a DSL rounding mode with different ties.

Compare all named outputs: names/arity, shape, dtype, alias behavior, defined lanes
and values. Choose exact, bitwise or tolerant comparison from the unit contract.
Zero tolerances are valid for an exact contract. There is no universal float tolerance
or ban on exact comparisons. Each nonzero tolerance needs a reason and scoped cases.
Check NaN/Inf positions, signed zero or saturation when those affect the contract;
random normal inputs alone do not test overflow, cancellation or subnormal behavior.

The mathematical target may have a different declared precision from stored output.
An `allclose` rule may explicitly set `reference_dtype` to `float32` or `float64`;
the output/stage descriptor still declares the actual floating storage dtype, and
the runner checks both independently. `exact` and `bitwise` reject this field;
without it, actual/reference dtypes must match. M058
restores the original KDA scaling targets before BF16 rounding with unchanged
`rtol=atol=0.006`. Physical BF16 checkpoints consumed by later stages remain unchanged.
Preserve the source-defined formula and original numeric/L2 gates; this rule is not
permission to cast a wrong actual dtype into compliance or change every saved-state reference.

Test the validator with deliberately wrong outputs: all zeros, sign reversal, a
missing output, a dtype/shape change, one corrupt stage and non-finite values outside
the allowed domain. A large absolute tolerance can accept an all-zero result when a
stage's true values are small. Preserve the justified pointwise budget and, when that
failure occurs, add a justified `max_relative_l2` rule from the unit runner:
`norm(actual - expected) / norm(expected)`. A zero expected norm requires a zero
residual. Record measured valid errors and the rejected controls; do not pick a bound
only because the current candidate passes it.

Forward and backward state preparation can have distinct precision contracts. Delta
Rule's fused forward applies beta after a matrix product; backward preparation uses
a separately rounded bf16 `key * beta` boundary. A saved-state variant must name its
pre/post-chunk state, order, layout and dtype. Matching shapes do not make those two
states interchangeable, and backward preparation belongs inside the backward unit.

The A5 FP8 causal attention demo
([`a5_fp8_causal`](../../../kernels/ascriptor_kernels/attention/a5_fp8_causal)) computes its
row sum in FP32 before casting probabilities to e5m2 for the value product; its `formula`
says so and its `reference.py` is where that order is written down. The cast changes the
numerator, not the denominator, so a description that puts the row sum after the cast is
describing a different kernel. Preserve published rowmax/rowsum outputs and compare them
separately — the tight rowsum comparison is what catches the wrong order.

Packed formats need an independent bit-level reference. Two calls to the same codec
do not validate that codec. For a uint2 ABI, explicitly define whether consecutive
values `a,b,c,d` map to `a | (b << 2) | (c << 4) | (d << 6)` and whether only exact
integers in `[0,3]` are accepted. For FP4, record nibble order and signed-zero behavior.
For hif8/e8m0/MX, record carrier dtype, scale layout, rounding and the supported domain.
Register casts may leave sparse lanes needing pack/unpack before storage.

Use the current A5 operation contract when one of these boundaries is involved:

| Boundary | Authoring and verification rule |
|---|---|
| [Predicate routing (M10-051)](../../../library/docs/api/mask-write-semantics.md) | `compare` makes inactive logical lanes false; masked NOT/AND/OR/XOR/MOV zero inactive predicate bits — nothing preserves an inactive destination *register* lane, and the [per-operator table](../../../library/docs/api/mask-write-semantics.md) says which of the six roles each of the 66 masked operators takes. Data and predicate selectors choose both branches. A mask retains 256 physical bits: b32 observes every fourth bit, and pack/unpack route physical bits through a 128-bit half. Preserve the full payload when inspecting or moving masks. |
| [Online MX subnormals (M10-052)](../../../kernels/ascriptor_kernels/algorithms/online_mx) | The online MX demo's finite FP32 domain includes signed subnormals and magnitudes up to `2**40` — its `subnormal_payload` cases build them, and `reference.py` is where that domain is written down. Its repaired normalization multiplies by an exact normal FP32 reciprocal, which is a power of two; replacing that path with native division can change subnormal results. Preserve E8M0 scale bytes and compare raw payload/scale outputs independently. This is an algorithm/domain result, not a global VF subnormal guarantee. |
| Scalar square root (M10-053) and BF16 conversion (M10-055) | Use typed `scalar_sqrt` in the qualified finite nonnegative FP32/FP16 domain; SIMT FP32 is a separate path. The model rounds at the declared result type even when a following cell is eliminated. Ordinary BF16 scalar abs/sqrt on CCE/PTO use explicit FP32 widening and SDK narrowing; the retained boundary probes cover their repaired domain. PyPTO scalar restrictions remain upstream gaps. BF16 SIMT, register conversion and matrix arithmetic have separate qualification. |
| SIMT zero signs (M10-054) | Rint/round/floor/ceil/trunc retain the input sign when their result is zero. Compare FP32 bits, since floating equality cannot distinguish the zeros. The CCE wrapper restores only zero-result signs; its `fmod` truncation intermediate remains separate. Check the artifact-specific board receipt before claiming qualification. |
| [Native exp and HiFloat8 midpoint (M10-059)](../../../kernels/ascriptor_kernels/attention/a5_v8_p_stage) | The V8 P-stage demo states its measured exponent budget in its own `main.py`: native `exp` within one FP32 ULP of a correctly rounded exp, which at a quantization boundary picks the neighbouring code. `reference_candidates` propagates that interval into a set of admissible HiFloat8 codes rather than a widened tolerance; padding stays exact. Retain the raw outputs and rejection controls. The budget is that stage's on A5, not a universal native-exp accuracy guarantee. |

M10-051 raw routing is measured for b8/b16/b32. Unobserved fine bits produced by
compare/init/update and the single-register b64 raw layout remain model conventions.
Keep those limits with any derived physical-mask claim.

Do not infer NaN comparison behavior, one-ULP division accuracy, flush-to-zero behavior
or stochastic rounding from another architecture. A model-derived result and a
silicon-measured result carry separate source/toolchain/version evidence. Historical
competition checker thresholds and "fastest" claims are not this product's contract.
Use explicit edge cases to distinguish hypotheses before changing compensation or
non-finite guards.

Relevant library owners: `ascriptor/ir/saturation.py`,
`ascriptor/backends/sim/cast_rounding.py`, `ascriptor/backends/sim/cast_saturation.py`,
`ascriptor/backends/sim/vf_ops.py`, `ascriptor/backends/sim/vec_ops.py` and public
facade declarations. Their source is available for [white-box exploration](simulator-white-box.md).
Promote confirmed behavior to its library owner and update any affected unit budget.
