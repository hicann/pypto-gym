# Numerical patterns with explicit storage boundaries

Choose one pattern after [preflight](authoring-preflight.md); exact overloads
come from the accepted API. These are dataflow recipes with concrete example
owners, not a catalogue of universal supported shapes.

| Pattern | Derive before coding | First example |
|---|---|---|
| Elementwise rows and tails | Register live lanes versus store footprint; GM valid extent; per-core row ownership | [Unaligned rows](../../../library/examples/api/unaligned_rows) and [mask semantics](../../../library/examples/api/mask_semantics) |
| Row reduction and broadcast | Reduction axis; result lanes; scratch read by the next operation; identity on invalid lanes before reduction | [Register reductions](../../../library/examples/api/register_reductions); its code is [excerpted below](#register-code) |
| Stable softmax | Max mask, shift, exp, sum, normalization; all-masked row contract; cast after the required sum update | [PFA BF16 demo](../../../kernels/ascriptor_kernels/attention/a5_pfa_bf16), whole; [V8 P stage](../../../kernels/ascriptor_kernels/attention/a5_v8_p_stage) for the vector side alone |
| Matmul across K tiles | Operand layout, first-tile initialization, subsequent accumulation, bias exactly once, padded physical tile | [Cube bias](../../../library/examples/api/cube_bias) |
| Packed or low precision | Logical format versus carrier, bit order, defined lanes, rounding/saturation and actual packed-store bytes | [Physical formats](../../../library/examples/api/physical_formats) and [cast formats](../../../library/examples/api/cast_formats) |
| Saved-state composition | Stage signatures, state precision/version, producer/last consumer, independent leaf and whole references | [Decompose](../playbooks/decompose.md) |

## Reason through a short row

For 64 FP16 values, useful data is 128 bytes but a full unmasked B16 register
store can touch 256 bytes. Preserve the physical allocation and explicitly
select live lanes. At a final slot this becomes an allocation overrun; at an
earlier row it can silently overwrite a neighbor. Check canaries and the last
row/slot. Packed stores have a different carrier-to-output mapping; derive
each instruction separately using [memory and tails](memory-and-tails.md#specific-boundaries).

<a id="reason-reduction"></a>
## Reason through a reduction

Identify the first operation that can consume padded values. Supply its
identity there, then trace which result lanes the next operation reads. A
single logical scalar does not prove a one-element scratch allocation is
enough for a full-width consumer. Preserve accumulation precision and casts.
Test all-negative maxima, cancellation, nonuniform rows, ties and valid tails.
Do not silently change an all-masked row's defined behavior to pass a test.

## Reason through a cast boundary

Write the exact sequence, such as FP32 matmul accumulation → FP32 scale/bias →
ReLU → FP16 RNE → next matmul. Moving scale into a different instruction or
deferring the FP16 cast changes the comparison target unless independently
justified. Use bit-level host codecs for packed-format contracts and numerical
references for their separate mathematical targets. See [precision](precision.md).

An FP16 input computed in FP32 — every normalization starts this way — is one
round trip through the even lanes:

```text
ub_to_reg_unpack  64 dense f16 UB elements -> the even lanes of an f16 register
cast ZERO         those even lanes -> a 64-lane f32 register        (compute here)
cast ZERO         f32 -> the even lanes of an f16 register
reg_to_ub_downsample  even lanes -> 64 dense f16 UB elements
```

The whole round trip, plus the reduction's return path, is one runnable unit:
[row_norm_fp16](../../../library/examples/api/row_norm_fp16). It normalizes four FP16
rows in FP32 — `cadd` into lane 0, `reg_to_ub_single` out to a UB cell, `ub_to_reg_single` back
across every lane, `sqrt` then `div` for the missing `rsqrt` — and it is bitwise against an FP64
reference in `sim`, `pipesim` and on a card through CCE.

It is also where `eps` earns a case. Its `zero_row` case sends an all-zero row through: with the
bias the row is a finite zero, without it the row divides by zero and arrives NaN. In any
ordinary input domain `eps` sits far below one FP16 ulp, so a case matrix without that row
cannot tell an implementation that applies it from one that drops it — a control that cannot
fail is not a control.

`reg_layout` is a lane parity for the half/single pair — f16 and bf16 alike — not a half of
the register, and `ONE` (the odd lanes) needs a `MaskReg(DT.half)`: under a b32 mask the same
cast returns all zeros and raises nothing, on the card as well as in the model. `ONE` also does
not emit on PyPTO-Pro. Whether the field is read at all depends on the pair's shape family: the
64-bit and same-size forms carry no part selector, and there `ONE` silently means `ZERO`.
[The cast policy](../../../library/docs/api/formats.md#cast-policy-and-destination-state) has
the family table. [The cast
policy](../../../library/docs/api/formats.md#cast-policy-and-destination-state)
owns both facts and the measurements behind them. A5 VF has no `rsqrt` — the
[manifest](../../../library/docs/api/manifest.json) declares it only in the
`a2_vector` scope — so a normalization spells it `sqrt` then `div`.

For each recipe test the first relevant valid/invalid boundary, output
coverage, preserved neighbors and same-core reuse when storage repeats.
Keep a deliberately wrong cast/mask or omitted output as a comparator control.
Use [debug](../playbooks/debug.md) when a primitive, model or lowering disagrees.

<a id="register-code"></a>
## Ground the recipe in register code

This excerpt is `reduce_vf` from [register_reductions/kernel.py](../../../library/examples/api/register_reductions/kernel.py).
It shows load → reduction → stores in the result layout. The owner supplies full DMA, imports, input domain and reference.

<!-- code-anchor:reduction:start -->
```python
@vf()
def reduce_vf(xf: Tensor, xi: Tensor, xl: Tensor, of: Tensor, oi: Tensor, ol: Tensor):
    rf = Reg(DT.float)
    ri = Reg(DT.int)
    rl = Reg(DT.int64)
    rf <<= xf[0]
    ri <<= xi[0]
    rl <<= xl[0]
    for src, out, dt, cols in ((rf, of, DT.float, 64), (ri, oi, DT.int, 64), (rl, ol, DT.int64, 32)):
        r_add = Reg(dt)
        r_max = Reg(dt)
        r_min = Reg(dt)
        cadd(r_add, src)
        cmax(r_max, src)
        cmin(r_min, src)
        out[0] <<= r_add  # one full register per row (element offsets)
        out[cols] <<= r_max
        out[2 * cols] <<= r_min
```
<!-- code-anchor:reduction:end -->
