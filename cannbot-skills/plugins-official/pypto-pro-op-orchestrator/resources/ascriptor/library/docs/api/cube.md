# Cube tiles, bias, quantization and MX scales

`matmul(dst, a, b, ...)` uses logical `a[M,K] @ b[N,K].T -> dst[M,N]`.
Both inputs are staged in L1 and the accumulator lives in L0C. A transpose
view changes the logical operand presented to its consumer; it does not move
data. The [basic tile](../../examples/api/cube_matmul) has one independent
reference and shares its factory across A2, A3, A5 and A5PR.

`splitn` or `splitk` partitions work inside one core. It does not distribute
work across cores or supply a host merge. The caller owns global partitioning,
input/output storage and cross-side synchronization. Implicit L0 operand
buffers are compiler-managed; old `_l0a`/`_l0b` counter names are not required
author setup. Shapes, supported dtypes and physical transfer requirements are
checked by the chosen device/backend, separately from Python declarations.

For A2/A3 INT32-carrier `DT.int4` operands, MMAD physically consumes the
complete final byte: odd K also reads its high nibble. Clear unused padding
nibbles before staging the operands to obtain the logical product above.
`k=` alone does not mask half a byte. The [INT4 tail unit](../../examples/api/a2_int4_tail)
accepts arbitrary public padding by normalizing private carriers on the device;
M10-099 records silicon and model controls.

## Bias and output conversion

The [bias example](../../examples/api/cube_bias) stages a contiguous
FP32 `[1,N]` row with `gm_to_l1_pad`. A default ND2NZ copy would give that row
a different physical layout. Bias belongs on the initializing K tile: split-K
adds it once, and split-N selects the matching slice for each output column
tile. Its small integer products and quarter-valued bias have an exact FP32
reference. Device-specific BT limits still constrain other shapes.

INT32-carrier `DT.int4` operands take a fused bias on the same terms (A2/A3,
qualified on a card 2026-09-23; the bias table is INT32 because the
accumulator is). It was refused for part of one day while the c220 `mmad_bias`
specialization was missing — see [RFC-0008](../rfc/0008-a2-family.md#int4-fused-bias-2026-09-23),
which records why the refusal outlived its cause.

The [quantization example](../../examples/api/cube_quant) separates
these store meanings:

| Accumulator → output | Tested formula |
| --- | --- |
| FP32 → INT8/UINT8 | ties-to-even round of `0.5*x`, then offset 8, then signed/unsigned clamp |
| INT32 → INT8 | ties-to-even round of `0.5*x`, then signed clamp |
| INT32 → FP16 | half rounding of `0.25*x` |
| FP32 → FP16 | half rounding of `0.5*x` |

The accumulator/destination dtype pair selects the fixpipe mode. Scale and
offset are explicit `.requant(...)` riders on L0C. The sample uses parameters
exactly representable in the float19 scale field; it does not establish a
reference for arbitrary parameter truncation. Forced large positive/negative
products exercise saturation. [A2/A3 transfer examples](../../examples/api/a2_cube_transfers)
also check half conversion followed by another matmul, initialized atomic
accumulation and restoration to a normal store.

## Draining L0C into the vector side

`l0c_to_ub(dst_ub, src_l0c, M=, N=, N_dst=, M_src=, dual_mode=, sub_block_id=)` is the fixpipe's
direct route from the accumulator to a vector program's UB, and `ub <<= l0c[...]` selects the
same instruction. Every cube core is paired with **two** vector sub-blocks — a ratio that holds
on every shipped profile, unlike the core counts — and `dual_mode` decides how the M rows reach
the pair:

| `dual_mode` | Where the M rows go | When |
|---|---|---|
| `SPLITM` (IR default) | first M/2 to sub-block 0, second M/2 to sub-block 1, each into its own UB | a per-row consumer: `GetSubBlockIdx()` becomes part of an address, not a guard |
| `SPLITN` | the N extent splits instead | the per-sub-block work is along N — which is where a transposed product puts it |
| `SINGLE` | the whole block to the one `sub_block_id` names | one sub-block must see every M row — a reduction ACROSS M — or the move is not a plain copy, below |

Which axis to split is a question about the tile, not a habit. In the attention kernels the two
drains of one iteration use different modes: the score is computed transposed (`score^T = K @
Q^T`), so its query rows are the N extent and the drain is `SPLITN` with `N_dst` at half the
query block; the PV result is `[M=queries, N=D]`, so that drain is `SPLITM`. Both hand each
sub-block 64 query rows. The rule is to split whichever axis carries the rows a sub-block owns —
`attention/a5_pfa_qk_metadata` has both drains within fifty lines of each other, and records that
the `SINGLE` alternative measured +82 µs on its shape.

Both compile and both are correct, so the choice is not reported as an error. Choosing `SINGLE`
where a split mode would do costs twice: the vector work is no longer shared, **and** the landing
tile is sized for the whole M, which is usually what forces the M tile back down. A performance
lint names every such move.

**Split mode carries the same-type plain copy alone** — fp32→fp32 or int32→int32. The fixpipe's
scalar path rides a deqScalar that exists only with the dual destination control off, so split
mode accepts no fused `relu`, no non-default `scale`/`offset`, and **not even an unscaled float
downcast** (fp32→fp16/bf16). A drain that converts on the way has to be `SINGLE`, and that one is
forced rather than chosen. The reverse is refused before any backend reaches it: `verify` rejects
a split-mode move carrying any of those riders, because the hardware does something else and two
of the three backends would otherwise have printed it. A `scale` that folds to 1 is not a rider —
the frontend drops it — so a plain copy spelled with an explicit unit scale stays in split mode.

`l0c_to_gm.*` has no `dual_mode`: the pairing is a property of the vector destination.

## MX payload and scale storage

The [MX example](../../examples/api/cube_mx) supplies independent FP8
arithmetic and FP4 table decoders. Every K32 group has an explicit exponent
scale. For the finite codes used here, byte `s` means `2**(s-127)`; 126, 127
and 128 therefore mean 0.5, 1 and 2.

| View | Meaning |
| --- | --- |
| logical `[rows, ceil(K/32)]` | one scale per row and K32 group |
| packed `[row_tile, k64_block, 16, 2]` | 32-byte blocks with adjacent group pairs for each row |
| dense GM scale input | input to `gm_to_l1_mx_scale_nd2nz`, which constructs the compact L1 stream |
| packed GM `[num_blocks,32]` | input to `gm_to_l1_mx_scale` |

FP8 payload has one byte per logical value. FP4 uses low-nibble-first UINT8
carriers with two logical values per byte; local reinterpretation provides a
compute view. `DT.mx_e4m3`/`DT.mx_e5m2` are compatibility aliases for ordinary
FP8 storage. They supply no hidden scale metadata. The explicit MX load and
`mmad_mx`/`matmul_mx` operation establish the scale association.

The runnable example checks a fully aligned, untransposed `M=N=16,K=128` tile
through both shortcut and explicit L0 staging. Nonuniform scales expose wrong
row/group addressing. Its compact format domain does not establish all
transpose/split/exceptional-scale combinations mentioned in old guidance.
For transpose paths, scales must already follow the transposed logical rows;
the payload transpose does not transpose scale bytes. Physical source
alignment and explicit scale offsets remain separate obligations.

`zero_mxfp8_l1_padding` remains a located compatibility gap pending its own
semantic mapping. Do not treat name presence as a working compact-to-padded
packing path. Source emission, functional/pipe simulation and vendor execution
are reported as separate stages for every unit.

## Convolution and physical output rows

The [convolution example](../../examples/api/cube_conv) owns its
independent input packers. Feature maps use NC1HWC0. Weights are row-major
`[Cout_p,K]` with K order `c1,kh,kw,c0`; ordinary L1 staging produces the
operand layout. A prebuilt `[K,N]` fractal is not interchangeable with this
contract. Channel and Cout padding are explicit zeros.

`conv2d` covers one cube output tile, with the caller owning input staging,
outer output loops and publication. Its load3d source windows depend on stride,
padding and dilation. These spatial parameters do not reorder the weight K
axis. The small sample checks same padding, dilation and a stride-2 tile with
only four logical output rows.

Physical M-tail rows can contain real values: load3d continues its raster
window beyond logical `Ho*Wo`. This corrects the old guide's blanket zero-tail
claim, superseded by source decision D-223. The example independently extends
bottom zero padding and uses Torch convolution to calculate every physical
row. Consumers wanting the logical result select `output_rows`; the reference
does not ignore or mask the remaining physical outputs.

The [bidirectional A5 bridge](../../examples/api/cube_vector_roundtrip)
connects vector-produced compact NZ to L1, then direct L0C-to-UB results back
to vector code. It documents live rows versus physical NZ pitch and uses
separate ownership transfers in each direction. Its odd pitch does not imply
a measured performance gain.
