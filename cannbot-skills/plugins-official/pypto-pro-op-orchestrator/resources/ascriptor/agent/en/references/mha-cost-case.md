# MHA: separate reconstructed cost from measured performance

Use the existing [performance record](../../templates/performance-analysis.md)
after the [MLA → MHA/GQA contract transfer](attention-authoring.md#transfer-the-contract-to-mha-or-gqa).
MHA case numbers below belong to a different benchmark from the MLA examples.
The complete-kernel performance scope that was ever measured is exactly:

| MHA origin / declared case | Dtype / layout | B / SQ / SKV / H / D | Causal | Measured official FIA median, µs |
|---|---|---|---|---|
| 6 / `cross_bf16_short_kv` | BF16 / BSND | 2 / 512 / 128 / 16 / 128 | No | `18.070` |
| 7 / `prefill_fp16_causal_long` | FP16 / BSND | 4 / 1024 / 1024 / 32 / 128 | Yes | `159.360` |
| 14 / `decode_fp16_large_batch` | FP16 / BSND | 128 / 1 / 128 / 32 / 128 | No | `180.910` |

The three shape rows above are live: they are the `full_shape` entries of `CASES` in the
[MHA demo's `main.py`](../../../kernels/ascriptor_kernels/attention/a5_mha_fp16_bf16/main.py),
and `python main.py --list` names them. The µs column is not: the demo records no
measurement.

**Where the µs on this page come from.** Every measured number below belongs to the unit
this demo replaced, and restores intact from the library's
retired attention exports —
the historical records are outside this source snapshot. Paths in parentheses are
relative to the restored `examples/attention/`. Restoring a record recovers original bytes
and identities; it is never a current execution claim.

The maintainer selected the live official FIA targets (`a5_mha_fp16_bf16/fia-targets.json`)
in [D-258](../../../library/docs/decisions.md). Each is the median of three independent
FIA captures on the same device and profiling protocol. The user subsequently
accepted the then-frozen implementation for completion and ended optimization.
That historical performance record (`a5_mha_fp16_bf16/performance.json`)
keeps each measured median, strict FIA result and user acceptance separate;
`user_accepted_current_version` does not mean a FIA win. A performance selection
requires all selected cases to be strictly faster; otherwise the published
selection covers verified correctness only. Each scene uses independently
supplied Q/K/V and identity head mapping. Retain all three original-parser
observations, their median and min/max; do not round equality or a slower result
into a pass. The historical case6 scheduling control below retains its original
source/runtime and is separate from the final common-unit qualification. Historical case
15 below is an arithmetic counterexample, outside this full-kernel performance scope.
Published baseline metadata and earlier comparisons retain their original identities.
Pipe-instrumented durations do not enter the latency targets.

## What the historical evidence permits

The maintainer-supplied 2026-09-08 MHA xhigh recommendations report two geometries
and cost totals. The historical R7 source, raw `final-source-costs.json` and
original validation/profile report have been externally cleaned and are
unavailable. This note has not inspected or rerun those files. It recomputes the
following numbers from the reported shapes, tiling and causal rule; matching the
reported totals does not verify the missing implementation.

Assume ordinary MHA with one independent K/V head per Q head, distinct K/V
storage, two-byte input/output elements and equal QK/value dimension D. Each
head/query tile loads each visited complete K and V tile once. Only wholly
invisible causal suffix tiles are omitted. Those assumptions reconstruct a
requested-payload model, not actual HBM/L2 transactions.

| Recomputed quantity | Historical MHA 7 | Historical MHA 15 |
|---|---:|---:|
| B / SQ / SKV / H / D | 4 / 1024 / 1024 / 32 / 128 | 60 / 1 / 512 / 16 / 256 |
| Logical M / physical M / KV N | 128 / 128 / 128 | 1 / 16 / 256 |
| Independent head sequences, `B*H` | 128 | 960 |
| Logical output rows, `B*H*SQ` | 131,072 | 960 |
| Query tiles per head | 8 | 1 |
| Global KV tile visits | 4,608 | 1,920 |
| Unique K+V bytes | 67,108,864 | 503,316,480 |
| K+V payload implied by the stated load policy | 301,989,888 | 503,316,480 |
| Unique Q+output bytes | 67,108,864 | 983,040 |
| QK+PV FLOPs on unmasked pairs | 34,393,292,800 | 503,316,480 |
| QK+PV FLOPs implied by whole physical tiles | 38,654,705,664 | 8,053,063,680 |
| Measured HBM/L2 bytes / Vector pipe utilization | UNKNOWN | UNKNOWN |

One MAC is two FLOPs, so QK plus PV costs `4*D` per query/key pair here.
For right-aligned causality, query i has
`v_i=min(SKV,max(0,SKV-SQ+i+1))` visible keys. All keys are visible in a
noncausal row. The useful count is `4*D*B*H*sum_i(v_i)`.

For case 7, each head has eight M128 query tiles with KV-prefix lengths
1 through 8. Thus visits are `4*32*(1+2+...+8)=4608`. The exact causal
pair count per head is `1024*1025/2`; useful FLOPs are
`4*128*(4*32)*(1024*1025/2)`. Whole-tile FLOPs are
`4*128*128*128*4608`. Unique KV bytes are `2*4*32*1024*128*2`,
and implied KV requests are `4608*2*128*128*2` bytes.

For case 15, each of `60*16` heads has one query and two N256 KV tiles:
`60*16*2=1920` visits. With SQ1, right-aligned causality and noncausal
attention both expose all 512 keys, so the unspecified historical causal flag
does not affect these counts. Useful FLOPs are `4*256*60*16*1*512`;
physical-tile FLOPs are `4*256*16*256*1920`. Unique KV bytes are
`2*60*16*512*256*2`, equal to the reconstructed requests. Q plus output
uses `2*60*16*1*256*2` bytes.

Case 7's 4.5× KV request ratio does not establish 4.5× HBM traffic. Case 15's
16× physical-work inflation does not by itself establish a Cube bottleneck.
Actual launch/core assignment, buffer allocations and live slot counts are
UNKNOWN for these unavailable sources. Geometry alone cannot fill those cells.
Keep Vector/state work separate, and use the existing [Roofline rules](roofline.md)
without substituting requested bytes for measured traffic or nominal capacity
for sustained throughput.

## A capacity calculation still needs a lifetime proof

For the stated physical M16/N256/D256 and two-byte elements, one L1 slot is
Q=8,192 B, K=131,072 B, V=131,072 B and P=8,192 B:

| Illustrative L1 arrangement | Total bytes | What the arithmetic establishes |
|---|---:|---|
| Q1 / K2 / V2 / P2 | 548,864 | Exceeds the stated 524,288-B L1 budget |
| Q1 / K1 / V2 / P2 | 417,792 | Fits this L1 sum; arithmetic alone does not prove the schedule or other memories |
| Q1 / K1 / V1 / P1 | 278,528 | Smaller L1 sum; possible serialization is not a measured result |

The recommendations' reported 532,480 B at V allocation equals Q1+K2+V2
before P is included. Reconstructing that number does not replay the unavailable
historical allocator run. K1 is legal only if its final QK operand read retires
before overwrite; V and P can still serve a delayed PV. When K also serves as
V, its later reader changes that conclusion. Use the existing
[last-reader and slot derivation](pipeline-model.md#derive-physical-storage-and-credit)
and count every storage level, then check synchronization, backend acceptance and timing through the [evidence table](../common-language.md#evidence). The
[independent K/V slot example](../../../library/examples/api/independent_kv_slots#independent-last-readers)
has its own bounded source/model/emission/native records for these policies;
it is not full attention or a fourth performance scene.
Its current evidence establishes those allocations, the located oversized
refusal and six PyPTO native bitwise passes. Its [five-policy timing illustration](../../../library/examples/api/independent_kv_slots#preliminary-device-timing)
retains one acquisition per policy and includes cases where fewer slots or
lookahead were slower. These results concern the new primitive; the historical
R7 program remains unavailable.

## A completed case 6 scheduling comparison

The captured native comparison (`a5_mha_fp16_bf16/evidence/schedule-study/native-summary.json`)
changes the order of **current-item QK/softmax** and **previous-item PV/finish**.
Restoring that one phase swap makes the two source ASTs equal; the other
execution files, head mapping, inputs, M128/N128, grid, slot/credit counts,
cache tags and numerical order match. This compares two P2 schedules. The
earlier one-slot resident implementation has different V traffic and is not
the control for this attribution.
The enumerated source costs (`a5_mha_fp16_bf16/evidence/schedule-study/requested-cost.json`)
bind the following full-case quantities.

| Shared full-case quantity | Both schedules |
|---|---|
| Work / launch | 16,384 logical rows; 128 items; 28 active Cube / 56 Vector; 4–5 items per Cube, at most 640 output rows |
| L1 | Q and K: 128×128 BF16, one slot each; V and P: the same shape, two slots each; total 196,608 B |
| L0C | Score: 128×128 FP32, two slots; product: the same shape, one slot; total 196,608 B |
| L0A / L0B | Each owns two 32-KiB operand slots, 65,536 B per space; raw/typed aliases do not add storage |
| UB per Vector | Score 64×128 FP32 ×2; product 64×128 FP32 ×1; P 65×128 BF16 ×2; denominator 1×64 FP32 ×2; output 64×128 BF16 ×1; total 148,480 B |
| Credits | QK 2; P publication 2; product 1 |
| GM requests | Q 128 loads; K 52 loads; V 92 loads. K+V payload 4,718,592 B; Q+output 8,388,608 B |
| Cube work | 128 QK and 128 PV tile products; 1,073,741,824 FLOPs, also the useful dense count for this noncausal full case |
| FIX / publication payload | Score FIX 8,388,608 B; product FIX 8,388,608 B; P publication 4,194,304 B |
| Measured hardware stage times / HBM/L2 bytes / pipe utilization | UNKNOWN for this case6 comparison |

The P buffer's extra UB row supplies its NZ pitch. The FP32 denominator keeps
its own item slot until finish; retiring P's L1 reader does not retire that
denominator or the output UB. K and each V slot have separate head tags, which
explains why their load counts differ even though the shapes match. These are
source requests, not measured HBM traffic. Read the source pair and its proof
(`a5_mha_fp16_bf16/evidence/schedule-study/source-proof.json`)
before applying the same counts to another implementation.

| Native phase order | Three independent acquisition medians, µs | Median / range, µs |
|---|---|---|
| QK/softmax current, then PV/finish previous | 12.25, 12.38, 12.08 | 12.25 / 12.08–12.38 |
| PV/finish previous, then QK/softmax current | 15.14, 15.32, 15.27 | 15.27 / 15.14–15.32 |

Each acquisition retains the original profiler/parser, three warmup and five
active steps, exact source/artifact/runtime/input bindings, original precision
and one target kernel per active step. QK-first records a **1.2465× speedup**
over the PV-first control and a **19.78% latency reduction**. Speedup is
`15.27/12.25`; reduction is `(15.27-12.25)/15.27`. The 13.55-µs published
case6 baseline is historical metadata. The separately measured live official
FIA median is 18.07 µs. These two schedules and their six development acquisitions
retain their original execution identities. The unit's final results stay separately
bound in its historical performance record; the demo binds none of them.

The model interval record (`a5_mha_fp16_bf16/evidence/schedule-study/model-intervals.json`)
uses a smaller BF16 case: B1/SQ257/SKV128/H2/D128 on one Cube. Its six items
cover each head's query rows 0–127, 128–255 and the one-row tail 256; the
head/query/slot legend and both original 64-stage interval sets are retained.
For every Cube stage/item, intersect the relevant Vector intervals with that
Cube interval, then take their **union across the two Vector participants**.
The table sums those unions over the recorded item pairs.

| Model phase order | Softmax(current) ∩ PV(previous), union cycles | Finish(previous) ∩ QK(current), union cycles | Complete model cycles |
|---|---:|---:|---:|
| QK current first | 1,797 | 0 | 37,922 |
| PV previous first | 0 | 1,261 | 40,774 |

These intervals establish model scheduling. They are not hardware stage
durations, and their union or model-cycle difference cannot be converted into
the native microseconds saved. Case7's separately captured vendor PMU columns
belong to its own source and diagnostic protocol; they do not fill the missing
case6 hardware cells above.

## Keep case 7 PMU diagnostics in their own scope

The three captured development diagnostics (`a5_mha_fp16_bf16/evidence/diagnostics/index.json`)
retain each factory, raw CSV and source identity. The following are medians
of five instrumented target tasks, using the vendor's original column names:

| Captured case7 source | `aic_mac_ratio` | `aic_mte2_time(us)` | `aiv_vec_time(us)` | `cube_utilization(%)` |
|---|---:|---:|---:|---:|
| `preload-v1` | 0.270 | 114.215 | 207.300 | 94.696 |
| `preload-cache256-v1` | 0.300 | 56.521 | 194.690 | 95.158 |
| `preload-transposed-v1` | 0.358 | 56.613 | 128.111 | 95.252 |

Preserve these vendor definitions and denominators. They do not identify the
critical-path wait or measure peak-FLOPs efficiency; `cube_utilization(%)`
must not be relabelled as MAC efficiency. Hardware HBM/L2 bytes and sustainable
bandwidth remain UNKNOWN. Instrumented duration is excluded from performance
qualification. These source-specific diagnostics cannot supply case6 stage
times or establish code gains across a later hardware rebinding.

The historical recommendations also report no stable gain from an N256 VF-unroll change.
Its source comparison and raw timing samples are unavailable here. Retain this
as **reported / raw evidence unavailable**, not a reproduced regression or a
measured limit on loop optimization. A new source-matched comparison is needed
before assigning the current kernel's bottleneck.
