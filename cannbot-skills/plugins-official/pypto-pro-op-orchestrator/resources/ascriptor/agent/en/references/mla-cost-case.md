# MLA: record the cost of a tuning change

Start ordinary FP16/BF16 MLA with the
[standard MLA demo](../../../kernels/ascriptor_kernels/attention/a5_mla_fp16_bf16/metadata.json).
Its original four foundations teach single-query decode, causal prefill, BF16/Dn448, and direct
two-query BNSD addressing. The current seven-scene demo adds large-batch FP16 decode,
long BF16/BNSD prefill and large-batch BF16 causal decode. Read each case's exact parameters
where they are declared, in the `CASES` list of
[`main.py`](../../../kernels/ascriptor_kernels/attention/a5_mla_fp16_bf16/main.py);
their dtype/layout/shape combinations do not form an unrestricted product.
Use the [narrow roundtrip](../../../library/examples/api/cube_vector_roundtrip)
to diagnose P publication, LOWHALF stores and physical NZ pitch.

**Where the numbers on this page come from.** The demo records no measurement of any
kind. Every observation quoted below belongs to the units this demo replaced, and every
one of them restores intact from the library's
retired attention exports —
the historical records are not part of this source snapshot. Paths in parentheses below
are relative to the restored `examples/attention/`. Restoring a record recovers original
bytes and identities; it is never a current execution claim.

## A completed performance note

The historical case 7 record (`a5_mla_fp16_bf16/study/historical-case7.json`)
owns the observation identities and source summaries below. These are dated
PyPTO-Pro hardware observations of earlier candidates, separate from validation
of the current teaching source. Each row passed the recorded precision check.

| Candidate / collection | Latency, hardware μs | Result scope |
|---|---:|---|
| R8 ND, local comparison | 63.00 | Reference for the local layout experiments |
| R8 compact NZ, local comparison | 63.51 | No observed end-to-end improvement |
| R16 compact NZ, local comparison | 64.03 | Row grouping and work distribution also changed |
| R8 ND, full-suite collection | 63.38 | Separate collection; do not substitute for the local 63.00 |

The published case 7 baseline is 15.7 μs. The historical record retains its
baseline field and source; the published table permits proxy-filled values and
does not identify individual entries as same-card native measurements. These
candidate observations are single collections, not three independent repeats
of one candidate. Their small differences do not establish a statistically
significant slowdown or identify a dominant stage.

- **Bottleneck evidence:** full-kernel latency is available; stage measurements,
  valid counters and the dominant bottleneck remain UNKNOWN.
- **Hypothesis:** changing ND publication to compact NZ might reduce P movement
  or packing cost. Correct LOWHALF stores establish footprint correctness only.
- **Expected space:** UNKNOWN until the changed work and its critical-path
  contribution are measured; successful packing does not predict a speedup.
- **Constraints:** preserve score scaling, online state and P conversion,
  causal/layout semantics, physical footprints and publish/consume/reuse edges.
- **Result:** the local R8 NZ result provides no observed end-to-end benefit.
  Withdraw the claim that P publication is an established main bottleneck.
  Its cost remains a hypothesis to measure.

## Recompute the work picture

For this single-query case, `B=1`, `Nq=128`, `SKV=2048`, `Dn=512`, `Dr=64`.
One MAC counts as two FLOPs, so useful QK plus PV work is
`2 * 128 * 2048 * (512 + 64 + 512) = 570425344` cube FLOPs. Count softmax,
state updates, reductions and casts separately from that cube total.

| Quantity | What the experiment establishes / what to retain |
|---|---|
| Independent row blocks | R8: 16; R16: 8 |
| Active Cube cores | At most 16 or 8 respectively on the recorded 28-Cube target; actual activity needs evidence |
| Work per core | Derive from the actual block assignment; fewer blocks changes available parallel work |
| Padded compute | All three source matmuls use M16. R8's 8 useful rows give 1140850688 equivalent cube FLOPs; R16 gives 570425344. These are source-derived work counts, not hardware counters |
| KV iterations and state | KV64 gives 32 tiles per row block: 512 block/tile visits for R8, 256 for R16, with different rows per visit. Count vector work separately |
| Traffic | Separate unique input values, separately allocated V with values equal to K, requested transfers and per-core rereads |
| Residency | Count all live L1/UB roles and physical pitches, including buffers awaiting their final reader |
| Measured HBM/L2 bytes, stage time | UNKNOWN in these observations |

R16 uses both Vector participants and halves the independent block count. It
is not a controlled test of participant count alone. A prefill with many items
per core has different reuse and scheduling opportunities; this decode result
does not establish the benefit of a prefill layout or pipeline.

Compare ND/NZ with row grouping fixed first. For a grouping change, report the
changed padding, participant rows, block count and traffic together. Use
[Roofline](roofline.md) and the [performance template](../../templates/performance-analysis.md)
to retain source identities and remaining unknowns. Keep model cycles, hardware
μs, requested bytes and measured HBM/L2 traffic separate; isolated stage times
cannot simply be summed into the fused-kernel latency.

Current correctness is whatever `python main.py --case <id>` prints for the case you
run; the demo holds no timing and no qualification at all. The earlier unit's
`validation.json` and `performance.json` restore from the same archive as historical
records. Read their backend, hardware, variant, exact cases and measurement protocol
before quoting either; historical observations do not extend what the demo checks.

## Case 10: fill the tuning record

The historical fresh-evaluation record (`a5_mla_fp16_bf16/study/historical-fresh-case10.json`)
binds the saved DSL, generated tiles and reports for these three candidates.
This is the 2026-09-08 evaluation of the earlier library/guide snapshot, and
times nothing that the gallery carries today. All three passed the recorded
first/final accuracy and single-runtime checks with the same development seed.
Each latency is one original profiler/parser collection, not a median of three
independent collections. The published `baseline_perf_us` is 223.995 μs; its
metadata retains the possibility of unlabelled proxy values.

Case 10 is FP16/BSND, `B=60, SQ=1, Nq=128, SKV=2048, Dn=512, Dr=64`,
noncausal with Nkv1: 7,680 distinct output rows. Below, S is M64/N256 with
eight KV splits and four-row merge items; P64/P32 are M64/N128 and M32/N128
preload. All three have no padded query rows. KiB/MiB use powers of 1024.

| Dispatch and observation | S | P64 | P32 |
|---|---:|---:|---:|
| Historical latency, μs | 825.20 | 291.82 | 435.18 |
| Accuracy / one runtime kernel | PASS / PASS | PASS / PASS | PASS / PASS |
| Published 1.0× target | Below target | Below target | Below target |
| Row groups / producer items | 120 / 960 | 120 / 120 | 240 / 240 |
| Launch Cube / Vector | 16 / 32 | 28 / 56 | 28 / 56 |
| Cube cores assigned producer work | 16 | 28 | 28 |
| Busiest Cube: items / producer row-visits | 60 / 3,840 | 5 / 320 | 9 / 288 |
| Distinct rows touched by that Cube | 512 partial rows | 320 complete rows | 288 complete rows |
| Merge items / busiest Vector items / output rows | 1,920 / 60 / 240 | N/A | N/A |

The split's row-visits count the same row in multiple KV partitions; they are
not new independent output rows. Producer work uses balanced floor intervals;
merge assigns four-row groups by Vector index. Assigned cores are derived from
that source and launch, not an occupancy counter.

Record the buffers before attributing a timing change. A cell gives physical
`shape×slots`; Q/K/P and final output staging are FP16. Score, product, online
and merge state, including every private-GM partial, are FP32.
UB quantities are **per Vector**, L1/L0C per Cube. These saved sources' sums
also match the union of their emitted physical address ranges; typed aliases
are not additional allocations.

| Allocation | S | P64 | P32 |
|---|---|---|---|
| Q / L1 | 64×512×1 | 64×512×1 | 32×512×1 |
| RopeQ / L1 | 64×64×1 | 64×64×1 | 32×64×1 |
| K, reused as V / L1 | 256×512×1 | 128×512×2 | 128×512×2 |
| RopeK / L1 | 256×64×1 | 128×64×2 | 128×64×2 |
| P / L1 | 64×256×1 | 64×128×2 | 32×128×2 |
| Score / L0C | 64×256×1 | 64×128×2 | 32×128×2 |
| Product / L0C | 64×512×1 | 64×512×1 | 32×512×1 |
| Score / UB | 32×256×1 | 32×128×2 | 16×128×2 |
| Product / UB | 32×512×1 | 32×512×1 | 16×512×1 |
| Compact-NZ P / UB | 33×256×1 | 33×128×2 | 17×128×2 |
| Max and sum / UB, each | 1×64×1 | 1×64×1 | 1×64×1 |
| Rescale / UB | 1×64×1, recurrence unused | 1×64×2 | 1×64×2 |
| Accumulator / UB | 32×512×1, used for merge | 32×512×1 | 16×512×1 |
| Output staging / UB | 32×512×1 | 32×512×1 | 16×512×1 |
| Merge max/sum/weight / UB, each | 8×64×1 | None | None |
| Partial output / GM | 960×64×512×1 | None | None |
| Partial max/sum / GM, each | 1,920×1×64×1 | None | None |
| **L1 total, KiB** | **392** | **392** | **340** |
| **L0C total, KiB** | **192** | **192** | **96** |
| **UB per Vector total, KiB** | **215.25** | **209.5** | **105.5** |
| **Private GM total, MiB** | **120.9375** | **0** | **0** |

Compact P has one physical padding row per NZ column. Max/sum/rescale carry
32/32/16 live entries within 64-element rows; merge state carries four live
entries per row. Every candidate also reserves two 32-KiB shortcut slots in
**each** of L0A and L0B, or 64 KiB per operand memory. Its typed views alias
those reservations. Retaining two K slots while changing N128 to N256 alone
makes K occupy 512 KiB of L1; adding Q/RopeQ/RopeK/P gives 712/644/610 KiB
for M64/M32/M16. That source-derived sum already exceeds the 512-KiB L1 budget;
it is a capacity calculation, not a new backend-bug finding.

| Requested repeated work | S | P64 | P32 |
|---|---:|---:|---:|
| KV iterations per producer / global / busiest Cube | 1 / 960 / 60 | 16 / 1,920 / 80 | 16 / 3,840 / 144 |
| Full KV rereads per batch, across row groups | 2 | 2 | 4 |
| Full KV read equivalents on busiest Cube | 7.5 | 5 | 9 |
| Q + RopeQ requests, MiB | 67.5 | 8.4375 | 8.4375 |
| K + RopeK requests, MiB | 270 | 270 | 540 |
| Busiest Cube K + RopeK requests, MiB | 16.875 | 11.25 | 20.25 |
| Score / product FIX group publications, each | 960 | 1,920 | 3,840 |
| Score / product FIX bytes, MiB | 60 / 120 | 60 / 240 | 60 / 240 |
| P Vector publications / MiB | 1,920 / 30 | 3,840 / 30 | 7,680 / 30 |
| Softmax VF calls / row max-and-sum updates | 1,920 / 61,440 | 3,840 / 122,880 | 7,680 / 122,880 |
| Output recurrence VF calls / row updates | 0 / 0 | 3,840 / 122,880 | 7,680 / 122,880 |
| Partial GM publications | 5,760 | 0 | 0 |
| Merge state / product GM load calls | 3,840 / 15,360 | 0 / 0 | 0 / 0 |
| Merge partial-row contributions | 61,440 | 0 | 0 |
| Partial output GM write / read, MiB | 120 / 120 | 0 / 0 | 0 / 0 |
| Partial max+sum GM write / read, MiB | 0.9375 / 0.46875 | 0 / 0 | 0 / 0 |
| Collective all-Vector barriers | 1 | 0 | 0 |
| Final output GM stores / MiB | 1,920 / 7.5 | 240 / 7.5 | 480 / 7.5 |

A FIX group publication here is one source Cube transfer split to two Vectors,
not a vendor instruction count. GM byte counts describe logical requested
payloads; short UB destinations can have larger padded footprints. No HBM/L2
traffic or dominant hardware bottleneck was measured in these comparisons.

**Split → P64:** 120 complete-row items already provide work for all 28 Cube
cores. The retained preload removes partial GM storage/merge and eightfold Q
rereads, while N, recurrence, FIX traffic, buffering and launch also change.
The improvement is an overall implementation result; these records do not
isolate the contribution of merge or launch alone.

**P64 → P32:** the hypothesis was better parallelism or less row waste. There
was no query-row padding to remove, and assigned Cube cores remain 28. The
busiest core has fewer output rows, but twice as many row groups double global
KV requests. P and FIX **bytes**, total per-row recurrence and useful cube work
stay unchanged; publication/VF **calls** increase. The measured regression
rejects that tuning choice for this historical case, not smaller M in general.
Keep the confirmed bottleneck UNKNOWN and fill the
[same before/after template](../../templates/performance-analysis.md#fill-before-each-tuning-change)
for the next candidate.

## From development observations to final qualification

The historical optimization study (`a5_mla_fp16_bf16/study/optimization.md`)
connects the selected routes to their work and lifetime changes. The seven full
origins are 7/1/3/19/10/15/18; the demo's four serial controls and 25 shrunken
models add no benchmark origins. Keep these records separate:

| Record | What it establishes |
|---|---|
| Historical four-scene performance (`a5_mla_fp16_bf16/study/stage1-performance.json`) | The earlier source/runtime and four exact scenes; it does not qualify the three additions. |
| Fresh development observations (`a5_mla_fp16_bf16/study/fresh-observations.json`) | Individually bound source, compiler runtime, artifacts and original-parser acquisitions; PMU attempts have their own scope. |
| Historical performance record (`a5_mla_fp16_bf16/performance.json`) | The former declared scope and status, with each final latency computed from three independent original-parser acquisitions under the recorded source/runtime. It declared that scope for a source the gallery no longer carries; the demo declares none. |

Timing milestone of 2026-09-08, recorded against that earlier source: all 21 formal
profiles passed, and each of the seven three-acquisition medians met its fixed
published baseline. All seven separate fresh-seed checks using 73510929 also passed.
The native record holds 28 completed acquisitions for these seven tuples — 21
`fresh-round-<case>-profile-{1,2,3}.json` and 7 `fresh-round-<case>-fresh-seed.json`
under `a5_mla_fp16_bf16/evidence/`. The complete
export/isolation checks
(`docs/migration/fragments/mla-fresh-export-20260908.json`, in the library's
`completed-migration` archive) also passed against the owners committed then. All of
this binds the file hashes in that record's `source_identity`, none of which the demo
carries; it qualifies nothing you can run today.

A single acquisition's five active tasks are not three independent acquisitions.
Read accuracy, single-runtime and recommendation status for the exact case;
tie final timing qualification to its acquisitions using the [evidence table](../common-language.md#evidence). An unstarted attempt contributes no latency and
does not determine whether its requested metric is supported.

The M128-to-paired comparison changes the compiler runtime along with row
tiling, storage and scheduling. Its observed difference cannot isolate the
benefit of shared L0B. The subsequent paired-to-quantum comparison keeps the
compiler runtime and changes only the partition expression in executable
source. Its individual acquisitions still do not establish a statistical effect.

The pair-quantum argument (`a5_mla_fp16_bf16/study/optimization.md`, "Pair quantum and
its exact bound") preserves the original per-core item ceiling, hence its fixed-M
physical-row bound. The partition expression it reasons about is still live in the
demo, as `ceil_items` and `quantum` in `mla_online_paired`. Enumerate live rows, causal work and KV requests separately. In origin 10,
q=1 keeps the original intervals and counted work; a different observed latency
cannot be attributed to reduced partition work. Static paired costs for an
unmeasured scene remain static evidence.

Read counter definitions and parser provenance in the observation record.
`cube_utilization(%)` expresses aggregate executed Cube cycles over the task
interval and hardware capacity; it is not useful-FLOP or MAC roofline efficiency.
Pipe ratios can overlap and use different denominators. The installed CANN
parser bytes were not independently pinned to the public parser source, and
PMU captures do not enter the final latency median.
