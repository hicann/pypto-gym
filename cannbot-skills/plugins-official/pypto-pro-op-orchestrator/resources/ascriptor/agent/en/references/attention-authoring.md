# Author an attention kernel

**This page is long and most of it is about MLA. Take the section you came for:**

| You want | Section |
|---|---|
| The online-softmax recurrence and the order its steps must keep | [Freeze the numerical sequence](#freeze-the-numerical-sequence) |
| MHA or GQA head mapping from an MLA contract | [Transfer the contract to MHA or GQA](#transfer-the-contract-to-mha-or-gqa) |
| Gathering KV rows that are not contiguous | [Select KV rows by index](#select-kv-rows-by-index) |
| The causal mask as a predicate | [Derive the causal predicate](#derive-the-causal-predicate) |
| Who owns which buffer, and in what layout | [Make layout and ownership explicit](#make-layout-and-ownership-explicit) |
| Which of the named schedules your scene is | [Choose the state and schedule for the scene](#choose-the-state-and-schedule-for-the-scene) |

A worked example of the recurrence is the softmax row of
[numerical patterns](numerical-patterns.md): the whole prefill demo, and the vector side
of it alone.

Use this topic after [preflight](authoring-preflight.md). It connects attention's
math and state to the shared [memory](memory-and-tails.md),
[pipeline](pipeline-model.md) and [cost](roofline.md) rules; those pages retain
the general method. Start with the task's exact mathematical and layout contract.
The examples below come from the
[standard MLA demo](../../../kernels/ascriptor_kernels/attention/a5_mla_fp16_bf16/metadata.json);
every case's exact tuple is a literal in its
[`main.py`](../../../kernels/ascriptor_kernels/attention/a5_mla_fp16_bf16/main.py), and
`python main.py --list` prints the case ids with the schedule each one selects.
For MHA/GQA, first check the family-transfer table before copying
their row grouping, K reuse or slot counts.

| Existing MLA teaching case / origin | Route | What to check |
|---|---|---|
| `decode_single_query` / 7 | `decode_splitkv` | Count independent row work before paying for partial state and KV-split merges. |
| `prefill_causal` / 1 | `prefill_resident` | Reuse resident K across query items and remove recurrence for a complete single KV tile. |
| `prefill_bf16_dn448` / 3 | `prefill_resident` | Retain BF16 rounding boundaries and verify divisible feature splits on the selected backend. |
| `decode_bnsd_two_queries` / 19 | `decode_preload` | Group physical rows correctly and preserve the old KV/P/rescale versions needed by delayed PV. |
| `decode_large_batch` / 10 | `online_paired` | Count repeated KV reads when row work already fills the launch; pair contexts within batch/core boundaries. |
| `prefill_bf16_bnsd_long` / 15 | `online_prefetch` | Recover BNSD query positions and stream multiple KV tiles with M128 and fragmented PV. |
| `decode_bf16_causal_large_batch` / 18 | `online_paired` | Keep each BF16 context's causal state independent and recompute work after pair-aligned partitioning. |

The demo combines the original origins 7/1/3/19 with 10/15/18 for seven exact
full scenes; its other 29 cases are four serial controls and 25 shrunken
models, which add no benchmark origins. Read each case's tuple in `main.py`:
BF16+BNSD has one declared long-prefill case, not an unrestricted shape domain,
and the official scenes the demo does not name stay outside it. The E4M3 MLA
decode in [a5_mla](../../../kernels/ascriptor_kernels/attention/a5_mla/metadata.json)
keeps a different 16× output convention; use the
[semantic selector](patterns.md#select-one-example).

For a worked cost comparison, the [MLA note](mla-cost-case.md) retains
[case 10's split/reuse experiment](mla-cost-case.md#case-10-fill-the-tuning-record)
and the [seven-scene qualification boundary](mla-cost-case.md#from-development-observations-to-final-qualification).
Read these only when the current task needs that comparison. Case numbers identify
historical evidence; they do not choose an implementation for a new contract.

## Transfer the contract to MHA or GQA

Record the KV-head map `g(hq)` before choosing row groups. The MLA demo's Nkv1
and `V=K_nope` are conditions of that demo. Standard MHA reads the supplied
K and V independently for each head; it does not inherit those conditions.

| Decision | This MLA teaching contract | MHA / GQA task to establish |
|---|---|---|
| Q-head → KV-head | `g(hq)=0` | MHA has `g(hq)=hq`. For GQA, use the declared map; `floor(hq/G)` applies only to declared contiguous equal groups of integer size `G=Hq/Hkv`. |
| K/V numerical relation | V equals K_nope for every legal input | Keep both operands unless equality is part of the new contract. Equal values in one test are insufficient. |
| Actual storage alias | Numerical equality does not require shared allocation | Check backing ranges/index maps at each memory level. A GM alias does not establish L1 sharing; actual local aliases share allocation and all reader lifetimes. |
| Query rows that can share K/V | Rows within one batch share the single KV head; causal positions still differ | Reusing one logical KV-head payload requires the same `(batch, g(hq))`. Packed multi-head tiles must retain each head's separate K/V segment and exclude cross-head contributions. |
| L1 K retirement | Include both QK and delayed PV operand staging | With independent K/V, K's last reader can be QK's final operand copy. Include every query item that reuses resident K. |
| L1 V retirement | Uses the retained K allocation | Independent V remains live through its last PV operand copy, including delayed PV. Its slot count can differ from K's. |
| Physical addressing | Uses the demo's Nkv1 offsets | Apply the actual Q/K/V head counts, sequence lengths and strides. Address arithmetic alone does not establish backend tile support. |

For a tensor with dimensions B, S, H, D, use element offsets and strides:

| Layout | Element offset | Sequence stride / head stride |
|---|---|---|
| BSND | `((b*S+s)*H+h)*D+d` | `H*D` / `D` |
| BNSD | `((b*H+h)*S+s)*D+d` | `D` / `S*D` |

Q uses `S=SQ,H=Hq,D=Dqk`; K uses `S=SKV,H=Hkv,D=Dqk`; V uses
`S=SKV,H=Hkv,D=Dv`. Output uses Q's head/sequence positions and Dv.
Multiply by element size for byte offsets. In BSND, a fixed-head sequence tile
has stride `H*D`; treating adjacent heads as one shared logical KV head changes
MHA mathematics. Explicit head packing must prove both QK masking and PV value
ownership for every row. In GQA, even a legal shared head group can require strided
loads. Follow the existing [view-to-physical-tile rules](memory-and-tails.md#trace-a-view-into-its-physical-tile).

Fill the release table below for each new head/item mapping, then derive slots
from [live versions and credits](pipeline-model.md#derive-physical-storage-and-credit).
Independent K1/V2/P2 is a possible schedule to prove, not a consequence of the
name MHA. The [independent K/V primitive](../../../library/examples/api/independent_kv_slots#independent-last-readers)
provides bounded lifetime controls with separately recorded execution stages.
The [MHA cost note](mha-cost-case.md#a-completed-case-6-scheduling-comparison)
contains a completed comparison that changes only phase order, separate model
intervals and native timings, and recomputable historical geometry. Its MHA case
numbers are independent of the MLA case numbers above.

<a id="indexed-kv-rows"></a>
## Select KV rows by index

A sparse or top-k contract supplies its KV rows as indices instead of a window.
That changes how the tile is assembled and nothing in the mathematics above, so
read the route for three decisions and then return here. Its owner is the
[indexed row gather unit](../../../library/examples/api/indexed_row_gather):
one slot's index goes into a cell with `Var.GetValueFrom`, the cell becomes the
row subscript of a GM view, and one transfer moves that row.

| Recorded fact | What it decides |
|---|---|
| Each table is gathered at its own row width | `Dv < Dk` whenever the contract separates them, so no single width addresses both, and the wrong width has two different outcomes. Separately declared K and V tables reject it (`slice [0:64) outside extent 48`); value rows stored as a `Dv` prefix of a `Dk`-wide row accept the same over-wide read silently and return the next field's columns. The unit's `tests/check_row_widths.py` records both. |
| The gather copies are unconditional | A slot padded up to the tile height is dereferenced too, so its index has to name a legal row. What makes a slot padding is that the mask discards its score, not that the copy is skipped. |
| A clamped destination row is legal and wrong | The loop runs the padded slot count while the destination has `live` rows, so `min(row, live - 1)` keeps the address inside the tensor and overwrites the last real row. Derive the guard before the store from the row count, never from the tensor's declared height. |

Then the cost: one gathered row is one transfer descriptor, so a top-k of 1024
rows is 1024 of them where a contiguous window would be one. Count that in
[the transfer budget](roofline.md) before choosing the tile height, and read the
[indexing rules](../../../library/docs/api/storage.md#indexing-and-slicing) for
what a dynamic subscript accepts.

**What no unit covers.** The transfer above is the only part of an index-selected
attention kernel with an owner today, and this page does not imply the rest: a
bottom-right causal geometry whose fully masked rows must emit zeros without
producing NaN, grouped query attention sharing one `(b, n2)` gather result across
its queries, column blocking of the `PV` product when `Dv < Dk`, and the
combination of any of those with the online state below. Derive each from its own
contract and record what you establish. The nearest runnable material is the
[presence-mask demo](../../../kernels/ascriptor_kernels/attention/a5_presence_mask/metadata.json),
which rebuilds a dense predicate rather than moving rows.

<a id="causal-mask"></a>
## Derive the causal predicate

First fix the contract's query/key positions and alignment. Elementwise causality
permits `k_position <= q_position`; right-aligned sequences use local indices
`j <= i + SKV - SQ`. The
[Block32 demo](../../../kernels/ascriptor_kernels/attention/a2_block32_causal/metadata.json)
instead permits `floor(k_position / 32) <= floor(q_position / 32)` in its top-left
coordinates: query 0 sees key 31 within the same block. Preserve the chosen predicate
through tiling and test block boundaries and unequal sequence lengths. That demo is
an A2/A3 demo and states no backend conclusion of its own; select the current task's
own mask contract.

## Freeze the numerical sequence

For the standard MLA cases, scores combine the non-positional and positional
products before the declared scale. Mask before the maximum; invalid probability
positions contribute zero. A vector-register chunk is not a separate softmax
row: combine maxima and sums across every chunk of that row.

For a streamed KV block, keep this order explicit:

```text
m_new = max(m_old, rowmax(score))
alpha = exp(m_old - m_new)
P = exp(score - m_new), with invalid entries zero
l_new = alpha * l_old + sum(P)                 # FP32, before the P cast
O_new = alpha * O_old + cast_input_dtype(P) @ V # FP32 accumulation/state
out = cast_output_dtype(O_final / l_final)
```

The `serial_nd` schedule in the MLA demo's
[`kernel.py`](../../../kernels/ascriptor_kernels/attention/a5_mla_fp16_bf16/kernel.py)
(`make_serial_nd_kernel`) provides a correctness starting point. A single KV tile
can remove old-state rescaling and repeated accumulation, as in the same file's
`prefill_resident` schedule (`make_prefill_resident_kernel`).
It still keeps the unnormalized P cast and final division. Moving normalization
across a low-precision cast changes rounding; algebra alone does not qualify it.
The independent reference and original benchmark comparator remain separate
from an implementation-matching numerical attribution model.

A first-tile specialization may establish maximum, sum and rescale before any
old-state read. Prove its actual writes cover every lane that any later reader
uses, then remove only the redundant initializer and its unused VF. For a
64-lane FP32 state, inspect the full store distribution and mask, including
padding and an inactive participant that still executes the VF. Audit the
first-product path separately: it must avoid old-state reads or provide valid
initialization for them. Multiplying an uninitialized value by zero does not
remove that read. Retain Q/P padding needed by physical Cube operands.

Use nonzero inputs, consecutive single-tile queries on one core, state-slot
wraps and a tail/idle case. Record first writes and subsequent actual state
loads, and compare the independent reference. The recorded initialization
control removed VF work without reducing model duration. One native development
acquisition likewise does not establish a repeatable speedup; keep work counts,
model cycles and measured performance separate.

Integer edits to FP32 exponent bits are not a general replacement for rescale:
zero or a normal-to-subnormal transition violates the required bit relation.
For example, V=K_nope=0 with changing rope scores still requires zero output.
Keep the current FP32 state path unless a new numerical contract and its
boundary controls justify a different representation.

## Make layout and ownership explicit

The query row offset is `((b*SQ+i)*Nq+h)*D` in BSND and
`((b*Nq+h)*SQ+i)*D` in BNSD. A singleton axis can make a view equivalent;
SQ>1 makes the distinction observable. Test nonuniform values on both axes.
For the declared noncausal Nkv=1 path, physical query rows within one batch
can share K and be grouped contiguously. A causal extension must recover each
row's query position; it cannot reuse one mask count for different positions.

In this MLA demo, V equals K_nope numerically; it need not alias the same allocation. Reusing
the already staged K for PV is legal under that contract. Keep inputs immutable,
outputs contiguous, and host layout conversion outside a one-kernel design
unless the task explicitly permits and measures it.

Logical rows, physical cube M and Vector-owned rows are separate quantities.
Initialize padding that a physical matmul reads, while each Vector publishes
only its owned rows. Diagnose packing with the
[narrow roundtrip](../../../library/examples/api/cube_vector_roundtrip),
including its address table and guards. Derive the store from the actual
[register layout](memory-and-tails.md#match-the-mask-to-the-register-layout);
LOWHALF is not a universal repair for every 64-value conversion.
Keep the [UB instruction base aligned](memory-and-tails.md#align-the-ub-instruction-base)
independently of logical payload or masks. A short state load from a GM offset
of 16 bytes into aligned UB is legal; the adapter must represent its physical
carrier correctly.
Also [trace a view into its physical tile](memory-and-tails.md#trace-a-view-into-its-physical-tile):
two score windows can have aligned bases yet still require the wider parent's
row pitch. Keep the logical view, generated carrier and selected backend's
result separate when diagnosing FIX-to-UB copies.
Follow that reference's L0B/Right mapping and fractal-pitch rules when slicing
explicit QK/PV operands; their physical coordinates differ from UB rows.

For causal cropping, use the largest real query position to prove that omitted
keys are invisible to every participating row. Apply the same extent to QK,
softmax, P publication and PV, and check the actual P/V operand reads. A smaller
logical score is not automatically a smaller physical carrier.
The [static capacity contract](../../../library/docs/rfc/0013-pypto-native-synchronization.md#static-local-capacity)
forbids guessing a typed L0 axis from raw-byte storage, the cause of the historical
M10-072 silent truncation. FIX also needs its physical source pitch:
dynamic valid M and dynamic `M_src` are different requirements. The guarded
backend rejects an unproved typed bound or unrepresentable dynamic FIX pitch.
Use supported static branches and correctly sized carriers when needed, then
recheck combined capacity, event budget and native precision. Preserve the
located refusal; do not infer dynamic support from a passing model.

## Choose the state and schedule for the scene

The implementation-specific schedules below belong to the MLA demo;
the [MHA analysis](mha-cost-case.md) uses its own head mapping and evidence scope.

Fill the [before/after tuning tables](../../templates/performance-analysis.md#fill-before-each-tuning-change)
for the requested shape and actual factory parameters. Count complete-row work
before adding KV splits, then include private partials and merge owners.
The [historical case 10 record](mla-cost-case.md#case-10-fill-the-tuning-record)
shows why a small decode's launch and split count do not transfer automatically
to a large batch. It also separates a smaller tile's reduced capacity from its
increased calls and KV requests when all cores already have assigned work.

The `decode_preload` pipeline in
[`kernel.py`](../../../kernels/ascriptor_kernels/attention/a5_mla_fp16_bf16/kernel.py)
(`_build_decode`)
uses M64/N128, keeps FP32 output state in UB and pairs QK/softmax for the current
KV tile with PV/accumulation for the previous one. K, P, score and the delayed
rescale have two slots; the product needs one. Running max/sum follow softmax
order, while the output accumulator consumes the matching older rescale. Use the
[generic pipeline derivation](pipeline-model.md#group-consecutive-stage-pairs)
for its startup, drain and credit accounting. A boundary in the table below is what a
consumer's `free` has to sit after; the calls that enforce it are on
[cross-side handoff](cross-side-handoff.md), and `P in UB` is the one this session got
wrong by returning the slot before the PV matmul had read it.

| Role | Release boundary to preserve |
|---|---|
| Q in L1 | Final QK operand-staging read across the KV tiles that reuse this query tile |
| K shared by QK and PV | The delayed PV's final MTE1 read, not QK completion |
| Independent K in L1 | Final QK operand-staging read; include all declared resident-query reuse |
| Independent V in L1 | Final PV operand-staging read, including delayed PV |
| P in UB | The transfer publishing this version to L1 has completed its source read |
| P in L1 | PV operand staging has consumed that version |
| Rescale factor | The matching delayed accumulator update has read it |
| Score / product in L0C | Their FIX publications have completed before overwrite |
| Output accumulator | Recurrence order within its query item, followed by final normalization |
| Explicit L0B shared by contexts | Every participating context's MMAD reader of that fragment has finished |
| UB product aliased as output | Final conversion and the output MTE3 read have retired, followed by the Vector barrier |

[Version scalar metadata with its payload](pipeline-model.md#keep-scalar-metadata-with-its-payload).
Sharing one L0C allocation is useful for sequential prefill score/PV lifetimes;
overlapping live score/PV generations require separate storage. Removing a
state array can make a larger row block fit, but increasing M also reduces the
number of independent blocks. Recompute work per active core for each scene.

Before concluding that a row block does not fit, check which `dual_mode` the
score drain uses. With `SPLITM` a 128-row L0C tile lands as two 64-row tiles,
one per vector sub-block, so the UB it costs is half of what the row count
suggests. With `SINGLE` the whole 128 rows land in one sub-block's UB, and a
budget computed that way will report M128 as impossible when it is not — while
also halving the vector throughput, so the two losses compound rather than trade
off. The M128 scenes below assume the split drain; see
[device facts](facts-device.md#the-drain-into-the-vector-side-has-no-safe-default)
for when it is unavailable.
The two original short-prefill scenes use a complete N128 KV tile, with M128 for FP16
and M64 for BF16. KV stays resident across that core's query items within each
batch. Dn448 uses divisible feature splits; a mathematical tail is insufficient
if the chosen backend cannot materialize its window.
Use the [paired slot/window controls](../../../library/examples/api/cube_vector_roundtrip#slot-and-window-boundary-checks)
to check the actual boundary: for K256 b16 PV, an implicit L0B operand needs
`256 * splitn * 2` bytes. Reducing a different buffer does not repair an
oversized L0B slot, and one rejected window does not invalidate every subview.

The `online_prefetch` route in the same
[`kernel.py`](../../../kernels/ascriptor_kernels/attention/a5_mla_fp16_bf16/kernel.py)
(`_build_online_prefetch`)
serves origin 15 with M128/N128. It loads K(t), consumes PV(t-1), then produces
QK(t), preserving a prologue and final drain. Two K slots feed one score/P/state
context; two 256-column PV fragments bound product storage. Multiple KV tiles
retain FP32 recurrence. Select an unmasked softmax only when every real row
sees that whole tile; the first tile can establish the numerator directly.

The `online_paired` route (`_build_online_paired`)
serves origins 10 and 18 with one or two M64/N128 contexts per pack. Share
K/RopeK and explicit L0B staging across participating contexts, while keeping
each context's P, score, FP32 state and product credits separate. A pair never
crosses a batch or core boundary. A context that retires its causal prefix
first still holds its final product credit while its partner continues.
Both new routes alias the final FP32 UB product as b16 output storage; that
alias extends its last-reader lifetime through the output DMA.

For `I` original M64 items and `C` cores, the paired route uses
`T=ceil(I/C)` and `q=2-(T%2)`; `mla_online_paired` computes exactly that as
`ceil_items`, `quantum` and `group_count` before deriving each core's interval.
The partition preserves the per-core item ceiling T, hence the fixed physical
M64 row-slot bound. It does not generally preserve valid-row counts, causal
FLOPs or latency. With q=1 the original intervals are unchanged. Enumerate the
actual per-core KV prefixes and requests before claiming better balance or less
work.

The `decode_splitkv` schedule (`_build_split_decode`)
needs explicit partial-output/max/sum layout, initialization, merge owner and
completion protocol. Its small decode scene uses M64/N256 with eight KV splits:
16 producer items feed 32 independent four-row merge items. A 16-Cube/32-Vector
launch matches these counts; record the launch separately from total device
cores. Each partition contains one tile, so its numerator can publish directly
from the PV result without the old-state recurrence. That publication makes
MTE3 the handoff's last reader, replacing the Vector update's V endpoint.

Every Vector, including an idle owner, joins the all-Vector completion barrier
before merge reads private workspace. Atomic updates or fences alone do not
establish that rendezvous. For partials `(m_j, l_j, O_j)`, merge with
`m = max_j(m_j)`, `w_j = exp(m_j-m)` and
`out = cast(sum_j(w_j*O_j) / sum_j(w_j*l_j))`. The four-row merge batches the
cross-partition max and sum loads into two strided DMAs; a row group cannot
cross its producer's Vector-state or M-tile boundary. Preserve this ownership,
the final numerical order and the one-runtime-kernel contract when retuning.

## Turn observations into a reusable result

Use [retiling cost comparisons](roofline.md#compare-work-after-retiling) before
attributing a benefit to overlap. A larger KV tile changes recurrence and FIX
traffic counts; an N128 pipeline versus an N256 serial kernel is not a pure
scheduling control. Verify compute overlap using actual same-core stage/item intervals and the [evidence table](../common-language.md#evidence).

Retain negative controls for wrong layout, a lost drain, an incorrect delayed
slot, untouched output and input writes. Model cases should preserve the
selected implementation branch and exercise multiple wraps, multiple items,
padding and idle cores. Record each stage through the evidence table and preserve the original native comparator. A runtime
tail accepted by the frontend may still fail backend tile materialization.

For MHA, follow the
[demo folder](../../../kernels/ascriptor_kernels/attention/a5_mha_fp16_bf16/metadata.json)
and the exact tuple each case declares in its
[`main.py`](../../../kernels/ascriptor_kernels/attention/a5_mha_fp16_bf16/main.py).
A public FIA source revision can suggest tiling or stage distances; it does not
identify the selected kernel in an installed `torch_npu` binary. Bind the package,
hardware and actual dispatch separately. The demo declares no per-scene numerical,
launch or performance status, so obtain each one yourself: `python main.py --case <id>`
for the precision check, `--launcher board` on a machine with the card for anything
about hardware. The [MHA cost note](mha-cost-case.md) carries the measured FIA
targets and the case 6 scheduling comparison as historical records, with the route
that restores them.

The MLA demo likewise records no source identity, no qualification and no timing.
`python main.py` runs every case's precision check under the functional simulator,
`--launcher pipesim` runs the lowered pipeline under the event/hazard model, and
`--launcher board` is the only way to obtain a hardware result. The earlier unit's
validation, performance and optimization records are historical and restore from the
library's retired attention exports
under `examples/attention/a5_mla_fp16_bf16/`; read their backend, hardware, variant,
exact cases and measurement protocol before quoting them, and do not transplant
latency or qualification to a new shape or device. The
[evidence boundary](mla-cost-case.md#from-development-observations-to-final-qualification)
holds the MLA timing milestone with its sources, and the
[historical layout experiment](mla-cost-case.md#a-completed-performance-note)
illustrates how to retain a correctness repair that did not establish a speedup
or a dominant bottleneck.
