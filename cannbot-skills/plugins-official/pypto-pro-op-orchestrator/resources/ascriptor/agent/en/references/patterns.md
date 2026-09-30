# Choose a dataflow pattern

Derive the math and contract first. Patterns explain composition; a runnable demo
and the current declaration provide exact API calls. The kernels owner supplies runnable
demos and nothing else — no measured support and no backend matrix, so a demo that fits is
a starting point, never a support statement. The generated indexes are rebuilt from that
owner, not copied as a second source of truth.

| Dataflow | Invariant to preserve |
|---|---|
| Cube-only matmul | One owner per output tile; explicit initialized/accumulating K phases |
| A2 row reduce then broadcast | Reduce groups to one row scalar; scratch meets the next vector footprint |
| A2 independent group reduction | `count_per_rep` and repeat stride are distinct; never merge independent group scalars |
| A5 packed cast | Prove sparse register placement, live reinterpret alias and pack/unpack storage order |
| Online softmax | Score-domain mask before max, declared sum precision, delayed value-path cast |
| Cube/vector bridge | Device-specific physical layout and publish/consume/reuse edges |
| Lookahead and drain | Every delayed stage consumes its own work index; drain publishes final outputs |
| Shared slot roles | Enough simultaneously live storage, with matching credit depth and rotation |

For exact uint2, test all 256 four-value groups and all 256 carrier bytes; confirm
the bit order and permitted input domain. For FP4, include signed zero, tie modes,
defined carrier lanes and the boundary requiring packing. A decode scratch may need
the full register-store footprint even if fewer values are logically live.

Attention variants can differ in precision as well as scheduling. Choose the
[elementwise, right-aligned or Block32 predicate](attention-authoring.md#causal-mask).
A half/hif8/FP8 probability boundary changes the delayed value contract,
while a float row sum may remain unchanged. Rowmax/rowsum outputs and saved state
must be compared when public. Keep separate score/PV scratch if their lifetimes overlap.
Do not transfer old fastest-variant claims or hardware support to a successor result.

If no pattern covers a needed capability, inspect the facade, frontend/lowering,
simulator handler and closest composition. A minimal generated probe and, when needed,
emission/board evidence can establish a precise gap. An unsuccessful candidate or
an absent gallery entry cannot establish that no composition exists.

See [memory and tails](memory-and-tails.md), [precision](precision.md),
[synchronization](synchronization.md) and [debug](../playbooks/debug.md) on demand.
Use prior evidence for a matching archived failure signature;
its observations retain their original scope and are not new support declarations.

<a id="simt-start"></a>
## Start a SIMT composition

1. Map `(core, vector participant, thread, iteration)` to logical input/output elements
   using the [SIMT identity rules](../../../library/docs/api/simt.md). Prove one writer
   per ordinary output, or count atomic contributors and initialize their destination.
2. For every shared value, name its producer, consumers and required rendezvous scope.
   Choose ordering and rendezvous separately; cross-side publication follows
   [synchronization](synchronization.md).
3. Exercise work smaller and larger than the thread count, a tail and multiple participants.
   Check output poison and traced writers as well as values; equal duplicate writes can
   hide an ownership error. Start from [SIMT atomics](../../../library/examples/api/simt_atomics)
   or the matching owner example.

<a id="sort-start"></a>
## Start a sort or topk composition

For finite FP32 largest, unordered TopK, start with the public
[`radix_topk` composite](../../../library/docs/api/sorting.md#register-radix-selection)
and the [complete algorithm demo](../../../kernels/ascriptor_kernels/algorithms/a5_radix_topk).
Threshold selection stays in registers; one call exposes a complete VF body in IR.
Check fixed capacities, count/k, padding and ties before use, and settle the backend question
by running that folder's `main.py` with the backend you need rather than by looking it up.

1. Fix selected count, ordered/unordered output, tie policy and allowed score domain in
   the task contract. Choose [record operations](../../../library/docs/api/sorting.md)
   only after establishing those requirements.
2. Track score and ID bits together, derive record/scratch footprints and prove each
   merge input already has the required order. Preserve initialized padding and tails.
3. Check selected membership, required ordering, ID multiplicity and score–ID pairing
   before any permitted tie canonicalization. Include duplicate scores and boundary
   sizes; reject missing/duplicate IDs and deliberately mismatched pairs. The
   [record example](../../../library/examples/api/sort_records) supplies primitive controls.

## Select one example

For repeated CVC/VCV/CVCV/VCVC or longer graphs, read [the generic method](pipeline-model.md),
then [the CVC derivation](cube-vector-cube.md) and a matching graph in the
[mixed-pipeline demo](../../../kernels/ascriptor_kernels/tutorials/mixed_pipeline).
The [numerical recipes](numerical-patterns.md) cover tails, reductions and casts;
[Roofline](roofline.md) connects busy resources to removable work and scheduling space.

From the agent checkout, use the small navigation index rather than preload the gallery:

```bash
python tools/select_example.py --pattern VCVC --language en --limit 1
python tools/select_example.py --query 'tail footprint' --device a5 --limit 2
python tools/select_example.py --pattern mla --language en
python tools/select_example.py --pattern p-publish --language en
```

`--query`, `--pattern`, `--language`, `--device` (`a5|a2|a3`) and `--limit` are the whole
interface. There is no `--dtype`, `--layout` or `--backend` filter: a folder declares no dtype,
layout or backend for one to read.

The answer is the same for both owners, because both are four-file folders: the guide, the
**folder**, its `formula` or `surface`, `topology`, `tags`, a case **count**, a `run` line and a
`support_scope` saying it records no backend result — where it has run is what you run, on the
machine you run it from. No match triggers a focused source/probe investigation; it does not prove a
missing capability.

`--pattern mla` selects the
[MLA demo](../../../kernels/ascriptor_kernels/attention/a5_mla_fp16_bf16). Its exact shapes,
numerical convention and schedules live in that folder: `python main.py --list` prints every
case with the schedule it runs on and, for a model-shape case, the full-shape case it stands
in for; `--variant` keeps only the cases using one schedule. The
[attention topic](attention-authoring.md) connects that entry to a complete authoring method;
the cost case is a focused experimental example. An explicit pattern does not override a
semantic conflict, and a listed case is one exact shape, not a Cartesian-product promise.
See the [completed cost note](mla-cost-case.md).

Whether a backend can run a folder is in neither the index nor the selector. Run that folder's
`main.py` with the `--launcher` and `--backend` you need, on a machine that has the card.
Where a backend or launcher genuinely cannot run it, the reason is a comment in that folder's own
`main.py` naming what is refused and where, and the case is skipped with that reason printed rather
than widened or deleted; the index only records that such a comment exists — `pypto_pro_note` for a
demo, `refusals` for an API example, which also names which backend or launcher it is.

For the historical [E4M3 MLA](../../../kernels/ascriptor_kernels/attention/a5_mla), use
`--pattern mla-e4m3-16x`. Its original Q/K/P format and 16× output convention remain visible;
the old name `mla_hif8` is an exact navigation alias, not a HiFloat8 datatype declaration.
P publication diagnostics use `p-publish`, which selects the library's cube/vector roundtrip and
what that folder itself establishes, rather than inheriting a full MLA result.

## Discovery and scope

The [index guide](../../index/README.md) distinguishes the generated [complete directory](../../index/kernels.json)
from [bilingual topics](../../index/patterns.json). Query exact IDs, owner paths or topic phrases,
for example `--query SIMT`, `--query quantization` or `--pattern api.event_depths`.
Use the [API entry](../../../library/docs/api/README.md) for signatures, [API examples](../../../library/examples/api/README.md)
for primitives and the [kernel gallery](../../../kernels/README.md) for complete algorithms. Neither
owner's folder records a validation stage; a measurement is a
library receipt, scoped by the release record.

`matched` explains retrieval. `related` contains reading suggestions: matrix-product normalization
is not LayerNorm, the SIMT transpose example includes matmul, and unordered topk does not promise sorted output.
`deferred_material` retains deferred scope; existing GELU/SwiGLU units do not establish A5 support.
`reason` distinguishes absent candidates, related reading, deferred material, conflicting filters and insufficient metadata.
`fallback` preserves the request and supplies owner directories. Missing structured dtype/layout facts remain unknown.
