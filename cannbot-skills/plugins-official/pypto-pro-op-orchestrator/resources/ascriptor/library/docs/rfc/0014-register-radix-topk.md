# RFC-0014: Register radix TopK as a composite API

Status: implemented, with scoped validation. The maintainer chose
register threshold selection and frontend composition of existing operations.

## Contract

The A5 facade exports:

```python
radix_topk(dst_values, dst_indices, src, count, k, *, largest=True, sorted=False)
```

This is a kernel-body UB-to-UB composite call. The initial domain deliberately
matches the canonical four-level MSD algorithm: `src` is FP32 `[1,4096]`,
`dst_values` is FP32 `[1,512]`, and `dst_indices` is INT32 `[1,512]`. All three
are complete, unmodified UB buffers with distinct backing allocations. Their
ordinary 32-byte alignment remains the storage contract. Offset/window views
and aliasing, including different views of one allocation, are not admitted.

`count` and `k` are Python integer constants or device INT32 scalars/cells.
Require `1 <= count <= 4096` and `1 <= k <= min(count,512)`. The first `count`
inputs must be finite; the remaining input slots must contain negative
infinity. Only the first `k` entries of each output are defined. Inputs are
unchanged. Values are unordered and must preserve the exact bits at their
returned input indices. Indices are distinct and in `[0,count)`, and every
value strictly greater than the numerical threshold must be present. Tied
threshold membership is unspecified, including numerical ties between signed
zeros. Stable ordering and nonfinite active inputs are not part of this API.

Only the literal options `largest=True` and `sorted=False` are admitted in
this revision. Unsupported options, signatures, storage, dtypes, aliases,
function/device contexts and statically known scalar-domain violations produce
a diagnostic at the call site. Runtime scalar bounds and input data/padding
are caller preconditions; source emission does not insert or promise hardware
assertions or host-side selection. Complete examples validate those conditions
before launching. Dynamic bounds are not silently clipped.

## Representation and implementation

The frontend validates the call boundary and then compiles the library's DSL
helper through the existing inline/VF machinery. The Surface module contains
`cf.call` and a complete VF body built from existing `vf.*` and scalar
loop/address operations. It introduces no Surface or Lowered TopK opcode and
does not change the legacy unimplemented `vec.topk_radix` contract.

The algorithm uses bitwise monotone keys, two accumulated byte-histogram
registers at each of four unrolled byte levels, vector comparison/reduction
to select the threshold bucket, register prefix/remaining-count updates, and
rank/scatter compaction. There is no scalar threshold binary search, scalar
load of a UB histogram, or per-level V-to-S synchronization. Histograms and
threshold state require no extra UB scratch. Runtime traversal loops remain
loops. No CANN internal header or external algorithm archive is a dependency.

Read/write access summaries are obtained from the VF body, so caller autosync
and the normal simulator/backend paths observe the source read and both
output writes. Callers still order DMA and buffer reuse. Necessary ordinary
VF load/store ordering is retained. The initial complete launch is one vector
participant; broader launch configurations require their own unit contracts.

## Ownership and validation

The INT32 index output is carried through a UINT32 memory view created at the
call site, and its register payload is bit-reinterpreted to UINT32 for scatter.
This preserves address, index bits and predicates while avoiding the signed
offset cast emitted by the PyPTO version in [A5-UP-037](../upstream.md#a5-up-037).
It adds no arithmetic conversion or scalar threshold work.

Library owns the public composite API and its reusable implementation. Kernels
owns complete launch examples, input generation, independent references and
comparison/support metadata. Factoring the reusable implementation into
library avoids duplicate algorithms and a library dependency on kernels.

Validation covers independent numerical selection, raw index range and
uniqueness, exact index/value correspondence, ties, signed zeros, count/k
boundaries, unchanged inputs, and repeated calls. Inspect IR and source to
ensure threshold work stays in registers. Record simulation, pipeline checks,
three-backend source emission, vendor compilation and device runs separately.
Installed-wheel import/emit must work outside the checkout without importing
Torch, NumPy or vendor Python libraries until execution requires them.

Selection of an alternative TopK implementation is a future design question.
If a semantic Surface operation becomes useful, decompose it before autosync
and address allocation using shared passes and provenance, rather than hiding
the expansion in individual backend printers.
