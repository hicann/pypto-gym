# Sorting scores with identifier payloads

## Register radix selection

`ascriptor.a5.radix_topk(dst_values, dst_indices, src, count, k, *,
largest=True, sorted=False)` is a composite API called from an A5 kernel.
It compiles a maintained DSL implementation into an inspectable VF body,
with no opaque TopK operation or scalar threshold binary search. Histograms,
threshold, prefix and remaining count stay in registers; compaction uses
rank and scatter. The [standalone example](../../examples/api/radix_topk)
shows one call and independent raw-index/value checks.

The first revision accepts complete, separate UB allocations: FP32 source
`[1,4096]`, FP32 values `[1,512]`, and INT32 indices `[1,512]`. Require finite
values in the first `count` source slots and negative-infinity padding in all
remaining slots, with `1 <= count <= 4096` and `1 <= k <= min(count,512)`.
Inputs are unchanged; only the first k entries of each output are defined.
Outputs are unordered, indices must be unique and valid, and each output value
preserves its indexed input bits. Threshold ties, including signed zeros,
permit any legal membership. Scalar bounds may be dynamic INT32 parameters;
callers validate their runtime values and input data before launching.

Only literal `largest=True` and `sorted=False` are supported. Unsupported
flags, dtypes, shapes, offset views, aliases and known invalid scalar constants
are rejected at the call site. These allocation sizes bound the initial API;
they are not limits of the vector ISA. The operation owns no extra UB scratch
and creates no DMA: callers order input/output transfers and reuse with the
ordinary synchronization mechanisms. See [RFC-0014](../rfc/0014-register-radix-topk.md)
for the exact admission and ownership contract.

```sh
python radix_topk.py inspect --output tmp/ir
python radix_topk.py check --count 65 --k 7 --launcher pipesim
python radix_topk.py emit --backend pypto_pro --output tmp/source
```

The [complete demo](../../../kernels/ascriptor_kernels/algorithms/a5_radix_topk)
adds the full boundary/tie case matrix and independent reference; `python main.py --list`
in that folder prints its cases. Historical board results do not qualify a newly
factored library implementation.

## Sorting records

`sort32`, `mergesort4` and `mergesort_2seq` use UB FP32 carrier streams. The
even words contain scores; odd words contain identifier bits reinterpreted as
FP32. Copying the odd words numerically as floating values can destroy the
identifier. The [standalone sort example](../../examples/api/sort_records)
contains all three operations and an independent generated reference.

`sort32` reads 32 scores and 32 UINT32 identifiers per repeat and writes 64
interleaved words. `mergesort4` combines four descending runs;
`mergesort_2seq` combines two descending runs. Allocate their complete input
and output footprints. A merge is not an arbitrary-input sorting operation.

Equal scores do not establish a stable identifier order. The example validates
descending scores, the identifier multiset, and the exact score bits associated
with each identifier. Only then does it canonicalize identifier order inside
equal-score groups for comparison. This rejects duplicate, missing and
out-of-range identifiers, scores attached to the wrong identifier, and an
unsorted output even if a loose score-only comparison would pass.

The reference includes distinct-score and tied-score cases. It reads no
recorded output, imports no simulator implementation and accepts only the
declared finite FP32 domain. Product regressions in
`tests/product/test_api_sort_records.py` exercise the negative controls.

```sh
python run.py reference
python run.py check --launcher sim
python run.py check --launcher pipesim
python run.py emit --backend cce
```

These commands run after copying the directory alone. The contract records
device/backend scope; a shared facade export is not a hardware support result.
