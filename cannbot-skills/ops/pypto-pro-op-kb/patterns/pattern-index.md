# Pattern selector

The validation column controls how a pattern may be used:

- `validated skeleton`: the page cites a retained runnable implementation and a
  passing result that covers the claimed scope. The implementation citation
  **resolves** -- on disk, or in a branch the page names. `check_kb_integrity.py`
  checks local paths directly and uses `git cat-file -e` for fetched named
  branches. If a citation is absent locally, names one or more branches and none
  is fetched, the check reports `SKIP`, which proves nothing. Reviewers still match
  the result to the claimed scope. **Fetch the named branches and re-run the check
  before using their code.**
- `conceptual only`: use the equations and decision rules, and inspect any study
  code as unverified input; do not treat it as a validated starting point.

| Pattern | Use when | Validation |
|---|---|---|
| [cube-only.md](cube-only.md) | the kernel is a pure 2-D contraction | validated skeleton for a single K block; conceptual only for K-loop accumulation |
| [vec-row-reduce-broadcast.md](vec-row-reduce-broadcast.md) | each row reduces to a scalar and broadcasts it back | validated skeleton |
| [buffer-reuse-lifetime.md](buffer-reuse-lifetime.md) | tile groups rotate or persist across iterations | conceptual only; the retained sample shows group construction, but its recorded shape does not exercise slot rotation or wrap |
| [batched-2d-contraction.md](batched-2d-contraction.md) | host reshaping can reduce a batched contraction to 2-D | conceptual only |
| [quant-matmul-scaled-mm.md](quant-matmul-scaled-mm.md) | a quantized matmul uses explicit scale/round/clamp semantics | conceptual only; a runnable study exists, but no passing result was retained |
| [cv-quant-matmul-direct-epilogue.md](cv-quant-matmul-direct-epilogue.md) | an integer Cube contraction feeds a floating-point Vector epilogue directly | conceptual only |
| [vec-per-token-dynamic-quant.md](vec-per-token-dynamic-quant.md) | each row is quantized by a scale derived from that row, and the scale is an output | conceptual only |
| [online-softmax-tail.md](online-softmax-tail.md) | a softmax reduction streams across multiple score chunks | conceptual only |
| [vec-scan-prefix-dependent.md](vec-scan-prefix-dependent.md) | element i of the output depends on elements 0..i along one axis | validated skeleton only for the retained FP32 [8192, 128] cumsum by contraction; the two vector-unit dataflows are conceptual |
| [vec-paired-lane-rotation.md](vec-paired-lane-rotation.md) | two lanes form a pair rotated by a coefficient looked up by position | validated skeleton for the interleaved fp32 row loop; the table lookup, split-half pairing and runtime-scalar layout are conceptual |
| [vec-ub-strip-gather.md](vec-ub-strip-gather.md) | an index tensor selects, per element, along one axis — `torch.gather`, `GatherElements`, `take_along_dim`, the read half of `scatter` | conceptual addressing skeleton; the historical record establishes only 12/12 dtype compatibility runs on Ascend950PR_9579, not the exact `C`/`L` geometry or the general-modulo path |
| [vec-tensorlist-fixed-arity.md](vec-tensorlist-fixed-arity.md) | a contiguous, rank-insensitive elementwise/foreach input is declared `is_list: true`, so the number of tensors is a runtime value the DSL cannot express | conceptual only for the deliverable independent-`Ptr` ABI; the historical `Tensor`/host-view harness validates parameter capacity through arity 64 and empty-range behavior, not the corrected kernel-view ABI |
| [vec-scatter-owner-model.md](vec-scatter-owner-model.md) | an index tensor decides where an output element is written | conceptual only; the fp32 inner-tile result is historical because no runnable implementation was retained |
