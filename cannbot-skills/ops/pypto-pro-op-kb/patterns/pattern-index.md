# Pattern selector

The validation column controls how a pattern may be used:

- `validated skeleton`: the page cites a retained runnable implementation, and that
  citation **resolves** -- on disk, or in a branch the page names. This repo splits
  the knowledge base from the operator trees that validate it, so an artifact may sit
  on another branch; `check_kb_integrity.py` resolves it with `git cat-file -e` rather
  than trusting the prose. **Fetch that branch before using the page as a code
  starting point** -- the status promises the code exists, not that it is checked out.
- `conceptual only`: use the equations and decision rules, but do not copy code
  until the target SDK's official docs/examples provide a matching runnable
  reference.

| Pattern | Use when | Validation |
|---|---|---|
| [cube-only.md](cube-only.md) | the kernel is a pure 2-D contraction | validated skeleton for a single K block; conceptual only for K-loop accumulation |
| [vec-row-reduce-broadcast.md](vec-row-reduce-broadcast.md) | each row reduces to a scalar and broadcasts it back | validated skeleton |
| [buffer-reuse-lifetime.md](buffer-reuse-lifetime.md) | tile groups rotate or persist across iterations | validated skeleton for the indexed rotating groups; conceptual for other loop-carried state |
| [batched-2d-contraction.md](batched-2d-contraction.md) | host reshaping can reduce a batched contraction to 2-D | conceptual only |
| [quant-matmul-scaled-mm.md](quant-matmul-scaled-mm.md) | a quantized matmul uses explicit scale/round/clamp semantics | validated skeleton for the indexed int8 path |
| [vec-per-token-dynamic-quant.md](vec-per-token-dynamic-quant.md) | each row is quantized by a scale derived from that row, and the scale is an output | conceptual only |
| [online-softmax-tail.md](online-softmax-tail.md) | a softmax reduction streams across multiple score chunks | conceptual only |
| [vec-scan-prefix-dependent.md](vec-scan-prefix-dependent.md) | element i of the output depends on elements 0..i along one axis | validated skeleton for both dataflows, measured on Ascend950PR_9579 |
| [vec-paired-lane-rotation.md](vec-paired-lane-rotation.md) | two lanes form a pair rotated by a coefficient looked up by position | validated skeleton for the interleaved fp32 row loop; the table lookup, split-half pairing and runtime-scalar layout are conceptual |
| [vec-ub-strip-gather.md](vec-ub-strip-gather.md) | an index tensor selects, per element, along one axis — `torch.gather`, `GatherElements`, `take_along_dim`, the read half of `scatter` | validated skeleton, 12/12 dtype pairs bit-exact on Ascend950PR_9579 — **the sweep covered `C <= L` only**; the hoisted `c` term is wrong for `C > L` (see the load-bearing premise on that page) |
| [vec-tensorlist-fixed-arity.md](vec-tensorlist-fixed-arity.md) | an input is declared `is_list: true`, so the number of tensors is a runtime value the DSL cannot express — the `torch._foreach_*` family | validated skeleton for the padded fixed-arity ladder and the phase rotation, measured to 385 declared parameters on Ascend950PR_9579 |
| [vec-scatter-owner-model.md](vec-scatter-owner-model.md) | an index tensor decides where an output element is written | validated skeleton for the inner-tile layout; conceptual for the owner-batch and rank-1 layouts |
