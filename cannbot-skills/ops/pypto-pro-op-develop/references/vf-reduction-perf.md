# Vector reduction authoring and performance

## Contents

- [Selection rule](#selection-rule)
- [Correct `vf.load_align` and mask placement](#load-mask-placement)
- [Reduction checklist](#reduction-checklist)
- [Do not branch inside a vector function](#no-branch)
- [Per-call overhead and loop form](#call-overhead)
- [Load-issue cost model — a hypothesis to re-probe, not a fact](#load-cost-model)
- [Reviewable examples](#reviewable-examples)
- [Scratch stores need a barrier before the next vector load](#scratch-barrier)
- [Primary source order](#primary-sources)


Use this reference for row reductions and reduce-then-broadcast dataflows in a
PyPTO-Pro vector section. Candidate selection is defined by `DESIGN.md`.

## <a id="selection-rule"></a>Selection rule

Apply the workflow policy in this order:

1. **Correctness and API support.** Confirm the operation, dtype, mask, layout,
   and tail behavior in the API documentation installed for the target
   PyPTO/CANN version. Settle the accumulator's dtype here, before comparing
   candidates: the reduce primitives are same-type and cannot widen, so a
   narrow-dtype chain whose intermediates exceed the format's range has to be
   widened by the caller. See
   [`pypto-pro-op-kb/constraints/precision.md`](../../pypto-pro-op-kb/constraints/precision.md).
   A faster candidate that returns `inf` is not a candidate.
2. **Follow `DESIGN.md §1`.** Implement its single frozen choice. Use tile
   operations only when DESIGN cites an explicit selected-KB-template requirement.

Do not generalize a result from softmax, RMSNorm, L2Norm, LayerNorm, one shape,
or one platform to all vector operators. The profiler result for the current
kernel is authoritative.

## <a id="load-mask-placement"></a>Correct `vf.load_align` and mask placement

`vf.load_align` loads a register view and does not take a predicate register.
Apply masks to computation and store operations whose installed API signature
accepts them.

```python
loaded = vf.load_align(input_tile, offset)
accumulator = vf.add(accumulator, loaded, predicate)
result = vf.reduce_sum(accumulator, predicate)
vf.store_align(output_tile + offset, result, predicate)
```

This form is invalid:

```python
loaded = vf.load_align(input_tile, offset, predicate)
```

Verify the exact argument order against the target version's API page before
copying a call; names and signatures may differ across SDK versions.

## <a id="reduction-checklist"></a>Reduction checklist

- Keep reduction accumulators in the precision required by the numerical
  contract; use FP32 unless a documented and validated narrower path is
  acceptable.
- Exclude padding lanes from every reduction and output store.
- Treat register width, offset units, predicate construction, and reduction
  result lanes as API-version-specific facts.
- Budget the backing Vec tiles against the detected platform. VF registers do
  not remove the need to fit input/output/workspace tiles in on-chip memory.
- Re-run correctness after every unroll, accumulator, mask, buffering, or tile
  size change.
- Use profiler data to distinguish vector-compute, scalar/control, and
  data-movement limits before selecting an optimization.

## <a id="no-branch"></a>Do not branch inside a vector function

This is a property of the vector unit, not of any DSL over it: the vector pipe
has no branch, so an `if` inside a vector function
lowers to predicated execution — both arms issue, every lane pays for both, and
the loop around them pays the mask bookkeeping. A vector function that branches
per iteration is far slower than one that does not.

Write the alternative out of the vector function instead:

- **Choose the route in the kernel body.** A kernel-level conditional runs on
  the scalar unit and is a real branch. Better still, select by *emptying a
  loop*: give the route that does not apply an empty item range —
  `end = begin + (end - begin) * on`, `on` in `{0, 1}` — so both bodies stay
  straight-line and neither needs a condition. The empty range is genuinely
  empty in pypto-pro: `pl.range(start, 0, step)` executing zero times is
  measured directly in
  [`pypto-pro-op-kb/patterns/vec-tensorlist-fixed-arity.md`](../../pypto-pro-op-kb/patterns/vec-tensorlist-fixed-arity.md).
- **A zero-or-one-iteration loop replaces an `if`.** A loop bounded by
  `min(n, 1)` runs its body only when `n > 0`, costs one loop set-up, and does
  not predicate the lanes.
- **A clamp replaces a conditional value.** A min/max composition is a
  branchless 0/1 select, and a masked multiply by a 0/1 register zeroes a
  contribution that should not count.
- **A `select` replaces a conditional store.** A compare into a mask register
  followed by `vf.select` is a data-path operation, not a branch — and masks
  measured free (see the cost model below).

## <a id="reviewable-examples"></a>Reviewable examples

- Tile-operation softmax:
  [`pypto-pro-op-kb/examples/samples/softmax/softmax_impl.py`](../../pypto-pro-op-kb/examples/samples/softmax/softmax_impl.py)
- Vector-function softmax:
  [`pypto-pro-op-kb/examples/samples/vf_vs_tileop/vf_softmax_impl.py`](../../pypto-pro-op-kb/examples/samples/vf_vs_tileop/vf_softmax_impl.py)
- Other validated vector-function compositions:
  [`pypto-pro-op-kb/examples/kernel-index.md`](../../pypto-pro-op-kb/examples/kernel-index.md)

These examples establish API usage only for their recorded environment. They
are not proof that the same implementation level is fastest for another
operator or target.

## <a id="scratch-barrier"></a>Scratch stores need a barrier before the next vector load

Any VF vector store into UB scratch that a later vector load reads — inside
one vector function or across calls that share the scratch — needs
`vf.mem_bar()` (default `VST_VLD`, the RAW direction) at the producer's
tail. `auto_mutex` only arbitrates *across* pipes; it orders nothing inside
the V pipe, so the omission reads clean on aligned shapes and goes stale on
the shapes that reorder the schedule. The official
`test_quant_lightning_indexer_vf.py` sample carries six such barriers with
"ensure X visible to Y's vlds" comments — copy the discipline, not just the
arithmetic. Cost is ~16 ns per barriered round trip (measured in the scan
KB): irrelevant to throughput, decisive for correctness. The WAR direction
needs no barrier (register dependences cover it; the same sample reuses
scratch without one).

## <a id="primary-sources"></a>Primary source order

1. `$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/` for the installed `pl` and `vf`
   signatures and constraints. In the current documentation layout, verify
   `SIMD-API/vf_computation/data_movement/load_align.md`,
   `SIMD-API/vf_computation/reduction/reduce_sum.md`, and
   `SIMD-API/vf_computation/data_movement/store_align.md`.
2. The official examples listed in
   [`../../pypto-pro-material-explore/references/official_samples.md`](../../pypto-pro-material-explore/references/official_samples.md).
3. A correctness run and profiler capture on the detected target.

If these sources disagree, record the version and the observed result, then use
the behavior of the actual target environment for that implementation.
