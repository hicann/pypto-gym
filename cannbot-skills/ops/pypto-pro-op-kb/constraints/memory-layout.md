# Memory layout and lifetime

## Rule

Treat Vec, Mat, Left, Right, and Acc as separate address spaces. Within each
space, allocate non-overlapping byte ranges for tiles whose lifetimes overlap.

## Lifetime classes

- **Rotating:** multiple slots selected with `.next()` across iterations.
- **Resident:** one slot kept live across an inner loop and accessed through the
  same handle.
- **Scratch:** a temporary whose lifetime ends before an overlapping address is
  reused.

`mutex_ids` identify synchronization slots; they do not prove that two byte
ranges are safe to alias.

## Address review

For each memory space, maintain a table with:

```text
tile | layout | start_byte | size_bytes_per_slot | slots | lifetime | end_byte_exclusive
```

Record byte ranges as half-open intervals `[start_byte, end_byte_exclusive)`.
For one slot, `end_byte_exclusive = start_byte + size_bytes_per_slot`. If a row
aggregates contiguous slots, use `start_byte + slots * size_bytes_per_slot`;
otherwise list each slot's start and exclusive end separately.

Reject the design if live ranges overlap, any exclusive end exceeds that
space's capacity, an address violates alignment, or a layout conversion is
inferred rather than documented.

### A strided range's size cannot generally be inferred from its extents alone

A non-empty tile with non-negative parent strides, whose base points at its
lowest addressed element, and which covers `span_i` elements along each axis of
a wider parent, occupies a **linear element extent** of the following size when
`stride_i` is measured in elements:

```text
1 + Σ (span_i − 1) · stride_i
```

An empty tile has zero extent. For whole-byte storage, multiply the element
extent by `element_size_bytes`. In 2-D with unit inner stride, `M` rows of `N`
elements with parent row stride `S` occupy
`((M−1)·S + N)·element_size_bytes`. `Π span_i` cannot generally replace this
linear extent: it may under-state stride gaps or over-state overlapping views.

For a sub-byte dtype, follow the target packing rule and round the first-to-last
touched bit range outwards to aligned whole bytes; do not use a fractional size.

For bursts, `(n_burst−1)·step + burst_len` uses start-to-start `step`; if the API
gives a post-payload gap, use `step = burst_len + gap`, then convert API units to
bytes.

**The two numbers answer different questions and you need both.**

| Question | Use |
|---|---|
| does this view fit inside its parent storage? | prove `[view_start_byte, view_start_byte + extent_bytes)` is contained in the parent's half-open range; use the formula above for whole-byte storage and target packing for sub-byte storage |
| do two live allocations collide? | the **linear byte extent**, as a conservative convention — see the note below |
| how many logical payload bytes move? | `Π spans · element_size_bytes` for whole-byte storage, or the target-packed equivalent; count transfer padding separately |

**The overlap row is a convention, not arithmetic.** Two allocations interleaved
in each other's inter-burst gaps genuinely do not collide byte for byte — the
skipped tail really is untouched. Asserting on the linear extent will therefore
reject some layouts that would in fact work. Take that trade: an interleaved
layout depends on the exact stride pattern of every transfer that touches it and
breaks the moment one of them is retiled, and nothing in the DSL validates the
assumption. Where a design deliberately interleaves, say so at the assertion and
prove the byte sets are disjoint — do not silently weaken the check to `Π spans`,
which fails to catch real collisions as well.

Two consequences worth stating separately:

- **For a non-overlapping view with stride gaps, a bounds check written against
  `Π spans` is wrong in the unsafe direction.** It under-reports, so it
  green-lights the case it exists to catch.
- **This is what makes an exhaustive layout script toothless.**
  [investigation-discipline §12](../references/investigation-discipline.md)
  requires a per-instance *pairwise address-range non-overlap* assertion, and
  records that the defect which survived three review rounds was found by that
  check alone. For a layout with stride gaps, a script that computes each range
  as a dense `Π spans` block computes the wrong ranges and then reports no
  overlap — a green exhaustive sweep that proves nothing, which §12 calls out as
  more dangerous than no sweep.
  Assert the linear extent, and keep the negative control that shows the
  assertion can fire.

## Evidence

- rotating Vec groups:
  [softmax_impl.py](../examples/samples/softmax/softmax_impl.py)
- resident A operand across output columns:
  [bf16_matmul_operand_reuse_impl.py](../examples/samples/bf16_matmul_operand_reuse/bf16_matmul_operand_reuse_impl.py)
- vector→cube layout handoff:
  [vec_cube_abs_sqrt_matmul_impl.py](../examples/samples/vec_cube_abs_sqrt_matmul/vec_cube_abs_sqrt_matmul_impl.py)

Use the installed API documentation and matching official example to confirm
alignment and layout semantics for the target version.

---
