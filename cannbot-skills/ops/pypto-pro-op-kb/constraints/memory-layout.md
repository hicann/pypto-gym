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
tile | layout | start | size per slot | slots | lifetime | end
```

Reject the design if live ranges overlap, an address violates the documented
alignment, or layout conversion is inferred rather than documented.

### A strided range's size is not the product of its extents

The `size per slot` column above is the one people fill in wrong, and it is wrong
in a way that makes the overlap check pass when it should fail.

A tile that covers `span_i` elements along each axis of a wider parent, with
parent stride `stride_i`, occupies a **linear extent** of

```text
1 + Σ (span_i − 1) · stride_i
```

**not** `Π span_i`. For the ordinary 2-D case — `M` rows of `N` elements out of a
parent whose row stride is `S` — that is `(M−1)·S + N`, which exceeds `M·N`
whenever `S > N`, i.e. whenever the tile is narrower than the tensor it came
from. The same shape of arithmetic governs a burst transfer, where `n_burst`
bursts of `burst_len` at inter-burst `step` span `(n_burst − 1)·step + burst_len`.

**The two numbers answer different questions and you need both.**

| Question | Use |
|---|---|
| does this view fit inside its parent storage? | the **linear extent** — `Π spans` under-states it and will declare a view legal that runs off the end |
| do two live allocations collide? | the **linear extent**, as a conservative convention — see the note below |
| how many bytes actually move? | `Π spans` — the skipped tail between bursts is genuinely not touched, and it is legal for it to extend past storage |

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

- **A bounds check written against `Π spans` is not conservative, it is wrong in
  the unsafe direction.** It under-reports, so it green-lights the case it exists
  to catch.
- **This is what makes an exhaustive layout script toothless.**
  [investigation-discipline §12](../references/investigation-discipline.md)
  requires a per-instance *pairwise address-range non-overlap* assertion, and
  records that the defect which survived three review rounds was found by that
  check alone. A script that computes each range as a dense `Π spans` block
  computes the wrong ranges and then reports no overlap — a green exhaustive
  sweep that proves nothing, which §12 calls out as more dangerous than no sweep.
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
