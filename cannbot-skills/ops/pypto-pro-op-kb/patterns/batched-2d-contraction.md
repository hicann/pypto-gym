# Batched contraction reduced to 2-D

## Applies when

The left operand has leading batch dimensions, the right operand is shared by
every batch, and the batch dimensions can be collapsed without changing the
contraction:

```text
out[..., M, N] = left[..., M, K] @ right[K, N]
```

## Decision rule

Collapse the leading batch axes so the contraction runs as a 2-D
`[batch_product * M, K] x [K, N]`, and restore `[..., M, N]` on the way out.

**Collapse it inside the kernel, not on the host.** The batch product is
arithmetic on the shape, so the kernel can compute the flat offset from the real
shape and index the operand directly. Doing it on the host instead means a
`reshape` — and, whenever the input is not already contiguous, a `contiguous()`
that materializes the whole tensor as a measured device kernel. See
[wrapper-boundary.md](../constraints/wrapper-boundary.md), which is in force for
every class and takes precedence over anything on this page.

Then:

- preserve the original batch-axis order when restoring the output shape;
- do not apply the transform when the right operand also varies by batch;
- validate dynamic shapes and launch geometry on the target SDK.

A reshape that is a pure view of an already-contiguous tensor costs nothing and
does not appear in the profile. If you cannot show from the profile that a host
reshape produced no device operation, it produced one.

The kernel body then follows the ordinary 2-D contraction pattern in
[cube-only.md](cube-only.md).

## Validation status

**Conceptual.** No retained kernel in this KB implements the batched collapse
end to end. The retained
[matmul implementation](../examples/samples/matmul_float_mmad/matmul_float_mmad_impl.py)
is a 2-D contraction reference only; it prepares its layout on the host, which is
what this page now tells you not to do, so read it for the kernel body and not for
the boundary. Use the target SDK's official reshape and dynamic-shape examples
before implementing the collapse.
