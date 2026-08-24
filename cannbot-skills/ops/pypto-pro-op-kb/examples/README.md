# Validated study kernels

[kernel-index.md](kernel-index.md) is the only production selector. Every row
has:

- a general topology or technique;
- a canonical implementation path;
- `validated` state;
- reviewable evidence in the implementation itself or a companion golden.

Do not add failed, work-in-progress, diagnostic, or benchmark-only probes to
the selector. Re-run the retained validation on the actual target before
adopting a sample in production; a historical validation does not establish
compatibility with a different PyPTO/CANN version or platform.

## What to copy from a sample, and what not to

**Copy the kernel. Do not copy the driver.**

A sample's value is the `@pl.jit` kernel body and the dataflow around it. Some samples
also define a `*_wrapper` (or a `__main__` block) whose job is to make the file runnable
by itself: it allocates an output, reshapes an operand into the layout the kernel expects,
and calls `torch.npu.synchronize()` so a smoke test measures something. Those are
**harness** calls.

A delivered wrapper is held to a much narrower rule — `torch.empty` for output allocation
and nothing else, with layout work inside the kernel. See
[constraints/wrapper-boundary.md](../constraints/wrapper-boundary.md). A sample driver is
not an example of that rule and must not be read as one; copying its shape into
`custom/<op>/test_{op}.py` produces a boundary violation that the sample never claimed to
license.
