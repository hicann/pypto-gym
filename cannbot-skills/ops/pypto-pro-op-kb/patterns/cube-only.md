# Pure cube contraction

## Applies when

The kernel is a 2-D contraction with no vector reduction, activation, cast, or
elementwise epilogue.

## Dataflow

Stage operands from GM to L1, move them to L0A/L0B, accumulate in L0C, and
store the result:

```text
GM → Mat → Left/Right → Acc → GM
```

Confirm operand layouts and tile limits in the target SDK documentation before
adopting the sample.

## Validated single-block skeleton

The complete runnable reference is
[matmul_float_mmad_impl.py](../examples/samples/matmul_float_mmad/matmul_float_mmad_impl.py),
with [matmul_float_mmad_golden.py](../examples/samples/matmul_float_mmad/matmul_float_mmad_golden.py)
as its retained golden.

```python
with pl.section_cube():
    a_mat = a_l1.current()
    b_mat = b_l1.current()
    a_left = a_l0a.current()
    b_right = b_l0b.current()
    out_acc = acc.current()
    pl.load(a_mat, a, [0, 0])
    pl.load(b_mat, b, [0, 0])
    pl.move(a_left, a_mat)
    pl.move(b_right, b_mat)
    pl.matmul(out_acc, a_left, b_right)
    pl.store(out, out_acc, [0, 0])
```

## K-loop accumulation semantics

Conceptual guidance only: no retained KB sample validates the general K-loop
phase sequence. When the target SDK uses `AccPhase`, only the last K block may
use `Final`; every earlier block uses `Partial`.

```text
for block in K_blocks:
    is_first = block == 0
    is_last  = block == K_blocks - 1
    phase    = Final if is_last else Partial

    if is_first:
        matmul(acc, left, right, phase=phase)
    else:
        matmul_acc(acc, acc, left, right, phase=phase)
```

This covers the one-block case (`matmul(..., Final)`), the first block of a
multi-block case (`Partial`), all middle blocks (`Partial`), and only the last
block (`Final`). Validate the exact API signature and buffer rotation against
an official example for the detected SDK version before implementing it.

## Failure checks

- A vector section means the kernel is not cube-only.
- An L0/L1 capacity failure requires smaller tiles or a validated K-loop
  design.
- A wrong total across three or more K blocks usually means an intermediate
  block was finalized too early.

## Validation status

- Single K block: validated by the retained runnable reference.
- General K-loop: conceptual only; not a copyable validated skeleton.
