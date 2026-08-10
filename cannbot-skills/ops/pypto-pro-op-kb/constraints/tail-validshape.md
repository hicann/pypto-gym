# Dynamic tails and valid shapes

## Rule

Keep a compile-time legal physical tile and set a runtime valid window for each
partial work item. Apply the valid window to every input, output, reduction
result, and workspace whose operation observes the tail.

## Checklist

1. Compute work-item counts with ceiling division.
2. Compute each remaining extent from the actual tensor dimension.
3. Clamp the extent to the physical tile size.
4. Set the valid shape before load, compute, reduction, and store as required by
   the installed API.
5. Neutralize invalid lanes for reductions when valid shape alone does not
   define the reduction identity.
6. Test dimensions smaller than one tile, exactly one tile, multiple exact
   tiles, and multiple tiles with a remainder.

For streaming softmax, the final score chunk must exclude invalid lanes from
both maximum and sum. See
[online-softmax-tail.md](../patterns/online-softmax-tail.md), which is
conceptual only.

## Exception: `compact=1` tiles — the window is part of the layout

Checklist item 4 is wrong for any tile declared `compact=1`, and following it
there corrupts data silently. Under `compact=1` the layout is interpreted
against the *current* valid window (`CompactMode.md`). Narrowing the window
between the write and the read makes writer and reader decode two different
fractal layouts of the same bytes.

Two operators in one session hit this, each on a `compact=1` Acc:

- An accumulator was set to `[tile_m, TN] = [128, 256]` before `matmul`
  (`ceil(128/16) = 8` row blocks) and re-set to `[valid_m, valid_n]` before
  `pl.store` (`ceil(1/16) = 1` row block). Only column block `j = 0` has a
  coincident address, so exactly **16 of 256 cells per row survive and 240 are
  wrong** — a signature clean enough to identify the mechanism from the
  mismatch count alone.
- A second operator issued `set_validshape(acc, …)` after `matmul` and before
  extract, giving 4 written row blocks against 3 read. It reproduced only on
  odd-M tails, which is why it read as a shape special case rather than a
  layout rule.

The discriminant is whether `compact` is present at all, and it separates the
in-repo examples cleanly. `CompactMode.md`'s own example and
`lightning_indexer:546,611-621` are `compact=1` and hold the M window fixed
throughout. The official ASW dynamic sample
(`test_matmul_perf_asw_4k_dn_move_offset_dynamic.py:111-114,203-204`) *does*
narrow before store — and its Acc carries no `compact`, so the layout comes
from the static shape and narrowing is pure clipping. Both styles are legal
on their own; mixing them is the defect.

Contract for `compact=1` accumulators:

1. Set the window **once per output tile**, before the first `matmul`.
2. Pin it to the full physical tile `[tile_m, TN]`, not the valid extent.
3. Issue no `set_validshape` on that tile between `matmul` and `store`.
4. Keep the K axis true at all times.

The cost is a tail tile writing `(tile_m − valid_m) × TN × 4B` of padding, which
is zero for any case whose M and N divide the tile. Note that rounding the
window *up* to the fractal boundary is a no-op rather than a fix —
`ceil(ceil(vm/16)·16/16) == ceil(vm/16)` — so a prescription of that shape will
reproduce the failure byte for byte.

## Evidence

- dynamic row/column handling:
  [softmax_impl.py](../examples/samples/softmax/softmax_impl.py)
- independent mathematical reference:
  [softmax_golden.py](../examples/samples/softmax/softmax_golden.py)

Confirm the exact `set_validshape`, load/store, padding, and reduction behavior
in the installed API documentation and official tail examples.
