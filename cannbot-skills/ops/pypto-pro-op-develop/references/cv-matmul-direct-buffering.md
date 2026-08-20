# CV matmul direct handoff and rotating buffers

Use this reference for A5 Cube matmul kernels whose accumulator is consumed by
a Vector epilogue in the same launch. It states the double-buffer shape in
PyPTO-Pro terms -- `make_tile_group + auto_mutex`; it is not a drop-in
operator.

## Buffer map

For a two-level K loop, start from the target SDK's official dynamic-ASW roles:

| role | depth | cursor operation |
|---|---:|---|
| A/B Mat (L1) | 4 | `.next()` once per non-empty K1 iteration |
| A/B Left/Right (L0) | 2 | `.next()` once per non-empty K0 iteration |
| int32 Acc (L0C) | 1 | `.current()` once per output tile |
| direct int32 Vec tile | 1 | `.current()` once per output tile |

A scalar `addrs=` value is the first TileGroup slot base. PyPTO-Pro lays later
slots contiguously by physical tile bytes. Prove the high-water mark with the
largest compile-time tile family; do not count only the logical tail window.
L0A, L0B and L0C are different memories, while both Mat operands share L1.

Assign every simultaneously live TileGroup a unique mutex ID. In a fused
cross-core kernel, reserve the logical READY/FREE IDs and their AIV mirrors
before assigning buffer IDs; on the validated A5 mapping, logical 0/1 occupy
physical 0/1/16/17.

## Direct Acc-to-Vector handoff

If the final accumulator uses `pl.move(..., acc_to_vec_mode=...)` instead of an
Acc store:

- leave `phase` unspecified on `pl.matmul` and `pl.matmul_acc`; a TMOV has no
  `STPhase.Final` peer to reset the Acc unit flag;
- pin a compact Acc to its full physical window before the first matmul and do
  not narrow it before TMOV; use `[tile_m, tile_n]` so a gated wider family is
  not silently decoded with the baseline TN;
- use the matching hardware split only when each AIV share fits UB and final GM
  stores cannot share an unaligned 32-byte beat; otherwise select a proven
  single-AIV family;
- retain a matching-runtime-proven M-to-FIX dependency. Older deployments may
  require the explicit `sync_src/sync_dst(M,FIX)` compatibility pair even when
  a newer auto-mutex test shows the Acc release/FIX acquire edge.

For a one-slot Cube-to-Vector credit protocol, each AIV primes MTE3 FREE once,
waits READY once per task, and returns MTE3 FREE after the conditional output
store.  Empty shares still return their credit.  Cube waits both physical FREE
credits before TMOV, sets both READY credits after TMOV, and drains one final
FREE pair after the loop.  For `T` tasks per core the balance is `T+1` FREE
events and `T` READY events in each direction.  Inspect generated CCE: the
validated specializations contain two MTE3 FREE sites, no V FREE site and place
the per-task FREE after TSTORE.

## Wide-tile resource check

Evaluate `(TM,TK1,TN)` candidates symbolically before codegen:

```text
L1 bytes  = depth_l1 * (TM*TK1 + TK1*TN) * operand_bytes
L0A bytes = depth_l0 * TM*TK0 * operand_bytes
L0B bytes = depth_l0 * TK0*TN * operand_bytes
L0C bytes = TM*TN*accumulator_bytes
UB direct = hardware_share_rows*hardware_share_cols*accumulator_bytes
```

Reducing L1 depth from four to two is valid only as a separate measured
candidate. A larger TK/TN can consume the saved capacity; do not assume either
choice is faster.

### Gating a wider tile family behind a tiling key

Prefer a new tiling-key bit over changing the baseline family globally. The
gate is a compile-time predicate over shape and mode, of the form:

```text
wide_n = M >= <m_min> and K >= <k_min> and N % <wide_tile> == 0
         and not <per-token mode> and <baseline occupancy> >= <cores>
tile_n = <wide_tile> if wide_n else <base_tile>
```

Every threshold in it is per-operator and per-target: derive each from the
resource check above and the measured occupancy, and re-derive them when either
changes. What generalises is the *structure* — admit the wide tile only where
it is both resource-safe and occupancy-positive, and only for the modes whose
epilogue can absorb it.

Propagate `tile_n` into every physical allocation and compact-window assertion,
and record the worst-case byte high-water marks for L1, L0A, L0B, L0C and UB of
the map you validated; they move with the tile and are not transferable between
maps.

**Do not reuse a wide key for small or decode shapes.** A candidate can pass
codegen and full bit-exact correctness and still regress the small-shape probes
by more than 1.5x, because a wide tile on a short reduction buys nothing and
costs occupancy. Correctness is not evidence that the key is safe to widen.

### Beat-safe N-tail dual-AIV gate

With a 2-byte output and a full 256-column share, adjacent row segments have:

```text
gap_bytes = (N - 256) * 2
```

Enumerate all 32 possible byte offsets when proving beat ownership. **A gap of
at least 32 B is sufficient for disjoint 32 B beats**; below that, two AIVs can
share a beat and the split is unsound — construct the overlapping-beat
counterexample rather than assuming the margin.

That inequality, not a width list, is the gate: the smallest admissible
unaligned `N` is the smallest one whose `gap_bytes` reaches 32, and every
narrower tail keeps the single-AIV path. Aligned `N` already uses its normal
family, so the dual-AIV split only ever applies to unaligned widths at or above
that threshold. Recompute the threshold when the output dtype changes — it is
`2 B` per element in the expression above.

## Mandatory promotion gates

1. Compile representative dtype, tail-mode and tile-family keys. Inspect C++
   for the expected slot addresses, cursor modulo, TLOAD/TEXTRACT/TMATMUL chain,
   one Acc TMOV, no GM workspace and no Acc phase.
2. On the authorized device under the required lock, compare bit-exactly across
   K around both K0 and K1 boundaries, M/N tails, multiple output tiles, batch,
   optional epilogue inputs and repeated same-core reuse.
3. Measure the exact kernel with msprof/`kernel_details.csv`. Wrapper Event
   intervals are only a directional probe.
4. Promote only tile families that improve the measured shapes; add a host
   metadata gate for resource-unsafe or occupancy-poor geometries.

**What the correctness evidence has to cover.** A 4xL1/2xL0 rotation is
established bit-exact only against K values that straddle every boundary the
rotation has (1, one below / at / one above each K tile, and at least one value
past two full cycles), plus M/N tails, multi-tile reuse and batch. Anything less
leaves a boundary the rotation never crossed.

**Performance stays a per-operator profiler decision.** A double-buffered
rotation is worth roughly 1.2-1.7x on kernel-only medians for a mid-size
geometry, but the spread across shapes is wider than the mean, and a wider K/N
tile is rejected whenever its *ranking* is unstable across shapes rather than
when its mean is worse -- an unstable ranking means the next shape decides the
winner.

**Expect the end-to-end gain to be smaller than the kernel gain.** Startup and
pipeline costs dominate the shapes that remain slow, so a kernel ratio well
above 1 can leave the end-to-end figure nearly flat. That gap is a
pipeline/startup problem; it is not evidence for promoting an unvalidated wide
tile. Removing compatibility syncs, changing the initial credit protocol,
changing K1 depth, or specialising the VF epilogue are separate candidates, each
requiring the full locked-device promotion gates.

**Do not compare aggregate figures whose baseline you do not control.** A
baseline that moves between measurements makes the comparison meaningless. The
evidence that a gated revision is real is *agreement between two independent
measurements of it* -- a direct device-kernel comparison against the preceding
source, and a held-lock A/B on the same device, landing on the same ratio within
noise. Agreement is the evidence; either figure alone is not.

Use the layers in this order: same-device held-lock A/B first, then device-kernel
times measured in the same run as their own baseline. Discard measurements from
a device you did not hold exclusively.
