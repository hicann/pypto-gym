# Device facts, and the rules that follow from them

Numbers alone do not change a kernel. This page carries a device fact only when a decision
turns on it, and states that decision next to it. Capacities and core counts are read from
[the shipped profiles](../../../library/ascriptor/devices/profiles); throughput and clocks from
[performance parameters](../../../library/docs/performance-parameters.md). Neither is copied
here — a second copy of a number is a second thing to get wrong.

## What the profiles say

| Resource | a5 (`950`) | a5pr (`950pr`) | a2 (`b3`) |
|---|---:|---:|---:|
| Cube cores | 32 | 28 | 20 |
| Vector cores | 64 | 56 | 40 |
| **Vector sub-blocks per cube core** | **2** | **2** | **2** |
| UB | 256 KB | 256 KB | 192 KB |
| L0C | 256 KB | 256 KB | 128 KB |
| L0A / L0B | 64 KB | 64 KB | 64 KB |
| L1 | 512 KB | 512 KB | 512 KB |
| BT (bias table) | 4 KB | 4 KB | 512 B |

Read the ratio, not the counts. Two vector sub-blocks per cube core holds on all seven profiles
the library ships; the counts do not, and `ascriptor.a5` is the 32/64 part while a 950PR card is
28/56. A rule written as "56 AIV" is a rule that is wrong on the device its own facade names.

## The drain into the vector side has no safe default

The [A3 profile](../../../library/ascriptor/devices/profiles/a3.json) is also
C220, with 20 cube and 40 vector participants per die and the A2 capacities
shown above. A2/A3 use GM publication for cube-to-vector handoff. The direct
L0C-to-UB choices below describe A5; they are not an A2/A3 transfer path.

`l0c_to_ub(..., dual_mode=)` decides how the L0C tile's M rows reach the pair of sub-blocks, and
both answers compile, run, and give the right numbers.

- **`SPLITM`** — the first M/2 rows go to sub-block 0 and the second M/2 to sub-block 1, **each
  into its own UB**. `GetSubBlockIdx()` becomes part of an address, not a guard around the work.
  This is the IR's default and what a per-row consumer wants.
- **`SPLITN`** — the N extent splits instead. This is not an exotic case: a transposed product
  puts the rows a sub-block owns on N. In the attention kernels the score is computed as
  `score^T = K @ Q^T`, so its query rows ARE the N extent and the drain is `SPLITN` with `N_dst`
  at half the query block, while the PV result on the next page is `[M=queries, N=D]` and drains
  `SPLITM`. Both hand each sub-block 64 query rows.
- **`SINGLE`** — the whole M block goes to one sub-block, with `sub_block_id` naming which. The
  other one has no work. Correct only when a single sub-block must see every M row, which means
  a reduction ACROSS M, not along it.

So the rule is not "use SPLITM"; it is **split whichever axis carries the rows a sub-block owns**,
and which axis that is follows from how the matmul was arranged.
`attention/a5_pfa_qk_metadata` has both drains within fifty lines of each other and records that
the `SINGLE` alternative measured +82 µs on its shape; `attention/a5_v8_cube_stage` drains one
tile twice, `SPLITN` and `SINGLE`, and publishes both so a reference can compare them.

Choosing SINGLE where SPLITM would do costs twice, and the second cost is the one that hides:
the vector work is no longer shared, **and** the landing tile is sized for the whole M instead
of half of it. A landing tile twice as large is usually what forces the M tile back down, so the
two multiply. Measured on a sparse-attention kernel written entirely in SINGLE: about 4x, the
larger part of a 3.13x gap against a reference that made the same routing choices.

A performance lint names every such move; `docs/api/cube.md` has the operation.

## SINGLE is sometimes forced, and then it is not a choice

Split mode carries the **same-type plain copy** alone — fp32→fp32 or int32→int32. Not a fused
relu, not any non-default requant, and **not even an unscaled float downcast** (fp32→fp16/bf16),
because the fixpipe's scalar path rides a deqScalar that exists only with the dual destination
control off. So a drain that converts on the way, which is most attention kernels, has to be
SINGLE, and no lint asks about those. The reverse — split mode with any of those riders — is
refused before any backend, because the hardware does something else and two of the three
backends would have printed it.

## Cube work is quantised, and the quantum is large

The A5 model charges 4096 FP16-operand MACs per cycle with **67 cycles of MMAD setup**
(A2: 2048 and 21). M and N pad to 16 and K to its operand-width quantum before any MAC work is
charged. Two consequences worth carrying into a tiling decision: mathematical FLOPs understate a
tiled implementation, because padded work is real work; and a schedule of many small matmuls
pays that setup every time, so prefer fewer and larger ones at equal MAC count.

## A `@vf` entry is charged before the body runs

The A5 cycle model charges `vf_fixed_overhead_cycles` 46 plus
`instruction_head_overhead_cycles` 10 for every `@vf` call, before a single instruction of the
body issues, and the body then issues at roughly a cycle each
(`vf_pipe_issue_interval`: LD 1.14, ST 1.0, SU 0.48). So a body of about fifty instructions
spends half its time being entered. The numbers live in
[`a5_cycle_model.json`](../../../library/ascriptor/backends/sim/timing/a5_cycle_model.json);
`performance-parameters.md` points at that model rather than restating it, and so does this page.

The rule: **merge adjacent `@vf` bodies instead of splitting them**, and treat a `@vf` inside a
tile loop as paying that entry per tile. The same arithmetic decides recompute-versus-store
questions — rebuilding a quantity is cheap once you are already inside a `@vf`, and expensive if
it needs one of its own.

A handoff that has to become visible costs more again: `intra_core_sync_latency_cycles` is 200 on
a5 (1000 on a2), so a barrier at the tail of a short body outweighs the body. Measured on this
repository's sparse-attention kernels, rebuilding a predicate in a per-tile `@vf` of about seventy
instructions left the vector side at 722 us against the cube's 161 us.

## A layout-converting DMA is not one instruction

`ub_to_l1.nd2nz` composes one MTE3 burst per NZ fractal column of the ND rows. The cycle model
gives it 32 bytes per cycle where the plain `ub_to_l1` gets 256, and the board agreed by about
10x (D-084). Store the tile as compact NZ from the `@vf` instead — a strided `vf.store` writes
the fractal directly — and move it with the plain `ub_to_l1`.

The rework carries its own trap, so take both halves: the strided store's natural block stride is
the staging tile's row count, which is 16-aligned by nature and therefore the worst rung of the
bank ladder. Pad the tile by one row so that stride is odd. On the two kernels reworked in D-226
the padding was worth more than the move it rides on. A lint states both at the call site, with
the numbers; this page is here so the rule is readable before a kernel is written.

## Do not hardcode a core count

`block_dim` comes from the profile, not from a literal. The same kernel source runs on 32c/64v
and 28c/56v parts, and a launch that assumes one of them either idles cores or asks for cores
that are not there. The counts above are for reading a cost model, not for pasting into a
kernel.
