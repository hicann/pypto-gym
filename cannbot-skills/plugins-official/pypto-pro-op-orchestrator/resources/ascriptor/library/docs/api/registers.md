# A5 registers, expressions and predicates

Write register operations inside `vf` on A5/A5PR. `Reg(dtype, name="", reg_num=1)`
holds one 256-byte register. The accepted `reg_num=2` forms are INT64, UINT64,
COMPLEX32 and COMPLEX64; `MaskReg` must use matching geometry. The
[grouped-register example](../../examples/api/register_groups) checks
both halves, including values too large for exact FP64 integer representation.
It serializes observable outputs as byte carriers so complex imaginary parts
and integer high bits remain part of comparison.

`RegList(dtype, length)` creates a static positive number of ordinary registers.
It is a different abstraction from one grouped register. Indices and lengths
must be valid at compile time; source and destination list lengths must agree
for list expressions. The full list method surface is declared in
[the manifest](manifest.json). A declaration for a convenience method does not
claim every dtype or backend implementation.

## Load, compute and store

`reg <<= ub[offset]` loads, `destination <<= expression` materializes a register
expression, and `ub[offset] <<= reg` stores. Tensor offsets are element offsets;
the selected load/store instruction still determines the physical footprint,
tabulated per distribution below.
[AXPB](../../examples/api/axpb) stages exactly one FP32 register of two inputs,
performs explicit multiply/add operations and compares every output lane.

Arithmetic operators and methods such as `.abs()`, `.ln()` and `.cadd()` build
source expressions. `.fill(value)`, `.arange(start)` and gather methods mutate
their destination. Numeric `cast`/`.astype` needs a valid cast configuration;
`.reinterpret` changes the interpretation of the same bits. Packed compute
views and byte carriers are explained in [formats](formats.md).

Ordinary masked register arithmetic zeros inactive result lanes. An
accumulator operation still reads its initialized destination on active lanes:

```text
axpy:       dst = old_dst + src * scalar
muldstadd:  dst = old_dst * src0 + src1
muladddst:  dst = src0 * src1 + old_dst
```

The [reduction example](../../examples/api/register_reductions) checks
whole-register sum/min/max, per-block reductions and pairwise sums. These
publish a leading scalar, a block-result prefix or a half-width prefix,
respectively. Their zero result lanes are part of the example's reference.
[Logarithm cases](../../examples/api/register_logs) distinguish natural,
base-two and base-ten logarithms and declare a floating comparison budget.
An integer-only bitwise operation is not an implicit float reinterpretation.

A contiguous store applies lane predicates; a contiguous load has none.
`reg <<= ub[...]` and the `ub_to_reg_*` family compile to `vf.load_cont`, whose
declaration carries no mask attribute at all, so nothing narrows the read: its
extent follows the register and the distribution. `ub[...] <<= reg` and the
`reg_to_ub_*` family compile to `vf.store_cont`, which does take one. The
block-strided `ub_to_reg` / `reg_to_ub` accept a mask on either side. Do not
apply one blanket mask rule to every helper: for the non-`normal` store
distributions the predicate is rounded to 32-byte block granularity, except the
measured `pack_b32` of a b16 register, which gates each packed element and
leaves holes (M10-064).

The current A5 CCE/PTO-ISA stride-one `ub_to_reg` shortcut still reads a full
register before applying its mask ([M10-088](../defects/M10-088-masked-stride-one-load-footprint.md)).
Until repaired, use an actually narrow distribution or sufficient owned backing
when the load reaches an allocation boundary.

Each distribution has its own extent. For a register of L lanes, counted in
elements of the memory dtype from the view's element offset:

| Author call | UB elements touched | Lanes | Element widths |
| --- | --- | --- | --- |
| `ub_to_reg_normal`, `reg <<= ub[...]` | L | all L | b8, b16, b32, b64 |
| `ub_to_reg_downsample` | 2L | every other element fills L lanes | b8, b16 |
| `ub_to_reg_upsample` | L/2 | each element reaches two adjacent lanes | b8, b16 |
| `ub_to_reg_unpack` | L/2 | each element reaches one even lane | b8, b16, b32 |
| `ub_to_reg_unpack4` | L/4 | each element reaches every fourth lane | b8 |
| `ub_to_reg_brcb` | L/c0, `c0 = 32 / element bytes` | each element broadcast over one 32-byte block | b16, b32 |
| `ub_to_reg_single` | 1 | broadcast to all L | b8, b16, b32 |
| `reg_to_ub_normal`, `ub[...] <<= reg` | up to L, each lane at its own position | active lanes | b8, b16, b32, b64 |
| `reg_to_ub_downsample` | L/2, compacted from the offset | active even lanes | b8, b16, b32 |
| `reg_to_ub_pack4` | L/4, compacted from the offset | active every-fourth lanes | b8 |
| `reg_to_ub_single` | 1 | lane 0 | b8, b16, b32 |

A distribution outside its width column is a frontend diagnostic, not a slow
path: `ub_to_reg_unpack4` on an FP16 register is `load distribution 'unpack4'
is not defined for f16`.

A dual-register interleave consumes or produces two full lane sets, ordinarily
512 bytes. Native `pack` selects low bits into a register half; `.pack4()`
selects every fourth byte lane while storing to UB.

An FP16 or BF16 register holds 128 lanes, so a `normal` load through a
64-element row view reads 128 elements and a `normal` store writes 128. The
allocation behind the row has to back the instruction's footprint; a
`MaskReg(dtype, init_mode=MaskType.LOWHALF)` limits the store to 64 and cannot
shrink the load. Both simulators reject an access outside the allocation and
name the mode, the lanes and the elements reached.
The bounds review and
tail-store hardware receipt
retain the M10-057/M10-060 failures and repairs at their recorded source scopes.

The [narrow cube/vector roundtrip](../../examples/api/cube_vector_roundtrip)
loads a full 128-element FP16/BF16 row, then uses a same-dtype `LOWHALF` predicate
to store 64 elements into compact NZ. Its initialized load slack, strided store
addresses and guard readback make the distinct footprints observable.

## Execution masks and selectors

`MaskType` provides ALL/NONE, fixed prefixes, a half/quarter prefix and periodic
MULTI3/MULTI4 presets. They initialize lane predicates; they do not describe
packed mask bytes in UB. `MaskReg.update(count)` and `update_mask(mask, count)`
consume a UINT32 cell count: activate up to the mask width, then subtract that
width with a floor at zero.

**The count must be a cell, not a literal.** `update_mask(mask, 40)` emits on
`pypto_pro` and runs, but `cce` and `pto_isa` both refuse it with `mask_update needs a
scalar counter, not a literal` — the device instruction decrements its counter in place
and a literal has no storage to decrement. Write `cnt = Var(40)` and pass `cnt`. The
literal form therefore cannot be cross-checked against `cce`, which is the control
backend a board verdict is measured against, so the refusal surfaces late: after sim,
pipesim and a PyPTO board run have all passed.

**`dup` broadcasts; it is not a masked move.** `dup(dst, src, mask)` with a *register* source
writes **lane 0 of that source** into every active lane — it does not copy lane to lane. Measured
on a 64-lane f32 register under `MaskType.LOWHALF`, with the source holding `100, 101, 102, …`:
lanes 0–31 all come back `100`, and lanes 32–63 come back `0`, not their previous contents. So it
is a broadcast for the active half and a clear for the rest. To keep lanes distinct and zero only
the inactive ones, use masked ordinary arithmetic — `adds(dst, src, 0.0, mask)` — which the rule
at the end of this section already covers.

`compare` produces false in inactive destination lanes. `select` writes each
data lane from its first or second source according to a selector mask.
Likewise, `mask_sel` chooses a source for every physical predicate bit;
execution-masked not/and/or/xor/move zero inactive bits. `mask.select(a, b)` builds
a data-register select; `mask.sel(a, b)` builds a predicate select expression.

That sentence is the whole `mask_*` family in one line, and it is easy to read past.
[Mask write semantics](mask-write-semantics.md) is the same answer as a per-operator table
over all 66 masked operators, with the two mask-register writes pinned by execution: a masked
`mask_mov` clears a destination bit that was set, and only `mask_sel` writes every bit from a
source. Nothing preserves an inactive destination register lane; only memory can be skipped.

[The mask semantics unit](../../examples/api/mask_semantics) contains
independent Boolean formulas for these distinctions, pack/unpack and
interleave/deinterleave, plus a published remaining counter. Its contract and
actual validation evidence govern support; merely constructing an enum or
emitting an instruction is not a successful numerical check.

Mask routing uses one physical 256-bit predicate register. `mask_pack` selects
even physical source bits into the selected 128-bit half and clears the other
half. `mask_unpack` expands that half into even physical positions and clears
odd positions. Logical b32 lanes observe bits 0,4,...,252, so lower pack then
lower unpack preserves all those lanes. Interleave combines effective-width
bit groups from two inputs across two outputs; deinterleave reverses routing.
Supported grouped data registers share this physical predicate at their
effective width. Raw b8/b16/b32 routing was measured; unobserved producer fine
bits and single-register b64 raw fields remain outside that qualification.
See the physical-predicate controls
and the [mask rules](mask-write-semantics.md).

`move_mask_spr` reads the vector mask state set outside the VF. The retained
method name `.move_to_spr()` also builds this read; its name must not be taken
as a write to the SPR. SPR lane indices and the physical 32-byte mask storage
layout are separate contracts. Reset temporary vector mask state before a
later operation that expects the full vector domain.

## Access boundaries

[Unaligned rows](../../examples/api/unaligned_rows) shows streaming
cursor state and the stateless once forms. A streaming load primes its state;
a streaming store ends with `reg_to_ub_unalign_post` so buffered tail bytes
become visible. That Post is required, not an optimization: the buffered bytes
live in the hardware register, and a vf that returns while a store state still
holds them is an illegal program that corrupts the next store to reuse the
register (RFC-0001 §6.11). Cursor position, valid tail count and physical read
slack are declared explicitly. The reference observes the complete promised
output. An unaligned store and its Post always advance their cursor:
`post_mode=PostMode.NORMAL` remains accepted through 0.1.x and means
`PostMode.UPDATE`. Only unaligned loads have a form that leaves the cursor.

[Block gather](../../examples/api/block_gather64) uses explicit byte
addresses and validates its 64-bit carrier route. [Grouped interleave](../../examples/api/register_groups)
checks both destination halves and source alias behavior inside its model
domain. These units do not turn every historical simulator alias behavior
into a vendor guarantee.

`print_reg` is a simulation inspection aid with an explicit lane limit. It is
separate from an observable output contract. Host scalar code uses Python
`builtins.abs`; facade math names remain compiled DSL calls.
