# Memory footprints and tails

<a id="vector-tail"></a>
## Start with an ordinary vector tail

For contiguous elementwise rows in an ordinary dtype, complete these steps and
open one matching owner example:

1. State each row's valid elements, GM stride, physical UB pitch and allocation/slot size.
2. Compose the UB allocation, slot and offset into the actual base; check the instruction's 32-byte alignment.
3. Derive active load, compute and store lanes/bytes separately. Choose an explicit
   predicate for the distribution and fit actual accesses inside owned backing;
   calculate GM valid extents separately from physical UB extents.
4. Initialize or mask padding before its first consumer; reductions use the operation's identity.
5. Check distinct row values, output poison, neighboring guards and the final row,
   then run functional and pipesim checks.

Choose a matching dtype/distribution in [unaligned rows](../../../library/examples/api/unaligned_rows)
or [mask semantics](../../../library/examples/api/mask_semantics).
For reductions, add [the numerical reduction section](numerical-patterns.md#reason-reduction).
Packed/HiFloat8, NZ, subviews, splits, slots or cross-side protocols trigger their
specific boundaries below; an ordinary tail does not require this entire reference.

<a id="specific-boundaries"></a>
## Instruction and layout boundaries

For every operation distinguish logical live elements, the bytes the instruction
accesses, and the allocation backing those bytes. A mask or narrow view does not
automatically shrink the access footprint. Derive it for the selected dtype/layout
and instruction; inspect the source model and generated instruction when uncertain.

For an A5 FP16 register, an unmasked `NORM_B16` store writes 128 lanes even through
a 64-element row view. For a 64-lane write, create
`low = MaskReg(f16, init_mode=MaskType.LOWHALF)` and pass it explicitly to
`reg_to_ub_normal(dst_row, values, low)`. An in-range full store can overwrite the next
logical row; at the allocation's end it must fail before writing any bytes. The
M057 repair checks
the distribution and active destination addresses against the actual allocation.
It does not turn view extents into a mask. Keep untouched neighbors and the last slot
in the regression; packed-mask activation outside measured domains remains a model convention.
The [narrow roundtrip](../../../library/examples/api/cube_vector_roundtrip)
provides FP16/BF16 64-element cases, physical address tables and slot-wrap controls.
The load half is not symmetric: `vf.load_cont` carries no predicate at all, so read the
[footprint per distribution](../../../library/docs/api/registers.md#load-compute-and-store)
rather than each case's own numbers.

<a id="packed-writeback"></a>
## Derive a packed writeback

1. Trace logical values through cast/rounding, carrier lane placement and pack/unpack bit order.
   Use the selected [format](../../../library/docs/api/formats.md) and its owner example.
2. Apply the explicit mask to carrier lanes, then derive the store distribution's destination
   byte intervals. Check each row/slot's owned range and the allocation, including the tail.
3. Compare raw packed bytes, neighboring guards and the final row; check adjacent state
   stores separately. Choose each mask from its dtype and distribution.

M060 shows why the
distribution matters too: a full 256-byte HiFloat8 ZERO-layout carrier stored with
`PACK_B16` (CCE `PK_B16`) writes 128 packed bytes, but the V2/V4 tail's single-key
probability row owns 64 bytes. Its explicit HiFloat8 LOWHALF mask selects the first
128 carrier bytes, producing the required 64-byte row. The adjacent 128-half state
buffers still require their valid full 256-byte `NORM_B16` stores. Derive each store's
extent separately; a tail repair is not a rule to halve every store.

## Transfer and physical layout boundaries

Cache-coherence operations have their own cache-line scope. Do not infer that scope
from a DMA burst or the simulator's access-conflict block. Scalar GM publication
needs its versioned instruction and separate device evidence; see the reviewed
cache-scope note — historical record (`docs/migration/fragments/docs-closure.vendor-review.md`).

For a byte burst with positive burst count, the highest accessed end is
`(n_burst - 1) * step + burst_len` from its base. An empty transfer has zero footprint;
do not apply that expression blindly at zero. The gap between bursts is not read.
A padded destination can write an aligned span larger than the source payload.
For a nonempty strided view, coverage is `1 + sum((span - 1) * stride)` in elements
for nonnegative strides; validate each dimension and the parent storage separately.
Check the actual consumer's stride support instead of assuming every view is a DMA.

NZ/ZZ allocations must satisfy physical panel pitch/alignment; logical `M*N` alone
is insufficient. A view/reinterpret/layout marker describes storage; it does not
perform arbitrary packing or a numeric conversion. Packed/exotic dtype carriers
need bit order and logical-versus-carrier extent in the contract.

Keep physical local tiles stable across a tail when the instruction requires a
full tile, with separate valid extents. Gate GM reads/writes and initialize/mask every
padded lane that later computation can observe. A bridge's split is part of its ABI:
fixed physical A2-family workspace halves cannot be replaced with a compact split
of valid rows without changing both producer and consumer layout.

An unaligned store chain can keep a partial block in separate state until its
required final flush. Track cursor mutation, prime/flush pairing and the final
partial block; advancing a cursor is not a proof of the instruction's access extent.
Use the selected overload's rules and compare untouched neighboring bytes too.

Apply reduction identities before the first affected reduction. For online softmax,
mask invalid key/causal score columns before max; mask invalid query rows before exp
or prevent their consumption; update the sum in its declared precision before a
probability cast. Keep fully masked-row behavior explicit. A finite sentinel is valid
only when it bounds the allowed score domain and yields the required semantics.
Do not claim all infinities are unsupported or allow NaNs in public outputs by default.

A2 tensor-vector `count` selects the total element count in counter mode;
`count_per_rep` selects the live lanes inside each repeat. They are mutually
exclusive. Use repeat and repeat strides for several chunks. The current IR carries
the mode on each operation; a mode inferred from source order is unsafe across
branches and zero-trip loops. For fp32, a 32-byte block contains eight lanes.
Group32 with stride four begins at lanes 0, 32, 64, 96; it is not a full-row reduction.
Check the whole chain's scratch footprint: a scalar result may feed an instruction
with a full aligned vector access. Preserve [precision](precision.md) and
[synchronization](synchronization.md) at these boundaries.

Source lookup in the accepted library: `ascriptor/frontend/rules_mem.py`,
`ascriptor/frontend/rules_vec.py`,
`ascriptor/backends/sim/dma_ops.py`, `ascriptor/backends/sim/vec_ops.py`,
`ascriptor/backends/sim/interp.py` (`MemRef`, `Machine`) and
`ascriptor/backends/sim/pipesim.py` (`Access`). These are model-derived lookup
points, not new hardware measurements. A hardware discrepancy follows
[white-box debugging](simulator-white-box.md).

Test the smallest valid and first invalid footprint, one and multiple bursts,
aligned/tail shapes, untouched output canaries, and repeated use by the same core.
Do not hide an invalid access by clipping its view or weakening a simulator guard.

For a mixed pipeline, sum resident weights and each independently derived slot
family at the same memory level; see [CVC storage accounting](cube-vector-cube.md#concrete-lifetime-table).
A residency win and a deeper pipeline can compete for the same capacity.

## Align the UB instruction base

Ordinary register↔UB and GM↔UB instructions require a 32-byte-aligned UB
base after composing allocation, buffer slot, view/reinterpret and explicit
offsets. Each physical UB row of a multi-row DMA must also start aligned;
logical valid columns do not define that row's pitch. The GM side retains
element-granular offsets: a legal 16-byte GM offset does not violate the UB rule.

Keep logical payload and physical blocks separate. Each UB allocation/slot owns
its allocator-rounded backing. A four-FP32 DMA payload can use a whole 32-byte
carrier with four valid elements, but its complete padded footprint must stay
inside that backing. It cannot borrow a neighbor's padding. Register distributions
and predicates still determine their active lane effects; a narrow view supplies
no implicit mask.

An executed ordinary register/UB instruction must have an aligned base even
with every mask lane off. The mask suppresses selected data effects, not the
base-address precondition. Scalar `.single()`/`.single_value()` use element
alignment; explicit unaligned instruction families retain their own state,
range and flush rules. A zero-burst DMA and instructions never executed, such
as a zero-trip VF loop, perform no access. The owner specifies these distinctions
in the [IR memory contract](../../../library/docs/rfc/0001-ir.md).

The historical M065 adapter failure received four FP32 values from a legal
GM offset of 16 bytes into aligned UB, but emitted the invalid physical type
`Vec<float, 1, 4>`. The carrier regression retains this boundary. The repair belongs to
the adapter's [whole-block carrier and valid shape](../../../library/docs/rfc/0011-pto-isa-backend.md),
preserving the legal kernel address. Diagnose the generated carrier before
changing the kernel's layout; distinguish a model pass from native compilation.

## Trace a view into its physical tile

Base alignment and physical row pitch are separate obligations. For a FP32
UB parent `[16,256]`, left/right column windows `[16,128]` have these addresses:

| Window | Required row 0 / row 1, bytes | If incorrectly materialized as a compact `[16,128]` tile |
|---|---|---|
| Left, columns 0–127 | 0 / 1,024 | 0 / 512 |
| Right, columns 128–255 | 512 / 1,536 | 512 / 1,024 |

All four origins are 32-byte aligned; that does not make a compact carrier
preserve the parent's pitch. Follow allocation/slot → logical view origin and
strides → backend physical tile, valid shape and consumer offsets. A valid
shape describes live elements; it does not by itself retain a wider row pitch.
Keep the final physical footprint within the actual parent allocation too.

A functional/pipe model can preserve the lowered view while an emitter loses
its pitch. For FIX, also track the explicit `N_dst`: modeled writes and traced
byte intervals must follow that descriptor, rather than substitute the parent's
logical stride. Use row/column labels, at least two rows, both column offsets,
poisoned parent padding, repeated slot reuse and full readback to expose that
mapping. The [FIX destination comparison](../../../library/examples/api/cube_vector_roundtrip#fix-destinations-retain-the-parent-ub-pitch)
keeps contiguous, pitched-window and independent-UB cases together.
[M066](../../../library/examples/api/cube_vector_roundtrip#fix-destinations-retain-the-parent-ub-pitch) owns the
scoped comparison cases and evidence for the backend repair. Preserve supported subviews; if a backend
cannot express one, retain its located rejection and parameters instead of
silently changing the pitch or rejecting all subviews. A rejection on one
backend is not proof of the same failure on another.

L0B has a separate coordinate map: IR uses `[N,K]`, while native PyPTO `Right`
uses `[K,N]`. Allocation, reinterpretation, slot selection and slicing must carry
that mapping exactly once; selecting a static slot must not swap an already
translated shape again. A correct shape still does not prove the physical pitch:
a partial-N window spanning multiple K fractals retains gaps at the parent's N
pitch. Preserve those addresses when materializing it, or retain a located
backend refusal. UB row-block rules do not define the Right fractal layout.
Use the [owner mapping rules](../../../library/docs/rfc/0011-pto-isa-backend.md)
and M067 controls
for the precise supported windows and separately recorded stages.

Before turning a slot/window failure into an API rule, use the existing paired boundary checks:
their source and the
test that runs them, beside the
[roundtrip example](../../../library/examples/api/cube_vector_roundtrip) they came from.
For K256 b16 PV, `splitn=64/128` needs 32/64 KiB per implicit L0B slot,
whose capacity is 32 KiB. The first passes both models and all three emitters;
the second is rejected by pipesim and PyPTO, while CCE/PTO-ISA only emit source.
That emission is not a legal-runtime result. The same probes show a legal
same-byte UINT8 `[32,64]` → FP16 `[32,32]` L0B reinterpret. They separately
record lowering's rejection of simultaneous splitn/splitk, and PyPTO's accepted
`[16,64][:,16:32]` L0C strip versus its located rejection of
`[32,64][:16,16:32]`. Read each exact case and stage; these model/emission
controls do not establish native qualification or a blanket reinterpret/subview ban.

## Match the mask to the register layout

A predicate selects physical register lanes before the store distribution
packs them. For the FP32-to-b16 ZERO-layout conversion used by the
[attention example](attention-authoring.md), 64 converted values occupy 128
b16 register positions. `reg_to_ub_downsample` packs those 64 values; applying
a b16 LOWHALF mask at that point would retain only 32. Conversely, a dense
64-value b16 load followed by a normal store needs the explicit 64-lane mask.
The same logical width therefore does not imply the same mask or store.

Trace conversion placement, optional deinterleave/packing, store distribution
and destination pitch as one chain. Compare raw staging bytes and neighboring
guards, not just a final reduction that might hide dropped or duplicated lanes.
