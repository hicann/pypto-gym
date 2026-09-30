# RFC-0015: Instruction-preserving PyPTO Pro import

Status: development. Versioned transport, bounded instruction conversion and
local synchronization/storage and bounded cross-side import are implemented.
Device qualification remains a separate milestone.

## Boundary

Import specialized PyPTO Pro device operations into verified `lowered/1`.
Preserve typed operands, operation order within each participant, control flow,
physical storage, numerical modes, state changes and synchronization. A source
operation may expand into an explicitly justified sequence; Python lines and
machine instructions are not the correspondence unit.

```text
Pro source + specialization inputs
  -> native semantic parser
  -> target Programs + functions + semantic metadata
  -> importer + necessary operation legalization
  -> fully verified Lowered IR
  -> simulation / selected backend
```

Pro owns parsing and compile-time evaluation; library owns import and target semantics.
[RFC-0001](0001-ir.md) owns
IR semantics; [RFC-0006](0006-lowering-pipeline.md) owns ordinary lowering.

## Integration contract

| Layer | Required information and target |
| --- | --- |
| Entry | Revision, device, specialization, static/runtime distinction; Module and parameters |
| Structure | Cube/Vector Programs, VF/SIMT, control and mutation; side functions, `cf.*`, `scalar.*` |
| Storage | Physical identities, addresses, layout, valid extents, aliases and slots; types and `mem.*` |
| Instructions | Typed operands, attributes, numerical modes, effects; concrete device opcodes |
| Synchronization | Explicit and Pro-generated locks, flags, barriers, state; IDs, pipe, side and order |
| ABI/provenance | Parameter order/direction, shapes, launch/workspace, source spans; metadata and OpID origins |

Capture parser-owned buffer/mutex relations before parser disposal. Stable keys
link Cube/Vector declarations without merging participant storage. Version the
export and pin the actual Pro build; audit snapshots are not runtime dependencies.

Every conversion rule matches opcode, types, attributes and target; records
its output operations and effects; and reports unsupported forms at source.
Account for every admitted operation as direct, expanded or declarative-only.
If codegen performs semantic expansion, provide a justified legalization or a
later structured export. Do not parse emitted C++ or hide backend text in IR.

## Fidelity and admission

Use a dedicated import pipeline. Do not invoke ordinary lowering to reallocate
memory, replan synchronization, renumber events or coalesce instructions.
Preserve the source synchronization configuration, including `auto_mutex`
generated operations; manual and automatic configurations are separate cases.
Construct all Lowered invariants before publishing a module. Necessary target
rewrites use Rewriter, explanations and source-to-output correspondence.

Admit control, storage and numerical forms only with matching semantics. Extend
owner RFCs and registry/verifier/model/backend support together; never weaken
verification, discard operations or approximate a gap.

## Acceptance

### VF arithmetic coverage

FP32 arithmetic imports admit explicit/default ZEROING only. Source-to-target
tables in `importers/pypto_pro/vector.py` drive arithmetic dispatch: elementary
unary math, binary arithmetic, scalar min/max and non-datablock reduce_sum.
FP32 leaky_relu (scalar slope), prelu (register slope) and pair_reduce_sum
also preserve explicit/default b32 ZEROING. Pair sums retain the complete
64-lane result: inactive source lanes contribute zero, adjacent pairs produce
32 sums in the low half, and the high half is cleared. Odd prefix lengths keep
the final active source element without including its inactive partner.
Unknown attributes, other precision modes and MERGING remain located gaps.
Reduction preserves all result lanes; reduce_max/min
keep their native index lane (see SPR masks, register reductions and interleaved
transfers). Backend printing and reverse-import evidence remain distinct in
the generated [import support matrix](../pypto-pro-import-support.md).

B32 predicates admit ALL/ALLF and VL1..VL64, comparisons, and runtime update_mask.
The latter casts a snapshot to uint32 and decrements a private cell, matching
Pro's temporary counter; it never mutates the source count. Comparisons accept
same-dtype FP32/INT32 vectors or a matching scalar. Select preserves false lanes
from its second source; ZEROING arithmetic/broadcast clears inactive lanes.
Masked stores leave inactive UB elements unchanged. Register memory transfers
are full aligned 256-byte accesses: 64 FP32/INT32 or 128 FP16/BF16 lanes.
Store predicates must match the register width; alignment is 32 bytes.
B16 constructors admit ALL/ALLF and VL1..VL128, plus snapshot-based update_mask.
Selection, comparison, broadcast and arithmetic admission remain independently bounded.

Same-width FP32↔INT32 casts admit ZERO layout and rint/round/floor/ceil/trunc.
FP32→INT32 retains OFF/ON saturation; INT32→FP32 requires saturation OFF.
Source `set_ctrl_spr` on individual saturation bits maps to raw
`core.set_sat_flag` (see launch identity and CTRL readback); instruction RS
choices remain subject to CTRL, not an invented importer default. Multi-bit and
unrelated CTRL fields are rejected.
FP32↔FP16/BF16 additionally admit ZERO/ONE layout, RINT (nearest even) and
saturation OFF. Narrowing admits b32 predicates; widening admits b16 or b32.
Narrowing sends source i to destination 2*i+layout and tests its b32 predicate.
Widening reads source 2*i+layout and samples the predicate at that source's
physical bit position (2*(2*i+layout)). A b16 mask can activate either position;
a b32 mask only activates even source positions, so widening ONE with b32
produces zero. Preserve the predicate without reinterpreting its granularity.
ZEROING clears inactive result lanes and unused interleaved slots. Other cast
pairs, widths, layouts and rounding/saturation variants remain located gaps.

Signed INT32 bitwise and/or/xor/not and left/right shifts admit b32 ZEROING.
Shift dispatch follows the source amount type: INT32 register for per-lane
amounts, or INT32 scalar/integer literal for uniform amounts. Preserve Pro's
explicit int16 narrowing for the scalar form. Constants outside [0, 32767]
are rejected; runtime scalar amounts must be in that interval, and vector
amounts must be nonnegative. Negative source shifts are undefined by Pro and
are not assigned an importer guarantee. Signed right shifts extend the sign;
left shifts truncate to 32 bits. Counts at/above 32 are checked separately.
Dtype reinterpretation, unsigned data and other widths remain located gaps.

### Register arithmetic and fused accumulators

FP32 add/sub/mul/max/min over matching registers and muls with an FP32 scalar
preserve explicit/default b32 ZEROING. Numeric literal scalars round to FP32;
integer literals must be exact (|v| <= 2^24). Cancelling nonzero terms give +0;
max/min ties return the shared value. Signed-zero/NaN ordering is unqualified,
and Pro's MERGING add/max remain gaps.

axpy, mul_add_dst and mul_dst_add update an FP32 destination in place:
`dst = src*s + dst`, `dst = src0*src1 + dst` and `dst = dst*src0 + src1`.
A5 execution rounds each exact expression once to nearest even; separate
multiply/add rounding is a different result. Aliased operands read the prior
destination value. The destination must be written earlier on every path of
the same VF body: native registers are uninitialized, so a declaration alone is
rejected. ZEROING clears inactive destination lanes instead of preserving the
accumulator. Other widths, integer accumulators and MERGING remain gaps.

### Scalar bit operations and extrema

Pro casts binary scalar operands to the promoted type and prints C++ operators.
INT32/INT64 or/xor/not, shifts, negation and min/max require operands of that
type. Shifts follow C++17: counts in [0, width); a left operand is nonnegative
and its result fits the unsigned width before signed conversion; right shifts
are arithmetic. Constant violations are rejected and the model checks runtime
values, as for division; negating the signed minimum is outside the domain.
Integer stores into narrower GM keep the low bits. Logical not is scalar.not on
BOOL and `x == 0` on INT32/INT64/FP32. FP32 negation flips zero signs; NaN
results only remain NaN. FP32 min/max in vector and cube sections map to
scalar.min/max, whose IEEE rule A5 measured there (RFC-0001 §6.16). A5's compiler
rejects `max()` in a VF body (`max() in vector function only supports integer
types!`); there both are invalid A5 IR, and the module check refuses each site.
SIMT forms are unmeasured and refused at source. Pro has no
ceiling-division expression, so scalar.ceil_div has no Pro source form.

### Mat fills and literal spelling

Pro CCE prints a finite FP32 literal as `std::to_string(double)` plus `f`: six
fractional digits, then binary32 rounding by the compiler. Admit a literal only
when that spelling keeps its FP32 bits; `1.00000006` prints as `1.000000f` and
is rejected at source. Admitted literals round once to nearest even.

`expands` on a cube-side FP16 Mat tile maps to `dma.set_constant_to_l1` over the
whole tile. TEXPANDS takes its repeat count from the tile capacity and ignores
the valid extent, so the extent must equal the shape; the literal must be exact
in FP16. PTO issues n_blocks repeats of one block where the target packs one
repeat of n_blocks (RFC-0011 §4.9); both write the same bytes, and L1 capacity
stays below the 32767-repeat limit. Runtime values, other Mat dtypes, `full` and
Vec expands remain gaps. On silicon a fill and later transfers of that tile
reorder without a barrier; the import keeps only barriers written in the source.

Audited targets without a Pro source form: kernels have no tensor-list
parameter (list.count/item/item_dim); stores require ascending order and PTO's
Acc store has no DN arm (dma.l0c_to_gm.nz2dn); AtomicType is a per-store none/add
attribute with no atomic region (atomic.begin/end/set_type). The padding-only L1
fill that used to head this list, `dma.fillpad_l1`, left it by being retired
rather than by gaining a producer (library `f75ee23`); Mat fillpad still does not
compile natively (A5-UP-004).

### Fused casts, predicate spill and memory barriers

exp_sub maps FP32 sources with a b32 predicate and layout ZERO, or FP16 sources
with a b16 predicate, to an FP32 `vf.expsub`. FP16 results read source lane
2*i+layout and A5 tests the predicate at that source position; inactive lanes
are zero. vexpdif is approximate: models agree with A5 within 1e-6 relative.
muls_cast maps FP32 with a b32 predicate and an FP32 scalar to FP16
`vf.mulscast`: A5 rounds the exact product once, ties away from zero, into lane
2*i+layout and zeroes every other lane, including the other half.

A b32 predicate `and_` maps to `vf.mask_and` in ZEROING mode. A two-operand
`store_align`/`load_align` with a b32 or b16 predicate spills or fills its 256-bit
image through a whole UB tile of at least 32 bytes (A5 returned both unchanged).
`mem_bar` maps to `vf.barrier`. Its `mode` is a MemBarMode index, VST_VLD when
absent, and Pro's emitter prints the enumerator's name, so each of the twelve
indices takes that tag's (src, dst) pair; any other value fails at source.

A5 ran all twelve tags, and the imported CCE artifact matched native Pro bit for
bit. No tag has a measured ordering effect. A VF that stores a tile and reloads
it through a second register read the stored value under the documented tag,
under a tag that misses the pattern and with no barrier at all, so the probes
could not make a wrong mode visible; the model orders a VF's own accesses by
program order and keeps doing so. Across the VF boundary the documented order
failed: with VST_LD, VLD_ST or VS_ALL in place, a prefix of the S-pipe loop after
the call still took the other order (10, 12 and 2 of 64 elements at seed 7),
exactly as with no barrier. A `mem_bar` therefore discharges no dependency during
import: an S-pipe access beside a VF call stays an unordered hazard that pipe
simulation reports, and such kernels stay out of the gate.

Audited targets without a Pro source form: Pro has no register print
(debug.print_reg), no event object or cross-core mutex declaration, only raw
flags and cross-core calls (sync.event/set/wait/set_all/release/mutex), no
count-to-mask SPR call (vec.set_mask_by_count) and no VF gathermask.

### Launch identity and CTRL readback

On AIC, `get_block_idx`/`get_block_num` map to `core.cube_idx`/`core.cube_num`.
On AIV, Pro prints `get_block_idx()*get_subblockdim()+get_subblockid()`, which
is `core.vec_idx`; raw `get_block_num()` is `core.cube_num` in mixed launches
and `core.vec_num` in AIV-only ones; `get_subblock_idx` is `core.sub_block_idx`
in mixed launches. `get_subblock_num` prints 1 on AIC and `get_subblockdim()` on
AIV: `scalar.const` 1 on AIC and in AIV-only launches, otherwise `core.vec_num`
divided by `core.cube_num`. INT32 results are cast to the source INDEX type.

Range proofs bound these queries by the exported `block_dim` B: AIC `core.cube_idx` and AIV-only
`core.vec_idx` to [0, B-1], block counts to [1, B], and in mixed launches AIV `core.vec_idx` to
[0, S·B-1] and `core.sub_block_idx` to [0, S-1] for S AIVs per AIC. GM windows at offsets derived
from them import when proven. Every launch entry (executor, `OpExec`, `compile_kernel`, the CCE,
PTO ISA and PyPTO Pro printers, simulator, pipesim, host `SetBlockDim`) therefore uses B and
refuses any other before launch, naming the kernel and its source.

CTRL bits 48/50/53/59/60 map to `core.get_sat_flag`/`core.set_sat_flag` raw
bits. `get_saturation_flag` compares the bit with zero: FLOAT, FLOAT8 and CAST
are on at 0, INT at 1. Pro writes `value & 1`, so values not proven 0/1 pass
through `scalar.and`. Pro prints reads without outer parentheses and holds
`get_ctrl_spr` in `uint64` locals. Its parser binds nested operands to `auto`
temporaries, so a read must be a whole assignment value, and integers derived
from `get_ctrl_spr` must be proven nonnegative with their operands. A reassigned
variable that held one has no proven range.

### Cross-core collectives

Static `set_cross_core`/`wait_cross_core` keep their pipe and ID (0..10). Mode 0
(INTER_BLOCK) maps to `allcube_*` on AIC and `allvec_*` on AIV: every core of that
side in the launch. Mode 1 (INTER_SUBBLOCK) maps to `intracore_allvec_*` among the
AIVs of one core in mixed launches only. Both print the same packed flag as Pro.
Refuse AIC or AIV-only mode 1, mode 3 (UNICAST_BLOCK), IDs 11..15 and `sync_all`.
Also refuse AIV mode 0/1 sets on PIPE_S: native and imported CCE builds both fail the A5
compiler's check that `ffts_cross_core_sync` takes V, MTE2 or MTE3.
Balanced publishes and waits remain the source's responsibility.

### Debug calls

Debug calls are observation-only: outputs must not depend on them, and the CCE
backend prints them as comments. `printf` maps to `debug.print`, with Pro's
`[file:line] ` prefix when `loc` is set. `%p` is refused, and `%u`/`%x` need a
proven nonnegative argument. A failed `pto_assert` prints and continues in Pro,
so it becomes `cf.if(!cond)` around its prints. `debug.assert`, which stops the
model, has no Pro producer. A GM tensor dump keeps TPRINT's `PipeBarrier(ALL)`
before `debug.dump` of a proven window. Tile dumps and traps are refused.

### Data cache clean

`system.dcci` of a static INT32, FP32, FP16 or BF16 GM tensor parameter maps to
`core.clean_dcache`. Its `dst` window starts at the address Pro prints. Without an
offset, that is the parameter itself. Otherwise it is the element at `[row, col]` or
at a row-major element offset, proven inside the tensor. `cache_line` SINGLE/ENTIRE
sets `entire_type` SINGLE_CACHE_LINE/ENTIRE_DATA_CACHE. `dst` AUTO and CACHELINE_OUT
both set `dcci_dst` CACHELINE_OUT, which is what Pro prints for tensors; UB, ALL and
ATOMIC keep their names. These forms matched their references in native and imported
CCE A5 runs on lines that one core uses alone. Tiles, GM views and other element types
refuse until measured. Cross-core scalar stores that share a cache line need a
CACHELINE_OUT dcci after the store *and* a publication that cannot run before it;
either alone lost stores in every launch (I012), and other destinations are
unmeasured there. The model treats dcci as value-neutral.

### SIMT launches

Each launched `SimtVF` function converts once into a `simt` function. Scalar
parameters keep their type; static GM tensors and whole Vec tiles become `gm`
and `ub` parameters. Every AIV must reach `simt.launch`, so it must sit at the
top level of the vector section. It admits one thread dimension, at most
`max_threads` threads, and arguments of the parameter types. In the body,
`getval`/`setval` become `simt.load`/`simt.store` on FP32/INT32 storage and
UINT32 GM tensors. The row-major index must be proven inside the storage, and
integer or BOOL values of another integer type convert as in C. `syncthreads`
becomes `simt.barrier` and must sit at the top level of the body. The two fences
keep their names.

Pro context queries are UINT32 casts of the INT32 target queries:
- `linear_thread_idx` and `thread_idx().x` map to `simt.thread_id`, in [0, N-1].
- `block_dim().x` maps to `simt.thread_num`, in [1, N].
- In AIV-only launches, `block_idx().x` maps to `simt.block_idx` and
  `grid_dim().x` to `simt.block_num`.

Pro binds a three-axis tuple for a `.x` read and reuses its x call; the y and z
calls are declarations. UINT32 arithmetic must be proven inside [0, 2**32).
`max_threads` bounds the thread facts; the target prints the largest launch
count as the launch bound. Threads that share storage are the source's
responsibility; the model runs them in thread order between barriers. Refuse
other axes, core queries in mixed launches, tile valid-shape reads, helper
calls, early returns, nested launches and barriers, integer/float store
conversions and other element types.

### VF block copies and unaligned access

`load_align`/`store_align` with DATA_BLOCK_LOAD or DATA_BLOCK_COPY map to
`vf.load`/`vf.store` (`vsldb`/`vsstb`) on FP32, INT32 or FP16 tiles and registers
of that dtype. The literal block stride lies in [0, 32767] (absent is 0) and is
given once; the base is 32-byte aligned and every block lies in the tile. A5 runs
fixed the RFC-0001 rules for predicates of the register's lane width: counts,
compares and loaded images. A stride-1 load prints the lane copy, so its predicate
must be an unwritten `create_mask` of whole blocks. `post_update=True` copies at
the cursor of its tile and offset, which no unaligned access shares, and advances
it `repeat_stride` 32-byte blocks; import folds straight-line cursors into
offsets. `repeat_stride` without `post_update`, `dist` and BF16 are refused.

In a VF, a literal `tile + k`, named or inline (Pro CCE aliases Tile lets),
addresses element k of a single-row UB tile. Pro `vf.load(E[, s])` maps to a new
load `vf.unalign`, `vf.load_unalign_pre` and `vf.load_unalign`;
`vf.store(E, r[, n])` to a new store `vf.unalign`, `vf.store_unalign` of n lanes
(default all) and a stride-0 `vf.store_unalign_post`. `load_unalign_init`,
`unalign_reg_for_store`, `load_unalign_pre` and `load_unalign` map directly.
Strides and counts move `vf.ub_cursor`s keyed like Pro's post-update pointers:
one per tile and offset in a VF function, restarting in every call and advancing
in tile elements. Import proves that each load reads where its register is
primed, so chains continue only after whole-register strides; that access stays
in the tile; that a cursor only loads or only stores; and that a VF function has
at most the four unaligned registers Pro documents.

An alias `v = u` of an unaligned register names that register, as Pro CCE prints
it. Refuse unaligned access and post-updating block copies in VF control flow,
runtime offsets, strides or counts, data and predicate register aliases and the
`vf.load`/`vf.store` keywords Pro's emitters ignore.
Chained `store_unalign` stays refused: NORM `vstus`/`vstas` do not compile and
`vstur`/`vstar` follow the AR count and may hang. A chain flushed by
`store_unalign_post(post_update=True)` needs an IR extension: A5 keeps an
unflushed tail across registers and VF calls and writes it into the next
unaligned store, and Lowered IR has no such state. Future work, in order:
top-level chains with literal counts and strides that prove a flush before any
other unaligned store and before return; chains in counted loops with a trailing
flush, as Pro's attention tests write them; AR-counted `vstur`/`vstar` after
squeeze STORE_REG, mask-source `pstu` and post-update block copies with repeat
strides, which need new IR state and probes. Device probes decide partial
blocks, stride 0, chains after other strides, the register limit and unaligned
carrier types.

### GM views and tile aliases

`make_ptr`, `addptr` and `make_tensor` over a static two-dimensional GM
parameter, or over a view of one, are declarations. A view keeps the root's
element width and has a static shape, static strides with an innermost stride
of 1, rows that do not overlap and a declared extent proven inside the root.
It maps to one root-sourced `mem.view`. A same-width dtype folds into its
result type, and an identity view is the parameter itself. A view used as a
source gives its origin and dtype, not its strides or extent: Pro CCE reuses
its base pointer, and native A5 runs read there. Transfers take the view's row
stride as their GM pitch: the burst gap, `N_src` and `N_dst`. A pointer offset
that reads a reassigned scalar is snapshotted where the pointer is computed,
not where a view is declared or used: Pro CCE copies it into a local, and
native A5 runs read that value. Views are not dump operands. `getval`/`setval`
through a view address its pointer plus the flat index, whatever its strides
(Pro prints `*((T*)ptr + i)`; `view_scalar_probe`): `scalar.load`/`store` of the
root, or its same-width reinterpret, at origin + index. A SIMT function reads a
view argument of its parameter's shape row-major at that shape's pitch from the
view's origin, annotated or not (`view_simt_probe`): the operand is a
`mem.reshape` of a `mem.slice` of the flattened root. Dtype views and runtime
origins as SIMT arguments fail at source.

A later Vec `make_tile` at an allocation's address and inside its elements,
as `pl.reinterpret` produces, re-declares those bytes. It maps to
`mem.reinterpret` and/or `mem.reshape` of the first allocation and starts
with its own valid shape, as the CCE constructor does. Pro CCE declares every
tile in the kernel prologue and folds a Vec declaration identical to an earlier
one without valid-shape metadata into a reference that shares its valid state;
declarations with metadata, `[-1, -1]` (the whole shape) included, stay distinct
(`alias_identical_probe`). Import returns that handle. A re-declaration in control
flow keeps one descriptor across trips (`alias_loop_probe`), so import binds it
before the outermost statement, where the loop descriptor rules apply: that
probe's store reads the previous trip's narrowing and fails at source. Aliases
take part in UB transfers, valid-shape updates, VF captures and `getval`/`setval`,
which index their elements flat (`view_scalar_probe`); each side of a paired
kernel re-declares its linked root. Other allocations in control flow, identical
Mat, mixed-metadata or tile group slot re-declarations, L0 aliases, other Mat or
layout changes, partial overlaps and aliases used as subview, split-M or SIMT
operands remain gaps.

### Transposed Mats, bias rows and Mat inserts

Native A5 runs selected each rule. A ZN Mat `[a, b]` holds the NZ bytes of its
transpose, typed `l1<T, [b, a], nz>`, and a ZN re-declaration of an NZ Mat with
reversed shape is that tile. A Mat-to-L0 move transposes when exactly one of
its Mat and destination is typed reversed. `order=[1, 0]` GM loads map to
`dma.gm_to_l1.dn2nz` into NZ Mats and to `nd2nz` into ZN Mats. Other ZN Mat
loads do not compile natively: PTO's `TLoadCubeCheck` needs an NZ Mat for an ND
source (`tload_common.hpp:131`, `mat_zn_plain_probe`). A one-row FP16 Mat is a
flat row: loads map to `dma.gm_to_l1.pad`, fills to `dma.set_constant_to_l1` of
its own blocks (`mat_row_fill_probe`), Vec rows move in as one `dma.ub_to_l1`
burst (`vec_mat_row_probe`), moves into a one-row FP32 Bias table map to
`dma.l1_to_bt`, and `matmul_bias` adds that table in `cube.mmad`; other uses
remain gaps.

Cube-side inserts of whole FP32 Acc rows of the Mat's width map to
`dma.l0c_to_l1`. Vec-to-Mat moves of whole equal-shape tiles are one
`dma.ub_to_l1` burst into the Mat's NZ blocks or flat row
(I026). An NZ Vec alias holds
compact fractals of its rows and is admitted only as a paired-side insert
source (`dma.ub_to_l1.nz`).

### Sort records

`sort32` over three distinct Vec tiles maps to `vec.sort32`: FP32 scores,
UINT32 identifiers and two FP32 destination words per score. A one-row
source of 32g scores (g ≤ 255) is one instruction with `repeat` = g, and group k
writes descending (score, identifier bits) records with identifiers 32k..32k+31.
A source of R > 1 rows of 32g scores is one `repeat` = g instruction per row over
`mem.slice` row windows of source and destination, each with the whole identifier
row: A5 restarts every row there, and groups advance along it. An identifier tile
of R rows gives row r its row r (`sort32_id_rows_probe`). Native A5 runs
measured one and three groups on one row and one and two groups on two rows, with
source and identifier tiles unchanged. Agreeing valid extents sort only the valid
rows (`sort32_valid_rows_probe`) or whole groups of the valid width
(`sort32_valid_columns_probe`). Equal scores define no identifier order;
references use distinct scores. UINT32 Vec tiles are admitted only as
identifiers loaded from GM, and UINT32 GM parameters only as their sources or as
SIMT launch storage. FP16 records, tail groups with `tmp`, tile aliases, narrowing
rows and groups together and narrowed per-row identifiers fail at source. A5 read a
sort32 source aliasing its destination's head, and an in-place mrgsort's runs,
before writing (`sort_overlap_probe`); aliased operands still fail at source.

`mrgsort(dst, src, block_len=L)` counts L in storage elements. Over equal one-row
FP32 tiles of valid width W = 4Lr words (L even, r ≤ 255; `mrgsort_valid_probe`)
it maps to `vec.mergesort4` with
`length_per_seq` = L/2 and `repeat` = r: each 4L-word group of `dst` is the
descending merge of the four L-word runs at the same offsets of `src`, which stays
unchanged (A5: L = 32, r = 2 and L = 64, r = 1). Pro binds `mrgsort2` as
(dst, src0, tmp, src1) (A5-UP-040). Two whole one-row
FP32 sources of m and n records with `exhausted=False`, and `dst` and `tmp` of
2(m + n) words, map to `vec.mergesort_2seq(dst, src0, src1)` with `size1` = m and
`size2` = n (A5: 16 + 16, and 16 + 8 in `mrgsort2_unequal_probe`); the sources
stay unchanged. `tmp` is scratch whose contents are
unspecified after the call, and the imported merge neither reads nor writes it;
A5 left a copy of the merged records there. Runs must be descending: a merge is
not a sort. Other widths, tile aliases, `exhausted=True`, three or four sources
and partial records fail at source.

The Pro printer spells the same rule. `vec.mergesort4` with `length_per_seq` = n
and `repeat` = 1 prints one `mrgsort(dst, src, block_len=2n)` over one-row aliases
of the 8n-word footprints. `vec.mergesort_2seq` of two adjacent n-record runs prints
`block_len` = n over the 4n-word footprints, whose four runs are the halves of both
lists. Overlapping footprints merge into a scratch and back, with `block_len` = n
then 2n, or n twice. Odd n where the runs halve lists, `repeat` > 1, a footprint
past its tile and unequal or non-adjacent runs report located gaps (I027).

Audited targets without a Pro source form: GM-to-UB loads are 2-D TLOAD
descriptors without NDDMA strides or pad values (dma.gm_to_ub.nd); Vec-to-Vec
`move`/`ub_copy` print TMOV (Pro fe20d726 `backend_cce_block_out_ops.cpp:543-619,
1045-1049`), which PTO issues as TMovVecToVec, a VF register loop, not a UB DMA
(dma.ub_to_ub). Both refuse at source.

### Slot buffers

A tile group whose cursor advances, or with more than 16 slots and a runtime
selection, is one `buf<P, n>` allocated at its first slot: guarded branches can
neither follow a cursor nor exceed their bound, and Pro CCE indexes one tile
array. Its slots must be contiguous Vec or Mat tiles of one descriptor at the
aligned tile pitch, where CCE and PTO place buffer slots, in a single-side
kernel. Each selection is a `mem.get_buf` whose index snapshot is proven inside
[0, n): the IR index wraps modulo n and Pro's does not. The cursor is an `i64`
cell set to n - 1; `next()` stores its successor before selecting. The `% n` of
`next()`, `current()` and `previous()` needs the nonnegative dividend that P4
storage ranges prove, so a cursor advanced in a branch or nested loop of a
counted loop refuses there. Selections are load, store, move and VF operands.
Other groups keep one allocation per slot and their accepted forms. Aliases and
partial overlaps of slots, other slot operations, L0 and paired-side buffers
fail at source.

### Microscaling matrix products

FP8 E4M3/E5M2 data are whole NZ fractals (16-aligned rows, 32-aligned columns)
under the Mat, Left and Right rules; as for FP32, an NZ or ZN Mat loads with
`order=[1, 0]` (`mx_fp8_transposed`), and a ZN Mat moves only into Left. E8M0
GM parameters are static [rows, groups/2, 2] planes. An E8M0 scale Mat is ZZ
[rows, groups] or NN [groups, rows], typed `l1<e8m0, [rows, groups], nz>`, with
16-aligned rows and even groups. A ZZ load without `order`, or NN load with
`order=[1, 0]`, from `[r, p, 0]` reads the plane row-major: `mem.reshape` to
[rows, 2·pairs], `mem.slice` at [r, 2p], `dma.gm_to_l1.mx_scale_nd2nz`. A5 reads
the other two group-major, [groups/2, rows, 2] (`mx_group_major`,
`mx_zz_order`); no strided view or target load reads that, so both fail at source.

ScaleLeft [M, K/32] and ScaleRight [K/32, N] tiles are declaration-only at their
data tile's address >> 4. An FP8 Mat-to-Left/Right move and the move of its
plane from a whole E8M0 ZZ (Left) or NN (Right) Mat are one `dma.l1_to_l0.mx`
with that Mat as `src_mx`, whose wrapper issues the data then the plane
instruction (RFC-0011 §4.12); the ledger records both moves on it. The plane
move comes next, or the four moves follow Pro's order: Left data, Right data,
Left plane, Right plane. The target then swaps the middle two ST instructions,
which touch disjoint tiles and Mats, and A5 gives the paired product
(`mx_split_moves`). Other orders, or issuing statements between, fail at source.
Either Mat may be a runtime tile-group selection: the pair dispatches over both
(`mx_group_moves`); slot buffers fail at source. `matmul_mx` maps to
`cube.mmad.mx` with `is_init`; `matmul_mx_acc` names one FP32 accumulator twice;
both check each plane's space, shape and address, K % 64 = 0 and whole extents.
E8M0 host tensors are bytes, as Pro passes them. Mixed kernels keep MX
operations on the cube side (`mx_paired`).

A5 scales each K32 group once, (A_g·B_gᵀ)·2^(ea+eb−254), finite where either
side alone overflows FP32 (`mx_scale_extremes`,
I029); code 255 makes its
group NaN, stored as 0x7FFFFFFF (`mx_code_255`). Native Pro runs (seeds 7, 203)
match the seven p6 references and fixed the p7 readings named here. A Final-phase
`matmul_mx` directly followed by its Final-phase store stores the product
(`mx_phase`) through a unit flag, which the target lacks; the import adds no
barrier, so phases fail at source.

A whole scale Mat set to fewer ZZ rows or NN N with every group, loaded once and
set back to its whole shape later in the same block, loads `rows` = those rows;
statements that do not name the Mat may come between (`mx_cut_statements`).
Pro's `TLOAD` and the wrapper issue one DN builtin that writes code 0 in the rest
of its last 16-row box and leaves later boxes stale (`mx_tail_code`,
`mx_cut_boxes`, `mx_nn_tail`), from any `[r, p, 0]` for ZZ (`mx_tail_offsets`).
NN loads leaving a whole box or off the origin, other valid shapes on MX tiles
and FP4 fail at source.

Group terms add in ascending order to zero, or for `matmul_mx_acc` to the prior
accumulator; a subnormal partial sum flushes to signed zero (`mx_subnormal_sums`,
`mx_acc_flush`, I030).
Sums truncate (I034): products
24 bits below a group's largest vanish wherever they sit (`mx_element_rounding`),
as do term bits below the partial sum's 24-bit window (`mx_group_rounding`). The
model truncates every addend and carry to the larger addend's 24-bit window; how
opposite signs and carries round is unmeasured.

### NZ-packed GM parameters

A `pl.NZ` tensor [R, C] of FP16 or INT8 holds ceil(C/C0) column blocks of
align16(R) rows of C0 = 32/bytes elements: element (r, c) is storage element
(c//C0)·align16(R)·C0 + r·C0 + c%C0. Native A5 runs read that layout from
format-29 tensors, whose storage has that size, and from ND views of the same
bytes (`nz_gm_probe`, `nz_padded_probe`, INT8 `nz_quant_store_probe`). A static
tensor imports as that storage, `gm<T, [ceil(C/C0)·align16(R), C0]>`, with [R, C]
in `meta.pro_nz_parameters`; native runs take its logical-shape view, and pointers
address storage elements (`nz_view_probe`). Transfers take static windows of whole
column blocks inside the storage, reading its padding as stored (`nz_padding_probe`),
as `mem.slice`s of storage rows. A load into an FP16 NZ Mat [16k, 16m] of valid
[h, 16w] is one `dma.gm_to_l1` of w bursts of h rows, source stride align16(R) − h
and destination stride 16k − h. Native Pro and imported CCE runs on A5 match whole
Mats for w = 2, k = 1 (both probes), w = 3 (`nz_wide_load`) and k = 2
(`nz_tall_load`), seeds 7 and 203 (I018);
valid rows land at the Mat's pitch and valid columns take whole blocks, each
measured with the other axis whole (`nz_partial_load_probe`). A whole 16-aligned
FP16 NZ Vec alias holds its blocks as contiguous bytes, so its loads and stores are
w bursts of 32h bytes with GM stride 32(align16(R) − h), `dma.gm_to_ub.pad` and
`dma.ub_to_gm.pad` (`nz_vec_probe`). A scaled Acc store of 32-element blocks
into INT8 is `dma.l0c_to_gm.nz2nz` with `scale`: A5 multiplies, rounds half to even
and saturates (`nz_quant_store_probe`, one block at column 32), and block k starts
k·align16(R)·32 elements after the first (`nz_int8_blocks_probe`, three blocks at
column 0; I039). Pro refuses several
blocks from fewer than align16(R) Acc rows (`ValidateNZTransfer`; `nz_int8_strip_probe`
fails natively), and so does import. A store of several blocks from a later column
is unmeasured and fails at source. A5 stores FP32 NZ tensors as channel-split 8-element
blocks (`nz_store_probe`), which no IR store selects: they fail at the parameter.
Other dtypes, runtime shapes, narrowing both axes, relu or offset riders,
transposed, slot-buffer or run-time selected transfers and other uses fail at
source. PTO ISA has no printer for plain `dma.gm_to_l1`.

### SIMT FP32 math

In SIMT functions, `simt.exp`, `exp2`, `log1p`, `sin`, `cos`, `tanh` and `rsqrt`
of one FP32 operand map to the same target operation, whose CCE shim prints
Pro's A5 formula. A literal operand becomes an FP32 `scalar.const`. Pro prints
`log` as `(x > 0 && x < FLT_MIN) ? log(exp(23) * x) - 23 : log(x)` and `log2`
as that value divided by `log(2)`; the forward shim spells both differently.
Import expands them into `scalar.cmp`, `scalar.and`, `simt.exp(23.0)`,
`scalar.mul`, `simt.log`, `scalar.sub` and `scalar.select`, then `scalar.div`
by `simt.log(2.0)`. Imported CCE prints Pro's operations in Pro's order and
constants, one statement each; no product feeds an addition.

The formulas approximate the functions that models evaluate (RFC-0001 §6.13).
exp2 is inexact at integers from |x| = 13, log2 of a power of two need not be
integral, and log1p and tanh return 0 on [-2^-25, 2^-24]. References allow
1e-6 + 1e-5·|f(x)| for finite values and require equal infinities and NaN
positions; native and imported CCE outputs are compared bitwise. A later
rounding, comparison or cast can branch differently in the model than on A5.

Pro prints integral rounding as raw A5 builtins. Measured on A5, every zero
result is +0, including those of -0 and of operands in (-1, 0), and literal
operands give their runtime results; ties follow IEEE, and large magnitudes and
infinities are unchanged. The target operations keep the operand's zero sign
(RFC-0001 §6.3), so import follows each call with `scalar.cmp` against zero and
`scalar.select` of +0; nonzero, infinite and NaN results are unchanged. A5
returned every measured NaN result, elementary or integral, as 0x7fffffff,
whatever the operand's sign or payload. Policy switches refuse SIMT rounding or
its literal operands instead. FP16/BF16 variants have no target: SIMT storage
refuses them.

### SIMT atomics and exact math

`atomic_<op>(target, operands)` becomes `simt.atomic { op = <op> }` on one
element of a SIMT storage parameter; its row-major offset must be proven inside
the storage. Operands must have the element dtype, and literals Pro types to it
must keep their value. The call's type must be the element dtype, and `old` is
the element before the update. Pro's CAS `(compare, value)` becomes operands
`(value, compare)`; both printers call `atomicCAS(ptr, compare, value)`. INT32
GM tensors and Vec tiles admit add, sub, exch, max, min, and, or, xor and cas;
UINT32 GM tensors also admit inc and dec; FP32 storage admits add, sub, exch,
max, min and cas. UINT32 GM kernel parameters are admitted only as whole SIMT
launch arguments or as GM sources of sort32 identifier loads. Threads share an element only through atomics; a result that
depends on thread order is unspecified. FP16/BF16 atomics, which return no prior
value, INT64/UINT64 elements and UINT32 tiles are refused.

`mul_hi` admits two INT32 or two UINT32 operands and keeps their dtype; `fma`
and `fmod` admit FP32 operands. `isnan`, `isinf` and `isfinite` on FP32 become
`simt.is* : i32` compared `ne 0`; Pro keeps the builtin's raw value, which is 0
or 1 on A5. RFC-0001 §6.14-§6.15 give the value semantics.

### SPR masks, register reductions and interleaved transfers

`set_mask_norm`, `set_vec_mask(high, low)` and `reset_mask` map to `vec.set_mask_normal`,
`vec.set_mask` and `vec.reset_mask` at the top level of the vector section; writes need an
earlier `set_mask_norm`. Operands are INT32/INT64 scalars or integer literals, converted to
uint64 as in C++. Counter mode is refused: no c310 instruction uses it. `get_mask_spr` maps to
`vf.mask` plus `vf.mask_from_spr` (B32: MASK0 bit i to bits 4i..4i+3; B16: MASK0 then MASK1, two
bits each). A read sits at the top level of a VF section called at the top level, after a write.
A5 probes showed program order without barriers; pset, plt, pand, movp, vcmp_lt, vlds, psts, GM
stores and launch boundaries left the SPR unchanged. Other predicate producers, tile loads,
scalar tile access, tile helpers and SIMT launches between the write and the read are refused:
their effect is unmeasured.

`reduce_max/min` map to `vf.cmax/cmin` with `index = true`, and `datablock=True` reductions to
`vf.cgadd/cgmax/cgmin`, on matching FP32/INT32 registers under b32 ZEROING. A5 writes the
extremum to lane 0 and its first active lane to lane 1, or group g (8 lanes) to lane g. Empty
selections give 0 for sums and otherwise the dtype's lowest or highest value with index 0; an
active NaN gives 0x7FFFFFFF and its lane, and -0 orders below +0. A four-operand `load_align`
with absent or width-matching DINTLV dist maps to `vf.load_interleave` at a static aligned
element offset of a whole tile (A5 ran FP16 at 0, 32 and 128). `store_align(tile, even,
odd, pred)` maps to `vf.store_interleave` at offset 0 of a whole tile of at least 128 FP32/INT32
elements (INTLV or INTLV_B32) or 256 FP16 elements (INTLV, printed INTLV_B16). A5 wrote every pair
whatever the predicate: b32 or b16 on 32-bit registers (`interleave_fine_mask`), b16 on FP16
(`interleave_half`); other widths are unmeasured. Tiles match the register dtype;
block-copy/post-update attributes are refused. Every admission keeps an importer switch. BF16
and runtime offsets remain gaps.

### Register rearrangement

Unmasked register `vf.move` copies all 256 bytes for FP32/INT32/FP16/BF16.
Masked moves and explicit mode attributes remain gaps: native masked move is
MERGING, whereas the target masked copy is ZEROING.

`vf.bit_cast` admits the same-width FP32↔INT32 and FP16↔BF16 pairs, and INT32
to or from the UINT8/UINT16/UINT32 carriers of indexed register access. Preserve
all bits, including NaN payloads and signed zero. Native destination assignment
and materialized `auto` expressions copy values; target reinterpret is a byte
alias, so import emits a reinterpret followed by a full copy into the explicit
or temporary destination. Later source/destination writes must not affect the
saved value. Direct source register alias bindings remain unadmitted.

Interleave and de_interleave admit four matching single-register operands of
these four types. Destinations must be distinct from each other and the sources;
mask-register forms and dtype reinterpretation are rejected. Interleave zips
source streams and splits the result at one register; deinterleave concatenates
the two input streams, then extracts even and odd positions into separate outputs.
Two-output order, exact payloads and copy lifetime are checked independently.

### Increasing register indices

`vf.arange` admits single FP32/INT32 registers and matching scalar start values
(or numeric literals; INT32 literals must fit the signed domain). Its full
register ramp ignores execution predicates. Only absent/INCREASE_ORDER is
admitted; dtype overrides must match the destination. INT32 addition follows
32-bit wraparound. Decreasing order, smaller widths and implicit scalar
narrowing are not inferred from the native entry point.

### Indexed register access and unsigned carriers

UINT8, UINT16 and UINT32 registers and b8 predicates are VF-local carriers made
by declarations, `update_mask` counts and INT32 `bit_cast` views. A cross-width
view keeps all 256 bytes: lane k of a W-bit view is bytes [kW/8, (k+1)W/8),
little-endian. Tiles and GM tensors keep their element types, and carriers never
load or store directly.

Register `gather(src, index)` maps to `vf.gather` for FP32/INT32 data with
INT32/UINT32 indices, FP16 with UINT16 and UINT8 with UINT8. Whole-tile
`gather(tile, index, pred)` maps to `vf.gather_copy` for FP32/UINT32 and
FP16/UINT16; DATA_BLOCK_LOAD maps to `vf.gatherb`, whose UINT32 index lane b is
the byte offset of destination block b. `scatter(tile, src, index, pred)` maps to
`vf.scatter_copy` for the same pairs. A 32-byte-aligned literal `tile + k` moves
their base k elements (A5 runs). `squeeze(src, pred,
gather_mode=NO_STORE_REG)` maps to `vf.squeeze` without store for matching FP32,
FP16, UINT32 and UINT8 registers, and `unsqueeze(pred)` to `vf.unsqueeze` for
INT32 and the three carriers. `pack` maps INT32/UINT32 to UINT16 and UINT16 to
UINT8, LOWER to `lowest` and UPPER to `highest`. `histograms` needs a UINT8
source, a b8 predicate and a UINT16 destination written earlier; absent BinType
and HistType are BIN0 and ACCUMULATE, and import always spells the mode. Tiles
match the register dtype, predicates its lane width, and no destination aliases
its source or index.

A5 runs of these forms, native and imported, established the model rules.
gather_copy zeroes inactive lanes. gatherb reads block b whole when the
predicate bit of UINT32 index lane b is set (b32 lane b, b16 lane 2b) and zeroes
other blocks (I014). Scatter validates every active index before
writing, and masked lanes never write. A5 leaves the surviving duplicate writer
unspecified, so active duplicates must carry identical payload bits
(I015). squeeze zeroes its tail; unsqueeze writes, on every lane, the
number of active lanes before it; pack zeroes the half it does not fill.
ACCUMULATE adds the number of active values at most 128*bin+j, including values
below the BIN1 base as radix TopK measures; FREQUENCY adds exact matches; counts
wrap modulo 2^16. A5 read register gather indices modulo the source lanes for
FP32/INT32 data with INT32/UINT32 indices and FP16 with UINT16, as RFC-0001 §4.4
now defines. Pro's register gather requires an index as wide as its source.

Refuse the default STORED squeeze (it also writes the AR SPR), vgather2_bc (FP16
with UINT32 indices), b8 widening, DATA_BLOCK_COPY, INT32 scatter (A5-UP-037),
mask or 64-bit packs and other dtypes. The only admitted UINT16 broadcast is
Pro's histogram idiom `full(0, pred)` with a b16 predicate; A5 runs compiled it
and zeroed the register.

### P5 runtime entry

`import_kernel(bundle)` wraps verified Lowered IR as an entry for existing
runtime tools. Lowered entries are checked and never run the Surface pipeline.
`entry.executor(...)` fixes device/block_dim to the exported specialization and
seeds outputs when any direction is inout; disabling that seed is rejected.
The entry records the Pro kernel definition, where a reserved entry name is refused (RFC-0007 §1).
Tensor order, dtype and shape remain checked by runtime binding.
Native Pro and imported device runs use the same
runtime-generated inputs/reference and report separate compilation/run gates.

### Private GM workspace parameters

Pro passes scratch GM as ordinary Tensor parameters. No private ownership is
inferred from names. `export_kernel(workspace=[...])` explicitly declares each
private parameter with `parameter`, two-dimensional `shape`, `alignment=32` and
`init="uninitialized"`. A shape axis is a positive literal or
`{"tensor": "public_parameter", "axis": 0 or 1}`; workspace-to-workspace references
are rejected. Static source dimensions must agree; dynamic source dimensions
are bound by this declared allocation contract. Source direction must be inout.

Initially admit a single vector participant (`block_dim=1`). Private parameters
leave the public signature and become named `mem.workspace` values. Declaration
order defines distinct 32-byte-aligned byte ranges; no ordinary allocation pass
runs. Public shape scalars size the ranges, with checked signed-size arithmetic
at binding. Source uses of private shape variables resolve to the declared sizes.

The existing launcher allocates the reported user workspace plus CANN system
workspace, synchronizes execution, then frees it. Contents are unspecified on
every call: each read needs a prior source write and synchronization; no zero-fill
or persistent-state semantics are invented. Models poison scratch to expose
unwritten reads. Native comparison passes explicitly allocated scratch tensors.
Aliases with public tensors, custom initializers and cross-participant workspace
protocols remain gaps.

### Dynamic dimensions and bounded loop transfers

Pro INDEX shape variables become `DimValue` references to appended i64 parameters,
inferred from tensor shapes, after explicit scalars. Dimensions must be positive;
STATIC dimensions stay constants. Distinct axes keep distinct names; equal sizes
do not imply an equality contract.

Local capacity and pitch stay static. Straight-line UB valid-shape updates admit
positive extents bounded by capacity and GM dimensions, including nested minima.
Each update captures its scalar value;
DMA attributes and slice extents reference that snapshot. Single-row transfers
use zero row strides, allowing an unaligned valid byte count within an aligned
allocation. Multi-row transfers retain an exactly representable fixed UB pitch.
A load row ending inside a 32-byte block fills the rest of that block with the
row's first element (`gm_view_skew_probe`, `ub_row_tail_probe`,
I038); such multi-row stores remain gaps.
Also admit tiled loops `i in [0, (N-1)//B+1)` for positive N and fixed positive B.
The matched offset `i*B` is nonnegative and strictly below N without signed
overflow; `min(N-offset, B)` bounds each transfer. N may be a minimum of distinct
GM dimensions. Nonzero windows require this proved residual relation.

Loop-local UB valid-shape updates must dominate consumers on every iteration.
Branches and loops have isolated descriptor state; a possibly updated descriptor
is unknown at the join/exit until explicitly reset. No last-iteration value is
guessed, including for empty loops. Unknown owners, unproved extents/offsets,
general layouts and zero dimensions fail explicitly. Allocation and synchronization
remain unchanged; no shape specialization is introduced.

### P4 paired sides

Mixed import requires identical parameter ABIs. Exported declaration keys,
rechecked against source names/locations, link storage across sides; matched
types, addresses and extents retain one target identity. Foreign descriptors
require an owner-side declaration. UB remains private to each AIV participant.
Admit full FP32 L0C-to-UB split-M and FP16 UB-ND-to-L1-NZ inserts with proven
bounds. Preserve static cross-core flags in mode 2 (one AIC and both AIVs),
including source credit seeds/drains; collectives follow their own section.
Dynamic cross IDs stay rejected. Raw local flags retain their channels and IDs.
No handoff protocol is invented.

### P4 local synchronization and storage

Preserve Pro `system.mutex_lock_dyn/unlock_dyn` as explicit A5 mode-zero local
get/release operations. Snapshot ID expressions before emitting locks. For
multiple IDs, preserve first-occurrence order and the native inequality guards
across distinct owners; IDs belonging to one owner must be provably distinct.
Candidate metadata is checked, never interpreted as a reason to renumber IDs.
Do not reproduce native V-only lock elision: the exported IR is the import
boundary. Existing source barriers remain separate instructions. Invalid
executed lock sequences fail model validation; no autosync planner repairs them.
Bounded runtime tuple selection preserves each physical slot, including gaps.
An integer range proof must establish an in-range index; no implicit modulo is
inserted. Scalar columns use selects; tile consumers use guarded branches with
one executed alternative, with at most 16 combinations. Selection indices are
snapshots. A reassigned integer scalar has a range only at statements where it
is proven: an assignment gives its value's range and a branch joins its arms.
A constant-bound counted loop without break or continue that steps the scalar
once, by a constant S at the top level of its body, holds entry + S·k before
step k and leaves entry + S·trips. Other reassignments give no range. Static
UB subviews with fixed parents share backing and retain row pitch; they are
not allocations.
Unbounded indices, cursor mutation outside slot buffers, general descriptor
mutation and overlapping allocations other than admitted tile aliases remain
gaps. Cross-side handoffs have a separate gate.

### P3 instruction conversion

`import_module(bundle)` returns a fully verified `lowered/1` Module or a located
gap. Initial admission covers one explicit active side, fixed two-dimensional
GM/local storage, constant tuple/slot selection, rectangular transfers, scalar
cells with counted loops/if, outlined VF sections and basic FP16/FP32 matmul.
Source attributes are checked per conversion rule: a known opcode alone is not
admission. Unsupported layouts, extents, modes and unbounded memory selections,
while/early return fail before a module is returned.
Physical bank alignment, capacity and non-overlap are checked without allocating
new addresses. Overlapping declarations other than admitted tile aliases remain
gaps; admitted subviews and aliases retain parent backing. Pro BOOL is logical
`b1`, not a GM byte-type conversion. FP32 literals are rounded before use and
scalar snapshots use an identity cast so negative zero is preserved. Inout runs
retain supplied tensors.

Preserve snapshots, mutation, signedness and width; runtime descriptor changes
are not compile-time state. Division/remainder need a proven domain or matching
rounding. VF captures expose read/write effects; defining operations distinguish
registers from predicates. Translate Right coordinates to `[N,K]` exactly once.

No autosync, address planning or instruction scheduling runs during import.
Every emitted operation has a source location and import origin, and a module
ledger accounts for translated and declaration-only source operations. Complete
verification and text/JSON round trips precede functional and pipe simulation.
Hazards in a faithfully imported source program remain visible; numerical
equality alone is not a synchronization pass. Inputs/references are generated at
test time. Source compilation and device execution are separate evidence gates.

### P2 transport and preparation

`ascriptor.importers.pypto_pro` is experimental. Importing it requires only the
standard library; `export_kernel` loads Pro on demand. `loads` validates an
`ascriptor.pypto-pro-export/1` document; `prepare_import` constructs a source
operation ledger, never a partially valid Ascriptor Module. Missing conversion
rules raise a source-located diagnostic rather than silently dropping an op.

Transport preserves producer identity, specialization, launch/directions and
ordered typed per-side graphs. Node references and declaration identities are
explicit; allocation identity differs from physical address. JSON preserves
operands, attributes, types, views and synchronization. Unknown forms fail closed.

The pinned Pro adapter adds a per-call parser observer and a read-only native
attribute reader. The existing `Call.kwargs` binding omits list-valued metadata;
it is not a valid source for this transport. No process-global parser patching
or shared-package mutation is permitted. Build the helper in an isolated overlay
against the selected Pro headers/native ABI, recording its binary fingerprint.
Run requested Pro pipeline preparation before parsing; preserve source mutex
configuration. Missing producer compatibility or metadata must fail explicitly.
Export sets Pro's parse-time `PYPTOPRO_JIT_ARCH`/`PYPTOPRO_NPU_ARCH` as its jit
does, then restores the caller's values ([I017](../../ascriptor/importers/pypto_pro/export.py));
ZZ Left, Pro's A3 default, fails at source.

The compatibility profile pins Pro files, the native helper and the import
attribute whitelist; its fingerprint covers all three. Profile /2 whitelists
`mem_bar(mode)` and `store_unalign_post(post_update)`, which Pro's emitters read
although its registry declares no attributes ([A5-UP-041](../upstream.md#a5-up-041)).
Export reads only the pins and the fingerprint, never the whitelist, so a /1
export holds the content of a /2 export. Import accepts the /2 identity and /1,
whose fingerprint the profile must reproduce from /2 without the added
attributes; any other identity fails. Device runners compare a fresh export
with a /1 fixture except for that identity.

P2 checks repeated exports, round trips, manual/auto graphs, physical IDs,
provenance, malformed inputs and native list metadata. It proves neither
Lowered conversion nor hardware. Field definitions belong to
`ascriptor/importers/pypto_pro/schema.py`.

Compare effective accesses, identities and synchronization on matching paths
and participants. Qualify Pro native, simulation and backend execution separately.

Support claims name revisions, device, backend, operation forms, cases and gates.
Initial scope is A5. Detailed milestones and temporary reports belong in ignored
`tmp/pypto-pro-import/`. Any Pro export changes use a separate development worktree.
