# PyPTO Pro mapping, as built

This mapping reference describes instruction spellings and protocols from
`ascriptor/backends/pypto_pro/emit.py`. The [unified upstream report](upstream.md)
owns issue IDs, current evidence and the disposition of historical gaps.
The [historical survey](pypto-pro-coverage.md) locates the original investigation;
its measurements are not fresh successor qualification. Use the
[generated support table](pypto-pro-support.md) for emitted opcode coverage.
The September 7 reconciliation corrects the stale ND-to-NZ, Acc-to-Mat,
strided-offset and scalar-access entries below.

The default `sync_mode="auto_mutex"` emits `@pl.jit(auto_mutex=True)` and
copies Lowered IR allocation IDs into `pl.make_tile_group(..., mutex_ids=...)`.
IR local mutex operations emit no Python statements or comments. The backend preserves all other
explicit synchronization and does not allocate IDs or select event protocols.
`sync_mode="manual"` retains explicit local mutex calls for comparison. See
[RFC-0013](rfc/0013-pypto-native-synchronization.md).

The instruction mappings below describe explicit IR operations. In manual mode,
`pl.system.*` calls are IR decisions and PyPTO inserts no local synchronization.

## 1. Instruction mapping

`vec.mergesort4` is explicitly rejected in both synchronization modes; its former
multi-pass expansion and backend-only UB scratch allocation have been removed.
`vec.mergesort_2seq` still maps equal-length, adjacent runs to one `pl.mrgsort`
call. See the [sorting subset](rfc/0013-pypto-native-synchronization.md#sorting-subset).

### 1.1 DMA

> A rank > 2 strided GM read has no pl spelling and is UNROLLED: `pl.make_tensor`
> builds a 2-D `TileShape2D` view whatever the shape list holds, and `pl.load` overwrites
> the last two dims from the destination tile, so the printer emits one 2-D `pl.load` per
> outer index into its own sub-tile (`_nd_unroll`, D-131). A non-unit INNERMOST stride is a
> different matter and stays a gap - pto's inner dimension is a contiguous burst (D-130).

| ascriptor (lowered) | PyPTO Pro spelling | notes |
|---|---|---|
| `dma.gm_to_ub[.nd/.pad]`, `dma.gm_to_l1[.nd2nz/.pad]` | `pl.load(tile, tensor, offs)` | layout/pad ride the tile declaration, not the op. A contiguous GM run filling a `[N, 1]` COLUMN window loads through `pl.make_tensor(param, dims + [1])` at `offs + [0]` (behaviour #19); an arbitrary pad VALUE (`constant_value`, `nearest_value_mode`, the NDDMA edge pads) has no pl channel at all (behaviour #20) |
| `dma.gm_to_l1.dn2nz` (transposed B) | `pl.load(..., order=[rank-1, rank-2])` with the B chain declared `[K, N]` | variant-C coordinates, D-087. `order` is an AXIS SELECTION, not a transpose flag: one entry per TILE dimension, each naming a TENSOR axis (default = the tensor's last N axes), and the list is always 2 long (a 3-entry list is an `IndexError` in the parser). A transposed 2-D load is therefore the last two axes named in REVERSE, so `[1, 0]` is right only at rank 2 - on a rank-3 `[B, S, D]` tensor it reads `(S, BATCH)`: board 507015 when the stride leaves the allocation, silent garbage when it stays inside (D-102) |
| `dma.ub_to_gm[.pad]` | `pl.store(tensor, tile, offs, atomic=...)` | |
| `dma.l0c_to_gm.nz2nd` | `pl.store(tensor, acc_tile, offs, scale=..., relu_pre_mode=...)` | quant scale: f32 → plain float; scale+offset → the packed-INT64 ride-along (behaviour #10) |
| `dma.l0c_to_ub` | `pl.move(vec_tile, acc_tile, acc_to_vec_mode=..., scale=...)` | `DualModeSplitM/N`, `SingleModeVec0/1` from our dual_mode/sub_block attrs. Our `scale` attr is pl's per-tensor deqScalar `scale=<float>` (D-127); a runtime or per-channel scale is refused with what it would need, and so is a scale that arrives WITH a source window - the offset form lowers to `TEXTRACT`, which drops the scale (D-128) |
| `dma.l1_to_l0` | `pl.move(l0_view, l1_tile)` | equal shapes both sides; the on-chip transpose is a DECLARATION property, not an op form (behaviour #8) |
| `dma.ub_to_l1` | `pl.move(l1_tile, ub_tile)` | plain supported transfer |
| `dma.ub_to_l1.nd2nz` | one `pl.insert` per C0-wide source/Mat alias | `_ub_nd2nz` composes ND rows into NZ columns; preserve source pitch and restore narrowed validshape before later reads (M10-063). The historical blanket absence is superseded. |
| `dma.ub_to_l1.nz` (compact-NZ publish) | `pl.insert(l1_base, nz_alias, [r, c])` | the source is re-declared as an NZ alias of the same address; windows add `pl.set_validshape` (behaviour #7) |
| `dma.l1_to_bt` | `pl.move(bias_tile, l1_row)` into `MemorySpace.Bias` | the [1, N] L1 row declares NO layout (pto self-lowers ColMajor; explicit ND is refused on board), D-090 |
| `mem.reinterpret` of a GM PARAMETER | `pl.make_tensor(param, shape, strides, dtype=...)` | `pl.reinterpret` is tile-level, but `make_tensor` carries a `dtype=` of its own - the same re-description `mem.view`/`mem.workspace` use, with the element type changed instead of the strides. This is how a byte-carrier parameter (torch has no hif8/fp4) is re-typed, D-107 |
| `dma.set_constant_to_l1` (whole tile) | `pl.expands(tile, val)` + `pl.system.bar_mte2()` | Mat fills retain IR mutex IDs; native auto-mutex selects MTE2 for a Mat destination ([A5-UP-038](upstream.md#a5-up-038), fixed upstream in `6a652e733`). Float8 zero fills use a same-width integer view; D-124. |
| `dma.l0c_to_l1` | `pl.insert(mat, acc, [r, c])` | `pl.move` lacks Acc→Mat, but `pl.insert` reaches it (D-116). Whole-row FP32 is supported; the distinct PTO partial-row FP32 restriction is [A5-UP-007](upstream.md#a5-up-007). |
| `dma.gm_to_l1.mx_scale_nd2nz` (dense scale) | `pl.load(scale_tile, tensor, [r, c, 0], order=[0, 1] / [1, 0])` | the GM scale reaches pto as RANK 3 `[rows, k_pairs, 2]`, the trailing physical-phase axis statically 2 ("MX scale load requires at least two matrix axes and one physical phase axis"). Same bytes as our `[rows, k_groups]`; the driver reshapes (behaviour #16) |
| `dma.gm_to_l1.pad` into a scale tile (packed-block scale) | `pl.load(u8_alias, tensor, offs)` through a one-row byte alias | GM already holds the packed 32-byte blocks, so the fill is one flat burst; the fractal tile is the wrong window for it |
| `dma.l1_to_l0.mx` (scale half) | `pl.move(scale_l0, scale_l1)` into `ScaleLeft`/`ScaleRight` | the L0 scale tile's address is NOT free: `addr(scale) = addr(data_tile) >> 4` (behaviour #16) |
| `dma.l1_to_l0.img2col` | gap | upstream absence |

### 1.2 Cube

| ascriptor | PyPTO Pro | notes |
|---|---|---|
| `cube.mmad` (is_init) | `pl.matmul(acc, a, b)` | rhs declared `[K, N]`, loaded `order=[1, 0]` |
| `cube.mmad` (accumulating) | `pl.matmul_acc(acc, acc, a, b)` | |
| `cube.mmad` + bias | `pl.matmul(acc, a, b, bias_tile)` | init only - `matmul_acc` has no bias channel (gap when combined), D-090 |
| `cube.mmad.mx` | `pl.matmul_mx(acc, a, b, scale_a, scale_b)` | the scales are EXPLICIT operands read from their own stops - our IR hangs them on the `l1_to_l0.mx` move instead (behaviour #16) |
| `cube.mmad` sub-block dst offsets | gap | `dst_row0/col0 != 0` is a later phase |

### 1.3 VF (register lane)

Register groups are a located gap (D-230). The inspected PyPTO Pro VF API has no
native register-group type or register-count parameter: `load_align`, `astype`,
`arange` and the arithmetic APIs implicitly produce one register. This backend
refuses `reg<T, 2>` before emitting a declaration or use, including load/store-only
and cast-only functions. Grouped masks are also refused until their doubled lane
extent has an explicit mapping; the dtype-only mask spelling must not discard
`MaskType.n`. The check includes result types, operands and mask attributes.
`vf.mod` has no upstream VF remainder API and raises an upstream gap for either
register count. Existing single-register i64/u64 arithmetic and complex dtype
gaps remain in force. Int64 arange and casts with an i64/u64 register on either
side also refuse for one-register requests. The cast diagnostic preserves D-221's
board-measured wrong-lane evidence; arange has no native VF arithmetic carrier.
No lo/hi arithmetic emulation is introduced.

Checked 2026-09-05 against the local upstream mirror's
`python/pypto_pro/language/_vf_api.py`. All six register-group samples now refuse
at `vf.reg`; before the guard, `groups_cast` incorrectly emitted a complete file.
The backend tests also cover isolated uses, mask results/operands/attributes,
single-register remainder and the existing single-register smoke.

| ascriptor | PyPTO Pro | notes |
|---|---|---|
| `vf.reg` / `vf.mask` | name binding / `vf.create_mask(pattern=..., dtype=MASK_W_DT[width])` | the dtype picks the LANE WIDTH; the b32 fallback was the D-093 mask bug |
| `vf.load_cont` / `vf.store_cont` | `vf.load_align/store_align(tile, off, dist=...)` | `LoadDist`/`StoreDist` from our dist attr (UNPK/PK/BRC families; `pack_b64` via a `bit_cast` to b32 first) |
| `vf.load` / `vf.store` (strided, vsstb/vsldb) | `vf.load_align(base + offset, ...)` / `vf.store_align(base + offset, reg, mask, ...)` with `DATA_BLOCK_COPY` and `block_stride` | D-116 supports pointer arithmetic, including runtime origins. The old cursor-walk-only restriction is retracted; unaligned cursor initialization is separately [A5-UP-027](upstream.md#a5-up-027). A stride-1 load whose predicate may split a block prints `vf.load_align(tile, off)` and a zeroing `vf.and_` (I036). |
| `vf.unalign` family | `vf.load_unalign_pre/load_unalign/store_unalign/store_unalign_post` | cursor semantics shared with cce |
| binary/unary/scalar arith | `vf.add/sub/mul/div/max/min/and/or/xor/...`, `vf.abs/exp/sqrt/neg/...`, `vf.muls/adds/mins/maxs/...` | tables VF_BINARY/VF_UNARY/VF_SCALAR in emit.py |
| `vf.cgadd/cgmax/cgmin` | `vf.reduce_sum/reduce_max/reduce_min` | group reduces |
| `vf.cmp/cmps/select` | `vf.cmp(..., mode)` / `vf.select` | pto types `select` ("only supports BOOL/INT8/UINT8/INT16/UINT16/FP16/BF16/INT32/UINT32/FP32") though it is a per-lane bit mux, so an 8-bit FLOAT operand goes through its uint8 carrier - bit_cast in, select, bit_cast back, same lanes and same mask (D-127) |
| `vf.cast` | `vf.astype(src, mask, dtype=..., round_mode=..., layout=...)` | ROUND table covers rint/round/floor/ceil/trunc/odd/hybrid; CAST_LAYOUT only `zero` (pto's ONE/TWO/THREE are register quarter selectors we do not use). Gaps: f16↔hif8 hybrid/odd (upstream refusal), bf16→f16 (pto's vcvt printer drops the RS argument), both D-093 |
| `vf.deinterleave/interleave` | `vf.de_interleave/interleave` | no 8-bit form upstream (gap for b8) |
| `vf.gather_copy/gatherb/gather/scatter_copy` | `vf.gather/gatherb (DATA_BLOCK_LOAD)/gather/scatter` | |
| `vf.mask_to_ub / ub_to_mask` | `vf.store_align(tile + off, mask)` / `mask = vf.load_align(tile + off)` | the predicate spill/fill pair (`psts` / `plds`), reached as the MaskReg OVERLOADS of the aligned move rather than as their own APIs - pto removed its `vf.mask_store/mask_load` ops in favour of this dispatch (`ir/op/vf_ops.cpp`), which routes on `IsMaskRegVar`. The store takes NO predicate argument; the default dist is NORM on both, the VL/8 = 32-byte physical image cce prints. Two traps: `psts` is emitted with a literal `0` offset unless `post_update` is set, so an offset passed as an ARGUMENT is dropped silently - displace the pointer instead; and `load_align` is a UNIFIED op whose FRESH destination takes its kind from the SOURCES, so a mask name pto does not already know lowers to `vlds` and loads data where predicates belong, also silently (M10-075) |
| `vf.mask_update / mask_from_spr` | `vf.update_mask(cnt, dtype=MASK_W_DT[width])` / `vf.get_mask_spr` | the dtype picks the predicate's LANE WIDTH and pl DEFAULTS IT TO 32 when unstated - `plt_b32` where cce prints `plt_b16`, which leaves every odd 16-bit lane false and silently skips half the work the mask guards (D-129). Always printed; an unknown width is a gap. pl copies the count, so a cell read later gets a printed `c - pl.min(c, n)` decrement (I040) |
| `vf.barrier` | `vf.mem_bar(mode=pl.MemBarMode.<TAG>)` | `MEM_BAR_MODE` prints the tag of the `(src, dst)` pair - the twelve AscendC MemType combinations, VST_VLD for `vec_store -> vec_load`; any other pair is a gap. Import inverts the same table. A5 ran all twelve with matching bits, but no tag showed an ordering effect and the scalar tags did not order an S-pipe access against the VF at all (RFC-0015) |
| `vf.pack/unsqueeze/dup/arange/reinterpret` | `vf.pack/unsqueeze/full/arange/bit_cast` | `vf.arange` carries its start in the attr **`v`** (not `start`, not a second operand) and its direction in `mode` -> `index_order=pl.IndexOrder.DECREASE_ORDER`. Reading the wrong attr fails SILENTLY (every ramp starts at 0) and survived a whole board sweep before flash_attn's causal mask exposed it (D-102) |

### 1.4 Vec block level / SPR

`vec.set_mask/set_mask_by_count/set_mask_count/reset_mask` → the SPR spellings
(`pl.set_vec_mask` family); the mask ALGEBRA (`mask_and/or/not/xor`, `mask_sel`,
`mask_mov`, `mask_pack/unpack`, `mask_interleave/deinterleave`) has no pto surface (gap).
The SPILL pair is not part of that gap and never was: `mask_to_ub` / `ub_to_mask` map to
`store_align` / `load_align` (the row above, M10-075). The earlier reading that "masks are
not first-class there" came from looking for mask-named APIs; pto carries the predicate
moves as overloads of the ordinary ones instead, so a name search misses them.
`core.set_sat_flag/get_sat_flag` → the saturation SPR pair.

### 1.5 Scalar, control flow, core ids

| ascriptor | PyPTO Pro | notes |
|---|---|---|
| `scalar.*` supported arithmetic | Python operators and `pl.min/max`, folded when explicitly bound | Dynamic values remain runtime scalars. Ordinary scalar abs/sqrt/cast are separate [A5-UP-001/002/003](upstream.md#a5-up-001); VF and SIMT APIs do not supply those ordinary forms. |
| `cf.for` | `for v in pl.range(lo, hi, step):` | constant trips unroll at parse time |
| `cf.if` | `if cond:` | |
| `cf.call` (vf) | direct call of the `@pl.vector_function` | |
| `core.cube_idx` | `pl.get_block_idx()` on the cube side; `(pl.get_block_idx() // pl.get_subblock_num())` on the vec side | D-088: an undivided cube id dispatches one AIC's two AIVs as different cubes → crosscore hang (board 507014) |
| `core.vec_idx` | `pl.get_block_idx()` | the GLOBAL AIV index in both launch modes (D-087) |
| `core.sub_block_idx` / `core.vec_num` / `core.cube_num` | `pl.get_subblock_idx()` / `(pl.get_block_num() * pl.get_subblock_num())` / `pl.get_block_num()` | |
| `debug.*` | dropped | observation-only, same policy as the cce sim aids |

| `scalar.load` / `scalar.store` (GetValueFrom / SetValueTo) | `pl.getval(tile_or_tensor, linear_offset)` / `pl.setval(tile_or_tensor, linear_offset, value)` | GM/UB view origins and explicit indices are linearized. Both FP32 UB example spellings pass all three backends (M10-061). Only storage-only low-precision dtypes are [A5-UP-034](upstream.md#a5-up-034). |


#### Sort family

| ascriptor | PyPTO Pro | notes |
|---|---|---|
| `vec.sort32` | `pl.sort32(dst, src, idx)` | our `repeat` is carried by the tile shape |
| `vec.mergesort4` | explicit refusal | Multi-pass expansion and backend-only scratch allocation are removed; see the RFC-0013 sorting subset. |
| `vec.mergesort_2seq` | one `pl.mrgsort(dst, span, block_len=n)` over the 4n-word footprints | only when the runs are ADJACENT in UB, equally long and n is even: the group's four runs are the halves of both lists. A destination overlapping the span is refused: the second pass would need storage the IR never allocated (RFC-0013 sorting subset). `pl.mrgsort2` in its documented order dropped src0 on the board (16 of 64 records, D-109); its graph order is A5-UP-038 |

Every tile the sort family touches needs `Rows == 1` (`TMrgsort: the row of
Destination and Source tile must be 1`), so the printer passes one-row aliases of
the same bytes.

### 1.6 SIMT

Integer `scalar.and` maps to Python `&` inside a SIMT function, including masks
introduced by power-of-two remainder simplification. Operand widths and
signedness are preserved; the adapter adds no narrowing cast.
Boolean conjunction emits `and`; `scalar.select` emits a conditional expression.
These also carry the signed floor-division correction introduced before emission.
Integer literals in arithmetic and select branches become deduplicated, typed
SIMT parameters supplied by `pl.const` at the launch site. SIMT bodies reject
`pl.const`; raw literals promote INT32 arithmetic to INDEX/INT64 and produced
incorrect negative exact quotients on the qualified native stack. The launch
arguments preserve the IR dtype without narrowing runtime operands.

`simt.launch` → `fn[threads](args...)`, the subscripted call upstream replaced
`pl.simt.launch` with (2b49dbfad, 2026-09-15), positional arguments only; the template
is `@pl.vector_function(mode="simt", max_threads=N)`. `simt.thread_id/
block_idx/...` → `pl.simt.*`; `simt.load/store/atomic/cast/ffs/popc/threadfence/
sync_workitems` → their `pl.simt` spellings (D-089).

### 1.7 Sync (lowered `sync.*`)

| ascriptor | PyPTO Pro | notes |
|---|---|---|
| `sync.event` (decl, depth 1) | nothing at the site - the id is inlined at each use | |
| `sync.event` (depth > 1) | `_evN = [ids]  # <event>: SET -> WAIT` + `_evNs/_evNw` counters | short numbered names, IR event name in the comment (D-094) |
| `sync.set` / `sync.wait` | `pl.system.sync_src/sync_dst(set_pipe=P.X, wait_pipe=P.Y, event_id=...)` `[; _evNs += 1]` `# <event>.set/.wait` | rotation via trace-time counters - print-time freezing starves the second wait (board 507014) |
| preset | leading `sync_src` lines at the declaration, `# <event>.arm k/n` | cce's `Event(PRESET)` constructor equivalent |
| `sync.set_all` / `sync.release` | per-id expanded lines, `# <event>.set_all/.release k/n` | release = cce's destructor drain |
| `sync.barrier` | `pl.system.bar_m/bar_mte1/bar_mte2/bar_mte3/bar_fix/bar_all()` | no `bar_v` upstream (gap for V) |
| `sync.mutex` (cv protocol) | `pl.system.set_cross_core/wait_cross_core(..., INTRA_BLOCK)` | line-for-line the cce `CrossCoreSetFlag<0x4>` expansion |

### 1.8 Memory declarations

| ascriptor | PyPTO Pro | notes |
|---|---|---|
| `mem.alloc` (static) | `pl.make_tile(pl.TileType(shape, dtype, target_memory=MEMSPACE[...], [layout=], [pad=]), addr=...)` | our allocator's addresses inlined (TileType cannot cross the closure boundary - parser refuses). No `size=`: PyPTO derives the span from the TileType, the keyword was optional and upstream removed it (9528ff753, 2026-09-17); a declaration whose allocation is not that footprint is refused |
| `mem.alloc` (rotating slots) | `pl.make_tile_group(type=..., addrs=[...], depth=k)` + `grp[cnt % k]` or same-addr `make_tile` | |
| `mem.get_buf/reinterpret/reshape/view/slice` | same-address re-declaration (`make_tile` at the same addr, new shape/dtype/layout) | `reinterpret layout="nz"` is a pass-through marker consumed by `pl.insert` (compact-NZ). A window only re-declares when the RESULT is a legal tile at a parse-time address: an unaligned row, a narrow fractal or a runtime origin keeps it a window record (behaviour #18) |
| `mem.workspace` | GM workspace tensor parameter + offsets | |

Memory spaces: `ub→Vec, l1→Mat, l0a→Left, l0b→Right, l0c→Acc, bt→Bias`.

## 2. Behavioural mapping

Mechanisms where the two backends express one IR decision differently - or
identically, when the mechanism lives in the IR.

**#1 Front-end nature (the root difference).** bisheng is a real C++ compiler:
cce ships `tensorutils_cce.h` templates (`Event<>`, `DBuff`, `gm_to_l1_nd2nz`,
`pack_deq_scalar`) and the compiler folds dead branches. pto's `pl.jit` is an AST
PARSER: every call in the kernel body parses as a DSL op (a helper class dies as
`create_op_call('block.Event')` - box-proven, D-094), so no runtime library is
possible; what the parser accepts beyond ops: module-level enum aliases,
augmented assignment, two statements per line, list indexing, python arithmetic.
Constant `pl.range` loops unroll at parse time, which is why trace-time python
counters (rotation, armed cells) work at all.

**#2 Sync encapsulation.** cce: `DEvent<MTE2, MTE1, 0, 0, 1> ev; ev.set();
ev.wait();` - counters are private members, the destructor releases. pypto: the
same protocol flattened to counter lines with the encapsulated view in tail
comments (`# ev.set`). Same IR event allocation, same ids, same rotation - the
generated flag traffic is instruction-identical (D-094 board: requant bit-exact,
pfa_fd's diff unchanged to the last bit).

**#3 The armed protocol (retired 2026-09-17).** The edge planner guarded a pair
whose producer sat behind a skipping branch with a two-bit cell, and both backends
printed those cells from the IR. No route emits one now: A5 plans slot mutexes and
A2/A3 plan slot sessions, whose credits bound the producer on every path without a
guard (RFC-0005 §5.5). Nothing in either backend reads a cell any more.

**#4 Launch dimension.** `<<<blockDim>>>` counts AICs for a mix binary (each
carries its two AIVs) but counts AIVs DIRECTLY for an AIV-only binary (pto
jit.py `is_aiv_only`). The manifest therefore launches `2 x block_dim` for
`mode == "vec"` modules; the semantic block_dim (and `_core_range`) stays in
cube units. Miss this and the top half of the grid never runs (D-093 board:
rows 0-31 exact, 32-63 zero).

**#5 Core-id semantics.** `get_block_idx()` in a Vector section is the global
AIV id (pto codegens block*subblockdim+subblockid) - never re-derive with
`*2+sub` (D-087). A cube id asked from the vec side must divide back down
(D-088). Both facts hold in both launch modes (an AIV-only launch has
subblock_num == 1).

**#6 Rotating buffers.** cce: `DBuff<T, L0A>.get(cnt)` + `.as<T2>()` views.
pypto: `make_tile_group(addrs, depth)` + `grp[cnt % depth]`, with same-address
`make_tile` re-declarations for typed views. The slot counters are the same IR
values; a mix kernel's vec section may carry orphan counter mirrors (harmless).

**#7 Compact-NZ publish (`ub_to_l1.nz`).** cce composes per-C0-strip bursts in
`ub_to_l1_nz`. pto's TINSERT is the same engine (burst formulas match one for
one) but dispatches on the SOURCE declaration: an ND source copies flat, an
NZ-DECLARED alias takes the row-pitch-adapting path. So the printer loads
through an ND tile and inserts through an NZ alias of the same address;
`pl.set_validshape` covers short tiles; the (TKV+1)-row pack is the same
`block_stride=65` cursor as pto's own FA (D-091, probe bit-exact).

**#8 On-chip transpose (L1→L0).** One byte-identity underlies it:
compact-NZ(M) == ZN(M^T). cce asks for the transpose explicitly
(`l1_to_l0<transpose>`); pto realises it when the two DECLARATIONS disagree on
`SFractal` (NZ = ColMajor + SFractal RowMajor, ZN = RowMajor + SFractal
ColMajor → `load_cbuf_to_ca(..., 1)`, a free 16x16 in-box transpose network).
The printer declares the cube-side view of a transposed operand shape-reversed
with `layout=pl.ZN`. Shape-reversal alone is BLIND on square tiles - the pfa
128x128 P moved untransposed until the ZN gate (D-091).

**#8b The Right side's default pair ALREADY transposes.** `TExtractToLeft`/
`TExtractToRight` (TExtract.hpp:502/532) both dispatch on
`DstTileData::SFractal == SrcTileData::SFractal` - equal means a straight copy,
unequal a transposing one. The A5 defaults are `Mat=NZ, Left=NZ, Right=ZN`
(block_ops.py `_DEFAULT_LAYOUTS_A5`), so a default `pl.move(right, mat)` is
already the transposing form - but ONLY between equal DECLARED shapes. `TMOV`
static-asserts `dst::Rows == src::Rows && dst::Cols == src::Cols`
(TMov.hpp:640), so a Mat `[N, K]` moved into a Right `[K, N]` never compiles;
the SFractal difference decides HOW the bytes are read, never what shape they
are declared at. A narrow-N implicit-B chain therefore cannot "keep `[N, K]`
and let the move transpose" (D-103's reading, refuted on the board by D-106):
it declares the CUBE-SIDE view `[K, N]` with `layout=pl.ZN` over the load's own
`[N, K]` storage - `NZ(B)` and `ZN(B.T)` are literally the same bytes - and the
move into Right (ZN too) is then an equal-shape, equal-SFractal copy. Only the
tile the MOVE reads is re-declared; the tile the GM load fills keeps its natural
`[N, K]` NZ declaration. This is also the only route when the `[K, N]` flip
would break the Mat column rule (mxfp4's B is 16 fp4 = 8 bytes per flipped row).
Behaviour #8
describes the Left side; this inverted Right-side default is the half the
attention kernels ride on. Related: `pl.load` into a Mat tile declared ZN does
not compile without `order` (TLOAD_IMPL has no ZN-destination instantiation) -
pto's own canonical TN matmul only ever writes a ZN Mat tile through a
transposed load.

**#9 VF block cursors (vsstb / vsldb).** cce takes a software pointer with a free
byte offset; pto has NO offset argument, and all three shapes that look like one
are silent wrong answers rather than refusals (D-109, each board-measured):

* a scalar or `[row, col]` positional lands in the vsstb CONFIG WORD's block-stride
  half - pto prints `vsstb(reg, ptr, (off << 16u) | (0 & 0xFFFFU), mask)`, the
  `block_stride=` kwarg is dropped, every access writes at the tile base and the
  register's blocks scatter outside the tile. A compile-time constant offset does
  the same, so this is not a runtime-expression limit;
* an `AddrReg` from `vf.create_addr_reg(i, stride)` swaps the whole call for
  `vst` + `vag_bN(stride)` - the CONTIGUOUS auto-increment store, `data_copy_mode`
  and `block_stride` both dropped. It agrees with vsstb only at `block_stride=1`,
  which is how it passed its first probe;
* a fourth positional on the load side is the de-interleave form and is refused
  ("vf.load_align 4-arg (de-interleave) form does not support data_copy_mode");
  the three-argument form is `(src, mask, ...)`, `args[2]` required to be a mask.

The one address motion is the post-update cursor. pto prints
`(block_stride << 16) | (repeat_stride & 0xFFFF)` + POST_UPDATE, and the tile
pointer - one per tile inside the inlined vf, SHARED by every access through it -
advances `repeat_stride` 32-byte blocks per call. `repeat_stride` is a free step,
and 0 is a legal fixed address. `VfPrinter.cursor_plan` proves each tile's accesses
walk `0, d, 2d, ...` in execution order over the whole loop nest - k accesses per
innermost iteration step by d, the innermost loop's coefficient is `k*d`, the next
one's `k*T_inner*d`, and so on, since an enclosing loop carrying none of it would
let the cursor run away on its second iteration - and derives `repeat_stride` from
d. A slice with a non-zero origin, an access behind a branch and the terminal case,
several DISJOINT cursor regions on one tile, gap with the offending numbers in the
message.

**#10 Requant scale+offset.** cce packs `pack_deq_scalar` ([31:13] f32-scale
bits | offset int9 << 37, bit46 signed-select). pto only encodes the f32 scale -
but widens an INT64 runtime scalar AS-IS into `set_quant_pre(u64)`, so the full
word rides along: `scale=(pl.const(packed, DT_INT64) + pl.get_block_idx() * 0)`.
The zero term defeats the parser's constant re-encoding (a bare const folds to a
python int and re-encodes as a saturated f32 pattern). u8 requant is refused by
pto's parser while cce runs it green (D-092, board bit-exact).

**#11 Zero-fill.** cce: `set_constant_to_l1(view, 0, n)` instruction. pto:
`pl.expands(tile, 0)`, which is the same `create_cbuf_matrix` builtin - see #31.
(The `pad=pl.TilePad.zero` reading of D-090 and the "no fill at all" reading of
D-123 were both wrong; #30 records how.)

**#12 Masks.** Explicit masks carry their LANE WIDTH in `create_mask(dtype=)`;
a b32 mask on a b16 op predicates every other lane under MODE_ZEROING - the
D-093 "odd bytes zero" bug class. Default full masks are derived per-op from the
operand dtype.

**#13 VF functions.** `@vf()` → `@pl.vector_function`, inlined by pto (cce:
per-vf `.h`). Probe-shaped vf functions do not inline (no reg/ptr declarations
emitted upstream) - the reliable probe loop is editing a staged kernel_pypto.py
on the box.

**#14 Specialisation.** cce keeps scalars as runtime `int32_t` (bisheng folds).
pypto binds the golden case's scalars at emit time (`Specialised for: {...}`) -
python-side folding is more aggressive, so e.g. `cube_num > b` appears as
`4 > b`. Same IR, different fold point; neither survives to the machine code.

**#20 A HARDWARE BARRIER may not name more cores than the card has.** pto's
`sync_all` is an FFTS barrier: it waits for every core the launch names, so a
`<<<blockDim>>>` past the card's own count deadlocks - the a5 profile says 32
AICs / 64 AIVs while an `Ascend950PR` card has 28 / 56, and `simt_atomic_add`
hung for nine minutes at 64. Over-subscribing bought a second wave (the runtime
ran the extra blocks after the first and each core's slice is disjoint), which is
what some goldens were recorded with - `matmul_abs_add1_vf` splits M by
`GetCubeNum()` and steps by 128, which tiles cleanly at 32 cores (4096/32) and
OVERLAPS at 28 (147). pl now **rejects** a `block_dim` above the stream's budget
(upstream `859743bb4`, 2026-09-21), so there is no second wave to buy: EVERY
pinned launch is clamped to the board's own core count (`boards.json`'s
`cube_cores`), and the clamp lands on `block_dim` before it folds, so the
`core.cube_num` / `core.vec_num` constants agree with what actually runs. A
CvMutex is a pairwise flag, not a barrier, and does not care.

**#21 A hif8 block access needs the UINT8 CARRIER on the TILE.** `vsstb` /
`vsldb` have no `__ubuf__ hifloat8*` overload; cce prints the same access
through the uint8 carrier, and pto only half-applies it - it casts the REGISTER
to uint8 but leaves the tile pointer hif8, so the call matches no candidate
("no known conversion from `__ubuf__ hifloat8_*`"). A
`pl.reinterpret(tile, dtype=pl.DT_UINT8, shape=[...])` view fixes exactly that,
and board-proven: the register and the mask may stay hif8, only the tile has to
change. The same carrier answers `vf.deinterleave` on a b8 register - cce emits
`vdintlv((vector_u8&)...)`, so the intrinsic exists and only the TYPE was
missing; the printer bit_casts both sources to `DT_UINT8`, de-interleaves, and
bit_casts the two results back.

The carrier is built by the CALLER, never inside the vf (D-126). `pl.reinterpret`
of a tile that reached the callee as a runtime-indexed tile-group element folds to
the group's FIRST slot - pypto binds the view to a compile-time address, so every
rotation writes slot 0 and whoever reads the other slot gets memory the launch
never wrote. A vf that needs a carrier takes it as an extra parameter, and the
caller indexes a PARALLEL uint8 `make_tile_group` over the same addresses.

This one is a workaround for **pypto's** inliner, not for anything in our IR, and it
is worth retiring the day pypto stops needing it (measured on the box's pypto of
2026-09-02; the surveyed master is `bc593e0b`). To re-check: emit a kernel whose vf
reinterprets a slot buffer - `v8_allhif8` is the canonical one - and read the JIT's
own C++ at `build/<kernel>__a5/tk_none/kernel.cpp`. The behaviour is still there
while `_tg_<name>_grp_tiles_0[` appears only as its declaration and an
`__inline_<n>_<name>_u8_0` carries a constant `TASSIGN`; it is fixed once the
inlined body subscripts the group itself. Then the extra parameter, `u8_carrier`,
`vf_carriers` and `u8_groups` can all go and the view moves back inside the vf.

**#19 The AIV launch doubles only for a VEC-SHARDED kernel.** Our IR counts
CUBE cores and models a vec-sharded kernel as running over
`GetVecNum() == 2 * block_dim` participants, so an AIV-only binary needs
`2 * block_dim` AIVs for `pl.get_block_idx()` to cover the shard. A kernel that
never asks which vec participant it is just wants `block_dim` copies, and
doubling it doubles its WORK - `simt_atomic_incdec`'s ring atomics landed twice
(2 instead of 4 per column, and the decrements wrapped to 3 instead of 1) until
the launch was left alone; with `block_dim` unchanged it is bit-exact.

**#18 Tail windows narrow at RUNTIME.** A tail-safe kernel slices a partial
window on the last iteration (`x[m0:m0 + valid_m, ...]`, `matmul(..., k=valid_k)`)
and both halves need saying in pto. The load side: a GM slice whose extents do not
fold keeps its EXPRESSIONS, and the printer wraps the transfer in
`pl.set_validshape(tile, [e0, e1])` / restore - without it the tail iteration reads
a whole tile past the end of the tensor. The matmul side: `pl.matmul` reads the
DECLARED shape, so the M/N/K the op asks for narrow all three operands around the
call. Both clamp with `pl.min(expr, dim)` rather than trusting the interval
analysis, which loses the correlation between a tail extent and its own loop
variable (`(m0 + valid_m) - m0` bounds as `[-28, 128]` for a 64-row tile). Where a
window ALREADY narrows its base tile, the two fold into their elementwise min -
one descriptor holds one shape, and the min is what both of them mean.

**#17 A mask register may not be loop-carried.** Our IR models a VF mask as a
register CELL: the same value is assigned again and again, which printed
literally makes `rowmask` a loop-carried variable. pto's own vf codegen types a
loop-carried variable as `float`, and the C++ then refuses
(`cannot initialize a variable of type 'float' with an lvalue of type MaskReg`).
Each mask WRITE inside a loop therefore takes a fresh python name - exact
exactly when no read of the mask reaches back across the loop boundary, which
the printer proves by scanning each loop body as if nothing came in (a read
before the body's own write refutes it, and the kernel gaps instead).

**#16 MX per-group scales are OPERANDS, not passengers.** Our IR carries cce's
model: the scale rides into L0 as the `src_mx` attribute of `l1_to_l0.mx` and
`mad_mx` finds it there. pto makes it explicit - `pl.matmul_mx(acc, a, b,
scale_a, scale_b)` reads two more tiles, from `MemorySpace.ScaleLeft` and
`ScaleRight`. Four things about that path are not guessable and each cost a
board round (D-105/D-106):

* **Staging is a Mat tile, and its layout is side-specific.** A is
  `[rows, k_groups]` + `layout=pl.ZZ`, B is `[k_groups, rows]` + `layout=pl.NN`.
  The two are the same bytes read two ways (both put group `g` of row `r` at
  `2*(r%16) + g%2` inside box `(r//16, g//2)`, boxes in row-tile-major order),
  and the B orientation is forced: `pl.matmul_mx` checks `scale_b` against the
  rhs tile's own `[K, N]` coordinates ("scale_b shape must match rhs_tile MX
  groups, expected [2, 16] ... got [16, 2]").
* **The L0 scale tile has no address of its own.** The hardware locates it
  implicitly: `addr(scale) = addr(data_tile) >> 4`. Allocating one anywhere else
  compiles and computes with the wrong scales.
* **A dense GM scale is a RANK-3 tensor to pto:** `[rows, k_pairs, 2]`, the
  trailing physical-phase axis statically `2`. Our `[rows, k_groups]` is the
  same bytes, so the printer publishes a `reshape` entry in the manifest and the
  board driver reshapes before the call. The A side loads `order=[0, 1]`; the B
  side, whose GM is still `[rows, k_pairs, 2]` while its tile is
  `[k_groups, rows]`, loads `order=[1, 0]`.
* **The block route stays bytes.** `gm_to_l1_mx_scale` copies pre-packed 32-byte
  blocks, so its `pl.load` goes through a one-row `u8` alias of the same L1
  address - the fractal tile would stride its rows by the GM row length.

**#23 A tile's SHAPE is a C++ type; its EXTENT is a valid shape; its ADDRESS is a
parse-time constant.** Three separate rules, each board-verified, and together they
decide when a window can become a tile of its own and when it has to stay a window
narrowed with `pl.set_validshape`:

* *Row alignment.* A `Vec` tile's row must be a whole number of 32-byte blocks
  (`pto_tile.hpp:1509`, "BFractal_ is RowMajor and SFractal_ is NoneBox: Rows must be
  32 bytes align"), and a fractal tile's columns a whole number of inner-box columns
  (`pto_tile.hpp:1507`, `static_assert(Cols % InnerCols == 0)`, "Layout cols must be
  divisible by inner box cols" - board: `Tile<Acc, float, 32, 2>` and
  `Tile<Right, half, 32, 2>` both fail it). So a single-column or single-scalar UB
  window, and a narrow-N cube fragment, have no tile declaration at all.
* *Valid shape is the way to say "narrow".* Board probe: 16-wide `Right`/`Acc`
  declarations plus `pl.set_validshape(l0b, [32, 2])` and
  `pl.set_validshape(acc, [32, 2])` before `pl.matmul` compute exactly columns 0..1
  (2.9e-06 from the fp32 reference) and leave the rest of the accumulator untouched -
  the store honours the valid shape too. Narrow N is a valid shape, never a type.
* *Address.* `pl.make_tile(addr=)` is fixed while parsing ("'addr' must be a
  compile-time integer, got ...", board), so a window whose ORIGIN is a runtime value
  cannot be a `make_tile`. When the origin's value SET enumerates at parse time it
  becomes a `pl.make_tile_group(addrs=[...])` selected by the strip ordinal - which is
  how a sub-block-indexed UB half and a CTRL-flag-indexed column strip are printed. A
  saturation flag qualifies because it is one CTRL bit: `pl.get_ctrl_spr(bit, bit)` is
  0 or 1, so `ub[0:1, a*8 : a*8+8]` enumerates into a 2-entry group indexed by `a`.

**#24 A COLUMN tile is fed through a trailing unit axis, not through `order`.** A GM
window that is a contiguous run of N elements (`g[b, h, c, r0:r0+N]`, the AscendC
`n_burst=N, burst_len=1 element` copy) filling a `[N, 1]` UB window is not a shape
mismatch to route around: `pl.load` maps the tile's dimensions to the tensor's LAST
TWO axes, so the run has to BECOME the row axis. Appending a unit axis to the tensor
does exactly that and moves no byte - a trailing 1 multiplies no stride, so every
offset term stays where it was and the row stride becomes 1. Board: bit-exact for
both spellings (the parameter's own axes plus a trailing 1, and a flat `[total, 1]`
view at the linear offset). The obvious alternative, naming the axes in reverse
(`order=[rank-1, rank-2]`), is the DN form and has NO Vec destination - the board
answers with `tload_common.hpp:344`, "Src and dst layout must be same!", because a
`Tile<TileType::Vec, ..., BLayout::RowMajor>` only takes a `Layout::ND` tensor.

**#22 pl carries a pad MODE, never a pad VALUE.** `pl.TilePad` is
`null / zero / max / min` and `pl.TileType(pad=)` takes nothing else (board:
`pad=-1.5` -> "TileType.pad must be a enum TilePad or compile-time integer 0/1/2/3";
`pad=7` -> "must be one of TilePad.null/zero/max/min"). `pl.fillpad(out, src, mode=)`
chooses `NORMAL / EXPAND / INPLACE` - how the pad region is filled, not with what -
and the installed `pypto_pro` package has zero occurrences of `pad_value`,
`padValue`, `constant_value` or `fill_value`. The C++ Tile does carry a runtime pad
value (`pto_tile.hpp:1332` `GetPadValue` / `SetPadValue`), so this is a pl-surface
absence like D-104's nz2dn, not a hardware one. NDDMA edge padding
(`loop_left_pad` / `config_left_pad` / `nearest_value_mode`) has no pl channel either.

**#26 A strided block access addresses `tile + offset`.** `vf.store_align(tile
+ off, reg, mask, data_copy_mode=DATA_BLOCK_COPY, block_stride=N)` and the
matching `vf.load_align` take the offset as POINTER ARITHMETIC ON THE TILE - in
the tile's own elements, any runtime expression, no `repeat_stride` and no
`post_update`. That is what the old easyasc bridge emitted, and the board agrees
(a runtime offset with `block_stride=2` writes exactly the two disjoint slabs).
The traps the D-109 list catalogued are all about the offset as an ARGUMENT; they say
nothing about the address expression, so there is no walk constraint on a tile's
accesses at all.

**#25 An ADDRESS pl can compute is not an address pl can DECLARE.** Three
board-verified walls, all reached from the same wish - "give this iteration its
own tile / its own tensor":

- `pl.make_tile`'s `addr` **must be a compile-time integer** ("addr is fixed
  while parsing; pass a literal ... not a runtime value such as a tensor shape or
  loop index"). A per-iteration UB alias is therefore impossible.
- A `pl.make_tile_group` indexed by a **@vf loop variable** parses but does not
  codegen (bisheng: `use of undeclared identifier '__inline_0_g_0'`). The group's
  slot selection is resolved outside the inlined vf body, where that variable does
  not exist. Indexed by a SECTION-level counter it works - that is the rotating
  double-buffer idiom, and the MX scale group rides it.
- `ptr.make_ptr` takes a `PtrType` or a `TensorType`, never a **ScalarType**
  ("Use pl.Ptr[dtype] for pointer params ..."), so a pointer VALUE read out of
  device memory cannot become a tensor. That is what an AscendC `ListTensorDesc`
  is made of, and it is why a `GMList` parameter has no RUNTIME spelling. Packing
  the members into one buffer does not save it either: `pl.make_tensor` over
  `pl.addptr(base, off)` inside a device loop does NOT vary per iteration (a
  3-member probe read the LAST member on every iteration). What a GMList DOES get
  is the same treatment as every other shape here - **specialisation**: the arity
  and the member extents join the call's bindings, the parameter expands into one
  tensor parameter per member, and the member loop unrolls (`list.count` and
  `list.item_dim` fold to literals, `list.item` names a parameter). pypto-gym's
  own limitations note prescribes a fixed maximum arity padded with unused slots
  and measures 6.46x device time for it; per-call specialisation has no padding
  and no unused slots to execute. The cost is that the emitted kernel serves ONE
  arity - which is exactly what it already did for one set of shapes and scalars.

What DOES vary per tile is the vsstb/vsldb **post-update cursor**: it is per tile
EXPRESSION, not global - two tiles interleaved inside one `@pl.vector_function`
each keep their own walk (board). That is what makes the cursor plan of
`VfPrinter.cursor_plan` a per-tile analysis; it is also why the disjoint-region
gap cannot be fixed from the backend, since the only way to hand a loop a second
tile is one of the three walls above.

**#27 `pl.move`'s whitelist, verbatim, and where TINSERT goes further.** The
parser prints the whole list when it refuses: `Mat->Left, Mat->Right,
Mat->Scaling, Mat->Bias, Mat->ScaleLeft, Mat->ScaleRight, Acc->Vec, Vec->Vec,
Vec->Mat`. **Acc->Mat is absent**, so the documented "L0C -> UB/L1" data path is
not reachable through `pl.move`. `pl.insert(mat_tile, acc_tile, [row, col])`
reaches it - TINSERT writes Mat and Vec destinations - and that is how an L0C
result is published back to L1 without a UB round trip.

**#15 Known numeric deltas (tolerated, not mapped away).** hif8 byte 0x80
decodes to a different quiet-NaN payload than cce (f32: 0x7fffffff vs
0x7fc00000; f16: 0x7fff vs 0x7e00) - semantically equal, `equal_nan` tolerance
entries. The remaining tolerance families are rounding-order effects listed in
`tests/kernels/a5/corpus.json` board_tolerance.

**#28 A float immediate reaches the hardware with SIX DECIMALS, so a constant
that needs more of them rides in as its integer bit pattern.** pypto's native
CCE printer renders a float immediate the way `std::to_string` does - six digits
after the point - and nothing warns. `vf.muls(reg, 0.004464285714285714)` (the
frontend's spelling of `reg / 224.0`) arrives as `vmuls(..., 0.004464f, ...)`,
a relative error of -6.4e-05, a thousand times worse than fp32; `1/sqrt(128)`
loses 3.9e-06; and `vf.maxs(reg, 5.877471754111438e-39)`, a clamp away from
zero, arrives as `0.0` and stops clamping at all. An INTEGER immediate is
printed exactly, so a constant that does not survive the six-decimal rendering
is materialised as `vf.full(<int bits>, dtype=DT_INT32)` + `vf.bit_cast(...,
dtype=DT_FP32)` - hoisted to the top of the `@pl.vector_function`, where it
costs one `vbr` per function and no instruction for the cast - and the op moves
to its register form (`muls -> mul`, `adds -> add`, `mins -> min`, `maxs ->
max`, `cmps -> cmp`). Board-proven on `matmul_kmkn_blockwise_quant128`: the
scale output goes from 1.4e-05 to 7.4e-08 (the cce path's own accumulation-order
residual) and the e5m2 output from six differing bytes to BIT-EXACT.

**#29 One L1 tile, two B roles: the transposed one goes through the ZN alias.**
`pl.move` into Right transposes or not by SFractal INEQUALITY, never by the
shapes, so the choice is made in the source Mat tile's declared layout. When a
kernel uses one L1 tile as a straight B and, elsewhere, as `B.T` - MLA's K and V
are the same tensor - variant C has already declared it `[K, N]` for the straight
use, and the transposed use is then `[N, K]` against that declaration. Read as a
window on the NZ tile, pto's transposing extract (`TExtractToBTransCompact`)
addresses the source with `indexRow` on the K axis and `indexCol` on the N axis
and takes its extents from the DESTINATION's valid shape - so a splitn window
placed in `indexRow` silently reads the wrong keys, and `set_validshape` on the
source does nothing at all. The spelling that works is the reversed-shape
`layout=pl.ZN` alias over the same L1 bytes plus a non-transposing extract:
`pl.move(l0b_view, l1kn_zn[i], offset=[0, n])`.

**#30 A Mat tile's `pad` promises nothing and corrupts its loads.** pto sets a
tile's pad value under `pto_set_tload_pad_val<TileType::Vec>` only
(tload_common.hpp); the Mat load path never reads `PadVal`. Declaring
`pad=pl.TilePad.zero` on an L1 tile therefore guarantees nothing - and on the
board it also BREAKS the loads into that tile: v8_allhif8 was nondeterministic
garbage with the pad and is deterministic and finite without it. Nothing else in
the a5 set fills L1 (`TASSIGN` binds an address, `TFILLPAD` static-asserts a Vec
destination) and the cube cannot reach UB, so `set_constant_to_l1` has no pl
spelling at all: it prints as a dropped-op comment, and a kernel that reads the
rows no load covers gets whatever the previous kernel left there.

**#31 `set_constant_to_l1` IS `pl.expands` on a Mat tile.** #30 is right that the
`pad` is not the fill and wrong that there is no fill: pl's `expands(out, scalar)`
(`language/_api.py:385`) checks the destination's DTYPE and nothing else
(`_EXPANDS_DTYPES`, `ir/op/block_ops.py:1003`), so an L1 tile reaches `TEXPANDS`'s
cbuf form - `pto_create_cbuf_matrix`, the same builtin cce's `create_cbuf_matrix`
reaches. The generated CCE says so: `TEXPANDS(_tg_l1x_grp_tiles_0[_ix_0],
0.000000f)`. Three things have to line up. **The repeat count** is a parameter in
cce and a property of the tile in pto (`repeatTimes = Rows * Cols * sizeof(T) / 32`,
`TExpandS.hpp:186`), so the two describe the same transfer only for a whole-tile
fill - which is every corpus op, all of them zeroing an L1 operand before a partial
matmul writes into it; a partial clear refuses by name rather than filling a
different number of blocks. **`repeatConfig`** is encoded the other way round (cce:
one repeat of n blocks; pto: n repeats of one block, gap 0) and means the same
thing. **float8 has no TEXPANDS instantiation**, so such a tile fills through a
same-width INTEGER view - a reinterpretation of the TILE, never of the VALUE, so it
is taken only when the bit pattern is what is being written, i.e. zero. And one
thing ships with the instruction that is not part of it: `pl.system.bar_mte2()`
after every fill. autosync models this op on MTE2 because that is what cce's
`create_cbuf_matrix` is, so it puts no event between the fill and the load that
overwrites part of the same L1 slot; pto reaches a different builtin and on silicon
the two reorder.

**#32 A cast's saturation flag has to be stated, and a float8 destination refuses
it.** cce's `vcvt` takes `RS_ENABLE` exactly when the op's `saturate` attribute is
set, and nothing in the a5 corpus sets it. pl's `vf.astype(..., saturate=)` documents
`SaturateMode.OFF` as its default and does not honour it: omitting the kwarg printed
`RS_ENABLE` on 51 of `v8_allhif8`'s 55 casts. State it. The one place it cannot be
stated is a float8 / float4 DESTINATION, where pto refuses the value cce prints -
"vf.astype: FP32->FP8 conversion requires saturate=ON (RS_ENABLE), OFF is not
supported for this path" - so the kwarg is left off there and pto's forced saturation
stands. The two backends then differ only on values outside the destination type's
range: cce sends them to infinity, pto clamps them (D-125).

**#33 A zero-extent transfer is the kernel's job to skip, not pl's to ignore.** An
idle core computes zero rows and still reaches its load; the printer narrows the
destination and asks for the move, which on cce is a zero-burst DMA and in the
interpreter an empty slice — both no-ops. `pl.load` rejects it:
`pl.set_validshape(slot, [0, 512])` followed by `pl.load(slot, view, [0, 0])` raises
`InvalidShape: load: offsets[0]=0 exceeds tensor dim 0 size 0`
(`block_ops.py:1186`), because the bounds check reads an empty range as an
out-of-range index. This is a **decided restriction, not a defect to wait on**
([A5-UP-048](upstream.md#a5-up-048)): a kernel whose extent can reach zero guards its
own transfer, as `matrix_block_quant`'s `if rows > 0` and
`a5_mla_fp16_bf16/kernels/online_paired.py` do. The guard is exact rather than
defensive — those kernels already clear the destination on the same branch that makes
the extent zero, so the skipped transfer would have moved nothing. Only the backend
that refuses it pays: cce and PTO ISA print the same guarded form and lose nothing.

**#34 A floor division and remainder print as `//` and `%`, with no correction around them.**
`pl`'s integer `//` and `%` round toward negative infinity for signed operands - upstream
`5866c9b6f` (2026-09-14) aligned them with Python, and it is an ancestor of the `289942aa3`
minimum these sources target. That is what Ascriptor's floor `scalar.div` and `scalar.mod` mean,
so `integer_division` leaves them alone for this backend (`native_floor_divmod`) and the printer
writes the operator. Expanding them here instead printed the native ops as the truncating half of
a correction the reader then applied a second time: measured on the a5 box, `-7 // 3` returned
`-4` and `-7 % 3` returned `5`, wrong for every negative dividend whose remainder is not zero.
Ceiling division and alignment still expand, because `pl` has no operator for them. This also
removed [A5-UP-047](upstream.md#a5-up-047)'s trigger: the shape that made bisheng abort
instruction selection inside a vector function was the expansion, and the bare `%` compiles and
runs there - the three `a5_mla_fp16_bf16` cases that used to crash pass on their original source.
cce and PTO ISA keep the expansion they always had, and `compact_integer_mod` still keeps a floor
remainder a single typed helper call for them.
